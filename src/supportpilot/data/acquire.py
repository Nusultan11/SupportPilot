"""Download pinned source revisions into immutable `data/raw/{source_id}/{revision}/`.

- Files are listed at the pinned commit; each download is checked against the upstream hash
  (LFS sha256, or git blob sha1 for small files) before it is accepted.
- Work happens in `data/raw/{source_id}/.partial-{revision}/` (a stable name), so an
  interrupted run resumes: complete files are re-verified and kept, partial ones resume.
- The finished directory gets a `manifest.json` (RawManifest) and is renamed into place.
  An existing revision is only verified, never modified; if it is corrupt the run stops and
  points to the backup instead of silently re-downloading.
"""

from __future__ import annotations

import fnmatch
import hashlib
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel

from supportpilot.common.hashing import sha256_file
from supportpilot.common.logging import get_logger
from supportpilot.common.manifest import (
    FileProblem,
    read_model_json,
    scan_files,
    verify_files,
    write_model_json,
)
from supportpilot.common.schemas import RawManifest, SourceEntry
from supportpilot.data.policy import Purpose, check_use

log = get_logger(__name__)
MANIFEST = "manifest.json"
_HF_META_DIR = ".cache"


class AcquireError(Exception):
    """Download or verification failed; nothing was finalised."""


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    sha256: str | None = None  # LFS files
    git_sha1: str | None = None  # regular git blobs


class RemoteSource(Protocol):
    def list_files(self) -> list[RemoteFile]: ...

    def download(self, file: RemoteFile, dest_root: Path) -> None:
        """Write `dest_root / file.path` (resuming a previous partial download if possible)."""


class AcquireResult(BaseModel):
    source_id: str
    revision: str
    status: Literal["downloaded", "skipped"]
    path: str
    files: int
    bytes: int
    seconds: float


def raw_dir(data_dir: Path, entry: SourceEntry) -> Path:
    return data_dir / "raw" / entry.source_id / entry.revision


def git_blob_sha1(path: Path) -> str:
    h = hashlib.sha1(usedforsecurity=False)
    h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def check_remote_file(path: Path, rf: RemoteFile) -> str | None:
    """None if `path` matches the upstream size and hash, else a short problem string."""
    if not path.is_file():
        return "missing"
    size = path.stat().st_size
    if size != rf.size:
        return f"size {size} != upstream {rf.size}"
    if rf.sha256 is not None and sha256_file(path) != rf.sha256:
        return "sha256 differs from upstream"
    if rf.sha256 is None and rf.git_sha1 is not None and git_blob_sha1(path) != rf.git_sha1:
        return "git blob sha1 differs from upstream"
    if rf.sha256 is None and rf.git_sha1 is None:
        return "upstream provides no hash"
    return None


def select_files(files: list[RemoteFile], include: list[str]) -> list[RemoteFile]:
    chosen: dict[str, RemoteFile] = {}
    for pattern in include:
        hits = [f for f in files if fnmatch.fnmatchcase(f.path, pattern)]
        if not hits:
            raise AcquireError(f"include pattern {pattern!r} matches no file at this revision")
        chosen.update({f.path: f for f in hits})
    return sorted(chosen.values(), key=lambda f: f.path)


def verify_raw(path: Path) -> list[FileProblem]:
    """Deep check of an acquired revision against its manifest."""
    manifest_path = path / MANIFEST
    if not manifest_path.is_file():
        return [FileProblem(path=MANIFEST, problem="missing")]
    m = read_model_json(manifest_path, RawManifest)
    return verify_files(path, m.files, exclude=frozenset({MANIFEST}))


def _download_verified(
    remote: RemoteSource, rf: RemoteFile, staging: Path, retries: int, backoff_s: float
) -> None:
    target = staging / rf.path
    if check_remote_file(target, rf) is None:
        log.info("already complete, kept", extra={"file": rf.path})
        return
    last = "not attempted"
    for attempt in range(1, retries + 1):
        try:
            remote.download(rf, staging)
        except AcquireError:
            raise
        except Exception as exc:  # network errors etc.; retried
            last = f"{type(exc).__name__}: {exc}"
            log.warning(
                "download failed", extra={"file": rf.path, "attempt": attempt, "error": last}
            )
            time.sleep(min(backoff_s * 2 ** (attempt - 1), 30.0))
            continue
        problem = check_remote_file(target, rf)
        if problem is None:
            return
        last = problem
        log.warning("downloaded file rejected", extra={"file": rf.path, "problem": problem})
        target.unlink(missing_ok=True)
    raise AcquireError(f"{rf.path}: failed after {retries} attempts ({last})")


def acquire_source(
    entry: SourceEntry,
    data_dir: Path,
    remote: RemoteSource,
    *,
    retries: int = 3,
    backoff_s: float = 2.0,
) -> AcquireResult:
    check_use(entry, Purpose.ACQUIRE)
    if entry.fetch is None:
        raise AcquireError(f"{entry.source_id}: no fetch spec in the registry")
    start = time.perf_counter()
    final = raw_dir(data_dir, entry)

    if final.exists():
        problems = verify_raw(final)
        if problems:
            raise AcquireError(
                f"{final} exists but fails verification ({len(problems)} problems, first: "
                f"{problems[0].path} {problems[0].problem}). Raw data is immutable: restore it "
                "from backup (RUNBOOK §6) or move it aside manually."
            )
        m = read_model_json(final / MANIFEST, RawManifest)
        return AcquireResult(
            source_id=entry.source_id,
            revision=entry.revision,
            status="skipped",
            path=str(final),
            files=len(m.files),
            bytes=sum(f.bytes for f in m.files),
            seconds=round(time.perf_counter() - start, 2),
        )

    wanted = select_files(remote.list_files(), entry.fetch.include)
    staging = final.parent / f".partial-{entry.revision}"
    staging.mkdir(parents=True, exist_ok=True)
    for rf in wanted:
        _download_verified(remote, rf, staging, retries, backoff_s)

    shutil.rmtree(staging / _HF_META_DIR, ignore_errors=True)
    wanted_paths = {rf.path for rf in wanted}
    for leftover in scan_files(staging, exclude=frozenset({MANIFEST})):
        if leftover.path not in wanted_paths:  # e.g. from an earlier include list
            (staging / leftover.path).unlink()
    entries = scan_files(staging, exclude=frozenset({MANIFEST}))
    if {e.path for e in entries} != wanted_paths:
        raise AcquireError("staging content does not match the requested file list")

    manifest = RawManifest(
        source_id=entry.source_id,
        revision=entry.revision,
        url=entry.url,
        retrieved_at=datetime.now(UTC),
        licence=entry.licence,
        repo_id=entry.fetch.repo_id,
        files=entries,
    )
    write_model_json(staging / MANIFEST, manifest)
    staging.rename(final)
    log.info("source acquired", extra={"source_id": entry.source_id, "files": len(entries)})
    return AcquireResult(
        source_id=entry.source_id,
        revision=entry.revision,
        status="downloaded",
        path=str(final),
        files=len(entries),
        bytes=sum(e.bytes for e in entries),
        seconds=round(time.perf_counter() - start, 2),
    )


class HfDatasetRemote:
    """Hugging Face dataset repo at a pinned commit. Uses HF_TOKEN if set (gated repos)."""

    def __init__(self, repo_id: str, revision: str) -> None:
        self.repo_id = repo_id
        self.revision = revision

    def list_files(self) -> list[RemoteFile]:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

        try:
            info = HfApi().dataset_info(self.repo_id, revision=self.revision, files_metadata=True)
        except GatedRepoError as exc:
            raise AcquireError(
                f"{self.repo_id} is gated: accept its terms on huggingface.co with your account "
                "and set HF_TOKEN"
            ) from exc
        except RepositoryNotFoundError as exc:
            raise AcquireError(f"{self.repo_id}@{self.revision} not found") from exc
        out: list[RemoteFile] = []
        for s in info.siblings or []:
            if s.size is None:
                raise AcquireError(f"{s.rfilename}: upstream size unknown")
            sha256 = s.lfs.sha256 if s.lfs is not None else None
            out.append(RemoteFile(path=s.rfilename, size=s.size, sha256=sha256, git_sha1=s.blob_id))
        return out

    def download(self, file: RemoteFile, dest_root: Path) -> None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError

        try:
            hf_hub_download(
                self.repo_id,
                file.path,
                repo_type="dataset",
                revision=self.revision,
                local_dir=dest_root,
            )
        except GatedRepoError as exc:
            raise AcquireError(f"{self.repo_id} is gated: set HF_TOKEN after accepting") from exc
