"""Data / model snapshots outside the repository (RUNBOOK §4, ADR-013).

Layout: `<root>/<kind>/<name>/<snapshot_id>/{archive.tar, manifest.json}`
- `snapshot_id` = `<UTC yyyymmddTHHMMSSZ>-<content_sha256[:8]>`.
- One read pass per file: bytes are hashed while being written into the tar.
- A snapshot directory appears atomically (built under a `.partial-*` name, then renamed).
- Creating a snapshot whose content equals the newest one of that name is a no-op.
- Restore never overwrites: the target must not exist or be empty; files are verified
  against the manifest before the restored directory is moved into place.
- Retention: newest `keep` snapshots per (kind, name) plus all pinned ones survive `prune`.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, NonNegativeInt

from supportpilot.common.hashing import HashingReader, sha256_file, sha256_text
from supportpilot.common.ids import validate_slug
from supportpilot.common.manifest import FileProblem, iter_files, read_model_json, verify_files
from supportpilot.common.manifest import write_model_json as _write_model_json
from supportpilot.common.schemas import FileEntry, Sha256, Slug

Kind = Literal["data", "model"]
KINDS: tuple[Kind, ...] = ("data", "model")
ARCHIVE = "archive.tar"
MANIFEST = "manifest.json"


class BackupError(Exception):
    """Invalid backup operation (bad paths, missing snapshot, verification failure)."""


class SnapshotManifest(BaseModel):
    schema_version: Literal["snapshot/1"] = "snapshot/1"
    snapshot_id: str
    name: Slug
    kind: Kind
    created_at: AwareDatetime
    source_path: str
    source_is_file: bool
    code_commit: str | None
    note: str | None = None
    pinned: bool = False
    content_sha256: Sha256
    archive_bytes: NonNegativeInt
    archive_sha256: Sha256
    total_bytes: NonNegativeInt
    files: list[FileEntry]


class VerifyResult(BaseModel):
    snapshot_id: str
    ok: bool
    deep: bool
    problems: list[FileProblem]


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def ensure_backup_root(root: Path, project_root: Path) -> Path:
    root = root.resolve()
    if _is_within(root, project_root):
        raise BackupError(f"backup root {root} must be outside the repository {project_root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _content_sha256(files: list[FileEntry]) -> str:
    canonical = json.dumps([f.model_dump() for f in files], sort_keys=True, separators=(",", ":"))
    return sha256_text(canonical)


def snapshot_dirs(root: Path, kind: Kind | None = None, name: str | None = None) -> list[Path]:
    """Complete snapshot dirs (with a manifest), oldest first."""
    out: list[Path] = []
    for k in [kind] if kind else KINDS:
        base = root / k
        if not base.is_dir():
            continue
        names = [base / name] if name else sorted(p for p in base.iterdir() if p.is_dir())
        for ndir in names:
            if ndir.is_dir():
                out.extend(
                    d
                    for d in ndir.iterdir()
                    if d.is_dir() and not d.name.startswith(".") and (d / MANIFEST).is_file()
                )
    return sorted(out, key=lambda d: (d.parent.parent.name, d.parent.name, d.name))


def load_manifest(snapshot_dir: Path) -> SnapshotManifest:
    path = snapshot_dir / MANIFEST
    if not path.is_file():
        raise BackupError(f"not a snapshot (no {MANIFEST}): {snapshot_dir}")
    return read_model_json(path, SnapshotManifest)


def list_snapshots(
    root: Path, kind: Kind | None = None, name: str | None = None
) -> list[SnapshotManifest]:
    return sorted(
        (load_manifest(d) for d in snapshot_dirs(root, kind, name)),
        key=lambda m: (m.kind, m.name, m.created_at),
    )


def resolve_snapshot(root: Path, ref: str) -> Path:
    """Accept a snapshot directory path or `<kind>/<name>/<snapshot_id>` relative to root."""
    cand = Path(ref)
    for d in (cand, root / ref):
        if (d / MANIFEST).is_file():
            return d
    raise BackupError(f"snapshot not found: {ref}")


def create_snapshot(
    source: Path,
    *,
    name: str,
    kind: Kind,
    root: Path,
    project_root: Path,
    code_commit: str | None = None,
    note: str | None = None,
    pin: bool = False,
) -> tuple[SnapshotManifest, bool]:
    """Snapshot `source` (file or directory). Returns (manifest, created_new)."""
    root = ensure_backup_root(root, project_root)
    source = source.resolve()
    if not source.exists():
        raise BackupError(f"source does not exist: {source}")
    if _is_within(root, source):
        raise BackupError("backup root must not be inside the source")
    validate_slug(name, "snapshot name")
    if kind not in KINDS:
        raise BackupError(f"kind must be one of {KINDS}")
    name_dir = root / kind / name
    name_dir.mkdir(parents=True, exist_ok=True)
    partial = name_dir / f".partial-{uuid.uuid4().hex[:8]}"
    partial.mkdir()
    try:
        if source.is_file():
            members = [(source, source.name)]
        else:
            members = [(p, p.relative_to(source).as_posix()) for p in iter_files(source)]
        entries: list[FileEntry] = []
        with tarfile.open(partial / ARCHIVE, "w", format=tarfile.PAX_FORMAT) as tar:
            for path, arcname in members:
                info = tar.gettarinfo(str(path), arcname=arcname)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with path.open("rb") as fh:
                    reader = HashingReader(fh)
                    tar.addfile(info, fileobj=reader)
                if reader.bytes_read != info.size:
                    raise BackupError(f"file changed while archiving: {arcname}")
                entries.append(FileEntry(path=arcname, bytes=info.size, sha256=reader.hexdigest()))
        content = _content_sha256(entries)

        existing = list_snapshots(root, kind, name)
        if existing and existing[-1].content_sha256 == content:
            shutil.rmtree(partial)
            return existing[-1], False

        created = datetime.now(UTC)
        snapshot_id = f"{created:%Y%m%dT%H%M%SZ}-{content[:8]}"
        manifest = SnapshotManifest(
            snapshot_id=snapshot_id,
            name=name,
            kind=kind,
            created_at=created,
            source_path=str(source),
            source_is_file=source.is_file(),
            code_commit=code_commit,
            note=note,
            pinned=pin,
            content_sha256=content,
            archive_bytes=(partial / ARCHIVE).stat().st_size,
            archive_sha256=sha256_file(partial / ARCHIVE),
            total_bytes=sum(e.bytes for e in entries),
            files=entries,
        )
        _write_model_json(partial / MANIFEST, manifest)
        final = name_dir / snapshot_id
        if final.exists():
            raise BackupError(f"snapshot already exists: {final}")
        partial.rename(final)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return manifest, True


def verify_snapshot(snapshot_dir: Path, *, deep: bool = True) -> VerifyResult:
    """Quick: archive sha256. Deep: also hash every tar member against the manifest."""
    m = load_manifest(snapshot_dir)
    problems: list[FileProblem] = []
    archive = snapshot_dir / ARCHIVE
    if not archive.is_file():
        problems.append(FileProblem(path=ARCHIVE, problem="missing"))
    elif sha256_file(archive) != m.archive_sha256:
        problems.append(FileProblem(path=ARCHIVE, problem="sha256"))
    elif deep:
        expected = {e.path: e for e in m.files}
        seen: set[str] = set()
        with tarfile.open(archive, "r") as tar:
            for member in tar:
                if not member.isfile():
                    problems.append(FileProblem(path=member.name, problem="unexpected"))
                    continue
                seen.add(member.name)
                entry = expected.get(member.name)
                fh = tar.extractfile(member)
                if entry is None or fh is None:
                    problems.append(FileProblem(path=member.name, problem="unexpected"))
                    continue
                reader = HashingReader(fh)  # type: ignore[arg-type]
                while reader.read(1 << 20):
                    pass
                if reader.hexdigest() != entry.sha256:
                    problems.append(FileProblem(path=member.name, problem="sha256"))
        problems.extend(
            FileProblem(path=p, problem="missing") for p in sorted(set(expected) - seen)
        )
    return VerifyResult(snapshot_id=m.snapshot_id, ok=not problems, deep=deep, problems=problems)


def restore_snapshot(snapshot_dir: Path, target: Path) -> SnapshotManifest:
    """Restore into `target` (must not exist or be empty). Verified before it is moved in place."""
    m = load_manifest(snapshot_dir)
    quick = verify_snapshot(snapshot_dir, deep=False)
    if not quick.ok:
        raise BackupError(f"archive integrity check failed: {quick.problems}")
    target = target.resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise BackupError(f"restore target exists and is not an empty directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.restoring-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    try:
        with tarfile.open(snapshot_dir / ARCHIVE, "r") as tar:
            tar.extractall(staging, filter="data")
        problems = verify_files(staging, m.files)
        if problems:
            raise BackupError(f"restored files do not match the manifest: {problems}")
        if target.exists():
            target.rmdir()
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return m


def set_pinned(snapshot_dir: Path, pinned: bool) -> SnapshotManifest:
    m = load_manifest(snapshot_dir).model_copy(update={"pinned": pinned})
    _write_model_json(snapshot_dir / MANIFEST, m)
    return m


def prune(root: Path, *, keep: int, dry_run: bool = True) -> list[Path]:
    """Delete snapshots beyond the newest `keep` per (kind, name); pinned ones are kept."""
    if keep < 1:
        raise BackupError("keep must be >= 1")
    groups: dict[tuple[str, str], list[tuple[SnapshotManifest, Path]]] = {}
    for d in snapshot_dirs(root):
        m = load_manifest(d)
        groups.setdefault((m.kind, m.name), []).append((m, d))
    doomed: list[Path] = []
    for items in groups.values():
        items.sort(key=lambda x: x[0].created_at, reverse=True)
        doomed.extend(d for m, d in items[keep:] if not m.pinned)
    if not dry_run:
        for d in doomed:
            shutil.rmtree(d)
    return doomed
