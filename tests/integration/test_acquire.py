"""Acquisition: upstream hash checks, idempotent re-runs, resume after interruption,
immutability of finished revisions. Uses an in-memory remote (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from supportpilot.common.manifest import read_model_json
from supportpilot.common.schemas import RawManifest
from supportpilot.data.acquire import (
    AcquireError,
    RemoteFile,
    acquire_source,
    check_remote_file,
    git_blob_sha1,
    raw_dir,
    select_files,
    verify_raw,
)
from supportpilot.data.policy import PolicyError
from tests.data_helpers import FakeRemote, Interrupted, entry

CSV = b"instruction,intent,category,tags,response\nhi,greet,MISC,B,hello\n" * 2000
FILES = {"data.csv": CSV, "README.md": b"# card\n", ".gitattributes": b"*.csv lfs\n"}


def _remote() -> FakeRemote:
    return FakeRemote(dict(FILES), lfs=frozenset({"data.csv"}))


def _acquire(tmp: Path, remote: FakeRemote, **kw: object) -> object:
    return acquire_source(entry(), tmp, remote, backoff_s=0, **kw)  # type: ignore[arg-type]


def test_git_blob_sha1_matches_git(tmp_path: Path) -> None:
    p = tmp_path / "f"
    p.write_bytes(b"hello\n")
    assert git_blob_sha1(p) == "ce013625030ba8dba906f756967f9e9ca394464a"  # `git hash-object`


def test_acquire_writes_immutable_revision_with_manifest(tmp_path: Path) -> None:
    remote = _remote()
    result = _acquire(tmp_path, remote)
    final = raw_dir(tmp_path, entry())
    assert result.status == "downloaded"  # type: ignore[attr-defined]
    assert sorted(p.name for p in final.iterdir()) == ["README.md", "data.csv", "manifest.json"]
    assert (final / "data.csv").read_bytes() == CSV  # byte-identical to upstream
    m = read_model_json(final / "manifest.json", RawManifest)
    assert {f.path for f in m.files} == {"README.md", "data.csv"}  # .gitattributes not included
    assert m.licence == "cc-by-4.0"
    assert m.revision == entry().revision
    assert verify_raw(final) == []
    assert not list(final.parent.glob(".partial-*"))


def test_rerun_skips_without_touching_the_remote(tmp_path: Path) -> None:
    _acquire(tmp_path, _remote())
    remote = _remote()
    result = _acquire(tmp_path, remote)
    assert result.status == "skipped"  # type: ignore[attr-defined]
    assert remote.list_calls == 0
    assert remote.downloads == []


def test_transient_errors_are_retried(tmp_path: Path) -> None:
    remote = _remote()
    remote.fail_next = {"data.csv": 2}
    _acquire(tmp_path, remote, retries=3)
    assert remote.downloads.count("data.csv") == 3


def test_retries_exhausted_leaves_nothing_final(tmp_path: Path) -> None:
    remote = _remote()
    remote.fail_next = {"data.csv": 5}
    with pytest.raises(AcquireError, match="failed after 2 attempts"):
        _acquire(tmp_path, remote, retries=2)
    assert not raw_dir(tmp_path, entry()).exists()


def test_corrupted_download_is_rejected_and_refetched(tmp_path: Path) -> None:
    remote = _remote()
    remote.corrupt_next = {"data.csv": 1}
    _acquire(tmp_path, remote)
    assert remote.downloads.count("data.csv") == 2
    assert (raw_dir(tmp_path, entry()) / "data.csv").read_bytes() == CSV


def test_persistently_corrupted_upstream_fails(tmp_path: Path) -> None:
    remote = _remote()
    remote.corrupt_next = {"data.csv": 99}
    with pytest.raises(AcquireError, match="sha256 differs"):
        _acquire(tmp_path, remote)
    assert not raw_dir(tmp_path, entry()).exists()


def test_interrupted_run_resumes_and_keeps_completed_files(tmp_path: Path) -> None:
    remote = _remote()
    remote.interrupt_on = "data.csv"  # README.md is fetched first (sorted), then the crash
    with pytest.raises(Interrupted):
        _acquire(tmp_path, remote)
    final = raw_dir(tmp_path, entry())
    staging = final.parent / f".partial-{entry().revision}"
    assert not final.exists()
    assert (staging / "README.md").is_file()
    assert (staging / "data.csv").stat().st_size == len(CSV) // 2

    resumed = _remote()
    _acquire(tmp_path, resumed)
    assert resumed.downloads == ["data.csv"]  # README.md was verified and kept
    assert verify_raw(final) == []
    assert not staging.exists()


def test_stale_staging_files_are_dropped(tmp_path: Path) -> None:
    staging = tmp_path / "raw" / "demo_src" / f".partial-{entry().revision}"
    staging.mkdir(parents=True)
    (staging / "old_unwanted.bin").write_bytes(b"junk")
    _acquire(tmp_path, _remote())
    assert not (raw_dir(tmp_path, entry()) / "old_unwanted.bin").exists()


def test_existing_corrupt_revision_is_never_overwritten(tmp_path: Path) -> None:
    _acquire(tmp_path, _remote())
    target = raw_dir(tmp_path, entry()) / "data.csv"
    target.write_bytes(b"tampered")
    remote = _remote()
    with pytest.raises(AcquireError, match="immutable"):
        _acquire(tmp_path, remote)
    assert target.read_bytes() == b"tampered"  # left for a human / restore from backup
    assert remote.downloads == []


def test_include_pattern_must_match(tmp_path: Path) -> None:
    files = [RemoteFile("a.csv", 1, sha256="0" * 64)]
    with pytest.raises(AcquireError, match="matches no file"):
        select_files(files, ["*.parquet"])


def test_policy_blocks_acquisition_before_any_network(tmp_path: Path) -> None:
    remote = _remote()
    e = entry(status="rejected", allowed_use="reference_only")
    with pytest.raises(PolicyError):
        acquire_source(e, tmp_path, remote)
    assert remote.list_calls == 0


def test_check_remote_file_reports_problems(tmp_path: Path) -> None:
    p = tmp_path / "x"
    rf = RemoteFile("x", 3, git_sha1="0" * 40)
    assert check_remote_file(p, rf) == "missing"
    p.write_bytes(b"ab")
    assert "size" in str(check_remote_file(p, rf))
    p.write_bytes(b"abc")
    assert "git blob sha1" in str(check_remote_file(p, rf))
    assert check_remote_file(p, RemoteFile("x", 3)) == "upstream provides no hash"
