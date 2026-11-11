"""Full create → verify → restore cycle and the safety rules around it."""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from supportpilot.common.manifest import scan_files, verify_files
from supportpilot.ops import backup
from supportpilot.ops.backup import BackupError


@pytest.fixture
def project(tmp_path: Path) -> Path:
    p = tmp_path / "repo"
    p.mkdir()
    return p


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "backups"


@pytest.fixture
def source(tmp_path: Path) -> Path:
    s = tmp_path / "dataset"
    (s / "lang=kk").mkdir(parents=True)
    (s / "lang=kk" / "қайтару.txt").write_text("Қайтару 14 күн ішінде.\n", encoding="utf-8")
    (s / "train.bin").write_bytes(bytes(range(256)) * 9000)  # ~2.3 MB, > one read chunk
    (s / "empty.txt").write_bytes(b"")
    return s


def _create(source: Path, root: Path, project: Path, **kw: object) -> backup.SnapshotManifest:
    m, created = backup.create_snapshot(
        source,
        name="sft_sample",
        kind="data",
        root=root,
        project_root=project,
        **kw,  # type: ignore[arg-type]
    )
    assert created
    return m


def _dir(root: Path, m: backup.SnapshotManifest) -> Path:
    return root / m.kind / m.name / m.snapshot_id


def test_full_round_trip(source: Path, root: Path, project: Path, tmp_path: Path) -> None:
    m = _create(source, root, project, code_commit="abc", note="test")
    snap = _dir(root, m)
    assert sorted(p.name for p in snap.iterdir()) == ["archive.tar", "manifest.json"]
    assert {f.path for f in m.files} == {"empty.txt", "lang=kk/қайтару.txt", "train.bin"}
    assert m.total_bytes == sum(f.bytes for f in scan_files(source))

    assert backup.verify_snapshot(snap, deep=True).ok
    assert backup.verify_snapshot(snap, deep=False).ok

    target = tmp_path / "restored"
    backup.restore_snapshot(snap, target)
    assert verify_files(target, scan_files(source)) == []
    assert (target / "lang=kk" / "қайтару.txt").read_text(encoding="utf-8").startswith("Қайтару")


def test_single_file_source(tmp_path: Path, root: Path, project: Path) -> None:
    f = tmp_path / "adapter.safetensors"
    f.write_bytes(b"weights" * 1000)
    m, _ = backup.create_snapshot(f, name="lora", kind="model", root=root, project_root=project)
    assert m.source_is_file
    assert [e.path for e in m.files] == ["adapter.safetensors"]
    out = tmp_path / "out"
    backup.restore_snapshot(_dir(root, m), out)
    assert (out / "adapter.safetensors").read_bytes() == f.read_bytes()


def test_unchanged_content_is_not_duplicated(source: Path, root: Path, project: Path) -> None:
    first = _create(source, root, project)
    again, created = backup.create_snapshot(
        source, name="sft_sample", kind="data", root=root, project_root=project
    )
    assert not created
    assert again.snapshot_id == first.snapshot_id
    (source / "new.txt").write_text("x", encoding="utf-8")
    _, created = backup.create_snapshot(
        source, name="sft_sample", kind="data", root=root, project_root=project
    )
    assert created
    assert len(backup.list_snapshots(root, "data", "sft_sample")) == 2
    assert not any(p.name.startswith(".partial") for p in (root / "data" / "sft_sample").iterdir())


def test_restore_refuses_non_empty_target(
    source: Path, root: Path, project: Path, tmp_path: Path
) -> None:
    snap = _dir(root, _create(source, root, project))
    target = tmp_path / "busy"
    target.mkdir()
    (target / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(BackupError, match="not an empty directory"):
        backup.restore_snapshot(snap, target)
    assert (target / "keep.txt").read_text(encoding="utf-8") == "user data"
    empty = tmp_path / "empty"
    empty.mkdir()
    backup.restore_snapshot(snap, empty)  # an empty dir is fine


def test_corrupted_archive_detected_and_not_restored(
    source: Path, root: Path, project: Path, tmp_path: Path
) -> None:
    snap = _dir(root, _create(source, root, project))
    archive = snap / "archive.tar"
    data = bytearray(archive.read_bytes())
    data[1024] ^= 0xFF
    archive.write_bytes(bytes(data))
    result = backup.verify_snapshot(snap)
    assert not result.ok
    assert result.problems[0].problem == "sha256"
    with pytest.raises(BackupError, match="integrity"):
        backup.restore_snapshot(snap, tmp_path / "never")
    assert not (tmp_path / "never").exists()


def test_deep_verify_catches_manifest_member_mismatch(
    source: Path, root: Path, project: Path
) -> None:
    snap = _dir(root, _create(source, root, project))
    raw = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
    raw["files"][0]["sha256"] = "0" * 64
    (snap / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    assert backup.verify_snapshot(snap, deep=False).ok  # archive itself untouched
    deep = backup.verify_snapshot(snap, deep=True)
    assert not deep.ok
    assert deep.problems[0].problem == "sha256"


def test_archive_has_no_absolute_or_parent_paths(source: Path, root: Path, project: Path) -> None:
    snap = _dir(root, _create(source, root, project))
    with tarfile.open(snap / "archive.tar") as tar:
        names = tar.getnames()
    assert all(not n.startswith(("/", "..")) and ".." not in n.split("/") for n in names)


def _git(project: Path, *args: str) -> None:
    git = shutil.which("git")
    assert git is not None
    subprocess.run([git, "-C", str(project), *args], check=True, capture_output=True)


def test_backup_root_inside_repo_needs_gitignore(source: Path, project: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    inside = project / "backups"
    with pytest.raises(BackupError, match="not git-ignored"):  # not even a git repo
        backup.create_snapshot(source, name="x", kind="data", root=inside, project_root=project)
    _git(project, "init", "-q")
    with pytest.raises(BackupError, match="not git-ignored"):  # repo, but not ignored
        backup.create_snapshot(source, name="x", kind="data", root=inside, project_root=project)
    (project / ".gitignore").write_text("/backups/\n", encoding="utf-8")
    m, created = backup.create_snapshot(
        source, name="x", kind="data", root=inside, project_root=project
    )
    assert created
    assert (inside / "data" / "x" / m.snapshot_id / "manifest.json").is_file()
    with pytest.raises(BackupError, match="repository root itself"):
        backup.ensure_backup_root(project, project)


def test_invalid_inputs(source: Path, root: Path, project: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="snapshot name"):
        backup.create_snapshot(
            source, name="Bad Name", kind="data", root=root, project_root=project
        )
    with pytest.raises(BackupError, match="does not exist"):
        backup.create_snapshot(
            tmp_path / "missing", name="x", kind="data", root=root, project_root=project
        )
    with pytest.raises(BackupError, match="inside the source"):
        backup.create_snapshot(
            source, name="x", kind="data", root=source / "bk", project_root=project
        )
    with pytest.raises(BackupError, match="not found"):
        backup.resolve_snapshot(root, "data/x/nope")


def test_prune_keeps_newest_and_pinned(source: Path, root: Path, project: Path) -> None:
    ids = []
    for i in range(5):
        (source / "counter.txt").write_text(str(i), encoding="utf-8")
        ids.append(_create(source, root, project, pin=(i == 0)).snapshot_id)
    dry = backup.prune(root, keep=2, dry_run=True)
    assert len(dry) == 2  # ids[1], ids[2]; ids[0] is pinned
    assert len(backup.list_snapshots(root)) == 5  # dry run deleted nothing
    deleted = backup.prune(root, keep=2, dry_run=False)
    assert {d.name for d in deleted} == {ids[1], ids[2]}
    assert {m.snapshot_id for m in backup.list_snapshots(root)} == {ids[0], ids[3], ids[4]}
    backup.set_pinned(root / "data" / "sft_sample" / ids[0], False)
    assert {d.name for d in backup.prune(root, keep=2, dry_run=False)} == {ids[0]}
    with pytest.raises(BackupError):
        backup.prune(root, keep=0)
