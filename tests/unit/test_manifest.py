from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from supportpilot.common.manifest import read_model_json, scan_files, verify_files, write_model_json
from supportpilot.common.schemas import RawManifest


def _tree(root: Path) -> None:
    (root / "sub").mkdir(parents=True)
    (root / "a.csv").write_text("id,text\n1,Сәлем\n", encoding="utf-8")
    (root / "sub" / "b.bin").write_bytes(b"\x00\x01" * 100)


def test_scan_is_sorted_posix_and_verifies(tmp_path: Path) -> None:
    _tree(tmp_path)
    entries = scan_files(tmp_path)
    assert [e.path for e in entries] == ["a.csv", "sub/b.bin"]
    assert entries[1].bytes == 200
    assert verify_files(tmp_path, entries) == []


def test_verify_detects_each_problem(tmp_path: Path) -> None:
    _tree(tmp_path)
    entries = scan_files(tmp_path)
    (tmp_path / "a.csv").write_text("id,text\n1,Салем\n", encoding="utf-8")  # same size, new hash
    (tmp_path / "sub" / "b.bin").unlink()
    (tmp_path / "extra.txt").write_text("x", encoding="utf-8")
    problems = {(p.path, p.problem) for p in verify_files(tmp_path, entries)}
    assert problems == {("a.csv", "sha256"), ("sub/b.bin", "missing"), ("extra.txt", "unexpected")}


def test_verify_detects_size_change(tmp_path: Path) -> None:
    _tree(tmp_path)
    entries = scan_files(tmp_path)
    (tmp_path / "a.csv").write_text("changed and longer", encoding="utf-8")
    assert [p.problem for p in verify_files(tmp_path, entries)] == ["size"]


def test_manifest_json_roundtrip(tmp_path: Path) -> None:
    _tree(tmp_path / "raw")
    m = RawManifest(
        source_id="bitext_retail_ecom",
        revision="r1",
        url="https://example.org",
        retrieved_at=datetime.now(UTC),
        files=scan_files(tmp_path / "raw"),
    )
    path = tmp_path / "manifest.json"
    write_model_json(path, m)
    assert read_model_json(path, RawManifest) == m
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))  # atomic write leaves no temp files
