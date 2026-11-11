"""Build, write, read and verify file manifests (DATA_CONTRACTS §3, §9)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from supportpilot.common.hashing import sha256_file
from supportpilot.common.io import atomic_write_text
from supportpilot.common.schemas import FileEntry, SourceRegistry


class FileProblem(BaseModel):
    path: str
    problem: Literal["missing", "size", "sha256", "unexpected", "symlink"]
    detail: str = ""


def iter_files(root: Path, exclude: frozenset[str] = frozenset()) -> list[Path]:
    """Regular files under `root` (sorted, POSIX order). Symlinks are rejected, not followed."""
    files: list[Path] = []
    for p in sorted(root.rglob("*"), key=lambda q: q.relative_to(root).as_posix()):
        rel = p.relative_to(root).as_posix()
        if rel in exclude:
            continue
        if p.is_symlink():
            raise ValueError(f"symlinks are not supported in manifests: {rel}")
        if p.is_file():
            files.append(p)
    return files


def scan_files(root: Path, exclude: frozenset[str] = frozenset()) -> list[FileEntry]:
    return [
        FileEntry(
            path=p.relative_to(root).as_posix(), bytes=p.stat().st_size, sha256=sha256_file(p)
        )
        for p in iter_files(root, exclude)
    ]


def verify_files(
    root: Path, expected: list[FileEntry], exclude: frozenset[str] = frozenset()
) -> list[FileProblem]:
    """Compare files on disk with a manifest. Empty list means identical."""
    problems: list[FileProblem] = []
    expected_paths = {e.path for e in expected}
    for entry in expected:
        p = root / entry.path
        if not p.is_file():
            problems.append(FileProblem(path=entry.path, problem="missing"))
            continue
        size = p.stat().st_size
        if size != entry.bytes:
            problems.append(
                FileProblem(path=entry.path, problem="size", detail=f"{size} != {entry.bytes}")
            )
            continue
        if sha256_file(p) != entry.sha256:
            problems.append(FileProblem(path=entry.path, problem="sha256"))
    for p in iter_files(root, exclude):
        rel = p.relative_to(root).as_posix()
        if rel not in expected_paths:
            problems.append(FileProblem(path=rel, problem="unexpected"))
    return problems


def write_model_json(path: Path, model: BaseModel) -> None:
    atomic_write_text(path, model.model_dump_json(indent=2) + "\n")


def read_model_json[M: BaseModel](path: Path, model_cls: type[M]) -> M:
    return model_cls.model_validate_json(path.read_text(encoding="utf-8"))


def load_source_registry(path: Path) -> SourceRegistry:
    """Load `configs/sources.yaml` (DATA_CONTRACTS §2)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return SourceRegistry.model_validate(data)
