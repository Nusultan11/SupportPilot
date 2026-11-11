"""Guards against repository mistakes that tests of behaviour cannot see."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from supportpilot.common.config import find_project_root

ROOT = find_project_root(Path(__file__).parent)


def test_no_source_or_config_file_is_gitignored() -> None:
    """An unanchored pattern like `data/` once hid `src/supportpilot/data/` from git."""
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    paths = [
        p.relative_to(ROOT).as_posix()
        for base in ("src", "tests", "configs")
        for p in (ROOT / base).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    ]
    result = subprocess.run(
        [git, "-C", str(ROOT), "check-ignore", "--stdin"],
        input="\n".join(paths).encode("utf-8"),  # bytes: no CRLF translation on Windows
        capture_output=True,
        check=False,
    )
    ignored = result.stdout.decode("utf-8").split()
    assert ignored == [], f"ignored but required: {ignored}"
