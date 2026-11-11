"""Code provenance for manifests, experiment runs and snapshots."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _git(root: Path, *args: str) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell, git resolved via PATH
            [git, "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip()


def code_commit(root: Path) -> str | None:
    """Full HEAD sha, suffixed with `-dirty` if tracked files have uncommitted changes."""
    sha = _git(root, "rev-parse", "HEAD")
    if not sha:
        return None
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    return f"{sha}-dirty" if status else sha
