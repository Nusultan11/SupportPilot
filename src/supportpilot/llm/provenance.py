"""Runtime fingerprints bind checkpoint reuse to source and library versions."""

from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path

from supportpilot.common.hashing import sha256_text


def runtime_identity() -> str:
    source = Path(__file__).resolve().parents[1]
    files = {
        p.relative_to(source).as_posix(): sha256_text(p.read_text(encoding="utf-8"))
        for p in sorted(source.rglob("*.py"))
    }
    versions = {}
    for name in ("torch", "transformers", "peft", "trl", "lm-format-enforcer"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return sha256_text(
        json.dumps(
            {"source": files, "versions": versions, "python": platform.python_version()},
            sort_keys=True,
        )
    )
