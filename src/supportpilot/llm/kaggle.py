"""Create minimal private Kaggle experiment payloads, without credentials or test/gold."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from supportpilot.common.hashing import sha256_file
from supportpilot.common.io import read_json, write_json

REQUIREMENTS = (
    "transformers==5.20.2\npeft==0.21.3\ntrl==1.17.1\n"
    "lm-format-enforcer==0.11.3\nbitsandbytes==0.50.2\n"
    "polars==2.1.0\npydantic==2.14.3\npydantic-settings==2.16.2\n"
    "lingua-language-detector==2.2.0\nmlflow==3.18.0\npyyaml==6.0.3\n"
)

BOOTSTRAP = """import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

work = Path("/kaggle/working/source")
work.mkdir(parents=True, exist_ok=True)
payloads = list(Path("/kaggle/input").rglob("payload.zip"))
if payloads:
    with zipfile.ZipFile(payloads[0]) as archive:
        archive.extractall(work)
else:
    mounted = next(Path("/kaggle/input").rglob("payload_manifest.json"))
    allowlist = json.loads(mounted.read_text("utf-8"))
    for name in [*allowlist, "payload_manifest.json"]:
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        target = work / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(mounted.parent / relative, target)
manifest = json.loads((work / "payload_manifest.json").read_text("utf-8"))
for name, expected in manifest.items():
    assert hashlib.sha256((work / name).read_bytes()).hexdigest() == expected, name
subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                "-r", str(work / "requirements.txt")], check=True)
# The image's obsolete optional torchao breaks PEFT dispatch even for ordinary LoRA.
# These experiments use unquantized LoRA / bitsandbytes NF4, never torchao.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
sys.path.insert(0, str(work / "src"))
from supportpilot.common.io import read_json, write_json
from supportpilot.llm.experiments import execute_plan
output = Path("/kaggle/working/results")
execute_plan(read_json(work / "plan.json"), read_json(work / "inputs.json"), output)
with (output / "environment.txt").open("w") as environment:
    subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=environment, check=True)
write_json(output / "payload_manifest.json", manifest)
write_json(output / "checksums.json", {
    p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in output.rglob("*") if p.is_file() and p.name != "checksums.json"})
"""


def package(
    project: Path,
    inputs: dict[str, Any],
    plan: dict[str, Any],
    destination: Path,
    *,
    owner: str,
    slug: str,
    adapters: dict[str, Path] | None = None,
) -> Path:
    """Only an explicit file allowlist can enter the uploaded ZIP."""
    dataset = destination / "dataset"
    kernel = destination / "kernel"
    staging = destination / "source"
    for path in (dataset, kernel, staging):
        path.mkdir(parents=True, exist_ok=True)
    files: dict[str, bytes] = {
        p.relative_to(project).as_posix(): p.read_bytes()
        for p in (project / "src/supportpilot").rglob("*.py")
    }
    files.update(
        {
            "inputs.json": json.dumps(inputs, ensure_ascii=False).encode(),
            "plan.json": json.dumps(plan, ensure_ascii=False).encode(),
            "requirements.txt": REQUIREMENTS.encode(),
            "uv.lock": (project / "uv.lock").read_bytes(),
            "configs/taxonomy.yaml": (project / "configs/taxonomy.yaml").read_bytes(),
        }
    )
    for key, adapter in (adapters or {}).items():
        if Path(key).name != key or key in {".", ".."}:
            raise ValueError("unsafe adapter key")
        for file in adapter.rglob("*"):
            if file.is_symlink():
                raise ValueError("adapter upload does not follow symlinks")
            if file.is_file():
                relative = file.relative_to(adapter)
                if ".env" in relative.parts or file.suffix not in {
                    ".json",
                    ".safetensors",
                    ".txt",
                    ".model",
                    ".jinja",
                    ".md",
                }:
                    raise ValueError("unexpected file in adapter payload")
                files[f"adapters/{key}/{relative.as_posix()}"] = file.read_bytes()
    import hashlib

    manifest = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    submission = destination / "submission.json"
    if submission.exists():
        previous = read_json(submission)
        if previous["files"] != manifest:
            raise ValueError("existing payload is immutable; use a distinct destination")
        if sha256_file(dataset / "payload.zip") != previous["payload_sha256"]:
            raise ValueError("existing payload ZIP is corrupt")
        if (kernel / "run.py").read_text("utf-8") != BOOTSTRAP:
            raise ValueError("existing kernel source changed")
        return destination
    files["payload_manifest.json"] = json.dumps(manifest, sort_keys=True).encode()
    with zipfile.ZipFile(dataset / "payload.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(files.items()):
            archive.writestr(name, value)
    write_json(
        dataset / "dataset-metadata.json",
        {
            "title": slug,
            "id": f"{owner}/{slug}",
            "licenses": [{"name": "other"}],
            "description": (
                "Private reproducibility payload. Synthetic licensed SFT; source licences "
                "retained locally. No real customer PII or secrets. Not for redistribution."
            ),
        },
    )
    (kernel / "run.py").write_text(BOOTSTRAP, encoding="utf-8")
    write_json(
        kernel / "kernel-metadata.json",
        {
            "id": f"{owner}/{slug}-run",
            "title": slug + "-run",
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [f"{owner}/{slug}"],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        },
    )
    write_json(
        destination / "submission.json",
        {
            "payload_sha256": sha256_file(dataset / "payload.zip"),
            "files": manifest,
            "private": True,
            "plan": plan,
        },
    )
    return destination
