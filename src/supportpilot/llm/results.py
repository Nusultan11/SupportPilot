"""Verify exact remote payload identity before importing real results to MLflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from supportpilot.common.config import Settings
from supportpilot.common.hashing import sha256_file
from supportpilot.common.io import read_json, write_json
from supportpilot.common.tracking import RunSpec, start_run


class FailedTaskError(RuntimeError):
    """A retained failed experiment, not a failure of the import itself."""


def verify_results(output: Path, submission: Path) -> list[Path]:
    expected = read_json(submission)["files"]
    if read_json(output / "payload_manifest.json") != expected:
        raise ValueError("downloaded payload does not match submitted source/config/data")
    checksums = read_json(output / "checksums.json")
    actual = {
        p.relative_to(output).as_posix()
        for p in output.rglob("*")
        if p.is_file() and p.name != "checksums.json"
    }
    if actual != set(checksums):
        raise ValueError("remote output inventory mismatch")
    for name, digest in checksums.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe result path")
        if sha256_file(output / relative) != digest:
            raise ValueError("remote checksum mismatch: " + name)
    return sorted(output.rglob("attempt_*.json"))


def verify_kernel_source(local: Path, remote: Path) -> dict[str, str]:
    from supportpilot.common.hashing import sha256_text

    metadata = read_json(remote / "kernel-metadata.json")
    if metadata["is_private"] is not True:
        raise ValueError("experiment kernel was not private")
    submitted = (local / "run.py").read_text(encoding="utf-8").strip()
    downloaded = (remote / metadata["code_file"]).read_text(encoding="utf-8").strip()
    if submitted != downloaded:
        raise ValueError("downloaded kernel source differs from submission")
    return {
        "source_lf_sha256": sha256_text(submitted),
        "kernel": metadata["id"],
        "docker_image": metadata.get("docker_image", "UNKNOWN"),
    }


def import_results(
    output: Path,
    submission: Path,
    settings: Settings,
    *,
    kernel: str,
) -> dict[str, Any]:
    """Only call after explicit-version terminal Kaggle status has been checked."""
    import mlflow

    attempts = verify_results(output, submission)
    ledger = output.parent / "mlflow_import.json"
    imported = read_json(ledger) if ledger.exists() else {}
    for path in attempts:
        digest = sha256_file(path)
        if digest in imported:
            continue
        receipt = read_json(path)
        task = receipt["task"]
        cfg = task["config"]
        inference = cfg.get("inference", cfg)
        model = inference["model"]
        spec = RunSpec(
            experiment="llm_stage7",
            run_name=task["name"],
            hypothesis=task.get("hypothesis", "Measure multilingual structured support behavior"),
            dataset_version="sft-20261112-b0388c64",
            seed=inference["seed"],
            base_model=model["repo_id"] + "@" + model["revision"],
            control_run_id=task.get("control_run_id"),
            params={
                "kernel": kernel,
                "changed_params": str(task),
                "platform": "Kaggle",
                "kb_version": "training-variant-bound-by-payload",
                "cost": "free quota; allocated GPU time not exposed by CLI",
                "payload_identity": sha256_file(submission),
            },
        )
        run_id = ""
        try:
            with start_run(spec, settings) as run:
                run_id = run.info.run_id
                mlflow.log_dict(receipt, "receipt.json")
                mlflow.log_dict(read_json(submission), "submission.json")
                mlflow.log_metric("wall_seconds", receipt["elapsed_seconds"])
                mlflow.set_tag("semantic_quality", "NOT_EVALUATED")
                mlflow.set_tag("remote_environment", str(receipt.get("environment")))
                if receipt["status"] != "FINISHED":
                    mlflow.set_tag("conclusion", "failed attempt retained")
                    raise FailedTaskError(task["name"])
                result = receipt.get("result", {})
                for lang, summary in result.get("summary", {}).items():
                    for metric, item in summary["metrics"].items():
                        if item["value"] is not None:
                            mlflow.log_metric(lang + "." + metric, item["value"])
                    if summary.get("intent_macro_f1_35") is not None:
                        mlflow.log_metric(lang + ".intent_macro_f1", summary["intent_macro_f1_35"])
                for name in ("predictions.jsonl", "summary.json", "binding.json"):
                    if (path.parent / name).exists():
                        mlflow.log_artifact(str(path.parent / name))
                mlflow.set_tag("conclusion", "completed measurement; see per-language metrics")
                mlflow.set_tag("decision", "follow-up")
        except FailedTaskError:
            pass
        imported[digest] = {"run_id": run_id, "status": receipt["status"], "kernel": kernel}
        write_json(ledger, imported)
    return imported
