"""Predeclared validation selection and immutable final-evaluation seals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from supportpilot.common.hashing import sha256_file, sha256_text
from supportpilot.common.io import read_json, write_json
from supportpilot.llm.adapters import MANIFEST, validate_adapter
from supportpilot.llm.models import InferenceConfig


def choose(
    summaries: dict[str, dict[str, Any]],
    settings: dict[str, Any],
    *,
    control: str,
) -> dict[str, Any]:
    rules = settings["selection"]
    candidates = {}
    for key, summary in summaries.items():
        score = 0.0
        eligible = True
        for language, weight in rules["language_weights"].items():
            if not summary[language]["n"]:
                eligible = False
            for metric, metric_weight in rules["metric_weights"].items():
                value = summary[language]["metrics"][metric]["value"]
                if value is None:
                    eligible = False
                else:
                    score += weight * metric_weight * value
        for metric in ("schema_validity", "language_compliance"):
            value = summary["kk"]["metrics"][metric]["value"]
            baseline = summaries[control]["kk"]["metrics"][metric]["value"]
            if (
                value is None
                or baseline is None
                or value < baseline - rules["kk_guard_max_regression"]
            ):
                eligible = False
        candidates[key] = {"score": score, "eligible": eligible}
    eligible_keys = [key for key, item in candidates.items() if item["eligible"]]
    if not eligible_keys:
        raise ValueError("no candidate satisfies validation selection rules")

    def ranking(key: str) -> tuple[float, float, float, bool]:
        scopes = [summaries[key][lang] for lang in rules["language_weights"]]
        memory = [scope.get("resources", {}).get("peak_vram_bytes") for scope in scopes]
        latency = [scope.get("latency_p50") for scope in scopes]
        peak = max(memory) if all(v is not None for v in memory) else float("inf")
        warm = sum(latency) / len(latency) if all(v is not None for v in latency) else float("inf")
        return candidates[key]["score"], -peak, -warm, key == control

    selected = max(eligible_keys, key=ranking)
    return {
        "selected": selected,
        "candidates": candidates,
        "scope": "validation_only",
        "semantic_quality": "NOT_EVALUATED",
    }


def seal(
    path: Path,
    configurations: dict[str, InferenceConfig],
    *,
    prompt_modes: dict[str, str],
    decision: dict[str, Any],
    validation_reports: list[Path],
) -> dict[str, Any]:
    """No test/gold content may be opened until this immutable receipt exists."""
    payload = {
        "schema_version": "llm_selection/1",
        "configurations": {
            key: value.model_dump(mode="json") for key, value in configurations.items()
        },
        "prompt_modes": prompt_modes,
        "adapter_manifests": {
            key: sha256_file(value.adapter / MANIFEST)
            for key, value in configurations.items()
            if value.adapter is not None
        },
        "decision": decision,
        "validation_reports": {str(p): sha256_file(p) for p in validation_reports},
    }
    for config in configurations.values():
        if config.adapter is not None:
            validate_adapter(config.adapter, config.model)
    payload["identity"] = sha256_text(json.dumps(payload, sort_keys=True))
    if path.exists() and read_json(path) != payload:
        raise ValueError(
            "final selection is sealed; retuning requires a new independent evaluation"
        )
    write_json(path, payload)
    return payload


def require_seal(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = read_json(path)
    identity = payload.pop("identity")
    if sha256_text(json.dumps(payload, sort_keys=True)) != identity:
        raise ValueError("selection seal checksum mismatch")
    for name, digest in payload["validation_reports"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError("validation evidence changed after selection")
    for key, digest in payload["adapter_manifests"].items():
        config = InferenceConfig.model_validate(payload["configurations"][key])
        if config.adapter is None or sha256_file(config.adapter / MANIFEST) != digest:
            raise ValueError("adapter changed after selection")
        validate_adapter(config.adapter, config.model)
    payload["identity"] = identity
    return payload
