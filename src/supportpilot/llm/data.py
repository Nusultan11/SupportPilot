"""Verified SFT inputs; final-evaluation access is separate from development."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import polars as pl

from supportpilot.common.hashing import sha256_file, sha256_text
from supportpilot.common.io import read_json, write_json
from supportpilot.common.schemas import AssistantResponse, DatasetManifest


def load_sft(root: Path, split: Literal["train", "validation", "test"]) -> list[dict[str, Any]]:
    manifest = DatasetManifest.model_validate(read_json(root / "manifest.json"))
    name = f"{split}.parquet"
    entry = next(e for e in manifest.files if e.path == name)
    if sha256_file(root / name) != entry.sha256:
        raise ValueError(f"SFT checksum mismatch: {name}")
    rows = pl.read_parquet(root / name).to_dicts()
    if len(rows) != manifest.row_counts[split]:
        raise ValueError("SFT row count mismatch")
    for row in rows:
        if row["split"] != split:
            raise ValueError("wrong split in SFT file")
        row["messages"] = json.loads(row.pop("messages_json"))
        row["target"] = AssistantResponse.model_validate_json(row.pop("target_json")).model_dump(
            mode="json"
        )
        if row["messages"][-1]["role"] != "assistant":
            raise ValueError("missing SFT target")
        if json.loads(row["messages"][-1]["content"]) != row["target"]:
            raise ValueError("target/message disagreement")
    return rows


def assert_disjoint(train: list[dict[str, Any]], other: list[dict[str, Any]]) -> None:
    for key in ("example_id", "scenario_id", "family_id"):
        if {r[key] for r in train} & {r[key] for r in other}:
            raise ValueError("split leakage: " + key)


def demonstrations(train: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Fixed one complete short training example per language, no evaluation labels."""
    selected = []
    for lang in ("en", "ru", "kk"):
        pool = [r for r in train if r["lang"] == lang and r["task"] == "grounded_answer"]
        row = min(pool, key=lambda r: (len(json.dumps(r["messages"])), r["example_id"]))
        selected.append(row["messages"])
    return selected


def freeze_development(sft: Path, destination: Path) -> Path:
    train, validation = load_sft(sft, "train"), load_sft(sft, "validation")
    assert_disjoint(train, validation)
    # The test file and gold are deliberately never opened here.
    payload = {
        "dataset_version": sft.name,
        "train": train,
        "validation": validation,
        "demonstrations": demonstrations(train),
    }
    identity = sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))[:20]
    target = destination / identity
    if target.exists():
        if read_json(target / "inputs.json") != payload:
            raise ValueError("immutable development freeze mismatch")
        return target
    target.mkdir(parents=True)
    write_json(target / "inputs.json", payload)
    write_json(
        target / "manifest.json",
        {
            "schema_version": "llm_development/1",
            "identity": identity,
            "dataset_version": sft.name,
            "inputs_sha256": sha256_file(target / "inputs.json"),
            "source_manifest_sha256": sha256_file(sft / "manifest.json"),
            "counts": {"train": len(train), "validation": len(validation)},
            "review_status": "draft",
            "independent_fluent_review": False,
        },
    )
    return target
