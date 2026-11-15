"""Mechanical overlap audit; absence of lexical overlap never certifies independence."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import polars as pl

from supportpilot.common.config import Settings, load_settings
from supportpilot.common.hashing import sha256_file
from supportpilot.common.io import read_json, write_json
from supportpilot.service.evaluation import SCOPES, root, verify_freeze


def normalise(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def strings(value: Any) -> list[str]:
    """Extract user text only; expected answers/evidence must not become questions."""
    if isinstance(value, list):
        return [text for item in value for text in strings(item)]
    if isinstance(value, dict):
        if value.get("role") == "user" and isinstance(value.get("content"), str):
            return [value["content"]]
        return [
            text
            for key, item in value.items()
            if key not in {"target", "expected", "evidence", "context"}
            for text in strings(item)
        ]
    return []


def audit(settings: Settings, frozen: Path) -> Path:
    data = verify_freeze(frozen)
    files = [
        settings.root / "data/processed/sft/sft-20261112-b0388c64" / f"{split}.parquet"
        for split in ("train", "validation", "test")
    ]
    files.extend(sorted((settings.root / "data/benchmarks/llm_heldout").glob("**/inputs.json")))
    files.extend(sorted((settings.root / "data/benchmarks/llm").glob("*/inputs.json")))
    # Include earlier authored gold/safety and exposed Stage 8 linguistic fixtures as text.
    files.extend(sorted((settings.root / "authored/gold").glob("*.yaml")))
    files.extend(sorted((settings.root / "authored/safety").glob("*.yaml")))
    corpus: list[tuple[str, str]] = []
    for path in files:
        if path.suffix == ".parquet":
            texts = pl.read_parquet(path, columns=["user_text"])["user_text"].to_list()
        elif path.suffix == ".json":
            texts = strings(read_json(path))
        else:
            # A conservative lexical audit of YAML quoted lines, including non-query material.
            texts = re.findall(r'"([^"\n]{10,})"', path.read_text("utf-8"))
        corpus.extend(
            (str(path.relative_to(settings.root)), normalise(t))
            for t in texts
            if isinstance(t, str)
        )
    old_texts: list[tuple[str, str]] = []
    for path in [
        settings.root / "tests/integration/test_service.py",
        settings.root / "src/supportpilot/service/demo.py",
    ]:
        old_texts.extend(
            (str(path.relative_to(settings.root)), normalise(t))
            for t in re.findall(r'"([^"\n]{10,})"', path.read_text("utf-8"))
        )
        files.append(path)
    corpus.extend(old_texts)
    # Exact inverted shingle accounting: Jaccard >=.8 candidates need a shared shingle.
    grams = [{text[i : i + 5] for i in range(max(1, len(text) - 4))} for _, text in corpus]
    postings: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(grams):
        for gram in group:
            postings[gram].append(index)
    rows = []
    for need in data["needs"]:
        for scope, text in zip(SCOPES, need["texts"], strict=True):
            norm = normalise(text)
            query = {norm[i : i + 5] for i in range(max(1, len(norm) - 4))}
            common = Counter(index for gram in query for index in postings.get(gram, []))
            matches = []
            for index, intersection in common.items():
                score = intersection / len(query | grams[index])
                if score >= 0.8:
                    matches.append(
                        {
                            "source": corpus[index][0],
                            "jaccard": score,
                            "exact": corpus[index][1] == norm,
                        }
                    )
            rows.append(
                {
                    "need_id": need["need_id"],
                    "scope": scope,
                    "phase": need["phase"],
                    "matches": matches,
                }
            )
    target = root(settings) / "overlap_audit.json"
    write_json(
        target,
        {
            "freeze_identity": data["identity"],
            "sources": {str(p): sha256_file(p) for p in files},
            "reference_texts": len(corpus),
            "rows": rows,
            "flagged_rows": sum(bool(r["matches"]) for r in rows),
            "semantic_near_copy_audit": "NOT_EVALUATED",
            "limitations": (
                "lexical diagnostics only; YAML extraction is conservative; "
                "semantic decontamination unverified; no exclusions after freeze"
            ),
        },
    )
    return target


if __name__ == "__main__":
    settings = load_settings()
    frozen = Path(read_json(root(settings) / "freeze_pointer.json")["path"])
    print(json.dumps({"path": str(audit(settings, frozen))}))
