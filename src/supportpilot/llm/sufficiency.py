"""Source-traced generation probes from exposed Stage 6.1 DEVELOPMENT observations only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from supportpilot.common.hashing import sha256_file, sha256_text
from supportpilot.common.io import read_json, write_json
from supportpilot.common.schemas import Action, AssistantResponse, KBChunk


def build_probes(frozen: Path, development: Path, kb: Path, output: Path) -> list[dict[str, Any]]:
    metadata = read_json(frozen / "metadata.json")
    queries = {
        r["query_id"]: r
        for r in (
            json.loads(line) for line in (frozen / "queries.jsonl").read_text("utf-8").splitlines()
        )
    }
    observations = read_json(development / "observations.json")["bge_m3_weighted075_rerank20"]
    chunks = {
        r["chunk_id"]: KBChunk.model_validate_json(r["chunk_json"])
        for r in pl.read_parquet(kb / "chunks.parquet").to_dicts()
    }
    selected: list[tuple[str, dict[str, Any]]] = []
    for lang in ("en", "ru", "kk"):
        pool = [
            r
            for r in observations
            if queries[r["query_id"]]["lang"] == lang
            and metadata[r["query_id"]]["phase"] == "development"
        ]
        lost = [
            r for r in pool if r.get("full_evidence10") == 1 and r.get("packed_sufficiency") == 0
        ]
        multi = [
            r
            for r in pool
            if len(metadata[r["query_id"]]["required_groups"]) > 1
            and r.get("packed_sufficiency") == 1
        ]
        negative = [r for r in pool if not r["answerable"]]
        single = [r for r in pool if r.get("packed_sufficiency") == 1]
        for kind, candidates in (
            ("packing_lost", lost),
            ("multi_document", multi),
            ("unanswerable", negative),
            ("single", single),
        ):
            selected.extend((kind, r) for r in sorted(candidates, key=lambda r: r["query_id"])[:3])
    rows = []
    for kind, observation in selected:
        query = queries[observation["query_id"]]
        meta = metadata[query["query_id"]]
        if observation["phase"] != "development" or meta["phase"] != "development":
            raise ValueError("sufficiency development cannot consume retrieval holdout")
        groups = meta["required_groups"]
        modes = ["packed", "absent", "irrelevant"]
        if groups:
            modes += ["complete"]
            if len(groups) > 1:
                modes += ["partial"]
        for mode in modes:
            ids = list(observation["packed"])
            if mode == "complete":
                ids = list(dict.fromkeys([group[0] for group in groups] + ids))
            elif mode == "absent":
                ids = []
            elif mode == "irrelevant":
                support = {c for group in groups for c in group}
                ids = [c for c in ids if c not in support]
            elif mode == "partial":
                omit = set(groups[-1])
                ids = [
                    c for c in dict.fromkeys([group[0] for group in groups] + ids) if c not in omit
                ]
            sufficient = bool(groups) and all(set(ids) & set(group) for group in groups)
            reason = None if sufficient else "unverified_evidence"
            target = AssistantResponse(
                language=query["lang"],
                intent="product_info",
                action=Action.ANSWER_KB if sufficient else Action.ESCALATE,
                answer="Evaluation label; not supplied to the model.",
                needs_human=not sufficient,
                escalation_reason=reason,
            )
            passages = [
                {
                    "chunk_id": cid,
                    "text": chunks[cid].text,
                    "title": chunks[cid].title,
                    "effective_from": chunks[cid].effective_from.isoformat(),
                    "effective_to": str(chunks[cid].effective_to)
                    if chunks[cid].effective_to
                    else None,
                }
                for cid in ids
            ]
            rows.append(
                {
                    "example_id": sha256_text(query["query_id"] + "|" + mode)[:16],
                    "scenario_id": query["scenario_id"],
                    "family_id": meta["family_id"],
                    "lang": query["lang"],
                    "code_switch": query["code_switch"],
                    "user_text": query["query"],
                    "task": "complete" if sufficient else "partial",
                    "intervention": mode,
                    "source_kind": kind,
                    "target": target.model_dump(mode="json"),
                    "required_groups": groups,
                    "ranking": observation["ranking"],
                    "packed_ids": ids,
                    "retrieval_sufficient": observation.get("full_evidence10"),
                    "packed_sufficient": sufficient,
                    "diagnostic_only_labels": ["intent", "action"],
                    "messages": [
                        {"role": "system", "content": "Evidence sufficiency study."},
                        {"role": "user", "content": query["query"]},
                        {
                            "role": "tool",
                            "name": "kb_search",
                            "content": json.dumps({"results": passages}, ensure_ascii=False),
                        },
                        {"role": "assistant", "content": target.model_dump_json()},
                    ],
                }
            )
    # Equivalent needs may occur in two census categories; do not count repeated rows twice.
    rows = list({r["example_id"]: r for r in rows}.values())
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "probes.json", rows)
    write_json(
        output / "manifest.json",
        {
            "schema_version": "llm_sufficiency/1",
            "phase": "exposed_development_diagnostic",
            "rows": len(rows),
            "needs": len({r["scenario_id"] for r in rows}),
            "source_metadata_sha256": sha256_file(frozen / "metadata.json"),
            "source_observations_sha256": sha256_file(development / "observations.json"),
            "kb_sha256": sha256_file(kb / "chunks.parquet"),
            "review_status": "draft",
            "semantic_judgment": "NOT_EVALUATED",
            "intent_action_metrics": "not decision metrics; coarse probe labels only",
        },
    )
    return rows
