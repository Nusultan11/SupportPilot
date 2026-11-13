"""Open test/gold only after an immutable validation selection receipt exists."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from supportpilot.common.config import Settings
from supportpilot.common.hashing import sha256_file, sha256_text
from supportpilot.common.io import read_json, write_json
from supportpilot.common.schemas import Action, AssistantResponse, DatasetManifest, GoldItem
from supportpilot.llm.data import assert_disjoint, load_sft
from supportpilot.llm.selection import require_seal
from supportpilot.retrieval.search import assemble_context, open_retriever


def export_heldout(
    selection: Path,
    sft: Path,
    gold: Path,
    settings: Settings,
    output: Path,
    *,
    train: list[dict[str, Any]],
) -> dict[str, Any]:
    sealed = require_seal(selection)
    if output.exists():
        manifest = read_json(output / "manifest.json")
        if manifest["selection_identity"] != sealed["identity"]:
            raise ValueError("held-out export is bound to a different selection")
        if sha256_file(output / "inputs.json") != manifest["inputs_sha256"]:
            raise ValueError("held-out export checksum mismatch")
        restored: dict[str, Any] = read_json(output / "inputs.json")
        return restored
    test = load_sft(sft, "test")
    assert_disjoint(train, test)
    manifest = DatasetManifest.model_validate(read_json(gold / "manifest.json"))
    entry = next(entry for entry in manifest.files if entry.path == "gold.jsonl")
    if sha256_file(gold / "gold.jsonl") != entry.sha256:
        raise ValueError("gold checksum mismatch")
    items = [
        GoldItem.model_validate_json(line)
        for line in (gold / "gold.jsonl").read_text("utf-8").splitlines()
    ]
    rows = []
    with open_retriever(settings) as retriever:
        # The map belongs to exactly the active immutable KB, never to benchmark targets.
        kb_root = (
            settings.resolve(settings.paths.data_dir)
            / "kb"
            / settings.kb.name
            / "versions"
            / retriever.artifact.kb_version
        )
        evidence_map = read_json(kb_root / "evidence_map.json")
        for item in items:
            query = item.turns[-1].content
            ranking = retriever.retrieve(query, lang=item.lang)
            context = assemble_context(
                ranking,
                retriever.tokenizer,
                settings.retrieval.context_tokens,
                kb_version=retriever.artifact.kb_version,
                index_version=retriever.artifact.manifest.dataset_version,
            )
            groups = [
                evidence_map[f"{e.doc_id}.{e.section}"][item.lang] for e in item.accepted_evidence
            ]
            ranked = {r.chunk_id for r in ranking}
            packed = {r.chunk_id for r in context.evidence}
            target = AssistantResponse(
                language=item.lang,
                intent=item.intent,
                action=item.action,
                answer="Final evaluation label only; never supplied to the model.",
                needs_human=item.escalate,
                escalation_reason=item.escalation_reason,
                clarification_question="?" if item.action is Action.CLARIFY else None,
            )
            passages = [
                {
                    "chunk_id": r.chunk_id,
                    "title": r.chunk.title,
                    "text": r.chunk.text,
                    "effective_from": r.chunk.effective_from.isoformat(),
                    "effective_to": str(r.chunk.effective_to) if r.chunk.effective_to else None,
                }
                for r in context.evidence
            ]
            messages = [
                {"role": "system", "content": "Grounded store support evaluation."},
                *[turn.model_dump(mode="json") for turn in item.turns],
                {
                    "role": "tool",
                    "name": "kb_search",
                    "content": json.dumps({"results": passages}, ensure_ascii=False),
                },
            ]
            if item.tool_state is not None:
                messages.append(
                    {
                        "role": "tool",
                        "name": "gold_tool_fixture",
                        "content": json.dumps(
                            {
                                "call": {"name": "gold_tool_fixture", "arguments": {}},
                                "result": item.tool_state,
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
            messages.append({"role": "assistant", "content": target.model_dump_json()})
            rows.append(
                {
                    "example_id": item.gold_id,
                    "scenario_id": item.scenario_id,
                    "family_id": "gold_category_" + item.category,
                    "lang": item.lang,
                    "code_switch": item.code_switch,
                    "task": item.category,
                    "split": "gold",
                    "user_text": query,
                    "target": target.model_dump(mode="json"),
                    "messages": messages,
                    "required_facts": [f.model_dump() for f in item.required_facts],
                    "required_groups": groups,
                    "ranking": [r.chunk_id for r in ranking],
                    "packed_ids": list(packed),
                    "retrieval_sufficient": all(ranked & set(g) for g in groups)
                    if groups
                    else None,
                    "packed_sufficient": all(packed & set(g) for g in groups) if groups else None,
                    "review_status": item.review.status.value,
                    "semantic_near_train": "semantic_near_train" in item.review.notes,
                    "tool_scope": "conditional synthetic fixture"
                    if item.tool_state is not None
                    else "none",
                }
            )
    output.mkdir(parents=True)
    payload = {"test": test, "gold": rows, "selection_identity": sealed["identity"]}
    write_json(output / "inputs.json", payload)
    write_json(
        output / "manifest.json",
        {
            "schema_version": "llm_heldout/1",
            "selection_identity": sealed["identity"],
            "inputs_sha256": sha256_file(output / "inputs.json"),
            "gold_manifest_sha256": sha256_file(gold / "manifest.json"),
            "counts": {"test": len(test), "gold": len(rows)},
            "gold_family_scope": (
                "coarse category sensitivity; not independently reviewed paraphrase families"
            ),
            "export_identity": sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False)),
        },
    )
    return payload
