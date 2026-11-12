"""Gold benchmark build (Stage 4, EVALUATION §1).

Source: `authored/gold/*.yaml`, written separately from the training scenarios (new
situations and phrasings, typos, code-switching, multi-turn, adversarial items). One source
item is one scenario, written natively in en / ru / kk, so each language gets the same
scenarios (comparable per-language results). Expected facts refer to the demo-store KB
(kb_sources/demo_store/facts.yaml), never to the training KB.

Review status (honest by construction): every item is `draft` = authored by the Stage 4
agent (an LLM) and checked automatically. No independent fluent human reviewer was
available, so nothing is `human_reviewed` or `verified`, and an LLM re-read is not
reported as independent verification.

Output `data/gold/<benchmark_version>/gold.jsonl` + manifest + quality_report.json, after
the leakage check against every SFT split.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field

from supportpilot.common.config import Settings
from supportpilot.common.hashing import sha256_text
from supportpilot.common.io import write_json
from supportpilot.common.schemas import LANGUAGES, GoldItem
from supportpilot.data.langid import LanguageIdentifier
from supportpilot.data.split import exact_leaks, lexical_leaks, ngram_leaks, semantic_leaks
from supportpilot.data.steps import StepError, commit, content_hash, partial_dir

GOLD_VERSION = 1
AUTHOR = "stage4-agent (Claude, LLM)"
_DIGITS = re.compile(r"\d+")
# Fixed facts stated in the KB templates themselves (not variant values).
FIXED_FACTS = frozenset({"sms_retry_seconds", "cart_days", "payment_retry_minutes"})


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GoldSource(_Model):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    cat: str
    diff: Literal["easy", "medium", "hard"]
    intent: str
    action: str
    esc: str | None = None
    facts: dict[str, str] = Field(default_factory=dict)
    ev: list[str] = Field(default_factory=list)
    tool: dict[str, Any] | None = None
    must_not: list[str] = Field(default_factory=list)
    cs: list[str] = Field(default_factory=list)  # languages whose text is RU-KK code-switched
    en: str | list[str]
    ru: str | list[str]
    kk: str | list[str]


class GoldFile(_Model):
    schema_version: Literal["gold_src/1"] = "gold_src/1"
    items: list[GoldSource]


def load_gold_sources(directory: Path) -> list[GoldSource]:
    items: list[GoldSource] = []
    for p in sorted(directory.glob("*.yaml")):
        items.extend(GoldFile.model_validate(yaml.safe_load(p.read_text("utf-8"))).items)
    ids = [i.id for i in items]
    if dupes := sorted({i for i in ids if ids.count(i) > 1}):
        raise StepError(f"duplicate gold ids: {dupes}")
    return items


def to_items(sources: list[GoldSource]) -> list[GoldItem]:
    out: list[GoldItem] = []
    for s in sources:
        scenario = "scn_" + sha256_text(f"gold|{s.id}")[:16]
        for lang in LANGUAGES:
            raw = getattr(s, lang)
            texts = [raw] if isinstance(raw, str) else list(raw)
            if len(texts) % 2 == 0:
                raise StepError(
                    f"{s.id}/{lang}: turns must alternate user/assistant and end with user"
                )
            turns = [
                {"role": "user" if i % 2 == 0 else "assistant", "content": t}
                for i, t in enumerate(texts)
            ]
            category = s.cat
            out.append(
                GoldItem.model_validate(
                    {
                        "gold_id": f"g_{s.id}_{lang}",
                        "scenario_id": scenario,
                        "lang": lang,
                        "code_switch": lang in s.cs,
                        "category": category,
                        "difficulty": s.diff,
                        "turns": turns,
                        "intent": s.intent,
                        "action": s.action,
                        "escalate": s.esc is not None,
                        "escalation_reason": s.esc,
                        "required_facts": [{"key": k, "value": v} for k, v in s.facts.items()],
                        "tool_state": s.tool,
                        "accepted_evidence": [
                            {"doc_id": e.split(".")[0], "section": e.split(".", 1)[1]} for e in s.ev
                        ],
                        "must_not": s.must_not,
                        "review": {
                            "status": "draft",
                            "author": AUTHOR,
                            "reviewers": [],
                            "notes": "auto-checked (schema, facts vs demo KB, language, leakage); "
                            "independent fluent human review pending",
                        },
                    }
                )
            )
    return out


def _demo_value_ok(key: str, value: str, demo: dict[str, Any]) -> bool | None:
    """True/False when the fact is a demo-store key; None for tool/other facts."""
    if key not in demo:
        return None
    v = demo[key]
    if isinstance(v, list):
        return value in {str(x) for x in v} or value == "-".join(map(str, v))
    if isinstance(v, dict):
        return value in {str(x) for x in v.values()}
    return str(v) == value


def check_items(
    items: list[GoldItem], demo: dict[str, Any], docs_sections: set[str]
) -> dict[str, Any]:
    problems: Counter[str] = Counter()
    examples: list[str] = []
    lid = LanguageIdentifier()
    last_user = [next(t.content for t in reversed(i.turns) if t.role == "user") for i in items]
    res = lid.detect_many(last_user)
    for item, r in zip(items, res, strict=True):
        for f in item.required_facts:
            ok = _demo_value_ok(f.key, f.value, demo)
            if ok is False:
                problems["fact_not_demo_value"] += 1
                examples.append(f"{item.gold_id}:{f.key}={f.value}")
            in_tool = bool(item.tool_state) and f.value in json.dumps(item.tool_state)
            if ok is None and not in_tool and f.key not in FIXED_FACTS:
                problems["fact_without_source"] += 1
                examples.append(f"{item.gold_id}:{f.key}")
        for e in item.accepted_evidence:
            if f"{e.doc_id}.{e.section}" not in docs_sections:
                problems["unknown_evidence_section"] += 1
                examples.append(f"{item.gold_id}:{e.doc_id}.{e.section}")
        if item.category in ("grounded",) and not item.accepted_evidence:
            problems["grounded_without_evidence"] += 1
        if item.category == "tool" and not item.tool_state:
            problems["tool_without_state"] += 1
        if not (r.lang == item.lang or item.code_switch or r.code_switch or r.lang == "und"):
            problems[f"lid_mismatch:{item.lang}->{r.lang}"] += 1
            examples.append(item.gold_id)
    return {"problems": dict(problems), "examples": examples[:30]}


def gold_dir(data_dir: Path) -> Path:
    return data_dir / "gold"


def build_gold(
    settings: Settings, *, code_commit: str | None, semantic: bool = True
) -> dict[str, Any]:
    from supportpilot.data.sftbuild import current_sft

    cfg = settings.stage4
    data_dir = settings.resolve(settings.paths.data_dir)
    authored = settings.resolve(cfg.authored_dir)
    sources = load_gold_sources(authored / "gold")
    items = to_items(sources)
    demo = yaml.safe_load(settings.resolve(cfg.demo_facts).read_text(encoding="utf-8"))
    docs = yaml.safe_load((authored / "kb_train" / "docs.yaml").read_text(encoding="utf-8"))
    sections = {f"{d}.{s['id']}" for d, doc in docs["docs"].items() for s in doc["sections"]}
    checks = check_items(items, demo, sections)

    sft = current_sft(data_dir)
    if sft is None:
        raise StepError(
            "build the SFT dataset first (`supportpilot data sft`): gold is checked against it"
        )
    train = pl.concat(
        [pl.read_parquet(sft / f"{s}.parquet") for s in ("train", "validation", "test")]
    )
    train_texts = sorted(set(train["user_text"].to_list()))
    train_ids = set(train["scenario_id"].to_list()) | set(train["family_id"].to_list())
    gold_texts = [" ".join(t.content for t in i.turns if t.role == "user") for i in items]
    last = [next(t.content for t in reversed(i.turns) if t.role == "user") for i in items]
    semantic_flagged: list[str] = []
    leaks: dict[str, list[str]] = {
        "scenario_id": [i.gold_id for i in items if i.scenario_id in train_ids],
        "exact": [items[k].gold_id for k in exact_leaks(train_texts, last)],
        "lexical": [
            items[k].gold_id
            for k in lexical_leaks(train_texts, last, threshold=cfg.leak_jaccard, seed=cfg.seed)
        ],
        "ngram": [items[k].gold_id for k in ngram_leaks(train_texts, gold_texts, cfg.leak_ngram)],
    }
    if semantic:
        from supportpilot.data.embed import Embedder, EmbedderSpec

        emb = Embedder(EmbedderSpec.model_validate(cfg.embedder))
        tv = emb.encode([_DIGITS.sub("0", t) for t in train_texts])
        gv = emb.encode([_DIGITS.sub("0", t) for t in last])
        near = semantic_leaks(tv, gv, cfg.leak_cosine)
        # Semantic closeness of a short canonical request ("status of order 99812") to the
        # training intent distribution is not a copied item: it is flagged on the item and
        # reported, while copies (ids, exact, lexical, 13-gram) block the build (ADR-026).
        for k, cos in near.items():
            item = items[k]
            items[k] = item.model_copy(
                update={
                    "review": item.review.model_copy(
                        update={"notes": f"{item.review.notes}; semantic_near_train cos={cos:.3f}"}
                    )
                }
            )
        semantic_flagged = [items[k].gold_id for k in near]
    blocking = sum(len(v) for v in leaks.values())
    per_lang = Counter(i.lang for i in items)
    report = {
        "passed": not checks["problems"] and blocking == 0 and min(per_lang.values()) >= 300,
        "items": len(items),
        "scenarios": len(sources),
        "per_lang": dict(per_lang),
        "per_category": dict(Counter(i.category for i in items if i.lang == "en")),
        "per_difficulty": dict(Counter(i.difficulty for i in items if i.lang == "en")),
        "multi_turn": sum(len(i.turns) > 1 for i in items),
        "code_switch": sum(i.code_switch for i in items),
        "review_status": dict(Counter(i.review.status.value for i in items)),
        "checks": checks,
        "leakage": {k: len(v) for k, v in leaks.items()},
        "leakage_items": {k: v[:20] for k, v in leaks.items() if v},
        "semantic_near_train": {
            "checked": semantic,
            "threshold": cfg.leak_cosine,
            "items": len(semantic_flagged),
            "per_lang": dict(Counter(g.rsplit("_", 1)[1] for g in semantic_flagged)),
            "per_category": dict(
                Counter(i.category for i in items if i.gold_id in set(semantic_flagged))
            ),
            "ids": semantic_flagged,
        },
        "sft_dataset": sft.name,
    }
    rows = [i.model_dump(mode="json") for i in items]
    src_hash = sha256_text(json.dumps([s.model_dump() for s in sources], ensure_ascii=False))[:16]
    out = gold_dir(data_dir) / "current"
    with partial_dir(out) as partial:
        with (partial / "gold.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        write_json(partial / "quality_report.json", report)
        m = commit(
            partial,
            out,
            name="gold",
            data_schema_version="gold/1",
            code_commit=code_commit,
            inputs=[
                f"authored/gold@{src_hash}",
                f"sft@{sft.name}",
                "kb_sources/demo_store/facts.yaml",
            ],
            params={
                "gold_version": GOLD_VERSION,
                "leak": [cfg.leak_jaccard, cfg.leak_ngram, cfg.leak_cosine],
                "semantic": semantic,
            },
            row_counts={f"lang={k}": v for k, v in per_lang.items()} | {"items": len(items)},
            content_sha256=content_hash(rows),
            quality_report="quality_report.json",
            gate=lambda _p: bool(report["passed"]),
        )
    final = gold_dir(data_dir) / m.dataset_version
    if final.exists():
        import shutil

        shutil.rmtree(final)
    out.rename(final)
    return {"path": str(final), "benchmark_version": m.dataset_version, **report}
