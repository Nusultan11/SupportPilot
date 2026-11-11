"""Qualitative review: deterministic samples per source and the reviewer's annotations.

Per source: `review_random` uniformly random records plus up to `review_flagged` records
with a problem signal (clean rejects, pending conflicts, defect flags, label-blocked
near-duplicates). Flagged rows are drawn round-robin over signal types, rarest first, so
rare defects are always included; if fewer exist, all are taken.

Annotations live in `docs/reviews/stage3_review.yaml` (hand-written, committed). They are
reviews by an LLM agent and are reported as such, never as independent human review.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field

from supportpilot.common.config import Settings
from supportpilot.common.io import atomic_write_text, write_json
from supportpilot.common.schemas import SourceEntry

# Flags that indicate a possible defect (style flags such as `offensive` are also sampled:
# whether they hurt is exactly what the review decides).
PROBLEM_FLAGS = (
    "garbled_turn",
    "replacement_char",
    "mojibake",
    "html_tag",
    "brace_artifact",
    "lid_disagrees",
    "lid_undetermined",
    "lang_relabelled",
    "lid_low_conf",
    "code_switch",
    "starts_with_assistant",
    "no_assistant_turn",
    "weak_first_turn",
    "response_missing",
    "response_template_instruction",
    "response_malformed_placeholder",
    "pii:email",
    "pii:phone",
    "pii:card",
    "url",
    "repeated_chars",
    "all_caps",
    "long_text",
    "short_text",
    "offensive",
)


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str  # R001.. (random) / F001.. (flagged)
    record_id: str
    codes: list[str] = Field(min_length=1)
    note: str | None = None


class SourceReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str
    verdict: str
    recommendation: str
    annotations: list[Annotation] = Field(default_factory=list)


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str
    reviewer: str
    reviewed_at: str
    method: str
    codes: dict[str, str]
    hard_codes: list[str] = Field(default_factory=list)
    sources: dict[str, SourceReview]
    conclusions: str = ""


def load_review(path: Path) -> Review | None:
    if not path.is_file():
        return None
    return Review.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _signals(source_id: str, d: Any) -> dict[str, list[str]]:
    """record_id lists per problem signal for one source."""
    cl = d.clean.filter(pl.col("source_id") == source_id)
    out: dict[str, list[str]] = {}
    for rid, reason in (
        cl.filter(pl.col("reject_reason").is_not_null())
        .select("record_id", "reject_reason")
        .iter_rows()
    ):
        out.setdefault(f"rejected:{reason}", []).append(rid)
    pen = d.pending.filter(
        (pl.col("source_id") == source_id) & (pl.col("reason") == "label_conflict")
    )
    out["label_conflict"] = pen["record_id"].to_list()
    groups = d.groups.filter(pl.col("source_id") == source_id)
    rid_of = dict(zip(groups["example_id"].to_list(), groups["record_id"].to_list(), strict=True))
    blocked = d.pairs.filter(
        pl.col("same_source") & ~pl.col("compatible") & (pl.col("jaccard") >= 0.8)
    )
    for a, b in zip(blocked["example_a"].to_list(), blocked["example_b"].to_list(), strict=True):
        if a in rid_of:
            out.setdefault("cross_label_near_dup", []).append(rid_of[a])
        if b in rid_of:
            out.setdefault("cross_label_near_dup", []).append(rid_of[b])
    flags = cl.select("record_id", "flags").iter_rows()
    for rid, fl in flags:
        for f in fl:
            if f in PROBLEM_FLAGS:
                out.setdefault(f, []).append(rid)
    if source_id == "support_json_ru":
        # same customer text, different decision (context-conditioned by policies/facts)
        multi = (
            cl.group_by("text")
            .agg(pl.col("record_id"), pl.col("context_json"))
            .filter(pl.col("record_id").list.len() > 1)
        )
        for rids, ctxs in multi.select("record_id", "context_json").iter_rows():
            actions = {json.loads(c).get("action") for c in ctxs}
            if len(actions) > 1:
                out.setdefault("same_text_other_action", []).extend(rids)
    return {k: sorted(set(v)) for k, v in out.items() if v}


def sample(
    source_id: str, d: Any, n_random: int, n_flagged: int, seed: int
) -> tuple[list[str], list[tuple[str, str]]]:
    rec_ids = d.records[source_id]["record_id"].to_list()
    rng = random.Random(f"{seed}|{source_id}|random")  # noqa: S311 - sampling, not security
    random_ids = rng.sample(rec_ids, min(n_random, len(rec_ids)))
    signals = _signals(source_id, d)
    rng2 = random.Random(f"{seed}|{source_id}|flagged")  # noqa: S311
    pools = {
        k: rng2.sample(v, len(v))
        for k, v in sorted(signals.items(), key=lambda kv: (len(kv[1]), kv[0]))
    }
    chosen: list[tuple[str, str]] = []
    seen = set(random_ids)
    while len(chosen) < n_flagged and any(pools.values()):
        for sig, pool in pools.items():
            while pool and pool[-1] in seen:
                pool.pop()
            if pool and len(chosen) < n_flagged:
                rid = pool.pop()
                seen.add(rid)
                chosen.append((rid, sig))
    return random_ids, chosen


def _cut(text: str | None, n: int) -> str:
    if not text:
        return ""
    t = text.replace("\n", " / ")
    return t if len(t) <= n else t[: n - 1] + "…"


_NOISY_FLAGS = frozenset(
    {"repaired", "response_kb_fact", "response_ui_label", "placeholder_filled"}
)


def render_entry(tag: str, row: dict[str, Any], signal: str | None) -> str:
    """Compact two-line entry: metadata, then the content to judge."""
    key = row["record_id"].split(":", 1)[1]
    flags = [f for f in row["flags"] if f not in _NOISY_FLAGS]
    head = [f"**{tag}** `{key}`", f"{row['source_intent'] or row.get('source_category') or '-'}"]
    head.append(f"{row['lang']}/{row['lang_detected']} {row['lid_conf']:.2f}")
    if row.get("source_tags"):
        head.append(row["source_tags"])
    head.append(f"→{row.get('reject_reason') or row.get('destination')}")
    if signal:
        head.append(f"[{signal}]")
    if flags:
        head.append(",".join(flags))
    ctx = json.loads(row["context_json"]) if row.get("context_json") else {}
    rtype = row["record_type"]
    body = f"T: {_cut(row['text'], 170)}"
    if rtype == "dialogue":
        turns = ctx.get("messages") or []
        rest = " ‖ ".join(f"{t['role'][0]}: {_cut(t['content'], 70)}" for t in turns[1:3])
        tgt = ctx.get("target") or {}
        why = _cut(tgt.get("reason"), 70)
        body += f" ‖ {rest} ‖ ({len(turns)} turns; {tgt.get('core_type')}: {why})"
    elif rtype == "policy_decision":
        facts = ", ".join(f"{f.get('key')}={f.get('value')}" for f in ctx.get("facts", []))
        pol = _cut(" | ".join(str(p.get("text")) for p in ctx.get("policies", [])), 90)
        decision = f"{ctx.get('action')}/{ctx.get('recommended_action')}"
        reply = _cut(row["reference_response"], 70)
        body += f" ‖ facts: {facts} ‖ policy: {pol} ‖ {decision}: {reply}"
    elif rtype == "reading_comprehension":
        opts = ctx.get("options") or []
        ans = str(ctx.get("correct_answer_num") or "")
        correct = opts[int(ans) - 1] if ans.isdigit() and 0 < int(ans) <= len(opts) else "?"
        body += f" ‖ P: {_cut(ctx.get('passage'), 110)} ‖ correct: {_cut(correct, 70)}"
    else:
        body += f" ‖ A: {_cut(row['reference_response'], 100)}"
    return " · ".join(head) + "  " + chr(10) + body


def write_review_samples(
    settings: Settings, entries: Sequence[SourceEntry], d: Any, out: Path
) -> list[str]:
    cfg = settings.eda
    out.mkdir(parents=True, exist_ok=True)
    dest = dict(
        zip(d.groups["record_id"].to_list(), d.groups["destination"].to_list(), strict=True)
    )
    written = []
    index: dict[str, Any] = {}
    for e in entries:
        sid = e.source_id
        random_ids, flagged = sample(sid, d, cfg.review_random, cfg.review_flagged, cfg.seed)
        cl = d.clean.filter(pl.col("source_id") == sid)
        by_id = {r["record_id"]: r for r in cl.iter_rows(named=True)}
        parts = [
            f"# Review sample: {sid}",
            "",
            f"Seed {cfg.seed}. {len(random_ids)} random records (R) and {len(flagged)} records "
            "with a problem signal (F). Text shown after cleaning (placeholders filled with "
            "synthetic values, PII masked). Annotations: docs/reviews/stage3_review.yaml.",
            "",
            "## Random sample",
        ]
        items = []
        for k, rid in enumerate(random_ids, 1):
            row = dict(by_id[rid], destination=dest.get(rid, "rejected"))
            parts.append(render_entry(f"R{k:03d}", row, None) + chr(10))
            items.append({"id": f"R{k:03d}", "record_id": rid})
        parts += ["", "## Flagged sample", ""]
        sig_counts = Counter(s for _, s in flagged)
        parts.append("Signals: " + ", ".join(f"{k} ({v})" for k, v in sorted(sig_counts.items())))
        for k, (rid, sig) in enumerate(flagged, 1):
            row = dict(by_id[rid], destination=dest.get(rid, "rejected"))
            parts.append(render_entry(f"F{k:03d}", row, sig) + chr(10))
            items.append({"id": f"F{k:03d}", "record_id": rid, "signal": sig})
        path = out / f"{sid}.md"
        atomic_write_text(path, "\n".join(parts) + "\n")
        index[sid] = {
            "random": len(random_ids),
            "flagged": len(flagged),
            "signals_available": {k: len(v) for k, v in _signals(sid, d).items()},
            "items": items,
        }
        written.append(str(path))
    write_json(out / "samples.json", index)
    return written
