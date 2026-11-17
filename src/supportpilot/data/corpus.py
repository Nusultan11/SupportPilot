"""Build the versioned Stage 1 corpus (ADR-047, DATA_CONTRACTS §6-9).

raw revisions (verified) -> normalise, mask PII -> label from the recipe -> exact dedup ->
near-duplicate families -> split groups -> leakage checks -> reports -> atomic publish.

Same raw inputs + recipe + taxonomy + registry + code => byte-identical data files and the
same `content_sha256` (only `manifest.json` carries the build time).
"""

from __future__ import annotations

import csv
import io
import json
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from supportpilot.common.hashing import sha256_file, sha256_text
from supportpilot.common.logging import get_logger
from supportpilot.common.manifest import (
    load_source_registry,
    read_model_json,
    scan_files,
    verify_files,
    write_model_json,
)
from supportpilot.common.schemas import Action, SourceRegistry
from supportpilot.data.acquire import MANIFEST, raw_dir, verify_raw
from supportpilot.data.dedup import UnionFind, near_duplicate_pairs
from supportpilot.data.policy import Purpose, check_use
from supportpilot.data.readers import INTENT_READERS, RawItem, doc_family, read_doc2dial
from supportpilot.data.recipe import (
    Recipe,
    SourceRecipe,
    compile_pattern,
    decide_label,
    derive_action,
)
from supportpilot.data.records import (
    NOT_EVALUATED,
    CorpusManifest,
    Evidence,
    Exclusion,
    GroundedExample,
    IntentExample,
    LabelStatus,
    Origin,
    Pool,
    Quality,
    Split,
    Turn,
)
from supportpilot.data.taxonomy import Taxonomy, load_taxonomy
from supportpilot.data.text import (
    content_words,
    dedup_key,
    detect_language,
    has_mojibake,
    mask_intro_name,
    mask_pii,
    mask_values,
    normalize_text,
    residual_pii,
)

log = get_logger(__name__)

INTENTS_DIR = "intents"
GROUNDED_DIR = "grounded"
REPORTS_DIR = "reports"
REVIEW_DIR = "review"
LANGS = ("en", "ru", "kk")
_NOT_HASHED = frozenset({MANIFEST, "REPORT.md"})


class CorpusError(Exception):
    """The corpus cannot be built or failed a gate; nothing was published."""


@dataclass
class _Row:
    """Mutable working row before it becomes an `IntentExample` or an `Exclusion`."""

    source_id: str
    revision: str
    item_id: str
    source_split: str | None
    source_label: str
    lang: str
    text: str
    group: str
    parallel_id: str | None
    origin: Origin
    domain: str
    words: int
    lang_guess: str
    code_switch: bool
    pii: dict[str, int]
    flags: list[str]
    intent: str | None = None
    status: LabelStatus | None = None
    rule: str = ""
    action: Action | None = None
    escalation_reason: str | None = None
    action_rule: str = ""
    reason: str = ""
    detail: str = ""
    duplicate_of: str | None = None
    key: str = field(default="")

    @property
    def example_id(self) -> str:
        return "ex-" + sha256_text(f"{self.source_id}|{self.item_id}")[:16]


def _unit(seed: int, key: str) -> float:
    return int(sha256_text(f"{seed}:{key}")[:15], 16) / 16**15


def assign_split(recipe: Recipe, group_id: str, heldout: bool) -> Split:
    """Deterministic group split; held-out groups go to validation/test only."""
    u = _unit(recipe.seed, group_id)
    if heldout:
        return Split.VALIDATION if u < 0.5 else Split.TEST
    if u < recipe.split.train:
        return Split.TRAIN
    if u < recipe.split.train + recipe.split.validation:
        return Split.VALIDATION
    return Split.TEST


def balanced_splits(recipe: Recipe, sizes: dict[str, int]) -> dict[str, Split]:
    """Size-aware deterministic split for few, large groups (e.g. documents).

    Groups are visited in seeded-hash order; each goes to the split furthest below its
    target share. Used where plain hashing of ~hundreds of groups skews the ratios."""
    target = {
        Split.TRAIN: recipe.split.train,
        Split.VALIDATION: recipe.split.validation,
        Split.TEST: recipe.split.test,
    }
    filled = dict.fromkeys(target, 0)
    total = 0
    out: dict[str, Split] = {}
    for group in sorted(sizes, key=lambda g: (_unit(recipe.seed, g), g)):
        total += sizes[group]
        best = max(target, key=lambda s: (target[s] * total - filled[s], -list(target).index(s)))
        out[group] = best
        filled[best] += sizes[group]
    return out


# ---------------------------------------------------------------- inputs


@dataclass(frozen=True)
class Inputs:
    recipe: Recipe
    taxonomy: Taxonomy
    registry: SourceRegistry
    recipe_sha256: str
    taxonomy_sha256: str
    registry_sha256: str


def load_inputs(recipe_path: Path, taxonomy_path: Path, registry_path: Path) -> Inputs:
    from supportpilot.data.recipe import load_recipe

    recipe = load_recipe(recipe_path)
    registry = load_source_registry(registry_path)
    known = {s.source_id for s in registry.sources}
    unknown = sorted(set(recipe.sources) - known)
    if unknown:
        raise CorpusError(f"recipe sources missing from the registry: {unknown}")
    return Inputs(
        recipe=recipe,
        taxonomy=load_taxonomy(taxonomy_path),
        registry=registry,
        recipe_sha256=sha256_file(recipe_path),
        taxonomy_sha256=sha256_file(taxonomy_path),
        registry_sha256=sha256_file(registry_path),
    )


def _raw_root(data_dir: Path, inputs: Inputs, source_id: str, purpose: Purpose) -> Path:
    entry = inputs.registry.get(source_id)
    check_use(entry, purpose)
    raw = raw_dir(data_dir, entry)
    if not (raw / MANIFEST).is_file():
        raise CorpusError(f"{source_id}: raw revision not acquired ({raw})")
    problems = verify_raw(raw)
    if problems:
        raise CorpusError(f"{source_id}: raw verification failed: {problems[:3]}")
    return raw


# ---------------------------------------------------------------- intent rows


def _prepare(source_id: str, revision: str, src: SourceRecipe, item: RawItem) -> _Row:
    text = normalize_text(item.text)
    text, pii = mask_values(text, item.mask_values)
    text, auto = mask_pii(text)
    text, names = mask_intro_name(text)
    for k, v in auto.items():
        pii[k] = pii.get(k, 0) + v
    if names:
        pii["intro_name"] = names
    sig = detect_language(text)
    flags: list[str] = []
    if sig.guess not in (item.lang, "unknown"):
        flags.append(f"lang_guess_{sig.guess}")
    if sig.code_switch:
        flags.append("code_switch_detected")
    return _Row(
        source_id=source_id,
        revision=revision,
        item_id=item.source_item_id,
        source_split=item.source_split,
        source_label=item.source_label,
        lang=item.lang,
        text=text,
        group=f"{source_id}:{item.group_key}",
        parallel_id=item.parallel_id,
        origin=src.origin[item.lang],
        domain=src.domain,
        words=len(content_words(text)),
        lang_guess=sig.guess,
        code_switch=sig.code_switch,
        pii=pii,
        flags=flags,
        key=f"{item.lang}|{dedup_key(text)}",
    )


def _gate(row: _Row, recipe: Recipe) -> str | None:
    """Text-quality exclusion reason, or None."""
    if not row.text or row.words == 0:
        return "no_request_text"
    if has_mojibake(row.text):
        return "encoding_damage"
    if len(row.text) > recipe.max_chars:
        return "too_long"
    sig = detect_language(row.text)
    if (sig.latin if row.lang == "en" else sig.cyrillic) == 0:
        return "wrong_script_for_language"  # no letter of the expected script at all
    if residual_pii(row.text):
        return "residual_pii"
    return None


def _label(row: _Row, src: SourceRecipe, inputs: Inputs) -> None:
    d = decide_label(
        row.source_id, src, row.source_label, row.text, row.words, inputs.recipe.min_words
    )
    row.rule = d.rule
    if d.intent is None:
        row.reason, row.detail = "label_excluded", d.reason
        return
    row.intent, row.status = d.intent, d.status
    named = compile_pattern(src.named_product) if src.named_product else None
    action, esc, rule = derive_action(inputs.taxonomy, d.intent, row.text, d, named)
    row.action, row.escalation_reason, row.action_rule = action, esc, rule.value


def _collect_intent_rows(data_dir: Path, inputs: Inputs) -> tuple[list[_Row], dict[str, str]]:
    rows: list[_Row] = []
    raw_inputs: dict[str, str] = {}
    for source_id in sorted(inputs.recipe.sources):
        src = inputs.recipe.sources[source_id]
        if src.kind != "intent":
            continue
        purpose = Purpose.TRAINING if src.active else Purpose.ANALYSIS
        raw = _raw_root(data_dir, inputs, source_id, purpose)
        raw_inputs[source_id] = sha256_file(raw / MANIFEST)
        revision = inputs.registry.get(source_id).revision
        reader = INTENT_READERS[source_id]
        n = 0
        for item in reader(raw):
            n += 1
            row = _prepare(source_id, revision, src, item)
            if not src.active:
                row.reason, row.detail = "source_inactive", src.inactive_reason or ""
            elif (reason := _gate(row, inputs.recipe)) is not None:
                row.reason = reason
            else:
                _label(row, src, inputs)
            rows.append(row)
        log.info("source read", extra={"source_id": source_id, "rows": n})
    return rows, raw_inputs


def _exact_dedup(rows: list[_Row], uf: UnionFind) -> int:
    """Same language + same dedup key: keep one row; conflicting labels exclude all."""
    by_key: dict[str, list[_Row]] = defaultdict(list)
    for r in rows:
        if not r.reason:
            by_key[r.key].append(r)
    removed = 0
    for members in by_key.values():
        if len(members) < 2:
            continue
        intents = {m.intent for m in members}
        for m in members[1:]:
            uf.union(members[0].group, m.group)
        if len(intents) > 1:
            for m in members:
                m.reason = "label_conflict"
                m.detail = "same text labelled " + ", ".join(sorted(str(i) for i in intents))
                removed += 1
            continue
        members.sort(key=lambda m: (m.status is LabelStatus.UNVERIFIED, m.example_id))
        for m in members[1:]:
            m.reason, m.duplicate_of = "exact_duplicate", members[0].example_id
            removed += 1
    return removed


def _families(
    kept: list[_Row], recipe: Recipe, uf: UnionFind
) -> tuple[dict[str, str], list[tuple[str, str, float]]]:
    """Near-duplicate families per language; their groups are merged (one split unit)."""
    fam = UnionFind()
    pairs_out: list[tuple[str, str, float]] = []
    for lang in LANGS:
        part = [r for r in kept if r.lang == lang]
        keys = [r.key.split("|", 1)[1] for r in part]
        nd = recipe.neardup
        for i, j, score in near_duplicate_pairs(
            keys,
            shingle=nd.shingle,
            threshold=nd.jaccard,
            num_perm=nd.num_perm,
            bands=nd.bands,
            seed=recipe.seed,
        ):
            a, b = part[i], part[j]
            fam.union(a.example_id, b.example_id)
            uf.union(a.group, b.group)
            pairs_out.append((a.example_id, b.example_id, round(score, 4)))
    for r in kept:
        fam.add(r.example_id)
    return {r.example_id: fam.find(r.example_id) for r in kept}, pairs_out


def build_intent_examples(
    data_dir: Path, inputs: Inputs
) -> tuple[list[IntentExample], list[Exclusion], dict[str, Any], dict[str, str]]:
    rows, raw_inputs = _collect_intent_rows(data_dir, inputs)
    uf = UnionFind()
    for r in rows:
        uf.add(r.group)
    for r in rows:  # parallel translations share one split unit
        if r.parallel_id:
            uf.union(r.group, f"parallel:{r.parallel_id}")
    exact_removed = _exact_dedup(rows, uf)
    kept = sorted((r for r in rows if not r.reason), key=lambda r: r.example_id)
    families, pairs = _families(kept, inputs.recipe, uf)

    heldout_roots = {
        uf.find(r.group) for r in rows if inputs.recipe.sources[r.source_id].placement == "heldout"
    }
    examples: list[IntentExample] = []
    for r in kept:
        root = uf.find(r.group)
        if r.intent is None or r.status is None or r.action is None:
            raise CorpusError(f"{r.example_id}: kept row without a label")
        examples.append(
            IntentExample(
                example_id=r.example_id,
                source_id=r.source_id,
                source_revision=r.revision,
                source_item_id=r.item_id,
                source_split=r.source_split,
                source_label=r.source_label,
                text=r.text,
                text_sha256=sha256_text(r.text),
                lang=r.lang,
                code_switch=r.code_switch,
                origin=r.origin,
                is_translation=r.origin is Origin.HUMAN_LOCALISED,
                parallel_id=r.parallel_id,
                domain=r.domain,
                intent=r.intent,
                action=r.action,
                escalation_reason=r.escalation_reason,
                label_status=r.status,
                label_rule=r.rule,
                action_rule=r.action_rule,
                pool=Pool.UNVERIFIED if r.status is LabelStatus.UNVERIFIED else Pool.CORE,
                group_id="grp-" + sha256_text(root)[:16],
                family_id="fam-" + sha256_text(families[r.example_id])[:16],
                split=assign_split(inputs.recipe, root, root in heldout_roots),
                quality=Quality(
                    words=r.words,
                    chars=len(r.text),
                    lang_guess=r.lang_guess,
                    lang_consistent=r.lang_guess in (r.lang, "unknown"),
                    pii_masked=dict(sorted(r.pii.items())),
                    flags=r.flags,
                ),
            )
        )
    exclusions = [
        Exclusion(
            source_id=r.source_id,
            source_item_id=r.item_id,
            lang=r.lang,
            source_label=r.source_label,
            reason=r.reason,
            detail=r.detail,
            duplicate_of=r.duplicate_of,
        )
        for r in sorted(rows, key=lambda r: r.example_id)
        if r.reason
    ]
    dedup_stats = {
        "exact_removed": exact_removed,
        "near_duplicate_pairs": len(pairs),
        "pairs": pairs,
    }
    return examples, exclusions, dedup_stats, raw_inputs


# ---------------------------------------------------------------- grounded rows


def build_grounded_examples(
    data_dir: Path, inputs: Inputs
) -> tuple[list[GroundedExample], dict[str, int], dict[str, str]]:
    grounded = {k: v for k, v in inputs.recipe.sources.items() if v.kind == "grounded"}
    out: list[GroundedExample] = []
    stats: Counter[str] = Counter()
    raw_inputs: dict[str, str] = {}
    for source_id, src in sorted(grounded.items()):
        if source_id != "doc2dial":
            raise CorpusError(f"no grounded reader for {source_id}")
        raw = _raw_root(data_dir, inputs, source_id, Purpose.TRAINING)
        raw_inputs[source_id] = sha256_file(raw / MANIFEST)
        revision = inputs.registry.get(source_id).revision
        seen: set[str] = set()
        for item in read_doc2dial(raw):
            stats["read"] += 1
            if not item.evidence:
                stats["excluded_no_evidence"] += 1
                continue
            response, rp = mask_pii(normalize_text(item.response), item.doc_public)
            context: list[Turn] = []
            masked = sum(rp.values())
            for role, text in item.context:
                t, cp = mask_pii(normalize_text(text), item.doc_public)
                masked += sum(cp.values())
                context.append(Turn(role=role, text=t))
            key = dedup_key(context[-1].text) + "||" + dedup_key(response)
            if key in seen:
                stats["excluded_exact_duplicate"] += 1
                continue
            seen.add(key)
            flags = ["missing_evidence_spans"] if item.missing_spans else []
            if masked:
                flags.append("pii_masked")
                stats["pii_masked_rows"] += 1
            group = f"{source_id}:{doc_family(item.doc_id)}"
            out.append(
                GroundedExample(
                    example_id="gr-"
                    + sha256_text(f"{source_id}|{item.dialogue_id}|{item.turn_id}")[:16],
                    source_id=source_id,
                    source_revision=revision,
                    dialogue_id=item.dialogue_id,
                    turn_id=item.turn_id,
                    lang="en",
                    origin=src.origin["en"],
                    domain=item.domain,
                    doc_id=item.doc_id,
                    context=context,
                    response=response,
                    dialogue_act=item.dialogue_act,
                    evidence=[
                        Evidence(span_id=i, text=normalize_text(t)) for i, t in item.evidence
                    ],
                    group_id="grp-" + sha256_text(group)[:16],
                    split=Split.TRAIN,  # reassigned below by balanced_splits
                    flags=flags,
                )
            )
    sizes = Counter(g.group_id for g in out)
    split_of = balanced_splits(inputs.recipe, dict(sizes))
    out = sorted(
        (g.model_copy(update={"split": split_of[g.group_id]}) for g in out),
        key=lambda g: g.example_id,
    )
    stats["kept"] = len(out)
    return out, dict(sorted(stats.items())), raw_inputs


# ---------------------------------------------------------------- checks


def leakage_checks(
    examples: list[IntentExample],
    grounded: list[GroundedExample],
    pairs: list[tuple[str, str, float]],
) -> dict[str, int]:
    """Independent re-checks of the split invariants. Every value must be 0."""
    split_of = {e.example_id: e.split for e in examples}
    by: dict[str, set[Split]] = defaultdict(set)
    for e in examples:
        by[f"group:{e.group_id}"].add(e.split)
        by[f"family:{e.family_id}"].add(e.split)
        if e.parallel_id:
            by[f"parallel:{e.parallel_id}"].add(e.split)
        by[f"text:{e.lang}|{dedup_key(e.text)}"].add(e.split)
    for g in grounded:
        by[f"doc:{g.group_id}"].add(g.split)
    cross = Counter(k.split(":", 1)[0] for k, v in by.items() if len(v) > 1)
    heldout_in_train = sum(
        1 for e in examples if e.source_id == "xsid_kk" and e.split is Split.TRAIN
    )
    return {
        "groups_across_splits": cross["group"],
        "families_across_splits": cross["family"],
        "parallel_across_splits": cross["parallel"],
        "exact_text_across_splits": cross["text"],
        "near_dup_pairs_across_splits": sum(1 for a, b, _ in pairs if split_of[a] != split_of[b]),
        "grounded_docs_across_splits": cross["doc"],
        "heldout_rows_in_train": heldout_in_train,
        "residual_pii_rows": sum(1 for e in examples if residual_pii(e.text)),
    }


# ---------------------------------------------------------------- reports


def _counter(items: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(items).items()))


def language_report(
    examples: list[IntentExample], exclusions: list[Exclusion], taxonomy: Taxonomy
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    all_intents = sorted(taxonomy.intents)
    for lang in LANGS:
        ex = [e for e in examples if e.lang == lang]
        core = [e for e in ex if e.pool is Pool.CORE]
        excl = [x for x in exclusions if x.lang == lang]
        core_intents = Counter(e.intent for e in core)
        out[lang] = {
            "core_rows": len(core),
            "unverified_rows": len(ex) - len(core),
            "excluded_rows": len(excl),
            "independent_core_families": len({e.family_id for e in core}),
            "independent_core_groups": len({e.group_id for e in core}),
            "core_by_split": _counter(e.split.value for e in core),
            "core_by_intent": dict(sorted(core_intents.items())),
            "core_by_action": _counter(e.action.value for e in core),
            "core_by_source": _counter(e.source_id for e in core),
            "core_by_origin": _counter(e.origin.value for e in core),
            "core_by_label_status": _counter(e.label_status.value for e in core),
            "core_in_domain_rows": sum(
                1 for e in core if e.intent not in ("out_of_scope", "greeting")
            ),
            "core_translated_rows": sum(1 for e in core if e.is_translation),
            "intents_without_core_rows": [i for i in all_intents if core_intents[i] == 0],
            "unverified_by_candidate_intent": _counter(
                e.intent for e in ex if e.pool is Pool.UNVERIFIED
            ),
            "excluded_by_reason": _counter(x.reason for x in excl),
            "code_switch_detected_rows": sum(1 for e in ex if e.code_switch),
            "lang_guess_mismatch_rows": sum(1 for e in ex if not e.quality.lang_consistent),
            "pii_masked_rows": sum(1 for e in ex if e.quality.pii_masked),
            "pii_masked_by_kind": _counter(
                k for e in ex for k, n in e.quality.pii_masked.items() for _ in range(n)
            ),
            "human_review": NOT_EVALUATED,
        }
    mixed = [e for e in examples if e.code_switch]
    out["mixed_ru_kk"] = {
        "rows_with_detected_code_switch": len(mixed),
        "core_rows": sum(1 for e in mixed if e.pool is Pool.CORE),
        "note": "automatic lexicon signal only; no source provides natural RU-KK requests",
        "human_review": NOT_EVALUATED,
    }
    return out


def review_sample(examples: list[IntentExample], per_language: int, seed: int) -> dict[str, str]:
    """Blind review sheets (CSV) for an independent fluent reviewer: round-robin over intents."""
    sheets: dict[str, str] = {}
    for lang in LANGS:
        core = [e for e in examples if e.lang == lang and e.pool is Pool.CORE]
        by_intent: dict[str, list[IntentExample]] = defaultdict(list)
        for e in sorted(core, key=lambda e: _unit(seed, e.example_id)):
            by_intent[e.intent].append(e)
        picked: list[IntentExample] = []
        while len(picked) < per_language and any(by_intent.values()):
            for intent in sorted(by_intent):
                if by_intent[intent] and len(picked) < per_language:
                    picked.append(by_intent[intent].pop(0))
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(
            [
                "example_id", "lang", "text", "intent", "action",
                "intent_correct(y/n)", "action_correct(y/n)", "language_ok(y/n)",
                "natural_request(y/n)", "notes",
            ]
        )  # fmt: skip
        for e in sorted(picked, key=lambda e: e.example_id):
            w.writerow([e.example_id, e.lang, e.text, e.intent, e.action.value, "", "", "", "", ""])
        sheets[lang] = buf.getvalue()
    return sheets


def _markdown(report: dict[str, Any]) -> str:
    lines = [f"# Corpus report — {report['version']}", ""]
    lines += ["Automatic checks only. Human review: NOT_EVALUATED.", ""]
    lines += ["## Per language (intent corpus)", ""]
    head = "| Metric | " + " | ".join(LANGS) + " |"
    lines += [head, "|---|" + "---|" * len(LANGS)]
    langs = report["languages"]
    for key in (
        "core_rows",
        "independent_core_families",
        "core_in_domain_rows",
        "core_translated_rows",
        "unverified_rows",
        "excluded_rows",
        "code_switch_detected_rows",
        "lang_guess_mismatch_rows",
        "pii_masked_rows",
    ):
        lines.append(f"| {key} | " + " | ".join(str(langs[x][key]) for x in LANGS) + " |")
    lines += ["", f"Mixed RU-KK: {json.dumps(langs['mixed_ru_kk'], ensure_ascii=False)}", ""]
    for lang in LANGS:
        d = langs[lang]
        lines += [f"## {lang}", ""]
        lines.append(f"- core by split: {d['core_by_split']}")
        lines.append(f"- core by source: {d['core_by_source']}")
        lines.append(f"- core by intent: {d['core_by_intent']}")
        lines.append(f"- core by action: {d['core_by_action']}")
        lines.append(f"- intents without core rows ({len(d['intents_without_core_rows'])}): "
                     f"{', '.join(d['intents_without_core_rows'])}")  # fmt: skip
        lines.append(f"- excluded by reason: {d['excluded_by_reason']}")
        lines.append("")
    lines += ["## Leakage checks (all must be 0)", ""]
    lines += [f"- {k}: {v}" for k, v in report["leakage"].items()]
    lines += ["", "## Grounded dialogues (doc2dial, EN)", "", f"{report['grounded']}", ""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- publish


def _jsonl(models: Iterable[BaseModel]) -> str:
    return "".join(m.model_dump_json() + "\n" for m in models)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def content_hash(root: Path) -> str:
    """SHA256 over `path<TAB>sha256` lines of the data files (manifest and REPORT.md excluded)."""
    lines = [f"{f.path}\t{f.sha256}\n" for f in scan_files(root, _NOT_HASHED)]
    return sha256_text("".join(lines))


@dataclass(frozen=True)
class BuildResult:
    version: str
    path: Path
    created: bool
    counts: dict[str, int]
    leakage: dict[str, int]


def build_corpus(
    data_dir: Path,
    out_root: Path,
    inputs: Inputs,
    code_commit: str | None,
    now: datetime | None = None,
) -> BuildResult:
    examples, exclusions, dedup, raw_a = build_intent_examples(data_dir, inputs)
    grounded, gstats, raw_b = build_grounded_examples(data_dir, inputs)
    leakage = leakage_checks(examples, grounded, dedup["pairs"])
    if any(leakage.values()):
        raise CorpusError(f"leakage/PII gate failed: {leakage}")

    out_root.mkdir(parents=True, exist_ok=True)
    stage = out_root / f".partial-{sha256_text(str(datetime.now(UTC).timestamp()))[:8]}"
    if stage.exists():
        shutil.rmtree(stage)
    try:
        for split in Split:
            _write(
                stage / INTENTS_DIR / f"{split.value}.jsonl",
                _jsonl(e for e in examples if e.pool is Pool.CORE and e.split is split),
            )
            _write(
                stage / GROUNDED_DIR / f"{split.value}.jsonl",
                _jsonl(g for g in grounded if g.split is split),
            )
        _write(
            stage / INTENTS_DIR / "unverified.jsonl",
            _jsonl(e for e in examples if e.pool is Pool.UNVERIFIED),
        )
        _write(stage / INTENTS_DIR / "excluded.jsonl", _jsonl(exclusions))
        pairs = "".join(f"{a}\t{b}\t{s}\n" for a, b, s in dedup["pairs"])
        _write(stage / REPORTS_DIR / "near_duplicate_pairs.tsv", "id_a\tid_b\tjaccard\n" + pairs)
        for lang, sheet in review_sample(
            examples, inputs.recipe.review_sample_per_language, inputs.recipe.seed
        ).items():
            _write(stage / REVIEW_DIR / f"sample_{lang}.csv", sheet)
        counts = {
            "intent_core": sum(1 for e in examples if e.pool is Pool.CORE),
            "intent_unverified": sum(1 for e in examples if e.pool is Pool.UNVERIFIED),
            "intent_excluded": len(exclusions),
            "intent_exact_duplicates_removed": dedup["exact_removed"],
            "intent_near_duplicate_pairs": dedup["near_duplicate_pairs"],
            "grounded": len(grounded),
        }
        for split in Split:
            counts[f"intent_core_{split.value}"] = sum(
                1 for e in examples if e.pool is Pool.CORE and e.split is split
            )
            counts[f"grounded_{split.value}"] = sum(1 for g in grounded if g.split is split)
        report: dict[str, Any] = {
            "version": "pending",
            "counts": counts,
            "languages": language_report(examples, exclusions, inputs.taxonomy),
            "leakage": leakage,
            "grounded": {
                **gstats,
                "by_split": _counter(g.split.value for g in grounded),
                "by_domain": _counter(g.domain for g in grounded),
                "by_dialogue_act": _counter(g.dialogue_act for g in grounded),
                "independent_doc_families": len({g.group_id for g in grounded}),
                "human_review": NOT_EVALUATED,
            },
            "human_review": NOT_EVALUATED,
        }
        _write(
            stage / REPORTS_DIR / "stats.json",
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        )
        content = content_hash(stage)
        stamp = (now or datetime.now(UTC)).astimezone(UTC)
        version = f"corpus-{stamp:%Y%m%d}-{content[:8]}"
        # REPORT.md names the version, so it is written after hashing and is not hashed
        report["version"] = version
        md = _markdown(report)
        existing = sorted(out_root.glob(f"corpus-*-{content[:8]}"))
        if existing and verify_corpus(existing[0]) == []:
            shutil.rmtree(stage)
            m = read_model_json(existing[0] / MANIFEST, CorpusManifest)
            log.info("unchanged content; existing corpus reused", extra={"version": m.version})
            return BuildResult(m.version, existing[0], False, m.counts, leakage)
        _write(stage / "REPORT.md", md)
        manifest = CorpusManifest(
            version=version,
            created_at=stamp,
            code_commit=code_commit,
            recipe_sha256=inputs.recipe_sha256,
            taxonomy_sha256=inputs.taxonomy_sha256,
            registry_sha256=inputs.registry_sha256,
            seed=inputs.recipe.seed,
            inputs=dict(sorted({**raw_a, **raw_b}.items())),
            counts=counts,
            content_sha256=content,
            files=scan_files(stage, frozenset({MANIFEST})),
        )
        write_model_json(stage / MANIFEST, manifest)
        final = out_root / version
        stage.replace(final)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    log.info("corpus published", extra={"version": version, **counts})
    return BuildResult(version, final, True, counts, leakage)


def latest_corpus(out_root: Path) -> Path:
    """Most recent published version (versions sort by build date, then content hash)."""
    found = sorted(p for p in out_root.glob("corpus-*") if (p / MANIFEST).is_file())
    if not found:
        raise CorpusError(f"no published corpus under {out_root}")
    return max(found, key=lambda p: read_model_json(p / MANIFEST, CorpusManifest).created_at)


def verify_corpus(path: Path) -> list[str]:
    """Problems with a published corpus directory (empty list = intact)."""
    try:
        m = read_model_json(path / MANIFEST, CorpusManifest)
    except (OSError, ValueError) as exc:
        return [f"manifest: {exc}"]
    problems = [
        f"{p.path}: {p.problem}" for p in verify_files(path, m.files, frozenset({MANIFEST}))
    ]
    if content_hash(path) != m.content_sha256:
        problems.append("content_sha256 mismatch")
    return problems
