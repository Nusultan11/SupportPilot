"""Stage 4 orchestration: backbone + translations + authored scenarios -> SFT dataset.

Steps (each checkpointed, see steps.py):
  select   -> data/interim/stage4/selected/        (select.py)
  native   -> data/interim/stage4/native/          (native.py, ADR-028)
  kb       -> data/processed/kb_train/              (kbtrain.py)
  translate-> data/interim/stage4/mt/  (cache)       (mtrun.py, GPU, resumable)
  score    -> data/interim/stage4/translated/       (mtrun.py, QE)
  sft      -> data/processed/sft/<dataset_version>/ (this module + sftgates.py)

The SFT step joins the utterances, assigns splits (split.py, ADR-026), builds the examples
(sft.py), removes held-out items that leak into train, balances the language mix toward
EN 30 / RU 35 / KK 35 by down-sampling English only (never up-sampling), runs the gates and
writes train/validation/test Parquet with a manifest, a quality report and statistics.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import yaml

from supportpilot.common.config import Settings
from supportpilot.common.hashing import sha256_text
from supportpilot.common.io import write_json
from supportpilot.common.logging import get_logger
from supportpilot.common.manifest import read_model_json
from supportpilot.common.schemas import DatasetManifest, SourceEntry, Split
from supportpilot.data.kbtrain import CHUNKS, kb_dir
from supportpilot.data.locale import LocaleCatalog, load_locale, localize_en
from supportpilot.data.mtrun import translated_dir
from supportpilot.data.native import NATIVE, native_dir
from supportpilot.data.scenarios import AUTHORED_SOURCE, expand, load_scenarios
from supportpilot.data.select import BACKBONE, selected_dir
from supportpilot.data.sft import Resources, build_all
from supportpilot.data.split import (
    assign_authored,
    exact_leaks,
    lexical_leaks,
    ngram_leaks,
    semantic_leaks,
)
from supportpilot.data.steps import (
    MANIFEST,
    StepError,
    commit,
    content_hash,
    jsonable,
    partial_dir,
    write_parquet,
)

log = get_logger(__name__)
SFT_VERSION = 5
TARGET_MIX = {"en": 0.30, "ru": 0.35, "kk": 0.35}
SFT_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("example_id", pa.string()),
        ("scenario_id", pa.string()),
        ("family_id", pa.string()),
        ("lang", pa.string()),
        ("code_switch", pa.bool_()),
        ("task", pa.string()),
        ("intent", pa.string()),
        ("action", pa.string()),
        ("split", pa.string()),
        ("messages_json", pa.string()),
        ("target_json", pa.string()),
        ("context_chunk_ids", pa.list_(pa.string())),
        ("distractor_chunk_ids", pa.list_(pa.string())),
        ("source_id", pa.string()),
        ("record_id", pa.string()),
        ("origin", pa.string()),
        ("translation_of", pa.string()),
        ("translation_system", pa.string()),
        ("kb_variant", pa.string()),
        ("system_prompt_id", pa.string()),
        ("quality_flags", pa.list_(pa.string())),
        ("user_text", pa.string()),
        ("history_len", pa.int32()),
    ]
)
_DIGITS = re.compile(r"\d+")


def sft_root(data_dir: Path) -> Path:
    return data_dir / "processed" / "sft"


def _bitext_utterances(bb: pl.DataFrame, locale: LocaleCatalog) -> list[dict[str, Any]]:
    rows = []
    for r in bb.filter(pl.col("selected_en")).iter_rows(named=True):
        loc = localize_en(r["text"], locale)
        flags = [
            f for f in (r["stage3_flags"] or []) if f in ("offensive", "lid_low_conf", "repaired")
        ]
        rows.append(
            {
                "utterance_key": r["record_id"],
                "source_id": r["source_id"],
                "record_id": r["record_id"],
                "scenario_id": r["scenario_id"],
                "family_id": r["family_id"],
                "lang": "en",
                "code_switch": False,
                "text": loc.text,
                "amounts_kzt": loc.amounts_kzt,
                "history": [],
                "intent": r["intent"],
                "action": r["action"],
                "escalation_reason": r["escalation_reason"],
                "source_intent": r["source_intent"],
                "slots": json.loads(r["slots_json"]),
                "origin": "original",
                "quality_flags": [*flags, *loc.fixes, "src:bitext"],
            }
        )
    return rows


def _translated_utterances(tr: pl.DataFrame, bb: pl.DataFrame) -> list[dict[str, Any]]:
    src_intent = dict(zip(bb["example_id"].to_list(), bb["source_intent"].to_list(), strict=True))
    rows = []
    for r in tr.filter(pl.col("accept")).iter_rows(named=True):
        loc_amounts = [int(x) for x in _DIGITS.findall(r["localized_en"]) if len(x) >= 3]
        rows.append(
            {
                "utterance_key": r["translation_id"],
                "source_id": r["source_id"],
                "record_id": r["record_id"],
                "scenario_id": r["scenario_id"],
                "family_id": r["family_id"],
                "lang": r["lang"],
                "code_switch": False,
                "text": r["text"],
                "amounts_kzt": [
                    a for a in loc_amounts if a >= 500 and str(a) not in r["slots_json"]
                ],
                "history": [],
                "intent": r["intent"],
                "action": r["action"],
                "escalation_reason": r["escalation_reason"],
                "source_intent": src_intent.get(r["example_id"]),
                "slots": json.loads(r["slots_json"]),
                "origin": "machine_translated",
                "translation_of": r["example_id"],
                "translation_system": r["system"],
                "quality_flags": [*r["fixes"], *r["soft"], "src:bitext", f"mt:{r['system']}"],
            }
        )
    return rows


def _native_utterances(nat: pl.DataFrame) -> list[dict[str, Any]]:
    """Native-language rows of licensed sources (ADR-028): no MT, lineage kept."""
    rows = []
    for r in nat.iter_rows(named=True):
        flags = [f for f in (r["stage3_flags"] or []) if f in ("offensive", "lid_low_conf")]
        rows.append(
            {
                "utterance_key": r["record_id"],
                "source_id": r["source_id"],
                "record_id": r["record_id"],
                "scenario_id": r["scenario_id"],
                "family_id": r["family_id"],
                "lang": r["lang"],
                "code_switch": bool(r["code_switch"]),
                "text": r["text"],
                "amounts_kzt": [],
                "history": [],
                "intent": r["intent"],
                "action": r["action"],
                "escalation_reason": r["escalation_reason"],
                "source_intent": None,
                "slots": {},
                "origin": "original",
                "quality_flags": [*flags, f"src:{r['source_id']}", "src:native"],
            }
        )
    return rows


def _authored_utterances(authored: Path, locale: LocaleCatalog, seed: int) -> list[dict[str, Any]]:
    rows = expand(load_scenarios(authored / "scenarios"), locale, seed)
    for r in rows:
        r["utterance_key"] = r["record_id"]
        r["quality_flags"] = ["src:authored", *(["code_switch"] if r["code_switch"] else [])]
    return rows


def _attach_sections(utts: list[dict[str, Any]], templates: dict[str, Any]) -> None:
    label_sections = templates["label_sections"]
    for u in utts:
        if "sections" not in u:
            u["sections"] = list(label_sections.get(u.get("source_intent") or "", []))


def _balance(examples: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], int]:
    """Down-sample English Bitext train rows toward the 30/35/35 plan (never up-sample).

    Only Bitext rows are dropped: authored and native-source rows are rare, distinct data."""
    train = [e for e in examples if e["split"] == "train"]
    n = Counter(e["lang"] for e in train)
    other = min(n.get("ru", 0), n.get("kk", 0))
    if other == 0:
        return examples, 0
    en_target = round(other * TARGET_MIX["en"] / TARGET_MIX["ru"])
    surplus = n.get("en", 0) - en_target
    if surplus <= 0:
        return examples, 0
    pool = [e for e in train if e["lang"] == "en" and "src:bitext" in e["quality_flags"]]
    # drop evenly across intents: rank within intent, drop the highest ranks first
    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in pool:
        by_intent[e["intent"]].append(e)
    ranked: list[tuple[float, str]] = []
    for _intent, items in by_intent.items():
        items.sort(key=lambda e: sha256_text(f"{seed}|bal|{e['example_id']}"))
        for i, e in enumerate(items):
            ranked.append((1 - i / len(items), e["example_id"]))
    ranked.sort()
    drop = {eid for _, eid in ranked[: min(surplus, len(ranked))]}
    return [e for e in examples if e["example_id"] not in drop], len(drop)


def _leak_filter(
    examples: list[dict[str, Any]], settings: Settings, embedder: Any | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = settings.stage4
    train = [e for e in examples if e["split"] == "train"]
    held = [e for e in examples if e["split"] != "train"]
    train_texts = sorted({e["user_text"] for e in train})
    held_texts = sorted({e["user_text"] for e in held})
    # Preserve the independently assigned evaluation sets. Removing evaluation rows
    # erased entire language/intent cells; remove their near-copies from training instead.
    exact = {train_texts[i] for i in exact_leaks(held_texts, train_texts)}
    lex = lexical_leaks(held_texts, train_texts, threshold=cfg.leak_jaccard, seed=cfg.seed)
    lex_t = {train_texts[i] for i in lex}
    ngr = {train_texts[i] for i in ngram_leaks(held_texts, train_texts, cfg.leak_ngram)}
    sem_t: set[str] = set()
    sem_max: list[float] = []
    if embedder is not None and held_texts:
        mask = [_DIGITS.sub("0", t) for t in train_texts]
        hmask = [_DIGITS.sub("0", t) for t in held_texts]
        tv, hv = embedder.encode(mask), embedder.encode(hmask)
        sem = semantic_leaks(hv, tv, cfg.leak_cosine)
        sem_t = {train_texts[i] for i in sem}
        from supportpilot.data.embed import max_cosine

        best, _ = max_cosine(tv, hv)
        sem_max = [float(x) for x in best.tolist()]
    bad = exact | lex_t | ngr | sem_t
    kept, removed = [], Counter[str]()
    held_ids = {e["scenario_id"] for e in held} | {e["family_id"] for e in held}
    for e in examples:
        if e["split"] == "train" and (
            e["user_text"] in bad or e["scenario_id"] in held_ids or e["family_id"] in held_ids
        ):
            removed[e["split"]] += 1
            continue
        kept.append(e)
    report = {
        "held_out_texts": len(held_texts),
        "policy": "preserve_evaluation_remove_training_copies",
        "exact": len(exact),
        "lexical_jaccard": len(lex_t),
        "ngram": len(ngr),
        "semantic": len(sem_t),
        "semantic_checked": embedder is not None,
        "removed_examples": dict(removed),
        "semantic_max_cosine_quantiles": (
            {
                q: round(float(sorted(sem_max)[int(q * (len(sem_max) - 1))]), 4)
                for q in (0.5, 0.9, 0.99)
            }
            if sem_max
            else {}
        ),
        "thresholds": {
            "jaccard": cfg.leak_jaccard,
            "ngram": cfg.leak_ngram,
            "cosine": cfg.leak_cosine,
        },
    }
    return kept, report


def _row(e: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": e["schema_version"],
        "example_id": e["example_id"],
        "scenario_id": e["scenario_id"],
        "family_id": e["family_id"],
        "lang": e["lang"],
        "code_switch": e["code_switch"],
        "task": e["task"],
        "intent": e["intent"],
        "action": e["action"],
        "split": e["split"],
        "messages_json": json.dumps(e["messages"], ensure_ascii=False),
        "target_json": json.dumps(e["target"], ensure_ascii=False, sort_keys=True),
        "context_chunk_ids": e["context_chunk_ids"],
        "distractor_chunk_ids": e["distractor_chunk_ids"],
        "source_id": e["source_id"],
        "record_id": e["record_id"],
        "origin": e["origin"],
        "translation_of": e["translation_of"],
        "translation_system": e["translation_system"],
        "kb_variant": e["kb_variant"],
        "system_prompt_id": e["system_prompt_id"],
        "quality_flags": e["quality_flags"],
        "user_text": e["user_text"],
        "history_len": e["history_len"],
    }


def _gold_copy_filter(
    examples: list[dict[str, Any]], settings: Settings
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reserve gold questions for decontamination only; never read targets into SFT.

    An external greeting happened to equal a gold question. Reject copied SFT inputs,
    rather than changing gold, its expected outputs or the overlap thresholds.
    """
    from supportpilot.data.gold import load_gold_sources

    cfg = settings.stage4
    sources = load_gold_sources(settings.resolve(cfg.authored_dir) / "gold")
    gold_last: list[str] = []
    gold_all: list[str] = []
    for source in sources:
        for lang in ("en", "ru", "kk"):
            turns = getattr(source, lang)
            user_turns = [turns] if isinstance(turns, str) else turns[::2]
            gold_last.append(user_turns[-1])
            gold_all.append(" ".join(user_turns))
    texts = sorted({e["user_text"] for e in examples})
    bad = exact_leaks(gold_last, texts)
    bad |= set(lexical_leaks(gold_last, texts, threshold=cfg.leak_jaccard, seed=cfg.seed))
    bad |= ngram_leaks(gold_all, texts, cfg.leak_ngram)
    bad_texts = {texts[i] for i in bad}
    removed = Counter(e["split"] for e in examples if e["user_text"] in bad_texts)
    return [e for e in examples if e["user_text"] not in bad_texts], {
        "policy": "gold_questions_for_copy_exclusion_only",
        "removed_examples": dict(removed),
        "gold_targets_used": False,
    }


def _input_version(path: Path) -> str:
    return read_model_json(path / MANIFEST, DatasetManifest).dataset_version


def current_sft(data_dir: Path) -> Path | None:
    root = sft_root(data_dir)
    if not root.is_dir():
        return None
    dirs = sorted(
        (p for p in root.iterdir() if p.is_dir() and (p / MANIFEST).is_file()),
        key=lambda p: read_model_json(p / MANIFEST, DatasetManifest).created_at,
    )
    return dirs[-1] if dirs else None


def build_sft(
    settings: Settings,
    entries: Sequence[SourceEntry],
    *,
    code_commit: str | None,
    force: bool = False,
    semantic: bool = True,
) -> dict[str, Any]:
    from supportpilot.data.sftgates import run_sft_gates

    cfg = settings.stage4
    data_dir = settings.resolve(settings.paths.data_dir)
    authored = settings.resolve(cfg.authored_dir)
    locale = load_locale(settings.resolve(cfg.locale))
    for need in (
        selected_dir(data_dir),
        native_dir(data_dir),
        translated_dir(data_dir),
        kb_dir(data_dir),
    ):
        if not (need / MANIFEST).is_file():
            raise StepError(f"missing input step {need}; run the earlier steps first")
    inputs = [
        f"backbone@{_input_version(selected_dir(data_dir))}",
        f"native@{_input_version(native_dir(data_dir))}",
        f"translated@{_input_version(translated_dir(data_dir))}",
        f"kbtrain@{_input_version(kb_dir(data_dir))}",
        "authored@"
        + sha256_text(
            "".join(
                p.read_text(encoding="utf-8")
                for p in [
                    *sorted((authored / "scenarios").glob("*.yaml")),
                    authored / "responses" / "templates.yaml",
                ]
            )
        )[:16],
        "gold_decontamination@"
        + sha256_text(
            "".join(p.read_text("utf-8") for p in sorted((authored / "gold").glob("*.yaml")))
        )[:16],
    ]
    params = {
        "sft_version": SFT_VERSION,
        "seed": cfg.seed,
        "split_ratios": cfg.split_ratios,
        "leak": [cfg.leak_jaccard, cfg.leak_ngram, cfg.leak_cosine],
        "semantic": semantic,
        "target_mix": TARGET_MIX,
    }
    root = sft_root(data_dir)
    cur = current_sft(data_dir)
    if not force and cur is not None:
        m = read_model_json(cur / MANIFEST, DatasetManifest)
        if m.inputs == inputs and m.params == jsonable(params):
            from supportpilot.common.manifest import verify_files

            if not verify_files(cur, m.files, exclude=frozenset({MANIFEST})):
                return {"status": "skipped", "path": str(cur), "dataset_version": m.dataset_version}

    bb = pl.read_parquet(selected_dir(data_dir) / BACKBONE)
    tr = pl.read_parquet(translated_dir(data_dir) / "translations.parquet")
    nat = pl.read_parquet(native_dir(data_dir) / NATIVE)
    chunks = pl.read_parquet(kb_dir(data_dir) / CHUNKS)
    variants = yaml.safe_load((authored / "kb_train" / "variants.yaml").read_text("utf-8"))
    res = Resources(authored, chunks, variants["variants"], locale, cfg.seed)
    utts = [
        *_bitext_utterances(bb, locale),
        *_translated_utterances(tr, bb),
        *_native_utterances(nat),
        *_authored_utterances(authored, locale, cfg.seed),
    ]
    _attach_sections(utts, res.templates)
    langs: dict[str, set[str]] = defaultdict(set)
    for u in utts:
        if u["source_id"] == AUTHORED_SOURCE and not u["code_switch"]:
            langs[u["scenario_id"]].add(u["lang"])
    authored_scn = [
        (
            u["scenario_id"],
            u["intent"],
            u["action"],
            {"en", "ru", "kk"} <= langs[u["scenario_id"]],
        )
        for u in utts
        if u["source_id"] == AUTHORED_SOURCE
    ]
    split_of: dict[str, Split] = {u["scenario_id"]: Split.TRAIN for u in utts}
    split_of.update(assign_authored(authored_scn, cfg.split_ratios, cfg.seed))
    built = build_all(utts, res, split_of)
    embedder = None
    if semantic:
        from supportpilot.data.embed import Embedder, EmbedderSpec

        embedder = Embedder(EmbedderSpec.model_validate(cfg.embedder))
    clean_examples, gold_filter = _gold_copy_filter(built.examples, settings)
    examples, leak_report = _leak_filter(clean_examples, settings, embedder)
    leak_report["gold_decontamination"] = gold_filter
    examples, dropped_en = _balance(examples, cfg.seed)
    examples.sort(key=lambda e: (e["split"], e["lang"], e["source_id"], e["record_id"], e["task"]))
    rows = [_row(e) for e in examples]
    stats = _stats(rows, utts, built.rejected, leak_report, dropped_en)
    with partial_dir(root / "current") as partial:
        for split in ("train", "validation", "test"):
            write_parquet(
                partial / f"{split}.parquet", [r for r in rows if r["split"] == split], SFT_SCHEMA
            )
        write_json(partial / "stats.json", stats)
        report = run_sft_gates(partial, settings, entries, chunks, leak_report)
        write_json(partial / "quality_report.json", report)
        row_counts: Counter[str] = Counter()
        for r in rows:
            row_counts[f"{r['split']}|lang={r['lang']}|task={r['task']}"] += 1
            row_counts[f"{r['split']}"] += 1
        m = commit(
            partial,
            root / "current",
            name="sft",
            data_schema_version="sft/1",
            code_commit=code_commit,
            inputs=inputs,
            params=params,
            row_counts=dict(row_counts),
            content_sha256=content_hash(rows),
            quality_report="quality_report.json",
            gate=lambda _p: bool(report["passed"]),
        )
    final = root / m.dataset_version
    if final.exists():
        import shutil

        shutil.rmtree(final)
    (root / "current").rename(final)
    return {
        "status": "built",
        "path": str(final),
        "dataset_version": m.dataset_version,
        "gates_passed": report["passed"],
        "splits": {s: row_counts[s] for s in ("train", "validation", "test")},
    }


def _stats(
    rows: list[dict[str, Any]],
    utts: list[dict[str, Any]],
    rejected: Counter[str],
    leak: dict[str, Any],
    dropped_en: int,
) -> dict[str, Any]:
    def dist(key: str, sub: list[dict[str, Any]]) -> dict[str, int]:
        return dict(sorted(Counter(str(r[key]) for r in sub).items()))

    out: dict[str, Any] = {"examples": len(rows), "utterances": len(utts)}
    for split in ("train", "validation", "test"):
        sub = [r for r in rows if r["split"] == split]
        n = max(1, len(sub))
        langs = dist("lang", sub)
        out[split] = {
            "examples": len(sub),
            "lang": langs,
            "lang_share": {k: round(v / n, 4) for k, v in langs.items()},
            "task": dist("task", sub),
            "origin": dist("origin", sub),
            "intents": len({r["intent"] for r in sub}),
            "scenarios": len({r["scenario_id"] for r in sub}),
            "families": len({r["family_id"] for r in sub}),
            "distinct_user_texts": len({r["user_text"] for r in sub}),
            "code_switch": sum(r["code_switch"] for r in sub),
            "multi_turn": sum(r["history_len"] > 0 for r in sub),
            "lang_x_task": dict(sorted(Counter(f"{r['lang']}|{r['task']}" for r in sub).items())),
        }
    out["target_mix"] = TARGET_MIX
    out["english_dropped_for_mix"] = dropped_en
    out["rejected_while_building"] = dict(sorted(rejected.items()))
    out["leakage_filter"] = leak
    return out
