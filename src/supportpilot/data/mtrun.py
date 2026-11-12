"""Stage 4 step 2: MT comparison, translation of the backbone, QE scoring (ADR-024).

- `compare`: every candidate system translates the same stratified EN sample; outputs,
  back-translations and automatic QE go to `artifacts/reports/stage4/mt_compare/`. The
  reviewed annotations live in docs/reviews/stage4_mt_review.yaml (LLM review, labelled so).
- `translate`: the selected system translates the rows marked `mt_<lang>` (resumable cache),
  then the back-translation system translates the outputs back to English.
- `score`: QE (qe.assess) for every translation -> `data/interim/stage4/translated/`.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa

from supportpilot.common.config import Settings
from supportpilot.common.hashing import sha256_text
from supportpilot.common.io import write_json
from supportpilot.common.logging import get_logger
from supportpilot.data.embed import Embedder, EmbedderSpec
from supportpilot.data.langid import LanguageIdentifier
from supportpilot.data.locale import LocaleCatalog, load_locale, localize_en, target_forms
from supportpilot.data.mt import (
    Backend,
    HFTranslator,
    MTSystem,
    TranslateResult,
    lookup,
    translate_cached,
)
from supportpilot.data.qe import QEThresholds, assess, chrf
from supportpilot.data.select import BACKBONE, selected_dir
from supportpilot.data.steps import (
    StepError,
    commit,
    content_hash,
    counts,
    is_current,
    partial_dir,
    read_manifest,
    write_parquet,
)

log = get_logger(__name__)
SCORE_VERSION = 2
TRANSLATED_SCHEMA = pa.schema(
    [
        ("translation_id", pa.string()),
        ("example_id", pa.string()),
        ("record_id", pa.string()),
        ("source_id", pa.string()),
        ("scenario_id", pa.string()),
        ("family_id", pa.string()),
        ("lang", pa.string()),
        ("src_text", pa.string()),
        ("localized_en", pa.string()),
        ("mt_raw", pa.string()),
        ("text", pa.string()),
        ("back_translation", pa.string()),
        ("system", pa.string()),
        ("intent", pa.string()),
        ("action", pa.string()),
        ("escalate", pa.bool_()),
        ("escalation_reason", pa.string()),
        ("slots_json", pa.string()),
        ("lid_conf", pa.float64()),
        ("bt_chrf", pa.float64()),
        ("emb_sim", pa.float64()),
        ("fixes", pa.list_(pa.string())),
        ("hard", pa.list_(pa.string())),
        ("soft", pa.list_(pa.string())),
        ("accept", pa.bool_()),
    ]
)


def systems(settings: Settings) -> dict[str, MTSystem]:
    return {s.id: s for s in (MTSystem.model_validate(x) for x in settings.stage4.mt_systems)}


class Runner:
    """Owns loaded models (one at a time on the 4 GB GPU) and the translation cache."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.data_dir = settings.resolve(settings.paths.data_dir)
        self.systems = systems(settings)
        self._loaded: dict[str, HFTranslator] = {}

    def _backend(self, system_id: str) -> Backend:
        spec = self.systems[system_id]
        if spec.arch == "pivot":
            raise ValueError("pivot systems are run with `translate`, not as a backend")
        if system_id not in self._loaded:
            for other in list(self._loaded):
                self._loaded.pop(other).close()
            self._loaded[system_id] = HFTranslator(spec)
        return self._loaded[system_id]

    def translate(
        self, texts: Sequence[str], system_id: str, src: str, tgt: str, *, limit: int | None = None
    ) -> list[TranslateResult]:
        spec = self.systems[system_id]
        if f"{src}-{tgt}" not in spec.pairs:
            raise ValueError(f"{system_id} does not serve {src}-{tgt}")
        if spec.arch == "pivot":
            first, second = spec.via
            mid = "ru"
            r1 = self.translate(texts, first, src, mid, limit=limit)
            mids = [m for m in lookup(self.data_dir, first, src, mid, texts) if m is not None]
            r2 = self.translate(mids, second, mid, tgt, limit=limit)
            # store the composed result under the pivot id, so callers read one cache
            have = lookup(self.data_dir, first, src, mid, texts)
            pairs = [(t, m) for t, m in zip(texts, have, strict=True) if m is not None]
            finals = lookup(self.data_dir, second, mid, tgt, [m for _, m in pairs])
            done = {t: f for (t, _), f in zip(pairs, finals, strict=True) if f is not None}

            def composed(batch: Sequence[str], _s: str, _t: str) -> list[str]:
                return [done[b] for b in batch]

            ready = [t for t in texts if t in done]
            r3 = translate_cached(
                ready, system=spec, src=src, tgt=tgt, data_dir=self.data_dir, backend=composed
            )
            return [*r1, *r2, r3]
        return [
            translate_cached(
                texts,
                system=spec,
                src=src,
                tgt=tgt,
                data_dir=self.data_dir,
                backend=self._backend(system_id),
                limit=limit,
            )
        ]

    def close(self) -> None:
        for t in self._loaded.values():
            t.close()
        self._loaded.clear()


def _backbone(data_dir: Path) -> pl.DataFrame:
    path = selected_dir(data_dir) / BACKBONE
    if not path.is_file():
        raise StepError("no backbone; run `supportpilot data sft --step select` first")
    return pl.read_parquet(path)


def localized_sources(df: pl.DataFrame, locale: LocaleCatalog) -> list[str]:
    return [localize_en(t, locale).text for t in df["text"].to_list()]


def compare_sample(df: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    """Stratified sample: round-robin over intents, sha256 order inside each intent."""
    pool = df.filter(pl.col("selected_en")).with_columns(
        pl.col("example_id")
        .map_elements(lambda e: sha256_text(f"{seed}|cmp|{e}"), return_dtype=pl.String)
        .alias("_r")
    )
    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in pool.sort("_r").iter_rows(named=True):
        by_intent[r["intent"]].append(r)
    picked: list[dict[str, Any]] = []
    i = 0
    while len(picked) < n and any(i < len(v) for v in by_intent.values()):
        for intent in sorted(by_intent):
            if i < len(by_intent[intent]) and len(picked) < n:
                picked.append(by_intent[intent][i])
        i += 1
    return pl.DataFrame(picked).drop("_r")


def _thresholds(settings: Settings) -> QEThresholds:
    return QEThresholds(
        min_bt_chrf=settings.stage4.qe_min_bt_chrf, min_emb_sim=settings.stage4.qe_min_emb_sim
    )


def run_compare(settings: Settings) -> dict[str, Any]:
    cfg = settings.stage4
    data_dir = settings.resolve(settings.paths.data_dir)
    locale = load_locale(settings.resolve(cfg.locale))
    sample = compare_sample(_backbone(data_dir), cfg.mt_compare_n, cfg.seed)
    src = localized_sources(sample, locale)
    runner = Runner(settings)
    candidates = {
        "ru": ["opus_en_ru", "m2m100_418m", "madlad400_3b"],
        "kk": ["opus_en_trk", "m2m100_418m", "madlad400_3b", "madlad400_3b_pivot_ru"],
    }
    bt_system = cfg.back_translation
    for lang, ids in candidates.items():
        for sid in ids:
            runner.translate(src, sid, "en", lang)
    outputs: dict[tuple[str, str], list[str]] = {}
    for lang, ids in candidates.items():
        for sid in ids:
            hyps = lookup(data_dir, sid, "en", lang, src)
            outputs[(lang, sid)] = [h or "" for h in hyps]
    for lang, ids in candidates.items():
        all_hyps = sorted({h for sid in ids for h in outputs[(lang, sid)] if h})
        runner.translate(all_hyps, bt_system[lang], lang, "en")
    runner.close()
    emb = Embedder(EmbedderSpec.model_validate(cfg.embedder))
    src_vec = emb.encode(src)
    lid = LanguageIdentifier()
    th = _thresholds(settings)
    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    for (lang, sid), outs in outputs.items():
        backs = lookup(data_dir, bt_system[lang], lang, "en", outs)
        vec = emb.encode(outs)
        sims = (src_vec * vec).sum(axis=1)
        accepted, flags = 0, Counter[str]()
        chrfs = []
        for i, (s_txt, h, b, sim) in enumerate(zip(src, outs, backs, sims, strict=True)):
            keep = list(target_forms(s_txt, lang, locale).values())
            q = assess(s_txt, h, lang, lid=lid, back=b, emb_sim=float(sim), keep=keep, th=th)
            accepted += q.accept
            flags.update(q.hard)
            chrfs.append(chrf(b or "", s_txt))
            rows.append(
                {
                    "sample": i,
                    "example_id": sample["example_id"][i],
                    "intent": sample["intent"][i],
                    "lang": lang,
                    "system": sid,
                    "src": s_txt,
                    "mt": h,
                    "post_edited": q.text,
                    "back": b,
                    "bt_chrf": q.bt_chrf,
                    "emb_sim": round(float(sim), 4),
                    "hard": q.hard,
                    "soft": q.soft,
                    "fixes": q.fixes,
                    "accept": q.accept,
                }
            )
        metrics[f"{lang}|{sid}"] = {
            "n": len(outs),
            "auto_accept_rate": round(accepted / len(outs), 4),
            "mean_bt_chrf": round(float(np.mean(chrfs)), 4),
            "mean_emb_sim": round(float(np.mean(sims)), 4),
            "hard_flags": dict(flags.most_common()),
        }
    out = settings.resolve(settings.paths.reports_dir) / "stage4" / "mt_compare"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(out / "metrics.json", metrics)
    return {"out": str(out), "metrics": metrics}


def run_translate(
    settings: Settings, lang: str, *, limit: int | None = None, back: bool = True
) -> list[TranslateResult]:
    cfg = settings.stage4
    data_dir = settings.resolve(settings.paths.data_dir)
    locale = load_locale(settings.resolve(cfg.locale))
    df = _backbone(data_dir).filter(pl.col(f"mt_{lang}"))
    src = localized_sources(df, locale)
    runner = Runner(settings)
    sid = cfg.mt_selected[lang]
    results = runner.translate(src, sid, "en", lang, limit=limit)
    if back:
        hyps = [h for h in lookup(data_dir, sid, "en", lang, src) if h]
        results += runner.translate(hyps, cfg.back_translation[lang], lang, "en", limit=limit)
    runner.close()
    return results


def translated_dir(data_dir: Path) -> Path:
    return data_dir / "interim" / "stage4" / "translated"


def run_score(
    settings: Settings, *, code_commit: str | None, force: bool = False
) -> dict[str, Any]:
    """QE for every translation in the cache; missing translations are an error."""
    cfg = settings.stage4
    data_dir = settings.resolve(settings.paths.data_dir)
    locale = load_locale(settings.resolve(cfg.locale))
    bb = _backbone(data_dir)
    out = translated_dir(data_dir)
    params = {
        "score_version": SCORE_VERSION,
        "mt_selected": cfg.mt_selected,
        "back_translation": cfg.back_translation,
        "qe_min_bt_chrf": cfg.qe_min_bt_chrf,
        "qe_min_emb_sim": cfg.qe_min_emb_sim,
        "embedder": cfg.embedder,
        # the localised MT sources (localisation rules can change without a backbone change)
        "sources_sha256": {
            lang: sha256_text("\n".join(localized_sources(bb.filter(pl.col(f"mt_{lang}")), locale)))
            for lang in ("ru", "kk")
        },
    }
    inputs = [f"backbone@{read_manifest(selected_dir(data_dir)).dataset_version}"]
    if not force and is_current(out, inputs, params):
        return {"status": "skipped", "rows": read_manifest(out).row_counts}
    emb = Embedder(EmbedderSpec.model_validate(cfg.embedder))
    lid = LanguageIdentifier()
    th = _thresholds(settings)
    rows: list[dict[str, Any]] = []
    for lang in ("ru", "kk"):
        df = bb.filter(pl.col(f"mt_{lang}"))
        src = localized_sources(df, locale)
        sid = cfg.mt_selected[lang]
        hyps = lookup(data_dir, sid, "en", lang, src)
        missing = sum(h is None for h in hyps)
        if missing:
            raise StepError(
                f"{missing} of {len(hyps)} {lang} translations missing; "
                f"run `supportpilot data translate --lang {lang}`"
            )
        hyp_s = [h or "" for h in hyps]
        backs = lookup(data_dir, cfg.back_translation[lang], lang, "en", hyp_s)
        sims = (emb.encode(src) * emb.encode(hyp_s)).sum(axis=1)
        for r, s_txt, h, b, sim in zip(
            df.iter_rows(named=True), src, hyp_s, backs, sims, strict=True
        ):
            keep = list(target_forms(s_txt, lang, locale).values())
            q = assess(s_txt, h, lang, lid=lid, back=b, emb_sim=float(sim), keep=keep, th=th)
            rows.append(
                {
                    "translation_id": sha256_text(f"{r['example_id']}|{lang}|{sid}")[:16],
                    "example_id": r["example_id"],
                    "record_id": r["record_id"],
                    "source_id": r["source_id"],
                    "scenario_id": r["scenario_id"],
                    "family_id": r["family_id"],
                    "lang": lang,
                    "src_text": r["text"],
                    "localized_en": s_txt,
                    "mt_raw": h,
                    "text": q.text,
                    "back_translation": b,
                    "system": sid,
                    "intent": r["intent"],
                    "action": r["action"],
                    "escalate": r["escalate"],
                    "escalation_reason": r["escalation_reason"],
                    "slots_json": r["slots_json"],
                    "lid_conf": q.lid_conf,
                    "bt_chrf": q.bt_chrf,
                    "emb_sim": q.emb_sim,
                    "fixes": q.fixes,
                    "hard": q.hard,
                    "soft": q.soft,
                    "accept": q.accept,
                }
            )
    rows.sort(key=lambda r: (r["lang"], r["source_id"], r["record_id"]))
    hard = Counter(f"{r['lang']}|{h}" for r in rows for h in r["hard"])
    row_counts = {
        **counts(rows, "lang", "accept"),
        **{f"hard|{k}": v for k, v in sorted(hard.items())},
    }
    with partial_dir(out) as partial:
        write_parquet(partial / "translations.parquet", rows, TRANSLATED_SCHEMA)
        m = commit(
            partial,
            out,
            name="translated",
            data_schema_version="stage4_translated/1",
            code_commit=code_commit,
            inputs=inputs,
            params=params,
            row_counts=row_counts,
            content_sha256=content_hash(rows),
        )
    return {"status": "built", "dataset_version": m.dataset_version, "rows": m.row_counts}
