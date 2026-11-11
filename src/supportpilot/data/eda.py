"""Stage 3 EDA: measured profile of every source, written to `artifacts/reports/eda/`.

All numbers come from the data (raw records, the clean checkpoint and the canonical
build), never from dataset cards. Outputs:

    profile.json        every metric below, machine-readable
    report.md           the human report (supportpilot.data.report)
    figures/*.png       charts referenced by the report
    tables/*.csv        long tables (intent counts, placeholder inventory, sweeps)
    review/<source>.md  qualitative review samples (supportpilot.data.review)
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from pydantic import BaseModel
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from supportpilot.common.config import Settings, TokenizerSpec
from supportpilot.common.hashing import sha256_file
from supportpilot.common.io import write_json
from supportpilot.common.logging import get_logger
from supportpilot.common.schemas import SourceEntry
from supportpilot.data.acquire import MANIFEST
from supportpilot.data.build import QUALITY, canonical_dir
from supportpilot.data.clean import CLEAN, clean_dir
from supportpilot.data.langid import LanguageIdentifier
from supportpilot.data.neardup import build_shingles, exact_jaccard
from supportpilot.data.placeholders import PLACEHOLDER_RE
from supportpilot.data.policy import Purpose
from supportpilot.data.records import scan_records
from supportpilot.data.textnorm import exact_key

log = get_logger(__name__)
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
SWEEP = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
BITEXT_TAGS = {
    "B": "basic syntax",
    "C": "coordinated syntax",
    "E": "abbreviations",
    "I": "interrogative",
    "K": "keyword mode",
    "L": "lexical variation",
    "M": "morphological variation",
    "N": "negation",
    "P": "politeness",
    "Q": "colloquial",
    "S": "?",
    "V": "?",
    "W": "offensive",
    "Z": "typos/errors",
}
_POLITE_RE = re.compile(
    r"\b(please|pls|plz|could you|would you|kindly|thank|thanks"
    r"|пожалуйста|спасибо|рахмет|өтінем)\b",
    re.IGNORECASE,
)


class EdaSummary(BaseModel):
    report: str
    profile: str
    figures: int
    review_files: list[str]
    seconds: float


# ------------------------------------------------------------------ helpers


def quantiles(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    out = {f"p{int(q * 100)}": float(np.quantile(arr, q)) for q in QUANTILES}
    out["mean"] = float(arr.mean())
    out["max"] = float(arr.max())
    return {k: round(v, 2) for k, v in out.items()}


def _share(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


def load_tokenizer(spec: TokenizerSpec) -> Any:
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    path = Path(hf_hub_download(spec.repo_id, "tokenizer.json", revision=spec.revision))
    if sha256_file(path) != spec.sha256:
        raise ValueError(f"{spec.repo_id}@{spec.revision}: tokenizer.json sha256 mismatch")
    return Tokenizer.from_file(str(path))


def count_tokens(tok: Any, texts: Sequence[str]) -> np.ndarray:
    out = np.zeros(len(texts), dtype=np.int64)
    for start in range(0, len(texts), 4096):
        enc = tok.encode_batch(list(texts[start : start + 4096]), add_special_tokens=False)
        out[start : start + len(enc)] = [len(e.ids) for e in enc]
    return out


def _dialogue_text(ctx: str | None) -> str:
    if not ctx:
        return ""
    msgs = json.loads(ctx).get("messages") or []
    return "\n".join(f"{m['role']}: {m['content']}" for m in msgs)


# ------------------------------------------------------------------ loading


class Data(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    records: dict[str, pl.DataFrame]
    clean: pl.DataFrame
    groups: pl.DataFrame
    pairs: pl.DataFrame
    pending: pl.DataFrame
    rejected: pl.DataFrame
    canonical: pl.DataFrame
    quality: dict[str, Any]
    manifest: dict[str, Any]


def load_data(settings: Settings, entries: Sequence[SourceEntry]) -> Data:
    from supportpilot.common.manifest import load_source_registry

    data_dir = settings.resolve(settings.paths.data_dir)
    registry = load_source_registry(settings.resolve(settings.data.registry))
    records = {
        e.source_id: scan_records(registry, data_dir, [e.source_id], Purpose.ANALYSIS).collect()
        for e in entries
    }
    clean = pl.concat([pl.read_parquet(clean_dir(data_dir, e) / CLEAN) for e in entries])
    out = canonical_dir(data_dir)
    if not (out / MANIFEST).is_file():
        raise FileNotFoundError(f"no build at {out}; run `supportpilot data clean --all` first")
    files = sorted((out / "examples").glob("lang=*/*.parquet"))
    return Data(
        records=records,
        clean=clean,
        groups=pl.read_parquet(out / "groups.parquet"),
        pairs=pl.read_parquet(out / "pairs.parquet"),
        pending=pl.read_parquet(out / "pending.parquet"),
        rejected=pl.read_parquet(out / "rejected.parquet"),
        canonical=pl.concat([pl.read_parquet(f) for f in files]) if files else pl.DataFrame(),
        quality=json.loads((out / QUALITY).read_text(encoding="utf-8")),
        manifest=json.loads((out / MANIFEST).read_text(encoding="utf-8")),
    )


# ------------------------------------------------------------------ per source


def _templates(keys: Sequence[str], top: int = 12) -> dict[str, Any]:
    openers = Counter(" ".join(k.split()[:3]) for k in keys if k)
    n = len(keys)
    return {
        "distinct_openers_3w": len(openers),
        "top_opener_share": _share(openers.most_common(1)[0][1], n) if openers else 0.0,
        "top10_openers_share": _share(sum(c for _, c in openers.most_common(10)), n),
        "top_openers": [
            {"opener": o, "rows": c, "share": _share(c, n)} for o, c in openers.most_common(top)
        ],
    }


def source_profile(
    sid: str,
    entry: SourceEntry,
    rec: pl.DataFrame,
    cl: pl.DataFrame,
    d: Data,
    tokens: dict[str, dict[str, dict[str, np.ndarray]]],
) -> dict[str, Any]:
    n = rec.height
    g = d.groups.filter(pl.col("source_id") == sid)
    rej = d.rejected.filter(pl.col("source_id") == sid)
    pen = d.pending.filter(pl.col("source_id") == sid)
    can = d.canonical.filter(pl.col("source_id") == sid) if d.canonical.height else d.canonical
    kept_nodes = g.filter(pl.col("destination") != "rejected")
    conflicts = pen.filter(pl.col("reason") == "label_conflict")["record_id"].implode()
    semantic = kept_nodes.filter(~pl.col("record_id").is_in(conflicts))
    fam_sizes = kept_nodes.group_by("family_id").len()["len"].to_numpy()
    scn_sizes = kept_nodes.group_by("scenario_id").len()["len"].to_numpy()

    missing = {
        c: {"null": int(rec[c].null_count()), "empty": int((rec[c] == "").sum() or 0)}
        for c in (
            "text",
            "response",
            "source_intent",
            "source_category",
            "source_tags",
            "source_split",
        )
    }
    text_len = rec["text"].str.len_chars().to_numpy()
    words = np.array([len(t.split()) for t in rec["text"].to_list()])
    resp = rec["response"].drop_nulls()
    lengths: dict[str, Any] = {
        "text_chars": quantiles(text_len),
        "text_words": quantiles(words),
        "response_chars": quantiles(resp.str.len_chars().to_numpy()) if resp.len() else {},
    }
    for tname, per in tokens.items():
        if sid in per:
            for field, arr in per[sid].items():
                lengths[f"{field}_tokens_{tname}"] = quantiles(arr)
    if (cl["record_type"] == "dialogue").any():
        lengths["dialogue_turns"] = quantiles(cl["n_turns"].to_numpy())

    lang = cl.group_by("lang_declared", "lang_detected").len().sort("len", descending=True)
    conf = cl.filter(pl.col("lang_detected").is_in(["en", "ru", "kk"]))["lid_conf"].to_numpy()
    flags = Counter(f for fl in cl["flags"].to_list() for f in fl)
    intents = rec["source_intent"].drop_nulls().value_counts(sort=True)
    cats = rec["source_category"].drop_nulls().value_counts(sort=True)
    tag_counts: Counter[str] = Counter()
    for t in rec["source_tags"].drop_nulls().to_list():
        tag_counts.update(set(t))
    ph_text: Counter[str] = Counter()
    for t in rec["text"].to_list():
        ph_text.update(m.group(1) for m in PLACEHOLDER_RE.finditer(t))
    ph_resp: Counter[str] = Counter()
    ph_resp_names: dict[str, Counter[str]] = {}
    for js in cl["response_placeholders"].drop_nulls().to_list():
        for cls, names in json.loads(js).items():
            ph_resp[cls] += 1
            ph_resp_names.setdefault(cls, Counter()).update(names)
    texts = cl["text"].to_list()
    style = {
        "question_mark": _share(sum("?" in t for t in texts), n),
        "politeness_marker": _share(sum(bool(_POLITE_RE.search(t)) for t in texts), n),
        "offensive": _share(flags["offensive"], n),
        "emoji": _share(flags["emoji"], n),
        "all_caps": _share(flags["all_caps"], n),
        "short_text_le2_words": _share(flags["short_text"], n),
        "lowercase_start": _share(sum(t[:1].islower() for t in texts if t), n),
    }
    distinct_responses = resp.n_unique() if resp.len() else 0
    prof: dict[str, Any] = {
        "source_id": sid,
        "allowed_use": entry.allowed_use.value,
        "licence": entry.licence,
        "revision": entry.revision,
        "records": n,
        "record_types": dict(Counter(rec["record_type"].to_list())),
        "splits": dict(Counter(str(s) for s in rec["source_split"].to_list())),
        "missing": missing,
        "uniqueness": {
            "distinct_raw_texts": rec["text_sha256"].n_unique(),
            "raw_text_dup_rows": n - rec["text_sha256"].n_unique(),
            "distinct_normalised_texts": cl["exact_key"].n_unique(),
            "distinct_after_exact_dedup": kept_nodes["exact_group"].n_unique(),
            "exact_duplicates_removed": rej.filter(pl.col("reason") == "exact_duplicate").height,
            "families": len(fam_sizes),
            "singleton_family_share": _share(int((fam_sizes == 1).sum()), len(fam_sizes)),
            "family_size": quantiles(fam_sizes),
            "scenarios": len(scn_sizes),
            # label_conflict rows form one-text scenarios of their own; exclude them here
            "scenarios_without_conflict_singletons": semantic["scenario_id"].n_unique(),
            "rows_per_scenario": quantiles(scn_sizes),
        },
        "lengths": lengths,
        "language": {
            "declared_vs_detected": lang.to_dicts(),
            "lid_conf": quantiles(conf),
            "low_conf_share": _share(flags["lid_low_conf"], n),
            "code_switch_rows": flags["code_switch"],
            "scripts": dict(Counter(cl["script"].to_list())),
        },
        "labels": {
            "source_intents": intents.height,
            "source_categories": cats.height,
            "intent_counts": {
                r["source_intent"]: r["count"] for r in intents.iter_rows(named=True)
            },
            "category_counts": {
                r["source_category"]: r["count"] for r in cats.iter_rows(named=True)
            },
            "intent_rows_min_max": [
                min(intents["count"].to_list()),
                max(intents["count"].to_list()),
            ]
            if intents.height
            else [],
        },
        "register": {
            "bitext_tags": {
                f"{k} ({BITEXT_TAGS.get(k, '?')})": _share(v, n)
                for k, v in sorted(tag_counts.items())
            },
            "style": style,
        },
        "templates": _templates([exact_key(t) for t in cl["text"].to_list()]),
        "placeholders": {
            "text_rows_with_placeholder": sum(1 for t in rec["text"].to_list() if "{{" in t),
            "text_inventory": dict(ph_text.most_common()),
            "response_rows_by_class": dict(ph_resp),
            "response_distinct_names_by_class": {k: len(v) for k, v in ph_resp_names.items()},
            "response_top_names_by_class": {k: v.most_common(12) for k, v in ph_resp_names.items()},
        },
        "responses": {
            "rows_with_response": int(resp.len()),
            "distinct_responses": int(distinct_responses),
            "responses_per_distinct_text": round(
                resp.len() / max(1, rec["text_sha256"].n_unique()), 3
            ),
        },
        "quality": {
            "flags": {k: {"rows": v, "share": _share(v, n)} for k, v in flags.most_common()},
            "clean_rejects": dict(Counter(r for r in cl["reject_reason"].drop_nulls().to_list())),
        },
        "outcome": {
            "canonical": can.height,
            "pending": pen.height,
            "rejected": rej.height,
            "pending_reasons": dict(Counter(pen["reason"].to_list())),
            "rejected_reasons": dict(Counter(rej["reason"].to_list())),
            "canonical_families": can["family_id"].n_unique() if can.height else 0,
            "canonical_scenarios": can["scenario_id"].n_unique() if can.height else 0,
        },
    }
    return prof


# ------------------------------------------------------------------ cross-source


def cross_source(d: Data) -> dict[str, Any]:
    g = d.groups
    multi = (
        g.group_by("exact_group")
        .agg(pl.col("source_id").unique().sort().alias("sources"))
        .filter(pl.col("sources").list.len() > 1)
    )
    exact_overlap = Counter(" & ".join(s) for s in multi["sources"].to_list())
    src_of = dict(zip(g["example_id"].to_list(), g["source_id"].to_list(), strict=True))
    p = d.pairs.filter(~pl.col("same_source"))
    near: Counter[str] = Counter()
    for a, b, j, comp in zip(
        p["example_a"].to_list(),
        p["example_b"].to_list(),
        p["jaccard"].to_list(),
        p["compatible"].to_list(),
        strict=True,
    ):
        key = " & ".join(sorted((src_of[a], src_of[b])))
        bucket = "0.8+" if j >= 0.8 else "0.5-0.8"
        near[f"{key} | J {bucket} | {'merged-eligible' if comp else 'label-blocked'}"] += 1
    scn = g.group_by("scenario_id").agg(pl.col("source_id").n_unique().alias("n"))
    return {
        "exact_overlap_groups": dict(exact_overlap),
        "near_dup_pairs": dict(sorted(near.items())),
        "scenarios_spanning_2_sources": scn.filter(pl.col("n") > 1).height,
    }


def threshold_sweep(d: Data) -> list[dict[str, Any]]:
    """Family structure if the threshold were different (label guard on/off)."""
    reps = d.groups.filter(
        pl.col("example_id").is_in(d.pairs["example_a"].implode())
        | pl.col("example_id").is_in(d.pairs["example_b"].implode())
    )
    nodes = d.groups.filter(pl.col("destination") != "rejected")
    ids = nodes["example_id"].to_list()
    index = {e: i for i, e in enumerate(ids)}
    srcs = np.array(nodes["source_id"].to_list())
    del reps
    p = d.pairs.filter(pl.col("example_a").is_in(ids) & pl.col("example_b").is_in(ids))
    a = np.array([index[x] for x in p["example_a"].to_list()], dtype=np.int64)
    b = np.array([index[x] for x in p["example_b"].to_list()], dtype=np.int64)
    jac = p["jaccard"].to_numpy()
    comp = p["compatible"].to_numpy()
    same_src = p["same_source"].to_numpy()
    bitext = np.isin(srcs, ["bitext_retail_ecom", "bitext_customer_support"])
    out = []
    n = len(ids)
    for th in SWEEP:
        m = (jac >= th) & comp
        graph = sparse.coo_matrix((np.ones(m.sum()), (a[m], b[m])), shape=(n, n))
        _, lab = connected_components(graph, directed=False)
        sizes = np.bincount(lab)
        bl = np.unique(lab[bitext])
        bs = sizes[bl]
        out.append(
            {
                "threshold": th,
                "pairs_label_compatible": int(m.sum()),
                "pairs_label_incompatible_same_source": int(((jac >= th) & ~comp & same_src).sum()),
                "pairs_cross_source": int(((jac >= th) & ~same_src).sum()),
                "bitext_families": len(bl),
                "bitext_largest_family": int(bs.max()) if len(bs) else 0,
                "bitext_share_in_families_ge100": round(
                    float(bs[bs >= 100].sum() / max(1, bitext.sum())), 4
                ),
            }
        )
    return out


def lsh_recall(d: Data, settings: Settings, sample: int = 1500) -> dict[str, Any]:
    """Brute-force exact Jaccard on a random Bitext sample vs the LSH pairs that were found."""
    cfg = settings.clean
    nodes = (
        d.groups.filter(pl.col("destination") != "rejected")
        .filter(pl.col("source_id").is_in(["bitext_retail_ecom", "bitext_customer_support"]))
        .select("example_id")
        .join(d.clean.select("example_id", "sim_text"), on="example_id")
    )
    rng = np.random.default_rng(cfg.seed)
    idx = np.sort(rng.choice(nodes.height, size=min(sample, nodes.height), replace=False))
    sub = nodes[idx.tolist()]
    sm = build_shingles(sub["sim_text"].to_list(), cfg.shingle_size)
    i, j = np.triu_indices(sub.height, k=1)
    jac = exact_jaccard(sm, np.stack([i, j], axis=1).astype(np.int64))
    ids = sub["example_id"].to_list()
    found = {
        tuple(sorted(x))
        for x in zip(d.pairs["example_a"].to_list(), d.pairs["example_b"].to_list(), strict=True)
    }
    out: dict[str, Any] = {"sample_nodes": sub.height, "pairs_checked": len(jac)}
    for th in (cfg.pair_report_min, 0.6, 0.7, cfg.family_threshold):
        m = jac >= th
        truth = [tuple(sorted((ids[x], ids[y]))) for x, y in zip(i[m], j[m], strict=True)]
        hit = sum(1 for t in truth if t in found)
        out[f"J>={th}"] = {
            "true_pairs": len(truth),
            "found_by_lsh": hit,
            "recall": _share(hit, len(truth)),
        }
    return out


def lid_calibration(d: Data, settings: Settings) -> dict[str, Any]:
    """Belebele has gold language labels: measure LID on its questions and passages."""
    b = d.clean.filter(pl.col("source_id") == "belebele")
    if not b.height:
        return {}
    q = Counter(zip(b["lang_declared"].to_list(), b["lang_detected"].to_list(), strict=True))
    lid = LanguageIdentifier(
        other_min_conf=settings.clean.lid_other_min_conf, min_letters=settings.clean.lid_min_letters
    )
    passages = [json.loads(c)["passage"] for c in b["context_json"].to_list()]
    res = lid.detect_many(passages)
    p = Counter(zip(b["lang_declared"].to_list(), [r.lang for r in res], strict=True))
    acc_q = {
        lang: _share(q[(lang, lang)], sum(v for (g, _), v in q.items() if g == lang))
        for lang in ("en", "ru", "kk")
    }
    acc_p = {
        lang: _share(p[(lang, lang)], sum(v for (g, _), v in p.items() if g == lang))
        for lang in ("en", "ru", "kk")
    }
    return {
        "questions_accuracy": acc_q,
        "passages_accuracy": acc_p,
        "questions_confusion": {f"{g}->{x}": v for (g, x), v in sorted(q.items())},
        "passages_confusion": {f"{g}->{x}": v for (g, x), v in sorted(p.items())},
        "questions_code_switch_flagged": int(b["code_switch"].sum()),
    }


# ------------------------------------------------------------------ orchestration


def run_eda(
    settings: Settings, entries: Sequence[SourceEntry], *, tokens: bool = True
) -> EdaSummary:
    from supportpilot.data.report import write_report
    from supportpilot.data.review import write_review_samples

    start = time.perf_counter()
    out = settings.resolve(settings.paths.reports_dir) / "eda"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    d = load_data(settings, entries)
    tok_counts: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    tokenizer_info = []
    if tokens:
        for spec in settings.eda.tokenizers:
            tok = load_tokenizer(spec)
            tokenizer_info.append(spec.model_dump())
            per: dict[str, dict[str, np.ndarray]] = {}
            for sid, rec in d.records.items():
                cl = d.clean.filter(pl.col("source_id") == sid)
                fields = {"text": count_tokens(tok, rec["text"].to_list())}
                resp = rec["response"].drop_nulls().to_list()
                if resp:
                    fields["response"] = count_tokens(tok, resp)
                if (cl["record_type"] == "dialogue").any():
                    fields["dialogue"] = count_tokens(
                        tok, [_dialogue_text(c) for c in cl["context_json"].to_list()]
                    )
                per[sid] = fields
            tok_counts[spec.name] = per
    profiles = {
        e.source_id: source_profile(
            e.source_id,
            e,
            d.records[e.source_id],
            d.clean.filter(pl.col("source_id") == e.source_id),
            d,
            tok_counts,
        )
        for e in entries
    }
    profile = {
        "dataset_version": d.manifest["dataset_version"],
        "code_commit": d.manifest["code_commit"],
        "inputs": d.manifest["inputs"],
        "params": d.manifest["params"],
        "row_counts": d.manifest["row_counts"],
        "tokenizers": tokenizer_info,
        "sources": profiles,
        "cross_source": cross_source(d),
        "threshold_sweep": threshold_sweep(d),
        "lsh_recall": lsh_recall(d, settings),
        "lid_calibration": lid_calibration(d, settings),
        "gates": d.quality,
        "canonical": {
            "rows": d.canonical.height,
            "intents": dict(Counter(d.canonical["intent"].to_list())) if d.canonical.height else {},
            "actions": dict(Counter(d.canonical["action"].to_list())) if d.canonical.height else {},
            "families": d.canonical["family_id"].n_unique() if d.canonical.height else 0,
            "scenarios": d.canonical["scenario_id"].n_unique() if d.canonical.height else 0,
        },
    }
    write_json(out / "profile.json", profile)
    review_files = write_review_samples(settings, entries, d, out / "review")
    figures = write_report(settings, profile, d, tok_counts, out)
    return EdaSummary(
        report=str(out / "report.md"),
        profile=str(out / "profile.json"),
        figures=figures,
        review_files=review_files,
        seconds=round(time.perf_counter() - start, 2),
    )
