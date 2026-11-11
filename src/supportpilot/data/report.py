"""Render the EDA profile into `report.md`, `figures/*.png` and `tables/*.csv`.

Charts follow the project data-viz rules: categorical slots in fixed order, one axis per
chart (small multiples instead of dual axes), thin marks, recessive grid, legends for
2+ series, numbers also given in tables next to every chart.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from supportpilot.common.config import Settings
from supportpilot.common.io import atomic_write_text
from supportpilot.data.review import load_review

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SEQ = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
SHORT = {
    "bitext_retail_ecom": "Bitext retail",
    "bitext_customer_support": "Bitext CS",
    "whileai_ecommerce_intent": "while-ai",
    "support_json_ru": "support-json-ru",
    "belebele": "Belebele",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "font.size": 9,
            "legend.frameon": False,
            "legend.labelcolor": INK,
            "lines.linewidth": 2,
        }
    )


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1 else f"{v:,.1f}".rstrip("0").rstrip(".")
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(_fmt(c).replace("|", "\\|") for c in r) + " |" for r in rows]
    return "\n".join(out)


# ------------------------------------------------------------------ figures


def fig_volume(profile: dict[str, Any], path: Path) -> None:
    srcs = list(profile["sources"])
    measures: list[tuple[str, Callable[[dict[str, Any]], int]]] = [
        ("records", lambda p: p["records"]),
        ("distinct texts", lambda p: p["uniqueness"]["distinct_after_exact_dedup"]),
        ("paraphrase families", lambda p: p["uniqueness"]["families"]),
        ("scenarios", lambda p: p["uniqueness"]["scenarios_without_conflict_singletons"]),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    h = 0.19
    y = np.arange(len(srcs))
    for k, (name, get) in enumerate(measures):
        vals = [get(profile["sources"][s]) for s in srcs]
        ax.barh(y + (k - 1.5) * h, vals, height=h - 0.02, color=SERIES[k], label=name)
        for yy, v in zip(y + (k - 1.5) * h, vals, strict=True):
            ax.text(v * 1.08, yy, f"{v:,}", va="center", fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set_yticks(y, [SHORT.get(s, s) for s in srcs])
    ax.invert_yaxis()
    ax.set_xlabel("count (log scale)")
    ax.set_title("Volume is not diversity: records vs distinct texts vs families vs scenarios")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def fig_sweep(sweep: list[dict[str, Any]], path: Path) -> None:
    th = [s["threshold"] for s in sweep]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    a1.plot(
        th,
        [s["pairs_label_compatible"] for s in sweep],
        color=SERIES[0],
        marker="o",
        ms=4,
        label="same label",
    )
    a1.plot(
        th,
        [max(1, s["pairs_label_incompatible_same_source"]) for s in sweep],
        color=SERIES[1],
        marker="o",
        ms=4,
        label="different label (blocked)",
    )
    a1.set_yscale("log")
    a1.set_xlabel("Jaccard threshold (char 5-grams)")
    a1.set_ylabel("near-duplicate pairs, same source")
    a1.set_title("Lexically similar ≠ same request")
    a1.legend(fontsize=8)
    a2.plot(
        th,
        [100 * s["bitext_share_in_families_ge100"] for s in sweep],
        color=SERIES[0],
        marker="o",
        ms=4,
    )
    a2.set_xlabel("Jaccard threshold")
    a2.set_ylabel("% of Bitext texts in families ≥ 100")
    a2.set_title("Chaining: families merge below 0.8")
    _save(fig, path)


def fig_tokens(
    tok: dict[str, dict[str, dict[str, np.ndarray]]], name: str, field: str, path: Path
) -> None:
    per = tok.get(name, {})
    srcs = [s for s in per if field in per[s]]
    if not srcs:
        return
    fig, axes = plt.subplots(1, len(srcs), figsize=(2.3 * len(srcs), 2.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, srcs, strict=True):
        arr = per[s][field]
        hi = max(8, int(np.quantile(arr, 0.99)))
        ax.hist(np.clip(arr, 0, hi), bins=30, color=SERIES[0], edgecolor=SURFACE, linewidth=0.6)
        ax.axvline(np.median(arr), color=INK, linewidth=1, linestyle=":")
        ax.set_title(SHORT.get(s, s), fontsize=9)
        ax.set_xlabel(f"{field} tokens (≤ p99)")
        ax.text(
            0.97,
            0.92,
            f"median {np.median(arr):.0f}",
            transform=ax.transAxes,
            ha="right",
            fontsize=7,
            color=INK2,
        )
    axes[0].set_ylabel("rows")
    fig.suptitle(
        f"Qwen tokenizer ({name}) length of {field}", fontsize=10, color=INK, fontweight="bold"
    )
    fig.tight_layout()
    _save(fig, path)


def fig_family_sizes(d: Any, path: Path) -> None:
    g = d.groups.filter(pl.col("destination") != "rejected")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for k, sid in enumerate(SHORT):
        sizes = g.filter(pl.col("source_id") == sid).group_by("family_id").len()["len"].to_numpy()
        if not len(sizes):
            continue
        xs = np.sort(sizes)
        ccdf = 1 - np.arange(len(xs)) / len(xs)
        ax.step(xs, ccdf, where="post", color=SERIES[k], label=SHORT[sid])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("family size (distinct texts)")
    ax.set_ylabel("share of families ≥ size")
    ax.set_title("Paraphrase family sizes (J ≥ 0.8 + label guard)")
    ax.legend(fontsize=8)
    _save(fig, path)


def fig_intents(profile: dict[str, Any], path: Path) -> None:
    items = sorted(profile["canonical"]["intents"].items(), key=lambda kv: kv[1])
    if not items:
        return
    fig, ax = plt.subplots(figsize=(6.5, 0.24 * len(items) + 1))
    ax.barh([k for k, _ in items], [v for _, v in items], color=SERIES[0], height=0.7)
    for i, (_, v) in enumerate(items):
        ax.text(v + 30, i, f"{v:,}", va="center", fontsize=7, color=INK2)
    ax.set_xlabel("canonical rows")
    ax.set_title("Canonical rows per v0 intent (draft map)")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def fig_flags(profile: dict[str, Any], path: Path) -> None:
    srcs = list(profile["sources"])
    names = sorted(
        {
            f
            for p in profile["sources"].values()
            for f, v in p["quality"]["flags"].items()
            if v["share"] >= 0.001 and f != "repaired"
        }
    )
    mat = np.array(
        [
            [profile["sources"][s]["quality"]["flags"].get(f, {}).get("share", 0.0) for f in names]
            for s in srcs
        ]
    )
    fig, ax = plt.subplots(figsize=(0.42 * len(names) + 2, 0.5 * len(srcs) + 1.6))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("seq", ["#f4f8fd", *SEQ])
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=max(0.01, mat.max()), aspect="auto")
    ax.set_xticks(range(len(names)), names, rotation=70, ha="right", fontsize=7)
    ax.set_yticks(range(len(srcs)), [SHORT.get(s, s) for s in srcs])
    ax.grid(False)
    for i in range(len(srcs)):
        for j in range(len(names)):
            if mat[i, j] >= 0.005:
                ax.text(
                    j,
                    i,
                    f"{100 * mat[i, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=INK if mat[i, j] < 0.5 * mat.max() else SURFACE,
                )
    fig.colorbar(im, ax=ax, fraction=0.03, label="share of rows")
    ax.set_title("Quality flags per source (% of rows)")
    _save(fig, path)


def fig_lid(d: Any, path: Path) -> None:
    cl = d.clean.filter(pl.col("lang_detected").is_in(["en", "ru", "kk"]))
    srcs = [s for s in SHORT if cl.filter(pl.col("source_id") == s).height]
    fig, axes = plt.subplots(1, len(srcs), figsize=(2.3 * len(srcs), 2.5))
    for ax, s in zip(np.atleast_1d(axes), srcs, strict=True):
        v = cl.filter(pl.col("source_id") == s)["lid_conf"].to_numpy()
        ax.hist(v, bins=20, range=(0, 1), color=SERIES[0], edgecolor=SURFACE, linewidth=0.6)
        ax.set_title(SHORT[s], fontsize=9)
        ax.set_xlabel("LID confidence")
    np.atleast_1d(axes)[0].set_ylabel("rows")
    fig.suptitle(
        "Language-ID confidence of the chosen language", fontsize=10, color=INK, fontweight="bold"
    )
    fig.tight_layout()
    _save(fig, path)


# ------------------------------------------------------------------ markdown


def _q(qs: dict[str, float]) -> list[Any]:
    return [qs.get(k, "") for k in ("p5", "p50", "p95", "p99", "max")]


def _source_section(sid: str, p: dict[str, Any], d: Any) -> str:
    u, lab, reg, tpl, ph, q, o = (
        p[k]
        for k in (
            "uniqueness",
            "labels",
            "register",
            "templates",
            "placeholders",
            "quality",
            "outcome",
        )
    )
    s = [f"### {SHORT.get(sid, sid)} (`{sid}`)", ""]
    s.append(
        f"Licence `{p['licence']}`, allowed use **{p['allowed_use']}**, revision "
        f"`{p['revision'][:12]}`. "
        f"Record types: {', '.join(f'{k} {v:,}' for k, v in p['record_types'].items())}. "
        f"Splits: {', '.join(f'{k} {v:,}' for k, v in p['splits'].items())}."
    )
    s.append("")
    s.append(
        table(
            [
                "records",
                "distinct raw texts",
                "distinct normalised",
                "after exact dedup",
                "families",
                "singleton families",
                "scenarios",
            ],
            [
                [
                    p["records"],
                    u["distinct_raw_texts"],
                    u["distinct_normalised_texts"],
                    u["distinct_after_exact_dedup"],
                    u["families"],
                    f"{100 * u['singleton_family_share']:.1f}%",
                    u["scenarios"],
                ]
            ],
        )
    )
    s.append("")
    s.append(
        "**Missing values** (null / empty string): "
        + ", ".join(f"{c} {v['null']:,}/{v['empty']:,}" for c, v in p["missing"].items())
    )
    s.append("")
    rows = [[k, *_q(v)] for k, v in p["lengths"].items() if v]
    s.append(table(["length", "p5", "p50", "p95", "p99", "max"], rows))
    s.append("")
    lang = ", ".join(
        f"{r['lang_declared']}→{r['lang_detected']} {r['len']:,}"
        for r in p["language"]["declared_vs_detected"]
    )
    s.append(
        f"**Language** (declared→detected): {lang}. LID confidence p50 "
        f"{p['language']['lid_conf'].get('p50', '-')}, low-confidence share "
        f"{100 * p['language']['low_conf_share']:.1f}%, code-switch rows "
        f"{p['language']['code_switch_rows']}."
    )
    s.append("")
    if lab["source_intents"]:
        top = sorted(lab["intent_counts"].items(), key=lambda kv: -kv[1])
        s.append(
            f"**Labels**: {lab['source_intents']} intents, {lab['source_categories']} "
            "categories; rows per intent "
            f"{lab['intent_rows_min_max'][0]:,} to {lab['intent_rows_min_max'][1]:,}. Largest: "
            + ", ".join(f"{k} {v:,}" for k, v in top[:6])
            + "; smallest: "
            + ", ".join(f"{k} {v:,}" for k, v in top[-4:])
            + "."
        )
    else:
        s.append("**Labels**: no intent label in the source.")
    s.append("")
    st = reg["style"]
    s.append(
        f"**Register/style**: question mark {100 * st['question_mark']:.0f}%, politeness marker "
        f"{100 * st['politeness_marker']:.0f}%, offensive {100 * st['offensive']:.1f}%, emoji "
        f"{100 * st['emoji']:.1f}%, ≤2 words {100 * st['short_text_le2_words']:.1f}%, lowercase "
        f"start {100 * st['lowercase_start']:.0f}%."
        + (
            " Bitext tags (share of rows): "
            + ", ".join(f"{k} {100 * v:.0f}%" for k, v in reg["bitext_tags"].items())
            + "."
            if reg["bitext_tags"]
            else ""
        )
    )
    s.append("")
    s.append(
        f"**Templates**: {tpl['distinct_openers_3w']:,} distinct 3-word openers; top opener "
        f"covers {100 * tpl['top_opener_share']:.1f}% and the top 10 cover "
        f"{100 * tpl['top10_openers_share']:.1f}% of rows. Top: "
        + "; ".join(f"“{t['opener']}” {100 * t['share']:.1f}%" for t in tpl["top_openers"][:6])
        + "."
    )
    s.append("")
    if ph["text_inventory"] or ph["response_rows_by_class"]:
        s.append(
            f"**Placeholders**: {ph['text_rows_with_placeholder']:,} texts contain one ("
            + ", ".join(f"{k} {v:,}" for k, v in ph["text_inventory"].items())
            + ") → filled with synthetic values. Responses: rows by class "
            + ", ".join(f"{k} {v:,}" for k, v in ph["response_rows_by_class"].items())
            + "; distinct names "
            + ", ".join(f"{k} {v:,}" for k, v in ph["response_distinct_names_by_class"].items())
            + "."
        )
        s.append("")
    resp = p["responses"]
    if resp["rows_with_response"]:
        s.append(
            f"**Responses**: {resp['rows_with_response']:,} rows, {resp['distinct_responses']:,} "
            f"distinct; {resp['responses_per_distinct_text']} responses per distinct text."
        )
        s.append("")
    flags = sorted(q["flags"].items(), key=lambda kv: -kv[1]["rows"])
    s.append(
        "**Flags** (rows): "
        + ", ".join(f"{k} {v['rows']:,} ({100 * v['share']:.1f}%)" for k, v in flags[:16])
        + "."
    )
    s.append("")
    s.append(
        f"**Outcome**: canonical {o['canonical']:,} · pending {o['pending']:,} ("
        + ", ".join(f"{k} {v:,}" for k, v in o["pending_reasons"].items())
        + f") · rejected {o['rejected']:,} ("
        + ", ".join(f"{k} {v:,}" for k, v in o["rejected_reasons"].items())
        + ")."
    )
    return "\n".join(s)


def _blocked_examples(d: Any, n: int = 10) -> str:
    texts = dict(zip(d.clean["example_id"].to_list(), d.clean["text"].to_list(), strict=True))
    p = d.pairs.filter(
        pl.col("same_source") & ~pl.col("compatible") & (pl.col("jaccard") >= 0.8)
    ).sort("jaccard", descending=True)
    rows = []
    seen: set[tuple[str, str]] = set()
    for r in p.iter_rows(named=True):
        key = (r["label_a"], r["label_b"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            [
                f"{r['jaccard']:.2f}",
                f"{r['label_a']}: {texts.get(r['example_a'], '')[:80]}",
                f"{r['label_b']}: {texts.get(r['example_b'], '')[:80]}",
            ]
        )
        if len(rows) >= n:
            break
    return table(["J", "text A (label)", "text B (label)"], rows)


def _review_section(settings: Settings, profile: dict[str, Any], d: Any) -> str:
    review = load_review(settings.resolve(settings.eda.review_annotations))
    if review is None:
        return "_No annotations yet: `docs/reviews/stage3_review.yaml` is missing._"
    texts = dict(zip(d.clean["record_id"].to_list(), d.clean["text"].to_list(), strict=True))
    out = [
        f"Reviewer: **{review.reviewer}** ({review.reviewed_at}). {review.method}",
        "",
        "Codes: " + "; ".join(f"`{k}` {v}" for k, v in review.codes.items()),
        "",
    ]
    for sid, sr in review.sources.items():
        out.append(f"### {SHORT.get(sid, sid)}")
        out.append("")
        for part, prefix in (("random", "R"), ("flagged", "F")):
            ann = [a for a in sr.annotations if a.id.startswith(prefix)]
            if not ann:
                continue
            c = Counter(code for a in ann for code in a.codes)
            ok = sum(1 for a in ann if a.codes == ["OK"])
            hard = sum(1 for a in ann if set(a.codes) & set(review.hard_codes))
            details = ", ".join(f"{k} {v}" for k, v in c.most_common() if k != "OK")
            out.append(
                f"- **{part}** ({len(ann)} reviewed): OK {ok} ({100 * ok / len(ann):.0f}%), "
                f"with a hard defect {hard} ({100 * hard / len(ann):.0f}%); codes: {details}"
            )
        examples: dict[str, list[str]] = {}
        for a in sr.annotations:
            for code in a.codes:
                if code != "OK" and len(examples.setdefault(code, [])) < 2:
                    note = f" — {a.note}" if a.note else ""
                    examples[code].append(f"`{a.id}` “{texts.get(a.record_id, '')[:110]}”{note}")
        if examples:
            out.append(
                "- examples: " + " · ".join(f"**{k}**: {'; '.join(v)}" for k, v in examples.items())
            )
        out += [
            "",
            f"**Findings.** {sr.summary}",
            "",
            f"**Verdict.** {sr.verdict}",
            "",
            f"**Recommendation for Stage 4.** {sr.recommendation}",
            "",
        ]
    return "\n".join(out)


def write_report(
    settings: Settings, profile: dict[str, Any], d: Any, tok: dict[str, Any], out: Path
) -> int:
    _style()
    figs = out / "figures"
    fig_volume(profile, figs / "volume_vs_diversity.png")
    fig_sweep(profile["threshold_sweep"], figs / "threshold_sweep.png")
    for name in tok:
        fig_tokens(tok, name, "text", figs / f"tokens_text_{name}.png")
        fig_tokens(tok, name, "response", figs / f"tokens_response_{name}.png")
        fig_tokens(tok, name, "dialogue", figs / f"tokens_dialogue_{name}.png")
    fig_family_sizes(d, figs / "family_sizes.png")
    fig_intents(profile, figs / "canonical_intents.png")
    fig_flags(profile, figs / "flags.png")
    fig_lid(d, figs / "lid_confidence.png")
    n_figs = len(list(figs.glob("*.png")))

    tables = out / "tables"
    pl.DataFrame(profile["threshold_sweep"]).write_csv(tables / "threshold_sweep.csv")
    rows = [
        (sid, k, v)
        for sid, p in profile["sources"].items()
        for k, v in p["labels"]["intent_counts"].items()
    ]
    pl.DataFrame(rows, schema=["source_id", "source_intent", "rows"], orient="row").write_csv(
        tables / "source_intents.csv"
    )
    ph = [
        (sid, cls, name, cnt)
        for sid, p in profile["sources"].items()
        for cls, names in p["placeholders"]["response_top_names_by_class"].items()
        for name, cnt in names
    ]
    pl.DataFrame(ph, schema=["source_id", "class", "placeholder", "rows"], orient="row").write_csv(
        tables / "response_placeholders_top.csv"
    )

    srcs = profile["sources"]
    md = ["# Stage 3 EDA and data-quality report", ""]
    md.append(
        f"Dataset `{profile['dataset_version']}` · code `{str(profile['code_commit'])[:12]}` · "
        "every number below is computed from the data by `supportpilot data eda` (profile.json)."
    )
    md += ["", "## 1. Summary", ""]
    md.append(
        table(
            [
                "source",
                "allowed use",
                "records",
                "distinct texts",
                "families",
                "scenarios",
                "canonical",
                "pending",
                "rejected",
            ],
            [
                [
                    SHORT.get(s, s),
                    p["allowed_use"],
                    p["records"],
                    p["uniqueness"]["distinct_after_exact_dedup"],
                    p["uniqueness"]["families"],
                    p["uniqueness"]["scenarios_without_conflict_singletons"],
                    p["outcome"]["canonical"],
                    p["outcome"]["pending"],
                    p["outcome"]["rejected"],
                ]
                for s, p in srcs.items()
            ],
        )
    )
    c = profile["canonical"]
    md += [
        "",
        f"Canonical set: **{c['rows']:,}** EN rows, {len(c['intents'])} v0 intents, "
        f"{c['families']:,} paraphrase families, **{c['scenarios']} scenarios**. Actions: "
        + ", ".join(f"{k} {v:,}" for k, v in c["actions"].items())
        + ".",
        "",
        "![volume vs diversity](figures/volume_vs_diversity.png)",
        "",
        "Definitions (ADR-022): *distinct texts* = after exact dedup (normalised template or "
        "final text); "
        "*family* = connected near-duplicates (char-5-gram Jaccard ≥ "
        f"{profile['params']['family_threshold']}) with the same label; *scenario* = same source "
        "label "
        "and required action (or a source-provided scenario id), joined across sources only via "
        "identical or near-identical texts with the same v0 intent and action.",
        "",
        "## 2. Sources",
        "",
    ]
    for sid, p in srcs.items():
        md += [_source_section(sid, p, d), ""]
    tk = profile["tokenizers"]
    if tk:
        md += [
            "Token lengths use "
            + ", ".join(f"`{t['repo_id']}@{t['revision'][:8]}` ({t['name']})" for t in tk)
            + ".",
            "",
        ]
        for t in tk:
            for field in ("text", "response", "dialogue"):
                if (out / "figures" / f"tokens_{field}_{t['name']}.png").is_file():
                    md.append(
                        f"![tokens {field} {t['name']}](figures/tokens_{field}_{t['name']}.png)"
                    )
        md.append("")
    md += ["![flags](figures/flags.png)", "", "## 3. Overlap between sources", ""]
    cs = profile["cross_source"]
    md.append(
        table(
            ["sources sharing an exact text", "groups"],
            [[k, v] for k, v in cs["exact_overlap_groups"].items()],
        )
        if cs["exact_overlap_groups"]
        else "No exact overlap."
    )
    md.append("")
    md.append(
        table(
            ["near-duplicate pairs across sources", "pairs"],
            [[k, v] for k, v in cs["near_dup_pairs"].items()],
        )
        if cs["near_dup_pairs"]
        else "No cross-source near-duplicates ≥ 0.5."
    )
    md += [
        "",
        f"Scenarios spanning two sources: {cs['scenarios_spanning_2_sources']}.",
        "",
        "## 4. Deduplication and grouping",
        "",
    ]
    md.append(
        table(
            [
                "J ≥",
                "same-label pairs",
                "different-label pairs (same source)",
                "cross-source pairs",
                "Bitext families",
                "largest",
                "% in families ≥ 100",
            ],
            [
                [
                    s["threshold"],
                    s["pairs_label_compatible"],
                    s["pairs_label_incompatible_same_source"],
                    s["pairs_cross_source"],
                    s["bitext_families"],
                    s["bitext_largest_family"],
                    f"{100 * s['bitext_share_in_families_ge100']:.1f}%",
                ]
                for s in profile["threshold_sweep"]
            ],
        )
    )
    md += [
        "",
        "![threshold sweep](figures/threshold_sweep.png)",
        "",
        "Highest-similarity pairs that the label guard kept apart (one example per label pair):",
        "",
        _blocked_examples(d),
        "",
        "LSH recall against brute force on a random Bitext sample: "
        + ", ".join(
            f"{k}: {v['found_by_lsh']}/{v['true_pairs']} ({100 * v['recall']:.1f}%)"
            for k, v in profile["lsh_recall"].items()
            if isinstance(v, dict)
        )
        + f" ({profile['lsh_recall']['sample_nodes']} texts, "
        f"{profile['lsh_recall']['pairs_checked']:,} pairs).",
        "",
        "![family sizes](figures/family_sizes.png)",
        "",
        "## 5. Language identification",
        "",
    ]
    lc = profile["lid_calibration"]
    if lc:
        md.append(
            "Calibration on Belebele (gold language per row): questions "
            + ", ".join(f"{k} {100 * v:.2f}%" for k, v in lc["questions_accuracy"].items())
            + "; passages "
            + ", ".join(f"{k} {100 * v:.2f}%" for k, v in lc["passages_accuracy"].items())
            + f". Confusions (questions): {lc['questions_confusion']}. Code-switch flagged on "
            f"{lc['questions_code_switch_flagged']} Belebele questions."
        )
    md += [
        "",
        "![lid](figures/lid_confidence.png)",
        "",
        "## 6. Quality gates (DATA_CONTRACTS §10)",
        "",
    ]
    md.append(
        table(
            ["gate", "status", "metric", "threshold", "details"],
            [
                [
                    g["name"],
                    g["status"],
                    g.get("metric"),
                    g.get("threshold"),
                    json.dumps(g["details"], ensure_ascii=False)[:260],
                ]
                for g in profile["gates"]["gates"]
            ],
        )
    )
    md += [
        "",
        "Split and leakage gates are not applicable yet: splits are created in Stage 4.",
        "",
        "![canonical intents](figures/canonical_intents.png)",
        "",
        "## 7. Qualitative review",
        "",
        _review_section(settings, profile, d),
        "",
    ]
    atomic_write_text(out / "report.md", "\n".join(md) + "\n")
    return n_figs
