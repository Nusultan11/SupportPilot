"""Gates, statistics and versioned publication of the human corpus (ADR-042, DATA_CONTRACTS §20).

Gates re-read the written files. They fail the build on: schema/ID problems, a non-train or
non-human-documented source in an active split, any forbidden origin, invalid labels, PII,
exact duplicates, a family/scenario in two splits, cross-split exact/lexical copies and
historical overlap (exact/lexical/13-gram recomputed here; semantic taken from the build).
Coverage of the 35 intents and of en/ru/kk/mixed is reported, never padded.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from supportpilot.common.hashing import sha256_text
from supportpilot.common.ids import make_dataset_version, make_example_id
from supportpilot.common.io import write_json
from supportpilot.common.manifest import read_model_json, verify_files
from supportpilot.common.schemas import INTENTS, Action, DatasetManifest, SourceEntry
from supportpilot.data.build import canonical_dir
from supportpilot.data.human import (
    ACTIVE_ORIGINS,
    FORBIDDEN_ORIGINS,
    HUMAN_LABEL_STATUSES,
    HUMAN_VERSION,
    LABEL_STATUS,
    REVIEW_STATUS,
    REVIEWED,
    SCHEMA,
    SPLITS,
    STAGE3_NATIVE,
    TARGET_STATUS,
    BuildInputs,
    HumanRecipe,
    build_rows,
    pii_hits,
    write_outputs,
)
from supportpilot.data.humangrounded import (
    GROUNDED_SCHEMA,
    GROUNDING_STATUS,
    build_grounded,
    grounded_stats,
    write_grounded,
)
from supportpilot.data.humangrounded import SPLITS as GROUNDED_SPLITS
from supportpilot.data.humangrounded import pii_hits as grounded_pii
from supportpilot.data.policy import PolicyError, Purpose, check_use
from supportpilot.data.split import exact_leaks, lexical_leaks, ngram_leaks
from supportpilot.data.steps import MANIFEST, StepError, commit, partial_dir, read_manifest
from supportpilot.data.textnorm import sim_key

LANGS = ("en", "ru", "kk")


def human_root(data_dir: Path) -> Path:
    return data_dir / "processed" / "human"


def _gate(name: str, ok: bool, metric: float, threshold: float, **details: Any) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if ok else "fail",
        "metric": metric,
        "threshold": threshold,
        "details": details,
    }


def load_active(build: Path) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(build / f"{s}.parquet") for s in SPLITS], how="vertical")


def run_gates(
    build: Path,
    entries: Sequence[SourceEntry],
    recipe: HumanRecipe,
    history: Sequence[str],
    build_report: dict[str, Any],
) -> dict[str, Any]:
    df = load_active(build)
    unv = pl.read_parquet(build / "unverified.parquet")
    rows = df.to_dicts()
    reg = {e.source_id: e for e in entries}
    gates: list[dict[str, Any]] = []

    bad_schema = [r["example_id"] for r in rows if r["schema_version"] != SCHEMA]
    bad_ids = [
        r["example_id"]
        for r in rows
        if make_example_id(r["source_id"], r["record_id"], r["lang"], r["origin"], r["text"])
        != r["example_id"]
    ]
    dup_ids = len(rows) - len({r["example_id"] for r in rows})
    gates.append(
        _gate(
            "schema_ids",
            not (bad_schema or bad_ids or dup_ids),
            len(bad_ids) + dup_ids,
            0,
            bad_schema=bad_schema[:5],
            bad_ids=bad_ids[:5],
            duplicate_ids=dup_ids,
        )
    )

    prov: list[str] = []
    for sid in sorted({r["source_id"] for r in rows}):
        a = recipe.authorship.get(sid)
        if sid not in reg or a is None or not a.active:
            prov.append(f"{sid}: not an active documented-human source")
            continue
        try:
            check_use(reg[sid], Purpose.TRAINING)
        except PolicyError as exc:
            prov.append(str(exc))
    origins = Counter(r["origin"] for r in rows)
    bad_origin = {
        o: n for o, n in origins.items() if o not in ACTIVE_ORIGINS or o in FORBIDDEN_ORIGINS
    }
    gates.append(
        _gate(
            "licence_provenance",
            not prov and not bad_origin,
            len(prov) + len(bad_origin),
            0,
            problems=prov,
            forbidden_origins=bad_origin,
            origins=dict(origins),
        )
    )

    actions = {a.value for a in Action}
    bad_label = [
        r["example_id"]
        for r in rows
        if r["intent"] not in INTENTS
        or r["action"] not in actions
        or r["escalate"] != (r["action"] == Action.ESCALATE.value)
        or (r["escalate"] and not r["escalation_reason"])
        or (not r["escalate"] and r["escalation_reason"])
    ]
    gates.append(_gate("labels", not bad_label, len(bad_label), 0, examples=bad_label[:5]))

    # human-reviewed labels only from a commissioned intake source; everything else is mapped
    bad_review = [
        r["example_id"]
        for r in rows
        if not (
            (r["label_status"] == LABEL_STATUS and r["review_status"] == REVIEW_STATUS)
            or (
                r["label_status"] in HUMAN_LABEL_STATUSES
                and r["review_status"] == REVIEWED
                and r["source_id"] in recipe.authorship
                and recipe.authorship[r["source_id"]].intake
            )
        )
    ]
    gates.append(
        _gate(
            "review_provenance",
            not bad_review,
            len(bad_review),
            0,
            examples=bad_review[:5],
            review_status=dict(Counter(r["review_status"] for r in rows)),
        )
    )

    misplaced = [
        r["example_id"]
        for r in rows
        if recipe.heldout_only(r["source_id"]) and r["split"] == "train"
    ]
    gates.append(
        _gate("heldout_placement", not misplaced, len(misplaced), 0, examples=misplaced[:5])
    )

    pii = [
        r["example_id"]
        for r in [*rows, *unv.to_dicts()]
        if pii_hits(r["text"]) or pii_hits(r["context_json"])
    ]
    gates.append(_gate("pii", not pii, len(pii), 0, examples=pii[:5]))

    keys = Counter(sim_key(r["text"]) for r in rows)
    dups = sum(n - 1 for n in keys.values() if n > 1)
    gates.append(_gate("duplicates", dups == 0, dups, 0))

    fam_split = df.group_by("family_id").agg(pl.col("split").n_unique().alias("n"))
    scn_split = df.group_by("scenario_id").agg(pl.col("split").n_unique().alias("n"))
    crossing = int((fam_split["n"] > 1).sum()) + int((scn_split["n"] > 1).sum())
    train = [r["text"] for r in rows if r["split"] == "train"]
    held = [r["text"] for r in rows if r["split"] != "train"]
    cs_exact = len(exact_leaks(train, held))
    cs_lex = len(lexical_leaks(train, held, threshold=recipe.leak_jaccard, seed=recipe.seed))
    gates.append(
        _gate(
            "split_isolation",
            crossing + cs_exact + cs_lex == 0,
            crossing + cs_exact + cs_lex,
            0,
            groups_in_two_splits=crossing,
            exact=cs_exact,
            lexical=cs_lex,
            semantic_removed=build_report.get("cross_split_semantic_removed"),
        )
    )

    texts = [r["text"] for r in [*rows, *unv.to_dicts()]]
    h_exact = len(exact_leaks(history, texts))
    h_lex = len(lexical_leaks(history, texts, threshold=recipe.leak_jaccard, seed=recipe.seed))
    h_ngram = len(ngram_leaks(history, texts, recipe.leak_ngram))
    sem_ok = bool(build_report.get("semantic_checked"))
    gates.append(
        _gate(
            "historical_leakage",
            h_exact + h_lex + h_ngram == 0 and sem_ok,
            h_exact + h_lex + h_ngram,
            0,
            exact=h_exact,
            lexical=h_lex,
            ngram=h_ngram,
            semantic_checked=sem_ok,
            historical_texts=len(history),
            removed_while_building=build_report.get("historical_overlap_removed"),
        )
    )

    targets = Counter(r["sft_target_status"] for r in rows)
    gates.append(
        _gate(
            "no_substituted_targets",
            set(targets) <= {TARGET_STATUS},
            0,
            0,
            sft_target_status=dict(targets),
        )
    )
    if recipe.grounded is not None:
        gates.extend(grounded_gates(build, entries, recipe, history, build_report))
    ok = all(g["status"] == "pass" for g in gates)
    return {"status": "pass" if ok else "fail", "gates": gates}


def grounded_gates(
    build: Path,
    entries: Sequence[SourceEntry],
    recipe: HumanRecipe,
    history: Sequence[str],
    build_report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Gates of the external grounded-response component (ADR-043)."""
    g = recipe.grounded
    assert g is not None
    df = pl.concat([pl.read_parquet(build / f"grounded_{s}.parquet") for s in GROUNDED_SPLITS])
    rows = df.to_dicts()
    unv = pl.read_parquet(build / "grounded_unverified.parquet").to_dicts()
    reg = {e.source_id: e for e in entries}
    out: list[dict[str, Any]] = []

    prov: list[str] = []
    try:
        check_use(reg[g.source_id], Purpose.TRAINING)
    except (KeyError, PolicyError) as exc:
        prov.append(str(exc))
    bad = [
        r["example_id"]
        for r in rows
        if r["schema_version"] != GROUNDED_SCHEMA
        or r["source_id"] != g.source_id
        or r["origin"] != g.origin
        or r["grounding_status"] != GROUNDING_STATUS
        or r["namespace"] != g.namespace
    ]
    ids = len(rows) - len({r["example_id"] for r in rows})
    out.append(
        _gate(
            "grounded_provenance_namespace",
            not prov and not bad and ids == 0,
            len(prov) + len(bad) + ids,
            0,
            problems=prov,
            bad_rows=bad[:5],
            duplicate_ids=ids,
            namespace=g.namespace,
        )
    )

    weak = []
    for r in rows:
        ev = json.loads(r["evidence_json"])
        if (
            not ev
            or any(not e.get("text") for e in ev)
            or not r["target"].strip()
            or not r["query"].strip()
            or r["evidence_recall"] < g.min_evidence_recall
        ):
            weak.append(r["example_id"])
    out.append(_gate("grounded_evidence", not weak, len(weak), 0, examples=weak[:5]))

    pii = [
        r["example_id"]
        for r in rows
        if grounded_pii(r["query"]) or grounded_pii(r["target"]) or grounded_pii(r["context_json"])
    ]
    out.append(_gate("grounded_pii", not pii, len(pii), 0, examples=pii[:5]))

    fam = df.group_by("family_id").agg(pl.col("split").n_unique().alias("n"))
    crossing = int((fam["n"] > 1).sum()) if fam.height else 0
    out.append(_gate("grounded_split_isolation", crossing == 0, crossing, 0, documents=crossing))

    texts = [r["query"] for r in [*rows, *unv]] + [r["target"] for r in [*rows, *unv]]
    h_exact = len(exact_leaks(history, texts))
    h_lex = len(lexical_leaks(history, texts, threshold=recipe.leak_jaccard, seed=recipe.seed))
    h_ngram = len(ngram_leaks(history, texts, recipe.leak_ngram))
    sem_ok = bool(build_report.get("grounded", {}).get("semantic_checked"))
    out.append(
        _gate(
            "grounded_historical_leakage",
            h_exact + h_lex + h_ngram == 0 and sem_ok,
            h_exact + h_lex + h_ngram,
            0,
            exact=h_exact,
            lexical=h_lex,
            ngram=h_ngram,
            semantic_checked=sem_ok,
        )
    )
    return out


def corpus_stats(
    build: Path, build_report: dict[str, Any], previous: Path | None
) -> dict[str, Any]:
    df = load_active(build)
    unv = pl.read_parquet(build / "unverified.parquet")
    exc = pl.read_parquet(build / "excluded.parquet")
    total_in = sum(build_report["read"].values())
    covered: dict[str, list[str]] = {}
    for lang in LANGS:
        covered[lang] = sorted(set(df.filter(pl.col("lang") == lang)["intent"].to_list()))
    # RU-KK mixing only; LID code-switch flags on RU/EN names are reported separately
    mixed = df.filter(
        pl.col("code_switch")
        & pl.col("langs").list.contains("ru")
        & pl.col("langs").list.contains("kk")
    )
    stats: dict[str, Any] = {
        "input_records": build_report["read"],
        "input_total": total_in,
        "active": df.height,
        "unverified": unv.height,
        "excluded": exc.height,
        "excluded_share": round(exc.height / total_in, 4) if total_in else None,
        "not_active_share": round((exc.height + unv.height) / total_in, 4) if total_in else None,
        "by_split": {
            s: {
                "examples": int((df["split"] == s).sum()),
                "families": df.filter(pl.col("split") == s)["family_id"].n_unique(),
                "scenarios": df.filter(pl.col("split") == s)["scenario_id"].n_unique(),
                "lang": dict(Counter(df.filter(pl.col("split") == s)["lang"].to_list())),
            }
            for s in SPLITS
        },
        "independent_scenarios": df["scenario_id"].n_unique(),
        "families": df["family_id"].n_unique(),
        "lang": dict(sorted(Counter(df["lang"].to_list()).items())),
        "mixed_ru_kk": mixed.height,
        "code_switch_flagged": int(df["code_switch"].sum()),
        "origin": dict(sorted(Counter(df["origin"].to_list()).items())),
        "source": dict(sorted(Counter(df["source_id"].to_list()).items())),
        "action": dict(sorted(Counter(df["action"].to_list()).items())),
        "intent": dict(sorted(Counter(df["intent"].to_list()).items())),
        "lang_x_intent": dict(
            sorted(
                Counter(f"{a}|{b}" for a, b in zip(df["lang"], df["intent"], strict=True)).items()
            )
        ),
        "intents_covered": len(set(df["intent"].to_list())),
        "intents_missing": sorted(set(INTENTS) - set(df["intent"].to_list())),
        "intents_missing_by_lang": {
            lang: sorted(set(INTENTS) - set(covered[lang])) for lang in LANGS
        },
        "held_out_intents": {
            s: len(set(df.filter(pl.col("split") == s)["intent"].to_list())) for s in SPLITS[1:]
        },
        "multi_turn_context": int((df["context_json"] != "[]").sum()),
        "unverified_reasons": dict(sorted(Counter(unv["reason"].to_list()).items())),
        "excluded_reasons": dict(sorted(Counter(exc["reason"].to_list()).items())),
        "sft_target_status": dict(Counter(df["sft_target_status"].to_list())),
        "label_status": dict(Counter(df["label_status"].to_list())),
        "review_status": dict(Counter(df["review_status"].to_list())),
        "independently_reviewed_share": (
            round(float((df["review_status"] == REVIEWED).sum()) / df.height, 4)
            if df.height
            else None
        ),
        "build": build_report,
    }
    if (build / "grounded_train.parquet").is_file():
        stats["grounded"] = grounded_stats(build, build_report.get("grounded", {}))
    if previous is not None and previous.is_dir():
        prev = pl.concat(
            [
                pl.read_parquet(
                    previous / f"{s}.parquet",
                    columns=["origin", "source_id", "lang", "user_text", "task"],
                )
                for s in SPLITS
            ],
            how="vertical",
        )
        stats["previous_active"] = {
            "dataset_version": previous.name,
            "examples": prev.height,
            "distinct_user_texts": prev["user_text"].n_unique(),
            "origin": dict(sorted(Counter(prev["origin"].to_list()).items())),
            "source": dict(sorted(Counter(prev["source_id"].to_list()).items())),
            "lang": dict(sorted(Counter(prev["lang"].to_list()).items())),
            "task": dict(sorted(Counter(prev["task"].to_list()).items())),
            "human_documented_share": round(
                float(prev["source_id"].is_in(["massive_en_ru"]).sum()) / prev.height, 4
            ),
        }
    return stats


def build_human(
    inp: BuildInputs,
    *,
    code_commit: str | None,
    recipe_sha256: str,
    taxonomy_sha256: str,
    previous: Path | None,
) -> dict[str, Any]:
    recipe = inp.recipe
    result = build_rows(inp)
    history = result.history
    root = human_root(inp.data_dir)
    sources = sorted(
        f"{e.source_id}@{e.revision}" for e in inp.entries if e.source_id in recipe.authorship
    )
    inputs = list(sources)
    if any(sid in STAGE3_NATIVE for sid in recipe.authorship):  # rows come from Stage 3
        inputs.append(f"canonical@{read_manifest(canonical_dir(inp.data_dir)).dataset_version}")
    params = {
        "human_version": HUMAN_VERSION,
        "recipe_sha256": recipe_sha256,
        "taxonomy_sha256": taxonomy_sha256,
        "seed": recipe.seed,
        "history_files": [p.relative_to(inp.data_dir).as_posix() for p in inp.history_paths],
        "semantic_checked": inp.encode is not None,
    }
    grounded = None
    g = recipe.grounded
    if g is not None:
        by_id = {e.source_id: e for e in inp.entries}
        if g.source_id not in by_id:
            raise StepError(f"grounded source {g.source_id} is not in the registry")
        check_use(by_id[g.source_id], Purpose.TRAINING)
        grounded = build_grounded(
            by_id[g.source_id],
            inp.data_dir,
            dict(g.acts),
            g.min_evidence_recall,
            dict(recipe.split_ratios),
            recipe.seed,
            history=history,
            leak_jaccard=recipe.leak_jaccard,
            leak_ngram=recipe.leak_ngram,
            leak_cosine=recipe.leak_cosine,
            encode=inp.encode,
        )
        result.report["grounded"] = grounded.report
        inputs.append(f"{g.source_id}@{by_id[g.source_id].revision}")
    with partial_dir(root / "build") as partial:
        out = write_outputs(partial, result, recipe)
        content = out["content_sha256"]
        if grounded is not None and g is not None:
            gout = write_grounded(
                partial,
                grounded,
                source_id=g.source_id,
                origin=g.origin,
                evidence_text=g.evidence,
                namespace=g.namespace,
            )
            content = sha256_text(f"{content}|grounded|{gout['content_sha256']}")
            out["row_counts"]["grounded"] = len(gout["active"])
        version = make_dataset_version("human", datetime.now(UTC).date(), content)
        final = root / version
        if (final / MANIFEST).is_file():
            m = read_model_json(final / MANIFEST, DatasetManifest)
            if problems := verify_files(final, m.files, exclude=frozenset({MANIFEST})):
                raise StepError(f"{final} exists but fails verification: {problems[0]}")
            return {"status": "skipped", "dataset_version": version, "path": str(final)}
        report = run_gates(partial, inp.entries, recipe, history, result.report)
        write_json(partial / "quality_report.json", report)
        stats = corpus_stats(partial, result.report, previous)
        write_json(partial / "stats.json", stats)

        passed = bool(report["status"] == "pass")

        def gate(_p: Path) -> bool:
            return passed

        m = commit(
            partial,
            final,
            name="human",
            data_schema_version=SCHEMA,
            code_commit=code_commit,
            inputs=inputs,
            params=params,
            row_counts=out["row_counts"],
            content_sha256=content,
            quality_report="quality_report.json",
            gate=gate,
        )
    return {
        "status": "built",
        "dataset_version": m.dataset_version,
        "path": str(final),
        "gates": report["status"],
        "active": stats["active"],
        "unverified": stats["unverified"],
        "excluded": stats["excluded"],
        "lang": stats["lang"],
        "intents_covered": stats["intents_covered"],
        "grounded": stats.get("grounded", {}).get("active"),
    }


def verify_human(build: Path) -> dict[str, Any]:
    """Checksum verification of a published human corpus version."""
    m = read_model_json(build / MANIFEST, DatasetManifest)
    problems = verify_files(build, m.files, exclude=frozenset({MANIFEST}))
    report = json.loads((build / "quality_report.json").read_text(encoding="utf-8"))
    return {
        "dataset_version": m.dataset_version,
        "files": len(m.files),
        "checksums_ok": not problems,
        "problems": [p.model_dump() for p in problems],
        "gates": report["status"],
        "content_id": sha256_text("".join(f.sha256 for f in m.files))[:16],
    }
