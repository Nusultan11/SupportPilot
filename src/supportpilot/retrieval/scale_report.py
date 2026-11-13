"""Source-bound Stage 6.1 report and complete, overlapping error census."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from supportpilot.common.config import Settings
from supportpilot.common.hashing import sha256_file
from supportpilot.common.io import write_json
from supportpilot.common.tracking import RunSpec, start_run, tracking_uri
from supportpilot.data.retrievalset import RetrievalQuery
from supportpilot.retrieval.scale_data import verify_freeze
from supportpilot.retrieval.scale_metrics import paired_scale, summarise_scale


def error_labels(row: dict[str, Any]) -> list[str]:
    """Overlapping observable failure modes, never inferred human error causes."""
    if not row["answerable"]:
        return ["negative_returned" if row["unanswerable_fpr"] else "negative_abstained"]
    labels = []
    if row["recall5"] < 1:
        labels.append("qrel_missing_at5")
    if row["full_evidence10"] < 1:
        labels.append("required_group_missing_at10")
    if row["packed_sufficiency"] < 1:
        labels.append("packed_evidence_insufficient")
        if row["full_evidence10"] == 1:
            labels.append("packing_lost_sufficient_ranking")
    if row["answerable_fnr"]:
        labels.append("answerable_abstained")
    return labels or ["all_observed_criteria_pass"]


def interval(metric: dict[str, Any]) -> str:
    if metric["mean"] is None:
        return "UNKNOWN"
    if metric.get("scenarios", 0) < 2:
        return f"{metric['mean']:.4f} [CI UNKNOWN: fewer than 2 clusters]"
    return f"{metric['mean']:.4f} [{metric['low']:.4f}, {metric['high']:.4f}]"


def report_scale(settings: Settings, frozen: Path) -> dict[str, Any]:
    import mlflow

    verify_freeze(frozen)
    audit = json.loads((frozen / "audit.json").read_text())
    protocol = json.loads((frozen / "protocol.json").read_text())
    metadata = json.loads((frozen / "metadata.json").read_text())
    queries = {
        q.query_id: q
        for q in (
            RetrievalQuery.model_validate_json(line)
            for line in (frozen / "queries.jsonl").read_text("utf-8").splitlines()
        )
    }
    root = settings.resolve(Path(protocol["output"])) / frozen.name
    selection = json.loads((root / "selection.json").read_text())
    if selection["freeze_sha256"] != sha256_file(frozen / "freeze.json"):
        raise ValueError("selection freeze mismatch")
    if selection["development_sha256"] != sha256_file(root / "development/results.json"):
        raise ValueError("sealed development changed")
    name = selection["selected"]
    results, census, examples = {}, {}, []
    diagnostic: dict[str, Any] = {}
    phase_runs: dict[str, str] = {}
    review_sample = []
    bound = {"freeze.json": sha256_file(frozen / "freeze.json")}
    for phase in ("development", "holdout"):
        path = root / phase
        receipt = json.loads((path / "results.json").read_text())
        attempt = json.loads((path / "attempt.json").read_text())
        if attempt["status"] != "FINISHED" or receipt["kb_version"] != audit["kb_version"]:
            raise ValueError("phase incomplete or wrong KB")
        rows = json.loads((path / "observations.json").read_text())[name]
        expected = {qid for qid, meta in metadata.items() if meta["phase"] == phase}
        if {row["query_id"] for row in rows} != expected or len(rows) != len(expected):
            raise ValueError("incomplete selected population")
        results[phase] = receipt["evaluations"]
        phase_runs[phase] = receipt["run_id"]
        counts: Counter[str] = Counter()
        sliced: dict[str, Counter[str]] = defaultdict(Counter)
        samples: Counter[str] = Counter()
        reviewed_needs: dict[str, set[str]] = defaultdict(set)
        for row in sorted(rows, key=lambda row: row["query_id"]):
            labels = error_labels(row)
            counts.update(labels)
            meta, query = metadata[row["query_id"]], queries[row["query_id"]]
            if (
                row["phase"] != phase
                or row["kb_version"] != audit["kb_version"]
                or row["benchmark_version"] != audit["input_identity"]
                or row["answerable"] != query.answerable
                or row["family_id"] != meta["family_id"]
            ):
                raise ValueError("observation lineage disagrees with frozen query metadata")
            if (
                phase == "holdout"
                and query.answerable
                and (row["recall5"] < 1 or row["packed_sufficiency"] < 1)
                and len(reviewed_needs[query.lang]) < 50
                and query.scenario_id not in reviewed_needs[query.lang]
            ):
                reviewed_needs[query.lang].add(query.scenario_id)
                review_sample.append(
                    {
                        "query": query.model_dump(mode="json"),
                        "observation": row,
                        "metadata": meta,
                        "labels": labels,
                    }
                )
            for scope in row["slices"]:
                sliced[scope].update(labels)
            for label in labels:
                key = f"{label}|{query.lang}"
                if samples[key] >= 5 or label in (
                    "all_observed_criteria_pass",
                    "negative_abstained",
                ):
                    continue
                samples[key] += 1
                examples.append(
                    {
                        "phase": phase,
                        "label": label,
                        "query": query.model_dump(mode="json"),
                        "metadata": meta,
                        "observation": row,
                        "missing_packed_groups": [
                            group
                            for group in meta["required_groups"]
                            if not set(group) & set(row["packed"])
                        ],
                    }
                )
        census[phase] = {
            "queries": len(rows),
            "answerable": sum(row["answerable"] for row in rows),
            "unanswerable": sum(not row["answerable"] for row in rows),
            "overlapping_labels": dict(counts),
            "slice_labels": {scope: dict(values) for scope, values in sliced.items()},
            "scope": (
                "Observable retrieval/support failures; not independent human root-cause review."
            ),
        }
        diagnostic[phase] = {}
        for label, use_profile in (("profile", True), ("non_profile", False)):
            subset = [row for row in rows if ("profile_derived" in row["slices"]) == use_profile]
            diagnostic[phase][label] = summarise_scale(subset, seed=protocol["seed"])
        for filename in ("results.json", "observations.json", "attempt.json"):
            bound[f"{phase}/{filename}"] = sha256_file(path / filename)
    bound["selection.json"] = sha256_file(root / "selection.json")
    holdout_rows = json.loads((root / "holdout/observations.json").read_text())
    paired_holdout = paired_scale(
        holdout_rows[selection["control"]],
        holdout_rows[name],
        seed=protocol["seed"],
    )
    lines = [
        "# Stage 6.1: multilingual retrieval evaluation",
        "",
        f"KB: `{audit['kb_version']}`. Frozen benchmark: `{frozen.name}`.",
        f"Selected on development: `{name}`; control: `{selection['control']}`.",
        "",
        "All judgements are LLM-authored drafts. No independent human/fluent KK review. "
        "Most needs target synthetic specifications with correlated templates; family sensitivity "
        "is reported separately. Counts do not certify natural customer diversity.",
        "",
        "Intervals below are 95% scenario-cluster bootstrap (1,000 draws). "
        "Ranking and proof metrics use answerable queries; FPR uses negatives only; "
        "FNR means empty packed context on answerable queries.",
        "",
        "| Phase / slice | queries / needs / families | Recall@5 | MRR@10 | nDCG@10 | "
        "full evidence@10 | packed sufficiency | FPR | FNR |",
        "|---|---:|---|---|---|---|---|---|---|",
    ]
    scopes = (
        "all",
        "ru",
        "kk",
        "en",
        "cross_lingual",
        "code_switch",
        "rare_document",
        "multi_document",
        "multi_fragment",
        "temporal",
        "hard_negative",
        "pair_ru_kk",
        "pair_kk_ru",
        "pair_ru_en",
        "pair_en_ru",
        "pair_kk_en",
        "pair_en_kk",
    )
    metrics = (
        "recall5",
        "mrr10",
        "ndcg10",
        "full_evidence10",
        "packed_sufficiency",
        "unanswerable_fpr",
        "answerable_fnr",
    )
    for phase, evaluations in results.items():
        selected = evaluations[name]
        for scope in scopes:
            if scope not in selected:
                continue
            row = selected[scope]
            lines.append(
                f"| {phase}/{scope} | {row['queries']}/{row['scenarios']}/"
                f"{row['families']} | "
                + " | ".join(interval(row["metrics"][metric]) for metric in metrics)
                + " |"
            )
    lines.extend(
        [
            "",
            "## Family-cluster sensitivity on the selected holdout",
            "",
            "With fewer than two clusters, between-cluster uncertainty is not identifiable. "
            "The table marks that CI UNKNOWN; raw degenerate bootstrap bounds remain in JSON "
            "for reproducibility and are not evidence of generalization precision.",
            "",
            "| Slice | Recall@5 | packed sufficiency | FPR |",
            "|---|---|---|---|",
        ]
    )
    for scope in scopes:
        row = results["holdout"][name].get(scope)
        if row:
            lines.append(
                f"| {scope} | "
                + " | ".join(
                    interval(row["family_sensitivity"][metric])
                    for metric in ("recall5", "packed_sufficiency", "unanswerable_fpr")
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Development matrix (selection inputs only)",
            "",
            "| Configuration | RU Recall@5 | KK Recall@5 | EN Recall@5 | FPR | Guard failures |",
            "|---|---|---|---|---|---|",
        ]
    )
    for key, evaluation in results["development"].items():
        guards = selection["comparisons"].get(key, {}).get("guard_failures", [])
        columns = [interval(evaluation[lang]["metrics"]["recall5"]) for lang in ("ru", "kk", "en")]
        columns.append(interval(evaluation["all"]["metrics"]["unanswerable_fpr"]))
        lines.append(f"| {key} | " + " | ".join(columns) + " | " + "; ".join(guards) + " |")
    lines.extend(
        [
            "",
            "## Paired selected-minus-control on the untouched holdout",
            "",
            "| Slice | Recall@5 delta | nDCG@10 delta | packed sufficiency delta | FPR delta |",
            "|---|---|---|---|---|",
        ]
    )
    for scope in scopes:
        if scope in paired_holdout:
            lines.append(
                f"| {scope} | "
                + " | ".join(
                    interval(paired_holdout[scope]["metrics"][metric])
                    for metric in ("recall5", "ndcg10", "packed_sufficiency", "unanswerable_fpr")
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Abstention proxy on the selected holdout",
            "",
            "FPR means a nonempty packed context on a negative query. FNR means an empty "
            "context on an answerable query. Accuracy does not test whether a generated "
            "answer is correct, and can hide class imbalance; proof sufficiency is separate.",
            "",
            "| Slice | negative / answerable queries | FPR | FNR | abstention accuracy |",
            "|---|---:|---|---|---|",
        ]
    )
    for scope in ("all", "ru", "kk", "en", "cross_lingual", "code_switch"):
        metric = results["holdout"][name][scope]["metrics"]
        lines.append(
            f"| {scope} | {metric['unanswerable_fpr']['queries']}/"
            f"{metric['answerable_fnr']['queries']} | "
            + " | ".join(
                interval(metric[key])
                for key in ("unanswerable_fpr", "answerable_fnr", "abstention_accuracy")
            )
            + " |"
        )
    historical_root = settings.resolve(Path(protocol["baseline_report"]))
    historical_path = historical_root / "online_context_metrics.json"
    historical = json.loads(historical_path.read_text())
    bound["historical_report_manifest"] = sha256_file(historical_root / "manifest.json")
    bound["historical_context_metrics"] = sha256_file(historical_path)
    lines.extend(
        [
            "",
            "## Preserved Stage 6 results (different cohort, already exposed)",
            "",
            f"KB `{protocol['baseline_kb']}`, benchmark `{protocol['baseline_benchmark']}`. "
            "These are the original packed-context holdout metrics, not a paired comparison "
            "with the new benchmark and not fresh confirmation.",
            "",
            "| Slice | Recall@5 | MRR@10 | nDCG@10 |",
            "|---|---|---|---|",
        ]
    )
    for scope in ("ru", "kk", "en", "cross_lingual", "code_switch"):
        lines.append(
            f"| {scope} | "
            + " | ".join(
                interval(historical["holdout"][scope][metric])
                for metric in ("recall5", "mrr10", "ndcg10")
            )
            + " |"
        )
    profile_path = root / "profile/receipt.json"
    profile = json.loads(profile_path.read_text())
    for filename in (
        "receipt.json",
        "selected_online.json",
        "encoding_probe.json",
        "fixed_work.json",
        "server_memory.json",
    ):
        bound["profile/" + filename] = sha256_file(root / "profile" / filename)
    lines.extend(
        [
            "",
            "## Matched corpus-growth diagnostic",
            "",
            "Identical 681 historical variants, same uncached control and code; all phases "
            "of this previously exposed cohort are descriptive diagnostics only.",
            "",
            "| Slice | 369-chunk Recall@5 | 8865-chunk Recall@5 | paired growth delta |",
            "|---|---|---|---|",
        ]
    )
    for scope in ("ru", "kk", "en", "cross_lingual", "code_switch"):
        values = [
            interval(profile["results"][label]["metrics"]["full"][scope]["recall5"])
            for label in ("baseline", "expanded")
        ]
        values.append(interval(profile["paired_growth"][scope]["recall5"]))
        lines.append(f"| {scope} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Uncached online performance on this laptop",
            "",
            "| Workload | p50 ms | p95 ms | sequential requests/s |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, value in {
        **profile["results"],
        "selected_new_development_80": profile["selected_online"],
    }.items():
        lines.append(
            f"| {label} | {value['p50_ms']:.3f} | {value['p95_ms']:.3f} | "
            f"{value['sequential_qps']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The first two workloads share the historical query set; the 80-query "
            "selected probe uses balanced new development variants and is not paired with "
            "them. Model-load, component/language latency, process RAM/VRAM, sampled Qdrant "
            "cgroup memory including file cache, disk and full uncached encoding rates are "
            "in profile/receipt.json. No concurrency or production SLO is inferred.",
        ]
    )
    scope_path = root / "profile/measurement_scope_audit.json"
    if scope_path.is_file():
        scope_audit = json.loads(scope_path.read_text())
        bound["profile/measurement_scope_audit.json"] = sha256_file(scope_path)
        lines.extend(
            [
                "",
                "## Storage counter definitions and online parity",
                "",
                scope_audit["storage_scope"],
                "",
                "| Collection | live points | apparent bytes | allocated bytes |",
                "|---|---:|---:|---:|",
            ]
        )
        for label, value in scope_audit["storage"].items():
            lines.append(
                f"| {label} | {value['points']} | {value['apparent_bytes']} | "
                f"{value['allocated_bytes']} |"
            )
        lines.extend(
            [
                "",
                f"Cached/uncached selected development parity: "
                f"{scope_audit['cached_online_queries']} queries; "
                f"{scope_audit['ranking_differences']} ranking differences and "
                f"{scope_audit['packing_differences']} packing differences. "
                "This is a sampled check, not bitwise model reproducibility on every input. "
                "Full uncached document encoding drift is reported separately.",
            ]
        )
    for diagnostic_path in sorted((root / "diagnostics").glob("*/results.json")):
        additional = json.loads(diagnostic_path.read_text())
        status = (
            mlflow.MlflowClient(tracking_uri=tracking_uri(settings))
            .get_run(additional["run_id"])
            .info.status
        )
        if status != "FINISHED" or additional["selected"] != name:
            raise ValueError("wording diagnostic is incomplete or used another selection")
        verify_freeze(Path(additional["frozen"]))
        bound[diagnostic_path.relative_to(root).as_posix()] = sha256_file(diagnostic_path)
        lines.extend(
            [
                "",
                "## Post-selection development wording diagnostic",
                "",
                f"{additional['queries']} variants reuse "
                f"{additional['existing_development_needs']} existing development needs; "
                "zero new independent needs. 20 domains, five related families; "
                "draft native paraphrases and internal one-letter entity deletions. "
                "No holdout needs, tuning or fresh confirmation claim. Cross-language "
                "means below use a different diagnostic cohort from the primary benchmark.",
                "",
                "| Slice | variants / needs / families | Recall@5 | nDCG@10 | packed sufficiency |",
                "|---|---:|---|---|---|",
            ]
        )
        for scope in (
            "ru",
            "kk",
            "en",
            "cross_lingual",
            "code_switch",
            "diagnostic_paraphrase",
            "diagnostic_entity_typo",
        ):
            row = additional["summary"][scope]
            lines.append(
                f"| {scope} | {row['queries']}/{row['scenarios']}/{row['families']} | "
                + " | ".join(
                    interval(row["metrics"][m]) for m in ("recall5", "ndcg10", "packed_sufficiency")
                )
                + " |"
            )
        lines.extend(
            ["", "| Paired wording change | Recall@5 delta | nDCG@10 delta |", "|---|---|---|"]
        )
        for scope in ("diagnostic_paraphrase", "diagnostic_entity_typo"):
            row = additional["paired_wording_delta"][scope]["metrics"]
            lines.append(f"| {scope} | {interval(row['recall5'])} | {interval(row['ndcg10'])} |")
        lines.extend(
            [
                "",
                additional["pairing"],
                "",
                "All supplemental variants are answerable; they do not establish robustness "
                "of abstention. SFT/gold decontamination and the source/target plan are bound "
                "in the diagnostic freeze and diagnostic_plan.json.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Selection and all negative development alternatives are retained in selection.json. "
            "A nonempty context does not prove answerability: inspect FPR and required-group "
            "sufficiency separately. Semantic abstention remains unresolved unless those "
            "measurements support it. No production quality or throughput claim follows.",
            "",
            "The matched historical-cohort growth/latency diagnostic is in profile/receipt.json; "
            "the previous Stage 6 holdout was already exposed. Absolute old/new benchmark "
            "means are not paired evidence because their cohorts differ.",
            "",
            "error_census.json enumerates the whole selected population. examples.json is "
            "a deterministic inspection sample, not a complete independent annotation.",
        ]
    )
    out = root / "report"
    out.mkdir(exist_ok=True)
    write_json(out / "error_census.json", census)
    write_json(out / "examples.json", examples)
    write_json(out / "template_sensitivity.json", diagnostic)
    write_json(out / "paired_holdout.json", paired_holdout)
    write_json(out / "qualitative_sample.json", review_sample)
    (out / "comparison.md").write_text("\n".join(lines) + "\n", "utf-8")
    spec = RunSpec(
        experiment="retrieval_scale",
        run_name=f"{frozen.name}_report",
        hypothesis="Account for every observation; distinguish relevance and sufficient proof",
        dataset_version=frozen.name,
        seed=protocol["seed"],
        base_model=name,
        control_run_id=phase_runs["development"],
        params={
            "kb_version": audit["kb_version"],
            "selected": name,
            "control_configuration": selection["control"],
            "inputs": json.dumps(bound, sort_keys=True),
            "report_code_sha256": sha256_file(Path(__file__)),
        },
    )
    with start_run(spec, settings) as run:
        for source in (
            root / "selection.json",
            frozen / "protocol.json",
            frozen / "audit.json",
            frozen / "freeze.json",
        ):
            mlflow.log_artifact(str(source), artifact_path="inputs")
        for path in out.iterdir():
            if path.is_file() and path.name != "receipt.json":
                mlflow.log_artifact(str(path))
        receipt = {
            "run_id": run.info.run_id,
            "inputs": bound,
            "selected": name,
            "files": {
                p.name: sha256_file(p)
                for p in out.iterdir()
                if p.is_file() and p.name != "receipt.json"
            },
        }
        write_json(out / "receipt.json", receipt)
        mlflow.log_artifact(str(out / "receipt.json"))
        mlflow.set_tags({"decision": "source_bound_report", "cost_usd": "0"})
    return {"path": str(out), "run_id": receipt["run_id"], "selected": name}
