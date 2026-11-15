"""Recompute Stage 9 conclusions from retained observations, never success counts alone."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from supportpilot.common.config import load_settings
from supportpilot.common.hashing import sha256_file
from supportpilot.common.io import read_json, write_json
from supportpilot.common.schemas import INTENTS
from supportpilot.common.tracking import RunSpec, start_run
from supportpilot.data.taxonomy import load_taxonomy
from supportpilot.service.evaluation import SCOPES, paired, root, summary, verify_freeze
from supportpilot.service.semantic_evaluation import evaluate_reviews


def f1_uncertainty(rows: list[dict[str, Any]], group: str) -> dict[str, Any]:
    """Cluster the full confusion matrix, including missing predictions and absent classes."""
    labels = sorted(INTENTS)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[group]].append(row)
    if len(groups) < 2:
        return {"low": None, "high": None, "clusters": len(groups)}
    confusion = []
    for cohort in groups.values():
        matrix = np.zeros((35, 36))  # final column is an invalid/missing prediction
        for row in cohort:
            predicted = row.get("raw_intent")
            matrix[
                labels.index(row["intent"]), labels.index(predicted) if predicted in labels else 35
            ] += 1
        confusion.append(matrix)
    arrays = np.stack(confusion)
    draws = np.random.default_rng(20261115).integers(len(groups), size=(1000, len(groups)))
    values = []
    for draw in draws:
        matrix = arrays[draw].sum(axis=0)
        tp = np.diag(matrix[:, :35])
        denominator = matrix.sum(axis=1) + matrix[:, :35].sum(axis=0)
        values.append(
            float(np.divide(2 * tp, denominator, out=np.zeros(35), where=denominator > 0).mean())
        )
    return {
        "low": float(np.quantile(values, 0.025)),
        "high": float(np.quantile(values, 0.975)),
        "clusters": len(groups),
        "limitation": "fixed35 labels; sparse bootstrap may omit rare classes; exploratory",
    }


def wilson(success: int, count: int) -> list[float] | None:
    if not count:
        return None
    z = 1.959963984540054
    p = success / count
    center = p + z * z / (2 * count)
    width = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count))
    denominator = 1 + z * z / count
    return [
        max(0.0, float((center - width) / denominator)),
        min(1.0, float((center + width) / denominator)),
    ]


def failure_cause(row: dict[str, Any]) -> str:
    if row.get("http_status") != 200:
        return "http_or_execution"
    result = row["result"]
    reason = result["response"].get("escalation_reason")
    if row["critical"] and not row["queue_verified"]:
        return "critical_recognition_or_action"
    if row["expected_delivery"] == "tool" and reason == "independent_safety_guard":
        return "guard_false_escalation"
    if row["expected_delivery"] == "queued" and result["delivery"] == "clarification":
        return "routing_or_missing_action_integration"
    if row.get("raw_intent") != row["intent"]:
        return "intent_routing"
    return "delivery_action"


def main() -> None:
    import mlflow

    settings = load_settings()
    directory = root(settings)
    frozen_path = Path(read_json(directory / "freeze_pointer.json")["path"])
    frozen = verify_freeze(frozen_path)
    control_path = Path(read_json(directory / "holdout_stage8.json")["attempt"]) / "report.json"
    candidate_path = (
        Path(read_json(directory / "holdout_boundary_guard.json")["attempt"]) / "report.json"
    )
    control, candidate = read_json(control_path), read_json(candidate_path)
    if any(r["status"] != "FINISHED" or len(r["rows"]) != 400 for r in (control, candidate)):
        raise ValueError("incomplete confirmation accounting")
    if control["freeze_identity"] != candidate["freeze_identity"]:
        raise ValueError("confirmation identities disagree")
    taxonomy = load_taxonomy(settings.resolve(settings.stage4.taxonomy))
    scopes: dict[str, Any] = {}
    error_rows: list[dict[str, Any]] = []
    for report in (control, candidate):
        # Derive a common action rubric from original raw routes. Historical development
        # reports used the noncritical taxonomy action on critical rows; retain those files.
        for row in report["rows"]:
            expected = "escalate" if row["critical"] else taxonomy.intents[row["intent"]].action
            if expected == "needs_reference":
                expected = (
                    "call_tool"
                    if any(
                        ref in row["text"] for ref in ("SP-1001", "SP-1002", "SP-9999", "TV-5501")
                    )
                    else "clarify"
                )
            row["raw_action_correct"] = float(row.get("raw_route", {}).get("action") == expected)
        recomputed = summary(report["rows"])
        for scope in SCOPES:
            cohort = [r for r in report["rows"] if r["scope"] == scope]
            values = recomputed[scope]
            values["macro_f1_family_ci95"] = f1_uncertainty(cohort, "family_id")
            values["macro_f1_scenario_ci95"] = f1_uncertainty(cohort, "scenario_id")
            critical = [r for r in cohort if r["critical"]]
            values["critical_delivery_wilson_sensitivity"] = wilson(
                sum(int(r["queue_verified"]) for r in critical), len(critical)
            )
            followups = [r for r in cohort if r["scenario_id"] in {"h01", "h07", "h08", "h46"}]
            values["followup_delivered_tools"] = sum(
                int(r.get("followup_correct", 0)) for r in followups
            )
            values["followup_cases"] = len(followups)
            failures = [r for r in cohort if not r["delivery_correct"]]
            values["failure_components"] = dict(Counter(failure_cause(r) for r in failures))
            values["queue_metric_scope"] = (
                "queue_verified/all rows is a fraction; critical recall uses required critical rows"
            )
            values["targets"] = {
                "intent_macro_f1_090": (
                    "POINT_TARGET_MET" if values["intent_macro_f1_35"] >= 0.9 else "NOT_MET"
                ),
                "critical_recall_095": "POINT_TARGET_MET"
                if values["critical_delivery_recall"]["mean"] >= 0.95
                else "NOT_MET",
                "critical_generalization": "NOT_ESTABLISHED",
                "semantic_citation_precision_095": "NOT_EVALUATED",
            }
            if report["variant"] == "boundary_guard":
                error_rows.extend(
                    {
                        "scope": scope,
                        "need_id": r["scenario_id"],
                        "cause": failure_cause(r),
                        "query": r["text"],
                        "raw_route": r.get("raw_route"),
                        "result": r.get("result"),
                    }
                    for r in failures[:50]
                )
        scopes[report["variant"]] = recomputed
    attempts = []
    for path in sorted(directory.glob("attempts/*/report.json")):
        record = read_json(path)
        attempts.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "status": record["status"],
                "run_id": record.get("run_id") or record.get("receipt", {}).get("run_id"),
                "phase": record["phase"],
                "variant": record["variant"],
                "smoke": record["smoke"],
            }
        )
    performance = []
    for path in sorted(directory.glob("performance/*/report.json")):
        record = read_json(path)
        performance.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "status": record["status"],
                "action": record["action"],
                "run_id": record.get("run_id"),
            }
        )
    supplemental: dict[str, list[dict[str, Any]]] = {}
    for category, pattern in {
        "semantic_overlap": "semantic_overlap/*/report.json",
        "retrieval": "retrieval/*/report.json",
        "stress": "stress/*/report.json",
        "live_generation": "live_generation/*/receipt.json",
    }.items():
        supplemental[category] = []
        for path in sorted(directory.glob(pattern)):
            record = read_json(path)
            entry: dict[str, Any] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "status": record["status"],
                "run_id": record.get("run_id") or record.get("receipt", {}).get("run_id"),
            }
            if category == "semantic_overlap":
                entry["flagged_rows"] = record.get("flagged_rows")
                entry["near_train_rows"] = sum(
                    any(
                        str(s["source"]).endswith("train.parquet")
                        for m in r["matches"]
                        for s in m["sources"]
                    )
                    for r in record.get("rows", [])
                )
                entry["flags_by_phase_language"] = dict(
                    Counter(
                        f"{r['phase']}:{r['scope']}" for r in record.get("rows", []) if r["flagged"]
                    )
                )
            if category == "live_generation":
                entry["actual_model_calls"] = record.get("actual_model_calls")
            supplemental[category].append(entry)
    stress_ids = {r["run_id"] for r in supplemental["stress"] if r["run_id"]}
    for path in (directory / "source_versions").glob("*.json"):
        stress_ids.update(read_json(path).get("applies_to", []))
    scale_receipts = []
    for path in (settings.resolve(settings.paths.reports_dir) / "retrieval_scale/load").glob(
        "*/receipt.json"
    ):
        record = read_json(path)
        if record.get("run_id") in stress_ids:
            scale_receipts.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "status": record["status"],
                    "run_id": record["run_id"],
                }
            )
    failed_preflight = directory / "checks/semantic_overlap_preflight_failure.log"
    critical_losses = [
        r["id"]
        for r, s in zip(control["rows"], candidate["rows"], strict=True)
        if r["critical"] and s["queue_verified"] < r["queue_verified"]
    ]
    result = {
        "schema_version": "system_final_evaluation/1",
        "freeze_identity": frozen["identity"],
        "control": str(control_path),
        "candidate": str(candidate_path),
        "per_language": scopes,
        "paired_delivery": paired(control["rows"], candidate["rows"]),
        "critical_delivery_losses": critical_losses,
        "semantic": evaluate_reviews([]),
        "attempts": attempts,
        "performance_attempts": performance,
        "supplemental_attempts": supplemental,
        "scale_engine_receipts": scale_receipts,
        "preflight_failures": [
            {
                "path": str(failed_preflight),
                "sha256": sha256_file(failed_preflight),
                "execution_status": "FAILED",
                "model_calls": 0,
                "reason": "source differed; retry used SHA256-verified Git baseline",
                "tracking_scope": "failure imported by analysis, not an inference run",
            }
        ]
        if failed_preflight.exists()
        else [],
        "semantic_independence": "NOT_CERTIFIED",
        "independent_fluent_review": False,
        "generation_default": False,
        "stage10_readiness": "NOT_READY",
        "stage9_quality_acceptance": "NOT_ACCEPTED",
        "acceptance_limits": [
            "independent fluent labels/reviewers unavailable (owner confirmed)",
            "embedding overlap flags require independent adjudication; no certified fresh gold",
            "intent target missed; overall holdout delivery improvement not established",
        ],
        "operational_slo": {
            "scope": "warm tools/queue HTTP; concurrency1; isolated synthetic DB",
            "p95_target_seconds": 0.1,
            "freeform_or_generation_slo": "NOT_ESTABLISHED",
            "multiuser_capacity": "NOT_ACCEPTED",
        },
        "stage7_seal_unchanged": sha256_file(settings.root / "artifacts/reports/llm/selection.json")
        == frozen["stage7_seal_sha256"],
        "conclusion": (
            "Narrow guard fix; no confirmed overall holdout delivery improvement. "
            "Intent targets missed; semantics unverified; autonomous/production NOT_READY."
        ),
        "limitations": [
            "100 draft E2E needs, 50 critical per language; translations share global needs",
            "only20 supplemental needs; locator proxies are not semantic proof",
            "no independent fluent human calibration; no gold/human verification claim",
            "three identical inference repeats, not independent training seeds",
            "all-success bootstrap is degenerate; Wilson does not establish critical recall>=.95",
        ],
    }
    spec = RunSpec(
        experiment="system",
        run_name="final_analysis",
        hypothesis="Assess targets and readiness without promoting draft proxies to semantics",
        dataset_version=frozen["identity"],
        seed=20261115,
        params={"semantic_quality": "NOT_EVALUATED", "readiness": "NOT_READY"},
    )
    with start_run(spec, settings) as run:
        result["analysis_run_id"] = run.info.run_id
        write_json(directory / "final_evaluation.json", result)
        write_json(
            directory / "error_review.json",
            {"reviewer": "codex", "independent_human_review": False, "rows": error_rows},
        )
        mlflow.log_artifact(str(directory / "final_evaluation.json"))
        mlflow.log_artifact(str(directory / "error_review.json"))
        if failed_preflight.exists():
            mlflow.log_artifact(str(failed_preflight))
        for category, entries in supplemental.items():
            for entry in entries:
                artifact_path = f"execution_receipts/{category}/{Path(entry['path']).parent.name}"
                mlflow.log_artifact(entry["path"], artifact_path=artifact_path)
        for entry in scale_receipts:
            path = Path(entry["path"])
            mlflow.log_artifact(str(path), artifact_path="scale_receipts/" + path.parent.name)
        mlflow.set_tags({"decision": "not_ready_for_stage10", "semantic_quality": "NOT_EVALUATED"})
    print(json.dumps({"path": str(directory / "final_evaluation.json")}))


if __name__ == "__main__":
    main()
