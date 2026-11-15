"""Frozen draft system evaluation; raw routing and observed delivery stay separate."""

from __future__ import annotations

import argparse
import json
import re
import secrets
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import yaml
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, SecretStr

from supportpilot.api import create_app
from supportpilot.common.config import Settings, load_settings
from supportpilot.common.hashing import sha256_file, sha256_text
from supportpilot.common.io import read_json, write_json
from supportpilot.common.provenance import code_commit
from supportpilot.common.schemas import INTENTS, LanguageCode
from supportpilot.common.tracking import RunSpec, start_run
from supportpilot.retrieval.evaluate import clustered_ci
from supportpilot.retrieval.experiments import measure
from supportpilot.service import orchestrator
from supportpilot.service.runtime import open_service
from supportpilot.service.store import Store

SCOPES = ("en", "ru", "kk", "mixed")


class Need(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    need_id: str
    phase: str
    family: str
    intent: str
    delivery: str
    critical: bool
    texts: list[str]
    followups: list[str] = []


def root(settings: Settings) -> Path:
    return settings.resolve(settings.paths.reports_dir) / "system"


def protocol(settings: Settings) -> dict[str, Any]:
    return dict(yaml.safe_load((settings.root / "configs/system_eval.yaml").read_text("utf-8")))


def load_needs(path: Path) -> list[Need]:
    raw = yaml.safe_load(path.read_text("utf-8"))
    needs = [
        Need(
            need_id=r[0],
            phase=r[1],
            family=r[2],
            intent=r[3],
            delivery=r[4],
            critical=r[5],
            texts=r[6:],
            followups=raw.get("followups", {}).get(r[0], []),
        )
        for r in raw["needs"]
    ]
    if len({n.need_id for n in needs}) != len(needs):
        raise ValueError("duplicate need IDs")
    for n in needs:
        if n.intent not in INTENTS or len(n.texts) != 4 or (n.followups and len(n.followups) != 4):
            raise ValueError("invalid intent or language coverage")
        if n.phase not in {"development", "holdout"}:
            raise ValueError("invalid phase")
    families = [{n.family for n in needs if n.phase == p} for p in ("development", "holdout")]
    if families[0] & families[1]:
        raise ValueError("development/holdout families overlap")
    return needs


def freeze(settings: Settings) -> Path:
    cfg = protocol(settings)
    paths = [settings.root / cfg[k] for k in ("source", "protocol")]
    paths.append(settings.root / "configs/system_eval.yaml")
    identity = sha256_text(json.dumps({str(p): sha256_file(p) for p in paths}, sort_keys=True))
    directory = settings.resolve(settings.paths.data_dir) / "benchmarks/system" / identity[:20]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "freeze.json"
    if target.exists():
        verify_freeze(target)
        return target
    needs = load_needs(paths[0])
    manifest = {
        "schema_version": "system_freeze/1",
        "identity": identity,
        "files": {str(p): sha256_file(p) for p in paths},
        "code_commit": code_commit(settings.root),
        "stage7_seal_sha256": sha256_file(settings.root / "artifacts/reports/llm/selection.json"),
        "baseline_risk": orchestrator._RISK.pattern,
        "candidate_risk": cfg["candidate_risk"],
        "needs": [n.model_dump() for n in needs],
        "independent_human_review": False,
        "semantic_independence": "NOT_CERTIFIED",
        "counts": {
            p: {
                "needs": sum(n.phase == p for n in needs),
                "families": len({n.family for n in needs if n.phase == p}),
                "critical_needs": sum(n.phase == p and n.critical for n in needs),
            }
            for p in ("development", "holdout")
        },
    }
    manifest["content_sha256"] = sha256_text(json.dumps(manifest, sort_keys=True))
    write_json(target, manifest)
    write_json(root(settings) / "freeze_pointer.json", {"path": str(target)})
    return target


def verify_freeze(path: Path) -> dict[str, Any]:
    data = read_json(path)
    digest = data.pop("content_sha256")
    if sha256_text(json.dumps(data, sort_keys=True)) != digest:
        raise ValueError("freeze content changed")
    for source, expected in data["files"].items():
        if sha256_file(Path(source)) != expected:
            raise ValueError("frozen source changed")
    data["content_sha256"] = digest
    return dict(data)


def macro_f1(rows: list[dict[str, Any]]) -> float:
    values = []
    for intent in sorted(INTENTS):
        tp = sum(r["intent"] == intent and r.get("raw_intent") == intent for r in rows)
        fp = sum(r["intent"] != intent and r.get("raw_intent") == intent for r in rows)
        fn = sum(r["intent"] == intent and r.get("raw_intent") != intent for r in rows)
        values.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(values))


def summary(rows: list[dict[str, Any]], *, seed: int = 20261115) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scope in SCOPES:
        cohort = [r for r in rows if r["scope"] == scope]
        critical = [r for r in cohort if r["critical"]]
        metrics = ("delivery_correct", "raw_intent_correct", "raw_action_correct", "queue_verified")
        rng = np.random.default_rng(seed)
        estimates = []
        if len(cohort) > 1:
            for _ in range(1000):
                estimates.append(
                    macro_f1([cohort[int(i)] for i in rng.integers(len(cohort), size=len(cohort))])
                )
        result[scope] = {
            "rows": len(cohort),
            "needs": len({r["scenario_id"] for r in cohort}),
            "families": len({r["family_id"] for r in cohort}),
            "intent_coverage": len({r["intent"] for r in cohort}),
            "intent_macro_f1_35": macro_f1(cohort) if cohort else None,
            "intent_macro_f1_scenario_ci95": np.quantile(estimates, [0.025, 0.975]).tolist()
            if estimates
            else None,
            "metrics": {m: clustered_ci(cohort, m, seed=seed) for m in metrics},
            "family_sensitivity": {
                m: clustered_ci(
                    [{**r, "scenario_id": r["family_id"]} for r in cohort], m, seed=seed
                )
                for m in metrics
            },
            "critical_needs": len(critical),
            "critical_delivery_recall": clustered_ci(critical, "queue_verified", seed=seed),
            "p50_seconds": float(np.quantile([r["seconds"] for r in cohort], 0.5))
            if cohort
            else None,
            "p95_seconds": float(np.quantile([r["seconds"] for r in cohort], 0.95))
            if cohort
            else None,
            "error_census": [r["scenario_id"] for r in cohort if not r["delivery_correct"]][:50],
            "diagnostic_counts": dict(
                Counter(r.get("diagnostic", "execution_failure") for r in cohort)
            ),
            "semantic_citation_precision": "NOT_EVALUATED",
            "grounded_answer_accuracy": "NOT_EVALUATED",
            "unsupported_claim_rate": "NOT_EVALUATED",
        }
    return result


def paired(control: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    left, right = {r["id"]: r for r in control}, {r["id"]: r for r in candidate}
    if len(left) != len(control) or len(right) != len(candidate) or left.keys() != right.keys():
        raise ValueError("paired IDs differ or duplicate")
    differences = []
    for key, row in left.items():
        other = right[key]
        if any(
            row[k] != other[k]
            for k in (
                "scenario_id",
                "family_id",
                "scope",
                "intent",
                "expected_delivery",
                "critical",
                "text",
            )
        ):
            raise ValueError("paired labels/groups differ")
        differences.append({**row, "delta": other["delivery_correct"] - row["delivery_correct"]})
    return {
        s: {
            "scenario": clustered_ci(
                [r for r in differences if r["scope"] == s], "delta", seed=20261115
            ),
            "family": clustered_ci(
                [{**r, "scenario_id": r["family_id"]} for r in differences if r["scope"] == s],
                "delta",
                seed=20261115,
            ),
        }
        for s in SCOPES
    }


@contextmanager
def risk_control(pattern: str) -> Iterator[None]:
    previous = orchestrator._RISK
    orchestrator._RISK = re.compile(pattern, re.IGNORECASE)
    try:
        yield
    finally:
        orchestrator._RISK = previous


def run_evaluation(
    settings: Settings, frozen: Path, *, phase: str, variant: str, smoke: bool = False
) -> Path:
    import mlflow

    data = verify_freeze(frozen)
    if (
        sha256_file(settings.root / "artifacts/reports/llm/selection.json")
        != data["stage7_seal_sha256"]
    ):
        raise ValueError("Stage 7 seal changed")
    if phase == "holdout":
        selection = read_json(root(settings) / "selection.json")
        if selection["freeze_identity"] != data["identity"]:
            raise ValueError("selection/freeze mismatch")
        marker = root(settings) / f"holdout_{variant}.json"
        if marker.exists():
            raise ValueError(
                "holdout already attempted; preserve attempt and use explicit technical retry"
            )
    out = root(settings) / "attempts" / uuid4().hex
    out.mkdir(parents=True)
    if phase == "holdout":
        write_json(marker, {"attempt": str(out), "freeze_identity": data["identity"]})
    token = secrets.token_urlsafe(32)
    cfg = settings.serving.model_copy(
        update={
            "database": out / "service.sqlite",
            "generation_enabled": False,
            "tokens": {"demo_alice": SecretStr(token)},
            "max_sessions": 2000,
        }
    )
    local = settings.model_copy(update={"serving": cfg})
    Store(cfg.database).seed_demo()
    needs = [Need.model_validate(n) for n in data["needs"] if n["phase"] == phase]
    if smoke:
        needs = needs[:1]
    rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": "system_attempt/1",
        "status": "RUNNING",
        "phase": phase,
        "variant": variant,
        "smoke": smoke,
        "freeze_identity": data["identity"],
        "source_hashes": {
            str(p.relative_to(settings.root)): sha256_file(p)
            for p in (settings.root / "src/supportpilot").rglob("*.py")
        },
        "uv_lock_sha256": sha256_file(settings.root / "uv.lock"),
        "code_commit": code_commit(settings.root),
        "serving_config": cfg.model_dump(mode="json", exclude={"tokens"}),
        "retrieval_config": settings.retrieval.model_dump(mode="json"),
        "paid_cost_usd": 0,
        "rows": rows,
    }
    write_json(out / "report.json", report)
    spec = RunSpec(
        experiment="system",
        run_name=f"{phase}_{variant}_{'smoke' if smoke else 'full'}",
        hypothesis="Boundaries reduce false escalation while preserving critical delivery",
        dataset_version=data["identity"],
        seed=20261115,
        params={
            "phase": phase,
            "variant": variant,
            "smoke": smoke,
            "semantic_quality": "NOT_EVALUATED",
        },
    )
    try:
        with (
            start_run(spec, settings) as run,
            risk_control(data["baseline_risk"] if variant == "stage8" else data["candidate_risk"]),
            measure() as resources,
        ):
            report["run_id"] = run.info.run_id
            started = time.perf_counter()
            with open_service(local) as service, TestClient(create_app(local, service)) as client:
                client.headers["Authorization"] = "Bearer " + token
                report["load_seconds"] = time.perf_counter() - started
                artifact = getattr(service.retriever, "artifact", None)
                report["kb_version"] = artifact.kb_version if artifact else None
                report["index_version"] = artifact.manifest.dataset_version if artifact else None
                for n in needs:
                    for index, scope in enumerate(SCOPES):
                        lang: LanguageCode = "kk" if scope == "mixed" else scope  # type: ignore[assignment]
                        row: dict[str, Any] = {
                            "id": n.need_id + ":" + scope,
                            "scenario_id": n.need_id,
                            "family_id": n.family,
                            "scope": scope,
                            "intent": n.intent,
                            "critical": n.critical,
                            "text": n.texts[index],
                            "expected_delivery": n.delivery,
                            "delivery_correct": 0.0,
                            "raw_intent_correct": 0.0,
                            "raw_action_correct": 0.0,
                            "queue_verified": 0.0,
                        }
                        tick = time.perf_counter()
                        try:
                            raw = service.route(n.texts[index], lang, scope == "mixed")
                            row["raw_route"] = raw.model_dump(mode="json")
                            row["raw_intent"] = raw.intent
                            row["raw_intent_correct"] = float(raw.intent == n.intent)
                            expected_action = service.taxonomy.intents[n.intent].action
                            if expected_action == "needs_reference":
                                expected_action = (
                                    "call_tool"
                                    if re.search(r"\b(?:SP-\d{4}|TV-\d{4})\b", n.texts[index])
                                    else "clarify"
                                )
                            row["raw_action_correct"] = float(raw.action == expected_action)
                        except Exception as exc:
                            row["routing_error"] = type(exc).__name__
                        row["raw_route_seconds"] = time.perf_counter() - tick
                        tick = time.perf_counter()
                        try:
                            response = client.post(
                                "/chat",
                                json={
                                    "message": n.texts[index],
                                    "language": lang,
                                    "request_id": str(uuid4()),
                                },
                            )
                            row["http_status"] = response.status_code
                            row["result"] = response.json()
                            if response.status_code == 200:
                                result = response.json()
                                row["diagnostic"] = result["diagnostic"]
                                row["delivery_correct"] = float(result["delivery"] == n.delivery)
                                if result["escalation_id"]:
                                    with service.store.transaction() as db:
                                        found = db.execute(
                                            "SELECT status FROM escalations "
                                            "WHERE id=? AND customer=?",
                                            (result["escalation_id"], "demo_alice"),
                                        ).fetchone()
                                    row["queue_verified"] = float(found is not None)
                                if n.followups:
                                    follow = client.post(
                                        "/chat",
                                        json={
                                            "message": n.followups[index],
                                            "language": lang,
                                            "request_id": str(uuid4()),
                                            "session_id": result["session_id"],
                                        },
                                    )
                                    row["followup"] = {
                                        "http_status": follow.status_code,
                                        "result": follow.json(),
                                        "expected_delivery": "tool",
                                    }
                                    row["followup_correct"] = float(
                                        follow.status_code == 200
                                        and follow.json()["delivery"] == "tool"
                                    )
                        except Exception as exc:
                            row["error"] = type(exc).__name__
                        row["seconds"] = time.perf_counter() - tick
                        rows.append(row)
                        write_json(out / "report.json", report)
                    print(f"SYSTEM {phase} {variant} {n.need_id}", flush=True)
                report["ready_after"] = client.get("/ready").status_code
            report["summary"] = summary(rows)
            report["status"] = "FINISHED"
            mlflow.set_tags(
                {
                    "decision": "development_only"
                    if phase == "development"
                    else "assessment_no_retuning",
                    "semantic_quality": "NOT_EVALUATED",
                }
            )
            for scope, values in report["summary"].items():
                mlflow.log_metric(f"{scope}_macro_f1_35", values["intent_macro_f1_35"])
            mlflow.log_artifact(str(frozen))
    except BaseException as exc:
        report.update(status="FAILED", error=type(exc).__name__, detail=str(exc))
        raise
    finally:
        if "resources" in locals():
            report["resources"] = vars(resources)
        write_json(out / "report.json", report)
        if report.get("run_id"):
            mlflow.MlflowClient().log_artifact(report["run_id"], str(out / "report.json"))
    return out / "report.json"


def select(settings: Settings, control: Path, candidate: Path) -> Path:
    left, right = read_json(control), read_json(candidate)
    if any(
        r["phase"] != "development" or r["smoke"] or r["status"] != "FINISHED"
        for r in (left, right)
    ):
        raise ValueError("selection requires completed full development attempts")
    if (
        left["freeze_identity"] != right["freeze_identity"]
        or left["variant"] != "stage8"
        or right["variant"] != "boundary_guard"
    ):
        raise ValueError("control identities differ")
    comparison = paired(left["rows"], right["rows"])
    improved = any(v["scenario"].get("low", 0) > 0 for v in comparison.values())
    no_loss = all(
        sum(r["delivery_correct"] for r in right["rows"] if r["scope"] == s)
        >= sum(r["delivery_correct"] for r in left["rows"] if r["scope"] == s)
        for s in SCOPES
    )
    no_critical_loss = all(
        not a["critical"] or b["queue_verified"] >= a["queue_verified"]
        for a, b in zip(left["rows"], right["rows"], strict=True)
    )
    result = {
        "freeze_identity": left["freeze_identity"],
        "control": str(control),
        "candidate": str(candidate),
        "control_sha256": sha256_file(control),
        "candidate_sha256": sha256_file(candidate),
        "paired": comparison,
        "adopt": improved and no_loss and no_critical_loss,
        "no_critical_loss": no_critical_loss,
        "no_language_loss": no_loss,
        "decision_scope": "development_only; no semantic or unrestricted-risk acceptance",
    }
    target = root(settings) / "selection.json"
    if target.exists():
        raise ValueError("selection is already sealed")
    write_json(target, result)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "evaluate", "select"])
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--phase", choices=["development", "holdout"], default="development")
    parser.add_argument("--variant", choices=["stage8", "boundary_guard"], default="stage8")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--control", type=Path)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    settings = load_settings()
    if args.action == "freeze":
        result = freeze(settings)
    elif args.action == "select":
        if args.control is None or args.candidate is None:
            parser.error("select requires --control and --candidate")
        result = select(settings, args.control, args.candidate)
    else:
        frozen = args.frozen or Path(read_json(root(settings) / "freeze_pointer.json")["path"])
        result = run_evaluation(
            settings, frozen, phase=args.phase, variant=args.variant, smoke=args.smoke
        )
    print(json.dumps({"path": str(result)}))


if __name__ == "__main__":
    main()
