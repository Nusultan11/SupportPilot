"""Fresh Stage 9 continuation: paired observed operations with immutable controls."""

from __future__ import annotations

import argparse
import importlib.util
import json
import secrets
import shutil
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import numpy as np
import yaml
from fastapi.testclient import TestClient
from pydantic import SecretStr

from supportpilot.api import create_app
from supportpilot.common.config import Settings, load_settings
from supportpilot.common.hashing import sha256_file, sha256_text
from supportpilot.common.io import read_json, write_json
from supportpilot.common.schemas import LanguageCode
from supportpilot.common.tracking import RunSpec, start_run
from supportpilot.routing.models import RouteDecision
from supportpilot.service.evaluation import SCOPES, macro_f1
from supportpilot.service.orchestrator import SupportService
from supportpilot.service.performance import latency
from supportpilot.service.runtime import open_service
from supportpilot.service.store import Store

SEED = 20261115
METRICS = (
    "intent_correct",
    "action_correct",
    "delivery_correct",
    "critical_recall",
    "reason_correct",
    "false_critical",
    "tool_success",
    "dialogue_success",
)
CRITICAL_REASONS = {
    "independent_safety_guard",
    "safety_risk",
    "fraud_suspected",
    "privacy_request",
    "legal_threat",
    "repeated_failure",
    "payment_dispute",
    "customer_requested_human",
    "complaint",
    "technical_issue",
}


def location(settings: Settings) -> Path:
    return settings.resolve(settings.paths.reports_dir) / "system/quality"


def digest(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, ensure_ascii=False))


def runtime_hashes(settings: Settings) -> dict[str, str]:
    paths = list((settings.root / "src/supportpilot").rglob("*.py"))
    paths += [
        settings.root / p
        for p in (
            "uv.lock",
            "configs/base.yaml",
            "configs/taxonomy.yaml",
            "configs/sources.yaml",
            "artifacts/reports/llm/selection.json",
        )
    ]
    return {str(p.relative_to(settings.root)): sha256_file(p) for p in sorted(paths)}


def load_inputs(path: Path) -> dict[str, Any]:
    from supportpilot.common.schemas import INTENTS

    data: dict[str, Any] = yaml.safe_load(path.read_text("utf-8"))
    rows = data["needs"]
    if len({r[0] for r in rows}) != len(rows):
        raise ValueError("duplicate need")
    groups = []
    for phase in ("development", "confirmation"):
        cohort = [r for r in rows if r[1] == phase]
        if {r[3] for r in cohort} != INTENTS:
            raise ValueError("all 35 intents required in both splits")
        groups.append({r[2] for r in cohort})
    if groups[0] & groups[1]:
        raise ValueError("family overlap")
    for row in rows:
        if (
            len(row) != 9
            or len(row[8]) != 4
            or row[1] not in {"development", "confirmation"}
            or row[3] not in INTENTS
        ):
            raise ValueError("invalid need")
    if any(len(v) != 4 for v in data.get("followups", {}).values()):
        raise ValueError("invalid followup")
    return data


def freeze(settings: Settings) -> Path:
    cfg = yaml.safe_load((settings.root / "configs/system_quality.yaml").read_text("utf-8"))
    paths = [settings.root / cfg[k] for k in ("source", "protocol")]
    paths.append(settings.root / "configs/system_quality.yaml")
    hashes = {str(p.relative_to(settings.root)): sha256_file(p) for p in paths}
    data = load_inputs(paths[0])
    identity = digest(hashes)
    out = settings.root / "data/benchmarks/system_quality" / identity[:20]
    target = out / "freeze.json"
    if target.exists():
        verify(settings, target)
        return target
    out.mkdir(parents=True)
    # Trusted local Git object, never downloaded executable benchmark material.
    control = subprocess.run(  # noqa: S603 - fixed local Git command; no shell
        [
            shutil.which("git") or "git",
            "show",
            f"{cfg['baseline_commit']}:src/supportpilot/service/orchestrator.py",
        ],
        cwd=settings.root,
        check=True,
        capture_output=True,
    ).stdout
    (out / "control.py").write_bytes(control)
    frozen = {
        "identity": identity,
        "files": hashes,
        "inputs": data,
        "control_commit": cfg["baseline_commit"],
        "control_source": str(out / "control.py"),
        "control_sha256": sha256_file(out / "control.py"),
        "stage7_seal_sha256": sha256_file(settings.root / "artifacts/reports/llm/selection.json"),
        "source_review": "LLM_AUTHORED_DRAFT; NO_INDEPENDENT_HUMAN_REVIEW",
        "semantic_independence": "NOT_CERTIFIED",
    }
    write_json(target, {**frozen, "content_sha256": digest(frozen)})
    write_json(location(settings) / "freeze_pointer.json", {"path": str(target)})
    return target


def verify(settings: Settings, path: Path) -> dict[str, Any]:
    frozen = dict(read_json(path))
    checksum = frozen.pop("content_sha256")
    if digest(frozen) != checksum:
        raise ValueError("changed frozen content")
    for name, expected in frozen["files"].items():
        if sha256_file(settings.root / name) != expected:
            raise ValueError("changed frozen inputs")
    if sha256_file(Path(frozen["control_source"])) != frozen["control_sha256"]:
        raise ValueError("changed control source")
    if (
        sha256_file(settings.root / "artifacts/reports/llm/selection.json")
        != frozen["stage7_seal_sha256"]
    ):
        raise ValueError("changed Stage 7 seal")
    return frozen


def baseline_class(frozen: dict[str, Any]) -> type[SupportService]:
    spec = importlib.util.spec_from_file_location(
        "supportpilot_quality_control", frozen["control_source"]
    )
    if spec is None or spec.loader is None:
        raise ValueError("control import unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SupportService  # type: ignore[no-any-return]


def baseline_store(settings: Settings, frozen: dict[str, Any], database: Path) -> Store:
    """Retain the pre-intervention queue policy as well as its orchestrator."""
    git = shutil.which("git")
    if git is None:
        raise ValueError("Git unavailable")
    content = subprocess.run(  # noqa: S603 - fixed trusted Git object; no shell
        [git, "show", f"{frozen['control_commit']}:src/supportpilot/service/store.py"],
        cwd=settings.root,
        check=True,
        capture_output=True,
    ).stdout
    source = location(settings) / "control_store.py"
    if source.exists() and source.read_bytes() != content:
        raise ValueError("control store changed")
    source.write_bytes(content)
    spec = importlib.util.spec_from_file_location("supportpilot_quality_control_store", source)
    if spec is None or spec.loader is None:
        raise ValueError("control store unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Store(database)  # type: ignore[no-any-return]


def expected_facts(tool: str | None) -> dict[str, Any]:
    facts: dict[str, dict[str, Any]] = {
        "get_order": {"order_id": "SP-1001", "order_status": "processing"},
        "get_payment": {"order_id": "SP-1001", "payment_status": "paid"},
        "get_stock": {"sku": "TV-5501", "quantity": 7},
        "request_cancellation": {"order_id": "SP-1001", "cancellation_request": "recorded"},
    }
    return facts.get(tool or "", {})


def observed_turn(
    client: TestClient,
    service: SupportService,
    text: str,
    language: str,
    expected_delivery: str,
    tool: str | None,
    *,
    session: str | None = None,
    confirm: bool = True,
) -> dict[str, Any]:
    facts = expected_facts(tool)
    observed: dict[str, Any] = {
        "text": text,
        "expected_delivery": expected_delivery,
        "expected_tool": tool,
        "expected_facts": facts,
        "correct": 0.0,
        "tool_success": 0.0,
        "queue_verified": 0.0,
    }
    tick = time.perf_counter()
    original_route = service.route
    original_tool = service.store.tool
    routed: list[RouteDecision] = []
    tools: list[dict[str, Any]] = []

    def capture_route(*args: Any, **kwargs: Any) -> RouteDecision:
        route = original_route(*args, **kwargs)
        routed.append(route)
        return route

    service.route = capture_route  # type: ignore[method-assign]

    def capture_tool(*args: Any, **kwargs: Any) -> dict[str, str | int]:
        values = original_tool(*args, **kwargs)
        tools.append({"customer": args[1], "name": args[2], "reference": args[3], "values": values})
        return values

    service.store.tool = capture_tool  # type: ignore[method-assign]
    try:
        response = client.post(
            "/chat",
            json={
                "message": text,
                "language": language,
                "request_id": str(uuid4()),
                "session_id": session,
                "confirm_cancellation": confirm,
            },
        )
        observed.update(http_status=response.status_code, result=response.json())
        if response.status_code != 200:
            return observed
        result = response.json()
        answer = result["response"]
        observed["effective_intent"] = answer["intent"]
        observed["effective_action"] = answer["action"]
        state = service.sessions[UUID(result["session_id"])]
        # Both runtimes already retain their actual intent in allowlisted history.
        # Fallback response.intent may be unclear even after a correct routing class.
        if state.history:
            observed["effective_intent"] = (
                state.history[-1].split(";", 1)[0].removeprefix("intent=")
            )
        observed["history"] = list(state.history)
        if routed:
            observed["planned_route"] = routed[-1].model_dump(mode="json")
            observed["effective_intent"] = routed[-1].intent
            if answer.get("escalation_reason") in {"unverified_generation", "evidence_unavailable"}:
                observed["effective_action"] = routed[-1].action.value
        decision = getattr(state, "decision", None)
        if decision is not None:
            observed["decision"] = decision.model_dump(mode="json")
            observed["effective_intent"] = decision.intent
            if result["delivery"] != "clarification":
                observed["effective_action"] = decision.action.value
        reason = answer.get("escalation_reason")
        observed["reason"] = reason
        if result["escalation_id"]:
            with closing(service.store.connect()) as db:
                entry = db.execute(
                    "SELECT reason FROM escalations WHERE id=? AND customer=? AND session=?",
                    (result["escalation_id"], "demo_alice", result["session_id"]),
                ).fetchone()
            observed["queue_verified"] = float(entry is not None and entry["reason"] == reason)
        observed["tool_success"] = float(
            bool(facts)
            and len(tools) == 1
            and tools[0]["name"] == tool
            and tools[0]["customer"] == "demo_alice"
            and tools[0]["values"] == facts
            and result["delivery"] == "tool"
            and set(answer["facts_from_tools"]) == set(facts)
            and all(f"{key}={value}" in answer["answer"] for key, value in facts.items())
        )
        observed["correct"] = float(
            result["delivery"] == expected_delivery
            and (expected_delivery != "tool" or observed["tool_success"] == 1)
            and (expected_delivery != "queued" or observed["queue_verified"] == 1)
        )
    except Exception as exc:
        observed["error"] = type(exc).__name__
    finally:
        service.route = original_route  # type: ignore[method-assign]
        service.store.tool = original_tool  # type: ignore[method-assign]
        observed["tool_calls"] = tools
        observed["seconds"] = time.perf_counter() - tick
    return observed


def interval(values: list[float]) -> dict[str, Any]:
    return {
        "numerator": sum(values),
        "denominator": len(values),
        "rate": float(np.mean(values)) if values else None,
    }


def eligible(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    if metric in {"critical_recall", "reason_correct"}:
        return [r for r in rows if r["critical"]]
    if metric == "false_critical":
        return [r for r in rows if not r["critical"]]
    if metric == "tool_success":
        return [r for r in rows if r["expected_tool"]]
    if metric == "dialogue_success":
        return [r for r in rows if r["followup_expected"]]
    return rows


def bootstrap(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    metric: str,
    *,
    family: bool = False,
    samples: int = 1000,
) -> list[float] | None:
    key = "family_id" if family else "scenario_id"
    groups = sorted({r[key] for r in left})
    if len(groups) < 2:
        return None
    a = {g: [r for r in left if r[key] == g] for g in groups}
    b = {g: [r for r in right if r[key] == g] for g in groups}
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(samples):
        drawn = rng.choice(groups, size=len(groups)).tolist()
        x, y = [r for g in drawn for r in a[g]], [r for g in drawn for r in b[g]]
        values.append(
            macro_f1(y) - macro_f1(x)
            if metric == "macro_f1_35"
            else float(np.mean([r[metric] for r in y]) - np.mean([r[metric] for r in x]))
        )
    return np.quantile(values, [0.025, 0.975]).tolist()  # type: ignore[no-any-return]


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for scope in SCOPES:
        cohort = [r for r in rows if r["scope"] == scope]
        critical = eligible(cohort, "critical_recall")
        success, count = sum(r["critical_recall"] for r in critical), len(critical)
        wilson = None
        if count:
            p, z = success / count, 1.96
            centre = (p + z * z / (2 * count)) / (1 + z * z / count)
            half = (
                z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
            )
            wilson = [centre - half, centre + half]
        result[scope] = {
            "rows": len(cohort),
            "families": len({r["family_id"] for r in cohort}),
            "intent_coverage": len({r["intent"] for r in cohort}),
            "macro_f1_35": macro_f1(cohort) if cohort else None,
            "raw_macro_f1_35": macro_f1(
                [{**r, "raw_intent": r.get("encoder_intent")} for r in cohort]
            )
            if cohort
            else None,
            "metrics": {m: interval([r[m] for r in eligible(cohort, m)]) for m in METRICS},
            "critical_wilson95": wilson,
            "latency": latency([r["seconds"] for r in cohort]),
            "execution_errors": sum(r.get("http_status") != 200 for r in cohort),
            "errors": [r["scenario_id"] for r in cohort if not r["delivery_correct"]][:50],
            "intent_errors": [r["scenario_id"] for r in cohort if not r["intent_correct"]][:50],
            "action_errors": [r["scenario_id"] for r in cohort if not r["action_correct"]][:50],
            "critical_errors": [r["scenario_id"] for r in critical if not r["critical_recall"]],
            "dialogue_errors": [
                r["scenario_id"]
                for r in eligible(cohort, "dialogue_success")
                if not r["dialogue_success"]
            ],
            "semantic_quality": "NOT_EVALUATED",
        }
    return result


def compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    a, b = {r["id"]: r for r in left["rows"]}, {r["id"]: r for r in right["rows"]}
    if len(a) != len(left["rows"]) or len(b) != len(right["rows"]) or a.keys() != b.keys():
        raise ValueError("paired IDs differ or duplicate")
    for key in a:
        if any(
            a[key][k] != b[key][k]
            for k in (
                "intent",
                "family_id",
                "text",
                "scope",
                "critical",
                "expected_tool",
                "expected_delivery",
                "expected_action",
                "followup_expected",
            )
        ):
            raise ValueError("paired labels/groups differ")
    result = {}
    for scope in SCOPES:
        x, y = (
            [r for r in a.values() if r["scope"] == scope],
            [r for r in b.values() if r["scope"] == scope],
        )
        metrics = {}
        for metric in (*METRICS, "macro_f1_35"):
            cx = x if metric == "macro_f1_35" else eligible(x, metric)
            cy = y if metric == "macro_f1_35" else eligible(y, metric)
            delta = (
                macro_f1(cy) - macro_f1(cx)
                if metric == "macro_f1_35"
                else (
                    float(np.mean([r[metric] for r in cy]) - np.mean([r[metric] for r in cx]))
                    if cx
                    else None
                )
            )
            metrics[metric] = {
                "delta": delta,
                "ci95": bootstrap(cx, cy, metric),
                "family_ci95": bootstrap(cx, cy, metric, family=True),
                "denominator": len(cx),
            }
        result[scope] = metrics
    return result


def evaluate(settings: Settings, phase: str, variant: str, *, smoke: bool = False) -> Path:
    import mlflow

    base = location(settings)
    frozen = verify(settings, Path(read_json(base / "freeze_pointer.json")["path"]))
    source_hashes = runtime_hashes(settings)
    if phase == "confirmation":
        pointer = base / "selection_pointer.json"
        selection = read_json(
            Path(read_json(pointer)["path"]) if pointer.exists() else base / "selection.json"
        )
        if selection["identity"] != frozen["identity"] or selection["runtime"] != source_hashes:
            raise ValueError("candidate differs from sealed selection")
        marker = base / f"confirmation_{variant}.json"
        if marker.exists():
            raise ValueError("confirmation already attempted")
    out = base / "attempts" / uuid4().hex
    out.mkdir(parents=True)
    if phase == "confirmation":
        with marker.open("x", encoding="utf-8") as stream:
            json.dump({"attempt": str(out), "identity": frozen["identity"]}, stream)
    inputs = frozen["inputs"]
    needs = [r for r in inputs["needs"] if r[1] == phase]
    if smoke:
        needs = needs[:1]
    token = secrets.token_urlsafe(32)
    cfg = settings.serving.model_copy(
        update={
            "database": out / "service.sqlite",
            "generation_enabled": False,
            "max_sessions": 2000,
            "tokens": {"demo_alice": SecretStr(token)},
        }
    )
    local = settings.model_copy(update={"serving": cfg})
    Store(cfg.database).seed_demo()
    report: dict[str, Any] = {
        "status": "RUNNING",
        "identity": frozen["identity"],
        "phase": phase,
        "variant": variant,
        "smoke": smoke,
        "runtime": source_hashes,
        "config": cfg.model_dump(mode="json", exclude={"tokens"}),
        "seed": SEED,
        "rows": [],
        "paid_service_cost_usd": 0,
        "electricity_cost": "UNKNOWN",
    }
    write_json(out / "report.json", report)
    from supportpilot.retrieval.experiments import measure

    spec = RunSpec(
        experiment="system_quality",
        run_name=f"{phase}_{variant}_{'smoke' if smoke else 'full'}",
        hypothesis="Service action integration and contextual guards improve delivery",
        dataset_version=frozen["identity"],
        seed=SEED,
        params={
            "phase": phase,
            "variant": variant,
            "smoke": smoke,
            "semantic_quality": "NOT_EVALUATED",
            "control": frozen["control_commit"],
        },
    )
    try:
        with start_run(spec, settings) as run, measure() as resources:
            report["run_id"] = run.info.run_id
            tick = time.perf_counter()
            with open_service(local) as composed:
                service = composed
                if variant == "control":
                    service = baseline_class(frozen)(
                        baseline_store(settings, frozen, cfg.database),
                        cfg,
                        composed.taxonomy,
                        composed.router,
                        composed.retriever,
                    )
                report["load_seconds"] = time.perf_counter() - tick
                try:
                    with TestClient(create_app(local, service)) as client:
                        client.headers["Authorization"] = "Bearer " + token
                        for need in needs:
                            identity, _, family, intent, action, delivery, reason, tool, texts = (
                                need
                            )
                            for scope, text in zip(SCOPES, texts, strict=True):
                                lang = cast(LanguageCode, "kk" if scope == "mixed" else scope)
                                row: dict[str, Any] = {
                                    "id": identity + ":" + scope,
                                    "scenario_id": identity,
                                    "family_id": family,
                                    "intent": intent,
                                    "scope": scope,
                                    "text": text,
                                    "critical": reason is not None,
                                    "requested_action": action,
                                    "expected_action": "clarify"
                                    if delivery == "clarification"
                                    else action,
                                    "expected_delivery": delivery,
                                    "expected_tool": tool,
                                    "followup_expected": identity in inputs.get("followups", {}),
                                }
                                # Separate raw research routing observation.
                                try:
                                    assert service.router is not None
                                    raw = service.router.route(
                                        text, {}, language=lang, code_switch=scope == "mixed"
                                    )
                                    row["encoder_route"] = raw.model_dump(mode="json")
                                    row["encoder_intent"] = raw.intent
                                except Exception as exc:
                                    row["encoder_error"] = type(exc).__name__
                                observation = observed_turn(
                                    client,
                                    service,
                                    text,
                                    lang,
                                    delivery,
                                    tool,
                                    confirm=identity not in inputs.get("no_confirmation", []),
                                )
                                row.update(observation)
                                row["raw_intent"] = row.get("effective_intent")
                                row["intent_correct"] = float(row.get("effective_intent") == intent)
                                row["action_correct"] = float(
                                    row.get("effective_action") == row["expected_action"]
                                )
                                row["delivery_correct"] = row["correct"]
                                row["critical_recall"] = row["queue_verified"]
                                row["reason_correct"] = float(row.get("reason") == reason)
                                row["false_critical"] = float(row.get("reason") in CRITICAL_REASONS)
                                row["dialogue_success"] = 0.0
                                if row["followup_expected"]:
                                    session = row.get("result", {}).get("session_id")
                                    follow = (
                                        observed_turn(
                                            client,
                                            service,
                                            inputs["followups"][identity][SCOPES.index(scope)],
                                            lang,
                                            inputs["followup_delivery"][identity],
                                            inputs["followup_tools"][identity],
                                            session=session,
                                            confirm=identity != "c44",
                                        )
                                        if session
                                        else {"correct": 0.0, "error": "missing_session"}
                                    )
                                    row["followup"] = follow
                                    row["dialogue_success"] = (
                                        row["delivery_correct"] * follow["correct"]
                                    )
                                report["rows"].append(row)
                                write_json(out / "report.json", report)
                            print(f"QUALITY {phase} {variant} {identity}", flush=True)
                        report["ready_after"] = client.get("/ready").status_code
                finally:
                    if service is not composed:
                        service.close()
            report["summary"] = summarise(report["rows"])
            report["wall_seconds"] = time.perf_counter() - tick
            report["status"] = "FINISHED"
            mlflow.set_tags(
                {
                    "decision_scope": "development_only"
                    if phase == "development"
                    else "assessment_no_retuning",
                    "semantic_quality": "NOT_EVALUATED",
                }
            )
            for scope, values in report["summary"].items():
                mlflow.log_metric(f"{scope}_macro_f1_35", values["macro_f1_35"])
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


def select(settings: Settings, left_path: Path, right_path: Path) -> Path:
    base = location(settings)
    if list(base.glob("confirmation_*.json")):
        raise ValueError("confirmation consumed; no reselection")
    target = base / "selections" / (digest(runtime_hashes(settings)) + ".json")
    if target.exists():
        raise ValueError("selection already sealed")
    left, right = read_json(left_path), read_json(right_path)
    if (
        any(
            r["status"] != "FINISHED" or r["phase"] != "development" or r["smoke"]
            for r in (left, right)
        )
        or left["identity"] != right["identity"]
    ):
        raise ValueError("completed matching development required")
    if left["variant"] != "control" or right["variant"] != "candidate":
        raise ValueError("incorrect control/candidate")
    if right["runtime"] != runtime_hashes(settings):
        raise ValueError("candidate runtime drift")
    comparison = compare(left, right)
    positive = any(v["delivery_correct"]["ci95"][0] > 0 for v in comparison.values())
    no_loss = all(v["delivery_correct"]["delta"] >= 0 for v in comparison.values())
    a, b = {r["id"]: r for r in left["rows"]}, {r["id"]: r for r in right["rows"]}
    critical_loss = [
        k for k in a if a[k]["critical"] and a[k]["critical_recall"] > b[k]["critical_recall"]
    ]
    selection = {
        "identity": left["identity"],
        "runtime": right["runtime"],
        "control": str(left_path),
        "candidate": str(right_path),
        "control_sha256": sha256_file(left_path),
        "candidate_sha256": sha256_file(right_path),
        "paired": comparison,
        "critical_losses": critical_loss,
        "adopt": positive and no_loss and not critical_loss,
        "scope": "draft operational evidence only; semantics NOT_EVALUATED",
    }
    write_json(target, selection)
    write_json(
        base / "selection_pointer.json", {"path": str(target), "sha256": sha256_file(target)}
    )
    return target


def analyse(settings: Settings) -> Path:
    """Derive corrected action scoring and absolute uncertainty without changing raw attempts."""
    base = location(settings)
    pointer = read_json(base / "selection_pointer.json")
    if sha256_file(Path(pointer["path"])) != pointer["sha256"]:
        raise ValueError("changed selection")
    seal = read_json(Path(pointer["path"]))
    reports = {}
    sources = {}
    for role in ("control", "candidate"):
        path = Path(read_json(base / f"confirmation_{role}.json")["attempt"]) / "report.json"
        raw = read_json(path)
        if (
            raw["status"] != "FINISHED"
            or raw["phase"] != "confirmation"
            or raw["variant"] != role
            or raw["identity"] != seal["identity"]
            or raw["runtime"] != seal["runtime"]
        ):
            raise ValueError("terminal sealed confirmation required")
        rows = []
        for row in raw["rows"]:
            language = "kk" if row["scope"] == "mixed" else row["scope"]
            rows.append(
                {
                    **row,
                    "action_correct": float(row.get("effective_action") == row["expected_action"]),
                    "declared_language_correct": float(
                        row.get("result", {}).get("response", {}).get("language") == language
                    ),
                }
            )
        reports[role] = {**raw, "rows": rows, "summary": summarise(rows)}
        sources[role] = {"path": str(path), "sha256": sha256_file(path), "run_id": raw["run_id"]}
    paired = compare(reports["control"], reports["candidate"])
    result: dict[str, Any] = {
        "schema_version": "system_quality_analysis/1",
        "identity": seal["identity"],
        "selection": pointer,
        "sources": sources,
        "seed": SEED,
        "bootstrap_samples": 1000,
        "correction": (
            "Action scoring uses frozen expected_action, including confirmation clarification. "
            "Raw reports and delivery-based selection are unchanged. Finding registered in "
            "completion/entry.json before confirmation outcomes."
        ),
        "uncertainty_limit": (
            "Sparse all35 macro-F1 percentiles and all-success intervals are exploratory; "
            "family IDs do not certify semantic independence."
        ),
        "semantic_quality": "NOT_EVALUATED",
        "fluent_language_quality": "NOT_EVALUATED",
        "variants": {},
        "paired": paired,
    }
    for role, report in reports.items():
        result["variants"][role] = {}
        for scope in SCOPES:
            cohort = [r for r in report["rows"] if r["scope"] == scope]
            summary = report["summary"][scope]
            absolute = {}
            for metric in (*METRICS, "macro_f1_35", "declared_language_correct"):
                selected = (
                    cohort
                    if metric in {"macro_f1_35", "declared_language_correct"}
                    else eligible(cohort, metric)
                )
                zero = [{**r, metric: 0.0, "raw_intent": None} for r in selected]
                absolute[metric] = {
                    "ci95": bootstrap(zero, selected, metric),
                    "family_ci95": bootstrap(zero, selected, metric, family=True),
                }
            result["variants"][role][scope] = {
                **summary,
                "declared_language": interval([r["declared_language_correct"] for r in cohort]),
                "absolute_intervals": absolute,
                "confusion": {
                    intent: {
                        prediction: sum(
                            r["intent"] == intent and str(r.get("raw_intent")) == prediction
                            for r in cohort
                        )
                        for prediction in sorted({str(r.get("raw_intent")) for r in cohort})
                    }
                    for intent in sorted({r["intent"] for r in cohort})
                },
                "error_census": [
                    {
                        "id": r["id"],
                        "family_id": r["family_id"],
                        "intent": r["intent"],
                        "observed_intent": r.get("effective_intent"),
                        "expected_action": r["expected_action"],
                        "observed_action": r.get("effective_action"),
                        "expected_delivery": r["expected_delivery"],
                        "observed_delivery": r.get("result", {}).get("delivery"),
                        "reason": r.get("reason"),
                        "critical": r["critical"],
                        "tool_success": r["tool_success"],
                        "dialogue_success": r["dialogue_success"]
                        if r["followup_expected"]
                        else None,
                    }
                    for r in cohort
                    if not r["delivery_correct"]
                    or not r["intent_correct"]
                    or not r["action_correct"]
                ][:50],
            }
    path = base / "confirmation_analysis.json"
    write_json(path, result)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "evaluate", "select", "report", "analyse"))
    parser.add_argument("--phase", choices=("development", "confirmation"), default="development")
    parser.add_argument("--variant", choices=("control", "candidate"), default="candidate")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--control", type=Path)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    settings = load_settings()
    if args.stage == "freeze":
        path = freeze(settings)
    elif args.stage == "evaluate":
        if args.smoke and args.phase == "confirmation":
            raise ValueError("no confirmation smoke")
        path = evaluate(settings, args.phase, args.variant, smoke=args.smoke)
    elif args.stage == "select":
        if args.control is None or args.candidate is None:
            raise ValueError("two reports required")
        path = select(settings, args.control, args.candidate)
    elif args.stage == "analyse":
        path = analyse(settings)
    else:
        base = location(settings)
        left = read_json(
            Path(read_json(base / "confirmation_control.json")["attempt"]) / "report.json"
        )
        right = read_json(
            Path(read_json(base / "confirmation_candidate.json")["attempt"]) / "report.json"
        )
        if any(r["status"] != "FINISHED" for r in (left, right)):
            raise ValueError("confirmation incomplete")
        path = base / "confirmation_comparison.json"
        write_json(
            path,
            {
                "identity": left["identity"],
                "control": left["summary"],
                "candidate": right["summary"],
                "paired": compare(left, right),
                "no_retuning": True,
            },
        )
    print(json.dumps({"path": str(path)}))


if __name__ == "__main__":
    main()
