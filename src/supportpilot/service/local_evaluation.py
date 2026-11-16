"""Local-only fresh router comparison; self-authored fixtures never certify semantics."""

from __future__ import annotations

import argparse
import json
import re
import secrets
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np
import polars as pl
import yaml
from fastapi.testclient import TestClient
from pydantic import SecretStr

from supportpilot.api import create_app
from supportpilot.common.config import Settings, load_settings
from supportpilot.common.hashing import sha256_bytes, sha256_file
from supportpilot.common.io import read_json, write_json
from supportpilot.common.schemas import INTENTS, LanguageCode
from supportpilot.common.tracking import RunSpec, start_run
from supportpilot.retrieval.experiments import measure
from supportpilot.routing.router import EncoderRouter
from supportpilot.service.evaluation import SCOPES
from supportpilot.service.evaluation_audit import normalise, strings
from supportpilot.service.quality_evaluation import (
    compare,
    digest,
    observed_turn,
    runtime_hashes,
    summarise,
)
from supportpilot.service.runtime import open_service
from supportpilot.service.store import Store


def root(settings: Settings) -> Path:
    return settings.resolve(settings.paths.reports_dir) / "system/qualification"


def expectations(intent: str) -> tuple[str, str, str | None, str | None]:
    """Authored fixture expectations from taxonomy/product contracts, not model output."""
    reads = {
        "track_order": "get_order",
        "cancel_order": "request_cancellation",
        "change_order": "get_order",
        "delivery_problem": "get_order",
        "request_refund": "get_payment",
        "refund_status": "get_payment",
        "product_defect": "get_order",
        "payment_problem": "get_payment",
        "invoice": "get_payment",
        "availability": "get_stock",
    }
    critical = {
        "personal_data": "privacy_request",
        "technical_issue": "technical_issue",
        "human_agent": "customer_requested_human",
        "complaint": "complaint",
    }
    if intent in reads:
        return "call_tool", "tool", None, reads[intent]
    if intent in critical:
        return "escalate", "queued", critical[intent], None
    if intent in {"greeting", "unclear", "change_address"}:
        return "clarify", "clarification", None, None
    if intent == "out_of_scope":
        return "refuse_oos", "refusal", None, None
    return "answer_kb", "queued", None, None


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text("utf-8"))
    rows = []
    families: set[str] = set()
    for phase, intent, family, *texts in data["rows"]:
        if phase not in {"development", "confirmation"} or intent not in INTENTS:
            raise ValueError("invalid split or intent")
        if family in families or len(texts) != 4 or any(not t.strip() for t in texts):
            raise ValueError("duplicate family or missing language")
        families.add(family)
        action, delivery, reason, tool = expectations(intent)
        rows.append(
            {
                "id": f"local-{phase}-{family}",
                "phase": phase,
                "intent": intent,
                "family_id": family,
                "texts": texts,
                "expected_action": action,
                "expected_delivery": delivery,
                "reason": reason,
                "expected_tool": tool,
            }
        )
    for phase in ("development", "confirmation"):
        if Counter(r["intent"] for r in rows if r["phase"] == phase) != Counter(INTENTS):
            raise ValueError("exactly one screening need per class/split required")
    return rows


def freeze(settings: Settings) -> Path:
    base = root(settings)
    path = base / "local_freeze.json"
    if path.exists():
        verified(settings)
        return path
    source = settings.root / "authored/system/local_quality.yaml"
    protocol = settings.root / "docs/reviews/stage9_local_protocol.md"
    frozen = {
        "source_sha256": sha256_file(source),
        "protocol_sha256": sha256_file(protocol),
        "rows": load_rows(source),
        "runtime": runtime_hashes(settings),
        "heads": {
            name: sha256_file(base / "heads" / (name + ".json")) for name in ("control", "balanced")
        },
        "seed": 20261116,
        "authorship": "LLM_AUTHORED_DRAFT; NO_INDEPENDENT_HUMAN_REVIEW",
        "semantic_independence": "NOT_CERTIFIED",
    }
    write_json(path, {**frozen, "identity": digest(frozen)})
    return path


def verified(settings: Settings) -> dict[str, Any]:
    frozen = dict(read_json(root(settings) / "local_freeze.json"))
    identity = frozen.pop("identity")
    if digest(frozen) != identity:
        raise ValueError("changed freeze")
    for name, key in (
        ("authored/system/local_quality.yaml", "source_sha256"),
        ("docs/reviews/stage9_local_protocol.md", "protocol_sha256"),
    ):
        if sha256_file(settings.root / name) != frozen[key]:
            raise ValueError("changed frozen source/protocol")
    for name, expected in frozen["heads"].items():
        if sha256_file(root(settings) / "heads" / (name + ".json")) != expected:
            raise ValueError("changed head")
    return {**frozen, "identity": identity}


def audit(settings: Settings) -> Path:
    """Reuse frozen reference vectors in their exact original source/text order."""
    frozen = verified(settings)
    base = root(settings)
    out = base / "local_overlap"
    if out.exists():
        raise ValueError("overlap already attempted")
    out.mkdir()
    prior = (
        settings.root / "artifacts/reports/system/semantic_overlap/8279c216c374447cb0b24540ec921bd0"
    )
    old = read_json(prior / "report.json")
    references: dict[str, list[dict[str, Any]]] = {}
    for name, expected in old["reference_sources"].items():
        path = Path(name)
        content = path.read_bytes()
        if sha256_bytes(content) != expected:
            readback = old["reference_readbacks"][name]
            if not readback.startswith("git@"):
                raise ValueError("changed reference")
            content = subprocess.check_output(  # noqa: S603 - trusted local Git object, no shell
                [
                    shutil.which("git") or "git",
                    "show",
                    f"{readback[4:]}:{path.relative_to(settings.root).as_posix()}",
                ],
                cwd=settings.root,
            )
            if sha256_bytes(content) != expected:
                raise ValueError("reference readback mismatch")
        texts = (
            pl.read_parquet(path, columns=["user_text"])["user_text"].to_list()
            if path.suffix == ".parquet"
            else strings(json.loads(content))
            if path.suffix == ".json"
            else re.findall(r'"([^"\n]{10,})"', content.decode("utf-8"))
        )
        for i, text in enumerate(texts):
            if isinstance(text, str) and (key := normalise(text)):
                references.setdefault(key, []).append({"source": name, "row": i})
    vector_path = prior / "reference_vectors.npy"
    if sha256_file(vector_path) != old["vector_sha256"][vector_path.name]:
        raise ValueError("reference vector mismatch")
    vectors = np.load(vector_path, allow_pickle=False)
    if len(vectors) != len(references):
        raise ValueError("reference order/count mismatch")
    extra: list[str] = []
    for source, column in (
        ("authored/system/scenarios.yaml", None),
        ("authored/system/quality.yaml", 8),
    ):
        data = yaml.safe_load((settings.root / source).read_text("utf-8"))
        extra.extend(
            normalise(t)
            for row in data["needs"]
            for t in (row[6:] if column is None else row[column])
        )
    queries = [
        (r, scope, normalise(text))
        for r in frozen["rows"]
        for scope, text in zip(SCOPES, r["texts"], strict=True)
    ]
    cfg = settings.model_copy(
        update={
            "serving": settings.serving.model_copy(
                update={"retrieval_enabled": False, "database": out / "service.sqlite"}
            )
        }
    )
    report: dict[str, Any] = {
        "status": "RUNNING",
        "identity": frozen["identity"],
        "exclusions": [],
        "semantic_independence": "NOT_CERTIFIED",
    }
    try:
        with (
            start_run(
                RunSpec(
                    experiment="system_qualification",
                    run_name="local_overlap",
                    hypothesis="Screen new draft input surfaces without outcome-driven exclusions",
                    dataset_version=frozen["identity"],
                    seed=20261116,
                ),
                settings,
            ) as run,
            open_service(cfg) as service,
        ):
            report["run_id"] = run.info.run_id
            assert isinstance(service.router, EncoderRouter)
            encoder = service.router.encoder
            encoder.spec = encoder.spec.model_copy(update={"batch_size": 8})
            vectors = np.concatenate([vectors, encoder.encode(extra, query=True)])
            reference_texts = [*references, *extra]
            query_vectors = encoder.encode([t for _, _, t in queries], query=True)
            np.save(out / "query_vectors.npy", query_vectors, allow_pickle=False)
            grams = [{t[i : i + 5] for i in range(max(1, len(t) - 4))} for t in reference_texts]
            rows = []
            for i, (need, scope, text) in enumerate(queries):
                scores = vectors @ query_vectors[i]
                qgrams = {text[j : j + 5] for j in range(max(1, len(text) - 4))}
                lexical = [
                    j
                    for j, g in enumerate(grams)
                    if len(g & qgrams) / max(1, len(g | qgrams)) >= 0.8
                ]
                cross = [j for j, (r, _, _) in enumerate(queries) if r["phase"] != need["phase"]]
                cross_scores = query_vectors[cross] @ query_vectors[i]
                rows.append(
                    {
                        "id": need["id"] + ":" + scope,
                        "phase": need["phase"],
                        "lexical_matches": lexical,
                        "max_cosine": float(scores.max()),
                        "embedding_flag": bool(scores.max() >= 0.95),
                        "closest_reference": reference_texts[int(scores.argmax())],
                        "cross_split_cosine": float(cross_scores.max()),
                        "cross_split_flag": bool(cross_scores.max() >= 0.95),
                    }
                )
            report.update(
                status="FINISHED",
                rows=rows,
                reference_texts=len(reference_texts),
                lexical_flags=sum(bool(r["lexical_matches"]) for r in rows),
                embedding_flags=sum(r["embedding_flag"] for r in rows),
                cross_split_flags=sum(r["cross_split_flag"] for r in rows),
                source_sha256=old["reference_sources"],
                prior_report_sha256=sha256_file(prior / "report.json"),
            )
    except BaseException as exc:
        report.update(status="FAILED", error=type(exc).__name__, detail=str(exc))
        raise
    finally:
        write_json(out / "report.json", report)
    return out / "report.json"


def evaluate(settings: Settings, phase: str, variant: str, *, smoke: bool = False) -> Path:
    frozen = verified(settings)
    base = root(settings)
    if frozen["runtime"] != runtime_hashes(settings):
        raise ValueError("runtime differs from freeze")
    if read_json(base / "local_overlap/report.json")["status"] != "FINISHED":
        raise ValueError("terminal overlap audit required")
    if phase == "confirmation" and smoke:
        raise ValueError("no confirmation smoke")
    if phase == "confirmation":
        seal = read_json(base / "local_selection.json")
        if seal["identity"] != frozen["identity"] or seal["runtime"] != frozen["runtime"]:
            raise ValueError("confirmation binding mismatch")
    out = base / "local_attempts" / uuid4().hex
    out.mkdir(parents=True)
    if phase == "confirmation":
        with (base / f"local_confirmation_{variant}.json").open("x", encoding="utf-8") as stream:
            json.dump({"attempt": str(out), "identity": frozen["identity"]}, stream)
    token = secrets.token_urlsafe(32)
    cfg = settings.serving.model_copy(
        update={
            "database": out / "service.sqlite",
            "generation_enabled": False,
            "tokens": {"demo_alice": SecretStr(token)},
        }
    )
    local = settings.model_copy(update={"serving": cfg})
    Store(cfg.database).seed_demo()
    cohort = [r for r in frozen["rows"] if r["phase"] == phase]
    if smoke:
        cohort = cohort[:1]
    report: dict[str, Any] = {
        "status": "RUNNING",
        "phase": phase,
        "variant": variant,
        "smoke": smoke,
        "identity": frozen["identity"],
        "rows": [],
        "runtime": frozen["runtime"],
        "paid_service_cost_usd": 0,
        "semantic_quality": "NOT_EVALUATED",
    }
    write_json(out / "report.json", report)
    try:
        with (
            start_run(
                RunSpec(
                    experiment="system_qualification",
                    run_name=f"local_{phase}_{variant}_{smoke}",
                    hypothesis="Class balancing improves routing without safety loss",
                    dataset_version=frozen["identity"],
                    seed=20261116,
                    params={
                        "variant": variant,
                        "phase": phase,
                        "selection_scope": "fresh_development_only",
                    },
                ),
                settings,
            ) as active,
            measure() as resources,
            open_service(local) as service,
        ):
            report["run_id"] = active.info.run_id
            assert isinstance(service.router, EncoderRouter)
            service.router.head = read_json(base / "heads" / (variant + ".json"))["head"]
            with TestClient(create_app(local, service)) as client:
                client.headers["Authorization"] = "Bearer " + token
                for need in cohort:
                    for scope, text in zip(SCOPES, need["texts"], strict=True):
                        lang = cast(LanguageCode, "kk" if scope == "mixed" else scope)
                        row: dict[str, Any] = {
                            "id": need["id"] + ":" + scope,
                            "scenario_id": need["id"],
                            "family_id": need["family_id"],
                            "scope": scope,
                            "intent": need["intent"],
                            "text": text,
                            "critical": need["reason"] is not None,
                            "expected_tool": need["expected_tool"],
                            "expected_delivery": need["expected_delivery"],
                            "expected_action": need["expected_action"],
                            "followup_expected": False,
                        }
                        tick = time.perf_counter()
                        try:
                            raw = service.router.route(
                                text, {}, language=lang, code_switch=scope == "mixed"
                            )
                            row["encoder_intent"] = raw.intent
                        except Exception as exc:
                            row["encoder_error"] = type(exc).__name__
                        row["raw_route_seconds"] = time.perf_counter() - tick
                        row.update(
                            observed_turn(
                                client,
                                service,
                                text,
                                lang,
                                need["expected_delivery"],
                                need["expected_tool"],
                            )
                        )
                        row.update(
                            raw_intent=row.get("effective_intent"),
                            intent_correct=float(row.get("effective_intent") == need["intent"]),
                            action_correct=float(
                                row.get("effective_action") == need["expected_action"]
                            ),
                            delivery_correct=row["correct"],
                            critical_recall=row["queue_verified"],
                            reason_correct=float(row.get("reason") == need["reason"]),
                            false_critical=float(
                                row.get("reason")
                                in {
                                    "privacy_request",
                                    "technical_issue",
                                    "customer_requested_human",
                                    "complaint",
                                    "safety_risk",
                                    "legal_threat",
                                    "fraud_suspected",
                                    "repeated_failure",
                                    "payment_dispute",
                                    "independent_safety_guard",
                                }
                            ),
                            dialogue_success=0.0,
                        )
                        report["rows"].append(row)
                        write_json(out / "report.json", report)
                        if smoke:
                            break
                    print(phase, variant, need["id"], flush=True)
                report["ready_after"] = client.get("/ready").status_code
            report["summary"] = summarise(report["rows"])
            report["status"] = "FINISHED"
    except BaseException as exc:
        report.update(status="FAILED", error=type(exc).__name__, detail=str(exc))
        raise
    finally:
        if "resources" in locals():
            report["resources"] = vars(resources)
        write_json(out / "report.json", report)
    return out / "report.json"


def select(settings: Settings, control: Path, balanced: Path) -> Path:
    frozen = verified(settings)
    base = root(settings)
    if (base / "local_selection.json").exists() or list(base.glob("local_confirmation_*.json")):
        raise ValueError("decision already sealed")
    a, b = read_json(control), read_json(balanced)
    for role, report in (("control", a), ("balanced", b)):
        if (
            report["phase"] != "development"
            or report["status"] != "FINISHED"
            or report["smoke"]
            or report["variant"] != role
            or report["identity"] != frozen["identity"]
            or report["runtime"] != frozen["runtime"]
            or len(report["rows"]) != 140
        ):
            raise ValueError("matching full development required")
    paired = compare(a, b, seed=20261116)
    x, y = {r["id"]: r for r in a["rows"]}, {r["id"]: r for r in b["rows"]}
    losses = [
        k for k in x if x[k]["critical"] and x[k]["critical_recall"] > y[k]["critical_recall"]
    ]
    unexpected_writes = [
        r["id"]
        for r in b["rows"]
        if any(
            t["name"] == "request_cancellation" and r["expected_tool"] != "request_cancellation"
            for t in r["tool_calls"]
        )
    ]
    positive = any(p["delivery_correct"]["ci95"][0] > 0 for p in paired.values())
    no_loss = all(p["delivery_correct"]["delta"] >= 0 for p in paired.values())
    result = {
        "identity": frozen["identity"],
        "runtime": frozen["runtime"],
        "heads": frozen["heads"],
        "control": str(control),
        "balanced": str(balanced),
        "control_sha256": sha256_file(control),
        "balanced_sha256": sha256_file(balanced),
        "paired": paired,
        "critical_losses": losses,
        "unexpected_writes": unexpected_writes,
        "adopt": positive and no_loss and not losses and not unexpected_writes,
        "decision_scope": (
            "draft operational evidence only; no runtime promotion or human certification"
        ),
    }
    path = base / "local_selection.json"
    write_json(path, result)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "audit", "evaluate", "select", "report"))
    parser.add_argument("--phase", choices=("development", "confirmation"), default="development")
    parser.add_argument("--variant", choices=("control", "balanced"), default="control")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--control", type=Path)
    parser.add_argument("--balanced", type=Path)
    args = parser.parse_args()
    settings = load_settings()
    if args.action == "freeze":
        path = freeze(settings)
    elif args.action == "audit":
        path = audit(settings)
    elif args.action == "evaluate":
        path = evaluate(settings, args.phase, args.variant, smoke=args.smoke)
    elif args.action == "select":
        if args.control is None or args.balanced is None:
            raise ValueError("both full development reports required")
        path = select(settings, args.control, args.balanced)
    else:
        base = root(settings)
        a, b = [
            read_json(
                Path(read_json(base / f"local_confirmation_{v}.json")["attempt"]) / "report.json"
            )
            for v in ("control", "balanced")
        ]
        if any(r["status"] != "FINISHED" for r in (a, b)):
            raise ValueError("terminal confirmation required")
        path = base / "local_confirmation_analysis.json"
        write_json(
            path,
            {
                "control": a["summary"],
                "balanced": b["summary"],
                "paired": compare(a, b, seed=20261116),
                "semantic_quality": "NOT_EVALUATED",
                "independent_review": "NOT_EVALUATED",
            },
        )
    print(json.dumps({"path": str(path)}))


if __name__ == "__main__":
    main()
