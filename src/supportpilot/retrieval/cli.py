"""Retrieval CLI: index, query/context, experiments and chosen configuration."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from supportpilot.common.config import Settings
from supportpilot.ops.health import make_qdrant_client
from supportpilot.retrieval.encode import Transformer
from supportpilot.retrieval.index import build_artifact, default_recipe, sync_index
from supportpilot.retrieval.models import SearchFilters
from supportpilot.retrieval.search import open_retriever


def command(args: argparse.Namespace, settings: Settings) -> int:
    if args.action == "scale":
        from supportpilot.retrieval.scale_data import prepare_scale
        from supportpilot.retrieval.scale_run import run_scale
        from supportpilot.retrieval.stress import run_stress

        if args.stage == "author":
            from supportpilot.retrieval.scale_author import author_profiles

            result = author_profiles(settings)
        elif args.stage == "load":
            result = run_stress(settings, count=args.count)
        elif args.stage == "update-drill":
            from supportpilot.retrieval.stress import vector_update_drill

            result = vector_update_drill(settings)
        elif args.stage == "profile":
            from supportpilot.retrieval.scale_profile import profile_scale

            if args.frozen is None:
                raise ValueError("--frozen is required for profile")
            result = profile_scale(settings, settings.resolve(args.frozen))
        elif args.stage == "report":
            from supportpilot.retrieval.scale_report import report_scale

            if args.frozen is None:
                raise ValueError("--frozen is required for report")
            result = report_scale(settings, settings.resolve(args.frozen))
        elif args.stage == "recovery":
            from supportpilot.retrieval.scale_recovery import recover_scale

            if args.frozen is None or args.restore_root is None:
                raise ValueError("--frozen and --restore-root are required for recovery")
            result = recover_scale(
                settings, settings.resolve(args.frozen), settings.resolve(args.restore_root)
            )
        elif args.stage == "diagnostic":
            from supportpilot.retrieval.scale_diagnostics import run_diagnostics

            if args.frozen is None:
                raise ValueError("--frozen is required for diagnostic")
            result = run_diagnostics(settings, settings.resolve(args.frozen))
        elif args.stage in ("audit", "freeze"):
            result = prepare_scale(
                settings,
                freeze=args.stage == "freeze",
                semantic=args.semantic,
                kb_version=args.kb_version,
            )
        else:
            if args.frozen is None:
                raise ValueError("--frozen is required for development/holdout")
            result = run_scale(settings, settings.resolve(args.frozen), phase=args.stage)
    elif args.action == "experiment":
        from supportpilot.retrieval.experiments import run_experiments

        result = run_experiments(settings, stage=args.stage)
    elif args.action == "index":
        recipe = default_recipe(settings)
        client = make_qdrant_client(settings.qdrant, settings)
        encoder = Transformer(recipe.encoder, settings.retrieval.device)
        try:
            artifact = build_artifact(settings, recipe, version=args.kb_version, encoder=encoder)
            result = {
                "index_version": artifact.manifest.dataset_version,
                "kb_version": artifact.kb_version,
                "artifact": str(artifact.path),
                **sync_index(settings, artifact, client),
            }
        finally:
            encoder.close()
            client.close()
    else:
        with open_retriever(settings) as retriever:
            filters = SearchFilters(
                all_versions=getattr(args, "all_versions", False),
                corpus_lang=args.lang,
                doc_ids=args.doc_id,
                doc_types=args.doc_type,
                as_of=args.as_of,
            )
            context = retriever.context(args.query, filters=filters, k=args.k, budget=args.budget)
            result = {**context.model_dump(mode="json"), "timings": retriever.last_timings}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("passed", True) else 1


def add_parser(sub: Any) -> None:
    parser = sub.add_parser("retrieval", help="Stage 6: versioned multilingual evidence retrieval")
    actions = parser.add_subparsers(dest="action", required=True)
    index = actions.add_parser("index", help="build/resume/synchronise active KB index")
    index.add_argument("--kb-version", help="prepare a specific immutable KB version")
    query = actions.add_parser("query", help="retrieve source evidence within a context budget")
    query.add_argument("query")
    query.add_argument(
        "--lang", choices=["en", "ru", "kk"], help="corpus language; omit for all languages"
    )
    query.add_argument("--doc-id", action="append", default=[])
    query.add_argument("--doc-type", action="append", default=[])
    query.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    query.add_argument(
        "--all-versions",
        action="store_true",
        help="explicit historical comparison; retains validity metadata",
    )
    query.add_argument("--k", type=int, default=10)
    query.add_argument("--budget", type=int)
    experiment = actions.add_parser("experiment", help="reproduce Stage 6 comparisons and reports")
    experiment.add_argument(
        "--stage",
        choices=["prepare", "compare", "batch", "rerank", "report", "profile", "drill", "all"],
        default="all",
    )
    scale = actions.add_parser(
        "scale", help="Stage 6.1: audited inputs, frozen evaluation, isolated load"
    )
    scale.add_argument(
        "--stage",
        required=True,
        choices=[
            "author",
            "audit",
            "freeze",
            "development",
            "holdout",
            "load",
            "profile",
            "report",
            "update-drill",
            "recovery",
            "diagnostic",
        ],
    )
    scale.add_argument(
        "--semantic", action="store_true", help="compute semantic leakage diagnostics"
    )
    scale.add_argument("--frozen", type=Path, help="immutable Stage 6.1 benchmark directory")
    scale.add_argument("--count", type=int, help="load smoke size; default 100000")
    scale.add_argument("--kb-version", help="audit/freeze a staged KB without activating it")
    scale.add_argument("--restore-root", type=Path, help="prepared isolated backup restore runtime")
    parser.set_defaults(func=command)
