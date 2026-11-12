"""Retrieval CLI: index, query/context, experiments and chosen configuration."""

from __future__ import annotations

import argparse
import json
from datetime import date
from typing import Any

from supportpilot.common.config import Settings
from supportpilot.ops.health import make_qdrant_client
from supportpilot.retrieval.encode import Transformer
from supportpilot.retrieval.index import build_artifact, default_recipe, sync_index
from supportpilot.retrieval.models import SearchFilters
from supportpilot.retrieval.search import open_retriever


def command(args: argparse.Namespace, settings: Settings) -> int:
    if args.action == "experiment":
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
                corpus_lang=args.lang,
                doc_ids=args.doc_id,
                doc_types=args.doc_type,
                as_of=args.as_of,
            )
            context = retriever.context(args.query, filters=filters, k=args.k, budget=args.budget)
            result = {**context.model_dump(mode="json"), "timings": retriever.last_timings}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


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
    query.add_argument("--k", type=int, default=10)
    query.add_argument("--budget", type=int)
    experiment = actions.add_parser("experiment", help="reproduce Stage 6 comparisons and reports")
    experiment.add_argument(
        "--stage",
        choices=["prepare", "compare", "batch", "rerank", "report", "profile", "drill", "all"],
        default="all",
    )
    parser.set_defaults(func=command)
