"""`supportpilot kb ...` commands (Stage 5). Results are JSON on stdout.

kb render              specs + facts -> data/kb/<kb>/sources (idempotent)
kb build [--force]     render, then build and activate a new version if anything changed
kb validate [--version V]   re-run every gate on a stored version
kb status              current version, versions, whether sources match it
kb activate VERSION    switch the current version (rollback)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from supportpilot.common.config import Settings
from supportpilot.common.provenance import code_commit
from supportpilot.data.steps import StepError
from supportpilot.kb.blocks import ExtractionError
from supportpilot.kb.build import KBConfig, TokenCounter, activate, build, render, status, validate
from supportpilot.kb.spec import SpecError


def add_parser(sub: Any) -> None:
    kb = sub.add_parser("kb", help="Stage 5: demo-store knowledge base (render, build, validate)")
    ksub = kb.add_subparsers(dest="action", required=True)
    ksub.add_parser("render", help="specs + facts -> source documents (PDF/DOCX/XLSX/CSV/JSON/TXT)")
    b = ksub.add_parser("build", help="render, extract, chunk, gate and activate a new version")
    b.add_argument("--force", action="store_true", help="re-parse every document")
    v = ksub.add_parser("validate", help="re-run the gates on a stored version")
    v.add_argument("--version", help="KB version (default: current)")
    ksub.add_parser("status", help="current version and stored versions")
    a = ksub.add_parser("activate", help="make a stored version current (rollback)")
    a.add_argument("version")
    kb.set_defaults(func=run)


def kb_config(settings: Settings) -> KBConfig:
    cfg = settings.kb
    tok = _tokenizer_spec(settings)
    gold = settings.resolve(cfg.gold_dir) if cfg.gold_dir else None
    return KBConfig(
        name=cfg.name,
        source_root=settings.resolve(cfg.source_root),
        locale=settings.resolve(settings.stage4.locale),
        kb_dir=settings.resolve(settings.paths.data_dir) / "kb" / cfg.name,
        max_tokens=cfg.max_tokens,
        lid_min_letters=cfg.lid_min_letters,
        tokenizer=f"{tok.repo_id}@{tok.revision}",
        gold_dir=gold,
    )


def _tokenizer_spec(settings: Settings) -> Any:
    for spec in settings.eda.tokenizers:
        if spec.name == settings.kb.tokenizer:
            return spec
    raise StepError(f"kb.tokenizer {settings.kb.tokenizer!r} is not in eda.tokenizers")


def token_counter(settings: Settings) -> TokenCounter:
    from supportpilot.data.eda import load_tokenizer

    tok = load_tokenizer(_tokenizer_spec(settings))

    def count(texts: Sequence[str]) -> list[int]:
        return [len(e.ids) for e in tok.encode_batch(list(texts), add_special_tokens=False)]

    return count


def _emit(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")


def run(args: argparse.Namespace, settings: Settings) -> int:
    cfg = kb_config(settings)
    try:
        match args.action:
            case "render":
                _emit(render(cfg))
            case "build":
                rendered = render(cfg)
                report = build(cfg, token_counter(settings), code_commit=code_commit(settings.root),
                               force=args.force)  # fmt: skip
                _emit({"render": rendered, "build": report})
            case "validate":
                report = validate(cfg, args.version)
                _emit(report)
                return 0 if report["passed"] else 1
            case "status":
                _emit(status(cfg))
            case "activate":
                _emit(activate(cfg, args.version))
    except StepError as e:
        _emit({"error": str(e)})
        return 1 if "gates failed" in str(e) else 2
    except (SpecError, ExtractionError, FileNotFoundError) as e:
        _emit({"error": str(e)})
        return 2
    return 0
