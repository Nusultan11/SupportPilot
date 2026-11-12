"""`supportpilot` command line. Results are JSON on stdout; logs go to stderr.

Exit codes: 0 ok, 1 check/verification failed, 2 usage or operation error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from supportpilot.common.config import Settings, load_settings
from supportpilot.common.logging import configure_logging, get_logger
from supportpilot.common.manifest import load_source_registry, write_model_json
from supportpilot.common.provenance import code_commit
from supportpilot.common.schemas import SourceEntry, SourceStatus
from supportpilot.ops import backup

log = get_logger("supportpilot.cli")


def _emit(obj: BaseModel | dict[str, Any]) -> None:
    if isinstance(obj, BaseModel):
        text = obj.model_dump_json(indent=2)
    else:
        text = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    sys.stdout.write(text + "\n")


def _backup_root(settings: Settings) -> Path:
    return backup.ensure_backup_root(settings.resolve(settings.backup.root), settings.root)


def cmd_env(args: argparse.Namespace, settings: Settings) -> int:
    from supportpilot.ops.envcheck import collect

    report = collect(settings.root, gpu_benchmark=not args.no_benchmark)
    if args.save:
        out = settings.resolve(settings.paths.reports_dir) / "env"
        path = out / f"env-{report.created_at:%Y%m%dT%H%M%SZ}.json"
        write_model_json(path, report)
        log.info("env report saved", extra={"path": str(path)})
    _emit(report)
    return 0 if not report.problems else 1


def cmd_check(args: argparse.Namespace, settings: Settings) -> int:
    from supportpilot.ops import health

    if args.target == "mlflow":
        result = health.check_mlflow(settings)
    else:
        cfg = settings.qdrant
        if args.mode:
            cfg = cfg.model_copy(update={"mode": args.mode})
        result = health.check_qdrant(settings, cfg)
    _emit(result)
    return 0 if result.ok else 1


def cmd_backup(args: argparse.Namespace, settings: Settings) -> int:
    root = _backup_root(settings)
    match args.action:
        case "create":
            manifest, created = backup.create_snapshot(
                Path(args.source),
                name=args.name,
                kind=args.kind,
                root=root,
                project_root=settings.root,
                code_commit=code_commit(settings.root),
                note=args.note,
                pin=args.pin,
            )
            log.info(
                "snapshot created" if created else "unchanged content; existing snapshot reused",
                extra={"snapshot_id": manifest.snapshot_id, "files": len(manifest.files)},
            )
            path = root / manifest.kind / manifest.name / manifest.snapshot_id
            _emit(
                {
                    "created": created,
                    "path": str(path),
                    "manifest": manifest.model_dump(mode="json"),
                }
            )
            return 0
        case "list":
            items = backup.list_snapshots(root, args.kind, args.name)
            rows = [
                {
                    "ref": f"{m.kind}/{m.name}/{m.snapshot_id}",
                    "pinned": m.pinned,
                    "files": len(m.files),
                    "total_bytes": m.total_bytes,
                    "created_at": m.created_at.isoformat(),
                }
                for m in items
            ]
            _emit({"snapshots": rows})
            return 0
        case "verify":
            result = backup.verify_snapshot(
                backup.resolve_snapshot(root, args.snapshot), deep=not args.quick
            )
            _emit(result)
            return 0 if result.ok else 1
        case "restore":
            snap = backup.resolve_snapshot(root, args.snapshot)
            m = backup.restore_snapshot(snap, Path(args.target))
            target = str(Path(args.target).resolve())
            _emit({"restored": m.snapshot_id, "target": target, "files": len(m.files)})
            return 0
        case "pin":
            m = backup.set_pinned(backup.resolve_snapshot(root, args.snapshot), not args.unpin)
            _emit({"snapshot_id": m.snapshot_id, "pinned": m.pinned})
            return 0
        case "prune":
            keep = args.keep or settings.backup.keep_per_name
            doomed = backup.prune(root, keep=keep, dry_run=not args.yes)
            key = "deleted" if args.yes else "would_delete"
            _emit({"dry_run": not args.yes, "keep": keep, key: [str(d) for d in doomed]})
            return 0
    raise AssertionError(args.action)  # pragma: no cover


def _selected_entries(args: argparse.Namespace, settings: Settings) -> list[SourceEntry]:
    registry = load_source_registry(settings.resolve(settings.data.registry))
    if args.all:
        return [s for s in registry.sources if s.status is SourceStatus.SELECTED]
    if not args.sources:
        raise ValueError("name one or more source ids, or pass --all (all selected sources)")
    unknown = [sid for sid in args.sources if sid not in {s.source_id for s in registry.sources}]
    if unknown:
        raise ValueError(f"unknown source id(s): {unknown}")
    return [registry.get(sid) for sid in args.sources]


def cmd_data(args: argparse.Namespace, settings: Settings) -> int:
    from supportpilot.data.acquire import HfDatasetRemote, acquire_source, raw_dir, verify_raw
    from supportpilot.data.convert import convert_source, records_dir, verify_records

    data_dir = settings.resolve(settings.paths.data_dir)
    if args.action == "sources":
        registry = load_source_registry(settings.resolve(settings.data.registry))
        rows = []
        for s in registry.sources:
            if args.status and s.status.value != args.status:
                continue
            rows.append(
                {
                    "source_id": s.source_id,
                    "status": s.status.value,
                    "allowed_use": s.allowed_use.value,
                    "licence": s.licence,
                    "languages": s.languages,
                    "acquired": (raw_dir(data_dir, s) / "manifest.json").is_file(),
                    "converted": (records_dir(data_dir, s) / "manifest.json").is_file(),
                }
            )
        _emit({"sources": rows})
        return 0

    if args.action in ("clean", "gates", "eda"):
        return _cmd_stage3(args, settings)
    if args.action in STAGE4_ACTIONS:
        return _cmd_stage4(args, settings)
    entries = _selected_entries(args, settings)
    if args.action == "acquire":
        results = []
        for e in entries:
            if e.fetch is None:
                raise ValueError(f"{e.source_id}: registry entry has no fetch spec")
            remote = HfDatasetRemote(e.fetch.repo_id, e.revision)
            r = acquire_source(e, data_dir, remote, retries=settings.data.download_retries)
            results.append(r.model_dump())
        _emit({"acquired": results})
        return 0
    if args.action == "convert":
        commit = code_commit(settings.root)
        results = [
            convert_source(
                e,
                data_dir,
                max_reject_ratio=settings.data.max_reject_ratio,
                code_commit=commit,
                force=args.force,
            ).model_dump()
            for e in entries
        ]
        _emit({"converted": results})
        return 0
    # verify
    report = []
    failed = False
    for e in entries:
        raw, rec = raw_dir(data_dir, e), records_dir(data_dir, e)
        item: dict[str, Any] = {"source_id": e.source_id, "raw": None, "records": None}
        if raw.exists():
            raw_problems = verify_raw(raw)
            item["raw"] = {
                "ok": not raw_problems,
                "problems": [p.model_dump() for p in raw_problems],
            }
            failed |= bool(raw_problems)
        if rec.exists():
            rec_problems = verify_records(rec)
            item["records"] = {
                "ok": not rec_problems,
                "problems": [p.model_dump() for p in rec_problems],
            }
            failed |= bool(rec_problems)
        if item["raw"] is None:
            item["raw"] = "not acquired"
        report.append(item)
    _emit({"verify": report})
    return 1 if failed else 0


def _cmd_stage3(args: argparse.Namespace, settings: Settings) -> int:
    from supportpilot.data import pipeline
    from supportpilot.data.build import BuildError

    registry = load_source_registry(settings.resolve(settings.data.registry))
    selected = [s for s in registry.sources if s.status is SourceStatus.SELECTED]
    if args.action == "gates":
        report = pipeline.regate(settings, selected)
        _emit(report)
        return 0 if report.passed else 1
    if args.action == "eda":
        from supportpilot.data.eda import run_eda

        summary = run_eda(settings, selected, tokens=not args.no_tokens)
        _emit(summary)
        return 0
    entries = selected if args.all else _selected_entries(args, settings)
    try:
        result = pipeline.prepare(
            settings,
            entries,
            build=args.all,
            force=args.force,
            code_commit=code_commit(settings.root),
        )
    except BuildError as exc:
        log.error("build failed", extra={"error": str(exc)})
        return 1
    _emit(result)
    return 0 if result.gates is None or result.gates.passed else 1


STAGE4_ACTIONS = (
    "taxonomy",
    "select",
    "native",
    "kb-train",
    "mt-compare",
    "translate",
    "qe-score",
    "sft",
    "gold",
    "retrieval-set",
)


def _cmd_stage4(args: argparse.Namespace, settings: Settings) -> int:
    from supportpilot.data import stage4

    registry = load_source_registry(settings.resolve(settings.data.registry))
    selected = [s for s in registry.sources if s.status is SourceStatus.SELECTED]
    commit = code_commit(settings.root)
    force = getattr(args, "force", False)
    match args.action:
        case "taxonomy":
            _emit(stage4.taxonomy_report(settings))
            return 0
        case "select":
            _emit(stage4.run_select(settings, selected, code_commit=commit, force=force))
            return 0
        case "native":
            _emit(stage4.run_native(settings, selected, code_commit=commit, force=force))
            return 0
        case "kb-train":
            _emit(stage4.run_kb(settings, code_commit=commit, force=force))
            return 0
        case "mt-compare":
            from supportpilot.data.mtrun import run_compare

            _emit(run_compare(settings))
            return 0
        case "translate":
            from supportpilot.data.mtrun import run_translate

            res = run_translate(settings, args.lang, limit=args.limit, back=not args.no_back)
            _emit({"translated": [r.model_dump() for r in res]})
            return 0
        case "qe-score":
            from supportpilot.data.mtrun import run_score

            _emit(run_score(settings, code_commit=commit, force=force))
            return 0
        case "sft":
            from supportpilot.data.mtrun import run_score
            from supportpilot.data.sftbuild import build_sft

            steps = {
                "select": stage4.run_select(settings, selected, code_commit=commit, force=False),
                "native": stage4.run_native(settings, selected, code_commit=commit, force=False),
                "kb_train": stage4.run_kb(settings, code_commit=commit, force=False),
                "qe_score": run_score(settings, code_commit=commit, force=False),
            }
            result = build_sft(
                settings, selected, code_commit=commit, force=force, semantic=not args.no_semantic
            )
            _emit({"steps": steps, "sft": result})
            return 0 if result.get("gates_passed", True) else 1
        case "gold":
            from supportpilot.data.gold import build_gold

            report = build_gold(settings, code_commit=commit, semantic=not args.no_semantic)
            _emit(report)
            return 0 if report["passed"] else 1
        case "retrieval-set":
            from supportpilot.data.retrievalset import build_retrieval_set

            report = build_retrieval_set(
                settings, code_commit=commit, semantic=not args.no_semantic
            )
            _emit(report)
            return 0 if report["passed"] else 1
    raise AssertionError(args.action)  # pragma: no cover


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="supportpilot", description=__doc__)
    p.add_argument("--config", type=Path, help="YAML config (default: configs/base.yaml)")
    sub = p.add_subparsers(dest="command", required=True)

    env = sub.add_parser("env", help="hardware / CUDA report")
    env.add_argument("--save", action="store_true", help="write JSON to artifacts/reports/env/")
    env.add_argument("--no-benchmark", action="store_true", help="skip the GPU matmul test")
    env.set_defaults(func=cmd_env)

    chk = sub.add_parser("check", help="service round-trip checks")
    chk.add_argument("target", choices=["mlflow", "qdrant"])
    chk.add_argument("--mode", choices=["server", "embedded", "memory"], help="qdrant mode")
    chk.set_defaults(func=cmd_check)

    bk = sub.add_parser("backup", help="data/model snapshots (git-ignored backups/ dir)")
    bsub = bk.add_subparsers(dest="action", required=True)
    c = bsub.add_parser("create", help="snapshot a file or directory")
    c.add_argument("source")
    c.add_argument("--name", required=True, help="slug, e.g. bitext_raw")
    c.add_argument("--kind", choices=backup.KINDS, required=True)
    c.add_argument("--note")
    c.add_argument("--pin", action="store_true", help="exempt from pruning (stage-end)")
    ls = bsub.add_parser("list")
    ls.add_argument("--kind", choices=backup.KINDS)
    ls.add_argument("--name")
    v = bsub.add_parser("verify")
    v.add_argument("snapshot", help="path or <kind>/<name>/<snapshot_id>")
    v.add_argument("--quick", action="store_true", help="archive checksum only")
    r = bsub.add_parser("restore")
    r.add_argument("snapshot")
    r.add_argument("--target", required=True, help="new or empty directory")
    pn = bsub.add_parser("pin")
    pn.add_argument("snapshot")
    pn.add_argument("--unpin", action="store_true")
    pr = bsub.add_parser("prune", help="apply retention (dry run unless --yes)")
    pr.add_argument("--keep", type=int)
    pr.add_argument("--yes", action="store_true", help="actually delete")
    bk.set_defaults(func=cmd_backup)

    dt = sub.add_parser("data", help="source registry, acquisition, conversion")
    dsub = dt.add_subparsers(dest="action", required=True)
    src = dsub.add_parser("sources", help="registry with local acquisition/conversion state")
    src.add_argument("--status", choices=[s.value for s in SourceStatus])
    for name, helptext in (
        ("acquire", "download pinned revisions into data/raw (skips verified ones)"),
        ("verify", "deep-check raw data and records against their manifests"),
        ("convert", "raw files -> data/interim/records (skips up-to-date ones)"),
    ):
        a = dsub.add_parser(name, help=helptext)
        a.add_argument("sources", nargs="*", help="source ids (see `data sources`)")
        a.add_argument("--all", action="store_true", help="all sources with status=selected")
        if name == "convert":
            a.add_argument("--force", action="store_true", help="rebuild even if up to date")
    cl = dsub.add_parser(
        "clean",
        help="Stage 3: records -> data/interim/clean; with --all also build "
        "data/interim/canonical (dedup, groups, canonical/pending/rejected) and run the gates",
    )
    cl.add_argument("sources", nargs="*", help="clean only these sources (no build)")
    cl.add_argument("--all", action="store_true", help="all selected sources + build + gates")
    cl.add_argument("--force", action="store_true", help="rebuild even if up to date")
    dsub.add_parser("gates", help="re-run the quality gates on data/interim/canonical")
    ed = dsub.add_parser("eda", help="EDA report -> artifacts/reports/eda/")
    ed.add_argument("--no-tokens", action="store_true", help="skip Qwen token counts (offline)")
    dsub.add_parser("taxonomy", help="Stage 4: validate taxonomy v1 and report the relabelling")
    for name, helptext in (
        ("select", "Stage 4: v1 relabel + EN backbone selection -> data/interim/stage4/selected"),
        ("native", "Stage 4: native-language source rows -> data/interim/stage4/native"),
        ("kb-train", "Stage 4: training KB variant -> data/processed/kb_train"),
        ("qe-score", "Stage 4: QE of cached translations -> data/interim/stage4/translated"),
    ):
        a = dsub.add_parser(name, help=helptext)
        a.add_argument("--force", action="store_true", help="rebuild even if up to date")
    dsub.add_parser("mt-compare", help="Stage 4: compare MT systems on a sample (GPU)")
    tr = dsub.add_parser("translate", help="Stage 4: translate the backbone queue (GPU, resumable)")
    tr.add_argument("--lang", choices=["ru", "kk"], required=True)
    tr.add_argument("--limit", type=int, help="translate at most N new texts")
    tr.add_argument("--no-back", action="store_true", help="skip back-translation")
    sf = dsub.add_parser(
        "sft", help="Stage 4: build the SFT dataset -> data/processed/sft/<version>"
    )
    sf.add_argument("--force", action="store_true", help="rebuild even if up to date")
    sf.add_argument("--no-semantic", action="store_true", help="skip the e5 semantic leakage check")
    gd = dsub.add_parser("gold", help="Stage 4: build the gold benchmark -> data/gold/<version>")
    gd.add_argument("--no-semantic", action="store_true", help="skip the e5 semantic leakage check")
    rq = dsub.add_parser(
        "retrieval-set", help="build independent KB-bound retrieval queries and qrels"
    )
    rq.add_argument(
        "--no-semantic", action="store_true", help="skip diagnostic semantic comparison"
    )
    dt.set_defaults(func=cmd_data)

    from supportpilot.kb.cli import add_parser as add_kb_parser

    add_kb_parser(sub)
    from supportpilot.retrieval.cli import add_parser as add_retrieval_parser

    add_retrieval_parser(sub)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.config)
    except Exception as exc:
        sys.stderr.write(f"config error: {exc}\n")
        return 2
    configure_logging(settings.logging)
    log.debug("start", extra={"command": args.command, "at": datetime.now(UTC).isoformat()})
    from supportpilot.data.acquire import AcquireError
    from supportpilot.data.build import BuildError
    from supportpilot.data.converters import ConversionError
    from supportpilot.data.policy import PolicyError
    from supportpilot.data.steps import StepError

    try:
        return int(args.func(args, settings))
    except (
        backup.BackupError,
        AcquireError,
        BuildError,
        ConversionError,
        PolicyError,
        StepError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        log.error("command failed", extra={"error": str(exc)})
        return 2
