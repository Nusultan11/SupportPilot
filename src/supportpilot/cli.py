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

    result = health.check_mlflow(settings)
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
    from supportpilot.data.acquire import acquire_source, raw_dir, remote_for, verify_raw

    data_dir = settings.resolve(settings.paths.data_dir)
    if args.action == "sources":
        registry = load_source_registry(settings.resolve(settings.data.registry))
        rows = [
            {
                "source_id": s.source_id,
                "status": s.status.value,
                "allowed_use": s.allowed_use.value,
                "licence": s.licence,
                "languages": s.languages,
                "acquired": (raw_dir(data_dir, s) / "manifest.json").is_file(),
            }
            for s in registry.sources
            if not args.status or s.status.value == args.status
        ]
        _emit({"sources": rows})
        return 0
    if args.action == "taxonomy":
        from supportpilot.data.taxonomy import load_taxonomy

        tax = load_taxonomy(settings.resolve(settings.data.taxonomy))
        _emit({"version": tax.version, "intents": len(tax.intents), "actions": tax.actions})
        return 0
    if args.action == "corpus":
        return _cmd_corpus(args, settings, data_dir)
    if args.action == "collection":
        return _cmd_collection(args, settings, data_dir)
    if args.action == "review":
        return _cmd_review(args, settings, data_dir)
    entries = _selected_entries(args, settings)
    if args.action == "acquire":
        results = []
        for e in entries:
            if e.fetch is None:
                raise ValueError(f"{e.source_id}: registry entry has no fetch spec")
            r = acquire_source(e, data_dir, remote_for(e), retries=settings.data.download_retries)
            results.append(r.model_dump())
        _emit({"acquired": results})
        return 0
    # verify
    report: list[dict[str, Any]] = []
    failed = False
    for e in entries:
        raw = raw_dir(data_dir, e)
        if not raw.exists():
            report.append({"source_id": e.source_id, "raw": "not acquired"})
            continue
        problems = verify_raw(raw)
        failed |= bool(problems)
        report.append(
            {
                "source_id": e.source_id,
                "raw": {"ok": not problems, "problems": [p.model_dump() for p in problems]},
            }
        )
    _emit({"verify": report})
    return 1 if failed else 0


def _cmd_corpus(args: argparse.Namespace, settings: Settings, data_dir: Path) -> int:
    from supportpilot.data.corpus import build_corpus, latest_corpus, load_inputs, verify_corpus

    out_root = data_dir / "corpus"
    if args.corpus_action == "build":
        inputs = load_inputs(
            settings.resolve(settings.data.corpus_recipe),
            settings.resolve(settings.data.taxonomy),
            settings.resolve(settings.data.registry),
        )
        res = build_corpus(data_dir, out_root, inputs, code_commit(settings.root))
        _emit(
            {
                "version": res.version,
                "path": str(res.path),
                "created": res.created,
                "counts": res.counts,
                "leakage": res.leakage,
            }
        )
        return 0
    path = Path(args.path) if args.path else latest_corpus(out_root)
    problems = verify_corpus(path)
    _emit({"path": str(path), "ok": not problems, "problems": problems})
    return 1 if problems else 0


def _cmd_collection(args: argparse.Namespace, settings: Settings, data_dir: Path) -> int:
    from supportpilot.common.io import atomic_write_text
    from supportpilot.data import collection as col
    from supportpilot.data.taxonomy import load_taxonomy

    plan = col.load_plan(settings.resolve(settings.data.collection_plan))
    taxonomy = load_taxonomy(settings.resolve(settings.data.taxonomy))
    if args.collection_action == "briefs":
        briefs = col.make_briefs(plan, taxonomy)
        out = (
            Path(args.out)
            if args.out
            else settings.resolve(settings.paths.artifacts_dir) / "collection" / plan.plan_id
        )
        atomic_write_text(out / col.BRIEFS_FILE, col.briefs_csv(briefs))
        atomic_write_text(out / col.ROWS_FILE, ",".join(col.ROW_COLUMNS) + "\n")
        _emit({"path": str(out), "briefs": len(briefs), "slices": sorted(plan.slices)})
        return 0
    path = Path(args.path)
    if args.collection_action == "check":
        report = col.check_delivery(path, plan, taxonomy)
    else:
        entry = load_source_registry(settings.resolve(settings.data.registry)).get(plan.source_id)
        report = col.intake_delivery(path, data_dir, entry, plan, taxonomy)
    _emit(report.model_dump())
    return 0 if report.ok else 1


def _cmd_review(args: argparse.Namespace, settings: Settings, data_dir: Path) -> int:
    from supportpilot.common.io import atomic_write_text
    from supportpilot.data import review
    from supportpilot.data.acquire import raw_dir
    from supportpilot.data.collection import load_plan, writers_by_example
    from supportpilot.data.corpus import latest_corpus, load_examples
    from supportpilot.data.taxonomy import load_taxonomy

    plan = load_plan(settings.resolve(settings.data.collection_plan))
    if args.review_action == "package":
        corpus = Path(args.corpus) if args.corpus else latest_corpus(data_dir / "corpus")
        out = (
            Path(args.out)
            if args.out
            else settings.resolve(settings.paths.artifacts_dir) / "review" / corpus.name
        )
        only = frozenset(args.strata.split(",")) if args.strata else None
        files = review.make_package(load_examples(corpus), plan, only)
        for name, text in files.items():
            atomic_write_text(out / name, text)
        _emit({"path": str(out), "rows": files["key.csv"].count("\n") - 1})
        return 0
    path = Path(args.path)
    registry = load_source_registry(settings.resolve(settings.data.registry))
    writers: dict[str, str] = {}
    known = {s.source_id: s for s in registry.sources}
    if plan.source_id in known:
        raw = raw_dir(data_dir, known[plan.source_id])
        if raw.is_dir():
            writers = writers_by_example(raw, plan.source_id)
    tax = load_taxonomy(settings.resolve(settings.data.taxonomy))
    report = review.score_review(
        path, args.reviewer_a, args.reviewer_b, plan, frozenset(tax.intents), writers
    )
    atomic_write_text(path / "review_report.json", report.model_dump_json(indent=2) + "\n")
    atomic_write_text(path / "REVIEW.md", review.report_markdown(report))
    _emit(report.model_dump())
    return 0 if not report.problems else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="supportpilot", description=__doc__)
    p.add_argument("--config", type=Path, help="YAML config (default: configs/base.yaml)")
    sub = p.add_subparsers(dest="command", required=True)

    env = sub.add_parser("env", help="hardware / CUDA report")
    env.add_argument("--save", action="store_true", help="write JSON to artifacts/reports/env/")
    env.add_argument("--no-benchmark", action="store_true", help="skip the GPU matmul test")
    env.set_defaults(func=cmd_env)

    chk = sub.add_parser("check", help="service round-trip checks")
    chk.add_argument("target", choices=["mlflow"])
    chk.set_defaults(func=cmd_check)

    bk = sub.add_parser("backup", help="data/model snapshots (git-ignored backups/ dir)")
    bsub = bk.add_subparsers(dest="action", required=True)
    c = bsub.add_parser("create", help="snapshot a file or directory")
    c.add_argument("source")
    c.add_argument("--name", required=True, help="slug, e.g. abcd_raw")
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

    dt = sub.add_parser("data", help="source registry, acquisition, verification")
    dsub = dt.add_subparsers(dest="action", required=True)
    src = dsub.add_parser("sources", help="registry with local acquisition state")
    src.add_argument("--status", choices=[s.value for s in SourceStatus])
    for name, helptext in (
        ("acquire", "download pinned revisions into data/raw (skips verified ones)"),
        ("verify", "deep-check raw data against its manifest"),
    ):
        a = dsub.add_parser(name, help=helptext)
        a.add_argument("sources", nargs="*", help="source ids (see `data sources`)")
        a.add_argument("--all", action="store_true", help="all sources with status=selected")
    dsub.add_parser("taxonomy", help="validate configs/taxonomy.yaml")
    cp = dsub.add_parser("corpus", help="build or verify the versioned Stage 1 corpus")
    csub = cp.add_subparsers(dest="corpus_action", required=True)
    csub.add_parser("build", help="raw sources -> data/corpus/<version>/ (reuses identical)")
    cv = csub.add_parser("verify", help="re-hash a published corpus against its manifest")
    cv.add_argument("path", nargs="?", help="corpus directory (default: latest)")
    co = dsub.add_parser("collection", help="commissioned native-speaker data (ADR-049)")
    cosub = co.add_subparsers(dest="collection_action", required=True)
    cb = cosub.add_parser("briefs", help="scenario briefs + empty rows.csv for writers")
    cb.add_argument("--out", help="directory (default: artifacts/collection/<plan_id>)")
    for name, helptext in (
        ("check", "run all intake checks on a delivery directory"),
        ("intake", "check, then copy the delivery into immutable data/raw"),
    ):
        cc = cosub.add_parser(name, help=helptext)
        cc.add_argument("path", help="delivery directory")
    rv = dsub.add_parser("review", help="independent blind label review (ADR-049)")
    rvsub = rv.add_subparsers(dest="review_action", required=True)
    rp = rvsub.add_parser("package", help="blind reviewer sheets + owner key")
    rp.add_argument("--corpus", help="corpus directory (default: latest)")
    rp.add_argument("--out", help="directory (default: artifacts/review/<corpus version>)")
    rp.add_argument("--strata", help="comma-separated stratum names (default: all)")
    rs = rvsub.add_parser("score", help="agreement, kappa and disagreements of filled sheets")
    rs.add_argument("path", help="package directory with filled sheet_a/sheet_b")
    rs.add_argument("--reviewer-a", required=True, help="pseudonymous id of reviewer A")
    rs.add_argument("--reviewer-b", required=True, help="pseudonymous id of reviewer B")
    dt.set_defaults(func=cmd_data)
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
    from supportpilot.data.collection import CollectionError
    from supportpilot.data.corpus import CorpusError
    from supportpilot.data.policy import PolicyError
    from supportpilot.data.review import ReviewError

    try:
        return int(args.func(args, settings))
    except (
        backup.BackupError,
        AcquireError,
        CorpusError,
        CollectionError,
        ReviewError,
        PolicyError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        log.error("command failed", extra={"error": str(exc)})
        return 2
