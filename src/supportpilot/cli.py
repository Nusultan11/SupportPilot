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
from supportpilot.common.manifest import write_model_json
from supportpilot.common.provenance import code_commit
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

    bk = sub.add_parser("backup", help="data/model snapshots outside the repo")
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
    try:
        return int(args.func(args, settings))
    except (backup.BackupError, FileNotFoundError, ValueError) as exc:
        log.error("command failed", extra={"error": str(exc)})
        return 2
