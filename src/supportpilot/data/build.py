"""Stage 3 step 2: cleaned records of all sources -> canonical / pending / rejected.

Output `data/interim/canonical/` (DATA_CONTRACTS §4, §4c, §4d, §9):

    examples/lang=<xx>/part-00000.parquet   CanonicalExample rows (canonical/1)
    pending.parquet                          PendingExample rows (valid, not canonical yet)
    rejected.parquet                         RejectedRecord rows (defects, duplicates)
    groups.parquet                           every kept row -> exact group, family, scenario
    pairs.parquet                            verified near-duplicate pairs (EDA, gates)
    quality_report.json                      gate results (gates.py)
    manifest.json                            DatasetManifest

Every input record lands in exactly one of examples / pending / rejected (gate `accounting`).

Grouping (ADR-022):
1. Exact duplicates: equal `exact_key`. If all members have compatible labels, the first one
   (most permissive licence, then registry order, then row order) is kept and the others are
   rejected as `exact_duplicate`. Incompatible labels -> all members pending `label_conflict`.
2. Families: union of verified near-duplicate pairs with Jaccard >= `family_threshold`
   whose labels are compatible, plus source-provided paraphrase groups (`family_hint`).
3. Scenarios: union of families that share a scenario key (same source label and action, or
   a source-provided scenario id). Lexical similarity alone never merges different labels.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ValidationError

from supportpilot.common.config import CleanSettings
from supportpilot.common.hashing import sha256_file, sha256_text
from supportpilot.common.ids import make_dataset_version
from supportpilot.common.logging import get_logger
from supportpilot.common.manifest import read_model_json, scan_files, verify_files, write_model_json
from supportpilot.common.schemas import (
    USE_RANK,
    AllowedUse,
    CanonicalExample,
    CleanManifest,
    DatasetManifest,
    PendingExample,
    RecordType,
    SourceEntry,
)
from supportpilot.data.acquire import MANIFEST
from supportpilot.data.clean import CLEAN, CLEAN_VERSION, clean_dir, verify_clean
from supportpilot.data.labels import IntentMap, map_labels
from supportpilot.data.neardup import near_duplicates
from supportpilot.data.textnorm import exact_key

log = get_logger(__name__)
BUILD_VERSION = 1
EXAMPLES = "examples"
PENDING = "pending.parquet"
REJECTED = "rejected.parquet"
GROUPS = "groups.parquet"
PAIRS = "pairs.parquet"
QUALITY = "quality_report.json"
DRAFT_FLAG = "labels:intent_map_draft_v0"

QUALITY_T = pa.struct(
    [
        ("lid_conf", pa.float64()),
        ("mt_qe", pa.float64()),
        ("intent_agree", pa.bool_()),
        ("dup_cluster", pa.string()),
        ("flags", pa.list_(pa.string())),
    ]
)
SLOTS_T = pa.map_(pa.string(), pa.string())
CANONICAL_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("example_id", pa.string()),
        ("record_id", pa.string()),
        ("source_id", pa.string()),
        ("scenario_id", pa.string()),
        ("family_id", pa.string()),
        ("lang", pa.string()),
        ("code_switch", pa.bool_()),
        ("text", pa.string()),
        ("reference_response", pa.string()),
        ("intent", pa.string()),
        ("source_intent", pa.string()),
        ("source_category", pa.string()),
        ("action", pa.string()),
        ("escalate", pa.bool_()),
        ("escalation_reason", pa.string()),
        ("slots", SLOTS_T),
        ("origin", pa.string()),
        ("translation_of", pa.string()),
        ("translation_system", pa.string()),
        ("quality", QUALITY_T),
        ("split", pa.string()),
        ("created_at", pa.timestamp("us", tz="UTC")),
    ]
)
PENDING_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("example_id", pa.string()),
        ("record_id", pa.string()),
        ("source_id", pa.string()),
        ("revision", pa.string()),
        ("record_type", pa.string()),
        ("source_split", pa.string()),
        ("scenario_id", pa.string()),
        ("family_id", pa.string()),
        ("lang", pa.string()),
        ("code_switch", pa.bool_()),
        ("langs", pa.list_(pa.string())),
        ("text", pa.string()),
        ("reference_response", pa.string()),
        ("slots", SLOTS_T),
        ("source_intent", pa.string()),
        ("source_category", pa.string()),
        ("source_tags", pa.string()),
        ("candidate_intent", pa.string()),
        ("candidate_action", pa.string()),
        ("context_json", pa.string()),
        ("allowed_use", pa.string()),
        ("reason", pa.string()),
        ("reason_detail", pa.string()),
        ("quality", QUALITY_T),
    ]
)
REJECTED_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("record_id", pa.string()),
        ("source_id", pa.string()),
        ("revision", pa.string()),
        ("example_id", pa.string()),
        ("stage", pa.string()),
        ("reason", pa.string()),
        ("detail", pa.string()),
        ("duplicate_of", pa.string()),
        ("family_id", pa.string()),
        ("allowed_use", pa.string()),
        ("text_sha256", pa.string()),
    ]
)
GROUPS_SCHEMA = pa.schema(
    [
        ("example_id", pa.string()),
        ("record_id", pa.string()),
        ("source_id", pa.string()),
        ("destination", pa.string()),
        ("exact_group", pa.string()),
        ("exact_group_size", pa.int32()),
        ("family_id", pa.string()),
        ("scenario_id", pa.string()),
        ("scenario_key", pa.string()),
        ("label_key", pa.string()),
        ("family_size", pa.int32()),
        ("scenario_size", pa.int32()),
    ]
)
PAIRS_SCHEMA = pa.schema(
    [
        ("example_a", pa.string()),
        ("example_b", pa.string()),
        ("jaccard", pa.float64()),
        ("same_source", pa.bool_()),
        ("same_label", pa.bool_()),
        ("compatible", pa.bool_()),
        ("label_a", pa.string()),
        ("label_b", pa.string()),
    ]
)

_CLEAN_COLUMNS = (
    "example_id",
    "record_id",
    "source_id",
    "revision",
    "source_split",
    "record_type",
    "lang",
    "lid_conf",
    "code_switch",
    "langs",
    "text",
    "reference_response",
    "slots",
    "source_intent",
    "source_category",
    "source_tags",
    "context_json",
    "exact_key",
    "sim_text",
    "family_hint",
    "scenario_hint",
    "flags",
    "reject_reason",
    "text_sha256",
)


class BuildError(Exception):
    """The build cannot run, or its quality gates failed (output kept for inspection)."""


class BuildResult(BaseModel):
    status: Literal["built", "skipped"]
    path: str
    dataset_version: str
    row_counts: dict[str, int]
    gates_passed: bool
    seconds: float


def canonical_dir(data_dir: Path) -> Path:
    return data_dir / "interim" / "canonical"


def build_params(cfg: CleanSettings, intent_map_path: Path) -> dict[str, object]:
    return {
        "build_version": BUILD_VERSION,
        "clean_version": CLEAN_VERSION,
        "intent_map_sha256": sha256_file(intent_map_path),
        "shingle_size": cfg.shingle_size,
        "minhash_perms": cfg.minhash_perms,
        "lsh_bands": cfg.lsh_bands,
        "family_threshold": cfg.family_threshold,
        "pair_report_min": cfg.pair_report_min,
        "seed": cfg.seed,
        "min_validity": cfg.min_validity,
        "min_lid_agreement": cfg.min_lid_agreement,
    }


# ------------------------------------------------------------------ union-find


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[x] != root:
            self.parent[x], x = root, int(self.parent[x])
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def roots(self) -> np.ndarray:
        return np.fromiter((self.find(i) for i in range(len(self.parent))), dtype=np.int64)

    def copy(self) -> UnionFind:
        uf = UnionFind(0)
        uf.parent = self.parent.copy()
        return uf


def _union_by_key(uf: UnionFind, keys: Sequence[str | None]) -> None:
    first: dict[str, int] = {}
    for i, k in enumerate(keys):
        if k is None:
            continue
        if k in first:
            uf.union(first[k], i)
        else:
            first[k] = i


def _component_ids(roots: np.ndarray, names: Sequence[str], prefix: str) -> list[str]:
    """Stable id per component: hash of the smallest member name (example_id or key)."""
    smallest: dict[int, str] = {}
    for r, name in zip(roots.tolist(), names, strict=True):
        if r not in smallest or name < smallest[r]:
            smallest[r] = name
    return [f"{prefix}_{sha256_text(f'{prefix}|{smallest[r]}')[:16]}" for r in roots.tolist()]


# ------------------------------------------------------------------ loading and labels


def _load(entries: Sequence[SourceEntry], data_dir: Path) -> tuple[pl.DataFrame, list[str]]:
    frames = []
    inputs = []
    for rank, e in enumerate(entries):
        cdir = clean_dir(data_dir, e)
        if not (cdir / MANIFEST).is_file():
            raise BuildError(f"{e.source_id}: not cleaned yet ({cdir}); run `data clean` first")
        if problems := verify_clean(cdir):
            raise BuildError(f"{e.source_id}: clean output fails verification: {problems[:3]}")
        m = read_model_json(cdir / MANIFEST, CleanManifest)
        inputs.append(f"{e.source_id}@{e.revision}#clean:{m.content_sha256[:16]}")
        frames.append(
            pl.read_parquet(cdir / CLEAN, columns=list(_CLEAN_COLUMNS)).with_columns(
                pl.lit(rank).alias("source_rank"),
                pl.lit(USE_RANK[e.allowed_use]).alias("use_rank"),
                pl.lit(e.allowed_use.value).alias("allowed_use"),
                pl.int_range(pl.len()).alias("row_in_source"),
            )
        )
    df = pl.concat(frames, how="vertical")
    return df.with_row_index("gidx"), inputs


def _labels(df: pl.DataFrame, imap: IntentMap) -> pl.DataFrame:
    status, intent, action, esc, reason, detail, lkey, xkey, skey = ([] for _ in range(9))
    for r in df.iter_rows(named=True):
        slots = dict(r["slots"] or [])
        lab = map_labels(imap, r["source_id"], r["source_intent"], r["text"], slots)
        status.append(lab.status)
        intent.append(lab.intent)
        action.append(lab.action)
        esc.append(lab.escalate)
        reason.append(lab.escalation_reason)
        detail.append(lab.detail)
        rtype = r["record_type"]
        sid = r["source_id"]
        if rtype == RecordType.POLICY_DECISION.value:
            ctx = json.loads(r["context_json"] or "{}")
            key = f"{sid}:{ctx.get('action')}|{ctx.get('recommended_action')}"
        elif rtype == RecordType.READING_COMPREHENSION.value:
            key = sid
        else:
            key = f"{sid}:{r['source_intent']}|{lab.action or '-'}"
        lkey.append(key)
        xkey.append(f"v0:{lab.intent}|{lab.action}" if lab.status == "mapped" else None)
        if r["scenario_hint"]:
            skey.append(r["scenario_hint"])
        elif rtype == RecordType.SINGLE_TURN.value and r["source_intent"] is not None:
            skey.append(key)
        else:
            skey.append(None)
    return df.with_columns(
        pl.Series("label_status", status),
        pl.Series("intent", intent, dtype=pl.String),
        pl.Series("action", action, dtype=pl.String),
        pl.Series("escalate", esc, dtype=pl.Boolean),
        pl.Series("escalation_reason", reason, dtype=pl.String),
        pl.Series("label_detail", detail, dtype=pl.String),
        pl.Series("label_key", lkey, dtype=pl.String),
        pl.Series("xlabel_key", xkey, dtype=pl.String),
        pl.Series("scenario_key", skey, dtype=pl.String),
    )


def compatible(
    src_a: str, lab_a: str, x_a: str | None, src_b: str, lab_b: str, x_b: str | None
) -> bool:
    """Same source: same source label (and action). Across sources: same mapped v0 intent+action."""
    if src_a == src_b:
        return lab_a == lab_b
    return x_a is not None and x_a == x_b


# ------------------------------------------------------------------ grouping


class _Groups(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    frame: pl.DataFrame
    pairs: pl.DataFrame


def _group(df: pl.DataFrame, cfg: CleanSettings) -> _Groups:
    active = df.filter(pl.col("reject_reason").is_null()).sort(
        "use_rank", "source_rank", "row_in_source", descending=[True, False, False]
    )
    # 1. exact groups -> one node each. Two rows are exact duplicates when their template
    #    keys match, or (single messages) when their final texts match: a filled
    #    "{{Account Type}}" can produce the same text as a literal "gold" in another row.
    text_keys = [
        exact_key(t) if rt == RecordType.SINGLE_TURN.value else None
        for t, rt in zip(active["text"].to_list(), active["record_type"].to_list(), strict=True)
    ]
    uf = UnionFind(active.height)
    _union_by_key(uf, active["exact_key"].to_list())
    _union_by_key(uf, text_keys)
    roots = uf.roots()  # root = smallest position = highest-priority member
    active = active.with_columns(pl.Series("exact_root", roots))
    grp = active.group_by("exact_root", maintain_order=True).agg(
        pl.col("gidx"),
        pl.col("source_id"),
        pl.col("label_key"),
        pl.col("xlabel_key"),
        pl.col("scenario_key"),
        pl.col("example_id"),
    )
    node_of: dict[int, int] = {}
    rep_gidx: list[int] = []
    conflict: list[bool] = []
    member_keys: list[set[str]] = []
    for n, r in enumerate(grp.iter_rows(named=True)):
        members = sorted(set(zip(r["source_id"], r["label_key"], r["xlabel_key"], strict=True)))
        ok = all(
            compatible(*members[i], *members[k])
            for i in range(len(members))
            for k in range(i + 1, len(members))
        )
        conflict.append(not ok)
        member_keys.append({k for k in r["scenario_key"] if k is not None} if ok else set())
        rep_gidx.append(r["gidx"][0])
        for g in r["gidx"]:
            node_of[g] = n
    nodes = (
        df.filter(pl.col("gidx").is_in(rep_gidx))
        .with_columns(
            pl.col("gidx")
            .replace_strict(dict(zip(rep_gidx, range(len(rep_gidx)), strict=True)))
            .alias("node")
        )
        .sort("node")
    )
    n_nodes = nodes.height
    src = nodes["source_id"].to_list()
    lab = nodes["label_key"].to_list()
    xlab = nodes["xlabel_key"].to_list()
    exid = nodes["example_id"].to_list()

    # 1b. Texts that differ only in case, punctuation or inserted profanity (equal
    #     similarity key) but carry incompatible labels are label noise: no reliable
    #     intent exists for them, so they become conflicts too. They are not removed as
    #     duplicates, because the profane variant is a distinct register sample.
    by_sim: dict[str, list[int]] = {}
    for i, key in enumerate(nodes["sim_text"].to_list()):
        by_sim.setdefault(key, []).append(i)
    for idxs in by_sim.values():
        if len(idxs) < 2:
            continue
        if any(
            not compatible(src[a], lab[a], xlab[a], src[b], lab[b], xlab[b])
            for k, a in enumerate(idxs)
            for b in idxs[k + 1 :]
        ):
            for a in idxs:
                conflict[a] = True
                member_keys[a] = set()

    # 2. near-duplicates between nodes
    nd = near_duplicates(
        nodes["sim_text"].to_list(),
        shingle_size=cfg.shingle_size,
        num_perm=cfg.minhash_perms,
        bands=cfg.lsh_bands,
        seed=cfg.seed,
        min_jaccard=cfg.pair_report_min,
    )
    fam = UnionFind(n_nodes)
    pair_rows: dict[str, list[Any]] = {k: [] for k in PAIRS_SCHEMA.names}
    for (a, b), j in zip(nd.pairs.tolist(), nd.jaccard.tolist(), strict=True):
        comp = not (conflict[a] or conflict[b]) and compatible(
            src[a], lab[a], xlab[a], src[b], lab[b], xlab[b]
        )
        if comp and j >= cfg.family_threshold:
            fam.union(a, b)
        for k, v in (
            ("example_a", exid[a]),
            ("example_b", exid[b]),
            ("jaccard", round(j, 6)),
            ("same_source", src[a] == src[b]),
            ("same_label", lab[a] == lab[b]),
            ("compatible", comp),
            ("label_a", lab[a]),
            ("label_b", lab[b]),
        ):
            pair_rows[k].append(v)
    _union_by_key(fam, nodes["family_hint"].to_list())

    # 3. scenarios: families joined by scenario key (conflict nodes keep their own)
    #    An exact-duplicate group whose members carry several scenario keys (e.g. the same
    #    text labelled retail:change_account and cs:edit_account) proves those keys equal.
    scn = fam.copy()
    skeys = [None if conflict[i] else k for i, k in enumerate(nodes["scenario_key"].to_list())]
    first_node: dict[str, int] = {}
    for i, keys in enumerate(member_keys):
        for k in sorted(keys):
            if k in first_node:
                scn.union(first_node[k], i)
            else:
                first_node[k] = i
    fam_roots, scn_roots = fam.roots(), scn.roots()
    family_ids = _component_ids(fam_roots, exid, "fam")
    scn_names = [k or family_ids[i] for i, k in enumerate(skeys)]
    scenario_ids = _component_ids(scn_roots, scn_names, "scn")

    node_cols = pl.DataFrame(
        {
            "node": np.arange(n_nodes),
            "family_id": family_ids,
            "scenario_id": scenario_ids,
            "conflict": conflict,
            "rep_example_id": exid,
        }
    )
    frame = (
        active.with_columns(pl.col("gidx").replace_strict(node_of).alias("node"))
        .join(node_cols, on="node", how="left")
        .with_columns(
            (pl.col("example_id") == pl.col("rep_example_id")).alias("is_rep"),
            pl.len().over("node").alias("exact_group_size"),
        )
    )
    frame = frame.with_columns(
        pl.col("node").n_unique().over("family_id").alias("family_size"),
        pl.col("node").n_unique().over("scenario_id").alias("scenario_size"),
    )
    pairs = (
        pl.DataFrame(pair_rows, schema=pl.Schema(PAIRS_SCHEMA))
        if pair_rows["jaccard"]
        else pl.DataFrame(schema=pl.Schema(PAIRS_SCHEMA))
    )
    log.info(
        "grouped", extra={"nodes": n_nodes, "candidates": nd.candidates, "pairs": len(nd.pairs)}
    )
    return _Groups(frame=frame, pairs=pairs)


# ------------------------------------------------------------------ destinations


def _quality(r: dict[str, Any], extra_flags: Sequence[str] = ()) -> dict[str, Any]:
    size = r.get("exact_group_size") or 1
    dup = f"dup_{sha256_text(r['rep_example_id'])[:16]}" if size > 1 else None
    return {
        "lid_conf": float(r["lid_conf"] or 0.0),
        "mt_qe": None,
        "intent_agree": None,
        "dup_cluster": dup,
        "flags": sorted({*(r["flags"] or []), *extra_flags}),
    }


def _pending_reason(r: dict[str, Any]) -> tuple[str, str | None] | None:
    if r["conflict"]:
        return "label_conflict", "identical text with incompatible labels"
    if r["allowed_use"] != AllowedUse.TRAIN.value:
        return f"allowed_use:{r['allowed_use']}", "source licence/registry forbids training"
    if r["record_type"] != RecordType.SINGLE_TURN.value:
        return f"record_type:{r['record_type']}", "canonical/1 holds single customer messages"
    if r["label_status"] != "mapped":
        reason = {
            "unmapped": "intent_unmapped",
            "no_label": "no_source_label",
            "suspect": "label_suspect",
        }.get(r["label_status"], "intent_not_in_map")
        return reason, r["label_detail"]
    return None


def _rejected(r: dict[str, Any], stage: str, reason: str, **kw: Any) -> dict[str, Any]:
    return {
        "schema_version": "rejected/1",
        "record_id": r["record_id"],
        "source_id": r["source_id"],
        "revision": r["revision"],
        "example_id": r["example_id"],
        "stage": stage,
        "reason": reason,
        "detail": kw.get("detail"),
        "duplicate_of": kw.get("duplicate_of"),
        "family_id": kw.get("family_id"),
        "allowed_use": r["allowed_use"],
        "text_sha256": r["text_sha256"],
    }


def _canonical(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "canonical/1",
        "example_id": r["example_id"],
        "record_id": r["record_id"],
        "source_id": r["source_id"],
        "scenario_id": r["scenario_id"],
        "family_id": r["family_id"],
        "lang": r["lang"],
        "code_switch": bool(r["code_switch"]),
        "text": r["text"],
        "reference_response": r["reference_response"],
        "intent": r["intent"],
        "source_intent": r["source_intent"],
        "source_category": r["source_category"],
        "action": r["action"],
        "escalate": bool(r["escalate"]),
        "escalation_reason": r["escalation_reason"],
        "slots": dict(r["slots"] or []),
        "origin": "original",
        "translation_of": None,
        "translation_system": None,
        "quality": _quality(r, [DRAFT_FLAG]),
        "split": None,
        "created_at": None,
    }


def _pending(r: dict[str, Any], reason: str, detail: str | None) -> dict[str, Any]:
    return {
        "schema_version": "pending/1",
        "example_id": r["example_id"],
        "record_id": r["record_id"],
        "source_id": r["source_id"],
        "revision": r["revision"],
        "record_type": r["record_type"],
        "source_split": r["source_split"],
        "scenario_id": r["scenario_id"],
        "family_id": r["family_id"],
        "lang": r["lang"],
        "code_switch": bool(r["code_switch"]),
        "langs": list(r["langs"] or [r["lang"]]),
        "text": r["text"],
        "reference_response": r["reference_response"],
        "slots": dict(r["slots"] or []),
        "source_intent": r["source_intent"],
        "source_category": r["source_category"],
        "source_tags": r["source_tags"],
        "candidate_intent": r["intent"],
        "candidate_action": r["action"],
        "context_json": r["context_json"],
        "allowed_use": r["allowed_use"],
        "reason": reason,
        "reason_detail": detail,
        "quality": _quality(r),
    }


class _Split(BaseModel):
    canonical: list[dict[str, Any]]
    pending: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    destination: dict[str, str]  # record_id -> canonical | pending | rejected


def _route(df: pl.DataFrame, groups: pl.DataFrame) -> _Split:
    canonical: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    dest: dict[str, str] = {}
    for r in df.filter(pl.col("reject_reason").is_not_null()).iter_rows(named=True):
        rejected.append(_rejected(r, "clean", r["reject_reason"]))
        dest[r["record_id"]] = "rejected"
    for r in groups.iter_rows(named=True):
        if not r["is_rep"] and not r["conflict"]:
            rejected.append(
                _rejected(
                    r,
                    "dedup",
                    "exact_duplicate",
                    duplicate_of=r["rep_example_id"],
                    family_id=r["family_id"],
                    detail=f"same exact_key as {r['rep_example_id']}",
                )
            )
            dest[r["record_id"]] = "rejected"
            continue
        why = _pending_reason(r)
        if why is not None:
            row = _pending(r, *why)
            try:
                PendingExample.model_validate(row)
            except ValidationError as exc:
                rejected.append(_rejected(r, "validate", "schema", detail=_first_error(exc)))
                dest[r["record_id"]] = "rejected"
                continue
            pending.append(row)
            dest[r["record_id"]] = "pending"
            continue
        row = _canonical(r)
        try:
            CanonicalExample.model_validate(row)
        except ValidationError as exc:
            rejected.append(_rejected(r, "validate", "schema", detail=_first_error(exc)))
            dest[r["record_id"]] = "rejected"
            continue
        canonical.append(row)
        dest[r["record_id"]] = "canonical"
    return _Split(canonical=canonical, pending=pending, rejected=rejected, destination=dest)


def _first_error(exc: ValidationError) -> str:
    err = exc.errors()[0]
    return f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"[:300]


# ------------------------------------------------------------------ writing


def _write(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema) if rows else schema.empty_table()
    pq.write_table(table, path, compression="zstd")


def _content_hash(*parts: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for rows in parts:
        for row in rows:
            h.update(
                (json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n").encode()
            )
    return h.hexdigest()


def _is_current(out: Path, inputs: list[str], params: dict[str, object]) -> bool:
    if not (out / MANIFEST).is_file():
        return False
    m = read_model_json(out / MANIFEST, DatasetManifest)
    return (
        m.inputs == inputs
        and m.params == json.loads(json.dumps(params))
        and not verify_files(out, m.files, exclude=frozenset({MANIFEST}))
    )


def build_canonical(
    entries: Sequence[SourceEntry],
    data_dir: Path,
    cfg: CleanSettings,
    imap: IntentMap,
    params: dict[str, object],
    *,
    code_commit: str | None,
    force: bool = False,
    run_gates: Any = None,
) -> BuildResult:
    """Build `data/interim/canonical/`. `run_gates(partial_dir) -> (passed, report)`."""
    start = time.perf_counter()
    df, inputs = _load(entries, data_dir)
    out = canonical_dir(data_dir)
    for stale in out.parent.glob(".partial-canonical-*"):  # left by a killed run
        shutil.rmtree(stale, ignore_errors=True)
    if not force and _is_current(out, inputs, params):
        m = read_model_json(out / MANIFEST, DatasetManifest)
        report = json.loads((out / QUALITY).read_text(encoding="utf-8"))
        return BuildResult(
            status="skipped",
            path=str(out),
            dataset_version=m.dataset_version,
            row_counts=m.row_counts,
            gates_passed=bool(report.get("passed")),
            seconds=round(time.perf_counter() - start, 2),
        )
    df = _labels(df, imap)
    groups = _group(df, cfg)
    split = _route(df, groups.frame)
    if len(split.destination) != df.height:
        raise BuildError(f"accounting: {len(split.destination)} routed != {df.height} rows")

    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.parent / f".partial-canonical-{uuid.uuid4().hex[:8]}"
    partial.mkdir()
    try:
        canonical = sorted(
            split.canonical, key=lambda r: (r["lang"], r["source_id"], r["record_id"])
        )
        by_lang: dict[str, list[dict[str, Any]]] = {}
        for row in canonical:
            by_lang.setdefault(row["lang"], []).append(row)
        for lang, rows in sorted(by_lang.items()):
            _write(
                partial / EXAMPLES / f"lang={lang}" / "part-00000.parquet", rows, CANONICAL_SCHEMA
            )
        pending = sorted(split.pending, key=lambda r: (r["source_id"], r["record_id"]))
        rejected = sorted(split.rejected, key=lambda r: (r["source_id"], r["record_id"]))
        _write(partial / PENDING, pending, PENDING_SCHEMA)
        _write(partial / REJECTED, rejected, REJECTED_SCHEMA)
        g = (
            groups.frame.with_columns(
                pl.col("record_id").replace_strict(split.destination).alias("destination"),
                pl.col("rep_example_id")
                .map_elements(lambda k: f"dup_{sha256_text(k)[:16]}", return_dtype=pl.String)
                .alias("exact_group"),
            )
            .select(GROUPS_SCHEMA.names)
            .sort("source_id", "record_id")
        )
        pq.write_table(g.to_arrow().cast(GROUPS_SCHEMA), partial / GROUPS, compression="zstd")
        pairs = groups.pairs.sort("example_a", "example_b")
        pq.write_table(pairs.to_arrow().cast(PAIRS_SCHEMA), partial / PAIRS, compression="zstd")

        row_counts: Counter[str] = Counter()
        for row in canonical:
            row_counts[f"canonical|lang={row['lang']}"] += 1
            row_counts[f"canonical|source={row['source_id']}"] += 1
        for row in pending:
            row_counts[f"pending|reason={row['reason']}"] += 1
            row_counts[f"pending|source={row['source_id']}"] += 1
        for row in rejected:
            row_counts[f"rejected|reason={row['reason']}"] += 1
            row_counts[f"rejected|source={row['source_id']}"] += 1
        row_counts["canonical"] = len(canonical)
        row_counts["pending"] = len(pending)
        row_counts["rejected"] = len(rejected)
        content = _content_hash(canonical, pending, rejected)
        passed, report = run_gates(partial) if run_gates else (True, {"passed": True, "gates": []})
        (partial / QUALITY).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest = DatasetManifest(
            dataset_version=make_dataset_version("canonical", datetime.now(UTC).date(), content),
            data_schema_version="canonical/1",
            code_commit=code_commit or "unknown",
            inputs=inputs,
            params=params,
            row_counts=dict(sorted(row_counts.items())),
            files=scan_files(partial),
            created_at=datetime.now(UTC),
            quality_report=QUALITY,
        )
        write_model_json(partial / MANIFEST, manifest)
        if not passed:
            failed = out.parent / "canonical.failed"
            shutil.rmtree(failed, ignore_errors=True)
            partial.rename(failed)
            raise BuildError(f"quality gates failed; build kept at {failed} (see {QUALITY})")
        old = out.parent / f".old-canonical-{uuid.uuid4().hex[:8]}"
        if out.exists():
            out.rename(old)
        partial.rename(out)
        shutil.rmtree(old, ignore_errors=True)
        shutil.rmtree(out.parent / "canonical.failed", ignore_errors=True)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    log.info("canonical built", extra={"dataset_version": manifest.dataset_version})
    return BuildResult(
        status="built",
        path=str(out),
        dataset_version=manifest.dataset_version,
        row_counts=manifest.row_counts,
        gates_passed=passed,
        seconds=round(time.perf_counter() - start, 2),
    )
