"""Human-authored training corpus (ADR-042, DATA_CONTRACTS §20).

A versioned corpus of customer requests whose text was written by people, built from pinned
raw sources and the frozen v1 taxonomy. It replaces the synthetic/templated/machine-translated
Stage 4 SFT rows as the active training corpus; the historical SFT datasets stay untouched.

Admission (`configs/human_corpus.yaml`):
- the registry allows training (usage policy) AND the recipe documents human authorship of the
  customer text; a source without documented authorship goes to `unverified.parquet` only;
- no machine-translated, LLM-generated, post-edited or templated rows: ABCD turns that echo the
  scenario prompt (second-person prompt voice) or repeat verbatim across many dialogues are
  routed to `unverified` (`prompt_echo`, `template_cluster`);
- labels are `source_label_mapped`: the source's own label through an authored table split
  by regex cues; no row is reported as human-adjudicated;
- no target answers: there is no human-written answer grounded in this store's KB/tools, so
  `sft_target_status` is `no_human_target` (no synthetic target is substituted).

Steps: ABCD opening request per dialogue (greeting-only turns become context) -> PII masking
with the dialogue's own fictional values -> cleaning/language/length checks -> label mapping
-> template detection -> exact dedup -> near-copy families (char-5-gram Jaccard) + MASSIVE
EN/RU alignment -> family label-conflict check -> historical overlap removal (exact, lexical,
13-gram, e5 cosine vs every earlier SFT/gold/benchmark user text) -> family-disjoint
stratified split -> cross-split semantic decontamination -> immutable versioned output.
"""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from supportpilot.common.hashing import sha256_text
from supportpilot.common.ids import make_example_id
from supportpilot.common.schemas import Action, SourceEntry
from supportpilot.data.acquire import raw_dir, verify_raw
from supportpilot.data.build import PENDING, canonical_dir
from supportpilot.data.labels import has_reference
from supportpilot.data.neardup import near_duplicates
from supportpilot.data.policy import Purpose, check_use
from supportpilot.data.split import exact_leaks, lexical_leaks, ngram_leaks
from supportpilot.data.steps import StepError, content_hash, counts, write_parquet
from supportpilot.data.taxonomy import Taxonomy, relabel, resolve_action
from supportpilot.data.textnorm import (
    CARD_RE,
    EMAIL_RE,
    PHONE_RE,
    clean_text,
    garbled,
    sim_key,
    word_count,
)

HUMAN_VERSION = 2
SCHEMA = "human_example/1"
# `human_spoken_asr`: a person spoke the request; the text is an automatic transcription of
# that speech (human wording, machine transcription, ASR errors kept). Not a translation.
ACTIVE_ORIGINS = frozenset(
    {"human_roleplay", "human_written", "human_localised", "human_spoken_asr"}
)
FORBIDDEN_ORIGINS = frozenset({"machine_translated", "llm_generated", "post_edited", "original"})
LABEL_STATUS = "source_label_mapped"
REVIEW_STATUS = "unreviewed"
TARGET_STATUS = "no_human_target"
SPLITS = ("train", "validation", "test")
HELDOUT = ("validation", "test")
FILES = (*(f"{s}.parquet" for s in SPLITS), "unverified.parquet", "excluded.parquet")
Encoder = Callable[[Sequence[str]], npt.NDArray[np.float32]]


# ----------------------------------------------------------------------------- recipe


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HumanRule(_Model):
    pattern: str
    intent: str | None = None
    action: str | None = None
    escalation_reason: str | None = None  # required when `action: escalate` overrides an intent
    exclude: str | None = None

    @model_validator(mode="after")
    def _one(self) -> Self:
        if (self.intent is None) == (self.exclude is None):
            raise ValueError("a rule sets exactly one of intent / exclude")
        if self.action is not None and self.intent is None:
            raise ValueError("an action override needs an intent")
        re.compile(self.pattern)
        return self


class HumanLabel(_Model):
    intent: str | None = None
    action: str | None = None
    rules: list[HumanRule] = Field(default_factory=list)
    exclude: str | None = None
    cap: int | None = Field(default=None, ge=1)  # native sources: rows per (label, language)

    @model_validator(mode="after")
    def _one(self) -> Self:
        if sum((self.intent is not None, bool(self.rules), self.exclude is not None)) != 1:
            raise ValueError("a source label has exactly one of intent / rules / exclude")
        return self


class Authorship(_Model):
    origin: dict[str, str] = Field(min_length=1)
    active: bool
    evidence: str = Field(min_length=20)
    # `heldout`: rows may enter validation/test only (the publisher intends evaluation use)
    placement: Literal["all", "heldout"] = "all"
    max_per_label: int | None = Field(default=None, ge=1)  # overrides native_max_per_label

    @model_validator(mode="after")
    def _origins(self) -> Self:
        bad = set(self.origin.values()) & FORBIDDEN_ORIGINS
        if self.active and (bad or not set(self.origin.values()) <= ACTIVE_ORIGINS):
            raise ValueError(f"an active source needs documented human origins, got {self.origin}")
        return self


class GroundedRecipe(_Model):
    """Human-written, document-grounded responses (ADR-043): external evidence namespace."""

    source_id: str
    namespace: str = Field(min_length=3)
    origin: Literal["human_written"]
    evidence: str = Field(min_length=20)
    acts: dict[str, Literal["answer", "clarify"]] = Field(min_length=1)
    min_evidence_recall: float = Field(gt=0.0, le=1.0)


class HumanRecipe(_Model):
    schema_version: Literal["human_recipe/2"]
    seed: int
    split_ratios: dict[str, float]
    family_jaccard: float = Field(gt=0.0, le=1.0)
    template_cluster_min: int = Field(ge=2)
    leak_jaccard: float = Field(gt=0.0, le=1.0)
    leak_ngram: int = Field(ge=5)
    leak_cosine: float = Field(gt=0.0, le=1.0)
    min_words: int = Field(ge=1)
    native_max_per_label: int = Field(ge=1)
    authorship: dict[str, Authorship]
    prompt_echo_patterns: list[str]
    personal_name_patterns: list[str] = Field(default_factory=list)
    greeting_only: str
    sources: dict[str, dict[str, HumanLabel]]
    grounded: GroundedRecipe | None = None

    @model_validator(mode="after")
    def _ok(self) -> Self:
        if set(self.split_ratios) != set(SPLITS) or abs(sum(self.split_ratios.values()) - 1) > 1e-9:
            raise ValueError("split_ratios must give train/validation/test summing to 1")
        for p in (*self.prompt_echo_patterns, *self.personal_name_patterns, self.greeting_only):
            re.compile(p)
        return self

    def cap(self, source_id: str, label: str) -> int:
        lab = self.sources.get(source_id, {}).get(label)
        if lab is not None and lab.cap is not None:
            return lab.cap
        a = self.authorship.get(source_id)
        if a is not None and a.max_per_label is not None:
            return a.max_per_label
        return self.native_max_per_label

    def heldout_only(self, source_id: str) -> bool:
        a = self.authorship.get(source_id)
        return a is not None and a.placement == "heldout"

    def check_taxonomy(self, tax: Taxonomy) -> None:
        allowed = {*tax.actions, "needs_reference"}
        for sid, labels in self.sources.items():
            for name, lab in labels.items():
                targets = [(lab.intent, lab.action)] + [(r.intent, r.action) for r in lab.rules]
                for intent, action in targets:
                    if intent is not None and intent not in tax.intents:
                        raise ValueError(f"{sid}:{name}: unknown intent {intent!r}")
                    if action is not None and action not in allowed:
                        raise ValueError(f"{sid}:{name}: unknown action {action!r}")
                for r in lab.rules:
                    if r.escalation_reason is not None and (
                        r.escalation_reason not in tax.escalation_reasons
                    ):
                        raise ValueError(f"{sid}:{name}: unknown reason {r.escalation_reason!r}")
                    if (
                        r.action == Action.ESCALATE.value
                        and r.intent is not None
                        and tax.escalation_reason(r.intent) is None
                        and r.escalation_reason is None
                    ):
                        raise ValueError(f"{sid}:{name}: escalate override needs a reason")


def load_recipe(path: Path, tax: Taxonomy) -> HumanRecipe:
    recipe = HumanRecipe.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    recipe.check_taxonomy(tax)
    return recipe


# ----------------------------------------------------------------------------- rows


@dataclass
class Row:
    record_id: str
    source_id: str
    lang: str
    origin: str
    text: str
    source_label: str
    source_split: str
    context: list[dict[str, str]] = field(default_factory=list)
    slots: dict[str, str] = field(default_factory=dict)
    align: str | None = None  # cross-language alignment key (MASSIVE parallel id)
    flags: list[str] = field(default_factory=list)
    intent: str | None = None
    action: str | None = None
    escalation_reason: str | None = None
    rule: str | None = None
    family_id: str = ""
    scenario_id: str = ""
    split: str = ""
    lid_conf: float | None = None
    code_switch: bool = False
    langs: list[str] = field(default_factory=list)

    @property
    def example_id(self) -> str:
        return make_example_id(self.source_id, self.record_id, self.lang, self.origin, self.text)


EXAMPLE_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("example_id", pa.string()),
        ("record_id", pa.string()),
        ("source_id", pa.string()),
        ("source_split", pa.string()),
        ("origin", pa.string()),
        ("authorship_evidence", pa.string()),
        ("label_status", pa.string()),
        ("review_status", pa.string()),
        ("sft_target_status", pa.string()),
        ("lang", pa.string()),
        ("code_switch", pa.bool_()),
        ("lid_conf", pa.float64()),
        ("langs", pa.list_(pa.string())),
        ("text", pa.string()),
        ("context_json", pa.string()),
        ("slots_json", pa.string()),
        ("source_label", pa.string()),
        ("rule", pa.string()),
        ("intent", pa.string()),
        ("action", pa.string()),
        ("escalate", pa.bool_()),
        ("escalation_reason", pa.string()),
        ("scenario_id", pa.string()),
        ("family_id", pa.string()),
        ("split", pa.string()),
        ("flags", pa.list_(pa.string())),
        ("reason", pa.string()),
    ]
)
EXCLUDED_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("source_id", pa.string()),
        ("source_label", pa.string()),
        ("lang", pa.string()),
        ("text", pa.string()),
        ("reason", pa.string()),
        ("detail", pa.string()),
    ]
)


# ----------------------------------------------------------------------------- ABCD

_SUFFIX_RE = re.compile(r"_\d+$")
_ORDER_ID_RE = re.compile(r"(?<!\w)\d{10}(?!\w)")
_ACCOUNT_ID_RE = re.compile(r"(?<!\w)(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{10}(?!\w)")
# North-American numbers with separators, e.g. "(977) 625-2661"; textnorm.PHONE_RE misses them.
NANP_RE = re.compile(r"(?<![\w+])(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?!\w)")


def abcd_label(subflow: str) -> str:
    return _SUFFIX_RE.sub("", subflow)


def opening_request(
    turns: Sequence[Sequence[str]], greeting: re.Pattern[str]
) -> tuple[list[dict[str, str]], list[str]]:
    """(context, request turns): consecutive customer turns up to the agent's first reply to
    something that is not a greeting. Greeting-only groups and agent turns become context."""
    context: list[dict[str, str]] = []
    group: list[str] = []
    for speaker, text in turns:
        if speaker == "customer":
            group.append(text)
            continue
        if group:
            if any(not greeting.match(t.strip()) for t in group):
                return context, group
            context.extend({"role": "user", "content": t} for t in group)
            group = []
        if speaker == "agent":
            context.append({"role": "assistant", "content": text})
    if any(not greeting.match(t.strip()) for t in group):
        return context, group
    return context, []


def _scenario_values(scenario: Mapping[str, Any]) -> list[tuple[str, str]]:
    """(value, token) pairs of the dialogue's fictional customer data, longest first."""
    personal = scenario.get("personal", {}) or {}
    order = scenario.get("order", {}) or {}
    pairs: list[tuple[str, str]] = []

    def add(value: Any, token: str) -> None:
        if isinstance(value, str) and len(value.strip()) >= 3:
            pairs.append((value.strip(), token))

    add(personal.get("customer_name"), "<name>")
    for part in str(personal.get("customer_name") or "").split():
        add(part, "<name>")
    add(personal.get("email"), "<email>")
    add(personal.get("phone"), "<phone>")
    add(personal.get("username"), "<username>")
    add(personal.get("account_id"), "<account_id>")
    add(order.get("order_id"), "<order_id>")
    add(order.get("full_address"), "<address>")
    add(order.get("street_address"), "<address>")
    add(order.get("zip_code"), "<zip>")
    return sorted(set(pairs), key=lambda p: -len(p[0]))


def mask_text(text: str, values: Sequence[tuple[str, str]]) -> str:
    """Replace the dialogue's own fictional values, then generic e-mail/phone/id patterns."""
    out = text
    for value, token in values:
        out = re.sub(rf"(?<!\w){re.escape(value)}(?!\w)", token, out, flags=re.IGNORECASE)
    out = EMAIL_RE.sub("<email>", out)
    out = PHONE_RE.sub("<phone>", out)
    out = NANP_RE.sub("<phone>", out)
    out = _ORDER_ID_RE.sub("<order_id>", out)
    return _ACCOUNT_ID_RE.sub("<account_id>", out)


def read_abcd(
    entry: SourceEntry, data_dir: Path, recipe: HumanRecipe
) -> tuple[list[Row], list[dict[str, Any]]]:
    root = raw_dir(data_dir, entry)
    if problems := verify_raw(root):
        raise StepError(f"{entry.source_id}: raw revision fails verification: {problems[0]}")
    data = json.loads(gzip.decompress((root / "data" / "abcd_v1.1.json.gz").read_bytes()))
    greeting = re.compile(recipe.greeting_only, re.IGNORECASE)
    origin = recipe.authorship[entry.source_id].origin["en"]
    rows: list[Row] = []
    excluded: list[dict[str, Any]] = []
    for split in ("train", "dev", "test"):
        for convo in data[split]:
            rid = f"{entry.source_id}:convo-{convo['convo_id']}"
            label = abcd_label(convo["scenario"]["subflow"])
            values = _scenario_values(convo["scenario"])
            # mask first, so a turn that only gives the customer's name reads as no request
            turns = [(sp, clean_text(mask_text(t, values)).text) for sp, t in convo["original"]]
            ctx, request = opening_request(turns, greeting)
            if not request:
                excluded.append(_excl(rid, entry.source_id, label, "en", "", "no_request", None))
                continue
            text = "\n".join(request)
            slots = {"order_id": "<order_id>"} if "<order_id>" in text else {}
            rows.append(Row(rid, entry.source_id, "en", origin, text, label, split, ctx, slots))
    return rows, excluded


# ----------------------------------------------------------------------------- native sources


def read_native(entries: Sequence[SourceEntry], data_dir: Path, recipe: HumanRecipe) -> list[Row]:
    """MASSIVE / mangi rows from the Stage 3 build (already cleaned, LID'd, deduplicated)."""
    ids = [e.source_id for e in entries]
    if not ids:
        return []
    can = canonical_dir(data_dir)
    if not (can / "manifest.json").is_file():
        raise StepError(f"no Stage 3 build at {can}; run `supportpilot data clean --all`")
    df = (
        pl.scan_parquet(can / PENDING)
        .filter(
            pl.col("source_id").is_in(ids)
            & pl.col("reason").is_in(["intent_unmapped", "intent_not_in_map"])
        )
        .select("record_id", "source_id", "lang", "text", "source_intent", "source_split")
        .collect()
        .sort("source_id", "record_id")
    )
    langs = {e.source_id: set(e.languages) for e in entries}
    rows = []
    for r in df.iter_rows(named=True):
        sid = r["source_id"]
        origin = recipe.authorship[sid].origin.get(r["lang"])
        if r["lang"] not in langs[sid] or origin is None:
            continue
        key = r["record_id"].split(":", 1)[1]
        align = None
        if sid == "massive_en_ru":  # en-US.test.17 and ru-RU.test.17 are one utterance
            align = f"{sid}:" + key.split(".", 1)[1]
        rows.append(
            Row(
                r["record_id"],
                sid,
                r["lang"],
                origin,
                r["text"],
                r["source_intent"] or "",
                r["source_split"] or "",
                align=align,
            )
        )
    return rows


# ----------------------------------------------------------------------------- raw native sources

# Intent-labelled sources read straight from their pinned raw revision (ADR-043); they are not
# Stage 3 sources, so the sealed canonical build stays byte-identical.


def _raw_root(entry: SourceEntry, data_dir: Path) -> Path:
    root = raw_dir(data_dir, entry)
    if problems := verify_raw(root):
        raise StepError(f"{entry.source_id}: raw revision fails verification: {problems[0]}")
    return root


def _class_names(path: Path, column: str) -> list[str]:
    """ClassLabel names that the Hub stores in the parquet schema metadata."""
    meta = pq.read_schema(path).metadata or {}
    info = json.loads(meta[b"huggingface"])["info"]["features"][column]
    return list(info["names"])


def read_minds14(entry: SourceEntry, data_dir: Path, recipe: HumanRecipe) -> list[Row]:
    root = _raw_root(entry, data_dir)
    origin = recipe.authorship[entry.source_id].origin["ru"]
    rows: list[Row] = []
    for path in sorted(root.glob("ru-RU/*.parquet")):
        names = _class_names(path, "intent_class")
        table = pq.read_table(path, columns=["path", "transcription", "intent_class"])
        for r in table.to_pylist():
            key = Path(str(r["path"]).replace("~", "/")).stem
            text = clean_text(str(r["transcription"] or "")).text
            label = names[int(r["intent_class"])]
            rows.append(
                Row(f"{entry.source_id}:{key}", entry.source_id, "ru", origin, text, label, "train")
            )
    return rows


def read_xsid(entry: SourceEntry, data_dir: Path, recipe: HumanRecipe) -> list[Row]:
    """Only the human-translated dev/test files; `projectedTrain` (MT) is never fetched."""
    root = _raw_root(entry, data_dir)
    origin = recipe.authorship[entry.source_id].origin["kk"]
    rows: list[Row] = []
    for split in ("valid", "test"):
        for path in sorted(root.glob(f"data/*/kk.{split}.conll")):
            block: dict[str, str] = {}
            for line in [*path.read_text(encoding="utf-8").splitlines(), ""]:
                if line.startswith("# ") and " = " in line:
                    k, v = line[2:].split(" = ", 1)
                    block[k.strip()] = v
                elif not line.strip() and block:
                    text = clean_text(block.get("text", "")).text
                    rid = f"{entry.source_id}:{split}.{block.get('id', '')}"
                    rows.append(
                        Row(
                            rid, entry.source_id, "kk", origin, text, block.get("intent", ""), split
                        )
                    )
                    block = {}
    return rows


def read_clinc(entry: SourceEntry, data_dir: Path, recipe: HumanRecipe) -> list[Row]:
    root = _raw_root(entry, data_dir)
    origin = recipe.authorship[entry.source_id].origin["en"]
    rows: list[Row] = []
    for path in sorted(root.glob("plus/*.parquet")):
        split = path.name.split("-", 1)[0]
        names = _class_names(path, "intent")
        for i, r in enumerate(pq.read_table(path, columns=["text", "intent"]).to_pylist()):
            text = clean_text(str(r["text"] or "")).text
            rid = f"{entry.source_id}:{split}.{i}"
            rows.append(
                Row(rid, entry.source_id, "en", origin, text, names[int(r["intent"])], split)
            )
    return rows


RAW_READERS: dict[str, Callable[[SourceEntry, Path, HumanRecipe], list[Row]]] = {
    "minds14_ru": read_minds14,
    "xsid_kk": read_xsid,
    "clinc150": read_clinc,
}
STAGE3_NATIVE = frozenset({"massive_en_ru", "mangi_kk_support_qa"})


# ----------------------------------------------------------------------------- labels


@dataclass(frozen=True)
class Decision:
    status: Literal["mapped", "excluded", "unlabelled", "not_in_map"]
    intent: str | None = None
    action: str | None = None
    rule: str | None = None
    detail: str | None = None
    escalation_reason: str | None = None


def map_label(recipe: HumanRecipe, tax: Taxonomy, row: Row) -> Decision:
    if row.source_id not in recipe.sources:
        lab = relabel(tax, row.source_id, row.source_label, row.text, row.slots)
        return Decision(
            lab.status, lab.intent, lab.action, lab.rule, lab.detail, lab.escalation_reason
        )
    m = recipe.sources[row.source_id].get(row.source_label)
    if m is None:
        return Decision("not_in_map", detail=f"{row.source_id}:{row.source_label} not in recipe")
    if m.exclude is not None:
        return Decision("excluded", detail=m.exclude)
    intent, action, rule, why = m.intent, m.action, None, None
    if m.rules:
        for r in m.rules:
            if re.search(r.pattern, row.text, re.IGNORECASE):
                if r.exclude is not None:
                    return Decision("excluded", rule=r.pattern, detail=r.exclude)
                intent, action, rule, why = r.intent, r.action, r.pattern, r.escalation_reason
                break
        else:
            return Decision("unlabelled", detail="no cue of the source label in the request")
    assert intent is not None
    resolved = resolve_action(tax, intent, action, has_reference(row.text, row.slots))
    why = (why or tax.escalation_reason(intent)) if resolved == Action.ESCALATE.value else None
    return Decision("mapped", intent, resolved, rule, escalation_reason=why)


# ----------------------------------------------------------------------------- grouping


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def families(rows: Sequence[Row], jaccard: float, seed: int) -> list[int]:
    """Component index per row: near-copies (char-5-gram Jaccard) and aligned translations."""
    uf = _UnionFind(len(rows))
    if len(rows) > 1:
        res = near_duplicates(
            [sim_key(r.text) for r in rows],
            shingle_size=5,
            num_perm=128,
            bands=32,
            seed=seed,
            min_jaccard=jaccard,
        )
        for a, b in res.pairs.tolist():
            uf.union(int(a), int(b))
    first: dict[str, int] = {}
    for i, r in enumerate(rows):
        if r.align is not None:
            if r.align in first:
                uf.union(first[r.align], i)
            else:
                first[r.align] = i
    return [uf.find(i) for i in range(len(rows))]


def assign_splits(
    groups: Mapping[str, str],
    ratios: Mapping[str, float],
    seed: int,
    heldout: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """family id -> split, stratified by intent in sha256 order (no RNG state).
    An intent with >= 3 families gets at least one validation and one test family.
    `heldout` families (evaluation-only sources) go to validation/test only, in the
    validation:test ratio, and do not count towards the train/held-out stratification."""
    by_intent: dict[str, list[str]] = defaultdict(list)
    held_by_intent: dict[str, list[str]] = defaultdict(list)
    for fam, intent in groups.items():
        (held_by_intent if fam in heldout else by_intent)[intent].append(fam)
    out: dict[str, str] = {}
    for _intent, fams in sorted(by_intent.items()):
        fams.sort(key=lambda f: sha256_text(f"{seed}|split|{f}"))
        n = len(fams)
        n_test, n_val = round(n * ratios["test"]), round(n * ratios["validation"])
        if n >= 3:
            n_test, n_val = max(1, n_test), max(1, n_val)
        for i, fam in enumerate(fams):
            out[fam] = "test" if i < n_test else "validation" if i < n_test + n_val else "train"
    share_test = ratios["test"] / (ratios["test"] + ratios["validation"])
    for _intent, fams in sorted(held_by_intent.items()):
        fams.sort(key=lambda f: sha256_text(f"{seed}|heldout|{f}"))
        n_test = round(len(fams) * share_test)
        for i, fam in enumerate(fams):
            out[fam] = "test" if i < n_test else "validation"
    return out


# ----------------------------------------------------------------------------- history

_USER_KEYS = frozenset({"user_text", "query", "question", "text"})
_ALL_STRING_KEYS = frozenset({"texts", "needs", "followups"})


def _strings(obj: Any, out: list[str], *, take_all: bool = False) -> None:
    if isinstance(obj, str):
        if take_all:
            out.append(obj)
    elif isinstance(obj, list):
        for v in obj:
            _strings(v, out, take_all=take_all)
    elif isinstance(obj, dict):
        if "role" in obj and "content" in obj:
            if obj["role"] == "user" and isinstance(obj["content"], str):
                out.append(obj["content"])
            return
        for k, v in obj.items():
            if k in _USER_KEYS and isinstance(v, str):
                out.append(v)
            elif k == "turns" and isinstance(v, list):
                _strings(v, out, take_all=all(isinstance(t, str) for t in v))
            else:
                _strings(v, out, take_all=take_all or k in _ALL_STRING_KEYS)


def historical_sources(data_dir: Path) -> list[Path]:
    """Every earlier SFT split, gold benchmark and frozen benchmark/confirmation input."""
    paths = sorted((data_dir / "processed" / "sft").glob("sft-*/*.parquet"))
    paths += sorted((data_dir / "gold").glob("*/gold.jsonl"))
    bench = data_dir / "benchmarks"
    for pattern in ("*/**/*.json", "*/**/*.jsonl", "*/*.json", "*/*.jsonl"):
        paths += sorted(bench.glob(pattern))
    skip = ("manifest", "quality_report", "results", "metadata", "audit", "protocol")
    return sorted({p for p in paths if not any(s in p.name for s in skip)})


def historical_texts(paths: Iterable[Path]) -> list[str]:
    texts: list[str] = []
    for p in paths:
        if p.suffix == ".parquet":
            df = pl.read_parquet(p, columns=["user_text", "messages_json"])
            texts.extend(df["user_text"].to_list())
            for mj in df["messages_json"].to_list():
                _strings(json.loads(mj), texts)
        elif p.suffix == ".jsonl":
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    _strings(json.loads(line), texts)
        else:
            _strings(json.loads(p.read_text(encoding="utf-8")), texts)
    keyed = {sim_key(t): t for t in texts if isinstance(t, str) and len(sim_key(t)) >= 3}
    return [keyed[k] for k in sorted(keyed)]


# ----------------------------------------------------------------------------- build


def _excl(
    rid: str, sid: str, label: str, lang: str, text: str, reason: str, detail: str | None
) -> dict[str, Any]:
    return {
        "record_id": rid,
        "source_id": sid,
        "source_label": label,
        "lang": lang,
        "text": text,
        "reason": reason,
        "detail": detail,
    }


@dataclass
class BuildInputs:
    entries: Sequence[SourceEntry]
    data_dir: Path
    recipe: HumanRecipe
    tax: Taxonomy
    lid: Callable[[Sequence[str]], list[tuple[str, float, bool, tuple[str, ...]]]]
    encode: Encoder | None
    history_paths: Sequence[Path]


@dataclass
class BuildResult:
    active: list[Row]
    unverified: list[tuple[Row, str]]
    excluded: list[dict[str, Any]]
    report: dict[str, Any]
    history: list[str] = field(default_factory=list)


def build_rows(inp: BuildInputs) -> BuildResult:
    recipe, tax = inp.recipe, inp.tax
    by_id = {e.source_id: e for e in inp.entries}
    for sid in recipe.authorship:
        if sid not in by_id:
            raise StepError(f"recipe source {sid} is not in the registry")
        check_use(by_id[sid], Purpose.TRAINING)
    rows: list[Row] = []
    excluded: list[dict[str, Any]] = []
    if "abcd" in recipe.authorship:
        abcd_rows, abcd_excluded = read_abcd(by_id["abcd"], inp.data_dir, recipe)
        rows += abcd_rows
        excluded += abcd_excluded
    rows += read_native(
        [by_id[s] for s in recipe.authorship if s in STAGE3_NATIVE], inp.data_dir, recipe
    )
    for sid in recipe.authorship:
        if sid in RAW_READERS:
            rows += RAW_READERS[sid](by_id[sid], inp.data_dir, recipe)
    unknown = set(recipe.authorship) - {"abcd", *STAGE3_NATIVE, *RAW_READERS}
    if unknown:
        raise StepError(f"no reader for recipe sources {sorted(unknown)}")
    read_counts = Counter(r.source_id for r in rows) + Counter(x["source_id"] for x in excluded)

    unverified: list[tuple[Row, str]] = []
    echo = [re.compile(p, re.IGNORECASE) for p in recipe.prompt_echo_patterns]
    names = [re.compile(p, re.IGNORECASE) for p in recipe.personal_name_patterns]

    # 1. text quality and label mapping
    lid = inp.lid([r.text for r in rows])
    kept: list[Row] = []
    for r, (lang, conf, cs, langs) in zip(rows, lid, strict=True):
        # rounded: the detector can differ in the last float ulp between runs
        r.lid_conf, r.code_switch, r.langs = round(conf, 6), cs, sorted(langs)
        d = map_label(recipe, tax, r)
        if d.status != "mapped":
            excluded.append(
                _excl(r.record_id, r.source_id, r.source_label, r.lang, r.text, d.status, d.detail)
            )
            continue
        r.intent, r.action, r.rule = d.intent, d.action, d.rule
        r.escalation_reason = d.escalation_reason
        reason = None
        if not recipe.authorship[r.source_id].active:
            reason = "authorship_undocumented"
        elif "�" in r.text or garbled(r.text):
            reason = "corrupted_text"
        elif word_count(r.text) < recipe.min_words:
            reason = "too_short"
        elif any(p.search(r.text) for p in echo):
            reason = "prompt_echo"
        elif any(p.search(r.text) for p in names):
            reason = "possible_personal_name"
        elif lang != r.lang and not cs:
            reason = "language_mismatch"
        if reason is not None:
            unverified.append((r, reason))
        else:
            kept.append(r)

    # 2. template clusters and exact duplicates (per source, by comparison key)
    dialogues: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for r in kept:
        dialogues[(r.source_id, sim_key(r.text))].append(r)
    deduped: list[Row] = []
    for (sid, _key), members in sorted(dialogues.items()):
        members.sort(key=lambda r: sha256_text(f"{recipe.seed}|dedup|{r.record_id}"))
        if sid == "abcd" and len(members) >= recipe.template_cluster_min:
            unverified.extend((r, "template_cluster") for r in members)
            continue
        deduped.append(members[0])
        for r in members[1:]:
            excluded.append(
                _excl(
                    r.record_id,
                    sid,
                    r.source_label,
                    r.lang,
                    r.text,
                    "exact_duplicate",
                    members[0].record_id,
                )
            )

    # 3. native sources: one row per near-copy family, then a cap per (label, language).
    #    Not every source label is paraphrase-poor; the cap stops MASSIVE out-of-scope rows
    #    from dominating the corpus (rows are recorded as excluded, never deleted).
    comp = families(deduped, recipe.family_jaccard, recipe.seed)
    cap_groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for i, r in sorted(
        enumerate(deduped), key=lambda x: sha256_text(f"{recipe.seed}|native|{x[1].record_id}")
    ):
        if r.source_id != "abcd":
            cap_groups[(r.source_id, r.source_label, r.lang)].append(i)
    native_keep: set[int] = set()
    for _cap, idx in sorted(cap_groups.items()):
        seen: set[int] = set()
        for i in idx:
            if len(seen) < recipe.cap(_cap[0], _cap[1]) and comp[i] not in seen:
                native_keep.add(i)
                seen.add(comp[i])
    final: list[Row] = []
    for i, r in enumerate(deduped):
        if r.source_id == "abcd" or i in native_keep:
            final.append(r)
        else:
            excluded.append(
                _excl(
                    r.record_id,
                    r.source_id,
                    r.source_label,
                    r.lang,
                    r.text,
                    "label_cap",
                    f"one per family, at most {recipe.cap(r.source_id, r.source_label)}"
                    " per label/lang",
                )
            )
    comp = families(final, recipe.family_jaccard, recipe.seed)  # aligned EN/RU stay together

    # 4. family ids and label conflicts inside a family
    members_of: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(comp):
        members_of[c].append(i)
    active: list[Row] = []
    for _c, idx in sorted(members_of.items()):
        rs = [final[i] for i in idx]
        fam = "fam_" + sha256_text("|".join(sorted(r.record_id for r in rs)))[:16]
        labels = {(r.intent, r.action) for r in rs}
        for r in rs:
            r.family_id = fam
            r.scenario_id = "scn_" + sha256_text(f"human|{fam}")[:16]
        if len(labels) > 1:
            unverified.extend((r, "family_label_conflict") for r in rs)
        else:
            active.extend(rs)

    # 5. overlap with every historical SFT / gold / benchmark user text
    history = historical_texts(inp.history_paths)
    pool = active + [r for r, _ in unverified]
    texts = [r.text for r in pool]
    hits: dict[int, str] = {}
    match: dict[int, str] = {}
    for i in exact_leaks(history, texts):
        hits.setdefault(i, "historical_exact")
    for i in lexical_leaks(history, texts, threshold=recipe.leak_jaccard, seed=recipe.seed):
        hits.setdefault(i, "historical_lexical")
    for i in ngram_leaks(history, texts, recipe.leak_ngram):
        hits.setdefault(i, "historical_ngram")
    sem_q: dict[str, float] = {}
    history_vec = None
    if inp.encode is not None and history and texts:
        history_vec = inp.encode(history)
        pool_vec = inp.encode(texts)
        from supportpilot.data.embed import max_cosine

        best, arg = max_cosine(pool_vec, history_vec)
        for i, v in enumerate(best.tolist()):
            if v >= recipe.leak_cosine:
                hits.setdefault(i, "historical_semantic")
                match[i] = f"cosine={v:.4f}; nearest={history[int(arg[i])]}"
        sem_q = {q: round(float(np.quantile(best, float(q))), 4) for q in ("0.5", "0.9", "0.99")}
    leaked = {id(pool[i]): reason for i, reason in hits.items()}
    detail = {id(pool[i]): m for i, m in match.items()}
    for r in pool:
        if id(r) in leaked:
            excluded.append(
                _excl(
                    r.record_id,
                    r.source_id,
                    r.source_label,
                    r.lang,
                    r.text,
                    leaked[id(r)],
                    detail.get(id(r), "overlaps an earlier SFT/gold/benchmark user text"),
                )
            )
    active = [r for r in active if id(r) not in leaked]
    unverified = [(r, why) for r, why in unverified if id(r) not in leaked]

    # 6. family-disjoint stratified split, then cross-split semantic decontamination
    fam_intent = {r.family_id: r.intent or "" for r in active}
    heldout = frozenset(r.family_id for r in active if recipe.heldout_only(r.source_id))
    split_of = assign_splits(fam_intent, recipe.split_ratios, recipe.seed, heldout)
    for r in active:
        r.split = split_of[r.family_id]
    cross = 0
    if inp.encode is not None:
        train = [r for r in active if r.split == "train"]
        held = [r for r in active if r.split != "train"]
        if train and held:
            from supportpilot.data.embed import max_cosine

            best, arg = max_cosine(
                inp.encode([r.text for r in train]), inp.encode([r.text for r in held])
            )
            near = {
                id(train[i]): f"cosine={v:.4f}; held_out={held[int(arg[i])].record_id}"
                for i, v in enumerate(best.tolist())
                if v >= recipe.leak_cosine
            }
            drop = set(near)
            for r in train:
                if id(r) in drop:
                    excluded.append(
                        _excl(
                            r.record_id,
                            r.source_id,
                            r.source_label,
                            r.lang,
                            r.text,
                            "cross_split_semantic",
                            near[id(r)],
                        )
                    )
            cross = len(drop)
            active = [r for r in active if id(r) not in drop]

    report = {
        "read": dict(sorted(read_counts.items())),
        "historical_texts": len(history),
        "historical_files": len(inp.history_paths),
        "historical_overlap_removed": dict(sorted(Counter(hits.values()).items())),
        "semantic_checked": inp.encode is not None,
        "semantic_max_cosine_quantiles": sem_q,
        "cross_split_semantic_removed": cross,
    }
    return BuildResult(active, unverified, excluded, report, history)


def to_record(r: Row, recipe: HumanRecipe, reason: str | None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "example_id": r.example_id,
        "record_id": r.record_id,
        "source_id": r.source_id,
        "source_split": r.source_split,
        "origin": r.origin,
        "authorship_evidence": recipe.authorship[r.source_id].evidence,
        "label_status": LABEL_STATUS,
        "review_status": REVIEW_STATUS,
        "sft_target_status": TARGET_STATUS,
        "lang": r.lang,
        "code_switch": r.code_switch,
        "lid_conf": r.lid_conf,
        "langs": r.langs,
        "text": r.text,
        "context_json": json.dumps(r.context, ensure_ascii=False),
        "slots_json": json.dumps(r.slots, ensure_ascii=False, sort_keys=True),
        "source_label": r.source_label,
        "rule": r.rule,
        "intent": r.intent,
        "action": r.action,
        "escalate": r.action == Action.ESCALATE.value,
        "escalation_reason": r.escalation_reason,
        "scenario_id": r.scenario_id,
        "family_id": r.family_id,
        "split": r.split if reason is None else "unverified",
        "flags": sorted(set(r.flags)),
        "reason": reason,
    }


def pii_hits(text: str) -> int:
    patterns = (EMAIL_RE, PHONE_RE, NANP_RE, CARD_RE)
    return sum(len(p.findall(text)) for p in patterns)


def write_outputs(partial: Path, result: BuildResult, recipe: HumanRecipe) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    for r in sorted(result.active, key=lambda r: r.record_id):
        by_split[r.split].append(to_record(r, recipe, None))
    for s, rows in by_split.items():
        write_parquet(partial / f"{s}.parquet", rows, EXAMPLE_SCHEMA)
    unv = [
        to_record(r, recipe, why)
        for r, why in sorted(result.unverified, key=lambda x: x[0].record_id)
    ]
    write_parquet(partial / "unverified.parquet", unv, EXAMPLE_SCHEMA)
    exc = sorted(result.excluded, key=lambda x: (x["source_id"], x["record_id"], x["reason"]))
    write_parquet(partial / "excluded.parquet", exc, EXCLUDED_SCHEMA)
    all_rows = [row for rows in by_split.values() for row in rows]
    row_counts = {
        **counts(all_rows, "split", "lang", "intent"),
        **counts(unv, "reason", prefix="unverified|"),
        **counts(exc, "reason", prefix="excluded|"),
    }
    return {
        "row_counts": row_counts,
        "content_sha256": content_hash([*all_rows, *unv, *exc]),
        "by_split": by_split,
        "unverified": unv,
        "excluded": exc,
    }
