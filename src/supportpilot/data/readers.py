"""Read verified raw revisions into uniform items. Raw files are never modified.

Each reader yields `RawItem`s in a deterministic order. Text is returned as written by the
source (normalisation and masking happen in `corpus.py`). Readers know file layouts only;
labelling decisions live in the recipe.
"""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from supportpilot.common.schemas import LanguageCode
from supportpilot.data.text import pii_values


@dataclass(frozen=True)
class RawItem:
    source_item_id: str
    lang: LanguageCode
    text: str
    source_label: str
    source_split: str | None
    group_key: str  # split unit within the source (dialogue, parallel id, row)
    parallel_id: str | None = None
    mask_values: dict[str, list[str]] = field(default_factory=dict)


def _hf_class_names(path: Path, column: str) -> list[str]:
    meta = pq.read_schema(path).metadata or {}
    info = json.loads(meta[b"huggingface"])["info"]["features"][column]
    names = info["names"]
    if not isinstance(names, list):
        raise ValueError(f"{path}: no class names for {column}")
    return [str(n) for n in names]


# ---------------------------------------------------------------- ABCD

_GREETING_ONLY = re.compile(
    r"^\W*(?:hi+|hello+|hey+|hiya|yo|good (?:morning|afternoon|evening|day)|greetings|"
    r"thanks?|thank you|ok(?:ay)?|yes|yeah|sure|fine|great|i'?m (?:good|fine|well)|"
    r"(?:hi|hello|hey) there|how are you(?: doing)?(?: today)?|am i connected|"
    r"is (?:any|some)one there|i (?:need|have) (?:a question|some help|help)|"
    r"can you help(?: me)?|i need help with something|help)"
    r"(?:\W+(?:hi+|hello+|hey+|there|thanks?|thank you|how are you(?: doing)?(?: today)?))*"
    r"\W*$",
    re.IGNORECASE,
)


def is_greeting_only(text: str) -> bool:
    return _GREETING_ONLY.match(text.strip()) is not None


def abcd_opening(turns: list[list[str]], max_customer_turns: int = 3) -> str | None:
    """Customer text that states the request: consecutive customer turns from the first
    substantive one until the agent answers. Greeting-only turns before it are dropped."""
    collected: list[str] = []
    for speaker, text in turns:
        if speaker == "customer":
            if not collected and is_greeting_only(text):
                continue
            collected.append(text.strip())
            if len(collected) >= max_customer_turns:
                break
        elif collected:
            break
    return " ".join(collected) if collected else None


def _abcd_mask_values(scenario: dict[str, Any]) -> dict[str, list[str]]:
    personal = scenario.get("personal", {})
    order = scenario.get("order", {})
    name = str(personal.get("customer_name", ""))
    names = [name, *name.split()] if name else []
    return {
        "name": [n for n in names if len(n) >= 3],
        "email": [str(personal.get("email", ""))],
        "username": [str(personal.get("username", ""))],
        "phone": [str(personal.get("phone", ""))],
        "order_id": [str(order.get("order_id", ""))],
        "address": [str(order.get("street_address", "")), str(order.get("full_address", ""))],
        "zip": [str(order.get("zip_code", ""))],
    }


def read_abcd(raw: Path) -> Iterator[RawItem]:
    with gzip.open(raw / "data" / "abcd_v1.1.json.gz", "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    for split in sorted(data):
        for convo in sorted(data[split], key=lambda c: int(c["convo_id"])):
            opening = abcd_opening(convo["original"])
            if opening is None:
                continue
            scenario = convo["scenario"]
            subflow = re.sub(r"_\d+$", "", str(scenario["subflow"]))
            cid = str(convo["convo_id"])
            yield RawItem(
                source_item_id=f"convo-{cid}",
                lang="en",
                text=opening,
                source_label=subflow,
                source_split=split,
                group_key=f"convo-{cid}",
                mask_values=_abcd_mask_values(scenario),
            )


# ---------------------------------------------------------------- MASSIVE

_MASSIVE_LOCALES: dict[str, LanguageCode] = {"en-US": "en", "ru-RU": "ru"}


def read_massive(raw: Path) -> Iterator[RawItem]:
    for locale, lang in _MASSIVE_LOCALES.items():
        for split in ("train", "validation", "test"):
            path = raw / locale / split / "0000.parquet"
            names = _hf_class_names(path, "intent")
            rows = pq.read_table(path, columns=["id", "intent", "utt"]).to_pylist()
            for row in sorted(rows, key=lambda r: int(r["id"])):
                rid = str(row["id"])
                yield RawItem(
                    source_item_id=f"{locale}-{rid}",
                    lang=lang,
                    text=str(row["utt"]),
                    source_label=names[int(row["intent"])],
                    source_split=split,
                    group_key=f"id-{rid}",
                    parallel_id=f"massive-{rid}",
                )


# ---------------------------------------------------------------- CLINC150


def read_clinc(raw: Path) -> Iterator[RawItem]:
    for split in ("train", "validation", "test"):
        path = raw / "plus" / f"{split}-00000-of-00001.parquet"
        names = _hf_class_names(path, "intent")
        rows = pq.read_table(path, columns=["text", "intent"]).to_pylist()
        for i, row in enumerate(rows):
            yield RawItem(
                source_item_id=f"{split}-{i}",
                lang="en",
                text=str(row["text"]),
                source_label=names[int(row["intent"])],
                source_split=split,
                group_key=f"{split}-{i}",
            )


# ---------------------------------------------------------------- MINDS-14


def read_minds14(raw: Path) -> Iterator[RawItem]:
    path = raw / "ru-RU" / "train-00000-of-00001.parquet"
    names = _hf_class_names(path, "intent_class")
    rows = pq.read_table(path, columns=["path", "transcription", "intent_class"]).to_pylist()
    for row in sorted(rows, key=lambda r: str(r["path"])):
        item = str(row["path"]).rsplit("/", 1)[-1].removesuffix(".wav")
        yield RawItem(
            source_item_id=item,
            lang="ru",
            text=str(row["transcription"]),
            source_label=names[int(row["intent_class"])],
            source_split="train",
            group_key=item,
        )


# ---------------------------------------------------------------- xSID


def read_xsid(raw: Path) -> Iterator[RawItem]:
    for split in ("valid", "test"):
        path = raw / "data" / "xSID-0.7" / f"kk.{split}.conll"
        meta: dict[str, str] = {}
        for line in [*path.read_text(encoding="utf-8").splitlines(), ""]:
            if line.startswith("# ") and " = " in line:
                key, value = line[2:].split(" = ", 1)
                meta[key.strip()] = value.strip()
            elif not line.strip() and meta:
                rid = meta["id"]
                yield RawItem(
                    source_item_id=f"{split}-{rid}",
                    lang="kk",
                    text=meta["text"],
                    source_label=meta["intent"],
                    source_split=split,
                    group_key=f"{split}-{rid}",
                    parallel_id=f"xsid-{split}-{rid}",
                )
                meta = {}


# ---------------------------------------------------------------- Mäñgi KK QA


def read_mangi(raw: Path) -> Iterator[RawItem]:
    for path in sorted(raw.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            yield RawItem(
                source_item_id=str(row["id"]),
                lang="kk",
                text=str(row["question"]),
                source_label=f"{row['domain']}/{row['subdomain']}",
                source_split=None,
                group_key=str(row["id"]),
            )


INTENT_READERS: dict[str, Callable[[Path], Iterator[RawItem]]] = {
    "abcd": read_abcd,
    "massive_en_ru": read_massive,
    "clinc150": read_clinc,
    "minds14_ru": read_minds14,
    "xsid_kk": read_xsid,
    "mangi_kk_support_qa": read_mangi,
}


# ---------------------------------------------------------------- doc2dial (grounded)


@dataclass(frozen=True)
class GroundedItem:
    dialogue_id: str
    turn_id: int
    domain: str
    doc_id: str
    context: list[tuple[str, str]]
    response: str
    dialogue_act: str
    evidence: list[tuple[str, str]]
    missing_spans: int
    doc_public: frozenset[str]
    source_split: str


def doc_family(doc_id: str) -> str:
    """Variants of one page share the title before `#` (e.g. `Title#3_0` and `Title#3_0_1`)."""
    return doc_id.split("#", 1)[0].strip().lower()


def read_doc2dial(raw: Path) -> Iterator[GroundedItem]:
    base = raw / "data" / "doc2dial" / "v1.0.1"
    docs = json.loads((base / "doc2dial_doc.json").read_text(encoding="utf-8"))["doc_data"]
    for split in ("train", "validation"):
        dials = json.loads((base / f"doc2dial_dial_{split}.json").read_text(encoding="utf-8"))[
            "dial_data"
        ]
        for domain in sorted(dials):
            for doc_id in sorted(dials[domain]):
                spans = docs[domain][doc_id]["spans"]
                public = pii_values(str(docs[domain][doc_id]["doc_text"]))
                for dial in sorted(dials[domain][doc_id], key=lambda d: str(d["dial_id"])):
                    history: list[tuple[str, str]] = []
                    for turn in dial["turns"]:
                        role, text = str(turn["role"]), str(turn["utterance"])
                        act = str(turn["da"])
                        if role == "agent" and act.startswith("respond") and history:
                            ids = sorted({str(r["sp_id"]) for r in turn["references"]}, key=int)
                            ev = [(i, str(spans[i]["text_sp"])) for i in ids if i in spans]
                            yield GroundedItem(
                                dialogue_id=str(dial["dial_id"]),
                                turn_id=int(turn["turn_id"]),
                                domain=domain,
                                doc_id=doc_id,
                                context=list(history),
                                response=text,
                                dialogue_act=act,
                                evidence=ev,
                                missing_spans=len(ids) - len(ev),
                                doc_public=public,
                                source_split=split,
                            )
                        history.append((role, text))
