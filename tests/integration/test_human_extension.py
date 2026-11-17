"""Human corpus extension (ADR-043): raw native sources, held-out placement, grounded component."""

from __future__ import annotations

import io
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from supportpilot.common.config import find_project_root
from supportpilot.common.hashing import sha256_text
from supportpilot.data.human import BuildInputs, HumanRecipe, historical_sources, load_recipe
from supportpilot.data.humangates import build_human, verify_human
from supportpilot.data.humangrounded import evidence_recall
from supportpilot.data.taxonomy import load_taxonomy
from tests.data_helpers import entry, make_raw

ROOT = find_project_root(Path(__file__).parent)
MINDS_NAMES = ["address", "app_error", "balance", "card_issues"]
CLINC_NAMES = ["greeting", "weather", "order", "card_declined"]

MINDS = [
    ("address", "Я переехала и хочу изменить свой домашний адрес"),
    ("address", "Мне нужно поменять адрес по которому зарегистрирована моя карточка"),
    ("address", "как изменить адрес в личном кабинете на сайте"),
    ("app_error", "приложение зависает и я не могу нормально пользоваться"),
    ("app_error", "приложение не загружает баланс моего счёта"),
    ("balance", "скажите пожалуйста какой у меня остаток на счету"),
    ("balance", "здравствуйте меня зовут Светлана хочу узнать баланс на счету"),
    ("card_issues", "моя карта не работает в магазине"),
]
XSID = [
    ("weather/find", "Бүгін жаңбыр жауады ма екен?"),
    ("alarm/set_alarm", "Ертең таңғы жетіге оятқыш қойыңызшы"),
    ("PlayMusic", "Маған қазақтың халық әнін қосып беріңізші"),
    ("reminder/set_reminder", "Кешкі сағат алтыда дүкенге баруды еске салыңыз"),
]
CLINC = [
    ("greeting", "hello there, how is your day going"),
    ("greeting", "good morning to you my friend"),
    ("weather", "what is the forecast for tomorrow in denver"),
    ("weather", "will it snow this weekend up north"),
    ("order", "order me more paper towels please"),
    ("card_declined", "my card got declined when i tried to pay"),
    ("card_declined", "i was at walmart and my card got declined"),
]


def _hf_parquet(rows: list[dict[str, Any]], column: str, names: list[str]) -> bytes:
    meta = {"info": {"features": {column: {"names": names, "_type": "ClassLabel"}}}}
    table = pa.Table.from_pylist(rows).replace_schema_metadata(
        {b"huggingface": json.dumps(meta).encode()}
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def _xsid() -> str:
    out = []
    for i, (label, text) in enumerate(XSID, 1):
        out.append(
            f"# id = {i}\n# text-en = x\n# text = {text}\n# intent = {label}\n1\tx\t{label}\tO\n"
        )
    return "\n".join(out) + "\n"


DOC_TEXT = (
    "You must report a change of address to the DMV within 10 days. "
    "If you moved, update the address on your license and on every vehicle registration."
)


def _doc2dial() -> dict[str, bytes | str]:
    s1 = "You must report a change of address to the DMV within 10 days."
    s2 = "If you moved,"
    s3 = "update the address on your license and on every vehicle registration."
    spans = {}
    for sp_id, sp in (("1", s1), ("2", s2), ("3", s3)):
        start = DOC_TEXT.index(sp)
        spans[sp_id] = {
            "sp_id": sp_id,
            "start_sp": start,
            "end_sp": start + len(sp),
            "text_sp": sp,
            "title": "Change of address",
        }
    docs = {}
    dials: dict[str, Any] = {"train": {}, "validation": {}}
    for n in range(4):
        doc_id = f"Address change {n}#1_0"
        docs[doc_id] = {
            "doc_id": doc_id,
            "title": "Change of address",
            "doc_text": DOC_TEXT,
            "spans": spans,
        }
        turns = [
            {
                "turn_id": 1,
                "role": "user",
                "da": "query_solution",
                "references": [{"sp_id": "1", "label": "solution"}],
                "utterance": f"I moved last month to town {n}, what do I need to do?",
            },
            {
                "turn_id": 2,
                "role": "agent",
                "da": "query_condition",
                "references": [{"sp_id": "2", "label": "precondition"}],
                "utterance": "So you moved, right?",
            },
            {
                "turn_id": 3,
                "role": "user",
                "da": "response_positive",
                "references": [{"sp_id": "2", "label": "precondition"}],
                "utterance": "Yes I have",
            },
            {
                "turn_id": 4,
                "role": "agent",
                "da": "respond_solution",
                "references": [
                    {"sp_id": "1", "label": "solution"},
                    {"sp_id": "3", "label": "solution"},
                ],
                "utterance": (
                    "Report the change of address to the DMV within 10 days"
                    " and update your license."
                ),
            },
            {
                "turn_id": 5,
                "role": "user",
                "da": "query_solution",
                "references": [{"sp_id": "3", "label": "solution"}],
                "utterance": "Anything else?",
            },
            {
                "turn_id": 6,
                "role": "agent",
                "da": "respond_solution",
                "references": [{"sp_id": "3", "label": "solution"}],
                "utterance": "Wonderful weather today, enjoy your sunny afternoon.",
            },
        ]
        split = "train" if n < 3 else "validation"
        dials[split].setdefault("dmv", {})[doc_id] = [
            {"dial_id": f"d{n}", "doc_id": doc_id, "domain": "dmv", "turns": turns}
        ]
    base = "data/doc2dial/v1.0.1/"
    return {
        base + "doc2dial_doc.json": json.dumps({"doc_data": {"dmv": docs}}),
        base + "doc2dial_dial_train.json": json.dumps({"dial_data": dials["train"]}),
        base + "doc2dial_dial_validation.json": json.dumps({"dial_data": dials["validation"]}),
    }


def _encode(texts: Sequence[str]) -> npt.NDArray[np.float32]:
    out = np.zeros((len(texts), 512), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"\w+", t.lower()):
            out[i, int(sha256_text(w)[:6], 16) % 512] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.maximum(norms, 1e-9)).astype(np.float32)


def _lid(texts: Sequence[str]) -> list[tuple[str, float, bool, tuple[str, ...]]]:
    out: list[tuple[str, float, bool, tuple[str, ...]]] = []
    for t in texts:
        lang = (
            "kk"
            if re.search("[әғқңөұүһі]", t.lower())
            else "ru"
            if re.search("[а-я]", t.lower())
            else "en"
        )
        out.append((lang, 1.0, False, (lang,)))
    return out


@pytest.fixture
def setup(tmp_path: Path) -> BuildInputs:
    tax = load_taxonomy(ROOT / "configs" / "taxonomy.yaml")
    full = load_recipe(ROOT / "configs" / "human_corpus.yaml", tax)
    data = full.model_dump()
    keep = ("minds14_ru", "xsid_kk", "clinc150")
    data["authorship"] = {k: data["authorship"][k] for k in keep}
    data["sources"] = {k: data["sources"][k] for k in keep}
    data["split_ratios"] = {"train": 0.5, "validation": 0.25, "test": 0.25}
    recipe = HumanRecipe.model_validate(data)
    minds = entry("minds14_ru", languages=["ru"], pipelines=["human"])
    xsid = entry(
        "xsid_kk",
        licence="cc-by-sa-4.0",
        languages=["kk"],
        pipelines=["human"],
        fetch={"type": "github_repo", "repo_id": "org/xsid", "include": ["data/*"]},
    )
    clinc = entry("clinc150", licence="cc-by-3.0", pipelines=["human"])
    d2d = entry(
        "doc2dial",
        licence="cc-by-3.0",
        pipelines=["human"],
        fetch={"type": "github_repo", "repo_id": "org/d2d", "include": ["data/*"]},
    )
    make_raw(
        tmp_path,
        minds,
        {
            "ru-RU/train-00000-of-00001.parquet": _hf_parquet(
                [
                    {
                        "path": f"ru-RU~X/{i:04d}.wav",
                        "transcription": t,
                        "intent_class": MINDS_NAMES.index(lab),
                    }
                    for i, (lab, t) in enumerate(MINDS)
                ],
                "intent_class",
                MINDS_NAMES,
            )
        },
    )
    make_raw(
        tmp_path, xsid, {"data/xSID-0.7/kk.valid.conll": _xsid(), "data/xSID-0.7/kk.test.conll": ""}
    )
    make_raw(
        tmp_path,
        clinc,
        {
            "plus/train-00000-of-00001.parquet": _hf_parquet(
                [{"text": t, "intent": CLINC_NAMES.index(lab)} for lab, t in CLINC],
                "intent",
                CLINC_NAMES,
            )
        },
    )
    make_raw(tmp_path, d2d, _doc2dial())
    return BuildInputs(
        entries=[minds, xsid, clinc, d2d],
        data_dir=tmp_path,
        recipe=recipe,
        tax=tax,
        lid=_lid,
        encode=_encode,
        history_paths=historical_sources(tmp_path),
    )


def test_extension_sources_placement_and_grounded_component(setup: BuildInputs) -> None:
    res = build_human(
        setup, code_commit="abc", recipe_sha256="r", taxonomy_sha256="t", previous=None
    )
    assert res["status"] == "built", res
    assert res["gates"] == "pass"
    out = Path(res["path"])
    active = pl.concat(
        [pl.read_parquet(out / f"{s}.parquet") for s in ("train", "validation", "test")]
    )

    kk = active.filter(pl.col("lang") == "kk")
    assert kk.height == len(XSID)
    assert set(kk["split"]) <= {"validation", "test"}  # evaluation-only source
    assert set(kk["intent"]) == {"out_of_scope"}

    ru = active.filter(pl.col("source_id") == "minds14_ru")
    assert set(ru["origin"]) == {"human_spoken_asr"}
    by_text = dict(zip(ru["text"], ru["intent"], strict=True))
    assert by_text["Я переехала и хочу изменить свой домашний адрес"] == "change_address"
    assert by_text["как изменить адрес в личном кабинете на сайте"] == "account_help"
    assert by_text["приложение зависает и я не могу нормально пользоваться"] == "technical_issue"
    assert not ru["text"].str.contains("карточка|счёта|Светлана").any()

    unv = pl.read_parquet(out / "unverified.parquet")
    assert "possible_personal_name" in set(unv["reason"])
    exc = pl.read_parquet(out / "excluded.parquet")
    assert "i was at walmart and my card got declined" in set(exc["text"])
    assert "order me more paper towels please" in set(exc["text"])
    en = dict(zip(active["text"], active["intent"], strict=True))
    assert en["my card got declined when i tried to pay"] == "payment_problem"

    g = pl.concat(
        [pl.read_parquet(out / f"grounded_{s}.parquet") for s in ("train", "validation", "test")]
    )
    assert set(g["behaviour"]) == {"answer", "clarify"}
    assert set(g["namespace"]) == {"doc2dial_external"}
    assert g.group_by("family_id").agg(pl.col("split").n_unique().alias("n"))["n"].max() == 1
    for ev in g["evidence_json"]:
        assert json.loads(ev)
        assert all(e["text"] in DOC_TEXT for e in json.loads(ev))
    gu = pl.read_parquet(out / "grounded_unverified.parquet")
    assert set(gu["reason"]) == {"weak_evidence_overlap"}  # the off-topic "weather" answers
    assert gu.height == 4

    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert stats["grounded"]["active"] == g.height
    assert stats["lang"]["kk"] == len(XSID)
    report = json.loads((out / "quality_report.json").read_text(encoding="utf-8"))
    names = {x["name"] for x in report["gates"]}
    assert {"heldout_placement", "grounded_evidence", "grounded_split_isolation"} <= names
    assert verify_human(out)["checksums_ok"]


def test_evidence_recall_is_a_lexical_share() -> None:
    assert evidence_recall("Report the address change", ["report a change of address"]) == 1.0
    assert evidence_recall("sunny weather today", ["report a change of address"]) == 0.0
    assert evidence_recall("", ["x"]) == 0.0
