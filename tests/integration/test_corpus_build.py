"""End-to-end corpus build on tiny raw fixtures in the real source formats (no network)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from supportpilot.cli import main
from supportpilot.common.manifest import read_model_json
from supportpilot.data.corpus import CorpusError, build_corpus, load_inputs, verify_corpus
from supportpilot.data.policy import PolicyError
from supportpilot.data.records import CorpusManifest, GroundedExample, IntentExample, Split
from tests.data_helpers import entry, make_raw

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- fixtures


def _convo(cid: int, subflow: str, *customer: str, name: str = "Crystal Minh") -> dict[str, Any]:
    turns = [["agent", "Hi! How can I help?"], *[["customer", t] for t in customer]]
    turns += [["agent", "Sure, your full name?"], ["customer", name]]
    return {
        "convo_id": cid,
        "scenario": {
            "personal": {"customer_name": name, "email": "cm@email.com", "phone": "(977) 625-2661"},
            "order": {
                "order_id": "3348917502",
                "street_address": "6821 1st ave",
                "zip_code": "75227",
            },
            "flow": "x",
            "subflow": subflow,
        },
        "original": turns,
        "delexed": [],
    }


ABCD = [
    _convo(1, "return_size", "hello", "I need to return these jeans, they are the wrong size"),
    _convo(2, "return_size", "I need to return these jeans, they are the wrong size!"),  # exact dup
    _convo(3, "promo_code_invalid", "my promo code says invalid when I pay"),
    _convo(4, "promo_code_invalid", "hi I have a question about something"),  # no cue
    _convo(5, "recover_password", "Hi, my name is Crystal Minh and I forgot my password"),
    _convo(6, "status_service_added", "a subscription was added to my account"),  # excluded
    _convo(7, "missing", "I never received my package, order 3348917502"),
    _convo(8, "boots_how", "are the Guess boots waterproof or not"),
]
for i in range(9, 39):
    ABCD.append(_convo(i, "slow_speed", f"the website is so slow today number {i} please fix"))

MASSIVE_INTENTS = ["weather_query", "general_greet", "lists_query"]
MASSIVE = {
    "en-US": [
        (1, 0, "what is the weather like in almaty tomorrow"),
        (2, 1, "hello"),
        (3, 2, "what is on my shopping list"),
        (4, 0, "olly will it rain"),
    ],
    "ru-RU": [
        (1, 0, "какая погода будет завтра в алматы"),
        (2, 1, "привет"),
        (3, 2, "что в моем списке покупок"),
        (4, 0, "олли будет ли дождь"),
    ],
}

XSID = [("valid", 1, "Бүгін жаңбыр жауады ма?"), ("test", 2, "Ертең Астанада қар бола ма")]

DOC_TEXT = "Call us at 1-518-473-6586. You must report a change of address within 10 days."


def _write_parquet(path: Path, rows: list[tuple[int, int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "id": [str(r[0]) for r in rows],
            "intent": [r[1] for r in rows],
            "utt": [r[2] for r in rows],
        }
    )
    meta = {"info": {"features": {"intent": {"names": MASSIVE_INTENTS}}}}
    table = table.replace_schema_metadata({b"huggingface": json.dumps(meta).encode()})
    pq.write_table(table, path)


def _turn(tid: int, role: str, act: str, spans: list[str], text: str) -> dict[str, Any]:
    refs = [{"sp_id": sp} for sp in spans]
    return {"turn_id": tid, "role": role, "da": act, "references": refs, "utterance": text}


def _doc2dial_files() -> dict[str, bytes | str]:
    doc = {
        "doc_data": {
            "dmv": {
                f"Change address#{v}": {
                    "doc_text": DOC_TEXT,
                    "spans": {
                        "1": {"text_sp": "Call us at 1-518-473-6586."},
                        "2": {"text_sp": "You must report a change of address within 10 days."},
                    },
                }
                for v in ("1_0", "1_0_1")
            }
        }
    }

    def dial(did: str, user: str) -> dict[str, Any]:
        return {
            "dial_id": did,
            "turns": [
                _turn(1, "user", "query_condition", ["2"], user),
                _turn(
                    2,
                    "agent",
                    "respond_solution",
                    ["2", "1"],
                    "Report it within 10 days or call 1-518-473-6586.",
                ),
            ],
        }

    train = {"dial_data": {"dmv": {"Change address#1_0": [dial("d1", "I moved, what now?")]}}}
    val = {"dial_data": {"dmv": {"Change address#1_0_1": [dial("d2", "How fast must I report?")]}}}
    base = "data/doc2dial/v1.0.1/"
    return {
        base + "doc2dial_doc.json": json.dumps(doc),
        base + "doc2dial_dial_train.json": json.dumps(train),
        base + "doc2dial_dial_validation.json": json.dumps(val),
    }


RECIPE: dict[str, Any] = {
    "schema_version": "corpus_recipe/1",
    "seed": 11,
    "split": {"train": 0.8, "validation": 0.1, "test": 0.1},
    "min_words": 3,
    "review_sample_per_language": 10,
    "sources": {
        "abcd": {
            "domain": "retail_support",
            "origin": {"en": "human_roleplay"},
            "evidence": "test",
            "named_product": r"\bguess\b",
            "labels": {
                "return_size": {"rules": [{"cue": r"\breturn", "intent": "return_product"}]},
                "promo_code_invalid": {"rules": [{"cue": r"\bpromo", "intent": "policy_faq"}]},
                "recover_password": {"rules": [{"cue": r"passw", "intent": "account_help"}]},
                "status_service_added": {"exclude": "no v1 intent"},
                "missing": {"rules": [{"cue": r"never received", "intent": "delivery_problem"}]},
                "boots_how": {"rules": [{"cue": r"\bboots\b", "intent": "product_info"}]},
                "slow_speed": {"rules": [{"cue": r"\bslow\b", "intent": "technical_issue"}]},
            },
        },
        "massive_en_ru": {
            "domain": "virtual_assistant",
            "origin": {"en": "human_written", "ru": "human_localised"},
            "evidence": "test",
            "veto": [{"pattern": r"\bolly\b|\bолли\b", "reason": "wake word"}],
            "labels": {
                "weather_query": {"intent": "out_of_scope"},
                "general_greet": {"intent": "greeting", "min_words": 1},
                "lists_query": {"exclude": "shopping-like"},
            },
        },
        "xsid_kk": {
            "domain": "virtual_assistant",
            "origin": {"kk": "human_localised"},
            "placement": "heldout",
            "min_words": 2,
            "evidence": "test",
            "labels": {"weather/find": {"intent": "out_of_scope"}},
        },
        "doc2dial": {
            "kind": "grounded",
            "domain": "government_services",
            "origin": {"en": "human_written"},
            "evidence": "test",
        },
    },
}


@pytest.fixture
def project(tmp_path: Path) -> dict[str, Path]:
    data = tmp_path / "data"
    gh = {"type": "github_repo", "repo_id": "org/x", "include": ["*"]}
    entries = {
        "abcd": entry("abcd", fetch=gh),
        "massive_en_ru": entry("massive_en_ru", languages=["en", "ru"]),
        "xsid_kk": entry("xsid_kk", languages=["kk"], licence="cc-by-sa-4.0", fetch=gh),
        "doc2dial": entry("doc2dial", licence="cc-by-3.0", fetch=gh),
    }
    abcd_gz = gzip.compress(
        json.dumps({"train": ABCD, "dev": [], "test": []}).encode("utf-8"), mtime=0
    )
    make_raw(data, entries["abcd"], {"data/abcd_v1.1.json.gz": abcd_gz})
    xsid = {
        f"data/xSID-0.7/kk.{split}.conll": "".join(
            f"# id = {i}\n# text = {t}\n# intent = weather/find\n1\tx\tO\tO\n\n"
            for s, i, t in XSID
            if s == split
        )
        for split in ("valid", "test")
    }
    make_raw(data, entries["xsid_kk"], dict(xsid))
    make_raw(data, entries["doc2dial"], _doc2dial_files())
    massive_raw = make_raw(data, entries["massive_en_ru"], {"README.md": "x"})
    for locale, rows in MASSIVE.items():
        for split in ("train", "validation", "test"):
            part = rows if split == "train" else []
            _write_parquet(massive_raw / locale / split / "0000.parquet", part)
    (massive_raw / "manifest.json").unlink()
    make_raw(data, entries["massive_en_ru"], {})  # re-scan: manifest now covers the parquet files

    reg = {
        "schema_version": "sources/1",
        "sources": [e.model_dump(mode="json") for e in entries.values()],
    }
    (tmp_path / "sources.yaml").write_text(yaml.safe_dump(reg, allow_unicode=True), "utf-8")
    (tmp_path / "corpus.yaml").write_text(yaml.safe_dump(RECIPE, allow_unicode=True), "utf-8")
    return {
        "root": tmp_path,
        "data": data,
        "recipe": tmp_path / "corpus.yaml",
        "registry": tmp_path / "sources.yaml",
        "taxonomy": ROOT / "configs" / "taxonomy.yaml",
    }


def _inputs(p: dict[str, Path]):  # type: ignore[no-untyped-def]
    return load_inputs(p["recipe"], p["taxonomy"], p["registry"])


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


# ---------------------------------------------------------------- tests


def test_build_is_reproducible_and_leak_free(project: dict[str, Path]) -> None:
    res = build_corpus(project["data"], project["data"] / "corpus", _inputs(project), "abc")
    assert res.created
    assert all(v == 0 for v in res.leakage.values())
    again = build_corpus(project["data"], project["data"] / "corpus", _inputs(project), "abc")
    assert (again.created, again.version) == (False, res.version)
    other = build_corpus(project["data"], project["root"] / "elsewhere", _inputs(project), "x")
    m1 = read_model_json(res.path / "manifest.json", CorpusManifest)
    m2 = read_model_json(other.path / "manifest.json", CorpusManifest)
    assert m1.content_sha256 == m2.content_sha256
    assert {f.path: f.sha256 for f in m1.files} == {f.path: f.sha256 for f in m2.files}
    assert verify_corpus(res.path) == []


def test_damaged_version_with_same_content_is_not_overwritten(project: dict[str, Path]) -> None:
    out = project["data"] / "corpus"
    res = build_corpus(project["data"], out, _inputs(project), "abc")
    test_file = res.path / "intents" / "test.jsonl"
    test_file.write_bytes(test_file.read_bytes().replace(b'"', b"'", 1))
    with pytest.raises(CorpusError, match="fails verification"):
        build_corpus(project["data"], out, _inputs(project), "abc")
    assert [p.name for p in out.iterdir() if p.name.startswith(".partial")] == []


def test_contents_follow_the_contracts(project: dict[str, Path]) -> None:
    res = build_corpus(project["data"], project["data"] / "corpus", _inputs(project), None)
    core = [
        IntentExample.model_validate(r)
        for s in Split
        for r in _read(res.path / "intents" / f"{s.value}.jsonl")
    ]
    by_src = {e.source_item_id: e for e in core}
    # opening request: greeting skipped, scenario name masked, one exact duplicate removed
    assert by_src["convo-1"].text.startswith("I need to return")
    assert "convo-2" not in by_src
    assert "Crystal" not in by_src["convo-5"].text
    assert "<name>" in by_src["convo-5"].text
    # actions from taxonomy conditions
    assert by_src["convo-7"].action.value == "call_tool"  # order reference present
    assert by_src["convo-8"].action.value == "answer_kb"  # named product
    assert by_src["convo-3"].action.value == "answer_kb"
    # parallel EN/RU rows share a split; held-out KK rows never train
    en, ru = by_src["en-US-1"], by_src["ru-RU-1"]
    assert en.split == ru.split
    assert ru.is_translation
    assert not en.is_translation
    assert all(e.split is not Split.TRAIN for e in core if e.lang == "kk")
    assert by_src["en-US-2"].intent == "greeting"
    unverified = [
        IntentExample.model_validate(r) for r in _read(res.path / "intents/unverified.jsonl")
    ]
    assert [u.source_item_id for u in unverified] == ["convo-4"]
    excluded = {r["source_item_id"]: r for r in _read(res.path / "intents/excluded.jsonl")}
    assert excluded["convo-2"]["reason"] == "exact_duplicate"
    assert excluded["convo-6"]["detail"] == "no v1 intent"
    assert excluded["en-US-4"]["detail"] == "wake word"
    assert excluded["ru-RU-3"]["detail"] == "shopping-like"
    # grounded rows: both page variants are one document family -> one split
    grounded = [
        GroundedExample.model_validate(r)
        for s in Split
        for r in _read(res.path / "grounded" / f"{s.value}.jsonl")
    ]
    assert len(grounded) == 2
    assert len({g.split for g in grounded}) == 1
    assert "1-518-473-6586" in grounded[0].response  # published in the document, so kept
    stats = json.loads((res.path / "reports" / "stats.json").read_text("utf-8"))
    assert stats["languages"]["kk"]["core_rows"] == 2
    assert stats["human_review"] == "NOT_EVALUATED"
    assert (res.path / "review" / "sample_ru.csv").read_text("utf-8").startswith("example_id,")


def test_tampering_is_detected(project: dict[str, Path]) -> None:
    res = build_corpus(project["data"], project["data"] / "corpus", _inputs(project), None)
    target = res.path / "intents" / "train.jsonl"
    target.write_text(target.read_text("utf-8") + "\n", "utf-8")
    assert verify_corpus(res.path)


def test_policy_and_raw_integrity_gates(project: dict[str, Path]) -> None:
    reg = yaml.safe_load(project["registry"].read_text("utf-8"))
    for s in reg["sources"]:
        if s["source_id"] == "abcd":
            s["allowed_use"] = "eval_only"
    project["registry"].write_text(yaml.safe_dump(reg, allow_unicode=True), "utf-8")
    with pytest.raises(PolicyError):
        build_corpus(project["data"], project["data"] / "corpus", _inputs(project), None)
    for s in reg["sources"]:
        s["allowed_use"] = "train"
    project["registry"].write_text(yaml.safe_dump(reg, allow_unicode=True), "utf-8")
    raw = next((project["data"] / "raw" / "xsid_kk").iterdir())
    (raw / "data/xSID-0.7/kk.test.conll").write_text("tampered", "utf-8")
    with pytest.raises(CorpusError, match="raw verification failed"):
        build_corpus(project["data"], project["data"] / "corpus", _inputs(project), None)


def test_cli_build_and_verify(project: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    cfg = project["root"] / "cfg.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "paths": {"data_dir": project["data"].as_posix()},
                "data": {
                    "registry": project["registry"].as_posix(),
                    "corpus_recipe": project["recipe"].as_posix(),
                    "taxonomy": project["taxonomy"].as_posix(),
                },
            }
        ),
        "utf-8",
    )
    assert main(["--config", str(cfg), "data", "corpus", "build"]) == 0
    built = json.loads(capsys.readouterr().out)
    assert built["created"] is True
    assert main(["--config", str(cfg), "data", "corpus", "verify"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
