"""End-to-end human corpus build on a tiny ABCD-shaped fixture (ADR-042)."""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from supportpilot.common.config import find_project_root
from supportpilot.common.hashing import sha256_text
from supportpilot.data.human import BuildInputs, HumanRecipe, historical_sources, load_recipe
from supportpilot.data.humangates import build_human, verify_human
from supportpilot.data.taxonomy import load_taxonomy
from tests.data_helpers import entry, make_raw

ROOT = find_project_root(Path(__file__).parent)

REQUESTS = {
    "recover_password": [
        "I forgot my password and cannot log in",
        "can't remember the password for my account at all",
        "my password is lost, please help me get back into my profile",
        "need a password reset, locked out since monday",
        "password not working anymore, I think I forgot it",
        "help, forgot password again and the reset link never came",
    ],
    "slow_speed": [
        "your website is super slow today",
        "the site keeps loading forever on my laptop",
        "pages take a minute to load, the website lags",
        "everything on the site is slow, cannot browse",
        "website speed is terrible, nothing loads",
        "the site is lagging so much I gave up",
    ],
}
ECHO = "I would like to know about these boots because you are thinking about buying them"
TEMPLATE = "I forgot my username"
LEAKED = "where is my order number please tell me"


def _convo(cid: int, subflow: str, text: str) -> dict[str, Any]:
    return {
        "convo_id": cid,
        "scenario": {
            "personal": {
                "customer_name": "jo tester",
                "email": "jo@email.com",
                "phone": "(977) 625-2661",
            },
            "order": {"order_id": "1234567890"},
            "flow": "x",
            "subflow": subflow,
        },
        "original": [
            ["agent", "Hi! How can I help you?"],
            ["customer", "hello"],
            ["agent", "Hello, what is the issue?"],
            ["customer", text],
            ["agent", "Sure, may I have your name?"],
            ["customer", "Jo Tester, jo@email.com"],
        ],
    }


def _abcd_bytes() -> bytes:
    convos = []
    cid = 1
    for subflow, texts in REQUESTS.items():
        for t in texts:
            convos.append(_convo(cid, subflow, t))
            cid += 1
    convos.append(_convo(cid, "boots_how_1", ECHO))
    for i in range(4):
        convos.append(_convo(cid + 1 + i, "recover_username", TEMPLATE))
    convos.append(_convo(cid + 9, "recover_password", "only Jo Tester"))
    convos.append(_convo(cid + 10, "status_service_added", "a premium subscription was added"))
    convos.append(_convo(cid + 11, "status_delivery_time", LEAKED))
    data = {"train": convos, "dev": [], "test": []}
    return gzip.compress(json.dumps(data).encode())


def _encode(texts: Sequence[str]) -> npt.NDArray[np.float32]:
    out = np.zeros((len(texts), 512), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"\w+", t.lower()):
            out[i, int(sha256_text(w)[:6], 16) % 512] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.maximum(norms, 1e-9)).astype(np.float32)


def _lid(texts: Sequence[str]) -> list[tuple[str, float, bool, tuple[str, ...]]]:
    return [("en", 1.0, False, ("en",)) for _ in texts]


@pytest.fixture
def setup(tmp_path: Path) -> tuple[BuildInputs, HumanRecipe]:
    tax = load_taxonomy(ROOT / "configs" / "taxonomy.yaml")
    full = load_recipe(ROOT / "configs" / "human_corpus.yaml", tax)
    data = full.model_dump()
    data["authorship"] = {"abcd": data["authorship"]["abcd"]}
    data["sources"] = {"abcd": data["sources"]["abcd"]}
    data["grounded"] = None
    data["split_ratios"] = {"train": 0.6, "validation": 0.2, "test": 0.2}
    recipe = HumanRecipe.model_validate(data)
    abcd = entry(
        "abcd",
        licence="mit",
        pipelines=["human"],
        fetch={"type": "github_repo", "repo_id": "org/abcd", "include": ["data/*"]},
    )
    make_raw(tmp_path, abcd, {"data/abcd_v1.1.json.gz": _abcd_bytes()})
    gold = tmp_path / "gold" / "gold-x" / "gold.jsonl"
    gold.parent.mkdir(parents=True)
    gold.write_text(json.dumps({"turns": [{"role": "user", "content": LEAKED}]}) + "\n", "utf-8")
    inp = BuildInputs(
        entries=[abcd],
        data_dir=tmp_path,
        recipe=recipe,
        tax=tax,
        lid=_lid,
        encode=_encode,
        history_paths=historical_sources(tmp_path),
    )
    return inp, recipe


def test_build_publishes_gated_immutable_corpus(setup: tuple[BuildInputs, HumanRecipe]) -> None:
    inp, _ = setup
    res = build_human(inp, code_commit="abc", recipe_sha256="r", taxonomy_sha256="t", previous=None)
    assert res["status"] == "built", res
    assert res["gates"] == "pass"
    out = Path(res["path"])
    assert re.fullmatch(r"human-\d{8}-[0-9a-f]{8}", out.name)

    active = pl.concat(
        [pl.read_parquet(out / f"{s}.parquet") for s in ("train", "validation", "test")]
    )
    assert active.height == 12
    assert set(active["intent"]) == {"account_help", "technical_issue"}
    assert set(active["origin"]) == {"human_roleplay"}
    assert set(active["sft_target_status"]) == {"no_human_target"}
    assert set(active["split"]) == {"train", "validation", "test"}
    assert not active["text"].str.contains("Jo|Tester|@|1234567890").any()
    assert active["context_json"].str.contains("hello").all()

    unv = pl.read_parquet(out / "unverified.parquet")
    assert dict(zip(unv["reason"], unv["reason"], strict=True)).keys() >= {
        "prompt_echo",
        "template_cluster",
    }
    exc = pl.read_parquet(out / "excluded.parquet")
    reasons = set(exc["reason"])
    assert {"excluded", "historical_exact"} <= reasons
    assert LEAKED not in set(active["text"]) | set(unv["text"])
    # the name-only request is never treated as a request text
    assert "<name>" not in set(active["text"])

    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert stats["active"] == 12
    assert stats["lang"] == {"en": 12}
    assert "ru" in stats["intents_missing_by_lang"]

    ver = verify_human(out)
    assert ver["checksums_ok"]
    assert ver["gates"] == "pass"
    again = build_human(
        inp, code_commit="abc", recipe_sha256="r", taxonomy_sha256="t", previous=None
    )
    assert again["status"] == "skipped"
    assert again["dataset_version"] == res["dataset_version"]


def test_build_fails_gates_without_semantic_check(setup: tuple[BuildInputs, HumanRecipe]) -> None:
    from supportpilot.data.steps import StepError

    inp, _ = setup
    inp.encode = None
    with pytest.raises(StepError, match="quality gates failed"):
        build_human(inp, code_commit="abc", recipe_sha256="r", taxonomy_sha256="t", previous=None)
