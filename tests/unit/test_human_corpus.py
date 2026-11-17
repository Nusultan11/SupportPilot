"""Human-authored corpus transforms (ADR-042)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from supportpilot.common.config import find_project_root
from supportpilot.data.acquire import AcquireError, GitHubRepoRemote, remote_for
from supportpilot.data.human import (
    ACTIVE_ORIGINS,
    HumanRecipe,
    Row,
    _strings,
    abcd_label,
    assign_splits,
    families,
    load_recipe,
    map_label,
    mask_text,
    opening_request,
)
from supportpilot.data.taxonomy import Taxonomy, load_taxonomy
from tests.data_helpers import entry, git_sha1

ROOT = find_project_root(Path(__file__).parent)


@pytest.fixture(scope="module")
def tax() -> Taxonomy:
    return load_taxonomy(ROOT / "configs" / "taxonomy.yaml")


@pytest.fixture(scope="module")
def recipe(tax: Taxonomy) -> HumanRecipe:
    return load_recipe(ROOT / "configs" / "human_corpus.yaml", tax)


def _row(text: str, label: str, source: str = "abcd", **kw: Any) -> Row:
    return Row(
        f"{source}:x-{abs(hash(text))}", source, "en", "human_roleplay", text, label, "train", **kw
    )


def test_project_recipe_is_valid_and_admits_only_documented_humans(recipe: HumanRecipe) -> None:
    assert recipe.authorship["abcd"].active
    assert not recipe.authorship["mangi_kk_support_qa"].active
    for a in recipe.authorship.values():
        if a.active:
            assert set(a.origin.values()) <= ACTIVE_ORIGINS


def test_recipe_rejects_active_machine_origin(recipe: HumanRecipe) -> None:
    data = recipe.model_dump()
    data["authorship"]["abcd"]["origin"] = {"en": "machine_translated"}
    with pytest.raises(ValidationError, match="documented human origins"):
        HumanRecipe.model_validate(data)


def test_recipe_rejects_unknown_intent(recipe: HumanRecipe, tax: Taxonomy) -> None:
    data = recipe.model_dump()
    data["sources"]["abcd"]["recover_password"] = {"intent": "teleport"}
    with pytest.raises(ValueError, match="unknown intent"):
        HumanRecipe.model_validate(data).check_taxonomy(tax)


def test_opening_request_skips_greetings_into_context(recipe: HumanRecipe) -> None:
    greeting = re.compile(recipe.greeting_only, re.IGNORECASE)
    turns = [
        ("agent", "Hi! How can I help?"),
        ("customer", "hello"),
        ("agent", "Hello, what can I do for you?"),
        ("customer", "I need to return my boots"),
        ("customer", "they are too small"),
        ("agent", "Sure, your name?"),
        ("customer", "<name>"),
    ]
    ctx, req = opening_request(turns, greeting)
    assert req == ["I need to return my boots", "they are too small"]
    assert [m["role"] for m in ctx] == ["assistant", "user", "assistant"]


def test_opening_request_with_greetings_only_is_empty(recipe: HumanRecipe) -> None:
    greeting = re.compile(recipe.greeting_only, re.IGNORECASE)
    assert (
        opening_request([("customer", "hi"), ("agent", "hello"), ("customer", "<name>")], greeting)[
            1
        ]
        == []
    )


def test_mask_text_replaces_dialogue_values_and_patterns() -> None:
    values = [("crystal minh", "<name>"), ("crystal", "<name>"), ("3348917502", "<order_id>")]
    out = mask_text(
        "Crystal Minh here, order 3348917502, mail c.m@email.com, (977) 625-2661", values
    )
    assert "Crystal" not in out
    assert "3348917502" not in out
    assert "@" not in out
    assert "<name>" in out
    assert "<order_id>" in out
    assert "<email>" in out
    assert "<phone>" in out
    assert mask_text("account VUCUIRSG8C ok", []) == "account <account_id> ok"


def test_abcd_label_strips_numeric_suffix() -> None:
    assert abcd_label("boots_how_3") == "boots_how"
    assert abcd_label("refund_status") == "refund_status"


@pytest.mark.parametrize(
    ("text", "label", "intent", "action"),
    [
        ("I forgot my password", "recover_password", "account_help", "answer_kb"),
        ("I need to remove an item from my order", "manage_cancel", "change_order", "clarify"),
        ("remove item from order <order_id>", "manage_cancel", "change_order", "call_tool"),
        ("what's up with my refund?", "refund_update", "refund_status", "clarify"),
        (
            "I was charged but I never ordered anything",
            "mistimed_billing_never_bought",
            "payment_problem",
            "escalate",
        ),
        ("Are the Guess boots waterproof?", "boots_other", "product_info", "answer_kb"),
        ("Do these boots come in black?", "boots_other", "product_info", "clarify"),
        ("I was sent the wrong size", "return_size", "delivery_problem", "clarify"),
        ("The website is so slow", "slow_speed", "technical_issue", "escalate"),
    ],
)
def test_label_mapping(
    recipe: HumanRecipe, tax: Taxonomy, text: str, label: str, intent: str, action: str
) -> None:
    slots = {"order_id": "<order_id>"} if "<order_id>" in text else {}
    d = map_label(recipe, tax, _row(text, label, slots=slots))
    assert (d.status, d.intent, d.action) == ("mapped", intent, action)
    assert (d.escalation_reason is not None) == (action == "escalate")


@pytest.mark.parametrize(
    ("text", "label", "status"),
    [
        ("I want to return those jeans", "manage_cancel", "excluded"),
        ("I am interested in a premium subscription", "status_service_added", "excluded"),
        ("hello there friend", "refund_status", "excluded"),
        ("can you help", "unknown_flow", "not_in_map"),
    ],
)
def test_label_mapping_never_guesses(
    recipe: HumanRecipe, tax: Taxonomy, text: str, label: str, status: str
) -> None:
    assert map_label(recipe, tax, _row(text, label)).status == status


def test_prompt_echo_patterns_catch_second_person_prompt_text(recipe: HumanRecipe) -> None:
    pats = [re.compile(p, re.IGNORECASE) for p in recipe.prompt_echo_patterns]
    echoed = "I want to remove the second item from your order because your partner/spouse hates it"
    natural = "I want to remove the second item from my order, my wife hates it"
    assert any(p.search(echoed) for p in pats)
    assert not any(p.search(natural) for p in pats)


def test_families_join_near_copies_and_aligned_translations() -> None:
    rows = [
        _row("I would like to return an item because it is the wrong size", "return_size"),
        _row("I would like to return an item because it is the wrong size!!", "return_size"),
        _row("wake me up at five", "alarm_set", source="massive_en_ru", align="m:test.1"),
        _row("разбуди меня в пять", "alarm_set", source="massive_en_ru", align="m:test.1"),
        _row("totally different question about shipping fees", "cost"),
    ]
    comp = families(rows, 0.6, seed=1)
    assert comp[0] == comp[1]
    assert comp[2] == comp[3]
    assert len({comp[0], comp[2], comp[4]}) == 3


def test_assign_splits_is_deterministic_and_covers_heldout() -> None:
    groups = {f"fam_{i:02d}": "a" if i < 20 else "b" for i in range(23)}
    s1 = assign_splits(groups, {"train": 0.8, "validation": 0.1, "test": 0.1}, seed=7)
    assert s1 == assign_splits(groups, {"train": 0.8, "validation": 0.1, "test": 0.1}, seed=7)
    for intent in ("a", "b"):
        got = {s1[f] for f, i in groups.items() if i == intent}
        assert got == {"train", "validation", "test"}


def test_history_extraction_keeps_user_texts_only() -> None:
    obj: dict[str, Any] = {
        "test": [
            {
                "user_text": "where is my parcel",
                "messages": [
                    {"role": "system", "content": "SYSTEM PROMPT"},
                    {"role": "user", "content": "where is my parcel please"},
                    {"role": "assistant", "content": "ASSISTANT ANSWER"},
                ],
                "target": {"answer": "ANSWER TEXT"},
            }
        ],
        "needs": [["need one text", "need two text"]],
        "gold": [{"turns": [{"role": "user", "content": "gold question"}]}],
    }
    out: list[str] = []
    _strings(json.loads(json.dumps(obj)), out)
    assert set(out) == {
        "where is my parcel",
        "where is my parcel please",
        "need one text",
        "need two text",
        "gold question",
    }


def test_pipelines_field_is_not_serialised() -> None:
    e = entry("abcd", pipelines=["human"])
    assert "pipelines" not in e.model_dump_json()
    assert e.model_dump_json() == entry("abcd").model_dump_json()
    assert [p.value for p in e.pipelines] == ["human"]


def test_remote_for_selects_github(tmp_path: Path) -> None:
    gh = entry("abcd", fetch={"type": "github_repo", "repo_id": "org/abcd", "include": ["*"]})
    assert isinstance(remote_for(gh), GitHubRepoRemote)


class _FakeGitHub(GitHubRepoRemote):
    def __init__(self, files: dict[str, bytes]) -> None:
        super().__init__("org/repo", "0" * 40)
        self.files = files

    def _get(self, url: str) -> bytes:
        if "/git/trees/" in url:
            tree = [
                {"path": p, "type": "blob", "size": len(d), "sha": git_sha1(d)}
                for p, d in self.files.items()
            ]
            return json.dumps({"tree": tree, "truncated": False}).encode()
        return self.files[url.rsplit("/", 1)[1]]


def test_github_remote_lists_and_downloads_verified(tmp_path: Path) -> None:
    from supportpilot.data.acquire import check_remote_file

    remote = _FakeGitHub({"a.json": b'{"x": 1}'})
    (rf,) = remote.list_files()
    remote.download(rf, tmp_path)
    assert check_remote_file(tmp_path / "a.json", rf) is None


def test_github_remote_rejects_lfs_pointer(tmp_path: Path) -> None:
    remote = _FakeGitHub({"big.bin": b"version https://git-lfs.github.com/spec/v1\noid sha256:x"})
    (rf,) = remote.list_files()
    with pytest.raises(AcquireError, match="LFS pointer"):
        remote.download(rf, tmp_path)


def test_heldout_families_never_enter_train() -> None:
    groups = {f"f{i}": "out_of_scope" for i in range(40)}
    held = frozenset(f"f{i}" for i in range(20))
    out = assign_splits(groups, {"train": 0.8, "validation": 0.1, "test": 0.1}, 7, held)
    assert {out[f] for f in held} == {"validation", "test"}
    assert "train" in {out[f] for f in groups if f not in held}


def test_project_recipe_caps_placement_and_name_patterns(recipe: HumanRecipe) -> None:
    assert recipe.heldout_only("xsid_kk")
    assert not recipe.heldout_only("abcd")
    assert recipe.cap("clinc150", "greeting") == 60  # label cap
    assert recipe.cap("clinc150", "weather") == 6  # source cap
    assert recipe.cap("massive_en_ru", "weather_query") == recipe.native_max_per_label
    names = [re.compile(p, re.IGNORECASE) for p in recipe.personal_name_patterns]

    def hit(t: str) -> bool:
        return any(p.search(t) for p in names)

    assert hit("Hello my name is Crustal <name> and I want a refund")
    assert hit("здравствуйте меня зовут Елена")
    assert hit("Менің атым Айгүл")
    assert not hit("my name is spelled wrong on my account")
    assert not hit("my name is <name>")
    assert not hit("встречи между первым и шестнадцатым марта")
    assert recipe.grounded is not None
    assert recipe.grounded.namespace.endswith("_external")
