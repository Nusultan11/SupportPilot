"""Authored native-style scenarios -> utterances (Stage 4, ADR-025).

`authored/scenarios/*.yaml` holds situations written natively in en / ru / kk (and RU-KK
code-switching) by the Stage 4 agent (LLM), not translated from Bitext. Each situation is
one `scenario_id`; each template is one paraphrase `family_id`; each template is filled
`fills` times with synthetic Kazakhstan values (locale catalogue). Multi-turn situations
carry a `history` of earlier turns (F11: the order id is carried over from the history).

Templates use {order_id}, {product}, {sku}, {city}, {city_to}, {city_in}, {name},
{amount}, {amount_n}, {days}: language-specific surface forms are chosen here.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from supportpilot.common.hashing import sha256_text
from supportpilot.common.schemas import INTENTS, Action, LanguageCode
from supportpilot.data.locale import LocaleCatalog, rng_for
from supportpilot.data.textfmt import kzt, render

AUTHORED_SOURCE = "supportpilot_authored"
AUTHOR = "stage4-agent (Claude, LLM); human review pending"
MIX_KEYS = {"mix_kk": "kk", "mix_ru": "ru"}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Turn(_Model):
    user: dict[str, str]
    assistant: dict[str, str]


class Scenario(_Model):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    intent: str
    action: Literal["answer_kb", "call_tool", "clarify", "escalate", "refuse_oos"]
    task: str | None = None  # default from the action; e.g. injection_resist
    sections: list[str] = Field(default_factory=list)
    tool_status: list[str] = Field(default_factory=list)
    ask: str | None = None
    reason: str | None = None
    product: bool = False  # the message names a product (catalogue / stock)
    history: list[Turn] = Field(default_factory=list)
    fills: int = Field(default=3, ge=1, le=10)
    en: list[str] = Field(default_factory=list)
    ru: list[str] = Field(default_factory=list)
    kk: list[str] = Field(default_factory=list)
    mix_kk: list[str] = Field(default_factory=list)
    mix_ru: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _valid(self) -> Scenario:
        if self.intent not in INTENTS:
            raise ValueError(f"{self.id}: unknown intent {self.intent}")
        if self.action == "answer_kb" and not self.sections:
            raise ValueError(f"{self.id}: answer_kb needs sections")
        if self.action == "call_tool" and not self.tool_status:
            raise ValueError(f"{self.id}: call_tool needs tool_status")
        if self.action == "clarify" and not self.ask:
            raise ValueError(f"{self.id}: clarify needs ask")
        if self.action == "escalate" and not self.reason:
            raise ValueError(f"{self.id}: escalate needs reason")
        if not (self.en or self.ru or self.kk or self.mix_kk or self.mix_ru):
            raise ValueError(f"{self.id}: no templates")
        for t in self.history:
            if set(t.user) != set(t.assistant):
                raise ValueError(f"{self.id}: history turns need the same languages")
        return self

    def templates(self) -> Iterator[tuple[str, LanguageCode, bool, int, str]]:
        """(key, lang, code_switch, index, template)."""
        for key in ("en", "ru", "kk", "mix_kk", "mix_ru"):
            lang: LanguageCode = MIX_KEYS.get(key, key)  # type: ignore[assignment]
            for i, t in enumerate(getattr(self, key)):
                yield key, lang, key in MIX_KEYS, i, t


class ScenarioFile(_Model):
    schema_version: Literal["scenarios/1"] = "scenarios/1"
    scenarios: list[Scenario]


def load_scenarios(directory: Path) -> list[Scenario]:
    out: list[Scenario] = []
    for path in sorted(directory.glob("*.yaml")):
        out.extend(ScenarioFile.model_validate(yaml.safe_load(path.read_text("utf-8"))).scenarios)
    ids = [s.id for s in out]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate scenario ids: {dupes}")
    return out


def fill_values(locale: LocaleCatalog, lang: str, *seed: object) -> dict[str, Any]:
    rng = rng_for(*seed)
    city = rng.choice(locale.cities)
    product = rng.choice(locale.products)
    name = rng.choice(locale.names)
    amount = (
        rng.randrange(3000, 250000, 100) if rng.random() < 0.7 else rng.randrange(500, 3000, 50)
    )
    city_to = {"en": city.en, "ru": city.ru.to, "kk": city.kk.dat}[lang]
    city_in = {"en": city.en, "ru": city.ru.in_, "kk": city.kk.loc}[lang]
    pname = product.get(lang)
    return {
        "order_id": str(rng.randrange(100000, 1000000)),
        "order_id2": str(rng.randrange(100000, 1000000)),
        "product": pname,
        "product_cap": pname[:1].upper() + pname[1:],
        "sku": product.sku,
        "_sku": product.sku,
        "city": {"en": city.en, "ru": city.ru.nom, "kk": city.kk.nom}[lang],
        "city_to": city_to,
        "city_in": city_in,
        "_city_en": city.en,
        "name": name.get(lang),
        "amount": kzt(amount, lang) if lang != "en" else f"{amount:,} tenge",
        "amount_n": str(amount),
        "days": str(rng.randrange(3, 15)),
    }


def utterance_id(scenario: str, key: str, idx: int, fill: int) -> str:
    return f"{scenario}.{key}.{idx}.{fill}"


def expand(scenarios: list[Scenario], locale: LocaleCatalog, seed: int) -> list[dict[str, Any]]:
    """All utterances of all scenarios (deterministic)."""
    rows: list[dict[str, Any]] = []
    for sc in scenarios:
        scenario_id = "scn_" + sha256_text(f"{AUTHORED_SOURCE}|{sc.id}")[:16]
        for key, lang, cs, idx, tpl in sc.templates():
            family_id = "fam_" + sha256_text(f"{AUTHORED_SOURCE}|{sc.id}|{key}|{idx}")[:16]
            seen: set[str] = set()
            for fill in range(sc.fills):
                vals = fill_values(locale, lang, seed, sc.id, key, idx, fill)
                text = render(tpl, vals, lang)
                history: list[dict[str, str]] = []
                for turn in sc.history:
                    hkey = lang if lang in turn.user else "en"
                    history.append({"role": "user", "content": render(turn.user[hkey], vals, lang)})
                    history.append(
                        {"role": "assistant", "content": render(turn.assistant[hkey], vals, lang)}
                    )
                dedup = text + "|" + "|".join(h["content"] for h in history)
                if dedup in seen:
                    continue  # a template without slots yields one utterance, not copies
                seen.add(dedup)
                uid = utterance_id(sc.id, key, idx, fill)
                rows.append(
                    {
                        "utterance_id": uid,
                        "record_id": f"{AUTHORED_SOURCE}:{uid}",
                        "source_id": AUTHORED_SOURCE,
                        "scenario_key": sc.id,
                        "scenario_id": scenario_id,
                        "family_id": family_id,
                        "lang": lang,
                        "code_switch": cs,
                        "text": text,
                        "history": history,
                        "intent": sc.intent,
                        "action": sc.action,
                        "task": sc.task,
                        "sections": list(sc.sections),
                        "tool_status": list(sc.tool_status),
                        "ask": sc.ask,
                        "reason": sc.reason,
                        "product": sc.product,
                        "slots": {
                            "order_id": vals["order_id"],
                            "sku": vals["_sku"],
                            "city": vals["_city_en"],
                        },
                        "origin": "llm_generated",
                    }
                )
    return rows


def default_task(action: str) -> str:
    return {
        Action.ANSWER_KB.value: "grounded_answer",
        Action.CALL_TOOL.value: "tool_answer",
        Action.CLARIFY.value: "clarify",
        Action.ESCALATE.value: "escalate",
        Action.REFUSE_OOS.value: "refuse_oos",
    }[action]
