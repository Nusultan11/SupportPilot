"""Kazakhstan localisation catalogue (configs/locale_kz.yaml, ADR-024).

Used to localise Bitext slot values before machine translation (USD amounts -> KZT, names
and cities in the target-language form) and by the authored scenario expander, the training
KB and the mock tool states. All values are synthetic.
"""

from __future__ import annotations

import random
import re
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from supportpilot.data.textfmt import group_digits


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RuCity(_Model):
    nom: str
    to: str
    in_: str = Field(alias="in")
    from_: str = Field(alias="from")


class KkCity(_Model):
    nom: str
    dat: str
    loc: str
    abl: str


class City(_Model):
    en: str
    ru: RuCity
    kk: KkCity


class Trilingual(_Model):
    id: str | None = None
    en: str
    ru: str
    kk: str

    def get(self, lang: str) -> str:
        return str(getattr(self, lang))


class Product(_Model):
    sku: str = Field(pattern=r"^[A-Z]{2}-\d{4}$")
    cat: str
    en: str
    ru: str
    kk: str
    price: tuple[int, int]

    def get(self, lang: str) -> str:
        return str(getattr(self, lang))


class LocaleCatalog(_Model):
    schema_version: Literal["locale/1"] = "locale/1"
    currency: Literal["KZT"] = "KZT"
    usd_to_kzt: int = Field(gt=0)
    kzt_round_to: int = Field(gt=0)
    cities: list[City] = Field(min_length=5)
    names: list[Trilingual] = Field(min_length=5)
    carriers: list[Trilingual]
    payment_methods: list[Trilingual]
    products: list[Product] = Field(min_length=10)

    @model_validator(mode="after")
    def _unique(self) -> LocaleCatalog:
        for what, keys in (
            ("city", [c.en for c in self.cities]),
            ("carrier", [c.id for c in self.carriers]),
            ("payment method", [p.id for p in self.payment_methods]),
            ("sku", [p.sku for p in self.products]),
        ):
            if len(set(keys)) != len(keys) or None in keys:
                raise ValueError(f"duplicate or missing {what} ids")
        return self

    @cached_property
    def city_by_en(self) -> dict[str, City]:
        return {c.en: c for c in self.cities}

    @cached_property
    def name_by_en(self) -> dict[str, Trilingual]:
        return {n.en: n for n in self.names}

    def lookup(self, kind: str, key: str) -> dict[str, Any]:
        """Resolver for `textfmt.render(catalog=...)`."""
        if kind == "city":
            c = self.city_by_en[key]
            return {"en": c.en, "ru": c.ru.model_dump(by_alias=True), "kk": c.kk.model_dump()}
        table = {"carrier": self.carriers, "payment_method": self.payment_methods}[kind]
        for item in table:
            if item.id == key:
                return item.model_dump()
        raise KeyError(f"{kind} {key!r} not in the locale catalogue")

    def usd_to_kzt_amount(self, usd: float) -> int:
        step = self.kzt_round_to
        return max(step, round(usd * self.usd_to_kzt / step) * step)


def load_locale(path: Path) -> LocaleCatalog:
    return LocaleCatalog.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


KZT_SCALE_MIN = 1000  # amounts at or above this are taken as tenge values (see localize_en)
_USD_RE = re.compile(
    r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{2}))?|\b(\d+(?:\.\d{2})?)\s?(?:USD|dollars?)\b"
)


class Localized(BaseModel):
    text: str
    fixes: list[str] = Field(default_factory=list)
    amounts_kzt: list[int] = Field(default_factory=list)


def localize_en(text: str, catalog: LocaleCatalog) -> Localized:
    """EN source text -> EN text with Kazakhstan values (before translation).

    Dollar amounts become KZT at the catalogue rate. Names and cities in the Stage 3 slot
    catalogue are already Kazakhstan values and stay as they are (they are replaced by the
    target-language form after translation, see `target_forms`).
    """
    fixes: list[str] = []
    amounts: list[int] = []

    def money(m: re.Match[str]) -> str:
        raw = m.group(1) or m.group(3) or "0"
        value = float(raw.replace(",", ""))
        if value >= KZT_SCALE_MIN:
            # Stage 3 slot values ({{Refund Amount}}: 1,000-250,000) are already shop prices in
            # tenge scale; converting them at the USD rate gave 100+ million tenge refunds.
            kzt_value = int(value)
            fixes.append("locale:amount_as_kzt")
        else:
            kzt_value = catalog.usd_to_kzt_amount(value)
            fixes.append("locale:usd_to_kzt")
        amounts.append(kzt_value)
        return f"{group_digits(kzt_value, 'en')} KZT"

    out = _USD_RE.sub(money, text)
    return Localized(text=out, fixes=sorted(set(fixes)), amounts_kzt=amounts)


def target_forms(text_en: str, lang: str, catalog: LocaleCatalog) -> dict[str, str]:
    """Names and city names in `text_en` mapped to their `lang` nominative form."""
    if lang == "en":
        return {}
    forms: dict[str, str] = {}
    for en, name in catalog.name_by_en.items():
        if en in text_en:
            forms[en] = name.get(lang)
    for en, city in catalog.city_by_en.items():
        if re.search(rf"\b{re.escape(en)}\b", text_en):
            forms[en] = city.ru.nom if lang == "ru" else city.kk.nom
    return forms


def rng_for(*parts: object) -> random.Random:
    """Deterministic RNG for synthetic values (independent of PYTHONHASHSEED)."""
    from supportpilot.common.hashing import sha256_text

    return random.Random(int(sha256_text("|".join(map(str, parts)))[:16], 16))  # noqa: S311
