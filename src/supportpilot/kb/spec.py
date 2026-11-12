"""Demo-store KB sources: canonical facts, record data and document specs (ADR-027).

- `facts.yaml` (demo_facts/1): store policy values. Stage 4 reads the same file.
- `catalog.yaml` (demo_catalog/1): products and delivery zones.
- `docs/<doc_id>.yaml` (kb_doc/1): one document, its format and its text per language with
  `{fact:format}` placeholders.

`fact_registry` flattens facts and records into canonical leaf facts. `Filler` fills a
template and reports every fact it placed with the exact surface string, which is the ground
truth the extraction gates check against.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from supportpilot.common.schemas import LANGUAGES, DocFormat, DocType, LanguageCode, Slug
from supportpilot.data import textfmt

DOCS_DIR = "docs"
FACTS_FILE = "facts.yaml"
CATALOG_FILE = "catalog.yaml"


class SpecError(ValueError):
    """A KB source file is invalid or a template cannot be filled."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Tri(_Model):
    en: str = Field(min_length=1)
    ru: str = Field(min_length=1)
    kk: str = Field(min_length=1)

    def get(self, lang: LanguageCode) -> str:
        return str(getattr(self, lang))


Local = str | Tri  # a plain string is the same in every language


def pick(value: Local, lang: LanguageCode) -> str:
    return value if isinstance(value, str) else value.get(lang)


class TableSpec(_Model):
    columns: list[Tri] = Field(min_length=1)
    rows: list[list[Local]] = Field(min_length=1)

    @model_validator(mode="after")
    def _rectangular(self) -> Self:
        if any(len(r) != len(self.columns) for r in self.rows):
            raise ValueError("every table row needs one cell per column")
        return self


class SectionSpec(_Model):
    id: Slug
    heading: Tri
    level: Literal[1, 2] = 1
    paragraphs: list[Tri] = Field(default_factory=list)
    table: TableSpec | None = None

    @model_validator(mode="after")
    def _content(self) -> Self:
        if not self.paragraphs and self.table is None:
            raise ValueError(f"section {self.id!r} has no content")
        return self


class ColumnSpec(_Model):
    header: Tri
    value: str = Field(min_length=1)
    type: Literal["text", "int"] = "text"
    when: Literal["warranty"] | None = None


class RecordsSpec(_Model):
    source: Literal["products", "delivery_zones"]
    key: str
    sheet: Tri | None = None
    columns: list[ColumnSpec] = Field(min_length=2)

    @model_validator(mode="after")
    def _key_first(self) -> Self:
        if self.columns[0].value != f"{{rec.{self.key}}}":
            raise ValueError("the first column must be the record key")
        return self


class DocSpec(_Model):
    schema_version: Literal["kb_doc/1"] = "kb_doc/1"
    doc_id: Slug
    doc_type: DocType
    format: DocFormat
    title: Tri
    intro: list[Tri] = Field(default_factory=list)
    sections: list[SectionSpec] = Field(default_factory=list)
    records: RecordsSpec | None = None

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if bool(self.sections) == (self.records is not None):
            raise ValueError("a document has either sections or records")
        ids = [s.id for s in self.sections]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.doc_id}: duplicate section ids")
        if self.sections and self.sections[0].level != 1:
            raise ValueError(f"{self.doc_id}: the first section must be level 1")
        if self.doc_type is DocType.FAQ and any(s.level != 1 or s.table for s in self.sections):
            raise ValueError(f"{self.doc_id}: FAQ items are level-1 questions without tables")
        if self.records is not None and self.format not in (DocFormat.XLSX, DocFormat.CSV):
            raise ValueError(f"{self.doc_id}: record documents are xlsx or csv")
        if self.records is None and self.format in (DocFormat.XLSX, DocFormat.CSV):
            raise ValueError(f"{self.doc_id}: xlsx/csv documents are record documents")
        if self.format is DocFormat.JSON and self.doc_type is not DocType.FAQ:
            raise ValueError(f"{self.doc_id}: json documents are FAQ exports")
        return self

    @property
    def numbered(self) -> bool:
        """Policies and terms number their headings (1., 1.1); FAQs do not."""
        return self.doc_type in (DocType.POLICY, DocType.TERMS)


class ProductSpec(_Model):
    sku: str = Field(pattern=r"^[A-Z]{2}-\d{4}$")
    cat: str
    price: PositiveInt
    name: Tri
    desc: Tri


class ZoneSpec(_Model):
    id: Slug
    zone: Literal["fast", "region"]
    en: str
    ru: str
    kk: str


class CatalogSpec(_Model):
    schema_version: Literal["demo_catalog/1"] = "demo_catalog/1"
    warranty_categories: list[str]
    categories: dict[str, Tri]
    products: list[ProductSpec] = Field(min_length=1)
    zones: dict[Literal["fast", "region"], Tri]
    delivery_zones: list[ZoneSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _refs(self) -> Self:
        skus = [p.sku for p in self.products]
        if len(skus) != len(set(skus)):
            raise ValueError("duplicate SKU")
        if bad := {p.cat for p in self.products} - set(self.categories):
            raise ValueError(f"unknown categories {sorted(bad)}")
        if bad := set(self.warranty_categories) - set(self.categories):
            raise ValueError(f"unknown warranty categories {sorted(bad)}")
        ids = [z.id for z in self.delivery_zones]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate zone id")
        return self


@dataclass(frozen=True)
class KBSources:
    root: Path
    facts: dict[str, Any]
    catalog: CatalogSpec
    docs: list[DocSpec]

    @property
    def effective_from(self) -> date:
        return date.fromisoformat(str(self.facts["effective_from"]))


def malformed_facts(node: Any, path: str = "") -> list[str]:
    """Paths of empty values and non-string keys. A YAML flow mapping with an unquoted comma
    (`{ru: пр. Достык, 77}`) silently turns the tail into a key with a null value."""
    bad: list[str] = []
    if isinstance(node, Mapping):
        for k, v in node.items():
            here = f"{path}.{k}" if path else str(k)
            if not isinstance(k, str):
                bad.append(f"{here} (non-string key)")
            bad += malformed_facts(v, here)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            bad += malformed_facts(v, f"{path}[{i}]")
    elif node is None or (isinstance(node, str) and not node.strip()):
        bad.append(f"{path} (empty)")
    return bad


def load_sources(root: Path) -> KBSources:
    """Load and validate `facts.yaml`, `catalog.yaml` and `docs/*.yaml` under `root`."""
    facts = yaml.safe_load((root / FACTS_FILE).read_text(encoding="utf-8"))
    if not isinstance(facts, dict) or facts.get("schema_version") != "demo_facts/1":
        raise SpecError(f"{root / FACTS_FILE}: expected schema_version demo_facts/1")
    for key in ("store_name", "effective_from"):
        if key not in facts:
            raise SpecError(f"{root / FACTS_FILE}: missing {key!r}")
    if bad := malformed_facts(facts):
        raise SpecError(f"{root / FACTS_FILE}: malformed values {bad}")
    catalog = CatalogSpec.model_validate(
        yaml.safe_load((root / CATALOG_FILE).read_text(encoding="utf-8"))
    )
    docs = []
    for path in sorted((root / DOCS_DIR).glob("*.yaml")):
        try:
            spec = DocSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        except ValueError as e:
            raise SpecError(f"{path}: {e}") from e
        if spec.doc_id != path.stem:
            raise SpecError(f"{path}: doc_id {spec.doc_id!r} must match the file name")
        docs.append(spec)
    if not docs:
        raise SpecError(f"no document specs in {root / DOCS_DIR}")
    return KBSources(root=root, facts=facts, catalog=catalog, docs=docs)


# ---------------------------------------------------------------- canonical facts


def _is_tri(v: object) -> bool:
    return isinstance(v, Mapping) and set(LANGUAGES) <= set(v)


def fact_registry(sources: KBSources) -> dict[str, Any]:
    """Canonical leaf facts: key -> value (JSON-able). Localised values are {en, ru, kk}."""
    out: dict[str, Any] = {}
    for key, value in sources.facts.items():
        if key == "schema_version":
            continue
        if isinstance(value, Mapping) and not set(value) <= set(LANGUAGES):
            for sub, v in value.items():
                if sub not in LANGUAGES:
                    out[f"{key}.{sub}"] = v
            if _is_tri(value):  # e.g. sale: {en, ru, kk, from, to}
                out[f"{key}.name"] = {lang: value[lang] for lang in LANGUAGES}
        else:
            out[key] = value
    cat = sources.catalog
    for p in cat.products:
        for name, v in record_values(p, cat).items():
            out[f"product.{p.sku}.{name}"] = v
    for z in cat.delivery_zones:
        for name, v in zone_values(z, cat).items():
            if name != "delivery_days":  # a facts.yaml value (fast/region_delivery_days)
                out[f"zone.{z.id}.{name}"] = v
    return out


def record_values(p: ProductSpec, cat: CatalogSpec) -> dict[str, Any]:
    return {
        "sku": p.sku,
        "name": p.name.model_dump(),
        "category": cat.categories[p.cat].model_dump(),
        "price": p.price,
        "desc": p.desc.model_dump(),
    }


def zone_values(z: ZoneSpec, cat: CatalogSpec) -> dict[str, Any]:
    return {
        "id": z.id,
        "name": {"en": z.en, "ru": z.ru, "kk": z.kk},
        "zone_name": cat.zones[z.zone].model_dump(),
        "delivery_days": f"{z.zone}_delivery_days",
    }


# ---------------------------------------------------------------- template filling


@dataclass(frozen=True)
class Hit:
    """One fact placed in a text: canonical key, canonical value, exact surface string."""

    key: str
    value: Any
    surface: str


@dataclass(frozen=True)
class Record:
    """Template context of one record: `{rec.<field>}` -> values; `aliases` map a derived
    field to the facts.yaml key it comes from."""

    prefix: str
    values: dict[str, Any]
    aliases: dict[str, str] = field(default_factory=dict)


# formats whose value is one field of a structured fact (showroom, sale)
_SUBKEY = {
    "city": "city",
    "address": "address",
    "sale_name": "name",
    "sale_from": "from",
    "sale_to": "to",
}
_RU_SMALL_UNITS = {
    "second": ("секунда", "секунды", "секунд"),
    "minute": ("минута", "минуты", "минут"),
}
_KK_SMALL_UNITS = {
    "second": {"nom": "секунд", "abl": "секундтан", "dat": "секундқа", "loc": "секундта"},
    "minute": {"nom": "минут", "abl": "минуттан", "dat": "минутқа", "loc": "минутта"},
}


def small_unit(n: int, unit: str, lang: LanguageCode, case: str | None = None) -> str:
    """60 seconds / 60 секунд / 60 секундтан; units textfmt does not have."""
    if lang == "en":
        return f"{n} {unit}" + ("" if n == 1 else "s")
    if lang == "ru":
        return f"{n} {textfmt.ru_plural(n, _RU_SMALL_UNITS[unit])}"
    return f"{n} {_KK_SMALL_UNITS[unit][case or 'nom']}"


def full_date(d: date, lang: LanguageCode) -> str:
    """1 November 2026 / 1 ноября 2026 г. / 2026 жылғы 1 қараша."""
    day_month = textfmt.fmt_date(d, lang)
    if lang == "en":
        return f"{day_month} {d.year}"
    if lang == "ru":
        return f"{day_month} {d.year} г."
    return f"{d.year} жылғы {day_month}"


def effective_line(d: date, lang: LanguageCode) -> str:
    """The line under every document title that states when the document applies."""
    if lang == "en":
        return f"Effective from {full_date(d, lang)}"
    if lang == "ru":
        return f"Действует с {full_date(d, lang)}"
    return f"{d.year} жылғы {textfmt.fmt_date(d, lang, 'abl')} бастап қолданылады"


Lookup = Callable[[str, str], Mapping[str, Any]]


class Filler:
    def __init__(self, facts: Mapping[str, Any], lookup: Lookup) -> None:
        self.facts = facts
        self.lookup = lookup

    def fill(
        self, template: str, lang: LanguageCode, rec: Record | None = None
    ) -> tuple[str, list[Hit]]:
        hits: list[Hit] = []

        def one(m: Any) -> str:
            name, fmt, case = m.group(1), m.group(2), m.group(3)
            if case and lang != "kk":
                case = None
            key, value = self._resolve(name, rec)
            surface = self._format(value, fmt, case, lang)
            if fmt == "sale_name":
                key, value = f"{key}.name", {lang_: value[lang_] for lang_ in LANGUAGES}
            elif fmt in _SUBKEY:
                key, value = f"{key}.{_SUBKEY[fmt]}", value[_SUBKEY[fmt]]
            hits.append(Hit(key=key, value=value, surface=surface))
            return surface

        try:
            text = textfmt.PLACEHOLDER_RE.sub(one, template)
        except (KeyError, TypeError, ValueError) as e:
            raise SpecError(f"cannot fill {template!r} ({lang}): {e}") from e
        if "{" in text or "}" in text:
            raise SpecError(f"unfilled placeholder left in {text!r}")
        return text, hits

    def _resolve(self, name: str, rec: Record | None) -> tuple[str, Any]:
        root, _, rest = name.partition(".")
        if root == "rec":
            if rec is None:
                raise SpecError(f"{{{name}}} outside a record")
            if rest in rec.aliases:
                src = rec.aliases[rest]
                return src, self.facts[src]
            if rest not in rec.values:
                raise SpecError(f"record has no field {rest!r}")
            return f"{rec.prefix}.{rest}", rec.values[rest]
        if root not in self.facts:
            raise SpecError(f"unknown fact {root!r}")
        value = self.facts[root]
        for part in rest.split(".") if rest else []:
            value = value[part]
        return name, value

    def _format(self, value: Any, fmt: str | None, case: str | None, lang: LanguageCode) -> str:
        if fmt is None and isinstance(value, int) and not isinstance(value, bool):
            return textfmt.group_digits(value, lang)
        if fmt in ("seconds", "minutes"):
            return small_unit(int(value), fmt[:-1], lang, case)
        if fmt == "date_full":
            return full_date(date.fromisoformat(str(value)), lang)
        spec = "{v" + (f":{fmt}" if fmt else "") + (f"@{case}" if case else "") + "}"
        return textfmt.render(spec, {"v": value}, lang, catalog=self.lookup)
