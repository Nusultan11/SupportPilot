"""Explicit fictional product specifications; no random or arithmetically scaled facts."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from supportpilot.common.schemas import LANGUAGES, DocFormat, DocType, Slug
from supportpilot.kb.spec import DocSpec, SectionSpec, Tri


class Attribute(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label: Tri
    unit: Tri
    question: Tri | None = None
    family: Slug


class ProfileGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sku_prefix: str = Field(pattern=r"^[A-Z]{2}$")
    category: Slug
    format: Literal["txt", "docx", "pdf", "json"]
    fields: list[Slug] = Field(min_length=1)
    missing: list[Slug] = Field(min_length=1)
    # id, native EN/RU/KK name, demo price, followed by explicitly authored field values.
    rows: list[list[str | int | float]] = Field(min_length=1)

    @model_validator(mode="after")
    def rectangular(self) -> Self:
        if len(set(self.fields)) != len(self.fields):
            raise ValueError("duplicate profile field")
        if any(len(row) != 5 + len(self.fields) for row in self.rows):
            raise ValueError("profile row must contain id/names/price and one value per field")
        if any(not isinstance(row[4], int) or row[4] <= 0 for row in self.rows):
            raise ValueError("demo prices must be positive integers")
        for row in self.rows:
            values = dict(zip(self.fields, row[5:], strict=True))
            if any(not isinstance(v, int | float) or not math.isfinite(v) for v in values.values()):
                raise ValueError("technical attribute values must be finite numbers")
            for prefix in ("operating", "storage"):
                low, high = values.get(f"{prefix}_min_c"), values.get(f"{prefix}_max_c")
                if isinstance(low, int | float) and isinstance(high, int | float) and low > high:
                    raise ValueError("inverted technical temperature interval")
        return self


class Profiles(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["technical_profiles/1"]
    origin: Literal["llm_generated"]
    author: str
    review_status: Literal["draft", "llm_reviewed"]
    licence_note: str
    attributes: dict[Slug, Attribute]
    groups: dict[Slug, ProfileGroup]
    sku_registry: dict[Slug, str]

    @model_validator(mode="after")
    def references(self) -> Self:
        ids: list[str] = []
        skus: list[str] = []
        for group in self.groups.values():
            if set(group.fields) & set(group.missing):
                raise ValueError("a missing attribute must not have a supplied value")
            if (set(group.fields) | set(group.missing)) - self.attributes.keys():
                raise ValueError("unknown profile attribute")
            ids.extend(str(row[0]) for row in group.rows)
            for row in group.rows:
                sku = self.sku_registry.get(str(row[0]), "")
                if not re.fullmatch(group.sku_prefix + r"-\d{4}", sku):
                    raise ValueError("every profile needs an explicitly registered SKU")
                skus.append(sku)
        if len(ids) != len(set(ids)) or len(skus) != len(set(skus)):
            raise ValueError("profile IDs and SKUs must be unique")
        if set(ids) != self.sku_registry.keys():
            raise ValueError("SKU registry has missing or unreferenced profile IDs")
        return self


def load_profiles(path: Path) -> Profiles:
    return Profiles.model_validate(yaml.safe_load(path.read_text("utf-8")))


def profile_items(profiles: Profiles) -> list[dict[str, Any]]:
    result = []
    for group_id, group in profiles.groups.items():
        for row in group.rows:
            result.append(
                {
                    "id": str(row[0]),
                    "group": group_id,
                    "sku": profiles.sku_registry[str(row[0])],
                    "name": Tri(en=str(row[1]), ru=str(row[2]), kk=str(row[3])),
                    "price": int(row[4]),
                    "category": group.category,
                    "format": group.format,
                    "values": dict(zip(group.fields, row[5:], strict=True)),
                }
            )
    return result


def profile_sources(
    profiles: Profiles,
    categories: dict[str, Tri],
    existing_skus: set[str],
) -> tuple[dict[str, Any], list[DocSpec]]:
    """Extend the store through separate catalogue documents, retaining the old catalogue bytes."""
    facts, documents = {}, []
    for item in profile_items(profiles):
        if item["sku"] in existing_skus or item["category"] not in categories:
            raise ValueError("profile SKU collides or its category is unknown")
        slug, name = item["id"], item["name"]
        prefix = f"technical.{slug}"
        facts[slug] = {
            "sku": item["sku"],
            "name": name.model_dump(),
            "price": item["price"],
            "category": categories[item["category"]].model_dump(),
            **item["values"],
        }
        intro = Tri(
            en=f"Item {{{prefix}.name}}, SKU {{{prefix}.sku}}, category {{{prefix}.category}}, "
            f"demo catalogue price {{{prefix}.price:kzt}}.",
            ru=f"Товар {{{prefix}.name}}, артикул {{{prefix}.sku}}, "
            f"категория {{{prefix}.category}}, "
            f"цена демонстрационного каталога {{{prefix}.price:kzt}}.",
            kk=f"Тауар {{{prefix}.name}}, артикулы {{{prefix}.sku}}, санаты {{{prefix}.category}}, "
            f"демонстрациялық каталогтағы бағасы {{{prefix}.price:kzt}}.",
        )
        sections = []
        for key in item["values"]:
            attribute = profiles.attributes[key]
            paragraphs = {}
            for lang in LANGUAGES:
                label, unit = attribute.label.get(lang), attribute.unit.get(lang)
                title = name.get(lang)
                sentences = {
                    "en": f"The technical specification for {title}: "
                    f"{label} — {{{prefix}.{key}}} {unit}.",
                    "ru": f"Техническая характеристика товара «{title}»: "
                    f"{label} — {{{prefix}.{key}}} {unit}.",
                    "kk": f"«{title}» тауарының техникалық сипаттамасы: "
                    f"{label} — {{{prefix}.{key}}} {unit}.",
                }
                paragraphs[lang] = sentences[lang]
            sections.append(
                SectionSpec(id=key, heading=attribute.label, paragraphs=[Tri(**paragraphs)])
            )
        fmt = DocFormat(item["format"])
        documents.append(
            DocSpec(
                doc_id="technical_" + slug,
                doc_type=DocType.FAQ if fmt == DocFormat.JSON else DocType.CATALOG,
                format=fmt,
                title=Tri(
                    en=f"Technical sheet: {name.en}",
                    ru=f"Технические характеристики: {name.ru}",
                    kk=f"Техникалық сипаттамалар: {name.kk}",
                ),
                intro=[intro],
                sections=sections,
            )
        )
    return {"technical": facts}, documents
