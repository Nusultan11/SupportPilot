"""Compile explicit subject/attribute needs, not keyword-based relevance judgements."""

from __future__ import annotations

from collections import Counter
from typing import Any

import yaml

from supportpilot.common.config import Settings
from supportpilot.common.hashing import sha256_file
from supportpilot.common.io import atomic_write_text
from supportpilot.common.schemas import LANGUAGES
from supportpilot.kb.profiles import load_profiles, profile_items
from supportpilot.retrieval.scale_data import ScaleNeed, ScaleSource

MARKER = "# Compiled from explicit technical_profiles/1 facts and native LLM-authored templates.\n"


def profile_questions(name: Any, attribute: Any, style: int) -> dict[str, str]:
    """Wording and translations are variants, never extra independent needs."""
    words = attribute.question or attribute.label
    result: dict[str, str] = {}
    for lang in LANGUAGES:
        item, label, unit = name.get(lang), words.get(lang), attribute.unit.get(lang)
        templates = {
            "en": [
                f"For {item}, what is the specified {label.lower()} in {unit}?",
                f"I am comparing specifications. Find the {label.lower()} for {item}, in {unit}.",
                f"{item}: {label.lower()}, {unit}?",
                f"The cards look similar. What {label.lower()} is listed for {item} ({unit})?",
                f"Check the specifcation of {item} for {label.lower()} ({unit}), please.",
            ],
            "ru": [
                f"Для «{item}» какое значение указано для показателя «{label}» ({unit})?",
                f"Сравниваю характеристики. Нужен показатель «{label}» у «{item}», в {unit}.",
                f"«{item}»: {label.lower()}, {unit}?",
                f"Карточки похожи. Уточните показатель «{label}» именно для «{item}» ({unit}).",
                f"Проверьте характиристику «{item}»: интересует «{label}» ({unit}).",
            ],
            "kk": [
                f"«{item}» тауары үшін «{label}» көрсеткішінің мәні қандай ({unit})?",
                f"Сипаттамаларды салыстырып отырмын. «{item}» үшін «{label}» мәні қажет ({unit}).",
                f"«{item}»: {label.lower()}, {unit}?",
                f"Карточкалар ұқсас. «{item}» үшін нақты «{label}» көрсеткішін айтыңыз ({unit}).",
                f"«{item}» сипатамасынан «{label}» мәнін тексеріңізші ({unit}).",
            ],
        }
        result[lang] = templates[lang][style % 5]
    return result


def author_profiles(settings: Settings) -> dict[str, Any]:
    path = settings.resolve(settings.kb.source_root) / "technical_profiles.yaml"
    profiles, source_sha = load_profiles(path), sha256_file(path)
    grouped: dict[str, list[ScaleNeed]] = {key: [] for key in profiles.groups}
    ordinal = 0
    for index, item in enumerate(profile_items(profiles)):
        slug, name, group_id = item["id"], item["name"], item["group"]
        group, doc = profiles.groups[group_id], "technical_" + slug
        for field_index, key in enumerate(item["values"]):
            attribute, style = profiles.attributes[key], ordinal % 5
            other = group.fields[(field_index + 1) % len(group.fields)]
            payload: dict[str, Any] = {
                "id": f"catalog_{slug}_{key}",
                "family_id": "profile_" + attribute.family,
                "need_key": f"technical.{slug}.{key}|current",
                "category": "hard" if style in (3, 4) or "min" in key or "max" in key else "simple",
                **profile_questions(name, attribute, style),
                "relevant": [
                    {"section": f"{doc}.{key}", "grade": 3},
                    {"section": f"{doc}.{other}", "grade": 0},
                ],
                "required": [{"fact": f"technical.{slug}.{key}", "sections": [f"{doc}.{key}"]}],
                "rationale": f"Authored target technical.{slug}.{key}; neighbouring {other} "
                f"does not answer this predicate. Profile SHA256 {source_sha}.",
                "cross_lingual": ordinal % 11 == 0,
                "tags": ["profile_derived", "domain_" + group_id],
            }
            if style == 4:
                payload["tags"].append("typo")
            if index % 10 == 9:
                payload["tags"].append("rare_document")
            if ordinal % 7 == 0:
                if ordinal % 2:
                    payload["mix_ru"] = f"У «{name.ru}» показатель «{attribute.label.ru}» "
                    payload["mix_ru"] += f"({attribute.unit.ru}) нақтылап беріңізші, осы үлгі үшін."
                else:
                    payload["mix_kk"] = f"«{name.kk}» үшін «{attribute.label.kk}» "
                    payload["mix_kk"] += (
                        f"({attribute.unit.kk}) керек, уточните по этой модели, пожалуйста."
                    )
            grouped[group_id].append(ScaleNeed.model_validate(payload))
            ordinal += 1
        for key in group.missing:
            attribute = profiles.attributes[key]
            relevance = [{"section": f"{doc}.{group.fields[0]}", "grade": 0}]
            if key == "repair_parts_price":
                relevance.append({"section": "warranty_service.diagnostics", "grade": 0})
            grouped[group_id].append(
                ScaleNeed.model_validate(
                    {
                        "id": f"catalog_missing_{slug}_{key}",
                        "family_id": "profile_missing_" + attribute.family,
                        "need_key": f"technical.{slug}.{key}|not_stated",
                        "category": "unanswerable",
                        **profile_questions(name, attribute, 0),
                        "relevant": relevance,
                        "required": [],
                        "negative_reason": "partial_evidence"
                        if key == "repair_parts_price"
                        else "missing_fact",
                        "rationale": f"The source for {slug} does not state {key}. "
                        "Dimensions, the retail "
                        "price or a diagnostic fee cannot supply it. "
                        f"Profile SHA256 {source_sha}.",
                        "tags": ["profile_derived", "hard_negative", "domain_" + group_id],
                    }
                )
            )
        if index % 2 == 0:
            grouped[group_id].append(
                ScaleNeed.model_validate(
                    {
                        "id": f"catalog_pickup_mass_{slug}",
                        "family_id": "profile_pickup_weight_comparison",
                        "need_key": f"technical.{slug}.mass_g|pickup_parcel_weight_limit",
                        "category": "multi_fragment",
                        "en": f"What does {name.en} weigh, "
                        "and what parcel weight limit do pickup points state?",
                        "ru": f"Какая масса у «{name.ru}» "
                        "и какой вес посылки допускают пункты самовывоза?",
                        "kk": f"«{name.kk}» салмағы қандай "
                        "және беру пункттері сәлемдеменің қандай салмағына рұқсат береді?",
                        "relevant": [
                            {"section": f"{doc}.mass_g", "grade": 3},
                            {"section": "delivery_faq.pickup_weight", "grade": 3},
                        ],
                        "required": [
                            {"fact": f"technical.{slug}.mass_g", "sections": [f"{doc}.mass_g"]},
                            {
                                "fact": "pickup_max_weight_kg",
                                "sections": ["delivery_faq.pickup_weight"],
                            },
                        ],
                        "rationale": "The two requested quantities require two documents. "
                        "Item mass is not "
                        "packed-parcel mass and does not establish live pickup availability.",
                        "tags": [
                            "profile_derived",
                            "multi_document",
                            "multi_fragment",
                            "domain_" + group_id,
                        ],
                    }
                )
            )
        if index % 10 == 0:
            grouped[group_id].append(
                ScaleNeed.model_validate(
                    {
                        "id": f"catalog_prepublication_mass_{slug}",
                        "family_id": "profile_physical_mass",
                        "need_key": f"technical.{slug}.mass_g|2026_10_15_before_document_validity",
                        "category": "unanswerable",
                        "as_of": "2026-10-15",
                        "en": f"According to the instructions valid on 15 October 2026, "
                        f"what was the stated mass of {name.en}?",
                        "ru": f"По инструкциям, действовавшим 15 октября 2026 года, "
                        f"какая масса была указана для «{name.ru}»?",
                        "kk": f"2026 жылғы 15 қазанда қолданылған нұсқаулықтар бойынша "
                        f"«{name.kk}» үшін қандай салмақ көрсетілген еді?",
                        "relevant": [{"section": f"{doc}.mass_g", "grade": 0}],
                        "required": [],
                        "negative_reason": "expired",
                        "rationale": "The technical source starts on 1 November 2026. "
                        "Its current mass cannot establish a fact at the requested earlier date.",
                        "tags": [
                            "profile_derived",
                            "temporal",
                            "outside_validity",
                            "hard_negative",
                        ],
                    }
                )
            )
    out = settings.resolve(settings.stage4.authored_dir) / "retrieval_scale"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for key, items in grouped.items():
        source = ScaleSource(
            schema_version="retrieval_scale_source/1",
            origin="llm_generated",
            review_status="draft",
            author="Codex (LLM); explicit profiles and native templates compiled deterministically",
            allowed_use="eval_only",
            licence_note=profiles.licence_note,
            items=items,
        )
        target = out / f"profile_{key}.yaml"
        content = MARKER + yaml.safe_dump(
            source.model_dump(mode="json"), allow_unicode=True, sort_keys=False, width=100
        )
        if target.exists() and not target.read_text("utf-8").startswith(MARKER):
            raise ValueError("refusing to overwrite a manually authored query source")
        if not target.exists() or target.read_text("utf-8") != content:
            atomic_write_text(target, content)
            written.append(target.name)
    counts = Counter(need.category for batch in grouped.values() for need in batch)
    return {
        "source_sha256": source_sha,
        "products": len(profile_items(profiles)),
        "atomic_attributes": ordinal,
        "needs": sum(counts.values()),
        "categories": dict(counts),
        "written": written,
        "review_status": "draft",
        "human_review": "NOT_PERFORMED",
        "independence_unit": "subject/requested predicate/conditions; templates share a family",
    }
