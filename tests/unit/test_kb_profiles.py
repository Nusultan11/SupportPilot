from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from supportpilot.kb.profiles import Profiles, load_profiles, profile_items, profile_sources
from supportpilot.kb.spec import KBSources, fact_registry, load_sources

SOURCE = Path(__file__).resolve().parents[2] / "kb_sources/demo_store"


def test_explicit_catalogue_has_atomic_facts_and_reorder_stable_skus() -> None:
    profiles = load_profiles(SOURCE / "technical_profiles.yaml")
    items = {item["id"]: item for item in profile_items(profiles)}
    assert len(items) == 200
    assert sum(len(item["values"]) for item in items.values()) == 2600
    assert len({item["sku"] for item in items.values()}) == 200
    payload = profiles.model_dump(mode="json")
    for group in payload["groups"].values():
        group["rows"].reverse()
    assert {item["id"]: item for item in profile_items(Profiles.model_validate(payload))} == items


def test_recursive_registry_keeps_legacy_facts_and_places_profile_leaves() -> None:
    sources = load_sources(SOURCE)
    registry = fact_registry(sources)
    legacy = KBSources(
        root=sources.root,
        facts={k: v for k, v in sources.facts.items() if k != "technical"},
        catalog=sources.catalog,
        docs=sources.docs,
    )
    assert all(registry[key] == value for key, value in fact_registry(legacy).items())
    assert registry["technical.compact_keyboard.mass_g"] == 620
    assert registry["technical.car_usb_charger.operating_min_c"] == -10
    assert registry["technical.gan_charger.output_a"] == 3.25
    assert registry["payment_retry_minutes"] == 15
    assert registry["archive_payment_retry_minutes"] == 10


def test_profile_source_rejects_unknown_skus_and_nonfinite_values() -> None:
    profiles = load_profiles(SOURCE / "technical_profiles.yaml")
    payload = profiles.model_dump(mode="json")
    payload["sku_registry"].pop("compact_keyboard")
    with pytest.raises(ValidationError, match="registered SKU"):
        Profiles.model_validate(payload)
    payload = profiles.model_dump(mode="json")
    payload["groups"]["peripherals"]["rows"][0][5] = float("nan")
    with pytest.raises(ValidationError, match="finite"):
        Profiles.model_validate(payload)
    sources = load_sources(SOURCE)
    with pytest.raises(ValueError, match="collides"):
        profile_sources(
            profiles, sources.catalog.categories, {profiles.sku_registry["compact_keyboard"]}
        )


def test_authoring_is_idempotent_and_preserves_manually_owned_files(tmp_path: Path) -> None:
    from supportpilot.common.config import Settings
    from supportpilot.retrieval.scale_author import author_profiles
    from supportpilot.retrieval.scale_data import load_needs

    source = tmp_path / "kb_sources/demo_store"
    source.mkdir(parents=True)
    shutil.copyfile(SOURCE / "technical_profiles.yaml", source / "technical_profiles.yaml")
    settings = Settings(root=tmp_path)
    first = author_profiles(settings)
    out = tmp_path / "authored/retrieval_scale"
    before = {p.name: p.read_bytes() for p in out.glob("*.yaml")}
    assert first["written"]
    assert author_profiles(settings)["written"] == []
    assert {p.name: p.read_bytes() for p in out.glob("*.yaml")} == before
    _, needs = load_needs(out)
    assert needs["catalog_compact_keyboard_mass_g"].required[0].fact == (
        "technical.compact_keyboard.mass_g"
    )
    owned = out / "profile_peripherals.yaml"
    owned.write_text("manually authored source\n", "utf-8")
    with pytest.raises(ValueError, match="manually authored"):
        author_profiles(settings)
    assert owned.read_text("utf-8") == "manually authored source\n"
