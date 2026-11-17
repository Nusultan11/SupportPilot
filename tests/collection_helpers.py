"""Tiny commissioned deliveries for tests (fixture text only, never corpus data)."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import yaml

from supportpilot.data.collection import (
    BRIEFS_FILE,
    DELIVERY_MANIFEST,
    ROW_COLUMNS,
    ROWS_FILE,
    CollectionPlan,
    briefs_csv,
    make_briefs,
)
from supportpilot.data.taxonomy import Taxonomy, load_taxonomy

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "commissioned_kz_2026"


def taxonomy() -> Taxonomy:
    return load_taxonomy(ROOT / "configs" / "taxonomy.yaml")


def plan(**overrides: Any) -> CollectionPlan:
    data = yaml.safe_load((ROOT / "configs" / "collection.yaml").read_text(encoding="utf-8"))
    data.update(
        {
            "scenarios_per_intent": 1,
            "writers_per_scenario": 2,
            "min_writers_per_slice": 2,
            "max_flagged_fraction": 0.5,
        }
    )
    data.update(overrides)
    return CollectionPlan.model_validate(data)


def contributor(cid: str, role: str = "writer", native: list[str] | None = None) -> dict[str, Any]:
    return {
        "contributor_id": cid,
        "role": role,
        "native": native or ["kk", "ru"],
        "fluent": ["en"],
        "country": "KZ",
        "contract_ref": f"contract-{cid}",
        "qualification": {"test_id": "screen-v1", "score": 0.9, "passed": True},
        "attestation": {
            "original_work": True,
            "no_llm": True,
            "no_machine_translation": True,
            "no_real_personal_data": True,
            "rights_assigned": True,
        },
    }


# (writer, brief, slice, lang, text)
ROWS: list[tuple[str, str, str, str, str]] = [
    ("w01", "sc-track_order-01", "ru", "ru", "Где мой заказ <order_id>, жду уже неделю"),
    ("w02", "sc-track_order-01", "ru", "ru", "Подскажите, когда приедет посылка, номер <order_id>"),
    ("w01", "sc-track_order-01", "kk", "kk", "Тапсырысым қайда жүр, нөмірі <order_id>"),
    ("w02", "sc-track_order-01", "kk", "kk", "Сәлеметсіз бе, сәлемдемем қашан келеді?"),
    ("w01", "sc-track_order-01", "ru_kk", "kk", "Сәлем, заказ где сейчас? Қашан келеді ма"),
    ("w02", "sc-track_order-01", "ru_kk", "ru", "Мой заказ ещё не пришёл, қашан болады ма"),
    ("w01", "sc-cancel_order-01", "ru", "ru", "Хочу отменить заказ, передумал покупать"),
    ("w02", "sc-cancel_order-01", "ru", "ru", "Можно отменить покупку пока не отправили?"),
]


def write_delivery(
    path: Path,
    p: CollectionPlan,
    rows: list[tuple[str, str, str, str, str]] | None = None,
    contributors: list[dict[str, Any]] | None = None,
    duration: int = 40,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "commissioned_delivery/1",
        "plan_id": p.plan_id,
        "batch_id": "batch-01",
        "vendor": "test vendor",
        "licence": "commissioned-assignment",
        "licence_evidence": "contract section 5 (assignment incl. ML training)",
        "delivered_at": "2026-11-20",
        "contributors": contributors
        or [contributor("w01"), contributor("w02"), contributor("r01", "reviewer")],
    }
    (path / DELIVERY_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    (path / BRIEFS_FILE).write_text(briefs_csv(make_briefs(p, taxonomy())), encoding="utf-8")
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(ROW_COLUMNS)
    for i, (writer, brief, sl, lang, text) in enumerate(rows or ROWS, start=1):
        w.writerow([f"r{i:04d}", writer, brief, sl, lang, text, duration])
    (path / ROWS_FILE).write_text(buf.getvalue(), encoding="utf-8")
    return path
