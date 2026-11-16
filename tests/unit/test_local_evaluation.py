"""Fresh experiment identity and isolation, not linguistic/semantic quality."""

from pathlib import Path

import pytest
import yaml

from supportpilot.common.config import Settings
from supportpilot.common.io import read_json, write_json
from supportpilot.service.local_evaluation import freeze, load_rows, root, select, verified

ROOT = Path(__file__).resolve().parents[2]


def test_new_draft_has_all_classes_and_separate_declared_families() -> None:
    rows = load_rows(ROOT / "authored/system/local_quality.yaml")
    assert len(rows) == 70
    assert len({r["family_id"] for r in rows}) == 70
    assert all(len(r["texts"]) == 4 for r in rows)


@pytest.fixture
def experiment(tmp_path: Path) -> Settings:
    settings = Settings(root=tmp_path)
    for name in (
        "authored/system/local_quality.yaml",
        "docs/reviews/stage9_local_protocol.md",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    for name in (
        "uv.lock",
        "configs/base.yaml",
        "configs/taxonomy.yaml",
        "configs/sources.yaml",
        "artifacts/reports/llm/selection.json",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    for variant in ("control", "balanced"):
        write_json(root(settings) / "heads" / (variant + ".json"), {"fixture": variant})
    freeze(settings)
    return settings


def test_freeze_rejects_changed_labels_heads_and_content(experiment: Settings) -> None:
    path = root(experiment) / "local_freeze.json"
    frozen = read_json(path)
    assert frozen["authorship"].startswith("LLM_AUTHORED_DRAFT")
    assert frozen["semantic_independence"] == "NOT_CERTIFIED"
    frozen["rows"][0]["intent"] = "greeting"
    write_json(path, frozen)
    with pytest.raises(ValueError, match="changed freeze"):
        verified(experiment)


def test_selection_rejects_a_consumed_decision(experiment: Settings) -> None:
    write_json(root(experiment) / "local_selection.json", {"fixture": "consumed"})
    with pytest.raises(ValueError, match="already sealed"):
        select(experiment, Path("not-read"), Path("not-read"))


def test_duplicate_family_is_not_independence(tmp_path: Path) -> None:
    data = yaml.safe_load((ROOT / "authored/system/local_quality.yaml").read_text("utf-8"))
    data["rows"][1][2] = data["rows"][0][2]
    path = tmp_path / "inputs.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate family"):
        load_rows(path)
