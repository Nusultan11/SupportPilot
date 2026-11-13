from __future__ import annotations

from pathlib import Path

import pytest

from supportpilot.common.config import Settings
from supportpilot.common.io import write_json
from supportpilot.llm.heldout import export_heldout
from supportpilot.llm.models import InferenceConfig, ModelSpec
from supportpilot.llm.selection import choose, require_seal, seal


def test_final_selection_cannot_silently_change(tmp_path: Path) -> None:
    model = ModelSpec(repo_id="example/model", revision="a" * 40, licence="apache-2.0")
    report = tmp_path / "validation.json"
    write_json(report, {"observations": 3})
    path = tmp_path / "selection.json"
    initial = seal(
        path,
        {"base": InferenceConfig(model=model)},
        prompt_modes={"base": "schema"},
        decision={"selected": "base"},
        validation_reports=[report],
    )
    assert require_seal(path)["identity"] == initial["identity"]
    with pytest.raises(ValueError, match="sealed"):
        seal(
            path,
            {"base": InferenceConfig(model=model, max_new_tokens=256)},
            prompt_modes={"base": "schema"},
            decision={"selected": "base"},
            validation_reports=[report],
        )
    write_json(report, {"observations": 4})
    with pytest.raises(ValueError, match="changed after selection"):
        require_seal(path)


def test_heldout_cannot_be_opened_without_a_seal(tmp_path: Path, settings: Settings) -> None:
    # Even the directories do not exist; the selection guard must fail first.
    with pytest.raises(FileNotFoundError, match="selection"):
        export_heldout(
            tmp_path / "selection.json",
            tmp_path / "test",
            tmp_path / "gold",
            settings,
            tmp_path / "out",
            train=[],
        )


def test_selection_ties_use_measured_memory_then_latency() -> None:
    settings = {
        "selection": {
            "language_weights": {"kk": 1.0},
            "metric_weights": {"schema_validity": 1.0},
            "kk_guard_max_regression": 0.02,
        }
    }

    def summary(memory: int | None, latency: float) -> dict[str, object]:
        return {
            "kk": {
                "n": 3,
                "metrics": {
                    "schema_validity": {"value": 1.0},
                    "language_compliance": {"value": 1.0},
                },
                "resources": {"peak_vram_bytes": memory},
                "latency_p50": latency,
            }
        }

    result = choose({"base": summary(200, 1), "smaller": summary(100, 2)}, settings, control="base")
    assert result["selected"] == "smaller"
    result = choose({"base": summary(100, 2), "faster": summary(100, 1)}, settings, control="base")
    assert result["selected"] == "faster"
    result = choose(
        {"base": summary(100, 2), "unknown": summary(None, 1)}, settings, control="base"
    )
    assert result["selected"] == "base"
