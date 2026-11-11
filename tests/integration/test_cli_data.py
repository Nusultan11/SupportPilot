"""`supportpilot data ...` end to end with a temporary registry and data dir (no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from supportpilot.cli import main
from supportpilot.data.convert import RECORDS, records_dir
from tests.data_helpers import entry, make_raw

CSV = "instruction,intent,category,tags,response\nhi,a,B,C,D\nyo,a,B,C,D\n"


@pytest.fixture
def setup(tmp_path: Path) -> tuple[list[str], Path]:
    data = tmp_path / "data"
    good = entry("bitext_retail_ecom")
    rejected = entry("synth_junk", status="rejected", allowed_use="reference_only")
    reg = {
        "schema_version": "sources/1",
        "sources": [e.model_dump(mode="json") for e in (good, rejected)],
    }
    (tmp_path / "sources.yaml").write_text(yaml.safe_dump(reg), encoding="utf-8")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "paths": {"data_dir": data.as_posix()},
                "data": {"registry": (tmp_path / "sources.yaml").as_posix()},
            }
        ),
        encoding="utf-8",
    )
    make_raw(data, good, {"r.csv": CSV})
    return ["--config", str(cfg), "data"], data


def _run(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, dict[str, Any]]:
    code = main(argv)
    out = capsys.readouterr().out
    return code, json.loads(out) if out.strip() else {}


def test_sources_convert_verify_cycle(
    setup: tuple[list[str], Path], capsys: pytest.CaptureFixture[str]
) -> None:
    base, data = setup
    code, out = _run(capsys, [*base, "sources"])
    assert code == 0
    state = {s["source_id"]: s for s in out["sources"]}
    assert state["bitext_retail_ecom"]["acquired"] is True
    assert state["bitext_retail_ecom"]["converted"] is False
    assert state["synth_junk"]["status"] == "rejected"

    code, out = _run(capsys, [*base, "convert", "--all"])
    assert code == 0
    assert out["converted"][0]["rows"] == 2
    code, out = _run(capsys, [*base, "convert", "bitext_retail_ecom"])
    assert out["converted"][0]["status"] == "skipped"

    code, out = _run(capsys, [*base, "verify", "--all"])
    assert code == 0
    assert out["verify"][0]["raw"]["ok"] is True
    assert out["verify"][0]["records"]["ok"] is True

    parquet = records_dir(data, entry("bitext_retail_ecom")) / RECORDS
    parquet.write_bytes(parquet.read_bytes()[:-10])
    code, out = _run(capsys, [*base, "verify", "bitext_retail_ecom"])
    assert code == 1
    assert out["verify"][0]["records"]["ok"] is False


def test_usage_and_policy_errors_exit_2(
    setup: tuple[list[str], Path], capsys: pytest.CaptureFixture[str]
) -> None:
    base, _ = setup
    assert _run(capsys, [*base, "acquire"])[0] == 2  # no ids and no --all
    assert _run(capsys, [*base, "acquire", "nope"])[0] == 2  # unknown id
    assert _run(capsys, [*base, "acquire", "synth_junk"])[0] == 2  # rejected -> PolicyError
    assert (
        _run(capsys, [*base, "sources", "--status", "rejected"])[1]["sources"][0]["source_id"]
        == "synth_junk"
    )
