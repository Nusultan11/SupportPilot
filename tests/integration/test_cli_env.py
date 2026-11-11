"""CLI end to end (backup cycle through `main`) and the environment report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from supportpilot.cli import main
from supportpilot.ops.envcheck import collect


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, object]]:
    code = main(list(argv))
    out = capsys.readouterr().out
    return code, json.loads(out) if out.strip() else {}


def test_cli_backup_cycle(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"backup:\n  root: {(tmp_path / 'bk').as_posix()}\n", encoding="utf-8")
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("Сәлем", encoding="utf-8")
    base = ["--config", str(cfg), "backup"]

    code, out = _run(capsys, *base, "create", str(src), "--name", "demo", "--kind", "data")
    assert code == 0
    assert out["created"] is True
    snap_id = out["manifest"]["snapshot_id"]  # type: ignore[index]
    ref = f"data/demo/{snap_id}"

    code, out = _run(capsys, *base, "list")
    assert code == 0
    assert out["snapshots"][0]["ref"] == ref  # type: ignore[index]

    assert _run(capsys, *base, "verify", ref)[0] == 0
    assert _run(capsys, *base, "verify", ref, "--quick")[0] == 0
    code, out = _run(capsys, *base, "restore", ref, "--target", str(tmp_path / "r"))
    assert code == 0
    assert (tmp_path / "r" / "a.txt").read_text(encoding="utf-8") == "Сәлем"
    assert _run(capsys, *base, "pin", ref)[1]["pinned"] is True
    code, out = _run(capsys, *base, "prune", "--keep", "1")
    assert out["dry_run"] is True
    assert out["would_delete"] == []

    # operation errors exit with 2 and never with a traceback
    assert _run(capsys, *base, "restore", ref, "--target", str(tmp_path / "r"))[0] == 2
    assert _run(capsys, *base, "verify", "data/demo/missing")[0] == 2


def test_cli_rejects_missing_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", str(tmp_path / "none.yaml"), "backup", "list"]) == 2
    assert "config error" in capsys.readouterr().err


def test_env_report_without_benchmark(tmp_path: Path) -> None:
    report = collect(tmp_path, gpu_benchmark=False)
    assert report.logical_cores is not None
    assert report.logical_cores >= 1
    assert report.ram_total_gib > 0
    assert report.python.startswith("3.12")
    if report.cuda_available:
        assert report.gpus
        assert report.gpus[0].total_vram_gib > 0


@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_matmul_benchmark(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    report = collect(tmp_path, gpu_benchmark=True)
    gpu = report.gpus[0]
    assert gpu.matmul_ok is True
    assert gpu.matmul_tflops_fp16 is not None
    assert gpu.matmul_tflops_fp16 > 0
