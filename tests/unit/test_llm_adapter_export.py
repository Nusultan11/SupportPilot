import zipfile
from pathlib import Path

import pytest

from supportpilot.common.io import read_json
from supportpilot.llm.adapters import (
    export_inference_adapter,
    validate_adapter,
    write_adapter_manifest,
)
from supportpilot.llm.kaggle import package
from supportpilot.llm.models import ModelSpec


def test_inference_export_preserves_weights_and_card_without_training_pickle(
    tmp_path: Path,
) -> None:
    model = ModelSpec(repo_id="example/model", revision="a" * 40, licence="apache-2.0")
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(b"verified-adapter-test")
    (source / "adapter_config.json").write_text("{}", encoding="utf-8")
    (source / "README.md").write_text("Original card", encoding="utf-8")
    (source / "training_args.bin").write_bytes(b"opaque-training-state-not-for-inference")
    write_adapter_manifest(source, model, "training-id")
    with pytest.raises(ValueError, match="inside the source"):
        export_inference_adapter(source, source / "nested", model)
    assert not (source / "nested").exists()
    target = tmp_path / "export"
    export_inference_adapter(source, target, model, model_card="Measured model card\n")
    validate_adapter(target, model)
    assert (target / "adapter_model.safetensors").read_bytes() == (
        source / "adapter_model.safetensors"
    ).read_bytes()
    assert not (target / "training_args.bin").exists()
    assert (target / "README.md").read_text("utf-8") == "Measured model card\n"
    assert (source / "README.md").read_text("utf-8") == "Original card"
    assert (
        "training_args.bin" in read_json(target / "adapter_manifest.json")["excluded_source_files"]
    )
    export_inference_adapter(source, target, model, model_card="Measured model card\n")
    with pytest.raises(ValueError, match="immutable"):
        export_inference_adapter(source, target, model, model_card="Different card")
    (target / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        export_inference_adapter(source, target, model, model_card="Measured model card\n")


def test_clean_export_can_enter_private_payload_but_unknown_files_are_rejected(
    tmp_path: Path,
) -> None:
    model = ModelSpec(repo_id="example/model", revision="a" * 40, licence="apache-2.0")
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(b"weights")
    (source / "README.md").write_text("Card", encoding="utf-8")
    (source / "training_args.bin").write_bytes(b"training-only")
    write_adapter_manifest(source, model, "training-id")
    adapter = export_inference_adapter(source, tmp_path / "export", model)
    project = tmp_path / "project"
    (project / "src/supportpilot").mkdir(parents=True)
    (project / "src/supportpilot/example.py").write_text("x = 1\n", encoding="utf-8")
    (project / "configs").mkdir()
    (project / "configs/taxonomy.yaml").write_text("intents: {}\n", encoding="utf-8")
    (project / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    destination = package(
        project,
        {},
        {"tasks": []},
        tmp_path / "payload",
        owner="example",
        slug="test",
        adapters={"selected": adapter},
    )
    with zipfile.ZipFile(destination / "dataset/payload.zip") as archive:
        assert "adapters/selected/README.md" in archive.namelist()
        assert not any(name.endswith("training_args.bin") for name in archive.namelist())
    (source / "unexpected.bin").write_bytes(b"unsupported")
    write_adapter_manifest(source, model, "training-id")
    with pytest.raises(ValueError, match="unexpected"):
        export_inference_adapter(source, tmp_path / "rejected", model)
    assert not (tmp_path / "rejected").exists()
