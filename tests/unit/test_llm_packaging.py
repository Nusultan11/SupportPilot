from pathlib import Path

import pytest

from supportpilot.llm.kaggle import package


def test_payload_replay_is_byte_identical_and_changed_inputs_cannot_overwrite(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    source = project / "src/supportpilot"
    source.mkdir(parents=True)
    (source / "example.py").write_text("x = 1\n", encoding="utf-8")
    (project / "configs").mkdir()
    (project / "configs/taxonomy.yaml").write_text("intents: {}\n", encoding="utf-8")
    (project / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    destination = tmp_path / "payload"
    package(project, {"validation": []}, {"tasks": []}, destination, owner="example", slug="test")
    original = (destination / "dataset/payload.zip").read_bytes()
    submission = (destination / "submission.json").read_bytes()
    package(project, {"validation": []}, {"tasks": []}, destination, owner="example", slug="test")
    assert (destination / "dataset/payload.zip").read_bytes() == original
    with pytest.raises(ValueError, match="immutable"):
        package(
            project, {"validation": [1]}, {"tasks": []}, destination, owner="example", slug="test"
        )
    assert (destination / "submission.json").read_bytes() == submission
    (destination / "dataset/payload.zip").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        package(
            project, {"validation": []}, {"tasks": []}, destination, owner="example", slug="test"
        )
