from supportpilot.llm.results import execution_tags


def test_remote_provenance_does_not_claim_the_importers_commit_or_environment() -> None:
    tags = execution_tags(
        {"python": "3.13.16", "platform": "Linux"},
        {"code_commit": "local-commit", "python": "3.12.3", "platform": "Windows"},
        "manifest-digest",
    )
    assert tags["python"] == "3.13.16"
    assert tags["platform"] == "Linux"
    assert tags["code_commit"] == "UNKNOWN_REMOTE_COMMIT"
    assert tags["importer_code_commit"] == "local-commit"
    assert tags["importer_python"] == "3.12.3"
    assert tags["source_manifest_sha256"] == "manifest-digest"
    assert execution_tags({}, {}, "digest")["python"] == "UNKNOWN"
