from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from supportpilot.common.config import find_project_root
from supportpilot.common.manifest import load_source_registry
from supportpilot.common.schemas import LICENCE_MAX_USE, AllowedUse, SourceStatus
from supportpilot.data.converters import CONVERTERS
from supportpilot.data.policy import PolicyError, Purpose, check_use
from tests.data_helpers import entry

REGISTRY = find_project_root(Path(__file__).parent) / "configs" / "sources.yaml"


def test_project_registry_is_valid_and_consistent() -> None:
    reg = load_source_registry(REGISTRY)
    selected = [s for s in reg.sources if s.status is SourceStatus.SELECTED]
    assert len(selected) >= 1
    for s in reg.sources:
        assert s.licence in LICENCE_MAX_USE
        assert s.decision.strip()
    for s in selected:
        assert s.fetch is not None
        assert s.source_id in CONVERTERS, f"selected source {s.source_id} has no converter"
    blocked = [s for s in reg.sources if s.licence in {"unknown", "conflicting"}]
    assert all(s.allowed_use is AllowedUse.BLOCKED for s in blocked)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"licence": "proprietary"}, "not in the licence policy"),
        ({"licence": "cc-by-nc-4.0", "allowed_use": "train"}, "exceeds"),
        ({"licence": "unknown", "allowed_use": "eval_only"}, "exceeds"),
        ({"licence": "conflicting", "allowed_use": "reference_only"}, "exceeds"),
        ({"allowed_use": "blocked", "licence": "unknown"}, "cannot be selected"),
        ({"fetch": None}, "needs a fetch spec"),
        ({"revision": "main"}, "40-char commit sha"),
        ({"revision": "12dd624"}, "40-char commit sha"),
        ({"languages": ["de"]}, "languages"),
        ({"fetch": {"repo_id": "no-slash", "include": ["*"]}}, "repo_id"),
        ({"fetch": {"repo_id": "a/b", "include": []}}, "include"),
        ({"status": "maybe"}, "status"),
    ],
)
def test_invalid_registry_entries(overrides: dict[str, object], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        entry(**overrides)


def test_nc_licence_allows_eval_only() -> None:
    e = entry(licence="cc-by-nc-4.0", allowed_use="eval_only")
    check_use(e, Purpose.EVALUATION)
    with pytest.raises(PolicyError):
        check_use(e, Purpose.TRAINING)


ALLOWED = {
    "train": {"acquire", "convert", "analysis", "evaluation", "training"},
    "eval_only": {"acquire", "convert", "analysis", "evaluation"},
    "reference_only": {"acquire", "convert", "analysis"},
    "blocked": set(),
}


@pytest.mark.parametrize("use", list(ALLOWED))
@pytest.mark.parametrize("purpose", list(Purpose))
def test_use_matrix(use: str, purpose: Purpose) -> None:
    e = entry(
        allowed_use=use,
        status="candidate" if use == "blocked" else "selected",
        licence="unknown" if use == "blocked" else "cc-by-4.0",
    )
    if purpose.value in ALLOWED[use]:
        check_use(e, purpose)
    else:
        with pytest.raises(PolicyError):
            check_use(e, purpose)


def test_rejected_source_is_never_acquired() -> None:
    e = entry(status="rejected", allowed_use="reference_only")
    check_use(e, Purpose.ANALYSIS)
    with pytest.raises(PolicyError, match="status=rejected"):
        check_use(e, Purpose.ACQUIRE)
