"""Usage policy (ADR-018): what each pipeline step may do with a source.

`allowed_use` in the registry is capped by the licence policy (schemas.LICENCE_MAX_USE);
here every step states the minimum level it needs, so e.g. an `eval_only` source can be
downloaded and analysed but never loaded for training.
"""

from __future__ import annotations

from enum import StrEnum

from supportpilot.common.schemas import USE_RANK, AllowedUse, SourceEntry, SourceStatus


class Purpose(StrEnum):
    ACQUIRE = "acquire"
    CONVERT = "convert"
    ANALYSIS = "analysis"
    EVALUATION = "evaluation"
    TRAINING = "training"


MIN_USE: dict[Purpose, AllowedUse] = {
    Purpose.ACQUIRE: AllowedUse.REFERENCE_ONLY,
    Purpose.CONVERT: AllowedUse.REFERENCE_ONLY,
    Purpose.ANALYSIS: AllowedUse.REFERENCE_ONLY,
    Purpose.EVALUATION: AllowedUse.EVAL_ONLY,
    Purpose.TRAINING: AllowedUse.TRAIN,
}

# Statuses that may be downloaded. Rejected sources stay registry-only.
ACQUIRABLE = frozenset({SourceStatus.SELECTED, SourceStatus.CANDIDATE, SourceStatus.UNAVAILABLE})


class PolicyError(Exception):
    """The registry forbids this use of the source."""


def check_use(entry: SourceEntry, purpose: Purpose) -> None:
    needed = MIN_USE[purpose]
    if USE_RANK[entry.allowed_use] < USE_RANK[needed]:
        raise PolicyError(
            f"{entry.source_id}: allowed_use={entry.allowed_use.value} does not permit "
            f"{purpose.value} (needs {needed.value} or more; licence {entry.licence})"
        )
    if purpose is Purpose.ACQUIRE and entry.status not in ACQUIRABLE:
        raise PolicyError(f"{entry.source_id}: status={entry.status.value}; not acquired")
