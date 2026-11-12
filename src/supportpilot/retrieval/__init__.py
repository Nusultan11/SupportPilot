"""Version-bound evidence retrieval. No LLM generation or business API calls."""

from supportpilot.retrieval.models import (
    RetrievalContext,
    ScoredChunk,
    SearchFilters,
    SearchOptions,
)
from supportpilot.retrieval.search import Retriever, open_retriever

__all__ = [
    "RetrievalContext",
    "Retriever",
    "ScoredChunk",
    "SearchFilters",
    "SearchOptions",
    "open_retriever",
]
