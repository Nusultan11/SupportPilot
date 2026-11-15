"""Local service composition; research generation is an explicit resource-costly option."""

import json
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager

from supportpilot.common.config import Settings
from supportpilot.common.hashing import sha256_file, sha256_text
from supportpilot.common.io import read_json
from supportpilot.data.taxonomy import load_taxonomy
from supportpilot.llm.backend import TransformersBackend
from supportpilot.llm.models import InferenceConfig
from supportpilot.retrieval import open_retriever
from supportpilot.routing.router import EncoderRouter
from supportpilot.service.orchestrator import SupportService
from supportpilot.service.store import Store


@contextmanager
def open_service(settings: Settings) -> Iterator[SupportService]:
    cfg = settings.serving
    taxonomy = load_taxonomy(settings.resolve(settings.stage4.taxonomy))
    seal = read_json(settings.root / "artifacts/reports/llm/selection.json")
    identity = seal.pop("identity")
    if sha256_text(json.dumps(seal, sort_keys=True)) != identity:
        raise ValueError("selection identity mismatch")
    router_path = settings.resolve(cfg.router)
    if seal["validation_reports"].get(str(router_path)) != sha256_file(router_path):
        raise ValueError("router differs from sealed selection")
    with ExitStack() as stack:
        store = Store(settings.resolve(cfg.database))
        router = EncoderRouter.load(router_path, taxonomy, device="cpu")
        retriever = stack.enter_context(open_retriever(settings)) if cfg.retrieval_enabled else None
        backend = None
        if cfg.generation_enabled:
            inference = InferenceConfig.model_validate(seal["configurations"]["adapter"])
            if (
                inference.adapter is None
                or sha256_file(inference.adapter / "adapter_manifest.json")
                != seal["adapter_manifests"]["adapter"]
            ):
                raise ValueError("adapter differs from sealed selection")
            backend = TransformersBackend(
                inference.model_copy(
                    update={
                        "quantization": "nf4",
                        "max_seconds": min(45, cfg.request_timeout_s),
                    }
                )
            )
            stack.callback(backend.close)
        service = SupportService(store, cfg, taxonomy, router, retriever, backend)
        stack.callback(service.close)
        yield service
