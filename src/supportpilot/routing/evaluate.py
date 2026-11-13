"""Train/select an encoder routing head using train/validation only."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from supportpilot.common.hashing import sha256_text
from supportpilot.common.io import read_json, write_json
from supportpilot.common.schemas import AssistantResponse
from supportpilot.data.taxonomy import load_taxonomy
from supportpilot.llm.evaluate import summarize
from supportpilot.retrieval.encode import Transformer
from supportpilot.retrieval.models import ModelSpec
from supportpilot.routing.router import fit_head, policy_decision


def encoder_experiment(
    inputs: dict[str, Any],
    encoder: ModelSpec,
    taxonomy: Path,
    output: Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    tax = load_taxonomy(taxonomy)
    # One classifier feature vector per distinct training utterance, not per task augmentation.
    distinct = {(r["user_text"], r["intent"]): r for r in inputs["train"]}
    train = list(distinct.values())
    validation = inputs["validation"]
    identity = sha256_text(
        str(encoder.model_dump())
        + str([(r["example_id"], r["user_text"]) for r in train + validation])
    )
    cache = output / "features.npz"
    binding = output / "binding.json"
    model = Transformer(encoder, device=device)
    try:
        if cache.exists():
            if read_json(binding)["identity"] != identity:
                raise ValueError("routing feature cache mismatch")
            with np.load(cache, allow_pickle=False) as data:
                xtrain, xval = data["train"], data["validation"]
        else:
            started = time.perf_counter()
            xtrain = model.encode([r["user_text"] for r in train], query=True)
            xval = model.encode([r["user_text"] for r in validation], query=True)
            np.savez_compressed(cache, train=xtrain, validation=xval)
            write_json(
                binding,
                {
                    "identity": identity,
                    "encoder": encoder.model_dump(mode="json"),
                    "train": len(train),
                    "validation": len(validation),
                    "encoding_seconds": time.perf_counter() - started,
                },
            )
        results: dict[str, Any] = {}
        previous = output / "development.json"
        if previous.exists():
            old = read_json(previous)
            archive = output / "attempts" / sha256_text(str(old))[:20] / "head_scope.json"
            if not archive.exists():
                write_json(archive, old)
        model.load()
        warm_timings = []
        for source in validation:
            begin = time.perf_counter()
            model.encode([source["user_text"]], query=True)
            warm_timings.append(time.perf_counter() - begin)
        write_json(
            output / "warm_encoder_latency.json",
            {
                "scope": "single-request warm encoding; head timed separately",
                "load_seconds": model.load_seconds,
                "device": device,
                "observations": [
                    {"example_id": r["example_id"], "seconds": t}
                    for r, t in zip(validation, warm_timings, strict=True)
                ],
            },
        )
        for c in (0.1, 1.0, 10.0):
            head = fit_head(xtrain, [r["intent"] for r in train], regularization=c)
            rows = []
            for source, vector, encoding_seconds in zip(
                validation, xval, warm_timings, strict=True
            ):
                begin = time.perf_counter()
                logits = np.asarray(head["weights"]) @ vector + np.asarray(head["bias"])
                probabilities = np.exp(logits - logits.max())
                probabilities /= probabilities.sum()
                index = int(probabilities.argmax())
                route = policy_decision(
                    head["labels"][index],
                    float(probabilities[index]),
                    source["user_text"],
                    {},
                    source["lang"],
                    source["code_switch"],
                    tax,
                )
                target = AssistantResponse.model_validate(source["target"])
                rows.append(
                    {
                        "example_id": source["example_id"],
                        "scenario_id": source["scenario_id"],
                        "family_id": source["family_id"],
                        "lang": source["lang"],
                        "code_switch": source["code_switch"],
                        "task": source["task"],
                        "expected_intent": target.intent,
                        "predicted_intent": route.intent,
                        "intent_accuracy": float(route.intent == target.intent),
                        "action_accuracy": float(route.action == target.action),
                        "escalation_accuracy": float(route.escalate == target.needs_human),
                        "seconds": time.perf_counter() - begin + encoding_seconds,
                    }
                )
            key = "c" + str(c)
            results[key] = {"head": head, "summary": summarize(rows), "observations": rows}
        # Macro-language validation macro-F1; no test/gold access here.
        selected = max(
            results,
            key=lambda k: (
                sum(
                    results[k]["summary"][lang]["intent_macro_f1_35"] for lang in ("en", "ru", "kk")
                )
                / 3
            ),
        )
        report = {
            "identity": identity,
            "selected": selected,
            "results": results,
            "selection_scope": "validation_only",
            "encoder": encoder.model_dump(mode="json"),
        }
        write_json(output / "development.json", report)
        write_json(
            output / "router.json",
            {
                "schema_version": "encoder_router/1",
                "encoder": encoder.model_dump(mode="json"),
                "head": results[selected]["head"],
            },
        )
        return report
    finally:
        model.close()
