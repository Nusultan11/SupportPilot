"""Portable source-backed experiment runner; identical code runs locally and on Kaggle."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from supportpilot.common.hashing import sha256_file, sha256_text
from supportpilot.common.io import read_json, write_json
from supportpilot.data.langid import LanguageIdentifier
from supportpilot.llm.backend import TransformersBackend
from supportpilot.llm.evaluate import score_output, summarize
from supportpilot.llm.models import InferenceConfig, TrainingConfig
from supportpilot.llm.prompts import PromptMode, prepare_messages
from supportpilot.llm.provenance import runtime_identity
from supportpilot.llm.train import train_adapter


def run_inference(
    rows: list[dict[str, Any]],
    config: InferenceConfig,
    output: Path,
    *,
    prompt_mode: PromptMode = "schema",
    demonstrations: list[list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    binding = {
        "rows": rows,
        "config": config.model_dump(mode="json"),
        "prompt_mode": prompt_mode,
        "demonstrations": demonstrations or [],
        "runtime": runtime_identity(),
    }
    identity = sha256_text(json.dumps(binding, sort_keys=True, ensure_ascii=False))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "binding.json"
    if path.exists() and read_json(path)["identity"] != identity:
        raise ValueError("inference resume input/config mismatch")
    write_json(
        path,
        {
            "identity": identity,
            "config": config.model_dump(mode="json"),
            "prompt_mode": prompt_mode,
        },
    )
    predictions = output / "predictions.jsonl"
    observations = []
    if predictions.exists():
        observations = [json.loads(line) for line in predictions.read_text("utf-8").splitlines()]
        if [r["example_id"] for r in observations] != [
            r["example_id"] for r in rows[: len(observations)]
        ]:
            raise ValueError("prediction checkpoint is not an input prefix")
    backend = TransformersBackend(config)
    lid = LanguageIdentifier()
    tokenization = {}
    for lang in ("en", "ru", "kk"):
        texts = [str(r.get("user_text", "")) for r in rows if r["lang"] == lang]
        words = sum(len(text.split()) for text in texts)
        tokens = sum(
            len(backend.tokenizer.encode(text, add_special_tokens=False)) for text in texts
        )
        tokenization[lang] = {
            "words": words,
            "tokens": tokens,
            "tokens_per_word": tokens / words if words else None,
        }
    try:
        with predictions.open("a", encoding="utf-8", newline="\n") as handle:
            for row in rows[len(observations) :]:
                raw = None
                problem = None
                observation: dict[str, Any] = {}
                try:
                    messages = prepare_messages(
                        row["messages"][:-1],
                        mode=prompt_mode,
                        demonstrations=demonstrations or [],
                        language=row["lang"],
                    )
                    raw = backend.generate(messages)
                    observation = dict(backend.last_observation)
                except Exception as exc:
                    problem = f"{type(exc).__name__}:{exc}"
                result = {**score_output(row, raw, lid), **observation, "execution_error": problem}
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                observations.append(result)
        summary = summarize(observations)
        write_json(output / "summary.json", summary)
        return {
            "identity": identity,
            "count": len(observations),
            "summary": summary,
            "load_seconds": backend.load_seconds,
            "tokenization": tokenization,
            "status": "FINISHED",
        }
    finally:
        backend.close()


def execute_plan(plan: dict[str, Any], inputs: dict[str, Any], output: Path) -> None:
    """Keep every failed run; a runner failure does not erase earlier observations."""
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "plan.json", plan)
    if plan.get("workers"):
        # Two independent experiment chains use the two provisioned T4 devices.
        # A subprocess avoids CUDA fork state and Kaggle's non-importable __main__.
        input_path = output.parent / "worker_inputs.json"
        write_json(input_path, inputs)
        processes = []
        for index, worker in enumerate(plan["workers"]):
            worker_plan = output.parent / f"worker_plan_{index}.json"
            write_json(worker_plan, worker)
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
            environment["CUDA_VISIBLE_DEVICES"] = str(index)
            processes.append(
                subprocess.Popen(  # noqa: S603 - fixed module/argv, no shell
                    [
                        sys.executable,
                        "-m",
                        "supportpilot.llm.experiments",
                        "--inputs",
                        str(input_path),
                        "--plan",
                        str(worker_plan),
                        "--output",
                        str(output / f"gpu{index}"),
                    ],
                    env=environment,
                )
            )
        codes = [process.wait() for process in processes]
        write_json(
            output / "parallel_receipt.json", {"exit_codes": codes, "workers": len(processes)}
        )
        return
    for task in plan["tasks"]:
        destination = output / task["name"]
        destination.mkdir(exist_ok=True)
        started = time.time()
        attempt = destination / f"attempt_{time.time_ns()}.json"
        receipt: dict[str, Any] = {
            "task": task,
            "started_unix": started,
            "status": "RUNNING",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES", "unrestricted"),
        }
        write_json(attempt, receipt)
        try:
            if task["kind"] == "train":
                training_config = TrainingConfig.model_validate(task["config"])
                receipt["result"] = train_adapter(
                    inputs["train"],
                    training_config,
                    destination / "training",
                    resume=destination / "training" / task["resume"]
                    if task.get("resume")
                    else None,
                    stop_after=task.get("stop_after"),
                )
            else:
                config = InferenceConfig.model_validate(task["config"])
                if task.get("adapter_task"):
                    config = config.model_copy(
                        update={"adapter": output / task["adapter_task"] / "training/adapter"}
                    )
                if task.get("input_adapter"):
                    key = task["input_adapter"]
                    if Path(key).name != key or key in {".", ".."} or task.get("adapter_task"):
                        raise ValueError("ambiguous or unsafe adapter source")
                    config = config.model_copy(
                        update={
                            "adapter": Path(__file__).resolve().parents[3] / "adapters" / key,
                        }
                    )
                if task["kind"] == "routing":
                    from supportpilot.data.taxonomy import load_taxonomy
                    from supportpilot.routing.llm import routing_experiment

                    receipt["result"] = routing_experiment(
                        inputs[task.get("split", "validation")],
                        config,
                        load_taxonomy(
                            Path(__file__).resolve().parents[3] / "configs/taxonomy.yaml"
                        ),
                        destination,
                    )
                    receipt["status"] = "FINISHED"
                    continue
                receipt["result"] = run_inference(
                    inputs[task.get("split", "validation")],
                    config,
                    destination,
                    prompt_mode=task.get("prompt_mode", "schema"),
                    demonstrations=inputs.get("demonstrations"),
                )
            receipt["status"] = "FINISHED"
        except Exception:
            receipt["status"] = "FAILED"
            receipt["exception"] = traceback.format_exc()
        finally:
            receipt["elapsed_seconds"] = time.time() - started
            receipt["device_wall_hours"] = receipt["elapsed_seconds"] / 3600
            receipt["gpu_hours_scope"] = (
                "one assigned-device process wall time including loading; "
                "whole session allocation and active-compute hours UNKNOWN"
            )
            receipt["environment"] = {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "peft", "trl", "lm-format-enforcer")
            }
            write_json(attempt, receipt)
            write_json(destination / "receipt.json", receipt)
    write_json(
        output / "checksums.json",
        {
            p.relative_to(output).as_posix(): sha256_file(p)
            for p in output.rglob("*")
            if p.is_file() and p.name != "checksums.json"
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execute_plan(read_json(args.plan), read_json(args.inputs), args.output)


if __name__ == "__main__":
    main()
