"""Completion-only TRL/PEFT training with input-bound resumable checkpoints."""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from typing import Any

import psutil

from supportpilot.common.hashing import sha256_text
from supportpilot.common.io import read_json, write_json
from supportpilot.llm.adapters import write_adapter_manifest
from supportpilot.llm.backend import TransformersBackend
from supportpilot.llm.models import TrainingConfig
from supportpilot.llm.prompts import prepare_messages
from supportpilot.llm.provenance import runtime_identity


def prepare_training_rows(
    rows: list[dict[str, Any]], tokenizer: Any, config: TrainingConfig
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    selected, excluded = [], []
    ordered = sorted(rows, key=lambda r: sha256_text(str(config.inference.seed) + r["example_id"]))
    if config.train_limit is not None:
        ordered = ordered[: config.train_limit]
    for row in ordered:
        flags = row.get("quality_flags") or []
        if any(str(f).startswith("mt:") for f in flags):
            draw = int(sha256_text("mt|" + row["example_id"])[:8], 16) / 2**32
            if draw >= config.mt_weight:
                excluded.append({"example_id": row["example_id"], "reason": "mt_downweight"})
                continue
        if row.get("split") != "train" or (
            not row.get("kb_variant")
            and row["task"] in {"grounded_answer", "conflicting_context", "injection_resist"}
        ):
            raise ValueError("training requires train-only training-KB variants")
        messages = prepare_messages(
            row["messages"][:-1], mode=config.prompt_mode, language=row["lang"]
        )
        prompt = str(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        )
        completion = str(row["messages"][-1]["content"]) + tokenizer.eos_token
        size = len(tokenizer.encode(prompt + completion, add_special_tokens=False))
        if size > config.max_length:
            excluded.append(
                {"example_id": row["example_id"], "reason": "overlength", "tokens": size}
            )
            continue
        selected.append({"prompt": prompt, "completion": completion})
    if not selected:
        raise ValueError("no complete training targets fit; no silent truncation")
    return selected, excluded


def train_adapter(
    rows: list[dict[str, Any]],
    config: TrainingConfig,
    output: Path,
    *,
    resume: Path | None = None,
    stop_after: int | None = None,
) -> dict[str, Any]:
    datasets: Any = importlib.import_module("datasets")
    peft: Any = importlib.import_module("peft")
    trl: Any = importlib.import_module("trl")
    transformers: Any = importlib.import_module("transformers")
    if config.prompt_mode == "fewshot":
        raise ValueError("training demonstrations are not implicitly selected")
    identity = sha256_text(
        json.dumps(
            {"config": config.model_dump(mode="json"), "rows": rows, "runtime": runtime_identity()},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    binding = output / "binding.json"
    if binding.exists():
        if read_json(binding)["identity"] != identity:
            raise ValueError("checkpoint input/config identity mismatch")
        if resume is None:
            raise ValueError("existing training run requires explicit resume")
    elif resume is not None:
        raise ValueError("resume requires the original binding manifest")
    output.mkdir(parents=True, exist_ok=True)
    write_json(binding, {"identity": identity, "config": config.model_dump(mode="json")})
    backend = TransformersBackend(config.inference)
    try:
        prepared, excluded = prepare_training_rows(rows, backend.tokenizer, config)
        write_json(
            output / "data_accounting.json", {"included": len(prepared), "excluded": excluded}
        )
        model = backend.model
        model.config.use_cache = False
        if config.packing and model.config._attn_implementation != "flash_attention_2":
            raise ValueError(
                "packing requires isolated sequence attention; eager packing is unsafe"
            )
        if config.inference.quantization == "nf4":
            model = peft.prepare_model_for_kbit_training(model)
        args = trl.SFTConfig(
            output_dir=str(output),
            max_steps=config.max_steps,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation,
            learning_rate=config.learning_rate,
            warmup_steps=max(1, int(config.max_steps * 0.05)),
            max_length=config.max_length,
            packing=config.packing,
            completion_only_loss=True,
            save_steps=config.save_steps,
            save_total_limit=None,
            logging_steps=1,
            report_to="none",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            fp16=config.inference.dtype == "float16",
            bf16=config.inference.dtype == "bfloat16",
            seed=config.inference.seed,
            data_seed=config.inference.seed,
            optim="adamw_torch",
            dataloader_num_workers=0,
        )
        callbacks = []
        if stop_after is not None:

            class StopAtStep(transformers.TrainerCallback):  # type: ignore[misc]
                def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                    if state.global_step >= stop_after:
                        control.should_training_stop = True
                        control.should_save = True
                    return control

            callbacks.append(StopAtStep())
        trainer = trl.SFTTrainer(
            model=model,
            args=args,
            train_dataset=datasets.Dataset.from_list(prepared),
            processing_class=backend.tokenizer,
            peft_config=peft.LoraConfig(
                r=config.rank,
                lora_alpha=config.alpha,
                lora_dropout=0.05,
                target_modules=config.target_modules,
                task_type="CAUSAL_LM",
                bias="none",
            ),
            callbacks=callbacks,
        )
        # TRL may add EOS/BOS or filter rows. Re-read its prepared labels to prove that
        # no assistant JSON or terminal EOS was truncated after our length audit.
        for sample, source in zip(trainer.train_dataset, prepared, strict=True):
            labels = trainer.data_collator([sample])["labels"][0].tolist()
            supervised = [token for token in labels if token != -100]
            if not supervised or supervised[-1] != backend.tokenizer.eos_token_id:
                raise ValueError("prepared target lost terminal EOS")
            decoded = backend.tokenizer.decode(supervised, skip_special_tokens=True)
            expected = source["completion"].removesuffix(backend.tokenizer.eos_token)
            if json.loads(decoded) != json.loads(expected):
                raise ValueError("prepared assistant target was truncated or altered")
        # TRL 1.17 casts quantized adapters to bf16 unconditionally. Trainable fp32
        # parameters are required for fp16 GradScaler, including on non-bf16 T4 GPUs.
        for parameter in trainer.model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(backend.torch.float32)
        reference = trainer.data_collator([trainer.train_dataset[0]])
        reference = {key: value.to(config.inference.device) for key, value in reference.items()}

        def reference_loss() -> float:
            trainer.model.eval()
            with backend.torch.no_grad():
                value = float(trainer.model(**reference).loss.item())
            trainer.model.train()
            return value

        before_loss = reference_loss()
        cuda = config.inference.device.startswith("cuda")
        if cuda:
            backend.torch.cuda.reset_peak_memory_stats(config.inference.device)
        training = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        after_loss = reference_loss()
        trainer.save_model(str(output / "adapter"))
        backend.tokenizer.save_pretrained(str(output / "adapter"))
        write_adapter_manifest(output / "adapter", config.inference.model, identity)
        history = trainer.state.log_history
        nonfinite_gradients = sum(
            "grad_norm" in event and not math.isfinite(float(event["grad_norm"]))
            for event in history
        )
        # JSON does not permit NaN/Infinity. Preserve them as explicit strings, never drop steps.
        history = [
            {
                key: str(value) if isinstance(value, float) and not math.isfinite(value) else value
                for key, value in event.items()
            }
            for event in history
        ]
        receipt = {
            "identity": identity,
            "global_step": trainer.state.global_step,
            "metrics": training.metrics,
            "history": history,
            "nonfinite_gradient_steps": nonfinite_gradients,
            "resumed_from": str(resume) if resume else None,
            "config": config.model_dump(mode="json"),
            "included": len(prepared),
            "excluded": len(excluded),
            "trainable_dtype": "float32",
            "verified_complete_targets": len(prepared),
            "reference_loss_before": before_loss,
            "reference_loss_after": after_loss,
            "reference_input_sha256": sha256_text(str(reference["input_ids"].tolist())),
            "end_rss_bytes": psutil.Process().memory_info().rss,
            "peak_vram_bytes": backend.torch.cuda.max_memory_allocated(config.inference.device)
            if cuda
            else 0,
            "status": "FINISHED",
        }
        write_json(output / f"training_step_{trainer.state.global_step}.json", receipt)
        return receipt
    finally:
        backend.close()
