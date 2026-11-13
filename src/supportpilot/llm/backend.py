"""Lazy Transformers backend, pinned weights, bounded generation and measured resources."""

from __future__ import annotations

import gc
import importlib
import time
from typing import Any

import psutil

from supportpilot.llm.adapters import validate_adapter
from supportpilot.llm.decoding import prefix_filter, tokenizer_data
from supportpilot.llm.models import InferenceConfig
from supportpilot.llm.prompts import response_schema


class TransformersBackend:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        if config.adapter is not None:
            validate_adapter(config.adapter, config.model)
        self.torch: Any = importlib.import_module("torch")
        transformers: Any = importlib.import_module("transformers")
        self.torch.manual_seed(config.seed)
        self.last_observation: dict[str, Any] = {}
        started = time.perf_counter()
        self.tokenizer: Any = transformers.AutoTokenizer.from_pretrained(
            config.model.repo_id, revision=config.model.revision, trust_remote_code=False
        )
        kwargs: dict[str, Any] = {
            "revision": config.model.revision,
            "trust_remote_code": False,
            "dtype": getattr(self.torch, config.dtype),
            "device_map": {"": config.device},
            "attn_implementation": "eager",
        }
        if config.quantization == "nf4":
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=getattr(self.torch, config.dtype),
            )
        cls = (
            transformers.AutoModelForCausalLM
            if config.model.architecture == "causal"
            else transformers.AutoModelForImageTextToText
        )
        self.model: Any = cls.from_pretrained(config.model.repo_id, **kwargs)
        if config.adapter is not None:
            if not (config.adapter / "adapter_config.json").is_file():
                raise FileNotFoundError("adapter_config.json missing: " + str(config.adapter))
            peft: Any = importlib.import_module("peft")
            self.model = peft.PeftModel.from_pretrained(self.model, str(config.adapter))
        self.model.eval()
        self.load_seconds = time.perf_counter() - started
        self._tokenizer_data: Any = None
        if config.constrained:
            self._tokenizer_data = tokenizer_data(self.tokenizer)

    def render(self, messages: list[dict[str, str]]) -> str:
        return str(
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        )

    def generate(self, messages: list[dict[str, str]]) -> str:
        text = self.render(messages)
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_tokens = int(inputs["input_ids"].shape[1])
        if input_tokens > self.config.max_input_tokens:
            raise ValueError(f"input_overlength:{input_tokens}>{self.config.max_input_tokens}")
        inputs = inputs.to(self.config.device)
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "max_time": self.config.max_seconds,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.config.constrained:
            kwargs["prefix_allowed_tokens_fn"] = prefix_filter(
                self._tokenizer_data, response_schema()
            )
        cuda = self.config.device.startswith("cuda")
        if cuda:
            self.torch.cuda.reset_peak_memory_stats(self.config.device)
            self.torch.cuda.synchronize(self.config.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, **kwargs)
        if cuda:
            self.torch.cuda.synchronize(self.config.device)
        elapsed = time.perf_counter() - started
        generated = outputs[0, input_tokens:]
        raw = str(self.tokenizer.decode(generated, skip_special_tokens=True))
        self.last_observation = {
            "seconds": elapsed,
            "input_tokens": input_tokens,
            "output_tokens": len(generated),
            "tokens_per_second": len(generated) / elapsed,
            "rss_bytes": psutil.Process().memory_info().rss,
            "peak_vram_bytes": self.torch.cuda.max_memory_allocated(self.config.device)
            if cuda
            else 0,
            "reserved_vram_bytes": self.torch.cuda.max_memory_reserved(self.config.device)
            if cuda
            else 0,
            "token_limit_reached": len(generated) >= self.config.max_new_tokens,
        }
        return raw

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.config.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
