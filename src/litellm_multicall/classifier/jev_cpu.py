"""Prism-owned CPU loader around unmodified, pinned SemIf scoring."""

from __future__ import annotations

import inspect
import resource
import sys
import time

from .._vendor.semif import direct, shared
from ..errors import PrismError
from .types import DecisionBatch, result_from_score


def assert_cpu(model):
    tensors = list(model.parameters()) + list(model.buffers())
    if not tensors or any(t.device.type != "cpu" for t in tensors):
        raise PrismError(
            "CPU profile has non-CPU parameters/buffers", 3, "device_contract"
        )
    if any(t.is_floating_point() and str(t.dtype) != "torch.float32" for t in tensors):
        raise PrismError(
            "CPU parity profile requires float32 tensors", 3, "dtype_contract"
        )


class JevCPUBackend:
    def __init__(self, profile, artifact):
        self.profile = profile
        self.artifact = artifact
        self.model = self.tokenizer = None
        self.metadata = {}

    def load(self):
        if self.model is not None:
            return
        started = time.perf_counter()
        cpu_start = time.process_time()
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise PrismError(
                "install the classifier-jevcpu extra with platform constraints",
                4,
                "missing_dependency",
            ) from exc
        torch.set_num_threads(self.profile.threads)
        torch.set_num_interop_threads(self.profile.interop_threads)
        common = {"local_files_only": True, "trust_remote_code": False}
        source = self.artifact["snapshot"]
        tokenizer = transformers.AutoTokenizer.from_pretrained(source, **common)
        kwargs = {
            "dtype"
            if int(transformers.__version__.split(".")[0]) >= 5
            else "torch_dtype": torch.float32
        }
        model, loading = transformers.AutoModelForCausalLM.from_pretrained(
            source, use_safetensors=True, output_loading_info=True, **kwargs, **common
        )
        if any(
            loading.get(key)
            for key in (
                "missing_keys",
                "mismatched_keys",
                "unexpected_keys",
                "error_msgs",
            )
        ):
            raise PrismError(
                "checkpoint loading diagnostics indicate incomplete/unexpected weights",
                7,
                "incomplete_checkpoint",
            )
        model.to(device="cpu", dtype=torch.float32)
        assert_cpu(model)
        model.eval()
        self.model, self.tokenizer = model, tokenizer
        self.metadata = {
            "backend": "jev_cpu",
            "source": self.artifact["source"],
            "revision": self.artifact["revision"],
            "tokenizer_revision": self.artifact["tokenizer_revision"],
            "artifact_sha256": self.artifact["identity_sha256"],
            "dtype": "float32",
            "device": "cpu",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "load_seconds": time.perf_counter() - started,
            "load_process_cpu_seconds": time.process_time() - cpu_start,
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "prompt_version": direct.PROMPT_VERSION,
            "scoring_implementation": "semif-b49b5bf5776af78495fa4900996042726f8e7c10",
            "criterion_schema_version": 1,
            "state_builder": "native_decision_v1",
            "scoring_mode": self.profile.scoring_mode,
        }

    def capabilities(self):
        return {
            "device": "cpu",
            "dtype": "float32",
            "options": [2, 16],
            "scoring_modes": ["direct", "shared", "compact_generation"],
            "selective_position_logits": self.model is not None
            and "logits_to_keep" in inspect.signature(self.model.forward).parameters,
        }

    def _compact(self, row, limit):
        import torch

        started = time.perf_counter()
        ids, slots, prompt_hash = direct.encode_prompt(self.tokenizer, row, limit)
        inputs = {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }
        forward_start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=lambda batch_id, tokens: slots,
            )
        selected = output.scores[0][0, slots].float().tolist()
        from .._vendor.semif.core import softmax

        return {
            "id": row["id"],
            "option_ids": [o["id"] for o in row["options"]],
            "option_logits": selected,
            "probabilities": softmax(selected),
            "input_tokens": len(ids),
            "prompt_sha256": prompt_hash,
            "forward_seconds": time.perf_counter() - forward_start,
            "total_seconds": time.perf_counter() - started,
        }

    def score(self, request):
        if self.model is None:
            raise PrismError("model has not been loaded", 3, "unavailable")
        started, cpu_start = time.perf_counter(), time.process_time()
        rows = request.rows()
        limit = min(
            self.profile.max_input_tokens,
            getattr(
                self.model.config,
                "max_position_embeddings",
                self.profile.max_input_tokens,
            ),
        )
        mode = self.profile.scoring_mode
        self.metadata = {**self.metadata, "scoring_mode": mode}
        if mode == "shared":
            # Check allocation bounds before invoking the upstream cache/forward.
            encoded = [
                direct.encode_prompt(self.tokenizer, row, limit)[0] for row in rows
            ]
            prefix = shared._state_prefix(self.tokenizer, request.state)
            padded = len(rows) * max(len(ids) - len(prefix) for ids in encoded)
            if padded > self.profile.max_padded_suffix_tokens:
                raise PrismError(
                    "shared suffix padding budget exhausted", 6, "resource_budget"
                )
            raw, timing = shared.score_shared(
                self.model, self.tokenizer, rows, self.metadata, max_tokens=limit
            )
            forwards = 2
        else:
            scorer = (
                (lambda row: self._compact(row, limit))
                if mode == "compact_generation"
                else (
                    lambda row: direct.score(
                        self.model, self.tokenizer, row, self.metadata, max_tokens=limit
                    )
                )
            )
            raw = [scorer(row) for row in rows]
            timing = {"forward_seconds": sum(r["forward_seconds"] for r in raw)}
            forwards = len(rows)
        results = [
            result_from_score(request, c, r, mode)
            for c, r in zip(request.criteria, raw)
        ]
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        measurements = {
            **timing,
            "model_forward_passes": forwards,
            "semantic_decisions": len(rows),
            "logical_input_tokens": sum(r["input_tokens"] for r in raw),
            "generated_output_tokens": sum(r.generated_output_tokens for r in results),
            "backend_total_ms": (time.perf_counter() - started) * 1000,
            "process_cpu_seconds": time.process_time() - cpu_start,
            "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
            "gpu_active_seconds": None,
            "energy_joules": None,
            "measurement_limitations": [
                "GPU active time and energy are unavailable; RSS is process lifetime peak"
            ],
        }
        return DecisionBatch(
            request_id=request.request_id,
            results=results,
            measurements=measurements,
            backend_metadata=self.metadata,
        )

    def score_many(self, requests):
        return [self.score(request) for request in requests]

    def close(self):
        self.model = self.tokenizer = None
