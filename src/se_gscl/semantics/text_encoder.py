"""Frozen decoder-only LLM text encoding for semantic prototype construction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def masked_mean_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool valid token states while excluding padding positions."""

    if last_hidden_state.ndim != 3:
        raise ValueError("last_hidden_state must have shape [B,L,d].")
    if attention_mask.shape != last_hidden_state.shape[:2]:
        raise ValueError("attention_mask must have shape [B,L].")
    weights = attention_mask.to(last_hidden_state.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (last_hidden_state * weights).sum(dim=1) / denominator


class FrozenDecoderTextEncoder:
    """Load a Hugging Face decoder model as a frozen text feature extractor."""

    _DTYPES = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cpu",
        dtype: str = "float32",
        cache_dir: str | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required to build real LLM text embeddings."
            ) from exc

        if dtype not in self._DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype}")
        self.model_name_or_path = str(model_name_or_path)
        self.device = torch.device(device)
        self.dtype = self._DTYPES[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        model_kwargs: dict[str, Any] = {
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        if self.device.type == "cuda":
            model_kwargs["torch_dtype"] = self.dtype
        self.model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @torch.inference_mode()
    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 8,
        max_length: int = 256,
    ) -> torch.Tensor:
        """Return CPU float32 embeddings with shape [N,d]."""

        if not texts:
            raise ValueError("At least one text description is required.")
        outputs: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = [str(value) for value in texts[start : start + batch_size]]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            model_output = self.model(**encoded, return_dict=True)
            pooled = masked_mean_pool(
                model_output.last_hidden_state,
                encoded["attention_mask"],
            )
            outputs.append(pooled.float().cpu())
        return torch.cat(outputs, dim=0)
