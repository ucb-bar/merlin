"""DeepSeek-R1-Distill-Qwen-1.5B causal-LM -> MLIR.

A Qwen2 decoder, so it differs from the Llama-family workloads in ways the op inventory sees: QKV
projections carry a BIAS (Llama's do not), and the GQA ratio is 12 query heads to 2 KV heads.

Env:
    M2M_QWEN_LAYERS=N   truncate to N decoder layers. NOTE this also switches to from_config, i.e.
                        RANDOM INIT -- it is a smoke path, not a smaller version of the real model.
                        Left unset (the default) the pretrained checkpoint is loaded at full depth.
    M2M_SEQ=N           sequence length for the example input (default 8)
"""

from __future__ import annotations

import os

import torch
from torch import nn

_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


class _LogitsOnly(nn.Module):
    """Wrap a HF causal LM so export sees a clean tensor->tensor forward."""

    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        self.lm = lm

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm(input_ids=input_ids, use_cache=False).logits


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    from transformers import AutoConfig, AutoModelForCausalLM

    n_layers = os.environ.get("M2M_QWEN_LAYERS")
    seq = int(os.environ.get("M2M_SEQ", "8"))

    if n_layers:
        # Smoke path: real Qwen2 architecture, fewer layers, RANDOM INIT (no checkpoint).
        cfg = AutoConfig.from_pretrained(_MODEL_ID)
        cfg.num_hidden_layers = int(n_layers)
        cfg.use_cache = False
        cfg.tie_word_embeddings = False   # avoid a tied-weight swap during quantize_
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID, dtype=torch.float32, use_cache=False, tie_word_embeddings=False
        )

    # Break any residual weight tying so torchao quantize_ can swap weights.
    lm_head = getattr(model, "lm_head", None)
    if isinstance(lm_head, nn.Linear):
        lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())

    model = _LogitsOnly(model.eval()).eval()
    input_ids = torch.randint(0, model.lm.config.vocab_size, (1, seq), dtype=torch.long)
    return model, (input_ids,)
