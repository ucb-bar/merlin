"""TinyLlama (1.1B) causal-LM example -> MLIR.

    m2m coverage examples/tiny_llama.py
    m2m convert  examples/tiny_llama.py --out <path>
    m2m convert  examples/tiny_llama.py --quant int8_weight_only --out <path>

Env:
    M2M_LLAMA_LAYERS=N   truncate to N decoder layers (fast smoke; default: full)
    M2M_SEQ=N            sequence length for the example input (default 8)

Weights are loaded from the local HF cache (TinyLlama/TinyLlama-1.1B-Chat-v1.0).
"""

from __future__ import annotations

import os

import torch
from torch import nn

_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class _LogitsOnly(nn.Module):
    """Wrap a HF causal LM so export sees a clean tensor->tensor forward."""

    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        self.lm = lm

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.lm(input_ids=input_ids, use_cache=False)
        return out.logits


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    from transformers import AutoConfig, AutoModelForCausalLM

    n_layers = os.environ.get("M2M_LLAMA_LAYERS")
    seq = int(os.environ.get("M2M_SEQ", "8"))

    if n_layers:
        # Fast smoke path: real Llama architecture, fewer layers, random init.
        cfg = AutoConfig.from_pretrained(_MODEL_ID)
        cfg.num_hidden_layers = int(n_layers)
        cfg.use_cache = False
        cfg.tie_word_embeddings = False  # avoid tied-weight swap during quantize_
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
    vocab = model.lm.config.vocab_size
    input_ids = torch.randint(0, vocab, (1, seq), dtype=torch.long)
    return model, (input_ids,)
