"""A section of a pretrained Hugging Face causal LM: its first N decoder layers, real weights, one forward.

For a model whose whole int8 image does not fit a board's DRAM. Every shape a whole-model capture has
(hidden size, FFN width, heads, head dim, vocabulary) is kept, because none of them depends on the layer
count; only the repetition of the decoder layer is cut, so each contraction the section contains is a
contraction the whole model contains, at the same shape and with the same pretrained weights.

    CAUSAL_LM_MODEL=<hf repo id>   the checkpoint, read from the local HF cache (set HF_HUB_OFFLINE=1 to
                                   forbid a download; gated checkpoints need their own access first)
    CAUSAL_LM_LAYERS=N             keep the first N decoder layers (unset: all of them)
    CAUSAL_LM_ENTRY=ids|embeds     where the exported forward starts. ``ids`` (default): token ids ->
                                   logits. ``embeds``: the hidden state the embedding lookup produces;
                                   the table is replaced by a one-row placeholder so it is NOT exported
                                   (it is fp32 -- int8_weight_only quantizes nn.Linear only -- and does no
                                   arithmetic), and the example input is the REAL embedding rows of the
                                   drawn token ids, so the section sees the tensor the whole model would.
    CAUSAL_LM_ATTENTION=eager      the attention implementation (eager keeps the score matmuls visible)
    M2M_SEQ=N                      sequence length of the example input (default 8)

Works for any ``AutoModelForCausalLM`` whose decoder keeps its layers in a ``layers`` ModuleList (Llama,
Gemma, Qwen, Mistral, ...). It names no model: the checkpoint is an argument. Capture it with
``examples/kodiak/run.sh capture <bundle-name> --loader examples/workloads/causal_lm_section/loader.py``.
"""

from __future__ import annotations

import os

import torch
from torch import nn


class _Logits(nn.Module):
    """Tensor -> logits, entering at token ids or at the embedded hidden state."""

    def __init__(self, lm: nn.Module, entry: str) -> None:
        super().__init__()
        self.lm = lm
        self.entry = entry

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.entry == "embeds":
            return self.lm(inputs_embeds=x, use_cache=False).logits
        return self.lm(input_ids=x, use_cache=False).logits


def _keep_first_layers(lm: nn.Module, n: int) -> int:
    decoder = lm.get_decoder()
    layers = getattr(decoder, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise RuntimeError(f"{type(decoder).__name__} keeps no `layers` ModuleList; cannot cut a section")
    total = len(layers)
    if not 1 <= n <= total:
        raise RuntimeError(f"CAUSAL_LM_LAYERS={n}: the model has {total} decoder layers")
    decoder.layers = nn.ModuleList(list(layers)[:n])
    for cfg in {id(lm.config): lm.config, id(decoder.config): decoder.config}.values():
        cfg.num_hidden_layers = n
        # Per-layer attention kinds (sliding window / global) are indexed by layer: cut them with the layers.
        if isinstance(getattr(cfg, "layer_types", None), list):
            cfg.layer_types = cfg.layer_types[:n]
    return total


def get_model_and_inputs():
    from transformers import AutoModelForCausalLM

    model_id = os.environ.get("CAUSAL_LM_MODEL")
    if not model_id:
        raise RuntimeError("set CAUSAL_LM_MODEL to the Hugging Face id of a cached causal LM")
    entry = os.environ.get("CAUSAL_LM_ENTRY", "ids")
    if entry not in ("ids", "embeds"):
        raise RuntimeError(f"CAUSAL_LM_ENTRY must be ids or embeds, not {entry!r}")
    seq = int(os.environ.get("M2M_SEQ", "8"))
    lm = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        use_cache=False,
        attn_implementation=os.environ.get("CAUSAL_LM_ATTENTION", "eager"),
    ).eval()
    # Untie the head from the embedding AFTER loading (loading untied makes a tied checkpoint's head
    # MISSING and randomly initialized), so quantizing the head leaves the lookup alone.
    head = lm.get_output_embeddings()
    if isinstance(head, nn.Linear):
        head.weight = nn.Parameter(head.weight.detach().clone())
        lm.config.tie_word_embeddings = False
    layers = os.environ.get("CAUSAL_LM_LAYERS")
    total = _keep_first_layers(lm, int(layers)) if layers else len(lm.get_decoder().layers)
    kept = int(layers) if layers else total

    ids = torch.randint(0, lm.config.vocab_size, (1, seq), dtype=torch.long)
    if entry == "embeds":
        embed = lm.get_input_embeddings()
        with torch.no_grad():
            x = embed(ids).detach().clone()
        lm.set_input_embeddings(nn.Embedding(1, x.shape[-1]))
    else:
        x = ids
    model = _Logits(lm, entry).eval()
    model.session_provenance = {
        "capture_scope": "single_forward_section" if kept < total else "single_forward",
        "checkpoint": model_id,
        "checkpoint_revision": getattr(lm.config, "_commit_hash", None),
        "decoder_layers_kept": kept,
        "decoder_layers_total": total,
        "entry": entry,
        "full_checkpoint": kept == total and entry == "ids",
        "input_source": "seeded_synthetic_token_ids",
        "synthetic_inputs": True,
        "paper_ready": False,
    }
    return model, (x,)
