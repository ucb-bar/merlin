"""Real pretrained Qwen session for the model-export inventory workflow.

Set QWEN_CHECKPOINT to the pinned local snapshot. No model is reconstructed from
hand-written kernel definitions, and checkpoint loading never falls back to
random initialization. The default scope is 32-token prefill plus two decodes.
"""

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
PROMPT = (
    "Explain step by step how a compiler can improve the performance of a neural "
    "network accelerator by reusing matrix tiles, reducing memory transfers, "
    "and overlapping independent operations while preserving numerical accuracy."
)


class QwenSession(torch.nn.Module):
    def __init__(self, model, decode_steps):
        super().__init__()
        self.decoder = model
        self.decode_steps = decode_steps

    def forward(self, input_ids):
        result = self.decoder(input_ids=input_ids, use_cache=True, return_dict=True)
        cache = result.past_key_values
        scores = [result.logits[:, -1:, :]]
        token = torch.argmax(scores[-1], dim=-1)
        tokens = [token]
        for _ in range(self.decode_steps):
            result = self.decoder(
                input_ids=token, past_key_values=cache, use_cache=True, return_dict=True
            )
            cache = result.past_key_values
            scores.append(result.logits[:, -1:, :])
            token = torch.argmax(scores[-1], dim=-1)
            tokens.append(token)
        return torch.cat(tokens, dim=1), torch.cat(scores, dim=1)


def get_model_and_inputs():
    root = Path(__file__).resolve().parents[5]
    checkpoint = Path(os.environ.get("QWEN_CHECKPOINT", root / "out/qwen-checkpoint"))
    if not (checkpoint / "model.safetensors").is_file():
        raise RuntimeError(f"Pinned pretrained checkpoint is missing: {checkpoint}")
    model, loading = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    if any(
        loading.get(k)
        for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise RuntimeError(f"Checkpoint did not load exactly: {loading}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    length = int(os.environ.get("QWEN_PREFILL", "32"))
    tokens = tokenizer(PROMPT, return_tensors="pt", truncation=True, max_length=length)[
        "input_ids"
    ]
    if tokens.shape != (1, length):
        raise RuntimeError(
            f"Prompt produced {tuple(tokens.shape)}, expected (1,{length})"
        )
    session = QwenSession(
        model.eval().requires_grad_(False),
        int(os.environ.get("QWEN_DECODE_STEPS", "2")),
    ).eval()
    return session, (tokens,)
