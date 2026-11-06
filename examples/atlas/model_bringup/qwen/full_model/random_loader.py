"""Complete Qwen2 1.5B architecture with seeded random FP32 parameters, offline."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import torch
import transformers
from transformers import Qwen2Config, Qwen2ForCausalLM

_spec = importlib.util.spec_from_file_location("qwen_session_loader", Path(__file__).with_name("loader.py"))
_loader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_loader)
QwenSession = _loader.QwenSession

SEED = 20261105


def get_model_and_inputs():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    config = Qwen2Config(
        hidden_size=1536,
        intermediate_size=8960,
        num_hidden_layers=28,
        num_attention_heads=12,
        num_key_value_heads=2,
        vocab_size=151936,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).float().eval().requires_grad_(False)
    prefill = int(os.environ.get("QWEN_PREFILL", "32"))
    decodes = int(os.environ.get("QWEN_DECODE_STEPS", "2"))
    ids = torch.arange(3, prefill + 3, dtype=torch.long)[None, :]
    session = QwenSession(model, decodes).eval()
    weights = hashlib.sha256()
    parameters = 0
    for name, value in model.named_parameters():
        parameters += value.numel()
        weights.update(name.encode())
        weights.update(value.detach().numpy().tobytes())
    out = Path(os.environ["QWEN_RANDOM_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        model="Qwen2 1.5B architecture from the replayed Qwen configuration",
        checkpoint=None,
        tensor_source="seeded random FP32 weights; no pretrained checkpoint",
        seed=SEED,
        parameters=parameters,
        unique_parameter_stream_sha256=weights.hexdigest(),
        parameter_hash_format="named_parameters order: UTF-8 name then contiguous FP32 bytes",
        input_ids=ids.tolist(),
        input_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        prefill=prefill,
        decode_steps=decodes,
        torch=torch.__version__,
        transformers=transformers.__version__,
        dtype="float32",
        attention="eager",
        config=config.to_dict(),
    )
    (out / "loader-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return session, (ids,)
