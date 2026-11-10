"""Complete public reduced-depth graphs with explicit, unadmitted FP semantics.

Only repetition depth changes. Pinned upstream arithmetic methods are extracted
unchanged; packaging/processor adapters are declared and never fetch weights.
These sources are development inputs, not selected integer-model contracts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "merlin.phase0.public_reduced_model_sources.v1"
KINDS = ("residual_classifier", "grouped_decoder", "vision_action_flow")

_COMMON = """from __future__ import annotations
import ast, copy, hashlib, json, logging, math
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Type, Union
import torch
from torch import Tensor, nn
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parent
PINS = json.loads((ROOT / "source-pins.json").read_text())
def leaf(name):
    p = ROOT / name
    if hashlib.sha256(p.read_bytes()).hexdigest() != PINS[name]:
        raise ValueError("public source pin changed: " + name)
    return p
def definitions(name, selected, namespace):
    text = leaf(name).read_text()
    tree = ast.parse(text)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in selected]
    if {n.name for n in nodes} != set(selected):
        raise ValueError("public definition selection is incomplete")
    # Source segments preserve original bodies and decorators byte-for-byte.
    source = "from __future__ import annotations\\n" + "\\n\\n".join(ast.get_source_segment(text, n) for n in nodes)
    exec(compile(source, str(leaf(name)), "exec"), namespace)
def values(shape, offset=0):
    n = math.prod(shape)
    return (((torch.arange(n, dtype=torch.int64) + offset) % 17 - 8).float() / 8).reshape(shape)
def get_model_and_inputs():
    torch.manual_seed(71)
    return build().eval(), inputs()
"""

_RESIDUAL = """
# Logging is the only torchvision-only adapter; arithmetic is unchanged.
_log_api_usage_once = lambda obj: None
definitions("resnet.py", ["conv1x1", "conv3x3", "BasicBlock", "Bottleneck", "ResNet"], globals())
class Program(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = ResNet(Bottleneck, [1, 1, 1, 1], num_classes=1000)
    def forward(self, image):
        return (self.network(image),)
def build():
    return Program()
def inputs():
    return (values((1, 3, 224, 224)),)
"""

_DECODER = """
from transformers import LlamaConfig, LlamaForCausalLM
class Program(nn.Module):
    def __init__(self):
        super().__init__()
        config = json.loads(leaf("decoder.json").read_text())
        config["num_hidden_layers"] = 1
        config["_attn_implementation"] = "eager"
        self.network = LlamaForCausalLM(LlamaConfig(**config))
    def forward(self, tokens):
        result = self.network(input_ids=tokens, use_cache=True, return_dict=True)
        prefill_logits, cache = result.logits, result.past_key_values
        logits, generated = [], []
        token = result.logits[:, -1:].argmax(dim=-1)
        for step in range(32):
            result = self.network(input_ids=token, past_key_values=cache, use_cache=True, return_dict=True)
            logits.append(result.logits)
            generated.append(token)
            cache = result.past_key_values
            token = result.logits[:, -1:].argmax(dim=-1)
        # Carried state remains observable, including the complete prefix and decode.
        layer = cache.layers[0]
        return prefill_logits, torch.cat(logits, dim=1), torch.cat(generated, dim=1), layer.keys, layer.values

def build():
    return Program()
def inputs():
    return (((torch.arange(32, dtype=torch.int64) * 37 + 11) % 32000).reshape(1, 32),)
"""

_VISION_ACTION = """
from transformers import AutoModel, SmolVLMConfig, SmolVLMForConditionalGeneration
config_data = json.loads(leaf("vision_language.json").read_text())
# Two layers retain the public self/cross-attention period; only repeated periods are removed.
config_data["text_config"]["num_hidden_layers"] = 2
config_data["vision_config"]["num_hidden_layers"] = 1
config_data["text_config"]["_attn_implementation"] = "eager"
config_data["vision_config"]["_attn_implementation"] = "eager"
class AutoConfig:
    @staticmethod
    def from_pretrained(model_id):
        return SmolVLMConfig(**copy.deepcopy(config_data))
class AutoProcessor:
    @staticmethod
    def from_pretrained(model_id):
        # IDs are unused because the original add_image_special_tokens flag is false.
        return SimpleNamespace(tokenizer=SimpleNamespace(fake_image_token_id=0, global_image_token_id=0))
class AutoModelForImageTextToText:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        raise ValueError("public development source never loads pretrained weights")
definitions("smolvlm_with_expert.py", ["apply_rope", "get_intermediate_size", "SmolVLMWithExpertModel"], globals())
definitions("utils.py", ["get_safe_dtype"], globals())
definitions("modeling_smolvla.py", [
    "create_sinusoidal_pos_embedding", "make_att_2d_masks", "resize_with_pad", "pad_vector", "normalize",
    "unnormalize", "safe_arcsin", "aloha_gripper_to_angular", "aloha_gripper_from_angular",
    "aloha_gripper_from_angular_inv", "pad_tensor", "VLAFlowMatching"], globals())
class Program(nn.Module):
    def __init__(self):
        super().__init__()
        config = json.loads(leaf("vision_action.json").read_text())
        if config["add_image_special_tokens"] or config["self_attn_every_n_layers"] != 2:
            raise ValueError("processor-free adapter requires the pinned public source flags")
        config.update(device="cpu", load_vlm_weights=False, num_vlm_layers=2)
        self.network = VLAFlowMatching(SimpleNamespace(**config))
    def forward(self, image, tokens, state, noise, state_mean, state_std, action_mean, action_std):
        normalized_state = F.pad((state - state_mean) / state_std, (0, 26))
        actions = self.network.sample_actions(
            [image], [torch.ones((1,), dtype=torch.bool, device=image.device)], tokens,
            torch.ones_like(tokens, dtype=torch.bool), normalized_state, noise=noise)
        return actions, actions[:, :, :6] * action_std + action_mean

def build():
    return Program()
def inputs():
    return (values((1,3,512,512)), ((torch.arange(48, dtype=torch.int64)*37+11)%49152).reshape(1,48),
        values((1,6), 1), values((1,50,32), 2), values((6,), 3), values((6,),4).abs()+1,
        values((6,),5), values((6,),6).abs()+1)
"""


def render(kind):
    """Return an ordinary standalone PyTorch loader; no target schedule is added."""
    if kind not in KINDS:
        raise ValueError("select a complete public reduced-depth graph kind")
    return _COMMON + dict(zip(KINDS, (_RESIDUAL, _DECODER, _VISION_ACTION), strict=True))[kind]


def freeze(destination, *, implementations, configurations, transformers_package):
    """Freeze actual source loaders and pins, retaining precise admission refusals.

    The caller selects already disclosed public implementation/config directories.
    No model weights, validation artifacts or network access are used here.
    """
    destination, impl, configs, tf = map(Path, (destination, implementations, configurations, transformers_package))
    if destination.exists():
        raise ValueError("reduced source destination must be new")
    original = json.loads((impl / "provenance.json").read_text())
    supplement = json.loads((impl / "provenance-supplement.json").read_text())
    files = {
        "resnet.py": impl / "torchvision/torchvision/models/resnet.py",
        "modeling_smolvla.py": impl / "lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py",
        "smolvlm_with_expert.py": impl / "lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py",
        "configuration_smolvla.py": impl / "lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py",
        "utils.py": impl / "lerobot/src/lerobot/utils/utils.py",
        **{name + ".json": configs / (name + ".json") for name in ("decoder", "vision_action", "vision_language")},
    }
    # Original exact file identities, not an unchecked arbitrary source directory.
    expected = {}
    for record in original + supplement:
        expected[Path(record["path"]).name] = record["sha256"]
    for name in files:
        if name.endswith(".py") and hashlib.sha256(files[name].read_bytes()).hexdigest() != expected[name]:
            raise ValueError("public upstream implementation pin changed")
    config_provenance = json.loads((configs / "provenance.json").read_text())
    if {r["name"] for r in config_provenance} != {"decoder", "vision_action", "vision_language"}:
        raise ValueError("select the complete public configuration roster")
    for row in config_provenance:
        path = configs / (row["name"] + ".json")
        raw = path.read_bytes() if not path.is_symlink() else b""
        if not row.get("revision") or len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("public configuration identity changed")
    destination.mkdir(parents=True)
    pins = {}
    for name, source in files.items():
        data = source.read_bytes()
        (destination / name).write_bytes(data)
        pins[name] = hashlib.sha256(data).hexdigest()
    (destination / "source-pins.json").write_text(json.dumps(pins, indent=2) + "\n")
    dependencies = []
    for name in (
        "__init__.py",
        "cache_utils.py",
        "models/llama/modeling_llama.py",
        "models/llama/configuration_llama.py",
        "models/smolvlm/modeling_smolvlm.py",
        "models/smolvlm/configuration_smolvlm.py",
    ):
        p = tf / name
        dependencies.append(dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    members = []
    details = [
        dict(
            depth=dict(stage_bottlenecks=[1, 1, 1, 1]),
            inputs=["B1x3x224x224"],
            liveouts=["all1000_logits"],
            preserved=[
                "stem",
                "all_four_width_stages",
                "batch_norm",
                "residual_projection",
                "relu",
                "pool",
                "classifier",
            ],
            pending=["selected batch-normalization/ReLU/pooling/readout numerical contract"],
        ),
        dict(
            depth=dict(decoder=1),
            inputs=["B1x32_tokens", "32_greedy_decode_steps"],
            liveouts=["complete_prefill_logits", "complete_decode_logits", "generated_tokens", "complete_final_KV"],
            preserved=[
                "GQA",
                "RMSNorm",
                "RoPE",
                "mask",
                "softmax",
                "SwiGLU",
                "residual",
                "KV_carry",
                "vocabulary_head",
            ],
            pending=["selected RMSNorm/RoPE/softmax/SwiGLU and logits numerical contracts"],
        ),
        dict(
            depth=dict(vision=1, language=2, expert=2, attention_period=2),
            inputs=[
                "B1_camera512x512",
                "48_language_tokens",
                "state6_padded32",
                "action_chunk50x32",
                "10_denoise_steps",
                "explicit_public_mean_std",
            ],
            liveouts=["complete50x32_latent_actions", "complete50x6_decoded_actions"],
            preserved=[
                "vision_patch_encoder",
                "vision_softmax_MLP",
                "pixel_shuffle_connector",
                "language_GQA",
                "prefix_KV_reuse",
                "state_projection",
                "time_embedding",
                "self_cross_attention_period",
                "10_Euler_state_updates",
                "action_decode",
            ],
            pending=["selected vision/connector/normalization/attention/expert/Euler whole-graph numerical contracts"],
        ),
    ]
    for kind, detail in zip(KINDS, details, strict=True):
        source = render(kind).encode()
        (destination / (kind + ".py")).write_bytes(source)
        members.append(
            dict(
                identity=kind,
                source=kind + ".py",
                source_sha256=hashlib.sha256(source).hexdigest(),
                **detail,
                status="source_frozen_selected_numerical_contract_refused",
                reference="UNAVAILABLE; no selected whole-graph numerical policy",
                capture="NOT_RUN",
                compiler="NOT_ADMITTED",
                full_model_validation_run=False,
            )
        )
    result = dict(
        schema=SCHEMA,
        scope="public_development_only_not_validation",
        members=members,
        source_pins=pins,
        upstream_provenance=[original, supplement],
        config_provenance=config_provenance,
        selected_python_dependencies=dependencies,
        runtime_closure=(
            "Leaf pins identify implementation choices; original sealed runtime closure remains required."
        ),
        adaptation=[
            "fresh deterministic weights; no pretrained weights or calibration",
            "depth-only reduction retaining every distinct stage/attention period",
            "exact original extracted arithmetic methods",
            "torchvision logging disabled",
            "offline pinned config; unused processor token IDs only",
            "explicit public normalization operands; no private statistics",
        ],
        target_numerical_policy="UNAVAILABLE; do not reinterpret FP32 semantics as selected int8 or relax tolerances",
        formal_admission=False,
        timing_qualified=False,
    )
    (destination / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
