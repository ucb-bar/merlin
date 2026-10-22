"""One complete SmolVLA expert invocation with external prefix-cache fixtures.

The lightweight assembly reuses installed LeRobot method bodies and selected
checkpoint tensors. It never constructs the vision/language prefix encoder.
All selected weights/interfaces are explicitly FP32 before optional W8A8.
"""

import ast
import hashlib
from importlib.metadata import distribution
import math
from pathlib import Path
from types import SimpleNamespace


def reference_methods(torch):
    """Load only the named source methods, avoiding full-policy imports/creation."""
    root = Path(distribution("lerobot").locate_file("lerobot/policies/smolvla"))
    sources = {name: root / name for name in ("modeling_smolvla.py", "smolvlm_with_expert.py")}

    def cpu_dtype(dtype, device):
        if device != "cpu":
            raise ValueError("This development reference admits CPU only")
        return dtype

    namespace = {"__name__": __name__, "torch": torch, "nn": torch.nn, "F": torch.nn.functional,
                 "math": math, "Tensor": torch.Tensor, "get_safe_dtype": cpu_dtype}
    functions, bindings = {}, []
    selections = {
        "modeling_smolvla.py": {None: ("create_sinusoidal_pos_embedding", "make_att_2d_masks"),
                                "VLAFlowMatching": ("embed_suffix", "denoise_step")},
        "smolvlm_with_expert.py": {None: ("apply_rope",), "SmolVLMWithExpertModel":
                                  ("get_vlm_model", "get_model_layers", "forward", "forward_attn_layer",
                                   "forward_cross_attn_layer", "get_attention_interface", "eager_attention_forward")},
    }
    for name, path in sources.items():
        raw = path.read_bytes()
        tree = ast.parse(raw, filename=str(path))
        selected = []
        for owner, names in selections[name].items():
            nodes = tree.body if owner is None else next(node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
            for function in names:
                node = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == function)
                if node.decorator_list:
                    raise ValueError("Unexpected reference-method decorator")
                future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
                module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
                exec(compile(module, str(path), "exec"), namespace)
                functions[(owner, function)] = namespace[function]
                selected.append({"owner": owner, "function": function, "line": node.lineno})
        bindings.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "methods": selected})
    return functions, bindings


def make_denoiser(torch, linear, load, generator, *, expert_layers=16):
    if expert_layers not in (1, 16):
        raise ValueError("Select one original expert block or the complete 16-layer expert")
    methods, sources = reference_methods(torch)
    nn = torch.nn

    class Projection(nn.Module):
        def __init__(self, name, bias=False):
            super().__init__()
            weight = load(name + ".weight")
            self.operation = linear(weight, load(name + ".bias") if bias else None)
            # Public LeRobot methods query projection.weight.dtype at casts.
            self.register_buffer("weight", torch.empty(0, dtype=torch.float32))

        def forward(self, value):
            return self.operation(value)[0]

    class RMS(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.register_buffer("weight", load(name + ".weight"))

        def forward(self, value):
            return value * torch.rsqrt((value * value).mean(dim=-1, keepdim=True) + 1e-5) * self.weight

    class MLP(nn.Module):
        def __init__(self, prefix):
            super().__init__()
            self.gate_proj = Projection(prefix + ".gate_proj")
            self.up_proj = Projection(prefix + ".up_proj")
            self.down_proj = Projection(prefix + ".down_proj")

        def forward(self, value):
            return self.down_proj(nn.functional.silu(self.gate_proj(value)) * self.up_proj(value))

    class Attention(nn.Module):
        def __init__(self, prefix):
            super().__init__()
            self.head_dim = 64
            for name in ("q", "k", "v", "o"):
                setattr(self, name + "_proj", Projection(prefix + "." + name + "_proj"))

    class Layer(nn.Module):
        def __init__(self, index):
            super().__init__()
            prefix = "model.vlm_with_expert.lm_expert.layers." + str(index)
            self.input_layernorm = RMS(prefix + ".input_layernorm")
            self.post_attention_layernorm = RMS(prefix + ".post_attention_layernorm")
            self.self_attn = Attention(prefix + ".self_attn")
            self.mlp = MLP(prefix + ".mlp")

    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_expert = nn.Module()
            self.lm_expert.layers = nn.ModuleList([Layer(index) for index in range(expert_layers)])
            self.lm_expert.norm = RMS("model.vlm_with_expert.lm_expert.norm")
            text = SimpleNamespace(layers=[None] * expert_layers, norm=nn.Identity())
            self.vlm = SimpleNamespace(model=SimpleNamespace(text_model=text),
                                       config=SimpleNamespace(text_config=SimpleNamespace(head_dim=64)))
            self.num_vlm_layers = self.num_expert_layers = expert_layers
            self.num_attention_heads, self.num_key_value_heads = 15, 5
            self.self_attn_every_n_layers = 2
            self.attention_mode = "cross_attn"
            self.expert_hidden_size = 720

    for (owner, name), function in methods.items():
        if owner == "SmolVLMWithExpertModel":
            setattr(Expert, name, function)

    class Denoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(chunk_size=50, num_steps=10, use_cache=True, min_period=0.004, max_period=4.0)
            self.vlm_with_expert = Expert()
            for name in ("action_in_proj", "action_time_mlp_in", "action_time_mlp_out", "action_out_proj"):
                setattr(self, name, Projection("model." + name, bias=True))
            self.register_buffer("prefix_mask", torch.ones(1, 4, dtype=torch.bool))

        def forward(self, noisy_actions, timestep, keys, values):
            cache = {index: {"key_states": keys[index], "value_states": values[index]} for index in range(expert_layers)}
            velocity = self.denoise_step(self.prefix_mask, cache, noisy_actions, timestep)
            updated = noisy_actions - 0.1 * velocity
            return velocity, updated, updated[:, :, :6]

    for name in ("embed_suffix", "denoise_step"):
        setattr(Denoiser, name, methods[("VLAFlowMatching", name)])
    module = Denoiser().eval()
    inputs = (torch.randn(1, 50, 32, generator=generator), torch.tensor([0.5]),
              torch.randn(expert_layers, 1, 4, 5, 64, generator=generator),
              torch.randn(expert_layers, 1, 4, 5, 64, generator=generator))
    scope = "One complete 16-layer expert denoiser invocation" if expert_layers == 16 else "Original expert layer zero followed by the original final norm"
    description = {"scope": scope + ", action projection, Euler update and action slice; prefix K/V tensors are external synthetic fixtures",
                   "complete_denoiser": expert_layers == 16, "vision_language_prefix_encoder_included": False,
                   "source_precision_selection": "Explicit FP32 conversion of selected original BF16/F32 tensors before W8A8",
                   "chunk_size": 50, "prefix_tokens": 4, "expert_layers": expert_layers, "expert_hidden_size": 720,
                   "query_heads": 15, "kv_heads": 5, "head_dim": 64, "intermediate_size": 2048,
                   "float_output_names": ["velocity", "updated_latent", "action_slice"], "integer_outputs": [],
                   "expected_gemmini_contractions": 7 * expert_layers + 4, "public_reference_methods": sources,
                   "reference_assembly": "Unmodified selected public method bodies re-executed with torch/nn/math globals; lightweight module tree and CPU dtype selection; original model constructors not called"}
    return module, inputs, description
