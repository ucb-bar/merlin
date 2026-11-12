"""Closed independent PyTorch compositions; no loader or model imports.

These source programs carry observable tensor computations. They establish no
target ownership, persistent invocation context, physical alias or FENV behavior.
Scalar parameters are validated by the ordinary builtin source renderer.
"""

BODIES = {
    "attention_residual_norm": """
class Model(nn.Module):
    def forward(self, q, k, v, residual, w, b):
        scores = (q @ k.transpose(-2, -1)) / math.sqrt({K})
        if {causal}:
            mask = torch.triu(torch.full((q.shape[-2], k.shape[-2]), torch.finfo(scores.dtype).min,
                                        dtype=scores.dtype), 1)
            scores = scores + mask
        attended = scores.softmax(-1) @ v
        return torch.nn.functional.layer_norm(attended + residual, ({Dv},), w, b, {eps})
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r({N}, {K}), _r({N}, {Dv}), _r({M}, {Dv}),
                     _r({Dv}) * 0.5 + 1.0, _r({Dv}) * 0.1)
""",
    "mlp_residual": """
class Model(nn.Module):
    def forward(self, x, up, down, bias):
        hidden = torch.nn.functional.gelu(x @ up + bias, approximate="tanh")
        return hidden @ down + x
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r({K}, {N}), _r({N}, {K}), _r({N}))
""",
    "conv_residual_pool": """
class Model(nn.Module):
    def forward(self, x, w, residual):
        convolved = torch.nn.functional.conv2d(x, w, stride=1)
        combined = torch.relu(convolved + residual)
        return combined.reshape(1, {N}, {Ho} // {Pool}, {Pool}, {Wo} // {Pool}, {Pool}).mean((3, 5))
def get_model_and_inputs():
    return Model(), (_r(1, {Cin}, {Himg}, {Wimg}), _r({N}, {Cin}, {P}, {P}), _r(1, {N}, {Ho}, {Wo}))
""",
    "producer_quantizer_observer": """
class Model(nn.Module):
    def forward(self, x, w):
        producer = x * {producer_scale}
        rounded = torch.round(producer / {quant_scale})
        quantized = torch.clamp(rounded, {quant_min}, {quant_max}).to(torch.int8)
        codes = quantized.to(producer.dtype)
        decoded = codes * {quant_scale}
        consumed = decoded @ w
        return producer, codes, decoded, consumed
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r({K}, {N}))
""",
}

INPUT_NAMES = {
    "attention_residual_norm": ["Q", "K", "V", "Residual", "W", "B"],
    "mlp_residual": ["X", "Up", "Down", "B"],
    "conv_residual_pool": ["X", "W", "Residual"],
    "producer_quantizer_observer": ["X", "W"],
}


def parameters(spec):
    """Derive the closed quantizer's software storage bounds and exact scales."""
    import math

    from merlin.common.quant_formats import get
    from merlin.development.targetgen.input_palette import scalar

    if get(spec.get("dtype", "fp32")).name != get("f32").name:
        raise ValueError("producer_quantizer_observer requires explicitly typed f32 source arithmetic")
    scales = {name: spec.get(name, 1.0) for name in ("producer_scale", "quant_scale")}
    for name, value in scales.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"builtin source field {name} must be an explicit positive finite scalar")
        scales[name] = scalar(value, "f32")
    if math.frexp(scales["quant_scale"])[0] != 0.5:
        raise ValueError("closed quantizer source requires an exact power-of-two quant_scale")
    bits = get("i8").element_bits
    return {**scales, "quant_min": -(1 << (bits - 1)), "quant_max": (1 << (bits - 1)) - 1}


# Only source structure is described here. Numerical scenario effects must be
# verified against complete source operands and independently recorded outputs.
SOURCE_EFFECTS = {
    "attention_residual_norm": {
        "attention_composition",
        "two_contraction_composition",
        "scale_composition",
        "residual_composition",
        "normalization_composition",
        "nonlinear_composition",
    },
    "mlp_residual": {
        "two_contraction_composition",
        "residual_composition",
        "nonlinear_composition",
        "multiple_consumers",
    },
    "conv_residual_pool": {
        "convolution_source_window",
        "residual_composition",
        "nonlinear_composition",
        "pooling_composition",
    },
    "producer_quantizer_observer": {
        "shared_producer",
        "multiple_consumers",
        "escaped_use",
        "producer_quantizer_observer",
        "complete_producer_consumer_publication",
    },
}


def source_effects(entry):
    """Concrete closed-body source mechanisms; execution effects stay separate."""
    effects = set(SOURCE_EFFECTS.get(entry["op"], ()))
    if entry["op"] == "attention_residual_norm" and entry.get("causal") is True:
        effects.add("causal_mask")
    return effects
