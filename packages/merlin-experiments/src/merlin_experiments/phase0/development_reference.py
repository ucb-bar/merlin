"""Independent bounded NumPy integer references for fresh development sources."""

from __future__ import annotations

from math import prod

import numpy as np

from . import development_sources as D


def inputs(declaration):
    if declaration.get("schema") == "merlin.phase0.public_parameter_compile_stress.v1":
        from .development_parameter_stress import inputs as parameter_inputs

        return parameter_inputs(declaration)
    if declaration.get("schema") == "merlin.phase0.public_attention_semantic_source.v1":
        from .development_attention import inputs as attention

        return attention(declaration)
    rows = D.describe(declaration)["inputs"]
    return [
        (((np.arange(prod(row["shape"]), dtype=np.int64) * (1 + 2 * index % 16) + index) % 17) - 8)
        .astype(np.int8)
        .reshape(row["shape"])
        for index, row in enumerate(rows)
    ]


def evaluate(declaration, values):
    """Compute every live-out in wide independent arithmetic before exact casts."""
    if declaration.get("schema") == "merlin.phase0.public_parameter_compile_stress.v1":
        from .development_parameter_stress import evaluate as parameter_evaluate

        return parameter_evaluate(declaration, values)
    if declaration.get("schema") == "merlin.phase0.public_attention_semantic_source.v1":
        from .development_attention import evaluate as attention

        return attention(declaration, values)
    pattern, g = declaration["pattern"], declaration["geometry"]
    shift = declaration["arithmetic"]["readout_shift"]

    def readout(value):
        return np.clip(np.floor_divide(value + 2 ** (shift - 1), 2**shift), -128, 127).astype(np.int8)

    wide = [np.asarray(value, dtype=np.int64) for value in values]
    if pattern.startswith("matmul"):
        lhs, rhs = wide
        result = [lhs @ (rhs.T if pattern.endswith("nk") else rhs)]
    elif pattern.startswith("conv_"):
        image, weight = wide[:2]
        oh = (g["H"] + 2 * g["PH"] - g["KH"]) // g["SH"] + 1
        ow = (g["W"] + 2 * g["PW"] - g["KW"]) // g["SW"] + 1
        padded = np.pad(image, ((0, 0), (g["PH"], g["PH"]), (g["PW"], g["PW"]), (0, 0)))
        weights = weight.reshape(g["KH"], g["KW"], g["CI"], g["CO"])
        spatial = np.zeros((1, oh, ow, g["CO"]), dtype=np.int64)
        # An independent convolution sum, not reuse of the source's im2col/cat.
        for h in range(oh):
            for w in range(ow):
                patch = padded[:, h * g["SH"] : h * g["SH"] + g["KH"], w * g["SW"] : w * g["SW"] + g["KW"], :]
                spatial[:, h, w, :] = np.einsum("bhwi,hwio->bo", patch, weights, dtype=np.int64)
        state = readout(spatial)
        result = [spatial.reshape(oh * ow, g["CO"])]
        if pattern == "conv_transition_nhwc":
            result += [state, state.reshape(oh * ow, g["CO"]).astype(np.int64) @ wide[2]]
    elif pattern == "prefix_action_recurrence":
        prefix, weight, state, state_weight = wide
        encoded = prefix @ weight
        cache = np.clip(readout(encoded).astype(np.int64).sum(0, keepdims=True), -128, 127).astype(np.int8)
        result = [encoded, cache]
        for _ in range(g["steps"]):
            update = state @ state_weight
            state = np.clip(readout(update).astype(np.int64) + state + cache.astype(np.int64), -128, 127)
            result += [update, state.copy()]
        result.append(state[: g["decode"]].copy())
    elif pattern == "vision_prefix_action":
        from .development_composition import constituents

        convolution, action = constituents(declaration)
        first = evaluate(convolution, values[:3])
        prefix = readout(first[-1])
        result = [*first, *evaluate(action, [prefix, *values[3:]])]
    elif pattern == "resident_rhs_reuse":
        result = [lhs @ wide[-1] for lhs in wide[:-1]]
    elif pattern == D.PATTERN:
        cache_k, cache_v = wide[:2]
        result = []
        for step in range(g["steps"]):
            q, k, v = wide[2 + step * 3 : 5 + step * 3]
            cache_k, cache_v = np.concatenate((cache_k, k), axis=1), np.concatenate((cache_v, v), axis=1)
            for head in range(g["query_heads"]):
                group = head // (g["query_heads"] // g["kv_heads"])
                scores = q[head] @ cache_k[group].T
                result += [scores, readout(scores).astype(np.int64) @ cache_v[group]]
            result += [cache_k.copy(), cache_v.copy()]
    else:
        raise ValueError("no independent reference for this public pattern")
    description = D.describe(declaration)
    if len(result) != len(description["outputs"]):
        raise ValueError("independent reference lost complete live-outs")
    outputs = []
    for row, value in zip(description["outputs"], result, strict=True):
        if list(value.shape) != row["shape"]:
            raise ValueError("independent result geometry differs from the complete source ABI")
        limit = np.iinfo(row["dtype"])
        if np.any(value < limit.min) or np.any(value > limit.max):
            raise ValueError("independent source reference exceeds its exact declared storage")
        outputs.append(value.astype(row["dtype"]))
    return outputs
