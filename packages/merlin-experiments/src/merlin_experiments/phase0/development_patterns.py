"""Typed public integer source patterns; no target or model policy selection."""

from __future__ import annotations

from math import prod

from . import development_sources as D

SCHEMA = "merlin.phase0.public_development_source.v2"
PATTERNS = {"matmul", "matmul_rhs_nk", "conv_nhwc", "conv_transition_nhwc", "prefix_action_recurrence"}


def describe(declaration):
    if isinstance(declaration, dict) and declaration.get("pattern") == "resident_rhs_reuse":
        from .development_reuse import describe as reused

        return reused(declaration)
    if isinstance(declaration, dict) and declaration.get("pattern") == "vision_prefix_action":
        from .development_composition import describe as composed

        return composed(declaration)
    if (
        not isinstance(declaration, dict)
        or set(declaration) != {"schema", "scope", "name", "pattern", "geometry", "arithmetic", "max_logical_elements"}
        or declaration["schema"] != SCHEMA
        or declaration["scope"] != "public_development"
        or not D.workload_roster._label(declaration["name"])
        or declaration["pattern"] not in PATTERNS
    ):
        raise ValueError("select an explicit public typed development pattern")
    g, arithmetic, pattern = declaration["geometry"], declaration["arithmetic"], declaration["pattern"]
    fields = (
        {"M", "K", "N"}
        if pattern.startswith("matmul")
        else {"H", "W", "CI", "CO", "KH", "KW", "SH", "SW", "PH", "PW"}
        | ({"OUT"} if pattern == "conv_transition_nhwc" else set())
        if pattern.startswith("conv_")
        else {"prefix", "hidden", "action", "chunk", "steps", "decode"}
    )
    if (
        not isinstance(g, dict)
        or set(g) != fields
        or any(
            type(value) is not int or not (0 if key in {"PH", "PW"} else 1) <= value <= 65536
            for key, value in g.items()
        )
        or not isinstance(arithmetic, dict)
        or set(arithmetic) != {"contract", "readout_shift"}
        or arithmetic["contract"] != D.CONTRACT
        or type(arithmetic["readout_shift"]) is not int
        or not 1 <= arithmetic["readout_shift"] <= 30
        or type(declaration["max_logical_elements"]) is not int
        or not 1 <= declaration["max_logical_elements"] <= 2**32
    ):
        raise ValueError("public source needs bounded geometry and the exact declared integer contract")
    inputs, outputs, work, contractions, reductions = [], [], 0, 0, []

    def value(name, shape, dtype="int8"):
        return {"name": name, "shape": shape, "dtype": dtype}

    if pattern.startswith("matmul"):
        m, k, n = (g[key] for key in ("M", "K", "N"))
        inputs = [value("lhs", [m, k]), value("rhs", [n, k] if pattern.endswith("nk") else [k, n])]
        outputs, work, contractions, reductions = [value("wide", [m, n], "int32")], m * k * n, 1, [k]
    elif pattern.startswith("conv_"):
        if max(g["KH"], g["KW"]) > 32:
            raise ValueError("public spatial constructor has a bounded 32-position kernel per axis")
        oh = (g["H"] + 2 * g["PH"] - g["KH"]) // g["SH"] + 1
        ow = (g["W"] + 2 * g["PW"] - g["KW"]) // g["SW"] + 1
        if min(oh, ow) < 1:
            raise ValueError("convolution has no complete output")
        inputs = [
            value("image", [1, g["H"], g["W"], g["CI"]]),
            value("spatial_weight", [g["KH"] * g["KW"] * g["CI"], g["CO"]]),
        ]
        outputs = [value("spatial_wide", [oh * ow, g["CO"]], "int32")]
        reductions = [g["KH"] * g["KW"] * g["CI"]]
        work, contractions = oh * ow * reductions[0] * g["CO"], 1
        if pattern == "conv_transition_nhwc":
            inputs += [value("point_weight", [g["CO"], g["OUT"]])]
            outputs += [value("spatial_state", [1, oh, ow, g["CO"]]), value("point_wide", [oh * ow, g["OUT"]], "int32")]
            reductions += [g["CO"]]
            work, contractions = work + oh * ow * g["CO"] * g["OUT"], 2
    else:
        if g["steps"] > 4 or g["decode"] > g["chunk"]:
            raise ValueError("public recurrence depth/decode must preserve its finite state")
        inputs = [
            value("prefix", [g["prefix"], g["hidden"]]),
            value("prefix_weight", [g["hidden"], g["action"]]),
            value("state", [g["chunk"], g["action"]]),
            value("state_weight", [g["action"], g["action"]]),
        ]
        outputs = [value("prefix_wide", [g["prefix"], g["action"]], "int32"), value("prefix_cache", [1, g["action"]])]
        for step in range(g["steps"]):
            outputs += [
                value(f"action_wide{step}", [g["chunk"], g["action"]], "int32"),
                value(f"action_state{step}", [g["chunk"], g["action"]]),
            ]
        outputs.append(value("decoded", [g["decode"], g["action"]]))
        work = g["prefix"] * g["hidden"] * g["action"] + g["steps"] * g["chunk"] * g["action"] ** 2
        contractions, reductions = 1 + g["steps"], [g["hidden"], g["action"], g["prefix"]]
    # Prefix row sums operate on bounded i8 readouts; all products and the
    # rounding offset fit signed i32 for arbitrary declared input bytes.
    if max(reductions) * 128**2 + 2 ** (arithmetic["readout_shift"] - 1) > 2**31 - 1:
        raise ValueError("public integer program can overflow its declared accumulator")
    elements = sum(prod(row["shape"]) for row in inputs + outputs)
    if elements > declaration["max_logical_elements"]:
        raise ValueError("complete public source exceeds its declared logical element ceiling")
    return {
        "declaration": declaration,
        "declaration_sha256": D._digest(declaration),
        "inputs": inputs,
        "outputs": outputs,
        "logical_elements": elements,
        "scalar_products": work,
        "contractions": contractions,
        "status": "public_source_only_not_model_or_accelerator_qualification",
        "missing": ["target_execution", "timing", "full_model", "softmax", "normalization"],
    }


def render(declaration):
    if isinstance(declaration, dict) and declaration.get("pattern") == "resident_rhs_reuse":
        from .development_reuse import render as reused

        return reused(declaration)
    if isinstance(declaration, dict) and declaration.get("pattern") == "vision_prefix_action":
        from .development_composition import render as composed

        return composed(declaration)
    description = describe(declaration)
    g, pattern = declaration["geometry"], declaration["pattern"]
    shift = declaration["arithmetic"]["readout_shift"]

    def readout(expression):
        return f"_saturate_i8(torch.div(({expression}) + {2 ** (shift - 1)}, {2**shift}, rounding_mode='floor'))"

    body = []
    if pattern.startswith("matmul"):
        rhs = "rhs.transpose(0,1).contiguous()" if pattern.endswith("nk") else "rhs"
        body = [f"wide = torch._int_mm(lhs, {rhs})"]
    elif pattern.startswith("conv_"):
        oh = (g["H"] + 2 * g["PH"] - g["KH"]) // g["SH"] + 1
        ow = (g["W"] + 2 * g["PW"] - g["KW"]) // g["SW"] + 1
        body = [f"padded = torch.nn.functional.pad(image, (0,0,{g['PW']},{g['PW']},{g['PH']},{g['PH']}))"]
        parts = []
        for kh in range(g["KH"]):
            for kw in range(g["KW"]):
                parts.append(f"padded[:,{kh}:{kh + oh * g['SH']}:{g['SH']},{kw}:{kw + ow * g['SW']}:{g['SW']},:]")
        body += [
            f"patches = torch.cat(({','.join(parts)},), dim=-1).reshape({oh * ow},{g['KH'] * g['KW'] * g['CI']})",
            "spatial_wide = torch._int_mm(patches, spatial_weight)",
        ]
        if pattern == "conv_transition_nhwc":
            body += [
                f"spatial_state = {readout('spatial_wide')}.reshape(1,{oh},{ow},{g['CO']})",
                f"point_wide = torch._int_mm(spatial_state.reshape({oh * ow},{g['CO']}), point_weight)",
            ]
    else:
        body = [
            "prefix_wide = torch._int_mm(prefix, prefix_weight)",
            f"prefix_cache = _saturate_i8({readout('prefix_wide')}.to(torch.int32)"
            ".sum(0,keepdim=True,dtype=torch.int32))",
            "current = state",
        ]
        for step in range(g["steps"]):
            body += [
                f"action_wide{step} = torch._int_mm(current, state_weight)",
                f"action_state{step} = _saturate_i8({readout(f'action_wide{step}')}.to(torch.int32)"
                " + current.to(torch.int32)"
                " + prefix_cache.to(torch.int32))",
                f"current = action_state{step}",
            ]
        body += [f"decoded = current[:{g['decode']}]"]
    names = [row["name"] for row in description["inputs"]]
    lines = [
        '"""Fresh explicit public integer source; no model/timing qualification."""',
        "import torch",
        "",
        *D._integer_saturation_source(),
        f"# Declaration: {description['declaration_sha256']}",
        "class Program(torch.nn.Module):",
        "    def forward(self, " + ",".join(names) + "):",
    ]
    lines += ["        " + line for line in body]
    lines += [
        "        return (" + ",".join(row["name"] for row in description["outputs"]) + ",)",
        "",
        "def get_model_and_inputs():",
    ]
    for ordinal, row in enumerate(description["inputs"]):
        lines.append(
            f"    {row['name']} = (((torch.arange({prod(row['shape'])},dtype=torch.int64)"
            f"*{1 + (2 * ordinal) % 16}+{ordinal})%17)-8).to(torch.int8).reshape({row['shape']!r})"
        )
    lines.append("    return Program().eval(), (" + ",".join(names) + ",)")
    return "\n".join(lines) + "\n"
