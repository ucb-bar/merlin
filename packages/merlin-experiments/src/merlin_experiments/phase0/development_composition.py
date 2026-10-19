"""Closed vision-to-prefix-to-action composition with complete public live-outs."""

from __future__ import annotations

import ast
from math import prod

from . import development_patterns as P
from . import development_sources as D


def constituents(declaration):
    if (
        not isinstance(declaration, dict)
        or set(declaration) != {"schema", "scope", "name", "pattern", "geometry", "arithmetic", "max_logical_elements"}
        or declaration["schema"] != P.SCHEMA
        or declaration["scope"] != "public_development"
        or declaration["pattern"] != "vision_prefix_action"
        or not D.workload_roster._label(declaration["name"])
        or not isinstance(declaration["geometry"], dict)
        or set(declaration["geometry"]) != {"vision", "action"}
    ):
        raise ValueError("vision/action composition needs an explicit closed public declaration")
    base = {key: value for key, value in declaration.items() if key not in {"pattern", "geometry"}}
    convolution = {**base, "pattern": "conv_transition_nhwc", "geometry": declaration["geometry"]["vision"]}
    action = {**base, "pattern": "prefix_action_recurrence", "geometry": declaration["geometry"]["action"]}
    c, a = P.describe(convolution), P.describe(action)
    if c["outputs"][-1]["shape"] != a["inputs"][0]["shape"]:
        raise ValueError("complete vision output and recurrent prefix input must join exactly")
    return convolution, action


def describe(declaration):
    convolution, action = constituents(declaration)
    c, a = P.describe(convolution), P.describe(action)
    inputs, outputs = c["inputs"] + a["inputs"][1:], c["outputs"] + a["outputs"]
    elements = sum(prod(row["shape"]) for row in inputs + outputs)
    if elements > declaration["max_logical_elements"]:
        raise ValueError("composed complete public source exceeds its logical element budget")
    return {
        **a,
        "declaration": declaration,
        "declaration_sha256": D._digest(declaration),
        "inputs": inputs,
        "outputs": outputs,
        "logical_elements": elements,
        "scalar_products": c["scalar_products"] + a["scalar_products"],
        "contractions": c["contractions"] + a["contractions"],
    }


def render(declaration):
    convolution, action = constituents(declaration)
    description = describe(declaration)

    def body(source):
        tree = ast.parse(source)
        forward = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "forward")
        return [ast.unparse(node) for node in forward.body if not isinstance(node, ast.Return)]

    shift = declaration["arithmetic"]["readout_shift"]
    statements = [
        *body(P.render(convolution)),
        f"prefix = _saturate_i8(torch.div(point_wide + {2 ** (shift - 1)}, {2**shift}, rounding_mode='floor'))",
        *body(P.render(action)),
    ]
    names = [row["name"] for row in description["inputs"]]
    lines = [
        '"""Independent public vision/prefix/action source, not a full model."""',
        "import torch",
        "",
        *D._integer_saturation_source(),
        f"# Declaration: {description['declaration_sha256']}",
        "class Program(torch.nn.Module):",
        "    def forward(self," + ",".join(names) + "):",
    ]
    lines += ["        " + line for statement in statements for line in statement.splitlines()]
    lines += [
        "        return (" + ",".join(row["name"] for row in description["outputs"]) + ",)",
        "",
        "def get_model_and_inputs():",
    ]
    for index, row in enumerate(description["inputs"]):
        lines.append(
            f"    {row['name']} = (((torch.arange({prod(row['shape'])},dtype=torch.int64)"
            f"*{1 + 2 * index % 16}+{index})%17)-8).to(torch.int8).reshape({row['shape']!r})"
        )
    lines.append("    return Program().eval(), (" + ",".join(names) + ",)")
    return "\n".join(lines) + "\n"
