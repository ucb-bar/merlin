"""Explicit shared-RHS source uses; no assertion of physical device residency."""

from __future__ import annotations

from math import prod

from . import development_patterns as P
from . import development_sources as D


def describe(declaration):
    geometry = declaration.get("geometry")
    if (
        not isinstance(geometry, dict)
        or set(geometry) != {"M", "K", "N"}
        or not isinstance(geometry["M"], list)
        or not 1 <= len(geometry["M"]) <= 64
        or any(type(m) is not int or not 1 <= m <= 65536 for m in geometry["M"])
    ):
        raise ValueError("shared RHS source needs an explicit bounded complete M roster")
    base = {**declaration, "pattern": "matmul", "geometry": {**geometry, "M": geometry["M"][0]}}
    original = P.describe(base)
    inputs = [
        {"name": f"lhs{index}", "shape": [m, geometry["K"]], "dtype": "int8"} for index, m in enumerate(geometry["M"])
    ] + [{"name": "shared_rhs", "shape": [geometry["K"], geometry["N"]], "dtype": "int8"}]
    outputs = [
        {"name": f"wide{index}", "shape": [m, geometry["N"]], "dtype": "int32"} for index, m in enumerate(geometry["M"])
    ]
    elements = sum(prod(row["shape"]) for row in inputs + outputs)
    if elements > declaration["max_logical_elements"]:
        raise ValueError("complete RHS reuse source exceeds its element budget")
    return {
        **original,
        "declaration": declaration,
        "declaration_sha256": D._digest(declaration),
        "inputs": inputs,
        "outputs": outputs,
        "logical_elements": elements,
        "scalar_products": sum(geometry["M"]) * geometry["K"] * geometry["N"],
        "contractions": len(geometry["M"]),
    }


def render(declaration):
    description = describe(declaration)
    names = [row["name"] for row in description["inputs"]]
    lines = [
        '"""Fresh shared RHS source; physical retention requires target evidence."""',
        "import torch",
        "",
        f"# Declaration: {description['declaration_sha256']}",
        "class Program(torch.nn.Module):",
        "    def forward(self," + ",".join(names) + "):",
    ]
    for index in range(description["contractions"]):
        lines.append(f"        wide{index} = torch._int_mm(lhs{index},shared_rhs)")
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
