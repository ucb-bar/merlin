"""Public ownership and escaping-layout controls with complete exact live-outs.

These source semantics exercise legality decisions. Physical residency, queue
occupancy, bank conflicts and overlap require independent emitted-effect evidence.
"""

from __future__ import annotations

import ast
from math import prod

import numpy as np

from . import development_patterns as P
from . import development_sources as D

SCHEMA = "merlin.phase0.public_development_control.v1"
_RHS_MODES = {"shared", "distinct_equal", "distinct_changed", "updated_epoch", "intervening_clobber"}


def _base(declaration):
    if (
        not isinstance(declaration, dict)
        or set(declaration)
        != {"schema", "scope", "name", "pattern", "geometry", "arithmetic", "max_logical_elements", "controls"}
        or declaration["schema"] != SCHEMA
        or declaration["scope"] != "public_development"
        or not D.workload_roster._label(declaration["name"])
        or not isinstance(declaration["geometry"], dict)
        or declaration["pattern"] not in {"rhs_versions", "strided_conv_escape", "kv_epoch"}
    ):
        raise ValueError("ownership controls need an explicit public source declaration")
    return {key: value for key, value in declaration.items() if key != "controls"} | {"schema": P.SCHEMA}


def describe(declaration):
    base = _base(declaration)
    controls, geometry = declaration["controls"], declaration["geometry"]
    if declaration["pattern"] == "kv_epoch":
        if (
            not isinstance(controls, dict)
            or set(controls) != {"epoch"}
            or controls["epoch"] not in {"append_only", "replace_prefix"}
            or geometry.get("steps") != 2
        ):
            raise ValueError("KV epoch control requires two complete steps and an explicit state replacement selection")
        original = D.describe({**base, "schema": D.SCHEMA, "pattern": D.PATTERN})
        inputs, outputs = original["inputs"][:], original["outputs"][:]
        if controls["epoch"] == "replace_prefix":
            shape = [geometry["kv_heads"], geometry["prefix_tokens"] + geometry["query_tokens"], geometry["head_dim"]]
            inputs += [{"name": f"delta_{axis}", "shape": shape, "dtype": "int8"} for axis in ("k", "v")]
            outputs += [{"name": f"epoch_cache_{axis}", "shape": shape, "dtype": "int8"} for axis in ("k", "v")]
        work, contractions = original["scalar_products"], original["contractions"]
    elif declaration["pattern"] == "rhs_versions":
        if (
            not isinstance(controls, dict)
            or set(controls) != {"mode"}
            or controls["mode"] not in _RHS_MODES
            or not isinstance(geometry, dict)
            or set(geometry) != {"M", "K", "N"}
            or not isinstance(geometry["M"], list)
            or not 2 <= len(geometry["M"]) <= 8
            or any(type(m) is not int or not 1 <= m <= 65536 for m in geometry["M"])
        ):
            raise ValueError("RHS controls require two to eight explicit uses and one supported ownership mode")
        original = P.describe({**base, "pattern": "matmul", "geometry": {**geometry, "M": geometry["M"][0]}})
        inputs = [
            {"name": f"lhs{i}", "shape": [m, geometry["K"]], "dtype": "int8"} for i, m in enumerate(geometry["M"])
        ]
        rhs = {"shape": [geometry["K"], geometry["N"]], "dtype": "int8"}
        inputs += [{"name": "rhs0", **rhs}]
        if controls["mode"] != "shared":
            inputs += [{"name": "delta" if controls["mode"] == "updated_epoch" else "rhs1", **rhs}]
        outputs = [
            {"name": f"wide{i}", "shape": [m, geometry["N"]], "dtype": "int32"} for i, m in enumerate(geometry["M"])
        ]
        if controls["mode"] == "updated_epoch":
            outputs += [{"name": "rhs_epoch", **rhs}]
        work, contractions = sum(geometry["M"]) * geometry["K"] * geometry["N"], len(geometry["M"])
        if controls["mode"] == "intervening_clobber":
            inputs += [{"name": "clobber_lhs", "shape": [geometry["M"][0], geometry["K"]], "dtype": "int8"}]
            outputs += [{"name": "clobber_wide", "shape": [geometry["M"][0], geometry["N"]], "dtype": "int32"}]
            work += geometry["M"][0] * geometry["K"] * geometry["N"]
            contractions += 1
    else:
        if (
            not isinstance(controls, dict)
            or set(controls) != {"source_stride", "escape_packing"}
            or type(controls["source_stride"]) is not int
            or not 1 <= controls["source_stride"] <= 4
            or type(controls["escape_packing"]) is not bool
        ):
            raise ValueError("layout controls require an explicit source stride and packing live-out selection")
        original = P.describe({**base, "pattern": "conv_nhwc"})
        inputs = [dict(row, shape=list(row["shape"])) for row in original["inputs"]]
        inputs[0]["shape"][1] = (geometry["H"] - 1) * controls["source_stride"] + 1
        outputs = original["outputs"][:]
        if controls["escape_packing"]:
            rows = outputs[0]["shape"][0]
            outputs += [
                {"name": "patches", "shape": [rows, geometry["KH"] * geometry["KW"] * geometry["CI"]], "dtype": "int8"}
            ]
        work, contractions = original["scalar_products"], original["contractions"]
    elements = sum(prod(row["shape"]) for row in inputs + outputs)
    if elements > declaration["max_logical_elements"]:
        raise ValueError("complete control source exceeds its explicit logical element budget")
    return {
        **original,
        "declaration": declaration,
        "declaration_sha256": D._digest(declaration),
        "inputs": inputs,
        "outputs": outputs,
        "logical_elements": elements,
        "scalar_products": work,
        "contractions": contractions,
        "control_scope": "functional SSA ownership and layout semantics; no physical scheduling witness",
        "missing": [
            "physical_alias_mutation",
            "device_residency",
            "queue_occupancy",
            "bank_mapping",
            "overlap",
            "target_execution",
            "timing",
            "full_model",
        ],
    }


def _statements(source):
    tree = ast.parse(source)
    forward = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "forward")
    return [ast.unparse(node) for node in forward.body if not isinstance(node, ast.Return)]


def render(declaration):
    description = describe(declaration)
    controls = declaration["controls"]
    names = [row["name"] for row in description["inputs"]]
    if declaration["pattern"] == "kv_epoch":
        base = {**_base(declaration), "schema": D.SCHEMA, "pattern": D.PATTERN}
        body = []
        for statement in _statements(D.render(base)):
            body += [statement]
            node = ast.parse(statement).body[0]
            if (
                controls["epoch"] == "replace_prefix"
                and isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "cache_v"
                and isinstance(node.value, ast.Name)
                and node.value.id == "cache_v0"
            ):
                for axis in ("k", "v"):
                    body += [
                        f"epoch_cache_{axis} = _saturate_i8(cache_{axis}.to(torch.int32)"
                        f" + delta_{axis}.to(torch.int32))",
                        f"cache_{axis} = epoch_cache_{axis}",
                    ]
    elif declaration["pattern"] == "rhs_versions":
        mode = controls["mode"]
        body = (
            ["rhs_epoch = _saturate_i8(rhs0.to(torch.int32) + delta.to(torch.int32))"]
            if mode == "updated_epoch"
            else []
        )
        for i in range(len(declaration["geometry"]["M"])):
            rhs = (
                "rhs0"
                if i == 0 or mode in {"shared", "intervening_clobber"}
                else ("rhs_epoch" if mode == "updated_epoch" else "rhs1")
            )
            body += [f"wide{i} = torch._int_mm(lhs{i}, {rhs})"]
            if i == 0 and mode == "intervening_clobber":
                body += ["clobber_wide = torch._int_mm(clobber_lhs, rhs1)"]
    else:
        base = {key: value for key, value in _base(declaration).items()}
        body = [
            f"image = image[:, ::{controls['source_stride']}, :, :]",
            *_statements(P.render({**base, "pattern": "conv_nhwc"})),
        ]
    lines = [
        '"""Public source legality controls; no physical scheduling qualification."""',
        "import torch",
        "",
        *D._integer_saturation_source(),
        f"# Declaration: {description['declaration_sha256']}",
        "class Program(torch.nn.Module):",
        "    def forward(self," + ",".join(names) + "):",
    ]
    lines += ["        " + line for statement in body for line in statement.splitlines()]
    lines += [
        "        return (" + ",".join(row["name"] for row in description["outputs"]) + ",)",
        "",
        "def get_model_and_inputs():",
    ]
    for index, row in enumerate(description["inputs"]):
        if declaration["pattern"] == "rhs_versions" and controls["mode"] == "distinct_equal" and row["name"] == "rhs1":
            lines += ["    rhs1 = rhs0.clone()"]
        else:
            lines += [
                f"    {row['name']} = (((torch.arange({prod(row['shape'])},dtype=torch.int64)"
                f"*{1 + 2 * index % 16}+{index})%17)-8).to(torch.int8).reshape({row['shape']!r})"
            ]
    lines += ["    return Program().eval(), (" + ",".join(names) + ",)"]
    return "\n".join(lines) + "\n"


def inputs(declaration):
    rows = describe(declaration)["inputs"]
    values = [
        (((np.arange(prod(row["shape"]), dtype=np.int64) * (1 + 2 * i % 16) + i) % 17) - 8)
        .astype(np.int8)
        .reshape(row["shape"])
        for i, row in enumerate(rows)
    ]
    if declaration["pattern"] == "rhs_versions" and declaration["controls"]["mode"] == "distinct_equal":
        names = [row["name"] for row in rows]
        values[names.index("rhs1")] = values[names.index("rhs0")].copy()
    return values


def evaluate(declaration, values):
    from . import development_reference as R

    description = describe(declaration)
    by_name = dict(zip((row["name"] for row in description["inputs"]), values, strict=True))
    if declaration["pattern"] == "kv_epoch":
        base = {**_base(declaration), "schema": D.SCHEMA, "pattern": D.PATTERN}
        if declaration["controls"]["epoch"] == "append_only":
            result = R.evaluate(base, values)
        else:
            g = declaration["geometry"]
            first = {**base, "geometry": {**g, "steps": 1}}
            initial = R.evaluate(first, values[:5])
            epoch = [
                np.clip(value.astype(np.int64) + delta.astype(np.int64), -128, 127).astype(np.int8)
                for value, delta in zip(initial[-2:], values[-2:], strict=True)
            ]
            second = {**first, "geometry": {**g, "steps": 1, "prefix_tokens": g["prefix_tokens"] + g["query_tokens"]}}
            result = [*initial, *R.evaluate(second, [*epoch, *values[5:8]]), *epoch]
    elif declaration["pattern"] == "rhs_versions":
        mode = declaration["controls"]["mode"]
        rhs0 = by_name["rhs0"].astype(np.int64)
        epoch = np.clip(rhs0 + by_name["delta"].astype(np.int64), -128, 127) if mode == "updated_epoch" else None
        result = []
        for i in range(len(declaration["geometry"]["M"])):
            rhs = (
                rhs0
                if i == 0 or mode in {"shared", "intervening_clobber"}
                else (epoch if mode == "updated_epoch" else by_name["rhs1"].astype(np.int64))
            )
            result += [by_name[f"lhs{i}"].astype(np.int64) @ rhs]
        if epoch is not None:
            result += [epoch]
        if mode == "intervening_clobber":
            result += [by_name["clobber_lhs"].astype(np.int64) @ by_name["rhs1"].astype(np.int64)]
    else:
        g, controls = declaration["geometry"], declaration["controls"]
        image = values[0][:, :: controls["source_stride"], :, :]
        result = R.evaluate({**_base(declaration), "pattern": "conv_nhwc"}, [image, values[1]])
        if controls["escape_packing"]:
            padded = np.pad(image, ((0, 0), (g["PH"], g["PH"]), (g["PW"], g["PW"]), (0, 0)))
            oh, ow = (
                (g[axis] + 2 * g[pad] - g[kernel]) // g[stride] + 1
                for axis, pad, kernel, stride in (("H", "PH", "KH", "SH"), ("W", "PW", "KW", "SW"))
            )
            patches = [
                padded[0, h * g["SH"] : h * g["SH"] + g["KH"], w * g["SW"] : w * g["SW"] + g["KW"], :].reshape(-1)
                for h in range(oh)
                for w in range(ow)
            ]
            result += [np.stack(patches)]
    return [value.astype(row["dtype"]) for value, row in zip(result, description["outputs"], strict=True)]
