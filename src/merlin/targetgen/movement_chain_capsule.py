"""Movement chains: an aten op that is a short program of the declared movement commands, one interface program.

Three torch ops are NOT a single movement command; each is a composition of commands the movement row already
declares (``slice`` / ``reshape`` movement forms, ``insert_slice``), so each is written as the composition:

* ``select.int`` (``select``): slice ONE index of one dim (size 1, rank kept), then a rank-reducing ``reshape``
  that DROPS the selected dim, as ``aten.select`` does;
* ``split_with_sizes`` (``split``): the LIST output, one ``slice`` per chunk, the chunks partitioning one dim with
  different sizes (a multi-output program, as ``conv_backward_spatial``);
* ``select_scatter`` (``select_scatter``): the rank-reduced source (the dim ``select`` drops) is ``reshape``d to size 1
  in that dim, then ``insert_slice``d at ONE index of the destination.

The device work is the same ``copy`` primitive for every command (row-contiguous MVIN / MVOUT with an arbitrary row
stride, ``LoadController.scala:76,102,105``); a ``reshape`` holds the same bytes in the same order. The golden composes
``merlin.runtime.integer_ops.move`` / ``insert_slice`` in program order; it has no arithmetic of its own.

A stage is ``{"op": "movement" | "insert_slice", "out", "srcs", "attrs", "shape"}``; ``outs`` names the graded results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .corpus_spec import CorpusBinding

KINDS = ("select", "split", "select_scatter")


def _split_sizes(extent: int) -> list[int]:
    """Chunk sizes partitioning ``extent`` (>= 2): three different sizes when extent >= 6, else [1, extent - 1]."""
    if extent >= 6:
        a, b = extent // 5, extent // 3
        return [a, b, extent - a - b]
    return [1, extent - 1]


def _slice_attrs(shape: list[int], dim: int, offset: int, size: int) -> dict[str, Any]:
    return {
        "form": "slice",
        "offsets": [offset if d == dim else 0 for d in range(len(shape))],
        "sizes": [size if d == dim else shape[d] for d in range(len(shape))],
        "strides": [1] * len(shape),
    }


def chain_plan(kind: str, shape: list[int]) -> dict[str, Any]:
    """``{"inputs": [(name, shape, role)], "stages": [...], "outs": [...]}`` for ``kind`` over a ``[m, n]`` source."""
    m, n = shape
    if kind == "select":
        idx = m // 2  # an interior row of dim 0
        return {
            "inputs": [("X", [m, n])],
            "stages": [
                {"op": "movement", "out": "sel", "srcs": ["X"], "attrs": _slice_attrs([m, n], 0, idx, 1), "shape": [1, n]},
                {"op": "movement", "out": "Y0", "srcs": ["sel"], "attrs": {"form": "reshape"}, "shape": [n]},
            ],
            "outs": ["Y0"],
        }
    if kind == "split":
        # split along dim 0, or along dim 1 when dim 0 is too short to give two chunks of different sizes
        dim = 0 if m >= 3 or n < 3 else 1
        extent = shape[dim]
        if extent < 2:
            raise ValueError(f"split_with_sizes needs a dim of extent >= 2, got shape {shape}")
        sizes, stages, at = _split_sizes(extent), [], 0
        for k, size in enumerate(sizes):
            out_shape = [size if d == dim else shape[d] for d in range(2)]
            stages.append({"op": "movement", "out": f"Y{k}", "srcs": ["X"], "attrs": _slice_attrs([m, n], dim, at, size), "shape": out_shape})
            at += size
        return {"inputs": [("X", [m, n])], "stages": stages, "outs": [f"Y{k}" for k in range(len(sizes))], "dim": dim, "sizes": sizes}
    if kind == "select_scatter":
        idx = m // 2
        return {
            "inputs": [("D", [m, n]), ("S", [n])],
            "stages": [
                {"op": "movement", "out": "S1", "srcs": ["S"], "attrs": {"form": "reshape"}, "shape": [1, n]},
                {
                    "op": "insert_slice", "out": "Y0", "srcs": ["D", "S1"],
                    "attrs": {"offsets": [idx, 0], "strides": [1, 1]}, "shape": [m, n],
                },
            ],
            "outs": ["Y0"],
        }
    raise ValueError(f"movement chain kind must be one of {KINDS}, got {kind!r}")


def evaluate_stages(stages: list[dict], env: dict) -> dict:
    """Run the stages in program order on the exact integer semantics; returns ``env`` extended with every stage."""
    from merlin.runtime import integer_ops

    env = dict(env)
    for st in stages:
        srcs = [env[s] for s in st["srcs"]]
        if st["op"] == "movement":
            env[st["out"]] = integer_ops.move(srcs[0], dict(st["attrs"]), list(st["shape"]), op=f"movement {st['out']!r}")
        elif st["op"] == "insert_slice":
            env[st["out"]] = integer_ops.insert_slice(srcs[0], srcs[1], dict(st["attrs"]), op=f"insert_slice {st['out']!r}")
        else:
            raise ValueError(f"unknown movement-chain stage op {st['op']!r}")
    return env


def build_movement_chain(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    from .corpus_spec import _MOVE_FORM_KEYS, _iface_prelude, _mlir_attr, _numeric_policy, _tensor_type

    plan = chain_plan(entry["chain"], list(entry["shape"]))
    idt, midt = binding.cap_dtype(binding.operand_dtype), binding.mlir_dtype(binding.operand_dtype)
    outs = plan["outs"]
    shapes = {name: list(sh) for name, sh in plan["inputs"]}
    shapes.update({st["out"]: list(st["shape"]) for st in plan["stages"]})
    attrs = {
        "chain": entry["chain"], "stages": plan["stages"], "outs": outs, "out": outs[0],
        "output_dtype": idt, "semantic": "movement_chain",
    }  # fmt: skip
    for key in ("dim", "sizes"):
        if key in plan:
            attrs[key] = plan[key]
    cap = {
        "name": entry["name"],
        "kind": entry["kind"],
        "source_role": entry["source_role"],
        "source_reference": entry["source_reference"],
        "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [{"name": n, "role": "input", "shape": list(sh), "dtype": idt} for n, sh in plan["inputs"]],
        "operation": {"op": "movement_chain", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, binding.operand_dtype, None),
        "expected": {
            "instruction_classes": binding.classes_for(op="movement", movement=True),
            "modes": {"movement": True},
        },
        "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional",
        "firesim": "optional",
    }
    lines = _iface_prelude(binding.target, entry.get("comment", ""))
    lines += [
        f'  %{n} = merlin_iface.tensor {{name = "{n}", role = "input"}} : {_tensor_type(sh, midt)}'
        for n, sh in plan["inputs"]
    ]
    for st in plan["stages"]:
        a, srcs = st["attrs"], st["srcs"]
        sig = ", ".join(_tensor_type(shapes[s], midt) for s in srcs)
        operands = ", ".join("%" + s for s in srcs)
        if st["op"] == "movement":
            body = f'form = "{a["form"]}", ' + "".join(
                f"{k} = {_mlir_attr(a[k])}, " for k in _MOVE_FORM_KEYS[a["form"]] if k in a
            )
        else:
            body = f"offsets = {_mlir_attr(a['offsets'])}, strides = {_mlir_attr(a['strides'])}, "
        lines.append(
            f'  %{st["out"]} = merlin_iface.{st["op"]} {operands} {{name = "{st["out"]}", {body}output_dtype = "{idt}"}} '
            f": ({sig}) -> {_tensor_type(st['shape'], midt)}"
        )
    lines.append("}")
    return cap, "\n".join(lines) + "\n"
