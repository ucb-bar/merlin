"""A source-stated pooling region implemented by movement and scaled readout."""

import math

from .contract.interface_emit import emit_interface_mlir


def build(entry, binding):
    from .corpus_spec import _numeric_policy, _pool_epilogue

    if binding.cap_dtype(binding.operand_dtype) != "i8":
        raise ValueError("maxpool2d currently states signed INT8 input and output")
    if entry.get("epilogue") != ["acc_scale", "maxpool"]:
        raise ValueError("maxpool2d requires explicit acc_scale then maxpool")
    m, n = int(entry["M"]), int(entry["N"])
    scale = float(entry["acc_scale"])
    if min(m, n) <= 0 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("maxpool2d needs positive dimensions and a finite positive scale")
    pool, rows = _pool_epilogue(entry, rows=m, in_dims=None, op="maxpool2d")
    src, dst = entry.get("src", "X"), entry.get("out", "Y0")
    attrs = {**pool, "epilogue": ["acc_scale", "maxpool"], "acc_scale": scale, "output_dtype": "i8"}
    cb = {
        "abi_version": "0.1",
        "target": binding.target,
        "tensors": {
            src: {"role": "input", "shape": [m, n], "dtype": "i8"},
            dst: {"role": "output", "shape": [rows, n], "dtype": "i8"},
        },
        "commands": [{"opcode": "MAXPOOL2D", "operands": {"src": src, "dst": dst}, "attributes": attrs}],
    }
    cap = {
        "name": entry["name"],
        "kind": entry["kind"],
        "source_role": entry["source_role"],
        "source_reference": entry["source_reference"],
        "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [{"name": src, "role": "input", "shape": [m, n], "dtype": "i8"}],
        "operation": {"op": "maxpool2d", "attributes": {**attrs, "src": src, "out": dst}},
        "numeric_policy": _numeric_policy(binding, binding.operand_dtype, None),
        "expected": {
            "instruction_classes": binding.classes_for(op="movement", movement=True),
            "modes": {"movement": True, "maxpool": True},
        },
        "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional",
        "firesim": "optional",
    }
    return cap, emit_interface_mlir(cb)
