"""Emit a schema-valid command buffer (ABI v0.1) from the gemmini Program."""
from __future__ import annotations

from typing import Any

from .conv import conv_geometry
from .program import Program


def emit_command_buffer(prog: Program) -> dict[str, Any]:
    handle_name: dict[str, str] = {}   # value-id -> command-buffer handle name
    acc_n = 0

    tensors: dict[str, dict] = {}

    def need_tensor(name: str):
        if name in prog.tensors and name not in tensors:
            tensors[name] = dict(prog.tensors[name])

    commands: list[dict] = []
    im2col_recipes: list[dict] = []
    conv_n = 0
    for rec in prog.ops:
        k = rec["kind"]
        if k == "pack":
            need_tensor(rec["src"])
            hn = rec["src"] + "_res"
            handle_name[rec["dst"]] = hn
            commands.append({"opcode": "RES_PACK",
                             "operands": {"src": rec["src"], "dst": hn},
                             "attributes": {"layout": rec["layout"]}})
        elif k == "matmul":
            need_tensor(rec["lhs"])
            nonlocal_acc = f"acc{acc_n}"
            acc_n += 1
            handle_name[rec["dst"]] = nonlocal_acc
            commands.append({"opcode": "MATMUL_RESIDENT",
                             "operands": {"lhs": rec["lhs"],
                                          "rhs": handle_name[rec["rhs"]],
                                          "dst": nonlocal_acc}})
        elif k == "commit":
            attrs: dict[str, Any] = {"epilogue": rec["epilogue"],
                                     "output_dtype": rec["output_dtype"]}
            if rec.get("acc_scale") is not None:
                attrs["acc_scale"] = rec["acc_scale"]
            commands.append({"opcode": "COMMIT",
                             "operands": {"src": handle_name[rec["acc"]],
                                          "dst": rec["dst"]},
                             "attributes": attrs})
        elif k == "evict":
            commands.append({"opcode": "EVICT",
                             "operands": {"handle": handle_name[rec["handle"]]}})
        elif k == "movement":
            need_tensor(rec["src"])
            # the VECTOR_MAP output is collected by the reference only if it is a
            # declared output tensor (COMMIT outputs are collected directly).
            tensors[rec["dst"]] = {"shape": rec["shape"], "dtype": rec["dtype"],
                                   "role": "output"}
            commands.append({"opcode": "VECTOR_MAP",
                             "operands": {"lhs": rec["src"], "dst": rec["dst"]},
                             "attributes": {"combine": "identity", "activation": []}})
        elif k == "conv2d":
            geo = conv_geometry(rec)
            need_tensor(rec["ifm"])      # NHWC source leaf (gathered into im2col)
            need_tensor(rec["weight"])   # packed [Kh*Kw*Ci, Co] weight
            # the im2col activation is a derived input: declare its shape so the
            # runner allocates it, and emit the gather recipe (built from IFM).
            tensors[geo["im2col"]] = {"shape": [geo["M"], geo["K"]],
                                      "dtype": geo["ifm_dtype"], "role": "input"}
            im2col_recipes.append(geo["recipe"])
            wh = rec["weight"] + "_res"
            acc = f"conv_acc{conv_n}"
            conv_n += 1
            commands.append({"opcode": "RES_PACK",
                             "operands": {"src": rec["weight"], "dst": wh},
                             "attributes": {"layout": "packed_conv_rhs"}})
            commands.append({"opcode": "MATMUL_RESIDENT",
                             "operands": {"lhs": geo["im2col"], "rhs": wh, "dst": acc}})
            cattrs: dict[str, Any] = {"epilogue": rec["epilogue"],
                                      "output_dtype": rec["output_dtype"]}
            commands.append({"opcode": "COMMIT",
                             "operands": {"src": acc, "dst": rec["dst"]},
                             "attributes": cattrs})

    cb: dict[str, Any] = {"abi_version": prog.abi_version, "target": prog.target,
                          "backend": "gemmini_oot_xdsl",
                          "tensors": tensors, "commands": commands}
    if im2col_recipes:
        cb["params"] = {"im2col_recipes": im2col_recipes}
    return cb
