#!/usr/bin/env python3
"""A PROBE compiler for gemmini: exercises Merlin's grading path end to end. It is not a solution.

It accepts exactly two interface shapes -- one 16x16 movement (isa/A1_mvin_mvout) and one
single-tile weight-stationary matmul (isa/A2_single_tile_matmul) -- and refuses everything else.
Every encoding constant is read from ``gemmini_isa.h`` beside this file, which Merlin GENERATES from
the RTL facts and the target contract (merlin.targetgen.isa_header_gen): funct codes, the RoCC
opcode, command-register field offsets, CONFIG selector codes and local-address flag bits. The only
facts taken from the public Gemmini ISA description rather than the header are the protocol choices
a probe has to make: weight-stationary dataflow is value 1 of the CONFIG_EX dataflow field, unit
scales are float 1.0, and a COMPUTE operand of all ones means "no D operand".

It imports nothing from Merlin (integrity_exempt: false) and implements the ABI's four commands:

  --verify-diagnostics IN                         parse + verify
  --convert-iface-to-gemmini IN                   print the (trivial) target-level listing
  --convert-iface-to-gemmini --emit-command-buffer=OUT IN
  --convert-iface-to-gemmini --emit-target-artifact IN   print the LLVM-dialect kernel module
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

HEADER = Path(__file__).resolve().parent / "gemmini_isa.h"
TARGET = "gemmini"


class Refused(Exception):
    """The probe does not handle this input."""


def _defines(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "#define" and parts[2].lstrip("-").isdigit():
            out[parts[1]] = int(parts[2])
    return out


H = _defines(HEADER)


def h(name: str) -> int:
    key = f"GEMMINI_{name}"
    if key not in H:
        raise Refused(f"the generated ISA header defines no {key}")
    return H[key]


def field(bundle: str, name: str, value: int) -> int:
    offset, width = h(f"{bundle}_{name}_OFFSET"), h(f"{bundle}_{name}_WIDTH")
    if value < 0 or value >= 1 << width:
        raise Refused(f"{bundle}.{name}={value} does not fit {width} bits")
    return value << offset


F32_ONE = struct.unpack("<I", struct.pack("<f", 1.0))[0]
WEIGHT_STATIONARY = 1  # CONFIG_EX dataflow field value (public Gemmini ISA)
NO_OPERAND = (1 << 32) - 1  # all-ones local address: "no D operand" (public Gemmini ISA)


# ------------------------------------------------------------------------------------- the interface
def _tensor_type(text: str) -> tuple[list[int], str]:
    inner = text.split("tensor<", 1)[1].split(">", 1)[0]
    parts = inner.split("x")
    return [int(p) for p in parts[:-1]], parts[-1]


def _attr(line: str, key: str) -> str:
    return line.split(f'{key} = "', 1)[1].split('"', 1)[0]


def parse(path: Path) -> dict:
    """The command buffer of a supported interface, or :class:`Refused`."""
    text = path.read_text(encoding="utf-8")
    ops = [line.strip() for line in text.splitlines() if "= merlin_iface." in line or line.strip().startswith("merlin_iface.")]
    tensors: dict[str, dict] = {}
    values: dict[str, str] = {}
    commands: list[dict] = []
    kinds = []
    for op in ops:
        lhs, _, rhs = op.partition(" = ") if " = " in op else ("", "", op)
        kind = rhs.split()[0].split(".", 1)[1]
        kinds.append(kind)
        if kind == "tensor":
            name = _attr(rhs, "name")
            shape, dtype = _tensor_type(rhs.rsplit(":", 1)[1])
            tensors[name] = {"shape": shape, "dtype": dtype, "role": _attr(rhs, "role")}
            values[lhs] = name
        elif kind == "movement":
            src = values[rhs.split()[1]]
            dst = _attr(rhs, "name")
            shape, dtype = _tensor_type(rhs.rsplit("->", 1)[1])
            tensors[dst] = {"shape": shape, "dtype": dtype, "role": "output"}
            commands.append({"opcode": "MOVEMENT", "operands": {"src": src, "dst": dst}, "attributes": {}})
            values[lhs] = dst
        elif kind == "resident_pack":
            src = values[rhs.split()[1]]
            values[lhs] = f"{src}_res"
            commands.append(
                {"opcode": "RES_PACK", "operands": {"src": src, "dst": f"{src}_res"}, "attributes": {"layout": _attr(rhs, "layout")}}
            )
        elif kind == "matmul":
            a, b = (s.strip().rstrip(",") for s in rhs.split(":", 1)[0].split()[1:3])
            values[lhs] = lhs.lstrip("%")
            commands.append({"opcode": "MATMUL_RESIDENT", "operands": {"lhs": values[a], "rhs": values[b], "dst": values[lhs]}})
        elif kind == "commit":
            src = values[rhs.split()[1]]
            if "epilogue = []" not in rhs:
                raise Refused("the probe handles only a commit without epilogue")
            dst = _attr(rhs, "name")
            commands.append(
                {"opcode": "COMMIT", "operands": {"src": src, "dst": dst}, "attributes": {"epilogue": [], "output_dtype": _attr(rhs, "output_dtype")}}
            )
            values[lhs] = dst
        elif kind == "evict":
            commands.append({"opcode": "EVICT", "operands": {"handle": values[rhs.split()[1]]}})
        else:
            raise Refused(f"merlin_iface.{kind} is outside the probe's two shapes")
    cb = {"abi_version": "0.1", "target": TARGET, "tensors": tensors, "commands": commands}
    shape = _shape_of(cb, kinds)
    cb["_probe_shape"] = shape
    return cb


def _square(t: dict, dtype: str) -> bool:
    edge = h("MESH_ROWS")
    return t["shape"] == [edge, edge] and t["dtype"] == dtype and h("MESH_COLS") == edge


def _shape_of(cb: dict, kinds: list[str]) -> str:
    t = cb["tensors"]
    if kinds == ["tensor", "movement"]:
        (src,) = [n for n, s in t.items() if s["role"] == "input"]
        (dst,) = [n for n, s in t.items() if s["role"] == "output"]
        if _square(t[src], "i8") and _square(t[dst], "i8"):
            return "movement"
    if kinds == ["tensor", "tensor", "resident_pack", "matmul", "commit", "evict"]:
        commit = cb["commands"][2]
        names = [n for n in t]
        if (
            len(names) == 2
            and all(_square(t[n], "i8") for n in names)
            and commit["attributes"]["output_dtype"] == "i32"
        ):
            return "single_tile_matmul"
    raise Refused("only one DIMxDIM i8 movement or one DIMxDIM i8 x i8 -> i32 matmul is supported")


# ------------------------------------------------------------------------------------- the kernel
def _program(cb: dict) -> tuple[list[str], list[tuple]]:
    """``(argument names, instructions)``; an instruction is (funct, rs1, rs2) where an operand is an
    int constant or the name of a pointer argument."""
    edge = h("MESH_ROWS")
    rows_cols = lambda bundle: field(bundle, "NUM_ROWS", edge) | field(bundle, "NUM_COLS", edge)  # noqa: E731
    acc = 1 << h("LOCAL_ADDR_ACCUMULATOR_SELECT_BIT")
    full = 1 << h("LOCAL_ADDR_FULL_WIDTH_READOUT_BIT")
    # The CONFIG selector is the rs1 cmd_type field the contract names (rocc_operand_roles.config).
    config_ld = (
        field("CONFIGMVOUTRS1", "CMD_TYPE", h("CONFIG_LD"))
        | field("CONFIGMVINRS1", "STRIDE", edge)
        | field("CONFIGMVINRS1", "PIXEL_REPEATS", 1)
        | field("CONFIGMVINRS1", "SCALE", F32_ONE)
    )

    def config_st(row_bytes: int) -> tuple:
        rs1 = field("CONFIGMVOUTRS1", "CMD_TYPE", h("CONFIG_ST"))
        rs2 = field("CONFIGMVOUTRS2", "STRIDE", row_bytes) | field("CONFIGMVOUTRS2", "ACC_SCALE", F32_ONE)
        return (h("FUNCT_CONFIG_CMD"), rs1, rs2)

    flush = (h("FUNCT_FLUSH_CMD"), 0, 0)
    if cb["_probe_shape"] == "movement":
        src = next(n for n, s in cb["tensors"].items() if s["role"] == "input")
        dst = next(n for n, s in cb["tensors"].items() if s["role"] == "output")
        return [src, dst], [
            flush,
            (h("FUNCT_CONFIG_CMD"), config_ld, edge),
            config_st(edge),
            (h("FUNCT_LOAD_CMD"), src, rows_cols("MVINRS2") | 0),
            (h("FUNCT_STORE_CMD"), dst, rows_cols("MVOUTRS2") | 0),
        ]
    weight, lhs = cb["commands"][0]["operands"]["src"], cb["commands"][1]["operands"]["lhs"]
    out = cb["commands"][2]["operands"]["dst"]
    order = [n for n in cb["tensors"]] + [out]  # logical inputs in declaration order, then the result
    config_ex = (
        h("FUNCT_CONFIG_CMD"),
        field("CONFIGEXRS1", "CMD_TYPE", h("CONFIG_EX"))
        | field("CONFIGEXRS1", "DATAFLOW", WEIGHT_STATIONARY)
        | field("CONFIGEXRS1", "A_STRIDE", 1)
        | field("CONFIGEXRS1", "ACC_SCALE", F32_ONE),
        field("CONFIGEXRS2", "C_STRIDE", 1),
    )
    w_row, a_row = 0, edge
    return order, [
        flush,
        config_ex,
        (h("FUNCT_CONFIG_CMD"), config_ld, edge),
        (h("FUNCT_LOAD_CMD"), weight, rows_cols("MVINRS2") | w_row),
        (h("FUNCT_LOAD_CMD"), lhs, rows_cols("MVINRS2") | a_row),
        config_st(edge * 4),
        (h("FUNCT_PRELOAD_CMD"), rows_cols("PRELOADRS") | w_row, rows_cols("PRELOADRS") | acc),
        (h("FUNCT_COMPUTE_AND_FLIP_CMD"), rows_cols("COMPUTERS") | a_row, rows_cols("COMPUTERS") | NO_OPERAND),
        (h("FUNCT_STORE_CMD"), out, rows_cols("MVOUTRS2") | acc | full),
    ]


def emit_llvm(cb: dict) -> str:
    args, program = _program(cb)
    opcode = h("ROCC_OPCODE")
    lines = [
        "module {",
        f"  llvm.func @{TARGET}_kernel({', '.join(f'%{a}: !llvm.ptr' for a in args)}) {{",
    ]
    for a in args:
        lines.append(f"    %{a}_i = llvm.ptrtoint %{a} : !llvm.ptr to i64")
    # Bracket the accelerator program with fences: prior host stores are visible to the DMA before the
    # first command, and every command has retired before the kernel returns.
    lines.append('    llvm.inline_asm has_side_effects "fence", "" : () -> ()')
    for index, (funct, rs1, rs2) in enumerate(program):
        operands = []
        for slot, value in (("a", rs1), ("b", rs2)):
            if isinstance(value, str):
                operands.append(f"%{value}_i")
            else:
                lines.append(f"    %c{index}{slot} = llvm.mlir.constant({value} : i64) : i64")
                operands.append(f"%c{index}{slot}")
        lines.append(
            f'    llvm.inline_asm has_side_effects ".insn r {opcode}, 3, {funct}, x0, $0, $1", "r,r" '
            f"{operands[0]}, {operands[1]} : (i64, i64) -> ()"
        )
    lines += ['    llvm.inline_asm has_side_effects "fence", "" : () -> ()', "    llvm.return", "  }", "}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    flags = [a for a in argv if a.startswith("--")]
    inputs = [a for a in argv if not a.startswith("--")]
    if len(inputs) != 1:
        print("usage: probe_compiler.py [flags] INPUT.mlir", file=sys.stderr)
        return 2
    try:
        cb = parse(Path(inputs[0]))
    except (Refused, IndexError, KeyError, ValueError, StopIteration) as exc:
        print(f"probe_compiler: refused: {exc}", file=sys.stderr)
        return 1
    public = {k: v for k, v in cb.items() if not k.startswith("_")}
    if "--verify-diagnostics" in flags:
        return 0
    emitted = False
    for flag in flags:
        if flag.startswith("--emit-command-buffer="):
            Path(flag.split("=", 1)[1]).write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
            emitted = True
    if "--emit-target-artifact" in flags:
        sys.stdout.write(emit_llvm(cb))
        return 0
    if not emitted:
        sys.stdout.write(f"// gemmini probe: {cb['_probe_shape']}\n" + json.dumps(public["commands"]) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
