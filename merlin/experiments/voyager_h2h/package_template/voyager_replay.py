"""Replay precomputed Voyager schedules as Gemmini instructions. Generated package file; do not edit.

The bridge package is the certified reference backend with ONE change: a resident matmul's COMMIT is
not lowered by the package's own instruction selection, it is replayed from ``voyager_schedules.json``
-- the schedule the pinned Voyager compiler produced for that exact shape, already lowered (by
``merlin.baselines.voyager_schedule``) to block loads, preloads, computes and stores. This module only
packs those ops into the package's existing, certified RoCC encodings; it chooses nothing.

A shape with no schedule, or one the bridge refused, raises: the harness records that capsule as not
compiled by this arm rather than silently falling back to the reference package's own schedule, which
would score the reference compiler under Voyager's name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ir_ingest import InterfaceProgram
from lowering.isa import (ACC_ACCUMULATE, ACC_BASE, ACC_FULL, DIM, GARBAGE_ADDR, Address,
                          Instruction, _config_ex, _config_ld, _config_st, _elem_bytes, _row_stride,
                          _tile_word)

_SCHEDULES = json.loads(Path(__file__).with_name("voyager_schedules.json").read_text())


def schedule_key(m: int, k: int, n: int, epilogue: list[str], out_dtype: str) -> str:
    return f"{m}x{k}x{n}|{'+'.join(epilogue) or 'none'}|{out_dtype}"


def _voyager_matmul_trace(program: InterfaceProgram, lhs_name: str, weight_name: str,
                          out_name: str, attrs: dict[str, Any]) -> list[Instruction]:
    lhs, weight, out = (program.tensors[n] for n in (lhs_name, weight_name, out_name))
    m, k = lhs.shape
    _, n = weight.shape
    key = schedule_key(m, k, n, list(attrs.get("epilogue", [])), out.dtype)
    entry = _SCHEDULES["schedules"].get(key)
    if entry is None:
        raise ValueError(f"no Voyager schedule was produced for {key}")
    if entry.get("refused"):
        raise ValueError(f"the Voyager arm does not compile {key}: {entry['refused']}")
    names = {"lhs": lhs_name, "weight": weight_name, "out": out_name}
    specs = {"lhs": lhs, "weight": weight, "out": out}
    strides = {role: _row_stride(spec) for role, spec in specs.items()}
    elem = {role: _elem_bytes(spec.dtype) for role, spec in specs.items()}
    readout = ACC_BASE | (ACC_FULL if out.dtype == "i32" else 0)
    trace = [_config_ex(weight_stationary=True), _config_st(strides["out"], attrs, out.dtype)]
    for op in entry["ops"]:
        kind = op[0]
        if kind == "mvin":
            _, role, row, col, rows, cols, spad_row = op
            # Packed exactly as the reference package packs a load -- a CONFIG_LD before every MVIN --
            # so the arms differ only in the schedule Voyager chose, never in instruction packing.
            trace.append(_config_ld(strides[role], channel=0))
            trace.append(Instruction("MVIN", Address(names[role], row * strides[role]
                                                     + col * elem[role]),
                                     _tile_word(spad_row, cols, rows)))
        elif kind == "preload":
            _, weight_row, acc_row, accumulate, rows, cols = op
            b = _tile_word(GARBAGE_ADDR if weight_row is None else weight_row, DIM, DIM)
            c = readout | acc_row | (ACC_ACCUMULATE if accumulate else 0)
            trace.append(Instruction("PRELOAD", b, _tile_word(c, cols, rows)))
        elif kind == "compute":
            _, input_row, rows, fresh = op
            trace.append(Instruction("COMPUTE_PRELOADED" if fresh else "COMPUTE_ACCUMULATE",
                                     _tile_word(input_row, DIM, rows),
                                     _tile_word(GARBAGE_ADDR, DIM, DIM)))
        elif kind == "mvout":
            _, _role, row, col, rows, cols, acc_row = op
            trace.append(Instruction("MVOUT", Address(names["out"], row * strides["out"]
                                                      + col * elem["out"]),
                                     _tile_word(readout | ACC_ACCUMULATE | acc_row, cols, rows)))
        else:
            raise ValueError(f"unknown schedule op {kind!r} in {key}")
    trace.append(Instruction("FENCE"))
    return trace


def build_trace(program: InterfaceProgram) -> list[Instruction]:
    if not program.is_contract_module:
        return [Instruction("FENCE"), Instruction("FLUSH", 0, 0)]
    trace: list[Instruction] = [Instruction("FENCE"), Instruction("FLUSH", 0, 0)]
    resident_sources: dict[str, str] = {}
    pending: dict[str, tuple[str, str]] = {}
    for command in program.commands:
        opcode, ops = command["opcode"], command.get("operands", {})
        if opcode == "RES_PACK":
            resident_sources[ops["dst"]] = ops["src"]
        elif opcode == "MATMUL_RESIDENT":
            pending[ops["dst"]] = (ops["lhs"], resident_sources[ops["rhs"]])
        elif opcode == "COMMIT":
            lhs, weight = pending[ops["src"]]
            trace.extend(_voyager_matmul_trace(program, lhs, weight, ops["dst"],
                                               command.get("attributes", {})))
        elif opcode == "EVICT":
            continue
        else:
            raise ValueError(f"the Voyager arm lowers resident matmuls only; {opcode} is not one")
    return trace
