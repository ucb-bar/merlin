"""Gemmini timing-probe backend: turns a generic probe request into a RoCC command stream and C program.

Target edge of ``merlin_experiments.timing_probes``. The selector decides WHAT to measure (a price
class with its state predicate, or a whole-call size regime); this module decides HOW a Gemmini
program exercises it, with one parameterized tile pipeline (load, preload, compute, store) in the two
protocols the RTL implements -- weight- and output-stationary, ordered as in the pinned gemmini.h
``sp_tiled_matmul_ws``/``sp_tiled_matmul_os`` loops. There is no per-class kernel: a request only sets
the pipeline's mode fields from the class's state predicate and picks which load functs it uses.

Every funct code, operand field position and width, mesh/scratchpad/accumulator geometry and element
width is read from the selected RTL facts (through the timing space). Protocol values the facts do
not state -- selector codes, load-state ids, mode codes, local-address flags and the retain sentinel
-- come from the timing space's pinned source declarations and the reviewed support contract. The C
glue is the selected bare-metal runtime's console (``htif.h``), the one the capsule harness links, and
the program is built with the target backend's own harness recipe and operator toolchain.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from merlin_experiments.timing_probes.decode import Command
from merlin_experiments.timing_probes.selection import Realized, Unrealizable

from merlin.common.facts_view import interface
from merlin.targetgen.rocc.semantics import field_slot

SHAPES = ("full", "partial")
_UNIT_SCALE = struct.unpack("<I", struct.pack("<f", 1.0))[0]
#: The pipeline stages: the state each one reads and the instructions it can issue.
_STAGE_STATES = ("EX", "ST", "NORM")


@dataclass
class _Op:
    kind: str  # rocc | fence
    funct: str | None = None
    rs1: int | tuple | None = None  # an int, or ("dram", buffer, byte offset)
    rs2: int | None = None


@dataclass
class Program:
    ops: list[_Op] = field(default_factory=list)
    buffers: dict[str, int] = field(default_factory=dict)


@dataclass
class _Plan:
    dataflow: int
    ex_activation: int
    st_activation: int
    I: int = 1  # noqa: E741 -- tiled-matmul extent names (output rows, output cols, reduction)
    J: int = 1
    K: int = 1
    a_rows: int = 0
    k_cols: int = 0
    c_cols: int = 0  # 0: a full mesh row
    pair_preloads: bool = True  # False: reuse resident weights with a compute that has no preload
    c_in_spad: bool = False
    flushes: int = 1
    loads: tuple[str, ...] = ()
    pool: dict | None = None


class GemminiProbeBackend:
    SHAPES = SHAPES

    def __init__(self, *, space, facts: dict, contract: dict):
        self.space = space
        body = facts.get("facts", facts)
        table = interface(body, "funct_decode_table") or {}
        self.opcode, self.funct3 = table.get("custom_opcode"), table.get("funct3")
        if type(self.opcode) is not int or type(self.funct3) is not int:
            raise Unrealizable("facts carry no custom opcode / funct3")
        self.functs = {name: code for code, name in space.funct_names.items()}
        mesh = next((a for a in body.get("arrays") or () if a.get("name") == "mesh"), {})
        self.dim = mesh.get("rows")
        if type(self.dim) is not int or mesh.get("cols") != self.dim:
            raise Unrealizable("facts carry no square mesh")
        memories = {m.get("name"): m for m in body.get("memories") or ()}
        self.spad_rows = memories["scratchpad"]["banks"] * memories["scratchpad"]["depth"]
        self.acc_rows = memories["accumulator"]["banks"] * memories["accumulator"]["depth"]
        protocol = space.document.get("probe_protocol") or {}
        self.load_states = dict(protocol.get("load_state_ids") or {})
        self.defaults = dict(protocol.get("defaults") or {})
        self.unit_scale = {tuple(f.split(".")) for f in protocol.get("unit_scale_fields") or ()}
        reviewed_path = (space.data_dir / str(protocol.get("local_address_contract"))).resolve()
        self._reviewed_sha = hashlib.sha256(reviewed_path.read_bytes()).hexdigest()
        reviewed = yaml.safe_load(reviewed_path.read_text())
        roles = reviewed.get("rocc_operand_roles") or {}
        addr_len = (contract.get("encoding") or {}).get("addr_len") or (reviewed.get("encoding") or {}).get("addr_len")
        flags = roles.get("local_address_flags") or {}
        if type(addr_len) is not int or not flags:
            raise Unrealizable("no reviewed local-address flag layout")
        self.flag = {name: 1 << (addr_len - 1 - spec["msb_offset"]) for name, spec in flags.items()}
        sentinel = roles.get("retain_sentinel") or {}
        slot = field_slot(space.layouts.get(sentinel.get("bundle")), str(sentinel.get("field")))
        if sentinel.get("value") != "all_ones" or slot is None:
            raise Unrealizable("no reviewed retain sentinel")
        self.garbage = (1 << slot[1]) - 1
        self.codes = {name: entry["code"] for name, entry in space.states.items()}
        self.ws = self.defaults.get("EX.dataflow")
        self.no_activation = self.defaults.get("EX.activation")
        emitted = {
            space.config_instruction,
            "FLUSH_CMD",
            "PRELOAD_CMD",
            "STORE_CMD",
            "COMPUTE_AND_FLIP_CMD",
            "COMPUTE_AND_STAY_CMD",
            *self.load_states,
        }
        self.emitted = {name for name in emitted if name in self.functs} | set(space.pseudo_classes)

    def identity(self) -> dict:
        return {
            "backend_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "local_address_contract_sha256": self._reviewed_sha,
        }

    # -- encoding ------------------------------------------------------------------------------------
    def _pack(self, bundle: str, **values) -> int:
        values = {k: (_UNIT_SCALE if (bundle, k) in self.unit_scale and v == 1 else v) for k, v in values.items()}
        return self.space.pack(bundle, values)

    def _config(self, state: str, rs1: int, rs2: int) -> _Op:
        selector = self.space.selector
        if selector is None:
            raise Unrealizable("no config selector layout")
        return _Op("rocc", self.space.config_instruction, rs1 | (self.codes[state] << selector.offset), rs2)

    def _rocc(self, funct: str, rs1, rs2) -> _Op:
        if funct not in self.functs:
            raise Unrealizable(f"facts declare no funct {funct}")
        return _Op("rocc", funct, rs1, rs2)

    def _acc(self, row: int, accumulate: bool = False) -> int:
        return self.flag["accumulator_select"] | (self.flag["accumulate"] if accumulate else 0) | row

    # -- request -> plan -------------------------------------------------------------------------------
    def _value(self, key: str, predicate):
        if predicate is None:
            return self.defaults.get(key)
        if type(predicate) is int:
            return predicate
        if isinstance(predicate, list):
            return predicate[0]
        if key == "ST.pool_stride":  # {not: 0}: a 2x2 stride-2 pooling window
            return 2
        raise Unrealizable(f"cannot choose a value for {key} != {predicate['not']}")

    def _plan(self, request: dict) -> _Plan:
        unknown = sorted(set(request.get("instructions") or ()) - self.emitted)
        if unknown:
            raise Unrealizable(f"the tile pipeline issues no {unknown}")
        state = dict(request.get("state") or {})
        for key, predicate in state.items():
            if key.partition(".")[0] not in _STAGE_STATES:
                raise Unrealizable(f"no pipeline stage sets {key}")
            if key.startswith("NORM.") and predicate != 0:
                raise Unrealizable("normalization configuration is not emitted by this backend")
        get = {
            k: self._value(k, state.get(k)) for k in ("EX.dataflow", "EX.activation", "ST.activation", "ST.pool_stride")
        }
        full = request["shape"] == "full"
        plan = _Plan(
            dataflow=get["EX.dataflow"], ex_activation=get["EX.activation"], st_activation=get["ST.activation"]
        )
        # Two design points per class: full tiles with every compute preloaded (gemmini.h order), and
        # half tiles with a different command count whose weight-stationary reuse issues computes
        # with no preload (ExecuteController "single mul"), so per-command, per-unit and preload
        # prices do not move together.
        ws = plan.dataflow == self.ws
        plan.a_rows = plan.k_cols = self.dim if full else self.dim // 2
        plan.I, plan.K = (2, 2) if full else ((3, 1) if ws else (1, 3))
        plan.flushes = 2 if full else 0
        plan.pair_preloads = full or not ws
        # A compute-time activation applies on the mesh's scratchpad write path (ExecuteController).
        plan.c_in_spad = plan.ex_activation != self.no_activation
        if plan.c_in_spad and ws:
            plan.K = 1  # scratchpad outputs cannot accumulate across the reduction
        plan.loads = tuple(i for i in request.get("instructions") or () if i in self.load_states)
        if get["ST.pool_stride"]:
            out = 2 if full else 1
            plan.pool = {
                "pool_stride": get["ST.pool_stride"],
                "pool_size": 2,
                "pool_out_dim": out,
                "porows": out,
                "pocols": out,
                "orows": 2 * out,
                "ocols": 2 * out,
            }
            plan.I, plan.K = (2 if full else 1), 1
            plan.a_rows, plan.k_cols, plan.pair_preloads = 4 * out * out, self.dim, True
        return plan

    def _regime_plan(self, targets: dict) -> _Plan:
        d = self.dim
        tile_macs, tile_bytes = d**3, d * d
        base = dict(
            dataflow=self.ws, ex_activation=self.no_activation, st_activation=self.defaults.get("ST.activation")
        )
        if all(v <= (tile_macs if q == "macs" else tile_bytes) for q, v in targets.items()):
            return _Plan(**base, a_rows=1, k_cols=1, c_cols=1, flushes=0)  # the domain's small end
        tiles_out = max(1, math.ceil(targets.get("write_bytes", tile_bytes) / tile_bytes))
        i_tiles = min(self.acc_rows // d, tiles_out)
        j_tiles = max(1, math.ceil(tiles_out / i_tiles))
        k_tiles = max(1, math.ceil(targets.get("macs", 0) / (tile_macs * i_tiles * j_tiles)))
        k_tiles = max(k_tiles, math.ceil(targets.get("read_bytes", 0) / ((i_tiles * j_tiles + j_tiles) * tile_bytes)))
        return _Plan(**base, I=i_tiles, J=j_tiles, K=k_tiles, a_rows=d, k_cols=d, flushes=1)

    # -- plan -> program ---------------------------------------------------------------------------------
    def _program(self, plan: _Plan) -> Program:
        d, ops = self.dim, []
        rows, cols = min(plan.a_rows, d), plan.k_cols
        ws = plan.dataflow == self.ws
        # A (I*d x K*d) and B (K*d x J*d) share one DRAM row stride, so one load state serves both.
        stride = max(plan.K, plan.J) * d
        program = Program(
            ops, {"a": plan.I * d * stride, "b": plan.K * d * stride, "c": max(plan.I * d * plan.J * d, d * d)}
        )
        loads = plan.loads or ("LOAD_CMD",)
        a_load, b_load = loads[0], loads[1] if len(loads) > 1 else loads[0]
        alternate = loads[2:]
        ops.append(_Op("fence"))
        ops += [self._rocc("FLUSH_CMD", 0, 0) for _ in range(plan.flushes)]
        ops.append(
            self._config(
                "EX",
                self._pack(
                    "ConfigExRs1", dataflow=plan.dataflow, activation=plan.ex_activation, a_stride=1, acc_scale=1
                ),
                self._pack("ConfigExRs2", in_shift=0, c_stride=1),
            )
        )
        for funct in dict.fromkeys(loads):
            ops.append(
                self._config(
                    "LD",
                    self._pack("ConfigMvinRs1", state_id=self.load_states[funct], pixel_repeats=1, stride=d, scale=1),
                    stride,
                )
            )
        c_stride = d if plan.pool else plan.J * d
        ops.append(
            self._config(
                "ST",
                self._pack("ConfigMvoutRs1", activation=plan.st_activation, **(plan.pool or {})),
                self._pack("ConfigMvoutRs2", stride=c_stride, acc_scale=1),
            )
        )
        b_base, a_base = 0, plan.K * d
        a_slots = max(1, (self.spad_rows - a_base - plan.I * d) // d)
        c_spad = self.spad_rows - plan.I * d

        def a_addr(i, k):
            return a_base + ((i * plan.K + k) % a_slots) * d  # a ring beyond scratchpad capacity

        def mvin(funct, buffer, offset, local, n_rows, n_cols):
            ops.append(
                self._rocc(
                    funct,
                    ("dram", buffer, offset),
                    self._pack("MvinRs2", local_addr=local, num_cols=n_cols, num_rows=n_rows),
                )
            )

        def c_addr(i, accumulate=False):
            return c_spad + i * d if plan.c_in_spad else self._acc(i * d, accumulate)

        for j in range(plan.J):
            if ws:
                for k in range(plan.K):
                    mvin(b_load, "b", k * d * stride + j * d, b_base + k * d, cols, d)
                for k in range(plan.K):
                    for i in range(plan.I):
                        funct = alternate[(i + k) % len(alternate)] if alternate and (i + k) % 2 else a_load
                        mvin(funct, "a", i * d * stride + k * d, a_addr(i, k), rows, cols)
                        weights = b_base + k * d if i == 0 else self.garbage
                        if i == 0 or plan.pair_preloads:
                            ops.append(
                                self._rocc(
                                    "PRELOAD_CMD",
                                    self._pack("PreloadRs", local_addr=weights, num_cols=d, num_rows=cols),
                                    self._pack("PreloadRs", local_addr=c_addr(i, k > 0), num_cols=d, num_rows=rows),
                                )
                            )
                        ops.append(
                            self._rocc(
                                "COMPUTE_AND_FLIP_CMD" if i == 0 else "COMPUTE_AND_STAY_CMD",
                                self._pack("ComputeRs", local_addr=a_addr(i, k), num_cols=cols, num_rows=rows),
                                self._pack("ComputeRs", local_addr=self.garbage, num_cols=d, num_rows=d),
                            )
                        )
            else:
                for i in range(plan.I):
                    for k in range(plan.K):
                        mvin(a_load, "a", i * d * stride + k * d, a_addr(i, k), rows, cols)
                        mvin(b_load, "b", k * d * stride + j * d, b_base + k * d, cols, d)
                        out = c_addr(i) if k == plan.K - 1 else self.garbage
                        ops.append(
                            self._rocc(
                                "PRELOAD_CMD",
                                self._pack("PreloadRs", local_addr=self.garbage, num_cols=d, num_rows=d),
                                self._pack("PreloadRs", local_addr=out, num_cols=d, num_rows=rows),
                            )
                        )
                        ops.append(
                            self._rocc(
                                "COMPUTE_AND_FLIP_CMD" if k == 0 else "COMPUTE_AND_STAY_CMD",
                                self._pack("ComputeRs", local_addr=a_addr(i, k), num_cols=cols, num_rows=rows),
                                self._pack("ComputeRs", local_addr=b_base + k * d, num_cols=d, num_rows=cols),
                            )
                        )
            for i in range(plan.I):
                payload = self._pack(
                    "MvoutRs2", local_addr=c_addr(i), num_cols=plan.c_cols or d, num_rows=0 if plan.pool else rows
                )
                ops.append(self._rocc("STORE_CMD", ("dram", "c", i * d * plan.J * d + j * d), payload))
        ops.append(_Op("fence"))
        return program

    # -- backend API ----------------------------------------------------------------------------------------
    def realize(self, request: dict, space) -> Realized:
        if request.get("shape") == "regime":
            plan = self._regime_plan(dict(request.get("regime") or {}))
            description = f"WS tile stream I={plan.I} J={plan.J} K={plan.K} of {plan.a_rows}x{plan.k_cols} tiles"
        else:
            plan = self._plan(request)
            description = (
                f"{request.get('class')} {request['shape']}: dataflow={plan.dataflow} "
                f"ex_activation={plan.ex_activation} st_activation={plan.st_activation} "
                f"pool={plan.pool} I={plan.I} K={plan.K} tile {plan.a_rows}x{plan.k_cols} "
                f"loads={plan.loads or ('LOAD_CMD',)}"
            )
        program = self._program(plan)
        commands = [
            Command("FENCE")
            if op.kind == "fence"
            else Command(op.funct, op.rs1 if type(op.rs1) is int else None, op.rs2)
            for op in program.ops
        ]
        issued = sum(op.kind == "rocc" for op in program.ops)
        macs = plan.I * plan.J * plan.K * min(plan.a_rows, self.dim) * plan.k_cols * self.dim
        cost = 20.0 * issued + macs / self.dim**2 + sum(program.buffers.values()) / self.dim
        return Realized(commands=commands, cost=cost, description=description, program=program)

    def render(self, realized: Realized, timing, *, probe_id: str) -> str:
        """The probe's C translation unit with the reviewed kernel-call timing window around it."""
        program: Program = realized.program
        # Operands live in DRAM above the runtime's stack, never loaded or touched by the core: the
        # image stays small (the serial loader moves every loaded byte) and Gemmini timing does not
        # depend on operand values. Each buffer starts on a 64-byte boundary.
        base, offsets = 0, {}
        for name, size in program.buffers.items():
            offsets[name] = base
            base += (max(1, size) + 63) // 64 * 64
        lines = [
            f"/* Timing probe {probe_id}: operator reference program generated by the Gemmini",
            " * timing-probe backend from RTL facts. Not a kernel; never granted to an author.",
            f" * DRAM operand arena: {base} bytes above _stack_top. */",
            "#include <stdint.h>",
            '#include "htif.h"',
            timing.source_marker(),
            "extern char _stack_top[];",
            "#define MERLIN_PROBE_ARENA ((((uintptr_t)_stack_top) + 4095u) & ~(uintptr_t)4095u)",
            *timing.declarations(),
            "__attribute__((noinline)) void kernel_entry(void){",
        ]
        for op in program.ops:
            if op.kind == "fence":
                lines.append('  __asm__ volatile("fence" ::: "memory");')
                continue
            if type(op.rs1) is tuple:
                rs1 = f"(uint64_t)(MERLIN_PROBE_ARENA + {offsets[op.rs1[1]] + op.rs1[2]}u)"
            else:
                rs1 = f"UINT64_C({op.rs1:#x})"
            lines.append(
                f'  __asm__ volatile(".insn r {self.opcode}, {self.funct3}, {self.functs[op.funct]}, x0, '
                f'%0, %1" :: "r"({rs1}), "r"(UINT64_C({op.rs2:#x})) : "memory");'
            )
        lines += [
            "}",
            '__attribute__((noinline)) void kernel_complete(void){ __asm__ volatile("fence" ::: "memory"); }',
            "int main(void){",
            "  console_init();",
            *timing.begin(),
            "  kernel_entry();",
            "  kernel_complete();",
            *timing.end(),
            '  htif_puts("DONE\\n");',
            "  htif_exit(0);",
            "  return 0;",
            "}",
            "",
        ]
        return "\n".join(lines)


def bind(*, space, facts: dict, contract: dict) -> GemminiProbeBackend:
    return GemminiProbeBackend(space=space, facts=facts, contract=contract)
