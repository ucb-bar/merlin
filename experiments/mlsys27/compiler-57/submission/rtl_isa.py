"""GENERATED from RTL facts by merlin.targetgen.rtl.gen_isa_module — DO NOT hand-edit the tables.

RTL-derived RoCC ISA encoder scaffold. The funct codes, custom opcode, mesh dimensions and
scratchpad/accumulator capacities below are extracted deterministically from the target's elaborated RTL
(firtool --ir-hw + the target's ISA definition) — they are what the *hardware* accepts, not a reading of
headers.

Build your command-buffer / RoCC lowering on `emit()` + `Program`: only LEGAL funct codes can be emitted
(an illegal funct raises), and `Program.finalize()` enforces config-before-use ordering. This eliminates
the two most common structural bugs (UNKNOWN instruction; use-before-config) by construction. You still
implement the op ALGORITHM (tiling to DIM, conv->im2col, accumulation) — that is the target-specific work.
"""
from __future__ import annotations
from dataclasses import dataclass, field

CUSTOM_OPCODE = 0x7b   # RoCC custom-3
FUNCT3 = 3
DIM = 16   # systolic array is 16x16
SCRATCHPAD_ROWS = 4096   # banks=4 elem_bits=8
ACCUMULATOR_ROWS = 512   # lanes=16 lane_bits=32

# RTL-extracted legal funct codes (emitting any other custom-3 funct is rejected by HW).
FUNCT = {
    "CONFIG_CMD": 0,  # config
    "LOAD2_CMD": 1,
    "LOAD_CMD": 2,
    "STORE_CMD": 3,
    "COMPUTE_AND_FLIP_CMD": 4,  # compute
    "COMPUTE_AND_STAY_CMD": 5,  # compute
    "PRELOAD_CMD": 6,
    "FLUSH_CMD": 7,
    "LOOP_WS": 8,
    "LOOP_WS_CONFIG_BOUNDS": 9,  # config
    "LOOP_WS_CONFIG_ADDRS_AB": 10,  # config
    "LOOP_WS_CONFIG_ADDRS_DC": 11,  # config
    "LOOP_WS_CONFIG_STRIDES_AB": 12,  # config
    "LOOP_WS_CONFIG_STRIDES_DC": 13,  # config
    "LOAD3_CMD": 14,
    "LOOP_CONV_WS": 15,
    "LOOP_CONV_WS_CONFIG_1": 16,  # config
    "LOOP_CONV_WS_CONFIG_2": 17,  # config
    "LOOP_CONV_WS_CONFIG_3": 18,  # config
    "LOOP_CONV_WS_CONFIG_4": 19,  # config
    "LOOP_CONV_WS_CONFIG_5": 20,  # config
    "LOOP_CONV_WS_CONFIG_6": 21,  # config
    "CLKGATE_EN": 22,
    "STORE_SPAD_CMD": 23,
    "LOOP_WS_CONFIG_SPAD_AB": 24,  # config
    "COUNTER_OP": 126,
}
LEGAL_FUNCT = frozenset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 126])
CONFIG_FUNCTS = frozenset([0, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 24])
COMPUTE_FUNCTS = frozenset([4, 5])
STORE_FUNCTS = frozenset([3, 23])


@dataclass
class Instr:
    funct: int
    rs1: int = 0
    rs2: int = 0
    def __post_init__(self):
        if self.funct not in LEGAL_FUNCT:
            raise ValueError(f"illegal funct {self.funct}: not in RTL legal_funct table "
                             f"(would emit an UNKNOWN custom-3 the hardware rejects)")
    def name(self) -> str:
        return {v: k for k, v in FUNCT.items()}.get(self.funct, "?")


@dataclass
class Program:
    """An ordered RoCC instruction stream with config-before-use enforced at finalize()."""
    instrs: list = field(default_factory=list)
    def emit(self, funct_name: str, rs1: int = 0, rs2: int = 0) -> "Program":
        if funct_name not in FUNCT:
            raise ValueError(f"unknown funct name {funct_name!r}; legal: {sorted(FUNCT)}")
        self.instrs.append(Instr(FUNCT[funct_name], rs1, rs2))
        return self
    def finalize(self) -> list:
        """Validate config-before-use; return the instruction list. Raises on ordering violations —
        the same 'use before config' bug the CIRCT checker flags, caught here at emit time instead."""
        configured = False
        store_configured = any(n for n in () )  # placeholder; track per-type below
        seen_config = set()
        for k, ins in enumerate(self.instrs):
            if ins.funct in CONFIG_FUNCTS:
                seen_config.add(ins.funct)
            if ins.funct in COMPUTE_FUNCTS and not (CONFIG_FUNCTS & seen_config):
                raise ValueError(f"use-before-config: COMPUTE (funct {ins.funct}) at {k} before any CONFIG_* "
                                 f"— emit a CONFIG before the first compute")
        return [(ins.funct, ins.rs1, ins.rs2) for ins in self.instrs]


def emit(funct_name: str, rs1: int = 0, rs2: int = 0) -> Instr:
    """One legal instruction (raises on illegal funct)."""
    return Instr(FUNCT[funct_name], rs1, rs2)
