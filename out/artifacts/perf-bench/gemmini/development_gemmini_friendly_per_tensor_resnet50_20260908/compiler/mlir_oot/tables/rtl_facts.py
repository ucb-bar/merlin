"""RTL-derived hardware facts for `gemmini`.

Every number here is transcribed from the CIRCT static discovery bundle for this target
(`merlin.targetgen.rtl.facts.load_facts("gemmini")["facts"]`, generator
`rtl-introspect-v2-circt-hw`, method `decoder_icmp_fanout(mlc)`) and cross-checked against the
shipped ISA headers (`gemmini_params.h`, `gemmini.h`).  Nothing here is invented: the discovery
report is quoted in `docs/public_facts_used.md`.
"""
from __future__ import annotations

# --- systolic array -------------------------------------------------------------------------
#: mesh rows == mesh cols (facts.arrays[name=mesh]) and `DIM` in gemmini_params.h.
DIM = 16

# --- on-chip memories (facts.memories) ------------------------------------------------------
SCRATCHPAD_BYTES = 262144
ACCUMULATOR_BYTES = 65536

#: one scratchpad row holds DIM operand elements (i8, facts.datapaths[name=input]).
SPAD_ROW_BYTES = DIM * 1
#: one accumulator row holds DIM accumulator elements (i32, facts.datapaths[name=accumulator]).
ACC_ROW_BYTES = DIM * 4

SPAD_ROWS = SCRATCHPAD_BYTES // SPAD_ROW_BYTES      # 16384
ACC_ROWS = ACCUMULATOR_BYTES // ACC_ROW_BYTES       # 1024

# --- datapath dtypes (facts.datapaths) ------------------------------------------------------
OPERAND_DTYPE = "i8"
ACCUMULATOR_DTYPE = "i32"

# --- RoCC command interface (facts.interfaces[name=funct_decode_table]) ---------------------
CUSTOM_OPCODE = 123          # 0x7b, RISC-V custom-3
FUNCT3 = 3                   # xd=0, xs1=1, xs2=1 -> ROCC_INSTRUCTION_RS1_RS2

#: The functs the elaborated decoder actually accepts (26 of them).
LEGAL_FUNCTS = frozenset(
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 126)
)

# --- optional accelerator engines -----------------------------------------------------------
# UniversalResNet50Configs.arrayConfig enables the loop-convolution unroller.  The same
# elaborated GemminiConfig fixes hasIm2Col=false, so canonical convolution may use functs 15--21
# but must never claim ExecuteController's separate im2col datapath.
HAS_LOOP_CONV = True
HAS_IM2COL = False
HAS_TRAINING_CONVS = True
HAS_MAX_POOL = True

#: funct -> decoder name, straight from the discovery table.
FUNCT_NAMES = {
    0: "CONFIG_CMD", 1: "LOAD2_CMD", 2: "LOAD_CMD", 3: "STORE_CMD",
    4: "COMPUTE_AND_FLIP_CMD", 5: "COMPUTE_AND_STAY_CMD", 6: "PRELOAD_CMD", 7: "FLUSH_CMD",
    8: "LOOP_WS", 9: "LOOP_WS_CONFIG_BOUNDS", 10: "LOOP_WS_CONFIG_ADDRS_AB",
    11: "LOOP_WS_CONFIG_ADDRS_DC", 12: "LOOP_WS_CONFIG_STRIDES_AB",
    13: "LOOP_WS_CONFIG_STRIDES_DC", 14: "LOAD3_CMD", 15: "LOOP_CONV_WS",
    16: "LOOP_CONV_WS_CONFIG_1", 17: "LOOP_CONV_WS_CONFIG_2", 18: "LOOP_CONV_WS_CONFIG_3",
    19: "LOOP_CONV_WS_CONFIG_4", 20: "LOOP_CONV_WS_CONFIG_5", 21: "LOOP_CONV_WS_CONFIG_6",
    22: "CLKGATE_EN", 23: "STORE_SPAD_CMD", 24: "LOOP_WS_CONFIG_SPAD_AB",
}
