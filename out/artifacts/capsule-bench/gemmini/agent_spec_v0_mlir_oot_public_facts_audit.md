# agent_spec_v0_mlir_oot — public-facts parity audit

Every RoCC/Gemmini constant and encoding detail used by the package's RoCC lowering
(`mlir_oot/lib/Conversion/GemminiToLLVM/GemminiToLLVM.cpp`), classified as **public target
information** (also available to the raw baseline) or a **Merlin tooling/authoring advantage**
(ported pre-assembled from the certified native path). This makes the fairness boundary explicit
(Correction B); advantage facts are reported as advantages, never as parity information.

## Baseline information set (what the sandbox actually contains)

`setup_baseline_sandbox.sh` copies into the baseline sandbox: the full `bench_contract/`, the public
**spike-model** `software/libgemmini/gemmini.h` + `gemmini_params.h`, and the MLIR `examples/standalone`
template. **Verified fact (this audit):** that `gemmini.h` is the spike *extension model* header
(`#include <riscv/extension.h>`, `<riscv/rocc.h>`) — it declares the **funct codes** and method
signatures but does **not** contain the rs1/rs2 **bit-packing** of `config_ex/ld/st`, the accumulator
address bits, or the `GARBAGE` sentinel. Those live in the spike model `.cc` and the *bare-metal*
libgemmini encoder header (both public in the Chipyard tree, but **not** copied into the sandbox).

Consequence: the funct codes / opcode / DIM / acc_scale-is-f32 are **public and in-sandbox**; the
exact **bit-packing** is **derivable from public Gemmini sources but was not handed to the in-sandbox
baseline pre-assembled** — so for this package it is a **Merlin tooling advantage** (ported from
`merlin/python/merlin/runtime/backends/gemmini_codegen_mlir.py`).

## Audit table

All constants below are in `GemminiToLLVM.cpp` (line numbers as of this snapshot).

| # | constant / behavior | value | line | source | in baseline sandbox? | classification |
|---|---|---|---|---|---|---|
| 1 | `.insn r` custom-3 opcode form | `0x7b`, funct3 `0x3` | 108 | `XCUSTOM_ACC=3` (gemmini_params.h:7); RISC-V custom-3 | ✅ yes (params.h + ISA) | **public** |
| 2 | funct codes | CONFIG 0 / MVIN 2 / MVOUT 3 / COMPUTE_PRELOADED 4 / PRELOAD 6 / FLUSH 7 | 32 | gemmini.h:158-166 (const members) | ✅ yes (gemmini.h) | **public** |
| 3 | systolic dim / tile size | `DIM = 16` | 26 | gemmini_params.h:8 | ✅ yes | **public** |
| 4 | weight-stationary dataflow | WS bit in `CFG_EX_RS1` (`1<<2`) | 33 | Gemmini ISA docs; gemmini.h preload/compute | ✅ concept yes / ❌ exact bit no | **public concept; advantage for exact bit** |
| 5 | activation (relu) encoding | `acc_act = relu?1:0`, `(acc_act<<2)\|2` in config_st RS1 | 144,148 | Gemmini RELU activation (public); exact RS1 layout (native path) | ✅ concept / ❌ exact layout | **public concept; advantage for layout** |
| 6 | acc_scale is IEEE-f32 + clamp at i8 readout | — | 39,145 | gemmini_params.h:38,68,76-78 (`ACC_SCALE_T_IS_FLOAT`, EXP 8 / SIG 24) | ✅ yes | **public** |
| 7 | acc_scale f32 bit packing into config_st | `scaleBits = bit_cast<u32>(f32)`, `<<32` into RS2 | 39,145,148 | native path (gemmini_codegen_mlir.py:166,170) | ❌ exact packing not in sandbox | **Merlin tooling advantage** |
| 8 | i8 output readout encoding | `read_base = ACC_I8 (0x80000000)`, `elt=1` | 30,146,147 | native path (ACC addr without full_C); not in sandbox header | ❌ | **Merlin tooling advantage** |
| 9 | full-i32 readout address | `C_ACC = 0xA0000000` | 29 | native path (gemmini_codegen_mlir.py:27) | ❌ | **Merlin tooling advantage** |
| 10 | accumulator K-accumulate bit | `ACC_ACCUM = 0x40000000` (`C_ACC\|ACC_ACCUM` for kt>0) | 31,154 | native path (gemmini_codegen_mlir.py:28) | ❌ | **Merlin tooling advantage** |
| 11 | config_ex RS1/RS2 bit-packing | `(F1<<32)\|(1<<16)\|(1<<2)`, `(1<<48)` | 33,34 | native path (gemmini_codegen_mlir.py:33-34) | ❌ | **Merlin tooling advantage** |
| 12 | config_ld RS1 + row-stride RS2 (dual stride: np for W, kp for A) | `(F1<<32)\|(DIM<<16)\|(1<<8)\|1`; stride bytes | 35,120,125 | native path (gemmini_codegen_mlir.py:35,141-159) | ❌ | **Merlin tooling advantage** |
| 13 | config_st RS2 out-row-stride | `(scaleBits<<32)\|(np*elt)` | 148 | native path (gemmini_codegen_mlir.py:170) | ❌ | **Merlin tooling advantage** |
| 14 | scratchpad/accumulator tile descriptor | `pack(addr)=(DIM<<48)\|(DIM<<32)\|(addr&0xFFFFFFFF)` | 37 | native path (gemmini_codegen_mlir.py:41-42) | ❌ | **Merlin tooling advantage** |
| 15 | A-tile scratchpad slot convention | `a_slot = Kt*Nt*DIM` (after resident W tiles) | 121 | native path (gemmini_codegen_mlir.py:103) | ❌ | **Merlin tooling advantage** |
| 16 | preload → compute_preloaded → mvout sequence | per-tile WS sequence | 153-159 | ops public (gemmini.h); exact unrolled sequence from native path | ✅ ops / ❌ exact sequence | **public ops; advantage for exact sequence** |
| 17 | GARBAGE sentinel for compute bd-addr | `0xFFFFFFFF` | 28,156 | native path (gemmini_codegen_mlir.py:26) | ❌ | **Merlin tooling advantage** |
| 18 | identity scale bits | `F1 = 0x3F800000` (1.0f) | 27 | IEEE-754 (public) | ✅ | **public** |
| 19 | hardcoded dims / tile assumptions | `DIM=16`, `ceilDim` pads to 16, single resident weight | 26,38,121 | DIM public; tiling/residency layout from native path | ✅ DIM / ❌ layout | **public DIM; advantage for tile/residency layout** |

## Summary

- **Public (also available to the baseline):** the custom-3 opcode, all funct codes, `DIM=16`, the
  weight-stationary/relu *concepts*, that `acc_scale` is IEEE-f32 with clamp, and `F1=1.0f`.
- **Merlin tooling advantage (ported pre-assembled; not in the sandbox header):** the exact
  rs1/rs2 **bit-packing** of every config word, the `pack()` tile descriptor, the accumulator
  address bits (`C_ACC` / `ACC_I8` / `ACC_ACCUM`), the `GARBAGE` sentinel, the scratchpad slot
  convention, and the exact unrolled WS tile sequence.

**Honest framing:** the RoCC encoding was **ported** from the certified native Merlin path (a
tooling/authoring advantage), **not independently rediscovered** from the sandbox's public headers.
It is derivable from the broader public Gemmini sources, but the in-sandbox baseline would have to
do that derivation itself. The package's *novel* contribution is the dialect + pass architecture +
provenance, not the encoding.
