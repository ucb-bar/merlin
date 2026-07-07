# Target-Generation (Bringup) Journey — driving example, snippets, speaker notes

**Purpose.** The kernel-mining journey (`KERNEL_JOURNEY.md`) shows how we *improve the compiler*. This
is the parallel story for the *other* half of the deck (slides ~14–20, + backup 50/54): **bringing up a
NEW hardware target in MLIR.** Same idea — start from one concrete problem and follow it end-to-end,
with the **real committed artifact at every step**, instead of a cold diagram.

**The driving example: one Gemmini int8 single-tile matmul capsule, `A2_single_tile_matmul`.**
Gemmini is an int8 systolic accelerator (i8×i8→i32, no f32 — verified from the RTL facts). The "problem"
the target author starts from is: *here is a 16×16 int8 matmul I need to run on this accelerator I'm
bringing up — make it produce bit-exact i32, certified against the RTL.* Every step below is a real file.

The persona: *you're bringing up your accelerator. In a heavy framework (IREE), adding an external
dialect means re-linking the whole compiler and carrying a runtime not built for C production. Here you
write a small out-of-tree (OOT) package — 4 CLI entrypoints + your own dialect in C++/TableGen or IRDL —
and the shared harness certifies it against the real RTL.*

---

## The journey, step by step (real committed snippets)

### ① THE PROBLEM / INPUT — the capsule + the interface MLIR
*Path:* `experiments/gemmini_capsule_bench_v0/scripts/pilot_capsules/A2_single_tile_matmul/`
The author starts from a frozen, target-agnostic interface (`merlin_iface`) + a golden reference:
```yaml
# capsule.yaml — the task
name: A2_single_tile_matmul
operation: {op: matmul, attributes: {output_dtype: i32}}
inputs:
- {name: W,  role: weight, shape: [16,16], dtype: i8}
- {name: A0, role: input,  shape: [16,16], dtype: i8}
numeric_policy: {compare: exact_int, dtype: i32}
expected: {instruction_classes: [FLUSH, CONFIG_EX, CONFIG_LD, MVIN, CONFIG_ST, PRELOAD,
                                 COMPUTE_PRELOADED, MVOUT]}
```
```mlir
// capsule.interface.mlir — the frozen merlin_iface input (same for EVERY target)
%W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<16x16xi8>) -> !merlin_iface.resident
%acc0  = merlin_iface.matmul %A0, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
%Y0    = merlin_iface.commit %acc0 {output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
```

### ② THE CONTRACT — the only thing the author must satisfy
*Path:* `merlin/contract/mlir_oot_backend_contract.yaml`. Four CLI entrypoints + one kernel symbol —
invoked as subprocesses, never imported. That's the entire surface (vs re-linking IREE):
```yaml
entrypoints:
  parse:                     # verify interface.mlir
  lower_interface_to_target: # merlin_iface -> YOUR target dialect      (stdout)
  emit_command_buffer:       # interface     -> command_buffer.json
  lower_target_to_llvm:      # YOUR dialect  -> LLVM/RoCC MLIR           (stdout)
kernel_abi:
  symbol: gemmini_kernel
  signature: "void gemmini_kernel(ptr weight, ptr lhs_0.., ptr out_0..)"
```

### ③ THE OOT DIALECT — what the author fills in (their own dialect)
*Path:* `…/submission/mlir_oot/include/gemmini/Gemmini.td` (C++/TableGen) — or IRDL
(`merlin/targets/gemmini/contracts/irdl/merlin_iface.irdl.mlir`). The author defines target ops:
```tablegen
def G_PackOp   : GOp<"pack">   { let arguments=(ins AnyType:$src, StrAttr:$layout);
                                  let results=(outs G_ResidentType:$result); }
def G_MatmulOp : GOp<"matmul"> { let arguments=(ins AnyType:$lhs, G_ResidentType:$rhs);
                                  let results=(outs G_AccType:$result); }
def G_CommitOp : GOp<"commit"> { let arguments=(ins G_AccType:$src, StrAttr:$output_dtype, ...); }
```

### ④ THE LOWERING — interface → target dialect → RoCC
*Paths:* `…/Conversion/InterfaceToGemmini.cpp` (merlin_iface.matmul → gemmini.matmul, i8×i8→**i32** acc),
then `…/Conversion/GemminiToLLVM.cpp` emits the weight-stationary RoCC `.insn` sequence (real A2 order,
from the decoded trace):
```text
.insn r 0x7b, 0x3, <funct>, x0, $rs1, $rs2     # RoCC custom-3
FLUSH(7) → CONFIG_EX(0) → CONFIG_ST(0) → CONFIG_LD(0) → MVIN(2 weight)
        → CONFIG_LD(0) → MVIN(2 lhs) → PRELOAD(6) → COMPUTE_PRELOADED(4) → MVOUT(3)
```

### ⑤ THE COMMAND BUFFER — the frozen ABI artifact
*Path (canonical, verified):* `experiments/gemmini_capsule_bench_v0/runs/_audit/rb_pilot_cpp_01/runs/`
`gemmini-capsule-bench/A2_single_tile_matmul/generated/command_buffer.json` (schema-valid):
```json
{ "target": "gemmini", "abi_version": "0.1", "backend": "gemmini_oot_cpp",
  "tensors": {"W": {"dtype":"i8","role":"weight","shape":[16,16]},
              "A0": {"dtype":"i8","role":"input","shape":[16,16]}},
  "commands": [
    {"opcode": "RES_PACK",        "operands": {"src":"W","dst":"W_res"}, "attributes": {"layout":"packed_rhs"}},
    {"opcode": "MATMUL_RESIDENT", "operands": {"lhs":"A0","rhs":"W_res","dst":"acc0"}},
    {"opcode": "COMMIT",          "operands": {"src":"acc0","dst":"Y0"}, "attributes": {"output_dtype":"i32"}},
    {"opcode": "EVICT",           "operands": {"handle":"W_res"}} ] }
```

### ⑥ CERTIFY — the oracle ladder L0–L3 (the sign-off is real RTL)
*Paths (verified):* `…/runs/_audit/rb_pilot_cpp_01/…/A2_single_tile_matmul/generated/instruction_trace.json`;
cycles in `experiments/gemmini_capsule_bench_v0/reports/cycles_by_capsule.md` + `full_suite_audit.md`.
```text
L0 golden numeric  →  golden i32 == reference(cb)            (bit-exact)
L1 consistency     →  reference(cb) == simulate(cb)
trace check        →  decoded RoCC classes == capsule.expected   ← rocc_decode
L2 spike           →  spike_out == golden                   (~secs)
L3 verilator (RTL) →  rtl_out == golden  ·  A2 = 299 cycles  (THE sign-off, ~2.5 min)
```
```json
// instruction_trace.json — REAL decoded class sequence for A2 (custom-3 0x7b/0x3)
"abi": {"custom_opcode": "0x7b", "funct3": "0x3"},
"classes": ["FENCE","FLUSH","CONFIG_EX","CONFIG_ST","CONFIG_LD","MVIN",
            "CONFIG_LD","MVIN","PRELOAD","COMPUTE_PRELOADED","MVOUT","FENCE"]
```

### ⑦ THE PAYOFF — what bringup delivered
*Path:* `experiments/gemmini_capsule_bench_v0/reports/METHODOLOGY.md` + `full_suite_audit.json`.
- **25/25 capsules RTL-certified** (20 public + 5 hidden), 0 verilator sim-skips — across 3 families:
  **A** ISA primitives (config/movement/matmul/acc-scale/relu/padding), **B** layers (quantized linear,
  conv2d-im2col, relu), **C** model slices (MLP, attention Q/K/V·QK·PV). Real L3 cycles: A2=299, B0=335,
  C0=1209.
- **Language-agnostic, proven:** a C++ OOT `gemmini-opt` package AND a Python package both pass the
  *same* runner identically — the contract is real, not a Merlin-only moat.
- **Lightweight:** the production C runtime is ~511 LOC (no interpreter/scheduler); the command buffer is
  the ABI, not a dialect — so the bringup output is a bare-metal RV64 ELF.

---

## Slide integration (maps to the deck, `OSCAR Merlin-9.pdf`)
- The bringup section is slides **14–20** ("Bringup of your target in MLIR" → "MLIR dialect target
  bringup" → "Support for different backends" → "Agentic MLIR dialect + passes" → "Experiments Setup"),
  plus backup **50** (Saturn OPU: intrinsic → TableGen → LLVM-intrinsic → MLIR pattern) and **54**
  (Experiment results: RTL-checks vs oracle, "where each Gemmini oracle sits").
- Drive that whole section with **A2**: ① capsule/interface → ② contract → ③ your dialect →
  ④ lowering/RoCC → ⑤ command buffer → ⑥ oracle ladder (A2 = 299 cycles, RTL-signed-off) → ⑦ 25/25.
- Slide 18 (Agentic MLIR dialect + passes) and 54 already show the oracle ladder + RTL-checks confusion
  matrix (TN 242 · FP 0 · FN 17 · TP 124) — they are the ⑥/⑦ evidence; the new front (①–⑤) eases people
  in with the concrete capsule before those.

## Honest notes
- Gemmini is **int8-only** (i8×i8→i32) by construction — that's the architecture, not a limitation of
  the demo. (This is also the honest reason the RVV kernel-mining example stays f32: different track.)
- The artifacts are all real and committed; the only "writing" needed is the narrative glue (this doc).
- Golden output values are withheld for benchmark integrity; a public talk shows the *structure* (op
  classes, cycle counts, pass/fail), not the secret golden tensors.
