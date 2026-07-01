# Saturn-vectors: cross-family generalization findings (honest)

The go/no-go question: **does Merlin's targetgen/certification architecture survive a target
that does NOT fit the resident-matmul shape (`RES_PACK/MATMUL_RESIDENT/COMMIT/EVICT`)?** We took
a vector/SIMD family (`vector.map` elementwise, `vector.reduce`) all the way to certified RTL.

## Verdict: it generalizes — at the ABI, codegen, oracle, and isolation layers. With caveats.

### What held (positive evidence)

- **ABI / reference / simulator (SV1):** the command-buffer + `simulate`/`reference` expressed the
  vector family **additively, no fork** — new opcodes `VECTOR_MAP`/`VREDUCE`, an output-role
  convention, and three small `Tensor` ops (`ew_add`/`ew_mul`/`reduce_sum`). The matmul path was
  untouched (regression-green). VEC2 (a dot product as `mul→reduce`) shows a *contraction
  expressed in the vector family*, not as matmul.
- **MLIR-faithful codegen (SV-MLIR):** vector compute lowers as `linalg.generic` (+ a reduction)
  through merlin's **real MLIR→LLVM compiler** (`lower_model`) and is **bit-exact on host** for
  VEC0/VEC1/VEC2 — no hand-written kernel. Accelerator **custom instructions** are emitted from
  MLIR via `merlin.inline_asm → llvm.inline_asm → .insn` (verified: a Gemmini RoCC CUSTOM-3 op
  `0x00b5307b` in a real object, no LLVM fork).
- **RVV oracle (SV3):** the vector kernels run **bit-exact on spike rv64gcv** (functional) and on
  the **Saturn-OPU Verilator RTL** (`derived_from_rtl=true`: VEC0 202 cyc, VEC2 743 cyc) — full
  RTL parity with the Gemmini story, on a different oracle / ISA / op family.
- **Isolation + plug-and-play:** the family is an **isolated package**
  (`artifacts/targets/saturn_vec/hand_v0/`) with its own xDSL dialect + lowering, loaded
  dynamically — not hardcoded in core. Same isolation as Gemmini (which was un-baked from core).

### What did NOT generalize cleanly (honest caveats — findings, not failures)

1. **The three-way gate degenerates to two-way for vectors.** Matmul's reference *bypasses
   residency* to cross-check the simulator; the vector family has no residency optimization, so
   `reference ≡ simulator` trivially. The meaningful gate is **merlin ≡ RTL oracle**.
2. **Native RVV vectorizes only contractions.** Merlin's `transform-dialect` RVV schedule covers
   `linalg.matmul`/`batch_matmul`; elementwise/reduction `linalg.generic` currently lower to
   **scalar** RVV-target code (correct, certified, but not vector-instruction'd). Getting real RVV
   for the vector family needs an added elementwise/reduction transform schedule — a concrete,
   bounded next step, not an architectural blocker.
3. **The contract/schedule "decision" layers are matmul-specific.** They encode
   residency/packing decisions; the vector family lowers **directly** from its interface dialect
   to the command buffer (no residency decisions). The full upper-dialect plane was *not*
   rebuilt for vectors — by design: the certified interface is the command buffer.

### Agent autonomy (SV4) — nuanced

With a *full recipe* (Gemmini) the agent generalized to held-out shapes first try. With only
**RVV intrinsic signatures and no recipe** (vector), it **derived the elementwise kernel
correctly but failed first-shot on the held-out reduction** (crashed on an intermediate-tensor
structure it was never shown). When the reduction was **visible** (in the repair set), it got all
rungs right on round 0. So: the agent **interpolates within demonstrated structure but does not
extrapolate to structurally-novel held-out cases** — strong within coverage, weak beyond it.

## Bottom line for the thesis

This is the cross-family evidence the prior assessment said was missing: the same
architecture certified a **non-matmul accelerator family against real RTL**, the codegen is now
**merlin-faithful** (MLIR/LLVM + inline-asm, not C), and generation is **isolated/plug-and-play**.
The honest gaps are bounded and named (elementwise RVV schedule; matmul-shaped decision layers;
agent extrapolation limits) — none falsify the approach; they scope the next work. The thesis
("the architecture explores the HW/SW boundary across families") now has one real second family,
not just a matmul engine that trivially fit.
