# gemmini_universal target — Jack's "Universal" ResNet-50 Gemmini (Alveo U250 / FireSim)

A **separate device from `gemmini`**, not a variant of it. Same generator, same 16x16 int8 mesh, same
256 KiB scratchpad and 64 KiB accumulator — and four parameter values that each REMOVE programs
`gemmini` executes. **Do not mix the two.** Nothing in this package reads or writes
`merlin/targets/gemmini/`.

| | `gemmini` | `gemmini_universal` |
|---|---|---|
| dataflow | WS + OS | **WS only** (OS branch constant-folded dead) |
| accumulator readout | full-width **and** narrow | **narrow only** — store DMA `UInt<128>` vs `UInt<512>` |
| execute stage | may read acc / write spad | **neither** |
| D operand | live | **hardcoded GARBAGE_ADDR** |
| ABI header | `3758ae96…` (defines `ACC_READ_FULL_WIDTH`) | `d6db18f8…` (does not) |
| `prototype_scope.output_dtype` | `i32` | **`i8`** |

**The readout is the fact that matters.** No int32 tensor can leave this accelerator. That is why
this target's capsule-bench cohort is 28 graded rows against gemmini's 98 — see
`merlin/experiments/capsule_bench/targets/gemmini_universal/target_experiment.yaml`.

## What is in here
- `contracts/target_contract.yaml` — intent + the ABI encoding surface. Its `readout:` block is the
  one thing with no counterpart in gemmini's contract, and every value in it is derived with the
  artifact named. Read its `gaps:` before trusting anything.
- `contracts/rtl_facts/facts.json` — the **promoted pin**, hand-driven (mlc/CIRCT over the extracted
  Gemmini subtree + the raw harness FIRRTL), promoted from
  `out/artifacts/cache/rtl_introspect/gemmini_universal/derived_facts_partial.json`. It says so in
  `generator.name`: it is NOT a `circt_introspect` run and must not be attributed to one. Heavy run
  scratch stays in that purgeable cache, never here.
- `contracts/abi/` — the build's own elaborated `gemmini_params.h` plus `abi.yaml` recording its
  digest, its source, its corroborations and the exact one-line diff against merlin's gemmini header.

## Two UNKNOWNs closed by derivation, and how
- **`custom_opcode`** was UNKNOWN because it derives from a target contract that did not exist. It is
  now **derived from the elaboration**: module `RoccCommandRouter` in this config's harness FIRRTL
  carries exactly one opcode comparison, `eq(UInt<7>(0h7b), …)` — 0x7b is RISC-V custom-3, so the slot
  is 3. Uniqueness was checked (one module, one instance, one such node in 84 MB). Corroborated
  independently by the attested header's `#define XCUSTOM_ACC 3`. `OpcodeSet.custom3` in the Scala is
  a *declaration* and was not used as the evidence.
- **The ABI** was UNKNOWN because no generated header for this config was known. It is now the FPGA
  build's own `gemmini_params.elaborated.h`, and the reason that is provenance rather than shape
  agreement is that **this device's Spike extension was built from byte-identical bytes**.

## No reference backend, deliberately
`plugin.backend` is absent. Writing a backend for this device is the work under test; symlinking
gemmini's would be worse than nothing, because that codegen was written against a device *with*
full-width accumulator readout and would emit mvouts this hardware cannot service while claiming to be
this target's reference.

## L2 is not a faithful oracle for the readout width — MEASURED
`spike --extlib=<jack's libgemmini.so> --extension=gemmini` is the L2 model, declared with its digest
under `runner.spike_extension`. It **passes** a graded capsule bit-exactly (A4_acc_scale_i8, errors=0)
and it **also passes** a full-width int32 mvout that this RTL cannot build — the extension implements
the full-width accumulator read unconditionally off the address bit, not gated on
`ACC_READ_FULL_WIDTH`. Both measurements:
`out/artifacts/capsule-bench/gemmini_universal/l2_oracle_smoke_20260909T092925Z_328c2e2/`.
**Never cite L2 as evidence about this device's readout width.** L3 has never been run for this target.

## Provenance
Pins `gemmini_universal_rtl` + `gemmini_universal_chipyard`, root env **`MERLIN_UNIVERSAL_CHIPYARD`**.
Do **not** point `MERLIN_CHIPYARD` at that tree: it would silently re-attribute every existing gemmini
claim to this device. `verify()` reports both pins `ok=false` with `sources: []` because git refuses
another user's repository — the content half is measured out of band and recorded in the facts under
`provenance.declared_read_path_digests` (11/11 and 1/1 declared digests match; revision independently
confirmed with a per-invocation `git -c safe.directory=…`, never a shared config edit).
