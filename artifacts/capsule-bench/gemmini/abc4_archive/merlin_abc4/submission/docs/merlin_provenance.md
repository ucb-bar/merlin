# Merlin provenance — `merlin_assisted` pilot run

> Documents *how* the Merlin/xDSL authoring approach helped. Written from what actually happened.

## 1. Merlin tools used

| Tool (path) | Used? | What you used it for |
|---|---|---|
| `targetgen/synthesize/` | no | Did not invoke the synthesizer; the dialect/passes were authored by hand against `xdsl`. |
| `targetgen/generate/` (scaffold gen) | no | No scaffold generator was run; package layout was hand-authored. |
| `xdsl_dialects/` (dialect patterns) | yes (read, as reference) | Studied the IRDL op/type + `verify_` pattern and the rewrite-pass structure, then re-authored equivalent ops/types/passes for the two backend dialects against `xdsl` directly. No `merlin` import at runtime. |
| `targetgen/contract/interface_emit.py` | no | Not used; the `merlin_iface` input is parsed directly from the capsule `capsule.interface.mlir`. |

## 2. Files generated with Merlin tooling

| submission file | origin | notes |
|---|---|---|
| `gemmini_backend/dialects.py` | hand (xDSL IRDL) | Two IRDL dialects (`merlin_iface` input + `gemmini` target), ops/types with `verify_`; modeled on `merlin.xdsl_dialects` patterns, imports only `xdsl`. |
| `gemmini_backend/iface_ir.py` | hand | Parser from `capsule.interface.mlir` → in-memory `Program`. |
| `gemmini_backend/passes.py` | hand | Conversion pass `merlin_iface` module → `gemmini` target module (both `verify()`ed). |
| `gemmini_backend/cmdbuf.py` | hand | Command-buffer emitter (frozen ABI opcodes). |
| `gemmini_backend/rocc.py` | hand | Instruction selection → tiled OS RoCC micro-sequences (`.insn r 0x7b`). |
| `gemmini_backend/gemmini_opt.py` | hand | The 4 CLI entrypoints. |

## 3. Failures encountered, and which Merlin tooling diagnosed

| round | capsule | failure plane / violations | fix | Merlin tool that helped |
|---|---|---|---|---|
| this round | `B3_conv2d_im2col_i8`, `B4_conv2d_relu_i8` | `trace_check` / `protocol_violation`: all instruction classes missing (CONFIG_LD, MVIN, PRELOAD, COMPUTE_*, MVOUT), `MVOUT count 0 != expected 3`, `k_accumulate`/`relu` modes declared but absent | `plan_kernel` in `rocc.py` had no `Conv2d` branch — the conv path emitted only fence/flush/config_ex. Added a `Conv2d` branch that lowers im2col conv to the same tiled OS matmul (`_matmul_tiles`) with M=patches, K=kh·kw·ci, N=out_ch, carrying the conv's epilogue/output_dtype/acc_scale. | none (diagnosed from the self-check redacted `trace_check.violations` + decoded instruction histogram) |

The numeric/command-buffer path was already correct: `cmdbuf.py` declares the conv activation as the `[patches, kh·kw·ci]` im2col matrix, which the harness materializes identically (`materialize_inputs`) for reference, simulate, and device, so the conv reduces self-consistently to a `[M,K]·[K,N]` matmul everywhere. Only the RoCC instruction stream was missing.

## 4. Files changed per iteration

| round | files changed | result |
|---|---|---|
| this round | `gemmini_backend/rocc.py` (add `Conv2d` branch in `plan_kernel`) | better; spike 20/20, **verilator 20/20 all_pass** |

## 5. Final-artifact integrity (self-attestation — the grader verifies independently)

- Imports any Merlin runtime / oracle code? **no** — scanned `submission/gemmini_backend/*.py`: no `import merlin`, no `runtime.reference`/`runtime.simulator`/`reference_outputs`/`outputs_match`/`xdsl_dialects.lowering`. Imports only `xdsl`.
- Self-contained (graded only through its CLI entrypoints, no Merlin import at runtime)? **yes**
- Stray Merlin authoring artifacts in `submission/`? **none**

## 6. One-line summary

The xDSL IRDL pattern from `merlin.xdsl_dialects` was a useful authoring template for the two-dialect + lowering-pass structure; the only correctness gap this round (conv2d had no RoCC lowering) was diagnosed purely from the self-check's redacted `trace_check` violations, and fixed by routing im2col conv through the existing tiled matmul. No Merlin oracle/runtime is used in the final artifact.
