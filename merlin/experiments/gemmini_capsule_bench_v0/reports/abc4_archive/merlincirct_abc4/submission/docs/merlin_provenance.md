# Merlin provenance — `merlin_assisted` pilot run

## 1. Merlin tools used

| Tool (path) | Used? | What you used it for |
|---|---|---|
| `targetgen/synthesize/` | no | Dialect + passes authored by hand against the public `xdsl` API. |
| `targetgen/generate/` (scaffold gen) | no | No scaffold generator invoked. |
| `xdsl_dialects/` (dialect patterns) | read (as reference) | Followed the IRDL idioms in `merlin/xdsl_dialects/targets/toynpu.py` and `interface.py` — `irdl_op_definition`/`irdl_attr_definition`, field-annotation operands/attributes, `verify_()` checks, and `PatternRewriteWalker`/`RewritePattern` lowering — as the model for our two dialects and the lowering pass. Not imported at runtime. |
| `targetgen/contract/interface_emit.py` | no | Input interface MLIR read directly (contract states it is regex-parseable, decoupled from xDSL); we then build real xDSL IR from it. |

The only runtime framework dependency is the public `xdsl` library itself
(`xdsl.irdl`, `xdsl.pattern_rewriter`, `xdsl.dialects.builtin`). No `merlin`
package is imported.

## 2. Files generated with Merlin tooling

| submission file | origin | notes |
|---|---|---|
| `gemmini_backend/dialects.py` | hand | Two IRDL dialects (`merlin_iface` input, `gemmini` target) with ops/types/verifiers. |
| `gemmini_backend/frontend.py` | hand | Reads the interface MLIR and builds a verified `merlin_iface` xDSL module. |
| `gemmini_backend/passes.py` | hand | `PatternRewriteWalker`-driven rewrite patterns lowering `merlin_iface` → `gemmini`. |
| `gemmini_backend/program.py` | hand | Walks the verified target module into a small record list. |
| `gemmini_backend/cmdbuf.py` | hand | Emits the schema-valid command buffer (incl. `params.im2col_recipes` for conv). |
| `gemmini_backend/rocc.py` | hand | WS-matmul tile schedule → RoCC `.insn` `llvm.func @gemmini_kernel`. |
| `gemmini_backend/conv.py` | hand | im2col geometry (conv → resident matmul over the derived activation). |
| `gemmini_backend/cli.py` | hand | The four contract entrypoints. |

## 3. Failures encountered, and which Merlin tooling diagnosed

| round | capsule | failure plane / violations | fix | Merlin tool that helped |
|---|---|---|---|---|
| this run | A1_mvin_mvout | `command_buffer` — `KeyError: 'lhs'`, then `reference != None` | movement `VECTOR_MAP` operands must be `lhs`/`dst` with `combine:"identity"`; output tensor must be declared `role:"output"` so the reference collects it | none (read the ABI: `command_buffer_abi.yaml` + schema) |
| this run | A7_edge_padding | `spike` functional mismatch (only non-DIM-multiple capsule) | mvout DRAM offset must use the *padded* row stride `Np`, not logical `N` (the runner harness allocates/prints output padded to a DIM multiple) | none (derived from the runner harness DRAM layout contract) |
| this run | B3/B4 conv2d | `target_to_llvm` — `NotImplementedError` (conv stub) | lower conv to a derived im2col activation (`params.im2col_recipes`) + standard resident WS matmul over `[M,K]·[K,Co]`; reuse the matmul tile emitter | none (followed the `conv_im2col`/`im2col_recipes` ABI) |

## 4. Files changed per iteration

| round | files changed | result |
|---|---|---|
| this run | `manifest.yaml` (point at `gemmini_backend/cli.py`) | 16/20 → baseline on the xDSL backend |
| this run | `cmdbuf.py` (movement operands/role) | A1 pass → 17/20 |
| this run | `rocc.py` (padded mvout stride) | A7 pass → 18/20 |
| this run | `conv.py`, `cmdbuf.py`, `rocc.py` (conv im2col) | B3+B4 pass → 20/20 spike |

## 5. Final-artifact integrity (self-attestation — the grader verifies independently)

- Imports any Merlin runtime code? **no** — grep over `gemmini_backend/` finds no
  `import merlin` / `from merlin` / `reference_outputs` / `outputs_match` /
  `xdsl_dialects.lowering` / `pipeline.execute`.
- Self-contained (graded only through its 4 CLI entrypoints, only `xdsl` at runtime)? **yes**
- Stray Merlin authoring artifacts in `submission/`? **none** (a legacy `gemmini_oot/`
  string-emitter package is present but unused — the manifest points only at `gemmini_backend/`).

## 6. One-line summary

The xDSL framework gave us the dialect/verifier/rewrite-pass scaffolding idioms
(via the `xdsl_dialects` patterns) so the backend is a real IRDL dialect + lowering
pass rather than a string emitter; it offered no direct help on the three numeric/ABI
bugs (movement operand schema, padded output stride, conv im2col) — those came from
reading the command-buffer ABI and runner-harness contracts.
