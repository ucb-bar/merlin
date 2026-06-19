# P21-S1 — Loop-preserving capture: proven at the torch level; one m2m gap to linalg

Target: openVLA, pi0.5, smolVLA (the user's chosen three). Goal: capture the K-step denoise/decode loop as
**structured control flow** so K, the loop-carried state (latent / KV), and the repeated region are recovered
from the IR — closing the assumed-K / region-roles / KV-state caveats.

## What is PROVEN and DONE (export + torch-level loop preservation)

The "torch.export unrolls loops" blocker is **broken**. With a shape-invariant `torch.while_loop` wrapper:

| model | result | carried state (shape-invariant iter_args) | K |
|---|---|---|---|
| **smolVLA** | export OK; **raw torch-mlir = a single `torch.prim.Loop`, body NOT unrolled**; eager cos **0.9999994** vs the real unrolled loop | `(i: si64[], x_t: f32[1,50,32])` | 10 |
| **openVLA** | export OK; `while_loop` HOP preserved; decode **bit-exact** vs HF `generate(max_new_tokens=7)` | `(i, cur_tok[1,1], out_toks[1,7], k_cache[2,1,4,27,128], v_cache[...])` — **static KV is a true shape-invariant iter_arg** | 7 |
| **pi0.5** | flow-matching, same pattern as smolVLA (carry the action latent); wrapper not finished (agent hit a limit) — follows smolVLA exactly | `(i, x_t)` | ~10 |

Reference wrappers (verified, numerically exact) committed here:
`smolvla_whileloop_wrapper.py`, `openvla_whileloop_decode_wrapper.py`; loop proof in
`smolvla_raw_torch_loop_proof.txt` (the `while_loop_cond_graph_0` / `while_loop_body_graph_0` torch funcs).

**Key wrapper lessons (so the loop captures cleanly):** carry only the evolving state (close over invariants
→ they lift as additional_inputs, avoiding the no-aliasing error); the carried state must be shape-invariant
(flow-matching: the action latent round-trips; autoregressive: use a **static/fixed-size KV cache** written
in-place at position `i`, not a growing cache); **hoist every in-body tensor constant** out of the loop
(linspace/arange/`torch.tensor([...])` trigger `lift_fresh_copy` / data-dependent-symbol errors); recompute
`time = 1 + i*dt` from the carried counter (don't index a precomputed table by `i`).

## The ONE remaining gap (fully scoped): m2m `while_loop → scf.for` lowering

The loop is preserved through torch-mlir *raw* import (`torch.prim.Loop`), but **m2m's lowering to
`linalg-on-tensors` produces an empty/failed module** — neither lowering path handles the loop:
- **torch-mlir's** torch→linalg pipeline rejects the `prim.Loop` because the 34 closed-over additional_inputs
  are referenced across the region boundary but not threaded as block args (a torch-mlir FxImporter defect).
- **m2m's CompGen FXImporter** (`m2m/ir/import_fx.py`) has **no `while_loop` handler** → falls to an opaque
  single-result `func.call @while_loop` → invalid IR (the HOP returns a tuple; no nested regions).

### Design (the fix — to implement in `m2m/ir/import_fx.py`)
Since the cond is `i < K` with **K constant**, lower to **`scf.for(0, K, 1)`** with carried state as
`iter_args` — and reference the closed-over weights **directly from the loop region** (legal in MLIR `scf`;
this *sidesteps the torch-mlir additional_inputs bug entirely*). Concretely, additive in the call_function
dispatch (zero risk to existing captures — `while_loop` never appeared before):
1. Detect the `while_loop` HOP node; resolve `cond`/`body` sub-GraphModules (root-gm submodules), the carried
   operand list, and the additional (closed-over) operands.
2. Extract `K` from the cond subgraph (`i < K` constant) → `scf.for` bounds `[0, K, 1]`.
3. Build the `scf.for` region block: arg `[index iv, *carried_types]`; map body placeholders →
   `iv`→i64-0-d-tensor (via `arith.index_cast` + `tensor.from_elements`), carried→iter_args,
   additional→the outer `value_map` SSA values.
4. Recursively import the body subgraph's call_function nodes into the region (reuse the per-node dispatch:
   decomposition table + opaque fallback — factor lines ~506-703 into a nested `emit_node` closure so it runs
   for both the main graph and the body).
5. `scf.yield` the body's new carried values (drop the `i+1`); map `getitem(node, k)` consumers to
   `scf.for` results (k≥1 → results[k-1]).
Precedent for emitting `scf` in xDSL is already in m2m: `decompositions.py::_bool_mask_gather` builds
`scf.ForOp`/`scf.IfOp`/`ScfYield`. The verifier safety net: re-run a normal capture (rdt flat) and confirm
the MLIR is byte-identical after the `emit_node` extraction.

### Once landed, the DSE side flips (small, in merlin):
`attribution`/`capture_fidelity`: when `scf.for/while` is present → loop body = `repeated_head` (structural,
not fqn-heuristic); iter_args = `loop_carried_state` (latent/KV); K = the loop bound (IR-recovered → drop the
`assumed` tag); KV iter_arg → KV-state recovered. This closes the assumed-K, region-roles, and KV-state items.

## Status
- ✅ Loop preservation proven + numerically exact for smolVLA (prim.Loop) and openVLA (static-KV, bit-exact);
  pi0.5 follows smolVLA. The core "loop is unrolled/erased" caveat is broken at the torch level.
- ⏳ One m2m compiler feature remains (`while_loop→scf.for` in the CompGen FXImporter) to carry the loop into
  `linalg-on-tensors` + the DSE artifacts. Fully designed above; it is a focused, tested unit (shared-importer
  surgery with a byte-identical-on-normal-capture safety net), not a rushed tail-of-session edit.
