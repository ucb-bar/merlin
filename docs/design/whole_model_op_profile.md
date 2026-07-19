---
title: "Design note: whole-model per-op profiler and where model time actually goes"
kind: design
status: current
owner: core
last_verified: 2026-07-19
related: [expert_gap_attribution, runtime_escape_audit]
code_refs: [merlin/python/merlin/llvmlower/op_profile.py, merlin/runtime/c/merlin_op_prof.c, merlin/python/merlin/rvvgen/k1.py, build_tools/scripts/k1_op_profile.py, merlin/python/merlin/baselines/contract.py]
---

# Whole-model per-op profiler: where model time actually goes

## Why this instrument was missing

The only whole-model instrumentation the repo had was a **two-way** split — a matmul bucket
(`-DMERLIN_DISPATCH_TIMING`, `rdtime` inside the routed GEMM shim) versus "everything else"
(`out/artifacts/kernel-mining/rvv/bench/dispatch_breakdown_measured.json`). Measured on the K1, the
matmul bucket is **1.3-6 %** of a model once the kernel is fast. So 94-97 % of model time had never
been attributed to *anything*. This note documents the profiler that closes that gap and the ranked
breakdown it produced.

## The profiler (default-OFF, framework-agnostic join key)

The board runs one monolithic `_mlir_ciface_forward`, so there is no call boundary to hook.
`merlin.llvmlower.op_profile.instrument` creates one by rewriting the IR: it interleaves a
one-instruction-cost `call @merlin_prof_mark(%id)` between the **top-level ops of `func.func
@forward`** (plus a sentinel before `func.return`). The shim `runtime/c/merlin_op_prof.c` samples
`rdtime` at each mark and credits the elapsed ticks to the *previous* op — one call per op, so
`ticks[i]` is op `i`'s cost. `build_k1_binary(..., op_profile=True)` splices this in **after** any
kernel-backend rewrite and links the shim; the harness prints `PROF <id> <ticks> <hits>`.

Each op record carries the **cross-compiler join key** (`op_profile.join_key`): `prov.fqn` when
present (the deepest `nn.Module` path that aligns a Merlin region with the SAME layer in an
ExecuTorch/GGUF/ONNX export — see `baselines/contract.py` / `baselines/_et_export.py`), else
`prov.region_id`, else the MLIR op name. So the breakdown is a framework-agnostic *graph-node -> IR
-> emitted-asm -> measured-cycles* chain, not a Merlin-only scheme. (Today's captures carry
`region_id`/`family`/`aten` but not yet `fqn`; the key degrades gracefully and upgrades for free once
the frontend tags fqn.)

Two subtleties that make it correct:

- **Op-boundary detection** keys on SSA-assignment lines (`%r = <op>`) at the function's own brace
  depth. `linalg.reduce` opens its reduction region `(%a,%b){...}` on the line *after* its
  brace-balanced first line; keying on `%r =` avoids splicing a marker into the middle of it (which
  would produce invalid IR). Covered by `merlin/tests/rvv/test_op_profile.py`.
- The default K1 RVV pipeline does **not** run `linalg-fuse-elementwise-ops` (it is env-gated behind
  `MERLIN_FUSE_POST`), so interleaving side-effecting calls cannot suppress a fusion that would
  otherwise have happened.

### Perturbation guard (the profiler must not move what it measures)

Every model is also built and run **without** instrumentation (byte-identical path). The driver
reports the profiled-vs-unprofiled median-wall delta and applies a **one-sided** guard: since
instrumentation can only *add* cost, it fails only when the profiled wall is *slower* than the
control by more than the board noise floor (default 1.9 %). Measured deltas:

| config | profiled wall | control wall | delta | verdict |
|---|---|---|---|---|
| bitvla native | 2497.6 ms | 2503.7 ms | **-0.245 %** | ok |
| openvla native | 5846.7 ms | 5855.9 ms | **-0.156 %** | ok |
| openvla ours-routed | 692.7 ms | 689.3 ms | **+0.494 %** | ok |
| bitvla ours-routed | 178.2 ms | 182.4 ms | -2.289 % (profiled faster) | ok (noise, not a cost) |

Profiler coverage (`sum(per-op ticks) / rd_time bracket`) is 0.997-1.000 across all runs — the marks
account for essentially the whole compute region.

## What the breakdown shows

Two regimes, because the answer flips completely with kernel quality.

### Regime 1 — default compiler path (`hand_v0` native RVV): the matmul *lowering* dominates

| model | wall | contraction | scalar (non-contraction) |
|---|---|---|---|
| bitvla | 2497.9 ms | **2415.9 ms (96.7 %)** | 82.1 ms |
| openvla | 5854.7 ms | **5457.1 ms (93.2 %)** | 397.6 ms |

On our *own* default codegen the matmul is not 6 %, it is 93-97 %. Individual small-M matmuls are
catastrophic: one bitvla `linalg.matmul -> tensor<32x1024xf32>` measures **478 ms**; openvla's
72 matmuls sum to 5.5 s. An `rvv_audit` cross-check of the emitted `forward` symbol shows only
**20.8 % static vector coverage** — even the "contraction" ops are scalar-dominated (addressing,
per-tile `memrefCopy`, masked tails). This is the same defect named in `expert_gap_attribution` and
the K1-PMU memory (per-tile `memrefCopy`) and the small-M scalar-fallback finding — now *measured*
per op. It is the province of the kernel-gap workstream (`cg_`); this profiler quantifies it but does
not duplicate the fix.

### Regime 2 — fast kernel routed in (`ours` v3 shim): the non-matmul 94% is exposed

Routing the 2-D f32 matmuls to our accumulator-resident v3 shim drops openvla to 692 ms (matches the
existing dispatch-breakdown wall) and bitvla to 178 ms. *Now* the per-op profiler answers the
question the two-way split never could:

**openvla (692 ms), ranked by measured ms:**

| op | ms | % model | vectorized? |
|---|---|---|---|
| `linalg.transpose` (45 ops) | **392.8** | **56.7 %** | scalar |
| `call` (26 routed matmuls) | 149.0 | 21.5 % | v3 kernel |
| `linalg.generic` (190 ops) | 142.5 | 20.6 % | mixed (conv2d 41 ms, attention `batch_matmul`, elementwise) |

The two largest single ops are transposes: `576x2304` = **198 ms** and `1536x384` = **122 ms**.
**openvla spends more time transposing (393 ms) than doing all its matmuls (149 ms).**

**bitvla (178 ms):** `linalg.generic` 90.6 ms (incl. 4 attention `batch_matmul` at ~15 ms each = 61 ms,
which are *not* routed — only 2-D matmul routes), `linalg.transpose` 45.1 ms, routed matmul `call`
36.7 ms. Scalar non-matmul is **66 %**.

## Prioritized: what is left on the table (ranked by measured ms)

1. **`linalg.transpose` — ~393 ms on openvla (57 % of the model), ~45 ms on bitvla.** Scalar,
   unvectorized layout transposition, falling through `convert-linalg-to-loops`. The RVV schedule
   vectorizes only `matmul`/`batch_matmul`, so every transpose is scalar. Biggest lever by far: fold
   the transpose into its consumer matmul (transpose-`b` GEMM, so it never materializes) or emit a
   vectorized transpose. This is a *compiler-general* capability, not a kernel hand-tune.
2. **Non-2-D contractions stay on the slow native path — ~60-80 ms.** The fast routing (ours/xnnpack)
   covers only rank-2 `linalg.matmul`; attention `batch_matmul` (bitvla 61 ms, openvla ~37 ms) and
   `conv2d` (openvla 41 ms) still use the native lowering with the 20.8 %-coverage defect. Extending
   the routable set (or fixing the native contraction lowering) recovers this.
3. **The matmul lowering itself — 2.4-5.5 s on the default path.** Owned by `cg_`; measured here for
   completeness. Routing to the v3 shim already recovers ~8-14x, so the win is in the *lowering*
   (memrefCopy / scalar addressing), not the ISA.
4. **Elementwise `linalg.generic` — bitvla ~25 ms.** Smaller; a post-matmul fusion lever
   (`MERLIN_FUSE_POST`) already exists to collapse these.

## Honest limits

- `rdtime` is the 24 MHz platform counter (~41.7 ns/tick, `cycle_accurate=false`); per-op numbers are
  meaningful only in aggregate (by op/family), which is how they are reported.
- `buffer-hoisting` moves allocs toward the entry, so allocation cost drifts to the first mark
  interval; allocation is captured in aggregate by the wall, not per-op.
- The `vectorized` flag is the pipeline's *intent* (contraction family), not proof the asm is vector;
  the `rvv_coverage` cross-check (20.8 % on `forward`) is the measured corrective.
- Runs are `n=3`, cos-gated `>= 0.9999` each; for the 178 ms bitvla-routed config, board noise (~2 %)
  exceeds the floor, so its perturbation is inconclusive in magnitude (but negative — no slowdown).

## Artifacts

`out/artifacts/measurements/k1_spacemit/{bitvla,openvla}_fp32_consistent/op_profile.json` (native) and
`op_profile_ours.json` (fast-kernel), each with the full per-op table, family/op/join-key rollups, the
perturbation control, and the rvv-coverage cross-check. Regenerate:
`MERLIN_COMPILE_TIMEOUT_S=3600 .venv/bin/python build_tools/scripts/k1_op_profile.py --model <dir>
[-n 3] [--kernel-backend ours]`.
