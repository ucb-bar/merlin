---
title: RVV beam search — reproducible expert-driven compiler improvement, gated on real K1 speedup
kind: guide
status: current
owner: rvvgen
last_verified: 2026-07-20
related: [kernel_mining, rvv_e2e, adding_a_target, dse_guidance]
code_refs:
  - merlin/python/merlin/rvvgen/beam.py
  - merlin/python/merlin/rvvgen/beam_cli.py
  - merlin/python/merlin/rvvgen/wholemodel_proposer.py
  - merlin/python/merlin/rvvgen/fork_from_action.py
  - merlin/python/merlin/rvvgen/runner.py
  - merlin/python/merlin/rvvgen/k1.py
  - merlin/python/merlin/kernels/action_catalog.py
  - merlin/python/merlin/targetgen/publish.py
  - build_tools/scripts/rvv_chia_beam.py
  - merlin/tests/rvv/test_rvv_beam.py
---

# RVV beam search

The beam search is Merlin's **learn-from-experts loop**: it lifts a target-agnostic Common Compute
Abstraction (CCA) from an expert kernel's assembly, diffs it against the CCA our compiler currently
emits, and searches compiler edit-points (flags → knobs → heuristics → passes → codegen) to **close
the divergences**, keeping only forks that are numerically correct AND measurably faster on the real
SpacemiT K1 board than the frozen hand-written baseline.

Every claim in the loop is fail-closed: correctness (cos/rel vs a golden) gates first, and a speedup
is credited only when the gate passes and the measurement is real K1 silicon — never a proxy.

> **Current architecture (whole-model).** The objective is now the WHOLE-MODEL wall, not an isolated
> kernel (kernel-only ranking is anti-correlated with e2e — the `ours_v3` trap). `merlin-rvv-beam
> --model-dir <whole-model-bundle> --proposer wholemodel` seeds from frozen `hand_v0`, proposes the
> byte-traffic-ranked whole-model levers (transpose fusion, per-matmul MR, self-copy erase, reduction
> vectorization, activation), measures each on the board, and stacks the winners across depth. The
> full architecture — the two CCAs (expert vs compiler), the graph→IR→asm→cycles analysis, the
> action-catalog ladder, the correctness tiers, and the autonomous experiment — is in
> [CCA beam design](../design/beam_cca_architecture.md). Reproduction recipe: §5 of
> [reproducibility](reproducibility.md).

## The pieces

| concern | code | what it does |
|---|---|---|
| frozen control | `out/artifacts/targets/rvv/hand_v0/` | the hand-authored, UNoptimized baseline the beam forks FROM and measures AGAINST. Never modified (guarded by `test_impr_features`; the beam asserts it byte-unchanged pre/post). |
| expert CCA | `kernels/cca.lift_asm` + `kernels/decode/rvv` | deterministically lifts the CCA (compute/vector/memory facets) from a decoded expert objdump. **No LLM authors it.** |
| proposer | `rvvgen/fork_from_action.propose_forks_from_cca` | routes each CCA divergence through `kernels/action_catalog` to a concrete fork (a knob/feature override) or an honest deferred work-item. |
| certify | `rvvgen/runner.certify_rvv` | builds one fork, runs the K-ladder (K2 build, K3 spike correctness, K5 K1 build+run, K6 speedup), gates on cos/rel. K1 cos-vs-golden is the correctness gate when spike is absent. |
| engine | `rvvgen/beam.run_beam` | gen-by-gen: propose → mint fork → certify → rank on **real K1 speedup** → keep top-k → escalate unmet promises → next gen. Writes `beam_tree.yaml`. |
| escalation | `beam._escalations` + `action_catalog.route_escalated` | when a fork's asm did NOT achieve its promised facet, route the next-stronger class (knob → heuristic → pass → codegen). |
| orchestration | `rvvgen/beam_cli` (`merlin-rvv-beam`) | aet-instruments the beam (parent run + one child `summary_metrics.json` per fork) and serializes the board. |
| board lock | `rvvgen/k1.board_lock` | a host-wide file `flock` — serializes physical-board access across ALL processes/sessions (concurrent beams, other users). |
| publishing | `targetgen/publish` (`merlin-target-publish`) | branch-per-version: the frozen baseline → `baseline` branch, each certified champion → `stable/<pkg>`. Verified against a local `file://` remote; no GitHub push. |

## Reproduce a run

Every step below is fail-closed; a missing board / toolchain records `not_run`, never a false pass.

### 0. Prerequisites

- The frozen baseline exists at `out/artifacts/targets/rvv/hand_v0/`.
- The K1 board + SpacemiT toolchain are reachable — set `MERLIN_K1_HOST`, `MERLIN_K1_SSH_KEY`,
  `MERLIN_K1_TOOLCHAIN` in `.env` (see `rvvgen/k1.py`). Check with
  `.venv/bin/python -c "from merlin.rvvgen import k1; print(k1.available())"`.

### 1. Freeze + publish the baseline branch (the control)

```bash
# resolves to the `baseline` branch by policy; verify against a local bare remote (no GitHub push)
git init --bare /tmp/rvv-mlir.git
.venv/bin/python -m merlin.targetgen.publish publish --target rvv --champion hand_v0 \
    --remote file:///tmp/rvv-mlir.git --execute --no-gate
git -C /tmp/rvv-mlir.git branch --list        # -> baseline
```

### 2. Build the workload + lift the expert CCA

The workload is any single-op bundle (`rvvgen.workloads.gen_matmul_f32` etc.). The expert CCA is
lifted from a decoded objdump fixture under `merlin/tests/data/cca_asm/` — e.g.
`xnnpack_f32_gemm_rvv.objdump`. A tractable square regime (128³) keeps every fork's host→rv64
compile bounded; `MERLIN_COMPILE_TIMEOUT_S` fails-closed any fork whose schedule makes clang spin.

### 3. Run the instrumented beam

```bash
MERLIN_COMPILE_TIMEOUT_S=90 .venv/bin/merlin-rvv-beam \
    --model-dir <workload-bundle> \
    --expert-objdump merlin/tests/data/cca_asm/xnnpack_f32_gemm_rvv.objdump \
    --op matmul --targets k1 --width 3 --depth 2
```

A `k1` target defaults to `max_workers=1` (serial) — the single board cannot run concurrent forks;
the `board_lock` flock additionally serializes across other processes. For a Ray fan-out with a
single-slot board gate, run the chia driver under the chia venv:

```bash
build/chia-venv/bin/python build_tools/scripts/rvv_chia_beam.py \
    --model-dir <workload-bundle> --expert-objdump <objdump> --k1-slots 1
```

### 4. Read the results

```bash
aet runs --suite rvv/beam/matmul          # the parent + one child run per fork
aet compare --suite rvv/beam/matmul       # rank forks by their summary_metrics.json
cat out/runs/rvv/beam/<id>/beam_tree.yaml # the full LLM-digestible per-step record
```

Each fork's `metrics/summary_metrics.json` carries `speedup` (real K1 wall vs the frozen seed),
`gate_ok` (cos/rel), `lever`, `depth`, `cca_divergence_closed`, and any `escalated` axes. The best
fork is the correctness-gated one with the highest real K1 speedup over `hand_v0`.

### 5. Promote + publish a champion

```bash
.venv/bin/python -m merlin.targetgen.publish promote --target rvv --champion <best-pkg>
.venv/bin/python -m merlin.targetgen.publish publish --target rvv --champion <best-pkg> \
    --remote file:///tmp/rvv-mlir.git --execute      # -> stable/<best-pkg> branch
```

## A verified run

A CCA beam over a 128³ fp32 matmul, expert CCA lifted from `xnnpack_f32_gemm_rvv.objdump`, forking
the frozen `hand_v0`, produced (measured on the live K1, medians of 3 repeats each, all `cos=1.0`):

| package | median K1 wall | speedup vs frozen baseline |
|---|---|---|
| `hand_v0` (frozen seed) | 104,123,459 ns | 1.000× |
| champion (CCA `vector.lmul` knob) | 98,341,588 ns | **1.059× (5.88% faster)** |

The seed and champion wall distributions do not overlap (seed min 104.0 ms > champion max 98.6 ms),
so the speedup is real, not measurement noise. The winning fork closed the `vector.lmul` CCA
divergence (widen the N tile to raise LMUL); the `compute.contraction_form` outer-product feature
forks fail-closed on the compile timeout (an honest finding, not a silent skip), and the frozen
baseline was asserted byte-unchanged pre/post (`beam_tree.yaml: baseline_frozen.verified_unchanged`).

## Honest boundaries

- **Substrate asymmetry.** Whole-model / section comparisons are K1 **wall** both sides;
  kernel-to-kernel attainment is spike **cycles** both sides. Never collapse the two into one number.
- **A knob can leave a residual.** If a fork's emitted asm did not achieve the promised facet, the
  beam escalates to the next-stronger class rather than crediting the miss. The
  `accumulator_resident → CODEGEN` rung reaches the register-resident microkernel emitter.
- **Real vs fake speedup.** A fork that breaks numerics sorts last regardless of speed. Speedup is
  credited only from measured K1 silicon (the INLINED-VS-ROUTED discipline).
- **No external push.** Publishing is verified against a local `file://` remote only; a real GitHub
  push needs an explicit human go-ahead.
