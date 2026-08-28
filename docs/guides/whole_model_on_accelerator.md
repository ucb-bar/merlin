---
title: Compiling a whole model onto an accelerator
kind: guide
status: current
owner: compiler
last_verified: 2026-08-27
related: [compilation_strategies, targetgen, adding_a_target, gemmini_experiment]
code_refs: [merlin/python/merlin/compile_cli.py, merlin/python/merlin/runtime/dispatch_runtime.py, merlin/python/merlin/targetgen/routing.py]
---

# Compiling a whole model onto an accelerator

Three different things get called "compiling a model with Merlin", and they carry very different
evidence. Keep them apart; conflating them is the easiest way to overstate what the compiler does.

| Path | Unit | Runs on | Entrypoint |
|---|---|---|---|
| whole-model, CPU/vector | a captured model | RISC-V CPU + RVV | `merlin-compile --target rvv` |
| accelerator, per capsule | a capsule | the accelerator, graded up the oracle ladder | `merlin-compile --target <t>` |
| **whole model ON an accelerator** | a captured model | accelerator mesh + host scalar/RVV lane | `compile_model(...)`, Python only |

This guide covers the third. For the first two see `rvv_e2e.md` and `gemmini_experiment.md`.

## There is no CLI for this path

`merlin-compile` dispatches to exactly two places — `compile_rvv` for `--target rvv`, and
`compile_oot` for a per-capsule accelerator run. `--run` offers `{none,host,k1,spike,zephyr,verilator}`
and has **no `mesh` choice**, and there is no flag for a package override on that path. The whole-model
mesh entrypoint is reachable only from Python. Everything below was run that way.

## Compiling a model onto the mesh

```python
from merlin.compile_cli import compile_model

out = compile_model("small_llama", "int8", target="gemmini", run="mesh",
                    verify=True, package=None, auto_capture=True, timeout=1800,
                    routing_dtype="int8")

print(out["mesh_execution"])   # per-layer: on-mesh vs host fallback vs oracle-unavailable
print(out["verify"])           # the numeric gate
```

To run against a specific backend package (a graded submission, say) pass `mesh_package=<path>`.

Environment:

- `MERLIN_MESH_SIM=spike` selects the mesh oracle;
- `MERLIN_MESH_VERIFY=1` additionally certifies a synthesized `DxD` tile per mesh-routed matmul,
  gated bit-exact against the declared accumulator.

`run="mesh"` routes each op across the target's compute units (matmul/systolic tiles to the mesh;
norms, activations and elementwise to the vector/scalar lane), then executes each mesh matmul layer on
the target's own oracle with the real operands injected, handing every layer's on-device output to the
op that consumes it. An op no unit supports is an honest scalar/RVV fallback, never a silent drop.

## Reading the result

Two questions decide whether a whole-model result means anything, and they are separate.

**Did the work reach the accelerator?** `mesh_execution` counts
`matmul_layers_on_mesh` / `matmul_layers_host_fallback` / `matmul_layers_oracle_unavailable`. A model
that produced the right answer entirely on the CPU is not an accelerator result, which is what the
`must_accelerate` gate exists to catch. Note that *unavailable* is distinct from *wrong*: a layer whose
oracle could not run was never measured, and `NOT_RUN_IS_NOT_PASS` means it cannot count as a pass.

**Was the arithmetic checked, and how strongly?** In the gate block:

| field | meaning |
|---|---|
| `per_element_guarded` | whether any per-element bound actually applied; `false` means the verdict rests on aggregate cosine alone |
| `per_element_basis` | which bound vetted it: `relative` (strict tiers) or `quantization_excess` (derived) |
| `quant_excess` | the run's worst deviation over the deviation a correct host int8 reference already costs on that same output; ~1 is indistinguishable from correct quantized arithmetic, bound 4 |

The per-element bound is derived rather than fixed for a measured reason: across the tracked
recaptures the deviation of a **correct** host int8 reference from the fp32 golden spans 0.027 to 1.88
of the output RMS and 1.3 to 99.0 in per-element relative terms. No constant satisfies all of them, so
a fixed threshold is necessarily fitted to whichever model it was chosen against. Comparing a run to
its own model's quantization floor (`golden_w8a8` vs `golden`) needs no such choice, and where that
floor is unmeasurable the veto stays off and the verdict reports itself unguarded rather than
manufacturing a bound. See `merlin/python/merlin/runtime/backends/zephyr_model.py::_gate`.

## Residency and composition are different claims

**Residency** is "every routed matmul layer ran on the mesh". **Composition** is "the compiler split
the network across the mesh and the host lane and still got the right answer".

A real network on a matmul-only mesh *cannot* run entirely on the accelerator — norms and activations
have nowhere else to go. That is why interop capsules withhold `must_accelerate` and instead declare
`lanes.require`, so each named lane having carried work is the behaviour under test rather than a
violation. Quoting one claim as the other overstates both.

## Grading a package

Pass the target's **own** capsule roots, never their common parent:

```python
from merlin.targetgen.corpora import graded_capsule_roots
from merlin.targetgen.capsule_grade import grade

r = grade("<package>", capsules_root=[str(x) for x in graded_capsule_roots("gemmini")],
          runs_root="<runs>", target="gemmini", labels={"public"})
print(r["headline"])
```

A target's capsules are split across sibling category directories (`isa/`, `layers/`, `model/`,
`model_slices/`), and different targets keep them in different places. Passing the parent pulls in
every target's corpus at once: grading a gemmini package against `merlin/contract/capsules` reports
`22/84` with 89 capsules marked "outside this target's declared capability" — a number that reads like
a catastrophic regression and means nothing.

## Capacity obligations

Two on-chip stores bound a contraction and they bind different dimensions: the operand store holds the
weight tile plus the activation tile (growing as `K*(M+N)`), the accumulator holds the output tile
(growing as `M*N`). `capacity_fit` evaluates both and reports `operands_hold` / `output_holds`
separately, because they fail on different layers — a wide-output layer sits well inside a 256 KiB
scratchpad while overrunning a 64 KiB accumulator. Both bounds are derived from the target's own
RTL-discovered capacities; a store that cannot be classified leaves the obligation undecidable rather
than assumed to hold.
