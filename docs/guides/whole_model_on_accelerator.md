---
title: Compiling a whole model onto an accelerator
kind: guide
status: current
owner: compiler
last_verified: 2026-08-28
related: [compilation_strategies, targetgen, adding_a_target, gemmini_experiment]
code_refs: [merlin/python/merlin/compile_cli.py, merlin/python/merlin/runtime/dispatch_runtime.py, merlin/python/merlin/targetgen/routing.py, merlin/python/merlin/system, merlin/python/merlin/llvmlower/device_offload.py, merlin/python/merlin/llvmlower/device_build.py]
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

## Two ways to run a model on a device

They are genuinely different, and the difference is what a result means.

**Interpreted** (`compile_model(..., run="mesh")`). The host side is a Python tree-walking interpreter
over the driver function; each matmul layer becomes an interface capsule, is shipped out of process to
the target's oracle, and the result is read back as a numpy array. Nothing is emitted into a host
binary. This is the path the L3 capsule results were measured on, and it is the right one for grading:
the oracle is the same one the capsule ladder certifies against.

**Compiled** (`device=` on the whole-model build). The contractions are rewritten into calls to private
symbols, the target's own package emits a kernel per distinct extent, a generated shim adapts the MLIR
calling convention to the device's kernel ABI, and the objects are linked beside the model object. The
output is one artifact that runs on a board with no Python and no simulator in the loop.

There is no CLI for either: `merlin-compile` dispatches to exactly two places — `compile_rvv` for
`--target rvv` and `compile_oot` for a per-capsule accelerator run. `--run` offers
`{none,host,k1,spike,zephyr,verilator}` with **no `mesh` choice**. Both paths are reached from Python.

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

## The compiled path

Measured end to end: a single ELF carrying host code and four device kernels — 33 custom instructions
in `.text` — ran `small_llama` int8 to completion on spike, with 15 contractions executing on the
accelerator across 4 distinct extents.

| | cos vs fp32 | cos vs W8A8 | max\|d\|/max\|g\| | argmax |
|---|---|---|---|---|
| **compiled ELF** | 0.999929 | 0.999908 | 0.01308 / 0.01480 | match |
| interpreted path | 0.99993 | 0.99991 | 0.0131 / 0.0148 | match |

The two agree to five decimal places, which is the check that matters: the compiled artifact is not
merely *a* result, it is the same result the graded path produces.

One fix stood between those two rows and it is worth repeating, because it is the failure mode this
whole path is most exposed to. The kernel ABI requires operands zero-padded to the mesh tile edge, and
every offloaded layer here has M=8 against a 16-wide mesh. Handing over raw buffers does not fault —
the kernel strides by the padded width through unpadded data, reads a neighbouring row as its own, and
returns **cos 0.9847**: plausible, and wrong. Nothing catches that except comparing against a path
already known to be right.


```python
from merlin.llvmlower.device_build import DeviceRouting
from merlin.runtime.backends import spike_model

routing = DeviceRouting(device="<target>", package_dir="<backend package>",
                        operand_dtype="int8", accum_dtype="i32",
                        select=lambda shape: True)   # the placement decision, made elsewhere
spike_model.build(model_dir, work, device=routing, int8_compute=True)
```

What happens, and where to look when it does not:

1. `prepare_for_lowering` rewrites each selected contraction into a call to a private symbol and
   writes `device_signatures.json` beside the prepared module. One symbol per distinct `(M,N,K)` —
   MLIR function types are monomorphic, so two extents cannot share a callee.
2. The build reads that sidecar, runs the target's package once per signature to emit a device kernel,
   renames each kernel to a distinct symbol, generates the shim, and links them with the model object.
3. `select=None` (or `device=None`) makes the whole path inert. The placement decision belongs to
   `merlin.system.place`, and a build that took it for itself would disagree with the router.

**Why the rename matters.** Every emitted kernel carries the single entry name the backend contract
declares. Linking several without renaming is not an error — the linker binds every call to whichever
object it resolved first, so the model runs one layer's kernel for every layer, produces numbers, and
is wrong.

**Fail-closed points.** A device whose datapath cannot be derived offloads nothing rather than
assuming a precision. A signature whose dtype has no MLIR type, or a sub-byte format whose element
offset is not a byte count, is declined and reported rather than approximated. A sidecar carrying
signatures with no `device=` routing to build them against is refused with that stated, rather than
failing later as an unresolved symbol.

## What is RTL-backed, and what is not

A whole-model image cannot run on the Verilator harness (below), so the compiled artifact's RTL
evidence is obtained by asking a narrower question. The artifact's device side is a fixed set of
kernels at fixed extents — the offload mints one per distinct `(M,N,K)` — and those are
scratchpad-resident, with no sub-word DRAM traffic. They therefore certify on exactly the harness the
capsule ladder uses, at the extents the model actually calls.

Measured for `small_llama` int8, whose compiled ELF carries four kernels:

| kernel | M×N×K | oracle | derived_from_rtl | cycle-accurate | cycles |
|---|---|---|---|---|---|
| k0 | 8×128×128 | rtl_verilator | yes | yes | 2837 |
| k1 | 8×344×128 | rtl_verilator | yes | yes | 7363 |
| k2 | 8×128×344 | rtl_verilator | yes | yes | 7159 |
| k3 | 8×256×128 | rtl_verilator | yes | yes | 5339 |

All four pass, including `reference_outputs_vs_simulate`.

**State the claim at its real scope.** This certifies the DEVICE CODE the artifact contains, on
elaborated RTL, at the extents it runs. It is not a whole-model RTL run, and the host↔device
integration is not what it measures — that is what the spike numbers above cover. The two together
are: correct device code (RTL), assembled into a correct whole-model result (functional). A
whole-model RTL run remains unavailable on the substrates here.

## Which substrates can run a compiled whole model

| substrate | whole-model artifact | why |
|---|---|---|
| spike (`--extension=<target>`) | **yes** | measured; the numbers above |
| chipyard Verilator | **no** on either gemmini config tried | harness RAM rejects sub-word writes |
| GSIM | **no**, as configured | no ELF loader on that SoC build |

The Verilator block is worth stating precisely, because it looks like a compiler failure and is not.
The harness monitor aborts with `'A' channel carries PutPartial type which is unexpected` — a
sub-word write on the TileLink A channel. A whole-model image produces those routinely; the capsules
do not, because they are scratchpad-resident with no sub-word DRAM traffic, which is why the same sim
certifies capsules happily. **Measured control, on two configs:** the identical model built with NO device offload
aborts at the same monitor, so this is the sim's memory model and not the offload path. Running a
whole model on RTL needs a config whose memory accepts sub-word writes.

Checked on a second config rather than inferred from one. `GemminiAndOPUShuttleConfig` aborts at the
same monitor and the same simulation timestamp, so this is the harness memory these builds share, not
a property of the config first tried. (An offloaded image on that config exits quietly WITHOUT the
abort, which is not evidence the block is absent: gemmini sits on a different custom opcode slot when
the OPU is also present, so its kernels trap early and the run never reaches the writes that trip the
monitor.) FireSim is the substrate to try next -- it models DDR rather than this harness RAM, and its
gemmini bitstream exists.

The GSIM block is different in kind. That SoC build bakes its kernel into the BootROM constants
because SimTSI was pruned, so there is no path to load an ELF at all — a compiled artifact cannot be
handed to it regardless of size. DRAM preload plus a reset vector would have to be built into the
harness first.

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
