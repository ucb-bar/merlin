# The Experiment ABI — a fair, repo-independent benchmark for target-backend generation

## Why this exists

The thesis claim is that Merlin can *generate + certify accelerator targets from RTL facts + an
oracle*. To measure whether Merlin tooling actually helps an agent do that, we will compare two
Claude Code agents on the same task: a **raw baseline** (docs + contract only) versus a
**Merlin-assisted** agent (docs + contract + Merlin's targetgen tooling). That comparison is only
valid if both solve the *same* problem against the *same* grading harness — otherwise the
baseline mostly fights the framework's integration boundary while the assisted agent benefits
from implicit internal knowledge.

So, before either agent runs, we built the **experiment ABI**: a versioned contract bundle, a
generic out-of-tree (OOT) package runner, two hand-authored reference packages (one Python, one
real C++ MLIR), negative fixtures, and AET recording. The fairness boundary is a **subprocess +
file contract + integrity scan**: a package is invoked only through CLI entrypoints and may not
import the harness internals. This is language-agnostic — proven by a Python and a C++ package
passing the identical runner.

## The contract (`merlin/contract/`)

> The contract bundle lives at `merlin/contract/` (core infra). External package bundles and the
> `--contract merlin/contract` examples below reference it by that repo-root-relative path.


A package consumes an `*.interface.mlir` in the frozen `merlin_iface` grammar
(`interface_grammar.md`) and exposes four CLI entrypoints:

| entrypoint | in → out |
|---|---|
| `parse` | interface.mlir → exit 0 / diagnostics |
| `lower_interface_to_target` | interface.mlir → target MLIR (stdout) |
| `emit_command_buffer` | interface.mlir → `command_buffer.json` |
| `lower_target_to_llvm` | interface.mlir → LLVM/RoCC MLIR (stdout) |

The lowered LLVM must define `llvm.func @gemmini_kernel(weight*, lhs*…, out*…)` (the kernel ABI);
the **runner** owns the harness (embeds deterministic tensors by name, prints `OUT/METRIC/DONE`),
the link, and the oracle invocation. Schemas (`schemas/*.schema.json`) validate the manifest, the
command buffer, the telemetry, and the results — fail-closed.

## The runner (`merlin/targetgen/oot_runner.py`)

`certify(package, interface.mlir, …)` hooks any package in via subprocess + files and records an
AET run. It never imports the package.

## The K-ladder (`scoring.yaml`)

K0 schema validates · K1 builds · K2 entrypoints callable · K3 interface→target MLIR · K4 valid
command buffer · **K5 `reference_outputs(cb) == simulate(cb)`** (always; pure Python) · K6
→LLVM/RoCC object · **K7 spike bootstrap** (`derived_from_rtl: false`) · **K8 verilator
certification** (`derived_from_rtl: true`, `cycle_accurate`) · K9 telemetry complete · K10 broken
packages fail closed. K0–K6, K9, K10 are mandatory; K7/K8 are gated on simulator availability.
Integer workloads ⇒ exact `==` gates (no tolerance); the cert is three-way:
`oracle == reference == simulate`.

## Two package classes

- **`artifacts/targets/gemmini/merlin_native_v0/`** (Python) — the **reference backend**. It
  wraps Merlin's certified MLIR-faithful path; it is the *one* `integrity_exempt` package (it
  legitimately imports Merlin internals) and is the vehicle that migrates the existing battery
  through the contract. It is **not** a competitor entry.
- **`artifacts/targets/gemmini/hand_smoke_oot/`** (C++ MLIR) — a genuine out-of-tree `gemmini-opt`
  built against the pinned LLVM/MLIR 23 install (`third_party/llvm-install`, commit
  `a47bddccec30`) using the standalone OOT template. It registers a real `merlin_iface` ODS
  dialect (parses the grammar natively), reconstructs the command buffer by walking the IR, and
  builds the `.insn` RoCC kernel via the MLIR C++ builder. Covers g0/g1. It is **not**
  integrity-exempt — it proves the contract is satisfiable from *outside* Merlin.

## What's proven (a concrete g0 trail)

Running

```
python -m merlin.targetgen.oot_runner --contract merlin/contract \
  --package artifacts/targets/gemmini/hand_smoke_oot \
  --input merlin/contract/examples/g0_matmul.interface.mlir \
  --run-id contract_smoke_g0 --simulator verilator
```

produces a recorded run with the full artifact trail and `status: pass`,
`oracle: rtl_verilator, derived_from_rtl: true, cycle_accurate: true, cycles: 308`:

```
runs/<…>/contract_smoke_g0/
  run_manifest.yaml           # status, oracle, toolchain SHAs, cycle_accurate
  artifact_manifest.json      # origin-tagged: interface_mlir(GENERATED),
                              #   target/cb/llvm/object(COMPILER_GENERATED), console(ORACLE_OUTPUT)
  generated/input.interface.mlir → lowered.target.mlir → command_buffer.json
            → lowered.llvm.mlir → kernel.o → package_kernel.elf
  artifacts/console.log
  results.yaml                # K-ladder summary (schema-valid)
```

Both packages certify g0/g1/g2 (g2 = float `acc_scale` requant → i8) on spike **and** verilator,
matching the known anchors (C0=308, C1=308, Q0=250 cycles).

## What is and isn't measured

**Measured:** whether a package, built and invoked through the contract, produces a command buffer
that is internally consistent (L0) and a lowered kernel that is bit-exact on the RTL oracle (L2),
plus complete, attributable telemetry.

**Not (yet):** the agent comparison itself; cost-model fidelity (cycles are anchors, not a
calibrated model); FireSim (L3); CIRCT-derived facts; bias/accumulator-preload; tiled shapes for
the C++ package (it covers single-tile g0/g1; the full battery rides the native reference).

## Fairness & integrity (`integrity_policy.md`)

A non-exempt package that imports `merlin.runtime.reference`/`simulator`, reads the reference
outputs, emits only a C compute kernel, or omits the command buffer **fails closed** with a
plane-routed `FailureCategory`. The seven `tests/fixtures/broken_packages/` fixtures verify this:

| fixture | plane → category |
|---|---|
| missing_manifest / missing_entrypoint | contract → structural_invariant_violation |
| bad_command_buffer_schema / bad_output_format / c_compute_kernel_only | abi_schema → protocol_violation |
| cheating_import_reference | integrity → forbidden_pattern |
| target_mlir_but_no_command_buffer | artifact_class → structural_invariant_violation |

## Tests

- `merlin/tests/gemmini/test_bench_contract.py` — schemas, grammar round-trip, golden examples (K0).
- `merlin/python/tests/test_oot_runner_smoke.py` — native g0/g1/g2 + C++ g0/g1 through the runner
  (K1–K9); oracle gates skip-if-unavailable.
- `merlin/python/tests/test_oot_runner_negative.py` — every broken fixture fails closed (K10).
