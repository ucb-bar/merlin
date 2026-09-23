# Examples

## Target experiment inputs

Start with a target's workflow map. Each map links to its single definition;
the [experiment catalog](../experiments/catalog.yaml) owns the stable command IDs.

| Workflow | Catalog ID | Declared target |
| --- | --- | --- |
| [Gemmini](gemmini/README.md) | `gemmini-functional` | `gemmini` |
| [Gemmini Universal](gemmini_universal/README.md) | `gemmini-universal-functional` | `gemmini_universal` |
| [Atlas](atlas/README.md) | `atlas-functional` | `atlas` |
| [Radiance](radiance/README.md) | `radiance-functional` | `radiance` |
| [MX Gemmini](mx_gemmini/README.md) | `mx-gemmini-functional` | `mx_gemmini` |
| [Saturn OPU](saturn_opu/README.md) | `saturn-opu-functional` | `saturn_opu_mxv256d128` |
| [Saturn OPU RVV](saturn_opu_rvv/README.md) | `saturn-opu-rvv-functional` | `saturn_opu_mxv256d128_rvv` |

Further OPU/K1 resource cleanup is deferred. None of these entries claims a qualified
compiler, simulator or hardware installation. Existing phase settings are unchanged.
The Universal entry declares Phase 1 only: it records a distinct configuration
without inventing a Phase 0 recipe or Phase 2 hardware result.

Inspect a definition without starting execution:

```sh
merlin experiment inspect gemmini-functional
```

To navigate stored work after a run, list orchestration records and inspect a
specific run's frozen handoffs:

```sh
merlin experiment runs --target gemmini
merlin experiment lineage /absolute/path/to/orchestration-run
```

`lineage` reports the definition digest, selected public input digests, the
corpus-release selection when present, and recorded phase output paths. It reads
the frozen orchestration records only: it does not recheck live inputs, enumerate
private holdouts, or certify a compiler. `status` shows process attempts; use the
native engine evidence for scientific verdicts. Runs outside the canonical
`out/runs/<target>/<experiment>/` layout are inspectable by explicit path.

Definitions with an admitted Phase 0 own their public recipe at
`phase0/recipe.yaml`; every target owns its descriptor at
`target/descriptor.yaml`. Universal intentionally has no Phase 0 recipe or route
until its different hardware capability and cohort have been reviewed.
Gemmini's authored prompts now live in [phase1/task](gemmini/phase1/task), selected
by its descriptor's `task_root`; [the Phase 1 guide](gemmini/phase1/README.md)
explains task modes and the required fresh reviewed release.
Its curated runtime harness is also example-owned under `phase1/contracts/`, selected
by `contracts_root`. Machine-specific Chipyard bring-up links have not been migrated.
These are not self-contained examples yet: descriptor `resources_root` still points
to retained task/harness/bundle resources, and generated/private sidecars remain
explicit external inputs. The destination layout has
`target/` for shared target inputs, `phase0/`, `phase1/`, `phase2/`, and a
`whole-model/` walkthrough. Gemmini now has a [workflow map](gemmini/README.md),
a [Phase 2 handoff guide](gemmini/phase2/README.md), and a
[whole-model inspection walkthrough](gemmini/whole-model/README.md); the latter
does not claim independently qualified accelerator deployment. The Atlas, Radiance
and MX Gemmini maps now include their Phase 2 handoffs and whole-model entrypoints;
they do not claim target-specific Phase 2 or whole-model qualification.
Reusable target support stays OOT; generated capsules,
weights and compiler artifacts stay under the configured output root. Do not copy
private holdouts, goldens or credentials into examples.

Public tooling-path templates are example-owned too:
[Atlas](atlas/target/README.md) and [Radiance](radiance/target/README.md).
Their guides explain the selected live `experiment.env` destination. Templates are
not automatically sourced, and existing local environment files have not moved.
MX Gemmini's [Phase 1 example](mx_gemmini/phase1/README.md) owns its public ISA and
MMIO contract inputs; the descriptor selects them explicitly for newly generated bundles.
Radiance's [Phase 1 example](radiance/phase1/README.md) similarly groups its ISA
inputs, retained worked assembly reference and pinned kernel-library selection.
Its descriptor preserves the separate no-kernel and kernel-library treatments.
Atlas's [Phase 1 example](atlas/phase1/README.md) groups its public ISA documentation,
curated RTL, worked assembly and target-owned preflight fixtures. The descriptor
selects the moved fixture sources and adapter callables together.

Legacy `target_experiment.yaml` paths are compatibility links, not duplicate source.
New catalog runs select the example-owned descriptor and freeze its current bytes.
Historical runs retain their original descriptors and resource inventories; do not
rewrite their receipts to point at these moved files. Target discovery matches the
descriptor's declared identity, not its example folder name.

Each Phase 0 definition now names its inputs separately: `recipe` is authored
source; `performance_template` is the shared experiment template; `synth_profile`
and `smt_profile` are generated inputs; `hidden_profile` is private input, never
public example content. Optional sidecars may be absent. A frozen run records both
their declared paths and presence, so adding or removing a sidecar invalidates
resume. The shared template lives in `experiments/templates/phase0/performance.yaml`;
SMT/private sidecars retain their explicit existing locations. Retained synthesis
references live in `experiments/reference-data/phase0/`, unchanged and separate from
newly generated artifacts. No generated or private files were copied into examples.

The synthesis maintenance command `build_tools/scripts/synth_capsule_corpus.py --target TARGET --write`
creates a fresh versioned `synth.yaml` under the configured `out/artifacts/verification/` root;
`--json` reports its path. It never overwrites the declared `synth_profile` reference.
Use `--check` separately to detect drift, then review the artifact before selecting it
in a new definition. Existing references can contain reviewed manual policy overrides;
regeneration is not permission to discard them.

Authored `synthesis_model_gates` in a public recipe can select a synthesized model by
name and declare `after_op_pass_fraction` (0–1) plus a required `reason`. This controls
when that model becomes eligible to run, not its numerical acceptance. Synthesis
rejects stale/ambiguous names, non-model entries and unsupported fields, and records
the exact recipe hash and gate change in provenance. Atlas's previously manual
composition-model gate is now declared this way; its retained reference is unchanged.

Compare the public recipes selected by two or more definitions through the same
experiment entrypoint:

```sh
merlin experiment corpus compare gemmini-functional atlas-functional
```

This prints a JSON comparison manifest without generating capsules or opening
private holdouts. It describes the current declared public recipes, not the frozen
corpus of an earlier run. Definitions may also be supplied as file paths.

For a compiler-level example rather than phase orchestration, see
[`triton/`](triton/). Follow its prerequisites; compiling an example is separate
from certifying it against RTL.

| example | what it is |
| --- | --- |
| [`triton/`](triton/) | One stock `@triton.jit` kernel → an accelerator's command buffer → that accelerator's own RTL. The compiler-facing example: how a target is plugged in, how the route is chosen, and how the same kernel compiles for two different accelerators. **Needs only the repo and its venv** (RTL certification is optional). |

The other two are the delivery story: compiling a PyTorch model to int8 multicore-RVV binaries for a
**RISC-V tapeout you cannot attach to**, proving as much as possible in simulation, and packaging it so
the one person with the board can run it and mail back a log that explains itself.

| example | chip | what it exercises that the other does not |
|---|---|---|
| [`kodiak/`](kodiak/) | ucb-bar Kodiak tapeout, 3 harts (2 with a vector unit), 512 MB | A **host-assisted (HTIF)** console, heterogeneous harts, and scalar images for the core with no vector unit |
| [`gemmelos/`](gemmelos/) | Bearly ML 25, 2 harts | A chip with **no Zephyr port and no debug host** — the UART address, its baud divisor and the PLL sequence are all *derived from that chip's own SDK headers*, and there is a variant that raises the clock 10× |

They are deliberately the same five stages, so the difference between "a board Zephyr knows about" and
"a board nothing knows about" is visible as a diff rather than described.

## Run one

```bash
cd examples/kodiak
./run.sh preflight        # says which stages this machine can do, and what would fix the rest
./run.sh probe            # seconds: build the vector probe, self-check it on spike
./run.sh build            # minutes: build + gate one small model at 1 and 2 harts
./run.sh package          # assemble a delivery package and its zip
./run.sh grade <log>      # score a console log the board's owner sent back
```

`--dry-run` prints every command without running it, so the examples are readable as documentation on a
machine with no toolchain at all. `--full` swaps the cheap subset for the matrix that actually ships
(hours of functional simulation — the flag exists so that cost is always a choice).

## What each stage needs

Run `./run.sh preflight` for a verdict on *your* machine; it asks the library the same questions the real
build does rather than re-deriving them. In general:

| stage | needs | who can run it |
|---|---|---|
| `grade` | python + numpy | anyone |
| `probe` | spike + a riscv64 toolchain | anyone who can build the toolchain |
| `build`, `package` | + `clang-23`/`mlir-translate`, + the Zephyr tree below, + (gemmelos) that chip's SDK | anyone with those checkouts |
| running on the board | the silicon | the two people who have one |

The last row is the honest limit, and it is why the packaging half of this repo exists: everything the
board owner needs — both builds of every image, the references, a numpy-only grader, the expected console
— goes in one zip, because a round trip costs days.

## Set up a machine from scratch

`./run.sh preflight` names whichever of these is missing, with the variable that fixes it. Nothing here
is optional folklore — each one is read by the library at a specific point, and the default for each is
either a checkout path or a hard failure.

```bash
# 1. the repo and its interpreter (see the root README / docs/guides/getting_started.md)
uv sync --all-extras            # creates .venv with merlin editable; plain `python` is not on PATH
                                # without uv: pip install -e '.[dev,xdsl]'

# 2. spike + a riscv64 bare-metal toolchain. Any build works; point at each binary directly:
export MERLIN_SPIKE=/path/to/spike
export MERLIN_RISCV_GCC=/path/to/riscv64-unknown-elf-gcc
#    or, if you have a chipyard checkout whose .conda-env carries riscv-tools, just:
export MERLIN_CHIPYARD=/path/to/chipyard

# 3. LLVM/MLIR 23 -- clang-23 compiles every RISC-V object, and mlir-translate lowers the OpenMP IR
#    every MULTICORE image needs. It is NOT in a fresh clone (third_party/llvm-install is gitignored):
#    build it per docs/guides/llvm_toolchain.md, then nothing needs exporting. Already have an LLVM 23?
export MERLIN_CLANG=/path/to/bin/clang-23
export MERLIN_MLIR_TRANSLATE=/path/to/bin/mlir-translate

# 4. cmake + ninja. Taken from $MERLIN_CHIPYARD/.conda-env/bin when that exists (pinned, for
#    reproducibility), otherwise from PATH. Nothing to set if you have them the normal way.

# 5. Zephyr — a WEST WORKSPACE, not a bare clone. The bare clone has no zephyr_ws/zephyr in it:
git clone -b dev https://github.com/ucb-bar/zephyr-chipyard-sw.git
cd zephyr-chipyard-sw && west init -l . && west update       # populates zephyr_ws/
export ZEPHYR_BASE=$PWD/zephyr_ws/zephyr

# 6. the Zephyr SDK (0.17.0 is what this is exercised against)
export ZEPHYR_SDK_INSTALL_DIR=/path/to/zephyr-sdk-0.17.0

# 7. only for the gemmelos example — that chip's own SDK checkout
export GEMMELOS_SDK=/path/to/gemmelos-bringup
```

Each of `MERLIN_*`, `ZEPHYR_BASE` and `ZEPHYR_SDK_INSTALL_DIR` is also read from a repo-root `.env`, so a
configured checkout resolves without exporting anything per shell.

For the whole-repo version of this check — every experiment capability, not just these two examples —
`build_tools/scripts/check_repro_env.py` asks each capability's own availability guard and reports which
variable to set.

Two honest caveats about a from-scratch machine:

- Steps 5 and 6 are the ones people get stuck on, and neither is ours: a Zephyr *west workspace* plus a
  matching SDK. If `preflight` says the tree and SDK are fine but `build` still fails, the build's own
  error is the authority, not this list. Step 3 is the *slowest* — building LLVM 23 is tens of minutes on
  a many-core host and about 13 GB of disk — but it is unattended and only happens once.
- `_sdk_dir()`'s built-in default is a path on one developer's machine. If you have not set
  `ZEPHYR_SDK_INSTALL_DIR`, `preflight` may show a Zephyr SDK that only exists here. Set it explicitly.

## Inputs, and how to get each one

Nothing here is a binary blob you have to be given. Everything is either in this repo, clonable, or
regenerable with a documented command.

| input | where it comes from |
|---|---|
| **The tuned RVV schedule** — the compiler package these binaries are built with | Tracked in this repo at `out/artifacts/targets/rvv/impr_tuned_wholemodel_vf_int8/` (3 files). Also published standalone: `git clone -b stable/impr_tuned_wholemodel_vf_int8 https://github.com/ucb-bar/rvv-mlir.git`. The frozen before-picture is branch **`baseline-int8_w8a8`** on the same repo — diffing the two is the clearest statement of what the tuning actually changed. |
| **Zephyr** | `git clone -b dev https://github.com/ucb-bar/zephyr-chipyard-sw.git`, then point `ZEPHYR_BASE` at `zephyr_ws/zephyr`. The commit matters and is checkable from any returned log: the boot banner prints it (`*** Booting Zephyr OS build 852bb170cc56 ***`). |
| **Model capture bundles** (`out/artifacts/recaptures/<model>_int8_full/`) | Not shipped — they are large and regenerable. Generate with [model2MLIR](https://github.com/ucb-bar/model2MLIR): `python $MODEL2MLIR_DIR/workloads/capture.py <model> --formats int8`. See [`docs/guides/model2mlir.md`](../docs/guides/model2mlir.md); quantization happens in that repo, not this one. That checkout is needed for `build` even when you already have the bundles — the MLIR lowering runner executes inside its venv (`MERLIN_M2M_DIR` / `MERLIN_M2M_VENV`), which is why `preflight` lists it. |
| **spike + riscv64 toolchain** | Any build works; point `MERLIN_SPIKE` at the binary, or `MERLIN_CHIPYARD` at a chipyard checkout whose `.conda-env` provides riscv-tools. |
| **`clang-23` + `mlir-translate`** (LLVM/MLIR 23) | Built from the pinned `third_party/llvm-project` submodule into `third_party/llvm-install/`, which is **gitignored** — a fresh clone has to build it: [`docs/guides/llvm_toolchain.md`](../docs/guides/llvm_toolchain.md) has the exact cmake line, why each flag is there, and a cheap two-target build if all you want is these examples. LLVM 23 is upstream `main`, so no distro package substitutes; an existing LLVM 23 install works via `MERLIN_CLANG` / `MERLIN_MLIR_TRANSLATE`. |
| **The gemmelos SDK** | A third-party bring-up repo, not ours to redistribute. It supplies only *facts* — a UART base address, two clock rates, a PLL register map. The `facts` stage prints exactly what is read out of it, so if you cannot get the checkout you can see precisely what would have to be supplied another way. The Kodiak example needs none of this. |
| **The boards** | Two tapeouts. You will not have one, which is the premise. |

## Reading a log that came back

Both examples end at the same place, because in practice that is where the work is. `gemmelos/` ships a
**real returned failure** (`returned/whisper_h1_debug_500mhz_fault.txt`) so the diagnosis flow is
reproducible with nothing but python: four lines of console localise the bug to one wrong field in a board
descriptor. `gemmelos/README.md` walks it.

## See also

- [`docs/guides/getting_started.md`](../docs/guides/getting_started.md) — the repo itself
- [`docs/guides/llvm_toolchain.md`](../docs/guides/llvm_toolchain.md) — building the LLVM/MLIR 23 these
  examples compile through
- [`docs/guides/adding_a_target.md`](../docs/guides/adding_a_target.md) — a third board
- [`docs/guides/model2mlir.md`](../docs/guides/model2mlir.md) — where capture bundles come from
- [`docs/guides/reproducibility.md`](../docs/guides/reproducibility.md) — what each result rests on
