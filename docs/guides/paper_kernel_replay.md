---
title: Replay the paper's preserved kernel images
kind: guide
status: current
owner: experiments
last_verified: 2026-11-05
related: [artifact_evaluation, ../reference/repo_structure]
code_refs:
  - build_tools/scripts/replay_paper_kernels.py
  - examples/gemmini/phase2/artifact-replay.json
  - merlin/tests/infra/test_paper_kernel_replay.py
---

# Replay the paper's preserved kernel images

The replay configuration selects 29 preserved images by exact SHA-256. It includes
the Exo-assisted compiler evaluation, construction and CCA comparisons, and the
later 89-case comparison against the updated automatic Gemmini library. The image
names are the original measurement IDs, so an image can be joined directly to its
saved results and experiment records. This workflow runs fixed executables. It
does not launch a coding agent or change a compiler.

There are two separate tasks. **Image replay** uses the preserved executable and
its original inputs. **Source rebuilding** additionally requires the exact Python
runtime, compiler libraries, cross toolchains and their dependencies. The commands
below verify the preserved replay inputs. They do not establish that the source
rebuild dependency closure is complete.

## Restore the inputs

Use the release's verified archive restoration workflow first. Pass its `data/`
directory as `--root`. The layout retains the original product name:

```
data/
  perf-studies_gemmini_v4_20261028T232951Z_744bcbb/
    cases.json
    study.json
    sources/
    compiler-libraries/
    fixtures/
    kernels/
    measurements/
    scripts/
  additional-compiled-kernels/<system>/<case>/kernel.o
  fsm-rerun/
```

The additional kernel directory supplies objects from later continuations that
were not copied into the initial product's `kernels/` directory. Keep both object
collections. The validator rejects conflicting objects under the same system and
case rather than selecting whichever directory it finds first.

## Inspect and verify a selection

From the source checkout, select the restored directory explicitly:

```bash
python3 build_tools/scripts/replay_paper_kernels.py \
  --config examples/gemmini/phase2/artifact-replay.json \
  --root /absolute/path/to/data \
  plan --image cca-exo-r3-full-v1

python3 build_tools/scripts/replay_paper_kernels.py \
  --config examples/gemmini/phase2/artifact-replay.json \
  --root /absolute/path/to/data \
  validate --image cca-exo-r3-full-v1
```

`plan` prints the selection without qualifying its inputs. `validate` checks the
ELF and image-manifest hashes, every consumed kernel-object hash, each retained
compiler's native source digest, the selected interfaces and tensor fixtures, and
the runtime, harness and linker bytes. It also compares the archived hardware
summary with its separately pinned hardware record. Missing or changed inputs
return a nonzero exit status.

For the later automatic-library control, use `--image fsm-current-library-89`.
Its build manifest intentionally retains `built_not_executed`, the status when
that document was written. A separate execution summary identifies the subsequent
hardware run. Validation checks both documents without rewriting either. It
checks all 178 linked object hashes, the updated library headers and the original
Exo-assisted image to which this control refers. Its two opposite-order passes
share one boot. They are not independent boots.

A matching compiler-source digest does not prove that every framework or toolchain
dependency needed to rebuild it has been restored. Likewise, these byte checks do
not independently repeat the original numerical experiment.

## Run one preserved image

Select the reviewed `fpga-run-elf` wrapper by absolute path and SHA-256. Supply a
JSON platform selection containing `agfi` and `runtime_hw_config`, using the values
printed under `required_hardware` by validation. This records the intended
platform. Actual execution must still be checked against the new job's hardware
summary and invocation.

```bash
python3 build_tools/scripts/replay_paper_kernels.py \
  --config examples/gemmini/phase2/artifact-replay.json \
  --root /absolute/path/to/data \
  replay --image cca-exo-r3-full-v1 \
  --wrapper /absolute/path/to/fpga-run-elf \
  --wrapper-sha256 SHA256_OF_THE_REVIEWED_WRAPPER \
  --platform /absolute/path/to/platform.json \
  --output /absolute/path/to/out/artifacts/paper-replay/exo
```

This command launches hardware. The wrapper owns serialization, instance launch
and teardown. The adapter never starts or stops the manager directly. It copies
the verified ELF into a new output directory and records the exact wrapper
invocation, output and return code. An existing output directory is refused.

`wrapper_completed` means that the wrapper returned zero. It is not a correctness
or performance verdict. Collect the new job's complete UART, uploaded-ELF digest,
SSM invocation, `HW_CFG_SUMMARY` and `sim-run.sh`. Check exact ELF and hardware
identity, full-output comparisons, guards, input digests and completion before
using any new timing result. Keep new results separate from the preserved paper
measurements. Do not attach an old hardware receipt to a new UART log.

## Inspect a source rebuild

If dependency archives were restored separately, add `--dependency-root /absolute/path/to/runtime` and `--dependency-metadata-root /absolute/path/to/dependency-metadata` before the subcommand. The first directory contains `dependencies/<alias>/`; the second contains the five `<alias>.inventory.json` files. Each inventory retains its configured SHA-256. This selection never moves or changes archived files.

```bash
python3 build_tools/scripts/replay_paper_kernels.py \
  --config examples/gemmini/phase2/artifact-replay.json \
  --root /absolute/path/to/data \
  rebuild-plan --system F10-FULL-NOREF-NOCCA --case G01 \
  --workspace /absolute/path/to/out/build/paper-kernel-rebuild
```

This checks the exact archived `compile_one.py`, `build_batch_v2.py`, selected
compiler and interface, then prints their namespace and dependency requirements.
It does not compile or execute anything. Missing dependency trees and complete
verified inventories are listed, and the command returns nonzero when any are
missing. Once restored, it checks exact membership, file modes and bytes,
symlink destinations across the selected mounts, historical tool executable hashes,
and archived runtime headers. A passing plan qualifies these input bindings,
not the generated code or the original Python environment's complete history.

The scripts contain their original absolute paths. The printed plan mounts
restored data and toolchains at those paths, read-only, with a fresh build
directory as the writable destination. It never edits the preserved scripts.
The original `scripts/study.json` lists seven compiler snapshots, while the final
product's `study.json` contains 22. The plan explicitly reports and records the
selected study-file overlay. It does not silently replace the historical file.

To complete source rebuilding, supply the compiler Python environment and its
interpreter, LLVM installation, RISC-V toolchain and libraries, upstream runtime
headers, selected generic compiler libraries, and complete pinned dependency
inventories. The preserved `toolchains.json` records several exact executable
hashes. Those hashes alone do not inventory every loaded library or Python
package. After restoring that closure, independently rebuild the selected
objects and image, compare their bindings, and perform a new numerical execution.

## Probe the restored interpreter and rebuild one case

The retained Python environment contains editable path entries and startup hooks.
The original compiler namespace did not expose the framework source directories
named by those editable entries. Keep those paths inaccessible. Do not mount a
new framework checkout to make them resolve. The adapter records these path
entries and verifies hook bytes through the runtime inventory.

After restoring dependencies, use `probe` with the same `--system`, `--case`, and
`--workspace` options as `rebuild-plan`, plus `--bwrap` and `--bwrap-sha256`.
The probe starts only the interpreter and imports xDSL. It records loaded module
origins and refuses origins outside selected inputs and ordinary OS roots. It
does not import the authored compiler, compile a workload, or launch a simulator.
Its result qualifies startup and those imports only.

To rebuild a compiler/case, select the original comparison image as well:

```bash
python3 build_tools/scripts/replay_paper_kernels.py \
  --config examples/gemmini/phase2/artifact-replay.json \
  --root /absolute/path/to/data \
  rebuild --image cca-exo-r3-full-v1 --system CCA-EXO-R3 --case G01 \
  --workspace /absolute/path/to/out/build/exo-g01-rebuild \
  --bwrap /absolute/path/to/data/dependencies/tools/bwrap \
  --bwrap-sha256 SHA256_OF_THE_PRESERVED_BWRAP
```

This executes the original pinned `compile_one.py` using the explicitly selected
22-source study table. It therefore supports the later Exo-assisted compiler
without changing either archived script. The exact bwrap binary also serves the
original script's nested sandbox. No compiler, library, fixture or dependency
source tree is writable. Only the fresh build workspace is writable.

The receipt records the new object digest and whether it matches the object
consumed by the selected preserved image. A zero script exit alone is insufficient.
The adapter requires a compiled result, unchanged source, the selected interface,
and a matching emitted-object digest. It then reopens the source and dependency
checks. A different but successfully compiled object is reported as a new build,
not substituted into the paper's measurements. Numerical checks and any new
linked-image hardware evaluation remain separate.

A preserved optional runtime symlink may name a missing target within its selected tree. The planner records these under `preserved_dangling_dependency_links` and sets `full_installed_tree_link_closure` to false. It does not repair them or claim every installed entry point is usable. Selected tool executables must resolve completely, and links escaping selected mounts or forming cycles are always refused. The separate startup probe checks the imports actually loaded.
