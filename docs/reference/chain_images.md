# Retained-object chain images

`python -m merlin.model_check.chain_image --selection selection.json --out FRESH_DIRECTORY`
packages already completed program builds into one boot image. It never invokes the candidate
compiler or derives new references. Every program in the selected models file must have its
original completed `baremetal_model.json` and compilation recipe. A missing recurrent program
is an error; an earlier compiler's stage cannot substitute for it.

The versioned `merlin.chain_image.selection.v1` JSON selects:

| Field | Meaning |
|---|---|
| `phase`, `model` | Selected phase and models-file model identifier. |
| `compiler` | `{path, sha256}` frozen compiler tree. |
| `models`, `inputs`, `workload_contract` | Exact `{path, sha256}` file pins. |
| `programs` | Every declared program name mapped to its completed build receipt pin. |
| `transport` | `full_arrays_v1` or `all_element_comparison_v1`. |
| `reference`, `reference_identity`, `bundle` | Compact mode's admitted all-output supplement pin, its exact selected identity, and original sealed bundle root. |
| `layout` | Explicit `weights` base per program, `inputs` and `references` bases, and live `arena: {begin, bytes}`. |
| `memory_map`, `execution_identity` | Existing decoded-map pin and exact provider identity. |
| `counter` | `{program, symbol, contract}` selecting the retained counter and its pinned scope/clock contract. |
| `boot` | `{entry_symbol, mailbox_symbols}` selecting original entry and host mailboxes. |
| `command_timeout_s` | Positive finite tool ceiling, at most 600 seconds. |

All paths are absolute ordinary files; generated output must be fresh and outside the frozen
compiler. The caller owns a bounded process lifetime and CPU allocation. The original split-weights
linker format is required; unsupported layouts refuse. This path currently supports the existing
little-endian ELF64/LP64 memref runtime and static ranks up to its existing maximum.

The packager namespaces original graph/call/device/weight objects with selected binutils and
independently checks that every allocated instruction/data section is unchanged. It exposes the
original compiled argument and input/output pointer tables. A tiny designated-field ABI probe
must reproduce every byte of each actual argument table under the selected header; pointer-table
extents must also agree. The common descriptor runtime is rebuilt against that same header, so an
unrecorded old header is never silently assumed. Candidate model/device code is not recompiled.

Input bindings come from the models file and captured typed ABI. At every step, the driver copies
the declared input or previous **actual** output, resets only the temporary arena, and invokes the
original trampoline. Teacher-forced token indices and recurrent carry indices remain declarative.
Each `CHAIN_STEP v1` cycle window brackets the actual trampoline invocation, excluding descriptor
preparation, input transfer, comparison and UART. Raw cycles have no clock-to-time conversion.

`full_arrays_v1` preserves complete `OUT_BIN v1` values under unique program/step/output names.
The existing numerical checker remains required. `all_element_comparison_v1` instead embeds an
explicitly admitted quantized all-output supplement, independently reopens the original bundle
seal, and requires bit equality between its judged arrays and the sealed INT8 reference. It compares
**every element of every output at every step**, including carry state. Floating values must be
finite and satisfy the selected `atol + rtol * abs(reference)` rule; integers and canonical booleans
must match exactly. The comparator is compiled with fast/finite-only math disabled.

Compact output includes element/mismatch counts, finiteness, maximum/RMS error, observational
FNV digests and at most four mismatch samples per array. Digests do not replace comparisons;
floating values within tolerance, including opposite signed zeros, may have different digests.
All steps still execute on a mismatch, and any mismatch or runtime rank error fails the guest.
Expected tensors embedded in these images remain private artifacts.

`merlin_experiments.post_freeze.firesim` explicitly admits this compact transport only through the
pinned chain manifest. Its collector joins the original uploaded/consumed ELF and complete workload
roster, then decodes the actual UART. It keeps model validation incomplete pending independent
source placement, accelerator-kernel, instruction and hardware gates. A linked image or transport
comparison does not promote old failed native records, certify a bitstream, establish a target clock,
or prove full-system/DRAM timing equivalence.
