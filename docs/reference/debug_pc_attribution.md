---
title: Debug companion PC attribution
kind: reference
status: current
owner: core
last_verified: 2026-10-06
code_refs: [src/merlin/perf/debug_companion.py, src/merlin/perf/whole_model_group_timing.py,
            packages/merlin-experiments/src/merlin_experiments/group_inspect.py]
---

# Debug companion PC attribution

`verify_debug_companion` admits a debug-only companion for an existing binary
without executing a new measurement. It checks ordinary little-endian ELF64
identity, every allocated section's bytes, type, flags, alignment, size and
address, and normalized relocations targeting allocated sections. Relocation
symbol indices may differ when debug symbols are added; symbol identities and
addends must match. Unsupported formats and malformed metadata refuse.

Check both the original object versus its debug object (`relocatable=True`) and
the final original image versus its debug relink. A changed build-ID note is an
allocated-byte difference and refuses; do not silently discard it. Reproduction
must preserve the actual compiler/link recipe. Numerical source qualification,
ISA audit, tool provenance and execution-counter provenance remain caller duties.

The caller selects the target tool and passes its LLVM symbolizer JSON records
to `attribute_symbolized_pcs`. Each census address must appear exactly once.
Missing source lines remain visible, inline stacks are retained, and innermost
line/function totals conserve instruction counts. Inline source attribution is
not an isolated call timer. Whole-program counts may include initialization,
checks and consumers outside a measured region; do not label them ROI counts or
hardware cycles without independent counter evidence.

The utility does not compile, run simulators, parse target opcodes, select target
tools, alter optimization policy or infer a performance improvement.

## Where it is called

`merlin experiment inspect <candidate> --group gN --trace` is the production caller. The group
program build (`whole_model_group_timing.build_group_programs(..., debug_companion=True)`) links the
companion from the program's own model, kernels and recipe with the debug option appended to the
recorded flags; inspect checks both the program object and the final image against it, runs the
candidate's functional model with its PC histogram on, symbolizes every PC with the `llvm-symbolizer`
beside the compiler the target's build recipe names, and attributes the histogram. A missing machine,
companion or symbolizer, or a refused companion, leaves the attribution `UNKNOWN` with the reason. See
[compile_debugging](../guides/compile_debugging.md).
