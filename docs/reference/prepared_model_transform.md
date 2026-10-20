---
title: Prepared model transformation
kind: reference
status: current
owner: compiler
last_verified: 2026-10-07
related: [reference/architecture.md]
code_refs:
  - src/merlin/llvmlower/prepared_model_transform.py
  - src/merlin/runtime/backends/spike_model.py
  - src/merlin/runtime/backends/zephyr_model.py
  - merlin/tests/runtime/test_prepared_model_transform.py
---

# Prepared model transformation

`prepared_model_transform` is an optional callback on the ordinary whole-model
build and shared preparation APIs. It runs after quantization, device source
binding, destination reuse, scheduling tags and provenance removal, before
profiling and upstream lowering. The callback is local to one build invocation.
It requires no replacement of imported compiler functions.

The callback takes `(input_snapshot: Path, output_directory: Path)` and returns
the path of selected MLIR inside that private directory. Both the original
prepared file and its byte-identical snapshot must remain unchanged. Selected
MLIR must be a regular file and parse and verify with the shared typed context.
Every original public function definition must retain its function type, and no
new public definition may appear. Private helpers are permitted. Symlinks,
outside output paths, malformed modules and changed public types refuse before
publication of a success receipt.

The receipt binds original and selected file identities and lists public
definitions. The bare-metal compilation recipe includes that receipt before
upstream lowering. These checks establish source identity and public function
types; they do not establish equivalent arithmetic, floating effects, aliasing,
private helper ABI, provider behavior or performance. The caller must supply and
qualify those proofs independently, including the original model output gate.

Empty selection returns the original path without reading it, parsing it or
creating callback output. Explicit transforms must derive decisions from the
current IR semantics, shapes, layouts and contracts. This interface installs no
default selection or workload-specific route.
