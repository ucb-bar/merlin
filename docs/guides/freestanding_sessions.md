---
title: Run captured sessions without an operating system
kind: guide
status: current
owner: runtime
last_verified: 2026-10-21
related: [runtime]
code_refs:
  - src/merlin/llvmlower/session_freestanding.py
  - src/merlin/llvmlower/session_bundle.py
  - merlin/tests/runtime/test_session_freestanding.py
---

# Run captured sessions without an operating system

`session_freestanding.generate` produces C execution glue for an existing
version-two captured session. It reuses the session scheduler, stage adapters,
stream inputs and state bindings. A prefill stage followed by recurrent decode
therefore runs as one session, carrying the actual preceding outputs into the
next call. Replaying each stage with its separately captured input is a different
execution and cannot establish this state behavior.

This generator prepares artifacts. It does not compile stages, select an
accelerator, build an ELF or qualify a target. An integrating builder must still
perform ordinary lowering, object namespacing, linking and final-image admission.

## Supply the original session and explicit limits

Keep the captured bundle unchanged and choose a fresh output directory outside
it. Supply limits from the selected execution environment and resource policy:

```python
from merlin.llvmlower.session_freestanding import SessionMemoryLimits, generate


def prepare_session(bundle, work, *, weights_base, address_limit,
                    max_weights_bytes, max_static_io_bytes, max_calls):
    return generate(
        bundle,
        work,
        limits=SessionMemoryLimits(
            weights_base=weights_base,
            address_limit=address_limit,
            max_weights_bytes=max_weights_bytes,
            max_static_io_bytes=max_static_io_bytes,
            max_calls=max_calls,
        ),
    )
```

The weights base must satisfy the existing generated-weight alignment. The
weights interval must fit the supplied address limit. These inputs establish
accepted artifact bounds; they do not prove that the physical memory is
accessible or belongs to this invocation. Structural memory facts inform the
address window. The software/runtime contract establishes ownership and access.

Call-count and static tensor/descriptor checks precede stage generation. Weight
and static-I/O byte checks occur after the existing stage generator runs, so
they do not bound its temporary RAM or disk use. Code, stack, descriptors and
dynamic activations require separate budgets and remain listed as unbudgeted.

## Consume every output before the next call

The generated runner resets the session, prepares each program in its declared
order, executes all its steps and invokes a mandatory observer for every output
after each completed call:

```c
#include "merlin_session_freestanding.h"

static int observe(int program, long step, int output,
                   const void *data, size_t bytes, void *context) {
  /* Consume or copy this output now; a later call may reuse its storage. */
  return consume_output(context, program, step, output, data, bytes);
}

int run_session(void *context) {
  return merlin_session_run_freestanding(observe, context);
}
```

The caller supplies `consume_output`; it may serialize complete results or
compare them against an independently prepared reference. Reference values do
not enter stage compilation. Include logits, recurrent buffers and positions
when they are declared outputs. An observer returning nonzero stops the runner
with an error. Passing no observer also fails. The generator checks output byte
counts but does not interpret values or grant numerical agreement.

## Link the generated artifacts

The output contains the existing per-stage glue, scheduler and renamed MLIR,
plus:

| Artifact | Purpose |
|---|---|
| `merlin_session_freestanding.h` | Observer and runner ABI. |
| `merlin_session_freestanding.c` | Ordered execution and complete output observation. |
| `weights.bin` | One streamed image containing aligned, unchanged stage payloads. |
| `freestanding_session.json` | Limits, stage offsets, source hashes and remaining budgets. |

The default runner uses literal absolute stage addresses derived from the
selected weights base and recorded offsets. A native test can define
`MERLIN_SESSION_WEIGHTS_BASE_ADDR` as the address of its linked weight symbol.
An integrating builder is responsible for checking the actual linked placement
against the selected limits. Equal contents in different stage blobs are not
deduplicated by this generator.

Original source files and generated stage weight identities are checked across
generation. Source changes, stale weight metadata, unsupported static tensor
shapes or exceeded limits refuse with a diagnostic. The existing single-stage
compiler defaults are unchanged.

## Verified scope

The native integration test lowers two actual MLIR stages, namespaces and links
their objects, and runs one initial call plus three recurrent calls. It checks
all 16 output arrays, including previous state, stream inputs and weight echoes,
against an independent reference, with surrounding buffer canaries. Wrong
bindings, omitted state copies, repeated step zero, wrong stage weight bases
and missing observations are discriminating negative controls. The generated
runner also compiles with freestanding C flags.

These tests establish execution glue and state behavior for the tested native
path. They do not establish accelerator completion semantics, whole-model
numerical admission, Spike execution or hardware performance.
