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
  - src/merlin/runtime/backends/spike_session.py
  - merlin/tests/runtime/test_session_freestanding.py
  - merlin/tests/runtime/test_spike_session.py
---

# Run captured sessions without an operating system

`session_freestanding.generate` produces C execution glue for an existing
version-two captured session. It reuses the session scheduler, stage adapters,
stream inputs and state bindings. A prefill stage followed by recurrent decode
therefore runs as one session, carrying the actual preceding outputs into the
next call. Replaying each stage with its separately captured input is a different
execution and cannot establish this state behavior.

The generator prepares artifacts. The opt-in `spike_session.build_session`
builder compiles those stages through ordinary lowering and object construction,
isolates their helper symbols, and links one freestanding ELF. Neither entry
point qualifies a target or runs a performance experiment automatically.

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

## Build one executable image

Supply a caller-owned observer source implementing the ABI above, one original
`DeviceRouting` or `None` for each program, and the selected runtime's memory
intervals. For example, an integrating caller can invoke:

```python
from merlin.runtime.backends.spike_session import build_session

image = build_session(
    session_bundle,
    fresh_build_directory,
    limits=selected_limits,
    arena_bytes=selected_arena_bytes,
    stack_bytes=selected_stack_bytes,
    dram_base=selected_dram_base,
    dram_bytes=selected_dram_bytes,
    devices=stage_routes,
    observer_sources=(observer_c,),
    observer_inputs=reference_files,
    cflags_override=selected_runtime_cflags,
    execution_memory_map=selected_memory_binding,
)
```

The caller supplies these variables from its original session, runtime and
resource selection. Observer sources are compiled after all target compiler
invocations; reference files do not enter device compilation. The builder does
not choose a simulator or start a run. It returns the linked ELF, stage products,
invocation records, memory checks and any final-image admission result.

Selected device stages must use the same target and the exact same live
linked-ELF admission service. Admission applies to the combined final ELF, not
only to its stage objects. Descriptor-result forwarding is currently unsupported.
Unknown target ownership, numerical or completion guarantees remain unknown.

Actual ELF loads, stack and allocator reservations must fit their declared
intervals without overlap. Without an explicit memory-map binding, the memory
result establishes interval arithmetic only, not physical accessibility. Source,
tool, route, callback and compiled-product changes across build boundaries
refuse the build.

The image records separate cycle intervals for reset, program preparation and
each step. Output observation and metric printing are outside these intervals.
Their sum is not continuous session latency; functional simulator counters are
not accelerator timing. Hardware timing and physical completion need the
selected execution environment's independent checks.

## Verified scope

The native integration test lowers two actual MLIR stages, namespaces and links
their objects, and runs one initial call plus three recurrent calls. It checks
all 16 output arrays, including previous state, stream inputs and weight echoes,
against an independent reference, with surrounding buffer canaries. Wrong
bindings, omitted state copies, repeated step zero, wrong stage weight bases
and missing observations are discriminating negative controls. The generated
runner also compiles with freestanding C flags.

The ordinary builder test also compiles two scalar stages into one actual
RISC-V ELF and executes the same complete 16-array observation on Spike. Mutation
tests reject a rewritten host object, a later stage overwriting an earlier
object, and changed original build controls before linking.

These tests establish the tested native and scalar Spike execution paths.
They do not establish accelerator completion semantics, full-model numerical
admission or hardware performance. Device-stage integration still requires the
original target contracts and final-image admission described above.
