---
title: Observe complete bare-metal model outputs
kind: guide
status: current
owner: runtime
last_verified: 2026-10-21
related: [runtime]
code_refs:
  - src/merlin/compile/baremetal_model.py
  - src/merlin/runtime/backends/spike_model.py
  - src/merlin/runtime/out_b64.py
  - merlin/runtime/baremetal/full_readback_b64.h
  - merlin/tests/runtime/test_full_readback_base64.py
---

# Observe complete bare-metal model outputs

Full readback returns every declared forward result, including state outputs,
in its original element format. It is an observation mechanism, not a numerical
reference or a target qualification. The caller still compares complete results
against an independent reference using the selected numerical policy.

## Select a transport explicitly

The saved-model compiler defaults to binary framing. Select printable base64
when the execution path passes through a terminal that may alter raw bytes:

```python
from merlin.compile.baremetal_model import compile_saved_model

result = compile_saved_model(
    **original_build_arguments,
    readback="full",
    full_readback_transport="base64",
)
```

`original_build_arguments` supplies the original capture, target, board, runtime,
output directory, execution selection, reference and numerical policy. This
option also supports the existing explicit readonly argument observations.
Base64 requires full readback; unsupported transport names refuse the build.
The lower-level `spike_model.build` uses `full_readback=True` with the same
`full_readback_transport` option.

Binary `OUT_BIN v1` framing carries raw little-endian bytes and a checksum.
Printable `OUT_B64 v1` framing carries the same native container bits in bounded
chunks. Floating-point values travel as unsigned raw bits; signed integers
retain their declared width, including 64-bit extremes. Neither path rounds,
formats decimal floats, samples values or narrows containers based on their
contents. Base64 adds approximately one third to the payload size, plus framing;
existing console limits still apply.

Changing the transport changes the caller image and build identity. Base64
headers and the selected compilation define are recorded as actual compiler
inputs. Obtain admission for the final linked image and retain its invocation
and compilation records as usual. An earlier binary ELF is a different image.

## Decode retained console bytes

Keep the original console stream as bytes. For an externally run image, select
the same transport when decoding:

```python
from merlin.compile.baremetal_model import read_full_outputs

values, metrics = read_full_outputs(
    console_bytes,
    capture_directory,
    expected_build_hash,
    full_readback_transport="base64",
)
```

The decoder checks complete frames, native widths, output names and shapes
against the captured forward signature, the expected build identity, the
memref-rank diagnostic and final `DONE`. Mixed protocols, repeated outputs,
out-of-order chunks, malformed payloads and truncated frames refuse decoding.
Supply the original `readonly_input_observation` record when that observation
was selected, so every observed argument byte is checked as well.

Printable framing tolerates terminal CRLF and tab expansion because payload
characters contain no raw tabs or binary control bytes. A damaged binary stream
is not repaired by consulting expected values. Preserve that failed run and
build a fresh printable caller when necessary; do not guess the missing bytes.

Output serialization occurs after the model call's timing interval. It consumes
real execution time but is not included in that call counter. Report console
transport overhead separately when it matters. Functional Spike counters do
not establish hardware latency, and successful decoding does not establish
accelerator execution or completion semantics.

## Verified scope

Tests compile the C serializer and send complete native-format values through
a real local pseudo-terminal configured for CRLF and tab expansion. Coverage
includes floating-point raw bits, signed widths, malformed streams and complete
readonly observations. An ordinary tiny multi-output build also executes on
Spike and returns every value through the saved-model decoder.

These checks establish the tested transport and compiler integration. They do
not establish whole-model numerical quality, hardware performance, runtime
ownership or support for an otherwise undeclared target capability.
