---
title: Complete native session output transport
kind: design
status: current
owner: core
last_verified: 2026-10-18
code_refs: [src/merlin/llvmlower/session_outputs.py, merlin/tests/runtime/test_session_outputs.py]
---

# Complete native session output transport

`observed_output_dump_bytes` reads the existing native `SESSION_HEX` console
protocol without creating or changing observed files. Its caller first checks
the full session digest stream against the captured tensor ABI and execution
schedule. Given explicit output taps and a byte limit, the reader requires one
complete byte payload for every selected step, bound to the matching guest
digest. Missing, duplicate, unexpected, malformed, oversized or changed payloads
refuse. Duplicate tap selections and digest keys also refuse.

`assembled_output_bytes` independently reopens retained dumps against their
typed shape, dtype, size and recorded digest, rejecting path escapes, symlinks,
missing steps and duplicate steps. An evaluator can compare those bytes with
the console reader's result without rewriting either artifact. A runner can
use the same reader before initially materializing the dumps.

These are byte-transport checks. Callers still bind the actual executable,
invocation, input capture, compiler objects, simulator and timing environment.
Full output coverage does not establish numerical equivalence, state-routing
semantics, independent reference quality or a hardware timing improvement.
