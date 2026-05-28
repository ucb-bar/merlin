"""Implementation package for the `verify-output` subcommand.

The registered shim `tools/verify/cli.py` cross-checks a compiled
model's output against an onnxruntime golden via hash comparison. This
package adds higher-level verification flows.

Extension points:

- `int8.py` — INT8-specific correctness check via cross-hash comparison.
  Extend for new quantization schemes (uint8, fp8).
- `het_e2e.py` — end-to-end numerical verification harness for
  heterogeneous schedules. Runs scheduled and unscheduled VMFBs with the
  same input and compares hashes.

Anything that's a "verify X" workflow goes inside this package. If the
workflow is genuinely orthogonal to output correctness (signal-shape,
layout), it probably belongs in `tools/coverage/` instead.
"""
