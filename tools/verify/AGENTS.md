# `tools/verify/` — agent guide

## Mental model

`tools/verify_output.py` is the registered shim for `./merlin verify-output`.
It cross-checks a compiled model's output against an onnxruntime golden
via hash comparison. This package adds higher-level verification flows
that span multiple devices or numerical formats.

For the module map, read `__init__.py`.

## Pitfalls

- **REPO_ROOT** is `parents[2]` from here.
- A "verify X" workflow goes inside this package. If the workflow is
  orthogonal to output correctness (signal-shape, layout), it probably
  belongs in `tools/coverage/` instead.

## Cross-references

- Inputs: VMFBs from `tools/compile/` and `tools/run/`.
- Golden generation: `tools/quantize/` (pre-quantization float export),
  or a sidecar `.onnx` next to the source `.mlir`.
- For *which dispatches landed where* (not correctness) use
  `./merlin coverage-check`.
- MCP wrapper: `tools/mcp_servers/verify.py` exposes `verify_output` with parsed
  hashes + max_diff returns.

## Update triggers

Re-read this file and update it in the same turn if you edit:

- `tools/verify/cli.py` (new hash policy / output format) — refresh
  module map; touch `tools/mcp_servers/verify.py` parser.
- `tools/verify/het_e2e.py` / `int8.py` — cross-ref against any
  consumers in `tests/integration/`.
