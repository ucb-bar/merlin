---
title: Compact host rounding at bounded integer observations
kind: reference
status: current
owner: compiler
last_verified: 2026-10-18
related: [shared_compiler_passes, optional_passes, lowering_pipeline, roundeven_runtime, whole_model_op_profile]
code_refs:
  - src/merlin/llvmlower/automatic_rounding.py
  - src/merlin/llvmlower/quant_round.py
  - src/merlin/llvmlower/quant_round_check.py
  - src/merlin/llvmlower/lower.py
  - src/merlin/llvmlower/optional_passes.py
  - merlin/tests/ir/test_automatic_rounding.py
  - merlin/tests/ir/test_stock_cli.py
---

# Compact host rounding at bounded integer observations

Ordinary serial host lowering automatically recognizes a closed, exact
residual-rounding expression and selects the standard LLVM round-even intrinsic.
It runs after source row/destination selection and before profiling or upstream
bufferization. This preserves opportunities for destination reuse while giving
the selected native compiler a compact operation to implement.

## Inputs and meaning

The pass consumes typed tensor computation, literal integral clamp bounds,
integer/float widths, original index/data-layout selection and code-growth
budgets. Hardware encodings and cycle costs are not inputs to this source
identity. The selected native toolchain owns intrinsic lowering; any library
fallback must implement the [round-even value contract](roundeven_runtime.md).
RTL widths alone do not qualify that implementation.

For a clamped value `c`, the existing exact arithmetic computes truncation `t`,
an exact residual `c - float(t)`, and a correction for magnitude above one half
or an odd half tie. The new representation computes `fptosi(roundeven(c))`.
Both produce the same integer when the actual two-sided integral bounds fit the
destination and remain within the source float's exact-integer range. Source
scaling, multiplication/division order, clamps, tensor types and every other
observation remain unchanged. No incoming rounding mode is assumed or modified.
Constrained FP operations or observable flag/errno contracts are outside this
value-only route.

## Selection and verification

`automatic-host-rounding` defaults on in the supported serial host source
family. Use `--no-pass automatic-host-rounding` or
`MERLIN_AUTOMATIC_HOST_ROUNDING=0` for a paired control; explicit pass selection
takes precedence over the environment default. Parallel/vector schedules,
source-bound observer selections and unsupported layouts retain their original
route. Both the Python runner and stock scalar transport can carry the standard
LLVM operation.

The scalar body must be canonical and pure, in a direct single-block function.
Unknown operations, attributes or enclosing numerical contracts block selection.
Other users of the bounded operand are allowed; live rounding intermediates
block selection to avoid duplicating their work. Source labels, model
names, captured IDs and target names never select the strategy.

The default budgets are 4096 net added IR operations and 512 operations per
original scalar body. Each replacement adds at most one net operation before ordinary
dead-code removal. The policy favors a compact exact source representation;
there is no cycle estimate or universal speedup claim. Phase 2 can compare this
route with its retained arithmetic alternative under the same checks and inputs.

`PipelineSession` enforces source prerequisites, invalidates old analyses during
replacement, then checks the realized arithmetic. The checker reads actual
intrinsic operands and bounds and expands the observation into the independent
residual expression. Complete outer interfaces, operations, effects and dataflow
must correspond. The report's rewrite count must equal the actual added
intrinsics; a changed helper/checker library refuses publication.

The frontend carries this one known standard LLVM operation in generic syntax.
This grants no semantics to arbitrary unregistered operations or opaque backend
output. LLVM/source conformance and actual final native execution remain
separate verification levels.

## Products and scope

The ordinary lowerer writes `automatic-host-rounding.json`, returns
`stats["automatic_host_rounding"]`, and records `host-rounding-selected` in an
enabled IR audit. Source/output, current numerical-source context, data layout,
budgets and direct helper identities are bound in that report. No match,
unsupported selection or exhausted budget retains the exact source bytes with
a diagnostic. Invalid selected transformations fail rather than publish success.

Native pairs exercise f16/f32/f64 and several integer widths, rectangular tails,
aliased inputs, overwrites, output guards and three carried-state calls. They
check complete independent results and actual LLVM intrinsic/subtraction
presence. Mutation tests alter bounds, operands, intrinsic semantics, arithmetic,
attributes, observations and library/accounting state. The stock route also
executes the resulting padded quantization through its ordinary tools.

These tests establish the documented source mechanism and selected native
toolchain behavior. They do not qualify arbitrary target intrinsics, numerical
policies, device instructions, asynchronous execution or hardware timing.
