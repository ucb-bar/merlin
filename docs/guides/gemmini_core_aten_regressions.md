---
title: Gemmini compiler regressions and the Core ATen overlay
kind: guide
status: current
owner: targetgen
last_verified: 2026-10-23
related: [phase0_specification, generating_capsules]
code_refs: [packages/merlin-experiments/src/merlin_experiments/phase0/regression_capsules.py, src/merlin/targetgen/core_aten_overlay.py, build_tools/scripts/capture_core_aten_cases.py, build_tools/scripts/build_core_aten_batch.py]
---

# Gemmini compiler regressions and the Core ATen overlay

Operator how-tos for the [Gemmini Phase 0 walkthrough](../../examples/gemmini/phase0/README.md).

## Compiler-fix regression inputs

Compiler fixes have reviewed regression inputs in
[`compiler-regression-selection.yaml`](../../experiments/reference-data/phase0/compiler-regression-selection.yaml) and
the shared [`compiler-regression-calls.yaml`](../../experiments/reference-data/phase0/compiler-regression-calls.yaml).
The shared [`compiler-regression-fixes.yaml`](../../experiments/reference-data/phase0/compiler-regression-fixes.yaml)
maps each fix to a call obligation or an infrastructure pytest. These are public
development inputs, separate from default derived-only release membership.
The device selection links the existing transposed-RHS sealed capsule rather
than adding a duplicate. Input shapes and strides describe retained calls;
they do not override extracted hardware geometry or instruction encodings.

Generate the generic corpus with the controller environment and an explicitly
selected Torch/model2MLIR interpreter:

```sh
python -m merlin_experiments.phase0.regression_capsules \
  --spec experiments/reference-data/phase0/compiler-regression-calls.yaml \
  --output "$REGRESSION_OUTPUT" --m2m-dir "$MODEL2MLIR_ROOT" \
  --python "$CAPTURE_PYTHON"
```

Choose a fresh `REGRESSION_OUTPUT` beneath `merlin.common.paths.artifacts_dir()`.
For the device inputs, use `--selection experiments/reference-data/phase0/compiler-regression-selection.yaml`
with explicit `--cases` and `--captures` owners from the retained public overlay.
The existing writer packages the selected calls and keeps eager answers in a
separate owner-only directory. Unavailable captures remain explicit in
`packaging.json`; they cannot count as generated or passing capsules. Repeated
generation preserves program/input bytes for the same selected tool and inputs;
capture location receipts and private commitments remain run-specific.
Promoting these calls into a release still requires an explicit reviewed
retained-call cohort selection through `core_aten_stage`, followed by its
receipt-bound derivation. Neither this command nor the regression seeds bypass
`capsule_policy: derived_only` or certify numerical/device execution.

## The Core ATen overlay on Gemmini

The Gemmini OOT `merlin-support/core_aten/` provider supplies Gemmini's pinned configuration and candidate
pool to the generic additive Core ATen overlay (`merlin.targetgen.core_aten_overlay`);
Its `build_overlay.py` writes the overlay under the artifacts root. Its cases never replace portable
Core ATen obligations.

The overlay builder also writes `core_aten_overlay_cases.json`, which the generic capture and batch
commands consume directly. Variant captures and verdicts use `case_id`; cases sharing an overload
remain separate. For accelerator execution, use the handwritten implementation package and an
explicit facts pin. For example, from this worktree with its interpreter:

```sh
PY="$PWD/.venv/bin/python"
OVERLAY_ROOT="$($PY -c 'from merlin.common.paths import artifacts_dir; print(artifacts_dir() / "verification" / "core-aten" / "gemmini-overlay")')"
$PY "$GEMMINI_SUPPORT/core_aten/build_overlay.py" --output-dir "$OVERLAY_ROOT"
$PY build_tools/scripts/capture_core_aten_cases.py \
  --corpus "$OVERLAY_ROOT/core_aten_overlay_cases.json" --output-dir "$OVERLAY_ROOT/captures"
$PY build_tools/scripts/build_core_aten_batch.py \
  --corpus "$OVERLAY_ROOT/core_aten_overlay_cases.json" --captures "$OVERLAY_ROOT/captures" \
  --output-dir "$OVERLAY_ROOT/device-batch" --target gemmini \
  --execution-provider "$GEMMINI_SUPPORT/core_aten/execution_provider.py" \
  --device-package "$DEVICE_PACKAGE" --rtl-facts "$RTL_FACTS_FILE" --run-spike
```

`DEVICE_PACKAGE` selects the OOT checkout with `mlir_oot.golden_device_catalog`; `RTL_FACTS_FILE`
selects the hardware observation, also selectable through `MERLIN_RTL_FACTS`. Capture needs its
configured model2MLIR/PyTorch interpreter; overlay construction needs PyTorch and z3. The execution
provider selects the extension library beside the selected `MERLIN_SPIKE` installation and prepends
its enclosing environment's `bin` directory for dtc. Select a Spike build with that extension installed.
Provider selection is required through `--execution-provider` or
`MERLIN_CORE_ATEN_EXECUTION_PROVIDER`; no in-tree provider is discovered.
`GEMMINI_SUPPORT` names the OOT repository's `merlin-support/` directory.
Select the lane's host declarations with `--host-capabilities
examples/gemmini/platform/host-capabilities-core-aten.yaml`.
Leave `MERLIN_TARGET_PATH` unset so contract and runtime resolve through the in-repo data-only
`examples/gemmini/support`; the OOT directory supplies only the execution provider. Its executable
backend has no epilogue-stage routes for the contract's declared readout semantics and refuses them.

Device mode requires one case per shard. A closed wrapping int8 multiply/add contraction can be
widened exactly to the facts-derived accumulator type and truncated back to its original result type
on the host. The original MLIR and a `modular_widening.json` receipt are retained. This preserves
Core ATen wrapping results, including overflow, without using saturating accelerator readout.
Nonzero initialization, overflow promises and additional arithmetic refuse that preparation.
Contractions with unsupported numeric or source semantics
remain host work, with the decline reason in the case map and verdict. `passed_by_lane` separates
host passes from device passes; `_int_mm` remains a non-Core bridge. A device pass requires source-bound routed calls,
correct result bytes, and a retained Spike trace containing at least one target instruction at a
matching linked-ELF address. Regrading rechecks the trace, executable and facts hashes; missing or
changed evidence fails closed. All results are dumped as storage bytes after model timing, without
floating conversion. This is functional Spike-extension evidence, not an elaborated-RTL verdict.
Header-derived provisional fixtures remain diagnostic and must not be presented as final hardware
qualification. Omit `--target` and `--device-package` for the default scalar lane.
