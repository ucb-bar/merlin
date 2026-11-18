# Execute a compiled application on npu_model

This example packages an application compiled by the standard Merlin commands and runs it on the pinned npu_model, keeping compilation and execution receipts separate. It does not capture or compile a model, and it makes no exactness claim by itself: compare every raw output word with an independent reference.

The pinned model2MLIR revision `ebb1c600e0d7f3205c801d8f00acb960015b32f2` is available from [reed-nicolas/model2MLIR, branch feat/tinyllama](https://github.com/reed-nicolas/model2MLIR/tree/feat/tinyllama) until its upstream PR merges. The software verifier retains the canonical upstream origin: clone `https://github.com/ucb-bar/model2MLIR.git`, add `https://github.com/reed-nicolas/model2MLIR.git` as a named remote, fetch `feat/tinyllama`, and check out the pinned revision.

## Live capture and the standard application compiler

Use `merlin targetgen native-compile --application` to stage captured IR and parameters through the installed provider's `application_adapter` and `compile_region`. See the shared [application workflow](../../../../docs/guides/whole_model_on_accelerator.md). Install matching Merlin, model2MLIR, and atlas-mlir checkouts in the compiler environment, then install the standalone `atlas_fp8` adapter package with `python -m pip install -e "$ATLAS_ROOT/adapter"`.

After a worker capture with `--materialize-bundle --already-quantized --quant-adapter atlas_fp8`, the compiler commands are:

```sh
nice -n 10 merlin targetgen stage-capture --capture "$LIVE_CAPTURE" \
  --precision-policy "$PRECISION_POLICY" --out "$STAGED"
nice -n 10 merlin targetgen native-compile --engine merlin_native --support atlas \
  --snapshot "$SNAPSHOT" --target-source "$ATLAS_ROOT" --compile-inputs "$STAGED" \
  --application --already-quantized --application-parameters external \
  --application-name capsule --application-memory-bytes 536870912 \
  --io-memory-bytes 67108864 --abi auto --mode diagnostic --out "$COMPILED/application"
nice -n 10 merlin targetgen native-verify --engine merlin_native --support atlas \
  --snapshot "$SNAPSHOT" --target-source "$ATLAS_ROOT" --compile-inputs "$STAGED" \
  --application --already-quantized --application-parameters external \
  --io-memory-bytes 67108864 --mode diagnostic --artifact "$COMPILED/application" \
  --out "$COMPILED/application-verified.json"
```

`$PRECISION_POLICY` is a source-bound JSON policy as described in [model2MLIR capture](../../../../docs/guides/model2mlir.md): it selects source arithmetic and carries a `source_sha256` equal to the captured `model.mlir`. Search, allocation, proof, and tool limits retain their defaults. Build the source-pinned snapshot with `merlin targetgen native-build --engine merlin_native --support atlas --io-memory-bytes 67108864 --cargo-target-dir "$CARGO_TARGET" --source-revision "$MERLIN_REVISION" --out "$SNAPSHOT"`.

## Package and execute

The packager replays `native-verify`, checks capture, staging, and input identity against the frozen boundary receipt, and builds the host library from the emitted C. `$BOUNDARY_SOURCE` is a JSON file whose `tensors` object maps each boundary tensor name to its `sha256` (of the raw tensor bytes), `shape`, and `dtype` (a torch dtype name such as `torch.bfloat16`), and whose `input_keys` lists those names in application input order; `$FROZEN_BOUNDARY_SOURCE_SHA256` is the SHA-256 of that file. `$BOUNDARY_TENSORS` is a safetensors file holding exactly those tensors. Ordinary packaging refuses unsatisfied host-linkage obligations.

```sh
TOOLS=examples/atlas/whole-model/unchanged_fx
nice -n 10 "$COMPILER_PYTHON" "$TOOLS/prepare_live_execution.py" \
  --compiled "$COMPILED" --capture "$LIVE_CAPTURE" --staged "$STAGED" \
  --snapshot "$SNAPSHOT" --atlas-root "$ATLAS_ROOT" \
  --boundary-source "$BOUNDARY_SOURCE" \
  --expected-boundary-source-sha256 "$FROZEN_BOUNDARY_SOURCE_SHA256" \
  --inputs "$BOUNDARY_TENSORS"
nice -n 10 "$MODEL_PYTHON" "$TOOLS/execute_graph.py" \
  --compiled "$COMPILED" --atlas-root "$ATLAS_ROOT" --model-root "$MODEL_ROOT" \
  --backend functional --max-instructions 1000000 --compact-trace \
  --execution-output "$EXECUTION"
```

Use the pinned model's compatible interpreter, with `$ATLAS_ROOT` and `$ATLAS_ROOT/python` on its import path. Execution publishes `EXECUTED_UNCOMPARED`; the functional backend executes one delay slot after each taken branch or jump and rejects control flow inside a delay slot. Compare every raw BF16 word of `$EXECUTION/actual.npz` with the same-shape reference.

Select qualified execution with `--backend qualified_native --native-qualification "$MATH_QUALIFICATION" --memory-qualification "$MEMORY_QUALIFICATION" --native-cache "$NATIVE_CACHE"`. It authenticates both receipts and retains the same default 1,000,000-instruction limit per launch; missing qualification refuses.

## Qualification-only combined loop, cache and residency gate

The optional residency example is a qualification bridge, selected only by `execute_graph.py --program-residency` with a functional backend. Generated packages do not include or enable it. The recipe uses `resident_runtime.py`, `program_residency.py`, `program_residency_bridge.c`, and `examples/atlas/program_session.h`. The generic Merlin C session owns retention policy; Atlas supplies physical IMEM primitives. The bridge explicitly prepares its separate qualification host and exclusively owns that endpoint. Platform resets must use `atlas_slots_reset_completed` to invalidate its retained handles. The bridge decodes and executes the complete persistent IMEM from reset PC zero, including the selected JAL delay slot. This is functional qualification, with hardware and timing gates separate.

The reset-entry trampoline adds two executed instructions per resident launch beyond the body's proven instruction bound. No static proof currently combines that overhead with the body bound. Only the runtime instruction cap enforces the 1,000,000-instruction policy for the combined launch. Keep `--max-instructions 1000000`; a launch that reaches the cap without halting fails.

Compile with the loop and cache options, package with `prepare_live_execution.py` as above, then execute with residency:

```sh
nice -n 10 "$COMPILER_PYTHON" -m merlin.targetgen.cli native-compile \
  --engine merlin_native --support atlas --snapshot "$SNAPSHOT" \
  --target-source "$ATLAS_ROOT" --compile-inputs "$STAGED" \
  --application --already-quantized --application-parameters external \
  --application-name capsule --application-memory-bytes 536870912 \
  --io-memory-bytes 67108864 --abi auto --mode diagnostic \
  --materialization-cache --region-materialization-cache \
  --application-region-loop --deterministic-receipts \
  --out "$COMPILED/application" \
  --status-file "$COMPILED/compile-status.json"
nice -n 10 "$MODEL_PYTHON" "$TOOLS/execute_graph.py" \
  --compiled "$COMPILED" --atlas-root "$ATLAS_ROOT" --model-root "$MODEL_ROOT" \
  --backend qualified_native --native-qualification "$ARITHMETIC_QUALIFICATION" \
  --memory-qualification "$MEMORY_QUALIFICATION" --native-cache "$NATIVE_CACHE" \
  --program-residency --max-instructions 1000000 --compact-trace \
  --execution-output "$EXECUTION"
```

Packaging independently replays `native-verify`, including the loop option recorded in the manifest. The counted C driver still invokes every original wrapper and binding. The two materialization caches affect compilation; `--deterministic-receipts` selects reproducible receipt metadata explicitly, and `--native-cache` holds qualified execution support. None of these options retains operand writes. Check the manifest's loop selection, compile-status cache records, and execution residency counters when recording the combined gate.

## Bind generated outputs to later inputs

Pass an explicit JSON descriptor to `execute_graph.py --input-bindings bindings.json` to feed one execution's outputs to a separately compiled application, for example a prefill's cache outputs and greedy token to a decode:

```json
{
  "schema": "merlin.application_output_bindings.v1",
  "producer_application": "prefill64/compiled",
  "producer_execution": "qualified/prefill64",
  "bindings": [
    {"input_index": 3, "output_index": 1},
    {"input_index": 4, "output_index": 2}
  ],
  "greedy": {"input_index": 0, "output_index": 0}
}
```

Paths resolve relative to the descriptor. The helper checks producer identity, completed execution, output hashes, and exact input/output shape and dtype before exposing generated arrays, and the execution receipt records the producer and each consumed output. The optional `greedy` edge widens BF16 words exactly, selects the last sequence row, chooses the lowest vocabulary index on a tie, and refuses NaNs. This uses the existing synchronous, caller-owned application buffers: keep producer outputs alive through consumer completion and allocate separate consumer outputs; input/output aliasing is not promised. There is no new target ABI or persistent device-state requirement.

The default `python -m pytest` run collects these tests; `python -m pytest examples/atlas` runs only the Atlas examples.
