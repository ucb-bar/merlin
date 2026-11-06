# Attention capture execution

`third_party/atlas-kernels/lib/atlas_execution.py` owns reusable physical instruction execution. `Executor(model_root)` requires an explicit physical checkout of npu_model commit `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5`. It uses the original `DefaultHardwareConfig` and `ArchState` with `numerics="rtl"`, a fixed initialization seed, and physical instruction `exec` methods. It does not measure Core cycles. Python must provide NumPy, Torch, and the dependencies imported by the selected npu_model checkout.

`executor.execute(source, arena, output_offset, output_bytes, expected_arena=..., requested_dma=..., program_sha256=...)` compares the entire final arena, checks preservation outside the output region, and validates load/store DMA totals and assembled binary identity. The optional expected arena is omitted during capture, when this execution creates the reference. Every execution records the selected model revision, imported model source hashes, all eleven RTL unary table hashes, memory curve hash, helper hashes, and explicit RTL mode. Each model dependency must match its named revision's Git blob, using scoped `git cat-file` requests. Source bookends reject changes between executor initialization and completion. The module rejects model imports from another checkout, odd VPU pair destinations, binary VPU read-port aliases, oversized programs, and missing ECALL completion.

`tools/tiny_full/attention_worker.py --model-root /physical/npu-model` is the delivered persistent JSONL worker. It uses the shared executor, checks independently counted generator DMA traffic, writes full staged arenas, and returns attention results to Merlin. It accepts capture directories outside the source checkout and does not search for sibling helpers or models.

Use [model bringup setup](../../README.md#build), keeping `ATLAS_ROOT` and `ATLAS_MLIR_BIN` on the canonical compiler, then export `ATLAS="$VIEW"`, `ATLAS_SOURCE_ROOT="$VIEW"`, and `ATLAS_MLIR_ROOT="$VIEW"`, and use `cd "$VIEW"`. For Merlin's `examples/atlas/comparisons/capture_tinyllama_full.py`, use the existing historical, arithmetic, and output arguments together with these arguments:

```sh
--attention-worker-python /physical/python-environment/bin/python \
--attention-worker-script "$VIEW/tools/tiny_full/attention_worker.py" \
--attention-model-root /physical/npu-model
```

The worker interpreter must have the model execution dependencies; Merlin's capture interpreter may remain separate. No `PYTHONPATH` or job-local helper is required. The exported capture records the delivered worker, generator, execution module, and path guard as implementation sources. Existing captures and archived source pins remain valid for re-export; replacing their historical source records would invalidate their original receipts.

For export, pass `--model-root /physical/npu-model --qualify-launches` to `tools/tiny_full/export_capture.py` along with the capture hash, output directory, source archive, and any required relocation arguments. This uses the same executor and checks every staged full arena and requested DMA count. The captured instructions and arithmetic are unchanged by moving the execution helper.

Run the focused real-model checks from a checkout with no sibling helpers or model directory:

```sh
ATLAS_EXECUTION_MODEL_ROOT=/physical/npu-model /physical/python-environment/bin/python -B \
  -m unittest discover -s test -p test_atlas_execution.py -v
```

These checks execute a small attention program against the unmodified selected model, compare the complete arena with analytically known BF16 results, exercise rejection paths, and run two requests through the persistent worker with an external capture directory. The model path is never inferred from the test or repository parent.
