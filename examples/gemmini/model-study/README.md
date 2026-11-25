# Complete-model optimization study

This study starts from the measured four-hour construction plus four-hour Exo/CCA refinement backend. Keep its device compiler unchanged while establishing complete-model correctness and profiling. Compiler fixes and changes to the shared host pipeline are separate experimental conditions.

| Input | Required identity |
| --- | --- |
| Gemmini backend branch | `paper/refinement/exo-kernels-4plus4h` |
| Branch commit | `7cd030e13360fd0961bd83fa288bd9009ba813df` |
| Compiler source tree | `4994f7b478047decf473e8263f6246b493f5189f0fbfef70b921ba8f99b012f7` |
| Delivered generic library | `ae25c7ee1017586f40679f6e17b33a6ae1ee880067b8a87d4a038eb9496d7833` |

The source-tree digest hashes sorted relative paths, NUL, file bytes, NUL. It is different from a Git commit and from model_check's submission-tree digest. `experiment/source-files.json` in the backend branch identifies every original compiler file.

## Prepare the selected programs

Select a checkpoint-backed capture bundle and its original `configuration/models.yaml`. The retained full TinyLlama capture has 22 layers, 32 prefill tokens and 32 decode steps. Do not use the generic two-layer smoke recipe. The SmolVLA session contains prefix encoding, ten denoising steps and action decoding. Across the selected three models this is 46 forward invocations and 133 declared output arrays, including recurrent state.

`prepare.py` checks compiler membership, file hashes and modes, the generic library, the complete bundle seal index and selected catalog hashes, and capture-metadata consistency. It records the selected Python, dependency origins, solver library and an unsatisfiable Z3 smoke check. Optional lowering probes receive only interface IR, never input or expected tensors. Outputs and temporary caches must be fresh directories outside all frozen inputs.

```sh
"$PYTHON" examples/gemmini/model-study/prepare.py \
  --compiler "$BACKEND/compiler" \
  --compiler-inventory "$BACKEND/experiment/source-files.json" \
  --compiler-sha256 "$COMPILER_SOURCE_SHA256" \
  --compiler-library "$BACKEND/dependencies/generic-library" \
  --compiler-library-sha256 "$LIBRARY_SOURCE_SHA256" \
  --bundle "$BUNDLE" --bundle-manifest-sha256 "$BUNDLE_MANIFEST_SHA256" \
  --models "$BUNDLE/configuration/models.yaml" --python "$PYTHON" \
  --pythonpath "$BACKEND/compiler/compiler_library" \
  --pythonpath "$BACKEND/dependencies/generic-library" \
  --output "$FRESH_PREPARATION" --temporary "$FRESH_TASK_TMP" \
  --model resnet50 --model tiny_llama --model smolvla \
  --compile-probes --max-probes 4096 --timeout 180
```

The preparation record separately checks the sealed prepared programs, quantization policies and device signatures in `routing_artifacts`. A `prepared` result means preparation and requested lowering probes completed. It does not mean replay is ready. `run_check.py` refuses replay when these routing artifacts are missing or changed. Preserve dangling links and missing payload failures as evidence. Recover files against their sealed hashes, or regenerate a separate bundle with a new producer record and compare its interfaces and arithmetic policy with the original.

This finite metadata check is not verification of every large bundle payload. Operand and expected-output arrays can be missing even when routing metadata is present. Run ordinary `model_check bundle-verify` before treating the complete bundle as ready. Capture receipts and import smoke checks do not establish model accuracy or historical runtime equivalence.

## Replay before complete execution

`run_check.py` invokes ordinary model_check with fresh work and report paths. Set the selected LLVM, GCC, Spike, host package, target contract, RTL facts, DTS and current evaluator source roots explicitly. Candidate Python is separated from evaluator Python: only the known compiler entrypoint receives the archived library. Python runs that entrypoint with `-S` and explicit dependency roots, excluding editable-install startup hooks. An import guard rejects Merlin/backend modules outside frozen roots and records every imported module's origin and hash. This is provenance checking, not a filesystem sandbox.

```sh
"$PYTHON" examples/gemmini/model-study/run_check.py \
  --preparation "$FRESH_PREPARATION/preparation.json" \
  --preparation-sha256 "$PREPARATION_SHA256" \
  --output "$FRESH_REPLAY" --timeout 960 -- \
  replay-kernel --model tiny_llama --program prefill \
  --group merlin_dev_gemmini_0 --unique --split visible \
  --engine spike --jobs 1 --timeout-s 900 --no-cache \
  --host-package "$HOST_PACKAGE" --contract "$CONTRACT" --rtl-facts "$FACTS"
```

Use `--no-cache` to avoid reusing the historical missing-Z3 refusals. After representative replay passes, use `check --model MODEL --mode full --full-comparison quantized-reference --input visible/v00 --split visible --engine spike --jobs 1 --timeout-s 3600 --no-cache` with the same selections. Do not cap decode steps for a complete-model result.

Keep seven outcomes separate: input identity, proof runtime, compiler build, complete execution, all-output functional correctness, model quality, and device placement. The ordinary quantized-reference gate covers selected judged outputs. Use the existing `full_output_audit` supplement for all declared arrays and carried state. Do not loosen tolerances to hide dynamic-quantization drift. Historical reference-quality failures must remain visible even when compiled arithmetic is exact.

## Explain the profile

```sh
merlin experiment model-bottlenecks \
  --report "$MODEL_REPORT" --report-sha256 "$MODEL_REPORT_SHA256" \
  --models "$SELECTED_MODELS" --models-sha256 "$MODELS_SHA256" \
  --model "$MODEL" --input visible/v00 \
  --counter-kind spike_retired_instruction_proxy --out "$FRESH_DIAGNOSTICS"
```

The reader consumes existing full-model reports. It checks the complete program/step roster and reconciles routed-call intervals plus gaps with each forward counter. Missing profiles, failed steps and dropped calls remain incomplete. Dropped calls make the remaining gap unattributed, since it can contain device work.

`bottlenecks.json` retains operator diagnostics. `agent-feedback.json` exposes bounded scalar summaries and next checks without private paths, reference tensors or failure text. This command can be used by a model-optimization agent in a separately granted development workspace. It does not change existing private-model access or launch an agent campaign.

Routed-call intervals contain ABI, packing, readout and synchronization performed inside the implementation. Gaps contain host work and profiler bookkeeping. Neither is a pure hardware-engine counter. Forward sums omit inter-step transfers. The first reader reports raw ticks or Spike instruction proxies, never physical cycle qualification or predicted optimization savings.

## Controlled changes and hardware measurements

First join expensive intervals to the actual emitted preparation, readout, layout, grouping and synchronization routes. The frozen backend already requests host elementwise fusion, but disables weight prepacking and shared-input sibling grouping. Settings are candidates for inspection, not evidence of applied or missing rewrites.

Test one mechanism at a time with unchanged arithmetic: immutable-weight prepacking, shared input preparation, layout propagation, supported fusion, projection grouping, or dependence-checked configuration and fence elimination. Preserve a control and test unsafe cases such as aliases, insufficient storage and intervening configuration changes. Merlin must continue to emit primitive commands rather than FSM hardware loops.

Expert QK kernels can require pretransposed keys, and combined QKV/gate-up kernels have a different calling contract from separate projections. Include equivalent preparation and outputs in both timed regions. Supplied model ELFs and static CCA records alone do not establish equivalent full-model work or a numerical reference.

Use local correctness checks before hardware. The first study allows six serialized FireSim submissions through the managed wrapper, with no automatic retries. Keep complete session latency, per-forward intervals, initial preparation, output transport and instrumentation overhead separate. Retain inputs, compiler versions, build recipes, raw observations and failed attempts. Do not modify the archived compiler, sealed bundles or existing AE records.
