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

For a single captured forward, `full_model.py --help` describes the ordinary linked-image diagnostic. Select `--host-python` with the upstream `torch_mlir` bindings and select `--candidate-python` independently for the archived device compiler. The two interpreters need not be the same. The driver records their origins, builds through `full_build.build`, runs Spike, and compares every returned array against the supplied reference under the selected host arithmetic and tolerances. Its `result.json` records the ELF, outputs, numerical comparisons and unchanged input checks. One forward does not qualify a recurrent session.

Generate an independent captured-forward reference from a freshly prepared catalog before running that driver:

```sh
"$PYTHON" examples/gemmini/model-study/reference.py \
  --catalog "$FRESH_CATALOG" --models "$SELECTED_MODELS" \
  --bundle "$SELECTED_REFERENCE_BUNDLE" --model "$MODEL" --program "$PROGRAM" \
  --output "$FRESH_REFERENCE"
```

The reference uses the model's selected host arithmetic and interprets both the reference and ordinary device-rewritten programs without a candidate compiler. It checks every declared output and records their byte agreement. Declared Boolean outputs may use canonical one-byte integer storage internally. The driver validates every byte is zero or one and exposes the logical Boolean ABI without changing its bytes. It refuses other dtype conversions, shape changes and forbidden nonfinite outputs. `expected.npz` and its `result.json` are inputs to the evaluator, never to device compilation. This reference is a new derivation for the captured inputs, not a historical recovery or a complete recurrent-session reference.

The linked-forward driver requires a successful `captured_reference.v2` record. Admission binds the selected model and program, source capture, prepared catalog, numerical policy, producer, and expected-array bytes. It repeats these checks after execution. An older record without this evidence is not admitted automatically. Keep its original result, generate a new reference record, and compare saved outputs separately instead of changing the historical metadata.

Keep seven outcomes separate: input identity, proof runtime, compiler build, complete execution, all-output functional correctness, model quality, and device placement. The ordinary quantized-reference gate covers selected judged outputs. Use the existing `full_output_audit` supplement for all declared arrays and carried state. Do not loosen tolerances to hide dynamic-quantization drift. Historical reference-quality failures must remain visible even when compiled arithmetic is exact.

## Record a condition without promoting its claims

Use an immutable Merlin revision and retain each condition's source inventory, configuration, original failures and result records. `conditions.py` creates a small index of **explicitly named metadata files**, each with a caller-selected SHA-256. It reads no nested paths, discovers no dependencies, copies no tensor values and imports no verdicts from the indexed records. A failed or still-running result can be indexed without becoming a successful experiment.

```sh
"$PYTHON" examples/gemmini/model-study/conditions.py create \
  --label resnet-baseline \
  --result run "$RESULT_JSON" "$RESULT_SHA256" \
  --manifest image "$CHAIN_IMAGE_JSON" "$CHAIN_IMAGE_SHA256" \
  --config selection "$SELECTION_JSON" "$SELECTION_SHA256" \
  --config models "$SELECTED_MODELS" "$MODELS_SHA256" \
  --output "$FRESH_INDEX_JSON"

"$PYTHON" examples/gemmini/model-study/conditions.py verify \
  --index "$FRESH_INDEX_JSON" --index-sha256 "$INDEX_SHA256"
```

Creation prints `index_sha256`; retain it independently for verification. Select existing canonical ordinary files and a fresh output filename in the operator's artifact directory. The index accepts up to 64 JSON result/manifest or JSON/YAML/TOML config files, at most 2 MiB per file and 16 MiB total. It rejects symlinks, changed bytes, duplicate names and extra index claim fields. Only paths, roles, names, sizes and digests are copied. Paths themselves may be private: keep the index access controlled when necessary. This is neither a payload exporter nor a recursive bundle verifier. Its `identities_verified` status grants no numerical, quality, no-FSM, hardware or expert-comparison qualification.

## Reproduce a complete session

Choose which operation is intended before selecting sources:

| Operation | Required inputs | Resulting scope |
| --- | --- | --- |
| Rerun a saved image | Exact ELF/image manifest, referenced verification closure, selected engine and matching decoder | A new execution of those saved bytes; no model compilation |
| Rebuild a historical condition | Original producer/runtime sources, generated inputs, exact tools, flags, configuration and historical receipts | A rebuild attempt of that version; byte identity must be checked, not assumed |
| Build from current source | An immutable current revision, selected captures, new independent references, explicit condition settings and target/tool bindings | A new source condition with new receipts; historical verdicts do not transfer |

The supplied study artifacts retain original producer copies and staged measured protocols. Preserve them even after an additive protocol is promoted into canonical source. A current checkout is not a substitute for an older pinned producer. Imported source snapshots also do not by themselves freeze a build key: `MERLIN_REPO_ROOT` selects the source inventory hashed by the ordinary build machinery. Bind that root and its bytes explicitly.

All commands below use operator-supplied paths and fresh output directories. Provision evaluator Python, upstream torch-MLIR host Python and frozen-compiler candidate Python separately, with exact package/tool artifact identities. Keep the original LLVM tools, RISC-V GCC/binutils/libm, host package, board catalog, DTS, target contract and RTL facts selected. For Spike, bind the actual binary, extension, ISA, privilege and memory-map identities from the admitted invocation; an executable found on `PATH` is insufficient. Regenerating floating-point model references also requires each model's framework environment and pinned checkpoint/input workers.

### Prepare complete references and programs

Supply a source-bound complete-session reference bundle, its seal/admission record and the original selected models configuration. TinyLlama requires all 33 forwards and 99 output and state arrays; ResNet requires one forward and its 1,000-logit array; SmolVLA requires all 12 forwards and 33 arrays. Use the ordinary `chain.run_reference`/`ChainCache` APIs under `reference_math_context` when deriving a new complete reference, retaining every declared dtype, array and state transition. The standalone `reference.py` command above covers captured inputs only; it does not generate or seal a recurrent-session reference. Catalog generation, checkpoint/input access and a new complete reference seal remain explicit prerequisites, not files this guide synthesizes automatically.

An original bundle with absent payloads or a refused numerical-source identity remains refused. Regenerate in a separate location with producer, source, input, math and output identities; compare with preserved historical answers observationally without relabeling the new bytes as recovery. Never truncate layers, heads, prefill lengths or recurrent steps to make a complete-session command finish.

Build each declared program through `full_model.py` with its admitted captured-forward reference before packaging a session. Preserve the control and select each optimization explicitly. `--host-elementwise-fusion` enables the existing host fusion condition; `MERLIN_HOST_BOUNDED_RNE=1` selects the existing bounded conversion condition. Neither is implied by the backend preference file. A typed `--source-route` additionally needs the original normalized source hash and independently inspected borrowed-effect premises. Later derived rounding or quantization variants need their own identities and checks. Model labels identify experiments; production transformation decisions must follow source semantics and target capabilities.

Expected tensors belong to reference checking and the comparison guest, not the device-compiler input. Retain the actual compiler arguments and imported-module origins. These guards are provenance checks, not a claim of filesystem isolation. Saved session ELFs can embed private inputs, weights and expected arrays; give them the same access restrictions as those payloads rather than publishing them with the source guide.

### Package verified repeats from completed objects

The additive builder leaves the historical chain builder and guest available:

```sh
"$PYTHON" -m merlin.model_check.chain_image_verified \
  --selection "$SELECTION_JSON" --out "$FRESH_IMAGE_DIR"
```

The caller supplies a `merlin.chain_image.selection.v1` document. It binds models and input pins, every completed program receipt, frozen compiler identity, independent reference bundle/identity, workload and counter contracts, boot selection, admitted execution map and allocation layout. Every path must satisfy the ordinary canonical-file checks. For the measured repeat protocol select `transport: verified_repeats_v1`, `timing_repeats: 3` and `group_profile: false`. Verified repeats require `retained_group_trace_policy: bypass_verified_neutral` and exact regenerated neutral wrappers; it does not grant permission to discard arbitrary profiling/effect code.

The builder reopens completed objects and recipes, preserves the full program roster and typed state bindings, derives the schedule and checks actual allocation/ABI/memory compatibility. It compiles comparison/runtime support and links a new image; it does not recompile the model kernels. Different packaging sources produce a new condition even when the completed model objects are unchanged. Recheck its numerical outputs and actual final ELF instructions before hardware submission.

The initial pass checks every declared array at every step. Each timed repeat then executes the complete session and checks the last declared outputs of each program after its timer stops. Intermediate repeat arrays are not independently captured. Whole-session intervals include input/state copies, arena resets, descriptors, forwards and scalar bookkeeping; comparisons and console transport are excluded. Keep initial setup/weight preparation, warm repeats, whole-session intervals and summed-forward intervals separate.

### Check a console or rerun the saved verified image

```sh
"$PYTHON" examples/gemmini/model-study/session.py check \
  --image "$CHAIN_IMAGE_JSON" --image-sha256 "$CHAIN_IMAGE_SHA256" \
  --console "$RAW_CONSOLE" --console-sha256 "$CONSOLE_SHA256" \
  --output "$FRESH_CHECK"

"$PYTHON" examples/gemmini/model-study/session.py run-spike \
  --image "$CHAIN_IMAGE_JSON" --image-sha256 "$CHAIN_IMAGE_SHA256" \
  --timeout-s "$SPIKE_BUDGET_SECONDS" --output "$FRESH_RUN"
```

This CLI supports `verified_repeats_v1`. Historical compact/full-array transports still require their matching retained reader, or a separately packaged and qualified new image. Select a sufficient explicit budget; the captured-forward driver's default is not a complete-session budget.

`check` verifies the image closure and strict complete-output transport before and after decoding. A saved console alone does not establish where it ran: its counter kind is `unattributed_saved_console`. `run-spike` additionally verifies the actual consumed simulator command against the admitted binary/extension/map/ISA/privilege selection. Its counters remain Spike instruction proxies, not physical cycles. Both paths preserve failure output in a fresh result record. Boolean comparison checks canonical raw bytes before logical conversion; do not replace it with a permissive cast or digest-only comparison.

Strict image verification reopens recipes, producers, source pins, namespaced objects/mappings, invocations, counter support and selected input/reference blobs. Copying only the ELF and manifest is insufficient for this verified CLI. Restore the declared closure at its recorded paths, or produce a separate explicit restoration/revalidation binding; never edit historical paths or digests to make a verifier accept a different artifact. `conditions.py` deliberately does not perform that restoration.

Hardware remains an explicit serialized operator action through `fpga-run-elf`, with the exact selected ELF and wrapper arguments retained. Preserve command ID, input custody, raw UART, run script, observed hardware configuration and decoder identity. Keep source-to-AGFI build qualification false when it is unproved. Successful numerical decoding does not supply hardware provenance, a checkpoint-quality pass or a matched expert denominator. The study's ResNet baseline/optimized pair is an internal matched hardware comparison; the supplied expert artifacts remain unmatched, so neither the 10% target nor the 5% stretch target is an achieved claim. SmolVLA's original model-quality failure remains visible even when every quantized output and carried state agrees exactly.

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

First join expensive intervals to the actual emitted preparation, readout, layout, grouping and synchronization routes. The frozen backend's preference file requests host elementwise fusion, but ordinary whole-model builds take this setting from the selected models configuration. Inspect the actual build receipt rather than assuming that preference was applied. Likewise, disabling optional weight prepacking does not imply repeated weight quantization. The ordinary offload path can already append quantized immutable weights. Settings are candidates for inspection, not evidence of applied or missing rewrites.

Test one mechanism at a time with unchanged arithmetic: immutable-weight prepacking, shared input preparation, layout propagation, supported fusion, projection grouping, or dependence-checked configuration and fence elimination. Preserve a control and test unsafe cases such as aliases, insufficient storage and intervening configuration changes. Merlin must continue to emit primitive commands rather than FSM hardware loops.

`full_build.build(..., source_route=...)` accepts the existing typed `SiblingSourceRoute` or `HeadBatchSourceRoute`. The route binds the normalized original source and explicit borrowed-call premises. Its complete record is part of the build key, so a grouped image cannot reuse an unchanged baseline image. Omitting the argument retains ordinary complete-group routing. This is a mechanism for controlled compiler variants, not automatic permission to combine arbitrary operations. Keep host quantization, floating-point reconstruction and all externally observed results unchanged. The caller must independently check that the actual target implementation completes before returning, retains no caller buffers and releases none of them.

Check the emitted implementation before interpreting fewer calls as an improvement. A plural interface may still schedule each product independently. Inspect the selected backend path, operand preparation, transfers, result readouts and fences. Counted loops must be expanded or explicitly left unknown when reporting command counts. A source-level view does not establish that bufferization avoids a physical copy. Hardware measurements must include any added packing and preserve the baseline timing boundary.

Expert QK kernels can require pretransposed keys, and combined QKV/gate-up kernels have a different calling contract from separate projections. Include equivalent preparation and outputs in both timed regions. Supplied model ELFs and static CCA records alone do not establish equivalent full-model work or a numerical reference.

Use local correctness checks before hardware. Declare a bounded allocation of serialized FireSim submissions in the run plan, including any explicit amendment, with no automatic retries. Keep complete session latency, per-forward intervals, initial preparation, output transport and instrumentation overhead separate. Retain inputs, compiler versions, build recipes, raw observations and failed attempts. Do not modify the archived compiler, sealed bundles or existing AE records.
