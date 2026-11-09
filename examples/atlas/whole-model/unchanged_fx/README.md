# Unchanged quantized FX model regions

The pinned model2MLIR revision `e445587543bf9b84e6ee1371414f2202f2533d9c` is available from [reed-nicolas/model2MLIR, branch feat/tinyllama](https://github.com/reed-nicolas/model2MLIR/tree/feat/tinyllama) until its upstream PR merges. The software verifier retains the canonical upstream origin: clone `https://github.com/ucb-bar/model2MLIR.git`, add `https://github.com/reed-nicolas/model2MLIR.git` as a named remote, fetch `feat/tinyllama`, and check out the pinned revision.

## Live capture and the standard application compiler

The existing application front door is `merlin targetgen native-compile --application`.
It stages the capture's own IR and parameters, selects the installed provider's
`application_adapter`, and calls Merlin's semantic region planner and the provider's
`compile_region`. Saved FX boundary extraction is not required by that planner.
`merlin compile --model-build` is the separate host-baseline ELF workflow; its CLI
does not expose device placement. See the shared
[application workflow](../../../../docs/guides/whole_model_on_accelerator.md).

After a worker capture with `--materialize-bundle --already-quantized --quant-adapter
atlas_fp8`, the existing compiler commands are:

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

The precision policy explicitly selects the original arithmetic and binds the
capture's source SHA-256. Search, allocation, proof, and tool limits retain their
defaults. The worker now carries finite BF16 scalar/splat words exactly, including
signed zero. Standard-path decoder-0 prefill64 compilation, artifact verification, and
independent exactness pass for the fixed saved activation words: all 131,072 BF16
output words match the independent reference through both plain functional and
qualified execution. Decode65 also matches all 35,328 BF16 hidden and updated
K/V words through both plain and qualified execution, using authenticated
independent parent cache words. These are fixed-input decoder checks; whole-model end-to-end
compilation and execution remain separate.

For this already-quantized route, `stage="original"` deliberately identifies the
selected authenticated native ExportedProgram as the original numeric source.
The worker enables that snapshot only with an explicit external adapter and an
already-quantized exported graph. There is no floating original graph to attest,
and this receipt makes no float-to-quantized equivalence claim.

For a decoder-0 capture, `adapter_loader.py` selects
`ATLAS_CAPTURE_BOUNDARY=layer0`, with `ATLAS_INPUTS_SAFETENSORS` pointing at the
saved-boundary activation tensor archive and `ATLAS_BOUNDARY_SOURCE_JSON` plus its
previously frozen `ATLAS_EXPECTED_BOUNDARY_SOURCE_SHA256`. Do not derive this
expected digest from that same file during the capture run. The graph and
parameters are reconstructed and captured live; the activation words are saved
boundary inputs. This establishes a decoder-level route, not complete end-to-end
model capture and execution. The latter requires the whole-model capture.
An authenticated six-input cache boundary uses the existing tensor-cache append
semantics and returns hidden state, updated K, and updated V. Neither path loads
a saved graph. Loader and staging tests do not establish generated exactness.

To capture and stage either decoder-0 phase, select the independently authenticated
boundary receipt and its saved activation archive. Set the frozen boundary digest
before selecting capture files. Use separate fresh destinations for prefill64 and
decode65; run one native compile at a time.

```sh
TOOLS=examples/atlas/whole-model/unchanged_fx
export ATLAS_CAPTURE_BOUNDARY=layer0
export ATLAS_BOUNDARY_SOURCE_JSON="$BOUNDARY_SOURCE"
export ATLAS_EXPECTED_BOUNDARY_SOURCE_SHA256="$FROZEN_BOUNDARY_SOURCE_SHA256"
export ATLAS_INPUTS_SAFETENSORS="$BOUNDARY_TENSORS"
export ATLAS_EXPORT_DIR="$BUNDLE/export" ATLAS_ARCHITECTURE="$ARCHITECTURE"
export ATLAS_EXPECTED_CONTRACT_SHA256="$CONTRACT_SHA"
export ATLAS_EXPECTED_POLICY_SHA256="$POLICY_SHA"
nice -n 10 "$COMPILER_PYTHON" -m merlin.targetgen._m2m_capture_worker \
  --loader "$TOOLS/adapter_loader.py" --dtype bf16 --out "$LIVE_CAPTURE" \
  --materialize-bundle --already-quantized --quant-adapter atlas_fp8 \
  --contract "$BUNDLE/export/manifest.json" --policy "$BUNDLE/export/policy.json" \
  --expected-contract-sha256 "$CONTRACT_SHA" --expected-policy-sha256 "$POLICY_SHA"
nice -n 10 "$COMPILER_PYTHON" "$TOOLS/live_precision.py" \
  --capture "$LIVE_CAPTURE" --kind layer0 --out "$PRECISION_POLICY"
```

Use the stage and native-compile commands above, with `--out
"$COMPILED/application"`. Build a source-pinned snapshot through `merlin targetgen
native-build --engine merlin_native --support atlas --io-memory-bytes 67108864
--cargo-target-dir "$CARGO_TARGET" --source-revision "$MERLIN_REVISION" --out
"$SNAPSHOT"`. The standalone policy selector reuses the saved-boundary example's
existing arithmetic policy construction; it never exports or compiles a graph.

After successful standard compilation, package only its emitted C for the
existing example runtime. This command replays `native-verify`, checks the live
capture/staged identity and saved activation words, and builds the host shim
outside the authenticated native application inventory:

```sh
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

The pinned model requires a compatible model interpreter, independently of the
compiler interpreter. Add both `$ATLAS_ROOT` and `$ATLAS_ROOT/python` to its import
path. Execution publishes `EXECUTED_UNCOMPARED`; authenticate the same-shape
independent reference and use `cache_composition.compare_outputs` to establish
exact word equality. A decode run driven by saved cache input words does not
establish generated parent-cache continuity. For decode65 comparison, authenticate
the independent oracle's `inputs.json`, parent reference/expected-array hashes,
and raw `runtime_inputs.npz` words. Use the existing
`composition.verify_stage_inputs` check and pass its preserved cache words with
`execute_graph.py --input-overrides VERIFIED_CACHE_INPUTS.npz`. Captured framework
cache words cannot substitute for the independent parent cache.

The existing qualified execution implementation may be selected explicitly with
`--backend qualified_native --native-qualification "$MATH_QUALIFICATION"
--memory-qualification "$MEMORY_QUALIFICATION" --native-cache "$NATIVE_CACHE"`.
It verifies pinned arithmetic and memory receipts before execution, preserves
program/control instructions and the same per-launch instruction limit, and
records the changed arithmetic implementation. The prefill64 plain and qualified
runs agree exactly on output words, launch counts, and executed instructions.

The source LM head now uses a closed host region admitted by the Atlas provider
and emitted through Merlin's existing semantic host-linear path. Original BF16
operands and their widened source graph types remain intact. The selected policy,
`bf16-exact-product-f32-ltr-rne-bf16-rne-v1`, widens the operands exactly, uses a
positive-zero f32 seed, and performs separately rounded binary32 multiplication
and addition in ascending K order, without contraction or fast math. It rounds
once to BF16 at the output; extreme products retain source binary32 underflow
semantics. This is host arithmetic, with no device acceleration claim.

The saved-head application independently matches all 2,048,000 raw BF16 logits
against reference v2 at an authenticated generated final-norm boundary, using
67,108,864 I/O bytes and 536,870,912 application bytes. This checks the head stage,
not a new upstream whole-model execution. Full saved-head artifact replay still
refuses the existing source-intake and arena-plan guards; no guard is disabled.
The tiny host-only application separately passes compile, full API replay, and C
execution at default limits. The declared ascending-K order does not claim
bit-exact agreement with torch's summation order. Reference v1 and
whole-model/cache results under the obsolete VPU head arithmetic are historical;
the old policy is removed from shared emission and refused by the provider.

Use fresh compiled output trees when switching head policy. Reuse checks require
the current policy in precision metadata and matching recorded host numeric
semantics; missing, historical, or inconsistent declarations refuse reuse. Keep
old artifacts and their receipts as historical evidence.

An optional live head-only capture still refuses `capture contains no native
Atlas calls`: the external adapter's no-native-call exception currently requires
a saved export. Keep that guard. Passing saved-head application execution does
not establish live head-only capture or complete live whole-model execution.

Rotary preparation is admitted as host source arithmetic through Merlin's
selected unary-math route, linked to the pinned CORE-MATH archive in
[the host-rotary example](../../rotary-host/README.md). Its BF16 outputs match
Torch over positions 0..2047; binary32 framework equivalence and positions beyond
2047 are unqualified. Chipyard newlib was rejected because one sine result
changes a BF16 word at position 1897. The head and rotary are admitted, while
complete live embedding/all 22 layers/norm/head through the standard front door
remains the next end-to-end step. A separate integration effort is addressing
the next non-rotary RMSNorm broadcast multiplication refusal. Generated cache
continuity and full-protocol quality remain separate gates.

The saved-boundary tools below are regeneration and regression aids. Their
historical qualification does not qualify the live application path.

## Saved-boundary regression details

These scripts extract, compile, and execute the 22 decoder layers, final normalization, and LM head of the fixed-shape TinyLlama export while preserving original parameter bytes. Source embedding lookup, masks, and rotary inputs are captured from the source graph for the initial boundary. This is a functional bringup path, and timing, hardware qualification, independent exactness, and quality require separate evidence.

Run `extract_capsules.py --bundle BUNDLE --output REGIONS --whole-model --threads 2` in an environment with the original `atlas_fx_ops` registration and PyTorch export support. The extractor verifies bundle hashes, replays the source fixture, copies each region's authored FX nodes, checks lifted tensors against the source export, and records source boundary names and ordered producer links. The region manifest records any authored alias between final normalization and the head. Each new regional capture archives inherited capture identities and an authenticated source-node map in `source.json.parent_capture`, then starts a fresh local capture namespace.

Run `whole_model.py --regions REGIONS --compiled COMPILED --execution EXECUTION --atlas-root ATLAS_ROOT --compiler-python COMPILER_PYTHON --model-python MODEL_PYTHON --model-root MODEL_ROOT --stop-after norm` with model2MLIR, Merlin, atlas-mlir's Python binding, and the custom op registration on the compiler import path. The model interpreter must be the unchanged pinned npu_model revision authenticated by `model_runtime.py`. A reference checkout can be added with `--reference-root`. Use `--stop-after head` to include the source-order generated host BF16 head policy; select an independent reference using the same declared arithmetic.

Compilation uses existing `convert`, `AtlasApplicationAdapter`, external parameter capture and staging, and `compile_semantic_application` APIs. The driver reuses completed compilations only after checking their saved source model hashes, selected policies, and recorded host-head numeric semantics. Use fresh output trees for a policy change. Every execution receives immutable source boundary fixtures plus explicit overrides from earlier generated outputs. The driver authenticates those outputs, passes their raw words, and checks the receiving stage's input hashes. A missing generated predecessor is an error. Later framework hidden-state fixtures are never used to replace generated activations.

The source LM head has an ordinary BF16 weight matrix. It explicitly selects `SOURCE_POLICY` from `merlin.compile.c_bf16_linear`, preserving original words without FP8 recoding and using the declared ascending-K binary32 multiply/add order described above. Structurally closed widened source regions are admitted without changing source tensor types. Other unqualified floating immutable contractions are refused before ordinary FP8 packing. Independent comparison must use reference v2; historical VPU-policy head exactness does not qualify this arithmetic. Complete upstream execution and quality still require their own gates.

Run `python test_whole_model.py` for focused continuity, corruption, and missing-predecessor checks. Execution receipts deliberately use `EXECUTED_UNCOMPARED`, and report pending independent comparison and quality. The runner supports `--compile-only` for intake checks without generated execution.

`composition.py` provides bounded concurrent compilation and composition checks. First run it with the same path and interpreter arguments plus `--compile-only --workers 2` to compile the selected stages. Then run it in a fresh execution directory with `--oracle ORACLE` to consume `runtime_inputs.npz`, `inputs.json`, `expected.npz`, and authenticated reference receipts from an independent chained original-FX oracle. The generated stages may execute concurrently. A final pass checks every generated output against that oracle and every generated producer output hash against the exact successor input consumed. This establishes pure-function composition for the fixed input, and the receipt explicitly disclaims temporally sequential execution. It rejects an undeclared later fixture boundary, mismatched oracle origin, modified raw input, modified expected output, or unclosed seam. `test_composition.py` covers raw carrier views, wrong producer attribution, wrong outputs, and seam corruption.

Add `--sequential --workers 1` for temporal execution with independent comparison. Each successor loads its linked inputs from the completed generated predecessor's `actual.npz`, verifies those bytes against the execution publication, and passes them unchanged to generated execution. This mode never opens oracle `runtime_inputs.npz` or `inputs.json`; immutable initial preparation boundaries are authenticated against source tensor hashes, and an optional initial embedding uses the independent raw source gather. Oracle `expected.npz` remains an output comparison target. The final seam check requires predecessor completion before consumer start. A RUNNING receipt does not certify completion; only a final PASS establishes exactness for all requested regions. Separate stage processes establish temporal dataflow, with shared-device persistent state and cached decode requiring separate gates.

To overlap an independent compiler producer with execution, use `composition.py --wait-for-compiled --workers 6 --model-worker-limit-file LIMIT.json`. Consumers wait for a complete, parsed `compiled.json` and never compile into shared destinations. The limit file is a JSON integer in `1..workers`, initially `2`; replace it atomically when changing the number of permitted model subprocesses. Waiting threads consume no model slot, and cancellation releases acquired slots. The receipt records the maximum observed model concurrency. A lower limit applies to new starts after already running calls complete.

For producer recovery, `--archive-incomplete` preserves every unfinished selected stage under the fresh execution directory and records the reversible moves. Completed compilations remain available for source and policy validation. Producers take an exclusive directory lock, so recovery cannot race another producer using this driver. Preserve and finish any producer started by older tooling before enabling recovery; consumers may continue reading completed publications.

The drivers default to the original `functional` validation backend. Selecting `--backend qualified_native` requires separate `--native-qualification` and `--memory-qualification` receipts, with an optional explicit `--native-cache` directory. The selected backend and both receipt hashes are recorded, and missing qualification never causes a silent fallback. This choice must follow the arithmetic and memory adapter qualification gates; it changes the validation implementation, and it does not change source parameters, generated programs, or placement.

`verify_model_transport.py --compiled COMPILED --capsules REGIONS --oracle ORACLE --execution EXECUTION --out AUDIT` audits all 22 decoder stages using the independent retained original-FX arrays and the inherited `verify_fp8_transport.py` checks. It authenticates every FP8 operand tile, every accumulator read/continuation-seed frame, final-K accumulator values, generated quantizers, and external parameter bytes. Intermediate-K arithmetic, final normalization, the head, hardware behavior, and quality are separate gates. The batch wrapper binds source, oracle, compiled metadata, quantizer library, program images and plans, parameter files, and referenced runtime payload files before and after validation. Cached results require the identical complete binding; legacy or incomplete receipts are revalidated and preserved under content-named prior-receipt files. Run `python test_verify_model_transport.py` for lightweight cache and traversal checks without model execution.

The example compiler and ordinary and cached composition drivers default to the generic allocator's 1,000 ms wall budget. An explicit `--allocation-timeout-ms` override is recorded in compilation and driver receipts, and an explicit allocator `timeout_ms` takes precedence. UNKNOWN remains a failure.

Select an RMS reduction domain with `--reduction-policy`. The default is `bf16-adjacent-fp32-tree-rne-seeded-tile-sum-ltr-chop-v1`. The explicit alternative `bf16-nonnegative-normal-adjacent-fp32-tree-rne-seeded-tile-sum-ltr-chop-v1` admits positive zero and nonnegative normal BF16 inputs without changing the reduction arithmetic. The Atlas provider owns both domain definitions and validates the selected policy through `AtlasApplicationAdapter`; the generic `compile_semantic_application` API is unchanged. `precision.json` and `compiled.json` record the exact policy identifier, and drivers refuse to reuse a compilation with a different reduction selection. Numerical exactness and quality still require separate gates.

Compilation uses generic `SearchLimits()` defaults, including 24 candidate nodes and a 60-second search wall limit. Target ABI and pointwise allocation inherit the active generic wall budget and deterministic allocation-order policy. The drivers default to I/O and application capacities of 67,108,864 and 536,870,912 bytes. A 4,325,376-byte I/O window selects the K32 tile route for Q, O, up, gate, and down, producing about 42,000 host-driven launches per layer.

For a declared text window, pass `--initial-inputs`, `--initial-preparation`, `--source-bundle`, and, when available, `--initial-tokens` and `--selection-manifest`. The canonical quality authenticator loads only the source graph and original BF16 embedding table, verifies the token and source hashes, and independently gathers the raw words. It checks that the embedding is the original direct parameter lookup and that every other captured preparation boundary is independent of token values. Only layer0 `input_0` may change. The concurrent composition mode also binds that input to the oracle's preparation and token hashes; sequential mode authenticates it directly against the unchanged source gather; mask, rotary, and later hidden-state fixtures retain their existing strict checks. No source graph arithmetic is executed during authentication.

`cache_source.py` supplies tensor-only source boundaries around an already authenticated, reference-patched Llama model. `TensorPrefillLogits` accepts token IDs, absolute positions, and an additive BF16 mask, and initializes K/V internally without zero-sized tensor inputs. `TensorCacheLogits` accepts token IDs, absolute positions, an additive BF16 mask, and one ordered BF16 K/V pair per layer; it returns source logits and appended K/V after RoPE and before GQA expansion. Call `validate_prefill_phase` or `validate_phase` before capture or execution to admit the corresponding unpadded, contiguous append contract. `phase_positions_and_mask` prepares the fixed positions and mask. The explicit `empty_cache` boundary remains useful for source-only compatibility checks; generated prefill uses the three-input interface. Source attention dynamically quantizes the complete BF16 prefix on every call. Tiny tests compare prefill and two decode steps against the source framework's DynamicCache, check cache words and unchanged state, and replay tensor-only and native uint8 attention exports. These are synthetic source semantics gates. Validate checkpoint parity, extracted cache links, independent arithmetic, and actual generated prefill/decode through their separate receipts.

`raw_capture.py --bundle BUNDLE --export-dir EXPORT --architecture CONFIG --seqlen LENGTH --out OUTPUT` reconstructs already quantized source modules from authenticated raw buffers and unchanged ordinary parameters. `--intake-only` checks architecture, policy, source state, and byte identity without forward execution. The original-length capture compares both original fixtures and requires native capture, reload verification, and post-capture state authentication; a different length additionally requires its completed `--source-shape-gate`. Conversion entry points are guarded, and unsupported folds, residual policies, aliases, and weight layouts refuse. Capture and intake receipts establish source preparation; generated exactness and quality require separate runs.

`capture_cache_source.py` requires that completed raw64 gate and checks two source chains with a four-token prefill followed by two one-token decode phases. Use `--parity-only` for the first source check; native phase capture uses a fresh output directory. Each phase publishes an authenticated cache ABI sidecar for `extract_capsules.py --cache-abi SIDECAR --cache-capture RECEIPT`. Wait for the complete `SOURCE_CACHE_CAPTURED` root receipt before extraction. Extraction preserves source operators and records temporal cache inputs separately from source fixtures. Ordinary whole-model runners refuse cached manifests until a phase runner authenticates their generated parent-cache state. These tools prepare source artifacts and carry no generated decode or full-window quality claim.

`cache_composition.py` compiles a captured phase with `--compile-only`, then executes it sequentially against a complete independent `--oracle`. Decode requires both `--parent-execution` and `--parent-regions`; every K/V input comes from the authenticated parent's actual output publication, while an initial prefill uses its authenticated three-input source ABI with an internally empty cache. The driver checks source roles, phase lengths, all output publications, the oracle's ordered stage bindings, consumed words, and final artifact hashes before PASS. `test_cache_composition.py` and its integration suite use synthetic artifacts and mocked dispatch; full-model cached exactness is a separate numerical gate.

`cache_reference.py` independently interprets source cache phases with the explicitly selected frozen original-FX helper and its qualified numerical policies. It requires the helper's completed binding and original arithmetic qualification receipts. Nonempty cache inputs come from a completed parent independent reference via `--parent-reference` and `--parent-capsules`; generated executions and captured framework cache values cannot supply those inputs. All hidden, logits, and updated cache words are published for separate generated comparison. `--qualify-only` authenticates numerical helpers without running a model graph.

Functional execution accepts a positive `--max-instructions` limit per native invocation. The default remains 1,000,000. Select a larger bounded limit only after checking the actual compiled programs; the limit does not establish numerical exactness, transport correctness, or target timing. Both ordinary and cached composition drivers record the selected budget.

Selected ordinary-operation input domains remain strict. Valid source tensors can produce an intermediate outside an admitted code interval. Preserve that refusal separately from successful cases, and qualify an explicit alternative policy before rerunning. A refusal does not establish a numerical mismatch, and a successful subset does not establish full-protocol quality.

For external-adapter intake, add `--adapter-export EXPORT --expected-contract-sha256 CONTRACT_SHA --expected-policy-sha256 POLICY_SHA` to `composition.py --compile-only` or to `compile_capsule.py`. Take both expected values from the frozen source receipt before selecting files for capture. Put the adapter package root and the source custom-op module on `PYTHONPATH`. The compiler authenticates the capsule's selected saved graph and original tensors through the installed `atlas_fp8` entry, passes the same external configuration to `convert` with `quantization_preapplied=True`, and runs the Atlas provider normalizer before bundle publication and staging. The original trace represents this already-native graph. The selected configuration, manifest, exact source hash, and native call census remain in the capture receipt; the staged IR carries both `prov.quantization_manifest_sha256` and module `prov.quantization`. Exact source joins consume temporary opaque dispatch, argument-binding, and value-alias metadata, then emit the authored typed semantics and source provenance. Norm and head capsules have no native custom calls. Final normalization may lower to device programs; the BF16 head uses its explicit host policy and original parameter bytes.

After capture, staging, signature, and every original tensor payload have been authenticated, the Atlas provider constructs a compiler-owned input view. It verifies the staged SHA-256 and exact parser/serializer roundtrip, checks the module manifest digest and scheme against the selected manifest, and consumes exactly `prov.quantization_manifest_sha256` and module `prov.quantization` into `compiler-input-view.json`. The archived `staged/program.mlir` retains both attributes. Semantic translation and parameter binding receive the owned program text and its actual hash, so runtime launch identities describe exactly the compiler input. The receipt records both consumed attribute names and values, the full staged hash, compiler hash, byte counts, and exact byte delta. No generated image, launch file, or comparison result is rewritten.

For live model reconstruction, use `adapter_loader.py` with the generic capture worker and `--already-quantized --quant-adapter atlas_fp8 --contract EXPORT/manifest.json --policy EXPORT/policy.json`, plus the two expected SHA-256 flags. Set `ATLAS_EXPORT_DIR`, `ATLAS_ARCHITECTURE`, and `ATLAS_INPUTS_SAFETENSORS`, and supply the same frozen hashes through `ATLAS_EXPECTED_CONTRACT_SHA256` and `ATLAS_EXPECTED_POLICY_SHA256`. The loader constructs the architecture on the meta device, installs authenticated raw codes, exponents, and ordinary tensors, and exports inside the original native-site context without loading a saved source program.

Validation provenance: historical adapter validation uses a regenerated bundle at source revision `a6a1aa4fef1106fc12f0ea4afc035310a2d4b7ff` that passes `verify_bundle`; it does not use the original artifact files. That validation saved the live loader's native export, extracted 24 boundaries, and ran `compile_capsule.py --adapter-export`. The live decoder route above instead consumes the worker's own IR through standard staging and application compilation, with live graph and parameter capture driven by saved-boundary activation words. Complete whole-model capture, compilation, and execution remain a separate end-to-end gate. Retain trace and compilation provenance differences even when executable, launch, and parameter bytes match.

Layer-0 compilation is load-sensitive under the unchanged 30-second per-tool wall inherited from the Atlas binding. Prefill64 sizing has taken about 23 seconds on a quiet host. A timeout is a failed compile; retain the failure receipt and retry under lower load without raising the limit.

## Saved-boundary layer0 regression and regeneration

Use matching `feat/tinyllama` branches of atlas-mlir, Merlin, and [model2MLIR](https://github.com/reed-nicolas/model2MLIR/tree/feat/tinyllama). Install the standalone `atlas_fp8` package from the atlas-mlir checkout with `python -m pip install -e "$ATLAS_ROOT/adapter"`. The matching model2MLIR head is `e445587543bf9b84e6ee1371414f2202f2533d9c`, also pinned in `merlin/contract/software_pins.yaml`. Retain the selected atlas-mlir and Merlin branch SHAs beside each run receipt; documentation commits can advance these draft branch heads. These are draft branches; distinguish live decoder captures driven by saved activation words from the saved-graph regression route below and from complete whole-model capture.

Regenerate from Loren's [atlas-torchao exporter](https://github.com/ucb-ee290c/atlas-torchao/blob/a6a1aa4fef1106fc12f0ea4afc035310a2d4b7ff/docs/algos/howto-model-to-fx-graph.md) at `a6a1aa4fef1106fc12f0ea4afc035310a2d4b7ff`, with the pinned checkpoint and calibration data. Run these commands in that checkout with a fresh `$BUNDLE`:

```sh
nice -n 10 "$CAPTURE_PYTHON" src/eval.py --model "$CHECKPOINT" \
  --seqlen 2048 --calib-n 16 --calib-seed 1234 --modes ref_conv \
  --export-only --algo atlas_v1_awq_tile_static --device cpu \
  --export-dir "$BUNDLE/export" --out "$BUNDLE/export_cli.json"
nice -n 10 "$CAPTURE_PYTHON" src/fx_export.py --model "$CHECKPOINT" \
  --export-dir "$BUNDLE/export" --out-dir "$BUNDLE/fx" --device cpu --seqlen 64 --native-fp8
nice -n 10 "$CAPTURE_PYTHON" scripts/tile_flow/verify_bundle.py \
  "$BUNDLE/export" "$BUNDLE/fx" "$BUNDLE/verify_bundle.json"
"$CAPTURE_PYTHON" scripts/tile_flow/make_bundle_manifest.py "$BUNDLE"
```

Keep the original unfolded parameter bytes. In the compiler environment, install matching local checkouts, including exactly one adapter entry point:

```sh
"$COMPILER_PYTHON" -m pip install -e "$M2M_ROOT" -e "$MERLIN_ROOT" -e "$ADAPTER_ROOT"
export PYTHONPATH="$ATLAS_ROOT/python:$SOURCE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
CONTRACT_SHA=$(sha256sum "$BUNDLE/export/manifest.json" | cut -d ' ' -f 1)
POLICY_SHA=$(sha256sum "$BUNDLE/export/policy.json" | cut -d ' ' -f 1)
```

Here `$SOURCE_ROOT` is the exporter checkout. Save the two hashes in the frozen source receipt before compilation and compare them on every reuse. From Merlin, extract prefill64 regions from the verified FX directory:

```sh
TOOLS=examples/atlas/whole-model/unchanged_fx
nice -n 10 "$CAPTURE_PYTHON" "$TOOLS/extract_capsules.py" \
  --bundle "$BUNDLE/fx" --output "$PREFILL_REGIONS" --whole-model --threads 1
```

Decode65 uses a separately authenticated query-one, past-length-64 native capture and its extracted `$DECODE_REGIONS`; a fixed64 no-cache graph cannot stand in for it. The cache capture and extraction entry points are described above. Set `$COMPILED64` and `$COMPILED65` to fresh destinations. These commands reproduce the layer-0 gate settings: prefill64 does not merge adjacent regions, and decode65 does.

```sh
compile_layer0() {
  nice -n 10 "$COMPILER_PYTHON" "$TOOLS/compile_capsule.py" \
    --capsule "$1/layer0" --output "$2/layer0" --external-parameters \
    --adapter-export "$BUNDLE/export" --expected-contract-sha256 "$CONTRACT_SHA" \
    --expected-policy-sha256 "$POLICY_SHA" --atlas-root "$ATLAS_ROOT" \
    --allocation-timeout-ms 1000 --io-memory-bytes 67108864 \
    --application-memory-bytes 536870912 \
    --reduction-policy bf16-adjacent-fp32-tree-rne-seeded-tile-sum-ltr-chop-v1 \
    --softmax-mask-policy bf16-minfinite-zero-rowspan80-nonempty-v1 "${@:3}"
}
compile_layer0 "$PREFILL_REGIONS" "$COMPILED64"
compile_layer0 "$DECODE_REGIONS" "$COMPILED65" --merge-adjacent-device-regions
```

Use Bash for this function. Build atlas-mlir in Release mode before compilation, and expose its tools through the target's configured tool paths. `compile_capsule.py` calls model2MLIR, Merlin, and the Atlas binding; its selected target invokes atlas-mlir tools. Search, allocation, proof, instruction, and tool-wall limits retain their defaults.

For generated execution against an independent same-shape oracle, run `composition.py --regions "$PREFILL_REGIONS" --compiled "$COMPILED64" --execution "$EXECUTION64" --oracle "$ORACLE64" --wait-for-compiled --sequential --workers 1 --stop-after layer0 --backend functional --max-instructions 1000000`, with the same compiler paths, adapter hashes, memory capacities, and arithmetic policies above, plus `--model-root "$MODEL_ROOT" --model-python "$MODEL_PYTHON" --compiler-python "$COMPILER_PYTHON"`. The model interpreter and checkout must match `model_runtime.py`'s pin. A qualified native backend instead requires both arithmetic and memory qualification receipts. For a whole-model run, compile with `composition.py --compile-only --stop-after head` and execute with `--stop-after head`; the embedding gather and original BF16 LM head execute on the host. Actual cache chains use `cache_composition.py`, an independent phase oracle, and authenticated parent execution and region publications. A direct `execute_graph.py` decode run consumes saved oracle input words and needs a separate exact comparison; it does not establish generated parent-cache continuity.

Run `python -m pytest merlin/tests` for Merlin and `python -m pytest tests` from each model2MLIR and adapter checkout. Run the regional and quality tests in separate processes so schema-only quality registration cannot conflict with native source-op registration:

```sh
python -m pytest "$TOOLS"/test_*.py
python -m pytest "$TOOLS/quality"
ctest --test-dir "$ATLAS_BUILD" --output-on-failure
```

Test results and functional instruction counts do not establish target timing or full-protocol quality. Retain exact checked-out revisions, compiled artifacts, launch schedules, and separate numerical comparison receipts.
