# Unchanged quantized FX model regions

The pinned model2MLIR revision `ebb1c600e0d7f3205c801d8f00acb960015b32f2` is available from [reed-nicolas/model2MLIR, branch feat/tinyllama](https://github.com/reed-nicolas/model2MLIR/tree/feat/tinyllama) until its upstream PR merges. The software verifier retains the canonical upstream origin: clone `https://github.com/ucb-bar/model2MLIR.git`, add `https://github.com/reed-nicolas/model2MLIR.git` as a named remote, fetch `feat/tinyllama`, and check out the pinned revision.

## Live capture and the standard application compiler

Use `merlin targetgen native-compile --application` to stage captured IR and
parameters through the installed provider's `application_adapter` and
`compile_region`. See the shared [application workflow](../../../../docs/guides/whole_model_on_accelerator.md).

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

The precision policy selects source arithmetic and binds the capture's source
SHA-256. Search, allocation, proof, and tool limits retain their defaults. The
already-quantized worker's `stage="original"` authenticates the selected native
ExportedProgram; it does not claim float-to-quantized equivalence.

For decoder-0 prefill64 or decode65, freeze the independent boundary receipt's
digest before capture. `adapter_loader.py` reconstructs the graph and parameters
live using the saved activation archive, without loading a saved graph. Decode
uses the authenticated six-input BF16 cache boundary. Use fresh destinations for
each phase and run one native compile at a time:

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

Build the source-pinned snapshot with `merlin targetgen native-build --engine
merlin_native --support atlas --io-memory-bytes 67108864 --cargo-target-dir
"$CARGO_TARGET" --source-revision "$MERLIN_REVISION" --out "$SNAPSHOT"`.
After the standard compile, package emitted C for the example runtime; the
packager replays `native-verify` and checks capture, staging, and input identity.

Ordinary packaging refuses unsatisfied host-linkage obligations. For certified
functional execution-host math, follow the explicit archive and equality-receipt
selection in [the host-rotary example](../../rotary-host/README.md).

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

Use the pinned model's compatible interpreter, with `$ATLAS_ROOT` and
`$ATLAS_ROOT/python` on its import path. Execution publishes
`EXECUTED_UNCOMPARED`; compare with the authenticated same-shape independent
reference using `cache_composition.compare_outputs`. Decode65 also requires
verified independent parent-cache words through `composition.verify_stage_inputs`
and `execute_graph.py --input-overrides VERIFIED_CACHE_INPUTS.npz`. This fixed-input
check does not establish generated parent-cache continuity.

Select qualified execution with `--backend qualified_native --native-qualification
"$MATH_QUALIFICATION" --memory-qualification "$MEMORY_QUALIFICATION" --native-cache
"$NATIVE_CACHE"`. It authenticates both receipts and retains the same default
1,000,000-instruction limit per launch; missing qualification refuses.

## Qualification-only combined loop, cache and residency gate

The optional residency example is a qualification bridge, selected only by
`execute_graph.py --program-residency` with a functional backend. Generated
packages do not include or enable it. The recipe uses `resident_runtime.py`,
`program_residency.py`, `program_residency_bridge.c`, and
`examples/atlas/program_session.h`. The generic Merlin C session owns retention
policy; Atlas supplies physical IMEM primitives. The bridge explicitly prepares
its separate qualification host and exclusively owns that endpoint. Platform
resets must use `atlas_slots_reset_completed` to invalidate its retained handles.
The bridge decodes and executes the complete persistent IMEM from reset PC zero,
including the selected JAL delay slot. This is functional qualification, with hardware and timing gates separate.

The reset-entry trampoline adds two executed instructions per resident launch
beyond the body's proven instruction bound. No static proof currently combines
that overhead with the body bound. Only the runtime instruction cap enforces the
1,000,000-instruction policy for the combined launch. Keep
`--max-instructions 1000000`; a launch that reaches the cap without halting fails.

For the fixed64 gate, first prepare the authenticated whole-model capture,
precision policy, rotary archive/equality receipt, source inputs and snapshot
using [the host-rotary recipe](../../rotary-host/README.md). Use a fresh
`$COMPILED` and `$EXECUTION`. Run from the Merlin repository root, with `$TOOLS`
set as above, `$ROTARY_TOOLS=examples/atlas/rotary-host`, and the compiler/model
interpreters and qualification paths set for those environments:

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
nice -n 10 "$COMPILER_PYTHON" "$TOOLS/prepare_live_execution.py" \
  --compiled "$COMPILED" --capture "$CAPTURE" --staged "$STAGED" \
  --snapshot "$SNAPSHOT" --atlas-root "$ATLAS_ROOT" \
  --boundary-source "$BOUNDARY" \
  --expected-boundary-source-sha256 "$BOUNDARY_SHA256" --inputs "$INPUTS" \
  --functional-validation \
  --execution-host-math-archive "$ROTARY_TOOLS/out/execution-host-math/libcore-math-execution-host.a" \
  --execution-host-math-equivalence "$ROTARY_TOOLS/out/execution-host-math/equivalence-receipt.json" \
  --execution-host-math-rv64-output "$ROTARY_TOOLS/out/rv64-rotary.bin"
nice -n 10 "$MODEL_PYTHON" "$TOOLS/execute_graph.py" \
  --compiled "$COMPILED" --atlas-root "$ATLAS_ROOT" --model-root "$MODEL_ROOT" \
  --backend qualified_native --native-qualification "$ARITHMETIC_QUALIFICATION" \
  --memory-qualification "$MEMORY_QUALIFICATION" --native-cache "$NATIVE_CACHE" \
  --program-residency --max-instructions 1000000 --compact-trace \
  --execution-output "$EXECUTION"
```

Packaging independently replays `native-verify`, including the loop option
recorded in the manifest. The counted C driver still invokes every original
wrapper and binding. The two materialization caches affect compilation;
`--deterministic-receipts` selects reproducible receipt metadata explicitly, and
`--native-cache` holds qualified execution support. None of these options retains
operand writes. Check the manifest's loop selection, compile-status cache records,
and execution residency counters when recording the combined gate.

Authenticate the independent reference's source inputs and parameters as in the
host-rotary gate, then compare every raw BF16 output word. For its fixed64 head
`expected.npz`, set `$REFERENCE_OUTPUTS` and run:

```sh
nice -n 10 "$MODEL_PYTHON" - "$EXECUTION/actual.npz" "$REFERENCE_OUTPUTS" <<'PYTHON'
from hashlib import sha256
import sys
import numpy as np

with np.load(sys.argv[1], allow_pickle=False) as archive:
    actual = archive["output_0"].copy()
with np.load(sys.argv[2], allow_pickle=False) as archive:
    expected = archive["output_0"].copy()
if actual.dtype != np.uint16 or expected.dtype != np.uint16:
    raise SystemExit("expected raw BF16 words")
if actual.shape != expected.shape or actual.shape != (1, 64, 32000):
    raise SystemExit("fixed64 output shape differs")
if not np.array_equal(actual, expected):
    raise SystemExit("independent output differs")
digest = sha256(actual.tobytes()).hexdigest()
if digest != "b81d4a7eb71d12ca9a1a601a1ddecb7d0354be968a1143f235acea7761e3aeba":
    raise SystemExit("fixed64 output digest differs")
print("PASS", digest)
PYTHON
```

## Saved-boundary regression

Install matching Merlin, model2MLIR, and atlas-mlir checkouts in the compiler
environment, then install the standalone `atlas_fp8` adapter package from
`$ATLAS_ROOT/adapter` with `python -m pip install -e "$ATLAS_ROOT/adapter"`.
Put the provider's Python directory and
the source custom-op registration on `PYTHONPATH`. `$BUNDLE/fx` must be a verified
native FP8 export with original unfolded codes, exponents, and ordinary tensors.
Freeze `$CONTRACT_SHA` and `$POLICY_SHA` from its source receipt before capture.

```sh
TOOLS=examples/atlas/whole-model/unchanged_fx
nice -n 10 "$CAPTURE_PYTHON" "$TOOLS/extract_capsules.py" \
  --bundle "$BUNDLE/fx" --output "$REGIONS" --whole-model --threads 1
nice -n 10 "$COMPILER_PYTHON" "$TOOLS/composition.py" \
  --regions "$REGIONS" --compiled "$COMPILED" --execution "$COMPILE_RECEIPT" \
  --atlas-root "$ATLAS_ROOT" --compiler-python "$COMPILER_PYTHON" \
  --model-python "$MODEL_PYTHON" --model-root "$MODEL_ROOT" \
  --adapter-export "$BUNDLE/export" --expected-contract-sha256 "$CONTRACT_SHA" \
  --expected-policy-sha256 "$POLICY_SHA" --compile-only --stop-after head
```

The compiler uses `convert`, `AtlasApplicationAdapter`, external parameter
capture and staging, and `compile_semantic_application`. Completed compilations
are reused only after source, policy, and host-head arithmetic checks. Use fresh
compiled output trees when changing policies. The host BF16 head retains source
words and selects `bf16-exact-product-f32-ltr-rne-bf16-rne-v1`: exact widening,
separate binary32 multiply/add in ascending K order, then one BF16 RNE. Historical
VPU head arithmetic remains refused; no exact torch summation-order claim is made.

Execute with the same arguments, a fresh `--execution` destination,
`--oracle "$ORACLE" --sequential --stop-after head`, and without `--compile-only`.
For a layer-0 check, use `--stop-after layer0` in both commands. `--sequential`
consumes authenticated generated predecessor words; without it, stages use the
independent oracle's authenticated runtime inputs. Both modes compare every
selected output and close generated seam hashes before PASS. Source embedding
gathers can be authenticated through `--initial-inputs`, `--initial-preparation`,
`--source-bundle`, and optional `--initial-tokens` and `--selection-manifest`.
Only the layer-0 embedding may replace its captured fixture.

## Cache phases and source preparation

`raw_capture.py` reconstructs the source from authenticated raw buffers;
`--intake-only` checks source identity without forward execution.
`cache_source.py` defines tensor-only prefill and appended BF16 K/V boundaries.
`capture_cache_source.py` requires the completed raw64 source gate and publishes
authenticated phase ABI sidecars. Pass those to `extract_capsules.py --cache-abi
SIDECAR --cache-capture RECEIPT`; decode65 requires its own query-one,
past-length-64 source capture.

`cache_reference.py` interprets source cache phases independently. Nonempty cache
inputs require a completed parent reference through `--parent-reference` and
`--parent-capsules`. `cache_composition.py` compiles with `--compile-only` and then
executes against `--oracle`; decode additionally requires `--parent-execution`
and `--parent-regions`. Every K/V input comes from authenticated actual parent
outputs. Ordinary composition refuses cached manifests. Source parity, generated
cache continuity, and full-protocol quality remain separate checks.

Run the regional and embedding tests in separate processes to avoid conflicting
source-op registration:

```sh
python -m pytest "$TOOLS"/test_*.py
python -m pytest "$TOOLS/quality"
```

The consolidated `test_cache_composition.py` covers phase roles, parent-cache
publications, driver source and memory selection, oracle lineage, and mutation
refusal using synthetic artifacts and mocked dispatch. Numerical exactness,
hardware behavior, quality, and timing require their own qualification evidence.

## Generated cache through standard applications

For a live four-input prefill boundary, select
`ATLAS_CAPTURE_BOUNDARY=layer0-prefill-cache` before the capture worker. This
returns hidden state, post-RoPE/pre-GQA K, and V as ordinary application outputs.
Use the same authenticated source tensors, original weights, precision policy,
`stage-capture`, and `native-compile --application` commands above. The existing
`layer0` selection retains its original hidden-only prefill ABI and six-input
decode ABI.

Execute the generated prefill application first. Bind its generated cache outputs
to decode with an explicit JSON descriptor:

```json
{
  "schema": "merlin.application_output_bindings.v1",
  "producer_application": "prefill64",
  "producer_execution": "prefill-execution",
  "bindings": [
    {"input_index": 3, "output_index": 1},
    {"input_index": 5, "output_index": 2}
  ]
}
```

Paths are relative to the descriptor. Pass it to `execute_graph.py
--input-bindings bindings.json` when executing the separately compiled decode65
application. This descriptor selects generated cache inputs directly;
`cache_composition.py` continues to use override archives. The example helper
checks producer identity, completed execution, output hashes, and exact
input/output shape and dtype before exposing generated arrays.
Decode receipts record the producer and each consumed output. Independent
references remain comparison-only; they do not supply these cache words.

This uses the existing synchronous, caller-owned application buffers. Keep
producer outputs alive through consumer completion and allocate separate decode
outputs; input/output aliasing is not promised. The example helper validates
each edge's shape and dtype independently. There is no new target ABI or
persistent device-state requirement. Exact comparisons of all three outputs and
backend qualification remain separate gates.

For all decoder layers, select `ATLAS_CAPTURE_BOUNDARY=model-prefill-cache` or
`model-decode-cache`. These reuse the tensor-cache source wrappers with the
original model hierarchy and unchanged parameter words. Their authenticated
source receipt has `kind="model"`, ordered `input_keys`, and exact `tensors`
records. Prefill takes `input_0` token IDs, `input_1` absolute positions, and
`input_2` additive BF16 mask. Decode adds ordered past K/V pairs from `input_3`.
Both return logits at output zero, followed by K/V pairs in layer order.

For 22 layers, bind prefill outputs 1 through 44 to decode inputs 3 through 46:
`input_index = output_index + 2`. Compile each fixed phase through the same
standard application commands and retain separate updated-cache outputs.
Independent fixture caches used during capture do not establish continuity;
execute decode with all 44 generated-output bindings. Compare logits and every
cache tensor, authenticate every producer edge, and check each old cache prefix.
The [independent whole-cache reference](../../rotary-host/reference/README.md)
computes its own prefill and decode state from original weights and tokens,
including v3 rotary preparation.

Whole-model generated-cache numerical evidence uses `qualified_native` only;
layer-0 continuity is checked separately on both backends. The regional
`cache_reference.py` remains for saved-capsule layer-0 and composition regressions;
the whole-cache reference interprets complete live source captures.

### Greedy decoding with decoder layer 0

For an explicit one-decoder-layer experiment, a model-cache source receipt may
set `"decoder_layers": [0]`. This selects the unchanged source embedding,
rotary, decoder layer 0, final normalization, and logits head. It is a truncated
model experiment; its logits are not the full 22-layer model's logits. Omitting
the field preserves the existing whole-model boundary. Existing layer-only
boundaries retain their original hidden-state interfaces.

Compile separate static prefill64, decode65, and decode66 applications. Each
model-cache decode has IDs, positions, mask, K, and V as its five inputs, and
logits, K, and V as outputs. The per-position fixture supplies positions and
mask. The existing `--input-bindings` descriptor can supply both caches and a
greedy ID from the preceding generated publication:

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

Paths resolve relative to the descriptor. Greedy selection widens BF16 words
exactly, selects the last sequence row, chooses the lowest vocabulary index
on a tie, and refuses NaNs. The execution receipt records the selected token
and authenticates its producer. No application ABI or device state changes.

Run `cache_decode_reference.py` first for prefill, then for each decode with
`--parent-reference PREVIOUS_REFERENCE --greedy-parent`. The reference chooses
from its own authenticated expected logits and consumes only its own K/V.
After execution, run `greedy_chain_gate.py --step APP EXEC REF` three times in
one invocation, in prefill64, decode65, decode66 order. The gate requires exact
outputs at all steps, independently matching greedy choices, and complete
cache bindings. Run the qualified backend chain before the plain functional
backend chain with the default execution limits.
