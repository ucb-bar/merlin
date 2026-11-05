# TVM–Gemmini comparison

This directory owns model recipes, independent numerical references, input preparation and session verification. The independent pinned TVM fork owns ONNX import, Relax optimization, integer decomposition, primitive scheduling, bounded learned search and static export. Its [integration guide](../../../../third_party/baselines/tvm-gemmini/apps/gemmini/README.md) documents backend APIs, adapter builds and exported-graph execution. TVM is not a Merlin submodule; all commands below run from the Merlin comparison checkout.

| Workload | Recorded development evidence | Remaining qualification |
| --- | --- | --- |
| ResNet50 v1.5 / IMAGENET1K_V2 | V6 passes exact baseline/optimized host gates: development top-one 16/20, matching FP32; frozen final100 87/100 versus FP32 88/100. Selected complete-model images pass matched-Spike repeats/controls. | Device execution of the full stream, canonical ImageNet accuracy, deployment and timing. |
| TinyLlama-1.1B-Chat-v1.0 | Opt-in v3/source-rotary/source-SiLU passes complete 128+32 host sessions in both modes, two repeats/reset: logits, full KV caches and positions are raw-exact. Native recurrence and unchanged development language gates pass. | Candidate adoption, complete static packaging, functional device checks, deployed memory fit and paper performance. |
| SmolVLA / SmolVLM2-500M | Recorded TRAIN observation passes retained prefix, ten denoising/Euler updates and actions in both modes, two fresh VMs and two resets each. Flow states/timesteps/actions are native-exact; three BF16 key-cache words differ within the unchanged tolerance. | Combined public export composition, original-policy quality/units, device, deployed memory and timing. |

These results qualify the recorded host/functional workloads only. Keep TRAIN/calibration, development and held-out inputs separate, freeze recipes before final evaluation and preserve numerical gates. Workload order is ResNet50, TinyLlama, then SmolVLA. Final timing requires the selected FireSim deployment; host and functional Spike supply no timing labels. Store weights, captures, binaries and immutable receipts under the configured `out/` roots, never alongside these sources.

## Target and numerical contract

[target.yaml](target.yaml) is the source/arithmetic/runtime declaration. Its provisional integer sources are Gemmini `6ad65b90`, gemmini-rocc-tests `7c540b3a` and libgemmini `ea8f7ed7`; full commits/digests live in YAML. The parameter header SHA256 is `3758ae967af3a179497660970201093a7fb624be00173990ce33d3f5c38da924`: 16×16 geometry, 256-KiB scratchpad and 64-KiB accumulator. Adapter verification stages exact Git blobs and checks resolved includes; the live floating-point header is incompatible. Hardware revision/generated configuration, usable guest DRAM and deployed completion/coherence remain unconfirmed.

The device boundary is dense row-major signed INT8 operands and INT32 results, positive static shapes and K≤131071. Generic linear lowering flattens left leading dimensions against shared rank-two weights; convolution uses explicit CPU packing/restoration. Bias, scales, activation, attention and state remain where each recipe declares them. TVM schedules C-library primitives in output-stationary mode; WS/FSM/hardware-loop paths are forbidden, including final linked ELFs. The separate whole-matmul reference adapter uses one-array reduction tiles to protect provisional 20-bit PE output and INT32 outer accumulation. The library CPU branch is not a full-accumulator oracle; independent bounded int64 references check integer arithmetic.

The selected development ABI is baremetal RV64GC/lp64d. YAML paths are relative to Merlin; environment references are declarations, not automatic expansions. The 8-GiB map is hypothetical, with no confirmed capacity. Functional export defaults to an explicitly requested 256-MiB map. Tensor admission includes aligned constants, serial scratch/workspace and simultaneously live routed state/outputs; executable, stack, runtime/platform reservations and retained fixtures need separate budgets. Shared-stage export deduplicates constants and refuses hidden runtime allocators. Hashes bind selected inputs/sources/artifacts and loaded libraries, not complete runtime source closure or target fit.

## Pinned host environment

Use absolute `TVM_ROOT` and `TVM_BUILD` paths. The newest TVM pin remains local until the fork is published: reuse a checkout/local clone containing it; a fresh remote clone cannot yet reproduce it. Tested tools are Python 3.10.14, GCC 11.5, CMake 3.26.5 and Ninja 1.13.2. Do not source Chipyard's environment for this build. Override tools with absolute `TVM_HOST_PYTHON`, `TVM_HOST_CMAKE`, `TVM_HOST_CC`, `TVM_HOST_CXX`; `TVM_BUILD_JOBS` defaults to 4.

For a new checkout, apply mandatory case-insensitive name exclusions before materializing files; preserve an existing matching checkout. Reject any supplied path containing either prohibited name before invoking tools. Optional dependencies must not be initialized recursively.

```bash
merlin_root="$PWD"
setup_dir="$merlin_root/examples/gemmini/comparisons/tvm"
export TVM_ROOT=/absolute/path/to/tvm-gemmini TVM_BUILD=/absolute/path/to/tvm-host-build
tvm_commit="$(awk '$1=="source:" {selected=1; next} selected && $1=="commit:" {print $2; exit}' "$setup_dir/target.yaml")"
git clone --filter=blob:none --no-checkout https://github.com/reed-nicolas/tvm.git "$TVM_ROOT"
git -C "$TVM_ROOT" sparse-checkout set --no-cone --stdin <<'PATTERNS'
/*
!*[hH][aA][mM][mM][eE][rR]*
!*[vV][lL][sS][iI]*
PATTERNS
git -C "$TVM_ROOT" checkout --detach "$tvm_commit"
test "$(git -C "$TVM_ROOT" rev-parse HEAD)" = "$tvm_commit" || { echo 'TVM source pin mismatch' >&2; exit 1; }
```

Initialize only `dmlc-core`, `dlpack` and `rang` at their recorded TVM gitlinks. For fresh dependencies, clone without checkout and use the same exclusions:

```bash
git -C "$TVM_ROOT" submodule init -- 3rdparty/dmlc-core 3rdparty/dlpack 3rdparty/rang
while read -r dependency repository; do
  revision="$(git -C "$TVM_ROOT" rev-parse "HEAD:3rdparty/$dependency")"
  git clone --no-checkout "$repository" "$TVM_ROOT/3rdparty/$dependency"
  git -C "$TVM_ROOT/3rdparty/$dependency" sparse-checkout set --no-cone --stdin <<'PATTERNS'
/*
!*[hH][aA][mM][mM][eE][rR]*
!*[vV][lL][sS][iI]*
PATTERNS
  git -C "$TVM_ROOT/3rdparty/$dependency" checkout --detach "$revision"
done <<'DEPENDENCIES'
dmlc-core https://github.com/dmlc/dmlc-core.git
dlpack https://github.com/dmlc/dlpack.git
rang https://github.com/agauniyal/rang.git
DEPENDENCIES
git -C "$TVM_ROOT" submodule status -- 3rdparty/dmlc-core 3rdparty/dlpack 3rdparty/rang
```

Each status line must start with a space; `-`, `+` or `U` requires repair. Prepare an isolated environment and helper-derived output root:

```bash
host_python="${TVM_HOST_PYTHON:-$(command -v python3.10)}"
host_cmake="${TVM_HOST_CMAKE:-$(command -v cmake)}"
host_cc="${TVM_HOST_CC:-$(command -v gcc)}"
host_cxx="${TVM_HOST_CXX:-$(command -v g++)}"
: "${host_python:?Set TVM_HOST_PYTHON}" "${host_cmake:?Set TVM_HOST_CMAKE}"
: "${host_cc:?Set TVM_HOST_CC}" "${host_cxx:?Set TVM_HOST_CXX}"
build_jobs="${TVM_BUILD_JOBS:-4}"
host_tools_path="$(dirname "$host_cmake"):$(dirname "$host_cc"):$(dirname "$host_cxx"):/usr/bin:/bin"
unset LD_LIBRARY_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH
unset CMAKE_PREFIX_PATH PYTHONPATH PYTHONHOME CC CXX CFLAGS CXXFLAGS LDFLAGS
export PYTHONDONTWRITEBYTECODE=1
build_root="$(PYTHONPATH="$merlin_root/src" "$host_python" -c 'from merlin.common.paths import build_dir; print(build_dir() / "baselines" / "tvm-gemmini")')"
mkdir -p "$build_root"/{cache/pip,tmp,logs}
export PIP_CACHE_DIR="$build_root/cache/pip" TMPDIR="$build_root/tmp"
"$host_python" -m venv "$build_root/venv"
source "$build_root/venv/bin/activate"
python -m pip install -r "$setup_dir/host-requirements.txt"
host_ninja="$build_root/venv/bin/ninja"
export PATH="$build_root/venv/bin:$host_tools_path"
```

Prefer an existing complete LLVM 18 SDK with X86/RISCV, matching headers and shared libraries; set `llvm_config` and `llvm_lib_dir` to it. If unavailable, this is the tested LLVM 18.1.8 source-build fallback (commit `3b5b5c1ec4a3095ab096dd780e84d7ab81f3d7ff`):

```bash
llvm_source="$build_root/llvm-project"
llvm_build="$build_root/llvm-build"
git clone --filter=blob:none --no-checkout --depth 1 --branch llvmorg-18.1.8 https://github.com/llvm/llvm-project.git "$llvm_source"
git -C "$llvm_source" sparse-checkout set --no-cone --stdin <<'PATTERNS'
/*
!/*/
/llvm/
/cmake/
/third-party/
!/llvm/test/
!/llvm/unittests/
!*[hH][aA][mM][mM][eE][rR]*
!*[vV][lL][sS][iI]*
PATTERNS
git -C "$llvm_source" checkout --detach llvmorg-18.1.8
"$host_cmake" -S "$llvm_source/llvm" -B "$llvm_build" -G Ninja -DCMAKE_MAKE_PROGRAM="$host_ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$host_cc" -DCMAKE_CXX_COMPILER="$host_cxx" \
  '-DLLVM_TARGETS_TO_BUILD=X86;RISCV' -DLLVM_ENABLE_PROJECTS= -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DLLVM_DYLIB_COMPONENTS=all -DBUILD_SHARED_LIBS=OFF \
  -DLLVM_INCLUDE_TOOLS=ON -DLLVM_BUILD_TOOLS=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_ZLIB=OFF
"$host_cmake" --build "$llvm_build" --target LLVM llvm-config --parallel "$build_jobs"
"$host_cmake" -E create_symlink libLLVM.so.18.1 "$llvm_build/lib/libLLVM-18.so"
llvm_config="$llvm_build/bin/llvm-config"
llvm_lib_dir="$llvm_build/lib"
```

Keep source/generated include directories when consuming this build tree. Configure TVM using the checked-in host cache and LLVM RPATH:

```bash
"$llvm_config" --version --targets-built --shared-mode --link-shared
"$host_cmake" -S "$TVM_ROOT" -B "$TVM_BUILD" -G Ninja -C "$setup_dir/host-config.cmake" -DCMAKE_MAKE_PROGRAM="$host_ninja" -DCMAKE_C_COMPILER="$host_cc" -DCMAKE_CXX_COMPILER="$host_cxx" -DUSE_LLVM="$llvm_config --link-shared" -DCMAKE_BUILD_RPATH="$llvm_lib_dir"
"$host_cmake" --build "$TVM_BUILD" --parallel "$build_jobs"
export PYTHONPATH="$TVM_ROOT/python" TVM_LIBRARY_PATH="$TVM_BUILD" TVM_FFI=ctypes
export TEST_DATA_ROOT_PATH="$build_root/cache/tvm-test-data"
python "$TVM_ROOT/apps/gemmini/verify_host.py" --output "$build_root/host-smoke.json"
python -m pip install -r "$setup_dir/frontend-requirements.txt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TVM_NUM_THREADS=1
```

`host-config.cmake` opts into `USE_HOST_ONLY_AUTO_COPY_GUARD`, OFF upstream by default. This local guard preserves unannotated IR and rejects automatic-copy markers because the restricted checkout lacks `LowerAutoCopy`; it implements no optimization and must not coexist with the full implementation. Record the enabled patch and actual C++ build identity; later Python source changes do not relabel an older library. In a new shell, restore the paths, activate the same venv and reapply these exports. Every numerical run requires a fresh output directory.

## Frontend, graph and search checks

Frontend pins are Torch 2.10.0 (tested `+cu128`, executing on CPU), ONNX 1.17.0, protobuf 5.29.5 and NumPy 1.26.4. Legacy export is explicit `dynamo=False`; ONNX 1.17 retains the importer API. Synthetic opset17/18 fixtures compare Torch, ONNX ReferenceEvaluator and Relax LLVM at `rtol=1e-4, atol=1e-5`; importer integer dtype/shape/values are exact. Neither suite qualifies models/devices.

```bash
python "$TVM_ROOT/apps/gemmini/verify_onnx.py" --output-dir "$build_root/frontend/new-run"
python "$TVM_ROOT/apps/gemmini/verify_onnx.py" --suite importer --output-dir "$build_root/frontend/importer-new-run"
python -m pip install 'xgboost==1.7.6' 'pytest==8.3.5'
```

Focused learned-search regressions live in TVM's `tests/python/relax/test_backend_contrib_gemmini_tuning.py`; use a test launcher that excludes workspace ancestor imports and unrelated discovery. TVM's `prepare_gemmini_graph` folds constants/fuses CPU work while preserving the device boundary. `BoundedGemminiSearch`/`GemminiModelSearch` use `PerStoreFeature`/`XGBModel`, finite reproducible budgets and candidate-bound repeated measurement requests. Invalid measurements preserve search state; synthetic labels check machinery only. Real timing requires one compiler/adapter/platform/protocol domain and independent semantic/device gates. The [graph simulator recipe](../../../../third_party/baselines/tvm-gemmini/apps/gemmini/README.md#complete-exported-graphs-in-gemmini-spike) supports baseline/optimized modes; its generated graph checks also admit `--search-seed 7` and `--workload conv-residual --conv-kernel 3` (kernels 1/7 are checked). These are bounded fixtures, not performance winners.

## ResNet50 artifacts and frozen evaluation

[verify_resnet.py](verify_resnet.py) defaults to a complete random-weight torchvision ResNet50 v1.5 structural diagnostic: 53 convolutions, one classifier, `[1,3,224,224]`→`[1,1000]`, opset17 and unchanged FP32 `rtol=atol=1e-4`. Supply model2MLIR loader revision `7915e23475c6db446a3c404847b11e8bc72c8a27`; frontend requirements include torchvision 0.25.0/Pillow 12.1.0. `--graph-mode both` checks baseline and explicit `zero`/`default_build` optimization. `--batchnorm-folding` checks a separate evaluation-only folded copy before export.

Supplied mode requires `--checkpoint`, `--inputs`, `--input-source` and `--preprocessing` together. Checkpoints are plain finite CPU state dicts loaded with `weights_only=True`; exact keys/dtypes/shapes are required. NPZ `images` must be finite float32 `[N,3,224,224]` or `[N,1,3,224,224]`, N>0; no conversion/preprocessing occurs. Optional labels JSON has exactly `input_sha256`, 1000 unique ordered `class_ids`, and one `{id, label}` sample per image (integer 0–999, no Boolean). Reports preserve every occurrence and rank ties by class index. Labels/dataset/training truth remain declarations.

The v6 integer recipe in [resnet_quantized.py](resnet_quantized.py) uses signed symmetric INT8, power-of-two per-channel weights/per-tensor activations, checked INT32 contractions/bias, ties-away requantization, accumulator residuals and integer pooling. Scales coarsen for overflow instead of saturating bias. TRAIN-only bounded MSE calibration records ranges/bounds, propagates sole-consumer ReLU ranges and rounds scaled spatial sums once. All learned convolutions/classifier use Gemmini; packing, requantization, residuals, pooling and logit dequantization are CPU work. Evaluation never selects coefficients.

Set `model2mlir_source` to the selected local checkout. Obtain the local checkpoint/archive and prepare disjoint TRAIN80/development20/final100 streams:

```bash
model_cache="$(PYTHONPATH="$merlin_root/src:$PYTHONPATH" python -c 'from merlin.common.paths import artifacts_dir; print(artifacts_dir() / "cache")')" || exit 1
test -n "$model_cache" || exit 1
mkdir -p "$model_cache/resnet50"
curl --fail --location --retry 3 --continue-at - --output "$model_cache/resnet50/resnet50-11ad3fa6.pth" https://download.pytorch.org/models/resnet50-11ad3fa6.pth
curl --fail --location --retry 3 --continue-at - --output "$model_cache/resnet50/imagenette2-320.tgz" https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
input_run="$model_cache/resnet50/train80-final100-new"
python "$setup_dir/prepare_resnet_inputs.py" --archive "$model_cache/resnet50/imagenette2-320.tgz" \
  --calibration-per-class 8 --evaluation-per-class 2 --final-per-class 10 --selection-seed resnet50-v2-train-final-v1 --output-dir "$input_run"
export RESNET_CHECKPOINT="$model_cache/resnet50/resnet50-11ad3fa6.pth" RESNET_TRAIN_INPUTS="$input_run/calibration.npz"
export FROZEN_QUANTIZATION="$input_run/frozen-quantization.json"
PYTHONPATH="$setup_dir:$merlin_root/src:$PYTHONPATH" python - <<'PYTHON'
import json, os
from pathlib import Path
import numpy as np
import torch, torchvision
import verify_resnet, resnet_quantized

torch.set_num_threads(2)
model = torchvision.models.resnet50(weights=None).cpu().eval()
verify_resnet.load_checkpoint(model, os.environ["RESNET_CHECKPOINT"], torch)
calibration, identity = verify_resnet.load_images(os.environ["RESNET_TRAIN_INPUTS"], np)
plan = resnet_quantized.calibrate_resnet50(model, calibration,
    calibration_ids=[f"{identity['sha256']}:{i}" for i in range(len(calibration))],
    checkpoint_declaration="local supplied state_dict", activation_estimator="mse", residual_policy="accumulator")
with Path(os.environ["FROZEN_QUANTIZATION"]).open("x") as stream:
    json.dump(plan.manifest(), stream, indent=2, sort_keys=True)
PYTHON
python "$setup_dir/verify_resnet.py" --model2mlir-root "$model2mlir_source" --tvm-source "$TVM_ROOT" \
  --checkpoint "$RESNET_CHECKPOINT" --inputs "$input_run/evaluation.npz" --labels "$input_run/labels.json" \
  --input-source 'Imagenette320 val; ordered samples in preparation.json' --preprocessing 'torchvision V2 RGB/resize232/crop224/normalization' \
  --quantized --calibration-inputs "$RESNET_TRAIN_INPUTS" --calibration-source 'Imagenette320 train; ordered samples in preparation.json' \
  --activation-estimator mse --residual-policy accumulator --frozen-quantization "$FROZEN_QUANTIZATION" \
  --graph-mode both --export-baremetal --max-top1-drop 0.02 --memory-limit-bytes 268435456 --output-dir "$build_root/resnet-v6-development-new"
```

The [preparer](prepare_resnet_inputs.py) reads without archive extraction, applies installed V2 transforms and refuses unsafe entries/duplicate/overlapping content. It binds archive/member/transform bytes without authenticating the official archive or training provenance. For final evaluation, replace inputs/labels with `final.npz`/`final_labels.json`, add `--evaluation-role held-out` and use a fresh output directory; recomputed TRAIN coefficients must exactly match the frozen manifest. The unchanged 0.02 top-one rate-loss gate compares the original FP32 model and is a development proposal, not paper approval. All images require exact raw INT32 and dequantized FP32 agreement. `--device-sample-indices 0,18` (development) or predeclared `0,90` (final) selects fixtures after full host gates; the device stream remains a subset. Exported `reference.npz`/`reference.json` bind independent integer results; execute `baremetal_baseline`/`baremetal_optimized` with the TVM graph-simulator guide. Export alone executes no device workload.

## Stateful verifier

[verify_session.py](verify_session.py) consumes `get_model_and_inputs` plus model2MLIR's `ExternalRuntimeSession` protocol (optional `get_session_spec` for a single program). Static signatures track changing state shapes and are reused across repeats. Torch/TVM state is independently routed; every output/state update and full reset must pass. Factory arguments are a JSON keyword object; repeat `--artifact` for additional local files. Factory loading disables Python network/HuggingFace downloads, without providing a process sandbox or implicit weight conversion.

```bash
python "$setup_dir/verify_session.py" --model2mlir-root "$model2mlir_source" --tvm-source "$TVM_ROOT" --tvm-build "$TVM_BUILD" \
  --factory /absolute/path/to/loader.py --factory-arguments /absolute/path/to/arguments.json \
  --artifact /absolute/path/to/checkpoint --output-dir "$build_root/session-new-run"
```

Legacy export defaults to opset17; `--onnx-exporter dynamo` defaults to opset18, with explicit `--opset` (BF16 convolution requires 22). Optional modern dependencies are `ml_dtypes==0.5.4`, `onnxscript==0.5.6`, `onnx_ir==0.1.12`. Export copies stages to protect native cache/reset state, preserves BF16 raw uint16 payloads and typed FP64 scalars, validates external-weight paths/ranges and rechecks selected files after execution. `get_verification_artifacts` binds factory helpers/checkpoint/tokenizer/input sources; `get_onnx_export_options` may supply reviewed translations without overriding verification controls. Failed comparisons retain metrics and bounded 64-MiB raw snapshots; input-state bytes are diagnostic accounting, not peak memory.

Opt-in `--bf16-fused-export` retains public linear/attention/single-output LayerNorm/SiLU boundaries, BF16 interfaces and FP32 internal math while disabling ONNX optimization. It checks masks/causal scale/static normalized-affine shapes and rejects dropout, grouped-query expansion and incompatible dtypes. It avoids an extra BF16 SiLU sigmoid store; it does not guarantee all native CPU reduction/rounding behavior or relax source gates.

Only after host gates pass, `--export-baremetal --memory-limit-bytes BYTES` derives shared static stages and `baremetal/session/{route.json,fixture.npz,provenance.json}` from original Torch outputs with exact reset bytes. TVM's device verifier consumes `--session-route`, `--fixture`, `--fixture-provenance`; host allclose does not imply exact-byte device acceptance. Nonfinite tensors/implicit conversions/hidden allocators are refused. Bind CPU math closure explicitly with repeated `--cpu-runtime-library /absolute/path/to/library.a`; no library is selected by default. Bounded INT8/math/BF16/FP64 proofs do not qualify full models; a nontrivial sinf fixture retains a one-ULP device failure. ELF/stack/platform and retained-output/fixture budgets remain separate.

## TinyLlama host recipe

[tinyllama_session.py](tinyllama_session.py) requires the full [checkpoint/tokenizer](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/tree/fe8a4ea1ffedaf415f4da2f062534de366a451e6), all 22 layers and local UTF-8 corpus. Explicit FP32 conversion of official BF16 weights is a compiler/development diagnostic, not original-precision quality qualification. Use Transformers 5.4.0's cache API/modern exporter in a path isolated from SmolVLA:

```bash
python -m pip install --target "$build_root/tinyllama-python-v5" 'transformers==5.4.0' 'huggingface_hub==1.5.0' \
  'tokenizers==0.22.1' 'safetensors==0.6.2' 'numpy==1.26.4' 'onnx==1.17.0' 'ml_dtypes==0.5.4' 'onnxscript==0.5.6' 'onnx_ir==0.1.12'
export PYTHONPATH="$build_root/tinyllama-python-v5:$merlin_root/src:$TVM_ROOT/python"
```

The default factory is an FP32 8+3 frontend diagnostic, `quantized: false`, with native arithmetic/rotary/SiLU. Explicit W8A8 (`quantized: true`) covers all 155 learned projections using per-output-channel INT8 weights, dynamic token INT8 activations, exact INT32 contractions and FP32 dequantization; other math/state is CPU FP32. Activation scale defaults to `power_of_two`; `maxabs` retains the older diagnostic. Complete-K integer references use actual operands.

Development language requires 128 prefill+32 teacher-forced decode, two repeats/reset, eight distinct attributed 128-token articles, NLL increase≤0.20 nats and top-one agreement≥0.80 versus the unchanged explicitly FP32-converted checkpoint. Prepare the [pinned WikiText validation parquet](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/validation-00000-of-00001.parquet) locally (`pyarrow` required), setting the five environment variables below to local snapshots/new outputs:

```bash
PYTHONPATH="$setup_dir:$merlin_root/src:$PYTHONPATH" python - <<'PYTHON'
import json, os
from pathlib import Path
from tinyllama_integer import prepare_wikitext_corpus
arguments = prepare_wikitext_corpus(os.environ["WIKITEXT_PARQUET"], os.environ["TINYLLAMA_CHECKPOINT"], os.environ["CORPUS_OUTPUT"])
arguments.update(checkpoint_dir=os.environ["TINYLLAMA_CHECKPOINT"], model2mlir_root=os.environ["MODEL2MLIR_ROOT"], precision="float32", cpu_threads=2,
    arithmetic_policy="fixed_tree_fp32_v3_softmax_reduce", rotary_policy="source_table_v1", silu_export_policy="torch_avx512_sleef_v1")
Path(os.environ["FACTORY_ARGUMENTS"]).write_text(json.dumps(arguments, indent=2) + "\n")
PYTHON
for graph_mode in baseline optimized; do
  python "$setup_dir/verify_session.py" --model2mlir-root "$model2mlir_source" --tvm-source "$TVM_ROOT" --tvm-build "$TVM_BUILD" \
    --factory "$setup_dir/tinyllama_session.py" --factory-arguments "$FACTORY_ARGUMENTS" --graph-mode "$graph_mode" \
    --onnx-exporter dynamo --repeats 2 --max-signatures 2 --output-dir "$build_root/tinyllama-v3-$graph_mode-new-run" || exit 1
done
```

Preparation verifies pinned parquet bytes and selects the first eight eligible articles without model feedback. `workload: synthetic_feasibility` separately permits 8+3/128+32 without a quality corpus. Native defaults and failed historical receipts remain intact; the candidate is unadopted. Recorded v3 quality is NLL increase 0.01698/top-one agreement 90.059% across 1,016 predictions.

| Explicit option | Arithmetic/scope limit |
| --- | --- |
| `arithmetic_policy: fixed_tree_fp32_v1` | Separately rounded FP32 products/adjacent balanced sums for QK and RMSNorm; no initial zero, FMA/reassociation/hidden FP64. Version-1 `merlin.fixed_order` helpers require positive static FP32/power-of-two K≤4096, finite nonoverflowing operands and visible `4*K` scratch. |
| `fixed_tree_fp32_v2_pv` / `fixed_tree_fp32_v3_softmax_reduce` | Add ordered PV, then active-mask max/subtract/backend-exp/ordered-sum/division softmax. Common 160-token bound pads positive zero to 256 leaves (may change signed zero); reject incompatible masks/cache before mutation. Backend exponentials remain distinct. |
| `rotary_policy: source_table_v1` | Configuration-derived native words for positions 0–159, checked full/cached equality; 80 KiB constants. Export requires admitted source masks/positions. |
| `silu_export_policy: torch_avx512_sleef_v1` | Binds pinned Torch/library/headers, AVX512 dispatch and two threads; proved 5,632-column 128-token/one-token calls. Source exp/eight FMAs/scaling/direct division, no Torch/SLEEF runtime/scratch. Dimension guards do not prove arbitrary thread-partition equivalence; LLVM only, C unsupported, target execution pending. |

All candidate CPU work, padding, scratch/reshape/output allocations count toward eventual memory/timing. Native/quality references and `rtol=atol=1e-4` gates remain unchanged.

## SmolVLA host recipe and source profiles

[smolvla_session.py](smolvla_session.py) preserves original LeRobot precision, including BF16 prefix/cache, for the full [policy](https://huggingface.co/lerobot/smolvla_base/tree/c83c3163b8ca9b7e67c509fffd9121e66cb96205) and supplied [backbone](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/tree/dc831d5df58eb65ae2fc25b50b7f5a896edec069). The neural boundary is preprocessed observations→ten denoising steps→`[1,50,6]` actions, excluding robot execution/postprocessing. Training-time backbone/camera identity is unverified. Checkpoint namespaced statistics do not map to plain action/state keys; observed transforms are identity, with no silent namespace selection or physical-unit qualification.

Use a separate Python 3.12 environment with original LeRobot/Transformers versions:

```bash
python3.12 -m venv "$build_root/smolvla-venv"
"$build_root/smolvla-venv/bin/python" -m pip install 'lerobot[smolvla]==0.5.1' 'torch==2.10.0' 'transformers==4.57.1' \
  'onnx==1.19.1' 'onnxscript==0.7.2' 'numpy==1.26.4' 'ml_dtypes==0.5.4' 'pyarrow==21.0.0' 'pandas==2.3.3' 'av==15.1.0'
PYTHONPATH="$setup_dir:$merlin_root/src:$TVM_ROOT/python" "$build_root/smolvla-venv/bin/python" "$setup_dir/prepare_smolvla_observation.py" \
  --dataset-root "$SO100_DATASET" --checkpoint-dir "$SMOLVLA_CHECKPOINT" --backbone-dir "$SMOLVLM_BACKBONE" \
  --frame-index 0 --split train --noise-seed 0 --video-backend pyav --output-dir "$OBSERVATION_OUTPUT"
```

The [preparer](prepare_smolvla_observation.py) uses the [pinned SO100 dataset](https://huggingface.co/datasets/lerobot/svla_so100_pickplace/tree/3d6d687a25cdf1565cdf24550814f72d999a861d), saved processor and explicit top→camera1/wrist→camera2 mapping. It binds schema/parquet/video/decoder identities, never invents a camera/refits normalization and separates splits by whole episodes. Recorded replay supplies no closed-loop quality result. Create a factory arguments JSON with local paths:

```json
{
  "checkpoint_dir": "/absolute/path/to/smolvla-policy",
  "backbone_dir": "/absolute/path/to/smolvlm-backbone",
  "model2mlir_root": "/absolute/path/to/model2MLIR",
  "fixture_kind": "dataset", "input_npz": "/absolute/path/to/observation.npz",
  "input_manifest": "/absolute/path/to/observation.json",
  "input_source": "pinned revision/episode/frame and preprocessing",
  "selected_cameras": ["observation.images.camera1", "observation.images.camera2"],
  "seed": 0, "cpu_threads": 1, "precision": "original"
}
```

```bash
for graph_mode in baseline optimized; do
  PYTHONPATH="$merlin_root/src:$TVM_ROOT/python" TVM_FFI=ctypes "$build_root/smolvla-venv/bin/python" "$setup_dir/verify_session.py" \
    --model2mlir-root "$model2mlir_source" --tvm-source "$TVM_ROOT" --tvm-build "$TVM_BUILD" \
    --factory "$setup_dir/smolvla_session.py" --factory-arguments /absolute/path/to/smolvla-arguments.json \
    --graph-mode "$graph_mode" --onnx-exporter dynamo --opset 22 --repeats 2 --max-signatures 3 --output-dir "$build_root/smolvla-$graph_mode-new-run" || exit 1
done
```

Each graph mode requires a fresh process/output directory; an integer run also needs a distinct `integer_diagnostic_dir` per invocation. Defaults remain original/native; synthetic staging uses `fixture_kind: synthetic`. Explicit W8A8 requires `quantized: true` and a fresh `integer_diagnostic_dir` beneath Merlin artifacts. A target copy replaces learned linears with INT8/INT32, FP32 dequantization/bias and one original BF16/FP32 interface cast; other work keeps original precision. Eager/reset, sampled complete-K checks and same-state velocity/independent trajectory checks do not establish original-policy quality.

The following opt-in factory options require that integer recipe, original precision and one native thread. Each binds pinned native source/library/headers/dispatch, checks the admitted original graph/layout and preserves native references/gates; unmatched geometry retains ordinary export or is refused. Fused profiles require explicit Dynamo/`--bf16-fused-export`. These are bounded CPU source-route proofs, not general native/RISC-V equivalence. Finite admitted intermediates, round-to-nearest and gradual underflow are proof assumptions, not arbitrary runtime-value scans.

| Factory key and non-native value | Admitted source scope and visible storage |
| --- | --- |
| `source_mean_policy`: `torch_cpu_mean720_v1`, `torch_cpu_mean720_960_v1` | FP32 keepdim `[1,50,720]`, plus `[1,177,960]`; pinned AVX2 cascade/direct division. 260-byte reusable scratch plus outputs; square/epsilon/rsqrt/affine/casts unchanged. |
| `source_softmax_policy`: `torch_cpu_softmax177_v1`, `torch_cpu_softmax177_227_v1`, `torch_cpu_softmax177_227_prefix_v1` | FP32 `[1,15,50,177/227]`, plus `[1,15,177,177]`; AVX512 lane/exp/reciprocal order, 836-byte scratch. |
| `source_layernorm_policy`: `torch_cpu_layernorm768_v1` | BF16 `[1,1024,768]`, original BF16 affine/epsilon `1e-6`; AVX2 Welford/cascade/FMA, 320-byte scratch. |
| `source_conv_policy`: `torch_cpu_conv3x16_v1` | BF16 `[1,3,512,512]`, `[768,3,16,16]` weights/bias, stride16/padding0/dilation1/groups1; oneDNN software-BF16 order, 128-byte scratch. Reject unproved BF16/AMX dispatch and ISA/fpmath overrides. |
| `source_sgemm_policy`: `torch_cpu_sgemm177_v1`, `torch_cpu_attention177_227_v1`, `torch_cpu_attention177_227_prefix_v1` | Manual QK/PV177, add QK/PV227, then prefix PV177. FP32 increasing-K or software-BF16 paired FMA/odd-tail order; 4-byte scratch, bound physical layouts/highest precision. Alpha1/beta0 is inferred for ordinary MatMul; original internal variant/buffers were not observed. |
| `source_scalar_policy`: `torch_cpu_language_scale960_v1` | One BF16 `[1,48,960]` embedding site/original INT64 tokens: native FP32 sqrt960 scalar/opmath and one BF16 cast on a copied ExportedProgram. |
| `source_rotary_policy`: `torch_cpu_denoise_rotary64_table_v1` | Boolean prefix177/suffix50 position DAG, D64/wavelength10000, all reachable positions; source-derived FP32 sine/cosine tables occupy 58,368 bytes. Prefix rotary/FP64 time trigonometry unchanged; not portable MKL. |
| `source_flash_policy`: `torch_cpu_flash_bf16_vision1024_v1` | Static strided vision Q/K/V/positive-zero mask DAG; query256/key512 tiling, BF16 exp stores/FP32 online normalization, 985,472-byte serial scratch and pinned host `expf`. No Torch/MKL runtime; newlib/device unqualified. |
| `source_silu_policy`: `torch_cpu_silu_fp32_v1` | Contiguous FP32 complete 32-element vector groups, source exp/direct division; refuse offsets/strides/symbolic shapes/scalar tails. LLVM `fmaf`/`floorf` bind libm bytes; recorded process addresses unavailable. |

Recorded TRAIN qualification used authenticated retained ExportedPrograms/compiled prefix artifacts and freshly compiled flow/action stages (Comparisons `fddec47f`, TVM `6b376cc889`). Actual compiled outputs feed subsequent stages; references are comparison-only. Prefix masks/all value-cache words are native-exact; only three key-cache words differ within `rtol=atol=1e-4` (maximum `3.8147e-6`). Byte identity across modes/VMs/resets and source/input/artifact/IEEE checks pass. First vision-block proofs select a prequalified LayerNorm slot; they do not establish arbitrary combined public StageCompiler composition, whose overlap refusal remains. The generic command above is a fresh qualification attempt, not reproduction of the retained composition. Full public packaging, original-policy quality, device/memory fit and timing remain pending. CPU casts, helpers, tables and all live storage count in eventual budgets.

[diagnose_smolvla_backends.py](diagnose_smolvla_backends.py) records original native flash/MATH trajectories, processor/statistics inventory and review-only numerical proposals without changing the active `1e-4` gate. Use `--help` for dataset/manifest attribution and `--velocity-replay`; output must be fresh beneath configured artifacts. Proposal inputs are TRAIN/development only, report `NO_ACCEPTANCE` and do not authenticate declared fixture identity or select action/robot limits. Ordered-arithmetic explorations remain unadopted local proofs; complete-policy native equivalence is not established by them. Preserve immutable failed receipts under `out/`.

## Target, simulator and verification entry points

```bash
python "$setup_dir/check_target.py" --config "$setup_dir/target.yaml"
python "$TVM_ROOT/apps/gemmini/verify_matmul.py" --gemmini-repo "$GEMMINI_ROOT" --build-root "$build_root" \
  --cc "$RISCV_BIN/riscv64-unknown-elf-gcc" --objdump "$RISCV_BIN/riscv64-unknown-elf-objdump"
python "$setup_dir/build_simulator.py" --libgemmini-repo "$LIBGEMMINI_REPO" --spike-prefix "$SPIKE_PREFIX" \
  --cxx "$HOST_CXX" --runtime-lib-dir "$SPIKE_SUPPORT/lib" --runtime-bin-dir "$SPIKE_SUPPORT/bin" \
  --smoke-elf "$CPU_PROOF/model.elf" --build-root "$build_root" --output-dir "$build_root/simulator-new"
python "$TVM_ROOT/apps/gemmini/verify_matmul_runtime.py" --adapter-receipt "$ADAPTER_BUILD/receipt.json" --simulator-receipt "$build_root/simulator-new/receipt.json" \
  --riscv-gcc "$RISCV_BIN/riscv64-unknown-elf-gcc" --spike "$SPIKE_PREFIX/bin/spike" --dtc "$SPIKE_SUPPORT/bin/dtc" \
  --spike-library-dir "$SPIKE_SUPPORT/lib" --plugin "$build_root/simulator-new/libgemmini.so" --output-dir "$build_root/matmul-runtime-new"
```

Supply existing matching reference repositories/tool installations and a passing CPU ELF. Adapter/simulator stages extract exact integer Git blobs without modifying references, check headers/tools/dependencies and bind immutable identities. Plugin loading alone executes no Gemmini arithmetic. Runtime fixtures check extremes/tails/strides, guards, repeated-call recovery and intentional failures; audit the final ELF for no FSM. Link audited `matmul.o`, never diagnostic `matmul.raw.o`. Final deployment/coherence/capacity and timing remain separate.

`check_target.py --memory-plan /path/to/reservations.json` accepts disjoint coarse `regions` with `name`, integer `address`, integer `size_bytes`; account for code/static data, weights, workspace, stack and runtime without double-counting arena children. Gaps are not measured free memory; omitted allocations stay unknown. The checker rejects incompatible arithmetic/WS/loop policy/ranges/confirmed-capacity requests, but does not verify hardware execution.

Use each CLI's `--help` and the TVM integration guide for focused cases. Comparison regression entry points are `merlin/tests/gemmini/test_tvm_{target,resnet_artifacts,resnet_quantized,session}.py`, `test_tinyllama_session.py` and `test_smolvla_session.py`; invoke explicit files with the selected model environment and `PYTHONPATH="$merlin_root/src:$PYTHONPATH"`. Optional real integration fixtures use `TVM_SESSION_MODERN_ONNX=1 TVM_SESSION_MODEL2MLIR="$model2mlir_source"` or `TVM_TINYLLAMA_INTEGER_ONNX=1`; Smol staging uses `MODEL2MLIR_ROOT`. Run the source/library-bound TVM regressions appropriate to a backend change. Static inspection does not substitute for numerical execution, device qualification or timing.
