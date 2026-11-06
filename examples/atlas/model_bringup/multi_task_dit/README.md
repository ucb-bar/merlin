# Multi-Task DiT float capture

This example captures LeRobot 0.6.1 Multi-Task DiT through Model2MLIR and torch-mlir, beside the existing Qwen bringup example. DiT and text-projection weights are randomly initialized, CLIP encoders are pretrained, and no accuracy claim is made. It performs no quantization and compiles no Atlas executable.

The full forward includes observation encoding, both CLIP towers, all six DiT blocks at all ten DDIM timesteps, scheduler updates, and the final action slice. The ABI takes already normalized observations and explicit initial noise. Dataset normalization, policy queues, and stochastic noise generation are outside the boundary. Camera count and robot dimensions are workload choices. DDIM and ten inference steps override the scheduler defaults; the architecture is retained.

The source is `lerobot.policies.multi_task_dit.modeling_multi_task_dit.MultiTaskDiTPolicy`. Both encoders and the tokenizer load the local `openai/clip-vit-base-patch16` snapshot at `57c216476eefef5ab752ec549e440a49ae4ae5f3`. The receipt hashes its checkpoint, configuration, and tokenizer files. Offline flags are set before importing the Hugging Face clients.

| Input | Shape | Dtype |
| --- | --- | --- |
| images | 1 x 2 x 2 x 3 x 224 x 224 | float32 |
| state | 1 x 2 x 14 | float32 |
| language tokens | 1 x 77 | int64 |
| language attention mask | 1 x 77 | int64 |
| initial noise | 1 x 32 x 14 | float32 |
| output actions | 1 x 24 x 14 | float32 |

The horizon is 32, observation history is 2, hidden width is 512, and each of six DiT blocks has eight 64-channel heads. Timesteps are `[90, 80, 70, 60, 50, 40, 30, 20, 10, 0]`. Python, NumPy, and PyTorch seeds are 0, deterministic algorithms are required, and model construction uses `torch.manual_seed(0)`. A separate generator seeded with 0 draws uniform images followed by normal noise; state is zero. The instruction is `Pick up the red block and place it in the bowl.`

## Offline reproduction

Python 3.12, torch 2.10.0+cpu, torch-mlir 20260531.828, Model2MLIR [`8893a37`](https://github.com/ucb-bar/model2MLIR/tree/8893a37480b4560f701089e5bf408644573a906f), transformers 5.5.4, diffusers 0.39.0, and xDSL 0.74.0 were used. Install them in a fresh Python 3.12 virtual environment (`venv/` below), with `WHEELS` pointing at a local wheel directory if you install offline: `pip --no-index --find-links "$WHEELS" torch==2.10.0+cpu torch-mlir==20260531.828 transformers==5.5.4 diffusers==0.39.0 xdsl==0.74.0`. Set `HF_HOME` to a Hugging Face cache that holds the pretrained CLIP encoders, and `M2M` to a Model2MLIR checkout at the pin above.

```bash
export TMPDIR="$PWD/tmp"
export HF_HOME="${HF_HOME:?set to a Hugging Face cache with the CLIP encoders}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export MTDIT_TORCHVISION_SCHEMA_COMPAT=1
M2M="${M2M:?set to a Model2MLIR checkout at 8893a37}"
EXAMPLE=merlin/examples/atlas/model_bringup/multi_task_dit
nice -n 10 venv/bin/python "$EXAMPLE/capture.py" \
  --m2m-dir "$M2M" --threads 4 --out "$PWD/out/capture-full"
nice -n 10 venv/bin/python "$EXAMPLE/capture.py" \
  --m2m-dir "$M2M" --threads 4 --region block --out "$PWD/out/capture-block"
nice -n 10 venv/bin/python -m pytest -q -p no:cacheprovider "$EXAMPLE/test_capture.py"
nice -n 10 venv/lib/python3.12/site-packages/torch_mlir/_mlir_libs/torch-mlir-opt \
  --verify-each out/capture-full/model.mlir -o /dev/null
```

Capture output directories must be fresh. Other genuine first-step regions are available with `--region vision`, `text`, or `dit`; their inputs come from hooks on the seeded policy. The block capture is the first real DiT block, with its complete conditioning and rotary inputs.

The cached torchvision 0.26.0 wheel targets torch 2.11 and CUDA. This CPU experiment uses only torchvision's Python tensor transforms. The explicit compatibility flag defines the absent NMS operator schema for import-time fake registration; it supplies no NMS implementation. The model never invokes NMS or the unavailable compiled image extension. This is an environment workaround, recorded in every receipt.

## Capture artifacts and checks

`capture.py` calls the public `m2m.convert(..., backend="torch_mlir", allow_fallback=False, capture_trace=True)` API described by Merlin's capture guide. It supplies the already exported program and requires the original and Model2MLIR-prepared programs to match eager exactly on the saved inputs. Both checks passed with max absolute error 0.0 for the full policy and first block. The test also checks complete denoising-loop scheduling and action slicing with a small independent fake policy.

Pinned Model2MLIR's torch-mlir bridge prints a structural view with resource payloads elided. The example instruments only the bridge's public `export_and_import` call to save the same returned MLIR module before that print, with all payloads present. No model operations or lowering passes are replaced. `model.mlir` is the complete resource-backed text, and `model.mlirbc` is its bytecode. Both persisted forms parse and verify with the pinned torch-mlir bindings; the wheel's LLVM 23 `torch-mlir-opt --verify-each` also verifies the text. No LLVM source build is required.

`structural.mlir` is Model2MLIR's elided inspection view. Its hash is the one bound by `frontend-trace.json`; it must not be executed as a weight-complete model. The separate `receipt.json` binds the unelided text, bytecode, raw exported program, weights, inputs, eager output, inventory, package versions, source scripts, and CLIP checkpoint files. `weights.safetensors` retains the raw export's named state; the executable MLIR uses frozen resource constants directly. This companion output has its own receipt schema and is not advertised as a Merlin staged capture bundle.

Entry tensors and weights are FP32. The lowering widens normalization intermediates to f64 and returns FP32; it also retains integer token/index tensors and boolean masks. There are no FP8, BF16, FP16, quantize, or dequantize operations in the captured MLIR. The detailed dtype and shape counts are in `inventory.json`. This is a floating-point capture with widened temporaries, so it is not an exclusively f32 IR.

The existing Merlin `_m2m_capture_worker.py` was tried separately. Its fixed FX-importer path initially fails on the xDSL 0.74 `Printer.print` method rename. A diagnostic printer adapter lets it materialize a same-conversion bundle, but the parsed bundle metadata and private function calls retain 38 unresolved `aten.select.int` calls in scheduler indexing. Its worker tail incorrectly reports zero because its text census misses generic-form calls. That bundle remains diagnostic; the successful capture here explicitly selects Model2MLIR's torch-mlir backend.

Host numerical MLIR execution is PENDING. A targeted call to the `llvm_m2m_toolchain` probe in `build_tools/scripts/check_repro_env.py`, using `src/merlin/llvmlower/toolchain.py`, finds the selected compiler interpreter but no clang-23. No host executable was built, and no MLIR-versus-eager numerical agreement is claimed. The original and prepared PyTorch exported-program comparisons are separate passing checks. Initial zero adaLN gates make the first block an identity; random initialization and seeded output equality do not establish trained behavior or internal numerical fidelity.

## Atlas family coverage

The reference is atlas-kernels `0333da05f4b172b126f7203ec1077a2cd6f5ede7`, `coverage/mapping.yaml`, SHA-256 `b70b5562ae808b158ba9e94d177d14d407a7767208ccd9c79963ffa94dcb2a2f`. `inventory.py` parses and verifies MLIR, recursively counts operations including scalar regions, inventories operand/result tensor dtype and shape occurrences, records contraction shapes and generic scalar bodies, and labels conservative family candidates. Repeated SSA uses count as repeated type occurrences.

The full graph contains 23,307 recursive operations: 485 `linalg.matmul`, 168 `linalg.batch_matmul`, one `linalg.conv_2d_nchw_fchw`, and 5,034 `linalg.generic`. The 654 named contractions include the full ten-step denoising loop. The standalone first block has seven contractions. `out/operator-inventory.md` and each capture's `inventory.json` retain all operation counts and contraction shapes.

| Captured computation | Existing family relationship | Remaining qualification or gap |
| --- | --- | --- |
| 485 matrix projections | `projection_gemm` geometry | Existing kernels require FP8 E4M3 operands and BF16 accumulation; full K/N panels, padding, biases, and composition remain unqualified here. |
| 168 batched attention contractions | `attention` geometry at head dimension 64 | 24 vision score/value contractions, 24 text score/value contractions, and 120 DiT score/value contractions. Existing GQA, masking, layout, softmax, and FP8/BF16 boundaries need qualification. |
| One 16 x 16 stride-16 vision patch convolution | Projection after patch gather is a potential route | The existing `resnet_patch_gather` family covers 3 x 3 patches, so it does not supply this gather. |
| DiT tanh GELU and gated/residual arithmetic | Portions resemble `smol_fusion`, `epilogue`, and `rne_add` | Fused boundary restrictions apply; general adaLN scale, shift, gates, and FP32 residual arithmetic lack a qualified route. |
| CLIP QuickGELU, layer normalization, and embedding lookup | No complete matching family | LayerNorm is not RMSNorm. Generic sigmoid/QuickGELU and embedding require additional implementations or an explicit host route. |
| DiT rotary and timestep math | Rotary portions resemble `rope` | Source rotary layout, table ownership, sin/cos, timestep embeddings, and scheduler arithmetic remain unqualified. |
| Tensor reshapes, transposes, slices, concatenation, and mask assertions | Layout or host-control obligations | An inventory association does not implement these boundaries or a whole-model execution schedule. |

Under the requested no-quantization policy, zero of these float contractions has a directly supported Atlas execution route in that mapping. Family resemblance does not admit FP32 or f64 to the FP8/BF16 kernels. The capture remains floating point, and no precision-policy transformation, Atlas simulation, performance measurement, or trained accuracy result is claimed.
