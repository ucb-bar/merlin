# Ownership and attribution

The TVM/Gemmini baseline on this branch is built on the bringup by
Nicolas Reed ([@reed-nicolas](https://github.com/reed-nicolas)), done
2026-10-07 to 2026-10-13 on two branches:

| Source | Tip | What it contributed |
| --- | --- | --- |
| Merlin [PR #53](https://github.com/ucb-bar/merlin/pull/53) (`comparisons/tvm-gemmini`) | [`a4745c66a`](https://github.com/ucb-bar/merlin/commit/a4745c66aaead245a58979f0d1d85cf244aee117) | Host setup, target pin, model sessions and verification for ResNet50, TinyLlama and SmolVLA |
| TVM [`reed-nicolas/tvm:gemmini/bringup`](https://github.com/reed-nicolas/tvm/tree/gemmini/bringup) | [`6b376cc88`](https://github.com/reed-nicolas/tvm/commit/6b376cc8895a83387ec38cc06ef8f7b09827b071) | Gemmini Relax BYOC backend, bare-metal runtime and verifiers, BF16/ONNX/arith fixes |

The curated branches took these files over rather than merging the commits, so
the per-file history below records where they came from. "Unchanged" means the
file on the curated branch is byte-identical to the bringup; "adapted" means it
started from the bringup and was edited afterwards.

## Merlin (this directory and `merlin/tests/gemmini`)

All 24 files from PR #53 are present here.

- Unchanged (13): `HOST_SETUP.md`, `bf16_export.py`, `build_simulator.py`,
  `check_target.py`, `diagnose_smolvla_backends.py`, `frontend-requirements.txt`,
  `host-config.cmake`, `host-requirements.txt`, `prepare_smolvla_observation.py`,
  `tinyllama_integer.py`, `test_tvm_resnet_artifacts.py`,
  `test_tvm_resnet_quantized.py`, `test_tvm_target.py`
- Adapted (11): `README.md`, `prepare_resnet_inputs.py`, `resnet_quantized.py`,
  `smolvla_session.py`, `target.yaml`, `tinyllama_session.py`, `verify_resnet.py`,
  `verify_session.py`, `test_smolvla_session.py`, `test_tinyllama_session.py`,
  `test_tvm_session.py`

## TVM ([`grammy117/tvm:gemmini/direct-intrinsics`](https://github.com/grammy117/tvm/tree/gemmini/direct-intrinsics))

All 31 files the bringup added or changed are present on the curated TVM branch.

- Unchanged (17): `apps/gemmini/{auto_copy_guard.cc, verify_baremetal_cpu.py,
  verify_exported_graph.py, verify_host.py, verify_matmul_runtime.py,
  verify_onnx.py}`, `python/tvm/relax/backend/contrib/gemmini_schedule.py`,
  `tests/python/relax/test_backend_contrib_gemmini{,_conv}.py`, and the upstream
  fixes in `CMakeLists.txt`, `LICENSE`, `src/arith/rewrite_simplify.cc`,
  `src/tir/transforms/unsupported_dtype_legalize.cc` with their tests
  (`test_arith_rewrite_simplify.py`, `test_frontend_onnx.py`,
  `test_tir_transform_{bf16,fp8}_legalize.py`)
- Adapted (14): `apps/gemmini/{baremetal.py, matmul.c, matmul.h, README.md,
  verify_graph.py, verify_matmul.py}`,
  `python/tvm/relax/backend/contrib/{gemmini.py, gemmini_conv.py, gemmini_tuning.py}`,
  `python/tvm/relax/frontend/onnx/onnx_frontend.py`, and
  `tests/python/relax/test_{apps_gemmini_exported_graph, backend_contrib_gemmini_baremetal,
  backend_contrib_gemmini_schedule, backend_contrib_gemmini_tuning}.py`

Graham Anderson ([@grammy117](https://github.com/grammy117)) curated the bringup
into these branches and owns the tuning, FireSim and simplified-backend work that
followed.
