---
title: RVV end-to-end — lower a model through model2MLIR and run it on the Merlin runtime
kind: guide
status: current
owner: runtime
last_verified: 2026-07-16
related: [model2mlir, reproducibility, getting_started, kernel_mining]
code_refs:
  - merlin/python/merlin/runtime/dispatch_runtime.py
  - merlin/python/merlin/llvmlower/pipeline.py
  - merlin/python/merlin/llvmlower/passes_quant_int.py
  - merlin/python/merlin/runtime/backends/spike.py
  - merlin/python/merlin/rvvgen/k1.py
  - merlin/tests/rvv/test_rvv_spike.py
  - merlin/tests/rvv/test_smolvla_rvv.py
  - merlin/tests/rvv/test_vla_models_rvv.py
---

# RVV end-to-end

The flagship reproduction path: take a PyTorch model, lower it through **model2MLIR** into Merlin's
flow, compile it with native **RISC-V Vector (RVV)** codegen, and run it end-to-end on the Merlin
runtime — first as a host oracle, then bit-exact on **spike rv64gcv**, then (optionally) on the real
**SpacemiT K1** board. Every stage is fail-closed: if a toolchain is missing the step skips or records
`not_run`, never a fabricated pass.

This guide stitches the four stages into one runnable sequence. For the capture-bundle format and the
honest quantization status see [model2MLIR frontend](model2mlir.md); for the workflow index see
[Reproducibility](reproducibility.md).

## 0. Environment

All external toolchains are resolved from the gitignored `.env` (a process env var wins, then `.env`,
then a default — via `merlin.common.paths.env`). Copy `.env.example` → `.env` and set the paths for the
stages you want. Then confirm what will really run vs skip:

```bash
.venv/bin/python build_tools/scripts/check_repro_env.py
```

The RVV path uses these capabilities (from the preflight): `xdsl`, `llvm_m2m_toolchain` (model2MLIR +
clang-23), `spike_rv64gcv` / `saturn_vec` (the spike oracle), and `k1_board` (the real board — needs it
reachable, not just env). Key `.env` keys: `MERLIN_M2M_DIR`, `MERLIN_M2M_VENV`, `MERLIN_CHIPYARD` (or
`MERLIN_SPIKE`/`MERLIN_RISCV_GCC`), and for the board `MERLIN_K1_HOST` + `MERLIN_K1_SSH_KEY`.

## 1. Capture — model → MLIR bundle (model2MLIR)

Quantization and export happen in **model2MLIR, not Merlin**. The capture harness writes a
framework-neutral bundle under `out/artifacts/recaptures/<model>_<variant>_consistent/`:

```bash
build_tools/scripts/capture_matrix.sh          # runs $MERLIN_M2M_DIR workloads per model:variant
# each bundle: model.mlir + weights.safetensors(+manifest) + inputs.npz + golden.npy + extra.npz
```

The bundle is resolved in-code via `merlin.common.artifacts.recaptures_dir()` /
`merlin.baselines.bundle.resolve(model, variant)`. `int8` (W8A8) is the only measured-working quant
format today; `fp8`/`int4` are a documented plan (`unavailable`).

## 2. Compile + run e2e (dispatch runtime, native RVV)

The dispatch runtime outlines each dispatch to a per-kernel compile, walks the driver in numpy, and
gates the whole-model output against the torch `golden.npy`:

```python
from merlin.runtime import dispatch_runtime as dr
from merlin.common.artifacts import recaptures_dir

bundle = recaptures_dir() / "smolvla_fp32_consistent"
res = dr.run_model(bundle, workdir="out/build/rvv_e2e/smolvla")           # f32 native-RVV path
# int8 W8A8 integer datapath (vwmacc i8xi8->i32) instead of dequant-to-f32:
res_i8 = dr.run_model(bundle, workdir="out/build/rvv_e2e/smolvla_i8", int8_compute=True)
print(res["cos"], res["ok"], res["n_kernels"])   # cos vs torch golden; ok = within tolerance
```

Native RVV is baked into the MLIR by the transform-dialect vectorize schedule in
`llvmlower/pipeline.py` (`RVV_TRANSFORM_SCHEDULE`: tile + `structured.vectorize` the contraction to
fixed-width `vector<Nxf32>`, then lower to `vfmacc`); `clang -march=rv64gcv` emits the real
`vsetvli`/`vle32.v`/`vfmacc.vv`. The int8 datapath is `llvmlower/passes_quant_int.py`.

## 3. Verify bit-exact on spike rv64gcv

The spike backend compiles each command buffer to a bare-metal HTIF ELF and runs it under
`spike --isa=rv64gcv_zfh_zvfh`, gating the outputs against the Merlin reference:

```bash
.venv/bin/python -m pytest merlin/tests/rvv/test_rvv_spike.py -q          # full pipeline on spike (multicore)
.venv/bin/python -m pytest merlin/tests/runtime/test_saturn_vec.py -q -k spike
```

Verified (2026-07-16, `.env`-wired spike): `test_rvv_spike.py` = 6 passed; the saturn_vec rv64gcv cert
= 3 passed. The whole-model RVV runs (spike proxy) are gated behind `MERLIN_RUN_SLOW=1`:

```bash
MERLIN_RUN_SLOW=1 .venv/bin/python -m pytest \
  merlin/tests/rvv/test_smolvla_rvv.py merlin/tests/rvv/test_vla_models_rvv.py -q
```

## 4. Verify on the real K1 board (optional, cycle truth)

The K1 backend (`rvvgen/k1.py`) cross-compiles the kernel (`-march=rv64gcv -mabi=lp64d`, VLEN=256,
glibc Linux userspace), scp's it, and runs it over SSH — cycle counts via `rdtime`. It is **fail-closed**:
`k1.available()` is False (→ rung `not_run`, never a false pass) unless `MERLIN_K1_HOST` is set and the
board is reachable.

```bash
MERLIN_K1_HOST=root@<board-ip> MERLIN_K1_SSH_KEY=/path/to/key \
  .venv/bin/python -m pytest merlin/tests/rvv/test_k1.py -q
```

## What each stage proves

| stage | oracle | what it establishes | gate |
|---|---|---|---|
| host `run_model` | torch `golden.npy` | the lowering is numerically correct | `cos`/`rel` tolerance |
| spike rv64gcv | Merlin reference | real RVV instructions, bit-exact | output equality |
| K1 board | on-device run | real silicon cycles (rdtime) | `not_run_is_not_pass` |

The pattern to reproduce is stage-over-stage agreement (host ≈ spike ≈ board within tolerance), not a
single absolute number — the folder name is convenience; `run_record.json` / the captured `cos` is the
source of truth.
