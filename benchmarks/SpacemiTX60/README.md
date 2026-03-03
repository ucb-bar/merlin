# SpacemiTX60 Dual-Model Runtime Flow

This folder provides scripts for deploying and running the new dual-model async
runtime sample on a SpacemiTX60 board via SSH.

## What gets run

- Runtime executable: `baseline-dual-model-async-run`
- VMFB artifacts:
  - `dronet.vmfb` (module name `dronet`)
  - `mlp.vmfb` (module name `mlp`)

The executable loads both VMFBs into the same IREE session, runs:
- Dronet as fast as possible.
- MLP at a configured periodic frequency.
- Synthetic sensor generators continuously produce new input tensors at
  configurable sensor rates.

## 1) Build executable (host or cross build tree)

The target name introduced by this sample is:

`merlin_baseline_dual_model_async_run`

Output binary name:

`baseline-dual-model-async-run`

## 2) Compile VMFB artifacts

Use:

```bash
benchmark/target/SpacemiTX60/compile_dual_model_vmfb.sh \
  --iree-compile /path/to/iree-compile \
  --target spacemit-riscv \
  --out-dir benchmark/target/SpacemiTX60/artifacts/vmfb
```

This script also rewrites module names so both can coexist in one runtime
session.

Via unified entrypoint:

```bash
python3 tools/merlin.py benchmark spacemitx60 compile-dual-vmfb -- \
  --iree-compile /path/to/iree-compile \
  --target spacemit-riscv \
  --out-dir benchmark/target/SpacemiTX60/artifacts/vmfb
```

## 3) Configure board connection

Copy and edit:

```bash
cp benchmark/target/SpacemiTX60/spacemitx60.env.example \
   benchmark/target/SpacemiTX60/spacemitx60.env
```

Set at least:
- `REMOTE_HOST`
- `REMOTE_USER`

## 4) Upload and run over SSH

```bash
benchmark/target/SpacemiTX60/run_dual_model_remote.sh \
  --binary /path/to/baseline-dual-model-async-run \
  --dronet-vmfb benchmark/target/SpacemiTX60/artifacts/vmfb/dronet.vmfb \
  --mlp-vmfb benchmark/target/SpacemiTX60/artifacts/vmfb/mlp.vmfb \
  --mlp-hz 20 \
  --duration-s 30 \
  --report-hz 1 \
  --dronet-sensor-hz 60 \
  --mlp-sensor-hz 20
```

Optional profiling:

```bash
... --profile-cmd "perf stat -d"
```

Via unified entrypoint:

```bash
python3 tools/merlin.py benchmark spacemitx60 run-dual-remote -- \
  --binary /path/to/baseline-dual-model-async-run \
  --dronet-vmfb benchmark/target/SpacemiTX60/artifacts/vmfb/dronet.vmfb \
  --mlp-vmfb benchmark/target/SpacemiTX60/artifacts/vmfb/mlp.vmfb
```

Results are stored locally under:

`benchmark/target/SpacemiTX60/results/<run_id>/`

including:
- `runtime.log`
- `run_meta.txt`
- `summary.json` (parsed stats)
