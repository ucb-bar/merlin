# TinyLlama capture, qualification, build, and replay

The repository's D2 capture and `d1-v2-qualify.py` route is the historical 906-launch model: 840 projection panels and 66 attention panels. Its qualifier explicitly requires that inventory. The current kit uses T-on-F3-h0 with 1,038 launches, including 132 norm launches and decode q/k RoPE; the external reproduction route is described below. Running the D2 capture driver does not regenerate the current kit. The maintained `AttentionBundle` and `build_attention.py` admit both inventories, plus the 972-launch post-attention-norm inventory.

Build the verified execution view using [model bringup setup](../../README.md#build), keeping generated outputs outside the view and canonical repositories. Set physical absolute paths in `ATLAS_ROOT`, `MERLIN_ROOT`, `ATLAS_NPU_MODEL_ROOT`, and `TINY_WORK_ROOT`. The model must be unmodified original `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5`. `TINY_WORK_ROOT` holds generated captures and receipts. Initialize a Python environment with the model, NumPy, and capture dependencies; `PYTHON` names that interpreter, and `HOST_CC` names the host C compiler. The input parameter capture, projection scale contract, and attention scale contract are external checksum-bound inputs, not downloaded by these commands. Their schemas and fixed contracts are checked by Merlin's driver. Keep the model and data roots physical, without symlinks. Use at most 16 numerical workers.

```sh
export ATLAS_ROOT=/absolute/path/to/atlas-mlir
export ATLAS="$VIEW" ATLAS_SOURCE_ROOT="$VIEW" ATLAS_MLIR_ROOT="$VIEW"
cd "$VIEW"
export MERLIN_ROOT=/absolute/path/to/merlin
export ATLAS_NPU_MODEL_ROOT=/absolute/path/to/npu-model
export TINY_WORK_ROOT="$MODEL_OUTPUT_ROOT/tiny-work"
export PYTHON=/absolute/path/to/model-venv/bin/python
export ATLAS_MLIR_BIN="$ATLAS_ROOT/build/bin"
export PYTHONPATH="$ATLAS_NPU_MODEL_ROOT:$ATLAS/tools:$MERLIN_ROOT/src"
export HOST_CC=/absolute/path/to/cc
export SOURCE_CAPTURE=/absolute/path/to/parameter-capture.json
export PROJECTION_SCALES=/absolute/path/to/projection-scale-contract.json
export ATTENTION_SCALES=/absolute/path/to/attention-scale-contract.json
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ATLAS_NPU_MODEL_NATIVE=0
export EXPECT_PROGRAMS=N  # distinct programs this export must produce; see the notes below
mkdir -p "$TINY_WORK_ROOT/tmp"
export TMPDIR="$TINY_WORK_ROOT/tmp"
ATLAS_LAYOUT_SHA256=$(sha256sum "$ATLAS/tools/atlas_host_layout.py" | cut -d ' ' -f 1)
ATLAS_ROOT="$ATLAS" "$PYTHON" -B "$MERLIN_ROOT/examples/atlas/comparisons/d1-capture-c-v2.py" \
  --source-capture "$SOURCE_CAPTURE" \
  --scale-contract "$PROJECTION_SCALES" \
  --attention-scale-contract "$ATTENTION_SCALES" \
  --attention-model-root "$ATLAS_NPU_MODEL_ROOT" \
  --atlas-layout "$ATLAS/tools/atlas_host_layout.py" \
  --atlas-layout-sha256 "$ATLAS_LAYOUT_SHA256" \
  --worker-script "$MERLIN_ROOT/examples/atlas/comparisons/d1-worker.py" \
  --worker-python "$PYTHON" \
  --attention-worker-script "$ATLAS/tools/tiny_full/attention_v2_worker.py" \
  --attention-worker-python "$PYTHON" \
  --cc "$HOST_CC" --layer-count 22 --decode-count 2 \
  --out "$TINY_WORK_ROOT/d2-capture"
CAPTURE_SHA256=$(sha256sum "$TINY_WORK_ROOT/d2-capture/receipt.json" | cut -d ' ' -f 1)
"$PYTHON" -B "$ATLAS/third_party/atlas-kernels/lib/atlas_backend.py" run \
  --backend atlas-mlir --atlas-mlir-bin "$ATLAS_MLIR_BIN" \
  --min-tier stream-verified --expect-min-programs "$EXPECT_PROGRAMS" \
  --receipt "$TINY_WORK_ROOT/export-backend.json" -- \
  "$ATLAS/tools/tiny_full/export_capture.py" \
  --capture "$TINY_WORK_ROOT/d2-capture/receipt.json" --capture-sha256 "$CAPTURE_SHA256" \
  --model-root "$ATLAS_NPU_MODEL_ROOT" --out "$TINY_WORK_ROOT/d2-assets" \
  --dram-base 0x80000000 --dram-bytes 0x80000000
"$PYTHON" -B "$ATLAS/tools/tiny_full/d1-v2-qualify.py" \
  --assets "$TINY_WORK_ROOT/d2-assets" \
  --out "$TINY_WORK_ROOT/d1-v2-qualification/full" --workers 16
```

`--out` for capture and export must be fresh. The 2 GiB DRAM example must match the actual target placement. The public export entry point is `export_capture.py`; `attention_export.py` is its implementation, not a standalone CLI. The C capture driver uses `d1-worker.py` for projections and `tools/tiny_full/attention_v2_worker.py` for attention, with generators imported from the independently selected atlas-kernels root staged in the view. It captures positive-zero residual surrogates for o/down; the exporter emits the final plain programs, and the qualifier executes those final programs on original47. The original-model path needs no native proof arguments. `--qualify-launches` on the exporter is an alternative direct per-launch execution gate; the separate qualifier adds its aggregate coverage checks.

The explicit `atlas_backend.py run --backend atlas-mlir` export command selects the machine IR, verifier, and emitter route using `atlas-opt` and `atlas-emit` in `ATLAS_MLIR_BIN`; the backend receipt records the encodes performed in that process. Set `EXPECT_PROGRAMS` to the number of distinct device programs the selected inventory must produce. No count is recorded for this historical 906-launch export; the recorded wrapped runs of the current kit cover 210 `check_programs.py` programs and 850 programs for each fast variant, as listed in [bringing up a model](../../docs/bringing-up-a-model.md). A wrapped run writes the count it encoded to `summary.programs` in its `--receipt` JSON, and `check_programs.py` prints `unique_programs` for the assets it checks, so take the value from a reviewed run of the same inventory; the wrapper exits nonzero on fewer programs or on any program below `stream-verified`. The separate qualifier launches subprocesses, whose encodes are not covered by the parent's export receipt. Require the independent exhaustive program-word comparison gate described in [bringing up a model](../../docs/bringing-up-a-model.md) for every distinct program in the selected default and fast builds before treating the compiler backend as qualified. Numerical exactness, generated source/binary equality, and backend word identity are separate gates.

Without `--reuse-records`, every final program executes on original47. Repeating the qualifier command with `--resume` audits saved rows before reuse. `--audit-only` audits existing results without filling missing executions. Qualification output must be below `$TINY_WORK_ROOT/d1-v2-qualification/`. The stream drivers and workers live beside the Merlin capture driver and retain their interfaces for optional concurrent backfill.

Aggregate qualification receipts encode path pins with @atlas/, @merlin/, @model/, and @data/ roots. build_attention.py takes --model-root and --receipt-data-root to resolve these roots, derives the atlas root from its checkout, and derives the Merlin root from --merlin (the comparisons directory). SHA-256 checks and HEAD blob authentication run after resolution. Legacy receipts can be read with repeatable --receipt-relocate OLD=NEW arguments; relocation never substitutes a digest. Preserve historical source bytes when relocating historical pins.

For the host build, use [the complete build command](../../docs/tiny-attention-build.md) with `--assets "$TINY_WORK_ROOT/d2-assets"`, the aggregate `receipt.json`, its SHA-256, `--merlin "$MERLIN_ROOT/examples/atlas/comparisons"`, `--model-root "$ATLAS_NPU_MODEL_ROOT"`, and `--receipt-data-root "$TINY_WORK_ROOT"`. Supply the external platform root and GNU RISC-V toolchain explicitly. The build output contains `host/`, `sources.cmake`, `build.json`, and `build/atlas-tinyllama.elf`. Captures may require substantial disk space; no storage reuse wrapper is required.

The qualifier admits projection reuse only for the original capture and stream schemas with exact=True and full_arena_exact=True. It also runs one additional original47 execution per unique program. Every byte in the output extent starts as the complement of the expected byte, declared spans must become the expected bytes, holes must retain their second pre-fill, and all other arena bytes must remain unchanged. Coverage reports are included in the aggregate receipt and are audited by --audit-only.

## Current 1,038-launch kit

The current 1,038-launch kit route requires a preserved artifact workspace. It is not self-contained in the public checkout. Qualification combines 88 fresh original47 runs and 950 historical exact joins. The qualified-kit scripts and inputs are retained outside the repositories, and they are not committed here. Ask Nicolas for their location, then set these variables to the retained copies:

| Variable | Fills |
| --- | --- |
| `TINY_KIT_ROOT` | The retained `T` workspace: the owner source `atlas-f3h0/`, its original model `npu-model/`, the `f3h0/` preparation, qualification, packaging, build, and replay scripts, and the `bulk/` assets and builds. |
| `F3H0_ROOT` | The retained F3-h0 qualification tree, with `qual-full/receipt.json` and `results.json`. |
| `F3H0_ASSETS` | The unchanged F3-h0 baseline assets (`assets-full`) that the overlay's `storage.json` inherits. |
| `KIT_PYTHON` | The Python 3.14 interpreter with npu_model `47b8599`'s locked dependencies and NumPy. |
| `IDENTITY_KIT` | The retained identity scripts: `identity_build.py`, `ORCH-RUN-tiny-resnet.sh`, and `ORCH-RUN-smol.sh`. |
| `TINY_KIT_ELFS` | The retained kit ELF directory, including `atlas-tinyllama-v2-sims.elf`. |
| `QEMU_RISCV64` | The `qemu-system-riscv64` binary of the recorded replay. |
| `RISCV_BIN` | The GNU RISC-V tool directory of the owner recipes, containing `riscv64-unknown-elf-gcc`. |

The recorded receipts name the original absolute locations of these inputs; ask Nicolas which copies keep them valid before relocating. In the commands below, the exact owner source is `$TINY_KIT_ROOT/atlas-f3h0`, its original model is `$TINY_KIT_ROOT/npu-model`, and the Python environment is `$KIT_PYTHON`. The unchanged baseline assets must exist at `$F3H0_ASSETS`, with `$F3H0_ROOT/qual-full/receipt.json` and all pinned historical source bytes. `f3h0/prepare.py` creates a checksum-bound overlay over that asset authority, changing 88 decode q/k launches. Its 950 other launches retain exact source, binary, initial arena, output, and complete final-arena joins. `storage.json` records the inherited asset root, so copying only the overlay directory is insufficient.

The following are the recorded owner interfaces, shown for a fresh isolated copy of that preserved workspace. Their fixed roots, archived compile recipes, external platform tree, and output locations must all be present. Existing receipts and builds must be preserved; these commands are not a request to overwrite them.

```sh
cd "$TINY_KIT_ROOT"
PY="$KIT_PYTHON"
export TMPDIR="$PWD/tmp" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ATLAS_NPU_MODEL_NATIVE=0
"$PY" -B f3h0/prepare.py --out "$PWD/bulk/f3h0-assets" --model-root "$PWD/npu-model"
"$PY" -B f3h0/qualify.py --assets "$PWD/bulk/f3h0-assets" --out "$PWD/f3h0/qual/full" --workers 16
"$PY" -B f3h0/package_qualification.py
"$PY" -B f3h0/build.py default
```

The qualifier's `--plan-only` checks the full inventory and writes exact reuse rows without executing the 88 changed launches; `--audit-only` requires existing complete results. The full command resumes by auditing completed rows. `package_qualification.py` archives the 1,038 worker records and logs into `qual/original-receipts.zip`, losslessly compresses the aggregate into `qual/receipt.json.gz`, and records both compressed and uncompressed hashes in `qual/GZIP.json`. `build.py default`, `build.py chip`, and `build.py fast` replay archived compilation/link recipes; their `bulk/f3h0/MODE/build.json` records source bookends, commands, and ELF SHA-256. This is the owner recipe interface, distinct from a new `build_attention.py` build.

For exact linked ELF identity, the retained `$IDENTITY_KIT/ORCH-RUN-tiny-resnet.sh` invokes `identity_build.py tiny-default`, `tiny-kit3-fast`, `tiny-m1-fast`, `tiny-m2-fast`, and `resnet`, and `$IDENTITY_KIT/ORCH-RUN-smol.sh` builds the 6 SmolVLA rows. They pin the source tips, use explicit external platform roots, reuse frozen generated assets and platform archives, and require expected SHA-256 plus full `cmp` on every row before writing `PASS.json`. At the gate commit `23d7f2b` with atlas-kernels `e7bbf7c`, identity passed on 16 of 16 rows: 10 TinyLlama and ResNet50 rows and 6 SmolVLA rows. This identity gate does not recapture, regenerate programs, qualify arithmetic, or replay the host. Adapt a copy when changing source roots, preserving the original recipe and its evidence.

The actual default QEMU replay runner is external `$TINY_KIT_ROOT/f3h0/replay/replay_v2_complete.py`, with the 1,038-launch overlay as `--bundle-root`. A fresh replay of an available matching default ELF uses the complete recorded arguments below; `REPLAY_OUT` must name a fresh output directory. The GNU compiler used by the owner recipes is `$RISCV_BIN/riscv64-unknown-elf-gcc`.

```sh
REPLAY_OUT=/absolute/path/to/fresh-tiny-replay
"$KIT_PYTHON" -B "$TINY_KIT_ROOT/f3h0/replay/replay_v2_complete.py" \
  --elf "$TINY_KIT_ROOT/bulk/f3h0/default/atlas-tinyllama.elf" \
  --bundle-root "$TINY_KIT_ROOT/bulk/f3h0-assets" --qemu "$QEMU_RISCV64" \
  --out "$REPLAY_OUT" --wall-seconds 7200 \
  --console-write-symbol _write --exit-symbol _exit --icount --gdb-stdio
```

The historical default ELF was retired after hashing, so supply a rebuilt byte-identical default ELF or the matching kit provider at `$TINY_KIT_ELFS/atlas-tinyllama-v2-sims.elf` through `--elf`. The external fast runner is `$TINY_KIT_ROOT/f3h0/replay/run_fast.py --out /absolute/path/to/fresh-fast-replay`; it additionally requires the linked fast ELF, generated programs, and `generated/admission.json` at its fixed owner paths. The generic default runner alone does not establish the fast transport admission proof.

Evidence flows from the capture and checksum manifest, through original47 numerical qualification, to build/source pins, linked fast admission when applicable, and the replay receipt. `$TINY_KIT_ROOT/f3h0/replay/default-full.command.json` and `default-full.inputs.json` preserve the measured replay command and input hashes; `default-full/receipt.json` records `PASS_CAPTURED_REPLY_REPLAY`, 1,038 exact launches, 137 exact final buffers, `host_elf_executed=true`, and `numerical_device_execution=false`. The guest tokens of this default ELF (SHA-256 `3b75125b5f552eeb93cdab158bd09349170e7b23894dc8f3a717040a732eb081`) are `[278, 3711, 5652]` at position 130. The `[278, 14263, 658]` in [task accuracy](../../docs/task-accuracy.md) belongs to the earlier 840-launch projection-only host, not this kit. `$TINY_KIT_ROOT/f3h0/GREEN.json` binds qualification, replay, corruption, static regeneration, and memory gates, and the retained kit README `KIT5/candidate/README.md` describes the kit's required terminal checks. Replay executes host instructions with captured Atlas replies, while the separate qualifier establishes exact arithmetic. The 3,528,847,056 cycles in `$F3H0_ROOT/results.json` at `derived_Atlas_cycles.total_cycles` belong to the older host-RoPE F3-h0 baseline and combine 730 D2 timing reuses, 106 F2 timing reuses, and 202 fresh original47 Core timings. The current synthetic-input bench totals 3,537,507,930 npu_model model cycles using reused full-shape timing runs; the [bench reconciliation](https://github.com/ucb-bar/atlas-kernels/blob/feat/kernel-library/coverage/examples/tinyllama/MODEL-BENCH.md#L105-L109) explains the program and timing-convention differences. None of these artifacts establishes full-model FireSim execution or physical device timing.
