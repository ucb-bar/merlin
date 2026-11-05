# Using the TVM–Gemmini comparison

This guide covers installing and running the TVM–Gemmini toolchain on this
branch: building the TVM fork, building kernel and ResNet50 ELFs (untuned or
tuned), checking them on Spike, and tuning them on a FireSim pool. To reproduce
the published F2 numbers, follow [REPRODUCING_F2_RESULTS.md](REPRODUCING_F2_RESULTS.md);
it uses the same tools with fixed pins. For what the compiler does, see
[IMPLEMENTATION.md](IMPLEMENTATION.md).

## 1. What you need

| Piece | Source | Notes |
| --- | --- | --- |
| Merlin | this branch, `mlsys27/feat-SoTA-tvm` (formerly `comparisons/tvm-workload-tuning`) | Provides `src/` (artifact helpers) and this directory, `examples/gemmini/comparisons/tvm/` (`$COMPARISON`) |
| TVM fork | submodule `third_party/tvm`, pinned at `f3cfd1a7d` on [`grammy117/tvm` `gemmini/direct-intrinsics`](https://github.com/grammy117/tvm/tree/gemmini/direct-intrinsics) | Apache TVM v0.19.0 plus the Gemmini backend (`python/tvm/relax/backend/contrib/gemmini*.py`) and tools (`apps/gemmini/`). The pin is the exact source of the F2 results. The branch tip adds only a Python refactor and documentation. |
| LLVM | 18.x with `llvm-config` | Built into `libtvm.so`; produces the RV64 operator objects |
| RISC-V GCC | `riscv64-unknown-elf-gcc` (Chipyard toolchain) | Compiles the C adapter and links the bare-metal ELF. F2 used 16.1.0; the a26 host uses 13.2.0. Both build working ELFs, but the bytes differ. |
| Gemmini software | gemmini `6ad65b90`, gemmini-rocc-tests `7c540b3a`, rocc-software `fddb795a` | Headers for the adapter (`gemmini_params.h` must match the hardware image) |
| Spike + libgemmini | libgemmini `ea8f7ed7` | Optional; functional checks only |
| FireSim pool | a qualified pool file (see [ADAPTER_CONTRACT.md](ADAPTER_CONTRACT.md)) | Needed only for tuning and timing |
| Python | 3.12 virtual environment | TVM Python deps (numpy 1.26, scipy, cloudpickle, tornado, decorator, attrs, packaging, psutil, pyyaml, jsonschema, typing_extensions, ml_dtypes) and pytest |

Use single-threaded BLAS and bounded build workers on shared machines.

## 2. Build TVM

```sh
git -C "$MERLIN" submodule update --init third_party/tvm     # f3cfd1a7d, the F2 compiler (name the path:
export TVM_SOURCE="$MERLIN/third_party/tvm"                  #  a bare --init also clones llvm-project)
git -C "$TVM_SOURCE" submodule update --init 3rdparty/dlpack 3rdparty/dmlc-core 3rdparty/rang
cmake -S "$TVM_SOURCE" -B "$TVM_BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DUSE_LLVM="$LLVM18/bin/llvm-config --link-shared" -DUSE_OPENMP=OFF -DUSE_LIBBACKTRACE=OFF \
  -DUSE_FALLBACK_STL_MAP=OFF -DUSE_HOST_ONLY_AUTO_COPY_GUARD=ON
cmake --build "$TVM_BUILD" --target tvm tvm_runtime --parallel 4
```

These are the flags of the a26 build (LLVM 18.1.8 from conda).
[host-config.cmake](host-config.cmake) is the bringup's equivalent cache file,
which also disables GPU, RPC and BLAS back ends. The Gemmini backend is
Python-only. The C++ sources are identical from `c50c4e269` through the branch
tip, so one `libtvm.so` serves every revision on the branch. Keep the submodule
at the F2 pin; to work on the newer tip, use a separate clone of
`gemmini/direct-intrinsics` and move the pin only together with new measurements.

## 3. Environment

```sh
export MERLIN=<merlin checkout>               # this branch
export TVM_SOURCE="$MERLIN/third_party/tvm"  TVM_BUILD=<tvm build dir>
export TVM_LIBRARY_PATH="$TVM_BUILD"
export PY=<venv>/bin/python
export RISCV_GCC=<riscv64-unknown-elf-gcc>
export ADAPTER_RECEIPT=<adapter build>/receipt.json   # from step 4
export COMPARISON="$MERLIN/examples/gemmini/comparisons/tvm"
export PYTHONPATH="$MERLIN/src:$TVM_SOURCE/python:$TVM_SOURCE/apps/gemmini:$COMPARISON"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TVM_NUM_THREADS=1
"$PY" -c "import tvm; print(tvm.__file__, tvm._ffi.base._LIB._name)"   # confirm both paths
```

On a26, `source /scratch/grammy/blast/tvm-tuning-env.sh` sets these variables.
Its `TVM` points at `tvm-f2-followup` (`c50c4e269` plus local edits), which is
not the F2 pin. Override `TVM_SOURCE` and `PYTHONPATH` with
`$MERLIN_DELIVERY/third_party/tvm` (the submodule: exact F2 sources, same tree
as `$BASELINE/tvm-f2-pin-f3cfd1a7d`) or `$BASELINE/tvm-direct-intrinsics`
(branch tip).

## 4. Build the Gemmini adapter

The adapter (`apps/gemmini/matmul.c`) is the C runtime that issues Gemmini
commands. Build it once per toolchain/header set:

```sh
"$PY" "$TVM_SOURCE/apps/gemmini/verify_matmul.py" \
  --gemmini-repo <gemmini> --rocc-tests-repo <gemmini-rocc-tests> --rocc-software-repo <rocc-software> \
  --build-root <dir ending in baselines/tvm-gemmini> --output-dir <that dir>/adapter-<name> --cc "$RISCV_GCC"
```

The receipt records the source set, header hashes, exported symbols and an
instruction audit. `compile_package.py` refuses an adapter receipt whose
`matmul.c`/`matmul.h` no longer match the TVM checkout.

## 5. Build ELFs

Every builder writes `package/` (operator objects, constants, ABI, placement and
memory audits), `guest/valid.elf`, `guest/bad-oracle.elf` and a reference
binding. Use a fresh output directory for every build.

**Kernel cases** (G01–G06, C01–C03, T01–T07):

```sh
"$PY" "$COMPARISON/prepare_kernel_benchmarks.py" list
"$PY" "$COMPARISON/prepare_kernel_benchmarks.py" prepare --case G01 --output-dir "$WORK/kernel-inputs/G01"
"$PY" "$COMPARISON/prepare_kernel_benchmarks.py" build --prepared "$WORK/kernel-inputs/G01" \
  --output-dir "$WORK/baseline/G01" --riscv-gcc "$RISCV_GCC" --adapter-receipt "$ADAPTER_RECEIPT"
```

Kernel builds refuse CPU fallback for the measured operation.

**ResNet50 chunks and the connected parent:**

```sh
"$PY" "$COMPARISON/restore_resnet_inputs.py" --merlin-root "$MERLIN" --output-dir "$WORK/resnet-inputs"
"$PY" "$COMPARISON/build_shared_chunk_elf.py" --tvm-source "$TVM_SOURCE" \
  --restored-root "$WORK/resnet-inputs" --chunk resnet-stem --output-dir "$WORK/baseline/resnet-stem" \
  --riscv-gcc "$RISCV_GCC" --adapter-receipt "$ADAPTER_RECEIPT" \
  --host-math-policy native --tensor-memory-bytes 268435456
```

Chunk ids are `resnet-stem`, `resnet-stage{1..4}-block{N}`, `resnet-classifier`
(18 in all), plus `full-resnet` for the connected parent.

**Tuned builds** add `--tuning-database <dir>` and the graph plan that database
was tuned for: `--graph-fusion-level {0..3}`, plus `--graph-cse` when tuned with CSE.
The compiler refuses a database with zero structural hits. Tasks without a
record keep the default schedule, and the placement report says which happened.

## 6. Check functionally on Spike (optional)

Build the plugin once with `build_simulator.py` (`--libgemmini-repo`,
`--spike-prefix`, `--runtime-lib-dir`, `--runtime-bin-dir`, `--smoke-elf`,
`--build-root`). Then:

```sh
"$PY" "$COMPARISON/check_kernel_spike.py" --guest "$WORK/baseline/G01/guest" \
  --simulator-receipt <simulator>/receipt.json --seconds 120
```

It passes only if `valid.elf` prints `DSE_PASS` and exits 0, and `bad-oracle.elf`
exits 22 with `TENSORIR_NUMERICAL_FAILURE`. Spike is a functional check, not a
timing source. `build_simulator.py` builds the plain libgemmini plugin. The
repaired-RNE plugin used for [SPIKE_HANDOFF_EVIDENCE.md](SPIKE_HANDOFF_EVIDENCE.md)
is recorded only by its patch hash and cannot be rebuilt from this branch.

## 7. Tune on FireSim

1. **Qualify the pool** under [ADAPTER_CONTRACT.md](ADAPTER_CONTRACT.md):
   repeat agreement, warm-call scope, wrong-oracle rejection, remote wall
   cancellation, and safe slot reuse. The F2 run's `tools/qualify.py` is a
   working example.
2. **Prepare a study** from prepared inputs. Owners with structurally identical
   contractions share records:

   ```sh
   "$PY" "$COMPARISON/prepare_active_study.py" --kind kernel|resnet \
     --input-root "$WORK/<kind>-inputs" --output-dir "$WORK/<kind>-study" [--initial-trials 20]
   ```

3. **Run it.** The command without `--execute` writes the plan only:

   ```sh
   "$PY" "$COMPARISON/run_firesim_study.py" --study "$WORK/<kind>-study/study.json" \
     --toolchain toolchain.json --pool pool.json --output-dir "$WORK/tune-<kind>" \
     --campaign-kind <kind> --target-clock-hz <counter Hz> \
     --candidate-bootstrap-seconds <s> --candidate-timeout-ceiling-seconds <s> \
     --deadline-epoch <epoch> --jobs <n> --ledger ledger-<kind>.json --execute
   ```

   `toolchain.json` names `gcc`, `adapter_receipt`, `hardware_receipt`,
   `timing_policy: "fixed-warm"` and `cycle_watchdog: false`. Keep one ledger per
   campaign across restarts and phases (`--phase continuation` for reserved
   graph-plan trials). At the deadline admissions stop and active runs finish
   under their frozen watchdogs.
4. **Use the results.** Each owner directory has `trials.jsonl` (every proposal
   with build/run errors and cycles), `baselines.json`, `timeout-history.json` and
   `validated-database/`. Build tuned ELFs from the validated database (step 5)
   and time the **complete** unit; extracted-task cycles are labels for search,
   not results.

## 8. Tool reference

| Tool | Purpose |
| --- | --- |
| `prepare_kernel_benchmarks.py list/prepare/build` | Generate the 16 kernel cases and build their ELFs |
| `restore_resnet_inputs.py` | Restore the committed ResNet50 split archive (no large-model capture) |
| `build_shared_chunk_elf.py` | Build a ResNet chunk or `full-resnet` ELF |
| `prepare_active_study.py` | Extract tasks and owners into a ReplayTrace study |
| `run_firesim_study.py` | Schedule per-owner campaigns over a pool with budgets, ledger and deadline |
| `tune_firesim_chunks.py` | One owner's campaign (called by the study runner) |
| `firesim_job_pool.py`, `firesim_timeout.py`, `firesim_runner_seconds.py` | Slot pool, median wall watchdog, cycles to seconds |
| `check_kernel_spike.py`, `check_shared_chunks.py` | Spike functional check; chunk semantics/owner inventory |
| `$TVM_SOURCE/apps/gemmini/compile_package.py` | Compile a program into a static package (used by the builders) |
| `$TVM_SOURCE/apps/gemmini/tune_models.py --extract-only` | Show extracted tasks and exclusions for a program |

Older tools in this directory (`verify_session.py`, `tinyllama_*`, `smolvla_*`,
`FASED_SWEEPS.md` flows) serve the backlogged full-model work and the deprecated
Verilator sweeps; the F2 campaign did not use them.

## 9. Tests

```sh
"$PY" -m pytest -q -n 8 merlin/tests/gemmini/test_tvm_*.py                       # from $MERLIN
"$PY" -m pytest -q -n 8 tests/python/relax/test_*gemmini*.py                      # from $TVM_SOURCE
```

Known inherited failures: TinyLlama/SmolVLA session tests need packages not in
the compiler environment (`transformers`, `onnxscript`, `lerobot`,
`safetensors`). TVM's `test_apps_gemmini_memory_model.py` needs the removed FASED
`mem_perturb.pinned_latency_pipe` helper. Four more TVM tests fail at both the F2
pin and the tip. Two expect features that were removed on purpose: automatic
bias preload and an INT32-add TensorIntrin. One expects an old tuning preset to
match a new prefill shape. One expects a static-graph `ValueError` that is no
longer raised; that last failure has not been investigated. The [handoff
regression product](../../../../out/artifacts/verification/gemmini-tvm-handoff-regression/v1/)
lists them all.
