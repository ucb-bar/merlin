# Smol fast SIMS builds

The verified build remains the default. Fast builds use immutable weights already linked into the admitted default ELF, skip the weight copy and weight preservation scan, and retain activation, guard, fusion-extra, finite-output, and padded-output validation.

Use the exported model, interpreter, and tool variables in [model bringup setup](../README.md#build) and [Inputs and environments](bringing-up-a-model.md#inputs-and-environments). The wrapper requires the 537 distinct programs that the recorded fast asset generation produces and at least `stream-verified` for each; inspect its receipt as described there. Generate bound programs from the complete full-model bundle and its default full SIMS ELF. Set `PYTHONPATH` to the admitted npu-model checkout so its assembler is available. Every baseline assembly and binary must match the bundle, and every linked weight range must match the bundle member before the tool emits a bound program. Fusion metadata, including its FP8 scale, participates in the variant key.

```sh
PYTHONPATH="$MODEL47:$ATLAS/tools" "$MODEL_PYTHON" "$ATLAS/third_party/atlas-kernels/lib/atlas_backend.py" run --backend atlas-mlir \
  --atlas-mlir-bin "$ATLAS_MLIR_BIN" --min-tier stream-verified --expect-min-programs 537 \
  --receipt "$MODEL_OUTPUT_ROOT/smol-fast-backend.json" -- \
  "$ATLAS/tools/generate_smol_fast_assets.py" --bundle "$SMOL_BUNDLE" --elf "$SMOL_DEFAULT_ELF" --out "$SMOL_FAST_ASSETS"
```

The builder consumes explicit CMake compile and link recipes without invoking the build system. It compiles the maintained raw runtime and context header against the generated package's other host and platform inputs. Large immutable asset objects use a transient memory file, and the builder removes intermediate objects after recording its receipt. Generated package inputs must be the admitted source inputs for the default ELF used during generation.

```sh
python tools/build_smol_fast.py --platform-root /absolute/path/to/platform-support --build-dir /absolute/path/to/build --target-dir /absolute/path/to/build/penguin26-tests/atlas_delivery_smolvla --link-command /absolute/path/to/build/penguin26-tests/atlas_delivery_smolvla/CMakeFiles/atlas-delivery-smolvla.dir/link.txt --out /absolute/path/to/default --expect-sha256 EXPECTED_DEFAULT_SHA256
python tools/build_smol_fast.py --platform-root /absolute/path/to/platform-support --build-dir /absolute/path/to/build --target-dir /absolute/path/to/build/penguin26-tests/atlas_delivery_smolvla --link-command /absolute/path/to/build/penguin26-tests/atlas_delivery_smolvla/CMakeFiles/atlas-delivery-smolvla.dir/link.txt --out /absolute/path/to/fast --generated /absolute/path/to/generated --fast --expect-sha256 EXPECTED_FAST_SHA256
```

Each output directory contains `smolvla.elf`, a link map, compiler logs, and `build.json`. The receipt records source hashes, commands, recipe hashes, elapsed time, and the ELF hash. An expected-hash mismatch exits unsuccessfully after recording `FAIL_IDENTITY`. ELF identity does not replace numerical qualification or captured-reply replay.
