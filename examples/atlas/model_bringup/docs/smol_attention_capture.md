# SmolVLA attention recapture and SIMS build

The maintained entrypoint is `tools/capture_smolvla_attention.py`, backed by `tools/smol_capture/`. Merlin's matching entrypoint imports this package from `--atlas-root`. The continuation runs the actual Merlin C graph, including its controls, norms, RoPE, cache joins, and Euler updates. Its default worker executes the configured original `npu_model` with explicit `numerics="rtl"`; no native MXU or NumPy attention substitute is required. Capture is reference generation, not ELF qualification, hardware execution, or a policy-quality claim.

Use physical paths without symlink components for source and output arguments. A virtualenv interpreter may be a symlink: its targets are separately screened before use. The interpreter needs NumPy and a CPU PyTorch compatible with the selected original model. A native C11 GCC compiler, Git for model identity, and the build toolchains below are required. The package rejects prohibited path components before access.

## Inputs

Build the view with [model bringup setup](../README.md#build), preserving canonical `ATLAS_ROOT` and `ATLAS_MLIR_BIN`. Set these variables to the teammate's independent checkouts and providers. None refers to a job directory, and the output directory must be writable. The capture backend uses the fixed compiled-observation bundle, so a raw checkpoint is not required.

```sh
PY=/absolute/path/to/venv/bin/python
export ATLAS="$VIEW" ATLAS_SOURCE_ROOT="$VIEW" ATLAS_MLIR_ROOT="$VIEW"
MERLIN=/absolute/path/to/merlin
MODEL=/absolute/path/to/original-npu-model
MODEL_PIN=47b8599d8ffbe5ddd96059ffe2b5be2427a410f5
INPUT_BUNDLE=/absolute/path/to/smol-f2-seg
BASELINE=/absolute/path/to/smol
WORK="$MODEL_OUTPUT_ROOT/smol-work"
```

`INPUT_BUNDLE` is the v1 FullBundle, with `SHA256SUMS`, `contract.json`, `host/reference-replay.json`, and its declared parameters, weights, programs, and reference buffers. Every consumed member is checked against `SHA256SUMS`, and embedded host parameters are compared byte for byte against the bundle before the C graph runs. These parameters include the compiled fixed observation, model weights, and initial state. This is the input-provider path used by the qualified trajectory after migration. Historical raw checkpoint and observation archives are recorded as historical metadata and are not claimed revalidated.

`BASELINE` supplies `host-baseline-v4-build/{main.c,assets.S}`, `host-baseline-v4/{SHA256SUMS,assets/}`, and `phase-canonical-v4/{compile.json,phase-reference.json}`. The generated main and assembler declarations are authenticated against the existing compilation receipt, and every assembler payload is authenticated against the asset checksum file before compilation. These are explicit immutable data providers, not imported job helpers. All production graph C providers come from `$MERLIN/examples/atlas/comparisons` and `$ATLAS/runtime`; the native transport and wrapper are maintained within this package. `--native-source` can explicitly override the two native transport source files.

If the baseline receipt and assembler declarations contain a historical path prefix, add `--relocate OLD=NEW` for each move. The mapping is explicit, longest prefix first, and does not weaken byte checks. The current input tree itself must use physical paths.

## Capture and small original-model check

```sh
cd "$VIEW"
"$PY" -B "$ATLAS/tools/capture_smolvla_attention.py" --help
"$PY" -B "$ATLAS/tools/smol_capture_smoke.py" \
  --python "$PY" --atlas-root "$ATLAS" --model-root "$MODEL" --model-pin "$MODEL_PIN" \
  --out "$WORK/small-capture"
"$PY" -B "$ATLAS/tools/capture_smolvla_attention.py" \
  --python "$PY" --atlas-root "$ATLAS" --model-root "$MODEL" --model-pin "$MODEL_PIN" \
  --bundle "$INPUT_BUNDLE" --baseline-root "$BASELINE" \
  --providers "$MERLIN/examples/atlas/comparisons" \
  --out "$WORK/capture" --prepare --execute
```

The small check captures and executes a real 32 by 32 projection and a one-head, 32-key attention program, checks known constant outputs exactly, and preserves full arenas and execution metadata. A full original-model capture can be expensive. `--phase-limit 1` produces a deliberate incomplete first-phase capture for integration smoke checks. A completed full capture must contain 1,421 projection calls, 2,715 projection panels, 200 attention launches, and 2,915 combined launches. Its `run/phase-reference.json` contains the initial state plus 200 completed phase boundaries, each with 56 live regions. `run/manifest.json`, `replay.json`, `device-events.jsonl`, `compiled-input-provider.json` in the capture root, and `baseline-differences.json` retain identities and scope. Final actions and all intermediate reference buffers are regenerated from the current host graph.

The original model commit and every currently imported model source must match the selected Git HEAD before execution. Neither imported modifications nor a substituted model root is silently accepted. The original numerical gate and final guest admission remain separate steps even when reference generation uses original arithmetic.

To reuse already admitted replies while regenerating all current host math and boundaries, add `--replies /absolute/path/device-events.jsonl --replies-sha256 EXACT_SHA256`. Every event must match the complete current staged arena hash and extent, actual runtime program hash and extent, ordered type and output geometry, output digest, whole preserved arena digest, and DMA request/observation. This is an identity-bound replay and records `fresh_model_executed=false` and `original_model_all_launch_gate=false`. The caller supplies the ledger admitted by its strict original-model audit; supplying a hash alone does not confer numerical admission. `--resume` applies the same current-state recomputation to a durable local prefix after interruption.

To reduce storage during a reply regeneration, add `--shared-phase-reference /absolute/path/phase-reference.json --shared-phase-reference-sha256 EXACT_SHA256`. Exact current blob bytes are compared with the immutable recorded source before creating a hardlink. The source is never reopened for writing, and unavailable cross-filesystem links fall back to copies. Event arenas and replies are also shared by exact identity when `--replies` is supplied. No executable or generated model output is copied into the source repositories.

`--observation ARCHIVE --observation-sha256 EXACT_SHA256` optionally authenticates an accompanying raw archive. It does not replace the compiled-observation input provider or assert that a raw checkpoint was reconstructed.

## Export and standalone build

Set `SUPPORT` to the RV64 support directory containing `rv64_support.c` and `include/`, `CLANG` to the RISC-V-capable compiler, `LLD` to its linker, and `TABLE_PYTHON` to an interpreter with the required table generation dependencies. These are toolchain inputs rather than repository-relative assumptions. Use the compiler and environment variables from [Inputs and environments](bringing-up-a-model.md#inputs-and-environments). When continuing the capture above, set `MODEL_PYTHON=$PY`, `MODEL47=$MODEL`, `SMOL_CAPTURE=$WORK/capture`, `SMOL_ASSETS=$WORK/kernel-assets`, `SMOL_BUNDLE=$WORK/bundle`, and `SMOL_BUNDLE_AUDIT=$WORK/audit-standalone.json`. Each wrapped command requires its own reviewed `$EXPECT_PROGRAMS` count for the selected profile and writes a separate backend receipt; inspect these as described in [the backend gate](bringing-up-a-model.md#4-check-kernels-at-the-actual-shapes).

```sh
PYTHONPATH="$MODEL47:$ATLAS/tools" "$MODEL_PYTHON" -B "$ATLAS/third_party/atlas-kernels/lib/atlas_backend.py" run --backend atlas-mlir \
  --atlas-mlir-bin "$ATLAS_MLIR_BIN" --min-tier stream-verified --expect-min-programs "$EXPECT_PROGRAMS" \
  --receipt "$MODEL_OUTPUT_ROOT/smol-export-backend.json" -- \
  "$ATLAS/tools/export_smol_policy_kernels.py" --native-capture \
  --replay "$SMOL_CAPTURE/run/replay.json" --attention-capture "$SMOL_CAPTURE/run/manifest.json" \
  --model-root "$MODEL47" --out "$SMOL_ASSETS"
PYTHONPATH="$MODEL47:$ATLAS/tools" "$MODEL_PYTHON" -B "$ATLAS/third_party/atlas-kernels/lib/atlas_backend.py" run --backend atlas-mlir \
  --atlas-mlir-bin "$ATLAS_MLIR_BIN" --min-tier stream-verified --expect-min-programs "$EXPECT_PROGRAMS" \
  --receipt "$MODEL_OUTPUT_ROOT/smol-host-backend.json" -- \
  "$ATLAS/tools/build_smol_full_host.py" \
  --assets "$SMOL_ASSETS" --replay "$SMOL_CAPTURE/run/replay.json" \
  --merlin "$MERLIN/examples/atlas/comparisons" --attention "$MERLIN/examples/atlas/comparisons" \
  --runtime "$ATLAS" --bundle-tools "$ATLAS/tools/host_iss" \
  --support "$SUPPORT" --compiler "$CLANG" --linker "$LLD" --table-python "$TABLE_PYTHON" \
  --model-root "$MODEL47" --host-input-bundle "$INPUT_BUNDLE" \
  --uart-kind sifive --uart-base 0x10020000 --uart-divisor 134 --target sim \
  --dram-base 0x80000000 --dram-bytes 0x80000000 --arena-base 0x90000000 \
  --traffic-reuse --rodata-after-arena --phase-reference "$SMOL_CAPTURE/run/phase-reference.json" \
  --out "$SMOL_BUNDLE"
PYTHONPATH="$MODEL47:$ATLAS/tools" "$MODEL_PYTHON" -B "$ATLAS/third_party/atlas-kernels/lib/atlas_backend.py" run --backend atlas-mlir \
  --atlas-mlir-bin "$ATLAS_MLIR_BIN" --min-tier stream-verified --expect-min-programs "$EXPECT_PROGRAMS" \
  --receipt "$MODEL_OUTPUT_ROOT/smol-bundle-check-backend.json" -- \
  "$ATLAS/tools/check_smol_full_bundle.py" \
  --bundle "$SMOL_BUNDLE" --model-root "$MODEL47" --out "$SMOL_BUNDLE_AUDIT"
```

The builder creates a sibling build-work directory (`$WORK/bundle-build` for the capture-workspace mapping above); use that directory and `$SMOL_BUNDLE` as the packaging inputs below. Its validated v2 manifest selects the attention registry, v2 cycle storage, and the 2,915-event execution profile. Its explicit `--host-input-bundle` avoids dependence on historical raw archives. An optional immutable `--shared-assets PROVIDER --shared-assets-sha256 SHA256SUMS_DIGEST` can share exact assets from an existing qualified bundle; it hardlinks them, so the provider must be on the same filesystem as `--out`.

An opt-in fusion profile such as `fc1-bias-gelu-fp8` needs a capture made with the same `--fusion-set`, and the same `--fusion-set` on both the exporter and the builder. The bundle provenance then carries the profile, so `package.py` and `build-smolvla.sh` need no extra option.

## SIMS packaging and cross compilation

Set `PLATFORM_ROOT` to a physical board-support source tree with the penguin26 platform and glossy startup, and `RISCV_BIN` to the GCC toolchain bin directory. The maintained integration templates, linker transformation, and blob extractor ship in `tools/smol_sims/`. A custom template can be selected explicitly with `--template`; there is no machine-local KIT lookup.

```sh
"$PY" -B "$ATLAS/tools/smol_sims/package.py" \
  --bundle "$WORK/bundle" --platform-root "$PLATFORM_ROOT" --build-work "$WORK/bundle-build" --out "$WORK/package"
SMOLVLA_PLATFORMS=SIMS SMOLVLA_PHASE=42 \
  bash "$ATLAS/tools/smol_sims/build-smolvla.sh" "$WORK/package" "$PLATFORM_ROOT" "$WORK/sims-42" "$RISCV_BIN"
SMOLVLA_PLATFORMS=SIMS SMOLVLA_PHASE=199 \
  bash "$ATLAS/tools/smol_sims/build-smolvla.sh" "$WORK/package" "$PLATFORM_ROOT" "$WORK/sims-199" "$RISCV_BIN"
SMOLVLA_PLATFORMS=SIMS \
  bash "$ATLAS/tools/smol_sims/build-smolvla.sh" "$WORK/package" "$PLATFORM_ROOT" "$WORK/sims-full" "$RISCV_BIN"
```

The package hardlinks the standalone ELF and extracts content-addressed assembler inputs directly from that ELF, checking each extracted symbol's SHA-256 filename. Source and output directories must be distinct, and each work directory must be fresh. The copied unused smoke target is disabled so an unrelated historical smoke package cannot become a build dependency. Both production graph compilation paths define `SMOL_ATTENTION_RUNTIME=1` and link the single authoritative atlas runtime registry. The SIMS link separates RX code, R constants/assets, and RW state, retains the low arena base, and reserves a 64 KiB stack. Cross compilation alone establishes no new numerical or hardware qualification. Preserve the strict original-model audit plus final gate, actual guest replay, layout audits, and their stated evidence scope before shipping ELFs.

## Fast immutable-weight SIMS builds

Fast mode keeps every program upload, dynamic A pack, output poison, decode, and output validation, but projection launches read their FP8 B weights in place from the immutable assets segment instead of copying them into the arena, and the verification-only A, B, and guard scans are skipped. `tools/smol_sims/generate_fast.py` regenerates each default projection program, checks it against the bundle, and reissues it with an `atlas_weight_source.WeightSource` bound to the weight symbol's address in a linked default SIMS ELF. The assets segment has a linker-fixed address, so one generation from the default full ELF serves the full, 42, and 199 variants of the same package; the runtime rejects a table whose encoded weight base differs from the linked panel before START.

```sh
"$PY" -B "$ATLAS/tools/smol_sims/generate_fast.py" --bundle "$WORK/bundle" \
  --elf "$WORK/sims-full/platform_root/build-smolvla-sims/penguin26-tests/atlas_delivery_smolvla/atlas-delivery-smolvla.elf" \
  --model-root "$MODEL" --out "$WORK/fast-assets"
SMOLVLA_PLATFORMS=SIMS SMOLVLA_PHASE=42 SMOLVLA_FAST_ASSETS="$WORK/fast-assets" \
  bash "$ATLAS/tools/smol_sims/build-smolvla.sh" "$WORK/package" "$PLATFORM_ROOT" "$WORK/fast-42" "$RISCV_BIN"
SMOLVLA_PLATFORMS=SIMS SMOLVLA_PHASE=199 SMOLVLA_FAST_ASSETS="$WORK/fast-assets" \
  bash "$ATLAS/tools/smol_sims/build-smolvla.sh" "$WORK/package" "$PLATFORM_ROOT" "$WORK/fast-199" "$RISCV_BIN"
SMOLVLA_PLATFORMS=SIMS SMOLVLA_FAST_ASSETS="$WORK/fast-assets" \
  bash "$ATLAS/tools/smol_sims/build-smolvla.sh" "$WORK/package" "$PLATFORM_ROOT" "$WORK/fast-full" "$RISCV_BIN"
```

`SMOLVLA_FAST_ASSETS` turns on the template's `SMOLVLA_FAST` option, which compiles `smol_fast_assets.c` and `smol_fast_images.S` from that directory and defines `ATLAS_FAST_MODE=1`; without it the default build is unchanged. The generated addresses belong to that package's SIMS layout and are not portable to another platform, linker script, or bundle. The generator accepts only default-profile bundles and refuses a fusion profile.

## Original-model backfill and safe resume

`tools/smol_capture_backfill.py` is the maintained original-model backfill entrypoint. It shares the strict per-receipt checks in `tools/smol_capture_audit.py` between resume and final aggregation. A reusable PASS must match the current runner, model harness, clean pinned model checkout, frozen event, runtime completion program, CSR sequence, exact full arena, output, and DMA. A separate receipt-bound worker completion record must contain an integer zero subprocess return code. Stale receipts are removed before dispatch, and a previous summary is replaced with INCOMPLETE before work starts.

```sh
"$PY" -B "$ATLAS/tools/smol_capture_backfill.py" \
  --python "$PY" --model-root "$MODEL" \
  --ledger "$WORK/capture/run/device-events.jsonl" --out "$WORK/original-audit" --workers 40
```

The default complete inventory is 2,915 launches, including 2,715 projections and 200 attention launches. An explicit `--expected-launches` supports small isolated validation inventories and reports that scope. A partial capture cannot produce a complete gate. `--watch` follows an actively appended capture ledger. The output `gate.json` is a strict backfill admission for the declared inventory, not an ELF or end-to-end hardware gate.
