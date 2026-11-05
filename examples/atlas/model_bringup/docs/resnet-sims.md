# ResNet50 host package and ELFs

`tools/resnet_sims/` turns one fused ResNet50 device capture into board-support host ELFs for SIMS (HTIF console), CHIP (UART at 3686400 baud with tohost exit), and QEMU (validation console only, never for F2). The capture is the `model.json` written by Merlin's `examples/atlas/comparisons/refresh/capture_resnet_device.py` without `--prepare-only`, together with the program, arena, expected-output, input, bias, and logit payloads it names. The package embeds every captured program, initial arena, and expected output; the guest stages each launch, checks every output byte, and consumes the actual device bytes. The stem max pool is its own weightless launch (`third_party/atlas-kernels/kernels/resnet_pool/atlas_resnet_pool.py`): the guest stages the stem output like a direct input, and the device returns the exact FP8 byte maximum. Every convolution after the stem gathers its producer's FP8 rows on device (`third_party/atlas-kernels/kernels/resnet_patch_gather/atlas_resnet_input.py`); the 3x3 convolutions order K as (ky, kx, channel), with the captured weights permuted to match.

External inputs are the capture, a physical board-support tree with the penguin26 platform and glossy startup (`PLATFORM_ROOT`), the RISC-V GCC bin directory (`RISCV_BIN`), and, for fast mode, an npu_model checkout (`MODEL`) and its Python environment (`PY`). Nothing is executed; these steps only cross-compile.

```sh
python3 -B "$ATLAS/tools/resnet_sims/package.py" --model "$CAPTURE/model.json" --out "$WORK/package"
python3 -B "$ATLAS/tools/resnet_sims/build.py" --package "$WORK/package" --platform-root "$PLATFORM_ROOT" \
  --toolchain "$RISCV_BIN" --work "$WORK/default"
```

`build.py` copies the board-support tree into the fresh work directory, skipping `.git` and `build*` entries and refusing symlinks. It attaches only the ResNet50 test directory and disables the copied delivery targets, which may include machine-local smoke packages. ELFs are written to `$WORK/default/build-default-{sims,chip,qemu}/penguin26-tests/atlas_resnet50/atlas-resnet50.elf`. Set `RESNET50_BUILD_JOBS` to change build concurrency.

## Immutable-weight fast mode

The same package also builds with `RESNET50_FAST=ON`. Each fast program reads immutable B directly from its embedded read-only arena blob instead of from the staged arena copy. The host stages the rest of the captured arena, refreshes direct inputs and every dynamic auxiliary tensor, and keeps the complete output comparison. It skips the staged-arena preverification and the postlaunch preservation comparison. The host rejects a launch whose bound weight base differs from the linked blob.

```sh
"$PY" -B "$ATLAS/tools/resnet_sims/fast.py" --package "$WORK/package" --model "$CAPTURE/model.json" \
  --npu-model "$MODEL" --platform-root "$PLATFORM_ROOT" --toolchain "$RISCV_BIN" --work "$WORK/fast-qemu" --platform qemu
```

Bind each platform separately, because the linked weight addresses differ between platforms. `fast.py` copies the package and emits every fast program with the original x30/x31 hook scratch through `legacy_epilogue_assembly`, first against placeholder weight bases. It checks that the default emission still equals each captured program. It then rebuilds and regenerates against the linked B symbols until the ELF hash is stable, and records the bound bases and program hashes in `$WORK/fast-*/generated/relocations.json`. The final ELF is `$WORK/fast-*/build-fast-<platform>/penguin26-tests/atlas_resnet50/atlas-resnet50.elf`.

A binding is a fixed point of the link, and one link can have more than one. Weight bases set the length of each program's address constants, program lengths shift the data sections, and the global pointer then decides whether linker relaxation shortens the host text, which moves every embedded blob by one 64-byte alignment step. A fresh run always reaches the same fixed point. Pass `--seed-relocations` with an archived `relocations.json` to reproduce that archived binding instead; the run fails unless the seed bases regenerate the archived program hashes.
