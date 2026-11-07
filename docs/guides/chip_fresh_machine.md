---
title: Start a Kodiak or Iris kernel project on a new machine
kind: guide
status: current
owner: targetgen
last_verified: 2026-11-07
related: [chip_packages, model2mlir, zephyr]
code_refs: [examples/kodiak/run.sh, examples/iris/run.sh, examples/lib/chip_setup.py, examples/lib/chip_flow.py, examples/kodiak/compile_elf.py]
---

# Start a Kodiak or Iris kernel project on a new machine

This guide starts with a fresh Linux checkout. Merlin owns scalar compilation and
the build driver, its pinned `model2MLIR` checkout captures models, `rvv-mlir`
lowers the vector fallback, and the selected chip package supplies the OPU or
VME kernel, hardware contract, and board. Setup fetches the pinned components;
clone a chip package separately when editing its kernels.

## Set up the compiler

Install `git`, `uv`, `make`, `g++`, and `dtc`. Have LLVM/MLIR 23 at the
[Merlin source pin](llvm_toolchain.md), Chipyard with `riscv-tools`, and Zephyr
SDK 0.17.0 available. These large tools are found and checked by setup, which
prints commands for anything missing. The chip's RTL and Spike source must also
match the pins reported by setup.

```sh
git clone --branch feat/OPU-VME https://github.com/ucb-bar/merlin.git
cd merlin
examples/kodiak/run.sh setup --llvm /path/to/llvm-install \
  --chipyard /path/to/chipyard --zephyr-sdk /path/to/zephyr-sdk-0.17.0
examples/kodiak/run.sh preflight --require package --require-pin saturn_opu_int8
```

For Iris, run `examples/iris/run.sh setup` and
`examples/iris/run.sh preflight --require package` instead. Iris setup also
checks out its pinned Chipyard source and Zvt Spike source, then builds that
Spike. Re-run setup after following any printed missing-prerequisite command.
An exit status of zero and a passing preflight are the setup gate. Setup writes
paths to the ignored `.env`; an exported shell variable takes precedence, so
remove stale exports reported with `!` by preflight.

Setup fetches the exact `model2MLIR`, RVV, and chip-package revisions Merlin
pins. A separate model2MLIR clone is useful for frontend development but is
not needed to compile models. A different model2MLIR revision must be
explicitly pinned and recaptured; pointing setup at an arbitrary checkout
will fail its capture pin check.

## Obtain or capture a model

The built-in operator loop needs no external model or checkpoint:

```sh
examples/kodiak/run.sh op gemm 64 64 64
# or: examples/iris/run.sh op gemm 64 64 64
```

For SpectFormer, clone the [upstream source](https://github.com/badripatro/SpectFormers)
and set `SPECTFORMER_DIR` to that checkout. The source repository does not
provide the trained checkpoint used by the published result. Obtain that
checkpoint from its owner and set `SPECTFORMER_CKPT` to the file. Without it,
the loader reports **random initialization**: useful for compiler smoke
testing, not a trained-model or accuracy result.

```sh
export SPECTFORMER_DIR=/path/to/SpectFormers
export SPECTFORMER_CKPT=/path/to/checkpoint_best.pth
test -f "$SPECTFORMER_CKPT"
examples/kodiak/run.sh capture spectformer
# or: examples/iris/run.sh capture spectformer
```

For the published trained-model source identity, verify the checkpoint
SHA-256 in the [SpectFormer source contract](../../examples/kodiak/TORCHAO_W8A8_SOURCE.md).
Another checkpoint can be compiled, but its outputs require a new numerical
gate.

For TinyLlama, preload the public
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` checkpoint at revision
`fe8a4ea1ffedaf415f4da2f062534de366a451e6` into the same Hugging Face
cache used by the capture environment. The
[source-capture guide](../../examples/workloads/causal_lm_section/README.md)
gives the snapshot and file-hash checks. From the Merlin root after setup:

```sh
uv pip install --python out/build/venv/model2mlir/bin/python huggingface_hub
out/build/venv/model2mlir/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", revision="fe8a4ea1ffedaf415f4da2f062534de366a451e6")'
```

Check the downloaded `config.json` and `model.safetensors` hashes with
the source-capture guide before enabling offline mode. The full checkpoint's weights exceed
physical Kodiak's 512 MiB; this two-layer section keeps real pretrained
weights and the full layer shapes:

```sh
export CAUSAL_LM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
export CAUSAL_LM_REVISION=fe8a4ea1ffedaf415f4da2f062534de366a451e6
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CAUSAL_LM_LAYERS=2 CAUSAL_LM_ENTRY=embeds M2M_SEQ=64
examples/kodiak/run.sh capture tinyllama_l2_s64 \
  --loader examples/workloads/causal_lm_section/loader.py
```

Run the same capture command with `examples/iris/run.sh` for Iris. Use an
absolute loader path if calling from outside the Merlin checkout. A capture
writes `out/artifacts/recaptures/<name>_int8_full/` with MLIR, weights, inputs,
a capture receipt, and a W8A8 reference. The ordinary `int8` capture starts
from weight-only TorchAO quantization; Merlin's W8A8 stage supplies the device
integer arithmetic. A separate true-source TorchAO W8A8 experiment has its
own [source contract](../../examples/kodiak/TORCHAO_W8A8_SOURCE.md) and must not
be conflated with this built-in capture. A new capture has a new identity and
needs its own numerical gate.

## Produce an ELF and iterate on a kernel

Kodiak's compile command builds one Zephyr ELF and a `.build.json` receipt;
it does not run Spike, FireSim, or the physical chip:

```sh
examples/kodiak/run.sh compile spectformer --opu --output /path/to/spectformer.elf
examples/kodiak/run.sh compile tinyllama_l2_s64 --opu --output /path/to/tinyllama.elf
examples/kodiak/run.sh compile spectformer --opu --board firesim_kodiak_opu \
  --output /path/to/spectformer-firesim.elf
```

The physical and FireSim boards have different memory maps. Physical Kodiak
execution is not currently qualified by the independent BootAddr positive
control. The [Kodiak example](../../examples/kodiak/README.md) explains the
numerical Spike gate and the separate FireSim job flow. The [Iris example]
(../../examples/iris/README.md) gives its direct `merlin-compile` command and
Zvt Spike gate; no Iris FPGA image is claimed.

To edit a kernel, clone [Kodiak-MLIR](https://github.com/ucb-bar/kodiak-mlir)
or [Iris-MLIR](https://github.com/ucb-bar/iris-mlir) at
`handwritten-implementation` with `--recurse-submodules`, then select that
checkout and its `base/rvv-mlir` with `MERLIN_TARGET_PATH`. The package's
`kernels/kernels.yaml` registers a C function with the documented matrix
ABI. Test the C file or registry name first, then test a compiler-routed
convolution and a model:

```sh
examples/kodiak/run.sh kernel-test /path/to/my_kernel.c --quick
examples/kodiak/run.sh op conv2d 16 64 16 3 --matrix-kernel /path/to/my_kernel.c
# Iris uses examples/iris/run.sh with the same kernel-test and op syntax.
```

The [Kodiak kernel guide](../../examples/kodiak/README.md) and
[Iris kernel guide](../../examples/iris/README.md) explain each unit's
generated header, edge tiles, signed-int8 arithmetic, registry constraints,
and where the routed ELF and report are kept. A passing `kernel-test`
certifies the selected kernel on its functional simulator; use the
compiler-routed operator or model gate to prove the compiler actually picked
it.
