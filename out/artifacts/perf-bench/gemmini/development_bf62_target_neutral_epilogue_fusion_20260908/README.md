# Merlin Phase-2 compiler checkpoint: native scalar epilogue development

The current candidate, its exact 22.98% whole-model Spike proxy improvement, and its q534
**1,537,416,019-cycle exact FireSim result** are documented in
[NATIVE_SCALAR_EPILOGUE.md](NATIVE_SCALAR_EPILOGUE.md). The text below is retained as the
provenance report for the slot-complete plus reusable-host-workspace predecessor on which it is
built.

## Predecessor report

This fresh isolated artifact combines two target-general compiler repairs:

1. Every overlapping Gemmini `LOOP_WS` launch emits a complete slot-local descriptor. Ordinary
   global `CONFIG_EX/LD/ST` values remain cached; no blanket serialization was introduced.
2. Multi-element host-lane tensor temporaries use one 64-byte-aligned internal workspace with
   ordered-host-segment lifetime reuse. Only bounded scalar temporaries remain on the stack, and
   no workspace pointer was added to the ABI.

The full prepared ResNet-50 graph compiles with 1,240 source operations, 109 tasks, 393 ABI
pointers, 3,787 complete `LOOP_WS` descriptors, and the same 1,050 dependency fences as the
descriptor-complete source. The command buffer is byte-for-byte equivalent after removing only
the new `params.host_storage` receipt.

The compiler-reported static frame is 816 bytes under the 65,536-byte target policy. It is 32
bytes above the workspace-only 784-byte result because complete descriptor emission slightly
changes register pressure, but it has 64,720 bytes of policy headroom. An exact local run using
Gemmini's normal 128 KiB bare-metal stack completed one untimed warm invocation, reset the mutable
arena outside timing, then completed one measured invocation at 976,190,118 Spike cycles. All
1,000 logits matched exactly (`bad=0`, `nonfinite=0`, top-1 258, checksum
`c6e777c3fe0aae90`). Spike is functional/comparative proxy evidence, not FPGA performance
evidence; this combined artifact has not been run in FireSim.

## Verify the immutable compiler snapshot

From this directory:

```sh
./verify.py
./validation/verify_combined.py
./verify_native_scalar_epilogue.py

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/compiler:$(git rev-parse --show-toplevel)/merlin/python" \
  $(git rev-parse --show-toplevel)/.venv/bin/python -m pytest -q \
  tests/test_reduction_resident_loop_ws.py \
  tests/test_host_workspace.py
```

The expected current compiler tree is
`0b1e14e9b61dd3a4eb3472a87702cc53c5a05b98198c08cec3bfbce85363db8a` over 40 files.
The bundled Merlin support snapshot is
`0714adc43f25efb2421afc37519190f9a2b58c75ad75d7a08b9ff3537936498a` over 1,016 files.

## Invoke the optimized compiler directly

`run-gemmini-opt` pins this artifact's compiler and bundled support snapshot ahead of the mutable
working tree. Point `MERLIN_PYTHON` at a Python environment containing xDSL and the repository's
declared dependencies:

```sh
artifact=/scratch/agustin/projects/oscar-merlin/out/artifacts/perf-bench/gemmini/development_bf62_target_neutral_epilogue_fusion_20260908
input=/absolute/path/to/model.mlir
output_dir=/absolute/path/to/output
mkdir -p "$output_dir"

MERLIN_PYTHON=/scratch/agustin/projects/oscar-merlin/.venv/bin/python \
  "$artifact/run-gemmini-opt" \
  --source-convolution \
  --convert-iface-to-gemmini \
  --emit-command-buffer="$output_dir/command_buffer.json" \
  --emit-target-artifact \
  -o "$output_dir/target.mlir" \
  "$input"
```

For already-normalized interface input that must not enable source-convolution recognition, omit
`--source-convolution`. The four raw entrypoint templates remain in `compiler/manifest.yaml`.

The command above emits the two compiler-owned products needed by the runner:

- `target.mlir`: LLVM-dialect target module defining `@gemmini_kernel`.
- `command_buffer.json`: whole-program ABI, tensor bindings, task plan, storage encodings, and the
  deterministic `params.host_storage` receipt.

Native object/ELF construction remains runner-owned because it needs the selected target's pinned
LLVM, RISC-V ISA, linker script, harness ABI, and hardware facts. In this checkout, the standard
path is `merlin.targetgen.contract.compile.llvm_mlir_to_object(..., target="gemmini")` followed by
the target-owned harness/link recipe; that path enforces the 64 KiB kernel-frame policy and writes
`kernel.stack_frame.json`.

## Evidence and limitations

Machine-readable evidence is in [validation/combined_receipt.json](validation/combined_receipt.json).
The full generated ResNet products are in `validation/resnet50/`; the exact local console and
standard-stack ELF are in `validation/resnet50/local_spike/`.

The current compiler recompiled all four portfolio graphs. ResNet-50 passed exact warm/measured
Spike and external sibling package q534 passed exact warm/measured FireSim; TinyLLaMA, LSTMNetViT,
and SmolVLA preserve their prior command-buffer identities. The q534 package and receipt remain
separate immutable evidence and are hash-bound by `verify_native_scalar_epilogue.py`.
