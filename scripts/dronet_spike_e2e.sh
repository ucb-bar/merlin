#!/usr/bin/env bash
# Phase F end-to-end: dronet → kernel-embedded vmfb for bare-metal Spike.
#
# This script runs the discovery → compile path. The ELF-embedding step
# (turning the resulting .vmfb into a bare-metal RISC-V binary) reuses the
# existing firesim sample at samples/SaturnOPU/simple_embedding_ukernel/ —
# follow the .incbin pattern used by TinyLlama there to embed an
# externally-compiled vmfb (see tinyllama_vmfb_embed.S.in for the template).
#
# Usage:
#   ./scripts/dronet_spike_e2e.sh                  # full flow
#   ./scripts/dronet_spike_e2e.sh --discover-only  # just dump kernel stubs
#   ./scripts/dronet_spike_e2e.sh --compile-only   # just produce vmfb

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT="${OUT:-build/dronet_spike_e2e}"
KERNEL_DIR="benchmarks/SaturnOPU/kernels"

echo "════════════════════════════════════════════════════════════════"
echo " Step 1 — Discover dronet's op inventory + minimum-cover"
echo "════════════════════════════════════════════════════════════════"
conda run -n merlin-dev uv run python -m tools.kernels.discover \
    models/dronet/dronet.mlir \
    --target saturn_opu_spike --hw SPIKE \
    --output "/tmp/dronet_phase_f_kernels" \
    --minimum-cover --auto-fuse \
    2>&1 | tail -25

if [[ "${1:-}" == "--discover-only" ]]; then
  exit 0
fi

echo
echo "════════════════════════════════════════════════════════════════"
echo " Step 2 — Compile dronet to kernel-embedded vmfb"
echo "════════════════════════════════════════════════════════════════"
rm -rf "$OUT"
./merlin compile models/dronet/dronet.mlir \
    --target saturn_opu_spike --hw SPIKE \
    --kernels-dir "$KERNEL_DIR" \
    --dump-phases \
    --output-dir "$OUT" 2>&1 | tail -10

echo
echo "════════════════════════════════════════════════════════════════"
echo " Step 3 — Verify kernel rewrites + linked .o"
echo "════════════════════════════════════════════════════════════════"
echo
echo "Kernel call sites in dronet's flow phase:"
grep -oE "@call_saturnopu_[a-z_0-9]+" "$OUT/phases/dronet.6.flow.mlir" \
  | sort | uniq -c
echo
echo "Symbols linked into the vmfb:"
strings "$OUT/dronet.vmfb" | grep -E "kb_saturnopu|_workgroup$" | sort -u | head -20
echo
echo "vmfb size:"
ls -la "$OUT/dronet.vmfb"

if [[ "${1:-}" == "--compile-only" ]]; then
  exit 0
fi

echo
echo "════════════════════════════════════════════════════════════════"
echo " Step 4 — Embed vmfb into a bare-metal RISC-V ELF"
echo "════════════════════════════════════════════════════════════════"
echo
echo "The firesim sample at samples/SaturnOPU/simple_embedding_ukernel/"
echo "already builds bare-metal IREE+model ELFs via iree_cc_binary. To"
echo "embed THIS vmfb (with our kernels linked) follow the TinyLlama"
echo "pattern in CMakeLists.txt (tinyllama_vmfb_embed.S.in):"
echo
echo "    set(_DRONET_SPIKE_VMFB \"$REPO_ROOT/$OUT/dronet.vmfb\")"
echo "    configure_file(...)  # generate dronet_vmfb_embed.S"
echo "    add_custom_command(... incbin into .vmfb_data section ...)"
echo "    iree_cc_binary(NAME bench_dronet_spike SRCS device_embedded_sync.c"
echo "                   model_benchmark.c dronet_vmfb_loader.c"
echo "                   DEPS iree::base iree::hal iree::hal::drivers::"
echo "                        local_sync::sync_driver iree::hal::local::"
echo "                        loaders::embedded_elf_loader iree::modules::hal"
echo "                        iree::vm iree::vm::bytecode::module ...)"
echo
echo "Then build via:"
echo "    ./merlin build --profile firesim --cmake-target bench_dronet_spike"
echo
echo "And run on Spike:"
echo "    spike --isa=rv64gcv build/firesim-merlin-release/.../bench_dronet_spike"
echo
echo "Note: htif.ld provides the boot path; no pk needed. The"
echo "dronet.vmfb already contains the linked SaturnOPU kernel .o files."
