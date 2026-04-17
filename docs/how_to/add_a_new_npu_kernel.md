# How To: Add a New NPU Kernel

This walkthrough shows how to add a new per-tile kernel to the
`npu_model.kernel_library` subpackage, wire a golden fixture + Program
class, and verify the kernel in both merlin's and npu_model's test
suites. See
[`docs/architecture/npu_compilation_pipeline.md`](../architecture/npu_compilation_pipeline.md)
for the underlying model.

## When to follow this guide

  * Adding a new **per-tile kernel** (32×32 atomic op) — this guide.
  * Adding a new **full-layer program** built from existing per-tile
    kernels — skip to [Full-layer programs](#full-layer-programs).
  * Changing the **compiler** side (a new tiling pattern, new IR op) —
    see `compiler/src/merlin/Dialect/NPU/` and its tests.

## Repo layout

```
merlin/
├── compiler/src/merlin/Dialect/NPU/scripts/
│    ├── gen_matmul_acc_variants.py       — manifest-entry generator
│    ├── gen_attention_acc_variants.py    — (ditto)
│    ├── annotate_kernel_patch_points.py  — fills dram_in/out patch_points
│    ├── fix_manifest_addi_overflow.py    — lui/addi split for large imms
│    └── run_kernel_golden_tests.py       — per-kernel test runner
└── third_party/npu_model/npu_model/kernel_library/      ← canonical home
     ├── manifest.json                 ← per-tile kernel ISA database
     ├── manifest_loader.py            ← load_kernel, load_imm
     ├── stitch.py                     ← stitcher + stitch_attention_chain
     ├── attention_acc_kernels.py      ← flash-attention body generators
     ├── fixtures.py                   ← KernelFixture registry
     ├── <kernel>.py (matmul, silu, …) ← thin wrappers returning load_kernel(name)
     └── tests/                        ← pytest suite runnable in npu_model CI
          ├── test_manifest_round_trip.py
          ├── test_kernel_goldens.py
          └── test_chains.py
```

## Step 1 — Write the kernel body

Pick a home under `npu_model/kernel_library/`. For generated-from-body
kernels (e.g. matmul K-accumulator variants), write a generator
function in Python that returns `list[Instruction]`. For hand-written
bodies, keep the explicit instruction stream.

Minimal example — a hypothetical absolute-value kernel on a 32×32 bf16
tile (split halves, reading from DRAM 0x0, writing to DRAM 0xb00):

```python
# third_party/npu_model/npu_model/kernel_library/vabs.py
from typing import Any
from npu_model.software import Instruction
from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs

def test() -> list[Instruction[Any]]:
    insts: list[Instruction[Any]] = []
    # Scalar setup: VMEM and DRAM addresses, transfer size.
    insts += [
        Instruction("lui",  ScalarArgs(rd=1, imm=0x2)),   # x1 = 0x2000 (VMEM in)
        Instruction("lui",  ScalarArgs(rd=2, imm=0x3)),   # x2 = 0x3000 (VMEM out)
        Instruction("addi", ScalarArgs(rd=3, rs1=0, imm=0)),       # x3 = 0 (DRAM in)
        Instruction("lui",  ScalarArgs(rd=4, imm=0x1)),            # x4 = 0x1000
        Instruction("addi", ScalarArgs(rd=4, rs1=4, imm=-1280)),   # x4 = 0xb00 (DRAM out)
        Instruction("addi", ScalarArgs(rd=5, rs1=0, imm=2047)),
        Instruction("addi", ScalarArgs(rd=5, rs1=5, imm=1)),       # x5 = 2048 (transfer size)
    ]
    # DMA load + vloads + vabs.bf16 (hypothetical) + vstores + DMA store.
    insts += [
        Instruction("dma.config.ch<N>", DmaArgs(channel=0)),
        Instruction("dma.wait.ch<N>",   DmaArgs(channel=0)),
        Instruction("dma.load.ch<N>",   DmaArgs(rd=1, rs1=3, rs2=5, channel=0)),
        Instruction("dma.wait.ch<N>",   DmaArgs(channel=0)),
        Instruction("vload",  VectorArgs(vd=0, rs1=1, imm12=0)),
        Instruction("vload",  VectorArgs(vd=1, rs1=1, imm12=32)),
        # ... vabs.bf16 for each half (not a real ISA op — illustrative) ...
        Instruction("vstore", VectorArgs(vd=0, rs1=2, imm12=0)),
        Instruction("vstore", VectorArgs(vd=1, rs1=2, imm12=32)),
        Instruction("dma.store.ch<N>", DmaArgs(rd=4, rs1=2, rs2=5, channel=0)),
        Instruction("dma.wait.ch<N>",  DmaArgs(channel=0)),
    ]
    return insts
```

## Step 2 — Register in the manifest

For a one-off generator like the example, just:

```bash
# Edit manifest.json directly, OR write a generator like
# compiler/src/merlin/Dialect/NPU/scripts/gen_matmul_acc_variants.py
```

For a complex body, prefer a generator script that emits a stable
instruction list and serializes to `manifest.json`. Example pattern:
`gen_attention_acc_variants.py` imports
`npu_model.kernel_library.attention_acc_kernels` and dumps each body
into the manifest.

## Step 3 — Populate patch_points

After any manifest edit:

```bash
conda run -n merlin-dev uv run python \
    compiler/src/merlin/Dialect/NPU/scripts/annotate_kernel_patch_points.py
```

This scans the manifest for each kernel's DMA ops and records the
scalar register chain that produces each DMA address. The stitcher
uses these to rewrite addresses per invocation.

## Step 4 — Fix `addi` overflow (if needed)

Any `addi` with an immediate outside the 12-bit signed range
`[-2048, 2047]` needs expansion to `lui + addi`:

```bash
conda run -n merlin-dev uv run python \
    compiler/src/merlin/Dialect/NPU/scripts/fix_manifest_addi_overflow.py
```

Safe for `rs1 == 0` cases. Leaves `rs1 == rd` cumulative bumps alone
(the autocomp generator relies on sign-extension there).

## Step 5 — Write a fixture

Add a `KernelFixture` entry to `fixtures.py`. The fixture records the
canonical input tensors, their DRAM offsets (matching the kernel's
`dram_in_*` from step 3), the expected output address, and a
torch reference:

```python
# Inside _build_registry() in npu_model/kernel_library/fixtures.py
torch.manual_seed(SEED + 999)
x = torch.randn(32, 32, dtype=torch.bfloat16)
x_h0, x_h1 = _split_halves(x)
expected = x.abs()

info = _manifest_kernel_info("vabs")  # reads patch_points from manifest
reg["vabs"] = KernelFixture(
    kernel="vabs",
    inputs=[(info["dram_in"][0], x_h0), (info["dram_in"][1], x_h1)],
    output_addr=info["dram_out"][0],
    expected=expected,
    description="y = |x| on a 32x32 bf16 tile (split halves)",
)
```

A `KernelGolden_vabs` Program class is auto-generated by
`npu_model/configs/programs/kernel_golden_suite.py` — no extra file
needed for the per-tile test.

## Step 6 — Run the tests

```bash
# Merlin side (uses compiler scripts):
conda run -n merlin-dev uv run python \
    compiler/src/merlin/Dialect/NPU/scripts/run_kernel_golden_tests.py

# npu_model side (pytest, runnable upstream in the submodule's CI):
cd third_party/npu_model
conda run -n merlin-dev uv run pytest npu_model/kernel_library/tests/ -x
```

Both should report PASS for the new kernel.

## Full-layer programs

For a multi-kernel schedule (e.g. a real transformer block):

1. **Decide the layer shape** (seq, hidden, head_dim). Divisible by 32
   where possible.
2. **Compose the schedule** via the stitcher. Use
   `stitch_attention_chain` for flash-attention K/V tiling,
   explicit `load_overrides`/`store_overrides` dicts for other
   composition (see `composition_smoke.py`, `composition_mini_block.py`,
   `matmul_k_tile_smoke.py` for patterns).
3. **Place inputs + weights** in `memory_regions = [(addr, tensor), ...]`.
4. **Set a golden_result** tuple `(output_addr, expected_tensor)`.
5. **Set a tolerance attribute** — the test harness honors one of
   `kernel_tolerance`, `chain_tolerance`, `composition_tolerance`
   (defaults to `(1e-2, 1e-2)`).

Example skeleton:

```python
class MyLayerProgram(Program):
    instructions = my_compose_fn(shape=...)
    memory_regions = [(DRAM_X, x), (DRAM_W, w), ...]
    golden_result = (DRAM_OUT, expected)
    composition_tolerance = (5e-2, 3.0)
```

Auto-discovered by `scripts/test_programs.py` via
`npu_model.configs.programs.__all__`.

## Memory layout + scheduling cheat sheet

See
[`docs/architecture/npu_compilation_pipeline.md`](../architecture/npu_compilation_pipeline.md)
for the authoritative table. Quick reminders:

  * **32 × 32 bf16 tile** = 2048 bytes, laid out as **two 32×16 halves**
    stacked 1024 B each (cols 0-15 first, then cols 16-31).
  * **32 × 32 fp8 tile** = 1024 bytes row-major, what matmul inputs expect.
  * **Column-blocked [128, 16]** — four 32×16 halves stacked; attention
    uses this for 32×64 Q / K^T / V tiles.
  * **imm12 on vload/vstore** is in **units of 32 bytes** → `imm12=32`
    means +1024 bytes = one half.
  * **MXU accumulator** is persistent across invocations; matmul chain
    variants exploit this. Attention chain spills state to DRAM instead.

## Running long simulations

For programs with ≥100k cycles (multi-block workloads), use
`--fast-sim` to skip per-cycle JSON trace writing:

```bash
conda run -n merlin-dev uv run python \
    third_party/npu_model/scripts/test_programs.py --fast-sim --max-cycles 10000000
```

Numerics are unaffected; wall-clock scales proportionally to the
amount of I/O skipped.
