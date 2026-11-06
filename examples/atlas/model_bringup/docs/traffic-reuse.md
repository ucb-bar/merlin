# Traffic reuse recipes

Use `atlas_program.assembly(..., traffic_reuse=True)` for the shapes in `_TRAFFIC_REUSE_MODULES`.

Admission requires integer dimensions, the original guarded output offset and an integer BF16 exponent in `[-126,127]`.

Unknown shapes and combinations with legacy residency or pretranspose flags fail closed; reuse defaults off.

| Recipe module suffix | Storage and reuse |
| --- | --- |
| `resident` | A and one transposed B column share bank 1; paired rows share each weight push. |
| `stream` | A fills bank 1 then bank 0; a transposed B column and output scratch use the remaining bank 0 space. |
| `tiny` | A fills bank 1; bank 0 holds one transposed B column and output scratch for two row pairs. |
| `tiny_breadth` | One or two resident rows use alternating 4-KiB B buffers with transpose at use. |
| `blocked` | Bank-bounded A blocks reuse transposed B columns across paired rows, including an odd tail. |
| `smol_connector` | One A row panel spans both banks; alternating B tile buffers stream the long reduction. |

Resident-column loaders prefetch only the next disjoint tile on channel 1 and wait before use.

For `12544/160/64`, blocked reuse loops over contiguous row pairs within one bank; cross-bank pairs remain explicit.

## Exactness and limits

Every accumulator visits K in increasing order: first overwrite, then accumulation, one BF16 pop and optional rescale.

The public API requires rescaling; direct stream and Tiny helpers also allow unscaled output.

One static `vli.all m4` supports the existing exponent patch protocol; m4/m5 never overlap popped output halves.

Operands, packed output order, the 4-GiB DRAM base and both 256-byte guards are preserved.

Recipes use only VMEM banks 0-1, bytes `[0,524288)`; VMEM operands count four-byte words and DRAM operands count bytes.

Each DMA or vector tile stays in one bank; buffers remain inaccessible until their owning DMA channel is waited.

Exclusive raw ownership and quiescence are required because compiled clients may allocate in the same banks.

| Consumer | Program limit |
| --- | --- |
| Raw host | Raw words plus five marker words must fit 32,768 words. |
| ResNet graph host | Actual raw bytes and descriptor counts must fit the 32,763-word copy buffer. |

All admitted programs fit these limits; builders validate actual bytes, structure and declared counts before linking.

Shared symbolic tests cover every public shape, ordered reductions, DMA ownership, bank limits and output halves.

Mutation controls guard waits, transpose, accumulation, rescale, buffer ownership and output placement.

Shared symbolic tests check tile origins and output halves; encoded tests guard the unique rescale patch site.

## Running

Follow [model bringup setup](../README.md#build), including bytecode suppression; export `ATLAS="$VIEW"`, `ATLAS_SOURCE_ROOT="$VIEW"`, and `ATLAS_MLIR_ROOT="$VIEW"`, and use `cd "$VIEW"`. Put `tools` on `PYTHONPATH` and use the same physical dimensions and exponent as the captured contraction. `import kernel_library` puts the generators in the execution view's pinned `third_party/atlas-kernels` library on `sys.path`. Use its source resolver for the relocated helper layout.

```python
import kernel_library
from atlas_program import assembly, dma_load_bytes
from kernel_library import code_sources

m, k, n = 128, 3072, 768
sources = code_sources(traffic_reuse=True)
source = assembly(m, k, n, m*k + k*n + 256, -15, traffic_reuse=True)
loads = dma_load_bytes(m, k, n, traffic_reuse=True)
assert code_sources(traffic_reuse=True) == sources
```

The source closure includes the recipe modules and their shared loader/output helper; retain it with generated programs.

Run the reuse gates with `ATLAS_NPU_MODEL_ROOT="$MODEL_ROOT" PYTHONPATH="$MODEL_ROOT":tools:test "$PYTHON" -m unittest discover -s "$KERNELS_ROOT/kernels/projection_gemm/tests" -p 'test_program_reuse*.py'`.

Run dispatch and capacity gates with `PYTHONPATH=tools:test "$PYTHON" -m unittest test_program_traffic_reuse`.

Run `ATLAS_NPU_MODEL_ROOT="$MODEL_ROOT" PYTHONPATH="$MODEL_ROOT":tools:test "$PYTHON" -m unittest test_graph_program_bounds` for graph and hardware bounds.

## Host handoff

`tools/chipyard/smol_layer.py --help` lists the checksum-bound inputs for a live seven-projection selected Smol layer.

The builder retains nonlinear C, checks complete physical DRAM after each projection and final output/key/value caches.

It emits `ATLAS_KERNEL_*` and `ATLAS_LAYER_*` records; build success alone does not qualify device execution.

See [host checking](host-iss.md) for the opt-in checker, stable-span preconditions and fail-closed behavior.

The selected Smol layer uses `runtime/smol_host_bits.h` for freestanding BF16 conversion. C union copies preserve FP32 representation and the BF16 rounding policy without `memcpy`. Widening signaling NaNs preserves sign and payload; narrowing sets the quiet bit. Run `PYTHONPATH=tools:test "$PYTHON" -m unittest test_host_bits` with an RV64 compiler to check all BF16 patterns, rounding boundaries, and undefined object symbols.
