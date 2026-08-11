"""The canonical Triton kernels every arm of the frontend work compiles, and their specs.

There is exactly one copy of each kernel because the claim under test is that the *same* source
descends to RVV, to Gemmini and to Radiance with nothing changed but the target. A per-bucket copy
would let the arms drift and quietly turn that claim into three separate results.

The kernels live in a real file on disk (not a string) because ``@triton.jit`` reads the decorated
function's source with ``inspect.getsourcelines``; an ``exec``-ed definition raises
``@jit functions should be defined in a Python file``.

Nothing here imports ``merlin.triton.source`` or triton internals at module scope, so a bucket
without the ``triton`` extra installed can still import this module to reach the specs.
"""
from __future__ import annotations

import importlib.util

from merlin.triton.spec import GridSpec, KernelArg, TritonKernelSpec

HAS_TRITON = importlib.util.find_spec("triton") is not None

# ``tl.dot`` rejects anything smaller: "Input shapes should have M >= 1, N >= 1 and K >= 32". So the
# smallest legal single tile is 16x32x16, and that is a hard floor on what the one-tile accelerator
# proof can mean — not a number chosen for convenience.
TILE_M, TILE_N, TILE_K = 16, 16, 32

VECTOR_ADD_N = 1024
VECTOR_ADD_BLOCK = 256

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def vector_add(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        """Elementwise add with a masked tail — the smallest kernel that is not matmul-family."""
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, x + y, mask=mask)

    @triton.jit
    def matmul_one_tile(a_ptr, b_ptr, c_ptr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """One static i8 x i8 -> i32 tile: no grid, no tails, no epilogue, static strides.

        Deliberately the exact shape the staged pipeline accepts today, so that a failure on an
        accelerator attributes to the target descent rather than to an unsupported access pattern.
        """
        offs_m = tl.arange(0, BM)
        offs_n = tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * BK + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * BN + offs_n[None, :])
        tl.store(c_ptr + offs_m[:, None] * BN + offs_n[None, :], tl.dot(a, b, out_dtype=tl.int32))

    @triton.jit
    def matmul_one_tile_f32(a_ptr, b_ptr, c_ptr, BM: tl.constexpr, BN: tl.constexpr,
                            BK: tl.constexpr):
        """The f32 twin of :func:`matmul_one_tile`, for targets whose unit is float, not integer."""
        offs_m = tl.arange(0, BM)
        offs_n = tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * BK + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * BN + offs_n[None, :])
        tl.store(c_ptr + offs_m[:, None] * BN + offs_n[None, :], tl.dot(a, b, out_dtype=tl.float32))
else:  # pragma: no cover - exercised only where the optional extra is absent
    vector_add = matmul_one_tile = matmul_one_tile_f32 = None


def vector_add_spec(n: int = VECTOR_ADD_N, block: int = VECTOR_ADD_BLOCK) -> TritonKernelSpec:
    """Spec for :func:`vector_add` over ``n`` elements; ``n`` need not be a multiple of ``block``."""
    return TritonKernelSpec(
        function=vector_add,
        args=(
            KernelArg("x_ptr", "pointer", "fp32", shape=(n,), effect="read"),
            KernelArg("y_ptr", "pointer", "fp32", shape=(n,), effect="read"),
            KernelArg("out_ptr", "pointer", "fp32", shape=(n,), effect="write"),
            KernelArg("n_elements", "scalar", "i32"),
        ),
        grid=GridSpec(dims_fn=lambda ce, rt: (-(-rt["n_elements"] // ce["BLOCK_SIZE"]),)),
        constexprs={"BLOCK_SIZE": block},
    )


def matmul_one_tile_spec(dtype: str = "i8", acc_dtype: str = "i32") -> TritonKernelSpec:
    """Spec for the one-tile matmul; ``dtype='fp32'`` selects the float twin."""
    fn = matmul_one_tile_f32 if dtype == "fp32" else matmul_one_tile
    return TritonKernelSpec(
        function=fn,
        args=(
            KernelArg("a_ptr", "pointer", dtype, shape=(TILE_M, TILE_K), effect="read"),
            KernelArg("b_ptr", "pointer", dtype, shape=(TILE_K, TILE_N), effect="read"),
            KernelArg("c_ptr", "pointer", acc_dtype, shape=(TILE_M, TILE_N), effect="write"),
        ),
        grid=GridSpec(dims=(1,)),
        constexprs={"BM": TILE_M, "BN": TILE_N, "BK": TILE_K},
    )
