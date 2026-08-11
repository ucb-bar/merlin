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
    def repeated_rhs_matmul(a0_ptr, a1_ptr, w_ptr, c0_ptr, c1_ptr, BM: tl.constexpr,
                            BN: tl.constexpr, BK: tl.constexpr):
        """Two activations against one immutable weight — Merlin's own reference workload.

        Written in Triton so that the accelerator arms compile the workload their target packages
        are certified on: one shared right-hand side is what makes the weight worth making resident,
        and a weight-stationary array's driver is built around exactly that. It is also the pair for
        the convergence test, where the hand-authored `build_input_module(reuse=2)` must produce the
        same interface, target and command buffer.
        """
        offs_m = tl.arange(0, BM)
        offs_n = tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        w = tl.load(w_ptr + offs_k[:, None] * BN + offs_n[None, :])
        a0 = tl.load(a0_ptr + offs_m[:, None] * BK + offs_k[None, :])
        a1 = tl.load(a1_ptr + offs_m[:, None] * BK + offs_k[None, :])
        out = offs_m[:, None] * BN + offs_n[None, :]
        tl.store(c0_ptr + out, tl.dot(a0, w, out_dtype=tl.int32))
        tl.store(c1_ptr + out, tl.dot(a1, w, out_dtype=tl.int32))

    @triton.jit
    def batched_matmul(a_ptr, b_ptr, c_ptr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """One independent matmul per program, over disjoint slices.

        Every pointer IS covered exactly once by the launch, so the addressing analysis is happy —
        and whole-tensor normalization would still be wrong, because a stack of small products is
        not one big product. This is what makes the contraction-under-a-grid guard load-bearing
        rather than theoretical.
        """
        pid = tl.program_id(axis=0)
        offs_m = tl.arange(0, BM)
        offs_n = tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        a = tl.load(a_ptr + pid * BM * BK + offs_m[:, None] * BK + offs_k[None, :])
        b = tl.load(b_ptr + pid * BK * BN + offs_k[:, None] * BN + offs_n[None, :])
        acc = tl.dot(a, b, out_dtype=tl.float32)
        tl.store(c_ptr + pid * BM * BN + offs_m[:, None] * BN + offs_n[None, :], acc)

    @triton.jit
    def vector_add_unmasked(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        """The same add with the bounds check removed — correct only when BLOCK divides the extent.

        Kept as a fixture because it is the sharpest test of the coverage analysis: it must be
        ACCEPTED when the block tiles the tensor exactly and REFUSED when it would run past the end.
        A bridge that ignored masks would accept both.
        """
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(out_ptr + offsets, tl.load(x_ptr + offsets) + tl.load(y_ptr + offsets))

    @triton.jit
    def transposed_store(x_ptr, out_ptr, BM: tl.constexpr, BN: tl.constexpr):
        """Reads row-major, writes column-major: full coverage, wrong order."""
        offs_m = tl.arange(0, BM)
        offs_n = tl.arange(0, BN)
        x = tl.load(x_ptr + offs_m[:, None] * BN + offs_n[None, :])
        tl.store(out_ptr + offs_m[:, None] + offs_n[None, :] * BM, x)

    @triton.jit
    def atomic_add_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        """An op the bridge has no translation for — it must say so rather than approximate."""
        offsets = tl.arange(0, BLOCK_SIZE)
        tl.atomic_add(out_ptr + offsets, tl.load(x_ptr + offsets))

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
    vector_add_unmasked = transposed_store = atomic_add_kernel = batched_matmul = None
    repeated_rhs_matmul = None


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
        # The extent reaches the kernel as a runtime scalar, so the bridge cannot see that it equals
        # the declared shape unless the caller says so. Declaring it is what lets the masked tail be
        # checked instead of assumed.
        assumptions={"n_elements": n},
    )


def vector_add_unmasked_spec(n: int, block: int = VECTOR_ADD_BLOCK) -> TritonKernelSpec:
    """Spec for :func:`vector_add_unmasked` — legal only when ``block`` divides ``n``."""
    return TritonKernelSpec(
        function=vector_add_unmasked,
        args=(
            KernelArg("x_ptr", "pointer", "fp32", shape=(n,), effect="read"),
            KernelArg("y_ptr", "pointer", "fp32", shape=(n,), effect="read"),
            KernelArg("out_ptr", "pointer", "fp32", shape=(n,), effect="write"),
        ),
        grid=GridSpec(dims=(-(-n // block),)),
        constexprs={"BLOCK_SIZE": block},
    )


def repeated_rhs_matmul_spec(m: int = TILE_M, k: int = TILE_K, n: int = TILE_N,
                            dtype: str = "i8", acc_dtype: str = "i32") -> TritonKernelSpec:
    """Spec for :func:`repeated_rhs_matmul`.

    Argument order matches `build_input_module(reuse=2)` exactly — activations, then the shared
    weight, then the outputs — so the two frontends are comparable value by value.
    """
    return TritonKernelSpec(
        function=repeated_rhs_matmul,
        args=(
            KernelArg("a0_ptr", "pointer", dtype, shape=(m, k), effect="read"),
            KernelArg("a1_ptr", "pointer", dtype, shape=(m, k), effect="read"),
            KernelArg("w_ptr", "pointer", dtype, shape=(k, n), effect="read"),
            KernelArg("c0_ptr", "pointer", acc_dtype, shape=(m, n), effect="write"),
            KernelArg("c1_ptr", "pointer", acc_dtype, shape=(m, n), effect="write"),
        ),
        grid=GridSpec(dims=(1,)),
        constexprs={"BM": m, "BN": n, "BK": k},
    )


def batched_matmul_spec(batch: int = 2) -> TritonKernelSpec:
    """Spec for :func:`batched_matmul`: ``batch`` independent TILE_M x TILE_K x TILE_N products."""
    return TritonKernelSpec(
        function=batched_matmul,
        args=(
            KernelArg("a_ptr", "pointer", "fp32", shape=(batch * TILE_M, TILE_K), effect="read"),
            KernelArg("b_ptr", "pointer", "fp32", shape=(batch * TILE_K, TILE_N), effect="read"),
            KernelArg("c_ptr", "pointer", "fp32", shape=(batch * TILE_M, TILE_N), effect="write"),
        ),
        grid=GridSpec(dims=(batch,)),
        constexprs={"BM": TILE_M, "BN": TILE_N, "BK": TILE_K},
    )


def transposed_store_spec(m: int = 8, n: int = 4) -> TritonKernelSpec:
    return TritonKernelSpec(
        function=transposed_store,
        args=(
            KernelArg("x_ptr", "pointer", "fp32", shape=(m, n), effect="read"),
            KernelArg("out_ptr", "pointer", "fp32", shape=(n, m), effect="write"),
        ),
        grid=GridSpec(dims=(1,)),
        constexprs={"BM": m, "BN": n},
    )


def atomic_add_spec(n: int = 64) -> TritonKernelSpec:
    return TritonKernelSpec(
        function=atomic_add_kernel,
        args=(
            KernelArg("x_ptr", "pointer", "fp32", shape=(n,), effect="read"),
            KernelArg("out_ptr", "pointer", "fp32", shape=(n,), effect="readwrite"),
        ),
        grid=GridSpec(dims=(1,)),
        constexprs={"BLOCK_SIZE": n},
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
