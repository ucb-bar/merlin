"""External kernel table: the kernels model2MLIR emits as ``prov.kernel`` calls, and the bodies to link.

model2MLIR lowers a registered custom op (today: a Triton ``@triton_op``) to a ``func.call`` of a private,
body-less ``func.func`` tagged ``prov.kernel = "<ns>::<name>"`` (``m2m/ir/kernel_registry.py``). That call
is a placeholder: the real code is supplied here, at link time, for whatever machine is being compiled
for. This module is only the DATA — which kernel, in which file, launched how. ``kernel_link`` does the
work.

**Keyed by ``prov.kernel``, never by symbol.** m2m suffixes the declaration's symbol (``_1``, ``_2``, …)
each time the same kernel appears at a new shape, so the symbol is per-shape and the attribute is the
identity. Entries are keyed by the same qualname as m2m's ``KernelSpec``, and
``test_table_keys_match_m2m_registry`` keeps the two tables in step. m2m emits only the identity and the
operand binding; everything about the implementation (file, entry, launch) is recorded here.

**A body is the Triton kernel itself, in MLIR's ``llvm`` dialect.** The file is what triton-cpu emits for
the ``@triton.jit`` function the ``@triton_op`` launches — that function, not a CPU twin of it: its
``llvm.func`` keeps the Python name and Triton's specialization attributes (``tt.divisibility``).
Triton is how the kernel was written; nothing here runs on a GPU. ``kernel_link`` merges the
``llvm.func`` into the model's MLIR before lowering, so it is optimized and code-generated for the
model's target together with the model, and carries no triple of its own.

**``args`` + ``grid`` are the ``@triton_op`` wrapper's launch, as data.** The Python wrapper allocates
the result, computes the program grid with ``triton.cdiv``, and calls the kernel with each tensor's data
pointer and its runtime scalars, in the ``@triton.jit`` signature's order; triton-cpu appends six launch
integers (program id x/y/z, grid x/y/z) to every kernel. A body records the first two parts
(``args``, ``grid``); ``kernel_link.emit_wrappers`` turns them into the MLIR twin of that wrapper. The
kernel's own signature is the check: a wrapper whose call does not match the ``llvm.func`` fails MLIR
verification at link time, by name.

Invariants:

- Every body is portable: generic ``llvm`` dialect only — no arch intrinsic namespace, no inline asm, no
  address spaces. ``test_every_body_file_exists_and_is_portable`` checks the files.
- Each result appears exactly once in ``args`` (the wrapper returns what the kernel wrote there).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

__all__ = ["KERNEL_TABLE", "KernelArg", "KernelBody", "KernelEntry", "KernelTableError", "body_path", "lookup"]


class KernelTableError(LookupError):
    pass


@dataclass(frozen=True)
class KernelArg:
    """One argument of the kernel's ``llvm.func`` before triton-cpu's six launch integers, in order.

    ``kind``: ``operand`` (data pointer of the tensor operand named ``name`` — the names are m2m's
    ``prov.kernel_operands``), ``result`` (data pointer of result ``index``, which the wrapper
    allocates), ``dim`` (extent ``index`` of operand ``name``, passed as ``i32``: the type Triton gives an
    integer argument that fits in 32 bits).
    """

    kind: str
    name: str = ""
    index: int = 0


@dataclass(frozen=True)
class KernelBody:
    """The linkable implementation of a kernel: one triton-cpu export in MLIR's ``llvm`` dialect.

    ``mlir`` is relative to the ``merlin.kernels`` package directory; ``entry`` is the ``llvm.func`` it
    defines. ``grid`` has one ``(operand, axis, block)`` per program axis (at most three; a missing axis
    is 1): axis ``i`` runs ``ceil(operand.shape[axis] / block)`` programs — the wrapper's
    ``triton.cdiv``. ``specialization`` lists the extents the body was compiled for (``None`` = read at
    run time): Triton bakes ``tl.constexpr`` arguments into the code, so a use at another shape is
    refused instead of computed wrong.
    """

    mlir: str
    entry: str
    args: tuple[KernelArg, ...]
    grid: tuple[tuple[str, int, int], ...] = ()
    specialization: tuple[tuple[str, tuple[int | None, ...]], ...] = ()
    dtype: str = "f32"
    description: str = ""


@dataclass(frozen=True)
class KernelEntry:
    qualname: str
    body: KernelBody
    description: str = ""


# model2MLIR workloads/mlp_basic_triton/kernels.py: the `triton_exp::fused_mlp2` wrapper launches
#   _mlp2_kernel[(cdiv(M, BLOCK_M),)](x, w1, b1, w2, b2, out, M, K=K, H1=H1, H2=H2, BLOCK_M=16)
# K/H1/H2/BLOCK_M are constexpr, so they are compiled in and `specialization` pins them. Weights are in
# nn.Linear's [out, in] layout; the kernel loads them transposed itself.
_MLP2 = KernelBody(
    mlir="triton_llvmir/mlp2_wrapper_llvm.mlir",
    entry="_mlp2_kernel",
    args=(
        KernelArg("operand", "x"),
        KernelArg("operand", "w1"),
        KernelArg("operand", "b1"),
        KernelArg("operand", "w2"),
        KernelArg("operand", "b2"),
        KernelArg("result", index=0),
        KernelArg("dim", "x", 0),
    ),
    grid=(("x", 0, 16),),
    specialization=(("x", (None, 16)), ("w1", (32, 16)), ("b1", (32,)), ("w2", (32, 32)), ("b2", (32,))),
    description="triton-cpu llvm-dialect MLIR of _mlp2_kernel, before LLVM optimization",
)

_BUILTIN: tuple[KernelEntry, ...] = (
    KernelEntry(
        "triton_exp::fused_mlp2",
        body=_MLP2,
        description="linear+relu+linear+relu in one Triton kernel (model2MLIR workloads/mlp_basic_triton)",
    ),
)

KERNEL_TABLE: dict[str, KernelEntry] = {e.qualname: e for e in _BUILTIN}


def lookup(qualname: str) -> KernelEntry:
    """The entry for a ``prov.kernel`` name; raises naming every known kernel when there is none."""
    try:
        return KERNEL_TABLE[qualname]
    except KeyError:
        raise KernelTableError(
            f"no external kernel {qualname!r} in merlin's kernel table (known: {sorted(KERNEL_TABLE)}); "
            "add an entry to src/merlin/llvmlower/kernel_table.py"
        ) from None


def body_path(body: KernelBody) -> Path:
    """Where the body's MLIR lives. Read through the package (works installed or in a checkout)."""
    return Path(str(resources.files("merlin.kernels").joinpath(body.mlir)))
