"""External kernel table: the kernels model2MLIR emits as ``prov.kernel`` calls, and the bodies to link.

model2MLIR lowers a registered custom op (today: a Triton ``@triton_op``) to a ``func.call`` of a private,
body-less ``func.func`` tagged ``prov.kernel = "<ns>::<name>"`` (``m2m/ir/kernel_registry.py``). That call
is a placeholder: the real code is supplied here, at link time, for whatever machine is being compiled
for. This module is only the DATA — which body, in which file, called how. ``kernel_link`` does the work.

**Keyed by ``prov.kernel``, never by symbol.** m2m suffixes the declaration's symbol (``_1``, ``_2``, …)
each time the same kernel appears at a new shape, so the symbol is per-shape and the attribute is the
identity. Entries mirror m2m's ``KernelSpec(qualname, backend, symbol, description)`` so the two tables
read side by side; ``test_table_keys_match_m2m_registry`` keeps them in step.

**``backend`` is the authoring language, not the execution target.** Triton is how the kernel was
written; nothing here runs on a GPU. A body is LLVM IR (or its MLIR ``llvm`` dialect form) for a CPU.

**A body states its own calling convention** (``abi``) because it is not the model's. The model reaches
the kernel through MLIR's C interface — one memref descriptor per tensor — while a triton-cpu export
takes raw pointers, the runtime row count, and six launch integers (program id x/y/z, grid x/y/z), and
computes one ``block`` of rows per call. ``kernel_link`` generates the shim between the two from this
description, so a second body with a different convention is a table edit, not new code.

**Portable bodies.** A triton-cpu export is written for the host it was produced on (an x86 triple and
datalayout) but its code is target-neutral: generic ``llvm.*`` intrinsics only, no arch intrinsics, no
inline asm, no address spaces. ``portable=True`` records that claim; the link step then drops the
triple/datalayout and the IR is compiled for the model's target. ``test_every_body_file_exists_and_is_
portable`` checks the claim against the file. A body that is NOT portable must carry the arch it was
written for in ``triple`` and is only linked for that arch.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

__all__ = ["AbiParam", "KERNEL_TABLE", "KernelBody", "KernelEntry", "KernelTableError", "body_path", "lookup"]


class KernelTableError(LookupError):
    pass


@dataclass(frozen=True)
class AbiParam:
    """One parameter of a body's entry point, in order.

    ``kind``: ``operand`` (data pointer of the tensor operand named ``name`` — the names are
    m2m's ``prov.kernel_operands``), ``result`` (data pointer of result ``index``), ``dim`` (extent
    ``index`` of operand ``name``), ``program_id`` / ``grid`` (launch coordinate / launch extent on
    axis ``index``), ``null`` (a pointer the body does not use, e.g. a scratch buffer).
    ``ctype`` is its C spelling in the generated shim.
    """

    kind: str
    ctype: str
    name: str = ""
    index: int = 0


@dataclass(frozen=True)
class KernelBody:
    """One linkable implementation of a kernel.

    ``form`` is ``llvm_ir`` (textual ``.ll``) or ``llvm_mlir`` (MLIR in the ``llvm`` dialect, translated
    with ``mlir-translate`` first). ``ir`` is relative to the ``merlin.kernels`` package directory.
    ``specialization`` lists the extents the body was compiled for (``None`` = read at run time), so a
    use at another shape is refused instead of computed wrong. ``block = (operand, axis, size)`` means
    one call per ``size`` rows of that operand's axis — the triton program grid.
    """

    form: str
    ir: str
    entry: str
    abi: tuple[AbiParam, ...]
    specialization: tuple[tuple[str, tuple[int | None, ...]], ...] = ()
    block: tuple[str, int, int] | None = None
    portable: bool = False
    triple: str = ""
    dtype: str = "f32"
    description: str = ""


@dataclass(frozen=True)
class KernelEntry:
    qualname: str
    backend: str = "triton"
    description: str = ""
    bodies: tuple[KernelBody, ...] = ()

    def body(self, form: str | None = None) -> KernelBody:
        """The body of ``form``, or the first (preferred) body when ``form`` is None."""
        for b in self.bodies:
            if form is None or b.form == form:
                return b
        raise KernelTableError(
            f"kernel {self.qualname!r} has no {form!r} body (available: {[b.form for b in self.bodies] or 'none'})"
        )


# triton-cpu's launch tail: every exported kernel takes these six after its own arguments.
_TRITON_CPU_LAUNCH = (
    AbiParam("program_id", "int32_t", index=0),
    AbiParam("program_id", "int32_t", index=1),
    AbiParam("program_id", "int32_t", index=2),
    AbiParam("grid", "int32_t", index=0),
    AbiParam("grid", "int32_t", index=1),
    AbiParam("grid", "int32_t", index=2),
)

# workloads/mlp_basic_triton/kernels.py:_mlp2_kernel — (x, w1, b1, w2, b2, out, M; K/H1/H2/BLOCK_M constexpr).
# Weights are in nn.Linear's [out, in] layout; the kernel loads them transposed itself.
_MLP2_ABI = (
    AbiParam("operand", "const float *", name="x"),
    AbiParam("operand", "const float *", name="w1"),
    AbiParam("operand", "const float *", name="b1"),
    AbiParam("operand", "const float *", name="w2"),
    AbiParam("operand", "const float *", name="b2"),
    AbiParam("result", "float *", index=0),
    AbiParam("dim", "int32_t", name="x", index=0),
    *_TRITON_CPU_LAUNCH,
)
_MLP2_SPEC = (("x", (None, 16)), ("w1", (32, 16)), ("b1", (32,)), ("w2", (32, 32)), ("b2", (32,)))

_BUILTIN: tuple[KernelEntry, ...] = (
    KernelEntry(
        "triton_exp::fused_mlp2",
        backend="triton",
        description="linear+relu+linear+relu in one Triton kernel (model2MLIR workloads/mlp_basic_triton)",
        bodies=(
            KernelBody(
                form="llvm_ir",
                ir="triton_llvmir/mlp2.ll",
                entry="fused_mlp2_kernel",
                abi=_MLP2_ABI,
                specialization=_MLP2_SPEC,
                block=("x", 0, 16),
                portable=True,
                description="triton-cpu LLVM IR, optimized for the exporting x86 host",
            ),
            KernelBody(
                form="llvm_mlir",
                ir="triton_llvmir/mlp2_llvm.mlir",
                entry="fused_mlp2_kernel",
                abi=_MLP2_ABI,
                specialization=_MLP2_SPEC,
                block=("x", 0, 16),
                portable=True,
                description="triton-cpu llvm-dialect MLIR, before LLVM optimization",
            ),
        ),
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
    """Where the body's IR lives. Read through the package (works installed or in a checkout)."""
    return Path(str(resources.files("merlin.kernels").joinpath(body.ir)))
