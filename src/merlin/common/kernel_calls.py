"""Structural predicates for model2MLIR external-kernel calls, shared by every pass that walks a module.

A model2MLIR capture can call a kernel it does not define: ``func.call @<sym>`` tagged ``prov.kernel``,
to a private ``func.func`` with no body (see ``llvmlower/kernel_table.py``). The body arrives at link
time. Until then every pass that reasons about a module's functions or calls has to tell that call and
that declaration apart from the entry function and from outlined dispatches — which is what these do,
on the IR (no text matching).
"""
from __future__ import annotations

__all__ = ["ExternalKernelNotExecutable", "entry_function", "is_external_decl", "is_kernel_call",
           "kernel_qualname"]


class ExternalKernelNotExecutable(RuntimeError):
    """Raised by a runtime that executes merlin's own dispatches when it meets an external kernel.

    Those runtimes have no body for it: the body is linked only on the LLVM route
    (``llvmlower.kernel_link``), which is where such a model has to be run."""

    def __init__(self, qualname: str):
        self.qualname = qualname
        super().__init__(
            f"external kernel {qualname!r} has no body in this runtime; its body is linked only on the "
            "LLVM route (llvmlower.kernel_link) -- run the model through lower_model / spike_model")


def is_kernel_call(op) -> bool:
    """A call to an external kernel: ``func.call`` carrying model2MLIR's ``prov.kernel`` identity."""
    return op.name == "func.call" and "prov.kernel" in op.attributes


def kernel_qualname(op) -> str:
    """The ``prov.kernel`` name (``"<ns>::<name>"``) of a kernel call or declaration."""
    return op.attributes["prov.kernel"].data


def is_external_decl(op) -> bool:
    """A ``func.func`` with no body: a declaration whose code is supplied elsewhere."""
    return op.name == "func.func" and not op.body.blocks


def entry_function(module):
    """The module's one function with a body (``@forward`` for a capture).

    Raises ``ValueError`` when there is none or more than one, rather than guessing: "the first
    func.func" stopped being the entry the moment captures could carry kernel declarations.
    """
    bodied = [op for op in module.walk() if op.name == "func.func" and not is_external_decl(op)]
    if len(bodied) != 1:
        names = [op.sym_name.data for op in bodied]
        raise ValueError(f"expected exactly one func.func with a body, found {len(bodied)}: {names}")
    return bodied[0]
