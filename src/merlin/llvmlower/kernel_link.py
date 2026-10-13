"""Link external kernel bodies into a lowered model: model2MLIR ``prov.kernel`` calls → one LLVM module.

A model2MLIR capture marks each registered custom kernel as a ``func.call`` of a private, body-less
``func.func`` tagged ``prov.kernel`` (see :mod:`kernel_table`). The whole-model pipeline already carries
such a call through bufferization and LLVM lowering untouched — MEASURED on ``mlp_basic_triton``: the
declaration survives one-shot-bufferize, and since ``buffer-results-to-out-params`` leaves body-less
callees alone the callee still RETURNS its result buffer. What is missing is only the code behind the
symbol. This module supplies it in two halves around :func:`pipeline.lower_to_llvm_ir`:

**Before the pipeline — one ABI, whichever preprocess ran.** :func:`prepare_kernel_declarations` puts
``llvm.emit_c_interface`` and a read-only ``bufferization.access`` on every argument of each kernel
declaration. Both were measured to matter:

* without it the two preprocess paths disagree. ``passes_xdsl.add_c_interface`` marks every function,
  so the xDSL path calls ``_mlir_ciface_<sym>(ptr res, ptr arg...)`` (one pointer per descriptor);
  ``preprocess_text_textual`` marks only the entry, so the textual path — the one ``spike_model``
  uses — calls ``<sym>`` with every descriptor field expanded (31 scalars for mlp2). The shim would
  have had to match whichever path ran;
* without the access annotation one-shot-bufferize copied all four weights into fresh buffers before
  the call (4 of the module's 5 ``llvm.memcpy``), the same defensive copy ``passes_opu`` documents.

**After the pipeline — shim + body + llvm-link** (``link_kernels``).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from . import kernel_table as KT

__all__ = ["KernelLinkError", "KernelUse", "find_kernel_uses", "prepare_kernel_declarations"]


class KernelLinkError(RuntimeError):
    pass


@dataclass(frozen=True)
class KernelUse:
    """One kernel declaration the module calls, with what its call sites say about it."""

    symbol: str
    qualname: str
    operand_names: tuple[str, ...]
    arg_types: tuple[tuple[tuple[int, ...], str], ...]
    result_types: tuple[tuple[tuple[int, ...], str], ...]
    kernel_args: dict[str, str] = field(default_factory=dict)


def _str_attr(attrs, key: str) -> str | None:
    a = attrs.get(key)
    return getattr(a, "data", None) if a is not None else None


def find_kernel_uses(mlir_text: str) -> tuple[KernelUse, ...]:
    """Every ``prov.kernel`` declaration in ``mlir_text``, in module order; ``()`` when there is none.

    Raises :class:`KernelLinkError` for a kernel merlin's table does not know, so an unlinkable model
    fails here, naming the kernel, rather than at the final link with an undefined symbol.
    """
    # A kernel-free module (every model but these) must not pay a whole-module parse — whisper's is
    # seconds — just to learn there is nothing to do. The attribute name is the cheapest exact test.
    if "prov.kernel" not in mlir_text:
        return ()
    from ..common.mlir_query import parse, type_shape_dtype

    module = parse(mlir_text)
    calls: dict[str, tuple[tuple[str, ...], dict[str, str]]] = {}
    decls = []
    for op in module.walk():
        if op.name == "func.call" and "prov.kernel" in op.attributes:
            callee = op.callee.root_reference.data
            names = op.attributes.get("prov.kernel_operands")
            args = op.attributes.get("prov.kernel_args")
            calls.setdefault(callee, (
                tuple(n.data for n in names.data) if names is not None else (),
                {k: str(v) for k, v in args.data.items()} if args is not None else {}))
        elif op.name == "func.func" and not op.body.blocks and "prov.kernel" in op.attributes:
            decls.append(op)

    uses = []
    for fn in decls:
        symbol = fn.sym_name.data
        qualname = _str_attr(fn.attributes, "prov.kernel")
        try:
            KT.lookup(qualname)
        except KT.KernelTableError as e:
            raise KernelLinkError(f"@{symbol}: {e}") from None
        names, kargs = calls.get(symbol, ((), {}))
        ftype = fn.function_type
        uses.append(KernelUse(
            symbol=symbol, qualname=qualname, operand_names=names,
            arg_types=tuple((tuple(s), d) for s, d in map(type_shape_dtype, ftype.inputs.data)),
            result_types=tuple((tuple(s), d) for s, d in map(type_shape_dtype, ftype.outputs.data)),
            kernel_args=kargs))
    return tuple(uses)


def _split_top_level(inner: str) -> list[str]:
    """Split an argument list on the commas that separate arguments (not those inside ``<>``/``{}``)."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(inner):
        if ch in "<{[(":
            depth += 1
        elif ch in ">}])":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(inner[start:i].strip())
            start = i + 1
    tail = inner[start:].strip()
    if tail:
        parts.append(tail)
    return parts


_READ = '{bufferization.access = "read"}'


def prepare_kernel_declarations(upstream_text: str, uses: Sequence[KernelUse]) -> str:
    """Mark each kernel declaration ``llvm.emit_c_interface`` and its arguments read-only.

    Text in, text out, on the module that is about to enter the upstream pipeline, so it applies
    after EITHER preprocess path. Anchored on the exact ``func.func private @<symbol>(`` this module
    found and done with ``str`` operations — the discipline ``passes_opu.patch_declaration_arg_attrs``
    uses for the same repair, and needed for the same reason: xDSL drops ``arg_attrs`` when it prints a
    body-less declaration, so the annotation cannot be put on the IR object. Idempotent.
    """
    out = upstream_text
    for use in uses:
        head = f"func.func private @{use.symbol}("
        at = out.find(head)
        if at < 0:
            continue
        open_paren = at + len(head) - 1
        close = out.find(")", open_paren)
        inner = out[open_paren + 1:close]
        if "bufferization.access" not in inner:
            annotated = ", ".join(f"{p} {_READ}" for p in _split_top_level(inner))
            out = out[:open_paren + 1] + annotated + out[close:]
            close = open_paren + 1 + len(annotated)
        line_end = out.find("\n", close)
        line_end = len(out) if line_end < 0 else line_end
        tail = out[close:line_end]
        if "llvm.emit_c_interface" in tail:
            continue
        attrs_at = tail.find("attributes {")
        if attrs_at >= 0:
            insert = close + attrs_at + len("attributes {")
            out = out[:insert] + "llvm.emit_c_interface, " + out[insert:]
        else:
            out = out[:line_end] + " attributes {llvm.emit_c_interface}" + out[line_end:]
    return out
