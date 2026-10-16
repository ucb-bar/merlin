"""Link external kernel bodies into a model: model2MLIR ``prov.kernel`` calls → one MLIR module.

A model2MLIR capture marks each registered custom kernel as a ``func.call`` of a private, body-less
``func.func`` tagged ``prov.kernel`` (see :mod:`kernel_table`). A kernel's body is a triton-cpu export in
MLIR's ``llvm`` dialect, so the link is done in MLIR, on the module about to enter the upstream pipeline
(:func:`link_kernels`, called by ``lower.lower_model`` after either preprocess path):

1. each declaration is REPLACED by a definition of the same symbol, so every call site stays as
   model2MLIR wrote it. The definition is the MLIR twin of the ``@triton_op`` Python wrapper
   (:func:`emit_wrappers`): it takes each operand's buffer read-only, allocates the result, and calls
   the kernel's ``llvm.func`` once per program of the grid with data pointers, runtime extents and
   triton-cpu's six launch integers;
2. the body's ``llvm.func`` is moved into the model module.

What reaches the pipeline is then an ordinary closed module — no external symbol, no generated C, no
second translation and no ``llvm-link`` — and the pipeline's own passes do what a hand-written adapter
had to. MEASURED on ``mlp_basic_triton`` (``merlin/tests/rvv/test_kernel_link.py`` pins each):

* one-shot-bufferize sees the wrapper's body, so it knows the operands are only read
  (``to_buffer read_only``) and copies no weight before the call;
* ``buffer-results-to-out-params{hoist-static-allocs}`` hoists the result allocation into ``@forward``,
  so the kernel writes straight into the caller's buffer and the model's deallocation frees it;
* the wrapper's arguments are identity-layout memrefs (``function-boundary-type-conversion=
  identity-layout-map``), so an operand with any other layout is copied dense by bufferization at the
  call instead of being handed to the raw-pointer kernel.

The merge runs in the compiler's Python (``toolchain.m2m_python``, the MLIR bindings the pipeline itself
uses) because a body is MLIR that xDSL cannot read. It is structural — symbols and operations, never
string surgery — so it does not care whether the preprocess printed the declaration pretty or generic.
The merged module is printed without locations, so a body's debug info (the exporting checkout's
source paths) never reaches the image.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import kernel_table as KT

__all__ = ["KernelLinkError", "KernelUse", "emit_wrappers", "find_kernel_uses", "link_kernels"]


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
            calls.setdefault(
                callee,
                (
                    tuple(n.data for n in names.data) if names is not None else (),
                    {k: str(v) for k, v in args.data.items()} if args is not None else {},
                ),
            )
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
        uses.append(
            KernelUse(
                symbol=symbol,
                qualname=qualname,
                operand_names=names,
                arg_types=tuple((tuple(s), d) for s, d in map(type_shape_dtype, ftype.inputs.data)),
                result_types=tuple((tuple(s), d) for s, d in map(type_shape_dtype, ftype.outputs.data)),
                kernel_args=kargs,
            )
        )
    return tuple(uses)


def _check_use(use: KernelUse, body: KT.KernelBody) -> None:
    """Refuse, by name, every use the body cannot compute correctly — before any code is generated."""
    where = f"@{use.symbol} ({use.qualname})"
    if use.kernel_args:
        raise KernelLinkError(
            f"{where}: non-tensor kernel_args {sorted(use.kernel_args)} are not passed to kernel bodies yet"
        )
    if len(use.operand_names) != len(use.arg_types):
        raise KernelLinkError(
            f"{where}: {len(use.arg_types)} operands but prov.kernel_operands names {len(use.operand_names)}"
        )
    shapes = dict(zip(use.operand_names, use.arg_types))
    results = [(f"result {i}", t) for i, t in enumerate(use.result_types)]
    for name, (shape, dtype) in [*shapes.items(), *results]:
        if dtype != body.dtype:
            raise KernelLinkError(f"{where}: {name} is {dtype} but the {body.entry} body is compiled for {body.dtype}")
        if any(d < 0 for d in shape):
            raise KernelLinkError(f"{where}: {name} has a dynamic shape {shape}; only static shapes link")
    # The wrapper returns what the kernel wrote, so a result the body never writes would be returned
    # uninitialized: the body must name every result exactly once.
    written = sorted(a.index for a in body.args if a.kind == "result")
    if written != list(range(len(use.result_types))):
        raise KernelLinkError(
            f"{where}: {len(use.result_types)} result(s), but the {body.entry} body writes result(s) {written}"
        )
    for name in [a.name for a in body.args if a.kind in ("operand", "dim")] + [g[0] for g in body.grid]:
        if name not in shapes:
            raise KernelLinkError(
                f"{where}: the {body.entry} body reads operand {name!r}, which this call does not have "
                f"(operands: {list(use.operand_names)})"
            )
    for name, want in body.specialization:
        if name not in shapes:
            raise KernelLinkError(
                f"{where}: the body is specialized on operand {name!r}, which this call "
                f"does not have (operands: {list(use.operand_names)})"
            )
        got = shapes[name][0]
        if len(got) != len(want) or any(w is not None and w != g for w, g in zip(want, got)):
            raise KernelLinkError(
                f"{where}: operand {name} is {list(got)} but the {body.entry} body was "
                f"compiled for {list(want)} (None = any)"
            )


# --------------------------------------------------------------------------- the wrapper
#
# One `func.func` per use, in the tensor domain the pipeline starts in, built only from upstream ops
# whose bufferization is defined. Its body is the @triton_op wrapper's: operand buffers, result
# allocation, the program grid, one `llvm.call` per program. Shapes are static (`_check_use`), so every
# extent and the grid are constants and the trip-1 loops fold away.

_LAUNCH_AXES = 3  # triton-cpu appends program id x/y/z, then grid x/y/z, to every kernel's arguments


def _shaped(kind: str, shape: tuple[int, ...], dtype: str) -> str:
    return f"{kind}<{''.join(f'{d}x' for d in shape)}{dtype}>"


def _wrapper(use: KernelUse, body: KT.KernelBody) -> tuple[list[str], str]:
    """The wrapper's lines, and the argument types of the ``llvm.call`` it makes (the body's signature)."""
    index = {n: i for i, n in enumerate(use.operand_names)}
    tensors = [_shaped("tensor", *t) for t in use.arg_types]
    buffers = [_shaped("memref", *t) for t in use.arg_types]
    out_tensors = [_shaped("tensor", *t) for t in use.result_types]
    out_buffers = [_shaped("memref", *t) for t in use.result_types]
    params = ", ".join(f"%a{i}: {t}" for i, t in enumerate(tensors))
    # Internal linkage: `private` alone does not reach LLVM (convert-func-to-llvm drops visibility), and
    # an external wrapper keeps a dead out-of-line copy in the object after clang inlines it.
    lines = [
        f"func.func private @{use.symbol}({params}) -> ({', '.join(out_tensors)}) "
        f'attributes {{llvm.linkage = #llvm.linkage<internal>, prov.kernel = "{use.qualname}"}} {{'
    ]
    # `read_only` is the promise that lets bufferization pass the caller's buffer instead of a copy.
    lines += [
        f"  %b{i} = bufferization.to_buffer %a{i} read_only : {t} to {m}"
        for i, (t, m) in enumerate(zip(tensors, buffers))
    ]
    lines += [f"  %r{j} = memref.alloc() {{alignment = 64 : i64}} : {m}" for j, m in enumerate(out_buffers)]
    args, types = [], []
    for k, a in enumerate(body.args):
        if a.kind in ("operand", "result"):
            buf, ty = (
                (f"%b{index[a.name]}", buffers[index[a.name]])
                if a.kind == "operand"
                else (f"%r{a.index}", out_buffers[a.index])
            )
            lines.append(f"  %i{k} = memref.extract_aligned_pointer_as_index {buf} : {ty} -> index")
            lines.append(f"  %w{k} = arith.index_cast %i{k} : index to i64")
            lines.append(f"  %p{k} = llvm.inttoptr %w{k} : i64 to !llvm.ptr")
            types.append("!llvm.ptr")
        elif a.kind == "dim":
            lines.append(f"  %p{k} = arith.constant {use.arg_types[index[a.name]][0][a.index]} : i32")
            types.append("i32")
        else:
            raise KernelLinkError(f"{body.entry}: unknown kernel argument kind {a.kind!r}")
        args.append(f"%p{k}")
    if len(body.grid) > _LAUNCH_AXES:
        raise KernelLinkError(f"{body.entry}: a {len(body.grid)}-axis grid; Triton programs have {_LAUNCH_AXES} axes")
    grid = [-(-use.arg_types[index[name]][0][axis] // block) for name, axis, block in body.grid]
    grid += [1] * (_LAUNCH_AXES - len(grid))
    lines += ["  %c0 = arith.constant 0 : i32", "  %c1 = arith.constant 1 : i32"]
    lines += [f"  %g{ax} = arith.constant {n} : i32" for ax, n in enumerate(grid)]
    # Triton programs are independent and unordered, so the grid runs as a sequential nest, x innermost.
    for depth, ax in enumerate(reversed(range(_LAUNCH_AXES))):
        lines.append(f"  {'  ' * depth}scf.for %pid{ax} = %c0 to %g{ax} step %c1 : i32 {{")
    launch = [*args, *(f"%pid{ax}" for ax in range(_LAUNCH_AXES)), *(f"%g{ax}" for ax in range(_LAUNCH_AXES))]
    signature = ", ".join([*types, *["i32"] * (2 * _LAUNCH_AXES)])
    lines.append(f"  {'  ' * _LAUNCH_AXES}llvm.call @{body.entry}({', '.join(launch)}) : ({signature}) -> ()")
    lines += [f"  {'  ' * depth}}}" for depth in reversed(range(_LAUNCH_AXES))]
    lines += [
        f"  %t{j} = bufferization.to_tensor %r{j} restrict writable : {m} to {t}"
        for j, (m, t) in enumerate(zip(out_buffers, out_tensors))
    ]
    lines.append(f"  return {', '.join(f'%t{j}' for j in range(len(out_tensors)))} : {', '.join(out_tensors)}")
    lines.append("}")
    return lines, signature


def emit_wrappers(uses: Sequence[KernelUse]) -> str:
    """One MLIR module holding the wrapper of every use, plus one ``llvm.func`` declaration per kernel
    entry so the module verifies on its own (the bodies define those symbols at link time).

    Every use is checked against its body first (:func:`_check_use`), so an unsupported use is refused
    by name before anything is generated."""
    out = ["// Generated by merlin.llvmlower.kernel_link -- do not edit.", "module {"]
    entries: dict[str, str] = {}
    for u in uses:
        body = KT.lookup(u.qualname).body
        _check_use(u, body)
        lines, signature = _wrapper(u, body)
        out += [f"  {line}" for line in lines]
        entries.setdefault(body.entry, signature)
    out += [f"  llvm.func @{entry}({signature})" for entry, signature in entries.items()]
    out.append("}")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- the merge
#
# Runs in the compiler's Python: argv = upstream module, wrappers module, output, body files...

_MERGE_SRC = r"""
import sys

from torch_mlir import ir

src, wrappers_path, dst, *bodies = sys.argv[1:]
ctx = ir.Context()
module = ir.Module.parse(open(src).read(), ctx)
wrappers = ir.Module.parse(open(wrappers_path).read(), ctx)


def symbol(op):
    return ir.StringAttr(op.attributes["sym_name"]).value if "sym_name" in op.attributes else None


top = {symbol(op): op for op in module.body.operations if symbol(op) is not None}
# Each kernel declaration is replaced in place by its wrapper (same symbol). The wrappers module's
# llvm.func declarations only let it verify alone; the bodies define those symbols.
for op in list(wrappers.body.operations):
    if op.operation.name != "func.func":
        continue
    name = symbol(op)
    decl = top.get(name)
    if decl is None or decl.operation.name != "func.func" or len(decl.regions[0].blocks):
        sys.exit(f"kernel_link: the module has no body-less declaration @{name} to replace")
    op.operation.move_before(decl.operation)
    decl.operation.erase()
    top[name] = op
# Every top-level op of a body moves in: the kernel, and any helper it defines or declares.
for path in bodies:
    body = ir.Module.parse(open(path).read(), ctx)
    for op in list(body.body.operations):
        name = symbol(op)
        if name in top:
            sys.exit(f"kernel_link: {path} defines @{name}, which the model module already has")
        module.body.append(op.operation)
        top[name] = op
try:
    module.operation.verify()
except ir.MLIRError as e:
    sys.exit(f"kernel_link: the linked module does not verify (e.g. a wrapper's llvm.call against its body):\n{e}")
open(dst, "w").write(str(module.operation))
"""


def link_kernels(upstream_text: str, uses: Sequence[KernelUse], work) -> str:
    """``upstream_text`` with every kernel declaration replaced by its wrapper and every body merged in.

    Leaves in ``work`` what it ran: ``kernel_wrappers.mlir`` (:func:`emit_wrappers`), ``kernel_link.py``
    and ``model.unlinked.mlir`` (the module as the preprocess left it). Raises :class:`KernelLinkError`
    naming the kernel for a use its body cannot compute, a missing body file, or a merge MLIR refuses
    (a wrapper whose call does not match the body's ``llvm.func``, a symbol the module already has)."""
    import subprocess

    from .toolchain import m2m_python

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    wrappers = work / "kernel_wrappers.mlir"
    wrappers.write_text(emit_wrappers(uses), encoding="utf-8")
    bodies: list[str] = []
    for u in uses:
        path = KT.body_path(KT.lookup(u.qualname).body)
        if not path.is_file():
            raise KernelLinkError(f"@{u.symbol} ({u.qualname}): kernel body {path} is missing")
        if str(path) not in bodies:
            bodies.append(str(path))
    src, dst, script = work / "model.unlinked.mlir", work / "kernel_link.out.mlir", work / "kernel_link.py"
    src.write_text(upstream_text, encoding="utf-8")
    script.write_text(_MERGE_SRC, encoding="utf-8")
    dst.unlink(missing_ok=True)
    got = subprocess.run(
        [str(m2m_python()), str(script), str(src), str(wrappers), str(dst), *bodies],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if got.returncode != 0 or not dst.is_file():
        raise KernelLinkError(
            f"linking {sorted({u.qualname for u in uses})} failed:\n{got.stdout[-1000:]}{got.stderr[-3000:]}"
        )
    text = dst.read_text(encoding="utf-8")
    dst.unlink()
    return text
