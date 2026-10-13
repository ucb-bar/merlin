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

**After the pipeline — shim + body + llvm-link** (:func:`link_kernels`). :func:`emit_shim` generates
the C that defines ``_mlir_ciface_<symbol>``: it checks each descriptor is dense row-major, allocates
the result, and calls the body the way its table entry says (for triton-cpu: once per block of rows).
The body is read from ``kernels/triton_llvmir`` (``llvm_mlir`` bodies go through ``mlir-translate``
first), and ``llvm-link`` merges shim + model + body into ``model.linked.ll``, which every existing
consumer (``codegen.compile_ll``, ``spike_model``, ``kernel_spike``) compiles unchanged.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import kernel_table as KT

__all__ = ["KernelLinkError", "KernelUse", "emit_shim", "find_kernel_uses", "link_arch", "link_kernels",
           "prepare_kernel_declarations", "select_body"]


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


# --------------------------------------------------------------------------- after the pipeline
#
# The model now calls `_mlir_ciface_<sym>(desc *res, desc *arg0, ...)`. The body takes whatever its
# table entry says (for triton-cpu: data pointers, the runtime row count and six launch integers, one
# call per block of rows). The shim is the generated C between the two; llvm-link then makes model +
# shim + body ONE module, so clang compiles (and may inline) them together.

_ELEM_BYTES = {"f64": 8, "f32": 4, "f16": 2, "bf16": 2, "i64": 8, "i32": 4, "i16": 2, "i8": 1}


def link_arch(target: str) -> str:
    """The arch a link for ``target`` (``riscv`` | ``host``) produces, as an LLVM triple's first field.

    Derived from what the compile will actually use — the ``--target=`` in ``codegen.RISCV_FLAGS``, the
    machine Python runs on for the host — so a body's triple is compared with the real target rather
    than with a name written here."""
    if target == "host":
        import platform
        return platform.machine()
    from .codegen import RISCV_FLAGS
    for flag in RISCV_FLAGS:
        if flag.startswith("--target="):
            return flag.partition("=")[2].partition("-")[0]
    raise KernelLinkError("codegen.RISCV_FLAGS names no --target=; cannot tell which arch to link for")


def _check_use(use: KernelUse, body: KT.KernelBody) -> None:
    """Refuse, by name, every use the body cannot compute correctly — before any code is generated."""
    where = f"@{use.symbol} ({use.qualname})"
    if len(use.result_types) != 1:
        raise KernelLinkError(f"{where}: {len(use.result_types)} results; the shim supports exactly one "
                              "result per kernel so far")
    if use.kernel_args:
        raise KernelLinkError(f"{where}: non-tensor kernel_args {sorted(use.kernel_args)} are not passed to "
                              "kernel bodies yet")
    if len(use.operand_names) != len(use.arg_types):
        raise KernelLinkError(f"{where}: {len(use.arg_types)} operands but prov.kernel_operands names "
                              f"{len(use.operand_names)}")
    shapes = dict(zip(use.operand_names, use.arg_types))
    for name, (shape, dtype) in [*shapes.items(), *(("result", t) for t in use.result_types)]:
        if dtype != body.dtype:
            raise KernelLinkError(f"{where}: {name} is {dtype} but the {body.form} body is compiled for "
                                  f"{body.dtype}")
        if any(d < 0 for d in shape):
            raise KernelLinkError(f"{where}: {name} has a dynamic shape {shape}; only static shapes link")
    for name, want in body.specialization:
        if name not in shapes:
            raise KernelLinkError(f"{where}: the body is specialized on operand {name!r}, which this call "
                                  f"does not have (operands: {list(use.operand_names)})")
        got = shapes[name][0]
        if len(got) != len(want) or any(w is not None and w != g for w, g in zip(want, got)):
            raise KernelLinkError(f"{where}: operand {name} is {list(got)} but the {body.form} body was "
                                  f"compiled for {list(want)} (None = any)")


def _descriptor_type(rank: int) -> str:
    if rank == 0:
        return "typedef struct { void *allocated, *aligned; long offset; } merlin_kmr0;"
    return (f"typedef struct {{ void *allocated, *aligned; long offset; long sizes[{rank}]; "
            f"long strides[{rank}]; }} merlin_kmr{rank};")


def _entry_source(use: KernelUse, body: KT.KernelBody) -> str:
    """One ``_mlir_ciface_<symbol>`` definition: validate layout, allocate the result, run the grid."""
    ranks = [len(s) for s, _ in use.arg_types]
    (res_shape, res_dtype), = use.result_types
    params = ", ".join([f"merlin_kmr{len(res_shape)} *res",
                        *(f"merlin_kmr{r} *a{i}" for i, r in enumerate(ranks))])
    index = {n: i for i, n in enumerate(use.operand_names)}
    lines = [f"void _mlir_ciface_{use.symbol}({params}) {{"]
    for i, r in enumerate(ranks):
        if r:
            lines.append(f"  if (!merlin_k_dense(a{i}->sizes, a{i}->strides, {r})) __builtin_trap();")
    count = 1
    for d in res_shape:
        count *= d
    nbytes = count * _ELEM_BYTES[res_dtype]
    # The result is a fresh allocation the model's deallocation passes own and free, exactly as if
    # the callee had been an MLIR function: m2m's declaration RETURNS its tensor and
    # buffer-results-to-out-params leaves body-less callees alone. Over-allocate to align it to 64 B.
    lines.append(f"  char *raw = (char *)malloc({nbytes} + 63);")
    lines.append("  if (!raw) __builtin_trap();")
    lines.append("  res->allocated = raw;")
    lines.append("  res->aligned = (void *)(((unsigned long)raw + 63) & ~63UL);")
    lines.append("  res->offset = 0;")
    stride = 1
    for axis in range(len(res_shape) - 1, -1, -1):
        lines.append(f"  res->sizes[{axis}] = {res_shape[axis]}; res->strides[{axis}] = {stride};")
        stride *= res_shape[axis]
    if body.block is not None:
        name, axis, size = body.block
        lines.append(f"  const long grid0 = (a{index[name]}->sizes[{axis}] + {size} - 1) / {size};")
    else:
        lines.append("  const long grid0 = 1;")
    args = []
    for p in body.abi:
        if p.kind == "operand":
            a = f"a{index[p.name]}"
            args.append(f"(({p.ctype}){a}->aligned) + {a}->offset")
        elif p.kind == "result":
            args.append(f"({p.ctype})res->aligned")
        elif p.kind == "dim":
            args.append(f"({p.ctype})a{index[p.name]}->sizes[{p.index}]")
        elif p.kind == "program_id":
            args.append(f"({p.ctype})pid" if p.index == 0 else f"({p.ctype})0")
        elif p.kind == "grid":
            args.append(f"({p.ctype})grid0" if p.index == 0 else f"({p.ctype})1")
        elif p.kind == "null":
            args.append(f"({p.ctype})0")
        else:
            raise KernelLinkError(f"{body.entry}: unknown ABI parameter kind {p.kind!r}")
    lines.append("  for (long pid = 0; pid < grid0; ++pid)")
    lines.append(f"    {body.entry}({', '.join(args)});")
    lines.append("}")
    return "\n".join(lines)


def emit_shim(uses: Sequence[KernelUse], target: str = "riscv", form: str | None = None) -> str:
    """The C translation unit that defines ``_mlir_ciface_<symbol>`` for every use.

    Freestanding (the bare-metal harness has no libc headers): ``malloc`` is declared, not included,
    and a violated precondition traps rather than printing. ``target`` only selects which body is
    acceptable (see :func:`select_body`); the C itself is target-independent."""
    bodies = {u.symbol: select_body(u, target, form) for u in uses}
    for u in uses:
        _check_use(u, bodies[u.symbol])
    ranks = sorted({len(s) for u in uses for s, _ in (*u.arg_types, *u.result_types)})
    out = ["/* Generated by merlin.llvmlower.kernel_link -- do not edit. */",
           "#include <stdint.h>",
           "extern void *malloc(unsigned long);",
           *(_descriptor_type(r) for r in ranks),
           "",
           "/* Row-major and dense: the only layout a raw-pointer kernel body can index. */",
           "static int merlin_k_dense(const long *sizes, const long *strides, int rank) {",
           "  long expect = 1;",
           "  for (int i = rank - 1; i >= 0; --i) {",
           "    if (sizes[i] != 1 && strides[i] != expect) return 0;",
           "    expect *= sizes[i];",
           "  }",
           "  return 1;",
           "}",
           ""]
    seen: set[str] = set()
    for u in uses:
        b = bodies[u.symbol]
        if b.entry in seen:
            continue
        seen.add(b.entry)
        out.append(f"void {b.entry}({', '.join(p.ctype for p in b.abi)});")
    out.append("")
    out.extend(_entry_source(u, bodies[u.symbol]) + "\n" for u in uses)
    return "\n".join(out)


def _ir_triple(ll_text: str) -> str:
    for line in ll_text.splitlines():
        if line.startswith("target triple"):
            return line.partition("=")[2].strip().strip('"')
    return ""


def select_body(use: KernelUse, target: str, form: str | None = None) -> KT.KernelBody:
    """The body of ``use`` to link for ``target``: the requested ``form`` (default: the table's first),
    refused unless it is portable or was written for that arch."""
    body = KT.lookup(use.qualname).body(form)
    if body.portable:
        return body
    arch = link_arch(target)
    if not body.triple.startswith(arch):
        raise KernelLinkError(
            f"{use.qualname}: {body.ir} is {body.triple or 'of no stated arch'} IR, not portable, so it "
            f"cannot be linked into a {arch} image; register a {arch} body or mark this one portable "
            "only if it holds no arch intrinsics, inline asm or address spaces")
    return body


def _body_ll(body: KT.KernelBody, work, target: str) -> Path:
    """The body as LLVM IR ready to link for ``target``: translated from MLIR if needed, and with the
    exporter's triple/datalayout removed when it is portable (the module then takes the target's)."""
    import subprocess

    from . import toolchain
    src = KT.body_path(body)
    if not src.is_file():
        raise KernelLinkError(f"kernel body {src} is missing")
    if body.form == "llvm_mlir":
        out = Path(work) / f"{body.entry}.{body.form}.ll"
        got = subprocess.run([str(toolchain.mlir_translate()), "--mlir-to-llvmir", str(src), "-o", str(out)],
                             capture_output=True, text=True)
        if got.returncode != 0:
            raise KernelLinkError(f"mlir-translate failed on {src.name}:\n{got.stderr[-2000:]}")
        text = out.read_text(encoding="utf-8")
    elif body.form == "llvm_ir":
        text = src.read_text(encoding="utf-8")
    else:
        raise KernelLinkError(f"{src.name}: unknown body form {body.form!r}")
    if body.portable:
        text = "\n".join(line for line in text.splitlines()
                         if not line.startswith(("target triple", "target datalayout")))
    else:
        arch = link_arch(target)
        found = _ir_triple(text)
        if not found.startswith(arch):
            raise KernelLinkError(f"{src.name}: declares target triple {found or '(none)'}, not {arch}")
    out = Path(work) / f"{body.entry}.{body.form}.link.ll"
    out.write_text(text + "\n", encoding="utf-8")
    return out


def link_kernels(model_ll, uses: Sequence[KernelUse], work, *, target: str = "riscv",
                 form: str | None = None, out_name: str = "model.linked.ll") -> Path:
    """Merge the lowered model, the generated shim and every kernel body into ``work/out_name``.

    The shim is compiled to IR with the same flags the target's objects get (``codegen``), and it goes
    FIRST on the llvm-link line: the lowered model carries no triple of its own, so the merged module
    takes the shim's, i.e. the target's."""
    import subprocess

    from . import toolchain
    from .codegen import RISCV_FLAGS, X86_FLAGS
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    shim_c = work / "kernel_shim.c"
    shim_c.write_text(emit_shim(uses, target, form), encoding="utf-8")
    shim_ll = work / "kernel_shim.ll"
    flags = RISCV_FLAGS if target == "riscv" else X86_FLAGS
    got = subprocess.run([str(toolchain.clang()), *flags, "-S", "-emit-llvm", str(shim_c), "-o", str(shim_ll)],
                         capture_output=True, text=True)
    if got.returncode != 0:
        raise KernelLinkError(f"the generated kernel shim did not compile:\n{got.stderr[-3000:]}")
    bodies: dict[str, Path] = {}
    for u in uses:
        b = select_body(u, target, form)
        if b.ir not in bodies:
            bodies[b.ir] = _body_ll(b, work, target)
    out = work / out_name
    cmd = [str(toolchain.llvm_link()), "-S", str(shim_ll), str(model_ll), *map(str, bodies.values()),
           "-o", str(out)]
    got = subprocess.run(cmd, capture_output=True, text=True)
    if got.returncode != 0 or not out.is_file():
        raise KernelLinkError(f"llvm-link failed:\n{' '.join(cmd)}\n{got.stderr[-3000:]}")
    return out
