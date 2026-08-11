"""``@triton.jit`` -> TTIR. The ONLY module in merlin.triton that touches Triton internals.

Triton's public entry point is ``triton.compile()``, which insists on a ``GPUTarget``, builds a
backend, and runs the whole stage chain down to a GPU binary. None of that is wanted here: Merlin
needs the *machine-independent* tile-level IR and nothing after it. So this module calls the AST ->
TTIR step directly (``ASTSource.make_ir``), which needs no device and stops exactly at the boundary
we want.

The blast radius of that choice is deliberately one file. ``ASTSource``/``make_ir`` are compiler
internals with no stability promise, so everything downstream consumes :class:`TTIRModule` and never
imports triton.

A ``GPUTarget`` still has to be supplied because the frontend's signature demands one. It selects
which backend provides the codegen callbacks, NOT what the IR means — TTIR is pre-layout,
pre-scheduling, and carries no target dialect. ``test_ttir_is_independent_of_the_nominal_backend``
holds that claim to account rather than trusting it.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .spec import TritonKernelSpec
from .toolchain import require

# The nominal frontend target. Any registered backend works; this one is present in every stock
# wheel. Compute capability and warp size influence backend option DEFAULTS, not the AST -> TTIR
# translation, and the values are recorded as provenance so a result stays reproducible.
_NOMINAL_BACKEND = "cuda"
_NOMINAL_ARCH = 80
_NOMINAL_WARP_SIZE = 32


class TritonFrontendError(RuntimeError):
    """The kernel could not be translated to TTIR (a Triton-side failure, surfaced with context)."""


@dataclass(frozen=True)
class TTIRModule:
    """TTIR for one kernel: deterministic text, the live module, and an op inventory.

    ``text`` is triton's debug-info-free printing. The default printing embeds ``loc(...)`` pointing
    at absolute source paths, which would make the artifact hash depend on where the checkout lives
    and where the kernel file happens to sit.

    ``module`` is the live IR. It is kept because it can be WALKED — ops expose name, operands,
    results and typed attributes, and values expose ``get_type()`` and an SSA ``id`` — so the bridge
    reads structure and types from the IR itself instead of re-parsing MLIR text.
    """

    kernel_name: str
    text: str
    module: Any
    ops: tuple[str, ...]
    digest: str
    triton_version: str
    provenance: dict[str, Any]

    def has_op(self, name: str) -> bool:
        return name in self.ops


def _signature(spec: TritonKernelSpec) -> dict[str, str]:
    """Triton's signature form: declared args, plus every constexpr marked as such."""
    sig = dict(spec.signature())
    for name in spec.constexprs:
        sig[name] = "constexpr"
    return sig


def make_ttir(spec: TritonKernelSpec) -> TTIRModule:
    """Translate ``spec``'s ``@triton.jit`` function to TTIR. No GPU required.

    Raises :class:`TritonFrontendError` with the kernel name attached when Triton rejects the
    kernel, so a tile-shape or type rejection is attributable rather than an opaque traceback from
    inside triton.
    """
    probe = require()
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource, make_backend

    fn = spec.function
    if not hasattr(fn, "cache_key"):
        raise TritonFrontendError(
            f"{spec.name!r} is not a @triton.jit function (no cache_key). Note that triton reads the "
            "decorated function's SOURCE, so it must be defined in a real .py file, not exec'd.")

    target = GPUTarget(_NOMINAL_BACKEND, _NOMINAL_ARCH, _NOMINAL_WARP_SIZE)
    backend = make_backend(target)
    options = backend.parse_options({})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    try:
        src = ASTSource(fn=fn, signature=_signature(spec), constexprs=dict(spec.constexprs))
        module = src.make_ir(target, options, backend.get_codegen_implementation(options),
                             backend.get_module_map(), context)
    except Exception as exc:  # noqa: BLE001 — re-raised with the kernel named
        raise TritonFrontendError(f"triton could not translate kernel {spec.name!r}: {exc}") from exc

    text = module.str_nodebug() if hasattr(module, "str_nodebug") else str(module)
    ops: list[str] = []
    module.walk(lambda op: ops.append(op.get_name()))
    return TTIRModule(
        kernel_name=spec.name,
        text=text,
        module=module,
        ops=tuple(sorted(set(ops))),
        digest=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        triton_version=probe.installed or "unknown",
        provenance={"nominal_backend": _NOMINAL_BACKEND, "nominal_arch": _NOMINAL_ARCH,
                    "warp_size": _NOMINAL_WARP_SIZE, "constexprs": dict(spec.constexprs)},
    )


def walk_ops(ttir: TTIRModule) -> list[Any]:
    """Every op in the module, in walk order — the bridge's input."""
    out: list[Any] = []
    ttir.module.walk(lambda op: out.append(op))
    return out


def entry_block_args(ttir: TTIRModule) -> list[Any]:
    """The kernel function's parameters, in declaration order.

    Triton's region API exposes no way down to a block, but every op knows its own block, so the
    entry block is reached through the first op of the body. Ordered parameters matter because they
    are what binds TTIR back to the caller-declared :class:`~merlin.triton.spec.TritonKernelSpec`.
    """
    for op in walk_ops(ttir):
        if op.get_name() in ("builtin.module", "tt.func"):
            continue
        block = op.get_block()
        return [block.get_argument(i) for i in range(block.get_num_arguments())]
    raise TritonFrontendError(f"kernel {ttir.kernel_name!r} has an empty body")


def tensor_shape(type_text: str) -> tuple[int, ...]:
    """``tensor<16x32xi8>`` -> ``(16, 32)``; a scalar type -> ``()``.

    MLIR type syntax, split structurally. The in-memory type API exposes only ``is_integer`` and
    ``is_fp16``, so shapes have to come from the printed form.
    """
    text = str(type_text)
    if not text.startswith("tensor<") or not text.endswith(">"):
        return ()
    dims = text[len("tensor<"):-1].split("x")[:-1]
    try:
        return tuple(int(d) for d in dims)
    except ValueError:
        raise TritonFrontendError(f"non-static tensor type {text!r}") from None


def pointee_dtype(type_text: str) -> str | None:
    """``!tt.ptr<f32>`` -> ``"f32"``; anything that is not a pointer type -> ``None``."""
    text = str(type_text)
    head, sep, rest = text.partition("<")
    if head != "!tt.ptr" or not sep or not rest.endswith(">"):
        return None
    return rest[:-1]


def _parse_splat_literal(attr_text: str, type_text: str) -> int | float | bool:
    """One MLIR constant attribute in its printed form -> a Python value.

    Handles the scalar and the *splat* elements form (``dense<0>``); a non-splat ``dense<[...]>`` is
    refused, because a per-element table is not a constant the addressing analysis can reason about.
    """
    body = attr_text.strip()
    if body.startswith("dense<") and body.endswith(">"):
        body = body[len("dense<"):-1].strip()
        if body.startswith("["):
            raise TritonFrontendError(f"non-splat constant {attr_text!r} is not supported")
    if body in ("true", "false"):
        return body == "true"
    if "." in body or "e" in body or "E" in body:
        if "x" not in type_text and "i" not in type_text.rsplit(">", 1)[-1]:
            return float(body)
        try:
            return int(body)
        except ValueError:
            return float(body)
    return int(body)


def constant_table(ttir: TTIRModule) -> dict[int, int | float | bool]:
    """SSA id -> Python value for every ``arith.constant`` in the module.

    Triton's in-memory op API exposes ``get_int_attr`` but nothing that reads a *dense* attribute,
    so a splat constant such as ``dense<32> : tensor<16x1xi32>`` — which is how a folded tile stride
    reaches the IR — is invisible to the walk. Its value is therefore taken from the printed module,
    which is the only other rendering of the same IR.

    Correlating the two by position would be a guess, so it is *verified* instead: the printed
    constants and the walked constants must agree in count, in result type, and — for every constant
    the walk can read directly — in value. If they ever diverge, this raises rather than returning a
    table that silently pairs the wrong values.
    """
    walked = [op for op in walk_ops(ttir) if op.get_name() == "arith.constant"]
    printed: list[tuple[str, str]] = []
    for raw in ttir.text.splitlines():
        name, sep, rhs = raw.strip().partition(" = ")
        if not sep or not rhs.startswith("arith.constant "):
            continue
        # `dense<0> : tensor<4xi32>` carries its type; `true` does not, because MLIR prints a
        # boolean constant bare. An absent type is recorded as None rather than mis-split.
        remainder = rhs[len("arith.constant "):]
        attr_text, sep, type_text = remainder.rpartition(" : ")
        printed.append((attr_text, type_text) if sep else (remainder, None))

    if len(printed) != len(walked):
        raise TritonFrontendError(
            f"{len(walked)} arith.constant op(s) in the walked IR but {len(printed)} in the printed "
            "IR — the two renderings disagree, so constants cannot be read reliably")

    table: dict[int, int | float | bool] = {}
    for op, (attr_text, type_text) in zip(walked, printed):
        result = op.get_result(0)
        if type_text is not None and str(result.get_type()) != type_text.strip():
            raise TritonFrontendError(
                f"constant type mismatch between walked ({result.get_type()}) and printed "
                f"({type_text.strip()}) IR — the two renderings are not in the same order")
        value = _parse_splat_literal(attr_text, type_text)
        direct = op.get_int_attr("value")
        # A one-bit `true` reads back as -1 through the integer accessor (all bits set), so booleans
        # are compared on truth rather than on the two's-complement value.
        if isinstance(value, bool) and direct is not None:
            direct = direct != 0
        if direct is not None and direct != value:
            raise TritonFrontendError(
                f"constant value mismatch: walked IR says {direct}, printed IR says {value}")
        table[result.id()] = value
    return table
