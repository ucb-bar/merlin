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
