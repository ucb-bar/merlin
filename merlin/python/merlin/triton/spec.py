"""The frontend's data model: what a caller must state about a kernel, and what compiling it produced.

A ``@triton.jit`` function is not self-describing. Its parameters are untyped, its pointer arguments
carry no shape, its grid lives at the call site, and which buffers it *writes* is visible only inside
the body. So a kernel cannot be compiled from source alone — the caller supplies the missing
semantic ABI, and :class:`TritonKernelSpec` is that contract.

Effects are declared rather than inferred, and are richer than a list of outputs, because the ABI
verifier for model kernel replacement needs to reject a mismatch (a kernel that mutates an input the
surrounding graph believes is read-only is a miscompile, not a detail). Inference from the body is
deliberately not attempted: a wrong guess here is silent.

Everything in this module is plain data with no Triton import, so it stays testable without the
wheel installed.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Effect = Literal["read", "write", "readwrite"]
ArgKind = Literal["pointer", "scalar"]

_EFFECTS: tuple[str, ...] = ("read", "write", "readwrite")
_KINDS: tuple[str, ...] = ("pointer", "scalar")


class KernelSpecError(ValueError):
    """A kernel spec that cannot be compiled as stated (fail closed, never guess)."""


@dataclass(frozen=True)
class KernelArg:
    """One kernel parameter: a pointer to a tensor, or a scalar.

    ``shape``/``strides``/``effect`` are required for pointers and meaningless for scalars. A
    pointer with no shape cannot be re-raised to a tensor argument, which is the whole job of the
    bridge, so the omission is an error rather than a dynamic-shape fallback.
    """

    name: str
    kind: ArgKind
    dtype: str
    shape: tuple[int, ...] | None = None
    strides: tuple[int, ...] | None = None
    effect: Effect | None = None

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise KernelSpecError(f"arg {self.name!r}: kind must be one of {_KINDS}, got {self.kind!r}")
        if not self.dtype:
            raise KernelSpecError(f"arg {self.name!r}: dtype is required")
        if self.kind == "pointer":
            if not self.shape:
                raise KernelSpecError(
                    f"arg {self.name!r}: a pointer needs a static shape — the bridge re-raises it to a "
                    "tensor-typed function argument and cannot do that for an unknown extent")
            if self.effect not in _EFFECTS:
                raise KernelSpecError(
                    f"arg {self.name!r}: a pointer needs an explicit effect {_EFFECTS} — mutation is "
                    "declared, never inferred from the body")
            if self.strides is not None and len(self.strides) != len(self.shape):
                raise KernelSpecError(
                    f"arg {self.name!r}: {len(self.strides)} strides for a rank-{len(self.shape)} shape")
        else:
            if self.shape or self.strides:
                raise KernelSpecError(f"arg {self.name!r}: a scalar carries no shape/strides")
            if self.effect is not None:
                raise KernelSpecError(f"arg {self.name!r}: a scalar carries no effect")

    @property
    def is_written(self) -> bool:
        return self.effect in ("write", "readwrite")

    @property
    def rank(self) -> int:
        return len(self.shape or ())

    def numel(self) -> int:
        n = 1
        for d in self.shape or ():
            n *= int(d)
        return n


@dataclass(frozen=True)
class GridSpec:
    """The SPMD launch grid: how many program instances exist, in up to 3 dimensions.

    Held as either concrete extents or a callable over (constexprs, runtime scalars) — the same two
    forms Triton's own ``kernel[grid](...)`` accepts. :meth:`resolve` collapses it to three ints
    before anything downstream sees it, because the bridge normalizes the grid into ordinary
    whole-function semantics and needs concrete trip counts to do so.

    Note what this type does NOT decide: whether those instances become a sequential loop, vector
    lanes, or warps. That is a *schedule* decision made per target inside Merlin — hard-coding it
    here would make one accelerator work and another impossible.
    """

    dims: tuple[int, ...] | None = None
    dims_fn: Callable[[Mapping[str, Any], Mapping[str, Any]], Sequence[int]] | None = None

    def __post_init__(self) -> None:
        if (self.dims is None) == (self.dims_fn is None):
            raise KernelSpecError("GridSpec takes exactly one of dims= or dims_fn=")
        if self.dims is not None:
            if not 1 <= len(self.dims) <= 3:
                raise KernelSpecError(f"grid must be 1..3 dimensional, got {len(self.dims)}")
            if any(int(d) < 1 for d in self.dims):
                raise KernelSpecError(f"grid extents must be >= 1, got {self.dims}")

    def resolve(self, constexprs: Mapping[str, Any] | None = None,
                runtime: Mapping[str, Any] | None = None) -> tuple[int, int, int]:
        """Concrete (x, y, z) extents, right-padded with 1s."""
        dims = self.dims
        if dims is None:
            assert self.dims_fn is not None
            raw = self.dims_fn(dict(constexprs or {}), dict(runtime or {}))
            dims = tuple(int(d) for d in raw)
            if not 1 <= len(dims) <= 3:
                raise KernelSpecError(f"grid callable returned {len(dims)} dimensions, expected 1..3")
            if any(d < 1 for d in dims):
                raise KernelSpecError(f"grid callable returned a non-positive extent: {dims}")
        padded = tuple(int(d) for d in dims) + (1, 1, 1)
        return padded[0], padded[1], padded[2]


@dataclass(frozen=True)
class TritonKernelSpec:
    """A ``@triton.jit`` function plus the semantic ABI needed to compile it.

    ``provenance`` carries values that must be accepted to satisfy the Triton frontend but that are
    NOT portable target semantics — ``num_warps`` and ``num_stages`` are CUDA scheduling knobs, and
    treating them as though they meant something on a systolic array would be a lie. They are
    recorded so a result is reproducible, and ignored when choosing a schedule. ``BLOCK_*`` /
    ``GROUP_*`` constexprs are different: those are portable meta-parameters Merlin may later treat
    as schedule candidates, so they live in ``constexprs``.
    """

    function: Any
    args: tuple[KernelArg, ...]
    grid: GridSpec
    constexprs: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    assumptions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.args:
            raise KernelSpecError("a kernel spec needs at least one argument")
        seen: set[str] = set()
        for a in self.args:
            if a.name in seen:
                raise KernelSpecError(f"duplicate argument name {a.name!r}")
            seen.add(a.name)
        if not any(a.kind == "pointer" for a in self.args):
            raise KernelSpecError("a kernel spec needs at least one pointer argument")
        if not self.outputs:
            raise KernelSpecError(
                "no argument is declared written — a kernel with no output would compile to a no-op; "
                "declare effect='write' (or 'readwrite') on the destination")

    @property
    def name(self) -> str:
        return getattr(self.function, "__name__", "kernel")

    @property
    def pointers(self) -> tuple[KernelArg, ...]:
        return tuple(a for a in self.args if a.kind == "pointer")

    @property
    def scalars(self) -> tuple[KernelArg, ...]:
        return tuple(a for a in self.args if a.kind == "scalar")

    @property
    def inputs(self) -> tuple[KernelArg, ...]:
        """Pointers the kernel reads (a readwrite buffer is both an input and an output)."""
        return tuple(a for a in self.pointers if a.effect in ("read", "readwrite"))

    @property
    def outputs(self) -> tuple[KernelArg, ...]:
        return tuple(a for a in self.pointers if a.is_written)

    def arg(self, name: str) -> KernelArg:
        for a in self.args:
            if a.name == name:
                return a
        raise KernelSpecError(f"no argument named {name!r}")

    def signature(self) -> dict[str, str]:
        """Triton's ``{name: type}`` signature form (``*fp32`` for pointers, ``i32`` for scalars)."""
        return {a.name: (f"*{a.dtype}" if a.kind == "pointer" else a.dtype) for a in self.args}


@dataclass
class KernelArtifact:
    """What a compilation produced, and enough provenance to reproduce or attribute it.

    The per-stage hashes are the point: they make "the same kernel compiled the same way" a
    checkable claim, and they are what lets model kernel replacement prove the override was
    actually used rather than silently ignored.
    """

    kernel_name: str
    target: str
    stage_hashes: dict[str, str] = field(default_factory=dict)
    stage_paths: dict[str, Path] = field(default_factory=dict)
    triton_version: str | None = None
    merlin_revision: str | None = None
    target_contract_hash: str | None = None
    dialect_plan_hash: str | None = None
    capability_report: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    root: Path | None = None

    def manifest(self) -> dict[str, Any]:
        return {
            "kernel_name": self.kernel_name,
            "target": self.target,
            "triton_version": self.triton_version,
            "merlin_revision": self.merlin_revision,
            "target_contract_hash": self.target_contract_hash,
            "dialect_plan_hash": self.dialect_plan_hash,
            "stage_hashes": dict(sorted(self.stage_hashes.items())),
            "stage_paths": {k: str(v) for k, v in sorted(self.stage_paths.items())},
            "capability_report": self.capability_report,
            "validation": self.validation,
        }
