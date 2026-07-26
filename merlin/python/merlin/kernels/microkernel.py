"""Target-agnostic MICRO-KERNEL codegen granularity — the knobs an expert kernel exposes, as a
general compiler capability rather than an RVV-only feature.

An expert micro-kernel (XNNPACK f32-gemm-7x4v, OpenBLAS sgemm 8x8, a Gemmini tile program, …) is a
POINT in a small space of codegen decisions. Merlin's job is to make that space expressible BY CODE
GENERATION for ANY target — never by shipping a hand-written ukernel in the compiler (hand kernels
stay ceiling REFERENCES only, per the mining contract).

The spec below is deliberately target-neutral: it names WHAT the decision is, not how a particular
backend realizes it. Each target registers a RESOLVER that turns a :class:`MicrokernelSpec` into that
target's own realization (RVV -> transform-schedule features; Gemmini -> tile-program knobs; …). A
target that cannot express an axis says so honestly (``UnsupportedAxis``) instead of silently ignoring
it — the beam then records it as an open divergence rather than crediting a change that never happened.

ESCAPE HATCH (important): when the toolchain lacks a lowering for an axis (e.g. MLIR's scalable-vector
-> RVV path is incomplete, so a dynamic ``vsetvli`` loop cannot be emitted the ordinary way), a target
may realize it through ``llvmlower.custom_isa`` — a ``merlin.inline_asm`` marker lowered to real
``llvm.inline_asm`` / ``llvm.call_intrinsic``. That keeps the capability in CODE GENERATION (we emit
the instruction) with no llvm-project fork and no hand ukernel.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Callable

#: how the vector length is chosen for the inner loop.
VL_FIXED = "fixed"        # a compile-time vector length (e.g. RVV vsetivli) — simple, wastes tail lanes
VL_DYNAMIC = "dynamic"    # VL-agnostic loop sizing to the hardware VL with the tail folded in
VL_STRATEGIES = (VL_FIXED, VL_DYNAMIC)


class UnsupportedAxis(NotImplementedError):
    """A target cannot (yet) express a requested micro-kernel axis. Raised, never silently ignored:
    an unrealized axis must stay an OPEN divergence, not a fake win."""


@dataclass(frozen=True)
class MicrokernelSpec:
    """One point in the target-agnostic micro-kernel codegen space.

    MR / NR / KC     register-block rows / cols and reduction blocking (the classic tiling triple).
    unroll_m         hold the M rows as INDEPENDENT accumulators (fully unrolled) instead of one
                     2-D ``vector<MRxNR>`` value. Expert kernels do this — it is why XNNPACK can use
                     MR=7 while a 2-D-vector formulation needs a vectorization-friendly MR (measured
                     on K1: MR 3/5/6/7 collapse to 193-279x off, MR 4 is 5.0x).
    vl_strategy      VL_FIXED | VL_DYNAMIC (see above). VL_DYNAMIC may require the ISA escape hatch.
    pack             pre-pack operands into unit-stride panels (kills strided/transposed inner reads).
    k_block          actually BLOCK the reduction by KC for cache reuse (vs only register-tiling it).
                     Measured motivation: the emitted inner loop walks B by a full row stride per K
                     step, touching a new cache line every iteration -- same instruction count as the
                     expert kernel but ~5x slower, i.e. memory-stalled. Cache blocking is the standard
                     fix, and KC was previously a DEAD parameter (the schedule ignored it entirely).
    """

    MR: int = 4
    NR: int = 16
    KC: int = 16
    unroll_m: bool = False
    vl_strategy: str = VL_FIXED
    pack: bool = False
    k_block: bool = False

    def __post_init__(self) -> None:
        if self.vl_strategy not in VL_STRATEGIES:
            raise ValueError(f"vl_strategy must be one of {VL_STRATEGIES}, got {self.vl_strategy!r}")
        for name in ("MR", "NR", "KC"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")

    @classmethod
    def from_knobs(cls, mk: dict[str, Any]) -> "MicrokernelSpec":
        """Build from a package's ``microkernel`` knob block (unknown keys rejected loudly)."""
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(mk) - known
        if unknown:
            raise ValueError(f"unknown microkernel knob(s) {sorted(unknown)}; known: {sorted(known)}")
        return cls(**{k: mk[k] for k in mk})

    def to_knobs(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

    def with_(self, **kw: Any) -> "MicrokernelSpec":
        """A neighbouring point (what a beam mutation produces)."""
        return replace(self, **kw)


#: The micro-kernel axes a beam/proposer MAY explore, and the ones PRUNED because they were measured
#: inert or structurally wrong. Every axis stays RESOLVABLE (``resolve`` still emits a schedule for it,
#: so a package or a test may pin it), but a proposer must not spend beam budget PROPOSING a pruned
#: axis as a candidate win — it burns a certify slot to (best case) reproduce the parent, (worst case)
#: regress. Two levers were inert/wrong exactly this way (they looked wired at every layer while the
#: emitted code was flat or slower — see rvvgen/beam._emitted_digest):
#:   * ``KC``       — INERT. The v3 recipe tiles the reduction K by 1 regardless of KC (KC only names
#:                    the outer register-block trip conceptually), so proposing different KC values on
#:                    the default recipe changes no emitted instruction (measured flat across KC). The
#:                    genuine reduction-blocking lever is ``k_block`` (a distinct axis), not KC-tuning.
#:   * ``unroll_m`` — STRUCTURALLY WRONG. Holding M as MR independent accumulators emits MR sequential
#:                    K-loops with B-reuse=1 (measured ~2.4x SLOWER than the 2-D vector<MRxNR> block),
#:                    so raising MR under unroll_m spends budget on a known regression.
#: Prove a lever by a measured emitted-code delta (PMU instret / decoded stream), never by schedule
#: text — that is what caught both of these. See also docs/design/expert_gap_attribution.md.
PRUNED_AXES: dict[str, str] = {
    "KC": "inert: the v3 schedule tiles K by 1 regardless of KC (measured flat emitted code); use "
          "the k_block axis for genuine reduction blocking",
    "unroll_m": "structurally wrong: MR sequential K-loops, B-reuse=1, measured ~2.4x slower than "
                "the 2-D vector<MRxNR> register block",
}

#: The full set of tunable micro-kernel axes (every MicrokernelSpec field that names a codegen
#: decision — the ``op`` identity is not one). ``proposable_axes`` is this minus the pruned ones.
_TUNABLE_AXES: tuple[str, ...] = ("MR", "NR", "KC", "unroll_m", "vl_strategy", "pack", "k_block")


def proposable_axes() -> list[str]:
    """The micro-kernel axes a beam/proposer SHOULD explore — the tunable axes minus the pruned
    (inert/structurally-wrong) ones. A proposer over the micro-kernel space MUST consult this so it
    does not burn certify budget on a lever that cannot produce a win."""
    return [a for a in _TUNABLE_AXES if a not in PRUNED_AXES]


def is_axis_proposable(axis: str) -> bool:
    """True if ``axis`` is a micro-kernel axis worth PROPOSING (tunable and not pruned)."""
    return axis in _TUNABLE_AXES and axis not in PRUNED_AXES


#: target name -> resolver. A resolver maps a spec to that target's realization — for a
#: schedule-driven target that is a list of compiler-feature names; other targets may return their
#: own directive objects. The beam only passes it through, so the type is target-defined.
_RESOLVERS: dict[str, Callable[[MicrokernelSpec], Any]] = {}


def register_resolver(target: str, fn: Callable[[MicrokernelSpec], Any]) -> None:
    """Register the per-target realization of the micro-kernel space (idempotent overwrite)."""
    _RESOLVERS[target] = fn


def registered_targets() -> list[str]:
    return sorted(_RESOLVERS)


def resolve(target: str, spec: MicrokernelSpec) -> Any:
    """Realize ``spec`` for ``target``. Raises if the target has no resolver (never silently no-ops)."""
    fn = _RESOLVERS.get(target)
    if fn is None:
        raise UnsupportedAxis(
            f"target {target!r} has no micro-kernel resolver registered (have: {registered_targets()}). "
            "Register one so the micro-kernel granularity is expressible for this target.")
    return fn(spec)


# ---------------------------------------------------------------------------------------------
# SHAPE AWARENESS — the workload side of the same space.
#
# A register block is not legal in the abstract: it is legal FOR A SET OF CONTRACTIONS. When a
# blocking factor does not divide the extent of the dim it blocks, the tail iteration is PARTIAL and
# the vectorizer must MASK it — and masking a PARALLEL dim is what breaks (or de-optimizes) real
# backends, while masking the REDUCTION dim is harmless (the reduction tail just contributes fewer
# terms). That asymmetry is a property of the contraction, not of RVV, so it lives here.
#
# Measured on the RVV/LLVM-23 path (`llvmlower.lower`, int8 W8A8 whole-model pipeline), 57 synthetic
# `linalg.matmul` cells over MR x NR x shape plus 18 whole-model lowerings (small_llama / openvla /
# bitvla):
#     * masking the REDUCTION dim (K) is always fine   -- M=8,N=128,K=344 lowers at MR=4,NR=16;
#     * masking a PARALLEL dim (M by MR, or N by NR) is a HARD lowering failure
#       ('vector.mask' op expects only one operation to mask / an unlowerable
#       builtin.unrealized_conversion_cast), EXCEPT when the block is 1 along the OTHER parallel dim
#       (MR=1 -> the C tile is effectively rank-1 and the masked transfer_write does lower) and the
#       block does not EXCEED the extent it masks.
# The fp32 pipeline does not hard-fail on the same cells -- it silently degrades instead (spike
# instret at M=17,N=128,K=128: 159,999 unmasked vs 5,435,000 masked, ~34x). So "does not mask a
# parallel dim" is a CORRECTNESS gate on one datapath and a 30x PERFORMANCE gate on the other.
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ContractionShape:
    """One contraction the micro-kernel space is asked to realize, named by ITERATOR TYPE.

    Deliberately NOT "M/N/K": a target-agnostic policy only needs to know which extents belong to
    PARALLEL iteration dims (the ones a register block tiles and whose tail must be masked) and which
    belong to REDUCTION dims (whose tail is harmless). ``op`` is the op class the target schedules
    separately (``linalg.matmul`` vs ``linalg.batch_matmul`` get independent tile sizes), so a policy
    can solve each class on its own instead of forcing one blocking on both.

    ``parallel`` is outer-to-inner as the op writes them (matmul -> (M, N); batch_matmul ->
    (B, M, N)), which is the order a target's tile-size vector uses.
    """

    op: str
    parallel: tuple[int, ...]
    reduction: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for f in ("parallel", "reduction"):
            v = getattr(self, f)
            if not isinstance(v, tuple) or any(int(x) <= 0 for x in v):
                raise ValueError(f"{f} must be a tuple of positive extents, got {v!r}")


def masked_parallel_dims(block: "tuple[int, ...]", extents: "tuple[int, ...]") -> tuple[int, ...]:
    """Indices of the parallel dims ``block`` would MASK on ``extents`` (partial tail iteration).

    A tile size of 0 (or 1) never masks: 0 means "do not tile this dim" and 1 always divides. The
    result is the general hazard set every target's policy reasons about."""
    return tuple(i for i, (b, e) in enumerate(zip(block, extents))
                 if int(b) not in (0, 1) and int(e) % int(b) != 0)


def largest_divisor_at_most(n: int, cap: int) -> int:
    """The largest divisor of ``n`` that is <= ``cap`` (always >= 1).

    This is how a shape-aware policy DERIVES a legal blocking from the knob space instead of
    hard-coding one: the requested factor is an upper bound (bigger register blocks are better up to
    the register file), and the workload's extents decide how much of it is reachable without
    masking. Applied to ``gcd`` over a set of extents it yields the largest block legal for ALL of
    them at once."""
    n, cap = int(n), int(cap)
    if cap < 1:
        raise ValueError(f"cap must be >= 1, got {cap}")
    for d in range(min(cap, n), 0, -1):
        if n % d == 0:
            return d
    return 1


#: target name -> shape-aware policy. Same contract as a resolver (the realization type is
#: target-defined), but it additionally sees the contractions it must be legal for.
_SHAPE_POLICIES: dict[str, Callable[[MicrokernelSpec, "tuple[ContractionShape, ...]"], Any]] = {}


def register_shape_policy(
        target: str,
        fn: Callable[[MicrokernelSpec, "tuple[ContractionShape, ...]"], Any]) -> None:
    """Register ``target``'s SHAPE-AWARE realization of the micro-kernel space (idempotent)."""
    _SHAPE_POLICIES[target] = fn


def has_shape_policy(target: str) -> bool:
    return target in _SHAPE_POLICIES


def resolve_for_shapes(target: str, spec: MicrokernelSpec,
                       shapes: "Sequence[ContractionShape]" = ()) -> Any:
    """Realize ``spec`` for ``target`` given the contractions it must cover.

    Falls back to the shape-BLIND :func:`resolve` when the caller observed no shapes or the target
    registered no policy — so a target that has not opted in keeps byte-identical behavior."""
    fn = _SHAPE_POLICIES.get(target)
    if fn is None or not shapes:
        return resolve(target, spec)
    return fn(spec, tuple(shapes))
