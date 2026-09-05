"""Fork-scoped, **default-off** compiler-feature registry for ``impr_<target>_vX`` forks.

The baseline RVV compiler (``pipeline.RVV_TRANSFORM_SCHEDULE`` + ``build_rvv_pipeline`` pass list)
is FROZEN — kernel-mining experiments never edit it. KNOB/FLAG/PATTERN improvements ride the
existing ``transform_schedule=``/``cflags_override=`` seams. But PASS- and HEURISTIC-class
improvements need actual compiler code, which still must not perturb the baseline. This registry
is how: each improvement is a NAMED feature with a hook that edits the pipeline pass list and/or
the transform schedule; an ``impr_`` fork's manifest lists the ``compiler_features`` it enables,
threaded through ``build_app`` -> ``lower_*`` -> ``build_rvv_pipeline(features=...)``.

Invariant: with ``features == frozenset()`` (the baseline / any non-impr build), the hooks are
never invoked, so the emitted pipeline string and schedule are **byte-identical** to today
(guarded by ``test_impr_features``). A feature only changes codegen when a fork explicitly enables
it, so it can be measured against the immutable baseline.
"""
from __future__ import annotations

from .copy_expand import FEATURE as _EXPAND_COPY_FEATURE
from .selfcopy import FEATURE as _SELF_COPY_FEATURE
from .transpose_fuse import FEATURE as _FUSE_TRANSPOSE_FEATURE
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ImprFeature:
    """One named, default-off compiler change.

    ``edit_pipeline`` rewrites the ordered MLIR pass list (PASS/HEURISTIC at the pipeline level).
    ``edit_schedule`` rewrites the transform-dialect schedule text (PATTERN at the schedule level
    that goes beyond what a hand-written ``schedule.mlir`` knob expresses). Either may be None.
    Both are pure functions (input -> new value); they must be deterministic.

    ``schedule_replace`` flags an ``edit_schedule`` that IGNORES its input text and emits a full
    replacement schedule (e.g. the tiled-vfmacc / accumulator-resident recipes whose structure
    differs fundamentally from the tile+vectorize baseline). Two such features cannot COMPOSE — the
    last one in sorted order would silently clobber the other's clamp — so ``apply_schedule`` refuses
    to apply more than one full-replacement schedule feature at once (the composed, whole-model-safe
    config is a SINGLE feature with all the clamps inherent). Additive edits (``schedule_replace`` =
    False, e.g. ``lmul_widen_n``'s string substitution) layer freely on top of a replacement.

    ``edit_cflags`` rewrites the clang flag list for the model object (KNOB at the BACKEND level).
    It exists because some codegen decisions are not expressible in the IR at all: the width of the
    vector register group the auto-vectorizer asks for is a backend query, not a type, so no tile
    size or schedule edit reaches it directly -- the only way this registry could move it before was
    to widen the N tile and let the group width follow, which changes the vectorized shapes as a side
    effect and can push a transfer into a masked form the backend rejects (see ``lmul_group``). A
    ``None`` hook leaves the flags untouched, so a build that names no cflag-editing feature compiles
    byte-identically.

    ``implies`` names features this one CANNOT be measured without -- lowering hygiene that is a
    property of the recipe's SHAPE rather than a tuning choice. It exists because a default-off
    feature whose payoff is cancelled by a separate default-off fix is, in practice, an inert lever:
    the beam has to discover the conjunction, and everyone who names the feature directly in
    ``compiler_features`` gets the cancelled version. See ``_tile_epilogue_hygiene``.
    """
    name: str
    action_class: str  # "PASS" | "HEURISTIC" | "PATTERN"
    description: str
    edit_pipeline: Callable[[list[str]], list[str]] | None = None
    edit_schedule: Callable[[str], str] | None = None
    edit_cflags: Callable[[list[str]], list[str]] | None = None
    schedule_replace: bool = False
    implies: frozenset[str] = frozenset()


_REGISTRY: dict[str, ImprFeature] = {}


def register(feature: ImprFeature) -> ImprFeature:
    if feature.name in _REGISTRY:
        raise ValueError(f"duplicate impr feature {feature.name!r}")
    _REGISTRY[feature.name] = feature
    return feature


def _try_lazy_register(name: str) -> bool:
    """Auto-register a v3 micro-kernel tuning point from its NAME (accum_resident_v3_<MR>_<NR>_<KC>).

    The register block is a continuous, beam-tunable knob space, but the lowering runs in a SUBPROCESS
    that re-imports this module — so a point registered on demand in the parent is invisible there and
    the feature fails to resolve (observed: every dynamically-registered MR failed at K2 while the
    pre-registered grid worked). Deriving the point from the name makes resolution reproducible in ANY
    process, which is what actually makes the space continuous.

    The same holds for the two other on-demand v3 families, which were previously NOT derivable from
    their name (their prefix is ``accum_resident_v3vl_`` / ``accum_resident_v3p_``, so the
    ``accum_resident_v3_`` test rejected them): a package that names one of those points directly in
    ``compiler_features`` -- rather than reaching it through a ``microkernel`` knob block, which
    registers it as a side effect of resolving -- failed with an "unknown impr feature" KeyError.
    Each family is registered from its own arity, so a name is either derivable or an honest error."""
    parts = name.split("_")
    tails: dict[str, tuple[int, "Callable[..., str]"]] = {
        "accum_resident_v3_": (3, ensure_v3_microkernel),          # MR, NR, KC
        "accum_resident_v3vl_": (3, ensure_v3_scalable_microkernel),  # MR, NR, KC
        "accum_resident_v3p_": (5, ensure_v3_perop_microkernel),   # MR_mm, NR_mm, MR_bmm, NR_bmm, KC
    }
    for prefix, (arity, make) in tails.items():
        if not name.startswith(prefix):
            continue
        args = parts[len(prefix.rstrip("_").split("_")):]
        if len(args) != arity:
            return False
        try:
            make(*(int(a) for a in args))
        except ValueError:
            return False
        return name in _REGISTRY
    return False


def get(name: str) -> ImprFeature:
    if name not in _REGISTRY and not _try_lazy_register(name):
        raise KeyError(f"unknown impr feature {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def known() -> list[str]:
    return sorted(_REGISTRY)


def normalize(features) -> frozenset[str]:
    """Accept None / list / set / frozenset -> validated frozenset (every name must be registered),
    closed under ``ImprFeature.implies``.

    The closure is here, and not at each call site, because ``feats = normalize(...)`` is the single
    point every consumer reads: the runner selection, the schedule edits, and the argv gates for the
    self-copy erase / transpose fusion all key off this one set. Expanding anywhere later would let
    them disagree about which features are on.

    Empty in => empty out, so the frozen baseline stays byte-identical.
    """
    if not features:
        return frozenset()
    fs = frozenset(features)
    for n in fs:
        get(n)  # raises on unknown
    while True:
        grown = fs | frozenset().union(*(get(n).implies for n in fs)) if fs else fs
        if grown == fs:
            return fs
        for n in grown - fs:
            get(n)  # an implied name must be registered too
        fs = grown


def apply_pipeline(passes: list[str], features: frozenset[str]) -> list[str]:
    """Apply each enabled feature's pipeline edit, in a stable (sorted) order. Empty -> unchanged
    list object content (identity), so the joined string is byte-identical to the baseline."""
    if not features:
        return passes
    out = list(passes)
    for name in sorted(features):
        f = get(name)
        if f.edit_pipeline is not None:
            out = f.edit_pipeline(out)
    return out


class CompositionError(ValueError):
    """Two features both fully REPLACE the transform schedule -> they cannot compose (one would
    silently clobber the other). The whole-model-safe composed config is a SINGLE feature with all
    clamps inherent; raise instead of letting sorted order pick a winner."""


def apply_schedule(schedule_text: str, features: frozenset[str]) -> str:
    """Apply each enabled feature's schedule edit so features COMPOSE rather than clobber.

    Full-schedule-REPLACEMENT features (``schedule_replace=True``, e.g. the tiled-vfmacc /
    accumulator-resident recipes whose ``edit_schedule`` ignores its input) are applied FIRST, and
    at most ONE is allowed — two would make the result depend on sorted order, silently discarding
    the other's clamp (the bug WORK-ITEM 2 fixes). Additive edits (``schedule_replace=False``, pure
    text transforms like ``lmul_widen_n``) then layer on top in stable order. Empty -> unchanged text.
    """
    if not features:
        return schedule_text
    replacers = [get(n) for n in sorted(features)
                 if get(n).edit_schedule is not None and get(n).schedule_replace]
    if len(replacers) > 1:
        raise CompositionError(
            "cannot compose multiple full-schedule-replacement features "
            f"{[f.name for f in replacers]}: each emits a complete transform schedule and would "
            "clobber the others. Enable the single composed feature that carries all clamps "
            "inherent (e.g. accumulator_resident_wholemodel) instead of stacking replacements.")
    out = schedule_text
    if replacers:                              # the one replacement runs first (input ignored)
        out = replacers[0].edit_schedule(out)
    for name in sorted(features):              # then additive edits layer on top
        f = get(name)
        if f.edit_schedule is not None and not f.schedule_replace:
            out = f.edit_schedule(out)
    return out


def apply_cflags(cflags, features: frozenset[str]) -> list[str]:
    """``cflags`` with every enabled feature's ``edit_cflags`` hook applied, in stable name order.

    Empty ``features`` (the baseline, and every build that names only IR-level features) returns the
    list unchanged, so the emitted object is byte-identical -- the same invariant
    :func:`apply_pipeline` / :func:`apply_schedule` carry, extended to the backend flags.

    Unlike a schedule replacement there is no composition hazard here: each hook appends its own
    ``-mllvm`` option and the flags are independent, so two cflag features layer the way two additive
    schedule edits do. A feature that wanted to CONTRADICT another on the same option would have to
    be caught by whoever adds it; today the only such option is the register-group width, and the
    registrar mints one feature per width so two of them cannot be named without saying so.

    WHO CALLS THIS. Every builder that turns the lowered ``.ll`` into the model object must, or a
    cflag feature is INERT on that path -- named, resolved, and emitting byte-identical code, which
    is the failure mode hardest to notice from the outside. Wired today:
    ``runtime.backends.zephyr_model.build_app`` and ``runtime.backends.spike_model.build``. NOT yet
    wired: ``mining.k1.build_k1_binary`` (and its staged sibling), which assembles its model-object
    flag list inline; a cflag feature therefore has no effect on that path until the same call is
    added there.
    """
    out = list(cflags)
    if not features:
        return out
    for name in sorted(features):
        f = get(name)
        if f.edit_cflags is not None:
            out = list(f.edit_cflags(out))
    return out


# ---- registered features ------------------------------------------------------------
# Keep this list small and evidence-justified. Each entry corresponds to a typed CompilerAction
# (PASS/HEURISTIC/PATTERN) surfaced by the action catalog from a mined kernel divergence.

# The vfmacc-forming schedule (developed by MLIR-level iteration, R-cont): vectorize the func's
# contraction to a real `vector.contract`, lower it via the OUTERPRODUCT strategy to a chain of
# `vector.outerproduct{kind=add}` (acc + a⊗b — the fused MAC), then `lower_outerproduct` turns each
# into `vector.fma` -> `llvm.intr.fmuladd` -> the RISC-V backend emits `vfmacc`. The baseline's
# [4,8,1] K=1 tiling never forms a contraction (so outerproduct/K-tile/fast-math were all measured
# no-ops); vectorize_children DOES form it. Appropriate for kernel-sized contraction workloads (the
# mining testbed) — NOT a whole transformer (vectorize_children there explodes vector.extract, per
# pipeline.py). Gated to the impr fork; baseline schedule untouched.
_VFMACC_SCHEDULE = """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %v = transform.structured.vectorize_children_and_apply_patterns %f : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %v {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
      transform.apply_patterns.vector.lower_outerproduct
      transform.apply_patterns.vector.lower_masked_transfers
      transform.apply_patterns.vector.lower_transpose
      transform.apply_patterns.vector.lower_shape_cast
    } : !transform.any_op
    transform.yield
  }
}
"""


def _vfmacc_schedule_edit(text: str) -> str:
    """Replace the baseline schedule with the vfmacc-forming recipe (full replacement — the
    structure differs fundamentally from the tile+vectorize baseline)."""
    return _VFMACC_SCHEDULE


# Combined / production-scalable variant: the CORRECT, SCALABLE, bounded-code tiled vfmacc.
#
# The earlier `fused_vfmacc_tiled` draft tiled [4,16,0] (K UNTILED) then ran
# `vectorize_children_and_apply_patterns` on the whole func. That had three fatal flaws, all now
# fixed here:
#   (a) K untiled => the [MR,NR,K] contraction read FULL-K vectors (vector<4x64>, vector<64x16>)
#       and fully unrolled the K reduction (64 outerproducts * MR = 256 fma), so .text grew with K
#       (blowing the JAL +-1 MB reach at 128^3) AND the oversized vectors spilled. ROOT CAUSE of the
#       M>=64 `tohost=1337` spike fault: a `trap_store_access_fault` (epc in vprintfmt, tval far
#       outside RAM) — the register allocator spilled those 4 KB vector<64x16>/vector<4x64> temps
#       and at M>=64 the spill overran the stack, corrupting adjacent BSS (printf's byte-count
#       counter loaded as 0x40040030 -> store to 0xc1058630 faulted). It "passed" at 32^3 only
#       because the smaller spill stayed in bounds. Bounding the tile (KC<=16) removes the oversized
#       vectors and the spill, so the fault is gone at 64 AND 128.
#   (b) `vectorize_children` on the WHOLE func scalarizes every non-contraction generic (the
#       transposes / elementwise of a real model) into tens of thousands of `vector.extract`s —
#       the whole-model explosion `pipeline.py` warns about. (Measured: 4120 extracts on a
#       matmul+transpose+add toy vs 84 with the scoped vectorize here.)
#   (c) running form-contract AND lower-contraction/outerproduct in ONE greedy `apply_patterns`
#       block left the contract un-lowered (mul+add, no fma).
#
# The fixed recipe (developed by mlir-opt iteration, verified bit-exact + bounded at 32/64/128):
#   1. TILE matmul [MR=4, NR=16, KC=16] and batch_matmul [1,4,16,16] -> M, N AND K are all scf.for
#      LOOPS. KC>1 (not 1) so a real reduction tile survives (KC=1 collapses to a bare mul+add and
#      no contract can form). KC=16 (over KC=4) cuts K-loop trip count 4x -> ~4x fewer retired
#      instructions on the spike proxy (5.2M->1.33M @ 64^3) while staying bounded; KC=16 keeps the
#      tile vectors small (vector<4x16>, vector<16x16>) so no spill/fault.
#   2. `transform.structured.vectorize %t` (SCOPED to the tiled op handle ONLY — never
#      vectorize_children on the func) => the non-contraction generics are left untouched, so NO
#      vector.extract explosion; they lower scalar via convert-linalg-to-loops as in the baseline.
#   3. `transfer_permutation_patterns` + `reduction_to_contract` rebuild a real `vector.contract`
#      from the scoped-vectorize `mulf`+`multi_reduction` (scoped vectorize alone never forms a
#      contract — that is exactly why the old recipe needed vectorize_children).
#   4. lower_contraction(outerproduct) and lower_outerproduct in SEPARATE apply_patterns blocks
#      -> vector.outerproduct{add} -> vector.fma -> llvm.intr.fmuladd -> vfmacc.
# Result: the inner body is a CONSTANT MR*KC = 64 fma regardless of M/N/K (64 fma @ 32, 64 AND
# 128) — bounded .text => no JAL wall, no register-spill stack-overrun fault, and whole-model-safe
# (84 extracts on the mixed toy vs 4120 before).
_VFMACC_TILED_SCHEDULE = """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t, %l:3 = transform.structured.tile_using_for %mm tile_sizes [4, 16, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %t vector_sizes [4, 16, 16] : !transform.any_op
    %bm = transform.structured.match ops{["linalg.batch_matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt, %bl:4 = transform.structured.tile_using_for %bm tile_sizes [1, 4, 16, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %bt vector_sizes [1, 4, 16, 16] : !transform.any_op
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    } : !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    } : !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_outerproduct
    } : !transform.any_op
    transform.yield
  }
}
"""


register(ImprFeature(
    name="lmul_widen_n",
    action_class="KNOB",
    description="mined lmul_grouping_policy: widen the matmul N tile/vector 8->16 so the emitted "
                "vector group uses a higher LMUL (m2->m4). Composes with other features in autotune.",
    edit_schedule=lambda t: t.replace("tile_sizes [4, 8, 1]", "tile_sizes [4, 16, 1]").replace(
        "vector_sizes [4, 8, 1]", "vector_sizes [4, 16, 1]"),
))


#: Prefix of the DIRECT register-group-width features. ``lmul_widen_n`` above reaches LMUL the only
#: way this registry could before: widen the N tile and let the backend's worst-case group sizing
#: follow. That is an INDIRECT route and it moves two things at once -- MEASURED on
#: small_llama_int8, taking the M-pad block from mr1_nr32 to mr1_nr64 to chase the expert's LMUL
#: went 5,180,908 ns -> 20,450,661 ns and put `_mlir_ciface_forward` on a scalar fallback (the wider
#: tail transfer becomes a masked multi-op `vector.mask` LLVM 23 rejects, so the transform
#: interpreter raises and the whole model lowers scalar: identical numerics, 4x the wall).
#:
#: These features set the group width itself and touch no tile size, so the vectorized shapes -- and
#: therefore the tail masking -- are exactly what they were. See :mod:`merlin.llvmlower.lmul_group`
#: for the derivation and for the measurement that the emitted `vsetvli` actually moves.
LMUL_GROUP_PREFIX = "lmul_group_m"

#: The REQUEST for a derived register-group width, as a feature name. Deliberately NOT registered:
#: like ``PEROP_BLOCK_NAME``, it is a sentinel that ``zephyr_model.prepare_for_lowering`` resolves
#: against the PREPARED IR -- the element widths the pipeline will actually see, plus the board's
#: VLEN -- and swaps for the concrete ``lmul_group_m<N>`` before lowering. Reaching ``normalize``
#: unresolved SHOULD raise, because a request nobody resolved must not read as a width.
#:
#: This is the name the ``vector.lmul`` action-catalog seam points at: the route knows the axis, the
#: IR knows the arithmetic, and neither of them should be writing down a number.
LMUL_GROUP_SENTINEL = "lmul_register_group"


def lmul_group_feature(lmul: int) -> str:
    """The name of the default-off feature that pins the auto-vectorizer's register group to ``lmul``.

    Every whole-register width is registered eagerly (there are four), so a name resolves in the
    lowering SUBPROCESS as well as in the parent -- the same reproducibility requirement
    :func:`_try_lazy_register` exists for.
    """
    from .lmul_group import LMUL_LADDER
    if int(lmul) not in LMUL_LADDER:
        from .lmul_group import LmulDerivationError
        raise LmulDerivationError(f"LMUL={lmul!r} is not one of {LMUL_LADDER}")
    return f"{LMUL_GROUP_PREFIX}{int(lmul)}"


def ensure_lmul_group(*, operand_bits: int, acc_bits: int, vlen: int | None = None,
                      max_group_elems: int | None = None) -> str:
    """DERIVE the register-group width from the datapath and return the feature that pins it.

    The width is ``lmul_group.group_lmul`` -- the smallest whole-register group at which the
    narrowest operand's group stops being a fraction, capped by what the target's VLEN can usefully
    hold. Nobody types a 4: an ``i8 x i8 -> i32`` contraction derives 4 (which is what the expert
    XNNPACK qd8 kernel runs at), an ``f32`` one derives 1, a ``bf16 -> f32`` one derives 2.
    """
    from .lmul_group import group_lmul
    return lmul_group_feature(group_lmul(operand_bits=operand_bits, acc_bits=acc_bits,
                                         vlen=vlen, max_group_elems=max_group_elems))


def ensure_lmul_group_for_elem_types(a: str, b: str, c: str, *, vlen: int | None = None,
                                     max_group_elems: int | None = None) -> str:
    """:func:`ensure_lmul_group` for a contraction named by its MLIR element types ``a x b -> c``."""
    from .lmul_group import group_lmul_for_elem_types
    return lmul_group_feature(group_lmul_for_elem_types(a, b, c, vlen=vlen,
                                                        max_group_elems=max_group_elems))


def _register_lmul_groups() -> list[str]:
    from .lmul_group import LMUL_LADDER, lmul_cflags
    names = []
    for _lmul in LMUL_LADDER:
        _name = f"{LMUL_GROUP_PREFIX}{_lmul}"
        register(ImprFeature(
            name=_name, action_class="KNOB",
            description=(
                f"Pin the vector REGISTER-GROUP WIDTH of auto-vectorized code to LMUL={_lmul}, "
                f"directly (an -mllvm backend option), without changing any tile or vector size. "
                f"This is the seam the `vector.lmul` divergence actually wants: the N-tile route to "
                f"the same axis also widens the tail transfer, which is a whole-model scalar-fallback "
                f"cliff. Derive the width with `ensure_lmul_group(...)` rather than naming a number "
                f"-- it is acc_bits/operand_bits, capped by what the VLEN can hold. Default-off; a "
                f"build that does not name it gets byte-identical flags."),
            edit_cflags=(lambda c, _l=_lmul: [*c, *lmul_cflags(_l)]),
        ))
        names.append(_name)
    return names


#: The whole ladder, registered eagerly. Naming a width directly is legal (a search may want to
#: bracket the derived one); ``ensure_lmul_group`` is how a caller gets the derived answer.
LMUL_GROUP_NAMES: tuple[str, ...] = tuple(_register_lmul_groups())


register(ImprFeature(
    name="fused_vfmacc_tiled",
    action_class="PASS",
    description="CORRECT, SCALABLE, bounded-code tiled vfmacc: tile matmul [MR=4,NR=16,KC=4] and "
                "batch_matmul [1,4,16,4] so M,N,K are all scf.for LOOPS, SCOPED-vectorize the tile "
                "only (no whole-func vectorize_children => whole-model-safe, no vector.extract "
                "explosion), rebuild the contract via reduction_to_contract, then "
                "outerproduct->vector.fma->vfmacc. Inner body = constant MR*KC=16 fma at any "
                "M/N/K => .text bounded (no JAL +-1MB wall) and applicable to whole models. "
                "Benchmarked vs the full-unroll fused_vfmacc_contraction.",
    edit_schedule=lambda _t: _VFMACC_TILED_SCHEDULE,
    schedule_replace=True,
))


# Same recipe, surfaced under the name the kernel-policy mining task refers to. `fused_vfmacc_tiled`
# is kept for the cross-framework matrix column that already names it; `fused_vfmacc_scalable` is the
# preferred name going forward (it is the whole-model-safe, bounded-code vfmacc).
register(ImprFeature(
    name="fused_vfmacc_scalable",
    action_class="PASS",
    description="Alias of the fixed fused_vfmacc_tiled recipe: bounded-code (constant MR*KC inner "
                "body), K-as-loop, scoped-vectorize (whole-model-safe) tiled vfmacc. Correct + "
                "bit-exact at 32/64/128; the whole-model-safe vfmacc for e2e.",
    edit_schedule=lambda _t: _VFMACC_TILED_SCHEDULE,
    schedule_replace=True,
))


# ---- parameterized tiled-vfmacc tuning grid -----------------------------------------
# The fixed recipe above hard-codes the tile (MR=4, NR=16, KC=16). To CLOSE the gap to the expert
# GEMMs (OpenBLAS/XNNPACK) by tuning the register-tile, this factory emits the SAME bounded-code,
# whole-model-safe recipe with an arbitrary (MR, NR, KC) tile. Larger MR amortizes the A-reload
# across the K loop; larger KC cuts K-loop trip count (fewer loop-overhead instructions retired);
# both are bounded as long as the MR*NR accumulator + the tile vectors stay in the vector register
# file (register-pressure ceiling) — past it the allocator spills and faults, recorded not_run.
def vfmacc_tiled_schedule(MR: int, NR: int, KC: int) -> str:
    """Return the bounded tiled-vfmacc transform schedule for register-tile (MR, NR, KC).

    matmul       -> tile_sizes [MR, NR, KC]      + scoped vectorize [MR, NR, KC]
    batch_matmul -> tile_sizes [1, MR, NR, KC]   + scoped vectorize [1, MR, NR, KC]
    Identical structure to _VFMACC_TILED_SCHEDULE (transfer_permutation + reduction_to_contract ->
    lower_contraction(outerproduct) -> lower_outerproduct -> vector.fma -> vfmacc) so the only
    variable is the tile.
    """
    return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t, %l:3 = transform.structured.tile_using_for %mm tile_sizes [{MR}, {NR}, {KC}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %t vector_sizes [{MR}, {NR}, {KC}] : !transform.any_op
    %bm = transform.structured.match ops{{["linalg.batch_matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt, %bl:4 = transform.structured.tile_using_for %bm tile_sizes [1, {MR}, {NR}, {KC}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %bt vector_sizes [1, {MR}, {NR}, {KC}] : !transform.any_op
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    }} : !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.lower_outerproduct
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


def _register_tiled_grid() -> list[str]:
    """Register a GRID of default-off tuning features `vfmacc_t_<MR>_<NR>_<KC>`.

    Grid: MR in {4,8}, NR in {16,32}, KC in {16,32,64}. Every tile divides 64 cleanly (the sweep
    shape), so no tail path. Each is its own ImprFeature whose edit_schedule emits
    vfmacc_tiled_schedule(MR,NR,KC); all default-off, so the baseline stays byte-identical.
    Returns the list of registered names (sweep order).
    """
    names: list[str] = []
    for MR in (4, 8):
        for NR in (16, 32):
            for KC in (16, 32, 64):
                nm = f"vfmacc_t_{MR}_{NR}_{KC}"
                register(ImprFeature(
                    name=nm,
                    action_class="PASS",
                    description=f"Tiled-vfmacc tuning point: register-tile (MR={MR}, NR={NR}, "
                                f"KC={KC}). Bounded-code, whole-model-safe tiled vfmacc with this "
                                f"tile (inner body = MR*KC fma). Default-off tuning-grid feature.",
                    edit_schedule=(lambda _t, _MR=MR, _NR=NR, _KC=KC:
                                   vfmacc_tiled_schedule(_MR, _NR, _KC)),
                    schedule_replace=True,
                ))
                names.append(nm)
    return names


TILED_GRID_NAMES: list[str] = _register_tiled_grid()


# ---- operand-PACKING tiled vfmacc ---------------------------------------------------
# The (MR,NR,KC) register-tile tuning sweep proved that tuning the tile closes ~none of the 15.7x
# gap to OpenBLAS/XNNPACK (1.32M vs 84,483 cyc @ 64^3). The residual is the strided per-tile vector
# transfers + loop overhead that the experts eliminate by PACKING operands into contiguous tiled
# panels (the #1 mined `packed_rhs_policy`: OpenBLAS A-ncopy / B-tcopy, XNNPACK goi-prepack). This
# feature reproduces that in the transform dialect:
#
#   1. `transform.structured.pack %matmul packed_sizes=[MR,NR,KC]` -> A becomes
#      tensor<M/MR x K/KC x MR x KC>, B becomes tensor<K/KC x N/NR x KC x NR>, C becomes
#      tensor<M/MR x N/NR x MR x NR>: the [MR,NR,KC] register tile lives as the CONTIGUOUS inner
#      dims of each packed panel (exactly OpenBLAS ncopy/tcopy / XNNPACK goi-prepack layout).
#   2. Tile the 6-loop packed generic's OUTER (M/MR, N/NR, K/KC) dims to 1 -> the loop body is ONE
#      [MR,NR,KC] register tile reading the contiguous packed panels.
#   3. `lower_pack` / `lower_unpack` -> the pack/unpack become pad + expand_shape + linalg.transpose
#      copy loops (convert-linalg-to-loops lowers them; the pipeline has no pack-lowering pass).
#   4. `fold_unit_extent_dims_via_reshapes` collapses the unit outer dims so the inner op is a clean
#      rank-3 [MR,KC]x[KC,NR] contraction over CONTIGUOUS tensor<MRxKC>/<KCxNR> slices.
#   5. Scoped-vectorize the inner op + transfer_permutation/reduction_to_contract ->
#      lower_contraction(outerproduct) -> lower_outerproduct -> vector.fma -> llvm.fmuladd -> vfmacc.
#
# Net: the inner-loop transfer_reads are UNIT-STRIDE contiguous (vector<MRxKC> from tensor<MRxKC>,
# vector<KCxNR> from tensor<KCxNR>) — no strided vector.transfer gather, no per-tile address
# recomputation — which is the lever the tile-tuning sweep could not pull. Default-off; baseline
# byte-identical (test_impr_features stays green). NOTE: pack of A/B/C is INSIDE the compiled
# function here (a microbench), so the measured PACK-INCLUDED cycles carry the one-time pack cost;
# the harness also reports PACK-EXCLUDED (inner-compute) by timing the matmul region only, which is
# the apples-to-apples vs the experts (who hoist the pack / use resident pre-packed weights).
def vfmacc_packed_schedule(MR: int, NR: int, KC: int, KS: int = 4) -> str:
    """Operand-packing tiled-vfmacc schedule for register-tile (MR, NR, KC), KC streamed by KS.

    pack matmul [MR,NR,KC] -> tile packed outer dims to 1 -> lower pack/unpack -> fold unit dims ->
    stream the inner KC reduction by KS -> scoped-vectorize the rank-3 inner op ->
    contract(outerproduct) -> vfmacc. Inner transfers are contiguous packed panels. KS<KC keeps the
    live B panel small (vector<KS x NR>) so the outerproduct does not spill the full vector<KCxNR>
    (the spill overran a buffer -> the M>=64 fault). Only `linalg.matmul` is packed (batch_matmul
    left to the scalable tiled recipe); the matmul microbench is the gap-closing target.
    """
    KS = min(KS, KC)
    return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %packed = transform.structured.pack %mm packed_sizes = [{MR}, {NR}, {KC}] : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    // B (operand 1) packs with inner tile [KC,NR]; transpose its inner dims to [NR,KC] so the
    // contraction needs NO per-iteration runtime vector.transpose of the B panel (the transpose
    // would otherwise scalarize into a vector.extract storm that spills at M>=64 -> wild-store
    // fault). With B pre-transposed in the pack the inner body is a clean MR*KC fma chain.
    %bpack = transform.get_producer_of_operand %packed[1] : (!transform.op<"linalg.generic">) -> (!transform.op<"linalg.pack">)
    %ptg, %bp2, %bu2 = transform.structured.pack_transpose %bpack with_compute_op(%packed) inner_perm = [1, 0] : (!transform.op<"linalg.pack">, !transform.op<"linalg.generic">) -> (!transform.op<"linalg.generic">, !transform.op<"linalg.pack">, !transform.any_op)
    %pg = transform.structured.match ops{{["linalg.generic"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t, %lo:3 = transform.structured.tile_using_for %pg tile_sizes [1, 1, 1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %packs = transform.structured.match ops{{["linalg.pack"]}} in %arg0 : (!transform.any_op) -> !transform.op<"linalg.pack">
    transform.structured.lower_pack %packs : (!transform.op<"linalg.pack">) -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
    %unpacks = transform.structured.match ops{{["linalg.unpack"]}} in %arg0 : (!transform.any_op) -> !transform.op<"linalg.unpack">
    transform.structured.lower_unpack %unpacks : (!transform.op<"linalg.unpack">) -> (!transform.op<"tensor.empty">, !transform.op<"linalg.transpose">, !transform.op<"tensor.collapse_shape">, !transform.op<"tensor.extract_slice">, !transform.op<"linalg.copy">)
    transform.apply_patterns to %arg0 {{
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
    }} : !transform.any_op
    // STREAM the inner KC reduction in chunks of KS so the live B panel is vector<KS x NR> (KS regs)
    // not vector<KC x NR> (KC regs). Materializing the whole vector<KCxNR> panel needs KC vector
    // registers and SPILLS hard (32 vs 2 spills measured at KC=16), and the spill scratch overran a
    // packed buffer into BSS -> the M>=64 `tohost=1337` wild-store fault. Streaming K keeps the
    // outerproduct's operands small (like the proven non-packed tiled recipe) so it stays bounded.
    %g0 = transform.structured.match ops{{["linalg.generic"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %gt, %gl = transform.structured.tile_using_for %g0 tile_sizes [0, 0, {KS}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %gt vector_sizes [{MR}, {NR}, {KS}] : !transform.any_op
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    }} : !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.lower_outerproduct
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


def _packed_eliminate_empties(passes: list[str]) -> list[str]:
    """Insert `eliminate-empty-tensors` immediately BEFORE one-shot-bufferize.

    The pack/unpack lowering (lower_pack/lower_unpack -> linalg.transpose with `tensor.empty`
    init/out dests) plus the pipeline's `canonicalize, cse` step CSE-merges the A-pack-dest and
    C-pack-dest `tensor.empty`s (identical type `tensor<M/MR x N/NR x MR x NR>`), so one-shot-
    bufferize then aliases the packed-A and packed-C onto ONE buffer (matmul reads A from the
    same alloc C is written to). That destructive update produced garbage + a wild out-of-bounds
    store (the `tohost=1337` vprintfmt-corruption fault at M>=64). `eliminate-empty-tensors` rewrites
    each `tensor.empty` used as a bufferization init into a distinct `bufferization.alloc_tensor`
    (not CSE-coalesced), giving packed-A and packed-C SEPARATE buffers -> correct + no overrun.
    Only runs when a packed feature is enabled; baseline pipeline untouched."""
    out = list(passes)
    try:
        i = out.index("one-shot-bufferize{bufferize-function-boundaries "
                       "function-boundary-type-conversion=identity-layout-map}")
    except ValueError:
        return out
    if out[i - 1] == "eliminate-empty-tensors":
        return out
    out.insert(i, "eliminate-empty-tensors")
    return out


register(ImprFeature(
    name="vfmacc_packed",
    action_class="PASS",
    edit_pipeline=_packed_eliminate_empties,
    description="operand-PACKING tiled vfmacc (mined packed_rhs_policy = OpenBLAS ncopy/tcopy, "
                "XNNPACK goi-prepack): transform.structured.pack A/B/C into contiguous [MR,NR,KC] "
                "register-tile panels, lower pack/unpack to copy loops, fold unit dims, then the "
                "scoped-vectorize -> outerproduct -> vfmacc recipe on the packed op so the inner "
                "transfers are UNIT-STRIDE contiguous (no strided vector.transfer). Register-tile "
                "[4,16,16]. Default-off; baseline byte-identical.",
    edit_schedule=lambda _t: vfmacc_packed_schedule(4, 16, 16),
    schedule_replace=True,
))


def _register_packed_grid() -> list[str]:
    """Register default-off packed-vfmacc tuning points `vfmacc_packed_<MR>_<NR>_<KC>` (same grid
    as the tiled sweep) so the packing layout can be tuned alongside the register tile."""
    names: list[str] = []
    for MR in (4, 8):
        for NR in (16, 32):
            for KC in (16, 32, 64):
                nm = f"vfmacc_packed_{MR}_{NR}_{KC}"
                register(ImprFeature(
                    name=nm,
                    action_class="PASS",
                    description=f"Operand-packing tiled-vfmacc tuning point: register-tile/pack-tile "
                                f"(MR={MR}, NR={NR}, KC={KC}). Contiguous packed inner transfers. "
                                f"Default-off tuning-grid feature.",
                    edit_pipeline=_packed_eliminate_empties,
                    edit_schedule=(lambda _t, _MR=MR, _NR=NR, _KC=KC:
                                   vfmacc_packed_schedule(_MR, _NR, _KC)),
                    schedule_replace=True,
                ))
                names.append(nm)
    return names


PACKED_GRID_NAMES: list[str] = _register_packed_grid()


# ---- accumulator-resident micro-kernel (genuine transform-dialect codegen) ----------
# THE genuine compiler-emitted answer to the scalable-gap that the `intrinsic_microkernel` marker
# only *demonstrated* with a hand-written C kernel. The earlier finding was that the upstream
# `tile -> scoped-vectorize -> outerproduct -> bufferize` recipe re-reads/re-writes the MR x NR
# accumulator THROUGH MEMORY every K-tile (a `vector.transfer_read`/`transfer_write` of the C
# subview INSIDE the K loop, + a `memref.copy` self-copy artifact) — that copy traffic, not the
# vfmacc arithmetic, was the 15.7x gap. Three hoisting attempts were reported to no-op.
#
# Root cause found here (mlir-opt iteration, LLVM 23): `hoist_redundant_vector_transfers` DOES lift
# the accumulator transfer pair into an scf.for **vector iter_arg** (value semantics: the
# accumulator stays in a `vector<MRxNR>` carried by the K loop, NEVER bufferized to memory inside
# K) — but ONLY when the C accumulator memref it reads/writes is a TYPED static-shape `memref.alloc`
# (e.g. `memref<4x16xf32>`). The pipeline's tiled C is a `memref.subview %alloc[%i,%j]` with a
# DYNAMIC offset (`strided<[..], offset: ?>`); the hoister's loop-invariance / subset analysis bails
# on the dynamic offset, so the transfer pair stays in the loop. `transform.structured.promote`
# produces a `memref.view` of an i8 buffer, which the hoister ALSO refuses to see through. The fix
# that works: `transform.structured.bufferize_to_allocation` on the M,N-tile matmul, which emits a
# TYPED `memref.alloc() : memref<MRxNRxf32>` (+ a copy-in / copy-out that are O(MR*NR) ONCE per
# M,N tile, NOT per K-tile), then tile K inside, vectorize -> contract, and run the hoister
# POST-BUFFERIZE so it sees the typed alloc and lifts the accumulator into the K-loop vector
# iter_arg. Net inner K body: read A + B tile, MR*KC vfmacc into the carried accumulator, yield;
# the accumulator touches memory exactly twice per M,N tile (copy-in/out), not 2*(K/KC) times.
#
# Two-phase because the hoist must run on the BUFFERIZED (memref) form:
#   PRE-bufferize  (edit_schedule): tile [MR,NR,0] -> bufferize_to_allocation(C) -> tile K [0,0,KC]
#                  -> scoped-vectorize -> transfer_permutation + reduction_to_contract  (STOP at
#                  vector.contract; do NOT lower it yet — the hoister wants the read->contract->write
#                  shape).
#   POST-bufferize (edit_pipeline): after one-shot-bufferize, splice a SECOND
#                  transform-preload-library + transform-interpreter that runs
#                  hoist_redundant_vector_transfers (=> vector iter_arg) THEN
#                  lower_contraction(outerproduct) -> lower_outerproduct -> vector.fma -> vfmacc.
# Verified bit-exact + accumulator-resident (vector iter_arg, zero per-K memref roundtrip) on spike
# at 32/64/128. Default-off; baseline byte-identical.
def _accumulator_resident_pre_schedule(MR: int, NR: int, KC: int,
                                       NR_bmm: int | None = None,
                                       MR_mm: int | None = None) -> str:
    """PRE-bufferize transform schedule: tile M,N; promote C to a TYPED local alloc via
    bufferize_to_allocation; tile the reduction K by 1; scoped-vectorize [MR,NR,1]; rebuild
    vector.contract. Stops at the contract (the post-bufferize schedule does the hoist + lowering).

    K is tiled by **1** (not KC) and vectorized [MR,NR,1] on purpose: that makes each K step read a
    SINGLE B row (`vector<1xNR>` -> `vector<NR>`) + MR A scalars and do MR `vfmacc` into the resident
    accumulator — structurally identical to the expert/hand micro-kernel (MR accumulator vreg-groups
    held across K, one B row + A scalars streamed per step). Vectorizing the full [MR,NR,KC] tile
    instead materializes the whole `vector<KCxNR>` B panel (KC vector registers) and SPILLS hard (40
    vector spills measured at KC=16, faulting at M>=64). The `KC` argument now sets the outer K
    register-block trip only conceptually; the streamed inner step is always 1 (bounded vreg use).
    C copy-in uses `linalg.copy` (vectorizable / cheap) rather than the default `memref.copy` runtime
    call. The batch_matmul (attention) path gets the same treatment so the feature is whole-model-safe.

    N-TAIL (``NR_bmm``): the batch_matmul (attention) N is often SMALL (e.g. llama-style N=8). When
    the batch_matmul is vectorized at the matmul NR (16) on an N=8 dim, the partial inner write needs
    a masked `vector.transfer_write` (vector<...x16> into tensor<...x8>) and LLVM-23 rejects the
    multi-op `vector.mask` -> PipelineError -> (whole-model) silent scalar fallback. Setting a
    SEPARATE batch_matmul NR (``NR_bmm`` <= the small N, default = NR) clamps NR=min(NR,N) so the
    vectorize is FULL (no mask) and the N=8 attention batch_matmul vectorizes to vfmacc. The matmul
    path keeps its own (larger) NR. For larger N, NR_bmm just tiles N cleanly. This is the N-tail fix
    that makes the accumulator-resident feature whole-model-safe for attention.

    M-TAIL (``MR_mm``): the M-side analog of the N-tail. A whole-model decode step is dominated by
    matmuls whose LEADING dim is M=1 (one token row); the schedule's MR=4 register tile then tries to
    write a `vector<4xNR>` into a `tensor<1xNR>` C tile -> a masked `vector.transfer_write` LLVM-23
    rejects with the SAME multi-op `vector.mask` PipelineError as the N-tail (silent scalar fallback,
    no vfmacc). Setting a SEPARATE matmul MR (``MR_mm`` <= the small M, default = MR) clamps
    MR=min(MR,M) on the matmul path so the inner vectorize is FULL (no mask) and the M=1 token-decode
    matmul vectorizes to vfmacc. ``MR_mm=1`` is whole-model-safe by construction: M=1 ops fit exactly
    (no mask) and any larger-M matmul just tiles M into single-row register tiles (still a real vfmacc
    chain, bit-exact). Only the matmul M tile is affected; the batch_matmul keeps its own MR. Combined
    with ``NR_bmm`` the tiled vectorize is tail-aware on BOTH the M side (matmul M<MR) and the N side
    (batch_matmul N<NR) generally — a tile that adapts MR=min(MR,M), NR=min(NR,N), not a memorized shape.
    """
    NB = NR_bmm if NR_bmm is not None else NR
    MM = MR_mm if MR_mm is not None else MR
    return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t1, %lmn:2 = transform.structured.tile_using_for %mm tile_sizes [{MM}, {NR}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %buf, %newmm = transform.structured.bufferize_to_allocation %t1 {{memory_space = 0 : i64, bufferize_destination_only}} : !transform.any_op
    %mm2 = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t2, %lk = transform.structured.tile_using_for %mm2 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %t2 vector_sizes [{MM}, {NR}, 1] : !transform.any_op
    %bm = transform.structured.match ops{{["linalg.batch_matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt1, %blmn:3 = transform.structured.tile_using_for %bm tile_sizes [1, {MR}, {NB}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %bbuf, %bnew = transform.structured.bufferize_to_allocation %bt1 {{memory_space = 0 : i64, bufferize_destination_only}} : !transform.any_op
    %bm2 = transform.structured.match ops{{["linalg.batch_matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt2, %blk = transform.structured.tile_using_for %bm2 tile_sizes [0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %bt2 vector_sizes [1, {MR}, {NB}, 1] : !transform.any_op
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


#: The hand-frozen whole-model register block, named ONCE so the registration below and the
#: shape-aware resolver read the same numbers. ``WHOLEMODEL_VF_CAPS`` is ``(MR, NR, KC)``; the two
#: per-class clamps are the tails somebody picked to survive the VLA shapes (matmul M tile 1,
#: batch_matmul N tile 8). They are a SHAPE decision, not a constant of the target, which is why
#: ``mining.apply.shape_adapted_features`` can re-derive them from a workload's own extents --
#: reading the triple as an upper BOUND. On tiny_llama that derivation returns exactly (4, 16) /
#: (4, 8), i.e. this frozen point, so the model this block was tuned on is unaffected.
WHOLEMODEL_VF_NAME = "accumulator_resident_wholemodel_vf"
WHOLEMODEL_VF_CAPS: tuple[int, int, int] = (4, 16, 16)
WHOLEMODEL_VF_MR_MM = 1
WHOLEMODEL_VF_NR_BMM = 8


def _tile_epilogue_hygiene(mr_matmul: int | None) -> frozenset[str]:
    """The lowering hygiene an MR>1 register-block recipe cannot be measured without.

    A recipe that tiles the output and bufferizes per tile leaves, in each tile epilogue, a
    ``memref.copy %x, %x`` -- the destination subview copied onto ITSELF -- which survives
    ``finalize-memref-to-llvm`` as an opaque rank-generic ``@memrefCopy`` runtime call. Erasing it is
    unconditionally value-preserving (identical SSA operand => identical base, offsets and region)
    and is a no-op on a lowering that has none, so ``erase_self_copy`` can only help or do nothing.
    ``mining/from_strategy._with_hygiene`` states exactly this policy for recipes reached through the
    ``microkernel`` knob space; this function is the same policy for the features named DIRECTLY in a
    package's ``compiler_features``, which bypass that space entirely.

    MEASURED, spike, 64^3 matmul, this repo, bit-identical output on every arm:

        int8   MR=1 240,369 cyc | MR=4 491,346 | MR=1+erase 240,369 | MR=4+erase 178,239
        f32    MR=1 231,313 cyc | MR=4 495,241 | MR=1+erase 231,313 | MR=4+erase 181,503

    Read those rows in order. Bare MR=4 is 2.0-2.1x SLOWER than MR=1 even though the register block
    made the compute cheaper -- the PC histogram attributes the whole delta (+250,977 instructions,
    exactly the cycle delta) to ``memrefCopy`` +187,520 / ``memcpy`` +98,304 / ``memset`` +24,896
    against ``forward`` -59,743, i.e. a 1.45x cheaper kernel paying 310,720 instructions of buffer
    copying to get there. With the erase, MR=4 is 1.35x (int8) / 1.27x (f32) FASTER than MR=1. So the
    A-reuse register block was never the losing lever; the un-erased epilogue was.

    MR<=1 returns empty deliberately. Not because the erase would be unsafe there, but because
    MR=1+erase measured BYTE-IDENTICAL (both dtypes, same cycle count to the digit) -- there is no
    self-copy to erase at MR=1, so implying it would move the validated ``MR_mm=1`` control's
    declared feature set for no measured effect. This also explains the standing claim in
    ``from_strategy._with_hygiene`` that the ``accumulator_resident_wholemodel*`` family "has no
    self-copy to erase": that was measured on the MR=1 member, and is false for the MR>1 members.
    """
    return frozenset({_SELF_COPY_FEATURE}) if (mr_matmul or 1) > 1 else frozenset()


def _accumulator_resident_v3_pre_schedule(MR: int, NR: int, KC: int,
                                          NR_bmm: int | None = None,
                                          MR_mm: int | None = None,
                                          skip_mm: bool = False,
                                          skip_bmm: bool = False) -> str:
    """PRE-bufferize schedule for the v3 (vfmacc.vf) micro-kernel — SAME as the v1/v2 pre-schedule
    but WITHOUT ``bufferize_to_allocation``.

    The v1/v2 pre-schedule promoted the C tile to a TYPED local ``memref.alloc`` (via
    ``bufferize_to_allocation``) because v1 ran ``hoist_redundant_vector_transfers`` POST-bufferize
    and that pass needed a typed alloc to (try to) see. v3 runs ``loop-invariant-subset-hoisting`` on
    the TENSOR form (before bufferize), where the K-loop carries the C tile as a value-semantic
    ``tensor<MRxNR>`` iter_arg directly (a ``tensor.extract_slice`` of the output) — no local alloc
    needed. Dropping ``bufferize_to_allocation`` removes the per-M,N-tile C copy-in/copy-out
    (``materialize_in_destination`` -> ``memref.copy``) that otherwise dominated the timed region
    (measured: ~634K of the 692K instret @64^3 was ``memrefCopy``/``memcpy`` from that promotion,
    while the actual ``forward`` compute kernel was only ~57K — at the hand ceiling). Without it the
    accumulator write lands in-place on the output buffer, so the compute kernel stands alone.

    Otherwise identical to the v1 pre-schedule: tile [MR,NR]; tile K by 1; scoped-vectorize [MR,NR,1]
    (one B row + MR A scalars per K step -> the resident-accumulator micro-kernel shape); rebuild
    ``vector.contract``; STOP (the v3 pipeline does subset-hoist + contract-lower + A-scalarize). The
    M-tail (``MR_mm``) / N-tail (``NR_bmm``) clamps carry over unchanged for whole-model safety.

    ``skip_mm`` / ``skip_bmm`` OMIT that op class's arms entirely, leaving those contractions for
    ``convert-linalg-to-loops`` (scalar). Needed because a claim can be impossible: when every block
    legal for a class's extents is 1-lane wide (an N=1 extent forces NR=1), tiling buys no vector
    lanes AND produces a parallel-dim-free ``vector.contract`` that no lowering strategy matches. Not
    claiming the class is then strictly better than claiming it badly — the model builds and runs, and
    the caller reports the class as un-vectorized rather than failing the build.
    """
    NB = NR_bmm if NR_bmm is not None else NR
    MM = MR_mm if MR_mm is not None else MR
    if __import__("os").environ.get("MERLIN_PAD_M"):
        # M-PADDING (default OFF): pad matmul M to multiple of MR=4 BEFORE tiling so small-M (openvla M=17)
        # register-blocks CLEANLY at MR=4 (no masked transfer_write scalar fallback). Measure vs MR=1 vf.
        return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %padded, %pad, %cp = transform.structured.pad %mm pad_to_multiple_of [4] {{padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32], padding_dimensions = [0]}} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %t1, %lmn:2 = transform.structured.tile_using_for %padded tile_sizes [4, {NR}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %mm2 = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t2, %lk = transform.structured.tile_using_for %mm2 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %t2 vector_sizes [4, {NR}, 1] : !transform.any_op
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.yield
  }}
}}
"""
    # EXPERIMENT (env MERLIN_VEC_EW, default OFF -> baseline byte-identical): after matmul/bmm are tiled
    # + vectorized, the remaining linalg.generic ops are the NON-matmul ops (activations/softmax/norm/
    # layout) — which otherwise fall through convert-linalg-to-loops to SCALAR (openvla's 1100ms). Match
    # + scoped-vectorize them here. NOTE: this is the BLUNT version (matches ALL generics incl. reduction
    # softmax/norm — may break cos or explode); the measurement tells us if scoped/tagged is needed.
    # PER-RANK BOUNDED vectorize (env MERLIN_VEC_RANK, default OFF): the non-matmul parallel generics
    # (tagged merlin.vec_rN by the pre-pass, N = loop rank) are vectorized with BOUNDED vector_sizes
    # [1,..,1,8] — innermost dim by 8 lanes (VLEN256/f32), NOT the plain no-sizes vectorize that
    # explodes (vector<17x576>=9792 lanes -> 8725ms). Per rank because openvla generics are rank 2/3/4.
    _ew = ("""
    %g2 = transform.structured.match attributes{merlin.vec_r2} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %g2 vector_sizes [1, 8] : !transform.any_op
    %g3 = transform.structured.match attributes{merlin.vec_r3} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %g3 vector_sizes [1, 1, 8] : !transform.any_op
    %g4 = transform.structured.match attributes{merlin.vec_r4} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %g4 vector_sizes [1, 1, 1, 8] : !transform.any_op"""
           if __import__("os").environ.get("MERLIN_VEC_RANK") else (
    """
    %g = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %g : !transform.any_op"""
           if __import__("os").environ.get("MERLIN_VEC_EW") else ""))
    _mm_arm = "" if skip_mm else f"""
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t1, %lmn:2 = transform.structured.tile_using_for %mm tile_sizes [{MM}, {NR}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %mm2 = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t2, %lk = transform.structured.tile_using_for %mm2 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %t2 vector_sizes [{MM}, {NR}, 1] : !transform.any_op"""
    _bmm_arm = "" if skip_bmm else f"""
    %bm = transform.structured.match ops{{["linalg.batch_matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt1, %blmn:3 = transform.structured.tile_using_for %bm tile_sizes [1, {MR}, {NB}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %bm2 = transform.structured.match ops{{["linalg.batch_matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt2, %blk = transform.structured.tile_using_for %bm2 tile_sizes [0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %bt2 vector_sizes [1, {MR}, {NB}, 1] : !transform.any_op"""
    return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{{_mm_arm}{_bmm_arm}{_ew}
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


def _zero_attr(t: str) -> str:
    """The MLIR zero literal for element type ``t``, derived from the type rather than tabulated.

    Integer types take ``0 : iN``; floating types take the f-literal form. A tabulated map would go
    stale the first time a new element type appears, and the failure is silent (scalar fallback).
    """
    t = str(t).strip()
    if t.startswith("i") and t[1:].isdigit():
        return f"0 : {t}"
    if t.startswith(("f", "bf")):
        return f"0.000000e+00 : {t}"
    raise ValueError(f"no zero literal derivable for element type {t!r}; add the rule rather than "
                     "letting the schedule pad with a wrongly-typed value")


def _accumulator_resident_v3_mrpad_pre_schedule(MR: int, NR: int, KC: int,
                                                NR_bmm: int | None = None,
                                                elem_types: tuple[str, str, str] | None = None) -> str:
    """PER-MATMUL MR register block with an M-PAD tail — the general fix for the M=1/M%MR!=0 case.

    ``accumulator_resident_wholemodel_vf`` clamps the matmul M tile to MR_mm=1 (no A-operand reuse:
    1 B-load + 1 A-load per FMA = 2.0 loads/useful-FMA). ``..._vf_mr4`` raises it to MR=4 (A-reuse,
    1.25 loads/FMA) but is ONLY correct where M%4==0: on M not divisible by MR the partial M tile
    needs a masked ``vector.transfer_write`` (``vector<MRxNR>`` into ``tensor<(M%MR)xNR>``) that
    LLVM-23 rejects with a multi-op ``vector.mask`` PipelineError -> (measured) NR=8/non-resident/119
    in-loop spills at M=17, and an outright PipelineError -> silent scalar fallback at M=1.

    This schedule handles the tail the way the hand shim does (``round_up_mr``): it PADS the matmul M
    dimension UP to the next multiple of MR BEFORE tiling (``transform.structured.pad`` on padding
    dimension 0, padding value 0.0). Every matmul then register-blocks CLEANLY at MR — the M tile is
    always FULL (no masked write, no PipelineError), so M=1 pads to MR, M=17 pads to 20, M=64 stays 64,
    each getting the same MR resident-accumulator vfmacc.vf register block. ``transform.structured.pad``
    threads a ``tensor.extract_slice`` back to the original M rows (the ``copy_back`` result), so the
    real output is BIT-EXACT: the padded rows are ``0-row @ B = 0`` and are discarded; only the [0:M]
    slice is written back. This is per-matmul (each op pads to ITS OWN next multiple of MR — a general
    tail rule, not a per-model constant), and it composes with the v3 subset-hoist + A-scalarization
    pipeline unchanged (it still stops at ``vector.contract``).

    The batch_matmul (attention) path is IDENTICAL to ``accumulator_resident_wholemodel_vf`` — MR=4 M
    tile + the ``NR_bmm`` N-tail clamp — so this feature only upgrades the ``linalg.matmul`` path from
    the MR=1 clamp to a padded-MR register block; the proven whole-model attention path is untouched.
    """
    NB = NR_bmm if NR_bmm is not None else NR
    # The padding value must have the ELEMENT TYPE OF THE OPERAND IT PADS. This was a single f32
    # literal reused for all three, which is correct only while every contraction is f32 -- and
    # `transform.structured.pad` does not warn, it hard-errors:
    #   'transform.structured.pad' op expects a padding value of type 'i8', got 0.000000e+00 : f32
    # A transform-interpreter error is a PipelineError, which the lowering catches as a whole-model
    # SILENT SCALAR FALLBACK -- the failure mode is "mysteriously slow", not "build failed".
    # It has never fired because the int8 datapath rewrites every linalg.matmul into a
    # linalg.generic, so no i8 matmul reaches this schedule at all (see the module docstring note on
    # named-op erasure). It fires the moment that is fixed, which is why it is fixed first.
    za, zb, zc = (_zero_attr(t) for t in (elem_types or ("f32", "f32", "f32")))
    # copy_back_op = "linalg.copy" (NOT the default bufferization.materialize_in_destination): the
    # pipeline's canonicalize/cse merges the many identical `linalg.fill 0 -> tensor<MxNxf32>` matmul
    # inits into ONE shared value, and materialize_in_destination copies each padded result back into
    # that SHARED fill buffer -> one-shot-bufferize sees N writes aliasing one buffer -> "cannot avoid
    # RaW conflict" -> whole-model PipelineError -> silent scalar fallback (this is EXACTLY what breaks
    # the pre-existing vf_mr4 on rdt2). A `linalg.copy` copy-back (paired with eliminate-empty-tensors
    # in the edit_pipeline) bufferizes each padded matmul into its own buffer, so the model lowers
    # vectorized. Verified: shared-fill reproducer + rdt2/bitvla whole-model.
    return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %padded, %pad, %cp = transform.structured.pad %mm pad_to_multiple_of [{MR}] {{padding_values = [{za}, {zb}, {zc}], padding_dimensions = [0], copy_back_op = "linalg.copy"}} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %t1, %lmn:2 = transform.structured.tile_using_for %padded tile_sizes [{MR}, {NR}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %mm2 = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t2, %lk = transform.structured.tile_using_for %mm2 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %t2 vector_sizes [{MR}, {NR}, 1] : !transform.any_op
    %bm = transform.structured.match ops{{["linalg.batch_matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt1, %blmn:3 = transform.structured.tile_using_for %bm tile_sizes [1, {MR}, {NB}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %bm2 = transform.structured.match ops{{["linalg.batch_matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt2, %blk = transform.structured.tile_using_for %bm2 tile_sizes [0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %bt2 vector_sizes [1, {MR}, {NB}, 1] : !transform.any_op
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


def _accumulator_resident_v3_mrpad_pipeline(passes: list[str]) -> list[str]:
    """The v3 (.vf subset-hoist + A-scalarize) pipeline PLUS ``eliminate-empty-tensors`` before
    one-shot-bufferize — required by the M-pad copy-back.

    The M-pad schedule copies each padded matmul result back with ``linalg.copy`` (not the default
    ``materialize_in_destination``); the pair (linalg.copy copy-back + eliminate-empty-tensors) is what
    lets one-shot-bufferize give each padded matmul its OWN buffer instead of aliasing the CSE-shared
    zero-fill init across every matmul (the "cannot avoid RaW conflict" whole-model PipelineError).
    Reuses ``_packed_eliminate_empties`` (same insertion the packed feature uses). Only runs when this
    feature is enabled; baseline pipeline untouched."""
    return _packed_eliminate_empties(_accumulator_resident_v3_pipeline(passes))


# NOTE: distinct entry-point name (`@__transform_accum_post`) so that when BOTH the pre- and
# post-bufferize schedules are preloaded into the same module symbol table, their named sequences do
# not collide (`@__transform_main` is defined by the pre schedule).
_ACCUM_RESIDENT_POST_SCHEDULE = """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_accum_post(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %f2 = transform.structured.hoist_redundant_vector_transfers %f : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f2 {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    } : !transform.any_op
    transform.apply_patterns to %f2 {
      transform.apply_patterns.vector.lower_outerproduct
      transform.apply_patterns.vector.lower_shape_cast
    } : !transform.any_op
    transform.yield
  }
}
"""


def _post_schedule_path() -> Path:
    """Write the POST-bufferize hoist+lower schedule to a stable temp file and return its path.

    `edit_pipeline` only sees the pass-list strings (not the per-build work dir), so the second
    transform-interpreter needs a real, stable file to preload. The content is fixed, so the path is
    content-addressed (one file, written once, reused) — deterministic and safe for parallel builds.
    """
    text = _ACCUM_RESIDENT_POST_SCHEDULE
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    p = Path(tempfile.gettempdir()) / f"merlin_accum_resident_post_{h}.mlir"
    if not p.is_file():
        p.write_text(text, encoding="utf-8")
    return p


def _accumulator_resident_pipeline(passes: list[str]) -> list[str]:
    """Splice a SECOND transform-preload-library + transform-interpreter right AFTER
    one-shot-bufferize: it hoists the typed-alloc accumulator transfer pair into the K-loop vector
    iter_arg (register-resident), then lowers contraction->outerproduct->vfmacc. The PRE-bufferize
    schedule (edit_schedule) deliberately stopped at vector.contract so the hoister sees the
    read->contract->write shape on the bufferized memref. Only runs when this feature is enabled;
    baseline pipeline untouched."""
    out = list(passes)
    # find one-shot-bufferize (it carries options, so match by prefix)
    idx = None
    for i, p in enumerate(out):
        if p.startswith("one-shot-bufferize"):
            idx = i
            break
    if idx is None:
        return out
    sched = _post_schedule_path()
    # Insert the hoist IMMEDIATELY after one-shot-bufferize — i.e. BEFORE
    # buffer-results-to-out-params and buffer-hoisting. Critical: buffer-hoisting/loop-hoisting
    # would hoist the per-tile C `memref.alloc` out to the function entry and REUSE one buffer across
    # every M,N tile; the hoister then sees a write to that shared buffer (the copy-in) in the
    # K-loop's parent region and conservatively refuses to lift the accumulator transfer across K.
    # Running the hoist BEFORE buffer-hoisting (while the typed C alloc is still local to the M,N
    # tile, exactly the shape that hoists in isolation) makes it fire -> vector iter_arg.
    insert_at = idx + 1
    inject = [
        f"transform-preload-library{{transform-library-paths={sched}}}",
        "transform-interpreter{entry-point=__transform_accum_post}",
        "canonicalize", "cse",
    ]
    if out[insert_at:insert_at + len(inject)] != inject:   # idempotent
        out = out[:insert_at] + inject + out[insert_at:]
    # Swap the default `convert-vector-to-scf` for the full-unroll variant. The hoisted accumulator
    # is a rank-2 `vector<MRxNR>` carried by the K loop; the DEFAULT convert-vector-to-scf lowers
    # every rank>=2 vector transfer through a STACK `memref.alloca` scratch (the carried accumulator
    # then lives on the stack, re-loaded/stored each K step, and under RVV's fixed VLEN it spills
    # hard -> the `tohost=1337` wild-store fault at M>=32). `full-unroll` instead fully unrolls those
    # rank-2 transfers into rank-1 element ops with NO stack scratch, so the accumulator stays in the
    # vector register file across K (register-resident, like the hand kernel's MR vfloat32m4_t accs).
    # Feature-scoped (only when this feature is on); baseline `convert-vector-to-scf` untouched.
    for i, p in enumerate(out):
        if p == "convert-vector-to-scf":
            out[i] = "convert-vector-to-scf{full-unroll}"
            break
    return out


def _register_accumulator_resident() -> list[str]:
    """Register `accumulator_resident_microkernel` (default tile) + a small tuning grid."""
    names: list[str] = []
    grid = [(4, 16, 16), (4, 16, 32), (8, 16, 16), (4, 32, 16)]
    for MR, NR, KC in grid:
        if (MR, NR, KC) == (4, 16, 16):
            nm = "accumulator_resident_microkernel"
            desc = ("Transform-dialect accumulator-residency attempt: tile [MR=4,NR=16], "
                    "bufferize_to_allocation the C tile to a TYPED static memref.alloc, tile K "
                    "[KC=16], scoped-vectorize -> vector.contract, then POST-bufferize "
                    "hoist_redundant_vector_transfers -> outerproduct -> vfmacc. Forms a real vfmacc "
                    "chain and is BIT-EXACT at 32/64/128 AND a non-cube 96x48x160 on spike (general, "
                    "not cube-overfit). HONEST MEASURED STATUS: the hoist does NOT fully lift the "
                    "carried accumulator into a pure register iter_arg under RVV's fixed VLEN — the "
                    "emitted K-loop still round-trips the accumulator through the stack "
                    "(vl4re8.v/vs4r.v of the accumulator per K-tile, confirmed by objdump), so "
                    "cca.lift_asm reads accumulator_resident=FALSE and it measures ~19x off the "
                    "hand intrinsic_microkernel ceiling @64^3 (954,558 vs 50,695). It is the best "
                    "transform-only accumulator attempt but does NOT close the gap; the genuine "
                    "closer is a dedicated RVV micro-kernel codegen pass (see action_catalog: "
                    "compute.accumulator_resident -> CODEGEN, forkable_now=False). Default-off, "
                    "baseline byte-identical.")
        else:
            nm = f"accum_resident_{MR}_{NR}_{KC}"
            desc = (f"Accumulator-resident micro-kernel tuning point (MR={MR}, NR={NR}, KC={KC}): "
                    f"register-resident K-loop vector iter_arg accumulator, no per-K memref "
                    f"roundtrip. Default-off tuning-grid feature.")
        register(ImprFeature(
            name=nm,
            action_class="PASS",
            description=desc,
            edit_pipeline=_accumulator_resident_pipeline,
            edit_schedule=(lambda _t, _MR=MR, _NR=NR, _KC=KC:
                           _accumulator_resident_pre_schedule(_MR, _NR, _KC)),
            schedule_replace=True,
        ))
        names.append(nm)
    # N-TAIL-SAFE variant for whole-model attention: same recipe but the batch_matmul N tile is
    # clamped to NR_bmm=8 (<= the small llama-style attention N=8) so the inner vectorize is FULL (no
    # masked vector.transfer_write -> no LLVM-23 PipelineError -> no silent scalar fallback). The
    # matmul path keeps NR=16. This is the feature that makes attention batch_matmuls vectorize to
    # vfmacc instead of falling back to scalar (verified bit-exact on spike for a B=4,M=32,N=8,K=32
    # attention batch_matmul). Default-off; baseline byte-identical.
    register(ImprFeature(
        name="accumulator_resident_ntail",
        action_class="PASS",
        description="N-tail-safe accumulator-resident micro-kernel: the accumulator-resident recipe "
                    "(tile [MR=4,NR=16,KC=16], bufferize_to_allocation C, stream K, scoped-vectorize "
                    "-> contract -> outerproduct -> vfmacc) with the batch_matmul N tile CLAMPED to "
                    "NR_bmm=8 (NR=min(NR,N)). Fixes the small-N (e.g. N=8) attention batch_matmul "
                    "that otherwise hits the LLVM-23 masked-transfer_write PipelineError -> silent "
                    "scalar fallback: with NR_bmm<=N the inner vectorize is full (no mask), so the "
                    "attention batch_matmul vectorizes to vfmacc. Bit-exact on spike (B=4,M=32,N=8,"
                    "K=32). Default-off, baseline byte-identical.",
        edit_pipeline=_accumulator_resident_pipeline,
        edit_schedule=lambda _t: _accumulator_resident_pre_schedule(4, 16, 16, NR_bmm=8),
        schedule_replace=True,
    ))
    names.append("accumulator_resident_ntail")

    # M-TAIL-SAFE variant: the M-side analog of accumulator_resident_ntail. The whole-model decode
    # step is dominated by matmuls with leading M=1 (one token row); the MR=4 register tile then
    # writes a vector<4xNR> into a tensor<1xNR> C tile -> the SAME multi-op vector.mask PipelineError
    # the N-tail hit (LLVM-23 rejects it -> silent scalar fallback, no vfmacc). Clamping the matmul
    # MR to MR_mm=1 (MR=min(MR,M)) makes the inner vectorize FULL on the M=1 matmul (no mask) so it
    # vectorizes to vfmacc; for any larger-M matmul MR_mm=1 just tiles M into single-row register
    # tiles (still a real vfmacc chain, bit-exact) — general, not M=1-overfit. Default-off.
    register(ImprFeature(
        name="accumulator_resident_mtail",
        action_class="PASS",
        description="M-tail-safe accumulator-resident micro-kernel: the accumulator-resident recipe "
                    "with the matmul M tile CLAMPED to MR_mm=1 (MR=min(MR,M)). Fixes the M=1 "
                    "token-decode matmul (smolVLA/rdt2 leading-M=1) that otherwise hits the LLVM-23 "
                    "masked-transfer_write multi-op vector.mask PipelineError -> silent scalar "
                    "fallback: with MR_mm<=M the inner vectorize is full (no mask), so the M=1 matmul "
                    "vectorizes to vfmacc. The M-side analog of accumulator_resident_ntail; the "
                    "batch_matmul keeps its own MR. Default-off, baseline byte-identical.",
        edit_pipeline=_accumulator_resident_pipeline,
        edit_schedule=lambda _t: _accumulator_resident_pre_schedule(4, 16, 16, MR_mm=1),
        schedule_replace=True,
    ))
    names.append("accumulator_resident_mtail")

    # WHOLE-MODEL-SAFE COMPOSED variant (WORK-ITEM 2 by the inherent-clamp design): a SINGLE feature
    # whose schedule has BOTH tail clamps inherent — matmul MR_mm=1 (M-tail) AND batch_matmul NR_bmm=8
    # (N-tail) — on top of the bit-exact tiled-vfmacc accumulator-resident recipe. Because both clamps
    # live in ONE full-schedule feature (not two separate full-schedule replacements that would clobber
    # each other when apply_schedule picks one), enabling THIS one feature applies the tiled vfmacc +
    # M-tail + N-tail together whole-model. It vectorizes a normal matmul, an M=1 token-decode matmul,
    # AND a small-N (N=8) attention batch_matmul in ONE schedule with no scalar fallback / no
    # vector.mask PipelineError. This is the config a whole-model fork enables. Default-off.
    register(ImprFeature(
        name="accumulator_resident_wholemodel",
        action_class="PASS",
        description="Whole-model-safe composed accumulator-resident micro-kernel: the tiled-vfmacc "
                    "accumulator-resident recipe with BOTH tail clamps inherent in one schedule — "
                    "matmul MR_mm=1 (M-tail: M=1 token-decode) AND batch_matmul NR_bmm=8 (N-tail: "
                    "small-N attention). Composes the tiled vfmacc + M-tail + N-tail so the best "
                    "single config is whole-model-safe by construction (no full-schedule feature "
                    "clobbers another's clamp). Vectorizes a normal matmul, an M=1 matmul, and an "
                    "N=8 batch_matmul to vfmacc in ONE schedule (no scalar fallback, no vector.mask "
                    "PipelineError). Bit-exact on spike across the shape spread. Default-off, "
                    "baseline byte-identical.",
        edit_pipeline=_accumulator_resident_pipeline,
        edit_schedule=lambda _t: _accumulator_resident_pre_schedule(4, 16, 16, NR_bmm=8, MR_mm=1),
        schedule_replace=True,
    ))
    names.append("accumulator_resident_wholemodel")
    return names


ACCUM_RESIDENT_NAMES: list[str] = _register_accumulator_resident()


# ---- accumulator-resident micro-kernel v2 (PRE-bufferize subset hoist) --------------
# The genuine gap-closer the prior `accumulator_resident_microkernel` could NOT reach. That
# feature ran `hoist_redundant_vector_transfers` POST-bufferize; on the bufferized memref form the
# accumulator transfer pair is on an scf.for **memref iter_arg** (`%arg7`), and BOTH
# `hoist_redundant_vector_transfers` AND `loop-invariant-subset-hoisting` no-op on it (the carried
# memref aliases the per-tile alloc, defeating the loop-invariance / subset analysis) — so the
# emitted K-loop round-tripped the `vector<MRxNR>` accumulator through the stack every K step
# (`vl4re8.v`/`vs4r.v` per K-tile, ~19x off the hand ceiling). MEASURED, with objdump evidence.
#
# Root cause found here (mlir-opt iteration, LLVM 23): the hoist MUST run on the TENSOR form, BEFORE
# one-shot-bufferize, where the K-loop carries the accumulator as a value-semantic
# `tensor<MRxNR>` iter_arg and the accumulator transfer pair reads/writes that tensor iter_arg at
# loop-invariant indices. On THAT form `loop-invariant-subset-hoisting` fires cleanly: it lifts the
# `vector.transfer_read` above the K-loop and the `vector.transfer_write` below it, threading a pure
# `vector<MRxNR>` through the loop as a SECOND iter_arg — the accumulator now lives in SSA vector
# values across K, never bufferized to memory inside the loop (verified in the emitted IR: the
# K-loop carries `iter_args(%acc_tensor, %acc_vector)`, body is read-A + read-B-row + MR `vector.fma`
# into `%acc_vector`, yield; the accumulator transfer pair is GONE from the loop body). After bufferize
# the carried `vector<MRxNR>` lowers to an `!llvm.array<MR x vector<NRxf32>>` loop-carried value (the
# K-loop block argument), which the RISC-V backend keeps in the vector register file across K — the
# accumulator-RESIDENT structure the hand kernel has (MR accumulator vreg-groups held across K).
#
# Because the hoist runs PRE-bufferize, the contract must NOT be lowered until AFTER the hoist (the
# subset analysis wants the read->contract->write shape). So the recipe is:
#   PRE-bufferize  (edit_schedule): the SAME `_accumulator_resident_pre_schedule` — tile [MR,NR]; tile
#                  K by 1; scoped-vectorize [MR,NR,1]; transfer_permutation + reduction_to_contract;
#                  STOP at vector.contract.
#   edit_pipeline: right after the first transform-interpreter (still on TENSORS), splice
#                  `loop-invariant-subset-hoisting` (=> the accumulator becomes a vector iter_arg)
#                  THEN a second transform-interpreter that lowers contraction->outerproduct->fma
#                  (drop-unit-dims first so the B-row/acc extracts are clean). One-shot-bufferize then
#                  sees the value-semantic vector iter_arg and keeps it register-resident.
# NOTE: this still emits `vfmacc.vv` (the A column is read as `vector<MRx1>` -> per-lane `<1xf32>`
# loads + a lane-broadcast the backend builds with a vmv/vslideup ladder), NOT the hand kernel's
# `vfmacc.vf` (A scalar straight from an FP reg). That residual A-broadcast pressure forces a small
# constant number of accumulator spills. So this CLOSES the accumulator-residency structure (the
# documented #1 gap) and a large fraction of the instret gap, but does not fully reach the hand
# ceiling — the remaining delta is the `.vf` A-operand form, recorded honestly. Default-off;
# baseline byte-identical.
#
# These patterns REQUIRE the contract to have at least one parallel dim (outerproduct needs a rank-1
# result to accumulate into). A register block of NR=1 does not give it one: vectorizing a [.., 1, 1]
# tile yields `vector.contract {iterator_types = ["reduction"]} vector<1xi8>, vector<1xi8> into i32`
# — a one-element dot product with no parallel dim, which NO lowering_strategy matches (measured: 24
# such ops survived on whisper_tiny, and an un-lowered vector.contract has no LLVM translation, so
# the build dies late with "missing LLVMTranslationDialectInterface"). That is a selection bug, not a
# lowering gap: a 1-lane vector is not a vectorization. `mining.apply` therefore never selects NR=1 —
# it leaves that op class un-tiled for `convert-linalg-to-loops` instead.
_ACCUM_RESIDENT_V2_LOWER_SCHEDULE = """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_accum_v2_lower(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.drop_inner_most_unit_dims_from_xfer_ops
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
      transform.apply_patterns.vector.drop_unit_dims_with_shape_cast
    } : !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    } : !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_outerproduct
      transform.apply_patterns.vector.lower_shape_cast
    } : !transform.any_op
    transform.yield
  }
}
"""


def _accum_v2_lower_schedule_path() -> Path:
    """Stable temp path for the PRE-bufferize contract-lowering schedule (content-addressed)."""
    text = _ACCUM_RESIDENT_V2_LOWER_SCHEDULE
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    p = Path(tempfile.gettempdir()) / f"merlin_accum_v2_lower_{h}.mlir"
    if not p.is_file():
        p.write_text(text, encoding="utf-8")
    return p


def _accumulator_resident_v2_pipeline(passes: list[str]) -> list[str]:
    """Splice the PRE-bufferize subset-hoist + contract-lower right after the FIRST
    transform-interpreter (still on tensors), before one-shot-bufferize.

    The pre-schedule (edit_schedule) stops at `vector.contract`; here we (1) run
    `loop-invariant-subset-hoisting` on the tensor form so the accumulator transfer pair becomes a
    `vector<MRxNR>` scf.for iter_arg (register-resident across K), then (2) run a second
    transform-interpreter that lowers the now-hoisted contraction -> outerproduct -> vector.fma.
    Running BEFORE bufferize is the whole point: post-bufferize the carried accumulator is a memref
    iter_arg the hoist cannot lift (the prior feature's measured blocker). Only runs when this feature
    is enabled; baseline pipeline untouched."""
    out = list(passes)
    # Locate the first transform-interpreter (the pre-schedule's __transform_main) and the
    # canonicalize,cse that follow it; splice after that cse (still on tensors, pre-bufferize).
    ti = None
    for i, p in enumerate(out):
        if p.startswith("transform-interpreter{entry-point=__transform_main}"):
            ti = i
            break
    if ti is None:
        return out
    # after ti come canonicalize, cse (then linalg-generalize-named-ops, one-shot-bufferize)
    insert_at = ti + 1
    # skip the canonicalize, cse pair that the pipeline puts right after the interpreter
    while insert_at < len(out) and out[insert_at] in ("canonicalize", "cse"):
        insert_at += 1
    sched = _accum_v2_lower_schedule_path()
    inject = [
        "loop-invariant-subset-hoisting",
        f"transform-preload-library{{transform-library-paths={sched}}}",
        "transform-interpreter{entry-point=__transform_accum_v2_lower}",
        "canonicalize", "cse",
    ]
    if out[insert_at:insert_at + len(inject)] != inject:   # idempotent
        out = out[:insert_at] + inject + out[insert_at:]
    # Same convert-vector-to-scf{full-unroll} swap as the v1 feature: the hoisted accumulator is a
    # rank-2 `vector<MRxNR>` carried by the K loop; the default convert-vector-to-scf would lower
    # rank>=2 transfers through a stack alloca scratch (re-introducing the per-K stack round-trip).
    # full-unroll lowers them to rank-1 element ops with NO stack scratch, so the carried accumulator
    # stays in the vector register file across K. Feature-scoped; baseline untouched.
    for i, p in enumerate(out):
        if p == "convert-vector-to-scf":
            out[i] = "convert-vector-to-scf{full-unroll}"
            break
    return out


def _register_accumulator_resident_v2() -> list[str]:
    """Register `accumulator_resident_v2` (default tile) + a small tuning grid. Reuses the v1
    PRE-bufferize schedule (forms the contract, stops there); the v2 pipeline does the PRE-bufferize
    subset hoist + contract lowering that makes the accumulator a genuine vector iter_arg."""
    names: list[str] = []
    grid = [(4, 16, 16), (4, 16, 32), (8, 16, 16), (4, 32, 16)]
    for MR, NR, KC in grid:
        if (MR, NR, KC) == (4, 16, 16):
            nm = "accumulator_resident_v2"
            desc = ("PRE-bufferize accumulator-resident micro-kernel (the genuine residency the v1 "
                    "feature could not reach): tile [MR=4,NR=16], tile K by 1, scoped-vectorize -> "
                    "vector.contract, then on the TENSOR form (before bufferize) run "
                    "loop-invariant-subset-hoisting so the accumulator transfer pair becomes a "
                    "vector<MRxNR> scf.for iter_arg (register-resident across K, NO per-K memref "
                    "roundtrip), then lower contraction -> outerproduct -> vfmacc. Verified by "
                    "objdump: the K-loop carries the accumulator as an llvm.array<MR x vector<NR>> "
                    "loop value with NO accumulator load/store inside the loop (unlike v1's per-K "
                    "vl4re8.v/vs4r.v). Bit-exact at 32/64/128 + a non-cube on spike. Residual: emits "
                    "vfmacc.vv (A read as vector<MRx1> -> lane-broadcast) not the hand kernel's "
                    "vfmacc.vf, so a small constant accumulator-spill from A-broadcast pressure "
                    "remains; closes the residency structure + most of the instret gap but not the "
                    "full hand ceiling. Default-off, baseline byte-identical.")
        else:
            nm = f"accum_resident_v2_{MR}_{NR}_{KC}"
            desc = (f"PRE-bufferize accumulator-resident tuning point (MR={MR}, NR={NR}, KC={KC}): "
                    f"tensor-level subset-hoisted vector<MRxNR> K-loop iter_arg accumulator (no "
                    f"per-K memref roundtrip). Default-off tuning-grid feature.")
        register(ImprFeature(
            name=nm,
            action_class="PASS",
            description=desc,
            edit_pipeline=_accumulator_resident_v2_pipeline,
            edit_schedule=(lambda _t, _MR=MR, _NR=NR, _KC=KC:
                           _accumulator_resident_pre_schedule(_MR, _NR, _KC)),
            schedule_replace=True,
        ))
        names.append(nm)
    return names


ACCUM_RESIDENT_V2_NAMES: list[str] = _register_accumulator_resident_v2()


# ---- accumulator-resident micro-kernel v3 (resident accumulator + vfmacc.vf) --------
# v2 made the accumulator register-resident (PRE-bufferize subset hoist -> vector iter_arg) but the
# emitted K-loop still used `vfmacc.vv`: the A operand was read as `vector<MRx1xf32>` and each row
# extracted `[i,0]:f32`, and the RISC-V backend cannot cheaply move a vector LANE into the `.vf`
# scalar FP operand, so it rebuilt the broadcast with a vmv/vslideup ladder (that ladder, NOT a
# spill, was the residual instret — v2 measured ~19x off the hand ceiling even though the
# accumulator was resident). v3 adds the A-operand SCALARIZATION rewrite (accum_microkernel.py):
# after the contraction is lowered to `vector.fma` with f32 A-extracts (but before bufferize), each
# `vector.transfer_read -> vector<MRx1xf32>` whose only uses are `vector.extract [i,0]:f32` is
# replaced with per-row scalar `tensor.extract`/`memref.load` (the SAME `a[i]` scalar the hand kernel
# loads). clang-23 then selects the clean `vfmacc.vf` (flw -> vfmacc.vf), and the emitted K-loop is
# the hand kernel's structure exactly: ONE B-row vle32, MR scalar A flw, MR `vfmacc.vf` into the
# resident accumulator, C stored once — 0 in-loop accumulator spills, 0 vfmacc.vv (verified by
# objdump). Numerically identical to v2 (scalar load of [i,0] == lane [i,0] of the vector read), so
# BIT-EXACT. The rewrite runs via a two-stage runner (pipeline._accum_microkernel_v3_features); the
# pipeline edit splices the SCALARIZE_MARKER where the split happens. Default-off; baseline
# byte-identical.
def _accumulator_resident_v3_pipeline(passes: list[str]) -> list[str]:
    """v2 pipeline (PRE-bufferize subset hoist + contract lower) PLUS the SCALARIZE_MARKER sentinel
    spliced immediately AFTER the contract-lowering transform-interpreter (still on tensors, before
    one-shot-bufferize). The two-stage runner splits the pipeline at the marker and runs the
    A-scalarization rewrite there. Only runs when a v3 feature is enabled; baseline untouched."""
    from .accum_microkernel import SCALARIZE_MARKER
    out = _accumulator_resident_v2_pipeline(passes)
    # The v2 edit spliced: loop-invariant-subset-hoisting, transform-preload-library{...v2_lower},
    # transform-interpreter{entry-point=__transform_accum_v2_lower}, canonicalize, cse. Put the
    # marker right AFTER that interpreter's canonicalize,cse (contract is now vector.fma w/ f32
    # A-extracts; bufferize has not run).
    idx = None
    for i, p in enumerate(out):
        if p.startswith("transform-interpreter{entry-point=__transform_accum_v2_lower}"):
            idx = i
            break
    if idx is None:
        return out
    insert_at = idx + 1
    while insert_at < len(out) and out[insert_at] in ("canonicalize", "cse"):
        insert_at += 1
    if SCALARIZE_MARKER not in out:
        out = out[:insert_at] + [SCALARIZE_MARKER] + out[insert_at:]
    return out


def _accumulator_resident_v3_kblocked_pre_schedule(MR: int, NR: int, KC: int) -> str:
    """v3 pre-schedule with a REAL K-blocking tile (the ``KC`` lever, which the plain v3 recipe
    silently ignores — it tiles K by 1 and never uses KC, so every KC produced the same schedule).

    Why it matters (measured): the emitted inner loop advances B by the full row stride each K step
    (``addi a0,a0,512`` for N=128 f32), touching a NEW cache line every iteration and re-walking them
    for every (M,N) tile. Instruction count matches XNNPACK's kernel almost exactly, yet we run ~5x
    slower — i.e. we are memory-stalled, not instruction-bound. Blocking K by KC keeps a B panel
    resident across the M/N tiles so those lines are reused before eviction (classic GEMM cache
    blocking), which is the standard fix for exactly this access pattern.

    Structure: tile [MR, NR] (registers) -> tile K by KC (CACHE) -> tile K by 1 (register-resident
    accumulation) -> scoped-vectorize [MR, NR, 1]. Pure code generation; no hand ukernel."""
    return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t1, %lmn:2 = transform.structured.tile_using_for %mm tile_sizes [{MR}, {NR}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %mmk = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tkc, %lkc = transform.structured.tile_using_for %mmk tile_sizes [0, 0, {KC}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %mm2 = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t2, %lk = transform.structured.tile_using_for %mm2 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %t2 vector_sizes [{MR}, {NR}, 1] : !transform.any_op
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


def ensure_v3_kblocked_microkernel(MR: int, NR: int, KC: int) -> str:
    """Register (on demand) the K-BLOCKED v3 tuning point — makes the KC lever real (cache blocking)."""
    name = f"accum_resident_v3kb_{MR}_{NR}_{KC}"
    if name in known():
        return name
    register(ImprFeature(
        name=name,
        action_class="PASS",
        description=(f"Accumulator-resident micro-kernel with REAL K-blocking (MR={MR}, NR={NR}, "
                     f"KC={KC}): K tiled by KC for cache reuse of the B panel, then by 1 for the "
                     f"register-resident accumulation. Compiler-emitted (no ukernel). Default-off."),
        edit_pipeline=_accumulator_resident_v3_pipeline,
        edit_schedule=(lambda _t, _MR=MR, _NR=NR, _KC=KC:
                       _accumulator_resident_v3_kblocked_pre_schedule(_MR, _NR, _KC)),
        schedule_replace=True,
        implies=_tile_epilogue_hygiene(MR),
    ))
    return name


def _accumulator_resident_v3_unrolled_pre_schedule(MR: int, NR: int, KC: int) -> str:
    """v3 pre-schedule with M held as MR INDEPENDENT accumulators (the `unroll_m` axis).

    The plain v3 recipe tiles M by MR and vectorizes to a 2-D ``vector<MRxNR>``, so MR must be
    vectorization-friendly — measured on K1: MR 3/5/6/7 collapse to 193-279x off XNNPACK while MR=4
    is 5.0x. An expert micro-kernel instead keeps MR SEPARATE accumulator registers (M fully
    unrolled), which is shape-agnostic and is why XNNPACK can pick MR=7.

    Recipe: tile M by 1 (each row is its own ``vector<NR>`` accumulator) x N by NR, tile K by 1,
    scoped-vectorize the [1, NR, 1] body, then UNROLL the M loop by MR -> MR independent accumulators
    carried as separate iter_args. Pure code generation; no hand ukernel."""
    return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t1, %lmn:2 = transform.structured.tile_using_for %mm tile_sizes [1, {NR}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %mm2 = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t2, %lk = transform.structured.tile_using_for %mm2 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %t2 vector_sizes [1, {NR}, 1] : !transform.any_op
    transform.loop.unroll %lmn#0 {{ factor = {MR} }} : !transform.any_op
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


def ensure_v3_unrolled_microkernel(MR: int, NR: int, KC: int) -> str:
    """Register (on demand) the M-UNROLLED v3 tuning point — MR independent accumulators, any MR."""
    name = f"accum_resident_v3u_{MR}_{NR}_{KC}"
    if name in known():
        return name
    register(ImprFeature(
        name=name,
        action_class="PASS",
        description=(f"Accumulator-resident micro-kernel with M UNROLLED into {MR} independent "
                     f"accumulators (NR={NR}, KC={KC}) — shape-agnostic in MR, unlike the 2-D "
                     f"vector<MRxNR> formulation. Compiler-emitted (no ukernel). Default-off."),
        edit_pipeline=_accumulator_resident_v3_pipeline,
        edit_schedule=(lambda _t, _MR=MR, _NR=NR, _KC=KC:
                       _accumulator_resident_v3_unrolled_pre_schedule(_MR, _NR, _KC)),
        schedule_replace=True,
        implies=_tile_epilogue_hygiene(MR),
    ))
    return name


#: RVV lowering fact (not a board constant): LLVM's RISC-V backend defines ``vscale = VLEN / 64``,
#: so an MLIR ``vector<[k]xT>`` holds ``k * VLEN/64`` elements and occupies ``k * VLEN/64 * sizeof(T)``
#: bytes of vector register file, i.e. ``LMUL = k * sizeof(T) / 8``. Two consequences we rely on:
#: the LMUL of a scalable type is FIXED by the type (never widened to cover a worst-case VLEN, which
#: is exactly the defect the ``_zvl`` pin worked around), and the element count at the RVV MINIMUM
#: VLEN of 128 bits is ``2 * k`` for EVERY element type. So ``k = NR // 2`` makes the ``NR`` knob mean
#: "lanes at the minimum VLEN" under VL_DYNAMIC, with the real lane count scaling with the hardware.
_VSCALE_LANES_PER_128B = 2


def scalable_lanes(NR: int) -> int:
    """The scalable multiplier ``k`` of ``vector<[k]xT>`` for a register block ``NR`` lanes wide at
    the RVV MINIMUM VLEN (128 bits). Dtype-independent — see :data:`_VSCALE_LANES_PER_128B`."""
    if NR % _VSCALE_LANES_PER_128B:
        raise ValueError(
            f"vl_strategy='dynamic' needs an even NR (NR counts lanes at the RVV minimum VLEN of "
            f"128 bits, and a scalable vector<[k]xT> holds 2k of them); got NR={NR}.")
    return NR // _VSCALE_LANES_PER_128B


def _accumulator_resident_v3_scalable_pre_schedule(MR: int, NR: int, KC: int) -> str:
    """v3 pre-schedule emitting a VL-AGNOSTIC (scalable) register block — the ``vl_strategy=dynamic``
    axis. Same recipe as :func:`_accumulator_resident_v3_pre_schedule` with the N tile made SCALABLE.

    Why this is the general fix. Our fixed-width codegen asks for ``vector<NRxf32>``, and the backend
    must size that register group for the worst VLEN the ``march`` string admits (``rv64gcv`` promises
    only the 128-bit minimum), so on a VLEN=256 part every value got DOUBLE the LMUL it needed and
    ``vl`` sat at half ``VLMAX``. Pinning ``_zvl256b`` fixes it for ONE board and miscompiles on a
    narrower one. A scalable ``vector<[k]xf32>`` has no such worst case: its LMUL is fixed by the type
    and its lane count is whatever the hardware reports, which is what the expert kernels do by
    calling ``__riscv_vsetvl_e32m4`` at run time. ``NR`` is reinterpreted as lanes at the MINIMUM
    VLEN (see :func:`scalable_lanes`), so ``NR`` still names the register-block width — it just names
    the LMUL rather than a lane count that only one board would honour.

    Two things the naive scalable schedule gets wrong, plus one PRECONDITION it inherits (all
    reproduced against LLVM/MLIR 23, see docs/design/vl_agnostic_codegen.md):

      * The N loop trip count is no longer a compile-time multiple of the tile, so ``vectorize``
        MASKS every transfer. Masked scalable transfers (a) block
        ``loop-invariant-subset-hoisting``, which is what makes the accumulator a register-resident
        ``scf.for`` iter_arg — without it the K loop round-trips C through memory, the exact defect
        v3 exists to remove — and (b) lower through a ``vector.transpose`` on
        ``vector<MRx1x[k]xf32>`` that has NO scalable lowering and survives to the LLVM edge
        (``error: Dialect 'vector' not found for custom op 'vector.transpose'``). THIS transpose,
        not the ``ub.poison`` the old note named, is the real residue of the "incomplete lowering".
        FIX: PEEL the N loop (``transform.loop.peel``) so the main loop's tile is exactly
        ``[k] * vscale`` wide, then vectorize it with ``assume_dynamic_dims_match_vec_sizes`` — the
        peel is what MAKES that assumption true. No masks, no transpose; the subset hoist fires
        exactly as in the fixed-width recipe.
      * The peeled N remainder keeps its ``linalg.matmul`` and falls through
        ``convert-linalg-to-loops`` to a scalar tail — correct for any N, zero-trip whenever the
        hardware VL divides N.
      * PRECONDITION ``MR | M`` (inherited, not introduced). ``assume_dynamic_dims_match_vec_sizes``
        asserts EVERY dynamic tile dim equals its vector size, INCLUDING M. The static ``MR`` M-tile
        has a partial last iteration when ``MR`` does not divide M, and the flag then makes the body
        write ``MR`` rows into a shorter tile (measured on the K1 at 130^3: ``malloc(): corrupted
        top size``; 128^3 and 100^3, where MR|M, are bit-exact). This is the SAME "MR must divide M"
        constraint the fixed 2-D ``vector<MRxNR>`` v3 recipe already carries (see
        expert_gap_attribution.md — small-M falls off a cliff). It is NOT fixed here by peeling M:
        ``transform.loop.peel`` FAILS on a statically-divisible loop, so an M peel would break the
        common MR|M case (M=128) to protect the MR∤M one. The M-tail is an orthogonal axis
        (pad/peel, shared with the fixed recipe); until it lands, MR∤M is fail-closed (the harness
        records not_run on the crash, never a false timing), not silently wrong.

    ``KC`` is carried for naming parity with the other v3 points; like the plain v3 recipe it does
    not tile the reduction (``k_block`` is the lever that does)."""
    k = scalable_lanes(NR)
    return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t1, %lm, %ln = transform.structured.tile_using_for %mm tile_sizes [{MR}, [{k}], 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %lnf = transform.cast %ln : !transform.any_op to !transform.op<"scf.for">
    %main, %tail = transform.loop.peel %lnf : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    %mms = transform.structured.match ops{{["linalg.matmul"]}} in %main : (!transform.any_op) -> !transform.any_op
    %t2, %lk = transform.structured.tile_using_for %mms tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %t2 vector_sizes [{MR}, [{k}], 1] {{assume_dynamic_dims_match_vec_sizes}} : !transform.any_op
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


def ensure_v3_scalable_microkernel(MR: int, NR: int, KC: int) -> str:
    """Register (on demand) the VL-AGNOSTIC v3 tuning point — a scalable (``vsetvli``-sized) N block.

    This is the realization of ``MicrokernelSpec.vl_strategy = VL_DYNAMIC``: the emitted inner loop
    sizes itself to the vector length the hardware reports, so it needs no ``_zvl`` march pin and is
    correct (and full-width) on ANY RVV part."""
    k = scalable_lanes(NR)
    name = f"accum_resident_v3vl_{MR}_{NR}_{KC}"
    if name in known():
        return name
    register(ImprFeature(
        name=name,
        action_class="PASS",
        description=(f"VL-AGNOSTIC accumulator-resident micro-kernel (MR={MR}, NR={NR} lanes at the "
                     f"RVV minimum VLEN -> vector<[{k}]xT>, KC={KC}): the N register block is a "
                     f"SCALABLE vector, so the emitted loop sizes to the hardware's runtime vector "
                     f"length (vsetvli against VLMAX) instead of a compile-time width that the "
                     f"backend must widen for the worst-case VLEN. Needs no _zvl march pin. N loop "
                     f"is peeled so the main body is unmasked; the remainder is a scalar tail. "
                     f"Compiler-emitted (no ukernel). Default-off."),
        edit_pipeline=_accumulator_resident_v3_pipeline,
        edit_schedule=(lambda _t, _MR=MR, _NR=NR, _KC=KC:
                       _accumulator_resident_v3_scalable_pre_schedule(_MR, _NR, _KC)),
        schedule_replace=True,
        implies=_tile_epilogue_hygiene(MR),
    ))
    return name


def ensure_v3_microkernel(MR: int, NR: int, KC: int) -> str:
    """Register (on demand) and return the name of the v3 accumulator-resident micro-kernel tuning
    point for ANY (MR, NR, KC) — turning the fixed 4-point grid into a CONTINUOUS, beam-tunable space.

    This is how the compiler reaches expert-kernel granularity by CODE GENERATION alone: the v3 recipe
    is fully compiler-emitted (tile [MR,NR] + K-by-1, scoped-vectorize -> vector.contract, PRE-bufferize
    subset-hoisting makes the accumulator a register-resident scf.for iter_arg, contraction ->
    outerproduct -> vector.fma, then A-scalarization -> vfmacc.vf). NO hand ukernel is involved — the
    intrinsic driver remains a ceiling REFERENCE only. Registering a point is idempotent."""
    name = ("accumulator_resident_microkernel_v3" if (MR, NR, KC) == (4, 16, 16)
            else f"accum_resident_v3_{MR}_{NR}_{KC}")
    if name in known():
        return name
    register(ImprFeature(
        name=name,
        action_class="PASS",
        description=(f"Accumulator-resident vfmacc.vf micro-kernel tuning point (MR={MR}, NR={NR}, "
                     f"KC={KC}), registered on demand so the beam can tune the register block "
                     f"continuously. Compiler-emitted (no ukernel). Default-off."),
        edit_pipeline=_accumulator_resident_v3_pipeline,
        edit_schedule=(lambda _t, _MR=MR, _NR=NR, _KC=KC:
                       _accumulator_resident_v3_pre_schedule(_MR, _NR, _KC)),
        schedule_replace=True,
        implies=_tile_epilogue_hygiene(MR),
    ))
    return name


def ensure_v3_perop_microkernel(MR_mm: int | None, NR_mm: int | None,
                                MR_bmm: int | None, NR_bmm: int | None,
                                KC: int) -> str:
    """Register (on demand) a v3 tuning point with an INDEPENDENT register block per op class.

    ``_accumulator_resident_v3_pre_schedule`` already emits four separate tile factors — the matmul
    arm uses ``[MR_mm, NR_mm]`` and the batch_matmul arm ``[1, MR_bmm, NR_bmm]`` — but until now the
    only way to reach them was the hand-frozen ``accumulator_resident_wholemodel_vf`` point, whose
    clamps (MR_mm=1, NR_bmm=8) are constants somebody picked once. They are a SHAPE decision: a
    model's matmuls and its attention batch_matmuls have different extents, so the largest
    non-masking block differs per class. This registrar makes that decision expressible, so a
    shape-aware policy can DERIVE the four factors from the workload instead of pinning them.

    ``accumulator_resident_wholemodel_vf`` == ``ensure_v3_perop_microkernel(1, 16, 4, 8, 16)``; that
    point keeps its own name (and its measured history) rather than being aliased here.

    A class's block may be ``None``, meaning DO NOT CLAIM this op class — its contractions are left
    to ``convert-linalg-to-loops``. That is the honest answer when no block wider than one lane is
    legal for the class's extents (see ``_accumulator_resident_v3_pre_schedule``); the alternative is
    a build that fails outright. Both classes ``None`` would vectorize nothing, so it is rejected."""
    skip_mm = MR_mm is None or NR_mm is None
    skip_bmm = MR_bmm is None or NR_bmm is None
    if skip_mm and skip_bmm:
        raise ValueError("ensure_v3_perop_microkernel: at least one op class must be claimed "
                         "(both blocks None vectorizes nothing — use the scalar backend instead)")
    tag_mm = "x_x" if skip_mm else f"{MR_mm}_{NR_mm}"
    tag_bmm = "x_x" if skip_bmm else f"{MR_bmm}_{NR_bmm}"
    name = f"accum_resident_v3p_{tag_mm}_{tag_bmm}_{KC}"
    if name in known():
        return name
    _mm_desc = "linalg.matmul UNCLAIMED (scalar)" if skip_mm else f"linalg.matmul [{MR_mm}, {NR_mm}]"
    _bmm_desc = ("linalg.batch_matmul UNCLAIMED (scalar)" if skip_bmm
                 else f"linalg.batch_matmul [1, {MR_bmm}, {NR_bmm}]")
    register(ImprFeature(
        name=name,
        action_class="PASS",
        description=(f"Accumulator-resident vfmacc.vf micro-kernel with a per-op-class register "
                     f"block ({_mm_desc}, {_bmm_desc}, KC={KC}), registered on demand so a "
                     f"shape-aware policy can pick a blocking that masks no parallel dim. "
                     f"Compiler-emitted (no ukernel). Default-off."),
        edit_pipeline=_accumulator_resident_v3_pipeline,
        edit_schedule=(lambda _t, _mm=MR_mm, _nm=NR_mm, _mb=MR_bmm, _nb=NR_bmm, _KC=KC,
                       _sm=skip_mm, _sb=skip_bmm:
                       _accumulator_resident_v3_pre_schedule(_mb or 1, _nm or 1, _KC,
                                                             NR_bmm=_nb or 1, MR_mm=_mm or 1,
                                                             skip_mm=_sm, skip_bmm=_sb)),
        schedule_replace=True,
        implies=_tile_epilogue_hygiene(None if skip_mm else MR_mm),
    ))
    return name


#: Op-class order encoded in a ``accum_resident_v3p_*`` name (two factors each, then KC).
_V3P_PREFIX = "accum_resident_v3p_"
_V3P_CLASS_ORDER = ("linalg.matmul", "linalg.batch_matmul")


def unclaimed_op_classes(feature: str) -> list[str]:
    """Contraction op classes a ``accum_resident_v3p_*`` point does NOT tile/vectorize.

    Lives next to :func:`ensure_v3_perop_microkernel` so the ``x_x`` spelling has ONE definition:
    a caller that wants to report "this class runs scalar" must not re-derive the name format.
    Returns [] for any other feature (including the hand-frozen points, which claim both classes).
    """
    if not feature.startswith(_V3P_PREFIX):
        return []
    parts = feature[len(_V3P_PREFIX):].split("_")
    if len(parts) != 2 * len(_V3P_CLASS_ORDER) + 1:
        return []
    return [cls for i, cls in enumerate(_V3P_CLASS_ORDER) if parts[2 * i] == "x"]


#: The non-contraction tail: every op the contraction-only schedule never matches (elementwise,
#: layout, im2col gather, pad) falls through convert-linalg-to-loops to SCALAR code on one core. It is
#: 86-89% of the linalg ops of every workload measured, which is why whole-model MAC/cycle sits at a
#: few percent of the datapath's peak while the matmul kernel itself looks fine.
NAMED_INT8_CONTRACTION_NAME = "named_int8_contraction"

register(ImprFeature(
    name=NAMED_INT8_CONTRACTION_NAME,
    action_class="PASS",
    description="Emit the canonical 2-D i8xi8->i32 contraction as a MIXED-TYPE linalg.matmul instead "
                "of a linalg.generic, so the named-op transform schedules can see it. The int8 quant "
                "pass otherwise leaves ZERO linalg.matmul in the module (measured: 15 -> 0 on "
                "small_llama_int8), and transform.structured.match on a name nothing carries returns "
                "an empty handle -- which makes every one of the 39 linalg.matmul/batch_matmul "
                "matchers in this file a vacuous no-op on int8 while still reporting as applied. An "
                "87-fork beam over those levers emitted only 21 distinct binaries. Batched, conv and "
                "non-canonical indexing keep the generic form: a named op ASSERTS an indexing "
                "convention, so claiming one the op does not have would be a correctness bug rather "
                "than a missed optimization. Default-off; baseline byte-identical.",
))

#: Quantize an activation BEFORE the pure data-movement op that expands it, not after.
#:
#: Registered EAGERLY, at import of this module, and listed in `wholemodel_proposer.RANKED_LEVERS`.
#: Both are load-bearing: `_composes` swallows the KeyError for an unregistered name and returns
#: False, so a lazily-registered lever is not declined by the search, it is INVISIBLE to it.
QUANTIZE_BEFORE_GATHER_NAME = "quantize_before_gather"

register(ImprFeature(
    name=QUANTIZE_BEFORE_GATHER_NAME,
    action_class="PASS",
    description=(
        "When a contraction's f32 activation operand is produced by a PURE data-movement op (an "
        "all-parallel linalg.generic whose body only yields its input, i.e. an element copy), "
        "quantize the op's SOURCE with a per-tensor scale instead of quantizing its expanded result "
        "with a per-parallel-row one. Quantization is elementwise, so quantize(G(A)) == G(quantize(A)) "
        "exactly for a single shared scale; what blocks the commutation today is only the per-row "
        "scale, under which one element of A carries a different scale in every im2col column it "
        "appears in. This is the case that matters on every convolutional model here: model2MLIR "
        "expands every conv into im2col + matmul before merlin sees it (190 such ops in deepjscc "
        "int8, 175 in lstmnetvit int8, zero fused conv2d), so the operand being quantized IS the "
        "expanded matrix -- deepjscc enc.net.1 quantizes a 147x4096 f32 matrix, ~41x the "
        "1x3x70x70 activation it was gathered from. With the scale moved, the abs-max and the "
        "quantize both run on the activation and the gather itself moves i8, 4x less traffic for the "
        "same trip count, and the f32 expansion is erased entirely. The abs-max is EXACT in both "
        "modes: over the source when the indexing maps and bounds PROVE the gather reads every "
        "element, otherwise reduced through the gather's own map (same reads, scalar result, no "
        "materialization) -- so a strided or dilated gather that skips elements is handled, not "
        "approximated with the coarser amax(A) >= amax(G(A)). Refuses and counts the reason for a "
        "computed body, a shared intermediate, a dynamic extent or a non-gather producer. "
        "NOT bit-identical: the per-tensor activation scale is a genuine numeric change against the "
        "shipped per-row scheme and must pass the accuracy gate on its own. Default-off; with the "
        "feature absent the int8 datapath is byte-identical."
    ),
))

VEC_NONCONTRACTION_NAME = "vectorize_non_contraction_generics"
#: Default lane count for the bare feature name. The lane width is a KNOB SPACE, not a
#: constant: `ensure_vec_noncontraction(lanes)` registers a point per width so a search can
#: pick it. The 1.28x regression this lever first measured was FLAT across 8/16/32 lanes, which
#: is the tell that the width was never the variable -- see `_vec_noncontraction_hygiene`.
VEC_NONCONTRACTION_LANES = 8


def _vec_noncontraction_hygiene() -> frozenset[str]:
    """The lowering hygiene the per-rank vectorize arms cannot be measured without.

    Each arm TILES the all-parallel generic on TENSORS and then vectorizes the tile, so every tile
    yields a ``tensor.insert_slice`` back into the loop-carried destination. Bufferization realizes
    that destructive update as a subview of the destination, a ``vector.transfer_write`` into it, a
    SECOND structurally identical subview, and a ``memref.copy`` between the two -- i.e. the tile is
    copied ONTO ITSELF, once per tile, INSIDE the innermost loop::

        %sv   = memref.subview %arg99[0, %i, %j, %k] [1, 1, 1, 8] [1, 1, 1, 1]
        vector.transfer_write %v, %sv[...]        // the result is already in place here
        %sv_0 = memref.subview %arg99[0, %i, %j, %k] [1, 1, 1, 8] [1, 1, 1, 1]
        memref.copy %sv, %sv_0                    // ...and this copies it onto itself

    That is the same defect ``selfcopy`` documents for the tiled CONTRACTION epilogue, and it is why
    this lever measured 1.28x SLOWER at bit-identical output with the vector count up 4.9x: the
    per-tile copy is emitted per 8 ELEMENTS, so the vector win is spent on a memcpy of the bytes the
    vector store just wrote. The earlier attribution missed it because it counted ``@memrefCopy``
    call sites -- these copies are contiguous in the innermost dim, so they lower to ``llvm.memcpy``
    instead and the memrefCopy count does not move at all.

    MEASURED at the post-bufferization split point (int8 recaptures, impr_tuned_wholemodel_vf_int8,
    host lowering), copy CALL SITES emitted inside ``forward``:

        deepjscc      lever off  72 memcpy | lever on  87 (+15, one per vectorized op)
                      + erase    52 memcpy | + erase   52 (+0)
        small_llama   lever off  52 memcpy | lever on  85 (+33, one per vectorized op)
                      + erase    39 memcpy | + erase   39 (+0)

    Erasing a ``memref.copy %x, %x`` is unconditionally value-preserving (identical SSA operand =>
    identical base, offsets and region) and is a no-op on a lowering that has none, so implying it
    can only help or do nothing -- and enabling it is what makes ``lower_to_llvm_ir`` splice the
    post-bufferization ``canonicalize,cse`` that collapses the two subviews into the one SSA value
    the erase keys on.

    CORRECTION (supersedes the closing claim of commit ffd3c40f, "the output is bit-identical on
    every arm"). That statement was WRONG for small_llama int8 and it was wrong because of the
    sample, not the reasoning: the copy census above was taken on deepjscc and small_llama, and only
    the OUTPUT DIGEST of deepjscc was compared. On small_llama the lever's output was cos 0.968247 /
    rel 0.46352 against the baseline's 0.999966 / 0.00836 -- a MISCOMPILE, not a rounding difference,
    and one that produced two different answers (``756d00f36c43``, ``d9d8a01aa32e``) from the SAME
    shared object depending on the host process's memory layout. The cause is unrelated to the
    self-copy erase (it is the sub-byte element hazard :func:`_vec_bytewise_matchers` documents and
    now refuses); what the erase claim got wrong is only that it generalized a bit-identity measured
    on one model to "every arm". With the refusal in place the output IS bit-identical to the
    baseline on all three int8 recaptures, in both arm placements, under three initial memory
    layouts -- which is the statement this note is willing to make.

    It is an ``implies`` rather than a note in the description for the reason ``_tile_epilogue_hygiene``
    gives: a default-off lever whose payoff is cancelled by a separate default-off fix is an inert
    lever, because everyone naming it directly in ``compiler_features`` gets the cancelled version.
    """
    return frozenset({_SELF_COPY_FEATURE})


#: Attribute the byte-addressable-element MATCHERS below annotate an op with. An arm matches
#: ``merlin.vec_r{rank}`` AND this, so an op only vectorizes when its own IR says every tensor it
#: reads and writes has a byte-or-wider element type. See :func:`_vec_bytewise_matchers`.
VEC_BYTEWISE_ATTR = "merlin.vec_ok"

#: Smallest element width, in bits, whose in-memory layout a ``vector.transfer`` and a scalar
#: ``memref`` access agree on. NOT a tuning choice and not a target fact: a `memref<...xT>` is
#: addressed at `sizeof(T)` rounded UP to a byte, while a `vector<NxT>` is PACKED, so at T=i1 a
#: `vector<8xi1>` occupies ONE byte where the scalar form occupies eight. Everything at 8 bits or
#: wider has the same layout in both.
VEC_BYTEWISE_MIN_BITS = 8

#: Largest ``ins`` arity the matchers below enumerate. The check has to hold for EVERY input, and
#: ``match.structured.elemental_bitwidth`` takes ONE value (a ``%s[all]`` handle is rejected by its
#: single-value trait), so the arity is enumerated and each input checked by index. An op with more
#: inputs than this matches NO matcher, is never annotated, and is therefore NOT vectorized -- the
#: refusal is what "fail closed" means here, and it is why the bound may be raised but never removed.
VEC_BYTEWISE_MAX_INPUTS = 6


def _vec_bytewise_matcher_prefix(text: str) -> str | None:
    """Symbol prefix for the matchers spliced into ``text``, derived from ITS OWN entry point.

    ``transform-preload-library`` merges every library it is given into one transform module, and the
    non-contraction arms are spliced into TWO of them (the pre-specialization library and the package
    schedule). A fixed symbol name would collide there. The entry-point symbol is what distinguishes
    the two libraries, so deriving the prefix from it makes the names unique without anyone choosing
    one. Returns None when the text has no named sequence to derive from (then nothing is spliced).
    """
    key = "transform.named_sequence @"
    for line in text.splitlines():
        head, sep, tail = line.partition(key)
        if not sep:
            continue
        name = tail.split("(", 1)[0].strip()
        if name:
            return f"__merlin_vec_bytewise_{name.lstrip('_')}_"
    return None


def _vec_bytewise_matchers(prefix: str, *, min_bits: int = VEC_BYTEWISE_MIN_BITS,
                           max_inputs: int = VEC_BYTEWISE_MAX_INPUTS) -> str:
    """Named matcher sequences that accept a ``linalg.generic`` iff EVERY tensor it reads or writes
    has an element at least ``min_bits`` wide.

    WHY THIS EXISTS -- a MISCOMPILE, measured, not inferred. The arms vectorize an op by tiling it and
    writing each tile with a ``vector.transfer_write``. For an element type narrower than a byte the
    two sides of that write disagree about the layout of the SAME buffer: LLVM stores a `vector<8xi1>`
    PACKED (one byte for eight lanes) while the memref the tile belongs to is addressed one byte per
    element, and every scalar consumer reads it that way. MEASURED on small_llama int8 (host lowering,
    old arm placement), the causal mask is a `tensor<8x8xi1>`::

        %3147 = call ptr @malloc(i64 128)          ; 64 elements, ONE BYTE EACH
        ...
        %3163 = getelementptr i1, ptr %3152, i64 %3157   ; %3157 = row * 8
        store <8 x i1> %3162, ptr %3164, align 1   ; ...but this writes ONE byte
        ...
        %3204 = getelementptr inbounds nuw i1, ptr %3152, i64 %3203
        %3205 = load i1, ptr %3204, align 1        ; scalar consumer, one byte per element
        %3208 = select i1 %3205, float 0xFFF0000000000000, float %3207

    8 of the 64 bytes are written and 56 are read back UNINITIALISED, straight into the attention
    mask's `select`. The result is wrong (cos 0.9682 / rel 0.4635 against a baseline 0.99997 / 0.0084)
    and depends on what happened to be in that allocation, so the same binary answers differently
    under a different initial stack/heap layout. Both directions are unsound: a vectorized WRITE of a
    sub-byte destination, and a vectorized READ of a sub-byte input some scalar loop wrote -- which is
    why the inputs are checked as well as the destination.

    Realized as matchers rather than as a tagging-time predicate because the tag is applied by the
    prepare pass and the arms are the last thing that decides; a schedule that carries the arms
    carries the refusal with them, and cannot be armed without it.

    One sequence per ``ins`` arity: ``match.structured.elemental_bitwidth`` takes a SINGLE value, and
    ``%s[all]`` is rejected by its single-value trait, so each input is checked by index and the arity
    is pinned with ``num_inputs``. An op outside the enumerated arities matches nothing and is
    refused. ``num_inits`` is pinned to 1 for the same reason -- a second destination would go
    unchecked otherwise.

    WHAT THIS IS NOT, checked and ruled out. The obvious suspect was PARTIAL COVERAGE -- the extra
    destination buffer the arms allocate per vectorized op, written only where the tile lands and
    read where it does not. It is not that. The tagging predicate admits an op only when its
    innermost extent is a whole multiple of the lane count and the arms tile every other dim by 1, so
    the tiles cover the destination exactly: MEASURED on small_llama int8 at the post-interpreter
    module, every one of the 33 (old placement) / 84 (new placement) tile writes carries
    ``in_bounds = [true...]``, there is not one ``affine.min`` trip-count clamp and not one
    ``vector.mask`` in the whole module. The uninitialised bytes come from the ELEMENT WIDTH, not
    from the tiling.
    """
    out: list[str] = []
    for n in range(max_inputs + 1):
        lines = [
            f"  transform.named_sequence @{prefix}{n}(%op: !transform.any_op "
            "{transform.readonly}) -> !transform.any_op {",
            '    transform.match.operation_name %op ["linalg.generic"] : !transform.any_op',
            "    transform.match.structured %op : !transform.any_op {",
            "    ^bb0(%s: !transform.any_op):",
            f"      %bits = transform.param.constant {min_bits} : i64 -> !transform.param<i64>",
            "      %one = transform.param.constant 1 : i64 -> !transform.param<i64>",
            f"      %arity = transform.param.constant {n} : i64 -> !transform.param<i64>",
            "      %ni = transform.match.structured.num_inputs %s : (!transform.any_op) -> "
            "!transform.param<i64>",
            "      transform.match.param.cmpi eq %ni, %arity : !transform.param<i64>",
            "      %no = transform.match.structured.num_inits %s : (!transform.any_op) -> "
            "!transform.param<i64>",
            "      transform.match.param.cmpi eq %no, %one : !transform.param<i64>",
            "      %init = transform.match.structured.init %s[0] : (!transform.any_op) -> "
            "!transform.any_value",
            "      %bwinit = transform.match.structured.elemental_bitwidth %init : "
            "(!transform.any_value) -> !transform.param<i64>",
            "      transform.match.param.cmpi ge %bwinit, %bits : !transform.param<i64>",
        ]
        for k in range(n):
            lines += [
                f"      %in{k} = transform.match.structured.input %s[{k}] : (!transform.any_op) -> "
                "!transform.any_value",
                f"      %bw{k} = transform.match.structured.elemental_bitwidth %in{k} : "
                "(!transform.any_value) -> !transform.param<i64>",
                f"      transform.match.param.cmpi ge %bw{k}, %bits : !transform.param<i64>",
            ]
        lines += ["    }", "    transform.yield %op : !transform.any_op", "  }"]
        out.append("\n".join(lines))
    return "\n".join(out) + "\n"


def _vec_bytewise_annotate(prefix: str, *, max_inputs: int = VEC_BYTEWISE_MAX_INPUTS) -> str:
    """The arm-body lines that run those matchers and mark what they accept."""
    return "".join(
        f"    %ok{n} = transform.collect_matching @{prefix}{n} in %arg0 : "
        f"(!transform.any_op) -> !transform.any_op\n"
        f'    transform.annotate %ok{n} "{VEC_BYTEWISE_ATTR}" : !transform.any_op\n'
        for n in range(max_inputs + 1))


#: Bounded per-rank vectorize of the tagged all-parallel generics. BOUNDED on purpose: a plain
#: no-sizes `vectorize` on a whole model explodes (vector<17x576> = 9792 lanes, measured 8725 ms), and
#: `vectorize_children` on the func scalarizes every generic into tens of thousands of vector.extracts
#: that this LLVM build does not lower. Innermost 8 lanes, one arm per loop rank, matched by the
#: `merlin.vec_r{rank}` attribute the prepare pass sets.
#: Each arm TILES to the vector width before vectorizing, exactly as the contraction arms do.
#: ``structured.vectorize`` does NOT tile: its sizes have to cover the iteration space, so vectorizing
#: an untiled 1x64 relu with [1, 8] fails the whole pipeline with "Attempted to vectorize, but failed"
#: (measured on deepjscc). Tiling first also means the vector shape is exactly the tile, so nothing is
#: masked -- the tagging predicate only admits extents that are whole multiples of the lane count.
def _vec_rank_arms(lanes: int, prefix: str | None = None) -> str:
    """The per-rank tile+vectorize arms at ``lanes`` innermost lanes.

    ``prefix`` names the byte-addressable-element matchers (see :func:`_vec_bytewise_matchers`); the
    arms then match ``merlin.vec_r{rank}`` AND the attribute those matchers annotate, so a sub-byte
    op is refused rather than mis-vectorized. Default None keeps the arms self-contained for the
    tests that read them without a module to splice matchers into.
    """
    gate = f", {VEC_BYTEWISE_ATTR}" if prefix else ""
    mark = _vec_bytewise_annotate(prefix) if prefix else ""
    return mark + f"""\
    %g2 = transform.structured.match attributes{{merlin.vec_r2{gate}}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %gt2, %gl2:2 = transform.structured.tile_using_for %g2 tile_sizes [1, {lanes}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %gt2 vector_sizes [1, {lanes}] : !transform.any_op
    %g3 = transform.structured.match attributes{{merlin.vec_r3{gate}}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %gt3, %gl3:3 = transform.structured.tile_using_for %g3 tile_sizes [1, 1, {lanes}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %gt3 vector_sizes [1, 1, {lanes}] : !transform.any_op
    %g4 = transform.structured.match attributes{{merlin.vec_r4{gate}}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %gt4, %gl4:4 = transform.structured.tile_using_for %g4 tile_sizes [1, 1, 1, {lanes}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %gt4 vector_sizes [1, 1, 1, {lanes}] : !transform.any_op
    %vecf = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %vecf {{
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
      transform.apply_patterns.vector.drop_unit_dims_with_shape_cast
      transform.apply_patterns.vector.lower_shape_cast
    }} : !transform.any_op
"""



def _splice_vec_rank_arms(text: str, lanes: int = VEC_NONCONTRACTION_LANES) -> str:
    """Insert the per-rank vectorize arms just before the schedule's func-level pattern block.

    ADDITIVE (not schedule_replace), so it layers on whatever micro-kernel recipe is in play: the
    contraction arms above it are untouched and only the previously-unmatched generics gain arms.
    A schedule that already carries them, or has no func anchor, is returned unchanged.
    """
    anchor = '    %f = transform.structured.match ops{["func.func"]}'
    if "merlin.vec_r" in text or anchor not in text:
        return text
    prefix = _vec_bytewise_matcher_prefix(text)
    head = _vec_module_header(text)
    if prefix is None or head is None:
        # Nowhere to put the refusal matchers => splice NO arms. Arming without them is the
        # miscompile `_vec_bytewise_matchers` documents, and a lever that is off is recoverable
        # where a lever that is silently wrong is not.
        return text
    text = text.replace(head, head + _vec_bytewise_matchers(prefix), 1)
    return text.replace(anchor, _vec_rank_arms(lanes, prefix) + anchor, 1)


def _vec_module_header(text: str) -> str | None:
    """The ``module ... {`` line the matcher sequences are inserted after, or None.

    Structural, not positional: the first line that OPENS a module. A schedule whose module this
    cannot find gets no arms at all rather than arms without their refusal matchers.
    """
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("module") and stripped.endswith("{"):
            return line
    return None


register(ImprFeature(
    name=VEC_NONCONTRACTION_NAME,
    action_class="PASS",
    description=(
        "Bounded per-rank vectorize of the NON-CONTRACTION all-parallel linalg.generics "
        "(elementwise / layout / im2col gather / pad) that the contraction-only schedule leaves for "
        "convert-linalg-to-loops to emit as scalar loops. Measured share of the loss: 86-89% of the "
        "linalg ops of every captured workload, against whole-model MAC/cycle of 0.40 (spectformer), "
        "0.22 (deepjscc), 0.067 (lstmnetvit) versus ~8 for a VLEN=128 int8 vwmacc datapath. Needs the "
        "prepare pass's merlin.vec_r{rank} tags, which build_app enables when this feature is on. "
        "MEASURED on deepjscc int8 (spike): emits 394 -> 1945 vector instructions (4.9x, so the "
        "lever is NOT inert), output BIT-IDENTICAL, and 484,690,000 -> 621,555,001 cycles, i.e. "
        "1.28x SLOWER, flat at 0.78x across 8/16/32 lanes. That flatness was the tell that the "
        "width was never the variable: the tile-and-vectorize-on-tensors realization emits a "
        "`memref.copy %x, %x` of each tile INSIDE the innermost loop -- a memcpy of the bytes the "
        "vector store just wrote, per 8 elements. It went unattributed because those copies lower "
        "to `llvm.memcpy`, not to the `@memrefCopy` the earlier check counted. This feature now "
        "IMPLIES `erase_self_copy`, which removes them: measured on the host lowering, the copy "
        "call sites the lever adds inside `forward` go from +15 (deepjscc) / +33 (small_llama) to "
        "+0 on both, with the vector-instruction gain kept. The output was reported bit-identical "
        "at that point and it was NOT: on small_llama int8 the lever answered cos 0.968247 / rel "
        "0.46352 against a baseline 0.999966 / 0.00836, and answered DIFFERENTLY from the same "
        "object under a different process memory layout, because a `vector<8xi1>` mask tile is "
        "stored PACKED into a buffer every scalar reader addresses one byte per element (56 of 64 "
        "bytes left uninitialised, read straight into the attention mask). The arms now refuse a "
        "sub-byte element type on the destination or on any input, and with that refusal the "
        "output IS bit-identical to the baseline on all three int8 recaptures, in both arm "
        "placements, under three memory layouts. See `_vec_bytewise_matchers`. "
        "NOT yet claimed as a speedup -- that is a board measurement, and the cycle number above "
        "is the one it has to beat. Two known limits remain, both MEASURED and neither fixable "
        "from this file: (a) COVERAGE -- `func.func(linalg-specialize-generic-ops)`, which runs "
        "before the transform interpreter so the contraction arms can match named ops, rewrites "
        "the tagged generics into `linalg.broadcast` and DROPS their `merlin.vec_r{rank}` "
        "attribute, so only 15 of 93 tagged ops (deepjscc) ever reach an arm while the prepare "
        "pass reports 93; (b) the arms still allocate one extra destination buffer per vectorized "
        "op (+12 allocs / +1.05 MB of 50.19 MB cumulative on deepjscc), which is malloc calls "
        "rather than per-element traffic."),
    edit_schedule=_splice_vec_rank_arms,
    implies=_vec_noncontraction_hygiene(),
))


def vec_noncontraction_lanes(features) -> int | None:
    """Innermost lane count the enabled non-contraction-vectorize feature asks for, else None.

    The TAGGING predicate (which admits an op only when its innermost extent is a whole multiple of the
    lane count, so nothing is masked) and the SCHEDULE arms must agree on this number; deriving it from
    the feature name is what keeps them from drifting apart.
    """
    for f in features or ():
        if f == VEC_NONCONTRACTION_NAME:
            return VEC_NONCONTRACTION_LANES
        if f.startswith(f"{VEC_NONCONTRACTION_NAME}_l"):
            tail = f.rsplit("_l", 1)[1]
            if tail.isdigit():
                return int(tail)
    return None


# ---------------------------------------------------------------------------------------------------
# `linalg.broadcast`, PRICED AND DECLINED. It surfaced as 8.95% of the profiled accelerator device leg
# -- third behind linalg.generic (62.48%) and linalg.transpose (12.52%), and a lever nobody had named --
# so it looked like a candidate for the vectorize lever right below. It is not, and the reason is the
# same reason that lever measured inert.
#
# Priced from the completed profiled leg (5,032 PROF lines joined against its op_profile_table.json,
# resolved coverage 1.00000, 300,868,459 of 3,359,861,701 ticks across 567 distinct ops), then read
# back out of the instrumented module. Every one is a RANK-EXPANDING materialization of a small vector
# into a large tensor -- `256xf32 -> 1x196x256xf32` (x70), `2048xf32 -> 2048x14xf32` (x64),
# `196xf32 -> 196x256xf32` (x48) -- i.e. a scale/bias/mask vector written out at full tensor size so
# the next elementwise op can read it index-for-index. Across all 567: 3.3 MiB read, 171.1 MiB
# WRITTEN, a 51.9x amplification, and ZERO arithmetic.
#
# So it is pure memory traffic, and "vectorize the broadcast loop" cannot be the fix: wider stores do
# not reduce the number of bytes stored. That is exactly what `vectorize_non_contraction_generics`
# measured -- 4.9x more vector instructions, bit-identical output, 1.28x SLOWER, flat across 8/16/32
# lanes -- because the cost was never vector width. The only real fix is to NOT materialize the tensor:
# have the consumer read the small operand through a broadcasting indexing map and drop the broadcast
# entirely. That is fusion / indexing-map rewriting, which is the lean-runtime fusion program and is
# explicitly out of scope here. Recorded so the next reader does not re-derive it and reach for the
# lever below.
# ---------------------------------------------------------------------------------------------------


def ensure_vec_noncontraction(lanes: int) -> str:
    """Register (on demand) the non-contraction vectorize point at ``lanes`` innermost lanes.

    Exposing the width as a registered point per value is what lets a search find one that pays
    instead of a human guessing. It is NOT the axis the first measurement moved: cycles were flat at
    0.78x across 8/16/32, because every width paid the same per-tile self-copy (see
    ``_vec_noncontraction_hygiene``). Each point implies the same hygiene as the bare name, so a
    width the search picks is measured in the same realization as the default one.
    """
    if lanes == VEC_NONCONTRACTION_LANES:
        return VEC_NONCONTRACTION_NAME
    name = f"{VEC_NONCONTRACTION_NAME}_l{lanes}"
    if name in known():
        return name
    register(ImprFeature(
        name=name, action_class="PASS",
        description=(f"Bounded per-rank vectorize of the non-contraction all-parallel generics at "
                     f"{lanes} innermost lanes (see {VEC_NONCONTRACTION_NAME}). Default-off; the width "
                     f"must be chosen by measurement, not assumed."),
        edit_schedule=lambda t, _l=lanes: _splice_vec_rank_arms(t, _l),
        implies=_vec_noncontraction_hygiene(),
    ))
    return name


#: Per-op register blocking: each contraction is tiled at the block legal for ITS OWN extents, instead
#: of one block per op CLASS. The class-wide decision is one too coarse -- whisper_tiny's single N=1
#: decode step forces its whole batch_matmul class (34% of the model's MACs) off the vector path. See
#: llvmlower/perop_blocks.py for why the tag has to be applied after specialization.
#: Promote small heap buffers to the STACK, after bufferization has created them and hoisting has
#: pulled them out of loops.
#:
#: WHY IT SHOULD MATTER HERE, and it is not the malloc cost. Bufferization gives every intermediate
#: its own `memref.alloc`; measured on small_llama int8 there are 209 of them, but they only price at
#: ~3% of wall, so removing the allocator calls is not the prize. The prize is LOCALITY: separate heap
#: buffers are scattered, so every intermediate is written and re-read through cache misses, while a
#: stack frame is one contiguous, hot region. That is structurally what ExecuTorch buys with its
#: memory-planning pass, which places EVERY activation of this model into a single 32,512-byte arena
#: (measured from its own AOT profile) -- small enough to stay in L1. We have no such planner wired:
#: `arena_plan.plan_arena` has exactly one caller, `runtime/program.py:97` (`build_program`), and the
#: replay engine that would consume its plan whole-model does not exist in the tree at all
#: (`merlin_program.c` is named by three docstrings and absent; `merlin_bump_linux.c` is a
#: measurement proxy). It was previously recorded here as "only reachable from the self-contained C
#: harness" -- that is a NAME COLLISION, not a reachability claim: `selfcontained_c_harness` defines
#: its own unrelated `build_program` and references no arena at all. What wiring it actually costs is
#: written up in docs/design/static_arena_wiring.md. Sized on the model we measure (small_llama
#: int8): 433 intermediate buffers, 3,114,112 B naive vs 265,536 B planned, 11.73x reuse.
#: Meanwhile this upstream pass gets a large part of the same effect for free.
#:
#: The size cap is the safety property, not a tuning knob: a promoted buffer lives on the stack for
#: the whole frame, so an uncapped promotion of a big intermediate overruns it. Zephyr's master stack
#: here is 8 MB and these intermediates are small (the largest activation on this model is 8x344 f32
#: = 11 KB), so the cap is what keeps a promotion from turning into a stack overflow on a model with
#: larger tensors.
#: Fuse elementwise producers into their consumers AFTER named ops have been generalized.
#:
#: The upstream pipeline runs `linalg-fuse-elementwise-ops` at position 2 and
#: `linalg-generalize-named-ops` at position 3 -- fusion BEFORE generalization. Fusion works on
#: `linalg.generic`; a NAMED op (linalg.broadcast, linalg.add, linalg.mul...) is not generalized until
#: the pass after, so it can never be fused. The ordering makes the fusion pass structurally unable to
#: see most of what it exists to fuse.
#:
#: MEASURED cost of that on small_llama int8: 39 `linalg.broadcast` ops, 23.9% of the model, each
#: MATERIALISING a per-channel quantization scale into a full weight-sized tensor -- a 344-element
#: scale vector splatted to tensor<344x128> (44,032 elements, a 128x amplification) and then read
#: straight back by the dequantize. Folding a broadcast into its consumer's indexing map is exactly
#: what elementwise fusion does, and it never got the chance.
#:
#: Added as a SECOND fusion after generalization rather than by reordering the existing one: the
#: pre-generalization pass still fuses the generics that are already generic, and leaving it in place
#: keeps this a strict addition to the baseline rather than a reshuffle whose effect is harder to
#: attribute. General, not target-specific -- every per-channel-quantized model has this shape.
FUSE_AFTER_GENERALIZE_NAME = "fuse_elementwise_after_generalize"


def _fuse_after_generalize(passes: list[str]) -> list[str]:
    """Insert a second ``linalg-fuse-elementwise-ops`` immediately after generalization."""
    out = list(passes)
    anchor = "func.func(linalg-generalize-named-ops)"
    try:
        i = out.index(anchor)
    except ValueError:
        raise ValueError(
            f"{FUSE_AFTER_GENERALIZE_NAME}: anchor {anchor!r} not in the pipeline; refusing to guess "
            f"where the second fusion belongs") from None
    out.insert(i + 1, "func.func(linalg-fuse-elementwise-ops)")
    return out


register(ImprFeature(
    name=FUSE_AFTER_GENERALIZE_NAME,
    action_class="PASS",
    description="run elementwise fusion a second time, AFTER linalg-generalize-named-ops. The "
                "upstream order fuses (pos 2) before generalizing (pos 3), and fusion only sees "
                "linalg.generic -- so every NAMED op is invisible to it. MEASURED on small_llama "
                "int8: 39 linalg.broadcast ops at 23.9% of the model, each materialising a "
                "per-channel quantization scale into a full weight-sized tensor (a 344-element "
                "vector splatted to 44,032 elements, then read straight back). "
                "MEASURED AND REFUTED as a whole-model lever on that same model: 3,543,517 -> "
                "4,313,041 ns, 1.22x SLOWER (sustained, n=3, cos identical). It does what it says -- "
                "the scalar bucket falls 3.1 -> 2.8 ms -- but the contraction RISES 1.8 -> 2.8 ms, "
                "because fusing the dequant chain into the producers perturbs the shape the "
                "contraction was vectorized into and costs more than the broadcast it removes. Kept "
                "registered so the finding is reproducible and the lever is not re-attempted blindly; "
                "the broadcast waste is real but needs a TARGETED fold of the broadcast into its "
                "consumer's indexing map, not blanket elementwise fusion. Default-off; baseline "
                "byte-identical.",
    edit_pipeline=_fuse_after_generalize,
))


PROMOTE_STACK_NAME = "promote_buffers_to_stack"
PROMOTE_STACK_BYTES_ENV = "MERLIN_PROMOTE_STACK_BYTES"
_PROMOTE_STACK_DEFAULT_BYTES = 16384


def _promote_stack_bytes() -> int:
    """Per-buffer promotion cap in bytes (env-overridable so the search can size it per model)."""
    raw = os.environ.get(PROMOTE_STACK_BYTES_ENV)
    if raw:
        try:
            v = int(raw)
        except ValueError:
            v = 0
        if v > 0:
            return v
    return _PROMOTE_STACK_DEFAULT_BYTES


def _promote_buffers_to_stack(passes: list[str], cap: int | None = None) -> list[str]:
    """Insert ``promote-buffers-to-stack`` right after the buffer hoisting.

    After hoisting (so a buffer already lifted out of a loop is promoted once, not per iteration) and
    before the loop lowering. Deallocation runs earlier in this pipeline (``__DEALLOC__``), and a
    stack buffer needs none -- promoting after it is what keeps the two consistent.

    ``cap`` names the per-buffer byte cap explicitly (the sized variants minted by
    :func:`ensure_promote_stack`); ``None`` keeps the env-or-default behaviour.
    """
    out = list(passes)
    anchor = "func.func(buffer-hoisting,buffer-loop-hoisting)"
    try:
        i = out.index(anchor)
    except ValueError:  # pipeline shape changed -> fail closed rather than insert somewhere wrong
        raise ValueError(
            f"{PROMOTE_STACK_NAME}: anchor {anchor!r} not in the pipeline; refusing to guess where "
            f"promote-buffers-to-stack belongs") from None
    nbytes = _promote_stack_bytes() if cap is None else int(cap)
    out.insert(i + 1,
               f"func.func(promote-buffers-to-stack{{max-alloc-size-in-bytes={nbytes}}})")
    return out


register(ImprFeature(
    name=PROMOTE_STACK_NAME,
    action_class="PASS",
    description="promote small bufferization allocs to the stack after hoisting. Targets LOCALITY, "
                "not allocator cost: bufferization gives each intermediate its own memref.alloc (209 "
                "of them on small_llama int8, ~3% of wall), and scattered heap buffers mean every "
                "intermediate is written and re-read through cache misses. ExecuTorch's memory "
                "planner places EVERY activation of the same model into ONE 32,512-byte arena, small "
                "enough to stay in L1; we have no whole-model planner wired, and this upstream pass "
                "buys much of the same effect. Per-buffer cap (MERLIN_PROMOTE_STACK_BYTES, default "
                "16384) is a stack-overflow guard, not a tuning knob. Default-off; baseline "
                "byte-identical.",
    edit_pipeline=_promote_buffers_to_stack,
))


def ensure_promote_stack(nbytes: int) -> str:
    """Register (on demand) a stack-promotion variant whose per-buffer cap is ``nbytes``.

    The cap is a LEVER, not just an overflow guard, and the difference is measured. On small_llama
    int8 (K1, sustained, cos identical on every arm) the SAME feature pays back very differently
    depending only on this number:

        <=  4 KB  4,904,957 ns  1.00x      <= 256 KB  3,649,518 ns  1.34x
        <= 16 KB  4,795,823 ns  1.03x      <=   1 MB  3,579,566 ns  saturated
        <= 64 KB  4,678,494 ns  1.05x

    Monotonic in the cap and flat past 256 KB, which locates the largest intermediate worth promoting
    between 64 KB and 256 KB -- a per-MODEL fact about intermediate sizes, not a constant.

    Why this function exists: the cap was reachable ONLY through ``MERLIN_PROMOTE_STACK_BYTES``, and
    no fork can vary an environment variable. So the beam always built at the 16 KB default, measured
    1.03x, and ranked the lever below levers it should beat -- in the run that prompted this, the
    stack-promotion fork came out SLOWER than its own parent and the search concluded the lever was
    not worth its width. The lever was never weak; the search could not reach the part of it that
    works. Naming the cap in the FEATURE puts it on the channel feature names already travel
    (proposer -> knobs.yaml ``compiler_features`` -> ``normalize`` -> build), so it needs no new
    plumbing and cannot silently disagree with the build the way an ambient env var can.
    """
    n = int(nbytes)
    if n <= 0:
        raise ValueError(f"{PROMOTE_STACK_NAME}: per-buffer cap must be positive, got {nbytes!r}")
    name = f"{PROMOTE_STACK_NAME}_{n}"
    if name in known():
        return name
    register(ImprFeature(
        name=name,
        action_class="PASS",
        description=(f"promote small bufferization allocs to the stack, per-buffer cap {n} bytes. "
                     f"Same pass as {PROMOTE_STACK_NAME}, with the cap NAMED so the search can vary "
                     f"it: the cap is model-dependent (measured 1.03x at 16 KB vs 1.34x at 256 KB on "
                     f"the same model and the same feature), and as an env var it was unreachable by "
                     f"any fork. Default-off; baseline byte-identical."),
        edit_pipeline=lambda passes, _n=n: _promote_buffers_to_stack(passes, cap=_n),
    ))
    return name


PEROP_BLOCK_NAME = "perop_register_block"


def _perop_sentinel_unresolved(_passes):
    """Reached only if the sentinel survived to lowering, which is a BUG -- so raise, loudly.

    ``perop_register_block`` is a REQUEST, not a lowering edit: the block table can only be derived
    from the PREPARED module, so ``zephyr_model.prepare_for_lowering`` derives it, tags the IR, and
    swaps this name for the concrete ``ensure_perop_block(...)`` feature. If the sentinel is still
    enabled at ``apply_pipeline`` time, the caller skipped that step -- and the consequence is
    invisible: nothing tagged the IR, so no schedule arm matches and EVERY contraction falls to
    ``convert-linalg-to-loops`` while the build reports success and the numbers stay correct. That is
    the measured deepjscc "2.56x regression that looks like a bad block but is an untagged build".
    Failing here is the only way that gets noticed.
    """
    raise RuntimeError(
        f"{PEROP_BLOCK_NAME!r} reached the lowering pipeline unresolved. It must be consumed by "
        "runtime.backends.zephyr_model.prepare_for_lowering, which derives the per-op block table "
        "from the prepared IR, tags the contractions, and replaces this sentinel with the concrete "
        "ensure_perop_block(...) feature. Lowering with it still set would leave every contraction "
        "untagged and silently scalar.")


PEROP_NR_FILL_NAME = "perop_nr_fill_register"


def _perop_nr_fill_unresolved(_passes):
    """Same contract as the block sentinel: a REQUEST consumed at preparation time, never a pass."""
    raise RuntimeError(
        f"{PEROP_NR_FILL_NAME!r} reached the lowering pipeline unresolved. It must be consumed by "
        "runtime.backends.zephyr_model.prepare_for_lowering, which is where the per-op block table is "
        "derived and is therefore the only place the board's vector length can widen an N cap.")


# A SEARCH KNOB, deliberately not a default, because its SIGN depends on the model. NR is an element
# count, so one number is a different fraction of the register file at each element width; this asks
# block_table to widen each contraction's N cap until its narrowest element fills a whole vector
# register (perop_blocks.nr_cap_for_dtypes). MEASURED on the live K1 (VLEN=256, whole model, per-op
# blocking on both arms, interleaved same-session arms, n=3, min-of-n, cos identical per model):
#
#   spectformer_int8_full        2,066,414,196 -> 1,781,260,404 cyc   1.160x FASTER
#   small_llama_int8_consistent      8,822,088 ->    10,547,988 cyc   1.196x SLOWER
#
# Both are far outside the board's 2.6% band, and they point OPPOSITE ways. The mechanism is visible in
# the emitted object and explains why: the accumulator is i32, not i8, so at VLEN=256 an NR=16 tile is
# already LMUL m4 and NR=32 pushes it to m8 -- the whole register file in one group. Decoded at 128^3
# int8 MR=4: NR=16 emits `e32,m4` with ZERO accumulator spill ops; NR=32 emits `e32,m8` with SIX
# (vs8r.v x3 + vl8re8.v x3), i.e. the carried accumulator round-trips through the stack every K-tile.
# Whether the wider tile wins is therefore a per-model question about how much of the model is wide
# enough to pay for that -- exactly the kind of question the beam exists to answer and a hand-picked
# default cannot. So it ships forkable and off.
#
# `implies` the block sentinel because it has no meaning without per-op blocking: there is no per-op N
# cap to widen otherwise. schedule_replace stays False -- it changes the TABLE, not the schedule shape,
# and the replacement schedule comes from the block feature it implies.
register(ImprFeature(
    name=PEROP_NR_FILL_NAME,
    action_class="KNOB",
    description="widen each contraction's per-op N cap until its NARROWEST element fills a whole "
                "vector register at the board's VLEN, instead of every op sharing one element count. "
                "MEASURED model-dependent on the K1: 1.160x faster on spectformer int8, 1.196x slower "
                "on small_llama int8 -- because the i32 accumulator is what sets LMUL, so a wider N "
                "tile can push it from m4 to m8 and spill (decoded: 0 -> 6 accumulator spill ops). A "
                "search knob, not a default. Default-off; baseline byte-identical.",
    edit_pipeline=_perop_nr_fill_unresolved,
    implies=frozenset({PEROP_BLOCK_NAME}),
))


# Registered so the SEARCH can reach it. The beam composes candidate feature sets through
# `impr_features.get`/`normalize`, so an unregistered name is not "rejected" -- it is silently never
# proposed (`wholemodel_proposer._composes` catches the KeyError and returns False). An unregisterable
# lever is an unsearchable one, which is the exact failure this whole line of work is about.
# `schedule_replace=True` is honest: what it resolves TO emits a complete transform schedule, so the
# composition rule must refuse stacking it with another replacement.
register(ImprFeature(
    name=PEROP_BLOCK_NAME,
    action_class="PASS",
    description="request PER-CONTRACTION register blocking: derive the widest block legal for each "
                "contraction's OWN extents (and its own narrowest element width) from the prepared "
                "IR, tag each contraction, and emit one tile+vectorize arm per distinct block. "
                "Resolved by prepare_for_lowering into a concrete, table-specific feature; a "
                "sentinel, so it must never reach lowering itself. Replaces the class-wide clamps "
                "(one degenerate extent in a class otherwise forces the whole class off the vector "
                "path). Default-off; baseline byte-identical.",
    edit_pipeline=_perop_sentinel_unresolved,
    schedule_replace=True,
))


#: Sentinel family that names the per-op MR cap. ``perop_register_block_mr<N>`` behaves exactly like
#: :data:`PEROP_BLOCK_NAME` except that it pins the cap ``block_table`` derives blocks under, instead
#: of reading the ambient ``MERLIN_PEROP_MR_CAP``.
PEROP_MR_SENTINEL_PREFIX = f"{PEROP_BLOCK_NAME}_mr"


def parse_perop_mr_sentinel(name: str) -> int | None:
    """The MR cap a ``perop_register_block_mr<N>`` sentinel names, or None if ``name`` is not one.

    Deliberately strict: the suffix must be ALL DIGITS. ``ensure_perop_block`` mints names in the same
    ``perop_register_block_*`` namespace (``..._3b_128_<hash>``), so a prefix-only test would read a
    RESOLVED feature as an unresolved sentinel and re-derive the table underneath it.
    """
    if not name.startswith(PEROP_MR_SENTINEL_PREFIX):
        return None
    suffix = name[len(PEROP_MR_SENTINEL_PREFIX):]
    if not suffix.isdigit():
        return None
    return int(suffix)


def perop_mr_sentinel(mr_cap: int) -> str:
    """Register (on demand) a per-op blocking request that PINS the MR cap to ``mr_cap``.

    Same contract as :data:`PEROP_BLOCK_NAME` -- a request consumed by
    ``zephyr_model.prepare_for_lowering``, never a lowering edit -- so it raises identically if it
    reaches the pipeline unresolved.

    The cap decides how much A-operand reuse each contraction can buy, and the right value is a
    property of the MODEL's shapes: the default 4 was measured at 128^3, where M is large and MR is a
    pure register-pressure question. On small_llama fp32 every contraction is M=8 with 2.0 MAC per
    byte, so the binding cost is weight TRAFFIC and a cap of 4 makes an M=8 op take two row-blocks,
    streaming the whole B matrix twice (measured K1, n=3, cos 0.9999999 on every arm: cap 4 ->
    5,762,513 ns, cap 8 -> 5,121,007 ns = 1.125x, cap 16 -> 5,153,090 ns, i.e. no further gain).
    Opposite regimes want opposite numbers, which is what makes it a search axis rather than a
    constant -- and as an env var no fork could vary it.
    """
    n = int(mr_cap)
    if n <= 0:
        raise ValueError(f"{PEROP_BLOCK_NAME}: MR cap must be positive, got {mr_cap!r}")
    name = f"{PEROP_MR_SENTINEL_PREFIX}{n}"
    if name in known():
        return name
    register(ImprFeature(
        name=name,
        action_class="PASS",
        description=(f"request PER-CONTRACTION register blocking with the MR cap pinned to {n}. "
                     f"Identical to {PEROP_BLOCK_NAME} except the cap is NAMED rather than read from "
                     f"MERLIN_PEROP_MR_CAP, so the beam can search it: the best cap is a property of "
                     f"the model's shapes (measured 1.125x from 4 -> 8 on an M=8, traffic-bound "
                     f"model; no further gain at 16). A sentinel, resolved by prepare_for_lowering. "
                     f"Default-off; baseline byte-identical."),
        edit_pipeline=_perop_sentinel_unresolved,
        schedule_replace=True,
    ))
    return name


#: The MR caps the SEARCH may name, registered EAGERLY for the same reason ``MRPAD_INT8_TILES`` is:
#: an on-demand-only registration is reachable from the proposer (which mints the name itself inside
#: ``refinement_forks``) but NOT from a ``--features`` list, a package's ``compiler_features``, or any
#: process that has not already called :func:`perop_mr_sentinel`. In those the name resolves to a
#: KeyError, which ``wholemodel_proposer._composes`` swallows as "does not compose" -- so the cap
#: reads as declined rather than as absent. Registration is default-off, so the frozen baseline is
#: unaffected; the ladder brackets the two regimes this repo has measured without asserting either:
#:
#:   cap 1  the EXPERT's MR (XNNPACK ``..._gemm_minmax_ukernel_1x4v__rvv`` is MR=1) on the per-op path
#:   cap 2  the rung between
#:   cap 8  one M-block for an M=8 model, i.e. each weight read ONCE instead of once per row-block
#:   cap 16 the saturation check -- on a model whose gcd(M) is 8 it derives the same table as cap 8
#:
#: 4 is not in the ladder because it is what the plain :data:`PEROP_BLOCK_NAME` sentinel already
#: derives under (``zephyr_model.perop_mr_cap()``); naming it again would register a second spelling
#: of the default and make two identical builds look like two arms.
PEROP_MR_LADDER: tuple[str, ...] = tuple(perop_mr_sentinel(_n) for _n in (1, 2, 8, 16))


def ensure_perop_block(table, kc: int) -> str:
    """Register (on demand) the per-op-blocked schedule for THIS model's block table.

    The schedule text is a function of the table (one tile+vectorize arm per distinct block), so the
    feature is registered per distinct table -- named by a hash of it, the same on-demand pattern
    ``ensure_v3_perop_microkernel`` uses for the register-block knob space.
    """
    import hashlib

    from . import perop_blocks as _pb

    blocks = _pb.distinct_blocks(table)
    key = hashlib.sha1(repr(sorted(table.items())).encode()).hexdigest()[:12]
    name = f"{PEROP_BLOCK_NAME}_{len(blocks)}b_{kc}_{key}"
    if name in known():
        return name
    text = _pb.schedule_text(table, kc)
    register(ImprFeature(
        name=name,
        action_class="PASS",
        description=(f"Per-op register blocking: {len(blocks)} distinct blocks over "
                     f"{len(table)} contraction geometries, KC={kc}. Each contraction is tiled at the "
                     f"widest block legal for its own extents, matched by the merlin.blk_<MR>x<NR> tag "
                     f"the prepare step applies after specialization. Replaces the per-op-CLASS block, "
                     f"whose smallest member otherwise clamps the whole class (measured: whisper_tiny "
                     f"claims 65.9% of its MACs per class vs 100% per op)."),
        edit_pipeline=_accumulator_resident_v3_pipeline,
        edit_schedule=lambda _t, _text=text: _text,
        schedule_replace=True,
        # The tile-epilogue hygiene, keyed on the WIDEST matmul block in the table. This site was
        # missed when `implies` was added to the eight other v3 registration points, and it is the one
        # that mattered most: it is the whole-model per-op path, so every board build went out paying
        # for a per-tile `memref.copy %x, %x` that does nothing. MEASURED on small_llama int8, spike
        # PC histogram over the linked ELF: `memrefCopy` was 28.15% of all retired instructions --
        # more than `forward` itself at 27.09% -- while the scalar-math routines everyone (including
        # this session) had been ranking first were 1.88%, i.e. inside the board's noise band. A static
        # instruction count had put those at 16.63%; the dynamic profile is what corrected it.
        implies=_tile_epilogue_hygiene(max((mr for mr, _nr in table.values()), default=1)),
    ))
    return name


def _register_accumulator_resident_v3() -> list[str]:
    """Register `accumulator_resident_microkernel_v3` (default tile) + a small tuning grid. Reuses
    the v1 PRE-bufferize schedule (forms the contract); the v3 pipeline does the v2 PRE-bufferize
    subset hoist + contract lowering AND splices the SCALARIZE_MARKER so the runner scalarizes the A
    operand -> the K-loop emits the hand kernel's accumulator-resident vfmacc.vf structure."""
    names: list[str] = []
    grid = [(4, 16, 16), (4, 16, 32), (8, 16, 16), (4, 32, 16)]
    for MR, NR, KC in grid:
        if (MR, NR, KC) == (4, 16, 16):
            nm = "accumulator_resident_microkernel_v3"
            desc = ("COMPILER-EMITTED accumulator-resident, register-blocked, vfmacc.vf RVV GEMM "
                    "micro-kernel — the genuine answer to the #1 scalable-gap the transform-only "
                    "v1/v2 features could not reach. Recipe: tile [MR=4,NR=16], tile K by 1, "
                    "scoped-vectorize -> vector.contract; PRE-bufferize loop-invariant-subset-"
                    "hoisting makes the accumulator a vector<MRxNR> scf.for iter_arg (register-"
                    "resident across K); lower contraction -> outerproduct -> vector.fma; then the "
                    "A-operand scalarization rewrite (accum_microkernel.py) replaces the A "
                    "vector<MRx1> read + extract:f32 with per-row scalar loads so the backend emits "
                    "vfmacc.vf (flw) not vfmacc.vv (vmv/vslideup). Emitted K-loop (objdump): ONE B "
                    "vle32 + MR A flw + MR vfmacc.vf into the resident accumulator + C stored once, "
                    "0 in-loop accumulator spills, 0 vfmacc.vv — the hand ceiling's structure, "
                    "compiler-emitted. BIT-EXACT vs scalar ref at 32/64/128 + non-cube on spike "
                    "(scalar load of A[i,0] == lane [i,0]). Default-off, baseline byte-identical.")
        else:
            nm = f"accum_resident_v3_{MR}_{NR}_{KC}"
            desc = (f"Accumulator-resident vfmacc.vf micro-kernel tuning point (MR={MR}, NR={NR}, "
                    f"KC={KC}): resident vector<MRxNR> K-loop iter_arg accumulator + scalar-A "
                    f"vfmacc.vf. Default-off tuning-grid feature.")
        register(ImprFeature(
            name=nm,
            action_class="PASS",
            description=desc,
            edit_pipeline=_accumulator_resident_v3_pipeline,
            edit_schedule=(lambda _t, _MR=MR, _NR=NR, _KC=KC:
                           _accumulator_resident_v3_pre_schedule(_MR, _NR, _KC)),
            schedule_replace=True,
            implies=_tile_epilogue_hygiene(MR),
        ))
        names.append(nm)

    # WHOLE-MODEL-SAFE vfmacc.vf composed variant (this iteration's gap-closer). The
    # `accumulator_resident_wholemodel` feature carries BOTH tail clamps (matmul MR_mm=1, batch_matmul
    # NR_bmm=8) so the resident-accumulator micro-kernel SURVIVES the small-M openvla/rdt2 matmuls
    # (M=17/20/28) and small-N attention — but it rides the v1 POST-bufferize hoist path, which still
    # emits `vfmacc.vv` + a per-K vslideup/vmv broadcast ladder (~20 inner-loop insns/FMA; the
    # measured openvla/rdt2 residual, see output/kernels/ceiling/kernel_breakdown.md). The
    # `accumulator_resident_microkernel_v3` feature emits the clean `vfmacc.vf` (~3 insns/FMA, at the
    # hand ceiling) via the PRE-bufferize subset hoist + A-scalarization — but its bare MR=4 M-tiling
    # trips the LLVM-23 masked-transfer_write PipelineError on small-M (M=17), degrading to
    # NR=8/non-resident/118-spill (the `ours_v3 @ openvla 17×192×576` breakdown row).
    #
    # THIS feature is the merge of the two proven pieces: the v3 PRE-bufferize subset-hoist +
    # contract-lower + A-scalarization (`_accumulator_resident_v3_pipeline`, emits vfmacc.vf) on top of
    # the v3 PRE-bufferize schedule WITH the wholemodel tail clamps inherent (MR_mm=1 + NR_bmm=8). The
    # clamps make MR=min(MR,M)=1 on the matmul path, so the small-M matmul vectorizes FULL (no masked
    # write -> no PipelineError -> no scalar fallback) AT NR=32, accumulator-resident; the A-scalarize
    # then turns the A `vector<MRx1>` lane-rebuild into per-row scalar loads so the K-loop emits
    # `vfmacc.vf` not `vfmacc.vv`. Net: the whole-model kernel's K-loop goes from ~20 ops/FMA
    # (vfmacc.vv + broadcast ladder) to ~3 ops/FMA (flw + vle32 + vfmacc.vf) while KEEPING the NR=32 +
    # residency it already had — closing the dominant share of the openvla/rdt2 gap. Both pieces are
    # proven separately (wholemodel = small-M survival at NR=32; v3 = vfmacc.vf at the hand ceiling);
    # this composes them in ONE schedule. Default-off; baseline byte-identical.
    register(ImprFeature(
        name=WHOLEMODEL_VF_NAME,
        action_class="PASS",
        description="Whole-model-safe vfmacc.vf accumulator-resident micro-kernel: the v3 "
                    "PRE-bufferize subset-hoist + A-scalarization recipe (emits vfmacc.vf, ~3 "
                    "inner-loop insns/FMA, at the hand ceiling) WITH the wholemodel tail clamps "
                    "inherent (matmul MR_mm=1, batch_matmul NR_bmm=8). The clamps make the small-M "
                    "openvla/rdt2 matmuls (M=17/20/28) vectorize FULL at NR=32 (no masked "
                    "transfer_write -> no LLVM-23 PipelineError -> no scalar fallback, unlike bare "
                    "v3 which degrades to NR=8/non-resident), and the A-scalarization turns the A "
                    "lane-rebuild into per-row scalar loads so the K-loop emits vfmacc.vf not "
                    "vfmacc.vv + the ~20-insn vslideup/vmv broadcast ladder. Net: takes the "
                    "whole-model kernel's K-loop from ~20 to ~3 ops/FMA while keeping NR=32 + "
                    "accumulator residency it already had — the openvla/rdt2 gap-closer "
                    "(output/kernels/ceiling/kernel_breakdown.md). Bit-exact (scalar A[i,0] == lane "
                    "[i,0]). Default-off; baseline byte-identical.",
        edit_pipeline=_accumulator_resident_v3_pipeline,
        edit_schedule=lambda _t: _accumulator_resident_v3_pre_schedule(
            *WHOLEMODEL_VF_CAPS, NR_bmm=WHOLEMODEL_VF_NR_BMM, MR_mm=WHOLEMODEL_VF_MR_MM),
        schedule_replace=True,
    ))
    names.append(WHOLEMODEL_VF_NAME)

    # ITERATION-3 (packing/memory residual): MR>1 register-block variant of the vf kernel for
    # A-OPERAND REUSE — the OpenBLAS lever. The memory-traffic decode (output/kernels/ceiling/
    # packing_residual.md) established that the iteration-2 `accumulator_resident_wholemodel_vf`
    # kernel ALREADY ties XNNPACK's inner-loop DATA MOVEMENT at every openvla/rdt2 shape: 1
    # unit-stride B load + 1 scalar A load = 2.0 loads/useful-FMA, unit_stride_only, 0 broadcast
    # ladder (the .vv vslideup/vmv ladder is gone). So vs XNNPACK there is NO per-FMA memory residual
    # left. The ONE remaining data-movement lever is the OpenBLAS MR>1 register block: holding MR
    # output rows in MR accumulator vreg-groups so ONE B-row load is shared across MR FMAs, dropping
    # loads/FMA from 2.0 (MR=1) toward 1+1/MR. This feature is that register block: matmul MR_mm=MR
    # (A-reuse) on the v3 vfmacc.vf path, with the batch_matmul NR_bmm=8 N-tail clamp retained.
    #
    # MEASURED (decode, this iteration): on a LARGE-M, MR-divisible matmul (M=20=5*4, M=64, M=128) the
    # MR=4 kernel emits MR=4 vfmacc.vf sharing 1 unit-stride B load => loads/FMA = 1.25 (1 B-load + 4
    # A-scalars over 4 FMAs), 0 spills, accumulator-resident — the OpenBLAS A-reuse shape, bit-exact.
    #
    # HONEST WHOLE-MODEL SCOPE (the reason this is a SEPARATE feature, not folded into wholemodel_vf):
    # MR>1 needs M >= MR with a CLEAN M-tile. The openvla/rdt2 matmuls are ALL small-M (the token/
    # batch dim: openvla M in {16,17,20}, rdt2 M in {1,28}); they have NO large-M matmul. On those
    # shapes MR=4 either has no clean tile (M=17,1 not divisible by 4 -> the M-tail trips the LLVM-23
    # masked-transfer_write PipelineError -> NR=8/non-resident/118-spill, MEASURED) or silently
    # scalar-falls-back (M=16,28 -> 0 vfmacc emitted, MEASURED). So MR>1 is NOT whole-model-safe for
    # the VLA decode/prefill matmuls and would REGRESS them — it is correct + a genuine A-reuse win
    # only on large-M GEMM (M>=MR, M%MR==0). It is therefore DEFAULT-OFF and intended for a large-M
    # workload; for the small-M openvla/rdt2 whole-model the safe config remains
    # `accumulator_resident_wholemodel_vf` (MR=1, already at XNNPACK's per-FMA traffic floor). The
    # residual A-reuse the VLAs leave on the table is a STRUCTURAL property of their small token dim,
    # not a matmul-kernel defect — closing it would need a dispatch-level layout/batching pass (group
    # multiple small-M matmuls into one large-M GEMM), out of scope for the matmul-kernel feature.
    register(ImprFeature(
        name="accumulator_resident_wholemodel_vf_mr4",
        action_class="PASS",
        description="MR=4 register-block variant of accumulator_resident_wholemodel_vf for A-operand "
                    "REUSE (the OpenBLAS MR>1 lever): matmul MR_mm=4 so ONE unit-stride B-row load is "
                    "shared across 4 vfmacc.vf into 4 resident accumulators, dropping K-loop "
                    "loads/useful-FMA from 2.0 (MR=1) to 1.25 (1 B-load + 4 A-scalars / 4 FMAs) — "
                    "MEASURED by the memory-traffic decode on large-M cube/M=20. batch_matmul NR_bmm=8 "
                    "N-tail clamp retained. CORRECT + bit-exact + A-reuse ONLY on large-M GEMM "
                    "(M>=MR and M%MR==0); on the small-M openvla/rdt2 matmuls (token dim 1-28) it has "
                    "no clean M-tile (M=17,1) -> LLVM-23 masked-write PipelineError, or scalar-falls-"
                    "back (M=16,28) -> would regress the whole model, so it is NOT whole-model-safe "
                    "for VLAs (use wholemodel_vf, already at XNNPACK's per-FMA traffic floor, there). "
                    "The openvla/rdt2 A-reuse residual is structural (small token dim), not a "
                    "matmul-kernel defect; closing it needs a dispatch-level large-M batching/layout "
                    "pass (output/kernels/ceiling/packing_residual.md). Default-off; baseline "
                    "byte-identical.",
        edit_pipeline=_accumulator_resident_v3_pipeline,
        edit_schedule=lambda _t: _accumulator_resident_v3_pre_schedule(4, 16, 16,
                                                                       NR_bmm=8, MR_mm=4),
        schedule_replace=True,
        implies=_tile_epilogue_hygiene(4),
    ))
    names.append("accumulator_resident_wholemodel_vf_mr4")

    # PER-MATMUL MR + M-PAD TAIL — the whole-model-safe MR>1 register block (this iteration).
    # `..._vf_mr4` proved the A-reuse register block (MR=4 -> loads/FMA 2.0->1.25) but is correct ONLY
    # where M%MR==0: on the small-/odd-M matmuls that dominate VLA decode (rdt2 M=1, openvla M=17) its
    # bare MR=4 M-tile trips the LLVM-23 masked-transfer_write PipelineError -> NR=8/non-resident/119
    # spills (M=17) or an outright scalar fallback (M=1) — MEASURED via the decoder. So it is NOT
    # whole-model-safe and would REGRESS the mixed-M models. This feature makes MR>1 whole-model-safe by
    # handling the M-tail the way the hand shim does (`round_up_mr`): PAD each matmul's M up to the next
    # multiple of MR before tiling, so EVERY matmul register-blocks cleanly at MR regardless of M (M=1
    # pads to 4, M=17 to 20, M=64 stays 64). Bit-exact — the padded rows are 0-row@B=0 and are sliced
    # off (transform.structured.pad's extract_slice copy_back), only the real [0:M] rows are written.
    # Per-matmul (each op pads to its own next multiple of MR — a general tail rule), so a model that
    # MIXES M%4==0 and M=1 matmuls (rdt2) gets the MR register block on ALL of them in ONE schedule.
    # The batch_matmul path is identical to `..._vf` (MR=4 + NR_bmm=8), so only the matmul path changes
    # from the MR=1 clamp to the padded MR register block. Default-off; baseline byte-identical.
    register(ImprFeature(
        name="accumulator_resident_wholemodel_vf_mrpad",
        action_class="PASS",
        description="Per-matmul MR>1 register block with an M-PAD tail — the whole-model-safe A-operand "
                    "reuse lever. Same v3 vfmacc.vf subset-hoist + A-scalarization recipe as "
                    "accumulator_resident_wholemodel_vf, but the matmul M tile is a padded MR=4 register "
                    "block instead of the MR=1 clamp: transform.structured.pad rounds each matmul's M up "
                    "to a multiple of MR (padding value 0) BEFORE tiling, so every matmul — INCLUDING "
                    "the M=1/M=17/M=28 VLA-decode matmuls that make bare vf_mr4 trip the LLVM-23 "
                    "masked-transfer_write PipelineError (M=1 scalar fallback, M=17 NR=8/119-spill) — "
                    "register-blocks cleanly at MR: ONE unit-stride B-row load shared across MR "
                    "vfmacc.vf into MR resident accumulators (loads/useful-FMA 2.0 -> 1.25). Bit-exact "
                    "(padded rows are 0-row@B=0, sliced off by pad's copy_back extract_slice; only real "
                    "[0:M] rows written). Per-matmul (each op pads to its own next MR multiple — general "
                    "tail rule, not a per-model constant), so a MIXED-M model (rdt2 M in {1,28}) gets "
                    "the MR block on every matmul in ONE schedule. batch_matmul path identical to _vf "
                    "(MR=4 + NR_bmm=8). Default-off; baseline byte-identical.",
        edit_pipeline=_accumulator_resident_v3_mrpad_pipeline,
        implies=_tile_epilogue_hygiene(4),
        edit_schedule=lambda _t: _accumulator_resident_v3_mrpad_pre_schedule(4, 16, 16, NR_bmm=8),
        schedule_replace=True,
    ))
    names.append("accumulator_resident_wholemodel_vf_mrpad")
    return names


ACCUM_RESIDENT_V3_NAMES: list[str] = _register_accumulator_resident_v3()

MRPAD_NAME = "accumulator_resident_wholemodel_vf_mrpad"


def ensure_mrpad_for_elem_types(a: str, b: str, c: str, MR: int = 4, NR: int = 16,
                                NR_bmm: int = 8) -> str:
    """Register (once) an M-pad register block whose padding values match the OPERAND TYPES.

    The default ``accumulator_resident_wholemodel_vf_mrpad`` pads with an f32 zero for all three
    operands, which is right only for an f32 contraction. On any other element type
    ``transform.structured.pad`` hard-errors, the transform interpreter raises, and the lowering
    catches that as a whole-model scalar fallback -- so the lever reads as "applied but slow" rather
    than "rejected". Naming the types makes the variant selectable by the search instead of being a
    property of whichever dtype the default happened to be written for.

    Returns the feature name to put in a feature set.
    """
    types = (str(a), str(b), str(c))
    MR, NR, NR_bmm = int(MR), int(NR), int(NR_bmm)
    if min(MR, NR, NR_bmm) <= 0:
        raise ValueError(f"MR/NR/NR_bmm must be positive, got {(MR, NR, NR_bmm)}")
    # The TILE is part of the identity. MR=4/NR=16 were chosen against f32; at one VLEN an i8 lane
    # holds four times the elements, so the f32 tile is not the int8 tile and a variant that hides
    # its shape behind the dtype name would silently pin the wrong one.
    name = f"{MRPAD_NAME}_" + "_".join(types) + f"_mr{MR}_nr{NR}_nb{NR_bmm}"
    if name in known():
        return name
    for t in types:                    # fail closed at REGISTRATION, not inside the interpreter
        _zero_attr(t)
    register(ImprFeature(
        name=name, action_class="PASS",
        description=(f"{MRPAD_NAME} at MR={MR}/NR={NR}/NR_bmm={NR_bmm} with padding values typed "
                     f"for a {types[0]}x{types[1]}->{types[2]} "
                     "contraction. Same recipe and same tail rule; only the pad literals differ, "
                     "because a wrongly-typed pad value is a transform-interpreter error and "
                     "therefore a silent whole-model scalar fallback."),
        edit_pipeline=_accumulator_resident_v3_mrpad_pipeline,
        implies=_tile_epilogue_hygiene(4),
        edit_schedule=(lambda _t, _ty=types, _mr=MR, _nr=NR, _nb=NR_bmm:
                       _accumulator_resident_v3_mrpad_pre_schedule(_mr, _nr, 16, NR_bmm=_nb,
                                                                   elem_types=_ty)),
        schedule_replace=True,
    ))
    return name

#: The canonical int8 contraction operand types. Registered eagerly so the typed M-pad variant is
#: NAMEABLE as a feature string and visible to the proposer -- an on-demand-only registration is
#: reachable from Python but not from a `--features` list or a search proposal, which is where it
#: has to be selectable from. Registration is default-off, so the frozen baseline is unaffected.
MRPAD_INT8_NAME = ensure_mrpad_for_elem_types("i8", "i8", "i32")

#: Tiles the SEARCH may try for the int8 named-op register block. MR=4/NR=16 is the f32 tile and it
#: is 5.1x behind the generic-level block on int8 -- an untuned shape, not a broken lever: against
#: its own control (no register block at all) it is 8.6x faster. An i8 lane holds 4x the elements of
#: an f32 lane at one VLEN, so the N strip in particular has no reason to match. Registered so the
#: proposer can refine along both axes instead of inheriting a constant chosen for another dtype.
#:
#: MR=1 IS IN THE LADDER BECAUSE THE EXPERT USES IT. XNNPACK's kernel is
#: `xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4v__rvv` -- 1x4v, i.e. MR=1 with a VLEN-scaled 4-group N
#: tile -- and the lifted CCA agrees: `compute.register_block expert=(1, ('vsetvlmax', 4.0))` against
#: our (4, ('vsetvlmax', 8.0)). The ladder started at MR=2 and the routed action's own text says
#: "RAISE the matmul register-block MR toward the expert MR", which is backwards here: the expert's
#: MR is BELOW ours, and its number was not in the search space at all. Our MR=4 variant measured
#: 1.61x SLOWER than the default, which is what that looks like from the outside.
MRPAD_INT8_TILES: tuple[str, ...] = tuple(
    ensure_mrpad_for_elem_types("i8", "i8", "i32", MR=_mr, NR=_nr, NR_bmm=_nb)
    for _mr, _nr, _nb in ((1, 16, 8), (1, 32, 8), (1, 64, 16),      # the EXPERT's MR
                          (2, 16, 8), (4, 16, 8), (8, 16, 8),
                          (4, 32, 8), (4, 64, 16), (8, 32, 16), (8, 64, 16)))



# ---- compiler-emitted register-blocked RVV intrinsic micro-kernel -------------------
# THE scalable-gap winner (output/kernels/ceiling/scalable_gap_result.md). The upstream
# tile->vectorize->bufferize lowering re-reads/re-writes the MR x NR accumulator THROUGH
# MEMORY every K-tile (a `memref.copy` inside the K loop that neither
# `hoist_redundant_vector_transfers` nor `loop-invariant-subset-hoisting` could lift, because
# bufferization recomputes the accumulator subview + a self-copy inside the loop). That
# operand-copy traffic — NOT the vfmacc chain (the compute kernel alone is ~56K instret @64^3,
# already ~1.5x OpenBLAS) — is the 15.7x scalable gap. The alternative the experts use, and that
# a dedicated RVV micro-kernel codegen pass would emit, is a register-blocked,
# accumulator-resident, K-streaming inner kernel: the MR x (NR vreg-group) accumulator lives in
# vector registers for the WHOLE K loop, only A scalars + a B row load per K-step (vfmacc.vf),
# and C stores once. Realized as a COMPILER-EMITTED riscv_vector.h intrinsic kernel
# (merlin/python/merlin/kernels/ceiling_drivers/ours_intrinsic_gemm_driver.c), scalable VLEN via
# vsetvl, MR=4 register block. MEASURED spike inner-compute (pack-excluded, resident-weight,
# head-to-head with OpenBLAS's hoisted pack), bit-exact at 32/64/128, bounded code, ZERO vector
# spills in the inner loop:
#     shape   OpenBLAS    ours-intrinsic   ratio
#     32^3     11,039        6,551         0.59x (ours 1.7x FASTER)
#     64^3     84,483       50,695         0.60x (ours 1.7x FASTER)
#     128^3   664,811      399,241        0.60x (ours 1.7x FASTER)
# This feature is a marker (no MLIR schedule/pipeline edit) recording that the gap-closing path is
# a dedicated RVV inner-kernel emitter, not the outerproduct lowering; the measured driver IS the
# emitter's output. Default-off; baseline byte-identical (it has no edit hooks).
register(ImprFeature(
    name="intrinsic_microkernel",
    action_class="CODEGEN",
    # HONEST LABEL: this is a CEILING REFERENCE, not a compiler-emitted feature. It is a marker with
    # NO MLIR schedule/pipeline edit (baseline byte-identical); the measured number comes from a
    # HAND-WRITTEN riscv_vector.h driver (ceiling_drivers/ours_intrinsic_gemm_driver.c), NOT from our
    # transform pipeline. It records the TARGET a dedicated RVV micro-kernel codegen pass should hit,
    # and quantifies how far the compiler-emitted accumulator_resident_microkernel still is from it.
    # The transform-dialect feature does NOT yet reach this (see accumulator_resident_microkernel:
    # the emitted asm still spills the carried accumulator through the stack inside the K loop —
    # vl4re8.v/vs4r.v of the accumulator per K-tile, so the CCA reads accumulator_resident=False, and
    # measured ~19x off this ceiling @64^3). Keeping the hand kernel ONLY as a labeled ceiling so the
    # gap is honest and the codegen work-item (action_catalog: compute.accumulator_resident ->
    # CODEGEN, forkable_now=False) is visible — never linked as if the compiler emitted it.
    description="CEILING REFERENCE (hand-written riscv_vector.h driver, NOT compiler-emitted): a "
                "register-blocked, accumulator-resident, K-streaming RVV GEMM micro-kernel (MR=4, "
                "NR=vsetvlmax) that keeps the MR x NR accumulator in vector registers across the "
                "whole K loop (vfmacc.vf chain, B row + A scalars streamed, C stored once). "
                "Spill-free, bit-exact 32/64/128, 1.7x faster than OpenBLAS on the spike proxy "
                "(pack-excluded). It is the TARGET for a dedicated RVV micro-kernel codegen pass; "
                "the transform-dialect accumulator_resident_microkernel does NOT yet reach it "
                "(~19x off @64^3 — still spills the accumulator per K-tile). Marker only (no "
                "schedule/pipeline edit); baseline byte-identical.",
    edit_pipeline=None,
    edit_schedule=None,
))


# ---- vectorized transcendental activation (GELU/sigmoid/SiLU/tanh) ------------------
# THE coverage gap from output/kernels/ceiling/cross_framework_ops_k1.md: our GELU (math.erf) and
# sigmoid/SiLU (math.exp) activations lower through convert-math-to-libm to a SCALAR libm call
# (erff/expf) in a loop, and the baseline transform schedule never vectorizes the elementwise
# activation generic — so the activation runs scalar and is ~11-18x behind XNNPACK's vectorized
# polynomial RVV kernels (f32-vgelu rational-12-10, f32-vsigmoid rr2-p5).
#
# This feature is the GENERAL compiler answer, in TWO composed parts (both default-off):
#   1. A math.exp/erf/tanh -> inline arith-POLYNOMIAL rewrite (act_poly.py), spliced into the
#      lowering runner BEFORE the pass manager (pipeline._activation_poly_runner, keyed on this
#      feature name). It replaces the transcendental with a minimax polynomial of mul/add/sub/div/
#      bitcast/shift ops — NOT a libm call. General over the math ops, so GELU (erf), sigmoid/SiLU
#      (exp via 1/(1+exp(-x)) / x*sigmoid(x)) and tanh (2/(1+exp(-2x))-1) all get it from ONE rewrite.
#   2. This schedule: the baseline matmul/batch_matmul vectorization PLUS vectorizing the
#      elementwise activation linalg.generic ([16] tile) — so the polynomial (now in the generic
#      body) vectorizes to vfmacc chains (the mul+add Horner pairs fuse) instead of a scalar loop.
#
# Accuracy is an APPROXIMATION (f32 minimax, ceiling-referenced to XNNPACK's exp rr2-p5 + the A&S
# 7.1.26 erf): cos=1.0 / max-abs-err <~1e-6 vs libm across gelu/sigmoid/silu/tanh — gated on cos/rel
# error, NOT bit-exact (the explicit activation accuracy tradeoff). Default-off; baseline
# byte-identical (the runner uses the plain _RUNNER and the schedule is unchanged when off).
_ACT_POLY_SCHEDULE = """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t, %l:3 = transform.structured.tile_using_for %mm tile_sizes [4, 8, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %t vector_sizes [4, 8, 1] : !transform.any_op
    %bm = transform.structured.match ops{["linalg.batch_matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt, %bl:4 = transform.structured.tile_using_for %bm tile_sizes [1, 4, 8, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %bt vector_sizes [1, 4, 8, 1] : !transform.any_op
    // Vectorize PRECISELY the activation generics — and ONLY those — the poly rewriter targeted.
    // The rewriter (act_poly.apply_activation_polynomial) replaces math.erf/exp/tanh with an arith
    // mul/add/fma chain ONLY inside a generic the PROVENANCE marks as an elementwise activation
    // (gelu/silu/sigmoid/tanh), and tags each such generic with a `merlin.act_vectorize` unit attr.
    // We match exactly those tagged generics and vectorize them (bare vectorize = rank-agnostic,
    // sizes inferred from the static iteration space) so the poly chain becomes a vector.fma/vfmacc
    // chain. A softmax-exp / normalization / any other generic is NOT tagged (its exp stayed on
    // libm), so it is NOT matched here and lowers exactly as in the baseline.
    //
    // This drops the previous BLANKET design (foreach over EVERY linalg.generic + failures(suppress)
    // + a hard-coded rank-1 tile). That design (a) rewrote+vectorized the softmax exp too, which
    // amplified the minimax error through the row-sum normalization -> openvla whole-model cos 0.541;
    // (b) used failures(suppress), which HID any vectorize miscompile; and (c) vectorized every
    // generic in the model, blowing clang -O2 compile time to 6+ min/config. Matching only the tagged
    // activation generics fixes all three: correct (softmax stays libm), no suppression (a genuine
    // vectorize failure surfaces), and bounded compile time (only the few activation generics
    // vectorize). No `failures(suppress)` — these tagged generics are static-shape elementwise and
    // always statically vectorize; if one ever didn't, we WANT the error, not a silent scalar fallback.
    %eg = transform.structured.match ops{["linalg.generic"]} attributes{"merlin.act_vectorize"} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.foreach %eg : !transform.any_op {
    ^bb_eg(%one_eg: !transform.any_op):
      transform.structured.vectorize %one_eg : !transform.any_op
    }
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_contraction
      transform.apply_patterns.vector.lower_masked_transfers
      transform.apply_patterns.vector.lower_transpose
      transform.apply_patterns.vector.lower_shape_cast
    } : !transform.any_op
    transform.yield
  }
}
"""


# NOTE: the feature no longer edits the pass list. The earlier `_act_poly_math_before_libm` inserted
# `convert-math-to-llvm` BEFORE `convert-math-to-libm` because the OLD poly emitted vector
# `math.fma`/`math.absf`/`math.roundeven` that libm would scalarize. But that same pass ALSO converted
# the (correctly un-rewritten) softmax `math.exp` to `llvm.intr.exp` (`llvm.exp.f32`), which the
# freestanding spike/RVV runtime cannot legalize -> a wild instruction -> the openvla whole-model
# "bad syscall" CRASH (run produced no OUT/DONE -> certify status=fail). The activation poly is now
# PURE ARITH (act_poly._ap_fma = arith mul+add, _ap_absf = max(x,-x), _ap_roundeven = add/sub-magic),
# so it carries NO math.* op; the only remaining transcendental is the softmax exp, which takes the
# baseline `convert-math-to-libm` -> scalar `expf` path (exact, crash-free). No pipeline edit needed.


register(ImprFeature(
    name="vectorized_transcendental_activation",
    action_class="PASS",
    description="GENERAL vectorized-activation lowering, PRECISELY TARGETED by provenance: the "
                "act_poly rewriter (spliced into the lowering runner before the pass manager) "
                "replaces math.erf/exp/tanh with an inline minimax ARITH polynomial ONLY inside a "
                "linalg.generic the provenance marks as an elementwise ACTIVATION (gelu/silu/sigmoid/"
                "tanh) — NOT a softmax/normalization (whose exp stays on the exact libm path; "
                "blanket-rewriting it drove openvla whole-model cos to 0.541). It TAGS each targeted "
                "generic (merlin.act_vectorize) and the schedule vectorizes exactly those (no blanket "
                "foreach over every generic, no failures(suppress) -> no masked miscompile, no "
                "6+min/config compile blowup). So GELU (erf) and sigmoid/SiLU (exp) vectorize to a "
                "vector fmul/fadd (vfmacc) chain while softmax stays correct. Closes the activation "
                "gap vs XNNPACK's polynomial RVV kernels (coefficient/structure CEILING REFERENCE; we "
                "emit the MLIR). APPROXIMATION: cos>0.999 / max-abs-err <~6e-7 vs libm on REALISTIC "
                "ranges (gated on cos/rel error, not bit-exact). Default-off; baseline byte-identical.",
    edit_schedule=lambda _t: _ACT_POLY_SCHEDULE,
    schedule_replace=True,
))


register(ImprFeature(
    name="fused_vfmacc_contraction",
    action_class="PASS",
    description="mined fma_broadcast_policy: form a real vector.contract -> outerproduct(kind=add) "
                "-> vector.fma -> llvm.fmuladd -> vfmacc (vectorize_children + lower_contraction "
                "outerproduct + lower_outerproduct). Closes the separate-vfmul.vv+vfadd.vv gap. For "
                "kernel-sized contraction workloads (vectorize_children explodes on whole models).",
    edit_schedule=_vfmacc_schedule_edit,
    schedule_replace=True,
))


# Erase the per-tile `memref.copy %x, %x` bufferization leaves behind. Pure lowering hygiene: it
# changes no schedule, only removes a runtime call that moves a buffer onto itself. Default-off so
# the frozen hand_v0 control stays byte-identical; the beam turns it on to close an
# `envelope.runtime_calls` divergence (see kernels/action_catalog.py).
#
# MEASURED, K1 f32 GEMM 128^3, kernel region, bit-exact:
#   instructions 1,710,650 -> 475,899 (3.59x)   ticks 41,195 -> 21,882 (1.88x)
#   vs XNNPACK   3.57x -> 1.90x
# The mechanism and why nothing upstream folds it: see llvmlower/selfcopy.py.
register(ImprFeature(
    name=_SELF_COPY_FEATURE,
    action_class="PASS",
    description="erase `memref.copy %x, %x` (a buffer copied onto itself) after bufferization/cse "
                "and before finalize-memref-to-llvm, where it would otherwise survive as an opaque "
                "@memrefCopy rank-generic runtime call costing ~79 retired instructions per OUTPUT "
                "ELEMENT. Removes the tile-epilogue copy; emits no new code.",
))


# Expand every static `memref.copy` into an emitted loop nest instead of a runtime call. The
# companion to `erase_self_copy`, and the reason that one alone cannot close the axis: the erase
# only removes copies that are REDUNDANT (`memref.copy %x, %x`), while a copy whose destination is a
# `memref.subview` of a larger buffer moves real data and survives as `@memrefCopy`, MLIR's
# rank-generic strided walker (~79 retired instructions per copied ELEMENT).
#
# MEASURED at the post-bufferization split point, small_llama_int8_consistent, hand_v0_int8:
#   self       19 copies (all in-loop),   608 elements  -> erase_self_copy removes these
#   diff-type  24 copies (prologue),     6144 elements  -> @memrefCopy, ~485K instructions/inference
#   same-type  40 copies (prologue),    21360 elements  -> memcpy
# So `envelope.runtime_calls` keeps both `memrefCopy` and `memcpy` however many self-copies are
# erased. Rewriting the copy to `linalg.copy` hands it to the `convert-linalg-to-loops` already in
# every pipeline, and finalize-memref-to-llvm then has nothing left to turn into a call.
# See llvmlower/copy_expand.py for the structural predicate and the fail-closed skip count.
register(ImprFeature(
    name=_EXPAND_COPY_FEATURE,
    action_class="PASS",
    description="rewrite every ranked, statically shaped `memref.copy` to a `linalg.copy` after "
                "bufferization and before finalize-memref-to-llvm, so the pipeline's own "
                "convert-linalg-to-loops emits an scf load/store nest instead of leaving a call to "
                "the rank-generic `@memrefCopy` runtime helper (or a copy-derived `memcpy`). "
                "Structure-keyed (ranked + static shape), no shape or model assumption; a copy it "
                "cannot prove static is left alone and counted. Default-off, baseline "
                "byte-identical.",
))


# Fold a `linalg.transpose` of a matmul's B operand INTO the matmul's access pattern (transpose-b
# GEMM). WHOLE-MODEL, cross-op: the whole-model profiler measured `linalg.transpose` at 393 ms = 57%
# of openvla -- more than every matmul combined -- and SCALAR (convert-linalg-to-loops). Every
# openvla matmul is a transposed-B addmm fed by a standalone weight transpose. Folding it eliminates
# the scalar transpose op AND its materialized buffer; the op stays `linalg.matmul` (transposed-B
# indexing_map) so the frozen RVV schedule still vectorizes it. Default-off; baseline byte-identical.
# The rewrite runs in the lowering runner (gated by argv[5]); see llvmlower/transpose_fuse.py.
register(ImprFeature(
    name=_FUSE_TRANSPOSE_FEATURE,
    action_class="PASS",
    description="fuse `matmul(A, transpose(B, [1,0]))` into a transpose-b `linalg.matmul` (repoint "
                "the B operand to the un-transposed weight + permute its indexing_map (k,n)->(n,k), "
                "then erase the dead transpose). Eliminates the standalone SCALAR weight transpose "
                "(393 ms / 57% of openvla) and its DRAM buffer with no op materialized; the matmul "
                "stays vectorized by the frozen schedule and reads B contiguously along k. "
                "Whole-model cross-op fusion; default-off, baseline byte-identical.",
))


# ---- vectorize a standalone reduction -> vfredusum/vredsum (the compute.reduction_form lever) ------
# The baseline RVV schedule vectorizes ONLY the contraction ops (linalg.matmul / batch_matmul); a
# standalone reduction (softmax max/sum, LayerNorm/RMSNorm mean/var, a `linalg.reduce`) is left as a
# `linalg.generic` with a `reduction` iterator and falls through `convert-linalg-to-loops` to a SCALAR
# accumulate loop. That is the 2nd-biggest byte-traffic op family (softmax ~3.85% of the census) going
# unvectorized -- and the `compute.reduction_form` CCA lever had NO route (a bijection orphan), so the
# beam could not act on it. This feature is that route.
#
# It vectorizes the reduction generic (matched by its `reduction` iterator, at the ranks that occur in
# practice -- pure reduce, softmax/norm row-reduce, batched 3-D) and lets the reduction lower to a
# HARDWARE vector reduction. Two pipeline knobs make that land as `vfredusum`/`vredsum` rather than a
# scalar tail or a lane-parallel add tree:
#   * `lower-vector-multi-reduction{lowering-strategy=inner-reduction}` lowers the
#     `vector.multi_reduction` (that `transform.structured.vectorize` produces) to a `vector.reduction`
#     (the single-instruction horizontal reduce) instead of the default inner-PARALLEL add tree.
#   * `convert-vector-to-llvm{reassociate-fp-reductions}` sets `reassoc` on the emitted
#     `llvm.intr.vector.reduce.fadd`, so the RISC-V backend selects the UNORDERED `vfredusum.vs`
#     (detectable as a real reduction by cca.lift_asm) instead of the ordered `vfredosum.vs`. Integer
#     reductions emit `vredsum.vs` directly (integer add is associative).
#
# The contraction blocks of the baseline schedule are LEFT INTACT (this edit only INSERTS the reduction
# match+vectorize before the func-level lowering-pattern block), so enabling the feature on a whole
# model does NOT regress the matmuls to scalar -- MEASURED: the emitted matmul object is byte-identical
# with and without this feature (65 vec ops, 4 vfmul, 4 vfadd, 0 vfmacc either way). Empty matches are
# harmless (vectorize on an empty handle is a no-op), so a reduction rank the model does not contain
# just does nothing.
#
# PROVEN on the EMITTED CODE (lower a gen_reduce_f32 / gen_softmax_f32 workload with the real
# RVV_CFLAGS `-fno-vectorize -fno-slp-vectorize`, so clang's own auto-vectorizer is OFF and every
# vector reduction is MLIR-emitted, then decode the objdump):
#   reduce_f32 64x256 :  vfredusum.vs x64 (baseline: 0 vector ops)   cca.reduction_form=vredsum_tree
#   softmax_f32 64x256:  vfredmax.vs x64 + vfredusum.vs x64          cca.reduction_form=vredsum_tree
# APPROXIMATION (not bit-exact): `reassociate-fp-reductions` reorders the fp sum, the standard tree /
# unordered reduction XNNPACK et al. also use -- gated on cos/rel error, never claimed bit-exact.
# Default-off; empty features => pipeline + schedule byte-identical (guarded by test_impr_features).

# The reduction match+vectorize block, inserted before the baseline schedule's func.func lowering-
# pattern block. One match per reduction rank we support (pure-reduction, [parallel, reduction],
# [parallel, parallel, reduction]); an arity absent from the module matches nothing and is skipped.
_VECTORIZE_REDUCTION_BLOCK = """\
    %redr1 = transform.structured.match ops{["linalg.generic"]} attributes{iterator_types = [#linalg.iterator_type<reduction>]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %redr1 : !transform.any_op
    %redr2 = transform.structured.match ops{["linalg.generic"]} attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %redr2 : !transform.any_op
    %redr3 = transform.structured.match ops{["linalg.generic"]} attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %redr3 : !transform.any_op
"""

# Anchor: the baseline schedule's func-level lowering-pattern match. Insert the reduction block just
# before it so the lowering patterns (lower_masked_transfers/transpose/shape_cast) clean up afterward.
_REDUCTION_ANCHOR = '    %f = transform.structured.match ops{["func.func"]}'

# A standalone reduction-only schedule, used when the input schedule has no func.func anchor to build
# on (so the feature is never a silent no-op — it always emits the reduction vectorization).
_VECTORIZE_REDUCTION_STANDALONE = f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
{_VECTORIZE_REDUCTION_BLOCK}\
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.lower_masked_transfers
      transform.apply_patterns.vector.lower_shape_cast
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


def vectorize_reduction_schedule(text: str) -> str:
    """Insert the reduction match+vectorize block into the baseline schedule (before its func.func
    lowering-pattern block), keeping the contraction handling intact. Falls back to a reduction-only
    schedule if the anchor is absent (never a silent no-op)."""
    if _REDUCTION_ANCHOR in text:
        return text.replace(_REDUCTION_ANCHOR,
                            _VECTORIZE_REDUCTION_BLOCK + _REDUCTION_ANCHOR, 1)
    return _VECTORIZE_REDUCTION_STANDALONE


def vectorize_reduction_pipeline(passes: list[str]) -> list[str]:
    """Make the emitted reduction a single-instruction horizontal vector reduce (`vfredusum`/`vredsum`):
    switch `lower-vector-multi-reduction` to the inner-reduction strategy (-> `vector.reduction`) and
    enable `reassociate-fp-reductions` on `convert-vector-to-llvm` (-> unordered `vfredusum.vs`). Only
    runs when this feature is enabled; the baseline pass list is untouched otherwise."""
    out: list[str] = []
    for p in passes:
        if p == "func.func(lower-vector-multi-reduction)":
            out.append("func.func(lower-vector-multi-reduction{lowering-strategy=inner-reduction})")
        elif p == "convert-vector-to-llvm":
            out.append("convert-vector-to-llvm{reassociate-fp-reductions}")
        else:
            out.append(p)
    return out


register(ImprFeature(
    name="vectorize_reduction",
    action_class="PASS",
    description="Vectorize a standalone reduction (softmax/norm row-reduce, `linalg.reduce`) so it "
                "lowers to a HARDWARE vector reduction (`vfredusum.vs` for fp / `vredsum.vs` for int) "
                "instead of the scalar convert-linalg-to-loops accumulate the baseline emits. Matches "
                "the reduction `linalg.generic` by its reduction iterator (ranks 1/2/3), vectorizes it "
                "to `vector.multi_reduction`, lowers that via the inner-reduction strategy to "
                "`vector.reduction`, and reassociates the fp reduce so the backend picks the unordered "
                "vfredusum. The contraction schedule is left intact (matmul object byte-identical), so "
                "it is whole-model-safe. The route for the compute.reduction_form CCA lever (previously "
                "a bijection orphan). APPROXIMATION (fp reassociation, cos-gated). Default-off; baseline "
                "byte-identical.",
    edit_pipeline=vectorize_reduction_pipeline,
    edit_schedule=vectorize_reduction_schedule,
    schedule_replace=True,
))


# ---- matrix-unit routing ------------------------------------------------------------
# Unlike every feature above, this one changes NEITHER the pipeline pass list NOR the transform
# schedule. It is an IR-level rewrite: `passes_opu.rewrite_prepared_file` replaces selected int8
# contractions with calls to the certified matrix microkernel, so those contractions leave the vector
# path entirely and the schedule's `linalg.matmul` arms simply match fewer ops.
#
# It is registered here anyway, with both hooks None, because this registry is the one place a build
# says which non-baseline compiler behaviour it wants -- `build_app` threads `compiler_features` through
# to the prepare step, and a feature that lived outside the registry would need a second, parallel way
# to be requested. Both hooks being None also means the byte-identity invariant is structural rather
# than merely tested: there is no edit to apply.
OPU_MATMUL_NAME = "opu_matmul"

register(ImprFeature(
    name=OPU_MATMUL_NAME,
    action_class="PASS",
    description=(
        "Route int8 rank-2 contractions with a zero accumulator init to the certified outer-product "
        "matrix microkernel, as calls to a generated translation unit that transposes the left operand "
        "K-major and reads its extents from the memref descriptors. NOT a schedule or pipeline edit: "
        "the rewrite happens on the prepared IR (llvmlower/passes_opu), which is why both hooks are "
        "None. Selection is a separate decision -- which contractions move is answered by the cost "
        "model / e-graph and passed in, so enabling this feature without a selector routes nothing. "
        "Coverage on spectformer int8: 90 of 106 contractions are legal (the 16 batch_matmuls are "
        "gapped by a matmul-only contract), and a tile-filling selector at edge 32 moves 41 of them, "
        # target-ok: names the hardware_pins.yaml entry this feature requires at build time — a pin
        # reference in prose, not a target this code routes on (selection is passed in, see above).
        "which is the shapes carrying ~88% of the arithmetic. Requires the pinned saturn revision "
        "carrying the unit (hardware_pins.yaml: saturn_opu_int8) at build time, because the "
        "instruction encodings are derived from its RTL rather than written down. Default-off."),
))


# ---- post-contraction elementwise fusion --------------------------------------------
# The stage this feature turns on ALREADY EXISTS in `pipeline.build_rvv_pipeline`, sitting directly
# above `linalg-generalize-named-ops` with a paragraph of measurement explaining why it is the
# load-bearing one. It was reachable only through the `MERLIN_FUSE_POST` environment variable -- and
# no fork can vary an environment variable. That is the same gap `vectorize_non_contraction_generics`
# was built to close for `MERLIN_VEC_RANK`: a lever the tuning loop cannot name is a lever the tuning
# loop cannot find, however well the pipeline supports it. Naming it here makes it selectable; the env
# var still forces it on for a manual A/B, and the splice below declines to double it when it does.
#
# WHAT IT IS FOR, measured on `small_llama_int8_consistent` at the config that is currently best
# (`prepack_weight_layout, perop_register_block, promote_buffers_to_stack, expand_memref_copy,
# cse_through_provenance`). A per-op board profile of that model puts `linalg.generic` at 16.5% of
# whole-model runtime across 191 ops in 26 structural classes -- a long tail with no dominant member,
# so there is nothing to special-case even if one wanted to. Re-priced with `cse_through_provenance`
# ON (which collapses the rotary cos/sin/powf duplicates 8:1 and takes the module 163 -> 112
# generics), what is LEFT at the top of the tail is the dynamic per-row activation quantization --
# an `absf`/`maximumf` amax scan, a scale divide, and a `divf`/`roundeven`/`min`/`max`/`fptosi`
# quantize pass -- together with the `sitofp`/`mulf`/`mulf` requant that `cse` does not touch at all.
#
# Read the IR of one of those and the cost is not the arithmetic. `linalg-specialize-generic-ops`
# (which runs just above, to recover the contraction NAMES the schedule matches on) un-fuses every
# elementwise chain, so each per-row scale is materialized into a full-size temporary before the op
# that divides by it:
#
#     %scale = ... -> tensor<8xf32>
#     %b = linalg.broadcast ins(%scale : tensor<8xf32>) outs(... : tensor<8x128xf32>) dimensions = [1]
#     %q = linalg.generic ins(%act, %b) { divf; roundeven; min; max; fptosi } -> tensor<8x128xi8>
#
# Counted over the whole prepared module: 50 `linalg.broadcast`, reading 17,476 bytes and WRITING
# 242,944 -- a 13.9x amplification for zero arithmetic, on a model whose weights are 602 KB. Running
# the stage takes that to 13 broadcasts, 3,012 bytes read and 60,160 written: 182,784 bytes per
# inference that are no longer materialized, and the consumer reads the small operand through its own
# indexing map instead.
#
# ON THE EMITTED CODE (K1 rv64gcv cross-compile, same package, same feature set, the ONLY difference
# being this stage; `rvv_audit.audit_binary` -> `compute_symbol()` = `forward`):
#
#     forward instructions      35,253 -> 31,348   (-11.1%)
#       vector                  14,053 -> 12,561
#       scalar compute          15,319 -> 13,802
#       vsetvl                   1,118 ->    954
#     forward vector fraction   0.4784 -> 0.4765   (flat -- see below)
#     model.o                  189,008 -> 170,664 bytes
#     stack alloca sites           118 ->    103   (344,640 -> 318,848 bytes)
#     model.ll digest       1cd391e1cae90ae3 -> d5868600a852e88f
#
# The vector fraction is FLAT on purpose, and reporting it as a win would be wrong: the broadcast
# loops were themselves partly vectorized, so deleting them removes vector and scalar instructions in
# roughly equal proportion. The claim is that ~11% of the instructions in `forward` were spent
# materializing values their consumer could read directly -- not that what remains is better
# vectorized.
#
# NUMERICS: bit-identical. On spike (bare-metal whole-model image, delivery configuration, VLEN=256),
# the output prefix digest is `cc60e8a90270ec1e` with and without the stage, and every gate figure
# agrees to the last digit against the independent references (fp32 cos 0.9999725818634033 / rel
# 0.007327893200253357 / argmax True; w8a8 cos 0.9999655485153198 / rel 0.008363386664235835;
# tiers ['fp32','w8a8'], tier_ok 'fp32_cos_only', ok True).
#
# THE WALL IS UNMEASURED. Everything above is static, and static evidence has been wrong here twice
# in one day: `fold_weight_transpose` removed ops and shrank the object and cost 1.09x, and a
# five-lever stack went 1.815x -> 1.943x. Fewer instructions is a reason to MEASURE this, not a
# result. It is default-off like every other lever, and it is ranked in the whole-model proposer so
# the beam prices it on the board instead of anyone assuming the sign.
#
# NOT `fuse_elementwise_after_generalize`, WHICH IS REFUTED -- read this before assuming the two are
# the same lever with different names. That one inserts a bare `linalg-fuse-elementwise-ops` AFTER
# `linalg-generalize-named-ops`, and it MEASURED 1.22x SLOWER on this same model (3,543,517 ->
# 4,313,041 ns): its scalar bucket fell 3.1 -> 2.8 ms but the CONTRACTION rose 1.8 -> 2.8 ms, because
# by then every named op is generic, the dequant chain fuses into the producers, and that perturbs the
# shape the transform schedule had already vectorized the contraction into. Its own note concludes the
# broadcast waste is real but needs something that does not disturb the contraction.
#
# This feature runs the stage on the OTHER SIDE of that anchor -- before generalization, while the
# contraction's neighbours are still named ops the blanket fusion cannot swallow -- and adds the
# `canonicalize`/`cse` the refuted one omits. That is not a cosmetic difference, and the emitted code
# says so. Decoded from both objects with `rvv_audit._insn_mnemonic`:
#
#     vwmacc     152 -> 152        the int8 contractions, byte for byte as vectorized as before
#     vmacc        4 ->   4
#     vredmax    104 -> 104        the amax reductions, likewise untouched
#     vle32.v  2,116 -> 1,729      \
#     vse32.v    809 ->   581       |  the temporaries that stop being written and re-read
#     vfmul.vv   540 ->   263      /
#     add      4,390 -> 3,763      \
#     li       2,142 -> 1,641       |  1,547 scalar instructions of loop bookkeeping for those loops
#     addi     2,475 -> 2,056      /
#
# The failure mode of the refuted sibling is a number this one can be checked against, and it does not
# move. WHAT DOES GET WORSE, recorded rather than buried: `fmul.s` 0 -> 82 and `fsw` 128 -> 201 -- a
# fused op leaves a small scalar f32 tail where none existed. It is dwarfed by the 277 `vfmul.vv` it
# replaces, but it is a scalarization, and if this lever ever measures badly that is the first place
# to look.
#
# STRUCTURE-KEYED: the hooks name an upstream MLIR pass and one anchor pass, no model, shape, dtype or
# target. Nothing here knows what a quantize is.
FUSE_ELEMENTWISE_NAME = "fuse_elementwise_post_contraction"

#: The stage, in the order `build_rvv_pipeline` documents for it. The `canonicalize`/`cse` are NOT
#: decoration: measured on the tagged IR of another model, fusion alone gives broadcast 277 /
#: tensor.empty 1415 and fuse+canonicalize+cse gives 245 / 68 -- most of the temporary collapse is
#: the cleanup, not the fusion.
FUSE_ELEMENTWISE_STAGE: tuple[str, ...] = (
    "func.func(linalg-fuse-elementwise-ops)", "canonicalize", "cse")

#: The pass the stage must sit IMMEDIATELY IN FRONT OF. Anchoring on this one rather than on an index
#: is what keeps the stage on the correct side of the two passes whose order is a correctness
#: property: it must run AFTER `transform-interpreter` (fusing earlier folds matmuls into generics and
#: `ops{["linalg.matmul"]}` then matches nothing -- a silent 0-vectorization) and BEFORE bufferization
#: (afterwards there are no producer/consumer tensors left to fuse).
FUSE_ELEMENTWISE_ANCHOR = "func.func(linalg-generalize-named-ops)"


def _fuse_elementwise_pipeline(passes: list[str]) -> list[str]:
    """Splice :data:`FUSE_ELEMENTWISE_STAGE` in just before :data:`FUSE_ELEMENTWISE_ANCHOR`.

    Two refusals, both fail-closed, because "the feature was enabled and changed nothing" is the
    failure mode this repo keeps re-learning:

    * the stage already being present means ``MERLIN_FUSE_POST`` put it there, so this returns the
      list unchanged rather than fusing twice (which is not merely wasteful -- a second
      ``canonicalize``/``cse`` pair changes the emitted code, so a manual A/B and a feature-driven one
      would not be comparing the same build);
    * the anchor being absent means this is not a pipeline the stage belongs in, and inserting at a
      guessed index would put the fusion on the wrong side of the transform interpreter. Raise, so the
      build fails loudly instead of quietly emitting the baseline while reporting the feature applied.
    """
    if FUSE_ELEMENTWISE_STAGE[0] in passes:
        return list(passes)
    if FUSE_ELEMENTWISE_ANCHOR not in passes:
        raise ValueError(
            f"{FUSE_ELEMENTWISE_NAME} was requested but the pass list carries no "
            f"{FUSE_ELEMENTWISE_ANCHOR!r} to anchor the fusion stage against, so there is no position "
            f"that is provably after the transform interpreter and before bufferization. Refusing to "
            f"insert at a guessed index and report the feature as applied.")
    at = passes.index(FUSE_ELEMENTWISE_ANCHOR)
    return [*passes[:at], *FUSE_ELEMENTWISE_STAGE, *passes[at:]]


register(ImprFeature(
    name=FUSE_ELEMENTWISE_NAME,
    action_class="PASS",
    description=(
        "Run `linalg-fuse-elementwise-ops` (+ canonicalize/cse) after the transform schedule has "
        "matched and vectorized the contractions, so the elementwise producer->consumer chains that "
        "`linalg-specialize-generic-ops` un-fuses collapse again and each line is touched once. The "
        "stage already existed in build_rvv_pipeline but was reachable only through the "
        "MERLIN_FUSE_POST environment variable, which no fork can vary -- so the tuning loop could "
        "not select it. Attacks the linalg.generic long tail (16.5% of int8 whole-model runtime "
        "across 191 ops in 26 classes): the per-row activation-quantize and requant generics each "
        "read a scale that `specialize` materialized into a full-size temporary first. MEASURED on "
        "small_llama_int8_consistent at the current best config -- linalg.broadcast 50 -> 13, "
        "242,944 -> 60,160 bytes written per inference for zero arithmetic (13.9x amplification "
        "removed); emitted `forward` 35,253 -> 31,348 instructions (-11.1%, vector 14,053 -> 12,561 "
        "and scalar 13,802 from 15,319, so the vector FRACTION is flat at 0.478 -> 0.477 by "
        "construction), model.o 189,008 -> 170,664 bytes, stack alloca 118 -> 103 sites. Output "
        "BIT-IDENTICAL on spike (prefix digest cc60e8a90270ec1e either way; tiers ['fp32','w8a8'], "
        "tier_ok 'fp32_cos_only', ok True, every gate figure equal to the last digit). NOT the "
        "refuted `fuse_elementwise_after_generalize`, which runs the stage on the far side of the "
        "generalize anchor and measured 1.22x SLOWER by perturbing the already-vectorized "
        "contraction: here vwmacc stays 152 and vredmax 104 while vle32.v 2,116 -> 1,729 and "
        "vse32.v 809 -> 581. (fmul.s 0 -> 82 is a small scalar tail this introduces.) THE WALL IS "
        "UNMEASURED -- fewer instructions is a reason to measure, not a result: on this same model a "
        "transpose fold that removed ops and shrank the object measured 1.09x SLOWER. Structure-keyed "
        "(names one upstream pass and one anchor pass; no model, shape, dtype or target). "
        "Default-off; baseline byte-identical."),
    edit_pipeline=_fuse_elementwise_pipeline,
))
