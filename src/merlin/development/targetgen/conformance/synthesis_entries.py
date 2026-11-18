"""Turn a DERIVED conformance requirement into capsule-profile entries.

The corpus pipeline was already deterministic everywhere except at its input. Every per-capsule field is
derived from the target's own facts -- tile dim, dtypes, instruction classes, oracle tiers, scale block,
semantic family, golden regime -- and regeneration is byte-stable. The *requirement* is derived too:
:func:`merlin.development.targetgen.conformance.requirements.required_cells` intersects what the capability manifest admits with
what real captured models contain, and :mod:`build_tools.checks.check_conformance_coverage` measures the
corpus against it.

The two never met. Which capsules exist was a hand-written ``capsules:`` list -- roughly 180 entries
across six profiles -- and that list is the one thing a new target's owner cannot reasonably be asked to
produce. This module closes the loop: requirement in, profile entries out, in exactly the shape
``generate_corpus.py`` already consumes, so binding, builders, goldens, operand rigor, scrubbing and the
provenance manifest are all reused untouched.

WHAT IS DERIVED AND WHAT IS DECLARED. Everything here is derived from the spec the conformance module
already writes. The only declared input is the workload spec (:func:`merlin.target.registry.experiment_descriptor`
``workload_spec``): the model roster, a dtype PREFERENCE ORDER used solely to break ties on axes no cell
pins, and a synthesis budget. A preference can never widen what a target supports -- it is filtered
against the admitted dtypes and a token that does not survive is dropped and reported.

FAIL CLOSED, ALWAYS. A cell whose family no available op expresses is an ERROR naming the cell, never a
silent omission: a requirement that quietly produces no capsule is indistinguishable from one that is
met. Same for exceeding the budget -- it raises rather than truncating, because a silently dropped point
reads downstream as a covered one.
"""

from __future__ import annotations

import copy
import hashlib
import json
from math import isqrt
from typing import Any

from merlin.development.targetgen.applications.operation_plan import _application_operation_plan, _cost
from merlin.development.targetgen.corpus.selection import (
    SynthesisError,
    _quantize,
    _tile_int,
    cap_to_affordable,
    extents_for,
    narrowest_admitted,
    quanta_by_dtype,
)

# ---- corpus_synth ------------------------------------------------------------------------------

SYNTH_PREFIX = "SY"


SOURCE_ROLE = "derived_sweep"


def declares_accelerator_placement(entry: dict) -> bool:
    """Whether an entry's DECLARED placement leaves any of its work to the accelerator.

    A public capsule grades the author's accelerator compiler. Host placement is Merlin's own generic
    host codegen, so a capsule whose declared placement is host-only grades nobody's accelerator work:
    an entry that forbids the mesh lane, or a probe declared neither required nor eligible to
    accelerate, carries no accelerator placement. Read from the declared lane and eligibility, never
    from a capsule name or family.
    """
    lanes = entry.get("lanes") or {}
    generalization = entry.get("generalization") or {}
    if "on_mesh" in (lanes.get("forbid") or ()):
        return False
    return not (generalization.get("must_accelerate") is False and generalization.get("eligible") is False)


def withhold_host_placed(entries: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split entries into publishable ones and host-placed ones reported instead of emitted.

    A withheld entry is recorded with its family and dtype so a host requirement stays visible in
    provenance; it is never emitted as a public capsule, because nothing in it is the author's work.
    Held-out members follow the same rule: a hidden capsule grades the same accelerator compiler.
    """
    kept, withheld = [], []
    for entry in entries:
        if declares_accelerator_placement(entry):
            kept.append(entry)
            continue
        withheld.append(
            {
                "name": entry.get("name"),
                "op": entry.get("op"),
                "operand_dtype": entry.get("operand_dtype"),
                "source": entry.get("source"),
                "generalization_axis": (entry.get("generalization") or {}).get("generalization_axis"),
                "reason": "declared placement carries no accelerator work; host placement is Merlin-supplied",
            }
        )
    return kept, withheld


def _op_family_map() -> dict[str, str]:
    from merlin.target.capabilities import semantic_families as sf

    return dict(sf._OP_FAMILY)


def available_ops() -> set[str]:
    """Ops that can actually be MATERIALIZED -- a direct-MLIR builder or a PyTorch body exists.

    An op in the schema's enum with neither is dead vocabulary: naming it in an entry would produce a
    capsule nothing can write.
    """
    from merlin.development.targetgen.corpus.model_builders import BUILDERS

    try:
        from merlin.capture.loaders import _OP_BODIES

        bodies = set(_OP_BODIES)
    except Exception:  # noqa: BLE001 -- no torch bodies is not zero ops
        bodies = set()
    return set(BUILDERS) | bodies


_BLOCK_SCALED_GOLDEN_ONLY = frozenset({"attention_mx"})


def _is_ieee_float(dtype: str) -> bool:
    """Whether this dtype routes to the IEEE-float (``simt``) golden engine.

    Same routing the generator applies (``corpus_spec.entry_binding``): block-scaled -> the MX engine, fp8 -> the
    specir refmodel, IEEE fp16/bf16/f32 -> simt, anything else -> the integer engine. Asked of the
    format registry rather than listed here, so a target declaring a new float width routes without an
    edit.
    """
    from merlin.common.fp8_formats import canonical_float
    from merlin.development.targetgen.corpus.binding import is_block_scaled

    if is_block_scaled(dtype):
        return False
    try:
        return canonical_float(dtype) in ("fp16", "bf16", "f32")
    except KeyError:
        return False


_SINGLE_FORMAT_GOLDEN_WITHIN_MX = {"gemv_batched": "fp8_e4m3"}


def _canonical(dtype: str) -> str | None:
    """``dtype``'s canonical float name, or ``None`` when it has none (an integer format, say)."""
    try:
        from merlin.common.fp8_formats import canonical_float

        return canonical_float(str(dtype))
    except Exception:  # noqa: BLE001 -- not a float format is a real answer
        return None


def ops_gradeable_at(dtype: str, pool: set[str]) -> set[str]:
    """``pool`` narrowed to the ops whose golden engine can grade this cell's dtype.

    Two exclusions, both measured, both silent before:

    * a BLOCK-SCALED-ONLY op at a dtype that is not block-scaled. The reverse is fine -- a plain
      contraction grades in the block-scaled engine like any other.
    * a PYTORCH-BODY-ONLY op at a dtype that is not IEEE float. Such an op has no direct-MLIR builder,
      so it can only be captured through torch, and the generator refuses a pytorch capsule outside the
      float regime unless the entry names a quantization scheme -- correctly, because the default
      weight-only capture emits a float matmul over dequantized weights and cannot grade an integer or
      fp8 datapath. Choosing one anyway produced an entry the generator then rejected.

    An op excluded here is not a defect; it is a reason the cell may have no expressible capsule, which
    ``synthesize`` reports as an unexpressable cell naming the dtype.
    """
    from merlin.development.targetgen.corpus.binding import is_block_scaled

    keep = set(pool)
    if not is_block_scaled(dtype):
        keep -= _BLOCK_SCALED_GOLDEN_ONLY
    # A THIRD exclusion, and the narrowest: an op whose golden grades exactly ONE operand format. Being
    # block-scaled is not enough for these -- the batched MX golden reconstructs its reference on the
    # mx_ref datapath and refuses any format but mxfp8, so choosing it for an mxfp4 cell produced an
    # entry that built its interface and then failed at the golden, leaving a capsule directory with no
    # golden in it.
    if is_block_scaled(dtype):
        for op, want in _SINGLE_FORMAT_GOLDEN_WITHIN_MX.items():
            if op in keep and _canonical(dtype) != want:
                keep.discard(op)
    if not _is_ieee_float(dtype):
        keep = {op for op in keep if source_for_op(op) != "pytorch"}
    return keep


def source_for_op(op: str) -> str | None:
    """``"pytorch"`` when ``op`` exists ONLY as a PyTorch body, else ``None`` (the direct-MLIR path).

    ``available_ops`` is the UNION of the two registries, so an op chosen from it may live in either --
    and the entry must say which, because the two paths are different writers. Measured: 18 radiance
    entries named a body-only op (`gelu`, `reduce_sum`, `softmax`, `attention_full`, ...) with no
    `source`, which routed them to the direct-MLIR builder that has no builder for them at all. They
    could never be written, and `test_every_chosen_op_can_actually_be_materialized` passed anyway
    because it checked membership in the union rather than agreement between op and source.

    Derived from registry membership, not a list here: adding a builder for a body-only op flips its
    source automatically, in the direction that prefers the direct path.
    """
    from merlin.development.targetgen.corpus.model_builders import BUILDERS

    if op in BUILDERS:
        return None
    try:
        from merlin.capture.loaders import _OP_BODIES
    except Exception:  # noqa: BLE001 -- no bodies available
        return None
    return "pytorch" if op in _OP_BODIES else None


_BATCHED_OPS = frozenset({"gemv_batched"})


_LAYOUT_OPS: frozenset = frozenset()


def op_for_shape(
    family: str,
    *,
    admitted_ops: set[str] | None = None,
    dtype: str | None = None,
    rank: int = 2,
    layout: str | None = None,
) -> str | None:
    """The op that exercises ``family`` at ``rank``/``layout``, or ``None`` when nothing materializes it.

    A thin constraint over :func:`op_for_family`: the family and dtype still decide WHICH op represents
    the cell, and this additionally refuses one that cannot express the region's shape. ``None`` is a
    reportable capability gap, never a reason to substitute a plain 2-D region -- a rank-3 requirement
    met by a rank-2 capsule is an uncovered point that reads as covered.
    """
    op = op_for_family(family, admitted_ops=admitted_ops, dtype=dtype)
    if op is None:
        return None
    if layout:
        return op if op in _LAYOUT_OPS else None
    if int(rank) >= 3:
        # NARROWED BY THE GOLDEN, not just by the builder. The one batched builder emits a BLOCK-SCALED
        # contraction, so it can be BUILT for any target and GRADED only where the dtype is block
        # scaled. Choosing it anyway produced a rank-3 entry on every target that declares batching,
        # which the generator would then have rejected downstream -- turning a reportable capability gap
        # into a capsule that silently disappears. `ops_gradeable_at` is the same predicate the cell
        # axis uses, so the two cannot disagree about what a dtype can grade.
        # THE BATCHED OP MUST BE OF THE FAMILY BEING ASKED FOR. This returned the first batched op in
        # the pool whatever family the probe named, so a batched CONTRACTION was emitted to evidence
        # `elementwise_map.batched`, `movement.batched` and `reduction.batched` alike -- three cells
        # covered by a capsule that computes something else. It was invisible while the only batched
        # golden was block-scaled (every such probe was a reported hole); giving the op a golden in
        # every engine turned the hole into a wrong capsule, which is worse. A family with no batched
        # op of its own is a hole again, and says so.
        pool = admitted_ops if admitted_ops is not None else available_ops()
        fam_map = _op_family_map()
        batched = sorted(
            op for op in (_BATCHED_OPS & ops_gradeable_at(str(dtype or ""), set(pool))) if fam_map.get(op) == family
        )
        return batched[0] if batched else None
    return op


_CONV_WINDOW_OUT = 4


def _image_for_window(
    kernel, stride, dilation, pad_before, pad_after, *, output_extent=_CONV_WINDOW_OUT
) -> tuple[int, ...]:
    """The input image that makes ``kernel`` produce ``output_extent`` outputs per axis.

    The inverse of :func:`merlin.runtime.engine.command_buffer.conv_out_dims`, which computes
    ``out = (H + pt + pb - (d*(k-1) + 1)) // s + 1``. Solving it for H at a chosen output is what keeps
    a 16x16 stride-16 window from walking off the default 8x8 image and producing a capsule with zero
    output rows -- which builds, runs, passes, and tests nothing.
    """
    out: list[int] = []
    for i, k in enumerate(kernel):
        s_ = stride[i] if i < len(stride) else 1
        d_ = dilation[i] if i < len(dilation) else 1
        before = pad_before[i] if i < len(pad_before) else 0
        after = pad_after[i] if i < len(pad_after) else 0
        span = d_ * (k - 1) + 1
        out.append(max(1, (output_extent - 1) * s_ + span - before - after))
    return tuple(out)


_CONV_OPS_BY_RANK: dict[int, tuple[str, ...]] = {
    1: ("conv1d",),
    2: ("conv2d", "convolution", "depthwise_conv2d"),
    3: ("conv3d",),
}


def op_for_family(
    family: str, *, admitted_ops: set[str] | None = None, dtype: str | None = None, writer: str | None = None
) -> str | None:
    """The cheapest op that exercises ``family`` AND can actually be written at ``dtype``.

    WRITABILITY COMES FIRST, cost second. Ranking by cost alone picked the cheapest op in the abstract
    and then discovered no writer could express it: measured, an elementwise cell chose `gelu` -- one
    operand, no direct-MLIR builder -- over `bias_add`, which has a builder, and then failed at an fp8
    dtype the PyTorch writer cannot take. Both are elementwise_map, so the cell had a writable
    representative all along and the ranking looked past it.

    ``dtype`` omitted keeps the old ordering, for callers asking the abstract question.

    ``writer`` NARROWS TO ONE WRITER, and the host axes need it. A host-lane capsule must classify as
    host work, and every op with a `merlin_iface` builder classifies as accelerator work by
    construction -- so those axes need an op the interface dialect CANNOT express, which is exactly
    `_writer_for(...) == "pytorch"`. Ranking without it picked `rmsnorm` (a builder) over `layernorm`
    (no builder) for the same family, and the generator then dropped the capsule as an unprovable
    forbid on three targets: the host-only lane was demanded, emitted, refused, and reported as an
    uncovered family the whole way down, while an op that could have proven it sat in the same family.
    """
    pool = admitted_ops if admitted_ops is not None else available_ops()
    cands = [op for op, fam in _op_family_map().items() if fam == family and op in pool]
    if writer is not None:
        cands = [op for op in cands if _writer_for({"op": op, "operand_dtype": dtype}) == writer]
    if dtype is None:
        return sorted(cands, key=_cost)[0] if cands else None
    # False sorts before True, so a writable op outranks an unwritable one whatever it costs.
    ranked = sorted(cands, key=lambda op: (_writer_for({"op": op, "operand_dtype": dtype}) is None, _cost(op)))
    return ranked[0] if ranked else None


def _fused_carrier(family: str, composed_with: tuple, pool: set[str]) -> tuple[str, list[str]] | None:
    """``(op, epilogue)`` for a family the manifest admits only IN COMPOSITION.

    A family declared ``composed_with: [contraction]`` is fused-only: the eligibility oracle refuses a
    standalone capsule for it as a false fallback, so the only capsule that can ever evidence that cell
    is a contraction carrying it as an epilogue. Emitting the standalone op would produce a capsule that
    is *wrong* rather than merely weak -- this is the single case where the obvious choice is a defect.

    THE STAGE MUST BELONG TO THE FAMILY IT IS COVERING, and a baked table got that wrong. This read
    ``{"elementwise_map": "relu", "reduction": "acc_scale"}`` -- but ``acc_scale`` is an
    ``elementwise_map``, so all three ``SY_reduction_i8_*`` capsules carried an elementwise stage,
    credited ``composed_families: [elementwise_map]``, and covered the elementwise cells they were not
    named for while the reduction cells they WERE named for stayed uncovered. Measured on gemmini:
    ``reduction/i8/partial`` had no capsule at all, and the other two reduction cells were covered
    incidentally by a maxpool epilogue capsule and two hand-authored pooling capsules.

    So the stage is derived: ask the taxonomy which family each buildable stage belongs to and take one
    that answers with this family. Same source the requirement's epilogue axis uses, so the two cannot
    disagree about what a stage is.
    """
    if "contraction" not in {str(c) for c in composed_with}:
        return None
    carrier = op_for_family("contraction", admitted_ops=pool)
    if carrier is None:
        return None
    from merlin.development.targetgen.corpus.binding import BUILDER_EPILOGUE_STAGES
    from merlin.target.capabilities import semantic_families as _sf

    # The stages a contraction builder can actually carry, read from the builder module itself (the same
    # tuple the requirement's epilogue axis walks) rather than re-typed here; the family each one belongs
    # to is DERIVED from the semantic taxonomy, so a new stage needs no table at all.
    stage = next((st for st in BUILDER_EPILOGUE_STAGES if _sf.from_op(st) == family), None)
    return (carrier, [stage]) if stage else None


SYNTH_ACC_SCALE = 0.0625


def _signed_stimulus_range() -> tuple[int, int]:
    from merlin.runtime.engine.command_buffer import SIGNED_STIMULUS_RANGE

    return SIGNED_STIMULUS_RANGE


def declare_pool_window(entry: dict) -> dict:
    """Give a pooling epilogue the geometry it needs, in place. Inert on every other entry.

    A pooling stage commits FEWER rows than it reads, so the builder must know what the contraction's
    rows mean spatially. Two things follow, and both belong here rather than in each axis:

    * the WINDOW is declared (a 2x2 with stride 2 is the shape every pooling datapath has) while the
      input geometry is left to the generator, which is the first place the tile edge is known -- the
      extents here are still tile-relative tokens by design;
    * ``M`` is pinned to one whole tile, because the generator factors those rows into ``H x W`` and a
      non-square count has no such reading. That does NOT flatten the alignment axis: occupancy is
      classified from the ragged extent, so ``K``/``N`` still carry aligned, partial and sub_tile
      (16x32x16, 16x32x15 and 16x4x4 classify as all three).
    """
    if "maxpool" not in [str(x) for x in (entry.get("epilogue") or ())]:
        return entry
    entry["pool_size"] = [2, 2]
    entry["pool_stride"] = [2, 2]
    entry["M"] = "tile"
    return entry


SYNTH_RESIDUAL_SCALE = 1.0


SYNTH_RESIDUAL_BOUND_LSB = 0


def declare_residual_add_params(entry: dict) -> dict:
    """Give a synthesized two-operand add the arithmetic it must declare, in place. Inert otherwise.

    The sibling of :func:`declare_pool_window`, and it exists for the same reason: an op the chooser
    may pick must be one the synthesizer can produce a COMPLETE entry for. ``build_residual_add``
    requires ``lhs_scale``/``rhs_scale``/``bound_lsb`` and refuses to default any of them, so an entry
    that named the op without them raised inside the builder -- turning a chosen op into a generation
    failure rather than a capsule.

    Latent until now and no longer: ``residual_add`` has been choosable for the elementwise family all
    along and lost the cost ranking to ``bias_add`` on every target measured, so nothing ever hit it.
    A target whose dtype cannot grade ``bias_add`` would have.
    """
    from merlin.development.targetgen.corpus.contraction_builders import build_residual_add
    from merlin.development.targetgen.corpus.model_builders import BUILDERS

    if BUILDERS.get(str(entry.get("op", ""))) is not build_residual_add:
        return entry
    entry.setdefault("lhs_scale", SYNTH_RESIDUAL_SCALE)
    entry.setdefault("rhs_scale", SYNTH_RESIDUAL_SCALE)
    entry.setdefault("bound_lsb", SYNTH_RESIDUAL_BOUND_LSB)
    return entry


def filtered_precision(preference: list[str], admitted: set[str]) -> tuple[list[str], list[str]]:
    """``(kept, dropped)`` -- the declared preference order, filtered against what the target admits.

    A preference RANKS what the hardware already has; it can never widen it. A token that does not
    survive is reported rather than dropped silently, because "we preferred int8 and the target has no
    int8 datapath" is a fact the reader of a synthesized corpus needs.
    """
    kept = [d for d in preference if d in admitted]
    return kept, [d for d in preference if d not in admitted]


def _exceeds_tile(token: str, tile: int) -> bool:
    """Does this extent token resolve to more than one tile edge?

    Resolve any supported multiplier, not just the first few common spellings:
    a capacity-derived boundary may need ``65*tile``. Unknown tokens remain
    non-evidence rather than being guessed larger than the edge.
    """
    resolved = _tile_int(token, tile) if tile else None
    return resolved is not None and resolved > tile


def pass_requirements_for(entry: dict, spec_doc: dict) -> list[str]:
    """The compiler obligations this entry actually exercises, as requirement CLASSES.

    `check_pass_obligations` rejects a catalogued pass no capsule requires, and rejects a pass that
    discharges no obligation -- but the loop was built and left empty: two of 248 capsules declared
    `pass_requirements`, both the capstone class, so the gate could not be turned on without rejecting
    most of the compiler. Deriving them from what an entry DOES is the only way the count moves without
    someone hand-labelling the corpus, and it keeps the declaration honest: a class appears because the
    shape demands it, not because an author remembered.

    Derived, never guessed:
      * ``target-isa-lowering`` -- the capsule executes, so something must lower to the target's ISA.
      * ``tile-schedule`` -- an extent exceeds the tile edge, so the work must be scheduled across
        tiles rather than issued as one.
      * ``host-seam`` -- the capsule declares a lane the accelerator does not own, so a boundary has to
        be materialized for the value to cross.
      * ``region-partition`` -- more than one region, so something must decide where each one goes.
      * ``block-movement`` -- an operand's CONTIGUOUS axis spans more than one tile, so a transfer of
        that operand carries several tiles' worth of adjacent bytes and the emitter has to decide how
        wide to make it.
    """
    from merlin.compiler.dialects.lowering import passes as _P

    out = [_P.TARGET_ISA_LOWERING]
    tile = 0
    for probe in (spec_doc.get("boundaries") or {}).get("extent_probes") or ():
        tile = max(tile, int(probe.get("edge") or 0))
    exceeds = {axis: _exceeds_tile(str(entry.get(axis) or ""), tile) for axis in ("M", "K", "N")}
    conv_windows_exceed_tile = False
    if entry.get("op") == "conv2d" and entry.get("Himg") is not None and entry.get("Wimg") is not None and tile:
        from merlin.runtime.engine.command_buffer import conv_out_dims

        oh, ow = conv_out_dims(
            int(entry["Himg"]),
            int(entry["Wimg"]),
            int(entry["kh"]),
            int(entry["kw"]),
            entry["stride"],
            entry["padding"],
            entry["dilation"],
        )
        conv_windows_exceed_tile = oh * ow > tile
    if any(exceeds.values()) or conv_windows_exceed_tile:
        out.append(_P.TILE_SCHEDULE)
    # THE CONTIGUOUS AXES, NOT ALL OF THEM. A transfer moves a rectangle whose width runs along the
    # operand's fastest-varying axis: the reduction axis K for the activation rows, the output axis N
    # for the weight and result rows. M only adds ROWS -- more transfers of the same width -- so an
    # entry that exceeds the edge on M alone demands tiling and nothing about movement width. K or N
    # past the edge is the case where one transfer COULD carry the next tile's bytes too, which is the
    # decision `block-movement` exists to demand. Derived from the entry's own extents, exactly like
    # `tile-schedule`; whether the target's DMA admits that width is the target's fact, checked
    # against the emitted stream by `trace_check.movement_findings`, not assumed here.
    if exceeds["K"] or exceeds["N"]:
        out.append(_P.BLOCK_MOVEMENT)
    lanes = entry.get("lanes") or {}
    if lanes.get("forbid") or any(str(lane) != "on_mesh" for lane in (lanes.get("require") or ())):
        out.append(_P.HOST_SEAM)
    if str(entry.get("kind")) == "model" or len(lanes.get("require") or ()) > 1:
        out.append(_P.REGION_PARTITION)
    return sorted(dict.fromkeys(out))


DEFAULT_PERMUTE_LAYOUT = {"shape": [1, 4, 8, 16], "permutation": [0, 2, 1, 3]}


def permute_probe_layout(observed: dict | None) -> dict:
    """``{shape, permutation}`` for a host-lane permutation probe: the most frequent rank-3/4 layout the
    captured regions carry, else :data:`DEFAULT_PERMUTE_LAYOUT`. Never the rank-2 transpose of a weight
    view, which is not the host movement the captures contain."""
    for layout in (observed or {}).get("layouts") or ():
        shape, order = list(layout.get("shape") or ()), list(layout.get("permutation") or ())
        if len(shape) in (3, 4) and sorted(order) == list(range(len(shape))) and order != sorted(order):
            return {"shape": shape, "permutation": order}
    return dict(DEFAULT_PERMUTE_LAYOUT)


def _host_probe_op(
    requirement: dict,
    family: str,
    dtype: str | None,
    pool: set[str],
    *,
    native_inventory=None,
    native_inventory_sha256=None,
) -> tuple[str | None, dict | None]:
    """``(op, observed row)`` for a host probe: an observed host operation a PyTorch writer can express.

    A requirement that carries ``observed_ops`` is answered ONLY from them, so the probe reproduces
    work real captures contain and stays inside declarations reviewed for exactly that work; an
    empty answer is reported as an uncovered pair. A historical requirement with no ``observed_ops``
    key keeps the family-level choice it was derived under.
    """
    rows = requirement.get("observed_ops")
    if rows is None:
        return op_for_family(family, admitted_ops=pool, dtype=dtype, writer="pytorch"), None
    from merlin.development.targetgen.applications.native_storage import host_source_declaration

    for observed in rows:
        op = observed.get("op")
        if op not in pool or _op_family_map().get(op) != family:
            continue
        candidate = {
            "op": op,
            "operand_dtype": dtype,
            "frontend_op": observed.get("frontend_op"),
            "generalization": {"must_accelerate": False, "eligible": False},
        }
        if _writer_for(candidate) == "pytorch":
            return op, observed
        native = host_source_declaration(observed, family, dtype, native_inventory, native_inventory_sha256)
        if native is not None and _writer_for({**candidate, **native}) == "pytorch":
            return op, {**observed, **native}
    return None, None


def _writer_for(entry: dict) -> str | None:
    from merlin.development.targetgen.corpus.selection import select_writer

    return select_writer(entry, source_for_op=source_for_op)


def _mark_source(entry: dict) -> None:
    from merlin.development.targetgen.corpus.selection import mark_source

    mark_source(entry)


def _exact_int_mm_group_candidates(
    group: dict, entries: list[dict], inventory_digest: str | None, application_order: tuple[str, ...]
) -> list[str]:
    """Name candidates only when every source occurrence has one digest-bound exact signature.

    Compact operation groups lose the individual tensor ABI. The entries retain each full sidecar
    signature hash and ordinal, so compare their complete occurrence multiset with the compact group.
    Equal family/dtype or a partial geometry match is insufficient.
    """
    from merlin.development.targetgen.applications.inventory import int_mm_source_is_qualified

    if not inventory_digest or not entries:
        return []
    raw_sources = group.get("sources") or {}
    if isinstance(raw_sources, dict):
        source_rows = [{"application": label, **record} for label, record in raw_sources.items()]
    elif isinstance(raw_sources, list):
        source_rows = raw_sources
    else:
        return []
    source_index = {name: index for index, name in enumerate(application_order)}
    expected: dict[tuple[int, str], int] = {}
    for source in source_rows:
        if not isinstance(source, dict):
            return []
        index = source_index.get(str(source.get("application") or ""))
        if index is None:
            return []
        key = (index, str(source.get("capture_sha256") or ""))
        count = source.get("count")
        if not key[1] or type(count) is not int or count < 1 or key in expected:
            return []
        expected[key] = count
    if not expected or sum(expected.values()) != group.get("count"):
        return []

    actual: dict[tuple[int, str], int] = {}
    ordinals: dict[tuple[int, str], set[int]] = {}
    names: list[str] = []
    for entry in entries:
        match = entry.get("application_signature_match") or {}
        if (
            entry.get("op") != "matmul"
            or entry.get("capture_op") != "int_matmul"
            or match.get("status") != "candidate_unverified"
            or match.get("full_inventory_sha256") != inventory_digest
            or not int_mm_source_is_qualified(match)
        ):
            return []
        if not match.get("expected_signature") or not match.get("sources"):
            return []
        names.append(str(entry["name"]))
        for source in match["sources"]:
            index = source.get("source_index")
            if type(index) is not int:
                return []
            key = (index, str(source.get("capture_sha256") or ""))
            count = source.get("count")
            indexes = source.get("ordinals")
            signature = source.get("signature_sha256")
            if (
                key not in expected
                or type(count) is not int
                or count < 1
                or not isinstance(indexes, list)
                or len(indexes) != count
                or any(type(index) is not int or index < 0 for index in indexes)
                or not isinstance(signature, str)
                or len(signature) != 64
            ):
                return []
            seen = ordinals.setdefault(key, set())
            if seen.intersection(indexes) or len(set(indexes)) != len(indexes):
                return []
            seen.update(indexes)
            actual[key] = actual.get(key, 0) + count
    return sorted(names) if actual == expected and len(set(names)) == len(names) else []


def exact_int_mm_entries(demands: dict | None, inventory: dict | None) -> tuple[list[dict], list[dict]]:
    """Return exact W8A8 integer-matmul slice candidates from a sealed application inventory.

    This is a candidate mapping, not a claim that the generated capsule or a submitted compiler runs.
    The writer must capture ``torch._int_mm`` and verify its linalg signature before it records a match.
    Distinct model weights with the same integer ABI need one slice; all source signatures stay linked.
    """
    if inventory is None:
        return [], []
    if not isinstance(demands, dict) or not demands.get("full_inventory_sha256"):
        raise SynthesisError("exact application slices require a compact digest-bound inventory")
    digest = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != demands["full_inventory_sha256"]:
        raise SynthesisError("exact application inventory digest differs from the selected conformance spec")
    if inventory.get("schema_version") not in (1, 2) or not isinstance(inventory.get("applications"), dict):
        raise SynthesisError("exact application inventory has an unsupported schema")
    compact_apps = demands.get("applications") or {}
    if set(inventory["applications"]) != set(compact_apps):
        raise SynthesisError("exact application inventory roster differs from the selected conformance spec")

    from merlin.common.digest import is_sha256
    from merlin.development.targetgen.applications.inventory import (
        exact_int_mm_geometry,
        verified_static_integerization,
    )

    # The arithmetic slice can be shared within one source scheme, never by aliasing two different
    # full-model quantization workflows that happen to have the same integer tensor ABI.
    by_geometry: dict[tuple[str, int, int, int], list[dict]] = {}
    refused: list[dict] = []
    for source_index, (label, app) in enumerate(sorted(inventory["applications"].items())):
        declared = compact_apps[label]
        if any(
            app.get(key) != declared.get(key)
            for key in ("capture_sha256", "capture_normalization", "capture_quantization")
        ):
            raise SynthesisError(f"exact application capture identity differs for {label!r}")
        scheme = app.get("capture_quantization")
        materialized = (app.get("capture_receipt") or {}).get("status") == "verified_materialized"
        static_proof = app.get("capture_integerization")
        receipt_sha = (app.get("capture_receipt") or {}).get("receipt_sha256")
        static_qualified = (
            materialized
            and is_sha256(receipt_sha)
            and verified_static_integerization(static_proof, receipt_sha256=receipt_sha)
            and static_proof["capture"]["sha256"] == app.get("capture_sha256")
        )
        for row in app.get("signatures") or ():
            if row.get("operation") != "aten._int_mm.default" or row.get("mlir_operation") != "linalg.generic":
                continue
            source = {
                # The exact private inventory keeps the source name. Public capsules use its
                # deterministic roster ordinal and capture hash, never a claim-model name.
                "source_index": source_index,
                "capture_sha256": app["capture_sha256"],
                "signature_sha256": hashlib.sha256(
                    json.dumps(
                        {k: v for k, v in row.items() if k not in {"count", "ordinals"}},
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "count": int(row.get("count") or 0),
                "ordinals": list(row.get("ordinals") or ()),
                "source_quantization": scheme,
            }
            if scheme == "int8_static_act_int8_weight":
                source.update(capture_receipt_sha256=receipt_sha, capture_integerization=static_proof)
            # A sealed model capture proves the operation's bytes and module-level quantization
            # identity. Older captures can still serve diagnostics when m2m supplied the per-op
            # quantized-weight origin; absence of both receipts is a refusal.
            geometry = exact_int_mm_geometry(row, require_quant_origin=not materialized)
            if (
                not (
                    scheme == "int8_dyn_act_int8_weight"
                    or (scheme == "int8_static_act_int8_weight" and static_qualified)
                )
                or row.get("disposition") != "hardware_admitted"
                or geometry is None
            ):
                refused.append(
                    {
                        **source,
                        "reason": (
                            "requires admitted rank-2 signed i8×i8→i32 with exact matmul maps/body, "
                            "a supported source scheme and a verified materialization receipt or "
                            "per-op weight-origin evidence; static sources additionally require "
                            "byte-bound finite integerization agreement with zero remaining contractions"
                        ),
                    }
                )
                continue
            signature = {
                key: row[key]
                for key in (
                    "operation",
                    "mlir_operation",
                    "frontend_op",
                    "provenance_op",
                    "semantic_family",
                    "operand_format",
                    "accumulator_dtypes",
                    "ordered_operand_types",
                    "ordered_result_types",
                    "result_shapes",
                    "contraction_shape",
                    "indexing_maps",
                    "iterator_types",
                    "body_operations",
                )
            }
            by_geometry.setdefault((scheme, *geometry), []).append({**source, "signature": signature})

    entries = []
    for (scheme, m, k, n), sources in sorted(by_geometry.items()):
        identity = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        entry = {
            "cat": "layers",
            "kind": "layer",
            "name": f"{SYNTH_PREFIX}_int_mm_m{m}_k{k}_n{n}_{identity[:10]}",
            "op": "matmul",
            "source": "pytorch",
            "capture_op": "int_matmul",
            # The isolated torch._int_mm body already takes quantized i8 operands; fp32 here only
            # disables the worker's automatic weight-only quantizer. The emitted IR is verified below.
            "capture_dtype": "fp32",
            "operand_dtype": "i8",
            "output_dtype": "i32",
            "M": m,
            "K": k,
            "N": n,
            "lhs": "A0",
            "weight": "W",
            "out": "Y0",
            "source_role": "model_derived",
            "source_reference": (
                f"exact normalized application aten._int_mm.default i8×i8→i32, {m}×{k}×{n}; "
                f"{sum(s['count'] for s in sources)} occurrence(s) in "
                f"{len({s['source_index'] for s in sources})} selected capture(s)"
            ),
            "label": "public",
            "generalization": {"generalization_axis": "application_operation"},
            "application_signature_match": {
                "status": "candidate_unverified",
                "full_inventory_sha256": digest,
                "source_quantization": scheme,
                "expected_signature": sources[0]["signature"],
                "sources": [{key: value for key, value in source.items() if key != "signature"} for source in sources],
            },
        }
        entries.append(entry)
    return entries, refused


TAIL_ISOLATION_SHAPES = (
    (
        "k_tail",
        {"M": "tile", "K": "2*tile-1", "N": "tile"},
        "only the reduction extent is ragged: M and N sit on the tile edge while K stops one short of "
        "two tiles, so a compiler that pads or masks only the output tile fails here alone",
    ),
    (
        "gemv_n_tail",
        {"M": 1, "K": "2*tile", "N": "2*tile-1"},
        "a single-row matrix-vector product whose output extent stops one short of two tiles, so the "
        "last output tile is partial while the reduction is whole",
    ),
)


def _tail_isolation_entries(op: str | None, dtype: str | None, spec_doc: dict, *, emitted: set[str]) -> list[dict]:
    """Contraction members that rag exactly one extent, on the dtype the geometry axis certifies.

    `partial` rags K and N together and the geometry classes are real model shapes, so no member asks
    for a ragged reduction under aligned M and N, or for a single-row product whose output alone is
    ragged. Both rest on the certified aligned sibling; without one they are not emitted.
    """
    sibling = f"{SYNTH_PREFIX}_contraction_{dtype}_aligned"
    if not op or not dtype or sibling not in emitted:
        return []
    members = []
    for label, extents, why in TAIL_ISOLATION_SHAPES:
        entry = {
            "cat": "isa",
            "kind": "isa",
            "name": f"{SYNTH_PREFIX}_contraction_{dtype}_{label}".replace("-", "_"),
            "op": op,
            "operand_dtype": dtype,
            "out": "Y0",
            "lhs": "A0",
            "weight": "W",
            "source_role": SOURCE_ROLE,
            "source_reference": f"synthesized for the tail-isolation axis: {why}",
            "label": "public",
            "generalization": {"generalization_axis": "tail_isolation"},
            **extents,
        }
        if _writer_for(entry) is None:
            continue
        if _writer_for(entry) == "pytorch":
            entry["source"] = "pytorch"
        reason = cap_to_affordable(entry, spec_doc, extends=sibling)
        if reason:
            entry["source_reference"] += f". {reason}"
        entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
        members.append(entry)
    return members


ACTIVATION_CONTRACTION_SHAPES = (
    (
        "aligned",
        {"M": "tile", "K": "2*tile", "N": "2*tile"},
        "every extent sits on the tile edge, so the only difference from the weight-stationary sibling "
        "is that neither operand is stored ahead of the call",
    ),
    (
        "decode",
        {"M": 1, "K": "2*tile", "N": "2*tile"},
        "a single-row product of two activations, the decode step's shape, so one output row is all the mesh is given",
    ),
    (
        "k_tail",
        {"M": "tile", "K": "2*tile-1", "N": "2*tile"},
        "the reduction extent both activations share stops one short of two tiles, and neither operand "
        "can be padded offline because neither is stored",
    ),
)


def _activation_contraction_entries(dtype: str | None, spec_doc: dict, *, emitted: set[str]) -> list[dict]:
    """Activation x activation contraction members whose product feeds a reshape.

    Written on the dtype the geometry axis certifies, as an exact integer program: operand-dtype
    inputs, an accumulator-dtype product compared exactly, no scale metadata. They rest on the
    certified aligned contraction sibling; without one, or on a dtype that is not an integer
    regime, they are not emitted.
    """
    from merlin.development.targetgen.corpus.binding import regime_for_dtype

    sibling = f"{SYNTH_PREFIX}_contraction_{dtype}_aligned"
    if not dtype or sibling not in emitted:
        return []
    try:
        if regime_for_dtype(str(dtype)) != "int":
            return []
    except Exception:  # noqa: BLE001 -- an unknown regime is not an exact integer one
        return []
    members = []
    for label, extents, why in ACTIVATION_CONTRACTION_SHAPES:
        entry = {
            "cat": "isa",
            "kind": "layer",
            "name": f"{SYNTH_PREFIX}_act_act_{dtype}_{label}".replace("-", "_"),
            "op": "activation_matmul",
            "operand_dtype": dtype,
            "out": "Y0",
            "lhs": "A0",
            "rhs": "B0",
            "source_role": SOURCE_ROLE,
            "source_reference": f"synthesized for the activation-contraction axis: {why}",
            "label": "public",
            "generalization": {"generalization_axis": "activation_contraction"},
            **extents,
        }
        if _writer_for(entry) != "builder":
            continue
        reason = cap_to_affordable(entry, spec_doc, extends=sibling)
        if reason:
            entry["source_reference"] += f". {reason}"
        entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
        members.append(entry)
    return members


def _multi_block_members(output_bound: dict, op: str, dtype: str, sibling: str, spec_doc: dict) -> list[dict]:
    """Members whose output spans several blocks on BOTH parallel axes, with tails, derived from facts.

    The accumulator-output boundary asks for one output row across many column tiles. A blocked
    schedule fails elsewhere too: per-block indexing of the accumulate decision, block edges that fall
    inside a tail, and the pass in which the live output tiles exceed what the accumulator holds. Two
    members, sized in tiles from the RTL-derived capacity (never a model shape): one whose output tile
    count stays inside one accumulator buffer with several blocks per axis, and one just beyond it.
    Both carry a non-multiple of the tile on M, N and K and a multi-tile reduction.
    """
    edge = int(output_bound.get("tile_edge") or 0)
    rows = int(output_bound.get("capacity_rows") or 0)
    if edge < 1 or rows < edge:
        return []
    capacity_tiles = rows // edge  # output tiles one accumulator buffer holds at once
    side = 1
    while (side + 1) * (side + 1) <= capacity_tiles:
        side += 1
    within = max(2, side - 1)
    beyond = side + 1  # beyond * beyond > capacity_tiles by construction
    members = []
    for label, m_tiles, n_tiles in (
        ("within", within, within - 1 if within > 2 else within),
        ("beyond", beyond, beyond),
    ):
        entry = {
            "cat": "layers",
            "kind": "layer",
            "name": f"{SYNTH_PREFIX}_multi_block_{label}",
            "op": op,
            "operand_dtype": dtype,
            "lhs": "A0",
            "weight": "W",
            "out": "Y0",
            "M": f"{m_tiles}*tile-3",
            "N": f"{n_tiles}*tile+5",
            "K": "3*tile+1",
            "stimulus_range": list(_signed_stimulus_range()),
            "source_role": SOURCE_ROLE,
            "source_reference": (
                f"synthesized for the multi-block axis: the RTL-derived accumulator holds {capacity_tiles} "
                f"output tiles; this member's output spans about {m_tiles}x{n_tiles} tiles "
                f"({'inside' if label == 'within' else 'beyond'} one buffer) with tails on M, N and K"
            ),
            "label": "public",
            "generalization": {"generalization_axis": "multi_block"},
        }
        why = cap_to_affordable(entry, spec_doc, extends=sibling)
        if why:
            entry["source_reference"] += f". {why}"
        entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
        members.append(entry)
    return members
