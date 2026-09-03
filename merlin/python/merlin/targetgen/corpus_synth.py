"""Turn a DERIVED conformance requirement into capsule-profile entries.

The corpus pipeline was already deterministic everywhere except at its input. Every per-capsule field is
derived from the target's own facts -- tile dim, dtypes, instruction classes, oracle tiers, scale block,
semantic family, golden regime -- and regeneration is byte-stable. The *requirement* is derived too:
:func:`merlin.targetgen.conformance.required_cells` intersects what the capability manifest admits with
what real captured models contain, and :mod:`build_tools.scripts.check_conformance_coverage` measures the
corpus against it.

The two never met. Which capsules exist was a hand-written ``capsules:`` list -- roughly 180 entries
across six profiles -- and that list is the one thing a new target's owner cannot reasonably be asked to
produce. This module closes the loop: requirement in, profile entries out, in exactly the shape
``generate_corpus.py`` already consumes, so binding, builders, goldens, operand rigor, scrubbing and the
provenance manifest are all reused untouched.

WHAT IS DERIVED AND WHAT IS DECLARED. Everything here is derived from the spec the conformance module
already writes. The only declared input is the workload spec (:func:`merlin.targetgen.target_experiment`
``workload_spec``): the model roster, a dtype PREFERENCE ORDER used solely to break ties on axes no cell
pins, and a synthesis budget. A preference can never widen what a target supports -- it is filtered
against the admitted dtypes and a token that does not survive is dropped and reported.

FAIL CLOSED, ALWAYS. A cell whose family no available op expresses is an ERROR naming the cell, never a
silent omission: a requirement that quietly produces no capsule is indistinguishable from one that is
met. Same for exceeding the budget -- it raises rather than truncating, because a silently dropped point
reads downstream as a covered one.
"""
from __future__ import annotations

from typing import Any

#: Emitted entries carry this prefix so a synthesized capsule can never collide with a hand-authored one
#: or with a sweep expansion (``expand_sweeps`` already raises on a duplicate name), and so a reader of
#: the corpus can tell at a glance which capsules a requirement produced.
SYNTH_PREFIX = "SY"

#: ``source_role`` for everything this module emits. Already in the capsule schema's closed enum, and
#: already means "expanded from a declarative rule rather than typed by hand".
SOURCE_ROLE = "derived_sweep"


class SynthesisError(ValueError):
    """A requirement that cannot be turned into a capsule, named so it can be fixed."""


def _op_family_map() -> dict[str, str]:
    from merlin.targetgen import semantic_families as sf
    return dict(sf._OP_FAMILY)


def available_ops() -> set[str]:
    """Ops that can actually be MATERIALIZED -- a direct-MLIR builder or a PyTorch body exists.

    An op in the schema's enum with neither is dead vocabulary: naming it in an entry would produce a
    capsule nothing can write.
    """
    from merlin.targetgen.corpus_spec import BUILDERS
    try:
        from merlin.targetgen.capsule_source import _OP_BODIES
        bodies = set(_OP_BODIES)
    except Exception:                              # noqa: BLE001 -- no torch bodies is not zero ops
        bodies = set()
    return set(BUILDERS) | bodies


def _cost(op: str) -> tuple:
    """Rank candidate ops for a family: fewest operands, then smallest default footprint, then name.

    A derived tie-break rather than a preference list, so adding an op cannot silently change which
    capsule a cell gets unless that op is genuinely cheaper.
    """
    from merlin.targetgen.corpus_spec import BUILDERS
    # Operand ROLES, not tensor count: `linear` carries an optional bias role that `matmul` does not,
    # so the bare contraction is the cheaper way to evidence a contraction cell.
    operands = {"matmul": 2, "linear": 3, "movement": 1, "rmsnorm": 1, "softmax": 1, "layernorm": 1,
                "add": 2, "reduce_sum": 1, "gelu": 1, "silu": 1, "bias_add": 2}
    return (operands.get(op, 3), 0 if op in BUILDERS else 1, op)


#: Ops whose GOLDEN exists only in the block-scaled engine. An op being buildable at a dtype says
#: nothing about being gradeable at it: the golden engine is selected by the entry's dtype, and these
#: two are implemented in the block-scaled one alone. Measured: radiance's `attention` cells resolve to
#: `attention_mx` -- the cheapest attention-family op -- while the cells are fp16/bf16/f32, so
#: generation died with "no SIMT golden for op 'attention_mx'" for six cells. Forcing MX geometry onto
#: those cells would be the wrong repair twice over: their dtype is not block-scaled, and the block
#: group (32) is not the tile edge (16), so the shapes would not fit either.
_BLOCK_SCALED_GOLDEN_ONLY = frozenset({"attention_mx", "gemv_batched"})


def _is_ieee_float(dtype: str) -> bool:
    """Whether this dtype routes to the IEEE-float (``simt``) golden engine.

    Same routing the generator applies (``_entry_regime``): block-scaled -> the MX engine, fp8 -> the
    specir refmodel, IEEE fp16/bf16/f32 -> simt, anything else -> the integer engine. Asked of the
    format registry rather than listed here, so a target declaring a new float width routes without an
    edit.
    """
    from merlin.runtime.fp8_formats import canonical_float
    from merlin.targetgen.corpus_spec import is_block_scaled

    if is_block_scaled(dtype):
        return False
    try:
        return canonical_float(dtype) in ("fp16", "bf16", "f32")
    except KeyError:
        return False


#: Ops whose golden grades exactly ONE operand format, as that format's canonical float name.
#:
#: Mirrors the golden engine's own guard rather than restating a rule: the batched MX golden raises
#: unless the operand canonicalizes to ``fp8_e4m3``, and a test asserts that this map and that guard
#: agree, so the two cannot drift into two different claims about what is gradeable.
_SINGLE_FORMAT_GOLDEN = {"gemv_batched": "fp8_e4m3"}


def _canonical(dtype: str) -> str | None:
    """``dtype``'s canonical float name, or ``None`` when it has none (an integer format, say)."""
    try:
        from merlin.runtime.fp8_formats import canonical_float
        return canonical_float(str(dtype))
    except Exception:                              # noqa: BLE001 -- not a float format is a real answer
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
    from merlin.targetgen.corpus_spec import is_block_scaled

    keep = set(pool)
    if not is_block_scaled(dtype):
        keep -= _BLOCK_SCALED_GOLDEN_ONLY
    # A THIRD exclusion, and the narrowest: an op whose golden grades exactly ONE operand format. Being
    # block-scaled is not enough for these -- the batched MX golden reconstructs its reference on the
    # mx_ref datapath and refuses any format but mxfp8, so choosing it for an mxfp4 cell produced an
    # entry that built its interface and then failed at the golden, leaving a capsule directory with no
    # golden in it.
    for op, want in _SINGLE_FORMAT_GOLDEN.items():
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
    from merlin.targetgen.corpus_spec import BUILDERS
    if op in BUILDERS:
        return None
    try:
        from merlin.targetgen.capsule_source import _OP_BODIES
    except Exception:                              # noqa: BLE001 -- no bodies available
        return None
    return "pytorch" if op in _OP_BODIES else None


#: Ops whose builder materializes a RANK-3 (batched) region.
#:
#: Derived rather than assumed: it is the set of builders that emit the dialect's batched contraction op
#: (``merlin_iface.matmul_batched``), and `test_perf_encoding_family`'s sibling in the shape-axis tests
#: builds one and looks for that op, so the claim is checked rather than restated. Today it holds one
#: entry, and that IS the finding the rank axis exists to surface: every target whose manifest declares
#: batching, except the block-scaled one, declares a capability the corpus cannot yet ask for.
_BATCHED_OPS = frozenset({"gemv_batched"})

#: Ops whose builder materializes a non-default operand LAYOUT (a transposed or repacked operand).
#:
#: EMPTY, deliberately. The interface declares a contraction's weight ``[K, N]``; a quantized
#: ``nn.Linear`` stores it ``[N, K]``, and the shape check correctly refuses the pair rather than
#: loosening. Closing it means teaching the builder the transposed-RHS layout -- the iface's
#: ``resident_pack`` already carries a ``layout`` attribute -- which is a feature, not a wiring gap. The
#: builders that DO name a layout (``packed_rhs``, ``nhwc``) fix one internally; they do not offer the
#: caller a choice, so they cannot exercise a layout axis.
_LAYOUT_OPS: frozenset = frozenset()


def _sf_primitives(family: str) -> tuple[str, ...]:
    """The primitive families ``family`` decomposes into, or itself when it is one."""
    from merlin.targetgen import semantic_families as sf
    try:
        return tuple(sf.primitives_of(family)) or (family,)
    except Exception:                              # noqa: BLE001 -- an unknown family is its own primitive
        return (family,)


def op_for_shape(family: str, *, admitted_ops: set[str] | None = None, dtype: str | None = None,
                 rank: int = 2, layout: str | None = None) -> str | None:
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
        pool = admitted_ops if admitted_ops is not None else available_ops()
        batched = sorted(_BATCHED_OPS & ops_gradeable_at(str(dtype or ""), set(pool)))
        return batched[0] if batched else None
    return op


def op_for_family(family: str, *, admitted_ops: set[str] | None = None,
                  dtype: str | None = None) -> str | None:
    """The cheapest op that exercises ``family`` AND can actually be written at ``dtype``.

    WRITABILITY COMES FIRST, cost second. Ranking by cost alone picked the cheapest op in the abstract
    and then discovered no writer could express it: measured, an elementwise cell chose `gelu` -- one
    operand, no direct-MLIR builder -- over `bias_add`, which has a builder, and then failed at an fp8
    dtype the PyTorch writer cannot take. Both are elementwise_map, so the cell had a writable
    representative all along and the ranking looked past it.

    ``dtype`` omitted keeps the old ordering, for callers asking the abstract question.
    """
    pool = admitted_ops if admitted_ops is not None else available_ops()
    cands = [op for op, fam in _op_family_map().items() if fam == family and op in pool]
    if dtype is None:
        return sorted(cands, key=_cost)[0] if cands else None
    # False sorts before True, so a writable op outranks an unwritable one whatever it costs.
    ranked = sorted(cands, key=lambda op: (_writer_for({"op": op, "operand_dtype": dtype}) is None,
                                           _cost(op)))
    return ranked[0] if ranked else None


def _fused_carrier(family: str, composed_with: tuple, pool: set[str]) -> tuple[str, list[str]] | None:
    """``(op, epilogue)`` for a family the manifest admits only IN COMPOSITION.

    A family declared ``composed_with: [contraction]`` is fused-only: the eligibility oracle refuses a
    standalone capsule for it as a false fallback, so the only capsule that can ever evidence that cell
    is a contraction carrying it as an epilogue. Emitting the standalone op would produce a capsule that
    is *wrong* rather than merely weak -- this is the single case where the obvious choice is a defect.
    """
    if "contraction" not in {str(c) for c in composed_with}:
        return None
    carrier = op_for_family("contraction", admitted_ops=pool)
    if carrier is None:
        return None
    stage = {"elementwise_map": "relu", "reduction": "acc_scale"}.get(family)
    return (carrier, [stage]) if stage else None


def _tile_int(token, tile: int):
    """A tile-relative extent as an integer, or ``None`` when it cannot be resolved here."""
    if isinstance(token, int):
        return token
    text = str(token or "").strip()
    if not text:
        return None
    if text == "tile":
        return int(tile)
    mult, sep, rest = text.partition("*tile")
    if sep and not rest:
        try:
            return int(mult) * int(tile)
        except ValueError:
            return None
    base, sep, delta = text.partition("-")
    if sep and base.strip() == "tile":
        try:
            return int(tile) - int(delta)
        except ValueError:
            return None
    try:
        return int(text)
    except ValueError:
        return None


def cap_to_affordable(entry: dict, spec_doc: dict, *, extends: str = "") -> "str | None":
    """Cap ``entry`` at the loop tier when its SIZE cannot be certified. Returns the reason, or None.

    One rule for every axis, applied to the shape rather than to the axis that produced it. The tier a
    capsule may demand is a fact about how long it takes to simulate, and until now it was a property
    of which code path emitted it -- so the accumulation-depth axis capped its deep capsules while the
    memory-regime axis, emitting shapes of the same order, demanded the cycle-accurate tier from all
    of them.

    Fails OPEN, deliberately: a target with no measured history, or an extent this function cannot
    resolve, leaves the tier untouched. Capping on a cost nobody measured would shrink the certified
    corpus on a guess, which is worse than an expensive capsule somebody can see in the bill.
    """
    aff = (spec_doc.get("cert_affordability") or {})
    ceiling = aff.get("max_elements")
    if not ceiling:
        return None
    tile = ((spec_doc.get("boundaries") or {}).get("tile_edge")) or 0
    if not tile:
        return None
    dims = {k: _tile_int(entry.get(k), tile) for k in ("M", "K", "N")}
    if any(v is None for v in dims.values()):
        return None
    # WRITTEN OUTPUT, which is what the cost law was calibrated on -- see `_cert_affordability`. The
    # reduction depth is deliberately absent from this product: a capsule drains `M x N` elements
    # whatever `K` is, so a deep accumulation is cheap to certify and must not be capped as though it
    # were large. Pricing by operand size instead (the first attempt) capped exactly the capsules most
    # worth certifying.
    elements = dims["M"] * dims["N"]
    if elements <= int(ceiling):
        return None
    entry["max_oracle_tier"] = "L2"
    if extends:
        entry["extends"] = extends
    return (f"{elements} operand elements exceeds the {int(ceiling)} a {aff.get('budget_s')}s "
            f"certification budget affords on this target, so it is graded at the loop tier and "
            f"rests on {extends!r}" if extends else
            f"{elements} operand elements exceeds the {int(ceiling)} a {aff.get('budget_s')}s "
            f"certification budget affords on this target, so it is graded at the loop tier")


def extents_for(alignment: str, probes: list[dict]) -> dict[str, str]:
    """Tile-relative extents for an alignment, spelled so the entry stays geometry-free.

    ``aligned`` sits on the boundary; ``partial`` rags exactly one axis by one element.
    ``cert_capsule_cover`` marks a capsule partial when ANY extent leaves a remainder, so ragging one
    axis is sufficient and keeps the working set predictable -- ragging all three would multiply the
    padding without covering anything more.

    Spelled ``tile`` / ``tile-1`` rather than resolved integers so the SAME entry describes the same
    shape on a target with a different edge; ``generate_corpus.resolve_extent`` parses the tokens
    structurally.
    """
    if not probes:
        raise SynthesisError(
            "no extent probe: this target's boundaries carry no tile edge, so an alignment-indexed "
            "requirement cannot be turned into a shape. Fix the boundary derivation rather than "
            "guessing an edge")
    if alignment == "partial":
        return {"M": "tile", "K": "2*tile", "N": "tile-1"}
    if alignment == "sub_tile":
        # BARELY OCCUPIED, which `partial` (ragged by one) cannot express. Spelled tile-relative like
        # every other extent, so a target with a 64-wide edge gets 16/32/16 from the same entry. K
        # stays at tile//2 rather than dropping to tile//4 so the capsule still asks for a real
        # reduction: a single-pass contraction would exercise accumulation not at all, and every other
        # capsule already carries one.
        return {"M": "tile/4", "K": "tile/2", "N": "tile/4"}
    return {"M": "tile", "K": "2*tile", "N": "tile"}


def filtered_precision(preference: list[str], admitted: set[str]) -> tuple[list[str], list[str]]:
    """``(kept, dropped)`` -- the declared preference order, filtered against what the target admits.

    A preference RANKS what the hardware already has; it can never widen it. A token that does not
    survive is reported rather than dropped silently, because "we preferred int8 and the target has no
    int8 datapath" is a fact the reader of a synthesized corpus needs.
    """
    kept = [d for d in preference if d in admitted]
    return kept, [d for d in preference if d not in admitted]


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
    """
    from merlin.xdsl_dialects.lowering import passes as _P

    out = [_P.TARGET_ISA_LOWERING]
    tile = 0
    for probe in ((spec_doc.get("boundaries") or {}).get("extent_probes") or ()):
        tile = max(tile, int(probe.get("edge") or 0))
    for axis in ("M", "K", "N"):
        token = str(entry.get(axis) or "")
        # `2*tile` and friends exceed the edge by construction; a resolved integer is compared directly.
        if token.startswith(("2*", "4*", "8*")) or (token.isdigit() and tile and int(token) > tile):
            out.append(_P.TILE_SCHEDULE)
            break
    lanes = entry.get("lanes") or {}
    if lanes.get("forbid") or any(str(l) != "on_mesh" for l in (lanes.get("require") or ())):
        out.append(_P.HOST_SEAM)
    if str(entry.get("kind")) == "model" or len(lanes.get("require") or ()) > 1:
        out.append(_P.REGION_PARTITION)
    return sorted(dict.fromkeys(out))


#: Dtype regimes the PyTorch writer can express. It grades a host-eager float reference with tolerance,
#: so an int or block-scaled datapath needs the direct-MLIR engine instead -- which is exactly why an op
#: with no builder cannot be written at those dtypes by anyone.
_PYTORCH_REGIMES = ("simt",)


def _writer_for(entry: dict) -> str | None:
    """Which writer can materialize this entry -- ``builder`` / ``pytorch`` / ``None`` for neither.

    ``available_ops`` is the union of the direct-MLIR BUILDERS and the PyTorch bodies, so an op can be
    "materializable" in the abstract and still have no writer AT THIS DTYPE: the PyTorch path is float
    only. A cell in that corner is a real capability gap and must be reported as an uncovered cell to
    argue about, never emitted as an entry nothing can write.
    """
    from merlin.targetgen.corpus_spec import BUILDERS, regime_for_dtype

    op = entry.get("op")
    if op in BUILDERS:
        return "builder"
    try:
        regime = regime_for_dtype(str(entry.get("operand_dtype") or ""))
    except Exception:                              # noqa: BLE001 -- an unknown regime is not a float one
        regime = None
    return "pytorch" if regime in _PYTORCH_REGIMES else None


def _mark_source(entry: dict) -> None:
    """Say which writer can materialize this entry's op, when it is not the direct-MLIR one.

    `available_ops` is the union of the direct-MLIR BUILDERS and the PyTorch bodies, so an op can be
    materializable and still have no builder -- and an entry that does not say so goes down the
    direct-MLIR path and dies with "no corpus builder for op". Measured: four synthesized capsules
    failed exactly that way on the first target whose elementwise family resolved to `gelu`.
    """
    from merlin.targetgen.corpus_spec import BUILDERS

    op = entry.get("op")
    if op and op not in BUILDERS:
        entry["source"] = "pytorch"


def synthesize(spec_doc: dict, *, workload_spec: dict | None = None,
               budget: int | None = None) -> dict:
    """``{"capsules": [entry...], "provenance": {...}}`` for one target's derived requirement.

    Pure: no I/O, no target name in the control flow. The spec is the derived conformance document, so
    everything about the target reaches this function as data.
    """
    target = str(spec_doc.get("target") or "")
    cells = list(spec_doc.get("cells") or ())
    probes = list((spec_doc.get("boundaries") or {}).get("extent_probes") or ())
    ws = dict(workload_spec or {})
    pool = available_ops()

    admitted_dtypes = {str(c.get("dtype")) for c in cells if c.get("dtype")}
    # A preference is declared in REGISTRY spelling ("int8"); a cell carries the CAPSULE spelling
    # ("i8"). Comparing them raw reported every token as dropped and the whole preference as inert.
    from merlin.targetgen.conformance import capsule_dtype
    _pref = []
    for tok in (ws.get("precision_preference") or ()):
        try:
            _pref.append(capsule_dtype(str(tok)))
        except Exception:                          # noqa: BLE001 -- an unmappable token is not a dtype
            _pref.append(str(tok))
    kept, dropped = filtered_precision(_pref, admitted_dtypes)

    cap_map: dict[str, Any] = {}
    try:
        from merlin.targetgen.eligibility import capability_map_for_target
        cap_map = capability_map_for_target(target) or {}
    except Exception:                              # noqa: BLE001 -- no map means no composed-with facts
        cap_map = {}

    entries: list[dict] = []
    unexpressable: list[str] = []
    unwritable: list[str] = []
    #: Shape-axis regions no writer materializes. SEPARATE from `unwritable`, which is about CELLS: the
    #: two are different claims and merging them would weaken a standing guarantee. Every cell every
    #: target requires currently has a writer, and a test asserts exactly that; folding a declared-but-
    #: unbuildable batched region into the same list would have made that assertion start failing for a
    #: reason it was never about.
    shape_unwritable: list[str] = []
    for cell in sorted(cells, key=lambda c: str(c.get("cell"))):
        family = str(cell.get("family") or "")
        dtype = str(cell.get("dtype") or "")
        alignment = str(cell.get("alignment") or "aligned")
        # `composed_with` is a tuple ATTRIBUTE, not a method. Calling it raised TypeError, a broad
        # `except` swallowed it, and every family looked standalone -- so the fused-only case silently
        # took the branch that produces a capsule the eligibility oracle refuses. Read the attribute,
        # and let anything unexpected surface.
        cap = cap_map.get(family)
        composed = tuple(getattr(cap, "composed_with", ()) or ()) if cap is not None else ()

        epilogue: list[str] = []
        fused = _fused_carrier(family, composed, ops_gradeable_at(dtype, pool)) if composed else None
        if fused is not None:
            op, epilogue = fused
        else:
            # The CELL'S dtype narrows the pool twice over, and both narrowings are needed: what the
            # GOLDEN ENGINE can grade at this dtype (ops_gradeable_at) and, within that, what a WRITER
            # can express (the dtype-aware ranking). Buildable is not gradeable and gradeable is not
            # writable; choosing on any one of the three alone produced an entry that died in the other.
            op = op_for_family(family, admitted_ops=ops_gradeable_at(dtype, pool), dtype=dtype)
        if op is None:
            unexpressable.append(f"{cell.get('cell')} (family {family!r})")
            continue

        name = f"{SYNTH_PREFIX}_{family}_{dtype}_{alignment}".replace("-", "_")
        entry: dict[str, Any] = {
            "cat": "isa", "kind": "isa", "name": name, "op": op,
            "operand_dtype": dtype, "out": "Y0", "lhs": "A0", "weight": "W",
            "source_role": SOURCE_ROLE,
            "source_reference": (
                f"synthesized for conformance cell {cell.get('cell')} "
                f"(basis={cell.get('basis')}, admitted_by={list(cell.get('admitted_by') or ())}); "
                f"extents from boundaries.extent_probes"
                + (f"; {family} is admitted only in composition with {list(composed)}, so it is carried "
                   f"as a fused epilogue on {op} -- a standalone capsule for it would be refused by the "
                   f"eligibility oracle as a false fallback" if fused else "")),
            "label": "public",
            "modes": {},
            **extents_for(alignment, probes),
        }
        if epilogue:
            entry["epilogue"] = epilogue
        _writer = _writer_for(entry)
        if _writer is None:
            unwritable.append(
                f"{cell.get('cell')} (op {op!r} has no direct-MLIR builder and dtype {dtype!r} is not "
                f"a regime the PyTorch writer can express)")
            continue
        if _writer == "pytorch":
            entry["source"] = "pytorch"
        entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
        entries.append(entry)

    # ---- the MEMORY-REGIME axis -------------------------------------------------------------------
    # A family/dtype/alignment cell says what arithmetic the corpus must contain; it says nothing about
    # the residency that arithmetic asks of the operand store. Measured on one target: 90.1% of 1829
    # real contraction regions spill the store while 100% of the public capsules fit it twice over, so
    # the corpus could not detect a memory-mapping failure of any kind. These entries close that.
    #
    # The extents are read from the spec, not computed here: the search needs the target's address
    # space, and this module is pure. A regime the spec resolves to null is one no capsule shape reaches
    # on this target -- recorded in provenance rather than dropped, because a silently absent regime
    # reads downstream as a covered one.
    mem_block = spec_doc.get("memory_mapping") or {}
    # A spec written before this axis existed carries no `regime_extents` AT ALL. That is a stale spec,
    # not an unreachable regime, and reporting it as "no capsule shape reaches this" would state a
    # measurement nobody made -- the precise failure this module exists to avoid.
    regime_extents = mem_block.get("regime_extents")
    regimes_resolved = regime_extents is not None
    regime_extents = regime_extents or {}
    regime_dtype = str(mem_block.get("regime_dtype") or "") or (
        sorted(admitted_dtypes)[0] if admitted_dtypes else "")
    unreachable_regimes: list[str] = []
    regime_op = op_for_family("contraction", admitted_ops=pool)
    for regime in sorted((mem_block.get("required") or {}) if regimes_resolved else {}):
        ext = regime_extents.get(regime)
        if not ext:
            unreachable_regimes.append(str(regime))
            continue
        if regime_op is None or not regime_dtype:
            unreachable_regimes.append(str(regime))
            continue
        entry = {
            "cat": "isa", "kind": "isa", "name": f"{SYNTH_PREFIX}_regime_{regime}",
            "op": regime_op, "operand_dtype": regime_dtype,
            "out": "Y0", "lhs": "A0", "weight": "W",
            "source_role": SOURCE_ROLE,
            "source_reference": (
                f"synthesized for memory regime {regime!r}: the declared inputs occupy "
                f"{ext.get('rows')} of {ext.get('capacity_rows')} operand-store rows "
                f"({ext.get('fraction_of_capacity')} of capacity), which is what puts this capsule in "
                f"that regime. Extents resolved by memory_regime.extents_for_regime with the same "
                f"sizing the coverage gate measures with"),
            "label": "public", "modes": {},
            "M": ext["M"], "K": ext["K"], "N": ext["N"],
        }
        # A REGIME THAT FILLS THE STORE CANNOT BE CERTIFIED, and saying so is what keeps the corpus
        # runnable. Measured on gemmini: `fits_single` and `spills` resolve to 131k and 262k operand
        # elements against a fitted history topping out at 4096 -- roughly 6.6 hours of extrapolated
        # simulation between them, demanded at the cycle-accurate tier by an axis that never asked
        # what it cost. They stay in the corpus at the loop tier, resting on the certified cell
        # capsule that proves the same arithmetic at a size somebody can afford.
        _sibling = f"{SYNTH_PREFIX}_contraction_{regime_dtype}_aligned"
        _why = cap_to_affordable(entry, spec_doc, extends=_sibling)
        if _why:
            entry["source_reference"] += f". {_why}"
        entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
        entries.append(entry)

    # ---- the NEGATIVE lane ------------------------------------------------------------------------
    # Families a real capture CONTAINS and this target's manifest does NOT admit. The compiler must leave
    # them on the host, and accelerating one is as much a defect as failing to accelerate an admitted
    # family -- but the requirement derived that set into `host_only` and nothing ever demanded it, so
    # the negative lane was covered only by whatever a hand-authored capsule happened to assert.
    #
    # These declare `forbid` alone. Requiring the host lane as well would add a demand no op-path grade
    # can measure, and an unmeasurable requirement turns the capsule into a permanent `incomplete`
    # instead of a test. The forbid IS enforceable: a decoded accelerator instruction violates it.
    _diag = dict(spec_doc.get("diagnostics") or {})
    host_block = spec_doc.get("host_only") or {}
    host_dtypes = dict(host_block.get("dtypes") or {})
    unsized_host: list[str] = []
    # THE FAMILY BEING UNADMITTED IS NOT ENOUGH. A capsule proves the negative lane only if NOTHING in
    # its program may be accelerated, and a composite family DECOMPOSES: `normalization` is a reduction
    # and an elementwise map, so on a target admitting either, a normalization capsule contains regions
    # the hardware legitimately takes. Measured: every `SY_host_only_normalization` this axis emitted
    # classifies `A` under `boundary.profile_capsule`, so the coverage gate correctly declined to count
    # it -- gemmini's negative lane was covered by a HAND-AUTHORED capsule the whole time, and atlas's
    # was not covered at all while a synthesized capsule sat there asserting it.
    #
    # Worse than uncounted: `forbid: [on_mesh]` on such a program is a claim the submission must NOT do
    # something it is entitled to do, so a compiler that correctly accelerates the admitted elementwise
    # regions would be recorded as violating the lane.
    # WIDER THAN `host_only`, AND KEYED ON THE PAIR. `host_only` carries families with no admitted dtype
    # at all; `host_lane` carries every (family, dtype) the captures contain and the manifest does not
    # admit -- which on a target admitting one narrow format is most of a real model. Measured on
    # gemmini: four admitted families, every one of them ALSO present at f32, 10,719 regions of host work
    # the requirement demanded no capsule of.
    #
    # Whether such a capsule can honestly forbid the mesh is NOT decided here. It is a property of the
    # written program -- `rmsnorm` classifies accelerator at both f32 and bf16 where `layernorm`
    # classifies host-only at bf16, so the op decides it and the family does not -- and the generator
    # classifies the capsule and refuses a forbid the classification will not support. A family-level
    # rule tried here first and was wrong in both directions: it blocked `normalization` on gemmini,
    # where a layernorm capsule proves the lane, and passed it on a target whose rmsnorm program
    # contains an eligible region anyway.
    for _pair in ((spec_doc.get("host_lane") or {}).get("required") or ()):
        family, dtype = str(_pair.get("family") or ""), str(_pair.get("dtype") or "")
        if not family or not dtype or family in {str(f) for f in (host_block.get("families") or ())}:
            continue                               # the narrow axis below already carries this family
        op = op_for_family(family, admitted_ops=pool, dtype=dtype)
        if op is None:
            unsized_host.append(f"{family}/{dtype} (no materializable op at this dtype)")
            continue
        entry = {
            "cat": "model_slices", "kind": "model_slice",
            "name": f"{SYNTH_PREFIX}_host_lane_{family}_{dtype}".replace("-", "_").replace(".", "_"),
            "op": op, "operand_dtype": dtype, "out": "Y0", "lhs": "A0", "weight": "W",
            "source_role": SOURCE_ROLE,
            "source_reference": (
                f"synthesized for the host lane: real captures contain {_pair.get('n_regions')} "
                f"{family!r} region(s) at {dtype}, and this target's manifest admits {family!r} at no "
                f"such dtype -- so every one of them must be placed on the host. A corpus with no "
                f"capsule here cannot tell a compiler that routes them correctly from one that does not"),
            "label": "public", "modes": {},
            "lanes": {"forbid": ["on_mesh"]},
            "semantic": {"must_accelerate": False, "eligible": False,
                         "generalization_axis": "host_lane",
                         "not_asserted_reason": (
                             "the target admits no datapath for this family at this dtype; the capsule "
                             "exists to prove the compiler does NOT accelerate it")},
            **extents_for("aligned", probes),
        }
        # AN OP WITH A `merlin_iface` BUILDER CANNOT SERVE THIS AXIS, whatever dtype it declares.
        # `merlin_iface` is the ACCELERATOR's interface dialect -- every program expressible in it is
        # accelerator work by construction -- and the capsule writer derives an iface interface for any
        # op that has a builder, even on the frontend path. The boundary classifier reads that interface,
        # so the capsule comes out `A`. Measured: a matmul capsule declaring genuine f32 operands
        # (tensor<16x32xf32>) still classified accelerator, while `gelu` and `reduce_sum` -- which have
        # no iface builder, so their interface stays linalg and eligibility is judged dtype-aware --
        # classified host-only.
        #
        # That is what `_writer_for` already means by "pytorch": an op with no builder. Reported rather
        # than written another way, because there IS no other way: covering host-lane contraction needs a
        # contraction op the interface dialect cannot express, which the corpus does not yet have.
        if _writer_for(entry) != "pytorch":
            unsized_host.append(
                f"{family}/{dtype} (op {op!r} has a merlin_iface builder, so its capsule carries an "
                f"accelerator-dialect interface and classifies as accelerator work whatever dtype it "
                f"declares; this axis needs an op of this family the interface dialect cannot express)")
            continue
        entry["source"] = "pytorch"
        entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
        entries.append(entry)

    for family in sorted(str(f) for f in (host_block.get("families") or ())):
        dtype = host_dtypes.get(family)
        op = op_for_family(family, admitted_ops=pool, dtype=dtype)
        if op is None or not dtype:
            unsized_host.append(f"{family} ({'no materializable op' if op is None else 'no observed dtype'})")
            continue
        entry = {
            "cat": "model_slices", "kind": "model_slice",
            "name": f"{SYNTH_PREFIX}_host_only_{family}".replace("-", "_"),
            "op": op, "operand_dtype": dtype, "out": "Y0", "lhs": "A0", "weight": "W",
            # PyTorch-sourced regardless of whether a direct-MLIR builder exists for the op: a host-only
            # capsule is a model slice at a float dtype, and the frontend-faithful lowering is what makes
            # it the program a real model would hand the host.
            "source": "pytorch",
            "source_role": SOURCE_ROLE,
            "source_reference": (
                f"synthesized for host-only family {family!r}: real captures contain it and this "
                f"target's capability manifest admits no capability for it, so the compiler must leave "
                f"it on the host lane. dtype {dtype} is the one the captures carry for this family"),
            "label": "public", "modes": {},
            "lanes": {"forbid": ["on_mesh"]},
            "semantic": {"must_accelerate": False, "eligible": False,
                         "not_asserted_reason": (
                             "the target declares no capability for this family; the capsule exists to "
                             "prove the compiler does NOT accelerate it")},
            **extents_for("aligned", probes),
        }
        entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
        entries.append(entry)

    # ---- the COMPOSITION axis -----------------------------------------------------------------------
    # The requirement derives which compositions real captures CONTAIN -- an isolated dispatch, adjacent
    # accelerator pairs, a host island, a full routing alternation -- and nothing consumed the set, so no
    # capsule anywhere demanded a host round trip. That is the shape a whole-model compiler actually gets
    # wrong, because it is the one where keeping an intermediate resident and paying to move it out
    # differ.
    #
    # ONE entry, not one per shape. The micro model's composition is whatever its derived layer inventory
    # produces once host layers are interleaved into the interior; emitting a capsule per declared shape
    # would mean inventing inventories no capture supports. The shapes this does not reach are reported.
    composition = dict((spec_doc.get("composition") or {}).get("required") or {})
    if composition:
        entry = {
            "cat": "model", "kind": "model", "name": f"{SYNTH_PREFIX}_micro_model",
            "micro_model": True,
            "source_role": SOURCE_ROLE,
            "source_reference": (
                "synthesized for the composition axis: one layer per admitted family, one per family a "
                "real capture contains that the manifest does not admit, sized to the target's own tile "
                "edge, with host layers interleaved into the interior. Inventory and extents come from "
                f"micro_model.spec; the shapes the requirement names are {sorted(composition)}"),
            "label": "public", "modes": {},
            "lanes": {"require": ["on_mesh"]},
            "semantic": {"generalization_axis": "composition"},
        }
        _mark_source(entry)
        entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
        entries.append(entry)

    # ---- the SHAPE-GENERALIZATION axis --------------------------------------------------------------
    # Rank and operand layout: the two things a `(family, dtype, alignment)` cell cannot say. A target's
    # manifest DECLARES that a unit batches, or transposes, or accepts a packed layout; `capability_probes`
    # has always turned those declarations into region descriptors, and nothing outside the fuzzer read
    # them -- so a target could claim batching and never be asked for a batched region by anything that
    # grades it.
    #
    # Reported, not raised, when no writer exists. A batched contraction is writable only through the
    # block-scaled builder today, and a transposed RHS needs a builder that does not exist (the interface
    # declares the weight [K, N] while a quantized `nn.Linear` stores it [N, K], which is a real feature
    # rather than a check to loosen). Those are capability gaps to argue about with the probe NAMED, and
    # the alternative -- leaving the axis out entirely, as before -- is the one that hides them.
    for req in ((spec_doc.get("shape_generalization") or {}).get("required") or ()):
        axis = str(req.get("axis") or "")
        family = str(req.get("family") or "")
        dtype = str(req.get("dtype") or "")
        probe = str(req.get("probe") or f"{family}.{axis}")
        # THE PROBE'S DTYPE FIRST, THEN THE TARGET'S OTHER ADMITTED ONES. This axis is about the SHAPE
        # capability -- can the unit be asked for a batched region at all -- so settling for a different
        # admitted format is a fair way to exercise it, where giving up would report a hole the target
        # does not have. `capability_probes` names the family's LEAD dtype, and on a multi-format target
        # that is not always the one a writer covers: mx-gemmini leads with mxfp4 while the batched
        # golden grades mxfp8 only, so the region is reachable and the lead dtype alone said it was not.
        _family_dtypes = [dtype] + sorted(
            {str(c.get("dtype")) for c in cells
             if c.get("dtype") and str(c.get("family")) == family and str(c.get("dtype")) != dtype})
        op, chosen_dtype = None, dtype
        for _dt in _family_dtypes:
            op = op_for_shape(family, admitted_ops=pool, dtype=_dt,
                              rank=int(req.get("rank") or 2), layout=req.get("layout"))
            if op is not None:
                chosen_dtype = _dt
                break
        dtype = chosen_dtype
        if op is None:
            shape_unwritable.append(
                f"{probe} ({axis} axis, family {family!r}, dtype {dtype!r}): no builder materializes a "
                f"{'rank-' + str(req.get('rank')) if axis == 'rank' else str(req.get('layout')) + '-layout'} "
                f"region for this family at this dtype")
            continue
        entry = {
            "cat": "model_slices", "kind": "model_slice",
            "name": f"{SYNTH_PREFIX}_{axis}_{probe.replace('.', '_')}",
            "op": op, "operand_dtype": dtype, "out": "Y0", "lhs": "A0", "weight": "W",
            "source_role": SOURCE_ROLE,
            "source_reference": (
                f"synthesized for the {axis} axis: this target's capability manifest declares "
                f"{family!r} handles a {probe.split('.')[-1]} region, and a (family, dtype, alignment) "
                f"cell cannot demand one. Shape and layout come from capability_probes"),
            "label": "public", "modes": {},
            "semantic": {"generalization_axis": axis},
        }
        # EXTENTS COME FROM THE BUILDER, NOT THE PROBE. This axis asks whether the unit can be asked for
        # a batched or transposed region at all; WHICH extent is the alignment axis's question, and the
        # cells already carry it. The probe's tile-relative shape is a reasonable default for a plain
        # 2-D region and not necessarily a legal one for the op that expresses this region: the batched
        # MX golden needs its contraction dim a multiple of 32 where the probe offers one tile, so
        # passing the probe's extents through produced an entry that built its interface and then failed
        # in the golden. The builder derives a legal shape from the same tile edge the probe used.
        if int(req.get("batch") or 1) > 1:
            # `B` is the key the batched builder reads. Spelling it `batch` produced an entry that built
            # at the builder's DEFAULT batch instead of the probe's, so the capsule would have carried a
            # batch nobody derived while looking like it carried the declared one.
            entry["B"] = int(req["batch"])
        if req.get("layout"):
            entry["layout"] = str(req["layout"])
        _mark_source(entry)
        entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
        entries.append(entry)

    # ---- the EPILOGUE axis --------------------------------------------------------------------------
    # WHICH stage rides the contraction. The cells already say that a fused-only family must be carried
    # as an epilogue rather than standalone -- but they cannot say WHICH one, so a corpus derived from
    # cells alone picks a single representative (gemmini got `relu` for elementwise_map and `acc_scale`
    # for reduction) and reports the fusion capability covered. The hand-authored corpus tests four
    # stages and their combinations, which is the level this axis is for.
    #
    # The requirement side evidences each stage from the manifest or from the target's own instruction
    # taxonomy; here we only have to write one contraction per stage.
    for _st in ((spec_doc.get("epilogue") or {}).get("required") or ()):
        stage = str(_st.get("stage") or "")
        if not stage:
            continue
        dtype = kept[0] if kept else (sorted(admitted_dtypes)[0] if admitted_dtypes else "")
        if not dtype:
            unexpressable.append(f"epilogue axis: no admitted dtype to build a {stage!r} stage at")
            continue
        entry = {
            "cat": "layers", "kind": "layer",
            "name": f"{SYNTH_PREFIX}_epilogue_{stage}",
            "op": "matmul", "operand_dtype": dtype,
            "lhs": "A0", "weight": "W", "out": "Y0",
            "epilogue": [stage],
            "source_role": SOURCE_ROLE,
            "source_reference": (
                f"synthesized for the epilogue axis: this target can fuse a {stage!r} stage onto a "
                f"contraction (evidenced by {_st.get('evidenced_by')}), and a (family, dtype, "
                f"alignment) cell cannot demand a particular stage -- so without this the capability is "
                f"reported covered by whichever single stage the cell axis happened to pick"),
            "label": "public", "modes": {},
            "semantic": {"generalization_axis": "epilogue"},
            **extents_for("aligned", probes),
        }
        # A POOLING STAGE COMMITS FEWER ROWS THAN IT READS, so the builder needs the geometry: without
        # `pool_in_dims` it cannot know what the contraction's rows mean spatially. It is NOT derived
        # here -- the extents at this point are tile-relative TOKENS (`tile`, `2*tile`), because the
        # whole reason synthesis writes them that way is that it does not know the target's edge. The
        # generator resolves them against the edge and derives the geometry there, which is the only
        # place both facts are in hand.
        if stage == "maxpool":
            entry["pool_size"] = [2, 2]
            entry["pool_stride"] = [2, 2]
        entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
        entries.append(entry)

    # ---- the ACCUMULATION-DEPTH axis ----------------------------------------------------------------
    # How deep a reduction the accumulator survives. The memory-regime axis proves the operand store is
    # LOADED to each residency level; this proves the ACCUMULATION that fills it completes -- different
    # questions, and only the first was derived. The hand-authored corpus tests K_tiles 1, 2 and 4 by
    # hand precisely because a single K says nothing about a multi-pass reduction.
    #
    # THE TIER SPLIT IS THE WHOLE POINT HERE. Filling the operand store takes thousands of elements of
    # K (measured on gemmini: 65k-262k operand elements against a certification history topping out at
    # 4096), so a residency-depth capsule is not something anyone can certify cycle-accurately. The
    # requirement side therefore sizes ONE certified point -- the deepest K the measured cost model puts
    # inside the budget -- and the residency points ride on it as L2-only perf extensions. That ordering
    # is load-bearing: emit the certified capsule first, and admit a deeper one only when it exists, so
    # a large capsule can never enter the corpus resting on nothing.
    _depth = ((spec_doc.get("memory_mapping") or {}).get("reduction_depth") or {})
    _depth_dtype = ((spec_doc.get("memory_mapping") or {}).get("regime_dtype")
                    or (sorted(admitted_dtypes)[0] if admitted_dtypes else ""))
    _depth_op = (op_for_family("contraction", admitted_ops=pool, dtype=_depth_dtype)
                 if _depth_dtype else None)
    _certified_depth_name = ""
    #: Depths this target could not size, kept OUT of `cells_no_writer_can_express`. That list means
    #: "a required cell has no writer" -- a capability gap in the corpus. A target with no measured
    #: certification history has no such gap: the writer exists and the evidence to SIZE it does not,
    #: which licenses a completely different action (run a cert and refit, not build a new builder).
    unsized_depth: list[str] = []

    def _depth_entry(name, point, *, tier, extends="", why=""):
        # TILE-RELATIVE, like every other sweep. The axis is about the NUMBER OF ACCUMULATION PASSES,
        # so `11*tile` says what the capsule is for in a way `176` does not -- and it keeps the entry
        # portable in spelling, which is the property `derived_sweep` promises. Only the application
        # axis bakes integers, because only there is the shape a model's rather than the target's.
        e = {
            "cat": "layers", "kind": "layer", "name": name,
            "op": _depth_op, "operand_dtype": _depth_dtype,
            "lhs": "A0", "weight": "W", "out": "Y0",
            "M": "tile", "K": f"{int(point['K_tiles'])}*tile", "N": "tile",
            "source_role": SOURCE_ROLE, "source_reference": why,
            "label": "public", "modes": {},
            "semantic": {"generalization_axis": "accumulation_depth"},
        }
        if tier != "L3":
            e["max_oracle_tier"] = tier
            e["extends"] = extends
        _mark_source(e)
        e["pass_requirements"] = pass_requirements_for(e, spec_doc)
        return e

    if _depth_op is None and (_depth.get("certified") or _depth.get("by_regime")):
        unsized_depth.append(f"no op materializes a contraction at {_depth_dtype!r}")
    elif _depth_op is not None:
        # THE ANCHOR IS A FALLBACK, NOT A FIXTURE. It guarantees a multi-pass reduction somewhere on
        # targets whose residency regimes yield no depth at all. Where the regimes DO yield one that
        # certifies -- which, since a reduction writes one output tile, is the normal case -- the
        # anchor is a strictly shallower duplicate of a capsule already in the corpus, and shipping it
        # would be one more certification buying nothing.
        _regime_depths = [pt for blk in (_depth.get("by_regime") or {}).values()
                          for pt in (blk.get("points") or ()) if pt.get("K")]
        _cert_pt = _depth.get("certified") if not _regime_depths else None
        if _cert_pt:
            _certified_depth_name = f"{SYNTH_PREFIX}_kdepth_certified"
            entries.append(_depth_entry(
                _certified_depth_name, _cert_pt, tier="L3",
                why=(f"synthesized for the accumulation-depth axis: {_cert_pt['K_tiles']} accumulation "
                     f"passes, the shallowest reduction that writes the accumulator more than once. "
                     f"Emitted only because this target's residency regimes yield no depth of their "
                     f"own; where they do, they produce deeper capsules and this would duplicate one. "
                     f"Costs {_cert_pt['predicted_seconds']}s against a {_cert_pt['budget_s']}s "
                     f"budget -- priced on the OUTPUT tile, which the reduction depth does not move")))
        elif _depth.get("certified_refusal"):
            unsized_depth.append(str(_depth["certified_refusal"]))

        for _regime, _blk in sorted((_depth.get("by_regime") or {}).items()):
            _points = [pt for pt in (_blk.get("points") or ()) if pt.get("K")]
            if not _points:
                continue
            _deep = max(_points, key=lambda pt: int(pt["K"]))
            # TIER DECIDED BY THE COST LAW, not asserted by this axis. These capsules were capped at
            # L2 on the belief that a deep reduction is expensive, which the operand-size metric
            # implied and the measured law refutes: a capsule drains `M x N` elements whatever `K` is,
            # so the deepest reduction the store admits writes ONE TILE and certifies for about the
            # price of the shallowest. That is the sweet spot this axis was looking for -- maximum
            # accumulation depth at minimum simulation cost -- and capping it would have thrown away
            # the cycle-accurate guarantee on the very behaviour hardest to get right.
            _e = _depth_entry(
                f"{SYNTH_PREFIX}_kdepth_{_regime}", _deep, tier="L3",
                why=(f"synthesized for the accumulation-depth axis: the deepest reduction the "
                     f"{_regime!r} residency regime admits ({_deep['K_tiles']} accumulation passes, "
                     f"{_deep.get('fraction_of_capacity')} of the operand store). The reduction moves "
                     f"the operands and not the result, so this certifies at one output tile"))
            _why = cap_to_affordable(
                _e, spec_doc,
                extends=(_certified_depth_name
                         or f"{SYNTH_PREFIX}_contraction_{_depth_dtype}_aligned"))
            if _why:
                _e["source_reference"] += f". {_why}"
                _e["pass_requirements"] = pass_requirements_for(_e, spec_doc)
            entries.append(_e)

    # ---- the APPLICATION axis -----------------------------------------------------------------------
    # The only axis whose capsules carry a shape a real model CONTAINS rather than a tile multiple.
    # Everything else here is tile-relative on purpose, so the same entry means the same thing on a
    # target with a different edge; an application shape is the opposite by design -- it is evidence
    # about one model, and it is not portable, which is why it carries its own provenance.
    #
    # The requirement side has already done the hard part: grouped the application's regions by what
    # the compiler must do with them, and sized each class against what a certification costs. Here we
    # only turn each sized capsule into an entry. A class that could not be sized is not in `required`
    # at all -- it is in the axis's `refused` list with its reason, which is reported rather than
    # raised because an unaffordable behaviour is a fact about the budget, not a broken corpus.
    _app = spec_doc.get("application_shapes") or {}
    #: Classes that actually got a cycle-accurate entry. An L2-only capsule is admissible ONLY as an
    #: extension of one, and the sizing side hands them over in pairs -- but a pair is not a
    #: guarantee here: if the deeper entry's op cannot be materialized we skip it, and iterating the
    #: list flat would then still emit its L2 partner, resting on nothing. That is precisely the
    #: failure the `extends` relation exists to prevent, arriving through the back door.
    _certified_classes: set = set()
    for _cap in (_app.get("required") or ()):
        _cls = str(_cap.get("class") or "")
        _tier = str(_cap.get("tier") or "L3")
        _dtype = _cls.split("/")[1] if "/" in _cls else ""
        if not _dtype:
            continue
        _op = op_for_family("contraction", admitted_ops=pool, dtype=_dtype)
        if _op is None:
            unwritable.append(f"application class {_cls}: no op materializes a contraction at {_dtype!r}")
            continue
        if _tier != "L3" and _cls not in _certified_classes:
            unwritable.append(
                f"application class {_cls}: its L2 capsule extends a cycle-accurate sibling that was "
                f"not emitted, so it would rest on nothing; dropped rather than shipped as a large "
                f"capsule nothing certifies")
            continue
        _slug = _cls.replace("/", "_").replace("-", "_")
        entry = {
            "cat": "layers", "kind": "layer",
            "name": f"{SYNTH_PREFIX}_app_{_slug}_{_tier.lower()}",
            "op": _op, "operand_dtype": _dtype,
            "lhs": "A0", "weight": "W", "out": "Y0",
            "M": int(_cap["M"]), "K": int(_cap["K"]), "N": int(_cap["N"]),
            # NOT `derived_sweep`: a sweep's shapes track the target's geometry, and this one tracks a
            # model's. Conflating them would tell a reader the extents move with the tile edge.
            "source_role": "model_derived",
            "source_reference": (
                f"synthesized for the application axis: behavioural class {_cls}, representing "
                f"{_cap['basis'].get('representative_of')} region(s) of "
                f"{_cap['basis'].get('source')}. Sized by {_cap['basis'].get('sized_by')}"
                + (f" against a {_app.get('cert_budget_s')}s certification budget"
                   if _cap["basis"].get("sized_by") == "measured_cost_model" else "")
                + (f"; extends {_cap['extends']}, which carries the cycle-accurate guarantee this "
                   f"larger shape rests on" if _cap.get("extends") else "")),
            "label": "public", "modes": {},
            "semantic": {"generalization_axis": "application"},
        }
        if _tier != "L3":
            # AN L2-ONLY CAPSULE IS AN EXTENSION, NEVER A SUBSTITUTE. The tier is capped because this
            # shape is too large to certify, and `extends` names the sibling that was -- so a reader
            # (and the gate) can tell a large capsule resting on a guarantee from one resting on
            # nothing.
            entry["max_oracle_tier"] = _tier
            entry["extends"] = str(_cap.get("extends") or "")
        _mark_source(entry)
        entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
        entries.append(entry)
        if _tier == "L3":
            _certified_classes.add(_cls)

    # ---- the ROSTER axis ----------------------------------------------------------------------------
    # The declared roster is the one thing the workload spec says that nothing consumed. Every capsule
    # above is a SLICE -- a cell, a regime, a lane, a derived micro model -- and the claim the whole
    # experiment builds toward is about the roster's real networks: "compile this model, at the best
    # format this target is certified for, and lower to the accelerator everything that can be".
    #
    # The format is DERIVED here, by `precision_policy.best_format`, from the same three things that
    # decide it anywhere: the manifest admits, the registry expresses, an accuracy gate certifies. The
    # admitted set is passed in rather than re-read, so this and the cells above cannot end up with two
    # answers to "what does this target support". A target whose preference names nothing it admits
    # synthesizes NO roster capsule and says so -- compiling a roster model in a format the hardware
    # lacks is not a weaker result, it is a different one.
    roster = [str(m) for m in (ws.get("models") or ())]
    contraction_dtypes = {str(c.get("dtype")) for c in cells
                          if c.get("dtype") and str(c.get("family")) == "contraction"}
    if roster and contraction_dtypes:
        from merlin.targetgen.precision_policy import best_format
        policy = best_format(target, preference=(ws.get("precision_preference") or None),
                             admitted=contraction_dtypes)
        chosen = policy.get("chosen") or {}
        if chosen.get("capsule_dtype"):
            # THE SCHEME, NOT THE DTYPE. A capture asked for "int8" quantizes WEIGHTS ONLY and emits a
            # float matmul over dequantized weights -- the wrong program for a datapath that consumes the
            # narrow format on both operands, and one no golden substitution can repair. The scheme is
            # derived from the format rather than declared; a format whose activation-quantizing scheme
            # is unknown raises there rather than silently capturing float arithmetic here.
            from merlin.targetgen.capsule_source import activation_quantizing_scheme
            scheme = activation_quantizing_scheme(chosen["capsule_dtype"])
            for model in roster:
                entry = {
                    "cat": "model", "kind": "model", "op": "model",
                    "name": f"{SYNTH_PREFIX}_model_{model}",
                    "model": model, "out": "Y0",
                    "operand_dtype": chosen["capsule_dtype"],
                    **({"quant_scheme": scheme} if scheme else {}),
                    "source_role": SOURCE_ROLE,
                    "source_reference": (
                        f"synthesized for the roster axis: whole model {model!r} at {chosen['format']}, "
                        f"the highest-ranked precision this target's manifest admits for a contraction "
                        f"out of the declared preference {policy.get('preference')}"
                        + (f", captured with {scheme} so the program contains the target's own "
                           f"arithmetic rather than a float matmul over dequantized weights"
                           if scheme else "")
                        + f". Accuracy in that format is {policy['certified']['status']}"),
                    "label": "public",
                    # Same deferral every whole-model capstone carries: a roster model is worth running
                    # only once the op suite it is made of passes, or the failure says nothing.
                    "gate": {"after_op_pass_fraction": 0.8},
                    # The mesh is REQUIRED, not hoped for. A roster capsule that graded numerics alone
                    # would pass a submission that ran the whole network on the host -- which is the
                    # vacuity the op capsules had removed and the capstones did not.
                    "lanes": {"require": ["on_mesh"]},
                    "semantic": {"generalization_axis": "roster"},
                }
                _mark_source(entry)
                entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
                entries.append(entry)
        else:
            unexpressable.append(
                f"roster axis: {policy.get('status')} -- the declared preference "
                f"{policy.get('preference')} names no format this target admits for a contraction "
                f"(admitted: {policy.get('admitted')}), so no roster model can be compiled at a format "
                f"the hardware has")

    if unexpressable:
        raise SynthesisError(
            "no materializable op expresses these required cells, so synthesizing would silently leave "
            "them uncovered: " + "; ".join(unexpressable) + ". Add a builder or a PyTorch body for the "
            "family, or establish that the requirement is wrong -- do not drop the cell")

    cap = int(budget if budget is not None else (ws.get("max_synthesized_capsules") or 160))
    if len(entries) > cap:
        raise SynthesisError(
            f"synthesis would emit {len(entries)} capsules against a budget of {cap}. Raise "
            f"workload_spec.max_synthesized_capsules deliberately, or narrow the roster -- never "
            f"truncate, because a silently dropped point reads downstream as a covered one")

    return {
        "capsules": entries,
        "provenance": {
            "generated_by": "merlin.targetgen.corpus_synth.synthesize",
            "target": target,
            "n_required_cells": len(cells),
            "n_entries": len(entries),
            "budget": cap,
            "precision_preference_kept": kept,
            "precision_preference_dropped": dropped,
            "cells_no_writer_can_express": unwritable,
            # The rank/layout regions the manifest declares and nothing can build. Reported here rather
            # than raised: unlike a cell, a declared shape capability with no builder is a gap to argue
            # about with the probe named, and leaving the axis out entirely -- which is what happened
            # before it existed -- is the version that hides it.
            "shape_regions_no_writer_can_express": shape_unwritable,
            "cells_no_writer_note": (
                "a required cell whose op has no direct-MLIR builder and whose dtype the PyTorch writer "
                "cannot express. Reported as an uncovered cell to argue about rather than emitted as an "
                "entry nothing can write; adding a builder for that op closes it"),
            "composition_shapes_required": sorted(composition),
            "composition_note": (
                "one micro-model capsule is synthesized for this axis, and its composition is whatever "
                "the derived inventory produces. A shape listed here that the micro model does not "
                "reach is not covered by synthesis -- emitting a capsule per shape would mean inventing "
                "inventories no capture supports"),
            "host_only_unsynthesizable": unsized_host,
            "host_only_note": (
                "a host-only family with no materializable op or no dtype observed in any capture. "
                "Reported rather than dropped: a negative lane nobody demanded reads downstream exactly "
                "like one nothing violates"),
            "accumulation_depth_unsizable": unsized_depth,
            "accumulation_depth_note": (
                "a reduction depth this target could not size. Kept separate from "
                "`cells_no_writer_can_express` because the two license different actions: a cell with "
                "no writer needs a builder, while an unsized depth needs a CERTIFICATION RUN -- the "
                "writer is there and the measured history to size it against is not. Reporting them "
                "together would make 'we never timed this target' read as 'the corpus cannot express "
                "this'"),
            "memory_regimes_status": "resolved" if regimes_resolved else "not_resolved",
            "memory_regimes_unreachable": unreachable_regimes,
            "memory_regime_note": (
                "the spec carries no `regime_extents`; it predates the axis and must be regenerated "
                "before a regime can be synthesized or declared unreachable"
                if not regimes_resolved else 
                "a regime with no capsule shape that reaches it on this target. `fits_on_reuse` is "
                "always here: a capsule's declared inputs are all live at once, so peak-live and total "
                "coincide and the regime that separates them cannot arise from inputs alone"),
            "precision_note": (
                "a preference RANKS the dtypes the target already admits and can never widen them; a "
                "dropped token is reported rather than silently ignored"),
        },
    }
