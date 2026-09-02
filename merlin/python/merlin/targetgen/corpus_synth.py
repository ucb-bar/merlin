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
    host_block = spec_doc.get("host_only") or {}
    host_dtypes = dict(host_block.get("dtypes") or {})
    unsized_host: list[str] = []
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
