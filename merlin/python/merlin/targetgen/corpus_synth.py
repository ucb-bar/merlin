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


def op_for_family(family: str, *, admitted_ops: set[str] | None = None) -> str | None:
    """The cheapest materializable op that exercises ``family``, or None when none does."""
    pool = admitted_ops if admitted_ops is not None else available_ops()
    cands = sorted((op for op, fam in _op_family_map().items() if fam == family and op in pool),
                   key=_cost)
    return cands[0] if cands else None


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

        # Narrowed by the CELL's dtype before anything is chosen. `op_for_family` picks the cheapest
        # op that can be BUILT, and buildable is not gradeable: the golden engine is selected by the
        # dtype, so a block-scaled-only op chosen for an fp16 cell produces an entry no engine can
        # grade -- which surfaced as a crash inside the writer rather than a choice made here.
        cell_pool = ops_gradeable_at(dtype, pool)
        epilogue: list[str] = []
        fused = _fused_carrier(family, composed, cell_pool) if composed else None
        if fused is not None:
            op, epilogue = fused
        else:
            op = op_for_family(family, admitted_ops=cell_pool)
        if op is None:
            unexpressable.append(
                f"{cell.get('cell')} (family {family!r}; no op that exercises it has a golden at "
                f"dtype {dtype!r})")
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
        source = source_for_op(op)
        if source:
            entry["source"] = source
        if epilogue:
            entry["epilogue"] = epilogue
        entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
        entries.append(entry)

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
            "precision_note": (
                "a preference RANKS the dtypes the target already admits and can never widen them; a "
                "dropped token is reported rather than silently ignored"),
        },
    }
