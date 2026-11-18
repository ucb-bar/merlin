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
from merlin.development.targetgen.conformance.geometry_entries import add_geometry_entries
from merlin.development.targetgen.conformance.semantic_entries import add_semantic_entries
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


def synthesize(
    spec_doc: dict,
    *,
    workload_spec: dict | None = None,
    budget: int | None = None,
    application_inventory: dict | None = None,
    capability_contract: dict | None = None,
) -> dict:
    """``{"capsules": [entry...], "provenance": {...}}`` for one target's derived requirement.

    Pass ``capability_contract`` for explicit-input derivation without ambient
    target-contract selection. Omitting it retains the legacy registry reader;
    callers freezing a new experiment must supply the selected contract.
    """
    from merlin.development.targetgen.conformance.synthesis_entries import (
        _CONV_OPS_BY_RANK,
        SOURCE_ROLE,
        SYNTH_ACC_SCALE,
        SYNTH_PREFIX,
        _activation_contraction_entries,
        _fused_carrier,
        _host_probe_op,
        _image_for_window,
        _mark_source,
        _multi_block_members,
        _signed_stimulus_range,
        _tail_isolation_entries,
        _writer_for,
        available_ops,
        declare_pool_window,
        declare_residual_add_params,
        exact_int_mm_entries,
        filtered_precision,
        op_for_family,
        op_for_shape,
        ops_gradeable_at,
        pass_requirements_for,
        permute_probe_layout,
        withhold_host_placed,
    )

    target = str(spec_doc.get("target") or "")
    floor_record = spec_doc.get("certification_floor")
    if floor_record is not None:
        floor = floor_record.get("tier") if isinstance(floor_record, dict) else None
        declared_tiers = [*(spec_doc.get("oracle_tiers") or ()), *(spec_doc.get("oracle_tiers_declared") or ())]
        if not isinstance(floor, str) or floor not in declared_tiers:
            raise SynthesisError("direct certification floor is absent from selected oracle-tier declarations")
    cells = list(spec_doc.get("cells") or ())
    probes = list((spec_doc.get("boundaries") or {}).get("extent_probes") or ())
    ws = dict(workload_spec or {})
    # A literal-name check misses ordinary aliases such as Tiny-Llama/tiny_llama.
    # This is an admission guard, not an anonymizer: held-out claim workloads
    # cannot become the source of the public derivation corpus in a verified run.
    if isinstance(application_inventory, dict) and isinstance(application_inventory.get("applications"), dict):
        roster_key = lambda name: "".join(ch for ch in str(name).casefold() if ch.isalnum())  # noqa: E731
        claims = {roster_key(name) for name in (ws.get("models") or ())}
        captures = {roster_key(name) for name in application_inventory["applications"]}
        if (claims - {""}) & (captures - {""}):
            raise SynthesisError("held-out claim model appears in the selected public derivation inventory")
    pool = available_ops()
    exact_entries, exact_refused = exact_int_mm_entries(spec_doc.get("application_demands"), application_inventory)

    admitted_dtypes = {str(c.get("dtype")) for c in cells if c.get("dtype")}
    # A preference is declared in REGISTRY spelling ("int8"); a cell carries the CAPSULE spelling
    # ("i8"). Comparing them raw reported every token as dropped and the whole preference as inert.
    from merlin.development.targetgen.conformance.requirements import capsule_dtype

    _pref = []
    for tok in ws.get("precision_preference") or ():
        try:
            _pref.append(capsule_dtype(str(tok)))
        except Exception:  # noqa: BLE001 -- an unmappable token is not a dtype
            _pref.append(str(tok))
    kept, dropped = filtered_precision(_pref, admitted_dtypes)

    cap_map: dict[str, Any] = {}
    try:
        from merlin.target.capabilities.eligibility import (
            capability_map_for_target,
            capability_map_from_contract,
        )

        cap_map = (
            capability_map_from_contract(capability_contract)
            if capability_contract is not None
            else capability_map_for_target(target)
        ) or {}
    except Exception:  # noqa: BLE001 -- no map means no composed-with facts
        cap_map = {}

    #: What granularity each dtype's own datapath imposes on an extent, published per cell by
    #: `conformance.spec`. Empty on a spec written before that derivation existed.
    _quanta = quanta_by_dtype(spec_doc)

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
            "cat": "isa",
            "kind": "isa",
            "name": name,
            "op": op,
            "operand_dtype": dtype,
            "out": "Y0",
            "lhs": "A0",
            "weight": "W",
            "source_role": SOURCE_ROLE,
            "source_reference": (
                f"synthesized for conformance cell {cell.get('cell')} "
                f"(basis={cell.get('basis')}, admitted_by={list(cell.get('admitted_by') or ())}); "
                f"extents from boundaries.extent_probes"
                + (
                    f"; {family} is admitted only in composition with {list(composed)}, so it is carried "
                    f"as a fused epilogue on {op} -- a standalone capsule for it would be refused by the "
                    f"eligibility oracle as a false fallback"
                    if fused
                    else ""
                )
            ),
            "label": "public",
            **extents_for(alignment, probes, quantum=_quanta.get(dtype)),
        }
        if epilogue:
            entry["epilogue"] = epilogue
            declare_pool_window(entry)
        declare_residual_add_params(entry)
        _writer = _writer_for(entry)
        if _writer is None:
            unwritable.append(
                f"{cell.get('cell')} (op {op!r} has no direct-MLIR builder and dtype {dtype!r} is not "
                f"a regime the PyTorch writer can express)"
            )
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
    regime_dtype = str(mem_block.get("regime_dtype") or "") or (sorted(admitted_dtypes)[0] if admitted_dtypes else "")
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
            "cat": "isa",
            "kind": "isa",
            "name": f"{SYNTH_PREFIX}_regime_{regime}",
            "op": regime_op,
            "operand_dtype": regime_dtype,
            "out": "Y0",
            "lhs": "A0",
            "weight": "W",
            "source_role": SOURCE_ROLE,
            "source_reference": (
                f"synthesized for memory regime {regime!r}: the declared inputs occupy "
                f"{ext.get('rows')} of {ext.get('capacity_rows')} operand-store rows "
                f"({ext.get('fraction_of_capacity')} of capacity), which is what puts this capsule in "
                f"that regime. Extents resolved by memory_regime.extents_for_regime with the same "
                f"sizing the coverage gate measures with"
            ),
            "label": "public",
            "M": ext["M"],
            "K": ext["K"],
            "N": ext["N"],
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

    # ---- the ACCUMULATOR-OUTPUT boundary ----------------------------------------------------------
    # Input residency and K depth do not bound the number of output tiles a
    # backend keeps live. Derive the first output shape beyond one addressable
    # accumulator buffer from RTL facts, without importing a held-out model's
    # shape or baking a target's row count into shared code.
    output_bound = spec_doc.get("accumulator_output_boundary") or {}
    output_bound_refusal = ""
    if output_bound.get("status") == "resolved":
        output_dtype = regime_dtype
        output_op = (
            op_for_family("contraction", admitted_ops=ops_gradeable_at(output_dtype, pool), dtype=output_dtype)
            if output_dtype
            else None
        )
        sibling = f"{SYNTH_PREFIX}_contraction_{output_dtype}_aligned"
        if not output_op or sibling not in {str(e.get("name")) for e in entries}:
            output_bound_refusal = f"no gradeable contraction and certified sibling at {output_dtype!r}"
        else:
            n_tiles = int(output_bound["N_tiles"])
            entry = {
                "cat": "isa",
                "kind": "isa",
                "name": f"{SYNTH_PREFIX}_accumulator_output_boundary",
                "op": output_op,
                "operand_dtype": output_dtype,
                "lhs": "A0",
                "weight": "W",
                "out": "Y0",
                "M": f"tile/{int(output_bound['tile_edge'])}",
                "K": "tile",
                "N": f"{n_tiles}*tile",
                "source_role": SOURCE_ROLE,
                "source_reference": (
                    "synthesized for the accumulator-output boundary: the RTL-derived addressable "
                    f"store has {output_bound['capacity_rows']} rows, while one output row by {n_tiles} "
                    f"N tiles requires {output_bound['output_rows_if_resident']} simultaneously "
                    "resident output rows. A streaming schedule may pass; a full-output-resident "
                    "schedule must not address beyond the store"
                ),
                "label": "public",
                "generalization": {"generalization_axis": "accumulator_output_capacity"},
            }
            why = cap_to_affordable(entry, spec_doc, extends=sibling)
            if why:
                entry["source_reference"] += f". {why}"
            entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
            entries.append(entry)
            entries.extend(_multi_block_members(output_bound, output_op, output_dtype, sibling, spec_doc))

    # ---- the NEGATIVE lane ------------------------------------------------------------------------
    # Families a real capture CONTAINS and this target's manifest does NOT admit. The compiler must leave
    # them on the host, and accelerating one is as much a defect as failing to accelerate an admitted
    # family -- but the requirement derived that set into `host_only` and nothing ever demanded it, so
    # the negative lane was covered only by whatever a hand-authored capsule happened to assert.
    #
    # These declare `forbid` alone. Requiring the host lane as well would add a demand no op-path grade
    # can measure, and an unmeasurable requirement turns the capsule into a permanent `incomplete`
    # instead of a test. The forbid IS enforceable: a decoded accelerator instruction violates it.
    composition, unsized_host = add_semantic_entries(
        SOURCE_ROLE=SOURCE_ROLE,
        SYNTH_ACC_SCALE=SYNTH_ACC_SCALE,
        SYNTH_PREFIX=SYNTH_PREFIX,
        _host_probe_op=_host_probe_op,
        _mark_source=_mark_source,
        _quanta=_quanta,
        _signed_stimulus_range=_signed_stimulus_range,
        _writer_for=_writer_for,
        admitted_dtypes=admitted_dtypes,
        application_inventory=application_inventory,
        cells=cells,
        declare_pool_window=declare_pool_window,
        declare_residual_add_params=declare_residual_add_params,
        entries=entries,
        kept=kept,
        op_for_shape=op_for_shape,
        pass_requirements_for=pass_requirements_for,
        permute_probe_layout=permute_probe_layout,
        pool=pool,
        probes=probes,
        shape_unwritable=shape_unwritable,
        spec_doc=spec_doc,
        unexpressable=unexpressable,
    )

    # ---- the CONVOLUTION-WINDOW axis ----------------------------------------------------------------
    # WHICH window rides the convolution, and how its border is treated. A cell says a contraction runs
    # at int8; it cannot say the contraction is a 3x3 stride-2 convolution whose border is padded, and
    # the padded border is where a lost padding identity is wrong and nowhere else.
    #
    # ⚠️ THE GAP THIS CLOSES IS BETWEEN TWO MEMBERS THAT EACH COVER HALF OF IT. The hand-authored
    # corpus declares a padded convolution and a strided convolution as SEPARATE entries and no
    # strided-AND-padded one, while the captures demand exactly that. A per-attribute corpus misses
    # that shape by construction; a per-CLASS corpus cannot, because the class IS the combination.
    #
    # A `padUNKNOWN` obligation is still emitted, at zero padding, with the unknown recorded in
    # `source_reference`: the capture applies that padding by a route the reader cannot see (a
    # reflection pad via an index gather), so its IDENTITY is not zero and a member asserting zero
    # would be making a claim the evidence does not support. What such a member does test is the
    # window and its stepping, which is real coverage and is stated as exactly that much.
    conv_capacity_boundary, unsized_geometry = add_geometry_entries(
        SOURCE_ROLE=SOURCE_ROLE,
        SYNTH_PREFIX=SYNTH_PREFIX,
        _CONV_OPS_BY_RANK=_CONV_OPS_BY_RANK,
        _activation_contraction_entries=_activation_contraction_entries,
        _image_for_window=_image_for_window,
        _quanta=_quanta,
        _signed_stimulus_range=_signed_stimulus_range,
        _tail_isolation_entries=_tail_isolation_entries,
        _writer_for=_writer_for,
        admitted_dtypes=admitted_dtypes,
        declare_pool_window=declare_pool_window,
        declare_residual_add_params=declare_residual_add_params,
        entries=entries,
        kept=kept,
        mem_block=mem_block,
        op_for_family=op_for_family,
        ops_gradeable_at=ops_gradeable_at,
        pass_requirements_for=pass_requirements_for,
        pool=pool,
        probes=probes,
        regime_dtype=regime_dtype,
        spec_doc=spec_doc,
        unexpressable=unexpressable,
    )

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
    _depth = (spec_doc.get("memory_mapping") or {}).get("reduction_depth") or {}
    _depth_dtype = (spec_doc.get("memory_mapping") or {}).get("regime_dtype") or (
        sorted(admitted_dtypes)[0] if admitted_dtypes else ""
    )
    _depth_op = op_for_family("contraction", admitted_ops=pool, dtype=_depth_dtype) if _depth_dtype else None
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
            "cat": "layers",
            "kind": "layer",
            "name": name,
            "op": _depth_op,
            "operand_dtype": _depth_dtype,
            "lhs": "A0",
            "weight": "W",
            "out": "Y0",
            "M": "tile",
            "K": f"{int(point['K_tiles'])}*tile",
            "N": "tile",
            "source_role": SOURCE_ROLE,
            "source_reference": why,
            "label": "public",
            "generalization": {"generalization_axis": "accumulation_depth"},
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
        _regime_depths = [
            pt for blk in (_depth.get("by_regime") or {}).values() for pt in (blk.get("points") or ()) if pt.get("K")
        ]
        _cert_pt = _depth.get("certified") if not _regime_depths else None
        if _cert_pt:
            _certified_depth_name = f"{SYNTH_PREFIX}_kdepth_certified"
            entries.append(
                _depth_entry(
                    _certified_depth_name,
                    _cert_pt,
                    tier="L3",
                    why=(
                        f"synthesized for the accumulation-depth axis: {_cert_pt['K_tiles']} accumulation "
                        f"passes, the shallowest reduction that writes the accumulator more than once. "
                        f"Emitted only because this target's residency regimes yield no depth of their "
                        f"own; where they do, they produce deeper capsules and this would duplicate one. "
                        + (
                            "Certification cost is unknown: this is a direct functional obligation, "
                            "not a cost-model affordability claim"
                            if _cert_pt.get("cost_status") == "unknown_not_priced"
                            else f"Costs {_cert_pt['predicted_seconds']}s against a {_cert_pt['budget_s']}s "
                            f"budget -- priced on the OUTPUT tile, which the reduction depth does not move"
                        )
                    ),
                )
            )
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
                f"{SYNTH_PREFIX}_kdepth_{_regime}",
                _deep,
                tier="L3",
                why=(
                    f"synthesized for the accumulation-depth axis: the deepest reduction the "
                    f"{_regime!r} residency regime admits ({_deep['K_tiles']} accumulation passes, "
                    f"{_deep.get('fraction_of_capacity')} of the operand store). The reduction moves "
                    f"the operands and not the result, so this certifies at one output tile"
                ),
            )
            _why = cap_to_affordable(
                _e, spec_doc, extends=(_certified_depth_name or f"{SYNTH_PREFIX}_contraction_{_depth_dtype}_aligned")
            )
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
    # The requirement side has already grouped the application's regions by what the compiler must
    # do with them and selected each shape. Without a direct floor it prices the anchor and caps a
    # larger extension; with a floor both the functional anchor and full shape require direct oracle
    # execution, with cost unknown. A class that could not be sized is not in `required`
    # at all -- it is in the axis's `refused` list with its reason, which is reported rather than
    # raised because an unaffordable behaviour is a fact about the budget, not a broken corpus.
    _app = spec_doc.get("application_shapes") or {}
    #: Classes that actually got a cycle-accurate entry. An L2-only capsule is admissible ONLY as an
    #: extension of one, and the sizing side hands them over in pairs -- but a pair is not a
    #: guarantee here: if the deeper entry's op cannot be materialized we skip it, and iterating the
    #: list flat would then still emit its L2 partner, resting on nothing. That is precisely the
    #: failure the `extends` relation exists to prevent, arriving through the back door.
    _certified_classes: set = set()
    for _cap in _app.get("required") or ():
        _cls = str(_cap.get("class") or "")
        _tier = str(_cap.get("tier") or "L3")
        _floor_tier = (spec_doc.get("certification_floor") or {}).get("tier")
        _direct_floor_member = (
            isinstance(_floor_tier, str)
            and _tier == _floor_tier
            and (_cap.get("basis") or {}).get("cost_status") == "unknown_not_priced"
        )
        _dtype = _cls.split("/")[1] if "/" in _cls else ""
        if not _dtype:
            continue
        _op = op_for_family("contraction", admitted_ops=pool, dtype=_dtype)
        if _op is None:
            unwritable.append(f"application class {_cls}: no op materializes a contraction at {_dtype!r}")
            continue
        if _tier != "L3" and not _direct_floor_member and _cls not in _certified_classes:
            unwritable.append(
                f"application class {_cls}: its L2 capsule extends a cycle-accurate sibling that was "
                f"not emitted, so it would rest on nothing; dropped rather than shipped as a large "
                f"capsule nothing certifies"
            )
            continue
        _slug = _cls.replace("/", "_").replace("-", "_")
        _member = str((_cap.get("basis") or {}).get("member") or "")
        _suffix = f"full_{_tier.lower()}" if _member == "full_shape" else _tier.lower()
        _anchor_name = f"{SYNTH_PREFIX}_app_{_slug}_{_tier.lower()}"
        entry = {
            "cat": "layers",
            "kind": "layer",
            "name": f"{SYNTH_PREFIX}_app_{_slug}_{_suffix}",
            "op": _op,
            "operand_dtype": _dtype,
            "lhs": "A0",
            "weight": "W",
            "out": "Y0",
            "M": int(_cap["M"]),
            "K": int(_cap["K"]),
            "N": int(_cap["N"]),
            # NOT `derived_sweep`: a sweep's shapes track the target's geometry, and this one tracks a
            # model's. Conflating them would tell a reader the extents move with the tile edge.
            "source_role": "model_derived",
            "source_reference": (
                f"synthesized for the application axis: behavioural class {_cls}, representing "
                f"{_cap['basis'].get('representative_of')} region(s) of "
                f"{_cap['basis'].get('source')}. Sized by {_cap['basis'].get('sized_by')}"
                + (
                    f" against a {_app.get('cert_budget_s')}s certification budget"
                    if _cap["basis"].get("sized_by") == "measured_cost_model"
                    else ""
                )
                + (
                    f"; has functional anchor {_anchor_name}, which also requires direct "
                    f"{_tier} certification; neither member is priced or certified by derivation"
                    if _member == "full_shape" and _cap.get("extends")
                    else f"; extends {_cap['extends']}, which carries the cycle-accurate guarantee this "
                    f"larger shape rests on"
                    if _cap.get("extends")
                    else ""
                )
            ),
            "label": "public",
            "generalization": {"generalization_axis": "application"},
        }
        if (_cap.get("basis") or {}).get("cost_status") == "unknown_not_priced":
            entry["certification_cost"] = "unknown_not_priced"
        if _tier != "L3" and not _direct_floor_member:
            # AN L2-ONLY CAPSULE IS AN EXTENSION, NEVER A SUBSTITUTE. The tier is capped because this
            # shape is too large to certify, and `extends` names the sibling that was -- so a reader
            # (and the gate) can tell a large capsule resting on a guarantee from one resting on
            # nothing.
            entry["max_oracle_tier"] = _tier
            # ⚠️ `extends` MUST NAME THE SIBLING CAPSULE, NOT THE CLASS IT BELONGS TO. The requirement
            # side names the behavioural CLASS ("contraction/i8/aligned/fits_double/rank2/
            # squareish_gemm") because that is its own vocabulary; `tier_policy.verify_extends` looks
            # up a CAPSULE and so could never resolve one. Measured: 23 of 28 members carrying the
            # field named a class, and every one of them verified as UNVERIFIED -- which the gate
            # correctly treats as WEAKER than naming nobody, because an unchecked extends reads as
            # certified. The anchor relation was declared everywhere and resolvable almost nowhere.
            #
            # The class stays, in `source_reference`, where it is provenance rather than a reference.
            entry["extends"] = f"{SYNTH_PREFIX}_app_{_slug}_l3"
        _mark_source(entry)
        entry["pass_requirements"] = pass_requirements_for(entry, spec_doc)
        entries.append(entry)
        if _tier == "L3" or _direct_floor_member:
            _certified_classes.add(_cls)

    # ---- the HELD-OUT CLAIM axis --------------------------------------------------------------------
    # The descriptor's model roster is for owner-side evaluation AFTER Phase 1
    # freezes. A public `model/*` capsule would enter the Phase-1 functional
    # selection and disclose the very model the compiler is said to generalize
    # to. Record the obligation, never synthesize a claim-model capsule here.
    roster = [str(m) for m in (ws.get("models") or ())]
    contraction_dtypes = {
        str(c.get("dtype")) for c in cells if c.get("dtype") and str(c.get("family")) == "contraction"
    }
    from merlin.development.targetgen.conformance import claim_applications as _claims

    evaluation_only = sorted(
        {str(m) for m in (ws.get("evaluation_only_models") or ())} | set(_claims.evaluation_only_models())
    )
    claim_model_evaluation: dict[str, Any] = {
        "schema": "claim_model_evaluation_v1",
        "source": "workload_spec.models",
        "model_count": len(roster),
        "visibility": "owner_only_after_phase1_freeze",
        "public_capsules_emitted": 0,
        # Whole models held out to check generality: evaluated on the same owner-only terms, never
        # derived from. Counted, not named in a public artifact beyond the reviewed registry.
        "evaluation_only_model_count": len(evaluation_only),
    }
    if roster and contraction_dtypes:
        from merlin.development.targetgen.policies.precision import best_format

        policy = best_format(target, preference=(ws.get("precision_preference") or None), admitted=contraction_dtypes)
        chosen = policy.get("chosen") or {}
        if chosen.get("capsule_dtype"):
            claim_model_evaluation.update(
                status="awaiting_phase1_freeze",
                capsule_dtype=chosen["capsule_dtype"],
                format=chosen["format"],
                format_selection_status=policy.get("status"),
            )
        else:
            claim_model_evaluation.update(
                status="format_unavailable",
                reason=(
                    f"{policy.get('status')}: preference {policy.get('preference')} names no admitted "
                    f"contraction format (admitted: {policy.get('admitted')})"
                ),
            )
    elif roster:
        claim_model_evaluation.update(
            status="format_unavailable",
            reason="derived requirement has no contraction cell for a whole-model accelerator claim",
        )
    else:
        claim_model_evaluation["status"] = "no_claim_models_declared"

    if unexpressable:
        raise SynthesisError(
            "no materializable op expresses these required cells, so synthesizing would silently leave "
            "them uncovered: " + "; ".join(unexpressable) + ". Add a builder or a PyTorch body for the "
            "family, or establish that the requirement is wrong -- do not drop the cell"
        )

    entries.extend(exact_entries)

    # ONE PASS OVER EVERY AXIS, so a new axis cannot forget it. A readout stage treats a negative
    # accumulator differently from a positive one (an activation clamps it, a requantization rounds
    # and saturates the other way, a pool's padding identity is wrong for it), and the default
    # stimulus is non-negative: on it none of that is observable, and the member cannot fail.
    # Measured: a relu member returned 126 of 256 outputs negative on hardware, activation never
    # applied, and passed. A scale stage with no multiplier is the same defect one step over: every
    # engine reads it as 1.0 and certifies a bare saturating cast.
    # The same holds one step earlier, at the operand load: a signed integer operand that is
    # zero-extended instead of sign-extended agrees with the reference on every non-negative input,
    # so a bare integer contraction, convolution or movement on the default stimulus cannot fail it.
    from merlin.development.targetgen.corpus.model_builders import BUILDERS
    from merlin.development.targetgen.corpus.selection import has_signed_integer_operands, is_sign_sensitive

    for entry in entries:
        if not is_sign_sensitive(entry):
            if str(entry.get("op") or "") in BUILDERS and has_signed_integer_operands(entry):
                entry.setdefault("stimulus_range", list(_signed_stimulus_range()))
            continue
        entry.setdefault("stimulus_range", list(_signed_stimulus_range()))
        staged = [entry, *(m for m in entry.get("matmuls") or () if isinstance(m, dict))]
        if "acc_scale_by_column" not in entry and any("acc_scale" in (item.get("epilogue") or ()) for item in staged):
            entry.setdefault("acc_scale", SYNTH_ACC_SCALE)

    # Placement decides publication: an entry with no accelerator placement is reported, not emitted.
    entries, host_placed = withhold_host_placed(entries)

    cap = int(budget if budget is not None else (ws.get("max_synthesized_capsules") or 160))
    if len(entries) > cap:
        raise SynthesisError(
            f"synthesis would emit {len(entries)} capsules against a budget of {cap}. Raise "
            f"workload_spec.max_synthesized_capsules deliberately, or narrow the public axes -- never "
            f"truncate, because a silently dropped point reads downstream as a covered one"
        )

    return {
        "capsules": entries,
        "provenance": {
            "generated_by": "merlin.development.targetgen.conformance.synthesis.synthesize",
            "target": target,
            "n_required_cells": len(cells),
            "n_entries": len(entries),
            "budget": cap,
            "precision_preference_kept": kept,
            "precision_preference_dropped": dropped,
            "claim_model_evaluation": claim_model_evaluation,
            "cells_no_writer_can_express": unwritable,
            # The rank/layout regions the manifest declares and nothing can build. Reported here rather
            # than raised: unlike a cell, a declared shape capability with no builder is a gap to argue
            # about with the probe named, and leaving the axis out entirely -- which is what happened
            # before it existed -- is the version that hides it.
            "shape_regions_no_writer_can_express": shape_unwritable,
            "cells_no_writer_note": (
                "a required cell whose op has no direct-MLIR builder and whose dtype the PyTorch writer "
                "cannot express. Reported as an uncovered cell to argue about rather than emitted as an "
                "entry nothing can write; adding a builder for that op closes it"
            ),
            "composition_shapes_required": sorted(composition),
            "composition_note": (
                "one micro-model capsule is synthesized for this axis, and its composition is whatever "
                "the derived inventory produces. A shape listed here that the micro model does not "
                "reach is not covered by synthesis -- emitting a capsule per shape would mean inventing "
                "inventories no capture supports"
            ),
            "host_placement_not_published": host_placed,
            "host_placement_note": (
                "an entry whose declared placement carries no accelerator work. Host placement is "
                "compiled by Merlin's own generic host codegen, so such a capsule would grade none of the "
                "author's work; the requirement stays listed here instead of becoming a public capsule"
            ),
            "host_only_unsynthesizable": unsized_host,
            "host_only_note": (
                "a host-only family with no materializable op or no dtype observed in any capture. "
                "Reported rather than dropped: a negative lane nobody demanded reads downstream exactly "
                "like one nothing violates"
            ),
            # Geometry classes real captures present that no entry could be emitted for. A separate
            # list because it licenses a separate action: not a builder and not a certification run,
            # but a shape the writer refused -- and a class silently absent here reads downstream as a
            # geometry the corpus covers.
            "geometry_classes_unsynthesizable": unsized_geometry,
            "geometry_note": (
                "a geometric class (aspect ratio) real models present that the corpus could not emit a "
                "capsule for. The classes that WERE emitted carry the real M/K/N of the heaviest region "
                "in the class, and their tier follows that size: one too large to certify inside the "
                "budget is graded at the loop tier and names the certified sibling it rests on"
            ),
            # A missing model-owned quantization representation must survive corpus synthesis. Without
            # this field, regeneration would honestly emit no MX application capsule but erase why,
            # leaving the absence indistinguishable from an axis nobody asked for.
            "application_missing_capabilities": list(_app.get("missing_capabilities") or ()),
            "application_operation_plan": _application_operation_plan(
                spec_doc.get("application_demands"), exact_entries=exact_entries
            ),
            "exact_application_slice_candidates": [entry["name"] for entry in exact_entries],
            "exact_application_slice_refusals": exact_refused,
            "exact_application_slice_note": (
                "candidate entries require PyTorch capture and exact emitted-linalg verification; "
                "neither these entries nor their capsule evidence proves whole-model compilation"
            ),
            "accumulation_depth_unsizable": unsized_depth,
            "accumulation_depth_note": (
                "a reduction depth this target could not size. Kept separate from "
                "`cells_no_writer_can_express` because the two license different actions: a cell with "
                "no writer needs a builder, while an unsized depth needs a CERTIFICATION RUN -- the "
                "writer is there and the measured history to size it against is not. Reporting them "
                "together would make 'we never timed this target' read as 'the corpus cannot express "
                "this'"
            ),
            "memory_regimes_status": "resolved" if regimes_resolved else "not_resolved",
            "accumulator_output_boundary": output_bound,
            "accumulator_output_boundary_refusal": output_bound_refusal,
            "conv_operand_capacity_boundary": conv_capacity_boundary,
            "memory_regimes_unreachable": unreachable_regimes,
            "memory_regime_note": (
                "the spec carries no `regime_extents`; it predates the axis and must be regenerated "
                "before a regime can be synthesized or declared unreachable"
                if not regimes_resolved
                else "a regime with no capsule shape that reaches it on this target. `fits_on_reuse` is "
                "always here: a capsule's declared inputs are all live at once, so peak-live and total "
                "coincide and the regime that separates them cannot arise from inputs alone"
            ),
            "precision_note": (
                "a preference RANKS the dtypes the target already admits and can never widen them; a "
                "dropped token is reported rather than silently ignored"
            ),
        },
    }
