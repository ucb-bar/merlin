"""Evaluate library-route admission against candidate shape and selected facts."""

from __future__ import annotations

from fractions import Fraction as Fraction

from merlin.analysis.kernels.mechanisms.metrics import (
    _SEPARATE,
    _SIDES,
    COMPARISON_SEMANTICS,
    FALLBACK,
    NO_STREAM,
    PORT,
    REGISTER,
    ROLE_FACTS,
    STREAM_FACTS,
    STREAM_SCOPE,
    _aggregate,
    _broadcast,
    _broadcasts,
    _buffering,
    _config,
    _consumer_ports,
    _count,
    _direction,
    _distance,
    _distance_value,
    _evidence_id,
    _extent,
    _fact,
    _first,
    _loop,
    _loops,
    _operand_gaps,
    _operand_groups,
    _order,
    _overlap,
    _overlap_value,
    _plain,
    _port_match,
    _position,
    _rate_status,
    _ratio,
    _role_entries,
    _role_fallback,
    _role_rate,
    _roles,
    _side_maps,
    _summarizer,
    _summary_value,
    _transfer,
    _unknowns,
    _width,
    stream_fact,
)
from merlin.analysis.kernels.mechanisms.stream import (
    _GROUP_BASIS,
    _INTERLEAVE,
    _STREAM_BASIS,
    _STREAM_METRIC,
    EXTRA_FACTS,
    STREAM_EFFECTS,
    _capacity_fraction,
    _extra_rows,
    _group_columns,
    _in_loop,
    _interleave_rows,
    _number,
    _on_accelerator,
    _output_group_rows,
    _per_role,
    _scalar,
    _scalar_map,
    _stream_direction,
    _stream_rows,
)

# ---- mechanism_admission -----------------------------------------------------------------------

_MAX_N_TILES = 256  # Bounded per-width admission evaluation; larger extents report the bound as unknown.


_DENSE = ("m", "k", "n")


_CONV = ("kh", "kw", "c", "oc")


def selected_geometry(target, facts):
    """Block edge, operand/accumulator rows and element widths from the selected RTL facts, or a reason."""
    from merlin.compiler.schedule import geometry_from_address_space
    from merlin.compiler.schedule.blocks import BlockScheduleError
    from merlin.target.capabilities.address_space import (
        accumulator_store,
        derive_address_space,
        operand_store,
    )

    if not target or facts is None:
        return None, "selected target or RTL facts unavailable"
    space = derive_address_space(target, facts=facts)
    try:
        geometry = geometry_from_address_space(space)
    except BlockScheduleError as error:
        return None, str(error)
    operand, accumulator = operand_store(space).store, accumulator_store(space).store
    if None in (operand, accumulator) or None in (
        operand.element_bits,
        accumulator.element_bits,
        operand.row_bytes,
        accumulator.row_bytes,
    ):
        return None, "operand or accumulator element width/row bytes not derived from the facts"
    return dict(
        block=geometry.block,
        operand_rows=geometry.operand_rows,
        accumulator_rows=geometry.accumulator_rows,
        operand_bank_rows=geometry.operand_bank_rows,
        operand_store=operand.name,
        accumulator_store=accumulator.name,
        operand_row_bytes=operand.row_bytes,
        accumulator_row_bytes=accumulator.row_bytes,
        operand_element_bits=operand.element_bits,
        accumulator_element_bits=accumulator.element_bits,
        sources=dict(geometry.sources),
    ), None


def _family(shape):
    keys = set(shape or ())
    if set(_DENSE) <= keys:
        return "dense"
    if set(_CONV) <= keys:
        return "convolution"
    return None


def _dense(shape, geometry):

    from merlin.compiler.schedule.dense.compact import DenseLoopStorage, DenseMemoryView
    from merlin.compiler.transforms.dense.planning import DenseProblem, DenseResources

    edge = geometry["block"]

    problem = DenseProblem(
        shape["m"],
        shape["k"],
        shape["n"],
        edge,
        edge,
        edge,
        geometry["operand_element_bits"] // 8,
        geometry["operand_element_bits"] // 8,
        geometry["accumulator_element_bits"] // 8,
    )

    mt, kt, nt = problem.tiles

    resources = DenseResources(
        geometry["operand_rows"] * geometry["operand_row_bytes"],
        geometry["accumulator_rows"] * geometry["accumulator_row_bytes"],
        max_m_blocks=mt,
        max_n_blocks=nt,
        max_k_blocks=kt,
        partial_output_tiles=True,
        partial_output_blocks=True,
    )

    def storage(first, first_rows):
        """One disjoint witness split of the operand store: ``first`` gets exactly ``first_rows``."""

        rest = geometry["operand_rows"] - first_rows

        if first_rows <= 0 or rest <= 0:
            return None

        other = "rhs" if first == "lhs" else "lhs"

        made = {
            first: DenseMemoryView(first, geometry["operand_store"], geometry["operand_row_bytes"], 0, first_rows),
            other: DenseMemoryView(other, geometry["operand_store"], geometry["operand_row_bytes"], first_rows, rest),
        }

        return DenseLoopStorage(
            made["lhs"],
            made["rhs"],
            DenseMemoryView(
                "acc", geometry["accumulator_store"], geometry["accumulator_row_bytes"], 0, geometry["accumulator_rows"]
            ),
        )

    return problem, resources, storage


def _m_panel(shape, geometry):
    from merlin.compiler.schedule.panels.pipeline import dense_m_panel_capacity

    problem, resources, storage = _dense(shape, geometry)
    mt, kt, nt = problem.tiles
    if nt > _MAX_N_TILES:
        return dict(status="unknown", missing=[f"bounded evaluation covers at most {_MAX_N_TILES} N tiles"])
    rows, satisfied, blockers = [], [], {}
    for n_tiles in range(1, nt + 1):
        rhs_rows = kt * n_tiles * problem.tile_k
        witness = storage("rhs", rhs_rows)
        if witness is None:
            blockers["rhs_private_rows"] = blockers.get("rhs_private_rows", 0) + 1
            rows.append(
                dict(
                    n_tiles=n_tiles,
                    maximum_m_tiles=0,
                    limiting_conditions=["rhs_private_rows"],
                    retained_rhs_rows=rhs_rows,
                )
            )
            continue
        result = dense_m_panel_capacity(problem, resources, witness, n_tiles=n_tiles)
        rows.append(
            dict(
                n_tiles=n_tiles,
                maximum_m_tiles=result["maximum_m_tiles"],
                limiting_conditions=result["limiting_conditions"],
                retained_rhs_rows=result["retained_rhs_rows"],
            )
        )
        if result["maximum_m_tiles"] >= 1 and all(result["conditions"].values()):
            satisfied.append(n_tiles)
        for name, value in result["conditions"].items():
            if not value:
                blockers[name] = blockers.get(name, 0) + 1
    return dict(
        status="satisfied" if satisfied else "violated",
        predicate="dense_panel_pipeline.dense_m_panel_capacity per original N width",
        satisfied_n_tiles=satisfied,
        violated_bounds=sorted(blockers),
        violated_bound_counts=blockers,
        per_n_tiles=rows,
        source_tiles=dict(m=mt, k=kt, n=nt),
        witness=(
            "for each N width the RHS view receives exactly its retained full-K rows and the LHS "
            "view the remaining operand rows; a construction condition, not an assignment choice"
        ),
    )


def _segmented(shape, geometry):
    from merlin.compiler.schedule.panels.segmented_pipeline import dense_segmented_blocks

    problem, resources, storage = _dense(shape, geometry)
    mt, kt, nt = problem.tiles
    lhs_rows = kt * problem.tile_m  # one complete M tile of the full-K LHS panel
    witness = storage("lhs", lhs_rows)
    if witness is None:
        return dict(
            status="violated",
            predicate="dense_segmented_pipeline.dense_segmented_blocks",
            violated_bounds=["full_lhs_panel_rows"],
            required_lhs_rows=lhs_rows,
            operand_rows=geometry["operand_rows"],
        )
    choices = dense_segmented_blocks(problem, resources, witness)
    blocks = sorted({(c.m, c.k, c.n) for c in choices})
    return dict(
        status="satisfied" if blocks else "violated",
        predicate="dense_segmented_pipeline.dense_segmented_blocks",
        admitted_blocks=[dict(m=m, k=k, n=n) for m, k, n in blocks[:32]],
        admitted_block_count=len(blocks),
        ordering="block dimensions, not preference",
        violated_bounds=[] if blocks else ["no M/N block holds a full-K LHS panel and two RHS K segments"],
        required_lhs_rows_per_m_tile=lhs_rows,
        source_tiles=dict(m=mt, k=kt, n=nt),
        witness="LHS view holds exactly one full-K M tile; RHS view the remaining operand rows",
    )


def _convolution(shape, geometry):
    from merlin.compiler.schedule.shared_operands.shared_rhs import plan_dense_rhs_residency

    edge = geometry["block"]
    chunks = shape["kh"] * shape["kw"] * -(-shape["c"] // edge)
    k_rows, n_cols = chunks * edge, -(-shape["oc"] // edge) * edge
    layout = plan_dense_rhs_residency(
        1,
        k_rows,
        n_cols,
        block=edge,
        operand_rows=geometry["operand_rows"],
        accumulator_rows=geometry["accumulator_rows"],
        rhs_base=edge,
        max_static_rhs_tiles=geometry["operand_rows"] // edge,
    )
    required = edge + chunks * (n_cols // edge) * edge
    return dict(
        status="satisfied" if layout is not None else "violated",
        predicate="convolution_exploration.plan_convolution_rhs_residency capacity owner "
        "(dense_rhs_residency.plan_dense_rhs_residency)",
        required_operand_rows=required,
        operand_rows=geometry["operand_rows"],
        violated_bounds=[] if layout is not None else ["full_rhs_rows_exceed_private_operand_rows"],
        witness=(
            "RHS base directly after one LHS tile; one column interval per output-channel tile. The "
            "original scalar-safe column intervals and any phased fallback stay the source's"
        ),
    )


def _transfer_bound(differences, structure):
    from merlin.analysis.kernels.effects.command_streams import _Protocol

    shapes = [
        shape
        for d in differences
        if d["effect"] == "operand_transfer_extent"
        for shape in (d.get("reference") or {}).get("shapes") or []
    ]
    if not shapes or not structure:
        return dict(status="unknown", missing=["reference transfer extents or selected operand protocol"])
    proto = _Protocol(structure)
    limits = {}
    for (klass, name), binding in proto.bindings.items():
        if "operand_load" not in ((structure.get("instruction_roles") or {}).get(klass) or ()):
            continue
        for axis in ("rows", "cols"):
            if name == proto.extent_field(klass, axis):
                width = proto.widths.get((binding.get("bundle"), binding.get("field")))
                if type(width) is int:
                    limits.setdefault(axis, {})[klass] = (1 << width) - 1
    if set(limits) != {"rows", "cols"}:
        return dict(status="unknown", missing=["declared transfer-bound field widths for operand_load classes"])
    needed = dict(rows=max(s["rows"] for s in shapes), cols=max(s["cols"] for s in shapes))
    fits = {axis: max(limits[axis].values()) >= needed[axis] for axis in needed}
    return dict(
        status="satisfied" if all(fits.values()) else "violated",
        predicate="selected RTL transfer-bound field widths encode the reference's observed extents",
        requested=needed,
        encodable=limits,
        violated_bounds=[f"{axis}_field_width" for axis, ok in fits.items() if not ok],
        limits=["encodability only; the address law and source pitch remain separate requirements"],
    )


def _admission(route, shape, geometry, differences, structure):
    name = route["admission"]
    if name is None:
        return dict(
            status="not_declared",
            capacity_predicate=None,
            reason="the capability declares execution laws only; no capacity predicate exists",
        )
    if name == "selected_transfer_extent_field":
        return _transfer_bound(differences, structure)
    if geometry is None:
        return dict(status="unknown", missing=["selected RTL geometry"])
    family = _family(shape)
    if family is None:
        return dict(
            status="unknown",
            predicate=name,
            missing=["candidate shape: dense m,k,n or convolution kh,kw,c,oc (--candidate-shape)"],
        )
    if family not in route["families"]:
        return dict(
            status="not_applicable",
            predicate=name,
            reason=f"route consumes {route['families']}; the candidate shape is {family}",
        )
    if name.endswith("dense_m_panel_capacity"):
        return _m_panel(shape, geometry)
    if name.endswith("dense_segmented_blocks"):
        return _segmented(shape, geometry)
    if name.endswith("plan_convolution_rhs_residency"):
        return _convolution(shape, geometry)
    return dict(status="unknown", missing=[f"no evaluator for declared predicate {name}"])


def _guarded_admission(route, shape, geometry, differences, structure):
    """A library predicate that raises on these inputs is reported, never mistaken for a decline."""
    try:
        return _admission(route, shape, geometry, differences, structure)
    except (ValueError, TypeError, ArithmeticError, AttributeError, KeyError) as error:
        return dict(
            status="unknown",
            predicate=route["admission"],
            missing=[f"predicate refused these inputs: {type(error).__name__}: {str(error)[:200]}"],
        )


# Preserve the original public lookup and pickle identities.
for _definition in (
    selected_geometry,
    _family,
    _dense,
    _m_panel,
    _segmented,
    _convolution,
    _transfer_bound,
    _admission,
    _guarded_admission,
):
    _definition.__module__ = "merlin.analysis.kernels.mechanisms.applicability"
del _definition


# ---- mechanism_applicability -------------------------------------------------------------------
# Mechanism differences between a reference occurrence and a candidate, with applicable library routes.
#
# Inputs are two static occurrence records (``analysis.json``-shaped: ``schedule_model.role_counts``,
# ``cfg`` and ``derived_effects`` from :mod:`merlin.analysis.kernels.effects.command_streams`), the scheduling
# catalog's declared effect table (``api_catalog.CAPABILITY_EFFECTS``), the selected RTL facts and the
# candidate's own source shape. Every difference carries an evidence pointer and a status:
# ``observed`` (decoded static sites or fields), ``inferred`` (a static site ratio read as a per-compute
# rate inside loops whose trip counts are unknown) or ``unknown``.
#
# For each difference the routes whose declared effect moves the candidate toward the reference are
# listed with their library admission predicate evaluated on the candidate's shape and the selected
# facts, plus the catalog's own usage and requirement text. A difference no route addresses is reported
# as ``missing_capability`` naming that effect. Nothing is ranked or chosen: routes keep catalog order,
# admission witnesses are construction conditions, and the author selects and composes any change.

SCHEMA = "merlin.mechanism_applicability.v1"


def partial_recipe(rows):
    """Set when one role or effect matches the reference while a role's width, buffering or overlap differs."""
    stream = [r for r in rows if r["effect"] in STREAM_EFFECTS]
    if not stream or all(r["direction"] == "unknown" for r in stream):
        return dict(status="unknown", reason=NO_STREAM, matched=[], still_differs=[])

    def key(row):
        return dict(
            effect=row["effect"],
            role=row.get("role") or row.get("operand"),
            direction=row["direction"],
            **({"needed_change": row["needed_change"]} if row.get("needed_change") else {}),
            **({"comparison": row["comparison"]} if row.get("comparison") else {}),
        )

    roled = [r for r in rows if r.get("role") or r.get("operand")]
    matched = [key(r) for r in roled if r.get("direction") == "candidate_matches"]
    beyond = [key(r) for r in roled if r.get("direction") == "candidate_differs" and r.get("beyond_reference")]
    differs = [key(r) for r in stream if r["direction"] == "candidate_differs" and not r.get("beyond_reference")]
    return dict(
        status=bool((matched or beyond) and differs),
        matched=matched,
        beyond_reference=beyond,
        still_differs=differs,
        unknown=[key(r) for r in stream if r["direction"] == "unknown"],
        basis="per-role rows of the proved command stream and per-operand transfer extents; a factual "
        "comparison with the reference, not a statement about speed",
    )


def mechanism_differences(reference, candidate, *, structure=None):
    """Every compared effect; equal values are kept as ``same`` so absence is never silent."""
    ref_id, cand_id = _evidence_id(reference), _evidence_id(candidate)
    rows = [
        _role_rate("operand_load_per_compute", "operand_load", reference, candidate, ref_id, cand_id),
        _role_rate("weight_load_per_compute", "weight_load", reference, candidate, ref_id, cand_id),
        *_transfer(reference, candidate, ref_id, cand_id, structure),
        *_broadcasts(reference, candidate, ref_id, cand_id),
        _config(reference, candidate, ref_id, cand_id),
        _count("sync_sites", "sync", reference, candidate, ref_id, cand_id),
        _loop(reference, candidate, ref_id, cand_id),
    ]
    for row in rows:
        row["difference"] = "unknown" if row["status"] == "unknown" else "differs" if row["needed_change"] else "same"
        row.update(_position(row))
    for row in _stream_rows(reference, candidate, ref_id, cand_id):
        row["difference"] = (
            "unknown"
            if row["direction"] == "unknown"
            else "same"
            if row["direction"] == "candidate_matches"
            else "differs"
        )
        rows.append(row)
    return rows + [
        dict(
            row,
            difference="unknown" if row["status"] == "unknown" else "differs",
            direction="unknown" if row["status"] == "unknown" else "candidate_differs",
        )
        for row in _unknowns(reference, candidate, ref_id, cand_id)
    ]


def command_stream(rows, decoder, *, max_steps=8_000_000, max_commands=1_000_000):
    """Decoded commands of a proved bounded scalar execution, or ``(None, reason)``.

    Feature-detected: until the machine stepper records commands and ``decode_stream`` exists, the
    stream facts stay unknown with :data:`NO_STREAM`."""
    import inspect

    from merlin.analysis.kernels.effects import command_streams as decoded_effects
    from merlin.analysis.observation.riscv import machine as riscv_machine

    counts = getattr(riscv_machine, "finite_control_counts", None)
    decode = getattr(decoded_effects, "decode_stream", None)
    if (
        counts is None
        or decode is None
        or "record_commands" not in inspect.signature(counts).parameters
        or not decoder
        or decoder.get("semantics") is None
    ):
        return None, NO_STREAM
    opcode = decoder["semantics"].isa_constants(decoder["target"]).get("CUSTOM_OPCODE")
    if opcode is None:
        return None, NO_STREAM + ": the selected ISA declares no custom opcode"
    counted = counts(
        list(rows), custom_opcode=opcode, max_steps=max_steps, max_commands=max_commands, record_commands=True
    )
    if counted.get("status") != "proved":
        return None, NO_STREAM + ": " + str(counted.get("reason") or counted.get("status"))
    stream = next((counted[k] for k in ("command_stream", "recorded_commands", "stream") if k in counted), None)
    if stream is None:
        return None, NO_STREAM + ": the stepper recorded no stream"
    if counted.get("command_stream_truncated"):
        return None, NO_STREAM + ": the recorded stream reached its bound (truncated)"
    return decode(stream, decoder=decoder), None


def stream_effects(analysis, structure, stream, *, evidence_id=None):
    """``derive_effects`` facts of one decoded proved stream, by whichever parameter the deriver takes."""
    import inspect

    from merlin.analysis.kernels.effects.derivation import derive_effects

    accepted = inspect.signature(derive_effects).parameters
    options = dict(stream=stream) if "stream" in accepted else dict(commands=stream)
    return derive_effects(analysis, structure, evidence_id=evidence_id, **options).get("facts") or {}


def applicability(reference, candidate, *, shape=None, geometry=None, geometry_reason=None, structure=None):
    """Differences, routes per difference with admission, and missing capabilities. Never ranked."""
    from merlin.compiler.schedule.api_catalog import capability_effects
    from merlin.compiler.schedule.library_sources import CURRENT_SDK_VERSION

    differences = mechanism_differences(reference, candidate, structure=structure)
    routes = capability_effects()
    family = _family(shape)
    out, missing = [], []
    for row in differences:
        entry = dict(row)
        effect = row.get("route_effect") or row["effect"]
        if row["difference"] != "differs" or not row.get("needed_change") or row.get("beyond_reference"):
            entry["routes"] = []
            if row.get("beyond_reference"):
                entry["routes_basis"] = (
                    "the candidate is already past the reference in the direction this "
                    "effect's routes move it (larger means "
                    + (row.get("larger_means") or "unknown")
                    + "); none is sought, since a route would move it further away"
                )
            if row["difference"] == "unknown":
                # Which routes would address this effect once both sides establish it; nothing is evaluated.
                entry["routes_if_established"] = [r["name"] for r in routes if effect in r["changes"]]
            out.append(entry)
            continue
        offered = [r for r in routes if r["changes"].get(effect) == row["needed_change"]]
        entry["routes"] = [
            dict(
                capability=r["name"],
                changes={effect: row["needed_change"]},
                families=r["families"],
                family_match=("unknown" if family is None else family in r["families"]),
                symbols=r["symbols"],
                usage=r["usage"],
                composition_and_scheduling_requirements=r["requires"],
                admission=_guarded_admission(r, shape, geometry, differences, structure),
                laws="required by the capability; not established by this report",
            )
            for r in offered
        ]
        if not offered:
            entry["missing_capability"] = dict(
                effect=effect,
                needed_change=row["needed_change"],
                **({"role": row["role"]} if row.get("role") else {}),
                reason="no catalog capability declares changing this effect in this direction",
            )
        elif family is not None and not any(family in r["families"] for r in offered):
            entry["missing_capability"] = dict(
                effect=effect,
                needed_change=row["needed_change"],
                family=family,
                reason="the capabilities that change this effect consume other source families",
            )
        if "missing_capability" in entry:
            missing.append(entry["missing_capability"])
        out.append(entry)
    return dict(
        schema=SCHEMA,
        differences=out,
        missing_capabilities=missing,
        partial_recipe=partial_recipe(out),
        candidate_shape=dict(value=shape, family=family, source="author_declared" if shape else "not_supplied"),
        selected_geometry=geometry if geometry is not None else dict(status="unknown", reason=geometry_reason),
        catalog=dict(
            api="merlin.compiler.schedule.api_catalog:capability_effects", inventory_version=CURRENT_SDK_VERSION
        ),
        ordering="differences in fixed effect order; routes in catalog order; nothing is ranked",
        choice="selecting, composing and implementing a route is the author's work",
        semantic_equivalence="unknown",
        performance="unknown",
    )
