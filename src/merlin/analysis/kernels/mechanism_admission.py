"""Evaluate library-route admission against candidate shape and selected facts."""

from __future__ import annotations

_MAX_N_TILES = 256  # Bounded per-width admission evaluation; larger extents report the bound as unknown.


_DENSE = ("m", "k", "n")


_CONV = ("kh", "kw", "c", "oc")


def selected_geometry(target, facts):
    """Block edge, operand/accumulator rows and element widths from the selected RTL facts, or a reason."""
    from merlin.compiler.schedule.blocks import BlockScheduleError
    from merlin.compiler.schedule.derive import geometry_from_address_space
    from merlin.development.targetgen.address_space import accumulator_store, derive_address_space, operand_store

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

    from merlin.compiler.schedule.compact_dense import DenseLoopStorage, DenseMemoryView
    from merlin.compiler.transforms.dense_planning import DenseProblem, DenseResources

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
    from merlin.compiler.schedule.dense_panel_pipeline import dense_m_panel_capacity

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
    from merlin.compiler.schedule.dense_segmented_pipeline import dense_segmented_blocks

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
    from merlin.compiler.schedule.dense_rhs_residency import plan_dense_rhs_residency

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
    from merlin.analysis.kernels.decoded_effects import _Protocol

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
    _definition.__module__ = "merlin.analysis.kernels.mechanism_applicability"
del _definition
