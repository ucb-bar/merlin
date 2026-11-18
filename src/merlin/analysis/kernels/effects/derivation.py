"""Derived static effects over decoded accelerator commands, from declared structure only.

Every fact is computed from the selected instruction-role roster, the selected
operand protocol (field names, register bundles, RTL field bindings and
interpretations), its RTL field layouts and the contract's declared epilogue
carriers. No instruction or field spelling, model table or target constant
selects a value. A fact the declared data cannot support stays ``unknown`` with
a reason; observed values are lexical static-listing facts, not dynamic issue
order, execution multiplicity, legality or performance.

Builder call (one per occurrence, after ``machine_projection``)::

    structure = selected_structure(decoder, contract_document)
    commands = decoded_commands(body, decoder=decoder)  # optional
    report["derived_effects"] = derive_effects(report, structure, commands=commands, evidence_id=evidence_id)

``report`` is the occurrence analysis record (``analysis.json``). Without
``commands`` only summary-supported facts are derived; ordered facts are unknown.
"""

from __future__ import annotations

from collections import Counter

from merlin.analysis.kernels.effects.static_geometry import (
    _DOMAIN,
    _MOVEMENT_ROLES,
    _VALUES,
    _WITNESSES,
    ORDERING,
    SCHEMA,
    _bits,
    _broadcast,
    _close,
    _digest,
    _epilogue,
    _fact,
    _field_domain,
    _loop_nest,
    _merge,
    _Protocol,
    _register_domain,
    _retention,
    _roles,
    _site,
    _summary_broadcast,
    _summary_windows,
    _symbolic_bits,
    _tiles,
    _unknown,
    _windows,
    decoded_commands,
    local_row_geometry,
    read_pinned_facts,
    read_pinned_sidecar,
    selected_structure,
)

# ---- decoded_effects ---------------------------------------------------------------------------


def _operand_buffering(view, stream, truncated, evidence_id):
    from merlin.analysis.kernels.effects.command_streams import (
        _BOUND,
        _RUNS_PREFIX,
        BUFFERING_DEFINITION,
        STREAM_SCOPE,
        _consumption,
        _rounds,
        _split_by_base,
        _stream_fact,
    )

    consumption = _consumption(view)
    if not consumption.loads:
        return _unknown("proved stream has no load with a resolved declared local-address field", scope=STREAM_SCOPE)
    value = {}
    for (role, operand), loads in sorted(_split_by_base(consumption.groups()).items()):
        bases = sorted({load["footprint"][0][0] for load in loads if load["footprint"]})
        entry = dict(
            definition=BUFFERING_DEFINITION,
            flags=list(loads[0]["flags"]),
            loads=len(loads),
            consumer_port=consumption.port(loads),
            distinct_base_addresses=len(bases),
            base_range=[bases[0], bases[-1]] if bases else None,
            block_steps=sorted({load["block_step"] for load in loads if load.get("block_step")}),
        )
        unresolved = Counter(load["reason"] for load in loads if load["footprint"] is None)
        if unresolved:
            entry.update(
                status="unknown",
                disjoint_windows=None,
                alternating=None,
                reason="; ".join(f"{n} loads: {why}" for why, n in unresolved.most_common()),
            )
            value.setdefault(role, {})[operand] = entry
            continue
        spaces = {load["space"] for load in loads}
        if spaces != {None}:
            space = next(iter(spaces))
            entry.update(windows_relative_to="allocation", allocation=space[0] if len(spaces) == 1 else None)
            if len(spaces) > 1:
                entry.update(
                    status="unknown",
                    disjoint_windows=None,
                    alternating=None,
                    allocations=len(spaces),
                    reason=f"loads address {len(spaces)} bases (exact or allocation-relative); "
                    "disjointness across allocations not established",
                )
                value.setdefault(role, {})[operand] = entry
                continue
            if any(hi > 1 << space[1] for load in loads for _, hi in load["footprint"]):
                entry.update(
                    status="unknown",
                    disjoint_windows=None,
                    alternating=None,
                    reason="a footprint wraps the allocation-relative offset width",
                )
                value.setdefault(role, {})[operand] = entry
                continue
        if any(not load["footprint"] or any(hi <= lo for lo, hi in load["footprint"]) for load in loads):
            entry.update(
                status="unknown",
                disjoint_windows=None,
                alternating=None,
                reason="a load declares zero transfer rows or columns; its round writes no row",
            )
            value.setdefault(role, {})[operand] = entry
            continue
        regions, sequence, straddling = _rounds(loads)
        visits = [w for i, w in enumerate(sequence) if i == 0 or w != sequence[i - 1]]
        if straddling:
            entry.update(
                status="unknown",
                disjoint_windows=None,
                alternating=None,
                fill_rounds=len(sequence),
                straddling_rounds=straddling,
                reason=f"{straddling} of {len(sequence)} fill rounds share rows with several regions or only "
                "part of one (loads issued into another region before the round was first read); "
                "the regions are not established",
            )
            value.setdefault(role, {})[operand] = entry
            continue
        entry.update(
            status="observed",
            fill_rounds=len(sequence),
            disjoint_windows=len(regions),
            windows=regions[:_BOUND],
            window_sequence_prefix=visits[:_RUNS_PREFIX],
            window_switches=len(visits) - 1,
            # More than one region, and fill rounds return to a region after leaving it.
            alternating=len(regions) > 1 and len(visits) > len(regions),
            unconsumed_loads=sum(1 for load in loads if load["first"] is None),
        )
        value.setdefault(role, {})[operand] = entry
    return _stream_fact(
        value,
        [
            f"operand_protocol#address_field={view.proto.address_field}",
            "operand_protocol#roles.local_address_flags",
            "structure#local_geometry",
            "structure#operand_field_kind_sources",
            f"{evidence_id}#command_stream",
        ],
        stream,
        truncated,
        definition="a fill round is the run of one operand's loads issued before any of their rows is first read "
        "by a later issue; rounds whose block footprints share a local row form one region; "
        "alternating: several regions and the rounds return to a region after leaving it",
        limits=[
            "footprints are declared rows x ceil(cols / derived row elements) blocks at the configured "
            "per-state block step; banks are not interpreted"
        ]
        + (
            [f"{n} loads unresolved: {why}" for why, n in consumption.unresolved.most_common()]
            if consumption.unresolved
            else []
        )
        + consumption.relative_limits(),
    )


def _distance(loads, consumption=None):
    from merlin.analysis.kernels.effects.command_streams import _Consumption, _median, _spread

    consumed = [load for load in loads if load["first"] is not None]
    ahead = [load for load in consumed if load["computes_between"] > 0]
    loads_between = [load["loads_between"] for load in consumed]
    relative = {}
    if consumption is not None and consumption.spaces:
        # An unread load later read through another base may have been consumed: not established.
        relative = dict(unestablished_consumer_loads=sum(1 for load in loads if consumption.unestablished(load)))
    return dict(
        **relative,
        loads=len(loads),
        consumed_loads=len(consumed),
        unconsumed_loads=len(loads) - len(consumed),
        unresolved_footprint_loads=sum(1 for load in loads if load["footprint"] is None),
        consumer_port=_Consumption.port(loads),
        loads_between=_spread(loads_between),
        computes_between=_spread([load["computes_between"] for load in consumed]),
        median_loads_between=_median(loads_between),
        min_loads_between=min(loads_between) if loads_between else None,
        fraction_with_compute_ahead=round(len(ahead) / len(consumed), 4) if consumed else None,
    )


def _prefetch_distance(view, stream, truncated, evidence_id):
    """Per operand: loads (and computes) issued between each load and the first issue reading its rows."""
    from merlin.analysis.kernels.effects.command_streams import STREAM_SCOPE, _consumption, _stream_fact

    consumption = _consumption(view)
    if not consumption.loads:
        return _unknown("proved stream has no load with a resolved declared local-address field", scope=STREAM_SCOPE)
    tensors = {}
    for (_, operand), loads in consumption.groups().items():
        tensors.setdefault(operand, []).extend(loads)
    if not any(load["first"] is not None for load in consumption.loads):
        return _unknown(
            "no later issue reads a row any resolved load wrote",
            scope=STREAM_SCOPE,
            unresolved=dict(consumption.unresolved),
        )
    return _stream_fact(
        dict(
            by_operand={k: _distance(v, consumption) for k, v in sorted(tensors.items())},
            all=_distance(consumption.loads, consumption),
        ),
        ["structure#instruction_roles", "structure#local_geometry", f"{evidence_id}#command_stream"],
        stream,
        truncated,
        partial=bool(consumption.unresolved),
        definition="for each load, the loads and the compute issues issued after it and before the first later "
        "issue whose declared local-address field reads a row the load wrote (latest writer); "
        "fraction_with_compute_ahead: consumed loads with at least one compute in between",
        limits=["issue order of a proved scalar execution; a measured distance, not a timing or speed claim"]
        + [f"{n} loads unresolved: {why}" for why, n in consumption.unresolved.most_common()]
        + consumption.relative_limits(),
    )


def _issue_overlap(view, stream, truncated, evidence_id):
    from merlin.analysis.kernels.effects.command_streams import (
        _RUNS_PREFIX,
        STREAM_SCOPE,
        _consumption,
        _median,
        _spread,
        _stream_fact,
    )

    runs, pending, loads = [], [], []
    computes = fences = barriers = 0
    for entry, klass, _, kind in view.entries:
        if kind == "local_staging":
            continue
        if runs and runs[-1][0] == kind:
            runs[-1][1] += 1
        else:
            runs.append([kind, 1])
        if kind == "barrier":
            barriers += 1
            fences += klass == "FENCE"
            pending = []
        elif kind == "compute":
            computes += 1
            for load in pending:
                load[1] += 1
        elif kind == "load":
            operand, _ = view.dram(klass, entry["fields"])
            load = [operand or "unattributed", 0]
            loads.append(load)
            pending.append(load)
    if not computes and not loads:
        return _unknown("proved stream has no compute or load issues", scope=STREAM_SCOPE)

    def overlap(rows):
        if not rows:
            return {}
        hit = sum(1 for _, n in rows if n) / len(rows)
        return dict(
            loads=len(rows),
            loads_with_compute_before_barrier=round(hit, 4),
            loads_without_compute_before_barrier=round(1 - hit, 4),
            computes_after_load_before_barrier=_spread([n for _, n in rows]),
        )

    compute_runs = [n for k, n in runs if k == "compute"]
    load_runs = [n for k, n in runs if k == "load"]
    # Interleave over the compute/load order alone (barriers and other issues between them are
    # ignored): a compute run followed by a load band, and compute runs between two load bands.
    bands = []
    for kind, n in runs:
        if kind not in ("compute", "load"):
            continue
        if bands and bands[-1][0] == kind:
            bands[-1][1] += n
        else:
            bands.append([kind, n])
    ratios = [m / n for (k, n), (j, m) in zip(bands, bands[1:], strict=False) if k == "compute" and j == "load"]
    between = [
        n
        for (p, _), (k, n), (q, _) in zip(bands, bands[1:], bands[2:], strict=False)
        if k == "compute" and p == q == "load"
    ]
    whole = overlap(loads)
    ports = _consumption(view).by_tensor_port()
    value = dict(
        runs_prefix=[list(run) for run in runs[:_RUNS_PREFIX]],
        run_count=len(runs),
        compute_run=_spread(compute_runs),
        load_run=_spread(load_runs),
        median_compute_run=_median(compute_runs),
        median_load_run=_median(load_runs),
        loads_per_compute_run=_median(ratios),
        compute_runs_followed_by_load_band=len(ratios),
        median_compute_run_between_load_bands=_median(between),
        computes=computes,
        loads=len(loads),
        loads_with_compute_before_barrier=whole.get("loads_with_compute_before_barrier"),
        loads_without_compute_before_barrier=whole.get("loads_without_compute_before_barrier"),
        by_operand={
            k: dict(overlap([row for row in loads if row[0] == k]), consumer_port=ports.get(k))
            for k in sorted({row[0] for row in loads})
        },
        fences=fences,
        barriers=barriers,
        computes_per_fence=round(computes / fences, 4) if fences else None,
        computes_per_barrier=round(computes / barriers, 4) if barriers else None,
    )
    return _stream_fact(
        value,
        ["structure#instruction_roles", f"{evidence_id}#command_stream"],
        stream,
        truncated,
        limits=["issue order of a proved scalar execution; hardware overlap and timing are not observed"],
    )


def _readout_pitch_fields(view, structure):
    """``{config class: field}`` of the stride-kind configuration fields whose RTL sink lives in the same
    module as a readout class's declared transfer fields (the memory row pitch of readouts)."""
    readout_bundles = set()
    for (klass, name), binding in view.proto.bindings.items():
        roles = _roles(structure, klass)
        if any(r in ("readout", "commit") for r in roles) and binding.get("field") == view.proto.extent["cols"]:
            readout_bundles.add(binding.get("bundle"))
    modules = {
        row.get("module")
        for row in structure.get("field_layouts") or ()
        if row.get("bundle") in readout_bundles and row.get("module")
    }
    found = {}
    for row in structure.get("operand_field_kind_sources") or ():
        klass, _, name = str(row.get("label") or "").partition(".")
        if row.get("kind") == "stride" and row.get("module") in modules and (klass, name) in view.proto.bindings:
            found.setdefault(klass, []).append(name)
    return {klass: names[0] for klass, names in found.items() if len(names) == 1}


def _output_group_extent(view, stream, truncated, evidence_id):
    """Per barrier-delimited block: computes and the readouts' declared extents and memory span.

    ``column_groups`` counts the distinct column positions (byte offset modulo the configured readout row
    pitch) a block's readouts start at; ``output_columns`` is that count times the readouts' common
    column extent, when the pitch and a uniform extent are established."""
    from merlin.analysis.kernels.effects.command_streams import _BOUND, STREAM_SCOPE, _median, _stream_fact

    pitch_fields = _readout_pitch_fields(view, view.structure)
    pitch = None
    blocks, block = [], None

    def fresh():
        return dict(
            computes=0,
            readouts=0,
            shapes=Counter(),
            elements=0,
            offsets=set(),
            rows=set(),
            starts=set(),
            row_starts=set(),
            cols=set(),
            pitch_unknown=0,
        )

    block = fresh()
    for entry, klass, _, kind in view.entries:
        if klass in pitch_fields:
            value = entry["fields"].get(pitch_fields[klass])
            pitch = value if type(value) is int and value > 0 else None
        if kind == "barrier":
            if block["computes"] or block["readouts"]:
                blocks.append(block)
            block = fresh()
        elif kind == "compute":
            block["computes"] += 1
        elif kind == "readout":
            r, c = view.extent(klass, entry["fields"])
            block["readouts"] += 1
            block["shapes"][(r, c)] += 1
            block["rows"].add(r)
            block["cols"].add(c)
            if r is not None and c is not None:
                block["elements"] += r * c
            operand, _ = view.dram(klass, entry["fields"])
            record = entry["fields"].get(view.operand_fields.get(klass) or "")
            offset = None
            if isinstance(record, dict):
                offset = record.get("byte_offset") if record.get("pointer_base") else record.get("raw")
                if type(offset) is int:
                    block["offsets"].add((operand, offset))
            if type(offset) is int and pitch:
                block["starts"].add((operand, offset % pitch))
                block["row_starts"].add((operand, offset // pitch))
            else:
                block["pitch_unknown"] += 1
    if block["computes"] or block["readouts"]:
        blocks.append(block)
    out = [b for b in blocks if b["readouts"]]
    if not out:
        return _unknown("proved stream has no readout-role issue with declared extents", scope=STREAM_SCOPE)

    def row(b):
        offsets = sorted(o for _, o in b["offsets"])
        single = len(b["rows"]) == 1 and None not in b["rows"]
        rows = next(iter(b["rows"])) if single else None
        uniform = len(b["cols"]) == 1 and None not in b["cols"]
        groups = len(b["starts"]) if not b["pitch_unknown"] else None
        return dict(
            computes=b["computes"],
            readouts=b["readouts"],
            shapes=[dict(rows=r, cols=c, count=n) for (r, c), n in b["shapes"].most_common(_BOUND)],
            readout_elements=b["elements"],
            row_extent=rows,
            column_extent_if_one_row_band=b["elements"] // rows if rows else None,
            column_groups=groups,
            row_bands=len(b["row_starts"]) if not b["pitch_unknown"] else None,
            output_columns=groups * next(iter(b["cols"])) if groups is not None and uniform else None,
            distinct_memory_offsets=len(offsets),
            memory_byte_span=offsets[-1] - offsets[0] if offsets else None,
        )

    rows = [row(b) for b in out]
    columns = [r["output_columns"] for r in rows]
    value = dict(
        blocks=len(rows),
        blocks_prefix=rows[:_BOUND],
        median_readouts_per_block=_median([r["readouts"] for r in rows]),
        median_readout_elements_per_block=_median([r["readout_elements"] for r in rows]),
        column_extents=dict(Counter(r["column_extent_if_one_row_band"] for r in rows).most_common(_BOUND)),
        row_extents=dict(Counter(r["row_extent"] for r in rows).most_common(_BOUND)),
        median_output_columns=_median(columns) if None not in columns else None,
        output_columns=dict(Counter(columns).most_common(_BOUND)),
        readout_pitch_fields=pitch_fields,
        median_computes_per_readout_block=_median([r["computes"] for r in rows]),
        compute_only_blocks=sum(1 for b in blocks if not b["readouts"]),
    )
    return _stream_fact(
        value,
        [
            "structure#instruction_roles",
            "field_layouts#transfer_bound",
            "structure#operand_field_kind_sources",
            f"{evidence_id}#command_stream",
        ],
        stream,
        truncated,
        limits=[
            "column_extent_if_one_row_band assumes a block's readouts share one row band; "
            "memory_byte_span lets a consumer check that against the declared row stride",
            "output_columns counts distinct column start positions modulo the configured readout row pitch "
            "times the common column extent; overlapping column tiles are not excluded",
        ]
        + (
            []
            if pitch_fields
            else ["no stride-kind configuration field shares the readout's RTL module; output_columns stays unknown"]
        ),
    )


def _extension_facts(stream, structure):
    """Facts of the optional ``stream_effects`` module over the same decoded stream."""
    from merlin.analysis.kernels.effects.command_streams import _STREAM_FACTS

    try:
        from merlin.analysis.kernels.effects import stream_facts as stream_effects
    except ImportError:
        return {}
    derive = getattr(stream_effects, "derive", None)
    if derive is None:
        return {}
    try:
        facts = derive(stream, structure) or {}
    except (ArithmeticError, LookupError, TypeError, ValueError) as error:
        return {"stream_effects": _unknown(f"stream_effects.derive failed: {type(error).__name__}: {error}")}
    return {name: fact for name, fact in facts.items() if name not in _STREAM_FACTS}


def stream_facts(stream, structure, *, evidence_id="occurrence", arguments=None, truncated=False):
    """The issue-order facts of one decoded proved stream; unknown without one."""
    from merlin.analysis.kernels.effects.command_streams import (
        _NO_STREAM,
        _STREAM_FACTS,
        STREAM_SCOPE,
        _role_transfer_width,
        _StreamView,
    )

    if stream is None:
        return {name: _unknown(_NO_STREAM, scope=STREAM_SCOPE) for name in _STREAM_FACTS}
    view = _StreamView(stream, structure or {}, _Protocol(structure or {}), arguments)
    facts = dict(
        role_transfer_width=_role_transfer_width(view, stream, truncated, evidence_id),
        operand_buffering=_operand_buffering(view, stream, truncated, evidence_id),
        issue_overlap=_issue_overlap(view, stream, truncated, evidence_id),
        prefetch_distance=_prefetch_distance(view, stream, truncated, evidence_id),
        output_group_extent=_output_group_extent(view, stream, truncated, evidence_id),
    )
    facts.update(_extension_facts(stream, structure or {}))
    return facts


def derive_effects(
    analysis, structure, *, commands=None, evidence_id=None, stream=None, arguments=None, stream_truncated=False
):
    """Return the ``derived_effects`` block for one occurrence analysis record.

    ``analysis``: the occurrence's ``analysis.json`` dict. ``structure``: from
    :func:`selected_structure` (role roster, operand protocol, RTL field layouts,
    load-state selection/ports, encoding, epilogue carriers). ``commands``: the
    optional complete ordered site list from :func:`decoded_commands`.
    ``stream``: the optional :func:`decode_stream` of a proved execution; it adds
    ``role_transfer_width``, ``operand_buffering``, ``issue_overlap``, ``prefetch_distance`` and
    ``output_group_extent`` (each unknown
    with ``no_proved_command_stream`` without it). ``arguments``: exact entry argument
    registers (``a0``..) that tie exact memory addresses to tensor arguments.
    """
    analysis, structure = analysis or {}, structure or {}
    evidence_id = (
        evidence_id or (analysis.get("evidence") or {}).get("evidence_id") or analysis.get("region_id") or "occurrence"
    )
    projection = analysis.get("mechanism_projection") or {}
    proto = _Protocol(structure)
    unknowns = []
    consistency = None
    if commands is not None:
        replay = Counter(r["instruction"] for r in commands)
        declared = Counter({k: v for k, v in (projection.get("instruction_classes") or {}).items() if v})
        consistency = dict(
            status="consistent" if not declared or replay == declared else "differs",
            replayed_sites=sum(replay.values()),
            analysis_classified_sites=sum(declared.values()),
        )
        if consistency["status"] == "differs":
            unknowns.append(dict(fact="*", reason="replayed commands differ from the analysis record"))
    if not projection:
        facts = {
            name: _unknown("occurrence has no selected accelerator projection")
            for name in (
                "operand_retention",
                "broadcast_loads",
                "transfer_tiles",
                "local_address_windows",
                "epilogue_configuration",
            )
        }
    elif commands is None:
        reason = "ordered decoded commands not supplied"
        facts = dict(
            operand_retention=_unknown(reason),
            broadcast_loads=_summary_broadcast(analysis, structure, proto),
            transfer_tiles=_unknown(reason + "; marginal profiles cannot pair rows with cols"),
            local_address_windows=_summary_windows(analysis, structure, proto),
            epilogue_configuration=_epilogue(None, structure, proto, evidence_id, analysis),
        )
    else:
        facts = dict(
            operand_retention=_retention(commands, structure, proto, evidence_id),
            broadcast_loads=_broadcast(commands, structure, proto, evidence_id),
            transfer_tiles=_tiles(commands, structure, proto, evidence_id),
            local_address_windows=_windows(commands, structure, proto, evidence_id),
            epilogue_configuration=_epilogue(commands, structure, proto, evidence_id, analysis),
        )
    roles = structure.get("instruction_roles") or {}
    observed = [k for k, v in (projection.get("instruction_classes") or {}).items() if v]
    if observed and all(roles.get(k) for k in observed):
        loops = sum(v for k, v in projection["instruction_classes"].items() if "loop_descriptor" in roles[k])
        facts["hardware_loop"] = _fact(
            dict(loop_descriptor_sites=loops, uses_hardware_loop=bool(loops)),
            "observed",
            scope=dict(static_sites=sum(projection["instruction_classes"].values())),
            evidence=["structure#instruction_roles", "analysis#mechanism_projection.instruction_classes"],
        )
    else:
        facts["hardware_loop"] = _unknown("role roster does not cover every observed instruction class")
    dataflow = [unit.get("dataflow") for unit in structure.get("compute_units") or () if unit.get("dataflow")]
    facts["dataflow"] = (
        _fact(sorted(set(map(str, dataflow))), "declared", evidence=["structure#compute_units"])
        if dataflow
        else _unknown("no selected protocol field or compute-unit declaration binds a dataflow selector")
    )
    facts["loop_nest"] = _loop_nest(projection, analysis, evidence_id)
    trips = (analysis.get("cfg") or {}).get("trip_counts", "unknown")
    facts["operation"] = _unknown(
        "operation kind and logical dimensions need a generic contract or loop-bound "
        "analysis; static trip counts are " + (trips if isinstance(trips, str) else str(trips.get("status"))),
        kind=None,
        logical_dims=None,
        dtypes=None,
    )
    facts["layer_identity"] = _unknown("no source or build correspondence; names are never inferred")
    facts.update(
        stream_facts(stream, structure, evidence_id=evidence_id, arguments=arguments, truncated=stream_truncated)
    )
    for name, fact in facts.items():
        if fact["status"] == "unknown":
            unknowns.append(dict(fact=name, reason=fact.get("reason")))
    return dict(
        schema=SCHEMA,
        evidence_id=evidence_id,
        basis="ordered_decoded_commands" if commands is not None else "analysis_summary",
        ordering=ORDERING,
        replay=consistency,
        inputs=dict(
            structure_sha256=_digest({k: v for k, v in structure.items() if k != "input_pins"}),
            input_pins=structure.get("input_pins") or [],
            commands_sha256=None if commands is None else _digest(commands),
            **({} if stream is None else dict(stream_sha256=_digest(stream), stream_entries=len(stream))),
        ),
        facts=facts,
        unknowns=unknowns,
        not_established=[
            "dynamic_execution_multiplicity",
            "optimization_legality",
            "numerical_equivalence",
            "performance",
        ],
    )
