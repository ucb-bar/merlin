"""Mechanism differences between a reference occurrence and a candidate, with applicable library routes.

Inputs are two static occurrence records (``analysis.json``-shaped: ``schedule_model.role_counts``,
``cfg`` and ``derived_effects`` from :mod:`merlin.analysis.kernels.decoded_effects`), the scheduling
catalog's declared effect table (``api_catalog.CAPABILITY_EFFECTS``), the selected RTL facts and the
candidate's own source shape. Every difference carries an evidence pointer and a status:
``observed`` (decoded static sites or fields), ``inferred`` (a static site ratio read as a per-compute
rate inside loops whose trip counts are unknown) or ``unknown``.

For each difference the routes whose declared effect moves the candidate toward the reference are
listed with their library admission predicate evaluated on the candidate's shape and the selected
facts, plus the catalog's own usage and requirement text. A difference no route addresses is reported
as ``missing_capability`` naming that effect. Nothing is ranked or chosen: routes keep catalog order,
admission witnesses are construction conditions, and the author selects and composes any change."""

from __future__ import annotations

from fractions import Fraction as Fraction

from merlin.analysis.kernels.mechanism_admission import (
    _CONV,
    _DENSE,
    _MAX_N_TILES,
    _admission,
    _convolution,
    _dense,
    _family,
    _guarded_admission,
    _m_panel,
    _segmented,
    _transfer_bound,
    selected_geometry,
)
from merlin.analysis.kernels.mechanism_metrics import (
    _SEPARATE,
    COMPARISON_SEMANTICS,
    NO_STREAM,
    ROLE_FACTS,
    STREAM_FACTS,
    STREAM_SCOPE,
    _broadcast,
    _broadcasts,
    _config,
    _count,
    _direction,
    _evidence_id,
    _extent,
    _fact,
    _loop,
    _loops,
    _order,
    _plain,
    _position,
    _rate_status,
    _ratio,
    _role_rate,
    _roles,
    _summary_value,
    _transfer,
    _unknowns,
    stream_fact,
)
from merlin.analysis.kernels.mechanism_ports import (
    _SIDES,
    FALLBACK,
    PORT,
    REGISTER,
    _aggregate,
    _buffering,
    _consumer_ports,
    _distance,
    _distance_value,
    _first,
    _operand_gaps,
    _operand_groups,
    _overlap,
    _overlap_value,
    _port_match,
    _role_entries,
    _role_fallback,
    _side_maps,
    _summarizer,
    _width,
)
from merlin.analysis.kernels.mechanism_stream import (
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

    from merlin.analysis import riscv_machine
    from merlin.analysis.kernels import decoded_effects

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

    from merlin.analysis.kernels.decoded_effects import derive_effects

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
