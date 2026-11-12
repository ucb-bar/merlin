"""Mechanism differences between a reference occurrence and a candidate, with applicable library routes.

Inputs are two static occurrence records (``analysis.json``-shaped: ``schedule_model.role_counts``,
``cfg`` and ``derived_effects`` from :mod:`merlin.kernels.decoded_effects`), the scheduling
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

from .mechanism_admission import (
    _CONV as _CONV,
)
from .mechanism_admission import (
    _DENSE as _DENSE,
)
from .mechanism_admission import (
    _MAX_N_TILES as _MAX_N_TILES,
)
from .mechanism_admission import (
    _admission as _admission,
)
from .mechanism_admission import (
    _convolution as _convolution,
)
from .mechanism_admission import (
    _dense as _dense,
)
from .mechanism_admission import (
    _family as _family,
)
from .mechanism_admission import (
    _guarded_admission as _guarded_admission,
)
from .mechanism_admission import (
    _m_panel as _m_panel,
)
from .mechanism_admission import (
    _segmented as _segmented,
)
from .mechanism_admission import (
    _transfer_bound as _transfer_bound,
)
from .mechanism_admission import (
    selected_geometry as selected_geometry,
)
from .mechanism_metrics import (
    _SEPARATE as _SEPARATE,
)
from .mechanism_metrics import (
    COMPARISON_SEMANTICS as COMPARISON_SEMANTICS,
)
from .mechanism_metrics import (
    NO_STREAM as NO_STREAM,
)
from .mechanism_metrics import (
    ROLE_FACTS as ROLE_FACTS,
)
from .mechanism_metrics import (
    STREAM_FACTS as STREAM_FACTS,
)
from .mechanism_metrics import (
    STREAM_SCOPE as STREAM_SCOPE,
)
from .mechanism_metrics import (
    _broadcast as _broadcast,
)
from .mechanism_metrics import (
    _broadcasts as _broadcasts,
)
from .mechanism_metrics import (
    _config as _config,
)
from .mechanism_metrics import (
    _count as _count,
)
from .mechanism_metrics import (
    _direction as _direction,
)
from .mechanism_metrics import (
    _evidence_id as _evidence_id,
)
from .mechanism_metrics import (
    _extent as _extent,
)
from .mechanism_metrics import (
    _fact as _fact,
)
from .mechanism_metrics import (
    _loop as _loop,
)
from .mechanism_metrics import (
    _loops as _loops,
)
from .mechanism_metrics import (
    _order as _order,
)
from .mechanism_metrics import (
    _plain as _plain,
)
from .mechanism_metrics import (
    _position as _position,
)
from .mechanism_metrics import (
    _rate_status as _rate_status,
)
from .mechanism_metrics import (
    _ratio as _ratio,
)
from .mechanism_metrics import (
    _role_rate as _role_rate,
)
from .mechanism_metrics import (
    _roles as _roles,
)
from .mechanism_metrics import (
    _summary_value as _summary_value,
)
from .mechanism_metrics import (
    _transfer as _transfer,
)
from .mechanism_metrics import (
    _unknowns as _unknowns,
)
from .mechanism_metrics import (
    stream_fact as stream_fact,
)
from .mechanism_ports import (
    _SIDES as _SIDES,
)
from .mechanism_ports import (
    FALLBACK as FALLBACK,
)
from .mechanism_ports import (
    PORT as PORT,
)
from .mechanism_ports import (
    REGISTER as REGISTER,
)
from .mechanism_ports import (
    _aggregate as _aggregate,
)
from .mechanism_ports import (
    _buffering as _buffering,
)
from .mechanism_ports import (
    _consumer_ports as _consumer_ports,
)
from .mechanism_ports import (
    _distance as _distance,
)
from .mechanism_ports import (
    _distance_value as _distance_value,
)
from .mechanism_ports import (
    _first as _first,
)
from .mechanism_ports import (
    _operand_gaps as _operand_gaps,
)
from .mechanism_ports import (
    _operand_groups as _operand_groups,
)
from .mechanism_ports import (
    _overlap as _overlap,
)
from .mechanism_ports import (
    _overlap_value as _overlap_value,
)
from .mechanism_ports import (
    _port_match as _port_match,
)
from .mechanism_ports import (
    _role_entries as _role_entries,
)
from .mechanism_ports import (
    _role_fallback as _role_fallback,
)
from .mechanism_ports import (
    _side_maps as _side_maps,
)
from .mechanism_ports import (
    _summarizer as _summarizer,
)
from .mechanism_ports import (
    _width as _width,
)
from .mechanism_stream import (
    _GROUP_BASIS as _GROUP_BASIS,
)
from .mechanism_stream import (
    _INTERLEAVE as _INTERLEAVE,
)
from .mechanism_stream import (
    _STREAM_BASIS as _STREAM_BASIS,
)
from .mechanism_stream import (
    _STREAM_METRIC as _STREAM_METRIC,
)
from .mechanism_stream import (
    EXTRA_FACTS as EXTRA_FACTS,
)
from .mechanism_stream import (
    STREAM_EFFECTS as STREAM_EFFECTS,
)
from .mechanism_stream import (
    _capacity_fraction as _capacity_fraction,
)
from .mechanism_stream import (
    _extra_rows as _extra_rows,
)
from .mechanism_stream import (
    _group_columns as _group_columns,
)
from .mechanism_stream import (
    _in_loop as _in_loop,
)
from .mechanism_stream import (
    _interleave_rows as _interleave_rows,
)
from .mechanism_stream import (
    _number as _number,
)
from .mechanism_stream import (
    _on_accelerator as _on_accelerator,
)
from .mechanism_stream import (
    _output_group_rows as _output_group_rows,
)
from .mechanism_stream import (
    _per_role as _per_role,
)
from .mechanism_stream import (
    _scalar as _scalar,
)
from .mechanism_stream import (
    _scalar_map as _scalar_map,
)
from .mechanism_stream import (
    _stream_direction as _stream_direction,
)
from .mechanism_stream import (
    _stream_rows as _stream_rows,
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
    from merlin.kernels import decoded_effects

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

    from merlin.kernels.decoded_effects import derive_effects

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
