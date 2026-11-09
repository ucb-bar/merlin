"""Static traffic, schedule verification and hardware-capacity observations."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

MOVEMENT_IN = ("operand_load", "weight_load")


DATA_MOVEMENT = ("operand_load",)


COMPUTE = "accumulate"


def mechanisms():
    """Accepted mechanism names: exactly the fact names ``derive_effects`` reports."""
    from merlin.kernels.decoded_effects import derive_effects

    return tuple(derive_effects({}, {})["facts"])


def traffic(rows, roles):
    """Per-role static sites and proved per-call executions; unknown when any site is unproved."""
    static, dynamic, unproved, unrostered = Counter(), Counter(), Counter(), Counter()
    for row in rows:
        names = tuple((roles or {}).get(row.get("class")) or ())
        if not names:
            unrostered[str(row.get("class"))] += 1
            continue
        count = row.get("executions")
        for role in names:
            static[role] += 1
            if type(count) is int and count >= 0:
                dynamic[role] += count
            else:
                unproved[role] += 1
    per_role = {
        role: dict(
            static_sites=static[role],
            dynamic=dynamic[role] if not unproved[role] else None,
            dynamic_status="proved" if not unproved[role] else "unknown",
            unproved_sites=unproved[role],
        )
        for role in sorted(static)
    }
    return dict(
        per_role=per_role,
        unrostered_sites=dict(unrostered),
        count_basis="static IR sites x counted-loop executions per kernel call (None: not proved)",
        ratios=dict(static=_ratio(per_role, "static_sites"), dynamic=_ratio(per_role, "dynamic")),
    )


def _ratio(per_role, key):
    # This census contains every observed role. An absent role has zero sites;
    # a present role without its count is unproved rather than zero.
    loads = [per_role[role].get(key) if role in per_role else 0 for role in MOVEMENT_IN]
    compute = per_role.get(COMPUTE, {}).get(key)
    if any(value is None for value in loads) or compute is None:
        return dict(loads_per_compute=None, status="unknown", reason="unproved_execution_count")
    if not compute:
        return dict(loads_per_compute=None, status="unknown", reason="no_accumulate_role_sites")
    by_role = {role: _round(Fraction(value, compute)) for role, value in zip(MOVEMENT_IN, loads)}
    moved = sum(value for role, value in zip(MOVEMENT_IN, loads) if role in DATA_MOVEMENT)
    preloads = sum(value for role, value in zip(MOVEMENT_IN, loads) if role not in DATA_MOVEMENT)
    return dict(
        loads_per_compute=_round(Fraction(sum(loads), compute)),
        by_role=by_role,
        data_movement_loads_per_compute=_round(Fraction(moved, compute)),
        preloads_per_compute=_round(Fraction(preloads, compute)),
        data_movement_roles=[r for r in MOVEMENT_IN if r in DATA_MOVEMENT],
        preload_roles=[r for r in MOVEMENT_IN if r not in DATA_MOVEMENT],
        loads=sum(loads),
        compute=compute,
        status="observed",
    )


def _round(fraction):
    return round(float(fraction), 6)


def effects(rows, decoder, structure, evidence_id):
    """``derive_effects`` over the decoded IR sites in IR order (lexical, not issue order)."""
    if structure is None:
        return None
    from merlin.kernels.decoded_effects import derive_effects

    commands = [
        dict(
            index=i,
            pc=i,
            instruction=row["class"],
            fields=dict(row.get("decoded") or {}),
            status="decoded" if row.get("class") != "UNKNOWN" else "unknown",
            operands=dict(rs1=row.get("rs1") or {}, rs2=row.get("rs2") or {}),
        )
        for i, row in enumerate(rows)
        if (decoder.get("structure") or {}).get("instruction_roles", {}).get(row.get("class")) is not None
    ]
    analysis = dict(mechanism_projection=dict(instruction_classes=dict(Counter(c["instruction"] for c in commands))))
    return derive_effects(analysis, structure, commands=commands, evidence_id=evidence_id)


def receipts(command_buffer):
    """Emitted compiler-lowering records: route selections, refusals, schedule IR (compiler-reported)."""
    out = []
    lowering = ((command_buffer or {}).get("params") or {}).get("compiler_lowering") or {}
    for family, records in sorted(lowering.items()):
        for record in records if isinstance(records, list) else [records]:
            if not isinstance(record, dict):
                continue
            row = dict(family=family, schema=record.get("schema"))
            for key, value in sorted(record.items()):
                if key.startswith("requested_") and isinstance(value, str):
                    suffix = key[len("requested_") :]
                    row.setdefault("routes", []).append(
                        dict(axis=suffix, requested=value, selected=record.get("selected_" + suffix))
                    )
                elif key.endswith("refusal") or key.endswith("refusals"):
                    row.setdefault("refusals", []).append(dict(field=key, value=value))
            schedule = record.get("schedule_ir")
            if isinstance(schedule, dict) and isinstance(schedule.get("text"), str):
                row["schedule_ir"] = dict(sha256=schedule.get("sha256"), text=schedule["text"])
            out.append(row)
    return out


def verify_schedule(text, declared_sha256):
    """The schedule dialect's own parser/verifier over one emitted schedule record."""
    actual = hashlib.sha256(text.encode()).hexdigest()
    if declared_sha256 and declared_sha256 != actual:
        return dict(verifier="schedule_dialect.verify", status="fail", reason="schedule_ir_digest_mismatch")
    try:
        from xdsl.utils.exceptions import ParseError, VerifyException

        from merlin.compiler.schedule.schedule_ir import parse_dense_schedule_plan
    except ImportError:
        return dict(verifier="schedule_dialect.verify", status="unknown", reason="verifier_unavailable")
    try:
        parse_dense_schedule_plan(text).verify()
    except (ParseError, VerifyException) as error:
        return dict(verifier="schedule_dialect.verify", status="fail", sha256=actual, reason=str(error)[:300])
    except (ValueError, TypeError, KeyError) as error:
        return dict(verifier="schedule_dialect.verify", status="unknown", sha256=actual, reason=str(error)[:300])
    return dict(verifier="schedule_dialect.verify", status="pass", sha256=actual)


def window_overlaps(effect_facts):
    """Lexical overlaps of operand-load windows of different classes in one declared flag set.

    Two movement-in classes filling overlapping local rows can clobber a live operand; a load
    feeding a compute or readout window is ordinary dataflow and is not reported."""
    fact = (effect_facts or {}).get("local_address_windows") or {}
    if fact.get("status") not in ("observed", "partial"):
        return dict(status="unknown", reason=fact.get("reason") or "local_address_windows_unknown", overlaps=[])
    windows = [w for w in fact["value"]["windows"] if type(w.get("min")) is int and MOVEMENT_IN[0] in w["roles"]]
    overlaps = []
    for i, left in enumerate(windows):
        for right in windows[i + 1 :]:
            if left["flags"] != right["flags"] or left["instruction"] == right["instruction"]:
                continue
            ends = [w.get("max_end") or w["max"] + 1 for w in (left, right)]
            if left["min"] < ends[1] and right["min"] < ends[0]:
                overlaps.append(
                    dict(
                        flags=left["flags"],
                        first=dict(
                            instruction=left["instruction"],
                            field=left["field"],
                            roles=left["roles"],
                            range=[left["min"], ends[0]],
                        ),
                        second=dict(
                            instruction=right["instruction"],
                            field=right["field"],
                            roles=right["roles"],
                            range=[right["min"], ends[1]],
                        ),
                    )
                )
    unresolved = fact["value"].get("unresolved_sites") or {}
    status = "overlap_observed" if overlaps else ("no_overlap_observed" if not unresolved else "unknown")
    return dict(
        status=status,
        overlaps=overlaps[:16],
        unresolved_address_sites=unresolved,
        scope=fact.get("scope"),
        basis="lexical IR windows; sequential reuse is not excluded",
    )


def _storage_status(record):
    verifiers = record["storage"]["library_verifiers"]
    windows = record["storage"]["local_address_windows"]["status"]
    if any(v["status"] == "fail" for v in verifiers):
        return "verifier_failed"
    if windows == "overlap_observed":
        return "operand_window_overlap_observed"
    if all(v["status"] == "pass" for v in verifiers) and windows == "no_overlap_observed":
        return "no_conflict_observed"
    return "unknown"


def capacities(facts):
    """Local memory capacities as RTL-fact data (name, banks, rows, bytes); empty when absent."""
    if facts is None:
        return []
    document = json.loads(Path(facts).read_bytes())
    spaces = (document.get("facts") or document).get("memory_address_spaces") or []
    return [
        dict(name=s.get("name"), banks=s.get("banks"), depth=s.get("depth"), bytes=s.get("bytes"))
        for s in spaces
        if isinstance(s, dict) and type(s.get("bytes")) is int
    ]


def _element_bytes(dtype):
    if not isinstance(dtype, str):
        return None
    digits = "".join(c for c in dtype if c.isdigit())
    return int(digits) // 8 if digits and int(digits) % 8 == 0 else None


def capacity_conditions(shape, spaces):
    rows = []
    for name, tensor in sorted(shape["tensors"].items()):
        size, width = 1, _element_bytes(tensor.get("dtype"))
        for extent in tensor.get("shape") or ():
            size *= extent if type(extent) is int else 0
        footprint = size * width if width and size else None
        rows.append(
            dict(
                tensor=name,
                role=tensor.get("role"),
                bytes=footprint,
                exceeds=[s["name"] for s in spaces if footprint is not None and footprint > s["bytes"]],
            )
        )
    return dict(
        tensors=rows,
        spaces=spaces,
        interpretation="whole-tensor footprint vs each RTL local memory; placement is not inferred",
    )


# Preserve established callable pickle identities through the explicit public exports.
for _export in (
    mechanisms,
    traffic,
    _ratio,
    _round,
    effects,
    receipts,
    verify_schedule,
    window_overlaps,
    _storage_status,
    capacities,
    _element_bytes,
    capacity_conditions,
):
    _export.__module__ = "merlin_experiments.phase1.feedback.candidate_verification"
