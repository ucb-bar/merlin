"""Separate per-invocation profiling from complete-chain numerical transport.

Gaps include wrapper bookkeeping. Dropped calls make host/device attribution
incomplete even when the complete counter window still reconciles.
"""

from __future__ import annotations

from merlin.runtime.whole_model_readback import parse_group_profile


def separate(text):
    ordinary, profiles, active, current_step = [], {}, None, None
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] in ("CHAIN_PROFILE_BEGIN", "CHAIN_PROFILE_END"):
            if len(fields) != 4 or fields[1] != "v1" or any(not n.isdecimal() for n in fields[2:]):
                raise ValueError("malformed chain group profile boundary")
            key = tuple(map(int, fields[2:]))
            if fields[0] == "CHAIN_PROFILE_BEGIN":
                if active is not None or key in profiles or key != current_step:
                    raise ValueError("duplicate, nested or displaced chain group profile")
                active = (key, [])
            else:
                if active is None or active[0] != key:
                    raise ValueError("unmatched chain group profile boundary")
                profile = parse_group_profile("\n".join(active[1]))
                if profile is None:
                    raise ValueError("empty chain group profile")
                profiles[key] = profile
                active = None
        elif active is not None:
            if fields and fields[0].startswith("CHAIN_"):
                raise ValueError("chain transport inside an unfinished group profile")
            active[1].append(line)
        else:
            ordinary.append(line)
            if fields and fields[0] == "CHAIN_STEP" and len(fields) == 6:
                current_step = tuple(map(int, fields[2:4]))
            elif fields and fields[0].startswith("CHAIN_"):
                current_step = None
    if active is not None:
        raise ValueError("truncated chain group profile")
    return "\n".join(ordinary), profiles


def reconcile(profiles, schedule, cycles, *, enabled):
    expected = {(step["program"], step["step"]) for step in schedule}
    if set(profiles) != (expected if enabled else set()):
        raise ValueError("chain group profiling differs from its selected complete step roster")
    result = []
    for step, counter in zip(schedule, cycles, strict=True):
        key = step["program"], step["step"]
        if key not in profiles:
            continue
        profile = profiles[key]
        group_cycles = sum(row["cycles"] for row in profile["groups"])
        gaps = profile["tail_gap"] + sum(row["gap_before"] for row in profile["groups"])
        if group_cycles + gaps != counter["cycles"]:
            raise ValueError("group and gap cycles do not reconcile with the invocation window")
        complete = profile["dropped"] == 0
        result.append({
            "program": counter["program"], "step": step["step"], **profile,
            "group_cycles": group_cycles,
            "gap_cycles": gaps,
            "host_gap_cycles": gaps if complete else None,
            "complete": complete,
            "scope": "instrumented routed calls and gaps including wrapper bookkeeping; no timing qualification",
        })
    return result
