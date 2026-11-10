"""Independent host-state gates for calibrated measurements.

The selected measurement contract supplies every threshold and expected state.
"""

from __future__ import annotations

from typing import Any


def state_ready(probe: dict[str, Any], contract: dict[str, Any]) -> bool:
    """Apply the frozen pre-measurement environmental gate to one external probe."""
    if probe.get("returncode") != 0:
        return False
    state = probe.get("state", {})
    if not isinstance(state, dict) or state.get("online") != contract.get("online"):
        return False
    try:
        governors = state["governors"]
        frequencies = [int(value) for value in state["frequencies_khz"].values()]
        temperatures = [int(value) for value in state["temperatures_millic"].values()]
        load = float(str(state["loadavg"]).split()[0])
        expected_frequency = int(contract["frequency_khz"])
        tolerance = float(contract["frequency_relative_tolerance"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    return (
        isinstance(governors, dict)
        and bool(governors)
        and set(governors.values()) == {contract.get("governor")}
        and len(frequencies) == int(contract.get("frequency_core_count", -1))
        and all(abs(value - expected_frequency) / expected_frequency <= tolerance for value in frequencies)
        and bool(temperatures)
        and max(temperatures) <= int(contract["maximum_temperature_millic"])
        and load <= float(contract["maximum_load_1m"])
    )


def paired_state_ok(before: dict[str, Any], after: dict[str, Any], contract: dict[str, Any]) -> bool:
    if not state_ready(before, contract) or not state_ready(after, contract):
        return False
    left, right = before["state"], after["state"]
    try:
        temperature_delta = max(
            abs(int(left["temperatures_millic"][key]) - int(right["temperatures_millic"][key]))
            for key in left["temperatures_millic"]
            if key in right["temperatures_millic"]
        )
        load_delta = abs(float(str(left["loadavg"]).split()[0]) - float(str(right["loadavg"]).split()[0]))
    except (KeyError, TypeError, ValueError):
        return False
    return (
        left.get("online") == right.get("online")
        and left.get("governors") == right.get("governors")
        and temperature_delta <= int(contract["maximum_pair_temperature_delta_millic"])
        and load_delta <= float(contract["maximum_pair_load_delta"])
    )
