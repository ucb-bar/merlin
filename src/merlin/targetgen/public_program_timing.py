"""Normal public whole-program selection; data supplies no execution authority.

The original model grader still owns package isolation, source placement, full
numerical checks and required tiers. Protected/validation sources are ineligible
for this pre-freeze selection.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

SCHEMA = "merlin.public_program_execution.v1"


def is_public_program(capsule):
    basis = ((capsule.get("performance") or {}).get("gate") or {}).get("source_execution_basis") or {}
    return (
        capsule.get("kind") == "model"
        and capsule.get("label") in ("public", "dev")
        and basis.get("schema") == "merlin.phase0.public_program_source_basis.v1"
        and basis.get("kind") == "public_program"
        and basis.get("boundary") == "complete_program_call"
    )


def verify_selection(capsule, selection):
    """Reopen the selected ordinary facts; never reconstruct a live owner."""
    if not is_public_program(capsule):
        raise ValueError("complete public timing requires the original independent public source basis")
    if (
        type(selection) is not dict
        or set(selection) != {"schema", "simulator", "rtl_facts", "board_config", "complete_call_timing"}
        or selection["schema"] != SCHEMA
        or (selection["simulator"], selection["complete_call_timing"]) not in (("gsim", True), ("spike", False))
        or type(selection["complete_call_timing"]) is not bool
        or (selection["simulator"] == "spike" and capsule.get("oracle_tier_ceiling") != "L2")
        or type(selection["board_config"]) is not str
        or not selection["board_config"]
    ):
        raise ValueError("public execution requires its closed ordinary engine/facts/timer selection")
    pin = selection["rtl_facts"]
    if type(pin) is not dict or set(pin) != {"path", "sha256"}:
        raise ValueError("public execution requires exact selected ordinary facts bytes")
    path = Path(pin["path"])
    if (
        not path.is_absolute()
        or path.resolve() != path
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > 16 << 20
        or hashlib.sha256(path.read_bytes()).hexdigest() != pin["sha256"]
    ):
        raise ValueError("public execution selected facts are unavailable or changed")
    return {
        "simulator": selection["simulator"],
        "rtl_facts": str(path),
        "board_config": selection["board_config"],
        "complete_call_timing": selection["complete_call_timing"],
        **({"oracle_tier_ceiling": "L2"} if selection["simulator"] == "spike" else {}),
    }


def candidate_execution_requested(result, capsule):
    """Preserve old acceleration demand and admit explicit public host programs."""
    if bool(((capsule or {}).get("semantic") or {}).get("must_accelerate")):
        return True
    selection = result.get("public_program_execution")
    if selection is None:
        return False
    verify_selection(capsule or {}, selection)
    return True


def numerical_screen_requested(result, capsule):
    selection = result.get("public_program_execution")
    if selection is None or selection.get("simulator") != "spike":
        return False
    verify_selection(capsule, selection)
    return True
