"""The FireSim leg driver, now in the tree instead of in /scratch tmp.

Twice the script behind a cited number has been a session scratchpad that was later purged:
``fs_legs.py`` is named in the provenance ``sources`` of the FireSim whole-model matrix-unit result and
no copy survives, and its successor lived as a loose ``zeph_leg.py`` under ``/scratch/agustin/tmp`` as
the only driver for building these legs. These tests pin the two properties that make it safe to keep:
it names no target, and its audit cannot pass an image that measures the wrong thing.

Build-free by construction — every assertion is on the CLI contract and the source, so this runs
without FireSim, a bitstream, or a bundle.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from merlin.common.paths import repo_root

SCRIPT = repo_root() / "build_tools/scripts/zephyr_firesim_leg.py"


def _src() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_the_driver_is_tracked_not_a_temp_file():
    assert SCRIPT.is_file()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(SCRIPT)],
                             cwd=repo_root(), capture_output=True, text=True)
    assert tracked.returncode == 0, "the driver must be TRACKED — that is the whole point"


def test_the_unit_and_its_config_are_arguments_not_literals():
    """The original hardcoded one unit name and one generator config. `build_tools/scripts/` is a
    check_no_target_name scan root, and more importantly a second matrix unit would have needed this
    file edited rather than a flag passed."""
    src = _src()
    assert '"--unit", required=True' in src
    assert '"--config", required=True' in src
    assert '"--board", required=True' in src
    assert '"--package", required=True' in src
    # and the encodings come from the unit's own contract, never a baked opcode
    assert "opu_shim.derive_encodings(opu_shim.load_contract(a.unit))" in src


def test_required_arguments_are_actually_enforced():
    r = subprocess.run([sys.executable, str(SCRIPT), "some_bundle", "device", "/tmp/x"],
                       capture_output=True, text=True, cwd=repo_root())
    assert r.returncode != 0
    assert "--unit" in (r.stderr + r.stdout)


def test_the_leg_choice_is_closed():
    r = subprocess.run([sys.executable, str(SCRIPT), "b", "sideways", "/tmp/x",
                        "--unit", "u", "--config", "c", "--board", "d", "--package", "p"],
                       capture_output=True, text=True, cwd=repo_root())
    assert r.returncode != 0 and "sideways" in (r.stderr + r.stdout)


def test_the_audit_fails_closed_in_BOTH_directions():
    """A device leg that routed nothing and a control leg that accidentally routed something both
    produce plausible numbers and a successful build. Neither may pass."""
    src = _src()
    assert "if routed and not counts:" in src
    assert "if not routed and counts:" in src
    assert src.count("raise SystemExit") >= 3
    # the failure text must say what went WRONG, not just that something did
    assert "carries NONE of the unit's instructions" in src
    assert "not a control" in src


def test_the_arena_is_sized_for_a_no_op_free_not_for_liveness_peak():
    """`free()` is a no-op in this runtime, so the arena must cover the SUM of allocations. Sizing it
    from the liveness peak is how a whole-model run dies at op 11,494 of 11,526."""
    src = _src()
    assert "activation_peak_bytes" in src
    assert "SUM of allocations, not the" in src
    assert "a.arena_mb or max(" in src          # explicit override still wins


def test_the_help_text_explains_the_board_argument_is_a_correctness_knob():
    r = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True,
                       cwd=repo_root())
    assert r.returncode == 0
    assert "DRAM" in r.stdout
