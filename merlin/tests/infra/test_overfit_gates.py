"""The gates that keep target coupling visible: the coupling scan and the lift detector.

These exist because the repo's cardinal rule ("derive, never hardcode") was being enforced by a check
with two blind spots that happened to cancel out. The whole-identifier name check matches ``gemmini``
but not ``gemmini_kernel`` or ``cycle_window_gemmini_region``, so vendor SYMBOL coupling was invisible;
and the coupling scan skips any file whose own path names a target, since self-reference is legitimate.
A module called ``<target>_<thing>.py`` therefore fell through both — which is how two fully general
modules (a fixed-format linker and boot builder, ``target`` a parameter throughout, derived ISA facts
end to end) sat in a vendor-named home for months with nothing asking whether they belonged there.

Tested against SYNTHETIC files rather than the live tree: a test that asserts today's counts turns into
a chore that gets bumped, and it would pass just as well if the scan stopped working.
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

from merlin.common.paths import repo_root

GATE = repo_root() / "build_tools" / "scripts" / "check_no_target_name.py"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("_check_no_target_name", GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ what "owned by a target" means
def test_a_path_naming_a_target_is_treated_as_that_targets_own_module(gate):
    assert gate._is_target_owned("merlin/python/merlin/runtime/backends/muon_link.py")
    assert not gate._is_target_owned("merlin/python/merlin/targetgen/fixed_format/link.py")


# ------------------------------------------------------------------ the coupling scan
def test_a_vendor_symbol_is_caught_even_though_it_is_not_a_bare_identifier(gate, tmp_path):
    """The leak the whole-identifier check cannot see, and the reason the second scan exists."""
    src = tmp_path / "generic.py"
    src.write_text('def emit():\n    return "METRIC cycle_window_gemmini_region 1"\n', encoding="utf-8")
    hits = gate._scan_coupling(src)
    assert [(k, t) for _, t, k, _ in hits] == [("symbol", "gemmini")]


def test_importing_a_target_module_from_a_generic_one_is_caught(gate, tmp_path):
    src = tmp_path / "generic.py"
    src.write_text("from merlin.runtime.backends import muon_capsule_runner\n", encoding="utf-8")
    assert [t for _, t, k, _ in gate._scan_coupling(src) if k == "import"] == ["muon"]


def test_a_docstring_mention_is_not_coupling(gate, tmp_path):
    """Prose describing a target is documentation. Counting it would flood the register with noise
    and train people to ignore it, which costs more than the few real hits it might also surface."""
    src = tmp_path / "generic.py"
    src.write_text('"""Originally written for gemmini; now generic."""\nX = 1\n', encoding="utf-8")
    assert gate._scan_coupling(src) == []


def test_an_inline_marker_suppresses_a_deliberate_mention(gate, tmp_path):
    src = tmp_path / "generic.py"
    src.write_text(f'NAMES = ("gemmini",)  # {gate.INLINE_MARKER} the set this gate hunts\n',
                   encoding="utf-8")
    assert gate._scan_coupling(src) == []


# ------------------------------------------------------------------ where the marker may sit
def test_the_marker_is_honoured_beside_the_mention_not_only_at_the_constants_start(gate, tmp_path):
    """Python reports an implicitly concatenated string at the line its FIRST fragment starts on.

    A long ``description=(...)`` can carry the target name a dozen lines below that, so anchoring the
    marker to the reported line put the annotation nowhere near what it explains — and someone who
    placed it correctly, beside the mention, watched the gate keep failing with no hint why.
    """
    src = tmp_path / "generic.py"
    src.write_text(
        "DESCRIPTION = (\n"
        '    "first fragment, no target here "\n'
        "    # target-ok: cites a pin, not a routing fact\n"
        '    "second fragment mentioning saturn "\n'
        ")\n", encoding="utf-8")
    assert gate._scan_file(src) == []


def test_an_unmarked_multi_line_constant_is_still_caught(gate, tmp_path):
    """The span rule must widen where a marker is ACCEPTED, never what gets scanned."""
    src = tmp_path / "generic.py"
    src.write_text(
        "DESCRIPTION = (\n"
        '    "first fragment, no target here "\n'
        '    "second fragment mentioning saturn "\n'
        ")\n", encoding="utf-8")
    assert [name for _, name, _ in gate._scan_file(src)] == ["saturn"]


# ------------------------------------------------------------------ the lift detector
def test_the_lift_detector_reports_the_live_tree_without_enforcing(gate):
    """A hit is a question ("is this general?"), not a violation — some answers are legitimately no.

    Asserting the CONTENTS here would be asserting today's debt; what must hold is that the detector
    runs over the real tree, returns reportable strings, and never fails the build.
    """
    assert gate.main(["--coupling"]) == 0
    for line in gate.lift_candidates():
        assert line.endswith("audit for a lift")
