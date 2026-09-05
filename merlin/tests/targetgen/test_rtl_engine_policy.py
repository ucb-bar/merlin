"""The cert tier is a fidelity; which simulator answers is a cost decision that must be RECORDED.

Binding a tier index to one binary (`L3 = verilator`) hid that decision and put two different fidelities
on the same rung across targets. These pin the policy: equal-fidelity engines are ordered by cost,
Verilator is never chosen while GSIM can run (~23x slower at corpus scale — 45 min vs 115 s per capsule),
and a tier that cannot run fails closed instead of quietly becoming a model tier.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import rtl_engine_policy as P

_UP = lambda why="ok": (lambda: (True, why))          # noqa: E731 - table-style probes read better inline
_DOWN = lambda why: (lambda: (False, why))            # noqa: E731


def test_gsim_is_preferred_over_verilator():
    """The whole point: Verilator must not be picked when GSIM can run."""
    sel = P.select("t", {"verilator": _UP("built"), "gsim": _UP("emu ready")})
    assert sel["engine"] == "gsim"


def test_vcs_wins_when_it_can_actually_run():
    sel = P.select("t", {"verilator": _UP(), "gsim": _UP(), "vcs": _UP("license free")})
    assert sel["engine"] == "vcs" and sel["passed_over"] == []


def test_an_unavailable_higher_priority_engine_is_skipped_with_its_reason():
    sel = P.select("t", {"vcs": _DOWN("no license"), "gsim": _UP("emu ready"), "verilator": _UP()})
    assert sel["engine"] == "gsim"
    assert sel["passed_over"] == ["vcs"]
    assert [c["reason"] for c in sel["considered"] if c["engine"] == "vcs"] == ["no license"]


def test_verilator_is_still_used_when_it_is_the_only_engine():
    """A target with no GSIM adapter yet must keep its cert tier, not lose it to a preference."""
    sel = P.select("atlas", {"verilator": _UP("vsim registered")})
    assert sel["engine"] == "verilator" and sel["fidelity"] == P.ELABORATED_RTL


def test_no_engine_fails_closed_rather_than_downgrading():
    with pytest.raises(P.NoEngineAvailable) as e:
        P.select("t", {"gsim": _DOWN("no adapter"), "verilator": _DOWN("no vsim")})
    assert "no adapter" in str(e.value) and "no vsim" in str(e.value)


def test_every_engine_reports_the_same_fidelity():
    """They all run the elaborated design; the tier must not grade one as weaker than another."""
    for name in P.ENGINE_PRIORITY:
        assert P.select("t", {name: _UP()})["fidelity"] == P.ELABORATED_RTL


def test_a_broken_probe_is_unavailable_not_a_crash():
    def boom():
        raise OSError("toolchain missing")
    sel = P.select("t", {"vcs": boom, "verilator": _UP()})
    assert sel["engine"] == "verilator"
    assert "OSError" in [c["reason"] for c in sel["considered"] if c["engine"] == "vcs"][0]


def test_lower_priority_probes_are_not_paid_once_one_is_available():
    """Probing an absent VCS license or building a Verilator model is not free."""
    called = []
    P.select("t", {"vcs": _UP("license"), "gsim": lambda: called.append("gsim") or (True, "x"),
                   "verilator": lambda: called.append("vl") or (True, "x")})
    assert called == []


def test_an_engine_the_policy_has_not_ranked_is_used_not_dropped():
    """A newly registered engine must still be selectable before anyone declares its priority."""
    sel = P.select("t", {"newsim": _UP("registered")})
    assert sel["engine"] == "newsim"


def test_an_engine_that_reports_available_with_no_reason_is_refused():
    """Peer review point, and the shape of several defects hit the same day: a tier that resolved to a
    different engine than the capsule asked for, with the reason living only in a log line, produces
    correct-looking numbers that cannot be audited. Absent reason must be a hard failure, not a default."""
    with pytest.raises(P.UnrecordedSelection):
        P.select("t", {"gsim": lambda: (True, "   ")})


def test_the_reason_survives_on_the_result_not_just_in_a_log():
    sel = P.select("t", {"vcs": _DOWN("no license"), "gsim": _UP("emu built at <path>")})
    assert sel["reason"] == "emu built at <path>"
    assert {c["engine"]: c["reason"] for c in sel["considered"]}["vcs"] == "no license"


# ---------------------------------------------------------------------------------------------
# The lineage gate must reach the SELECTION path, not only the module that owns the home layout.
# ---------------------------------------------------------------------------------------------

def test_a_refused_lineage_loses_the_selection_not_just_the_probe(tmp_path, monkeypatch):
    """Measured 2026-09-04: with MERLIN_GSIM_REQUIRE_RECEIPT=1, a target whose engine carried only an
    adoption record had gsim_emulator.probe() answer False and STILL certified on it, because the
    selection path asked only whether the wrapper file existed. A provenance gate the selection routes
    around is not a gate."""
    from merlin.targetgen import program_oracle as PO
    from merlin.targetgen import gsim_emulator as GE

    home = tmp_path / "gsim"
    home.mkdir()
    (home / "gsim_run.py").write_text("def run_program(*a, **k): ...", encoding="utf-8")
    monkeypatch.setattr(PO, "_rtl_engine_dir", lambda target, engine: home)
    monkeypatch.setattr(GE, "resolve", lambda target, **k: GE.Resolution(
        target=target, path=home / "gsim_run.py", source="derived", ok=False, refused=True,
        reason="lineage ADOPTED, not built-and-bound", flavour="wrapper", digest="d",
        receipt_status="adopted", receipt=None))
    available, reason = PO._rtl_engine_probe("t", "gsim")()
    assert available is False and "ADOPTED" in reason


def test_an_engine_this_module_does_not_lay_out_is_not_judged_by_it(tmp_path, monkeypatch):
    """A refusal authored by the wrong module would be worse than none: verilator's home is laid out
    and receipted by whoever built it, so the gsim lineage record must not be consulted for it."""
    from merlin.targetgen import program_oracle as PO
    from merlin.targetgen import gsim_emulator as GE

    home = tmp_path / "vsim"
    home.mkdir()
    (home / "verilator_run.py").write_text("def run_program(*a, **k): ...", encoding="utf-8")
    monkeypatch.setattr(PO, "_rtl_engine_dir", lambda target, engine: home)

    def _boom(*a, **k):
        raise AssertionError("the gsim lineage record was consulted for another engine")
    monkeypatch.setattr(GE, "resolve", _boom)
    assert PO._rtl_engine_probe("t", "verilator")()[0] is True
