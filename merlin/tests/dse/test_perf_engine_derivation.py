"""The engine set an overlap term is defined over, DERIVED from the target's own RTL.

A capability contract declares COMPUTE units, because that is what a compiler routes against. The
engines whose *concurrency* an analytical model prices are a different, larger set — and the target
exercised here declared exactly one, which made the only interesting performance term it has
(``engine_pair`` overlap) report ``uncalibratable`` for the reason "the target declares 1 engine(s)"
while its own elaboration carried three decoupled command controllers.

What is pinned here, in both directions:

* the derivation FIRES on evidence — a module that owns a detected control FSM *and* exposes a
  completion channel is an engine, and the union with the contract counts it once (the positive case,
  because a negative-only suite in this repo has repeatedly passed for the wrong reason);
* it REFUSES on absence — an FSM with no completion port, a completion port with no FSM, an
  unreadable elaboration, an extraction that was never run, and a contract that does not say which
  module realises its unit are each recorded as UNKNOWN *with the reason*, never as "no engine",
  which is the reading that flatters the result.
"""
from __future__ import annotations

import pytest

from merlin.perf.occupancy import derived_engines, engine_set
from merlin.targetgen.rtl.fsm import FsmRegister

# A contract in the real shape: one declared arithmetic unit, exactly the situation that made the
# overlap term unidentifiable. Nothing here is read out of a name.
ONE_UNIT_CONTRACT = {
    "name": "t_probe",
    "compute_units": [{"name": "mesh", "kind": "systolic", "dtypes": ["int8"], "ops": ["matmul"]}],
}
ALIASED_CONTRACT = {
    "name": "t_probe",
    "compute_units": [{"name": "mesh", "kind": "systolic", "dtypes": ["int8"], "ops": ["matmul"],
                       "rtl_module": "Exec"}],
}


def _fsm(*qualified):
    """``("Mod.reg", ...)`` -> the FSM inventory entries an extraction would report."""
    return [FsmRegister(module=m, register=r) for m, _, r in (q.partition(".") for q in qualified)]


def _ports(completing=(), decoupled=(), *, n_modules=100, status="derived", why=""):
    """A :func:`merlin.targetgen.rtl.ports.port_facts` record, in its own shape."""
    if status != "derived":
        return {"status": status, "why": why, "fields": {}}
    return {"status": "derived", "dialect": "fir", "n_modules": n_modules,
            "fields": {"completed": {"modules": sorted(completing),
                                     "decoupled": sorted(decoupled), "leaves": {}}}}


class TestTheDerivationFires:
    """The positive case. A rule that only ever refuses is indistinguishable from a broken one."""

    def test_a_module_that_sequences_and_completes_its_own_work_is_an_engine(self):
        got, basis = derived_engines(
            "t_probe",
            fsm_registers=_fsm("Ld.control_state", "St.control_state", "Exec.control_state"),
            ports=_ports(completing=("Ld", "St", "Exec", "Top"), decoupled=("Ld", "St")))
        assert sorted(got) == ["Exec", "Ld", "St"], "the intersection of both halves is the engine set"
        assert basis["status"] == "derived"
        for name in got:
            assert got[name]["rtl_module"] == name
            assert got[name]["kind"], "a derived engine must carry a kind"

    def test_the_derivation_says_which_evidence_made_each_engine(self):
        got, _ = derived_engines("t_probe", fsm_registers=_fsm("Ld.control_state"),
                                 ports=_ports(completing=("Ld",), decoupled=("Ld",)))
        basis = got["Ld"]["basis"]
        assert "Ld.control_state" in basis, "the register that evidenced it is named"
        assert "handshake" in basis, "whether the completion is tagged is part of the evidence"
        assert "ROLE" in basis, "compute-vs-movement is left UNKNOWN, and says so"

    def test_the_completion_field_is_a_parameter_not_a_law(self):
        # A target whose engines signal completion under another name is served by asking for it.
        ports = {"status": "derived", "dialect": "hw", "n_modules": 9,
                 "fields": {"done": {"modules": ["Ld"], "decoupled": [], "leaves": {}}}}
        got, basis = derived_engines("t_probe", fsm_registers=_fsm("Ld.control_state"),
                                     ports=ports, completion_field="done")
        assert sorted(got) == ["Ld"] and basis["completion_field"] == "done"


class TestItRefusesRatherThanInvents:
    """Each way the evidence can fail, and the reason it leaves behind."""

    def test_an_fsm_with_no_completion_port_is_refused_with_its_reason(self):
        got, basis = derived_engines("t_probe",
                                     fsm_registers=_fsm("Ld.control_state", "Inner.state"),
                                     ports=_ports(completing=("Ld",), decoupled=("Ld",)))
        assert sorted(got) == ["Ld"]
        assert "Inner" in basis["refused"], "a refused candidate is RECORDED, not dropped"
        assert "Inner.state" in basis["refused"]["Inner"]
        assert "UNKNOWN" in basis["refused"]["Inner"], "refused is not the same as shown absent"

    def test_a_completion_port_with_no_fsm_is_not_an_engine(self):
        # A wrapper and a command tracker both complete work they do not sequence.
        got, _ = derived_engines("t_probe", fsm_registers=_fsm("Ld.control_state"),
                                 ports=_ports(completing=("Ld", "Top", "Tracker")))
        assert sorted(got) == ["Ld"]

    def test_an_unreadable_elaboration_is_unknown_not_empty(self):
        got, basis = derived_engines("t_probe", fsm_registers=_fsm("Ld.control_state"),
                                     ports=_ports(status="unavailable", why="not on this host"))
        assert got == {} and basis["status"] != "derived"
        assert "UNKNOWN" in basis["why"] and "not on this host" in basis["why"]

    def test_an_extraction_that_found_nothing_is_unknown_not_a_design_with_no_engines(self):
        # Empty is a statement about the EXTRACTION. A machine nobody analysed must not read as a
        # machine with nothing in it.
        got, basis = derived_engines("t_probe", fsm_registers=[], ports=_ports(completing=("Ld",)))
        assert got == {} and basis["status"] != "derived"
        assert "UNKNOWN" in basis["why"] and "extraction" in basis["why"]

    def test_no_target_named_is_unknown(self):
        got, basis = derived_engines(None)
        assert got == {} and basis["status"] != "derived" and "UNKNOWN" in basis["why"]

    def test_a_derivation_that_clears_nothing_says_so_rather_than_confirming_the_declaration(self):
        # Both extractions RAN and no module clears the bar. That is a third state, and the record
        # must not let it read as "the contract's single engine is the whole machine".
        got, basis = derived_engines("t_probe", fsm_registers=_fsm("VectorFSM.state"),
                                     ports=_ports(completing=("Top",)))
        assert got == {} and basis["status"] == "derived"
        assert "UNKNOWN" in basis["why"]
        assert "VectorFSM" in basis["refused"]


class TestTheUnionCountsAnEngineOnce:
    """A declared unit and the controller that sequences it are one engine, asked rather than guessed."""

    def test_a_cross_checked_alias_folds_the_two_into_one_engine(self):
        engines, basis = engine_set(
            ALIASED_CONTRACT,
            fsm_registers=_fsm("Ld.control_state", "St.control_state", "Exec.control_state"),
            ports=_ports(completing=("Ld", "St", "Exec"), decoupled=("Ld", "St")))
        assert sorted(engines) == ["Ld", "St", "mesh"], "Exec IS the mesh; it is not a fourth engine"
        assert basis["declared_aliases"] == {"Exec": "mesh"}
        assert basis["unresolved_aliases"] == []
        assert "CROSS-CHECKED" in engines["mesh"]["basis"]
        assert engines["mesh"]["kind"] == "systolic", "the declaration's archetype survives the union"

    def test_an_unresolved_alias_is_reported_not_silently_merged_or_doubled(self):
        engines, basis = engine_set(
            ONE_UNIT_CONTRACT, fsm_registers=_fsm("Ld.control_state", "Exec.control_state"),
            ports=_ports(completing=("Ld", "Exec"), decoupled=("Ld",)))
        assert sorted(engines) == ["Exec", "Ld", "mesh"]
        assert basis["unresolved_aliases"] == ["mesh"]
        assert "UNKNOWN" in engines["mesh"]["basis"], "whether one of them IS the mesh is unknown"

    def test_an_rtl_module_the_elaboration_does_not_contain_fails_the_cross_check(self):
        # The contract is not trusted about the RTL: a module the elaboration was READ and found not
        # to contain is recorded as uncorroborated rather than accepted.
        contract = {"name": "t_probe",
                    "compute_units": [{"name": "mesh", "kind": "systolic", "dtypes": ["int8"],
                                       "rtl_module": "NotThere"}]}
        engines, _ = engine_set(contract, fsm_registers=_fsm("Ld.control_state"),
                                ports=_ports(completing=("Ld",), decoupled=("Ld",)))
        assert sorted(engines) == ["Ld", "mesh"], "the declared unit survives; it is just not confirmed"
        assert "FAILS the cross-check" in engines["mesh"]["basis"]

    def test_with_no_rtl_evidence_the_union_is_exactly_the_declaration(self):
        engines, basis = engine_set(ONE_UNIT_CONTRACT, fsm_registers=[],
                                    ports=_ports(status="unavailable", why="absent"))
        assert sorted(engines) == ["mesh"] and basis["status"] != "derived"
        assert "UNKNOWN" not in engines["mesh"].get("rtl_module", "")


class TestOnTheShippedTargets:
    """The end-to-end proof, on whatever target's RTL is actually readable on this host.

    Written over the registry rather than against one target: if adding a second target needed a
    branch here, the derivation would be overfit to the one it was written against.
    """

    def _derivable(self):
        from merlin.targetgen import target_registry as TR
        out = []
        for name in TR.all_targets():
            try:
                got, basis = derived_engines(name)
            except Exception:                                   # noqa: BLE001 -- unparseable target
                continue
            out.append((name, got, basis))
        return out

    def test_some_shipped_target_actually_derives_an_engine(self):
        fired = [(n, g) for n, g, b in self._derivable() if g]
        if not fired:
            pytest.skip("no shipped target's elaboration and FSM extraction are both on this host")
        for name, got in fired:
            for engine, spec in got.items():
                assert spec["basis"].startswith("DERIVED"), f"{name}.{engine} claims no evidence"

    def test_a_target_whose_rtl_evidences_engines_gets_a_pair_to_overlap(self):
        from merlin.perf import calibration as CAL
        from merlin.targetgen import target_registry as TR
        from merlin.targetgen.rtl.fsm import fsm_inventory

        fired = [n for n, g, _ in self._derivable() if g]
        if not fired:
            pytest.skip("no shipped target's elaboration and FSM extraction are both on this host")
        for name in fired:
            inv = CAL.engine_inventory(TR.load_contract(name), [], CAL.calibrate_idle([]),
                                       fsm_registers=list(fsm_inventory(name)))
            assert len(inv.declared) >= 2, (
                f"{name}: its RTL evidences {fired} engines, so the inventory is not one unit")
            cells = CAL.required_cells(inv, {})
            pairless = [c for c in cells if c.axis == CAL.ENGINE_PAIR_AXIS and "engine(s)" in c.why]
            assert not pairless, (
                f"{name}: still reports no pair to overlap: {[c.why for c in pairless]}")
            assert inv.derivation.get("status") == "derived"
            assert inv.derivation.get("rule"), "the record must carry the rule it applied"
