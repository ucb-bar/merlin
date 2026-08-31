"""End-to-end proof of the mechanism-calibration driver, on synthetic traces.

Synthetic per-cycle traces, deliberately. The real cycle-accurate tier is not available in this
environment (no built co-simulation model), and the point of these tests is not to re-measure hardware
but to prove that each way the calibration could FABRICATE a number is closed:

* an engine no instrument read must come back UNKNOWN, never idle -- the substitution that moved a
  corpus idle figure from 76.7% to 46.2% and one kernel's from 89.9% to 39.2%;
* a vector with fewer than two live engines must refuse eta, not report 0.0;
* a trace binding a column to an undeclared engine must void the reading, not reassign the column;
* a signal counted beside its own sub-signals must not manufacture overlap (204 fabricated cycles on
  one measured design);
* eta above 1 from three engines overlapping in disjoint pairs must be reported, not clipped;
* a regime with one capsule must be flagged an extrapolation, since one point cannot separate a rate
  from a fixed intercept.

Every trace here is written out cycle by cycle so the expected counts are checkable by hand.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from merlin.common.paths import merlin_dir, repo_root
from merlin.perf import calibration as CAL

# Two engines of DIFFERENT declared kinds, so the engine axis has a pair. The kinds come from the
# contract exactly as a real target's would; nothing here is read out of a name.
TWO_ENGINE_CONTRACT = {
    "compute_units": [
        {"name": "eng_a", "kind": "systolic", "dtypes": ["int8"], "ops": ["matmul"]},
        {"name": "eng_b", "kind": "vector", "dtypes": ["int8"], "ops": ["add"]},
    ]
}
ONE_ENGINE_CONTRACT = {
    "compute_units": [{"name": "eng_a", "kind": "systolic", "dtypes": ["int8"], "ops": ["matmul"]}]
}
THREE_ENGINE_CONTRACT = {
    "compute_units": [
        {"name": "eng_a", "kind": "systolic", "dtypes": ["int8"], "ops": ["matmul"]},
        {"name": "eng_b", "kind": "vector", "dtypes": ["int8"], "ops": ["add"]},
        {"name": "eng_c", "kind": "scalar", "dtypes": ["int8"], "ops": ["add"]},
    ]
}


def _bits(pattern: str) -> list[str]:
    """``'0011'`` -> the per-cycle port values. Ports are one bit; the trace carries their spelling."""
    return list(pattern)


def _trace(name, columns, binding, *, ports=(), states=(), **kw) -> CAL.MechanismTrace:
    return CAL.MechanismTrace(capsule=name, columns=columns, binding=binding,
                              port_columns=tuple(ports), state_columns=tuple(states),
                              completion_observable=kw.pop("completion_observable", True), **kw)


# --------------------------------------------------------------------------- idle calibration


def test_idle_encoding_is_derived_from_a_paired_port_not_assumed():
    """A state register paired cycle-exactly with a busy port pins the encoding. No constant assumed."""
    # port high on cycles 1,2; the state register holds '7' on exactly the cycles the port is low.
    tr = _trace("c", {"p": _bits("0110"), "s": ["7", "3", "3", "7"]},
                {"p": "eng_a", "s": "eng_b"}, ports=("p",), states=("s",))
    got = CAL.calibrate_idle([tr])
    assert got.idle_value == "7"
    assert got.basis == CAL.IDLE_DERIVED
    assert got.paired_with == ("p",)


def test_unpaired_state_column_is_unreadable_not_idle():
    """The 76.7%->46.2% bug, closed. No port varies, so nothing pins the encoding and the column is
    reported unreadable rather than contributing zero busy cycles."""
    tr = _trace("c", {"p": _bits("0000"), "s": ["0", "1", "1", "0"]},
                {"p": "eng_a", "s": "eng_b"}, ports=("p",), states=("s",))
    idle = CAL.calibrate_idle([tr])
    assert idle.idle_value is None and idle.basis == CAL.IDLE_UNESTABLISHED
    hot, unreadable = CAL.busy_vectors(tr, idle)
    assert "s" not in hot, "an uncalibrated state register must not enter the occupancy vector"
    assert "s" in unreadable and unreadable["s"]
    # ...and the engine it belongs to is UNOBSERVABLE with the reason, not an engine with 0 busy.
    inv = CAL.engine_inventory(TWO_ENGINE_CONTRACT, [tr], idle)
    assert inv.declared["eng_b"].observable is False
    assert "idle" in inv.declared["eng_b"].why


def test_producer_declared_idle_value_is_stamped_not_promoted():
    """A producer's declaration is an acceptable INPUT and is never quoted as a measurement."""
    tr = _trace("c", {"p": _bits("0000"), "s": ["0", "1", "1", "0"]},
                {"p": "eng_a", "s": "eng_b"}, ports=("p",), states=("s",))
    got = CAL.calibrate_idle([tr], declared_idle_value="0")
    assert got.idle_value == "0" and got.basis == CAL.IDLE_DECLARED
    assert "DECLARED, not derived" in got.detail


def test_idle_calibration_is_corpus_wide_not_per_trace():
    """A program that leaves the paired unit constant must not withdraw a calibration the rest of the
    corpus established -- per-trace calibration dropped the busiest unit exactly where it mattered."""
    pinning = _trace("pins", {"p": _bits("0110"), "s": ["7", "3", "3", "7"]},
                     {"p": "eng_a", "s": "eng_b"}, ports=("p",), states=("s",))
    silent = _trace("silent", {"p": _bits("0000"), "s": ["7", "7", "7", "7"]},
                    {"p": "eng_a", "s": "eng_b"}, ports=("p",), states=("s",))
    assert CAL.calibrate_idle([pinning, silent]).idle_value == "7"
    assert CAL.calibrate_idle([silent]).idle_value is None


# --------------------------------------------------------------------------- eta and the split


def _overlapping_trace(name="cap_overlap"):
    """8 cycles. eng_a busy 0-3, eng_b busy 2-5 -> 2 cycles of genuine overlap.

    busy_a=4, busy_b=4 -> available = second-largest = 4; realised = 2; eta = 0.5.
    """
    return _trace(name, {"a": _bits("11110000"), "b": _bits("00111100")},
                  {"a": "eng_a", "b": "eng_b"}, ports=("a", "b"), work="w1")


def test_eta_overlap_split_and_busy_are_measured_on_a_real_pair():
    rec = CAL.calibrate(target="t", contract=TWO_ENGINE_CONTRACT, traces=[_overlapping_trace()])
    cap = rec["capsules"][0]
    assert cap["eta"]["state"] == CAL.MEASURED
    assert cap["eta"]["value"] == pytest.approx(0.5)
    assert cap["overlap_observable"] is True
    assert cap["busy_cycles"]["value"] == {"eng_a": 4, "eng_b": 4}
    assert cap["overlap"]["realised_cycles"]["value"] == 2
    assert cap["overlap"]["available_cycles"]["value"] == 4
    assert cap["overlap"]["unrealised_cycles"]["value"] == 2
    assert sorted(cap["live_engines"]) == ["eng_a", "eng_b"]
    assert rec["ran_against_traces"] is True
    assert rec["audit"]["ok"], rec["audit"]["violations"]


def test_pair_cell_is_calibrated_and_names_the_run_it_would_spend_the_tier_on():
    cheap = _overlapping_trace("cheap")
    dear = _trace("dear", {"a": _bits("1" * 40 + "0" * 40), "b": _bits("0" * 20 + "1" * 40 + "0" * 20)},
                  {"a": "eng_a", "b": "eng_b"}, ports=("a", "b"), work="w1")
    rec = CAL.calibrate(target="t", contract=TWO_ENGINE_CONTRACT, traces=[dear, cheap],
                        points_per_cell=1)
    pair = [c for c in rec["calibration_set"]["cells"] if c["axis"] == CAL.ENGINE_PAIR_AXIS]
    assert len(pair) == 1 and pair[0]["key"] == "eng_a|eng_b"
    assert pair[0]["state"] == CAL.CALIBRATED
    assert pair[0]["capsules"] == ["cheap"], "the cheapest run that shows the mechanism wins"


def test_one_live_engine_refuses_eta_and_does_not_report_zero():
    """The zero this whole module exists to prevent: arithmetic from a vector that could not have
    shown overlap, indistinguishable from a machine that genuinely serialises."""
    tr = _trace("solo", {"a": _bits("11110000"), "b": _bits("00000000")},
                {"a": "eng_a", "b": "eng_b"}, ports=("a", "b"))
    rec = CAL.calibrate(target="t", contract=TWO_ENGINE_CONTRACT, traces=[tr])
    cap = rec["capsules"][0]
    assert cap["eta"]["state"] == CAL.UNKNOWN
    assert cap["eta"]["value"] is None and cap["eta"]["why"]
    assert cap["overlap"]["realised_cycles"]["state"] == CAL.UNKNOWN
    # busy IS measured -- eng_b was read and was zero, which is a different fact from unread.
    assert cap["busy_cycles"]["value"] == {"eng_a": 4, "eng_b": 0}


def test_engine_the_instrument_did_not_read_refuses_the_reading():
    tr = _trace("partial", {"a": _bits("11110000"), "b": _bits("00111100")},
                {"a": "eng_a", "b": "eng_b"}, ports=("a", "b"),
                unmeasured_units=("eng_c",))
    rec = CAL.calibrate(target="t", contract=TWO_ENGINE_CONTRACT, traces=[tr])
    cap = rec["capsules"][0]
    assert cap["eta"]["state"] == CAL.UNKNOWN
    assert "did not read" in cap["eta"]["why"]


def test_column_bound_to_an_undeclared_engine_voids_the_reading():
    """The trace and the contract then disagree about what the device IS; the column is not reassigned."""
    tr = _trace("bad", {"a": _bits("1100"), "z": _bits("0011")},
                {"a": "eng_a", "z": "not_declared"}, ports=("a", "z"))
    rec = CAL.calibrate(target="t", contract=TWO_ENGINE_CONTRACT, traces=[tr])
    cap = rec["capsules"][0]
    assert cap["eta"]["state"] == CAL.UNKNOWN
    assert "does not declare" in cap["eta"]["why"]
    assert rec["engine_inventory"]["binding_error"]
    assert rec["audit"]["ok"], rec["audit"]["violations"]


def test_a_signal_counted_beside_its_sub_signals_does_not_manufacture_overlap():
    """204 fabricated overlap cycles on one measured design. Subsumption is inside the same declared
    engine, so it folds; nesting across declared engines is structure and must not."""
    tr = _trace("nested",
                {"unit": _bits("11110000"), "half": _bits("11000000"), "b": _bits("00000110")},
                {"unit": "eng_a", "half": "eng_a", "b": "eng_b"}, ports=("unit", "half", "b"))
    rec = CAL.calibrate(target="t", contract=TWO_ENGINE_CONTRACT, traces=[tr])
    cap = rec["capsules"][0]
    assert "half" in cap["joint"]["subsumed_columns"], "a sub-signal of the same engine must fold"
    assert cap["overlap"]["realised_cycles"]["value"] == 0
    assert cap["overlap_observable"] is True, "two live engines: the zero here IS evidence"


def test_eta_above_one_is_reported_not_clipped():
    """Three engines overlapping in disjoint pairs: the numerator counts all pairs, the denominator is
    the top pair's ceiling."""
    # a busy 0-3, b busy 0-1, c busy 2-3 -> every cycle 0..3 has two engines busy.
    # busy = a:4 b:2 c:2 -> available = 2 (second largest); realised = 4 -> eta = 2.0
    tr = _trace("disjoint",
                {"a": _bits("111100"), "b": _bits("110000"), "c": _bits("001100")},
                {"a": "eng_a", "b": "eng_b", "c": "eng_c"}, ports=("a", "b", "c"))
    rec = CAL.calibrate(target="t", contract=THREE_ENGINE_CONTRACT, traces=[tr])
    cap = rec["capsules"][0]
    assert cap["eta"]["value"] == pytest.approx(2.0)
    assert cap["overlap"]["unrealised_cycles"]["value"] == -2
    eng = rec["composition"]["engine_axis"]
    assert eng["operator"]["value"] == "max"
    assert eng["eta"]["value"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- the composition operator


def test_engine_axis_operator_is_measured_and_the_kind_axis_refusal_is_recorded():
    """Both axes reported, and the kind axis is EXPECTED to refuse here: every declared compute unit
    is an arithmetic engine, so a by-kind grouping collapses the pair into one group."""
    rec = CAL.calibrate(target="t", contract=TWO_ENGINE_CONTRACT, traces=[_overlapping_trace()])
    comp = rec["composition"]
    assert comp["engine_axis"]["operator"]["value"] == "partial"
    assert comp["engine_axis"]["eta"]["value"] == pytest.approx(0.5)
    assert comp["kind_axis"]["operator"]["state"] == CAL.UNKNOWN
    assert comp["kind_axis"]["operator"]["why"]
    assert "two instruments" in comp["kind_axis_note"]


def test_one_unreadable_run_leaves_the_corpus_operator_unestablished():
    """UNKNOWN propagates, the way composition_operator propagates it: dropping the unmeasurable run
    would reweight the corpus towards whatever happened to be measurable."""
    solo = _trace("solo", {"a": _bits("11110000"), "b": _bits("00000000")},
                  {"a": "eng_a", "b": "eng_b"}, ports=("a", "b"))
    rec = CAL.calibrate(target="t", contract=TWO_ENGINE_CONTRACT,
                        traces=[_overlapping_trace(), solo])
    comp = rec["composition"]
    assert comp["engine_axis"]["operator"]["state"] == CAL.UNKNOWN
    assert comp["runs_without_a_reading"] == ["solo"]


# --------------------------------------------------------------------------- the cover


def test_single_engine_target_reports_the_pair_axis_uncalibratable():
    rec = CAL.calibrate(target="t", contract=ONE_ENGINE_CONTRACT, traces=[])
    pair = [c for c in rec["calibration_set"]["cells"] if c["axis"] == CAL.ENGINE_PAIR_AXIS]
    assert len(pair) == 1 and pair[0]["state"] == CAL.UNCALIBRATABLE
    assert "no pair to overlap" in pair[0]["why"]


def test_unobservable_half_makes_the_pair_uncalibratable_and_names_the_half():
    tr = _trace("c", {"a": _bits("1100"), "s": ["0", "1", "1", "0"]},
                {"a": "eng_a", "s": "eng_b"}, ports=("a",), states=("s",))
    rec = CAL.calibrate(target="t", contract=TWO_ENGINE_CONTRACT, traces=[tr])
    pair = [c for c in rec["calibration_set"]["cells"] if c["axis"] == CAL.ENGINE_PAIR_AXIS][0]
    assert pair["state"] == CAL.UNCALIBRATABLE
    assert "eng_b is not observable" in pair["why"]


def test_regime_cells_take_the_extremes_and_flag_a_single_point_cell():
    regimes = {"capacity_rows": 1000,
               "by_regime": {"fits_double": ["small", "mid", "big"], "spills": ["huge"]}}
    by_capsule = {"small": {"regime": "fits_double", "rows": 4},
                  "mid": {"regime": "fits_double", "rows": 200},
                  "big": {"regime": "fits_double", "rows": 480},
                  "huge": {"regime": "spills", "rows": 4000}}
    rec = CAL.calibrate(target="t", contract=TWO_ENGINE_CONTRACT, traces=[],
                        corpus_regimes=regimes, regime_by_capsule=by_capsule)
    cells = {c["key"]: c for c in rec["calibration_set"]["cells"]
             if c["axis"] == CAL.MEMORY_REGIME_AXIS}
    assert cells["fits_double"]["capsules"] == ["small", "big"], "the ENDS of the regime, not the mid"
    assert cells["spills"]["state"] == CAL.CALIBRATED
    assert "EXTRAPOLATION" in cells["spills"]["why"]
    # a regime no capsule occupies is UNCALIBRATABLE, not a silently passing cell
    assert cells["fits_single"]["state"] == CAL.UNCALIBRATABLE
    assert cells["fits_on_reuse"]["state"] == CAL.UNCALIBRATABLE
    assert rec["coefficient_domain"]["regimes_with_points"] == ["fits_double", "spills"]
    assert rec["coefficient_domain"]["regimes_without_points"] == ["fits_on_reuse", "fits_single"]


def test_undrivable_capacity_yields_one_uncalibratable_regime_cell():
    rec = CAL.calibrate(target="t", contract=TWO_ENGINE_CONTRACT, traces=[],
                        corpus_regimes={"capacity_rows": None, "by_regime": {}})
    cells = [c for c in rec["calibration_set"]["cells"] if c["axis"] == CAL.MEMORY_REGIME_AXIS]
    assert len(cells) == 1 and cells[0]["state"] == CAL.UNCALIBRATABLE
    assert "no operand-store capacity" in cells[0]["why"]


def test_plan_mode_reports_no_calibration_happened():
    rec = CAL.calibrate(target="t", contract=TWO_ENGINE_CONTRACT, traces=[])
    assert rec["ran_against_traces"] is False and rec["n_traces"] == 0
    assert "NO per-cycle trace" in rec["measurement_basis"]
    assert rec["capsules"] == []
    assert rec["composition"]["engine_axis"]["operator"]["state"] == CAL.UNKNOWN


# --------------------------------------------------------------------------- the FSM inventory


class _Reg:
    """Stands in for a :class:`~merlin.targetgen.rtl.fsm.FsmRegister`, same protocol."""

    def __init__(self, module, register, states=None, exported=False):
        self.module, self.register, self.states, self.exported = module, register, states, exported

    @property
    def qualified(self):
        return f"{self.module}.{self.register}"

    def matches_signal(self, path):
        parts = path.replace("/", ".").split(".")
        return len(parts) >= 2 and parts[-1] == self.register


def test_detected_but_undeclared_control_fsms_are_reported():
    """15 detected / 3 exported on one measured target, with the two controllers whose concurrency was
    the whole measurement among the 12 dropped. The inventory widens the engine set, never narrows it."""
    tr = _trace("c", {"x/state_a": _bits("1100")}, {"x/state_a": "eng_a"}, ports=("x/state_a",))
    regs = [_Reg("X", "state_a", 3, True), _Reg("Y", "state_b"), _Reg("Z", "state_c")]
    inv = CAL.engine_inventory(TWO_ENGINE_CONTRACT, [tr], CAL.calibrate_idle([tr]),
                               fsm_registers=regs)
    d = inv.to_dict()
    assert d["n_detected"] == 3
    assert sorted(d["detected_undeclared"]) == ["Y.state_b", "Z.state_c"]
    assert "DETECTED" in d["detected_basis"]


def test_absent_fsm_extraction_is_a_statement_about_the_extraction():
    inv = CAL.engine_inventory(TWO_ENGINE_CONTRACT, [], CAL.calibrate_idle([]))
    assert inv.detected == ()
    assert "NOT about the design" in inv.detected_basis


# --------------------------------------------------------------------------- the three-state invariant


def test_audit_catches_a_refusal_spelled_as_a_number():
    bad = {"eta": {"state": CAL.UNKNOWN, "value": 0.0, "why": "unreadable"}}
    got = CAL.audit(bad)
    assert not got["ok"] and any("carries a value" in v for v in got["violations"])


def test_audit_catches_a_refusal_with_no_reason():
    got = CAL.audit({"eta": {"state": CAL.UNKNOWN, "value": None, "why": ""}})
    assert not got["ok"] and any("no reason" in v for v in got["violations"])


def test_unknown_cannot_be_constructed_without_a_reason():
    with pytest.raises(ValueError):
        CAL.unknown("")
    with pytest.raises(ValueError):
        CAL.measured(None)


def test_every_record_this_module_emits_passes_its_own_audit():
    for contract, traces in ((TWO_ENGINE_CONTRACT, [_overlapping_trace()]),
                             (ONE_ENGINE_CONTRACT, []),
                             (TWO_ENGINE_CONTRACT, [])):
        rec = CAL.calibrate(target="t", contract=contract, traces=traces)
        assert rec["audit"]["ok"], rec["audit"]["violations"]


def test_record_is_json_serialisable():
    rec = CAL.calibrate(target="t", contract=TWO_ENGINE_CONTRACT, traces=[_overlapping_trace()])
    assert json.loads(json.dumps(rec))["schema_version"] == CAL.SCHEMA_VERSION


# --------------------------------------------------------------------------- the driver


DRIVER = merlin_dir() / "experiments" / "capsule_bench" / "harness" / "perf_calibrate.py"


def test_driver_help_runs():
    got = subprocess.run([sys.executable, str(DRIVER), "--help"], capture_output=True, text=True,
                         cwd=str(repo_root()), timeout=180)
    assert got.returncode == 0
    assert "mechanisms" in got.stdout


def test_driver_consumes_the_trace_seam_end_to_end(tmp_path):
    """The whole path: a trace file in the documented shape -> a written record with a measured eta.

    Run against a REAL target so the contract, corpus and regime cover are the real ones, with the
    engine names read out of that target's own declaration rather than written here.
    """
    from merlin.targetgen import target_registry as TR

    target = next((t for t in TR.all_targets()
                   if len(_kinds(TR, t)) >= 2), None)
    if target is None:
        pytest.skip("no reference target declares two engines")
    engines = sorted(_kinds(TR, target))
    trace = {"capsule": "synthetic_pair_probe",
             "columns": {"col_a": _bits("11110000"), "col_b": _bits("00111100")},
             "binding": {"col_a": engines[0], "col_b": engines[1]},
             "port_columns": ["col_a", "col_b"], "state_columns": [],
             "unmeasured_units": [], "work": "probe", "completion_observable": True,
             "provenance": "synthetic; proves the seam, not the hardware"}
    tf = tmp_path / "trace.json"
    tf.write_text(json.dumps(trace))

    got = subprocess.run([sys.executable, str(DRIVER), "--target", target,
                          "--traces", str(tf), "--dry-run"],
                         capture_output=True, text=True, cwd=str(repo_root()), timeout=900)
    assert got.returncode == 0, got.stderr[-3000:]
    assert "eta=0.5000" in got.stdout, got.stdout[-3000:]
    assert "audit ok=True" in got.stdout


def _kinds(TR, target):
    from merlin.perf.occupancy import declared_engines
    try:
        return declared_engines(TR.load_contract(target))
    except Exception:                                       # noqa: BLE001 -- unparseable contract
        return {}
