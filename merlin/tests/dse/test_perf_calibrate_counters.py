"""The SECOND calibration seam: aggregate readings from a target's own combination counters.

The trace seam needs a per-cycle instrument, and on a target with no built co-simulation model every
eta in a calibration record stays UNKNOWN however good the RTL is. A target whose RTL counts the cycles
each SUBSET of its engines was busy has already measured realised overlap, so this seam exists to let
that reach the record.

Two instruments in one record is exactly the shape that invites a fabricated number, so what is pinned
here is every way the counter path could invent one:

* a missing counter must make the whole reading UNKNOWN, never a smaller overlap -- and must propagate,
  leaving the CORPUS operator unestablished rather than derived from whatever was readable;
* a per-engine busy total must be the single counter PLUS every combination containing it, because
  reading the singles alone understates the busiest engine, which is eta's denominator;
* no overlappable time must be reported as 0/0 undefined, never as ``SUM``;
* the KIND axis must refuse while no resource kind is declared for the counter engines -- deriving
  "LD means movement" from a counter's spelling is the overfit the repo's cardinal rule forbids;
* readings taken over two different engine sets must not be summed into one corpus figure;
* and the counter block must NEVER set ``ran_against_traces``, enter the capsule cover, or be compared
  with the trace block. They are two instruments, and this repo has already paid for conflating two
  instruments that shared a name.

The counter values below are written out so the busy totals and eta are checkable by hand.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from merlin.common.paths import merlin_dir, repo_root
from merlin.perf import calibration as CAL
from merlin.perf import hw_counters as HC

ONE_ENGINE_CONTRACT = {
    "compute_units": [{"name": "eng_a", "kind": "systolic", "dtypes": ["int8"], "ops": ["matmul"]}]
}

#: A synthetic counter header in the shape a target ships one: three singles, three pairs and the
#: triple, over a prefix. The engine tokens are FACTORED OUT of the names by the library -- nothing
#: below tells it what the engines are called.
HEADER = """
#define UNIT_A_CYCLES 1
#define UNIT_B_CYCLES 2
#define UNIT_C_CYCLES 3
#define UNIT_A_B_CYCLES 4
#define UNIT_A_C_CYCLES 5
#define UNIT_B_C_CYCLES 6
#define UNIT_A_B_C_CYCLES 7
"""


def _counters():
    got = HC.derive_occupancy_counters(HEADER)
    assert got.complete(), got.to_dict()
    return got


#: One hand-checkable run. Busy totals are single + every combination containing that engine:
#:   A = 100 + 30 + 10 + 5 = 145
#:   B =  40 + 30 +  0 + 5 = 75
#:   C =  20 + 10 +  0 + 5 = 35
#: realised (>=2 engines) = 30 + 10 + 0 + 5 = 45
#: total = 255; available = min(255 - 145, 255 // 2) = min(110, 127) = 110
#: eta = 45 / 110 = 0.4090909...
VALUES = {"UNIT_A_CYCLES": 100, "UNIT_B_CYCLES": 40, "UNIT_C_CYCLES": 20,
          "UNIT_A_B_CYCLES": 30, "UNIT_A_C_CYCLES": 10, "UNIT_B_C_CYCLES": 0,
          "UNIT_A_B_C_CYCLES": 5}
BUSY = {"A": 145, "B": 75, "C": 35}
REALISED, AVAILABLE = 45, 110


def _reading(workload="w0", values=None, **kw):
    return CAL.CounterReading(workload=workload, values=dict(VALUES if values is None else values),
                              counters=_counters(), total_cycles=kw.pop("total_cycles", 300),
                              provenance=kw.pop("provenance", "synthetic counter block"), **kw)


# --------------------------------------------------------------------------- the reading itself


def test_busy_total_is_the_single_plus_every_combination_containing_it():
    """Reading the singles as whole-engine totals understates the busiest engine and inflates eta."""
    got = CAL.counter_calibration([_reading()])
    busy = got["runs"][0]["busy_cycles"]
    assert busy["state"] == CAL.MEASURED
    assert busy["value"] == BUSY
    # The naive reading would have been the singles alone; it must NOT be what came back.
    assert busy["value"] != {"A": 100, "B": 40, "C": 20}


def test_eta_and_the_operator_are_measured_from_the_counter_values():
    got = CAL.counter_calibration([_reading()])
    run = got["runs"][0]
    assert run["realised_cycles"]["value"] == REALISED
    assert run["available_cycles"]["value"] == AVAILABLE
    assert run["eta"]["value"] == pytest.approx(REALISED / AVAILABLE)
    axis = got["engine_axis"]
    assert axis["eta"]["value"] == pytest.approx(REALISED / AVAILABLE)
    assert axis["operator"]["value"] == "partial"
    assert axis["realised_cycles"]["value"] == REALISED
    assert axis["available_cycles"]["value"] == AVAILABLE


def test_the_engine_axis_comes_from_the_counter_header_not_from_a_contract():
    got = CAL.counter_calibration([_reading()])
    assert got["engines"] == ["A", "B", "C"]
    assert "counter header" in got["engine_axis_source"]


# --------------------------------------------------------------------------- the refusals


def test_a_missing_counter_makes_the_reading_unknown_never_a_smaller_overlap():
    partial = {k: v for k, v in VALUES.items() if k != "UNIT_A_B_CYCLES"}
    got = CAL.counter_calibration([_reading(values=partial)])
    run = got["runs"][0]
    for key in ("eta", "busy_cycles", "realised_cycles", "available_cycles"):
        assert run[key]["state"] == CAL.UNKNOWN, key
        assert run[key]["value"] is None, key
        assert run[key]["why"], key
    assert "UNIT_A_B_CYCLES" in run["eta"]["why"]


def test_one_unreadable_run_leaves_the_corpus_operator_unestablished():
    """UNKNOWN propagates. Dropping the unreadable run reweights the corpus towards the readable one."""
    partial = {k: v for k, v in VALUES.items() if k != "UNIT_B_C_CYCLES"}
    got = CAL.counter_calibration([_reading("good"), _reading("bad", values=partial)])
    assert got["runs_without_a_reading"] == ["bad"]
    for axis in ("engine_axis", "kind_axis"):
        assert got[axis]["operator"]["state"] == CAL.UNKNOWN
        assert got[axis]["operator"]["value"] is None
        assert "propagates" in got[axis]["operator"]["why"]


def test_no_overlappable_time_is_undefined_not_sum():
    """One engine busy alone cannot overlap with anything; 0/0 must not classify as ``SUM``."""
    solo = {k: 0 for k in VALUES}
    solo["UNIT_A_CYCLES"] = 200
    got = CAL.counter_calibration([_reading(values=solo)])
    run = got["runs"][0]
    assert run["eta"]["state"] == CAL.UNKNOWN
    assert "0/0" in run["eta"]["why"] or "no overlap was AVAILABLE" in run["eta"]["why"]
    assert got["engine_axis"]["operator"]["value"] is None


def test_readings_over_two_different_engine_sets_are_not_summed():
    other = HC.derive_occupancy_counters(
        "#define OTHER_X_CYCLES 1\n#define OTHER_Y_CYCLES 2\n#define OTHER_X_Y_CYCLES 3\n")
    a = _reading("a")
    b = CAL.CounterReading(workload="b", counters=other,
                           values={"OTHER_X_CYCLES": 10, "OTHER_Y_CYCLES": 10,
                                   "OTHER_X_Y_CYCLES": 5},
                           total_cycles=30, provenance="a second, different counter block")
    got = CAL.counter_calibration([a, b])
    assert got["engines"] == []
    assert got["engine_axis"]["operator"]["state"] == CAL.UNKNOWN
    assert "DIFFERENT engine sets" in got["engine_axis"]["operator"]["why"]


def test_no_counter_reading_at_all_is_unknown_with_a_reason():
    got = CAL.counter_calibration([])
    assert got["n_runs"] == 0
    for axis in ("engine_axis", "kind_axis"):
        for key in ("operator", "eta"):
            assert got[axis][key]["state"] == CAL.UNKNOWN
            assert got[axis][key]["value"] is None
            assert got[axis][key]["why"]


# --------------------------------------------------------------------------- the kind axis


def test_the_kind_axis_refuses_while_no_kind_is_declared_for_the_counter_engines():
    got = CAL.counter_calibration([_reading()])
    why = got["kind_axis"]["operator"]["why"]
    assert got["kind_axis"]["operator"]["state"] == CAL.UNKNOWN
    assert "cannot be derived from a counter's name" in why
    assert "'A'" in why or "A" in why


def test_the_kind_axis_resolves_once_the_producer_declares_the_kinds():
    kinds = {"A": "compute", "B": "movement", "C": "movement"}
    got = CAL.counter_calibration([_reading(kind_of=kinds)])
    axis = got["kind_axis"]
    assert axis["operator"]["state"] == CAL.MEASURED, axis["operator"].get("why")
    assert axis["eta"]["state"] == CAL.MEASURED
    assert got["declared_kinds"] == kinds


def test_a_declared_kind_is_recorded_as_the_producers_declaration_not_derived():
    """The kinds arrive as an INPUT. Nothing here may read them out of the counter's spelling."""
    got = CAL.counter_calibration([_reading(kind_of={"A": "compute", "B": "compute",
                                                     "C": "compute"})])
    # All one kind -> composition_operator collapses them into one group and correctly refuses.
    assert got["kind_axis"]["operator"]["state"] == CAL.UNKNOWN
    assert got["kind_axis"]["operator"]["why"]


# --------------------------------------------------------------------------- the two instruments


def test_counter_readings_never_set_ran_against_traces():
    rec = CAL.calibrate(target="t", contract=ONE_ENGINE_CONTRACT, counter_readings=[_reading()])
    assert rec["ran_against_traces"] is False
    assert rec["n_traces"] == 0
    assert rec["ran_against_counters"] is True
    assert rec["n_counter_runs"] == 1
    assert rec["measured_basis"][CAL.COUNTER_INSTRUMENT] is True
    assert rec["measured_basis"][CAL.TRACE_INSTRUMENT] is False
    assert rec["measured_basis"]["any"] is True


def test_the_measurement_basis_says_which_instrument_ran_and_which_did_not():
    rec = CAL.calibrate(target="t", contract=ONE_ENGINE_CONTRACT, counter_readings=[_reading()])
    basis = rec["measurement_basis"]
    assert "NO per-cycle trace" in basis
    assert "CounterReading seam" in basis
    plan = CAL.calibrate(target="t", contract=ONE_ENGINE_CONTRACT)
    assert "PLAN, not a calibration" in plan["measurement_basis"]
    assert plan["measured_basis"]["any"] is False


def test_the_trace_side_stays_unknown_when_only_counters_were_supplied():
    """No number crosses between the blocks: a counter eta must not fill the trace-side composition."""
    rec = CAL.calibrate(target="t", contract=ONE_ENGINE_CONTRACT, counter_readings=[_reading()])
    for axis in ("engine_axis", "kind_axis"):
        assert rec["composition"][axis]["operator"]["state"] == CAL.UNKNOWN
    assert rec["capsules"] == []
    assert rec["counter_calibration"]["engine_axis"]["operator"]["state"] == CAL.MEASURED


def test_the_record_says_the_two_instruments_are_not_comparable():
    rec = CAL.calibrate(target="t", contract=ONE_ENGINE_CONTRACT, counter_readings=[_reading()])
    assert CAL.INSTRUMENTS_NOT_COMPARABLE == rec["counter_calibration"]["not_comparable_with_traces"]
    assert "never merged" in CAL.INSTRUMENTS_NOT_COMPARABLE


def test_a_counter_only_record_passes_its_own_audit_and_serialises():
    rec = CAL.calibrate(target="t", contract=ONE_ENGINE_CONTRACT, counter_readings=[_reading()])
    assert rec["audit"]["ok"], rec["audit"]["violations"]
    assert json.loads(json.dumps(rec))["ran_against_counters"] is True


# --------------------------------------------------------------------------- the driver


DRIVER = merlin_dir() / "experiments" / "capsule_bench" / "harness" / "perf_calibrate.py"
PRODUCER = merlin_dir() / "experiments" / "performance_contract" / "counter_occupancy.py"


def test_both_drivers_expose_the_counter_seam():
    for script, needle in ((DRIVER, "--counters"), (PRODUCER, "--simulator")):
        got = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True,
                             cwd=str(repo_root()), timeout=180)
        assert got.returncode == 0, got.stderr[-2000:]
        assert needle in got.stdout, got.stdout


def test_driver_drops_an_empty_reading_rather_than_calling_it_zero_overlap(tmp_path):
    """A run the producer marked unusable must not become a run that measured no overlap."""
    target = _a_target_with_counters()
    doc = {"readings": [{"workload": "no_counters", "values": {},
                         "dropped": "the bracket did not fire"}]}
    f = tmp_path / "counters.json"
    f.write_text(json.dumps(doc))
    got = subprocess.run([sys.executable, str(DRIVER), "--target", target,
                          "--counters", str(f), "--dry-run"],
                         capture_output=True, text=True, cwd=str(repo_root()), timeout=900)
    assert got.returncode != 0
    assert "every supplied counter reading was dropped" in (got.stdout + got.stderr)


def test_driver_refuses_a_file_recorded_over_a_different_engine_set(tmp_path):
    target = _a_target_with_counters()
    doc = {"counter_block": {"counters": {"engines": ["NOPE"]}},
           "readings": [{"workload": "w", "values": {"X": 1}}]}
    f = tmp_path / "counters.json"
    f.write_text(json.dumps(doc))
    got = subprocess.run([sys.executable, str(DRIVER), "--target", target,
                          "--counters", str(f), "--dry-run"],
                         capture_output=True, text=True, cwd=str(repo_root()), timeout=900)
    assert got.returncode != 0
    assert "disagree" in (got.stdout + got.stderr)


def _a_target_with_counters() -> str:
    """A registered target whose OWN shipped header exposes combination counters, or skip.

    Resolved from the registry rather than named, so this test moves with the roster instead of
    pinning one target the way the repo's cardinal rule forbids.
    """
    from merlin.targetgen import target_registry as TR
    for name in TR.all_targets():
        try:
            if HC.counters_for_target(name).get("status") == "derived":
                return name
        except Exception:                       # noqa: BLE001 -- an unloadable target is not a match
            continue
    pytest.skip("no registered target exposes combination counters on this host")
    raise AssertionError("unreachable")


# --------------------------------------------------------------------------- the producer


@pytest.fixture(scope="module")
def producer():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_counter_occupancy_under_test", PRODUCER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_producer_names_no_target_in_its_workload(producer):
    """The command buffer is the generic ABI vocabulary; the target is only ever a parameter."""
    cb = producer.matmul_cb("some_target", 16, 16, 16)
    assert cb["target"] == "some_target"
    assert {c["opcode"] for c in cb["commands"]} == {"RES_PACK", "MATMUL_RESIDENT", "COMMIT", "EVICT"}


def test_provenance_says_the_revision_is_unrecorded_when_no_pin_was_declared(producer, tmp_path):
    header = tmp_path / "h.h"
    header.write_text(HEADER)
    got = producer.provenance("t", [], str(header))
    assert got["pins_declared"] == []
    assert "UNRECORDED" in got["pins_note"]
    assert got["all_pins_ok"] is None


def test_provenance_refuses_an_unknown_pin_rather_than_reporting_a_clean_run(producer, tmp_path):
    header = tmp_path / "h.h"
    header.write_text(HEADER)
    got = producer.provenance("t", ["no_such_pin_exists_anywhere"], str(header))
    assert "unavailable" in got
    assert "all_pins_ok" not in got


def test_a_target_with_no_counter_block_is_refused_not_defaulted(producer, monkeypatch):
    monkeypatch.setattr(producer.HC, "counters_for_target",
                        lambda target: {"status": "unavailable", "why": "no header could be read"})
    with pytest.raises(SystemExit) as exc:
        producer.counter_block("t")
    assert "unavailable" in str(exc.value)
