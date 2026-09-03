"""Persisting the PER-CYCLE columns, and joining a signal's instance to a declared engine.

The co-simulation occupancy driver measured three controllers every cycle and then wrote only the
totals. The calibration seam needs the columns, not the totals, so a record could sit beside a real
bit-exact measurement with every eta UNKNOWN -- and did. These tests pin the emission and, more
importantly, the join it depends on.

The join is the interesting part. A co-simulation state manifest addresses a signal by its INSTANCE
path (``ex_controller/control_state``); a capability contract and a synthesis FSM inventory both name a
unit by its MODULE (``ExecuteController``). Turning one into the other by resemblance is the guess the
repo's cardinal rule forbids, and it is wrong the moment a design instantiates a module under a name
that does not look like it. So the map comes from the design's own elaborated HW dialect, and a column
whose instance resolves to no DECLARED engine is left unbound and reported -- never attached to
whichever engine happens to be nearest, which would put three decoupled controllers' concurrency inside
one systolic array where it cannot be seen.
"""
from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from merlin.common.paths import merlin_dir

DRIVER = merlin_dir() / "experiments" / "performance_contract" / "gemmini_occupancy.py"

#: A design that instantiates its controllers under names that do NOT resemble their modules. If the
#: join were done by resemblance every one of these would be missed, which is the point.
HW_DIALECT = """
hw.module @Top(in %clock : i1) {
  %0 = hw.instance "u0" @ExecuteController(clock: %clock: i1) -> (x: i1)
  %1 = hw.instance "u1" @LoadController(clock: %clock: i1) -> (x: i1)
  %2 = hw.instance "widget" @SomeUndeclaredThing(clock: %clock: i1) -> (x: i1)
  hw.output
}
"""


@pytest.fixture(scope="module")
def driver():
    spec = importlib.util.spec_from_file_location("_gemmini_occupancy_under_test", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_checkout(tmp_path, monkeypatch, driver):
    """A modelling checkout laid out the way the driver resolves one, with a tiny HW dialect."""
    outputs = tmp_path / "runs" / "circt-arc" / "t" / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "T_core_hw.mlir").write_text(HW_DIALECT, encoding="utf-8")
    monkeypatch.setenv("MERLIN_MLC_DIR", str(tmp_path))
    return tmp_path


def test_instance_to_module_comes_from_the_design_not_from_resemblance(fake_checkout, driver):
    got = driver.instance_modules("t")
    assert got == {"u0": "ExecuteController", "u1": "LoadController",
                   "widget": "SomeUndeclaredThing"}


def test_an_absent_hw_dialect_binds_nothing_rather_than_guessing(tmp_path, monkeypatch, driver):
    (tmp_path / "runs" / "circt-arc" / "t" / "outputs").mkdir(parents=True)
    monkeypatch.setenv("MERLIN_MLC_DIR", str(tmp_path))
    assert driver.instance_modules("t") == {}


def test_a_column_with_no_declared_engine_is_left_unbound_and_reported(fake_checkout, driver,
                                                                      monkeypatch):
    """Unbound is the honest state. Binding it to the nearest engine would hide the concurrency."""
    monkeypatch.setattr(driver, "STATE_SIGNALS",
                        ("u0/control_state", "u1/control_state", "widget/control_state"))
    monkeypatch.setattr(driver, "PORT_SIGNALS", ())
    monkeypatch.setattr(driver, "engine_by_module",
                        lambda target: {"ExecuteController": "systolic_mesh",
                                        "LoadController": "LoadController"})
    trace = {"u0/control_state": ["0", "1"], "u1/control_state": ["0", "0"],
             "widget/control_state": ["1", "1"]}
    meta = {"shape": "2x2x2", "signals_present": sorted(trace), "signals_absent": [],
            "bit_exact": True}
    got = driver.mechanism_traces("t", [trace], [meta])[0]
    assert got["binding"] == {"u0/control_state": "systolic_mesh",
                              "u1/control_state": "LoadController"}
    assert "widget/control_state" not in got["binding"]
    assert "UNBOUND" in got["provenance"]
    assert "widget/control_state" in got["provenance"]


def test_the_emitted_trace_is_the_shape_the_calibration_seam_consumes(fake_checkout, driver,
                                                                     monkeypatch):
    from merlin.perf import calibration as CAL

    monkeypatch.setattr(driver, "STATE_SIGNALS", ("u0/control_state",))
    monkeypatch.setattr(driver, "PORT_SIGNALS", ("u0/io_busy",))
    monkeypatch.setattr(driver, "engine_by_module",
                        lambda target: {"ExecuteController": "systolic_mesh"})
    trace = {"u0/control_state": ["0", "1", "1", "0"], "u0/io_busy": ["0", "1", "1", "0"]}
    meta = {"shape": "2x2x2", "signals_present": sorted(trace), "signals_absent": ["u9/absent"],
            "bit_exact": True}
    got = driver.mechanism_traces("t", [trace], [meta])[0]
    # Round-trips through JSON, because that is how the driver hands it over.
    got = json.loads(json.dumps(got))
    mt = CAL.MechanismTrace(
        capsule=got["capsule"], columns=got["columns"], binding=got["binding"],
        port_columns=tuple(got["port_columns"]), state_columns=tuple(got["state_columns"]),
        unmeasured_units=tuple(got["unmeasured_units"]), work=got["work"],
        completion_observable=got["completion_observable"], provenance=got["provenance"])
    assert mt.sampled_cycles == 4
    assert mt.state_columns == ("u0/control_state",)
    assert mt.port_columns == ("u0/io_busy",)
    assert mt.unmeasured_units == ("u9/absent",)


def test_completion_observable_is_stated_false_not_defaulted_true(fake_checkout, driver, monkeypatch):
    """Defaulting it True is how an unmeasured trait becomes a satisfied concurrency gate."""
    monkeypatch.setattr(driver, "STATE_SIGNALS", ("u0/control_state",))
    monkeypatch.setattr(driver, "PORT_SIGNALS", ())
    monkeypatch.setattr(driver, "engine_by_module",
                        lambda target: {"ExecuteController": "systolic_mesh"})
    got = driver.mechanism_traces(
        "t", [{"u0/control_state": ["0", "1"]}],
        [{"shape": "2x2x2", "signals_present": ["u0/control_state"], "signals_absent": [],
          "bit_exact": True}])[0]
    assert got["completion_observable"] is False


def test_the_work_fingerprint_carries_whether_the_run_was_bit_exact(fake_checkout, driver,
                                                                    monkeypatch):
    """An occupancy vector from a run that computed the wrong thing is not the machine's behaviour."""
    monkeypatch.setattr(driver, "STATE_SIGNALS", ("u0/control_state",))
    monkeypatch.setattr(driver, "PORT_SIGNALS", ())
    monkeypatch.setattr(driver, "engine_by_module",
                        lambda target: {"ExecuteController": "systolic_mesh"})
    for flag in (True, False):
        got = driver.mechanism_traces(
            "t", [{"u0/control_state": ["0", "1"]}],
            [{"shape": "2x2x2", "signals_present": ["u0/control_state"], "signals_absent": [],
              "bit_exact": flag}])[0]
        assert f"bit_exact={flag}" in got["work"]


def test_the_checkout_context_manager_restores_the_working_directory(driver, tmp_path):
    """The modelling repo resolves its artifacts relative to CWD; a leaked chdir breaks the caller."""
    import os
    before = os.getcwd()
    with pytest.raises(RuntimeError):
        with driver._in_mlc_checkout(tmp_path):
            assert os.path.realpath(os.getcwd()) == os.path.realpath(str(tmp_path))
            raise RuntimeError("the model raised")
    assert os.getcwd() == before
