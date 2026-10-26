"""A budgeted gSIM run is bounded by its wall budget, not the selected command's cycle cap.

A fake harness stands in for the emulator: it simulates at a scaled 1000 cycles per second, honours its
``+max-cycles`` argument exactly as the real harness does (stop and exit with the cycle-cap status) and
needs 900 cycles to finish. The selected command caps it at 300 cycles -- the scaled analogue of a cap
that ends every run after ~10 minutes whatever its budget. Process output grants no RTL qualification.
"""

import json
import os
import subprocess
from contextlib import nullcontext

import pytest
from test_bound_chipyard_rocc_backend import _bound, _console, _file, _gsim, _pin

from merlin.runtime.backends import chipyard_rocc
from merlin.runtime.backends import chipyard_rocc_bound as B
from merlin_experiments.phase1 import gsim_engine_bound as E

pytest_plugins = ("test_bound_chipyard_rocc_backend",)

RATE, NEED, SELECTED_CAP = 1000, 900, 300


def _harness(tmp_path):
    body = "\n".join(
        (
            "#!/bin/sh",
            "cap=0",
            'for a do case "$a" in +max-cycles=*) cap="${a#+max-cycles=}";; esac; done',
            f'if [ "$cap" -lt {NEED} ]; then',
            f'  sleep "$(awk "BEGIN {{ print $cap / {RATE} }}")"',
            '  echo "GSIM timeout: no RTL/TSI completion after $cap cycles" >&2',
            f"  exit {chipyard_rocc._GSIM_CYCLE_CAP_EXIT}",
            "fi",
            f"sleep {NEED / RATE}",
            "printf '%s\\n' '" + _console().rstrip() + "'",
            f"printf 'GSIM model finished execution.\\nCycles: {NEED}\\n' >&2",
            "",
        )
    )
    path = _file(tmp_path, "fake-harness", body)
    path.chmod(0o700)
    return path


@pytest.fixture
def budgeted(selection, tmp_path, monkeypatch):
    from merlin.targetgen import rtl_engine_policy

    engine = selection[0]["runner"]["chipyard_rocc"]["engines"]["spike"]
    engine["binary"] = _pin(_harness(tmp_path))
    engine, _, _ = _gsim(selection, tmp_path)
    engine["argv"].append(f"+max-cycles={SELECTED_CAP}")
    monkeypatch.setattr(rtl_engine_policy, "gsim_runtime_slot", lambda **kwargs: nullcontext())
    monkeypatch.delenv(B.GSIM_BUDGET_MAX_CYCLES_ENV, raising=False)
    return _bound(selection), _file(tmp_path, "requested.elf", "owned original executable")


def _process_records(root):
    rows = [json.loads(p.read_text()) for p in root.glob("invocations/*/invocation.json")]
    return sorted((r for r in rows if r.get("stage") == "selected_rocc_process"), key=lambda r: r["started_ns"])


def test_the_selected_cycle_cap_ends_a_run_its_wall_budget_allows(budgeted, tmp_path):
    bound, elf = budgeted
    with pytest.raises(subprocess.CalledProcessError) as stopped:
        bound.run_elf(elf, simulator="gsim", timeout=4)
    assert stopped.value.returncode == chipyard_rocc._GSIM_CYCLE_CAP_EXIT
    (run,) = E.engine_runs(tmp_path)
    assert run["capped"] and run["cycles"] == SELECTED_CAP and run["wall_s"] < 4


def test_a_budget_derived_cycle_bound_lets_the_run_finish_within_its_wall_budget(budgeted, tmp_path):
    bound, elf = budgeted
    with pytest.raises(subprocess.CalledProcessError):
        bound.run_elf(elf, simulator="gsim", timeout=4)  # the first job learns the engine's rate
    rate = E.engine_rate({"first": {"engine_runs": E.engine_runs(tmp_path)}})
    assert rate["max_cycles_per_s"] == pytest.approx(RATE, rel=0.5)
    budget = E.engine_cycle_bound(4, rate)
    assert budget["max_cycles"] >= NEED and "wall budget 4 s" in budget["basis"]
    with E.engine_cycle_budget(budget["max_cycles"]):
        assert bound.run_elf(elf, simulator="gsim", timeout=4) == _console()
    last = _process_records(tmp_path)[-1]
    assert f"+max-cycles={budget['max_cycles']}" in last["argv"] and last["returncode"] == 0
    finished = [run for run in E.engine_runs(tmp_path) if not run["capped"]]
    assert [run["cycles"] for run in finished] == [NEED]
    assert B.GSIM_BUDGET_MAX_CYCLES_ENV not in os.environ  # scoped to the measurement


def test_a_budget_never_lowers_the_selected_cap_and_is_validated(budgeted, monkeypatch):
    bound, elf = budgeted
    monkeypatch.setenv(B.GSIM_BUDGET_MAX_CYCLES_ENV, "10")
    with pytest.raises(subprocess.CalledProcessError):
        bound.run_elf(elf, simulator="gsim", timeout=4)
    for malformed in ("0", "-5", "lots"):
        monkeypatch.setenv(B.GSIM_BUDGET_MAX_CYCLES_ENV, malformed)
        with pytest.raises(B.RoCCExecutionError, match="budgeted gSIM cycle bound"):
            bound.run_elf(elf, simulator="gsim", timeout=4)


def test_without_an_observed_rate_the_selected_bound_stays_and_says_so():
    budget = E.engine_cycle_bound(600, None)
    assert budget["max_cycles"] is None and "no gSIM engine rate observed" in budget["basis"]
    with E.engine_cycle_budget(None) as applied:
        assert applied is None
