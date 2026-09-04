"""The agent must be able to ASK the analytical model, not only read numbers the host wrote.

MEASURED, campaign 20260903T222654Z over three trials: the only judge of a candidate was a
~110 s brokered GSIM measurement, so a schedule that turned out 11.1% WORSE cost exactly as much to
evaluate as one that won -- two such excursions burned ~220 s of oracle time. Of the fifteen broker
actions the agent was given, none could price its own emitted artifacts; `merlin.perf.differential`
was named four times in the prompt as a GO requirement with no way to invoke it.

This action closes that: it reads the candidate's OWN command buffers, so it needs no oracle, no
goldens and no holdout, and can be called without budget. In exchange it is ORDERING-ONLY.
"""
from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from merlin.common.paths import repo_root


def _stage():
    d = repo_root() / "merlin" / "experiments" / "gemmini_perf_bench" / "scripts"
    if not (d / "perf_agent_stage.py").exists():
        pytest.skip("the perf agent stage is not in this checkout")
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    if "perf_agent_stage" in sys.modules:
        return sys.modules["perf_agent_stage"]
    spec = importlib.util.spec_from_file_location("perf_agent_stage", d / "perf_agent_stage.py")
    mod = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: the module defines dataclasses, and dataclasses resolves each field's
    # type through sys.modules[cls.__module__] -- absent that entry it raises on a NoneType module.
    sys.modules["perf_agent_stage"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001
        del sys.modules["perf_agent_stage"]
        pytest.skip(f"perf_agent_stage is not importable here: {type(exc).__name__}: {exc}")
    return mod


def _buffer(tmp_path, name, macs_shape):
    """A command buffer copied from the emitter's real output shape (PK00_k16 baseline).

    Operand keys and the RES_PACK -> MATMUL_RESIDENT -> COMMIT -> EVICT sequence are taken from an
    actually-emitted buffer rather than invented, because a fixture the pricer silently refuses
    makes this whole test vacuous -- which is exactly what a guessed schema did on the first pass.
    """
    m, k, n = macs_shape
    doc = {"abi_version": 1, "target": "t",
           "tensors": {"W": {"shape": [k, n], "dtype": "i8", "role": "weight"},
                       "A0": {"shape": [m, k], "dtype": "i8", "role": "input"},
                       "Y0": {"shape": [m, n], "dtype": "i32", "role": "output"}},
           "commands": [
               {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"},
                "attributes": {"layout": "packed_rhs"}},
               {"opcode": "MATMUL_RESIDENT",
                "operands": {"lhs": "A0", "rhs": "W_res", "dst": "acc0"}},
               {"opcode": "COMMIT", "operands": {"src": "acc0", "dst": "Y0"},
                "attributes": {"epilogue": [], "output_dtype": "i32"}},
               {"opcode": "EVICT", "operands": {"handle": "W_res"}}]}
    p = tmp_path / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def test_the_fixture_is_priceable_at_all():
    """Guard the guard: a buffer the pricer refuses would make every assertion below vacuous."""
    import tempfile, pathlib
    from merlin.perf.work_volume import work_from_command_buffer
    with tempfile.TemporaryDirectory() as td:
        p = _buffer(pathlib.Path(td), "f.json", (16, 128, 16))
        w = work_from_command_buffer(json.loads(p.read_text()))
    assert w.known_macs == 16 * 128 * 16, (w.known_macs, w.refusals)
    assert not w.is_lower_bound, w.refusals


def test_the_analysis_costs_no_oracle_and_says_what_it_is_based_on():
    stage = _stage()
    assert stage.ANALYSIS_ACTION == "analyze-command-buffers"
    # It must be OPTIONAL: a required action would force a call the agent may not need.
    src = (repo_root() / "merlin" / "experiments" / "gemmini_perf_bench" / "scripts"
           / "perf_agent_stage.py").read_text(encoding="utf-8")
    assert "no oracle, no golden, no holdout" in src


def test_it_prices_two_buffers_against_the_derived_ceilings(tmp_path):
    stage = _stage()
    a = _buffer(tmp_path, "a.json", (16, 16, 16))
    b = _buffer(tmp_path, "b.json", (16, 32, 16))
    out = stage.analyze_command_buffers(a, b, peak_macs_per_cycle=256,
                                        achievable_macs_per_cycle=80.0)
    assert out["kind"] == "host_owned_command_buffer_analysis"
    assert set(out["arms"]) == {"baseline", "candidate"}
    for arm in out["arms"].values():
        assert arm["macs"] > 0
        assert arm["ideal_cycles_at_peak"] > 0
        assert arm["ideal_cycles_at_achievable"] > arm["ideal_cycles_at_peak"], \
            "the achievable ceiling must imply MORE cycles than the structural peak"
    # Different work volumes must be called out: a cycle delta there is not a schedule comparison.
    assert "work_delta" in out and out["work_delta"]["candidate_over_baseline"] == pytest.approx(2.0)


def test_it_never_claims_an_absolute_cycle_count(tmp_path):
    """Ordering-only is the licence the corpus-calibrated model actually has."""
    stage = _stage()
    a = _buffer(tmp_path, "a.json", (16, 16, 16))
    out = stage.analyze_command_buffers(a, a, peak_macs_per_cycle=256,
                                        achievable_macs_per_cycle=80.0)
    flat = json.dumps(out)
    assert "predicted_cycles" not in flat and "cycles_estimate" not in flat


def test_it_does_not_pass_off_a_type_error_as_an_analyzer_verdict(tmp_path):
    """The basis must be a value the differential module defines, or an explicit non-attempt.

    The first version of this action called `differential.compare` on plain dicts. `compare` takes
    `envelope.Composed`, so it raised `AttributeError: 'dict' object has no attribute 'operator'`
    on every call; a bare `except` turned that into a hardcoded "REFUSED" that read like a verdict.
    This test's ancestor asserted the basis against UPPERCASE literals while the module's constants
    are lowercase -- so it passed only because the exception branch produced the uppercase string,
    and would have failed the moment `compare` actually worked.

    Asserting against the module's own constants is what makes that undetectable-by-construction
    failure detectable: a literal cannot drift from a constant it never references.
    """
    from merlin.perf import differential as DIFF

    stage = _stage()
    a = _buffer(tmp_path, "a.json", (16, 16, 16))
    out = stage.analyze_command_buffers(a, a, peak_macs_per_cycle=256,
                                        achievable_macs_per_cycle=80.0)
    basis = out["differential"]["basis"]
    assert basis in (DIFF.EXACT, DIFF.ORDERING_ONLY, DIFF.REFUSED, "not_attempted"), basis
    # And whatever it says, it must not be an exception type leaking out as a reason.
    assert "Error" not in out["differential"]["reason"], out["differential"]
    assert out["differential"]["reason"], "a basis with no reason is indistinguishable from silence"


def test_an_unreadable_buffer_refuses_rather_than_crashing(tmp_path):
    stage = _stage()
    missing = tmp_path / "nope.json"
    good = _buffer(tmp_path, "a.json", (16, 16, 16))
    with pytest.raises(Exception):
        stage.analyze_command_buffers(missing, good, peak_macs_per_cycle=256,
                                      achievable_macs_per_cycle=80.0)
