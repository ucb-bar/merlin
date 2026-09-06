"""A measurement must name the engine build that produced it, and be refused when it names another.

`validate_execution` checks three pin fields -- binary, FIRRTL and model -- that its own caller copies
out of the certificate before handing them back, so those three comparisons cannot fail. This suite
pins the one comparison that CAN: the pinned binary must appear among the digests of the engine build
the runner actually loaded from.
"""
from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace

import pytest

from merlin.common.paths import merlin_dir

_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _load(name: str):
    source = _SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = _load("perf_gsim_gate")
PAIRED = _load("run_paired_perf_bench")

_PINNED = "a" * 64
_OTHER = "b" * 64


def _certificate():
    pins = {name: {"sha256": _PINNED} for name in GATE.REQUIRED_PINS}
    return SimpleNamespace(sha256="c" * 64, pins=pins)


def _decision():
    return SimpleNamespace(certificate_sha256="c" * 64, admitted=True, selected_engine="gsim",
                           eligible=True, phase="development_correctness", refusal_reason=None,
                           to_dict=lambda: {"selected_engine": "gsim"})


def _execution(**overrides):
    execution = {"engine": "gsim", "status": "pass", "derived_from_rtl": True,
                 "cycle_accurate": True, "elf_sha256": "d" * 64, "cycles": 304,
                 "binary_sha256": _PINNED, "firrtl_sha256": _PINNED, "model_sha256": _PINNED}
    execution.update(overrides)
    return execution


def test_a_run_on_the_pinned_engine_build_is_admitted() -> None:
    """CONTROL. Without this, every refusal below could be the check firing on everything."""
    record = GATE.validate_execution(
        _certificate(), _decision(),
        _execution(observed_engine_binaries={"status": "observed",
                                             "digests": [_OTHER, _PINNED]}))
    assert record["admitted"] is True


def test_a_run_on_a_different_engine_build_is_refused() -> None:
    """The defect this exists for: the number came from a build the certificate does not pin."""
    with pytest.raises(GATE.GsimGateError, match="does not contain the pinned"):
        GATE.validate_execution(
            _certificate(), _decision(),
            _execution(observed_engine_binaries={"status": "observed", "digests": [_OTHER]}))


@pytest.mark.parametrize("observed", [
    {"status": "UNKNOWN", "reason": "engine build not established"},
    None,
    {"status": "observed"},
])
def test_unestablished_provenance_is_inert_not_agreement(observed) -> None:
    """A run whose engine home could not be resolved must not be REFUSED on that basis -- the check
    is evidence-driven. It must equally never be read as a match: the audit record carries the
    UNKNOWN through so a reader can tell an unverified run from a verified one."""
    execution = _execution()
    if observed is not None:
        execution["observed_engine_binaries"] = observed
    if observed == {"status": "observed"}:
        # Claimed observed with no digests is a malformed claim, not an absence: refuse it.
        with pytest.raises(GATE.GsimGateError, match="does not contain the pinned"):
            GATE.validate_execution(_certificate(), _decision(), execution)
        return
    record = GATE.validate_execution(_certificate(), _decision(), execution)
    assert record["admitted"] is True
    assert record["execution"].get("observed_engine_binaries") == observed


def test_the_runner_records_what_it_loaded_or_says_it_could_not() -> None:
    """The evidence the check consumes must be produced. An oracle carrying engine provenance yields
    its digests; one carrying none yields UNKNOWN rather than an empty match."""
    assert PAIRED._observed_engine_binaries(
        {"provenance": {"binaries": {"emu": _PINNED}}}) == {
            "status": "observed", "digests": [_PINNED]}
    for barren in ({}, {"provenance": {}}, {"provenance": {"binaries": {}}}, None, "not-a-mapping"):
        assert PAIRED._observed_engine_binaries(barren)["status"] == "UNKNOWN"
