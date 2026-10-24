"""Model component certification must retain the ordinary RTL admission boundary."""

from contextlib import contextmanager

import pytest

from merlin.targetgen import package_certification, rtl_simulator_execution


def _slot(events):
    @contextmanager
    def reserve(*, timeout):
        events.append("admitted")
        assert timeout == 30
        yield 17
        events.append("released")

    return reserve


def _verifier(events, pin_state):
    def verify(backend, engine, target):
        assert events[0] == "admitted"
        assert (engine, target) == ("verilator", "example")
        if pin_state == "invalid":
            raise ValueError("selected product changed")
        identity = "replacement" if pin_state == "replaced" and "executed" in events else "selected"
        events.append("verified")
        return {"execution_identity": identity}

    return verify


def _executor(events):
    def execute(cb, llvm, **options):
        assert events == ["admitted", "verified"]
        assert 0 < options["timeout"] <= 17
        assert options["inputs"] == {"Activation": [[3]]}
        events.append("executed")
        return {
            "outputs": {"Result": [[6]]},
            "oracle": {"kind": "elaborated_rtl", "derived_from_rtl": True},
            "timing": {"sim_active_s": 2},
        }

    return execute


@pytest.fixture
def certification_execution(monkeypatch, request):
    events = []
    monkeypatch.setattr(rtl_simulator_execution, "verilator_slot", _slot(events))
    monkeypatch.setattr(rtl_simulator_execution, "verify_selected", _verifier(events, request.param))
    monkeypatch.setattr(package_certification._runtime.oot_compile, "run_on_oracle", _executor(events))

    def certify():
        return package_certification._run_certificate_oracle(
            object(), {}, "LLVM", simulator="verilator", target="example", timeout=30, inputs={"Activation": [[3]]}
        )

    return certify, events, request.param


@pytest.mark.parametrize("certification_execution", ["stable"], indirect=True)
def test_certification_retains_verified_product_and_timing(certification_execution):
    certify, events, _ = certification_execution
    result = certify()
    assert result["oracle"]["provenance"]["execution_identity"] == "selected"
    published = package_certification._certificate_oracle_record(result, "verilator", True)
    assert published["provenance"] == result["oracle"]["provenance"]
    assert published["derived_from_rtl"] is True
    assert published["timing"]["sim_active_s"] == 2
    assert published["result"] == "pass"
    assert events == ["admitted", "verified", "executed", "verified", "released"]


@pytest.mark.parametrize("certification_execution", ["invalid", "replaced"], indirect=True)
def test_certification_rejects_changed_product(certification_execution):
    certify, events, pin_state = certification_execution
    with pytest.raises((ValueError, RuntimeError), match="changed"):
        certify()
    assert ("executed" in events) is (pin_state == "replaced")
