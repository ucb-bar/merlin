"""The selected OOT region boundary retains original caller custody checks."""

import hashlib
from contextvars import ContextVar

import pytest

from merlin.compiler.lower.regions import lower_region
from merlin.targetgen import capsule_common
from merlin.targetgen.package_runtime import BackendDeclined


def test_region_invocation_keeps_input_and_execution_owner(tmp_path, monkeypatch):
    owner = ContextVar("test_region_owner", default=None)
    token = owner.set(object())
    original_owner = owner.get()
    package = object()
    interface = tmp_path / "selected.iface.mlir"
    interface.write_text("module {}")
    events = []

    def invoke(pkg, source, directory, **options):
        assert (pkg, source, directory) == (package, interface, tmp_path)
        assert options == {"contract": None, "timeout": 17}
        assert owner.get() is original_owner
        events.append("invoke")
        return {"commands": []}, "admitted artifact"

    monkeypatch.setattr(capsule_common, "lower_interface", invoke)
    try:
        artifact, receipt, elapsed = lower_region(
            package, interface, tmp_path, timeout=17, check_inputs=lambda: events.append("check")
        )
    finally:
        owner.reset(token)
    assert (artifact, receipt) == ("admitted artifact", None)
    assert elapsed >= 0
    assert events == ["check", "invoke", "check"]


@pytest.mark.parametrize("kind", ["target_configuration", "target_schedule"])
def test_region_target_receipt_is_checked_before_return(tmp_path, monkeypatch, kind):
    target = object()
    package = object()
    receipt = b'{"selected": true}\n'
    events = []

    def invoke(pkg, source, directory, **options):
        assert options[kind] is target
        (directory / (kind + ".json")).write_bytes(receipt)
        events.append("invoke")
        return {}, "artifact"

    def current():
        events.append("target")
        return True

    def validate(pkg, directory, selected, *, expected_receipt_sha256):
        assert (pkg, directory, selected) == (package, tmp_path, target)
        assert expected_receipt_sha256 == hashlib.sha256(receipt).hexdigest()
        events.append("receipt")

    monkeypatch.setattr(capsule_common, "lower_interface", invoke)
    artifact, digest, _ = lower_region(
        package, tmp_path / "input", tmp_path, timeout=1,
        check_inputs=lambda: events.append("input"), target_kind=kind, target=target,
        target_is_current=current, revalidate_target=validate,
    )
    assert artifact == "artifact"
    assert digest == hashlib.sha256(receipt).hexdigest()
    assert events == ["target", "input", "invoke", "input", "target", "receipt", "input"]


@pytest.mark.parametrize("changed_before", [False, True])
def test_changed_original_target_refuses_artifact(tmp_path, monkeypatch, changed_before):
    current = not changed_before
    invoked = []

    def invoke(*args, **kwargs):
        nonlocal current
        current = False
        invoked.append(True)
        return {}, "artifact that must not escape"

    monkeypatch.setattr(capsule_common, "lower_interface", invoke)
    with pytest.raises(ValueError, match="target configuration context changed"):
        lower_region(
            object(), tmp_path / "input", tmp_path, timeout=1, check_inputs=lambda: None,
            target_kind="target_schedule", target=object(), target_is_current=lambda: current,
            revalidate_target=lambda *args, **kwargs: pytest.fail("stale target reached receipt validation"),
        )
    assert bool(invoked) is not changed_before


def test_backend_decline_is_not_an_artifact(tmp_path, monkeypatch):
    failure = BackendDeclined("not supported")

    def invoke(*args, **kwargs):
        raise failure

    monkeypatch.setattr(capsule_common, "lower_interface", invoke)
    with pytest.raises(BackendDeclined) as caught:
        lower_region(object(), tmp_path / "input", tmp_path, timeout=1, check_inputs=lambda: None)
    assert caught.value is failure
