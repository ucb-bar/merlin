"""Covered transformations cannot select or manufacture their verification authority."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.brokers.common_transforms import CommonTransformService

from merlin.common.paths import repo_root

SOURCE = """builtin.module {
  func.func @f(%x: i8) -> i8 {
    %zero = arith.constant 0 : i8
    %result = arith.addi %x, %zero : i8
    func.return %result : i8
  }
}"""


def _service():
    service = object.__new__(CommonTransformService)
    service.profile = {"host_index_bits": 32, "lean_toolchain": {"selection": "not_selected"}, "sha256": "a" * 64}
    service._verify = lambda: None
    return service


def _request(**options):
    return {"schema": "merlin.common_transform_request.v1", "source_ir": SOURCE, "options": options}


@pytest.mark.parametrize("key", ["semantics", "coverage", "toolchain", "verifier", "proof_result"])
def test_requests_cannot_supply_proof_authority(tmp_path, key):
    request = _request()
    request[key] = {"status": "verified"}
    with pytest.raises(ValueError, match="cannot select semantics"):
        _service().apply(request, work_root=tmp_path)


def test_index_width_is_owned_by_frozen_host_policy(tmp_path):
    with pytest.raises(ValueError, match="frozen host ABI index width"):
        _service().apply(_request(index_bits=64), work_root=tmp_path)
    service = _service()
    service.profile["host_index_bits"] = None
    with pytest.raises(ValueError, match="independently reviewed"):
        service.apply(_request(), work_root=tmp_path)


def test_missing_formal_toolchain_cannot_turn_into_a_success(tmp_path):
    with pytest.raises(FileNotFoundError, match="explicitly selected"):
        _service().apply(_request(), work_root=tmp_path)


@pytest.mark.parametrize(
    "outcome",
    [
        None,
        True,
        SimpleNamespace(status="refuted", reason="not a proof"),
        SimpleNamespace(verified=True, reason="candidate assertion", status="verified"),
    ],
)
def test_candidate_constructed_verified_receipt_cannot_discharge_a_change(tmp_path, monkeypatch, outcome):
    from merlin.verify import lean_checker, lean_contract

    service = _service()
    toolchain = lean_contract.LeanToolchain(
        tmp_path / "absent",
        "a" * 64,
        (tmp_path,),
        ((tmp_path / "absent", "a" * 64),),
        tmp_path,
    )
    service._toolchain = lambda _work: toolchain
    monkeypatch.setattr(
        lean_checker,
        "verify_pipeline_change",
        lambda _change, **_kwargs: outcome,
    )
    with pytest.raises(ValueError, match="not established by Lean"):
        service.apply(_request(reuse_buffers=False), work_root=tmp_path)


@pytest.mark.parametrize("fault", [None, "status", "domain", "scope", "semantics_sha256", "toolchain_sha256"])
def test_host_callback_keeps_fixed_scope_model_toolchain_and_returns_completion(tmp_path, monkeypatch, fault):
    """Mock only host completion; exercise the real acceptance gate and pipeline."""
    from dataclasses import replace

    from merlin.verify import lean_checker, lean_contract

    service = _service()
    toolchain = lean_contract.LeanToolchain(
        tmp_path / "absent",
        "a" * 64,
        (tmp_path,),
        ((tmp_path / "absent", "a" * 64),),
        tmp_path,
    )
    service._toolchain = lambda _work: toolchain

    def complete(change, **kwargs):
        assert kwargs["toolchain"] is toolchain
        assert isinstance(kwargs["deadline"], float)
        assert toolchain.timeout_s == 120
        receipt = lean_contract.LeanTransformReceipt(
            lean_contract.SCHEMA,
            "lean_kernel",
            lean_contract.DOMAINS[0],
            lean_contract.digest(change.before_ir),
            lean_contract.digest(change.after_ir),
            lean_checker.pipeline_semantic_identity(),
            "verified",
            tuple(change.certificate.scope),
            (),
            toolchain_sha256=toolchain.identity(),
        )
        if fault is not None:
            value = "refuted" if fault == "status" else ("wrong_scope",) if fault == "scope" else "stale"
            receipt = replace(receipt, **{fault: value})
        # A narrowly mocked private host completion tests metadata gates without
        # claiming any external Lean process was run by this control.
        monkeypatch.setitem(lean_checker._COMPLETED, id(receipt), receipt)
        return receipt

    monkeypatch.setattr(lean_checker, "verify_pipeline_change", complete)
    if fault is None:
        result = service.apply(_request(reuse_buffers=False), work_root=tmp_path, timeout_s=4)
        assert result["status"] == "verified" and result["receipts"]
    else:
        with pytest.raises(ValueError, match="not established by Lean"):
            service.apply(_request(reuse_buffers=False), work_root=tmp_path, timeout_s=4)


def _client():
    # This public client has no private imports and can be staged independently.
    source = repo_root() / "src/merlin/transforms/client.py"
    spec = importlib.util.spec_from_file_location("_independent_common_client", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase2_client_uses_declared_broker_and_removes_temporary_request(tmp_path, monkeypatch):
    client = _client()
    calls = []

    def execute(argv, **kwargs):
        calls.append(argv)
        request = tmp_path / argv[-1].removeprefix("request=")
        document = json.loads(request.read_bytes())
        assert set(document) == {"schema", "source_ir", "options"}
        response = {"status": "verified", "source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest()}
        return SimpleNamespace(returncode=0, stdout=json.dumps({"stdout": json.dumps(response)}))

    monkeypatch.setattr(client.subprocess, "run", execute)
    result = client.apply_transform(
        tmp_path,
        SOURCE,
        {},
        transport_command=("declared-python", "declared-broker"),
    )
    assert result["status"] == "verified"
    assert calls[0][:3] == ["declared-python", "declared-broker", "common-transform"]
    assert not list((tmp_path / "build").iterdir())


def test_phase2_client_rejects_response_for_different_ir(tmp_path, monkeypatch):
    client = _client()
    monkeypatch.setattr(
        client.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"status": "verified", "source_sha256": "b" * 64}),
        ),
    )
    with pytest.raises(RuntimeError, match="different input IR"):
        client.apply_transform(tmp_path, SOURCE, {}, transport_command=("declared-broker",))
    assert not list((tmp_path / "build").iterdir())


def test_inherited_phase2_action_is_an_independent_optional_formal_check():
    from merlin_experiments.phase2.common_transforms import action

    declaration = action()
    assert declaration.name == "common-transform"
    assert declaration.placeholders == ("request",)
    assert declaration.required is False
    assert "no simulator or performance claim" in declaration.purpose
