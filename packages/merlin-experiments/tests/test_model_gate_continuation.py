import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import model_gate_vouchers as vouchers

from merlin.targetgen import capsule_runner as runner
from merlin.targetgen.model_gate_continuation import ModelGateContinuation


def capsule(tmp_path):
    root = tmp_path / "cap"
    root.mkdir()
    (root / "capsule.yaml").write_text("name: model\nkind: model\ngate: {after_op_pass_fraction: 0.8}\n")
    return {"name": "model", "kind": "model", "__dir__": str(root), "gate": {"after_op_pass_fraction": 0.8}}


def test_model_only_continuation_schedules_new_native_check_but_flags_never_admit(tmp_path):
    cap = capsule(tmp_path)
    calls = []

    def run(c, workers):
        calls.append(c["name"])
        return {"capsule": c["name"], "kind": "model", "status": "incomplete"}

    kwargs = dict(max_workers=1, target="gemmini", oracle_adapters={"L2": object(), "L3": object()})
    ordinary = runner._run_streaming_model_gate([], [cap], run, **kwargs)
    assert not calls and ordinary[0]["status"] == "gated"
    token = ModelGateContinuation(
        "model",
        hashlib.sha256((Path(cap["__dir__"]) / "capsule.yaml").read_bytes()).hexdigest(),
        "gemmini",
        "a" * 64,
        "b" * 64,
    )
    results = runner._run_streaming_model_gate([], [cap], run, model_gate_continuations=(token,), **kwargs)
    assert calls == ["model"] and results[0]["status"] == "incomplete"  # Scheduling grants no pass.
    with pytest.raises(ValueError, match="operator-verified"):
        runner._run_streaming_model_gate([], [cap], run, model_gate_continuations=({"promoted": True},), **kwargs)


def test_private_voucher_requires_closed_l2_gate_and_exact_source_context(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(
        vouchers,
        "_context",
        lambda context, cert, contract=None: {"descriptor_sha256": "c" * 64, "instrument": "judge"},
    )
    monkeypatch.setattr(vouchers.promotion, "_submission_digest", lambda source: "a" * 64)
    from merlin.targetgen import target_experiment

    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda path: SimpleNamespace(target="gemmini"))
    cap = capsule(tmp_path)
    ws = tmp_path / "workspace"
    ws.mkdir()
    context = SimpleNamespace(descriptor="unused")
    record = {
        "capsule": "model",
        "kind": "model",
        "model_gate_admission": {"passed": True, "stage": "L2", "fraction": 1.0},
        "model_execution_check": {
            "kind": "owned_host_model_execution",
            "status": "incomplete",
            "candidate_required_tiers": {"tiers": {"L2": {"status": "pass"}}},
            "candidate_source_placement": {"status": "clean"},
            "candidate_completed_dispatch": {"status": "verified"},
            "candidate_source_coverage": {"status": "verified"},
        },
    }
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(record))
    path = vouchers.issue(ws, ws, record, capsule=cap, context=context, cert_tier="L3", receipt_path=receipt)
    assert not path.is_relative_to(ws)
    token = vouchers.verify(path, source_digest="a" * 64, capsules={"model"}, context=context, cert_tier="L3")
    assert token.capsule == "model"
    for digest, names in (("d" * 64, {"model"}), ("a" * 64, {"other"})):
        with pytest.raises(ValueError):
            vouchers.verify(path, source_digest=digest, capsules=names, context=context, cert_tier="L3")
    record["model_execution_check"]["candidate_required_tiers"]["tiers"]["L2"]["status"] = "unverified"
    receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="closed L2"):
        vouchers.issue(ws, ws, record, capsule=cap, context=context, cert_tier="L3", receipt_path=receipt)
