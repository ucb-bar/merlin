"""Actual selected workflow execution/qualification, without listeners or scientific engines."""

import json
import time
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import broker as B
from merlin_experiments.phase2 import broker_policy as P
from merlin_experiments.phase2 import corpus_feedback as CF
from merlin_experiments.phase2 import whole_model as WM
from merlin_experiments.phase2.contracts import StageGateError


def inputs(tmp_path):
    return dict(
        candidate=tmp_path,
        target_experiment=SimpleNamespace(target="fixture"),
        receipt_path=tmp_path / "control" / "receipts.jsonl",
    )


@pytest.mark.parametrize("identity", [None, "", "target-name", "corpus-feedback-v2"])
def test_unknown_selection_has_no_fallback(tmp_path, identity):
    with pytest.raises(StageGateError, match="unknown broker workflow"):
        P.select_workflow(identity, **inputs(tmp_path))
    with pytest.raises(StageGateError, match="unknown broker workflow"):
        P.action_registry(identity, tmp_path, SimpleNamespace())
    assert not (tmp_path / "control").exists()


def test_selection_and_bound_paths_refuse_wrong_authority(tmp_path):
    with pytest.raises(StageGateError, match="disagrees"):
        P.select_workflow(P.WHOLE_MODEL_V1, **inputs(tmp_path))
    with pytest.raises(StageGateError, match="disagrees"):
        P.select_workflow(P.CORPUS_FEEDBACK_V1, **inputs(tmp_path), global_experiment=object())
    policy = P.select_workflow(P.CORPUS_FEEDBACK_V1, **inputs(tmp_path))
    with pytest.raises(AttributeError):
        policy.workflow_id = P.WHOLE_MODEL_V1
    with pytest.raises(StageGateError, match="another invocation"):
        B.Broker(
            SimpleNamespace(),
            SimpleNamespace(),
            tmp_path / "other",
            (),
            tmp_path / "receipts",
            workflow=policy,
            deadline=0,
            max_calls=1,
            max_tool_seconds=1,
        )


def test_interleaved_actual_policies_preserve_scientific_order_and_receipt_authority(tmp_path):
    calls = []
    action = B.BrokerAction(P.E2E_ANALYSIS_ACTION, ("host",), (), "analysis", True)

    def stage_analysis(*args, **kwargs):
        calls.append("stage-analysis")
        assert kwargs["target"] == "fixture"
        return {"stage": True}

    def global_analysis(*args, **kwargs):
        calls.append("global-analysis")
        return {"private": True}

    def project(document, **kwargs):
        calls.append("global-projection")
        assert document == {"private": True}
        return {"global": True}

    brokers = []
    for name, identity, kwargs in [
        (
            "stage",
            P.CORPUS_FEEDBACK_V1,
            dict(
                services=P.BrokerServices(whole_model_analysis=stage_analysis),
                functional_base=tmp_path,
                e2e_sentinel=object(),
            ),
        ),
        (
            "global",
            P.WHOLE_MODEL_V1,
            dict(
                services=P.BrokerServices(global_analysis_view=project),
                global_experiment=SimpleNamespace(analysis=SimpleNamespace(analyze=global_analysis)),
            ),
        ),
    ]:
        root = tmp_path / name
        root.mkdir()
        context = inputs(root)
        workflow = P.select_workflow(identity, **context, **kwargs)
        broker = B.Broker(
            SimpleNamespace(),
            context["target_experiment"],
            root,
            (action,),
            context["receipt_path"],
            workflow=workflow,
            deadline=time.monotonic() + 20,
            max_calls=3,
            max_tool_seconds=5,
        )
        brokers.append(broker)
    for broker in (brokers[0], brokers[1], brokers[0]):
        result = broker.execute({"action": action.name, "workflow_id": "candidate-cannot-select"})
        assert result["returncode"] == 0
        rows = [json.loads(line) for line in broker.receipt_path.read_text().splitlines()]
        assert all(row["workflow_id"] == broker.workflow.workflow_id for row in rows)
    assert calls == ["stage-analysis", "global-analysis", "global-projection", "stage-analysis"]
    global_broker = brokers[1]
    row = json.loads(global_broker.receipt_path.read_text())
    audit = {"broker_invocations": [dict(action=action.name, bindings_sha256=row["bindings_command_sha256"])]}
    qualified = global_broker.workflow.verify_receipts(global_broker.receipt_path, actions=(action,), audit=audit)
    assert qualified["workflow_policy"] == {"status": "bound", "id": P.WHOLE_MODEL_V1}
    with pytest.raises(StageGateError, match="workflow policy"):
        CF.verify_broker_receipts(global_broker.receipt_path, (action,), audit)
    row["workflow_id"] = P.CORPUS_FEEDBACK_V1
    global_broker.receipt_path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="workflow policy"):
        WM.verify_global_broker_receipts(global_broker.receipt_path, actions=(action,), audit=audit)
    del row["workflow_id"]
    historical = json.dumps(row).encode() + b"\n"
    global_broker.receipt_path.write_bytes(historical)
    qualified = WM.verify_global_broker_receipts(global_broker.receipt_path, actions=(action,), audit=audit)
    assert qualified["workflow_policy"] == {"status": "historical_unbound", "id": None}
    assert global_broker.receipt_path.read_bytes() == historical


@pytest.mark.parametrize(
    "label,expected",
    [
        (" a  b ", "a-b"),
        ("é英a☃b", "a-b"),
        ("A._--B", "a._--b"),
        ("/工具/", ""),
        ("a\x00\nb", "a-b"),
        ("x١٢z", "x-z"),
    ],
)
def test_probe_slug_preserves_historical_ascii_runs(label, expected):
    assert P._probe_slug(label) == expected
