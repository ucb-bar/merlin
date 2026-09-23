"""Canonical broker contracts without a listener, compiler, provider or hardware."""

from __future__ import annotations

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import broker as B
from merlin_experiments.phase2 import broker_policy as BP
from merlin_experiments.phase2 import corpus_feedback as CF
from merlin_experiments.phase2 import whole_model as WM


def make_broker(tmp_path, *, actions=None, workflow_id=BP.CORPUS_FEEDBACK_V1, max_calls=3, **kwargs):
    candidate = tmp_path / "candidate"
    candidate.mkdir(exist_ok=True)
    return B.Broker(
        SimpleNamespace(argv=(), network="available_not_an_isolation_claim", clear_environment=True),
        SimpleNamespace(),
        candidate,
        actions or (B.BrokerAction(BP.INVENTORY_ACTION, ("inventory",), (), "inspect", False),),
        tmp_path / "control" / "receipts.jsonl",
        deadline=time.monotonic() + 30,
        max_calls=max_calls,
        max_tool_seconds=10,
        workflow=BP.select_workflow(
            workflow_id,
            candidate=candidate,
            target_experiment=SimpleNamespace(),
            receipt_path=tmp_path / "control" / "receipts.jsonl",
            **kwargs,
        ),
    )


def test_global_projection_is_explicit_and_receipted(tmp_path):
    seen = []

    def project(document, *, complete_evidence, context_provider_installed):
        seen.append((document, complete_evidence, context_provider_installed))
        return {"visible": "projection"}

    native = {"full": "scientific evidence"}
    experiment = SimpleNamespace(analysis=SimpleNamespace(analyze=lambda *_a, **_k: native))
    action = B.BrokerAction(BP.E2E_ANALYSIS_ACTION, ("analyze",), (), "full model", True)
    broker = make_broker(
        tmp_path,
        actions=(action,),
        workflow_id=BP.WHOLE_MODEL_V1,
        global_experiment=experiment,
        services=BP.BrokerServices(global_analysis_view=project),
    )
    result = broker.execute({"action": action.name})
    assert result["returncode"] == 0
    assert json.loads(result["stdout"]) == {"visible": "projection"}
    detail = broker.receipt_path.parent / "full_model_analysis_0000.json"
    assert json.loads(detail.read_text()) == native
    assert seen == [(native, "/perf-control/" + detail.name, False)]
    receipt = json.loads(broker.receipt_path.read_text())
    audit = {"broker_invocations": [{"action": action.name, "bindings_sha256": receipt["bindings_command_sha256"]}]}
    assert WM.verify_global_broker_receipts(broker.receipt_path, actions=(action,), audit=audit)[
        "all_required_succeeded"
    ]


def test_missing_projection_cannot_dynamically_import_native_controller(tmp_path, monkeypatch):
    import builtins

    original = builtins.__import__

    def forbid(name, *args, **kwargs):
        assert name not in {"perf_agent_stage", "run_global_perf_experiment"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid)
    action = B.BrokerAction(BP.E2E_ANALYSIS_ACTION, ("analyze",), (), "full model", True)
    broker = make_broker(
        tmp_path,
        actions=(action,),
        workflow_id=BP.WHOLE_MODEL_V1,
        global_experiment=SimpleNamespace(analysis=SimpleNamespace(analyze=lambda *_a, **_k: {})),
    )
    assert broker.execute({"action": action.name})["returncode"] == 125
    assert json.loads(broker.receipt_path.read_text())["state"] == "complete"


def test_concurrent_admission_never_exceeds_locked_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(CF, "inspect_compiler_package", lambda _: SimpleNamespace(to_dict=lambda: {"ok": True}))
    broker = make_broker(tmp_path, max_calls=2)

    def invoke(_):
        try:
            return broker.execute({"action": BP.INVENTORY_ACTION})["returncode"]
        except B.StageGateError as exc:
            assert "budget is exhausted" in str(exc)
            return "refused"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(invoke, range(12)))
    assert results.count(0) == 2
    assert results.count("refused") == 10
    rows = [json.loads(line) for line in broker.receipt_path.read_text().splitlines()]
    assert sorted(row["index"] for row in rows) == [0, 1]
    assert all(row["state"] == "complete" for row in rows)


@pytest.mark.parametrize("binding", ["../secret", "/secret", "x" * 8193, "bad\0path"])
def test_path_and_request_refusals_leave_terminal_receipts(tmp_path, binding):
    action = B.BrokerAction("compile", ("compiler", "{output}"), ("output",), "compile", False)
    broker = make_broker(tmp_path, actions=(action,))
    with pytest.raises(B.StageGateError):
        broker.execute({"action": "compile", "bindings": {"output": binding}})
    row = json.loads(broker.receipt_path.read_text())
    assert row["state"] == "rejected" and row["returncode"] != 0


@pytest.mark.parametrize(
    "token,length,body,status",
    [
        ("wrong", "2", b"{}", 403),
        ("correct", "1000001", b"", 400),
        ("correct", "2", b"[]", 400),
        ("correct", "2", b"{}", 200),
    ],
)
def test_http_contract_with_fake_server_no_socket(tmp_path, monkeypatch, token, length, body, status):
    broker = make_broker(tmp_path)
    events = []

    class FakeServer:
        def __init__(self, address, handler):
            assert address == ("127.0.0.1", 0)
            self.server_address = ("127.0.0.1", 1234)
            self.handler = handler

        def serve_forever(self):
            pass

        def shutdown(self):
            events.append("shutdown")

        def server_close(self):
            assert not self.daemon_threads and self.block_on_close
            events.append("joined")

    monkeypatch.setattr(B, "ThreadingHTTPServer", FakeServer)
    monkeypatch.setattr(broker, "execute", lambda request: {"returncode": 0})
    with broker.serving():
        handler = object.__new__(broker._server.handler)
        handler.connection = SimpleNamespace(settimeout=lambda _: None)
        handler.path = "/execute"
        handler.headers = {"Content-Length": length, "X-Perf-Token": broker.token if token == "correct" else token}
        handler.rfile = io.BytesIO(body)
        handler.wfile = io.BytesIO()
        observed = []
        handler.send_error = observed.append
        handler.send_response = observed.append
        handler.send_header = lambda *_: None
        handler.end_headers = lambda: None
        handler.do_POST()
        assert observed == [status]
    assert events == ["shutdown", "joined"]


def test_action_registry_and_shim_are_owned_together(tmp_path):
    root = tmp_path / "candidate"
    action = B.BrokerAction("parse", (str(root / "compiler"), "{input}"), ("input",), "parse", True)
    sealed = B.action_registry_contract((action,), root)
    assert B.actions_from_registry_contract(sealed, root) == (action,)
    control = tmp_path / "control"
    shim = B.stage_broker_shim(
        control, host="127.0.0.1", port=1234, token="private", tool_timeout_s=5, actions=(action,)
    )
    assert shim.read_text() == B._BROKER_SHIM
    assert json.loads((control / ".perf_broker.json").read_text())["actions"] == ["parse"]
    with pytest.raises(B.StageGateError, match="not fresh"):
        B.stage_broker_shim(control, host="127.0.0.1", port=1234, token="private", tool_timeout_s=5, actions=(action,))
