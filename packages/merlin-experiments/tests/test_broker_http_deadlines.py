"""Socket-free HTTP body deadlines preserve admitted action receipt joining.

The header socket timeout is idle-only, not an absolute slowloris shutdown bound.
"""

import io
import json
import socket
import subprocess
import threading
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import broker as B
from merlin_experiments.phase2 import broker_policy as BP
from merlin_experiments.phase2 import whole_model as WM


@pytest.fixture(autouse=True)
def no_processes_or_listeners(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("HTTP deadline tests cannot launch processes or bind sockets")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(socket.socket, "bind", refuse)


class Connection:
    def __init__(self):
        self.timeouts = []
        self.events = []

    def settimeout(self, value):
        self.timeouts.append(value)
        self.events.append("timeout")

    def makefile(self, *args):
        self.events.append("makefile")
        return io.BytesIO()


@pytest.fixture
def harness(tmp_path, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(B.time, "monotonic", lambda: now[0])
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    receipts = tmp_path / "control/receipts.jsonl"
    target = SimpleNamespace()
    state = SimpleNamespace(now=now, events=[], join=None, execute=lambda: {"static": "evidence"})

    def analyze(*args, **kwargs):
        return state.execute()

    action = B.BrokerAction(BP.E2E_ANALYSIS_ACTION, ("analyze",), (), "synthetic host analysis", True)
    workflow = BP.select_workflow(
        BP.WHOLE_MODEL_V1,
        candidate=candidate,
        target_experiment=target,
        receipt_path=receipts,
        global_experiment=SimpleNamespace(analysis=SimpleNamespace(analyze=analyze)),
        services=BP.BrokerServices(global_analysis_view=lambda document, **kwargs: document),
    )
    state.broker = B.Broker(
        SimpleNamespace(),
        target,
        candidate,
        (action,),
        receipts,
        deadline=130.0,
        workflow=workflow,
        max_calls=4,
        max_tool_seconds=10,
    )
    state.action = action

    class Server:
        def __init__(self, address, handler):
            assert address == ("127.0.0.1", 0)
            self.server_address = ("127.0.0.1", 1234)
            state.handler_type = handler

        def serve_forever(self):
            pass

        def shutdown(self):
            state.events.append("shutdown")

        def server_close(self):
            assert self.daemon_threads is False and self.block_on_close is True
            state.events.append("joining")
            if state.join:
                state.join()
            state.events.append("joined")

    monkeypatch.setattr(B, "ThreadingHTTPServer", Server)
    return state


def handler_for(harness, body, *, length=None):
    handler = object.__new__(harness.handler_type)
    handler.path = "/execute"
    handler.headers = {
        "X-Perf-Token": harness.broker.token,
        "Content-Length": str(len(body)) if length is None else length,
    }
    handler.connection = Connection()
    handler.rfile, handler.wfile = io.BytesIO(body), io.BytesIO()
    handler.status = []
    handler.send_response = handler.status.append
    handler.send_error = lambda code, *args: handler.status.append(code)
    handler.send_header = lambda *args: None
    handler.end_headers = lambda: None
    handler.close_connection = False
    return handler


def test_header_setup_installs_idle_timeout_before_reading(harness):
    with harness.broker.serving():
        handler = object.__new__(harness.handler_type)
        handler.request = Connection()
        handler.setup()
        assert handler.timeout == 5.0
        assert handler.request.timeouts == [5.0]
        assert handler.request.events[:2] == ["timeout", "makefile"]


@pytest.mark.parametrize("length", ["", "0", "-1", "1000001", "garbage", "2.0"])
def test_invalid_length_refuses_before_body_or_action(harness, length):
    class NoRead:
        def read1(self, size):
            pytest.fail("invalid length reached body read")

    with harness.broker.serving():
        handler = handler_for(harness, b"", length=length)
        handler.rfile = NoRead()
        handler.do_POST()
        assert handler.status == [400]
    assert harness.broker.calls == []
    assert harness.broker.receipt_path.read_bytes() == b""


@pytest.mark.parametrize("remaining", [30.0, 1.25])
def test_stalled_body_is_bounded_by_request_and_stage_deadline(harness, remaining):
    harness.broker.deadline = harness.now[0] + remaining
    with harness.broker.serving():
        handler = handler_for(harness, b"{}")

        class Stalled:
            def read1(self, size):
                assert 0 < size <= 65536
                timeout = handler.connection.timeouts[-1]
                assert timeout == min(5.0, remaining)
                harness.now[0] += timeout
                raise TimeoutError("synthetic stalled client")

        handler.rfile = Stalled()
        handler.do_POST()
        assert handler.status == [408]
        assert handler.close_connection is True
    assert harness.broker.calls == []
    assert harness.events == ["shutdown", "joining", "joined"]


def test_partial_body_does_not_reset_absolute_deadline(harness):
    with harness.broker.serving():
        handler = handler_for(harness, b"{}")

        class Trickle:
            count = 0

            def read1(self, size):
                self.count += 1
                if self.count == 1:
                    assert handler.connection.timeouts[-1] == 5.0
                    harness.now[0] += 3.0
                    return b"{"
                assert handler.connection.timeouts[-1] == 2.0
                harness.now[0] += 2.0
                raise TimeoutError("the second fragment missed the absolute deadline")

        handler.rfile = Trickle()
        handler.do_POST()
        assert handler.status == [408] and handler.close_connection
        assert handler.connection.timeouts == [5.0, 2.0]
    assert harness.broker.calls == []


def test_expired_stage_refuses_without_reading(harness):
    harness.broker.deadline = harness.now[0]

    class NoRead:
        def read1(self, size):
            pytest.fail("expired stage reached body read")

    with harness.broker.serving():
        handler = handler_for(harness, b"{}")
        handler.rfile = NoRead()
        handler.do_POST()
        assert handler.status == [408] and handler.close_connection
    assert harness.broker.calls == []


def test_final_fragment_arriving_after_deadline_is_not_admitted(harness):
    with harness.broker.serving():
        body = json.dumps({"action": harness.action.name}).encode()
        handler = handler_for(harness, body)

        class LateBody:
            def read1(self, size):
                harness.now[0] += 5.0
                return body

        handler.rfile = LateBody()
        handler.do_POST()
        assert handler.status == [408] and handler.close_connection
    assert harness.broker.calls == []


@pytest.mark.parametrize("body,length", [(b"{", "2"), (b"{}", "3"), (b"[]", "2"), (b"xx", "2")])
def test_truncated_or_invalid_document_never_allocates_action(harness, body, length):
    with harness.broker.serving():
        handler = handler_for(harness, body, length=length)
        handler.do_POST()
        assert handler.status == [400]
    assert harness.broker.calls == []


def test_timeout_response_disconnect_does_not_escape_handler(harness):
    attempted_writes = []

    class Stalled:
        def read1(self, size):
            raise TimeoutError("stalled")

    class Disconnected:
        def write(self, data):
            attempted_writes.append(data)
            raise BrokenPipeError("peer departed")

    with harness.broker.serving():
        handler = handler_for(harness, b"{}")
        handler.rfile, handler.wfile = Stalled(), Disconnected()

        def send_error(code):
            handler.status.append(code)
            handler.wfile.write(b"timeout response")

        handler.send_error = send_error
        handler.do_POST()
        assert handler.close_connection and handler.status == [408]
    assert harness.broker.calls == []
    assert attempted_writes == [b"timeout response"]


def test_completed_admitted_request_joins_before_receipt_sealing(harness):
    entered, release = threading.Event(), threading.Event()
    failures = []

    def analyze():
        entered.set()
        assert release.wait(2), "fake server failed to join admitted request"
        return {"static": "evidence"}

    harness.execute = analyze
    with harness.broker.serving():
        body = json.dumps({"action": harness.action.name}).encode()
        handler = handler_for(harness, body)

        def run():
            try:
                handler.do_POST()
            except BaseException as exc:
                failures.append(exc)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        assert entered.wait(2)
        assert harness.broker.calls[0]["state"] == "running"
        assert harness.broker.receipt_path.read_bytes() == b""

        def join():
            release.set()
            worker.join(2)
            assert not worker.is_alive()
            assert not failures
            assert handler.status == [200]
            assert json.loads(harness.broker.receipt_path.read_text())["state"] == "complete"

        harness.join = join
    assert harness.events == ["shutdown", "joining", "joined"]
    receipt = json.loads(harness.broker.receipt_path.read_text())
    audit = {
        "broker_invocations": [{"action": harness.action.name, "bindings_sha256": receipt["bindings_command_sha256"]}]
    }
    verified = WM.verify_global_broker_receipts(harness.broker.receipt_path, actions=(harness.action,), audit=audit)
    assert verified["all_required_succeeded"] is True
