"""Neutral host transport controls, not compiler or target qualification."""

import json
import os
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import authoring
from merlin_experiments.phase2 import broker as B
from merlin_experiments.phase2 import broker_file_channel as F
from merlin_experiments.phase2 import broker_policy as BP
from merlin_experiments.phase2 import corpus_feedback as CF
from test_phase2_broker import make_broker


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(CF, "inspect_compiler_package", lambda _: SimpleNamespace(to_dict=lambda: {"owners": []}))
    return make_broker(tmp_path)


def submit(channel, selected_token, *, file_id="a" * 32, **changes):
    document = {
        "schema": F.PROTOCOL,
        "request_id": file_id,
        "token": selected_token,
        "request": {"action": BP.INVENTORY_ACTION, "bindings": {}, "timeout_s": 5},
    }
    document.update(changes)
    path = Path(channel["request_dir"]) / ("req_" + file_id + ".json")
    path.write_text(json.dumps(document))
    return path


def reply(case, request_id="a" * 32):
    path = case.receipt_path.parent / F.REPLIES / ("resp_" + request_id + ".json")
    end = time.monotonic() + 3
    while not path.exists() and time.monotonic() < end:
        time.sleep(0.01)
    return json.loads(path.read_bytes())


def test_original_execute_receipt_and_replay_are_preserved(case):
    with case.serving_files(workspace=case.candidate.parent, control_dir=case.receipt_path.parent) as channel:
        B.stage_broker_shim(
            case.receipt_path.parent,
            host="",
            port=0,
            token=case.token,
            tool_timeout_s=5,
            actions=tuple(case.actions.values()),
            file_channel=channel,
        )
        request = submit(channel, case.token)
        response = reply(case)
        assert response["returncode"] == 0 and json.loads(response["stdout"]) == {"owners": []}
        request.unlink()
        submit(channel, case.token, request={"action": "changed-replay", "bindings": {}})
        time.sleep(0.06)
    receipts = [json.loads(line) for line in case.receipt_path.read_text().splitlines()]
    assert len(case.calls) == len(receipts) == 1
    assert receipts[0]["state"] == "complete" and receipts[0]["returncode"] == 0
    CF.verify_broker_action_receipts(
        case.receipt_path,
        tuple(case.actions.values()),
        {
            "broker_invocations": [
                {"action": BP.INVENTORY_ACTION, "bindings_sha256": receipts[0]["bindings_command_sha256"]}
            ]
        },
    )


@pytest.mark.parametrize("transport", ["tcp", "files.v1"])
def test_original_author_channel_stages_selected_shim(case, transport):
    with case.serving_author(
        transport=transport, workspace=case.candidate.parent, control_dir=case.receipt_path.parent,
        tool_timeout_s=5, actions=tuple(case.actions.values()),
    ) as record:
        config = json.loads((case.receipt_path.parent / ".perf_broker.json").read_bytes())
        assert (case.receipt_path.parent / "perf_tool.py").is_file()
        if transport == "files.v1":
            assert config["file_channel"]["schema"] == record["schema"] == F.PROTOCOL
            assert record["max_concurrent"] == 1
            assert "request_dir" not in record and "reply_dir" not in record
        else:
            assert record is None and "file_channel" not in config
            assert config["url"].startswith("http://127.0.0.1:")


@pytest.mark.parametrize("fault", ["wrong_token", "unicode_token", "extra_field", "wrong_schema", "wrong_id"])
def test_closed_authenticated_envelope_refuses_before_execute(case, fault):
    with case.serving_files(workspace=case.candidate.parent, control_dir=case.receipt_path.parent) as channel:
        change = {
            "wrong_token": {"token": "wrong"},
            "unicode_token": {"token": "é"},
            "extra_field": {"arbitrary": True},
            "wrong_schema": {"schema": "other"},
            "wrong_id": {"request_id": "b" * 32},
        }[fault]
        submit(channel, case.token, **change)
        assert reply(case)["returncode"] == 125
    assert not case.calls and case.receipt_path.read_bytes() == b""


@pytest.mark.parametrize("special", ["fifo", "socket", "symlink", "hardlink"])
def test_nonregular_or_aliased_request_cannot_block_or_dispatch(case, special):
    with case.serving_files(workspace=case.candidate.parent, control_dir=case.receipt_path.parent) as channel:
        path = Path(channel["request_dir"]) / ("req_" + "a" * 32 + ".json")
        sock = None
        if special == "fifo":
            os.mkfifo(path)
        elif special == "socket":
            # A short pathname is unnecessary: bind through the existing directory FD.
            fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                sock = socket.socket(socket.AF_UNIX)
                sock.bind("/proc/self/fd/" + str(fd) + "/" + path.name)
            finally:
                os.close(fd)
        else:
            other = path.parent / "other"
            other.write_text("{}")
            path.symlink_to(other) if special == "symlink" else os.link(other, path)
        try:
            assert reply(case)["returncode"] == 125
        finally:
            if sock:
                sock.close()
    assert not case.calls


def test_fifo_swap_between_stat_and_original_read_refuses_without_blocking(case, monkeypatch):
    original = F.IO._read

    def changed(descriptor, name, limit):
        os.unlink(name, dir_fd=descriptor)
        os.mkfifo(name, dir_fd=descriptor)
        return original(descriptor, name, limit)

    monkeypatch.setattr(F.IO, "_read", changed)
    with case.serving_files(workspace=case.candidate.parent, control_dir=case.receipt_path.parent) as channel:
        submit(channel, case.token)
        assert reply(case)["returncode"] == 125
    assert not case.calls


def test_request_replacement_after_read_refuses(case, monkeypatch):
    original = F.IO._read

    def changed(descriptor, name, limit):
        document = original(descriptor, name, limit)
        os.unlink(name, dir_fd=descriptor)
        F.IO._write(descriptor, name, document)
        return document

    monkeypatch.setattr(F.IO, "_read", changed)
    with case.serving_files(workspace=case.candidate.parent, control_dir=case.receipt_path.parent) as channel:
        submit(channel, case.token)
        assert reply(case)["returncode"] == 125
    assert not case.calls


def test_oversized_reply_is_transport_refusal_after_original_receipt(case, monkeypatch):
    monkeypatch.setattr(
        CF, "inspect_compiler_package", lambda _: SimpleNamespace(to_dict=lambda: {"owners": ["x" * F.MAX_REPLY_BYTES]})
    )
    with case.serving_files(workspace=case.candidate.parent, control_dir=case.receipt_path.parent) as channel:
        submit(channel, case.token)
        response = reply(case)
        assert response["returncode"] == 125 and "receipt retained" in response["stderr"]
    assert json.loads(case.receipt_path.read_text())["state"] == "complete"


def test_directory_replacement_refuses_at_final_boundary(case):
    with pytest.raises(B.StageGateError, match="file broker transport failed"):
        with case.serving_files(workspace=case.candidate.parent, control_dir=case.receipt_path.parent) as channel:
            path = Path(channel["request_dir"])
            path.rename(path.with_name("old-inbox"))
            path.mkdir()
    assert not case.calls


def test_original_action_budget_and_pool_close_are_preserved(case):
    closed = []
    case.max_calls = 120
    case.workflow.evaluation_pool = SimpleNamespace(close=lambda: closed.append(True))
    with case.serving_files(workspace=case.candidate.parent, control_dir=case.receipt_path.parent) as channel:
        assert channel["max_requests"] == 120 + F.BAD_REQUEST_ALLOWANCE
    assert closed == [True]


def test_inflight_dispatch_joins_before_original_pool_close(case, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    order = []
    original = case.execute

    def delayed(request):
        entered.set()
        assert release.wait(2)
        result = original(request)
        order.append("dispatch-complete")
        return result

    monkeypatch.setattr(case, "execute", delayed)
    case.workflow.evaluation_pool = SimpleNamespace(close=lambda: order.append("pool-closed"))
    with case.serving_files(workspace=case.candidate.parent, control_dir=case.receipt_path.parent) as channel:
        submit(channel, case.token)
        assert entered.wait(2)
        release.set()
    assert order == ["dispatch-complete", "pool-closed"]
    assert json.loads(case.receipt_path.read_text())["state"] == "complete"
    assert not any(t.name.startswith("original-file-broker") for t in threading.enumerate())


def test_delayed_first_call_keeps_original_verifier_order(case, monkeypatch):
    count = []

    def inspect(_candidate):
        index = len(count)
        count.append(index)
        if index == 0:
            time.sleep(0.08)
        return SimpleNamespace(to_dict=lambda: {"owners": [index]})

    monkeypatch.setattr(CF, "inspect_compiler_package", inspect)
    with case.serving_files(workspace=case.candidate.parent, control_dir=case.receipt_path.parent) as channel:
        assert channel["max_concurrent"] == 1
        submit(channel, case.token, file_id="f" * 32)
        time.sleep(0.002)
        submit(channel, case.token, file_id="a" * 32)
        assert json.loads(reply(case, "f" * 32)["stdout"]) == {"owners": [0]}
        assert json.loads(reply(case, "a" * 32)["stdout"]) == {"owners": [1]}
    rows = [json.loads(line) for line in case.receipt_path.read_text().splitlines()]
    assert [row["index"] for row in rows] == [0, 1]
    CF.verify_broker_action_receipts(
        case.receipt_path,
        tuple(case.actions.values()),
        {
            "broker_invocations": [
                {"action": BP.INVENTORY_ACTION, "bindings_sha256": row["bindings_command_sha256"]} for row in rows
            ]
        },
    )


@pytest.mark.parametrize(
    "rows,expected",
    [
        ([], False),
        ([{"feedback_successes": 0}], False),
        ([{"feedback_successes": 1, "feedback_receipts": ["verified"]}, {"refusal": "later"}], True),
    ],
)
def test_stage_feedback_bool_uses_joined_original_feedback(rows, expected):
    assert authoring._development_feedback_performed(rows) is expected
