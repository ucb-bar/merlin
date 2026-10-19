"""Original common-check transport controls; semantic verdicts are synthetic.

No Lean theorem, compiler author, numeric model, hardware or simulator executes.
"""

import hashlib
import json
import threading
import time
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.brokers import common_transforms as OWNER
from merlin_experiments.phase1.brokers import selfcheck as P1
from merlin_experiments.phase2 import broker_policy as BP
from merlin_experiments.phase2 import common_transforms as P2
from merlin_experiments.phase2 import functional_inputs as FI
from test_phase2_broker import make_broker

from merlin.transforms import client

BEFORE, AFTER = "actual before IR", "actual after IR"


def _response(before=BEFORE, after=AFTER):
    return {
        "status": "unsupported",
        "source_sha256": hashlib.sha256(before.encode()).hexdigest(),
        "target_sha256": hashlib.sha256(after.encode()).hexdigest(),
        "scope": "synthetic transport observation",
    }


@pytest.fixture
def channel(tmp_path, monkeypatch, request):
    ws = tmp_path / "workspace"
    ws.mkdir()
    capsules = tmp_path / "capsules"
    capsules.mkdir()
    manifest = tmp_path / "original.yaml"
    manifest.write_text("original operator selection")
    seen, failures = [], []

    class Owner:
        def __init__(self, **kwargs):
            assert kwargs["bundle_manifest"] == manifest
            self.profile = {"common_substrate": "merlin.common_compiler_substrate." + getattr(request, "param", "v3")}

        def check(self, request, **kwargs):
            assert set(request) == {"schema", "before_ir", "after_ir", "entry"}
            assert request["schema"] == client.CHECK_SCHEMA
            assert 0 < kwargs["timeout_s"] <= 5
            assert not kwargs["work_root"].is_relative_to(ws)
            seen.append((request, kwargs))
            return _response(request["before_ir"], request["after_ir"])

        def apply(self, *_args, **_kwargs):
            pytest.fail("common-check was dispatched as an apply request")

    monkeypatch.setattr(OWNER, "CommonTransformService", Owner)
    context = SimpleNamespace(repo=tmp_path, descriptor=tmp_path / "descriptor.yaml", readback_policy=None)

    def run():
        try:
            P1.main(
                ["--ws", str(ws), "--poll", "0.01", "--compiler-bundle-manifest", str(manifest)],
                context=context,
                capsules_root=capsules,
            )
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 3
    while not (ws / ".qa_channel/broker_health.json").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    yield SimpleNamespace(ws=ws, seen=seen)
    (ws / ".qa_channel/STOP").write_text("stop")
    thread.join(timeout=3)
    assert not thread.is_alive() and not failures


def test_original_phase1_channel_checks_candidate_ir_and_binds_request_lease(channel):
    result = client.check_transform(channel.ws, BEFORE, AFTER, entry="f", wait_seconds=5)
    assert result["status"] == "unsupported"
    request = json.loads(next((channel.ws / ".qa_channel").glob("req_*.json")).read_bytes())
    assert request["action"] == "common-check"
    assert result["selfcheck_request_id"] == request["request_id"]
    assert P1._requester_alive(request) is True
    assert request["requester"]["pid_namespace"] == P1._pid_namespace()
    assert channel.seen[0][0]["before_ir"] == BEFORE and channel.seen[0][0]["after_ir"] == AFTER


@pytest.mark.parametrize("fault", ["request", "source", "target"])
def test_channel_reply_cannot_substitute_a_different_request_or_ir(tmp_path, monkeypatch, fault):
    channel = tmp_path / ".qa_channel"
    channel.mkdir()

    def respond(_delay):
        request = json.loads(next(channel.glob("req_*.json")).read_bytes())
        response = {**_response(), "selfcheck_request_id": request["request_id"]}
        response[{"request": "selfcheck_request_id", "source": "source_sha256", "target": "target_sha256"}[fault]] = (
            "wrong"
        )
        (channel / f"resp_{request['request_id']}.json").write_text(json.dumps(response))
        (channel / f"done_{request['request_id']}").write_text("done")

    monkeypatch.setattr(client.time, "sleep", respond)
    with pytest.raises(RuntimeError, match="different"):
        client.check_transform(tmp_path, BEFORE, AFTER, entry="f", wait_seconds=1)


def test_channel_stop_and_timeout_do_not_become_proof(tmp_path, monkeypatch):
    channel = tmp_path / ".qa_channel"
    channel.mkdir()
    monkeypatch.setattr(client.time, "sleep", lambda _delay: (channel / "STOP").write_text("stop"))
    with pytest.raises(RuntimeError, match="stopped"):
        client.check_transform(tmp_path, BEFORE, AFTER, entry="f", wait_seconds=1)
    (channel / "STOP").unlink()
    ticks = iter((1, 1, 1, 1, 2_000_000_000))
    monkeypatch.setattr(client.time, "time_ns", lambda: next(ticks))
    result = client.check_transform(tmp_path, BEFORE, AFTER, entry="f", wait_seconds=1)
    assert result["status"] == "unknown" and "timed out" in result["reason"]


@pytest.mark.parametrize("key", ["semantics", "effects", "resource_constraints", "toolchain", "tactic", "registry"])
def test_client_has_no_candidate_authority_arguments(tmp_path, key):
    with pytest.raises(TypeError):
        client.check_transform(tmp_path, BEFORE, AFTER, entry="f", **{key: "author selection"})


@pytest.mark.parametrize("fault", [None, "request", "source", "target", "timeout", "refused"])
def test_phase2_declared_transport_binds_both_products_and_cleans_request(tmp_path, monkeypatch, fault):
    def run(argv, **kwargs):
        assert argv[:2] == ["declared-broker", "common-check"]
        path = tmp_path / argv[-1].removeprefix("request=")
        payload = json.loads(path.read_bytes())
        assert payload == {"schema": client.CHECK_SCHEMA, "before_ir": BEFORE, "after_ir": AFTER, "entry": "f"}
        response = {**_response(), "selfcheck_request_id": path.name}
        if fault in {"request", "source", "target"}:
            response[
                {"request": "selfcheck_request_id", "source": "source_sha256", "target": "target_sha256"}[fault]
            ] = "wrong"
        if fault == "timeout":
            raise client.subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return SimpleNamespace(returncode=125 if fault == "refused" else 0, stdout=json.dumps(response))

    monkeypatch.setattr(client.subprocess, "run", run)
    if fault in {"request", "source", "target"}:
        with pytest.raises(RuntimeError, match="different"):
            client.check_transform(tmp_path, BEFORE, AFTER, entry="f", transport_command=("declared-broker",))
    else:
        result = client.check_transform(tmp_path, BEFORE, AFTER, entry="f", transport_command=("declared-broker",))
        assert result["status"] == ("unknown" if fault in {"timeout", "refused"} else "unsupported")
    assert not list((tmp_path / "build").iterdir())


def test_phase2_fixed_provider_dispatch_and_receipt_are_the_original_broker(tmp_path):
    calls = []

    def check(path, timeout):
        calls.append((path, timeout))
        return _response()

    action = P2.check_action()
    broker = make_broker(tmp_path, actions=(action,), services=BP.BrokerServices(common_check=check))
    result = broker.execute({"action": action.name, "bindings": {"request": "check.json"}})
    assert result["returncode"] == 0 and json.loads(result["stdout"])["status"] == "unsupported"
    assert calls and calls[0][0] == "check.json"
    receipt = json.loads(broker.receipt_path.read_text())
    assert receipt["action"] == "common-check" and receipt["returncode"] == 0


@pytest.mark.parametrize("version", [None, "v1", "v2", "v3", "v4", "v5", "v6", "v7"])
def test_phase2_inherits_check_only_from_original_supported_owner(tmp_path, monkeypatch, version):
    selected_profile = None if version is None else {"common_substrate": "merlin.common_compiler_substrate." + version}
    selected = {
        "bundle": {"compiler_tool_profile": selected_profile},
        "workspace": tmp_path / "original_ws",
        "bundle_path": tmp_path / "original_manifest",
        "repo": tmp_path,
    }
    monkeypatch.setattr(FI, "_verify_private_functional_provenance", lambda _pin: selected)
    owners = []

    class Owner:
        def __init__(self, **kwargs):
            self.profile = selected_profile
            owners.append(kwargs)

        def apply(self, request, **kwargs):
            return {"status": "unchanged"}

        def check(self, request, **kwargs):
            return _response(request["before_ir"], request["after_ir"])

    monkeypatch.setattr(P2, "CommonTransformService", Owner)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    apply, check = P2.services(
        SimpleNamespace(host_provenance=object()), candidate=candidate, work_root=tmp_path / "host_work"
    )
    assert (check is not None) == (version in {"v3", "v4", "v5", "v6"})
    assert len(owners) == (0 if version is None else 1)
    if check is not None:
        path = candidate / "unique-request.json"
        path.write_text(
            json.dumps({"schema": client.CHECK_SCHEMA, "before_ir": BEFORE, "after_ir": AFTER, "entry": "f"})
        )
        result = check(path.name, 5)
        assert result["selfcheck_request_id"] == path.name
        linked = candidate / "linked.json"
        linked.symlink_to(path)
        with pytest.raises(BP.StageGateError, match="ordinary candidate-owned"):
            check(linked.name, 5)


@pytest.mark.parametrize("channel", ["v1", "v2"], indirect=True)
def test_original_phase1_dispatch_refuses_unadmitted_check_profiles(channel):
    result = client.check_transform(channel.ws, BEFORE, AFTER, entry="f", wait_seconds=5)
    assert result["status"] == "unavailable" and result["all_pass"] is False
    assert not channel.seen
    assert result["source_sha256"] == _response()["source_sha256"]
    assert result["target_sha256"] == _response()["target_sha256"]
