"""Agent-accessible selfchecks cannot escape an experiment-wide RTL-engine pin."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path

from merlin.common.paths import merlin_dir


HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


def _module(name: str):
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location(f"required_engine_{name}", HARNESS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_selfcheck_refuses_verilator_and_builds_gsim_adapter(monkeypatch):
    selfcheck = _module("agent_selfcheck")
    selected = []
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "gsim")
    monkeypatch.setattr(
        selfcheck.CR, "_spike_verilator_adapter",
        lambda engine, target: selected.append((engine, target)) or object(),
    )

    try:
        selfcheck._adapters("verilator", "gemmini", "chipyard")
    except ValueError as exc:
        assert "conflicts with MERLIN_REQUIRED_RTL_ENGINE='gsim'" in str(exc)
    else:
        raise AssertionError("a Verilator request escaped the GSIM pin")
    assert selected == [], "a refused request must not construct any simulator adapter"

    adapters, sim = selfcheck._adapters("gsim", "gemmini", "chipyard")
    assert sim == "gsim" and set(adapters) == {"L2", "L3"}
    assert selected == [("spike", "gemmini"), ("gsim", "gemmini")]
    assert selfcheck._default_sim() == "gsim"


def test_required_vcs_never_constructs_the_legacy_verilator_rung(monkeypatch):
    """The general engine pin is fail-closed even on VCS's historical L4 ladder."""
    selfcheck = _module("agent_selfcheck")
    selected = []
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "vcs")
    monkeypatch.setattr(
        selfcheck.CR, "_spike_verilator_adapter",
        lambda engine, target: selected.append((engine, target)) or object(),
    )

    try:
        selfcheck._adapters("vcs", "gemmini", "chipyard")
    except ValueError:
        pass  # An unavailable required VCS engine is the expected fail-closed result.
    assert selected == [("spike", "gemmini")]


def test_both_brokers_narrow_rtl_requests_to_gsim_but_keep_spike(monkeypatch):
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "gsim")
    sync = _module("selfcheck_broker")
    async_broker = _module("simjob_broker")

    assert sync._allowed_sims() == ("spike", "gsim")
    assert sync._default_sim() == "gsim"
    assert async_broker._allowed_sims() == ("spike", "gsim")
    assert "conflicts" in sync._sim_policy_error("verilator")


def test_gsim_pin_remaps_the_historical_l3_promotion_binding(monkeypatch):
    """The broker pin must not strand L3 behind the manifest's old Verilator label."""
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "gsim")
    promotion = _module("tier_promote")

    assert promotion.cert_sim("L3") == "gsim"


def _wait(path: Path, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not path.exists():
        time.sleep(0.01)
    assert path.exists(), f"timed out waiting for {path}"


def test_sync_broker_rejects_verilator_without_launching_a_child(tmp_path, monkeypatch):
    broker = _module("selfcheck_broker")
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "gsim")
    monkeypatch.setattr(
        broker.subprocess, "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("child must not launch")),
    )
    ws = tmp_path / "ws"
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True)
    (ch / "req_bad.json").write_text(json.dumps({
        "sim": "verilator", "capsules": "all", "timeout": 30,
    }))
    thread = threading.Thread(target=broker.main, args=(["--ws", str(ws), "--poll", "0.01"],))
    thread.start()
    try:
        _wait(ch / "done_bad")
        response = json.loads((ch / "resp_bad.json").read_text())
        assert response["all_pass"] is False
        assert response["required_rtl_engine"] == "gsim"
        assert "conflicts" in response["error"]
    finally:
        (ch / "STOP").write_text("stop")
        thread.join(timeout=3)
    assert not thread.is_alive()


def test_sync_broker_forwards_gsim_to_the_real_selfcheck_boundary(tmp_path, monkeypatch):
    broker = _module("selfcheck_broker")
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "gsim")
    ws = tmp_path / "ws"
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True)
    record = tmp_path / "forwarded.json"
    fake = tmp_path / "fake_selfcheck.py"
    fake.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "args=sys.argv[1:]\n"
        "Path(os.environ['FORWARD_RECORD']).write_text(json.dumps(args))\n"
        "out=args[args.index('--out')+1]\n"
        "Path(out).write_text(json.dumps({'all_pass': True, 'sim': args[args.index('--sim')+1]}))\n"
    )
    monkeypatch.setenv("FORWARD_RECORD", str(record))
    monkeypatch.setattr(broker, "SELFCHECK", fake)
    (ch / "req_good.json").write_text(json.dumps({
        "sim": "gsim", "capsules": "all", "workers": 1, "timeout": 30,
    }))
    thread = threading.Thread(target=broker.main, args=(["--ws", str(ws), "--poll", "0.01"],))
    thread.start()
    try:
        _wait(ch / "done_good")
        response = json.loads((ch / "resp_good.json").read_text())
        forwarded = json.loads(record.read_text())
        assert response == {"all_pass": True, "sim": "gsim"}
        assert forwarded[forwarded.index("--sim") + 1] == "gsim"
    finally:
        (ch / "STOP").write_text("stop")
        thread.join(timeout=3)
    assert not thread.is_alive()


def test_sandbox_shim_exposes_gsim_and_defaults_to_the_pin(monkeypatch):
    shim = _module("selfcheck_shim")
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "gsim")
    assert "gsim" in shim._SIMS
    assert shim._default_sim() == "gsim"
    assert shim._sim_policy_error("gsim") is None
    assert "conflicts" in shim._sim_policy_error("verilator")
