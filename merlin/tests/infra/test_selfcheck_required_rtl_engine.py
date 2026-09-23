"""Agent-accessible selfchecks cannot escape an experiment-wide RTL-engine pin."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest
from phase1_feedback import feedback_context, feedback_source

from merlin.common.paths import merlin_dir, module_source_path

pytestmark = pytest.mark.target("gemmini")

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


def _module(name: str):
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    source = (
        module_source_path("merlin_experiments.phase1.tools.selfcheck")
        if name == "selfcheck_shim"
        else feedback_source(name, HARNESS / f"{name}.py")
    )
    spec = importlib.util.spec_from_file_location(f"required_engine_{name}", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_selfcheck_refuses_verilator_and_builds_gsim_adapter(monkeypatch):
    selfcheck = _module("agent_selfcheck")
    selected = []
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "gsim")
    monkeypatch.setattr(
        selfcheck.CR,
        "simulator_adapter",
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
        selfcheck.CR,
        "simulator_adapter",
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
    assert async_broker._allowed_sims(
        feedback_context(merlin_dir() / "experiments/capsule_bench/targets/gemmini", target="gemmini")
    ) == ("spike", "gsim")
    assert "conflicts" in sync._sim_policy_error("verilator")


def test_gsim_pin_remaps_the_historical_l3_promotion_binding(monkeypatch):
    """The broker pin must not strand L3 behind the manifest's old Verilator label."""
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "gsim")
    promotion = _module("tier_promote")

    assert (
        promotion.cert_sim(
            "L3", context=feedback_context(merlin_dir() / "experiments/capsule_bench/targets/gemmini", target="gemmini")
        )
        == "gsim"
    )


def _wait(path: Path, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not path.exists():
        time.sleep(0.01)
    assert path.exists(), f"timed out waiting for {path}"


def test_sync_broker_rejects_verilator_without_launching_a_child(tmp_path, monkeypatch):
    broker = _module("selfcheck_broker")
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "gsim")
    monkeypatch.setattr(
        broker.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("child must not launch")),
    )
    ws = tmp_path / "ws"
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True)
    (ch / "req_bad.json").write_text(
        json.dumps(
            {
                "sim": "verilator",
                "capsules": "all",
                "timeout": 30,
            }
        )
    )
    thread = threading.Thread(
        target=broker.main,
        args=(["--ws", str(ws), "--poll", "0.01"],),
        kwargs={"context": feedback_context(tmp_path), "capsules_root": tmp_path},
    )
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
    monkeypatch.setattr(broker, "worker_command", lambda *args: [sys.executable, str(fake)])
    (ch / "req_good.json").write_text(
        json.dumps(
            {
                "sim": "gsim",
                "capsules": "all",
                "workers": 1,
                "timeout": 30,
            }
        )
    )
    thread = threading.Thread(
        target=broker.main,
        args=(["--ws", str(ws), "--poll", "0.01"],),
        kwargs={"context": feedback_context(tmp_path), "capsules_root": tmp_path},
    )
    thread.start()
    try:
        _wait(ch / "done_good")
        response = json.loads((ch / "resp_good.json").read_text())
        forwarded = json.loads(record.read_text())
        assert response["all_pass"] is True
        assert response["sim"] == "gsim"
        assert response["selfcheck_protocol"] == broker.PROTOCOL_VERSION
        assert response["selfcheck_request_id"] == "good"
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


def test_the_unpinned_default_is_the_engine_this_target_has(monkeypatch):
    """No pin: the self-check must default to the engine the GRADE would select, not to Verilator.

    THE DEFECT THIS PINS, measured on a live run. ``MERLIN_REQUIRED_RTL_ENGINE`` was unset, so the
    default fell through to the historical Verilator literal and EVERY self-check ran the slowest
    engine -- while the grader beside it selected GSIM by the declared ``rtl_engine_policy`` order and
    the target's engine home contained nothing else. Self-check turnaround across five requests went
    6 min, 7 min, 75 min, 96 min: a ~15x degradation of the loop the agent iterates in, with no flag,
    log line or verdict field naming the cause.
    """
    from merlin.targetgen.rtl_engine_policy import ENGINE_PRIORITY

    selfcheck = _module("agent_selfcheck")
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    # The engine the DECLARED priority ranks above the fallback -- named by the policy, not here.
    faster = [e for e in ENGINE_PRIORITY if e in selfcheck.SIM_TIER and e != "spike"]
    assert faster, "the engine policy declares no self-check-drivable elaborated-RTL engine"
    assert faster[0] != selfcheck.FALLBACK_RTL_SIM, "the policy must rank something above the fallback"
    monkeypatch.setattr(selfcheck, "_target_sim_via", lambda _context: ("some_target", "chipyard"))
    monkeypatch.setattr(
        selfcheck.CR,
        "describe_l3_engine",
        lambda target, sim_via: {"available": True, "engine": faster[0], "reason": "probe says so"},
    )
    context = feedback_context(HARNESS, target="some_target")
    assert selfcheck._default_sim(context) == faster[0]
    assert selfcheck._default_sim(context) != selfcheck.FALLBACK_RTL_SIM


def test_a_pin_still_beats_discovery(monkeypatch):
    """An experiment that declares its engine is stating what its results are attributable to."""
    from merlin.targetgen.rtl_engine_policy import ENGINE_PRIORITY

    selfcheck = _module("agent_selfcheck")
    drivable = [e for e in ENGINE_PRIORITY if e in selfcheck.SIM_TIER and e != "spike"]
    pinned, discovered = drivable[-1], drivable[0]
    assert pinned != discovered
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", pinned)
    monkeypatch.setattr(
        selfcheck.CR,
        "describe_l3_engine",
        lambda target, sim_via: {"available": True, "engine": discovered, "reason": "probe says so"},
    )
    assert selfcheck._default_sim() == pinned


def test_discovery_never_makes_spike_the_implicit_default(monkeypatch):
    """Spike is a correctness-only screen: resolving it here would downgrade an unflagged run."""
    selfcheck = _module("agent_selfcheck")
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    monkeypatch.setattr(selfcheck, "_target_sim_via", lambda _context: ("some_target", "chipyard"))
    monkeypatch.setattr(
        selfcheck.CR,
        "describe_l3_engine",
        lambda target, sim_via: {"available": True, "engine": "spike", "reason": "the screen is here"},
    )
    context = feedback_context(HARNESS, target="some_target")
    assert selfcheck._default_sim(context) != "spike"
    assert selfcheck._default_sim(context) == selfcheck.FALLBACK_RTL_SIM


def test_failed_discovery_falls_back_visibly(monkeypatch, capsys):
    """Fail-SAFE, not fail-silent: the fallback still runs, and it says why it fired."""
    selfcheck = _module("agent_selfcheck")
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    monkeypatch.setattr(selfcheck, "_target_sim_via", lambda _context: ("some_target", "chipyard"))
    monkeypatch.setattr(
        selfcheck.CR,
        "describe_l3_engine",
        lambda target, sim_via: {"available": False, "reason": "no elaborated-RTL engine available"},
    )
    assert selfcheck._default_sim(feedback_context(HARNESS, target="some_target")) == selfcheck.FALLBACK_RTL_SIM
    err = capsys.readouterr().err
    assert "no elaborated-RTL engine available" in err and selfcheck.FALLBACK_RTL_SIM in err


def test_an_unpinned_shim_asks_for_no_engine_at_all(monkeypatch):
    """The in-sandbox shim cannot see the target's engines, so it must not answer for them.

    Its ``verilator`` literal travelled in the request payload, and an explicit ``sim`` is a choice
    the broker must honour -- which is how the sandboxed path (i.e. every real run) overrode the
    driver-side discovery the default above installs.
    """
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    shim = _module("selfcheck_shim")
    broker = _module("selfcheck_broker")
    assert shim._default_sim() is None
    assert broker._default_sim() is None
    assert shim._sim_policy_error(None) is None


def test_the_shim_and_broker_carry_a_tier_request(tmp_path, monkeypatch):
    """``--tiers`` is how a caller names the CHEAP tier on a target whose cheap tier is not selectable
    by simulator name. It reached the real self-check from promotion jobs only; a sandboxed agent got
    ``unrecognized arguments``, the same class of gap ``--offload-census`` had."""
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    broker = _module("selfcheck_broker")
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
        "Path(out).write_text(json.dumps({'all_pass': True}))\n"
    )
    monkeypatch.setenv("FORWARD_RECORD", str(record))
    monkeypatch.setattr(broker, "worker_command", lambda *args: [sys.executable, str(fake)])
    (ch / "req_tiered.json").write_text(json.dumps({"capsules": "all", "tiers": "L2", "timeout": 30}))
    thread = threading.Thread(
        target=broker.main,
        args=(["--ws", str(ws), "--poll", "0.01"],),
        kwargs={"context": feedback_context(tmp_path), "capsules_root": tmp_path},
    )
    thread.start()
    try:
        _wait(ch / "done_tiered")
        forwarded = json.loads(record.read_text())
        assert forwarded[forwarded.index("--tiers") + 1] == "L2"
        assert "--sim" not in forwarded, "an unnamed engine must be resolved by the self-check, not the broker"
    finally:
        (ch / "STOP").write_text("stop")
        thread.join(timeout=3)
    assert not thread.is_alive()


def test_the_shim_reaches_the_tier_flag_and_names_no_engine(tmp_path, monkeypatch):
    """The request a sandboxed agent actually writes: a tier by name, and NO engine of its own."""
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    # Load the shim from where a run STAGES it (<ws>/agent_selfcheck.py) -- it derives the workspace
    # from its own path, so loading it in place would write a channel into the harness directory.
    staged = tmp_path / "agent_selfcheck.py"
    staged.write_text(module_source_path("merlin_experiments.phase1.tools.selfcheck").read_text())
    (tmp_path / "submission").mkdir()
    spec = importlib.util.spec_from_file_location("staged_shim", staged)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    # No broker is listening, so this returns `in_progress` once its wait budget expires. The request
    # it left on the channel is what is under test.
    shim.main(["--tiers", "L2", "--capsules", "A1", "--wait-budget", "1", "--timeout", "30"])
    reqs = list((tmp_path / ".qa_channel").glob("req_*.json"))
    assert len(reqs) == 1
    body = json.loads(reqs[0].read_text())
    assert body["tiers"] == "L2"
    assert "sim" not in body, "an unflagged shim must leave the engine to the driver, not pin verilator"
