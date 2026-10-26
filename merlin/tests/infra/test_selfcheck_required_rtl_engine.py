"""Agent-accessible selfchecks cannot escape an experiment-wide RTL-engine pin."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest
from phase1_feedback import MODULES, feedback_context, feedback_source

from merlin.common.paths import merlin_dir, module_source_path

pytestmark = pytest.mark.target("gemmini")

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


def _module(name: str):
    if name in {"selfcheck_broker", "simjob_broker"}:
        # The launcher invokes brokers with ``python -m``.  Loading a source file
        # under an invented top-level name breaks its legitimate relative imports.
        return importlib.import_module(f"merlin_experiments.phase1.brokers.{name.removesuffix('_broker')}")
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    source = (
        module_source_path("merlin_experiments.phase1.tools.selfcheck")
        if name == "selfcheck_shim"
        else feedback_source(name, HARNESS / f"{name}.py")
    )
    # A fresh, unregistered copy named for its package, so its own relative imports still resolve.
    canonical = "merlin_experiments.phase1.tools.selfcheck" if name == "selfcheck_shim" else MODULES.get(name, name)
    spec = importlib.util.spec_from_file_location(canonical, source)
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
    # The grade's own full-call ladder is a separate lookup (test_selfcheck_full_call_adapters.py).
    monkeypatch.setattr(selfcheck.CR, "oracle_adapters", lambda target, sim_via, **kw: {})

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


def test_required_vcs_never_constructs_the_legacy_verilator_rung(monkeypatch):
    """A VCS pin builds L3 on VCS itself (same rung as Verilator/GSIM) and never a Verilator rung."""
    selfcheck = _module("agent_selfcheck")
    selected = []
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "vcs")
    monkeypatch.setattr(
        selfcheck.CR,
        "simulator_adapter",
        lambda engine, target: selected.append((engine, target)) or object(),
    )
    # The grade's own full-call ladder is a separate lookup (test_selfcheck_full_call_adapters.py).
    monkeypatch.setattr(selfcheck.CR, "oracle_adapters", lambda target, sim_via, **kw: {})

    adapters, sim = selfcheck._adapters("vcs", "gemmini", "chipyard")
    assert selected == [("spike", "gemmini"), ("vcs", "gemmini")]
    assert sorted(adapters) == ["L2", "L3"] and sim == "vcs"
    assert selfcheck.SIM_TIER["vcs"] == "L3"


def test_both_brokers_narrow_rtl_requests_to_gsim_but_keep_spike(monkeypatch):
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "gsim")
    sync = _module("selfcheck_broker")
    async_broker = _module("simjob_broker")

    # The grade's own selection is always accepted: it honours the pin itself.
    assert sync._allowed_sims() == ("spike", "gsim", "required")
    assert sync._default_sim() == "required"
    assert async_broker._allowed_sims(
        feedback_context(merlin_dir() / "experiments/capsule_bench/targets/gemmini", target="gemmini")
    ) == ("spike", "gsim", "required")
    assert "conflicts" in sync._sim_policy_error("verilator")
    assert sync._sim_policy_error("required") is None


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
    (ws / "submission").mkdir()
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
        # The pinned engine's name is served by the grade's own selection (the same tiers), and the
        # response says which name was asked for.
        assert response["sim"] == "required" and response["requested_sim"] == "gsim"
        assert response["selfcheck_protocol"] == broker.PROTOCOL_VERSION
        assert response["selfcheck_request_id"] == "good"
        assert forwarded[forwarded.index("--sim") + 1] == "required"
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


def test_the_shims_pinned_default_is_served_by_the_grade_selection(monkeypatch):
    """Under a pin the staged shim names the pinned engine for an unflagged request. That name is the
    same tiers as the grade's selection, so the broker serves it with the grade's own adapters -- one
    request then certifies -- while the screen and every other name keep their narrower meaning."""
    from merlin_experiments.phase1.feedback import dispatch

    shim = _module("selfcheck_shim")
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "gsim")
    assert dispatch.grade_selection_for(shim._default_sim()) == dispatch.REQUIRED_SIM
    assert dispatch.grade_selection_for("spike") == "spike"
    assert dispatch.grade_selection_for(None) is None
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "vcs")
    # VCS now shares the L3 rung with Verilator/GSIM (selfcheck.SIM_TIER), so a VCS pin is served by the
    # grade's own selection exactly like a GSIM pin.
    assert dispatch.grade_selection_for("vcs") == dispatch.REQUIRED_SIM
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE")
    assert dispatch.grade_selection_for("gsim") == "gsim", "with no pin an explicit engine stays explicit"


def _selection_main_hands_to_adapters(tmp_path, monkeypatch, *argv):
    """Run the worker's ``main`` only as far as its adapter selection, and return what it asked for."""
    from types import SimpleNamespace

    selfcheck = _module("agent_selfcheck")
    seen = []

    def observe(sim, target, sim_via, **kwargs):
        seen.append((sim, target, sim_via))
        raise ValueError("selection observed")

    monkeypatch.setattr(
        selfcheck, "resolve_context", lambda *args: SimpleNamespace(target="some_target", repo=tmp_path,
                                                                    readback_policy=None)
    )
    monkeypatch.setattr(selfcheck, "_target_sim_via", lambda _context: ("some_target", "chipyard"))
    monkeypatch.setattr(selfcheck, "_adapters", observe)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "manifest.yaml").write_text("name: fixture\n")
    (tmp_path / "public").mkdir()
    rc = selfcheck.main(["--submission", str(tmp_path / "sub"), "--capsules-root", str(tmp_path / "public"),
                         "--capsules", "A1", *argv])
    assert rc == 2
    return seen, selfcheck


@pytest.mark.parametrize("pin", ["", "gsim"])
def test_the_unflagged_worker_selects_every_tier_the_grade_runs(tmp_path, monkeypatch, pin):
    """No ``--sim``: the grade's own selection, never one engine -- pinned, discovered or the screen."""
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", pin)
    seen, selfcheck = _selection_main_hands_to_adapters(tmp_path, monkeypatch)
    assert seen == [(selfcheck.REQUIRED_SIM, "some_target", "chipyard")]
    assert selfcheck.REQUIRED_SIM not in selfcheck.SIM_TIER, "the selection must not read as an engine"


def test_a_named_engine_remains_a_narrower_legacy_selection(tmp_path, monkeypatch):
    """Backwards compatibility: an explicit engine still reaches its single-engine adapters."""
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    seen, _selfcheck = _selection_main_hands_to_adapters(tmp_path, monkeypatch, "--sim", "spike")
    assert seen == [("spike", "some_target", "chipyard")]


def test_the_grade_selection_delegates_to_the_official_adapter_factory(monkeypatch):
    """The pin is honoured inside the grade's own selection, so the self-check names no engine."""
    selfcheck = _module("agent_selfcheck")
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "gsim")
    official = {"L2": object(), "L3": object()}
    monkeypatch.setattr(selfcheck.CR, "qa_checkpoint_adapters", lambda target, sim_via, **kwargs: official)
    monkeypatch.setattr(
        selfcheck.CR, "simulator_adapter",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no engine may be built directly")),
    )
    adapters, sim = selfcheck._adapters(selfcheck.REQUIRED_SIM, "gemmini", "chipyard")
    assert sim == selfcheck.REQUIRED_SIM and adapters == official
    monkeypatch.setattr(selfcheck.CR, "simulator_adapter", lambda engine, target, **kwargs: engine)
    spike_only, _ = selfcheck._adapters("spike", "gemmini", "chipyard")
    assert spike_only == {"L2": "spike"}, "the screen stays a screen"


def test_the_shim_and_broker_carry_a_tier_request(tmp_path, monkeypatch):
    """``--tiers`` is how a caller names the CHEAP tier on a target whose cheap tier is not selectable
    by simulator name. It reached the real self-check from promotion jobs only; a sandboxed agent got
    ``unrecognized arguments``, the same class of gap ``--offload-census`` had."""
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    broker = _module("selfcheck_broker")
    ws = tmp_path / "ws"
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True)
    (ws / "submission").mkdir()
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
        assert forwarded[forwarded.index("--sim") + 1] == "required", (
            "an unnamed engine selects every tier the grade runs, never one engine"
        )
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
