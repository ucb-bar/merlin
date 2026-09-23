"""Real launcher control flow with fake Chia transport: no cluster, agents or hardware."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from types import SimpleNamespace

import pytest
from merlin_experiments.execution import chia_native

from merlin.benchharness import chia_bridge
from merlin.common.paths import repo_root, runs_dir


@pytest.fixture
def launchers(tmp_path, monkeypatch):
    pytest.importorskip("aet.tracking.run_logger")
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path))
    monkeypatch.setenv("MERLIN_EXPERIMENT_PYTHON", sys.executable)
    state = {"returncode": 0, "submitted": [], "metrics": [], "collector": None, "ray": False}

    class Ref:
        def __init__(self, value, number):
            self.value, self.number = value, number

        def hex(self):
            return str(self.number)

    def get(ref, **kwargs):
        if isinstance(ref, list):
            return [get(item) for item in ref]
        if isinstance(ref.value, Exception):
            raise ref.value
        return ref.value

    def decorate(**options):
        def wrapper(function):
            def remote(*args, **kwargs):
                state["submitted"].append((function.__name__, args))
                if len(state["submitted"]) - 1 == state.get("dispatch_error_index"):
                    raise ValueError("synthetic dispatch failure")
                result = {
                    "run_id": args[-1] if function.__name__ != "run_coordinator" else "coordinator",
                    "returncode": state["returncode"],
                    "wall_s": 0.5,
                }
                if len(state["submitted"]) - 1 == state.get("worker_error_index"):
                    result = RuntimeError("synthetic worker failure")
                return Ref(result, len(state["submitted"]))

            function.chia_remote = remote
            function.options = lambda **kwargs: function
            return function

        return wrapper

    chia = types.ModuleType("chia")
    base = types.ModuleType("chia.base")
    functions = types.ModuleType("chia.base.ChiaFunction")
    functions.ChiaFunction = decorate
    functions.get = get
    trace = types.ModuleType("chia.trace")
    # The wrapper reads a source identity, but this fixture never claims upstream qualification.
    trace.__file__ = __file__
    chia.trace, chia.base = trace, base
    base.ChiaFunction = functions
    ray = types.ModuleType("ray")
    ray.ObjectRef = Ref
    ray.get = get
    ray.wait = lambda refs, **kwargs: (refs, [])
    ray.cancel = lambda ref, **kwargs: None
    ray.is_initialized = lambda: state["ray"]
    ray.init = lambda **kwargs: state.update(ray=True)
    ray.shutdown = lambda: state.update(ray=False)
    ray.cluster_resources = lambda: {}
    ray.get_runtime_context = lambda: SimpleNamespace(get_node_id=lambda: "owned-node")
    ray.nodes = lambda: [
        {
            "Alive": True,
            "NodeID": "owned-node",
            "Resources": {"CPU": 2, "verilator": 1, "codex_slots": 1, "gsim_slots": 1},
        }
    ]
    strategies = types.ModuleType("ray.util.scheduling_strategies")
    strategies.NodeAffinitySchedulingStrategy = lambda **kwargs: SimpleNamespace(**kwargs)
    ray.kill = lambda actor: None
    trace.get_collector = lambda: state["collector"]
    trace.start_collector = lambda **kwargs: state.update(collector=kwargs["log_dir"])
    trace.stop_collector = lambda: state.update(collector=None)
    profiler = types.ModuleType("chia.trace.profiler")
    profiler.reset_profiler = lambda: None
    profiler.get_collector = trace.get_collector
    profiler.start_collector = trace.start_collector
    profiler.stop_collector = trace.stop_collector
    metrics = types.ModuleType("chia.trace.metrics")
    metrics.MetricsBackend = type("MetricsBackend", (), {})
    for name, module in (
        ("chia", chia),
        ("chia.base", base),
        ("chia.base.ChiaFunction", functions),
        ("chia.trace", trace),
        ("chia.trace.metrics", metrics),
        ("chia.trace.profiler", profiler),
        ("ray", ray),
        ("ray.util.scheduling_strategies", strategies),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(chia_bridge, "_backend_cls", None)  # isolate the optional upstream base class
    context = SimpleNamespace(REPO=tmp_path, EXP=tmp_path / "target", RUNS=tmp_path / "runs", TARGET="fixture")
    batch = types.ModuleType("launch_ab_batch")
    batch.C, batch.ARMS = context, {"baseline": {}}
    batch._run_id = lambda arm, tag: f"{arm}_{tag}"
    batch._arm_cmd = lambda arm, rid, args, cond: [sys.executable, "synthetic_driver.py", "--run-id", rid]
    batch._run_preflight = lambda commands: state.get("preflight_returncode", 0)
    monkeypatch.setitem(sys.modules, "launch_ab_batch", batch)
    monkeypatch.setitem(sys.modules, "_common", context)
    repeat = types.ModuleType("run_repeatability")
    repeat.DRIVER = tmp_path / "synthetic_driver.py"
    repeat._load = lambda path: None
    repeat._agg = lambda values: {"values": values}
    monkeypatch.setitem(sys.modules, "run_repeatability", repeat)
    frozen = types.ModuleType("merlin_experiments.frozen_python")
    frozen.inherited_python_command = lambda command: command
    monkeypatch.setitem(sys.modules, "merlin_experiments.frozen_python", frozen)

    class NativeSession:
        source = None

        def __init__(self, endpoint):
            self.endpoint, self.invocations = endpoint, []
            state.setdefault("native_sessions", []).append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

        def reserve(self):
            invitation = SimpleNamespace(invocation=str(len(self.invocations)))
            self.invocations.append(invitation)
            return invitation

        def close(self):
            pass

        def receipt(self, invitation, **kwargs):
            return {
                "cleanup_complete": True,
                "guardian": {"native_started": state.get("native_started", True), "returncode": state["returncode"]},
            }

    monkeypatch.setattr(chia_native, "Session", NativeSession)
    originals = list(sys.path)

    def load(name):
        suite = "gemmini_perf_bench/scripts" if name == "chia_agentic_perf_experiment" else "capsule_bench/harness"
        source = repo_root() / "merlin/experiments" / suite / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"outcome_{name}", source)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        return module

    yield load, state, tmp_path
    sys.path[:] = originals


@pytest.mark.parametrize("returncode", [0, 17])
@pytest.mark.parametrize("launcher", ["chia_ab_batch", "chia_repeatability", "chia_agentic_perf_experiment"])
def test_child_outcome_controls_launcher_and_aet_run(launchers, launcher, returncode):
    load, state, _output = launchers
    state["returncode"] = returncode
    module = load(launcher)
    arguments = {
        "chia_ab_batch": [
            "--tag",
            "outcome",
            "--arms",
            "baseline",
            "--repeats",
            "2",
            "--managed-native-endpoint",
            "synthetic",
        ],
        "chia_repeatability": ["--n", "2", "--prefix", "outcome", "--managed-native-endpoint", "synthetic"],
        "chia_agentic_perf_experiment": [
            "--orchestration-run-id",
            "outcome",
            "--stub-seconds",
            "0.01",
            "--managed-native-endpoint",
            "synthetic",
        ],
    }[launcher]
    observed = module.main(arguments)
    expected = returncode if launcher == "chia_agentic_perf_experiment" else int(returncode != 0)
    assert observed == expected
    records = list(runs_dir().rglob("run_record.json"))
    assert len(records) == 1
    run_dir = records[0].parent
    events = [json.loads(line) for line in (run_dir / "logs/events.jsonl").read_text().splitlines()]
    finished = [event for event in events if event.get("event") == "run.finished"]
    assert finished[-1]["payload"]["status"] == ("error" if returncode else "ok")
    assert state["ray"] is False and state["collector"] is None
    assert len(state["submitted"]) == (1 if launcher == "chia_agentic_perf_experiment" else 2)
    if launcher == "chia_repeatability":
        receipt = json.loads((run_dir / "chia/repeats.json").read_text())
        assert [row["returncode"] for row in receipt["results"]] == [returncode, returncode]
        summary = json.loads((run_dir / "metrics/summary_metrics.json").read_text())
        assert len(summary["nonzero_returncodes"]) == (2 if returncode else 0)
    elif launcher == "chia_agentic_perf_experiment":
        summary = json.loads((run_dir / "metrics/summary_metrics.json").read_text())
        assert summary["result"]["returncode"] == returncode
        scalars = [json.loads(line) for line in (run_dir / "chia/metrics.jsonl").read_text().splitlines()]
        assert {"tag": "coordinator/returncode", "value": returncode, "step": 0} in scalars


def test_repeatability_keeps_successful_peer_when_one_worker_raises(launchers):
    load, state, _output = launchers
    state["worker_error_index"] = 0
    assert (
        load("chia_repeatability").main(["--n", "2", "--prefix", "partial", "--managed-native-endpoint", "synthetic"])
        == 1
    )
    run_dir = next(runs_dir().rglob("run_record.json")).parent
    results = json.loads((run_dir / "chia/repeats.json").read_text())["results"]
    assert results[0]["returncode"] == 1 and "synthetic worker failure" in results[0]["error"]
    assert results[0]["wall_s"] is None
    assert results[1]["returncode"] == 0 and results[1]["wall_s"] == 0.5


def test_batch_requires_explicit_service_but_dry_run_does_not(launchers):
    load, state, _ = launchers
    module = load("chia_ab_batch")
    with pytest.raises(SystemExit) as caught:
        module.main(["--tag", "missing", "--arms", "baseline"])
    assert caught.value.code == 2 and not state.get("native_sessions")
    assert module.main(["--tag", "plan", "--arms", "baseline", "--dry-run"]) == 0
    assert not state.get("native_sessions") and not state["submitted"]


def test_batch_missing_local_resources_refuses_before_dispatch(launchers, monkeypatch):
    load, state, _ = launchers
    monkeypatch.setattr(sys.modules["ray"], "nodes", lambda: [])
    with pytest.raises(RuntimeError, match="driver's own Ray node"):
        load("chia_ab_batch").main(
            ["--tag", "resources", "--arms", "baseline", "--managed-native-endpoint", "synthetic"]
        )
    assert not state["submitted"] and not state["ray"] and state["collector"] is None


def test_batch_partial_dispatch_retains_independent_reserved_evidence(launchers):
    load, state, _ = launchers
    state["dispatch_error_index"] = 1
    with pytest.raises(ValueError, match="synthetic dispatch failure"):
        load("chia_ab_batch").main(
            ["--tag", "partial", "--arms", "baseline", "--repeats", "2", "--managed-native-endpoint", "synthetic"]
        )
    run_dir = next(runs_dir().rglob("run_record.json")).parent
    native = json.loads((run_dir / "chia/native.json").read_text())
    tasks = json.loads((run_dir / "chia/tasks.json").read_text())
    assert len(native["tasks"]) == 2 and all(row["receipt"]["cleanup_complete"] for row in native["tasks"])
    assert tasks["tasks"][-1]["state"] == "dispatch_unknown"
    assert tasks["native_descendants"] == "not_verified"
    assert not state["ray"] and state["collector"] is None


def test_batch_return_without_native_execution_fails(launchers):
    load, state, _ = launchers
    state["native_started"] = False
    with pytest.raises(chia_native.CleanupIncomplete):
        load("chia_ab_batch").main(["--tag", "bypass", "--arms", "baseline", "--managed-native-endpoint", "synthetic"])
    run_dir = next(runs_dir().rglob("run_record.json")).parent
    assert not json.loads((run_dir / "chia/native.json").read_text())["cleanup_complete"]
    events = [json.loads(line) for line in (run_dir / "logs/events.jsonl").read_text().splitlines()]
    assert [event for event in events if event.get("event") == "run.finished"][-1]["payload"]["status"] == "error"


def test_repeat_requires_endpoint_only_for_planned_execution(launchers):
    load, state, _ = launchers
    module = load("chia_repeatability")
    with pytest.raises(SystemExit) as help_result:
        module.main(["--help"])
    assert help_result.value.code == 0 and not state.get("native_sessions")
    with pytest.raises(SystemExit) as caught:
        module.main(["--n", "1", "--prefix", "missing"])
    assert caught.value.code == 2 and not state.get("native_sessions")
    assert module.main(["--n", "1", "--prefix", "plan", "--dry-run"]) == 0
    complete = module.C.RUNS / "raw_baseline" / "complete_01"
    complete.mkdir(parents=True)
    (complete / "run_manifest.yaml").write_text("complete: true\n")
    assert module.main(["--n", "1", "--prefix", "complete"]) == 0
    assert not state.get("native_sessions") and not state["submitted"]


def test_repeat_missing_local_resources_refuses_before_dispatch(launchers, monkeypatch):
    load, state, _ = launchers
    monkeypatch.setattr(sys.modules["ray"], "nodes", lambda: [])
    with pytest.raises(RuntimeError, match="driver's own Ray node"):
        load("chia_repeatability").main(["--n", "1", "--managed-native-endpoint", "synthetic"])
    assert not state["submitted"] and not state["ray"] and state["collector"] is None


def test_repeat_builds_frozen_transport_on_driver_and_keeps_native_argv(launchers, monkeypatch):
    load, state, _ = launchers
    commands = []

    def bind(command):
        commands.append(list(command))
        return ["bound-python", "sealed-transport", *command]

    monkeypatch.setattr(sys.modules["merlin_experiments.frozen_python"], "inherited_python_command", bind)
    assert (
        load("chia_repeatability").main(["--n", "1", "--stub-seconds", "0.1", "--managed-native-endpoint", "synthetic"])
        == 0
    )
    assert len(commands) == 1 and commands[0] == [sys.executable, "-c", "import time; time.sleep(0.1)"]
    assert state["submitted"][0][1][0] == ["bound-python", "sealed-transport", *commands[0]]


def test_repeat_partial_dispatch_keeps_native_reservations_and_primary(launchers):
    load, state, _ = launchers
    state["dispatch_error_index"] = 1
    with pytest.raises(ValueError, match="synthetic dispatch failure"):
        load("chia_repeatability").main(["--n", "2", "--managed-native-endpoint", "synthetic"])
    run_dir = next(runs_dir().rglob("run_record.json")).parent
    native = json.loads((run_dir / "chia/native.json").read_text())
    tasks = json.loads((run_dir / "chia/tasks.json").read_text())
    assert len(native["tasks"]) == 2 and all(row["receipt"]["cleanup_complete"] for row in native["tasks"])
    assert tasks["tasks"][-1]["state"] == "dispatch_unknown"
    assert tasks["native_descendants"] == "not_verified"
    assert not state["ray"] and state["collector"] is None


def test_repeat_return_without_native_execution_fails(launchers):
    load, state, _ = launchers
    state["native_started"] = False
    with pytest.raises(chia_native.CleanupIncomplete):
        load("chia_repeatability").main(["--n", "1", "--managed-native-endpoint", "synthetic"])
    run_dir = next(runs_dir().rglob("run_record.json")).parent
    assert not json.loads((run_dir / "chia/native.json").read_text())["cleanup_complete"]
    events = [json.loads(line) for line in (run_dir / "logs/events.jsonl").read_text().splitlines()]
    assert [event for event in events if event.get("event") == "run.finished"][-1]["payload"]["status"] == "error"


def test_repeat_body_refuses_unmanaged_local_execution(launchers):
    load, _, _ = launchers
    with pytest.raises(RuntimeError, match="managed Chia setup hook"):
        load("chia_repeatability").run_repeat([sys.executable, "-c", "raise SystemExit(87)"], "/tmp", "refused")


def test_repeat_unavailable_service_refuses_before_accounting_or_dispatch(launchers, monkeypatch):
    load, state, _ = launchers

    def unavailable(endpoint):
        raise FileNotFoundError("provisioned endpoint unavailable")

    monkeypatch.setattr(chia_native, "Session", unavailable)
    with pytest.raises(FileNotFoundError, match="provisioned endpoint unavailable"):
        load("chia_repeatability").main(["--n", "1", "--managed-native-endpoint", "absent"])
    assert not state["submitted"] and not state["ray"] and state["collector"] is None
    assert not list(runs_dir().rglob("run_record.json"))


def test_repeat_existing_unfinished_run_keeps_resume_argv(launchers):
    load, state, _ = launchers
    module = load("chia_repeatability")
    (module.C.RUNS / "raw_baseline" / "resume_01").mkdir(parents=True)
    assert module.main(["--n", "1", "--prefix", "resume", "--managed-native-endpoint", "synthetic"]) == 0
    command = state["submitted"][0][1][0]
    assert command.count("--resume") == 1
    assert command[command.index("--run-id") + 1] == "resume_01"


def test_perf_requires_endpoint_but_help_and_dry_run_do_not(launchers):
    load, state, _ = launchers
    module = load("chia_agentic_perf_experiment")
    with pytest.raises(SystemExit) as help_result:
        module.main(["--help"])
    assert help_result.value.code == 0
    arguments = ["--orchestration-run-id", "admission", "--stub-seconds", "0.01"]
    with pytest.raises(SystemExit) as missing:
        module.main(arguments)
    assert missing.value.code == 2
    assert module.main([*arguments, "--dry-run"]) == 0
    assert not state.get("native_sessions") and not state["submitted"] and not state["ray"]


def test_perf_unavailable_endpoint_refuses_before_accounting(launchers, monkeypatch):
    load, state, _ = launchers

    def unavailable(endpoint):
        raise FileNotFoundError("service unavailable")

    monkeypatch.setattr(chia_native, "Session", unavailable)
    with pytest.raises(FileNotFoundError, match="service unavailable"):
        load("chia_agentic_perf_experiment").main(
            ["--orchestration-run-id", "unavailable", "--stub-seconds", "0.01", "--managed-native-endpoint", "absent"]
        )
    assert not state["submitted"] and not list(runs_dir().rglob("run_record.json"))


def test_perf_requires_both_resources_on_managed_node(launchers, monkeypatch):
    load, state, _ = launchers
    monkeypatch.setattr(
        sys.modules["ray"],
        "nodes",
        lambda: [
            {"Alive": True, "NodeID": "owned-node", "Resources": {"CPU": 1, "codex_slots": 1}},
            {"Alive": True, "NodeID": "elsewhere", "Resources": {"CPU": 1, "gsim_slots": 1}},
        ],
    )
    with pytest.raises(RuntimeError, match="driver's own Ray node"):
        load("chia_agentic_perf_experiment").main(
            ["--orchestration-run-id", "resources", "--stub-seconds", "0.01", "--managed-native-endpoint", "synthetic"]
        )
    assert not state["submitted"] and not state["ray"] and state["collector"] is None


def test_perf_keeps_native_command_separate_from_frozen_transport(launchers, monkeypatch):
    load, state, _ = launchers
    monkeypatch.setattr(
        sys.modules["merlin_experiments.frozen_python"],
        "inherited_python_command",
        lambda command: ["guarded-python", "sealed-transport", *command],
    )
    module = load("chia_agentic_perf_experiment")
    assert (
        module.main(
            ["--orchestration-run-id", "frozen", "--stub-seconds", "0.01", "--managed-native-endpoint", "synthetic"]
        )
        == 0
    )
    _, arguments = state["submitted"][0]
    command, _, plan, _ = arguments
    assert plan["command"] == command
    assert plan["transport_command"] == ["guarded-python", "sealed-transport", *command]
    assert plan["command_artifacts"] == module.chia_launch.command_artifacts(command)
    assert plan["launch_policy"] == module.chia_launch.policy_identity()
    unhashed = {key: value for key, value in plan.items() if key != "sha256"}
    assert module.hashlib.sha256(module._canonical(unhashed)).hexdigest() == plan["sha256"]


@pytest.mark.parametrize("failure", ["dispatch", "bypass"])
def test_perf_independent_native_refusal_preserves_failed_run(launchers, failure):
    load, state, _ = launchers
    if failure == "dispatch":
        state["dispatch_error_index"] = 0
    else:
        state["native_started"] = False
    with pytest.raises(ValueError if failure == "dispatch" else chia_native.CleanupIncomplete):
        load("chia_agentic_perf_experiment").main(
            ["--orchestration-run-id", "failed", "--stub-seconds", "0.01", "--managed-native-endpoint", "synthetic"]
        )
    run_dir = next(runs_dir().rglob("run_record.json")).parent
    native = json.loads((run_dir / "chia/native.json").read_text())
    assert len(native["tasks"]) == 1 and native["tasks"][0]["receipt"]["cleanup_complete"]
    if failure == "bypass":
        assert not native["cleanup_complete"]
    assert json.loads((run_dir / "chia/tasks.json").read_text())["native_descendants"] == "not_verified"
    events = [json.loads(line) for line in (run_dir / "logs/events.jsonl").read_text().splitlines()]
    assert [event for event in events if event.get("event") == "run.finished"][-1]["payload"]["status"] == "error"
    assert not state["ray"] and state["collector"] is None
