"""Installed probe admission and receipts, with synthetic execution and no hardware claims."""

import copy
import importlib.util
import socket
import subprocess
from pathlib import Path

import pytest
from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2 import emission_analysis as EA
from merlin_experiments.phase2 import portfolio_checkpoint as PC
from merlin_experiments.phase2 import portfolio_probes as PP

from merlin.common.digest import sha256_bytes
from merlin.perf import analysis_worker as AW
from merlin.targetgen import oot_runner


@pytest.fixture(autouse=True)
def refuse_processes_and_listeners(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("probe-owner tests must not execute tools or create listeners")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "bind", refused)


@pytest.fixture
def case(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "portfolio_analysis_fixtures", Path(__file__).with_name("portfolio_analysis_fixtures.py")
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    result = helper.build_case(tmp_path, monkeypatch)
    result.owner.analyze(result.candidate, hypothesis="synthetic previous revision")
    (result.candidate / "version.txt").write_text("1")
    result.owner.analyze(result.candidate, hypothesis="synthetic current revision")
    result.probes = PP.PortfolioProbes(result.owner)
    result.answers = tmp_path / "private-answers"
    result.answers.mkdir()
    result.overlay = tmp_path / "shared-overlay"
    result.overlay.mkdir()
    (result.overlay / "source.py").write_text("SHARED = True\n")
    for row in result.journal.iterations:
        policy = {
            "package_path": row["submitted_snapshot"],
            "compiler_dependencies": row["compiler_dependencies"],
            "command_prefix": ["never-executed-bwrap", "--clearenv", "never-executed-python"],
            "bwrap_argv_length": 2,
            "answer_surfaces": [{"path": str(result.answers), "kind": "dir"}],
            "overlay_trees": {str(result.overlay): C.exact_tree_record(result.overlay)["sha256"]},
        }
        result.journal.register_compiler_sandboxes(row["iteration"], {"candidate": policy})
    baseline_policy = {
        **copy.deepcopy(policy),
        "package_path": str(result.inputs.optimization_baseline),
        "compiler_dependencies": result.inputs.optimization_baseline_binding["compiler_dependencies"],
    }
    result.journal.register_optimization_baseline_sandbox(baseline_policy)
    result.owner.analyzer = AW.IsolatedAnalysisWorker(
        analysis_source=Path(EA.__file__),
        contract_root=result.inputs.contract_root,
        output=tmp_path / "unused-workers",
        sandbox_factory=lambda *_: pytest.fail("retained policy must not be recreated"),
    )
    return result


@pytest.mark.parametrize("arm", ["candidate", "previous", "baseline"])
@pytest.mark.parametrize("nested", [False, True])
def test_compilation_uses_exact_retained_arm_and_refreshes_action(case, monkeypatch, arm, nested):
    events, executions = [], []
    for name in ("refresh", "revalidate"):
        original = getattr(case.session, name)

        def recorded(candidate, *, original=original, name=name):
            events.append(name)
            return original(candidate)

        monkeypatch.setattr(case.session, name, recorded)
    monkeypatch.setattr(oot_runner, "load_package", lambda package: package)

    def execute(package, name, interface, output_json=None, **kwargs):
        executions.append((package, name, kwargs))
        assert interface.read_text() == "synthetic reduced interface"
        if output_json is not None:
            output_json.write_text('{"commands":[],"tensors":{}}')
        return subprocess.CompletedProcess([name], 0, "synthetic lowered", "")

    monkeypatch.setattr(AW, "run_sandboxed_entrypoint", execute)
    interface = case.root / "reduced.mlir"
    interface.write_text("synthetic reduced interface")
    method = {
        "candidate": case.probes.compile_probe_candidate,
        "previous": case.probes.compile_previous_probe_candidate,
        "baseline": case.probes.compile_optimization_baseline_probe_candidate,
    }[arm]

    def compile_pair():
        return method(case.candidate, interface, case.root / "compile-scratch", timeout_s=10, emit_command_buffer=True)

    if nested:
        with case.session.action(case.candidate):
            result = compile_pair()
    else:
        result = compile_pair()
    expected = (
        case.inputs.optimization_baseline
        if arm == "baseline"
        else Path(case.journal.iterations[0 if arm == "previous" else 1]["submitted_snapshot"])
    )
    assert [name for _, name, _ in executions] == ["lower_target_to_llvm", "emit_command_buffer"]
    assert all(package == expected for package, _, _ in executions)
    assert executions[0][2]["sandbox"] == executions[1][2]["sandbox"]
    assert 0 < executions[1][2]["timeout_s"] <= executions[0][2]["timeout_s"] <= 10
    assert result["command_buffer"] == {"commands": [], "tensors": {}}
    assert events == (["revalidate", "revalidate"] if nested else ["refresh"])


@pytest.mark.parametrize("mutation", ["policy", "overlay", "private", "linked_scratch", "package"])
def test_sandbox_admission_refuses_changed_or_private_inputs(case, mutation):
    scratch = case.root / "probe-scratch"
    scratch.mkdir()
    if mutation == "policy":
        case.journal.compiler_sandboxes[1]["candidate"]["command_prefix"].append("untrusted")
    elif mutation == "overlay":
        (case.overlay / "source.py").write_text("CHANGED = True\n")
    elif mutation == "private":
        scratch = case.answers / "inside-private"
        scratch.mkdir()
    elif mutation == "linked_scratch":
        alias = case.root / "scratch-link"
        alias.symlink_to(scratch, target_is_directory=True)
        scratch = alias
    else:
        policy = copy.deepcopy(case.journal.compiler_sandboxes[1])
        policy["candidate"]["package_path"] = str(case.candidate)
        case.journal.register_compiler_sandboxes(1, policy)
    with pytest.raises(ValueError):
        case.probes._probe_sandbox(case.candidate, scratch)


def test_compiler_return_does_not_bypass_full_input_refresh(case, monkeypatch):
    monkeypatch.setattr(oot_runner, "load_package", lambda package: package)

    def mutate(*args, **kwargs):
        (case.inputs.baseline / "version.txt").write_text("changed during synthetic compiler call")
        return subprocess.CompletedProcess([], 0, "synthetic", "")

    monkeypatch.setattr(AW, "run_sandboxed_entrypoint", mutate)
    interface = case.root / "reduced.mlir"
    interface.write_text("synthetic")
    with pytest.raises(ValueError, match="frozen compiler changed"):
        case.probes.compile_probe_candidate(case.candidate, interface, case.root / "probe", timeout_s=10)


@pytest.mark.parametrize("timeout,argv", [(0, ["never-run"]), (float("nan"), ["never-run"]), (1, [])])
def test_native_execution_refuses_invalid_admission_before_any_process(case, timeout, argv):
    scratch = case.root / "native-scratch"
    scratch.mkdir()
    with pytest.raises(ValueError, match="positive finite deadline"):
        case.probes.run_native_probe(case.candidate, scratch, argv, timeout_s=timeout)


def test_native_policy_export_is_detached_and_grants_only_public_scratch(case):
    scratch = case.root / "native-scratch"
    scratch.mkdir()
    policy = case.probes.native_probe_policy(case.candidate, scratch)
    original = copy.deepcopy(policy)
    assert policy["package_path"] == case.journal.iterations[-1]["submitted_snapshot"]
    assert policy["command_prefix"][2:5] == ["--bind", str(scratch), str(scratch)]
    policy["command_prefix"].append("caller mutation")
    assert case.probes.native_probe_policy(case.candidate, scratch) == original


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("build", "exact host-owned dependency capability"),
        ("execution", "exact trusted executable capability"),
        ("combined", "separate action bounded to 60 seconds"),
        ("execution_over_budget", "separate action bounded to 60 seconds"),
    ],
)
def test_native_dependency_grants_require_exact_capabilities_before_process(case, kind, reason):
    class FakeCapability:
        def extend(self, *args, **kwargs):
            pytest.fail("an untrusted dependency capability must never be invoked")

        def revalidate(self, *args, **kwargs):
            pytest.fail("an untrusted dependency capability must never be invoked")

    scratch = case.root / "capability-scratch"
    scratch.mkdir()
    options = {}
    if kind in ("build", "combined"):
        options["_build_dependencies"] = FakeCapability()
    if kind in ("execution", "combined", "execution_over_budget"):
        options["_execution_dependencies"] = FakeCapability()
    with pytest.raises(ValueError, match=reason):
        case.probes.run_native_probe(
            case.candidate,
            scratch,
            ["never-executed"],
            timeout_s=61 if kind == "execution_over_budget" else 10,
            **options,
        )


def test_compiler_exception_still_refreshes_the_outer_action(case, monkeypatch):
    monkeypatch.setattr(oot_runner, "load_package", lambda package: package)
    events = []
    original = case.session.refresh

    def refreshed(candidate):
        events.append("refresh")
        return original(candidate)

    def failed(*args, **kwargs):
        raise TimeoutError("synthetic compiler timeout")

    monkeypatch.setattr(case.session, "refresh", refreshed)
    monkeypatch.setattr(AW, "run_sandboxed_entrypoint", failed)
    interface = case.root / "reduced.mlir"
    interface.write_text("synthetic")
    with pytest.raises(TimeoutError, match="synthetic compiler timeout"):
        case.probes.compile_probe_candidate(case.candidate, interface, case.root / "probe", timeout_s=10)
    assert events == ["refresh"]


@pytest.mark.parametrize("outcome", ["success", "exception", "substituted_member"])
def test_member_scoped_semantic_receipt_and_failure_accounting(case, monkeypatch, outcome):
    row = case.journal.iterations[-1]
    initial_elapsed = row["elapsed_seconds"]
    clock = [100.0]
    monkeypatch.setattr(PP.time, "monotonic", lambda: clock[0])
    seen = []

    def provider(*, candidate, probes, timeout_s, portfolio_member):
        assert probes is case.probes and timeout_s > 0
        selected = probes.analysis.session.selected_changed_portfolio_context(candidate, portfolio_member)
        seen.append(portfolio_member["portfolio_index"])
        clock[0] += 2
        if outcome == "exception":
            raise RuntimeError("synthetic qualification unavailable")
        binding = {
            "selection": selected["selection"],
            "previous": selected["previous"]["member_binding"],
            "current": selected["current"]["member_binding"],
        }
        if outcome == "substituted_member":
            binding["current"] = {**binding["current"], "capsule_sha256": "0" * 64}
        return {"status": "synthetic_binding_only", "portfolio_member_binding": binding}

    if outcome == "success":
        receipt = case.probes.qualify_changed_region(case.candidate, provider=provider, timeout_s=30)
        assert receipt["full_model_numerics_qualified"] is False
        assert receipt["global_speedup_proven"] is False
        assert receipt["scope"] == "selected changed mechanism and tested reduced domain only"
        PC.verify_changed_region_semantic_receipt(
            receipt,
            iteration=row,
            portfolio_identity=case.inputs.portfolio_identity,
            target_sha256=case.inputs.target_sha256,
            experiment_root=case.output,
        )
        assert len(row["semantic_receipts"]) == 1
    else:
        with pytest.raises(RuntimeError if outcome == "exception" else ValueError):
            case.probes.qualify_changed_region(case.candidate, provider=provider, timeout_s=30)
        assert not row.get("semantic_receipts")
    assert seen == [0]
    assert row["elapsed_seconds"] == initial_elapsed + 2


def admission(case):
    from merlin.perf.activity_schedule import ActivityEvent
    from merlin.perf.execution_policy import SimulationBudget
    from merlin.perf.mechanism_probe import MechanismEvidence, derive_mechanism_signature
    from merlin.xdsl_dialects.lowering.global_plan import ValueRepresentation

    binding = case.session.current_probe_binding(case.candidate)
    signature = derive_mechanism_signature(
        representations=[ValueRepresentation("local", "row-major", "i8", "blocked")],
        events=[
            ActivityEvent("load", "dma", "movement", 0, movement_bytes=128),
            ActivityEvent("compute", "mesh", "compute", 0, depends_on=("load",)),
        ],
        capacity_regime={"scratch": "fits_double"},
        tile_shape=(8, 8, 8),
        edge_cases=("aligned",),
        repetition_semantics="independent tiles",
        instruction_semantics=[{"operation": "load"}, {"operation": "matmul"}],
    )
    model = MechanismEvidence(
        binding,
        signature,
        case.journal.iterations[-1]["analysis"]["emission"]["candidate_lowered_sha256"],
        1024,
        "synthetic host model",
    )
    probe = MechanismEvidence(binding, signature, sha256_bytes(b"synthetic-probe"), 4, "synthetic host probe")
    return dict(
        model=model,
        probe=probe,
        descriptor={"kind": "kernel"},
        budget=SimulationBudget(30, 30),
        estimated_cycles=100,
        measured_cycles_per_second=100,
    )


@pytest.mark.parametrize("outcome", ["success", "exception", "wrong_observation", "over_budget"])
def test_measurement_admission_receipt_scope_and_failure_charges(case, monkeypatch, outcome):
    from merlin.perf.execution_policy import WarmComputeReceipt, WarmProfileContract
    from merlin.perf.mechanism_probe import ProbeObservation

    inputs = admission(case)
    row = case.journal.iterations[-1]
    original = row["elapsed_seconds"]
    clock = [100.0]
    monkeypatch.setattr(PP.time, "monotonic", lambda: clock[0])
    elapsed = 31 if outcome == "over_budget" else 3

    def execute(*, timeout_s):
        assert timeout_s == 30
        clock[0] += elapsed
        if outcome == "exception":
            raise TimeoutError("synthetic bounded executor stopped")
        if outcome == "wrong_observation":
            return None
        return ProbeObservation(
            inputs["probe"],
            WarmComputeReceipt("synthetic-probe", 100, WarmProfileContract(), "synthetic counter receipt"),
            0.1,
            1,
            inputs["probe"].artifact_digest,
        )

    if outcome == "success":
        receipt = case.probes.measure_probe(case.candidate, admission_inputs=inputs, execute=execute)
        assert receipt["scope"] == "mechanism_probe_only" and receipt["full_model_cycles"] is None
        assert receipt["target_timing_authority"] is None
        assert receipt["warmup_runs"] == receipt["measured_runs"] == 1
        sealed = case.session.seal(case.candidate)
        assert (
            len(
                PC.consume_global_candidate(
                    sealed,
                    context=PC.CheckpointVerificationContext(
                        case.inputs.host_policy, case.inputs.compiler_shared_source_root
                    ),
                )["probe_receipts"]
            )
            == 1
        )
    else:
        with pytest.raises(TimeoutError if outcome == "exception" else ValueError):
            case.probes.measure_probe(case.candidate, admission_inputs=inputs, execute=execute)
        assert not row["probe_receipts"]
    assert row["elapsed_seconds"] == original + elapsed


def test_admission_cannot_replace_binding_or_name_different_model(case):
    inputs = admission(case)
    with pytest.raises(ValueError, match="cannot replace"):
        case.probes.measure_probe(
            case.candidate,
            admission_inputs={**inputs, "current_binding": inputs["model"].binding},
            execute=lambda **kwargs: pytest.fail("unadmitted probe executed"),
        )
    with pytest.raises(ValueError, match="different compiled artifact"):
        case.probes.measure_probe(
            case.candidate,
            admission_inputs={**inputs, "model": inputs["probe"]},
            execute=lambda **kwargs: pytest.fail("unadmitted probe executed"),
        )


@pytest.mark.parametrize("mutation", [None, "binding", "elf"])
def test_controlled_prefix_keeps_exact_artifact_and_non_global_scope(case, mutation):
    captured = case.session.current_artifacts(case.candidate)
    captured["decoded_trace"] = {"instructions": [{"synthetic": index} for index in range(3)]}
    binding = case.session.current_probe_binding(case.candidate)
    workdir = case.root / "synthetic-prefix"
    workdir.mkdir()
    elf = workdir / "primitive.elf"
    elf.write_bytes(b"synthetic artifact, never executed")
    source = {
        "timed_source_instruction_indices": [0],
        "source_instruction_indices": [0],
        "source_artifact_sha256": captured["candidate_lowered_sha256"],
        "slice_source_sha256": sha256_bytes(b"synthetic reduced source"),
        "full_model_executed": False,
        "full_layer_executed": False,
        "source_task_compute_pair_count": 2,
        "executed_compute_pair_count": 1,
    }
    prepared = {
        "workdir": str(workdir),
        "elf_sha256": C.sha256_file(elf),
        "wrapper_sha256": sha256_bytes(b"synthetic wrapper"),
        "primitive_mlir_sha256": sha256_bytes(b"synthetic primitive"),
        "domain_digest": sha256_bytes(b"synthetic domain"),
        "source_artifact_sha256": source["slice_source_sha256"],
        "measurement_scope": "controlled_source_prefix",
        "simulator_executed": False,
        "timed_instruction_count": 1,
        "host_input_bytes": 8,
        "output_storage_bytes": 8,
    }
    executed = []

    def provider(*, candidate, probes, timeout_s):
        assert probes is case.probes and candidate == case.candidate and timeout_s > 0

        def execute(*, timeout_s):
            executed.append(True)
            if mutation == "elf":
                elf.write_bytes(b"changed after admission")
            return {
                **prepared,
                "correct": True,
                "warmup_runs": 1,
                "measured_runs": 1,
                "full_model_executed": False,
                "full_source_probe_executed": False,
                "total_compute_cycles": 100,
            }

        return {
            "controlled_context_inputs": {
                "binding": None if mutation == "binding" else binding,
                "scope": "controlled_source_prefix",
                "model_artifact_sha256": captured["candidate_lowered_sha256"],
                "source_slice": source,
                "prepared": prepared,
            },
            "execute": execute,
        }

    if mutation is None:
        receipt = case.probes.profile_controlled_context(case.candidate, provider=provider, timeout_s=30)
        assert receipt["scope"] == "controlled_source_prefix"
        assert receipt["global_cost_validated"] is False and receipt["calibration_admissible"] is False
        assert receipt["global_speedup_proven"] is False and receipt["full_model_cycles"] is None
        pointer = case.journal.iterations[-1]["context_receipts"][0]
        assert C.sha256_file(Path(pointer["path"])) == pointer["sha256"]
    else:
        with pytest.raises(ValueError):
            case.probes.profile_controlled_context(case.candidate, provider=provider, timeout_s=30)
        assert not case.journal.iterations[-1].get("context_receipts")
    assert bool(executed) is (mutation != "binding")
