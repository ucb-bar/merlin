"""Installed global assembly with explicit roots and synthetic scientific observations."""

import importlib.util
import json
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2 import emission_analysis as EA
from merlin_experiments.phase2 import host_policy as HP
from merlin_experiments.phase2 import portfolio_checkpoint as PC
from merlin_experiments.phase2.global_experiment import GlobalPerfExperiment
from merlin_experiments.phase2.stage_inputs import StageE2ESentinel

from merlin.benchharness import hash_tree
from merlin.common.digest import sha256_bytes
from merlin.perf import phase2_edit_contract
from merlin.perf.historical_reference import reference_summary


@pytest.fixture(autouse=True)
def refuse_processes_and_listeners(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("installed assembly fixtures must not launch tools or listeners")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "bind", refused)


@pytest.fixture
def launch(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "portfolio_analysis_fixtures", Path(__file__).with_name("portfolio_analysis_fixtures.py")
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    analyzer = helper.SyntheticAnalyzer()
    monkeypatch.setattr(EA, "analyze_whole_model_emission", analyzer)
    baseline, candidate, shared = [tmp_path / name for name in ("baseline", "candidate", "shared")]
    for path in (baseline, candidate):
        path.mkdir()
        (path / "compiler.py").write_text("from merlin.helper import VALUE\n\ndef schedule():\n    return VALUE\n")
        (path / "version.txt").write_text("0")
        (path / "manifest.yaml").write_text("components:\n  emit: [compiler.py]\n")
    shared.mkdir()
    (shared / "__init__.py").write_text("")
    (shared / "helper.py").write_text("VALUE = 1\n")
    model = tmp_path / "model"
    model.mkdir()
    (model / "capsule.yaml").write_text("interface_mlir: capsule.interface.mlir\n")
    (model / "capsule.interface.mlir").write_text("module {}\n")
    sentinel = StageE2ESentinel("model", str(model), str(model), C.exact_tree_record(model)["sha256"], (), ())
    resources = tmp_path / "contract"
    for relative in HP.RESOURCE_FILES:
        path = resources / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"synthetic_resource": relative}))
    controller = tmp_path / "selected-controller.py"
    controller.write_text("SYNTHETIC_CONTROLLER = True\n")
    descriptor = tmp_path / "target.json"
    descriptor.write_text('{"synthetic_target":true}')
    kwargs = dict(
        baseline=baseline,
        baseline_sha256=hash_tree(baseline)["sha256"],
        sentinel=sentinel,
        target="synthetic-unregistered-target",
        target_sha256=C.sha256_file(descriptor),
        target_descriptor=descriptor,
        output=tmp_path / "run",
        analyzer=analyzer,
        controller_source=controller,
        contract_root=resources,
        compiler_shared_source_root=shared,
        prior_shared_source_relative=Path("compiler"),
        prior_shared_source_fallback=shared,
    )
    return SimpleNamespace(kwargs=kwargs, candidate=candidate, analyzer=analyzer, root=tmp_path)


def test_constructor_composes_one_concrete_state_and_exact_experiment_record(launch):
    experiment = GlobalPerfExperiment(**launch.kwargs)
    assert experiment.revision_session.inputs is experiment.inputs
    assert experiment.revision_session.journal is experiment.revisions
    assert experiment.revision_session.edit_authority is experiment.edit_authority
    assert experiment.analysis.session is experiment.revision_session
    assert experiment.probes.analysis is experiment.analysis
    assert experiment.static_analysis_import.analysis is experiment.analysis
    path = experiment.output / "experiment.json"
    document = C.mapping_file(path)
    assert path.read_bytes() == C.canonical_json(document)
    assert not path.stat().st_mode & 0o222
    assert document["schema"] == "global_perf_experiment_v1"
    assert document["launch_scope"] == "development_readiness_only"
    assert document["phase1_action"] == "reuse_exact_snapshot"
    assert document["portfolio"] == experiment.inputs.portfolio_identity
    assert document["portfolio_sha256"] == C.document_sha256(document["portfolio"])
    assert document["host_verification_policy"] == HP.build_record(
        controller_source=launch.kwargs["controller_source"], contract_root=launch.kwargs["contract_root"]
    )
    assert document["baseline_compiler_dependencies"] == experiment.inputs.baseline_dependencies
    assert document["full_model_simulation_allowed"] is False
    assert not launch.analyzer.calls


@pytest.mark.parametrize("failure", ["baseline", "target", "controller", "timeout", "existing_output"])
def test_constructor_refusal_does_not_create_or_overwrite_output(launch, failure):
    output = launch.kwargs["output"]
    if failure == "baseline":
        launch.kwargs["baseline_sha256"] = "0" * 64
    elif failure == "target":
        launch.kwargs["target_sha256"] = "bad"
    elif failure == "controller":
        alias = launch.root / "controller-link.py"
        alias.symlink_to(launch.kwargs["controller_source"])
        launch.kwargs["controller_source"] = alias
    elif failure == "timeout":
        launch.kwargs["timeout_s"] = 0
    else:
        output.mkdir()
        (output / "preserve").write_text("existing user evidence")
    with pytest.raises(ValueError):
        GlobalPerfExperiment(**launch.kwargs)
    if failure == "existing_output":
        assert [path.name for path in output.iterdir()] == ["preserve"]
    else:
        assert not output.exists()
    assert not launch.analyzer.calls


def test_early_input_admission_precedes_host_source_discovery(launch, monkeypatch):
    launch.kwargs["baseline_sha256"] = "0" * 64
    monkeypatch.setattr(HP, "build_record", lambda **kwargs: pytest.fail("policy observed before invalid baseline"))
    with pytest.raises(ValueError, match="compiler digest"):
        GlobalPerfExperiment(**launch.kwargs)
    assert not launch.kwargs["output"].exists()


def test_direct_constructor_analysis_probe_seal_and_installed_consumer(launch):
    from merlin.perf.activity_schedule import ActivityEvent
    from merlin.perf.execution_policy import SimulationBudget, WarmComputeReceipt, WarmProfileContract
    from merlin.perf.mechanism_probe import MechanismEvidence, ProbeObservation, derive_mechanism_signature
    from merlin.xdsl_dialects.lowering.global_plan import ValueRepresentation

    experiment = GlobalPerfExperiment(**launch.kwargs)
    row = experiment.analysis.analyze(launch.candidate, hypothesis="synthetic complete graph")
    binding = experiment.revision_session.current_probe_binding(launch.candidate)
    signature = derive_mechanism_signature(
        representations=[ValueRepresentation("local", "row-major", "i8", "blocked")],
        events=[
            ActivityEvent("load", "dma", "movement", 0, movement_bytes=128),
            ActivityEvent("compute", "mesh", "compute", 0, depends_on=("load",)),
        ],
        capacity_regime={"scratch": "fits"},
        tile_shape=(8, 8),
        edge_cases=("aligned",),
        repetition_semantics="independent tiles",
        instruction_semantics=[{"operation": "load"}, {"operation": "compute"}],
    )
    model = MechanismEvidence(
        binding,
        signature,
        row["analysis"]["emission"]["candidate_lowered_sha256"],
        100,
        "synthetic host model extraction",
    )
    probe = MechanismEvidence(
        binding, signature, sha256_bytes(b"synthetic-probe"), 2, "synthetic host probe extraction"
    )
    executed = []

    def execute(*, timeout_s):
        executed.append(timeout_s)
        return ProbeObservation(
            probe,
            WarmComputeReceipt("synthetic", 100, WarmProfileContract(), "synthetic counter observation"),
            0.1,
            1,
            probe.artifact_digest,
        )

    receipt = experiment.probes.measure_probe(
        launch.candidate,
        admission_inputs=dict(
            model=model,
            probe=probe,
            descriptor={"kind": "kernel"},
            budget=SimulationBudget(30, 30),
            estimated_cycles=100,
            measured_cycles_per_second=100,
        ),
        execute=execute,
    )
    assert executed and receipt["scope"] == "mechanism_probe_only"
    assert receipt["target_timing_authority"] is None
    sealed = experiment.revision_session.seal(launch.candidate)
    document = PC.consume_global_candidate(
        sealed,
        context=PC.CheckpointVerificationContext(
            experiment.inputs.host_policy, experiment.inputs.compiler_shared_source_root
        ),
    )
    assert len(document["probe_receipts"]) == 1
    assert document["global_speedup_proven"] is False
    assert len(launch.analyzer.calls) == 1


def test_proposal_run_coordinates_detached_history_and_exact_revisions(launch):
    experiment = GlobalPerfExperiment(**launch.kwargs)
    observed = []

    def propose(history):
        observed.append(len(history))
        if not history:
            return launch.candidate, "initial synthetic proposal"
        if len(history) == 1:
            history[0]["hypothesis"] = "caller cannot rewrite journal"
            (launch.candidate / "version.txt").write_text("1")
            return launch.candidate, "second synthetic proposal"
        return None

    records = experiment.run(propose, iterations=3)
    assert observed == [0, 1, 2]
    assert [row["iteration"] for row in records] == [0, 1]
    assert records[0]["hypothesis"] == "initial synthetic proposal"
    records[0]["hypothesis"] = "returned copy"
    assert experiment.revisions.iterations[0]["hypothesis"] == "initial synthetic proposal"
    with pytest.raises(ValueError, match="positive"):
        experiment.run(propose, iterations=0)


@pytest.mark.parametrize("mutation", ["controller", "contract", "shared"])
def test_selected_source_drift_refuses_analysis_before_analyzer(launch, mutation):
    experiment = GlobalPerfExperiment(**launch.kwargs)
    path = {
        "controller": launch.kwargs["controller_source"],
        "contract": launch.kwargs["contract_root"] / HP.RESOURCE_FILES[0],
        "shared": launch.kwargs["compiler_shared_source_root"] / "helper.py",
    }[mutation]
    path.write_text("CHANGED = True\n")
    with pytest.raises(ValueError):
        experiment.analysis.analyze(launch.candidate, hypothesis="refused drift")
    assert not launch.analyzer.calls and not experiment.revisions.iterations


@pytest.mark.parametrize("explicit_guidance", [False, True])
def test_edit_catalog_and_work_order_freeze_preserve_guidance_selection(launch, explicit_guidance):
    resources = launch.kwargs["contract_root"]
    manifest = resources / "schemas/manifest.schema.json"
    if explicit_guidance:
        manifest.write_text(
            json.dumps(
                {
                    "properties": {
                        "optimization_surfaces": {
                            "items": {
                                "properties": {
                                    "scope": {"enum": ["selected_scope"]},
                                    "effects": {"items": {"enum": ["selected_effect"]}},
                                }
                            }
                        }
                    }
                }
            )
        )
        launch.kwargs["guidance_contract"] = resources
    experiment = GlobalPerfExperiment(**launch.kwargs)
    contract = phase2_edit_contract.seal(
        {
            "schema": "compiler_edit_contract_v1",
            "existing_symbols": [{"surface_id": "schedule", "path": "compiler.py", "symbol": "schedule"}],
            "helper_extensions": [],
        }
    )
    declaration = {
        "id": "schedule",
        "path": "compiler.py",
        "symbol": "schedule",
        "scope": "selected_scope" if explicit_guidance else "codegen",
        "effects": ["selected_effect"] if explicit_guidance else ["movement"],
        "cca_axes": ["dispatch.dma_overlap"],
        "mechanism": "Synthetic scheduling",
        "emitted_delta": "Changed order",
        "validation": "Source dependencies",
        "abandonment": "Invalid dependencies",
    }
    binding = experiment.freeze_edit_scope(launch.candidate, contract, host_surface_declarations=[declaration])
    assert experiment.edit_authority.guidance_contract == (resources if explicit_guidance else None)
    assert ("guidance_contract_sha256" in binding) is explicit_guidance

    def seal(body):
        return {**body, "sha256": C.document_sha256(body)}

    catalog = seal(
        {
            "schema": "compiler_mechanism_catalog_v1",
            "contract_sha256": contract["sha256"],
            "mechanisms": [
                {"id": "schedule", "selectors": [{"kind": "function", "path": "compiler.py", "symbol": "schedule"}]}
            ],
        }
    )
    catalog_path = launch.root / "catalog.json"
    catalog_path.write_bytes(C.canonical_json(catalog))
    catalog_path.chmod(0o444)
    frozen = experiment.freeze_mechanism_catalog(catalog_path, C.sha256_file(catalog_path))
    assert frozen["catalog"] == catalog
    candidate_sha = binding["initial_candidate_sha256"]
    portfolio = experiment.inputs.portfolio_identity
    work = seal(
        {
            "schema": "host_prepared_mechanism_work_order_v1",
            "mechanism_id": "schedule",
            "catalog_sha256": catalog["sha256"],
            "contract_sha256": contract["sha256"],
            "initial_candidate_sha256": candidate_sha,
            "round_start_candidate_sha256": candidate_sha,
            "portfolio_sha256": experiment.inputs.portfolio_identity_sha256,
            "ordered_portfolio": portfolio["members"],
            "status": "ready_for_authoring",
            "source_operation_ids": [],
            "portfolio_site_bindings": [
                {
                    **member,
                    "compiler_sha256": candidate_sha,
                    "source_sha256": "a" * 64,
                    "plan_digest": "b" * 64,
                    "candidate_command_buffer_sha256": "c" * 64,
                    "candidate_lowered_sha256": "d" * 64,
                    "status": "ready",
                    "inventory": {},
                    "source_operation_ids": [0],
                    "chains": [],
                }
                for member in portfolio["members"]
            ],
        }
    )
    work_path = launch.root / "work.json"
    work_path.write_bytes(C.canonical_json(work))
    work_path.chmod(0o444)
    bound_work = experiment.freeze_mechanism_work_order(work_path, C.sha256_file(work_path), candidate=launch.candidate)
    assert bound_work["work_order"] == work
    assert not Path(bound_work["frozen_path"]).stat().st_mode & 0o222
    assert not launch.analyzer.calls


@pytest.mark.parametrize("inside_workspace", [False, True])
def test_historical_reference_stages_only_outside_writable_workspace(launch, inside_workspace):
    workspace = launch.root / "workspace"
    workspace.mkdir()
    source = (workspace if inside_workspace else launch.root) / "historical.json"
    source.write_bytes(
        C.canonical_json({"schema": "historical_reference_bundle_v1", "records": [], "summary": reference_summary([])})
    )
    launch.kwargs.update(historical_reference_path=source, historical_reference_sha256=C.sha256_file(source))
    experiment = GlobalPerfExperiment(**launch.kwargs)
    control = launch.root / "control"
    control.mkdir()
    if inside_workspace:
        with pytest.raises(ValueError, match="writable author workspace"):
            experiment.stage_historical_reference(control, workspace=workspace)
        assert not list(control.iterdir())
    else:
        experiment.stage_historical_reference(control, workspace=workspace)
        staged = control / "historical_reference.json"
        assert staged.read_bytes() == source.read_bytes()
        assert not staged.stat().st_mode & 0o222
