"""Experiment launch cannot turn a reference backend or JSON into authority."""

import json

import pytest
from component_launch_fixture import diagnostic_launch as diagnostic_launch
from component_launch_fixture import package as package
from component_launch_fixture import short_ipc_directory as short_ipc_directory
from merlin_experiments.phase1 import component_qualification as qualification
from merlin_experiments.phase2 import component_launch_inputs as launch
from merlin_experiments.phase2.contracts import StageGateError

from merlin.runtime.backends import base as backends


@pytest.fixture
def declaration(tmp_path):
    data = dict.fromkeys(launch._FIELDS)
    data.update(schema="merlin.component_launch_inputs.v1", candidate="/private/handwritten/compiler")
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("origin,support", [(None, None), (None, object()), (object(), None),
                                          ({"qualified": True}, {"independent": True}), (object(), object())])
def test_legacy_launch_refuses_before_reference_resolution_or_execution(declaration, monkeypatch, origin, support):
    def forbidden(*_args, **_kwargs):
        pytest.fail("reference backend/compiler resolution occurred before independent experiment authority")

    monkeypatch.setattr(backends, "get_backend", forbidden)
    monkeypatch.setattr(qualification, "qualify_component_compiler", forbidden)
    with pytest.raises(StageGateError, match="independent|from-scratch"):
        launch.load_component_launch_inputs(declaration, phase1_origin=origin, measurement_support=support)


def test_reading_launch_data_creates_no_candidate_or_authority(declaration):
    before = set(declaration.parent.iterdir())
    assert launch.read_component_launch_declaration(declaration)["candidate"] == "/private/handwritten/compiler"
    assert set(declaration.parent.iterdir()) == before


def _write_diagnostic(fixture):
    fixture.declaration.write_text(json.dumps(fixture.data))
    return fixture.declaration


def test_positive_assembly_reuses_issued_owners_without_running_a_compiler_or_author(diagnostic_launch, monkeypatch):
    from merlin_experiments.phase1.providers import codex_agent

    fixture = diagnostic_launch
    def forbidden(*_args, **_kwargs):
        pytest.fail("launch assembly must not run a compiler, author or backend factory")
    monkeypatch.setattr(backends, "get_backend", forbidden)
    monkeypatch.setattr(qualification, "qualify_component_compiler", forbidden)
    monkeypatch.setattr(codex_agent, "run_round", forbidden)
    before = set(fixture.declaration.parent.iterdir())
    inputs = launch.load_component_launch_inputs(
        _write_diagnostic(fixture), phase1_origin=fixture.origin, measurement_support=fixture.support,
    )
    assert inputs.candidate == fixture.execution.candidate
    assert inputs.edit_authority is fixture.execution.edit_authority
    assert inputs.qualification is fixture.admission.qualification
    assert inputs.policy.baseline_admission is fixture.admission
    assert inputs.policy.independent_runtime is fixture.support
    assert inputs.policy.component_corpus is fixture.admission.corpus
    assert inputs.sandbox_binary == fixture.origin.inputs.author_sandbox.source
    assert inputs.sandbox_binary != next(
        row.source for row in inputs.control_runtime if row.destination == "/usr/bin/bwrap"
    )
    assert set(fixture.declaration.parent.iterdir()) == before | {fixture.declaration}
    assert not inputs.stage_root.exists()
    analytical = next(arguments for role, arguments in fixture.calls if role == "analytical")
    assert analytical["calibration_adapter"] == fixture.support.qualification.calibration_adapter
    assert analytical["scope"] is fixture.support.qualification.scope
    assert analytical["dependencies"] == {}
    assert any(role == "runtime" and "rtl_executor" in arguments for role, arguments in fixture.calls)
    monkeypatch.undo()
    with pytest.raises(StageGateError, match="independently evaluated live authority"):
        launch.load_component_launch_inputs(
            fixture.declaration, phase1_origin=fixture.origin, measurement_support=fixture.support,
        )


@pytest.mark.parametrize("change", ["missing", "source", "destination", "control", "bytes", "origin"])
def test_phase2_outer_sandbox_cannot_substitute_its_fresh_origin_selection(diagnostic_launch, change):
    from dataclasses import replace

    fixture = diagnostic_launch
    inputs = launch.load_component_launch_inputs(
        _write_diagnostic(fixture), phase1_origin=fixture.origin, measurement_support=fixture.support,
    )
    selected = fixture.origin.inputs.author_sandbox
    if change == "missing":
        fixture.origin.inputs.author_sandbox = None
    elif change == "source":
        cloned = selected.source.with_name("cloned-outer")
        cloned.write_bytes(selected.source.read_bytes())
        grants = tuple(replace(row, source=cloned) if row == selected else row for row in inputs.control_runtime)
        inputs = replace(inputs, control_runtime=grants)
    elif change == "destination":
        fixture.origin.inputs.author_sandbox = replace(selected, destination="/usr/bin/bwrap")
    elif change == "control":
        inputs = replace(inputs, control_runtime=tuple(row for row in inputs.control_runtime if row != selected))
    elif change == "bytes":
        selected.source.write_text("changed outer bytes")
    else:
        inputs = replace(inputs, qualification=replace(inputs.qualification, compiler_origin=None))
    with pytest.raises(StageGateError, match="sandbox|source bytes changed"):
        inputs.sandbox_binary


@pytest.mark.parametrize("field", ["candidate", "corpus", "view", "descriptor", "source_root", "contract_root",
                                    "qualification_root", "edit_authority_root", "edit_contract"])
def test_launch_declaration_cannot_select_a_foreign_compiler_or_cohort(diagnostic_launch, field):
    fixture = diagnostic_launch
    fixture.data[field] = str(fixture.declaration.parent / "unowned-input")
    with pytest.raises(StageGateError, match="another independently admitted input"):
        launch.load_component_launch_inputs(
            _write_diagnostic(fixture), phase1_origin=fixture.origin, measurement_support=fixture.support,
        )
    assert not any(role in {"analytical", "cca"} for role, _ in fixture.calls)


@pytest.mark.parametrize("field", ["calibration_adapter", "qualification", "output", "lease_path"])
def test_launch_declaration_cannot_substitute_calibration_labels_or_feedback_outputs(diagnostic_launch, field):
    fixture = diagnostic_launch
    fixture.data["analytical"][field] = str(fixture.declaration.parent / "foreign-evidence")
    with pytest.raises(StageGateError, match="another independently admitted input"):
        launch.load_component_launch_inputs(
            _write_diagnostic(fixture), phase1_origin=fixture.origin, measurement_support=fixture.support,
        )
    assert not any(role in {"analytical", "cca"} for role, _ in fixture.calls)


@pytest.mark.parametrize("field", ["runtime", "control_runtime"])
def test_launch_declaration_cannot_add_an_external_tool_or_support_tree(diagnostic_launch, field):
    fixture = diagnostic_launch
    fixture.data[field].append({"source": "/private/reference-decoder", "destination": "/usr/bin/decoder",
                                "sha256": "1" * 64})
    with pytest.raises(StageGateError, match="tool grants differ"):
        launch.load_component_launch_inputs(
            _write_diagnostic(fixture), phase1_origin=fixture.origin, measurement_support=fixture.support,
        )


def test_launch_readiness_probe_cannot_execute_an_ungranted_program(diagnostic_launch):
    fixture = diagnostic_launch
    fixture.data["readiness"][0]["command"] = ["/usr/bin/foreign-probe"]
    with pytest.raises(StageGateError, match="outside the independently admitted tool grants"):
        launch.load_component_launch_inputs(
            _write_diagnostic(fixture), phase1_origin=fixture.origin, measurement_support=fixture.support,
        )


def test_functional_roles_cannot_enable_performance_launch(diagnostic_launch):
    fixture = diagnostic_launch
    object.__setattr__(fixture.support, "qualified_roles", ("grade", "stage_verifier"))
    with pytest.raises(StageGateError, match="lacks qualified requested roles"):
        launch.load_component_launch_inputs(
            _write_diagnostic(fixture), phase1_origin=fixture.origin, measurement_support=fixture.support,
        )


def test_launch_cannot_use_an_origin_other_than_the_actual_baseline(diagnostic_launch):
    from dataclasses import replace

    fixture = diagnostic_launch
    other_origin = replace(fixture.origin)
    with pytest.raises(StageGateError, match="exact fresh Phase 1 origin"):
        launch.load_component_launch_inputs(
            _write_diagnostic(fixture), phase1_origin=other_origin, measurement_support=fixture.support,
        )


def test_launch_private_stage_cannot_overlap_qualification_evidence(diagnostic_launch):
    fixture = diagnostic_launch
    fixture.data["stage_root"] = str(fixture.support.qualification.receipt.parent / "launch")
    with pytest.raises(StageGateError, match="private stage overlaps"):
        launch.load_component_launch_inputs(
            _write_diagnostic(fixture), phase1_origin=fixture.origin, measurement_support=fixture.support,
        )


def test_constructed_typed_owners_cannot_enable_launch(declaration):
    from dataclasses import fields

    from merlin_experiments.phase1.component_origin import FreshCompilerOrigin
    from merlin_experiments.phase2.component_runtime_authority import IndependentComponentRuntime

    origin = FreshCompilerOrigin(**dict.fromkeys(row.name for row in fields(FreshCompilerOrigin)))
    support = IndependentComponentRuntime(**dict.fromkeys(row.name for row in fields(IndependentComponentRuntime)))
    with pytest.raises(StageGateError, match="independently evaluated live authority"):
        launch.load_component_launch_inputs(declaration, phase1_origin=origin, measurement_support=support)


def test_measured_variants_cannot_join_two_candidate_edit_owners(diagnostic_launch):
    from dataclasses import replace

    fixture = diagnostic_launch
    first = fixture.support.qualification.variants[0]
    other = replace(first, execution=replace(fixture.execution))
    object.__setattr__(fixture.support.qualification, "variants", (first, other))
    with pytest.raises(StageGateError, match="different candidate edit owners"):
        launch.load_component_launch_inputs(
            _write_diagnostic(fixture), phase1_origin=fixture.origin, measurement_support=fixture.support,
        )


def test_measured_scope_cannot_differ_from_the_authorized_execution(diagnostic_launch):
    from dataclasses import replace

    fixture = diagnostic_launch
    object.__setattr__(fixture.support.qualification, "scope",
                       replace(fixture.execution.scope, timer_sha256="a" * 64))
    with pytest.raises(StageGateError, match="measurement scope"):
        launch.load_component_launch_inputs(
            _write_diagnostic(fixture), phase1_origin=fixture.origin, measurement_support=fixture.support,
        )


def test_candidate_must_still_be_the_qualified_fresh_compiler_at_launch(diagnostic_launch, monkeypatch):
    from merlin_experiments.phase2.contracts import exact_tree_record

    fixture = diagnostic_launch
    seed = exact_tree_record(fixture.execution.edit_authority.seed)["sha256"]
    source = fixture.execution.candidate / "compiler.py"
    source.write_text(source.read_text().replace("return 1", "return 3"))
    fixture.execution.edit_authority.validate_candidate(fixture.execution.candidate)
    def verify_original(self, *, candidate=None):
        if exact_tree_record(candidate)["sha256"] != seed:
            raise StageGateError("compiler changed after fresh domain qualification")
    monkeypatch.setattr(qualification.ComponentQualification, "verify", verify_original)
    with pytest.raises(StageGateError, match="changed after fresh domain qualification"):
        launch.load_component_launch_inputs(
            _write_diagnostic(fixture), phase1_origin=fixture.origin, measurement_support=fixture.support,
        )


def test_analytical_declaration_cannot_import_a_callback_factory(diagnostic_launch):
    fixture = diagnostic_launch
    fixture.data["analytical"]["factory"] = "reference_adapter:create"
    with pytest.raises(StageGateError, match="complete closed declaration"):
        launch.load_component_launch_inputs(
            _write_diagnostic(fixture), phase1_origin=fixture.origin, measurement_support=fixture.support,
        )


def test_loader_has_no_backend_or_callback_import_route():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(launch))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not any(name and ("backend_adapters" in name or name.startswith("merlin.backends")) for name in imports)
    assert not {"get_backend", "import_module", "qualify_component_compiler", "run_round"} & names
