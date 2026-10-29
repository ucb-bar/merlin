"""Synthetic public/protected and CPU fidelity admission; no engine execution."""

import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import evaluation_fidelity as F
from merlin_experiments.phase2.component_compiled_arms import CompiledStaticArmFeatures
from merlin_experiments.phase2.contracts import StageGateError, document_sha256, sha256_file

from merlin.perf.component_cost import COMPLETE_STAGES, ComponentCostScope
from merlin.perf.execution_policy import require_probe_execution


def registry(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"schema": F.REGISTRY_SCHEMA, "public": ["a" * 64], "protected": ["b" * 64]}))
    return F.ProgramAccessRegistry(path, sha256_file(path))


def test_compiled_static_sites_leave_complete_cost_and_physical_features_unknown(tmp_path):
    compiler, workspace = tmp_path / "compiler", tmp_path / "arm"
    compiler.mkdir()
    (compiler / "compiler.py").write_text("# synthetic compiler, never executed\n")
    (workspace / "features").mkdir(parents=True)
    (workspace / "features" / "static_features.json").write_text("{}\n")
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("synthetic target\n")
    scope = ComponentCostScope("1" * 64, "2" * 64, "3" * 64)
    owner = CompiledStaticArmFeatures(
        SimpleNamespace(scope=scope, baseline_admission=SimpleNamespace(target_descriptor=descriptor)),
        None,
        ("logical_load_sites",),
        "4" * 64,
    )
    observation = owner._observation(
        {"features": {"load_sites": 7}, "products": [{"sha256": "5" * 64}], "selection": {}},
        compiler,
        SimpleNamespace(source_sha256="6" * 64),
        SimpleNamespace(capsules_sha256="7" * 64),
        workspace,
    )
    assert observation.functional_status == observation.legality_status == "UNKNOWN"
    assert observation.cold == observation.warm
    assert observation.cold[0].stages == COMPLETE_STAGES
    context = observation.cold[0].context
    assert context["compiled_static"]["load_sites"] == 7
    assert context["physical_memory_traffic"] is None
    assert context["dynamic_instruction_counts"] is None
    assert context["cold_warm_costs"] is None


def test_private_registry_keeps_protected_and_unknown_programs_out_of_public_development(tmp_path):
    selected = registry(tmp_path)
    selected.require_public("a" * 64)
    with pytest.raises(StageGateError, match="before compiler freeze"):
        selected.require_public("b" * 64)
    with pytest.raises(StageGateError, match="original public source registry"):
        selected.require_public("c" * 64)
    selected.path.write_text(json.dumps({"schema": F.REGISTRY_SCHEMA, "public": ["b" * 64], "protected": []}))
    with pytest.raises(StageGateError, match="changed"):
        selected.require_public("b" * 64)


def test_registry_does_not_admit_overlapping_or_repeated_source_membership(tmp_path):
    selected = registry(tmp_path)
    for public, protected in ((["a" * 64], ["a" * 64]), (["a" * 64, "a" * 64], [])):
        selected.path.write_text(json.dumps({"schema": F.REGISTRY_SCHEMA, "public": public, "protected": protected}))
        with pytest.raises(StageGateError):
            F.ProgramAccessRegistry(selected.path, sha256_file(selected.path)).verify()


@pytest.mark.parametrize("engine", F.CPU_ENGINES)
def test_cpu_engine_identity_never_implies_model_numerical_or_timing_qualification(engine):
    result = F.inspect_program_fidelity(
        scope="whole_model", engine=engine, phase="public_development", public=True, configuration_matched=True
    )
    assert result["execution_substrate"] == "CPU"
    assert result["numerical_status"] == result["timing_status"] == result["physical_traffic_status"] == "UNKNOWN"
    assert result["promotion"] == "NONE"
    assert any("component qualification" in row for row in result["reasons"])
    assert any("cycle equivalence" in row for row in result["reasons"])


def test_public_cpu_policy_preserves_legacy_full_model_probe_refusal():
    for descriptor in (
        {"kind": "model"},
        {"operation": {"op": "model"}},
        {"performance": {"measurement_scope": "full_model"}},
    ):
        with pytest.raises(ValueError, match="compile-only search objective"):
            require_probe_execution(descriptor)
    result = F.inspect_program_fidelity(scope="whole_model", engine="spike", phase="public_development", public=False)
    assert any("protected program unavailable" in reason for reason in result["reasons"])
    assert any("not physical timing or traffic" in reason for reason in result["reasons"])


def test_selected_engine_reopens_actual_backend_and_binary_bytes(tmp_path, monkeypatch):
    binary = tmp_path / "ordinary-cpu-engine"
    binary.write_bytes(b"synthetic engine identity; never executed")
    identity = {
        "engine": "gsim",
        "binary": {"path": str(binary), "sha256": sha256_file(binary)},
        "selection_sha256": "d" * 64,
    }
    calls = []

    def observed(engine):
        calls.append(engine)
        return identity

    monkeypatch.setattr(
        "merlin.runtime.backends.base.get_backend", lambda target: SimpleNamespace(execution_engine_identity=observed)
    )
    selected = F.CpuEngineSelection("synthetic", "gsim", document_sha256(identity))
    assert selected.identity() == identity
    binary.write_bytes(b"changed engine")
    with pytest.raises(StageGateError, match="changed"):
        selected.identity()
    assert calls == ["gsim", "gsim"]


def test_saved_callback_and_pass_summary_cannot_supply_ordinary_public_execution(tmp_path):
    with pytest.raises(StageGateError, match="fixed ordinary live execution"):
        F.evaluate_public_program_cpu(
            execution=SimpleNamespace(execute=lambda **kwargs: {"numeric": "pass"}),
            selection=object(),
            registry=registry(tmp_path),
            compiler=tmp_path,
            member=object(),
            corpus=object(),
            workspace=tmp_path / "work",
            timeout_s=1,
        )
    with pytest.raises(StageGateError, match="exact ordinary execution"):
        CompiledStaticArmFeatures(SimpleNamespace(verify=lambda: "qualified"), object(), ("count",), "a" * 64).verify()


def test_configuration_pins_remain_required_after_matching_selection(tmp_path):
    paths = [tmp_path / name for name in ("memory", "cache", "clock")]
    for path in paths:
        path.write_text("public final configuration")
    config = F.CpuTimingConfiguration(*((str(path), sha256_file(path)) for path in paths))
    before = config.sha256
    assert config.sha256 == before
    paths[1].write_text("changed cache configuration")
    with pytest.raises(StageGateError, match="changed"):
        config.record()
