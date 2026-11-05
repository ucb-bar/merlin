"""Public normal routing and refusal controls; no engine or timing qualification."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import public_program_measurement as P

from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.public_program_timing import candidate_execution_requested, is_public_program, verify_selection


@pytest.fixture
def capsule():
    return {
        "name": "public_control",
        "kind": "model",
        "label": "dev",
        "semantic": {"must_accelerate": False},
        # Phase 0 publishes only capsules with accelerator placement; this one keeps no compute on the
        # host, so the candidate-whole-program route may score its complete call.
        "software_screen": {"status": "admitted", "placements": {"accelerator": 1}},
        "performance": {
            "gate": {
                "source_execution_basis": {
                    "schema": "merlin.phase0.public_program_source_basis.v1",
                    "kind": "public_program",
                    "boundary": "complete_program_call",
                    "capture_sha256": "a" * 64,
                    "receipt_sha256": "b" * 64,
                }
            }
        },
    }


@pytest.fixture
def rtl(tmp_path):
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps({"facts": {"source": {"config": "synthetic-config"}}}))
    return {"rtl_facts": {"path": str(facts), "sha256": hashlib.sha256(facts.read_bytes()).hexdigest()}}


def test_explicit_selection_uses_actual_normal_facts_and_unchanged_host_demand(capsule, rtl):
    selected = P.selection(capsule, rtl)
    assert verify_selection(capsule, selected)["complete_call_timing"] is True
    assert candidate_execution_requested({"public_program_execution": selected}, capsule)
    assert not candidate_execution_requested({}, capsule)
    assert capsule["semantic"]["must_accelerate"] is False
    assert not candidate_execution_requested({}, None)


@pytest.mark.parametrize("defect", ["validation", "protected", "source_role", "boundary", "facts", "callback"])
def test_unknown_source_or_changed_selection_has_no_public_timing(capsule, rtl, defect):
    selected = P.selection(capsule, rtl)
    if defect in ("validation", "protected"):
        capsule["label"] = defect
    elif defect == "source_role":
        capsule["performance"]["gate"]["source_execution_basis"]["kind"] = "validation_program"
    elif defect == "boundary":
        capsule["performance"]["gate"]["source_execution_basis"]["boundary"] = "precomputed_operands"
    elif defect == "facts":
        selected["rtl_facts"]["sha256"] = "f" * 64
    else:
        selected["executor"] = "arbitrary_callback"
    with pytest.raises(ValueError):
        verify_selection(capsule, selected)
    if defect not in ("facts", "callback"):
        assert not is_public_program(capsule)
        assert P.selection(capsule, rtl) is None


def test_saved_passing_grade_or_legacy_host_graph_cannot_supply_timing(capsule, rtl):
    selected = P.selection(capsule, rtl)
    with pytest.raises(ValueError, match="host-graph"):
        P.project(capsule, {"status": "pass", "public_program_execution": selected}, target="synthetic")
    with pytest.raises(ValueError, match="complete passing"):
        P.project(capsule, {"status": "incomplete"}, target="synthetic")


def test_original_byte_pin_reopens_and_rejects_product_or_link_drift(tmp_path):
    source = tmp_path / "product"
    source.write_bytes(b"actual private product")
    pin = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    assert P._reopen(pin) == source
    source.write_bytes(b"mutated product")
    with pytest.raises(ValueError, match="changed"):
        P._reopen(pin)
    alias = tmp_path / "alias"
    alias.symlink_to(source)
    with pytest.raises(ValueError, match="indirect"):
        P._reopen({**pin, "path": str(alias)})


def test_host_only_public_selection_uses_original_grade_and_preserves_required_gates(
    capsule, rtl, tmp_path, monkeypatch
):
    selected = P.selection(capsule, rtl)
    observed = {}
    monkeypatch.setattr(CR, "model_budget_seconds", lambda: None)
    monkeypatch.setattr(CR, "make_run_paths", lambda *a, **k: SimpleNamespace(run_path=tmp_path / "run"))
    monkeypatch.setattr(CR, "_grade_model_capsule", lambda *a, **k: pytest.fail("legacy host graph ran"))
    monkeypatch.setattr("merlin.targetgen.provenance.toolchain_shas", lambda _target: {"source": "a" * 40})

    def original_grade(member, **kwargs):
        observed.update(kwargs)
        assert member["semantic"] == {"must_accelerate": False}
        return {"status": "incomplete", "kind": "model"}

    monkeypatch.setattr(CR, "_grade_model_capsule_unlocked", original_grade)
    config = SimpleNamespace(
        target="synthetic",
        suite="control",
        dtype="f32",
        force_match_policy=None,
        fourth_output_name="lowered.llvm.mlir",
    )
    result = CR.run_capsule(
        copy.deepcopy(capsule),
        tmp_path,
        runs_root=tmp_path,
        target="synthetic",
        config=config,
        oracle_adapters={},
        model_execution_selection=selected,
    )
    assert result["status"] == "incomplete"
    assert observed["candidate_context"]["public_program_execution"] == selected
    assert observed["budget_s"] is None
    assert observed["candidate_context"]["package_prebuilt"] is False
    assert "executor" not in observed["candidate_context"]


def test_only_original_public_source_can_declare_zero_device_obligations(capsule, tmp_path, monkeypatch):
    import yaml

    from merlin.targetgen import target_experiment
    from merlin.targetgen.native_model_execution import NativeModelExecutionError, _frozen_model_policy

    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: synthetic\n")
    corpus = tmp_path / "corpus"
    (corpus / "isa").mkdir(parents=True)
    (corpus / "MANIFEST.yaml").write_text(
        "instruction_policy:\n  status: resolved\n  prohibited_instruction_roles: []\n"
    )
    source = tmp_path / "source"
    source.mkdir()
    capsule.update(expected={"instruction_classes": []}, required_oracle_tiers=["L0", "L1", "L2", "L3"])
    (source / "expected_instruction_coverage.yaml").write_text("instruction_classes: []\n")
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(descriptor))
    monkeypatch.setattr(
        target_experiment,
        "load_target_experiment",
        lambda _p: SimpleNamespace(target="synthetic", capsule_corpus=corpus / "isa",
                                  descriptor_sha256=hashlib.sha256(descriptor.read_bytes()).hexdigest()),
    )
    (source / "capsule.yaml").write_text(yaml.safe_dump(capsule))
    policy = _frozen_model_policy(source, target="synthetic")
    assert policy["public_zero_device_demand"] is True
    assert policy["expected_instruction_classes"] == []
    assert policy["required_tiers"] == ["L0", "L1", "L2", "L3"]
    for changed in ({"label": "validation"}, {"semantic": {"must_accelerate": True}}):
        (source / "capsule.yaml").write_text(yaml.safe_dump({**capsule, **changed}))
        with pytest.raises(NativeModelExecutionError, match="instruction coverage"):
            _frozen_model_policy(source, target="synthetic")


def test_unselected_clock_refuses_before_original_native_build(tmp_path, monkeypatch):
    from merlin.runtime.backends import base as backends
    from merlin.targetgen import native_model_execution as E
    from merlin.targetgen.contract import compile as C

    source, bundle = tmp_path / "source", tmp_path / "bundle"
    source.mkdir()
    bundle.mkdir()
    monkeypatch.setattr(backends, "get_backend", lambda _t: SimpleNamespace())
    monkeypatch.setattr(C, "compile_lowered_to_elf", lambda *a, **k: pytest.fail("native build reached"))
    result = E.execute_candidate_model(
        command_buffer={"kernel_abi": {"kind": "whole_program"}},
        lowered_mlir_text="module {}",
        capsule_dir=source,
        capture_bundle=bundle,
        target="synthetic",
        out_dir=tmp_path / "output",
        complete_call_timing=True,
    )
    assert result["status"] == "incomplete"
    assert "counter/clock" in result["failure"]["detail"]


def test_spike_selection_requires_explicit_ceiling_and_never_selects_timing(capsule, rtl):
    with pytest.raises(ValueError):
        P.selection(capsule, rtl, simulator="spike")
    capsule["oracle_tier_ceiling"] = "L2"
    selected = P.selection(capsule, rtl, simulator="spike")
    assert verify_selection(capsule, selected) == {
        "simulator": "spike",
        "rtl_facts": rtl["rtl_facts"]["path"],
        "board_config": "synthetic-config",
        "complete_call_timing": False,
        "oracle_tier_ceiling": "L2",
    }
    selected["complete_call_timing"] = True
    with pytest.raises(ValueError):
        verify_selection(capsule, selected)


def test_native_numerical_projection_leaves_full_grade_pending(capsule, rtl):
    from merlin.targetgen.native_model_policy import attach_numerical_screen

    capsule["oracle_tier_ceiling"] = "L2"
    native = {"simulator": "spike", "oracle_tier_ceiling": "L2", "numeric": {"status": "pass"}}
    result = {
        "public_program_execution": P.selection(capsule, rtl, simulator="spike"),
        "candidate_native_execution": native,
        "status": "incomplete",
    }
    check = {
        "status": "incomplete",
        "candidate_native_model_check": {
            "violations": ["candidate_required_tiers_unverified"],
            "candidate_required_tiers": {
                "required_tiers": ["L0", "L1", "L2", "L3"],
                "tiers": {tier: {"status": "pass"} for tier in ("L0", "L1", "L2")},
            },
        },
    }
    attach_numerical_screen(result, capsule, check)
    assert result["status"] == "incomplete"
    assert result["candidate_numerical_screen"]["timing_cycles"] is None
    assert result["numeric"] == {"status": "pass"}
    assert result["tiers"]["L3"]["status"] == "pending"
    assert all(row["derived_from_rtl"] is False and row["cycle_accurate"] is False for row in result["tiers"].values())
    check["candidate_native_model_check"]["violations"].append("candidate_completed_dispatch_unverified")
    attach_numerical_screen(result, capsule, check)
    assert result["candidate_numerical_screen"]["status"] == "unverified"
    assert result["tiers"]["L2"]["status"] == "unverified"


def test_numerical_caller_needs_no_clock_selection_and_restores_scope():
    from merlin.runtime.complete_call_timing import from_contract, optional_timing_for_numerics

    incomplete = {"logical_harness": {"complete_call_timing": {"schema": "invalid"}}}
    with pytest.raises(ValueError):
        from_contract(incomplete)
    with optional_timing_for_numerics():
        assert from_contract(incomplete) is None
    with pytest.raises(ValueError):
        from_contract(incomplete)


def test_fast_member_passes_original_spike_selection_and_exact_grade_to_product_join(
    capsule, rtl, tmp_path, monkeypatch
):
    """Original caller linkage only; all execution/product bodies are spies."""
    import time
    from contextlib import nullcontext
    from pathlib import Path

    from merlin_experiments.phase2 import ordinary_fast_feedback as F

    capsule["required_oracle_tiers"] = ["L0", "L1", "L2", "L3"]
    candidate, member = tmp_path / "candidate", tmp_path / "member"
    candidate.mkdir()
    member.mkdir()
    grade = {"status": "incomplete", "tiers": {tier: {"status": "pass"} for tier in ("L0", "L1", "L2")}}
    calls = []
    monkeypatch.setattr(F.CR, "load_capsule", lambda *a, **kw: copy.deepcopy(capsule))
    monkeypatch.setattr(F.CR, "simulator_adapter", lambda engine, target: (engine, target))
    monkeypatch.setattr(F.PC, "package_sandbox_policy", lambda *a: "original-selected-box")
    monkeypatch.setattr(F.PC, "boxed_entrypoints", lambda policy: nullcontext())
    monkeypatch.setattr(F, "hash_tree", lambda path: {"sha256": "a" * 64})
    monkeypatch.setattr(F, "exact_tree_record", lambda path: {"sha256": "b" * 64})

    def run(selected_capsule, source, **kwargs):
        assert source == candidate
        assert selected_capsule["required_oracle_tiers"] == capsule["required_oracle_tiers"]
        assert selected_capsule["oracle_tier_ceiling"] == "L2"
        assert kwargs["oracle_adapters"] == {"L2": ("spike", "synthetic")}
        assert kwargs["model_execution_selection"] == P.selection(selected_capsule, rtl, simulator="spike")
        assert kwargs["model_execution_selection"]["complete_call_timing"] is False
        calls.append("original_capsule_call")
        return grade

    def join(root, **kwargs):
        assert kwargs["original_grade"] is grade
        calls.append("same_grade_product_join")
        return {"status": "UNKNOWN"}

    monkeypatch.setattr(F.CR, "run_capsule", run)
    monkeypatch.setattr(F, "compiled_observation", join)
    if hasattr(F, "observe_passed_census"):
        monkeypatch.setattr(F, "observe_passed_census", lambda *a, **kw: {"status": "UNKNOWN"})
        monkeypatch.setattr(F.RTL_FACTS, "observed_facts", lambda *a: nullcontext())
    policy = SimpleNamespace(
        _contract_root=tmp_path,
        target_experiment=SimpleNamespace(target="synthetic"),
        feedback_evaluator=SimpleNamespace(rtl_identity=rtl, rtl_facts_path=Path(rtl["rtl_facts"]["path"])),
    )
    result = F._member_arm(
        policy=policy,
        revision=SimpleNamespace(source=candidate, source_sha256="a" * 64),
        member=SimpleNamespace(source_dir=member, source_sha256="b" * 64),
        workspace=tmp_path / "work",
        deadline=time.monotonic() + 20,
        workers=1,
    )
    assert calls == ["original_capsule_call", "same_grade_product_join"]
    assert result["pending_tiers"] == ["L3"] and result["timing_cycles"] is None


def test_original_execution_observer_records_only_actual_python_dispatch(tmp_path):
    from merlin.common import invocation_record as I
    from merlin.targetgen.native_model_policy import observed_execution

    elf, engine = tmp_path / "input", tmp_path / "selected-engine-bytes"
    elf.write_bytes(b"synthetic input identity")
    engine.write_bytes(b"synthetic dependency identity")
    calls = []

    def original_call(path, *, simulator, timeout):
        calls.append((path, simulator, timeout))
        return "actual Python control return\n"

    console, pin = observed_execution(
        SimpleNamespace(run_elf=original_call),
        elf,
        simulator="spike",
        citation={"binary": {"path": str(engine)}},
        directory=tmp_path / "outputs",
        timeout=1,
        capture_kwargs={},
    )
    record = I.verify(pin["path"])
    assert console == "actual Python control return\n"
    assert calls == [(elf, "spike", 1)]
    assert record["kind"] == "python_call" and record["outcome"] == "returned"
    assert record["stage"] == "execution"
    assert any(row["path"] == str(engine) for row in record["dependencies"])
    assert any(row["path"] == str(elf) for row in record["inputs"])


def test_numerical_native_dispatch_never_queries_or_runs_rtl(tmp_path, monkeypatch):
    """Routing spies supply no real compiler/engine/ELF or grade qualification."""
    from merlin.compile import model_execution_inputs as M
    from merlin.runtime import route_quality as Q
    from merlin.runtime.backends import base as B
    from merlin.targetgen import bundle_harness as H
    from merlin.targetgen import native_model_execution as E
    from merlin.targetgen import oracle_policy as O
    from merlin.targetgen.contract import compile as C

    source, bundle = tmp_path / "source", tmp_path / "bundle"
    source.mkdir()
    bundle.mkdir()
    (source / "capsule.yaml").write_text("numeric_policy: {compare: exact_int}\n")
    for name in ("golden.yaml",):
        (source / name).write_text("synthetic byte identity\n")
    for name in ("model.mlir", "weights.safetensors", "weights.safetensors.manifest.json", "inputs.npz"):
        (bundle / name).write_text("synthetic byte identity\n")
    calls = []

    def original_control_call(elf, *, simulator, timeout):
        calls.append(simulator)
        return "OUT out 1 1 1\nDONE\n"

    backend = SimpleNamespace(run_elf=original_control_call, parse_output=lambda _c: ({"out": [[1]]}, {}))
    recipe = SimpleNamespace(require_kernel_stack_frame=lambda: SimpleNamespace(entry_symbol="control"))
    service = SimpleNamespace(recipe=recipe, source_pins=())
    monkeypatch.setattr(B, "get_backend", lambda _t: backend)
    monkeypatch.setattr(B, "harness_build_recipe", lambda _t: recipe)
    monkeypatch.setattr(H, "is_executable_emission", lambda *a, **k: (True, "routing control"))
    monkeypatch.setattr(H, "emitted_entry_arity", lambda *a, **k: 1)
    monkeypatch.setattr(
        E, "_bind_inputs", lambda *a, **k: ({}, {"source_entry_binding": {"source_owned_mutables": []}})
    )
    monkeypatch.setattr(E, "_frozen_model_policy", lambda *a, **k: {"required_tiers": ["L0", "L1", "L2", "L3"]})
    monkeypatch.setattr(E, "_build_service_for", lambda *a, **k: service)
    monkeypatch.setattr(E, "_host_compute_report", lambda *a, **k: SimpleNamespace(to_dict=lambda: {}))
    monkeypatch.setattr(Q, "require_clean_host_compute", lambda _r: None)
    monkeypatch.setattr("merlin.targetgen.golden_store.load_golden", lambda _s: {"outputs": {"out": [[1]]}})
    monkeypatch.setattr(M, "selected_firrtl", lambda *a, **k: {"path": str(tmp_path / "facts"), "config": "control"})
    monkeypatch.setattr(M, "native_engine", lambda *a, **k: pytest.fail("RTL engine reached"))
    monkeypatch.setattr(O, "selected_l3_engine_report", lambda *a, **k: pytest.fail("RTL selection reached"))
    monkeypatch.setattr(E, "_functional_engine", lambda _t: (backend, {"engine": "spike"}, lambda: None))

    def original_compile_control(cb, text, output, **kwargs):
        assert kwargs["execution_deadline"] is not None
        output.mkdir()
        path = output / "unqualified-input"
        path.write_bytes(b"not an ELF; routing control only")
        return path

    monkeypatch.setattr(C, "compile_lowered_to_elf", original_compile_control)
    result = E.execute_candidate_model(
        command_buffer={
            "kernel_abi": {"kind": "whole_program", "args": [{"tensor": "out", "access": "write"}], "outputs": ["out"]},
            "tensors": {"out": {"shape": [1, 1], "dtype": "i32", "role": "output"}},
        },
        lowered_mlir_text="routing control",
        capsule_dir=source,
        capture_bundle=bundle,
        target="synthetic",
        out_dir=tmp_path / "output",
        simulator="spike",
        rtl_facts=tmp_path / "facts",
        board_config="control",
        complete_call_timing=False,
        oracle_tier_ceiling="L2",
    )
    assert calls == ["spike"]
    assert result["tiers"]["L2"]["numeric"]["status"] == "pass"
    assert "L3" not in result["tiers"] and "complete_call_timing" not in result
    assert result["execution_invocations"]["L2"]["sha256"]


def test_a_frozen_host_placement_routes_to_the_merlin_host_split(capsule, rtl):
    """The candidate never emits, and is never scored on, compute Phase 0 placed on the host."""
    capsule["software_screen"]["placements"] = {"accelerator": 2, "host": 1}
    assert P.selection(capsule, rtl) is None
    capsule["oracle_tier_ceiling"] = "L2"
    assert P.selection(capsule, rtl, simulator="spike") is None


@pytest.mark.parametrize("screen", [None, {"status": "admitted", "placements": {"host": 3}}])
def test_a_public_program_without_accelerator_placement_has_no_scorable_scope(capsule, rtl, screen):
    from merlin_experiments.phase2.metric_scope import MetricScopeRefusal

    capsule["software_screen"] = screen
    with pytest.raises(MetricScopeRefusal):
        P.selection(capsule, rtl)
