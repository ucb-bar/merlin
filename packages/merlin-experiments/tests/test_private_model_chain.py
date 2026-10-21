"""Static import never upgrades structural, diagnostic or incomplete partitions."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1.feedback import private_model_chain as C


def test_partition_projection_preserves_original_and_uses_full_native_validator(monkeypatch):
    original = {"mode": "canonical_model_partition", "selected_model": "a", "passed": False,
                "partition_passed": True, "required_models": ["a", "b"],
                "required_programs": {"a": ["forward"], "b": ["decode"]},
                "models": [{"model": "a", "status": "pass", "checks": {"source": "real"}}]}
    seen = []
    monkeypatch.setattr(C.models, "complete", lambda record, **kwargs: seen.append((record, kwargs)) or True)
    assert C._partition_complete(original, "a", "a" * 64)
    projection, arguments = seen[0]
    assert "mode" not in projection and projection["models"] is original["models"]
    assert arguments == {"required_models": ["a"], "required_programs": {"a": ["forward"]},
                         "candidate_sha256": "a" * 64}
    assert original["passed"] is False and original["required_models"] == ["a", "b"]
    for key, value in (("mode", "prebuilt_diagnostic"), ("passed", True), ("partition_passed", False),
                       ("selected_model", "b"), ("models", [])):
        assert not C._partition_complete({**original, key: value}, "a", "a" * 64)
    monkeypatch.setattr(C.models, "complete", lambda *args, **kwargs: False)
    assert not C._partition_complete(original, "a", "a" * 64)


def test_missing_original_profile_fails_before_instruction_or_compiler_work(tmp_path, monkeypatch):
    monkeypatch.setattr(C.profile, "functional_service", lambda run: None)
    monkeypatch.setattr(C.models, "run", lambda *args, **kwargs: pytest.fail("compiler must not start"))
    with pytest.raises(ValueError, match="original compiler-profile owner"):
        C.run_static(run_dir=tmp_path / "run", descriptor=tmp_path / "descriptor", submission=tmp_path / "compiler",
                     name="a", out=tmp_path / "partition")


def test_no_numerical_or_join_credit_without_all_declared_models(tmp_path, monkeypatch):
    monkeypatch.setattr(C.models, "requirements_for", lambda descriptor: ("a", "b"))
    monkeypatch.setattr(C.models, "program_requirements_for", lambda descriptor: {"a": ("f",), "b": ("g",)})
    with pytest.raises(ValueError, match="complete declared model roster"):
        C.join({"a": tmp_path / "fake.json"}, run_dir=tmp_path, descriptor=tmp_path / "descriptor",
               candidate_sha256="a" * 64)


def test_producer_binds_actual_stage_callable_request_and_output_digest(tmp_path, monkeypatch):
    path = tmp_path / "native-static.json"
    path.write_text("{}")
    record = {"run_dir": str(tmp_path), "selected_model": "a", "candidate_tree_sha256": "a" * 64}
    call = {"kind": "python_call", "stage": "native_static_chain_partition", "callable": C.__name__ + "._compile",
            "arguments": {"run_dir": str(tmp_path), "model": "a", "candidate_tree_sha256": "a" * 64},
            "outputs": [C._pin(path)]}
    monkeypatch.setattr(C.IR, "verify", lambda invocation: call)
    C._producer(path, record, tmp_path / "invocation.json")
    fused = {**record, "host_elementwise_fusion": True}
    with pytest.raises(ValueError, match="completed original producer"):
        C._producer(path, fused, tmp_path / "invocation.json")
    selected_call = {**call, "arguments": {**call["arguments"], "host_elementwise_fusion": True}}
    monkeypatch.setattr(C.IR, "verify", lambda invocation: selected_call)
    C._producer(path, fused, tmp_path / "invocation.json")
    monkeypatch.setattr(C.IR, "verify", lambda invocation: call)
    for key, value in (("stage", "structural"), ("callable", "foreign.producer"),
                       ("arguments", {"model": "a"}), ("outputs", [])):
        changed = {**call, key: value}
        monkeypatch.setattr(C.IR, "verify", lambda invocation: changed)
        with pytest.raises(ValueError, match="completed original producer"):
            C._producer(path, record, tmp_path / "invocation.json")
    monkeypatch.setattr(C.IR, "verify", lambda invocation: call)
    path.write_text("changed")
    with pytest.raises(ValueError, match="completed original producer"):
        C._producer(path, record, tmp_path / "invocation.json")


def test_live_provider_cannot_run_before_producer_completion(tmp_path):
    partition = C.NativePartition(tmp_path / "record.json", {}, SimpleNamespace(verify=lambda: None))
    with pytest.raises(ValueError, match="producer has not completed"):
        partition.verify()


@pytest.mark.parametrize("changed", [None, "input", "static", "receipt", "fusion"])
def test_native_record_reopens_serialized_paths_and_rejects_changed_bytes(tmp_path, monkeypatch, changed):
    from merlin_experiments.phase1.feedback import private_source_freeze

    source = tmp_path / "source"
    source.write_text("original input")
    receipt = tmp_path / "baremetal_model.json"
    receipt.write_text(json.dumps({"status": "compiled", "output": {
        "elf_sha256": "e" * 64, "compilation_recipe": {"sha256": "c" * 64}}}))
    static = tmp_path / "partition.json"
    static_record = {"candidate_tree_sha256": "a" * 64, "private_spec_sha256": "s" * 64,
                     "authored_source_freeze": {}, "required_programs": {"one": ["forward"]},
                     "models": [{"checks": {"build": {"programs": [{"program": "forward",
                         "elf_sha256": "e" * 64, "compilation_recipe": {"recipe_sha256": "c" * 64}}]}}}]}
    static.write_text(json.dumps(static_record))
    record = {"schema": C.SCHEMA, "scope": "canonical_static_only", "selected_model": "one",
              **{key: static_record[key] for key in (
                  "candidate_tree_sha256", "private_spec_sha256", "authored_source_freeze")},
              "inputs": [C._pin(source)], "static_partition_report": str(static),
              "static_partition_report_sha256": C._pin(static)["sha256"], "target": "test_target",
              "instruction_path": str(tmp_path / "selection"), "receipts": {"forward": C._pin(receipt)}}
    monkeypatch.setattr(private_source_freeze, "valid_binding", lambda *args: True)
    monkeypatch.setattr(C, "_partition_complete", lambda *args: True)
    monkeypatch.setattr(C.partitions, "_selection", lambda *args, **kwargs: None)
    if changed is not None:
        if changed == "fusion":
            record["host_elementwise_fusion"] = True
        else:
            {"input": source, "static": static, "receipt": receipt}[changed].write_text("{}")
        with pytest.raises(ValueError, match="changed|incomplete"):
            C._verify_record(record)
    else:
        assert C._verify_record(record) == static_record


def test_no_execution_gate_requires_explicit_private_criterion(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import private_numerical_selection as Q

    spec = tmp_path / "private.yaml"
    spec.write_text("models: []")
    with pytest.raises(ValueError, match="explicitly selected full-chain criterion"):
        C._build_options(None, spec=spec, name="one", programs={"one": ["forward", "host_only"]})
    monkeypatch.setattr(Q, "read", lambda path: {"mode": Q.MODE})
    options = C._build_options(None, spec=spec, name="one", programs={"one": ["forward", "host_only"]})
    assert options == {"one": {p: {"readback": "full", "host_llvm_bitcode": True, "group_profile": True}
                               for p in ("forward", "host_only")}}
    fused = C._build_options(None, spec=spec, name="one", programs={"one": ["forward", "host_only"]},
                             host_elementwise_fusion=True)
    assert fused == {"one": {p: {**options["one"][p], "host_elementwise_fusion": True}
                              for p in ("forward", "host_only")}}
    with pytest.raises(ValueError, match="explicit Boolean"):
        C._build_options(None, spec=spec, name="one", programs={"one": ["forward"]}, host_elementwise_fusion=1)


def test_actual_sealed_descriptor_without_whole_model_section_can_use_selected_chain(tmp_path):
    """Operator integration fixture: reopen the actual admitted descriptor, never synthesize its old gate."""
    from merlin_experiments.phase1.feedback import private_numerical_selection as Q

    selected = os.environ.get("MERLIN_TEST_PHASE1_RELEASE_DESCRIPTOR")
    if not selected:
        pytest.skip("requires an explicit existing sealed release descriptor")
    descriptor = Path(selected)
    programs = C.models.program_requirements_for(descriptor)
    assert programs and "whole_model" not in yaml.safe_load(descriptor.read_bytes()).get("phase1_gates", {})
    gate = C.execution.gate_for(descriptor, required_programs=programs)
    assert gate is None
    pins = {}
    for key in ("models", "reference_manifest", "reference_sums"):
        path = tmp_path / key
        path.write_text("immutable fixture selection")
        pins[key] = C._pin(path)
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    spec = tmp_path / "private.yaml"
    spec.write_text(yaml.safe_dump({"numerical_qualification": {
        "schema": Q.SCHEMA, "mode": Q.MODE, "config": str(config), "static_root": str(tmp_path / "static"),
        **pins, "comparison": "quantized-reference", "output_scope": "all_declared", "float_policy": Q.POLICY,
        "kernel_exact": True, "deadline_unix": 2000000000,
    }}))
    for name, stages in programs.items():
        options = C._build_options(gate, spec=spec, name=name, programs=programs)
        assert list(options[name]) == list(stages)
        assert all(o["readback"] == "full" and o["host_llvm_bitcode"] for o in options[name].values())


def test_native_compile_separates_private_metadata_from_selected_build_output(tmp_path, monkeypatch, capsys):
    from contextlib import nullcontext

    from merlin_experiments.phase1.feedback import private_facts, private_result_format_selection

    from merlin.targetgen import target_experiment

    run = tmp_path / "selected-out/runs/new-run"
    metadata = tmp_path / "private/operator/model"
    metadata.mkdir(parents=True, mode=0o700)
    selected_build = tmp_path / "selected-out/build"
    monkeypatch.setattr(C, "build_dir", lambda: selected_build)
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda _: SimpleNamespace(target="a"))
    monkeypatch.setattr(C.models, "requirements_for", lambda _: ["one"])
    monkeypatch.setattr(C.models, "program_requirements_for", lambda _: {"one": ["forward"]})
    monkeypatch.setattr(C.execution, "gate_for", lambda *a, **k: None)
    monkeypatch.setattr(C, "_build_options", lambda *a, **k: {})
    monkeypatch.setattr(C.instructions, "prepare", lambda *a, **k: SimpleNamespace(
        require_facts=lambda _: None, service=object()))
    monkeypatch.setattr(private_facts, "selected_input_facts", lambda *a, **k: nullcontext({}))
    monkeypatch.setattr(private_result_format_selection, "from_run", lambda *a, **k: None)
    expected = selected_build / "phase1-native-chain/new-run/one/static"
    seen = []

    class ReachedNativeBuild(Exception):
        pass

    def build(*args, **kwargs):
        seen.append(kwargs["out"])
        assert kwargs["out"] == expected
        assert not kwargs["out"].exists()
        assert not kwargs["out"].is_relative_to(metadata)
        kwargs["progress"]({"stage": "host_admission", "model": "one", "program": "forward", "current": 128,
                            "total": 314, "candidate_tree_sha256": "must_not_be_reported"})
        raise ReachedNativeBuild

    monkeypatch.setattr(C.models, "run", build)
    arguments = dict(run_dir=run, descriptor=tmp_path / "descriptor", submission=tmp_path / "compiler",
                     name="one", out=metadata, owner=None, spec=tmp_path / "spec", selected=object(), freeze={})
    with pytest.raises(ReachedNativeBuild):
        C._compile(**arguments)
    assert seen == [expected]
    journal = json.loads((metadata / C.PROGRESS).read_bytes())
    assert journal["stage"] == "host_admission" and journal["program"] == "forward"
    assert journal["current"] == 128 and journal["total"] == 314 and journal["sequence"] == 2
    assert journal["updated_at"].endswith("+00:00")
    assert "candidate_tree_sha256" not in journal and "passed" not in journal
    assert (metadata / C.PROGRESS).stat().st_mode & 0o777 == 0o600
    assert not (metadata / "native-static.json").exists()
    logged = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert [row["stage"] for row in logged] == ["instruction_selection", "host_admission"]
    expected.mkdir(parents=True)
    with pytest.raises(ValueError, match="fresh selected output"):
        C._compile(**arguments)
    assert seen == [expected]
