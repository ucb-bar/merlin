"""Static import never upgrades structural, diagnostic or incomplete partitions."""

from types import SimpleNamespace

import pytest
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
