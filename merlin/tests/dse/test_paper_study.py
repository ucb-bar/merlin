"""Frozen-compiler paper study: holdout, matrix, and lifecycle honesty contracts."""
from __future__ import annotations

from pathlib import Path

import pytest

from merlin.common.paths import bench_dir
from merlin.common.yaml import write_yaml
from merlin.compare import study
from merlin.compare.paper import PaperStudySpec, SessionSpec, validate_paper_result
from merlin.compare.session import validate_capture_session, validate_paper_input_binding


def _spec() -> PaperStudySpec:
    return PaperStudySpec.from_yaml(bench_dir() / "rvv_paper" / "study_v2.yaml")


def test_curated_study_is_holdout_safe_and_defers_old_models():
    spec = _spec()
    names = {m.name for m in spec.models}
    assert names == {"gemma2_2b", "tinyllama_1_1b", "smolvla", "resnet50_v1_5", "lstmnetvit"}
    assert names == set(spec.holdout_models)
    assert names <= set(spec.development_corpus["excluded_models"])
    assert not names & {"bitvla", "rdt2", "openvla"}
    assert spec.reporting["deferred_models"] == ["bitvla", "rdt2", "openvla"]
    assert spec.reporting["performance_scope_policy"]["primary_table"] == (
        "end_to_end_continuous_sessions")
    assert all(tuple(model.session.parameters["timed_stages"]) == model.session.stages
               for model in spec.models)
    assert all(model.session.measurement_repeats == 5 for model in spec.models)
    executorch = next(backend for backend in spec.backends if backend.name == "executorch_xnnpack")
    assert "run" in executorch.options["command"]
    assert "{framework_package}" in executorch.options["command"]
    assert "{framework_package_sha256}" in executorch.options["command"]
    assert "--session-contract" not in executorch.options["command"]
    assert executorch.options["command"][0] == "{python_executable}"
    assert executorch.options["python_executable"].endswith("/bin/python3.12")
    assert executorch.options["python_venv_argv0"] == ".venv/bin/python-paper-adapter"
    assert len(executorch.options["python_executable_sha256"]) == 64


def test_matrix_only_contains_supported_precision_pairs():
    spec = _spec()
    cells = spec.matrix()
    assert len(cells) == 60
    assert spec.core_counts == (1, 8)
    assert len({cell.key for cell in cells}) == len(cells)
    assert {backend.name: sum(cell.backend.name == backend.name for cell in cells)
            for backend in spec.backends} == {
                "hand_v0_int8": 10,
                "merlin_frozen": 20,
                "merlin_xnnpack": 10,
                "merlin_openblas": 10,
                "executorch_xnnpack": 10,
            }
    assert all(c.precision in c.model.precisions and c.precision in c.backend.precisions for c in cells)
    assert not any(c.backend.name in {"merlin_xnnpack", "merlin_openblas"} and
                   c.precision == "w8a8" for c in cells)
    assert not any(c.backend.name == "executorch_xnnpack" and c.precision == "w8a8"
                   for c in cells)
    gap = spec.reporting["unsupported_comparisons"]
    assert gap == [{
        "backend": "executorch_xnnpack", "precision": "w8a8", "status": "not_implemented",
        "reason": gap[0]["reason"],
    }]
    assert "may be inferred" in gap[0]["reason"]


@pytest.mark.parametrize("core_counts", [
    [1, 1],
    [1, True],
    [1, "8"],
    [1, 8.0],
    [0, 8],
])
def test_core_counts_are_unique_exact_positive_integers(core_counts):
    import yaml

    raw = yaml.safe_load((bench_dir() / "rvv_paper" / "study_v2.yaml").read_text())
    raw["core_counts"] = core_counts

    with pytest.raises(ValueError, match="core_counts.*unique exact positive integers"):
        PaperStudySpec.parse(raw)


def test_execution_order_is_frozen_block_randomized_without_changing_cells():
    spec = _spec()
    first = study.execution_matrix(spec)
    second = study.execution_matrix(spec)
    assert [cell.key for cell in first] == [cell.key for cell in second]
    assert {cell.key for cell in first} == {cell.key for cell in spec.matrix()}
    # Every model/precision/core block stays contiguous, while backend order is seed-derived.
    block_positions = {}
    for index, cell in enumerate(first):
        block_positions.setdefault((cell.model.name, cell.precision, cell.core_count), []).append(index)
    assert all(indices == list(range(min(indices), max(indices) + 1))
               for indices in block_positions.values())
    assert [cell.key for cell in first] != [cell.key for cell in spec.matrix()]


def test_draft_preflight_is_explicitly_blocked():
    preflight = _spec().preflight()
    assert not preflight.ready
    assert any("status is draft" in reason for reason in preflight.blockers)
    assert any("capture sha256" in reason for reason in preflight.blockers)


def test_missing_baseline_source_only_disables_causal_attribution_not_performance():
    import dataclasses

    spec = _spec()
    backends = []
    for backend in spec.backends:
        if backend.kind == "frozen_baseline":
            options = dict(backend.options)
            options.pop("source_paths", None)
            options.pop("kernel_source_sha256", None)
            backend = dataclasses.replace(backend, options=options)
        backends.append(backend)

    preflight = dataclasses.replace(spec, backends=tuple(backends)).preflight()

    assert any("causal attribution unavailable" in value for value in preflight.warnings)
    assert not any("frozen_baseline" in value for value in preflight.blockers)


def test_paper_input_binding_matches_exact_model_artifact_and_source(tmp_path):
    import dataclasses
    import hashlib
    import yaml
    import json

    model = _spec().models[0]
    artifact = tmp_path / "inputs" / model.name / "token_ids.npy"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"frozen token bytes")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    source = "fixture/wikitext@" + "a" * 40 + ":raw/test"
    model = dataclasses.replace(model, expected_provenance={
        **model.expected_provenance, "token_sha256": digest, "token_source": source,
    })
    relative = artifact.relative_to(tmp_path).as_posix()
    record = {
        "active_holdouts": [model.name],
        "models": {model.name: {
            "artifacts": [{"path": relative, "sha256": digest,
                           "bytes": artifact.stat().st_size}],
            "environment": {"M2M_GEMMA_TOKEN_IDS": "{bundle}/" + relative,
                            "M2M_GEMMA_TOKEN_SOURCE": source},
            "provenance": {"input_source": source},
        }},
    }
    (tmp_path / "paper_inputs.json").write_text(json.dumps(record), encoding="utf-8")

    assert validate_paper_input_binding(tmp_path, [model]) == []
    changed = dataclasses.replace(
        model, expected_provenance={**model.expected_provenance, "token_sha256": "0" * 64})
    errors = validate_paper_input_binding(tmp_path, [changed])
    assert any("token_sha256" in value and "differs from the paper bundle" in value
               for value in errors)


def test_paper_input_binding_fails_closed_on_model_set_and_path_escape(tmp_path):
    import json

    model = _spec().models[0]
    outside = tmp_path.parent / "outside-token.npy"
    outside.write_bytes(b"outside")
    record = {
        "active_holdouts": [model.name, "unrelated"],
        "models": {model.name: {
            "artifacts": [{"path": "../outside-token.npy", "sha256": "0" * 64,
                           "bytes": len(outside.read_bytes())}],
            "environment": {"M2M_GEMMA_TOKEN_IDS": "{bundle}/../outside-token.npy",
                            "M2M_GEMMA_TOKEN_SOURCE": "fixture/source"},
            "provenance": {"input_source": "fixture/source"},
        }, "unrelated": {}},
    }
    (tmp_path / "paper_inputs.json").write_text(json.dumps(record), encoding="utf-8")

    errors = validate_paper_input_binding(tmp_path, [model])
    assert any("model set differs" in value for value in errors)
    assert any("path is unsafe" in value for value in errors)


def test_environment_preflight_rejects_provenance_unrelated_to_pinned_bundle(monkeypatch):
    import dataclasses

    spec = _spec()
    first = dataclasses.replace(
        spec.models[0],
        expected_provenance={**spec.models[0].expected_provenance,
                             "token_sha256": "0" * 64})
    spec = dataclasses.replace(spec, models=(first, *spec.models[1:]))
    monkeypatch.setattr(study, "_capture_contract", lambda *_args, **_kwargs: (Path(), {}, []))

    preflight = study.environment_preflight(spec)

    assert any("token_sha256" in value and "differs from the paper bundle" in value
               for value in preflight.errors)


def test_freeze_does_not_open_invalid_private_bundle_before_package_authority(tmp_path):
    import dataclasses
    import json

    from merlin.compare.freeze import freeze_study, sha256_paths

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "paper_inputs.json").write_text(
        json.dumps({"active_holdouts": [], "models": {}}), encoding="utf-8")
    policy = tmp_path / "policy.yaml"
    runtime = tmp_path / "runtime.bin"
    policy.write_text("version: 1\n", encoding="utf-8")
    runtime.write_bytes(b"runtime")
    spec = dataclasses.replace(
        _spec(), paper_inputs={"path": str(inputs), "sha256": sha256_paths([inputs])})
    from merlin.compare.paper_toolchain_authority import write_toolchain_authority
    authority = write_toolchain_authority(
        tmp_path / "toolchain-authority.json", authority_id="invalid-input-test",
        target=spec.target, build_tool="/usr/bin/clang")
    source = tmp_path / "invalid-input-study.yaml"
    write_yaml(source, spec.canonical_dict())
    spec = PaperStudySpec.from_yaml(source)

    with pytest.raises(ValueError, match="package registration is absent"):
        freeze_study(spec, policy_path=policy, runtime_paths=[runtime],
                     toolchain_authority_path=authority,
                     output_path=tmp_path / "frozen.yaml")


def test_full_freeze_finishes_package_rebuild_before_private_validators(
        tmp_path, monkeypatch):
    """The full constructor, not only its evidence helper, enforces the ordering boundary."""
    import dataclasses
    from types import SimpleNamespace
    from merlin.compare import freeze
    from merlin.compare import paper_measurement_freeze
    from merlin.compare.paper_toolchain_authority import write_toolchain_authority

    spec = dataclasses.replace(_spec(), source_path=None)
    source = tmp_path / "draft.yaml"
    write_yaml(source, spec.canonical_dict())
    spec = PaperStudySpec.from_yaml(source)
    policy, runtime = tmp_path / "policy", tmp_path / "runtime"
    policy.write_text("policy", encoding="utf-8")
    runtime.write_text("runtime", encoding="utf-8")
    authority = write_toolchain_authority(
        tmp_path / "toolchain-authority.json", authority_id="freeze-order-test",
        target=spec.target, build_tool="/usr/bin/clang")
    registration = {
        "packages": [{"model": model.name, "precision": "fp32"}
                     for model in spec.models]}
    monkeypatch.setattr(
        freeze, "_require_external_package_registration",
        lambda *_args: (tmp_path / "registration.json", "a" * 64, registration))
    (tmp_path / "registration.json").write_text("{}", encoding="utf-8")
    from merlin.baselines import executorch_session
    monkeypatch.setattr(
        executorch_session, "load_session_package",
        lambda *_args, **_kwargs: SimpleNamespace())
    events = []

    def rebuilt(*_args, **_kwargs):
        events.append("packages_rebuilt")
        raise RuntimeError("stop after the ordering witness")

    def private_access(*_args, **_kwargs):
        events.append("private_access")
        raise AssertionError("private validation ran before package rebuild")

    monkeypatch.setattr(
        paper_measurement_freeze, "validate_packages_before_private_io", rebuilt)
    monkeypatch.setattr(freeze, "validate_paper_input_binding", private_access)
    monkeypatch.setattr(freeze, "validate_capture_session", private_access)
    with pytest.raises(RuntimeError, match="ordering witness"):
        freeze.freeze_study(
            spec, policy_path=policy, runtime_paths=[runtime],
            toolchain_authority_path=authority, output_path=tmp_path / "frozen.yaml")
    assert events == ["packages_rebuilt"]


def test_full_freeze_actual_session_loader_hashes_private_tree_only_after_rebuild_barrier(
        tmp_path, monkeypatch):
    """The real v3 loader hashes stream/reference bytes only after every package rebuild."""
    import dataclasses
    import hashlib
    import json

    from merlin.baselines import executorch_session as ets
    from merlin.compare import freeze
    from merlin.compare.freeze import sha256_paths
    from merlin.compare.paper_toolchain_authority import write_toolchain_authority

    package = tmp_path / "private-session-package-v3"
    package.mkdir()
    private_files = {
        "private-reset.bin": b"\0" * 8,
        "private-frame0.bin": b"\0" * 12,
        "private-stream.bin": b"\1" * 24,
        "private-correctness.npz": b"private correctness sentinel",
        "private-quality.npz": b"private eager reference sentinel",
    }
    (package / "step.pte").write_bytes(b"pte")
    for name, payload in private_files.items():
        (package / name).write_bytes(payload)
    manifest = package / "executorch_session.json"
    manifest.write_text(json.dumps({
        "schema": ets.SCHEMA, "protocol_version": 1, "kind": "recurrent_frames",
        "paper_ready": True, "precision": "fp32", "reset": "restore_initial_inputs",
        "observations": 2, "warmups": 1, "measurement_repeats": 3,
        "programs": [{"name": "step", "pte": "step.pte", "ptd": [],
                      "method": "forward", "inputs": [
                          {"dtype": "float32", "shape": [1, 2]},
                          {"dtype": "float32", "shape": [1, 3]}]}],
        "bindings": [
            {"target": {"program": "step", "index": 0}, "kind": "initial",
             "tensor": {"dtype": "float32", "shape": [1, 2]},
             "file": "private-reset.bin"},
            {"target": {"program": "step", "index": 1}, "kind": "initial",
             "tensor": {"dtype": "float32", "shape": [1, 3]},
             "file": "private-frame0.bin"},
            {"target": {"program": "step", "index": 1}, "kind": "stream",
             "tensor": {"dtype": "float32", "shape": [1, 3]},
             "file": "private-stream.bin"}],
        "routes": [{"source": {"program": "step", "index": 1},
                    "target": {"program": "step", "index": 0},
                    "tensor": {"dtype": "float32", "shape": [1, 2]}}],
        "execution_schedule": [
            {"stage": "step", "program": "step", "observation": 0, "timed": True},
            {"stage": "step", "program": "step", "observation": 1, "timed": True}],
        "observation_output": {"source": {"program": "step", "index": 0},
                               "tensor": {"dtype": "float32", "shape": [1, 2]}},
        "correctness": "private-correctness.npz", "correctness_key": "correctness",
        "quality": "private-quality.npz", "quality_key": "quality",
        "logical_stages": ["step"],
        "stage_schedule": [{"name": "step", "steps": 2,
                            "execution": "compiled_recurrent", "timed": True}],
        "parameters": {}, "provenance": {},
    }), encoding="utf-8")
    runner = package / "executorch_session_runner"
    elf = bytearray(64)
    elf[:6] = b"\x7fELF\x02\x01"
    elf[18:20] = (243).to_bytes(2, "little")
    runner.write_bytes(elf)
    runner.chmod(0o755)
    plan_identity = ets.session_identity_sha256(
        ets.plan_session_identity(ets.load_plan(manifest)))
    invocation = {"MERLIN_K1_TOOLCHAIN": "/exact/tc",
                  "MERLIN_K1_TOOLCHAIN_ROOT": "/exact/tc",
                  "MERLIN_MODEL2MLIR": "/exact/model2mlir",
                  "MERLIN_M2M_DIR": "/exact/model2mlir"}
    packages = ["executorch==test", "torch==test"]
    package_text = "\n".join(packages) + "\n"
    environment = {
        "invocation_environment": invocation,
        "invocation_environment_sha256": ets._json_sha256(invocation),
        "python": "Python test", "python_packages": packages,
        "executorch_identity": {"exporter_version": "test", "exporter_git_sha": "9" * 40,
                                "source_git_sha": "9" * 40, "matches": True},
        "python_packages_sha256": hashlib.sha256(package_text.encode()).hexdigest(),
        "toolchain_identity": {
            "root": "/exact/tc",
            "c_compiler": {"path": "/exact/tc/bin/clang", "sha256": "1" * 64,
                           "version": "clang test"},
            "cxx_compiler": {"path": "/exact/tc/bin/clang++", "sha256": "2" * 64,
                             "version": "clang test"}},
        "model2mlir_identity": {"path": "/exact/model2mlir", "git_sha": "c" * 40,
                                "loader_sha256": "d" * 64,
                                "capture_source_sha256": "e" * 64},
        "external_model_source": None,
    }
    (package / "session_package.json").write_text(json.dumps({
        "schema": ets.PACKAGE_SCHEMA, "model": "sentinel_model", "variant": "fp32",
        "precision": "fp32", "capture_sha256": "a" * 64,
        "capture_session_identity_sha256": plan_identity,
        "framework_source_sha256": "b" * 64,
        "build_environment_sha256": ets._json_sha256(environment),
        "build_invocation_environment_sha256": ets._json_sha256(invocation),
        "build_environment": environment, "xnnpack": True,
        "manifest": manifest.name, "runner": runner.name,
        "observations": 2, "warmups": 1, "measurement_repeats": 3,
    }), encoding="utf-8")
    package_digest = sha256_paths([package])

    base = _spec()
    model = base.models[0]
    external = next(backend for backend in base.backends if backend.kind == "external_runtime")
    external = dataclasses.replace(external, options={
        **external.options,
        "packages": {model.name: {"fp32": {
            "path": str(package), "sha256": package_digest,
            "build_environment_sha256": environment["invocation_environment_sha256"]}}},
    })
    paper_inputs = tmp_path / "paper-inputs"
    paper_inputs.mkdir()
    (paper_inputs / "paper_inputs.json").write_text("{}\n", encoding="utf-8")
    spec = dataclasses.replace(
        base, models=(model,), holdout_models=(model.name,),
        backends=tuple(external if backend.kind == "external_runtime" else backend
                       for backend in base.backends),
        paper_inputs={"path": str(paper_inputs), "sha256": sha256_paths([paper_inputs])},
        source_path=None)
    source = tmp_path / "sentinel-study.yaml"
    write_yaml(source, spec.canonical_dict())
    spec = PaperStudySpec.from_yaml(source)
    policy, runtime = tmp_path / "policy", tmp_path / "runtime"
    policy.write_text("policy", encoding="utf-8")
    runtime.write_text("runtime", encoding="utf-8")
    authority = write_toolchain_authority(
        tmp_path / "toolchain-authority.json", authority_id="sentinel-order-test",
        target=spec.target, build_tool="/usr/bin/clang")
    registration = tmp_path / "package-registration.json"
    registration.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        freeze, "_require_external_package_registration",
        lambda *_args: (registration, hashlib.sha256(registration.read_bytes()).hexdigest(),
                        {"packages": [{"model": model.name, "precision": "fp32"}]}))
    events = []

    def package_barrier(*_args, **_kwargs):
        events.append("all_packages_regenerated_and_relinked")

    class StopAfterRealLoader(RuntimeError):
        pass

    def stop_at_private_binding(*_args, **_kwargs):
        events.append("paper_input_binding")
        raise StopAfterRealLoader

    monkeypatch.setattr(
        "merlin.compare.paper_measurement_freeze.validate_packages_before_private_io",
        package_barrier)
    monkeypatch.setattr(freeze, "validate_paper_input_binding", stop_at_private_binding)
    original_open = Path.open
    sentinels = {(package / name).resolve() for name in private_files}

    def guarded_open(path, *args, **kwargs):
        if path.resolve() in sentinels:
            assert events and events[0] == "all_packages_regenerated_and_relinked"
            assert "paper_input_binding" not in events
            events.append(f"private_read:{path.name}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(StopAfterRealLoader):
        freeze.freeze_study(
            spec, policy_path=policy, runtime_paths=[runtime],
            toolchain_authority_path=authority, output_path=tmp_path / "frozen.yaml")
    private_events = [event for event in events if event.startswith("private_read:")]
    assert {event.removeprefix("private_read:") for event in private_events} == set(private_files)
    assert events[0] == "all_packages_regenerated_and_relinked"
    assert events[-1] == "paper_input_binding"


def test_plan_writes_versioned_protocol_without_running(tmp_path: Path):
    out = study.run(_spec(), live=False, out_dir=tmp_path / "plan")
    assert (out / "study.yaml").is_file()
    assert (out / "preflight.yaml").is_file()
    assert (out / "matrix.yaml").is_file()
    assert "BLOCKED" in (out / "report.md").read_text(encoding="utf-8")


def test_live_explicit_output_still_opens_and_closes_aet_run(monkeypatch, tmp_path: Path):
    from types import SimpleNamespace
    from merlin.compare.paper import Preflight

    explicit = tmp_path / "explicit-paper-run"
    aet_dir = tmp_path / "canonical-aet-run"
    handle = SimpleNamespace(run_dir=aet_dir, run_id="aet-run", timestamp="time", git_sha="sha")
    events = []
    monkeypatch.setattr(study, "environment_preflight",
                        lambda _spec: Preflight((), ("deliberately blocked",), ()))
    monkeypatch.setattr(study, "start_run", lambda **_kwargs: events.append("start") or handle)
    monkeypatch.setattr(
        study, "finish_run",
        lambda got, status, summary: events.append((got, status, summary)))

    with pytest.raises(study.StudyNotReady) as raised:
        study.run(_spec(), live=True, out_dir=explicit)

    assert raised.value.output_dir == explicit
    assert (explicit / "preflight.yaml").is_file()
    assert events[0] == "start"
    assert events[1][0] is handle and events[1][1] == "fail"


def test_live_study_api_rejects_arbitrary_contract_builder_injection(tmp_path):
    with pytest.raises(TypeError, match="unexpected keyword argument 'executor'"):
        study.run(_spec(), live=True, out_dir=tmp_path / "paper-output",
                  executor=lambda *_args: tmp_path / "forged-contract.yaml")


def _write_registered_template(tmp_path, cell, *, registry_id="merlin_compile_v1",
                               backend_adapter="merlin_compile",
                               require_worker_threads=False, hardcoded_trajectory=False,
                               hostile_compiler=False):
    import hashlib
    import json
    import os
    import shutil
    import struct
    import subprocess
    import yaml

    root = tmp_path / ("hostile-compiler-template" if hostile_compiler else
                       "hardcoded-template" if hardcoded_trajectory else "registered-template")
    root.mkdir()
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    canonical_sha = lambda value: hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    package_identity = "c" * 64 if registry_id == "merlin_compile_v1" else "2" * 64
    source_identity = "b" * 64 if registry_id == "merlin_compile_v1" else "1" * 64
    artifact = root / "runtime-artifact.bin"
    artifact.write_bytes(b"/* 3.0 7.0 */\n")
    tool = root / "clang"
    shutil.copy2(Path("/usr/bin/clang").resolve(), tool)
    tool.chmod(0o755)
    from merlin.compare.paper_toolchain_authority import write_toolchain_authority
    authority = write_toolchain_authority(
        tmp_path / f"{root.name}-toolchain-authority.json",
        authority_id=f"{root.name}-reviewed", target="unit-test", build_tool=tool)
    compiler_input = root / "compiler-input.json"
    compiler_input.write_text(json.dumps({
        "schema_version": 1, "kind": "paper_unit_test_affine_compiler_input_v1",
        "compiler_or_framework_source_sha256": source_identity,
        "capture_sha256": cell.model.artifacts[cell.precision]["sha256"],
        "runtime_artifact_sha256": sha(artifact), "work_iterations": 1000000,
    }, sort_keys=True), encoding="utf-8")
    from merlin.compare.paper_model_object_builder import (
        OBJECT_BUILD_ARGV, UNIT_TEST_RECIPE, regenerate_model_object,
    )
    model_object = root / "model_object.o"
    regeneration = regenerate_model_object(
        recipe=UNIT_TEST_RECIPE, registry_id=registry_id, target="unit-test",
        compiler_input=compiler_input, tool=tool, output=model_object,
        source_identity_sha256=source_identity,
        capture_sha256=cell.model.artifacts[cell.precision]["sha256"],
        runtime_artifact_sha256=sha(artifact))
    if hardcoded_trajectory:
        # Exact attack fixture: the supplied/digest-bound object ignores both model artifact and
        # input and emits the expected trajectory.  Every package hash is refreshed below, but the
        # registry must reject it because it cannot be regenerated from compiler-input.json.
        source = root / "malicious-model-object.c"
        source.write_text(r'''
#include <stddef.h>
int merlin_paper_step(const char *artifact_path, const unsigned char *input,
                      size_t input_size, unsigned char *output, size_t *output_size) {
  (void)artifact_path; (void)input; (void)input_size;
  static unsigned long step = 0;
  float preconstructed[2] = {10.0f + 3.0f * (float)step, 1.0f + (float)step};
  step++;
  if (*output_size < sizeof(preconstructed)) return 2;
  __builtin_memcpy(output, &preconstructed, sizeof(preconstructed));
  *output_size = sizeof(preconstructed);
  volatile unsigned long work = 1;
  for (unsigned long i = 1; i < 1000000; ++i) work = work * 33u + i;
  return (int)(work & 0u);
}
''', encoding="utf-8")
        subprocess.run([str(tool), "-O2", "-std=c11", "-c", str(source),
                        "-o", str(model_object)], check=True)
        regeneration = {**regeneration,
                        "generated_source_sha256": sha(source),
                        "model_object_sha256": sha(model_object)}
    from merlin.compare import paper_measurement_controller
    shipped_runner = Path(paper_measurement_controller.__file__).with_name(
        "paper_model_abi_runner.c")
    runner = root / "runner.c"
    shutil.copy2(shipped_runner, runner)
    executable = root / "expected-cell"
    subprocess.run([str(tool), "-O2", "-std=c11", str(runner), str(model_object),
                    "-o", str(executable)], check=True)
    if hostile_compiler:
        # Attack fixture: an executable ELF posing as clang ignores every source argument and
        # writes one preconstructed, input-independent trajectory object to the requested output.
        malicious_source = root / "preconstructed-trajectory.c"
        malicious_source.write_text(r'''
#include <stddef.h>
int merlin_paper_step(const char *artifact_path, const unsigned char *input,
                      size_t input_size, unsigned char *output, size_t *output_size) {
  (void)artifact_path; (void)input; (void)input_size;
  static const float known[2] = {10.0f, 1.0f};
  if (*output_size < sizeof(known)) return 2;
  __builtin_memcpy(output, known, sizeof(known)); *output_size = sizeof(known); return 0;
}
''', encoding="utf-8")
        preconstructed = root / "preconstructed-trajectory.o"
        subprocess.run(["/usr/bin/clang", "-O2", "-std=c11", "-c",
                        str(malicious_source), "-o", str(preconstructed)], check=True)
        payload = preconstructed.read_bytes()
        compiler_source = root / "hostile-compiler.c"
        compiler_source.write_text(
            "#include <stdio.h>\n#include <string.h>\n"
            f"static const unsigned char payload[] = {{{','.join(str(value) for value in payload)}}};\n"
            "int main(int argc,char **argv){const char *out=0;"
            "for(int i=1;i+1<argc;i++)if(!strcmp(argv[i],\"-o\"))out=argv[i+1];"
            "if(!out)return 2;FILE *f=fopen(out,\"wb\");if(!f)return 3;"
            "size_t n=fwrite(payload,1,sizeof(payload),f);return fclose(f)||n!=sizeof(payload);}\n",
            encoding="utf-8")
        hostile = root / "hostile-compiler"
        subprocess.run(["/usr/bin/clang", "-O2", "-std=c11", str(compiler_source),
                        "-o", str(hostile)], check=True)
        shutil.copy2(hostile, tool)
        shutil.copy2(preconstructed, model_object)
        subprocess.run([str(tool), "-O2", "-std=c11", str(runner), str(model_object),
                        "-o", str(executable)], check=True)
    package_receipt = root / "package-receipt.json"
    object_builder = Path(paper_measurement_controller.__file__).with_name(
        "paper_model_object_builder.py")
    package_receipt.write_text(json.dumps({
        "schema_version": 2, "kind": "paper_backend_package_receipt_v2",
        "status": "finalized", "registry_id": registry_id,
        "build_adapter": ("merlin_model_abi_c_v1" if registry_id == "merlin_compile_v1"
                          else "executorch_model_abi_c_v1"),
        "cell": {"model": cell.model.name, "backend": cell.backend.name,
                 "precision": cell.precision},
        "package_identity_sha256": package_identity,
        "compiler_or_framework_source_sha256": source_identity,
        "capture_sha256": cell.model.artifacts[cell.precision]["sha256"],
        "runtime_artifact_sha256": sha(artifact),
        "runner_source_sha256": sha(runner), "model_object_sha256": sha(model_object),
        "compiler_input_sha256": sha(compiler_input),
        "object_builder_source_sha256": sha(object_builder),
        "object_recipe": UNIT_TEST_RECIPE, "object_build_argv": OBJECT_BUILD_ARGV,
        "generated_model_source_sha256": regeneration["generated_source_sha256"],
        "build_tool_sha256": sha(tool),
        "build_source_identity_sha256": canonical_sha(
            {"compiler_input": sha(compiler_input), "model_object": sha(model_object),
             "object_builder": sha(object_builder), "runner": sha(runner)}),
        "build_argv": ["{tool}", "-O2", "-std=c11", "{source:runner}",
                       "{source:model_object}", "-o", "{output}"],
        "result_executable_sha256": sha(executable),
        "finalized_at": "2026-08-31T00:00:00+00:00",
    }, sort_keys=True), encoding="utf-8")
    # Private measurement inputs and references are materialized only after the executable/package
    # receipt is frozen above.  They are not inputs to the build adapter.
    inputs = root / "session-inputs.bin"
    input_values = [1.0 + (int.from_bytes(os.urandom(4), "little") % 100000) / 997.0
                    for _ in range(cell.model.session.observations)]
    inputs.write_bytes(b"MRLNFRM1" + b"".join(
        struct.pack("<Qf", 4, value) for value in input_values))
    reference = root / "reference.bin"
    reference.write_bytes(b"MRLNFRM1" + b"".join(
        struct.pack("<Qff", 8, value * 3.0 + 7.0, value) for value in input_values))
    manifest = root / "session-manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "kind": "paper_session_inputs_v1",
        "session_kind": cell.model.session.kind,
        "observations": cell.model.session.observations,
        "inputs": {"session_input": sha(inputs)},
        "records": [{"index": index, "payload_sha256": hashlib.sha256(
            struct.pack("<f", value)).hexdigest()}
                    for index, value in enumerate(input_values)],
    }), encoding="utf-8")
    io_receipt = root / "measurement-io-receipt.json"
    io_receipt.write_text(json.dumps({
        "schema_version": 1, "kind": "paper_measurement_io_generation_receipt_v1",
        "status": "finalized",
        "cell": {"model": cell.model.name, "backend": cell.backend.name,
                 "precision": cell.precision},
        "package_receipt_sha256": sha(package_receipt),
        "artifact_sha256": sha(artifact), "input_sha256": {"session_input": sha(inputs)},
        "session_manifest_sha256": sha(manifest),
        "reference_output_sha256": sha(reference), "reference_authority": "eager_fp32",
        "observations": cell.model.session.observations,
        "capture_sha256": cell.model.artifacts[cell.precision]["sha256"],
        "input_source_sha256": sha(inputs),
        "eager_reference_source_sha256": sha(reference),
        "eager_reference_key": "test_affine_reference",
        "generated_at": "2026-08-31T00:00:01+00:00",
    }, sort_keys=True), encoding="utf-8")
    template = {
        "schema_version": 3, "kind": "paper_backend_measurement_template_v3",
        "status": "frozen", "registry_id": registry_id,
        "backend_adapter": backend_adapter,
        "cell": {"model": cell.model.name, "backend": cell.backend.name,
                 "precision": cell.precision, "core_count": cell.core_count},
        "resources": {
            "package_receipt": {"path": package_receipt.name,
                                "sha256": sha(package_receipt)},
            "compiler_input": {"path": compiler_input.name,
                               "sha256": sha(compiler_input)},
            "model_object": {"path": model_object.name, "sha256": sha(model_object)},
            "build_tool": {"path": tool.name, "sha256": sha(tool)},
            "runtime_artifact": {"path": artifact.name, "sha256": sha(artifact)}},
        "environment": {}, "execution": {"mode": "rvv",
            "core_ids": [min(os.sched_getaffinity(0))],
            "require_worker_threads": require_worker_threads},
        "memory_policy": "resident", "timeout_seconds": 10,
    }
    path = root / "template.yaml"
    path.write_text(yaml.safe_dump(template, sort_keys=True), encoding="utf-8")
    io = {
        "artifact": {"path": str(artifact), "sha256": sha(artifact)},
        "inputs": {"session_input": {"path": str(inputs), "sha256": sha(inputs)}},
        "session_manifest": {"path": str(manifest), "sha256": sha(manifest)},
        "reference_output": {"path": str(reference), "sha256": sha(reference)},
        "generation_receipt": {"path": str(io_receipt), "sha256": sha(io_receipt)},
    }
    return path, sha(path), io, authority


def test_executorch_model_object_recipe_rejects_unreceipted_compiler_input(tmp_path):
    from merlin.compare.paper_model_object_builder import (
        expected_recipe,
        regenerate_model_object,
    )
    compiler_input = tmp_path / "compiler-input"
    tool = tmp_path / "tool"
    compiler_input.write_bytes(b"must not be accepted as a production compiler input")
    tool.write_bytes(Path("/bin/true").read_bytes())

    with pytest.raises(ValueError, match="ExecuTorch compiler input is invalid JSON"):
        regenerate_model_object(
            recipe=expected_recipe("executorch_v1", "k1"), registry_id="executorch_v1",
            target="k1",
            compiler_input=compiler_input, tool=tool, output=tmp_path / "object.o",
            source_identity_sha256="a" * 64, capture_sha256="b" * 64,
            runtime_artifact_sha256="c" * 64)


def test_registered_merlin_builder_produces_independently_replayable_live_result(
        monkeypatch, tmp_path):
    import dataclasses
    import hashlib
    import json
    from merlin.compare.paper import Preflight
    from merlin.compare import paper_attribution
    from merlin.compare.paper_contract_registry import build_registered_contract
    from merlin.common.yaml import load_yaml

    spec = dataclasses.replace(_spec(), status="frozen", target="unit-test",
                               freeze={**_spec().freeze, "policy_sha256": "d" * 64,
                                       "compiler_source_sha256": "b" * 64,
                                       "runtime_sha256": "e" * 64})
    model = spec.models[0]
    artifacts = {key: dict(value) for key, value in model.artifacts.items()}
    artifacts["fp32"]["sha256"] = "f" * 64
    model = dataclasses.replace(model, artifacts=artifacts)
    backend = next(value for value in spec.backends if value.name == "merlin_frozen")
    provisional_cell = dataclasses.replace(
        next(value for value in spec.matrix()
             if value.model.name == model.name and value.backend.name == backend.name
             and value.precision == "fp32" and value.core_count == 1), model=model)
    template_path, template_sha, measurement_io, authority = _write_registered_template(
        tmp_path, provisional_cell)
    options = {**backend.options, "timeout": 10, "package_sha256": "c" * 64,
               "measurement_contracts": {model.name: {"fp32": {"1": {
                   "path": str(template_path), "sha256": template_sha}}}}}
    backend = dataclasses.replace(backend, options=options)
    models = tuple(model if value.name == model.name else value for value in spec.models)
    backends = tuple(backend if value.name == backend.name else value for value in spec.backends)
    spec = dataclasses.replace(
        spec, models=models, backends=backends,
        freeze={**spec.freeze, "measurement_io": {
            backend.name: {model.name: {"fp32": measurement_io}}},
            "toolchain_authority_path": str(authority),
            "toolchain_authority_sha256": hashlib.sha256(authority.read_bytes()).hexdigest()})
    cell = next(value for value in spec.matrix()
                if value.model.name == model.name and value.backend.name == backend.name
                and value.precision == "fp32" and value.core_count == 1)
    forged_template = tmp_path / "template-with-authored-oracle.yaml"
    forged_document = load_yaml(template_path)
    forged_document["oracle"] = {"kind": "bytes_exact", "metric": "self_authored",
                                 "threshold": 0.0, "scope": "trajectory", "steps": 1}
    forged_document["reference_output"] = measurement_io["reference_output"]
    write_yaml(forged_template, forged_document)
    forged_options = {**backend.options, "measurement_contracts": {model.name: {"fp32": {
        "1": {"path": str(forged_template),
              "sha256": hashlib.sha256(forged_template.read_bytes()).hexdigest()}}}}}
    forged_backend = dataclasses.replace(backend, options=forged_options)
    forged_spec = dataclasses.replace(
        spec, backends=tuple(forged_backend if value.name == backend.name else value
                             for value in spec.backends))
    forged_cell = next(value for value in forged_spec.matrix()
                       if value.model.name == model.name and value.backend.name == backend.name
                       and value.precision == "fp32" and value.core_count == 1)
    with pytest.raises(ValueError, match="backend measurement template is closed"):
        build_registered_contract(
            forged_spec, forged_cell, run_id="forged", timestamp="20260831T000000Z",
            git_sha="deadbee", staging_dir=tmp_path / "forged-contract",
                base_result=study._base_result(
                    forged_spec, forged_cell, "forged", "20260831T000000Z", "deadbee"))
    hard_path, hard_sha, hard_io, hard_authority = _write_registered_template(
        tmp_path, provisional_cell, hardcoded_trajectory=True)
    hard_backend = dataclasses.replace(backend, options={
        **backend.options, "measurement_contracts": {model.name: {"fp32": {"1": {
            "path": str(hard_path), "sha256": hard_sha}}}}})
    hard_spec = dataclasses.replace(
        spec, backends=tuple(hard_backend if value.name == backend.name else value
                             for value in spec.backends),
        freeze={**spec.freeze, "measurement_io": {
            hard_backend.name: {model.name: {"fp32": hard_io}}},
            "toolchain_authority_path": str(hard_authority),
            "toolchain_authority_sha256": hashlib.sha256(
                hard_authority.read_bytes()).hexdigest()})
    hard_cell = next(value for value in hard_spec.matrix()
                     if value.model.name == model.name and value.backend.name == backend.name
                     and value.precision == "fp32" and value.core_count == 1)
    with pytest.raises(ValueError, match="model object differs from registry regeneration"):
        build_registered_contract(
            hard_spec, hard_cell, run_id="hardcoded", timestamp="20260831T000000Z",
            git_sha="deadbee", staging_dir=tmp_path / "hardcoded-contract",
            base_result=study._base_result(
                hard_spec, hard_cell, "hardcoded", "20260831T000000Z", "deadbee"))
    hostile_path, hostile_sha, hostile_io, hostile_authority = _write_registered_template(
        tmp_path, provisional_cell, hostile_compiler=True)
    hostile_backend = dataclasses.replace(backend, options={
        **backend.options, "measurement_contracts": {model.name: {"fp32": {"1": {
            "path": str(hostile_path), "sha256": hostile_sha}}}}})
    hostile_spec = dataclasses.replace(
        spec, backends=tuple(hostile_backend if value.name == backend.name else value
                             for value in spec.backends),
        freeze={**spec.freeze, "measurement_io": {
            hostile_backend.name: {model.name: {"fp32": hostile_io}}},
            "toolchain_authority_path": str(hostile_authority),
            "toolchain_authority_sha256": hashlib.sha256(
                hostile_authority.read_bytes()).hexdigest()})
    hostile_cell = next(
        value for value in hostile_spec.matrix()
        if value.model.name == model.name and value.backend.name == backend.name
        and value.precision == "fp32" and value.core_count == 1)
    with pytest.raises(
            ValueError, match="build tool differs from independent toolchain authority"):
        build_registered_contract(
            hostile_spec, hostile_cell, run_id="hostile-compiler",
            timestamp="20260831T000000Z", git_sha="deadbee",
            staging_dir=tmp_path / "hostile-compiler-contract",
            base_result=study._base_result(
                hostile_spec, hostile_cell, "hostile-compiler",
                "20260831T000000Z", "deadbee"))
    monkeypatch.setattr(study, "environment_preflight", lambda _spec: Preflight((), (), ()))
    monkeypatch.setattr(study, "execution_matrix", lambda _spec: (cell,))
    monkeypatch.setattr(paper_attribution, "attach_causal_attribution",
                        lambda _spec, _results: None)
    real_start = study.start_run
    monkeypatch.setattr(study, "start_run", lambda **kwargs: real_start(
        **kwargs, project_root=tmp_path / "aet-root"))

    output = study.run(spec, live=True, out_dir=tmp_path / "paper-output")

    result = load_yaml(output / "results.yaml")["results"][0]
    receipt = load_yaml(Path(result["measurement_receipt"]["path"]))
    raw = json.loads((Path(result["measurement_receipt"]["path"]).parent /
                      "raw_measurement.json").read_text())
    assert receipt["kind"] == "paper_controller_measurement_receipt_v6"
    assert raw["build_receipt"]["registry_id"] == "merlin_compile_v1"
    assert raw["memory"]["peak_rss_bytes"] > 0
    assert raw["functional_output_sha256"] != measurement_io["inputs"]["session_input"]["sha256"]
    assert len(raw["quality"]["per_step"]) == cell.model.session.observations
    assert all(row["gate_ok"] for row in raw["quality"]["per_step"])
    assert result["lifecycle"]["status"] == "pass"


def test_registered_executorch_builder_binds_framework_package_and_external_threads(
        tmp_path, monkeypatch):
    import dataclasses
    import hashlib
    import yaml
    from merlin.compare.paper_contract_registry import build_registered_contract
    from merlin.compare.paper_measurement_controller import normalize_receipt, produce_receipt

    base_spec = _spec()
    spec = dataclasses.replace(base_spec, status="frozen", target="unit-test",
                               freeze={**base_spec.freeze, "policy_sha256": "d" * 64,
                                       "compiler_source_sha256": "b" * 64,
                                       "runtime_sha256": "e" * 64})
    model = spec.models[0]
    artifacts = {key: dict(value) for key, value in model.artifacts.items()}
    artifacts["fp32"]["sha256"] = "f" * 64
    model = dataclasses.replace(model, artifacts=artifacts)
    backend = next(value for value in spec.backends if value.name == "executorch_xnnpack")
    provisional = dataclasses.replace(
        next(value for value in spec.matrix()
             if value.model.name == model.name and value.backend.name == backend.name
             and value.precision == "fp32" and value.core_count == 1), model=model)
    template_path, template_sha, measurement_io, authority = _write_registered_template(
        tmp_path, provisional, registry_id="executorch_v1", backend_adapter="executorch",
        require_worker_threads=True)
    options = {**backend.options, "timeout": 10, "framework_source_sha256": "1" * 64,
               "packages": {model.name: {"fp32": {
                   "path": "unused-by-controller", "sha256": "2" * 64}}},
               "measurement_contracts": {model.name: {"fp32": {"1": {
                   "path": str(template_path), "sha256": template_sha}}}}}
    backend = dataclasses.replace(backend, options=options)
    spec = dataclasses.replace(
        spec, models=tuple(model if value.name == model.name else value for value in spec.models),
        backends=tuple(backend if value.name == backend.name else value for value in spec.backends),
        freeze={**spec.freeze, "measurement_io": {
            backend.name: {model.name: {"fp32": measurement_io}}},
            "toolchain_authority_path": str(authority),
            "toolchain_authority_sha256": hashlib.sha256(
                authority.read_bytes()).hexdigest()})
    cell = next(value for value in spec.matrix()
                if value.model.name == model.name and value.backend.name == backend.name
                and value.precision == "fp32" and value.core_count == 1)
    base = study._base_result(spec, cell, "executorch-child", "20260831T000000Z", "deadbee")
    contract = build_registered_contract(
        spec, cell, run_id="executorch-child", timestamp="20260831T000000Z",
        git_sha="deadbee", staging_dir=tmp_path / "executorch-contract", base_result=base)
    from merlin.compare import paper_measurement_controller as controller
    contract_document = yaml.safe_load(contract.read_text(encoding="utf-8"))
    private_paths = {
        (contract.parent / contract_document["reference_output"]["path"]).resolve(),
        *{(contract.parent / ref["path"]).resolve()
          for ref in contract_document["inputs"].values()},
    }
    events = []
    original_build = controller._build_executable
    original_read_bytes = Path.read_bytes

    def observed_build(*args, **kwargs):
        result = original_build(*args, **kwargs)
        events.append("build_verified")
        return result

    def observed_read(path):
        if path.resolve() in private_paths:
            events.append(f"private_read:{path.name}")
            assert "build_verified" in events
        return original_read_bytes(path)

    monkeypatch.setattr(controller, "_build_executable", observed_build)
    monkeypatch.setattr(Path, "read_bytes", observed_read)
    result = normalize_receipt(produce_receipt(contract, tmp_path / "executorch-receipt"))

    assert result["lifecycle"]["status"] == "pass"
    assert result["provenance"]["framework_source_sha256"] == "1" * 64
    assert result["provenance"]["framework_package_sha256"] == "2" * 64
    assert result["execution"]["worker_thread_source"] == "proc_task_status"
    assert events[0] == "build_verified"
    assert any(event.startswith("private_read:") for event in events)


def test_freeze_constructor_derives_measurement_io_from_capture_and_package_receipts(tmp_path):
    import dataclasses
    import hashlib
    import json
    import numpy as np
    import yaml
    from merlin.compare.freeze import sha256_paths
    from merlin.compare.paper_measurement_freeze import (
        construct_measurement_evidence, write_capture_measurement_source_receipt,
    )

    base = _spec()
    model = base.models[0]
    backend = next(value for value in base.backends if value.name == "merlin_frozen")
    observations = model.session.observations
    capture = tmp_path / "capture"
    capture.mkdir()
    inputs = np.arange(observations * 2, dtype=np.float32).reshape(observations, 2)
    reference = np.stack((inputs[:, 0] + 1, inputs[:, 1] * 2), axis=1).astype(np.float32)
    np.savez(capture / "session_inputs.npz", stream=inputs)
    np.savez(capture / "quality.npz", eager=reference)
    (capture / "session_contract.yaml").write_text(yaml.safe_dump({
        "version": 1, "streams": [{"key": "stream"}], "inputs": "session_inputs.npz",
        "quality": {"scope": "trajectory", "reference": "eager_fp32",
                    "metric": model.quality["metric"], "golden": "quality.npz", "key": "eager",
                    "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest()},
    }), encoding="utf-8")
    source_receipt = write_capture_measurement_source_receipt(
        capture, model=model.name, precision="fp32", observations=observations)
    artifacts = {key: dict(value) for key, value in model.artifacts.items()}
    artifacts["fp32"]["sha256"] = sha256_paths([capture])
    artifacts["fp32"]["path"] = str(capture)
    model = dataclasses.replace(model, artifacts=artifacts, precisions=("fp32",))
    provisional = dataclasses.replace(
        next(value for value in base.matrix()
             if value.model.name == model.name and value.backend.name == backend.name
             and value.precision == "fp32" and value.core_count == 1), model=model)
    template, template_sha, _manual_io, authority = _write_registered_template(
        tmp_path, provisional)
    backend = dataclasses.replace(backend, precisions=("fp32",), options={
        **backend.options, "package_sha256": "c" * 64,
        "measurement_contracts": {model.name: {"fp32": {"1": {
            "path": str(template), "sha256": template_sha}}}}})
    spec = dataclasses.replace(
        base, target="unit-test", models=(model,), backends=(backend,), core_counts=(1,),
        holdout_models=(model.name,),
        freeze={**base.freeze, "compiler_source_sha256": "b" * 64})
    raw = spec.canonical_dict()
    raw["freeze"]["measurement_io"] = {"forged": "must be ignored"}

    derived, retained = construct_measurement_evidence(
        raw, capture_roots={(model.name, "fp32"): capture},
        output_path=tmp_path / "frozen.yaml", toolchain_authority_path=authority,
        toolchain_authority_sha256=hashlib.sha256(authority.read_bytes()).hexdigest())

    row = derived[backend.name][model.name]["fp32"]
    generation = json.loads(Path(row["generation_receipt"]["path"]).read_text())
    assert "forged" not in derived
    assert generation["capture_sha256"] == artifacts["fp32"]["sha256"]
    assert generation["reference_authority"] == "eager_fp32"
    assert source_receipt in retained
    source_receipt.unlink()
    with pytest.raises(ValueError, match="capture measurement-source receipt"):
        construct_measurement_evidence(
            raw, capture_roots={(model.name, "fp32"): capture},
            output_path=tmp_path / "missing-receipt.yaml",
            toolchain_authority_path=authority,
            toolchain_authority_sha256=hashlib.sha256(authority.read_bytes()).hexdigest())


def test_live_run_signature_has_no_contract_builder_capability():
    import inspect

    assert "executor" not in inspect.signature(study.run).parameters


def test_live_study_without_standalone_contract_builder_fails_closed(monkeypatch, tmp_path):
    import dataclasses
    import json
    from merlin.compare.paper import Preflight
    from merlin.common.yaml import load_yaml

    spec = dataclasses.replace(_spec(), status="frozen")
    monkeypatch.setattr(study, "environment_preflight", lambda _spec: Preflight((), (), ()))
    real_start = study.start_run
    monkeypatch.setattr(
        study, "start_run",
        lambda **kwargs: real_start(**kwargs, project_root=tmp_path / "aet-root"))
    explicit = tmp_path / "paper-output"

    with pytest.raises(ValueError, match="measurement_contracts"):
        study.run(spec, live=True, out_dir=explicit)

    link = load_yaml(explicit / "aet-parent.yaml")
    parent_events = [json.loads(line) for line in
                     (Path(link["canonical_run_dir"]) / "logs" / "events.jsonl")
                     .read_text().splitlines()]
    assert parent_events[-1]["event"] == "run.finished"
    assert parent_events[-1]["payload"]["status"] == "fail"


def test_live_study_rejects_an_authored_result_mapping_as_measurement_authority(
        monkeypatch, tmp_path):
    import dataclasses
    from merlin.compare.paper import Preflight

    spec = dataclasses.replace(_spec(), status="frozen")
    cell = next(c for c in spec.matrix()
                if c.backend.name == "merlin_frozen" and c.precision == "fp32"
                and c.core_count == 1)
    monkeypatch.setattr(study, "environment_preflight", lambda _spec: Preflight((), (), ()))
    monkeypatch.setattr(study, "execution_matrix", lambda _spec: (cell,))
    real_start = study.start_run
    monkeypatch.setattr(
        study, "start_run",
        lambda **kwargs: real_start(**kwargs, project_root=tmp_path / "aet-root"))

    def authored_result(_spec, _cell, run_id, timestamp, git_sha, _staging_dir):
        return study._base_result(_spec, _cell, run_id, timestamp, git_sha)

    with pytest.raises(TypeError, match="unexpected keyword argument 'executor'"):
        study.run(spec, live=True, out_dir=tmp_path / "paper-output",
                  executor=authored_result)


def test_environment_audit_collects_deeper_blockers_even_for_a_draft(monkeypatch, tmp_path: Path):
    spec = _spec()
    monkeypatch.setattr(study, "_capture_contract", lambda *_args, **_kwargs: (
        tmp_path, {}, ["session stage implementation is incomplete"]))
    preflight = study.environment_preflight(spec)
    assert any("status is draft" in value for value in preflight.blockers)
    assert any("session stage implementation is incomplete" in value
               for value in preflight.blockers)
    assert not any("stateful ExecuTorch session command" in value
                   for value in preflight.blockers)
    assert not any("digest mismatch" in value for value in preflight.blockers)


def test_environment_audit_checks_frozen_source_paths_and_stateful_command(monkeypatch, tmp_path):
    import dataclasses

    spec = _spec()
    source = tmp_path / "external.py"
    source.write_text("pass\n", encoding="utf-8")
    import hashlib

    python = tmp_path / "pinned-python"
    python.write_bytes(b"#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    command = ["{python_executable}", "-m", "merlin.baselines.executorch_session", "run",
               "--model", "{model}", "--variant", "{variant}",
               "--cores", "{cores}", "--package", "{framework_package}",
               "--package-sha256", "{framework_package_sha256}",
               "--warmups", "{warmups}", "--observations", "{observations}",
               "--measurement-repeats", "{measurement_repeats}",
               "--quality-metric", "{quality_metric}", "--quality-min", "{quality_min}",
               "--framework-source-sha256", "{framework_source_sha256}"]
    backends = []
    for backend in spec.backends:
        if backend.adapter == "executorch":
            options = {**backend.options, "command": command,
                       "python_executable": str(python),
                       "python_executable_sha256": hashlib.sha256(python.read_bytes()).hexdigest(),
                       "source_paths": [str(source)],
                       "framework_source_sha256": study.sha256_paths([source])}
            backend = dataclasses.replace(backend, options=options)
        backends.append(backend)
    spec = dataclasses.replace(spec, backends=tuple(backends))
    monkeypatch.setattr(study, "_capture_contract", lambda *_args, **_kwargs: (tmp_path, {}, []))
    preflight = study.environment_preflight(spec)
    assert not any("stateful ExecuTorch session command" in value for value in preflight.blockers)
    assert not any("executorch_xnnpack: framework_source_sha256 mismatch" in value
                   for value in preflight.blockers)

    command_without_package = [part for part in command if "framework_package}" not in part]
    broken = []
    for backend in spec.backends:
        if backend.adapter == "executorch":
            backend = dataclasses.replace(
                backend, options={**backend.options, "command": command_without_package})
        broken.append(backend)
    preflight = study.environment_preflight(dataclasses.replace(spec, backends=tuple(broken)))
    assert any("omits placeholders ['framework_package']" in value
               for value in preflight.blockers)

    build_in_measurement = []
    for backend in spec.backends:
        if backend.adapter == "executorch":
            backend = dataclasses.replace(
                backend, options={**backend.options,
                                  "command": ["build" if part == "run" else part
                                              for part in command]})
        build_in_measurement.append(backend)
    preflight = study.environment_preflight(
        dataclasses.replace(spec, backends=tuple(build_in_measurement)))
    assert any("run-only adapter" in value for value in preflight.blockers)

    ambient_python = []
    for backend in spec.backends:
        if backend.adapter == "executorch":
            backend = dataclasses.replace(
                backend, options={**backend.options,
                                  "command": ["python", *command[1:]]})
        ambient_python.append(backend)
    preflight = study.environment_preflight(
        dataclasses.replace(spec, backends=tuple(ambient_python)))
    assert any("command must start with the pinned {python_executable} placeholder" in value
               for value in preflight.blockers)


def test_environment_audit_recomputes_declared_frozen_baseline_source(monkeypatch, tmp_path):
    import dataclasses

    spec = _spec()
    source = tmp_path / "baseline-source.c"
    source.write_text("int baseline(void) { return 0; }\n", encoding="utf-8")
    backends = []
    for backend in spec.backends:
        if backend.kind == "frozen_baseline":
            backend = dataclasses.replace(backend, options={
                **backend.options, "source_paths": [str(source)],
                "kernel_source_sha256": "0" * 64,
            })
        backends.append(backend)
    spec = dataclasses.replace(spec, backends=tuple(backends))
    monkeypatch.setattr(study, "_capture_contract", lambda *_args, **_kwargs: (
        tmp_path, {}, []))

    preflight = study.environment_preflight(spec)

    assert any("hand_v0_int8: kernel_source_sha256 mismatch" in value
               for value in preflight.blockers)


def _passing_result() -> dict:
    return {
        "schema_version": 2, "run_id": "run", "timestamp": "20260830T000000Z",
        "git_sha": "abcdef0", "study_label": "study", "target": "board",
        "model": "model", "checkpoint": "checkpoint", "artifact_sha256": "a" * 64,
        "fidelity": "full", "backend": "compiler", "runtime": "runtime",
        "precision": "w8a8", "quantization": "static", "core_count": 8,
        "session": {"kind": "recurrent_frames", "observations": 2, "measurement_repeats": 3,
                    "stages": ["recurrent_step"]},
        "lifecycle": {"built": True, "ran": True, "status": "pass", "reason": None},
        "correctness": {"gate_ok": True, "cosine": 1.0},
        "quality": {"gate_ok": True, "metric": "output_cosine", "value": 0.999,
                    "scope": "trajectory", "steps": 2},
        "timing": {"unit": "ns", "sample_unit": "complete_session", "scope": "end_to_end",
                   "timed_stages": ["recurrent_step"], "excluded_stages": [],
                   "samples": [10, 11, 12], "median": 11, "p95": 12},
        "memory": {"policy": "resident", "peak_rss_bytes": 1024},
        "execution": {"mode": "rvv_openmp", "requested_mode": "rvv_openmp",
                      "fallback_used": False, "core_count": 8, "requested_core_count": 8,
                      "affinity_source": "sched_getaffinity", "semantic_session": True,
                      "same_input_repetition": False},
        "provenance": {"compiler_policy_sha256": "b" * 64, "runtime_sha256": "c" * 64,
                       "vlen_bits": 256, "vlen_source": "csr",
                       "board_conditions": {
                           "before": {"governor": "performance", "current_khz": 1600000,
                                      "max_khz": 1600000, "max_thermal_millic": 41000},
                           "after": {"governor": "performance", "current_khz": 1600000,
                                     "max_khz": 1600000, "max_thermal_millic": 42000},
                       }},
    }


def test_passing_result_requires_real_run_quality_and_exact_mode():
    result = _passing_result()
    validate_paper_result(result)
    result["execution"]["fallback_used"] = True
    with pytest.raises(ValueError, match="fallback"):
        validate_paper_result(result)


@pytest.mark.parametrize(("field", "value", "message"), [
    ("semantic_session", False, "semantic_session=true"),
    ("semantic_session", 1, "semantic_session=true"),
    ("same_input_repetition", True, "same_input_repetition=false"),
    ("same_input_repetition", 0, "same_input_repetition=false"),
])
def test_passing_result_cannot_lie_about_continuous_session_semantics(
        field, value, message):
    result = _passing_result()
    result["execution"][field] = value

    with pytest.raises(ValueError, match=message):
        validate_paper_result(result)


def test_passing_result_requires_observed_cpu_affinity():
    result = _passing_result()
    result["execution"]["core_count"] = 7
    with pytest.raises(ValueError, match="requested core count"):
        validate_paper_result(result)


def test_passing_executorch_result_requires_exact_worker_threads():
    result = _passing_result()
    result["runtime"] = "executorch"
    result["execution"].update(
        worker_threads=8, worker_thread_source="proc_task_status")
    validate_paper_result(result)
    result["execution"]["worker_threads"] = 1
    with pytest.raises(ValueError, match="worker-thread configuration"):
        validate_paper_result(result)
    result = _passing_result()
    result["execution"]["affinity_source"] = "requested_value"
    with pytest.raises(ValueError, match="sched_getaffinity"):
        validate_paper_result(result)


def test_passing_kernel_swap_requires_complete_declared_eligible_coverage():
    result = _passing_result()
    result["backend"] = "merlin_xnnpack"
    result["execution"].update(n_routed=3, n_eligible=3, n_candidates=5)
    validate_paper_result(result)

    result["execution"]["n_routed"] = 2
    with pytest.raises(ValueError, match="complete coverage"):
        validate_paper_result(result)

    result["execution"].update(n_routed=3, n_eligible=3, n_candidates=2)
    with pytest.raises(ValueError, match="complete coverage"):
        validate_paper_result(result)


def test_passing_result_requires_locked_frequency_and_thermal_evidence():
    result = _passing_result()
    result["provenance"]["board_conditions"]["after"]["current_khz"] = 1200000
    with pytest.raises(ValueError, match="current frequency equal to max"):
        validate_paper_result(result)
    result = _passing_result()
    result["provenance"]["board_conditions"]["before"]["governor"] = "ondemand"
    with pytest.raises(ValueError, match="performance governor"):
        validate_paper_result(result)


def test_build_only_cannot_be_pass():
    result = _passing_result()
    result["lifecycle"]["ran"] = False
    with pytest.raises(ValueError, match="built=true and ran=true"):
        validate_paper_result(result)


def test_passing_result_requires_exact_samples_statistics_and_trajectory():
    result = _passing_result()
    result["timing"]["samples"] = [10]
    with pytest.raises(ValueError, match="exactly 3 full-session timing samples"):
        validate_paper_result(result)
    result = _passing_result()
    result["timing"]["median"] = 999
    with pytest.raises(ValueError, match="median/p95"):
        validate_paper_result(result)
    result = _passing_result()
    result["quality"]["steps"] = 1
    with pytest.raises(ValueError, match="exact observation trajectory"):
        validate_paper_result(result)


def test_causal_attribution_records_are_structured_and_digest_bound():
    result = _passing_result()
    result["causal_attribution"] = {"schema_version": 1, "records": [{
        "comparator": "executorch_xnnpack", "status": "available",
        "why": "structural observation", "how": "frozen treatment",
        "evidence": {"binding_sha256": "a" * 64, "ablation_sha256": "b" * 64,
                     "structural_sha256": "c" * 64},
    }]}
    validate_paper_result(result)
    result["causal_attribution"]["records"][0]["evidence"]["ablation_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="invalid ablation_sha256"):
        validate_paper_result(result)


def test_external_adapter_cannot_rewrite_matrix_identity():
    spec = _spec()
    cell = next(value for value in spec.matrix() if value.backend.adapter == "executorch")
    result = study._base_result(spec, cell, "run", "timestamp", "git")
    with pytest.raises(ValueError, match="identity mismatch for model"):
        study._merge_external_result(result, {"model": "different_model"}, cell)


def test_external_adapter_must_supply_all_measured_sections_and_frozen_source():
    spec = _spec()
    cell = next(value for value in spec.matrix() if value.backend.adapter == "executorch")
    result = study._base_result(spec, cell, "run", "timestamp", "git")
    with pytest.raises(ValueError, match="omitted measured sections"):
        study._merge_external_result(result, {}, cell)
    measured = {
        "lifecycle": {}, "correctness": {}, "quality": {}, "timing": {}, "memory": {},
        "execution": {}, "provenance": {"framework_source_sha256": "wrong"},
    }
    with pytest.raises(ValueError, match="framework source digest differs"):
        study._merge_external_result(result, measured, cell)
    measured["provenance"] = {
        "framework_source_sha256": cell.backend.options["framework_source_sha256"],
        "framework_package_sha256": "wrong",
    }
    with pytest.raises(ValueError, match="package digest differs"):
        study._merge_external_result(result, measured, cell)


def test_external_adapter_revalidates_pinned_python_at_live_launch(
        monkeypatch, tmp_path):
    import dataclasses
    import hashlib

    spec = _spec()
    python = tmp_path / "pinned-python"
    python.write_bytes(b"#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    backends = []
    for backend in spec.backends:
        if backend.adapter == "executorch":
            backend = dataclasses.replace(backend, options={
                **backend.options,
                "python_executable": str(python),
                "python_executable_sha256": hashlib.sha256(python.read_bytes()).hexdigest(),
            })
        backends.append(backend)
    spec = dataclasses.replace(spec, backends=tuple(backends))
    cell = next(value for value in spec.matrix() if value.backend.adapter == "executorch")
    capture = tmp_path / "capture"
    capture.mkdir()
    monkeypatch.setattr(study, "_capture_contract",
                        lambda *_args, **_kwargs: (capture, {"version": 2}, []))
    seen = {}

    def refuse_launch(command, **_kwargs):
        seen["command"] = command
        seen["executable"] = _kwargs["executable"]
        raise RuntimeError("stop before external work")

    monkeypatch.setattr(study.subprocess, "run", refuse_launch)
    result = study.execute_cell(spec, cell, "run", "time", "sha")
    assert Path(seen["command"][0]).is_absolute()
    assert seen["command"][0].endswith("/.venv/bin/python-paper-adapter")
    assert seen["executable"].startswith("/proc/self/fd/")
    assert result["lifecycle"]["status"] == "error"

    python.write_bytes(b"#!/bin/sh\nexit 1\n")
    seen.clear()
    result = study.execute_cell(spec, cell, "run", "time", "sha")
    assert seen == {}
    assert "python executable digest mismatch" in result["lifecycle"]["reason"]


def test_external_python_rejects_symlink_escape_and_fd_survives_exact_swaps(
        tmp_path):
    import dataclasses
    import hashlib
    import os
    import subprocess

    spec = _spec()
    cell = next(value for value in spec.matrix() if value.backend.adapter == "executorch")
    source = tmp_path / "source-python"
    source.write_text("#!/bin/sh\nprintf ORIGINAL\n", encoding="utf-8")
    source.chmod(0o755)
    argv0 = Path(study.repo_root()) / ".venv" / "bin" / "python-paper-adapter"

    def with_source(value):
        backend = dataclasses.replace(cell.backend, options={
            **cell.backend.options, "python_executable": str(value),
            "python_executable_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "python_venv_argv0": str(argv0),
        })
        return dataclasses.replace(cell, backend=backend)

    symlink = tmp_path / "python-link"
    symlink.symlink_to(source)
    with pytest.raises(ValueError, match="escape or symlink"):
        study._external_python(with_source(symlink))
    (source.parent / "nested").mkdir()
    escaped = source.parent / "nested" / ".." / source.name
    with pytest.raises(ValueError, match="escape or symlink"):
        study._external_python(with_source(escaped))

    safe_cell = with_source(source)
    pinned, staged, execute_fd = study._stage_external_python(
        safe_cell, tmp_path / "private-python")
    source.write_text("#!/bin/sh\nprintf MALICIOUS_SOURCE\n", encoding="utf-8")
    staged.unlink()
    staged.write_text("#!/bin/sh\nprintf MALICIOUS_STAGE\n", encoding="utf-8")
    staged.chmod(0o755)
    try:
        proc = subprocess.run(
            [str(pinned.argv0)], executable=f"/proc/self/fd/{execute_fd}",
            pass_fds=(execute_fd,), capture_output=True, text=True, check=True)
    finally:
        os.close(execute_fd)
    assert proc.stdout == "ORIGINAL"


def test_private_fd_python_preserves_frozen_venv_semantics(tmp_path):
    import os
    import subprocess

    spec = _spec()
    cell = next(value for value in spec.matrix() if value.backend.adapter == "executorch")
    pinned, _staged, execute_fd = study._stage_external_python(
        cell, tmp_path / "private-python")
    try:
        proc = subprocess.run(
            [str(pinned.argv0), "-c",
             "import sys, numpy; print(sys.prefix); print(numpy.__version__)"],
            executable=f"/proc/self/fd/{execute_fd}", pass_fds=(execute_fd,),
            capture_output=True, text=True, check=True)
    finally:
        os.close(execute_fd)
    lines = proc.stdout.splitlines()
    assert lines[0] == str(Path(study.repo_root()) / ".venv")
    assert lines[1]


def test_environment_preflight_rejects_unbounded_cell_timeout(monkeypatch, tmp_path):
    import dataclasses

    spec = _spec()
    backend = dataclasses.replace(
        spec.backends[0], options={**spec.backends[0].options, "timeout": 86_401})
    spec = dataclasses.replace(spec, backends=(backend, *spec.backends[1:]))
    monkeypatch.setattr(study, "_capture_contract",
                        lambda *_args, **_kwargs: (tmp_path, {"version": 2}, []))
    preflight = study.environment_preflight(spec)
    assert any("invalid whole-cell timeout" in value and "86400" in value
               for value in preflight.blockers)


def test_forged_session_version_cannot_reach_an_adapter(monkeypatch, tmp_path):
    spec = _spec()
    cell = next(c for c in spec.matrix() if c.backend.name == "merlin_frozen")
    capture = tmp_path / "capture"
    capture.mkdir()
    monkeypatch.setattr(study, "_capture_contract",
                        lambda *_args, **_kwargs: (capture, {"version": 999}, []))
    from merlin import compile_cli
    monkeypatch.setattr(
        compile_cli, "compile_rvv",
        lambda *_args, **_kwargs: pytest.fail("forged session reached the compiler adapter"))

    result = study.execute_cell(spec, cell, "run", "time", "sha")

    assert result["lifecycle"]["status"] == "not_run"
    assert "validated version-2 session" in result["lifecycle"]["reason"]


def test_merlin_success_normalizes_backend_evidence_and_reports_session_behavior(
        monkeypatch, tmp_path):
    spec = _spec()
    cell = next(c for c in spec.matrix()
                if c.backend.name == "merlin_frozen" and c.precision == "fp32"
                and c.core_count == 1)
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "model.mlir").write_text("module {}\n")
    monkeypatch.setattr(study, "_capture_contract",
                        lambda *_args, **_kwargs: (capture, {"version": 2}, []))
    from merlin import compile_cli
    conditions = {"governor": "performance", "current_khz": 1600000,
                  "max_khz": 1600000, "max_thermal_millic": 41000}
    monkeypatch.setattr(compile_cli, "compile_rvv", lambda *_args, **_kwargs: {
        "status": "verified", "binary": "model", "verify": {"gate_ok": True},
        "trajectory_quality": {"scope": "trajectory",
                               "steps": cell.model.session.observations,
                               "min_cosine": 1.0},
        "iter_wall_ns": [10] * cell.model.session.measurement_repeats,
        "sustained_wall_ns": {"median": 10, "p95": 10},
        "execution": {"mode": "rvv", "requested_mode": "rvv", "fallback_used": False,
                      "core_count": 1, "requested_core_count": 1,
                      "affinity_source": "sched_getaffinity",
                      "semantic_session": True, "same_input_repetition": False,
                      "kernel_backend": None, "n_routed": None},
        "vlen": 256, "vlen_source": "csr", "peak_rss_bytes": 1,
        "board_conditions": {"before": conditions, "after": conditions},
    })

    result = study.execute_cell(spec, cell, "run", "time", "sha")

    assert result["lifecycle"]["status"] == "pass"
    assert "kernel_backend" not in result["execution"]
    assert result["execution"]["semantic_session"] is True
    assert result["execution"]["same_input_repetition"] is False


def test_merlin_forged_raw_kernel_backend_is_retained_and_demoted(monkeypatch, tmp_path):
    spec = _spec()
    cell = next(c for c in spec.matrix()
                if c.backend.name == "merlin_frozen" and c.precision == "fp32"
                and c.core_count == 1)
    capture = tmp_path / "capture"
    capture.mkdir()
    monkeypatch.setattr(study, "_capture_contract",
                        lambda *_args, **_kwargs: (capture, {"version": 2}, []))
    from merlin import compile_cli
    conditions = {"governor": "performance", "current_khz": 1600000,
                  "max_khz": 1600000, "max_thermal_millic": 41000}
    compiled = {
        "status": "verified", "binary": "model", "verify": {"gate_ok": True},
        "trajectory_quality": {"scope": "trajectory",
                               "steps": cell.model.session.observations,
                               "min_cosine": 1.0},
        "iter_wall_ns": [10] * cell.model.session.measurement_repeats,
        "sustained_wall_ns": {"median": 10, "p95": 10},
        "execution": {"mode": "rvv", "requested_mode": "rvv", "fallback_used": False,
                      "core_count": 1, "requested_core_count": 1,
                      "affinity_source": "sched_getaffinity", "semantic_session": True,
                      "same_input_repetition": False, "kernel_backend": "xnnpack",
                      "n_routed": 3, "n_eligible": 3, "n_candidates": 4},
        "vlen": 256, "vlen_source": "csr", "peak_rss_bytes": 1,
        "board_conditions": {"before": conditions, "after": conditions},
    }
    monkeypatch.setattr(compile_cli, "compile_rvv", lambda *_args, **_kwargs: compiled)
    evidence = {}

    result = study.execute_cell(spec, cell, "run", "time", "sha", evidence_sink=evidence)

    assert result["lifecycle"]["status"] == "fail"
    assert "kernel-backend evidence differs" in result["lifecycle"]["reason"]
    assert "kernel_backend" not in result["execution"]
    assert evidence["adapter_output"]["execution"]["kernel_backend"] == "xnnpack"


def test_kernel_swap_with_zero_routed_kernels_cannot_pass(monkeypatch, tmp_path):
    spec = _spec()
    cell = next(c for c in spec.matrix()
                if c.backend.kind == "kernel_swap" and c.precision == "fp32")
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "model.mlir").write_text("module {}\n")
    monkeypatch.setattr(study, "_capture_contract", lambda *_args: (
        capture, {"version": 2}, []))
    from merlin import compile_cli
    monkeypatch.setattr(compile_cli, "compile_rvv", lambda *_args, **_kwargs: {
        "status": "verified", "binary": "model", "verify": {"gate_ok": True},
        "trajectory_quality": {"scope": "trajectory", "steps": cell.model.session.observations,
                               "min_cosine": 1.0},
        "iter_wall_ns": [10] * cell.model.session.measurement_repeats,
            "sustained_wall_ns": {"median": 10, "p95": 10},
            "execution": {"mode": "rvv", "requested_mode": "rvv", "fallback_used": False,
                      "semantic_session": True, "same_input_repetition": False,
                      "kernel_backend": cell.backend.options["kernel_backend"], "n_routed": 0,
                      "n_eligible": 0, "n_candidates": 0},
        "vlen": 256, "vlen_source": "csr", "peak_rss_bytes": 1,
    })
    result = study.execute_cell(spec, cell, "run", "time", "sha")
    assert result["lifecycle"]["status"] == "fail"
    assert "complete nonempty eligible GEMM set" in result["lifecycle"]["reason"]
    assert result["timing"]["scope"] == "end_to_end"
    assert result["timing"]["timed_stages"] == ["prefill", "decode"]
    assert result["timing"]["excluded_stages"] == []


def test_off_frequency_cell_is_recorded_but_demoted(monkeypatch, tmp_path):
    spec = _spec()
    cell = next(c for c in spec.matrix()
                if c.backend.name == "merlin_frozen" and c.precision == "fp32")
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "model.mlir").write_text("module {}\n")
    monkeypatch.setattr(study, "_capture_contract", lambda *_args: (
        capture, {"version": 2}, []))
    from merlin import compile_cli
    conditions = {"governor": "performance", "current_khz": 1600000,
                  "max_khz": 1600000, "max_thermal_millic": 41000}
    off_frequency = {**conditions, "current_khz": 1200000}
    monkeypatch.setattr(compile_cli, "compile_rvv", lambda *_args, **_kwargs: {
        "status": "verified", "binary": "model", "verify": {"gate_ok": True},
        "trajectory_quality": {"scope": "trajectory",
                               "steps": cell.model.session.observations,
                               "min_cosine": 1.0},
        "iter_wall_ns": [10] * cell.model.session.measurement_repeats,
        "sustained_wall_ns": {"median": 10, "p95": 10},
        "execution": {"mode": "rvv", "requested_mode": "rvv", "fallback_used": False,
                      "core_count": cell.core_count,
                      "requested_core_count": cell.core_count,
                      "affinity_source": "sched_getaffinity",
                      "semantic_session": True, "same_input_repetition": False,
                      "kernel_backend": None},
        "vlen": 256, "vlen_source": "csr", "peak_rss_bytes": 1,
        "board_conditions": {"before": conditions, "after": off_frequency},
    })
    result = study.execute_cell(spec, cell, "run", "time", "sha")
    assert result["lifecycle"]["status"] == "fail"
    assert "board-condition endpoints" in result["lifecycle"]["reason"]
    assert result["provenance"]["board_conditions"]["after"] == off_frequency


def test_primary_study_rejects_a_stage_subset_as_the_headline_session():
    import yaml

    raw = yaml.safe_load((bench_dir() / "rvv_paper" / "study_v2.yaml").read_text())
    raw["models"][0]["session"]["parameters"]["timed_stages"] = ["decode"]
    with pytest.raises(ValueError, match="must time every stage"):
        PaperStudySpec.parse(raw)


def _session_spec(observations: int = 2) -> SessionSpec:
    return SessionSpec("recurrent_frames", 0, observations,
                       ("visual_encode", "recurrent_step", "predict"),
                       ("hidden_state", "cell_state"))


def _write_session(capture: Path, *, paper_ready: bool = True, steps: int = 2) -> None:
    import hashlib
    import numpy as np
    import yaml

    capture.mkdir()
    np.savez(capture / "session_inputs.npz", frames=np.zeros((steps, 1, 1)))
    correctness = np.zeros((steps, 1), np.float32)
    quality = np.ones((steps, 1), np.float32)
    np.savez(capture / "session_goldens.npz", output0=correctness)
    np.savez(capture / "session_quality_fp32.npz", output0=quality)
    contract = {
        "version": 1, "kind": "recurrent_frames", "paper_ready": paper_ready,
        "stages": ["visual_encode", "recurrent_step", "predict"],
        "inputs": "session_inputs.npz",
        "states": [
            {"name": "hidden_state", "input_arg": 10, "output_index": 1},
            {"name": "cell_state", "input_arg": 11, "output_index": 2},
        ],
        "streams": [{"name": "frame", "input_arg": 9, "key": "frames"}],
        "correctness": {"scope": "trajectory", "golden": "session_goldens.npz",
                        "key": "output0", "output_index": 0,
                        "reference": "eager_same_precision",
                        "reference_sha256": hashlib.sha256(correctness.tobytes()).hexdigest()},
        "quality": {"scope": "trajectory", "golden": "session_quality_fp32.npz",
                    "key": "output0", "output_index": 0, "reference": "eager_fp32",
                    "reference_sha256": hashlib.sha256(quality.tobytes()).hexdigest()},
        "provenance": {"checkpoint": "trained", "full_checkpoint": True,
                       "checkpoint_sha256": "a" * 64, "synthetic_session": False,
                       "session_sha256": "b" * 64, "session_source": "test/trajectory"},
    }
    (capture / "session_contract.yaml").write_text(yaml.safe_dump(contract), encoding="utf-8")


def test_session_capture_must_be_paper_ready_and_long_enough(tmp_path: Path):
    capture = tmp_path / "capture"
    _write_session(capture, paper_ready=False, steps=1)
    _, errors = validate_capture_session(capture, _session_spec())
    assert any("paper_ready=true" in error for error in errors)
    assert any("session has 1 observations" in error for error in errors)
    assert any("golden has 1 observations" in error for error in errors)


def test_session_capture_rejects_extra_state_and_accepts_exact_contract(tmp_path: Path):
    capture = tmp_path / "capture"
    _write_session(capture)
    _, errors = validate_capture_session(capture, _session_spec())
    assert errors == []

    import yaml
    path = capture / "session_contract.yaml"
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    contract["states"].append({"name": "undeclared", "input_arg": 12, "output_index": 3})
    path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    _, errors = validate_capture_session(capture, _session_spec())
    assert any("carried state differs" in error for error in errors)


def test_multi_program_session_requires_compiled_prefill_and_validates_child_trajectory(
        tmp_path, monkeypatch):
    import hashlib
    import json
    import numpy as np
    import yaml
    from merlin.llvmlower import session_bundle

    capture = tmp_path / "causal"
    for name in ("prefill", "decode"):
        stage = capture / "stages" / name
        stage.mkdir(parents=True)
        (stage / "model.mlir").write_text("module { func.func @forward() }\n")
        (stage / "weights.safetensors").write_bytes(b"weights")
        (stage / "weights.safetensors.manifest.json").write_text(json.dumps({}))
        np.savez(stage / "inputs.npz", in0=np.zeros(2), in1=np.zeros(2), in2=np.zeros(2))
    decode = capture / "stages" / "decode"
    np.savez(decode / "session_inputs.npz", tokens=np.zeros((3, 2), np.float32))
    correctness = np.zeros((3, 2), np.float32)
    quality = np.ones((3, 2), np.float32)
    np.savez(decode / "session_goldens.npz", logits=correctness)
    np.savez(decode / "session_quality_fp32.npz", logits=quality)
    (decode / "session_contract.yaml").write_text(yaml.safe_dump({
        "version": 1, "kind": "autoregressive_decode", "paper_ready": True,
        "stages": ["decode"], "steps": 3, "inputs": "session_inputs.npz",
        "states": [
            {"name": "kv_cache", "input_arg": 1, "output_index": 0},
            {"name": "position", "input_arg": 2, "output_index": 1},
        ],
        "streams": [{"name": "token", "input_arg": 0, "key": "tokens"}],
        "correctness": {"scope": "trajectory", "golden": "session_goldens.npz",
                        "key": "logits", "output_index": 0,
                        "reference": "eager_same_precision",
                        "reference_sha256": hashlib.sha256(correctness.tobytes()).hexdigest()},
        "quality": {"scope": "trajectory", "golden": "session_quality_fp32.npz",
                    "key": "logits", "output_index": 0, "reference": "eager_fp32",
                    "reference_sha256": hashlib.sha256(quality.tobytes()).hexdigest()},
    }))
    root = {
        "version": 2, "kind": "autoregressive_decode", "paper_ready": True,
        "stages": ["prefill", "decode"],
        "stage_schedule": [
            {"name": "prefill", "steps": 1, "execution": "compiled", "timed": True},
            {"name": "decode", "steps": 3, "execution": "compiled_recurrent", "timed": True},
        ],
        "parameters": {"prefill_tokens": 2, "decode_tokens": 3, "batch": 1,
                       "timed_stages": ["prefill", "decode"],
                       "paper_primary_scope": "end_to_end"},
        "programs": [
            {"name": "prefill", "bundle": "stages/prefill", "steps": 1},
            {"name": "decode", "bundle": "stages/decode", "steps": 3},
        ],
        "bindings": [
            {"name": "kv_cache", "from": {"program": "prefill", "output_index": 0},
             "to": {"program": "decode", "input_arg": 1}},
            {"name": "position", "from": {"program": "prefill", "output_index": 1},
             "to": {"program": "decode", "input_arg": 2}},
        ],
        "states": [{"name": "kv_cache"}, {"name": "position"}], "streams": [],
        "quality": {"scope": "trajectory", "program": "decode"},
        "provenance": {"checkpoint": "trained", "full_checkpoint": True,
                       "synthetic_tokens": False, "token_sha256": "a" * 64,
                       "token_source": "fixture/corpus"},
    }
    (capture / "session_contract.yaml").write_text(yaml.safe_dump(root))
    monkeypatch.setattr(session_bundle, "parse_forward_signature",
                        lambda _path: [([2], "f32"), ([2], "f32"), ([2], "f32")])
    monkeypatch.setattr(session_bundle, "forward_signature",
                        lambda _path: ([([2], "f32")], [([2], "f32"), ([2], "f32")]))
    expected = SessionSpec(
        "autoregressive_decode", 1, 3, ("prefill", "decode"),
        ("kv_cache", "position"), root["parameters"], 3)
    _, errors = validate_capture_session(capture, expected)
    assert errors == []

    root["stage_schedule"][0]["execution"] = "eager_reference_initial_state"
    (capture / "session_contract.yaml").write_text(yaml.safe_dump(root))
    _, errors = validate_capture_session(capture, expected)
    assert any("not executed by compiled code" in error for error in errors)
    assert any("multi-program session is not executable" in error for error in errors)


def test_action_session_can_be_stream_free_with_exact_transition_count(tmp_path: Path):
    capture = tmp_path / "capture"
    _write_session(capture, steps=2)
    import yaml
    path = capture / "session_contract.yaml"
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    contract.update(kind="action_chunk", stages=["flow_denoise"], streams=[], steps=2)
    contract["provenance"] = {"checkpoint": "trained", "full_checkpoint": True,
                              "synthetic_inputs": False, "input_sha256": "c" * 64,
                              "input_source": "test/actions"}
    contract["states"] = [
        {"name": "flow_state", "input_arg": 1, "output_index": 0},
        {"name": "timestep", "input_arg": 2, "output_index": 1},
    ]
    path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    expected = SessionSpec("action_chunk", 0, 2, ("flow_denoise",),
                           ("flow_state", "timestep"))
    _, errors = validate_capture_session(capture, expected)
    assert errors == []


def test_timed_session_stage_must_be_compiled(tmp_path: Path):
    capture = tmp_path / "capture"
    _write_session(capture)
    import yaml
    path = capture / "session_contract.yaml"
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    contract["stage_schedule"] = [
        {"name": "visual_encode", "execution": "eager_reference", "timed": True},
        {"name": "recurrent_step", "execution": "compiled_recurrent", "timed": True},
        {"name": "predict", "execution": "compiled", "timed": True},
    ]
    contract["parameters"] = {"timed_stages": [
        "visual_encode", "recurrent_step", "predict"]}
    path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    expected = SessionSpec(
        "recurrent_frames", 0, 2,
        ("visual_encode", "recurrent_step", "predict"),
        ("hidden_state", "cell_state"),
        {"timed_stages": ["visual_encode", "recurrent_step", "predict"]})
    _, errors = validate_capture_session(capture, expected)
    assert any("visual_encode" in error and "not executed by compiled code" in error
               for error in errors)


def test_paper_recurrent_session_rejects_synthetic_or_unhashed_provenance(tmp_path: Path):
    capture = tmp_path / "capture"
    _write_session(capture)
    import yaml
    path = capture / "session_contract.yaml"
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    contract["provenance"]["synthetic_session"] = True
    contract["provenance"]["session_sha256"] = "unresolved"
    path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    _, errors = validate_capture_session(capture, _session_spec())
    assert any("synthetic trajectory" in error for error in errors)
    assert any("trajectory has no valid SHA-256" in error for error in errors)
