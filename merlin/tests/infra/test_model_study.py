"""Model diagnostics must not confuse incomplete profiles with model failures."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.model_check.bottlenecks import analyze, feedback


def report():
    return {"schema": "merlin.model_check.report.v1", "model": "network", "status": "pass",
            "gates": {"quantized_reference": {"status": "pass"}},
            "programs": [{"program": "encode", "runs": [{
                "input": "visible/v00", "step": 0, "status": "completed", "cycles": 100,
                "group_profile": {"calls": 2, "dropped": 0, "group_cycles": 60, "gap_cycles": 40},
            }]}]}


def read(document, **options):
    return analyze(document, expected_steps={"encode": 1}, input_id="visible/v00",
                   counter_kind="spike_retired_instruction_proxy", **options)


def test_complete_profile_retains_scope_and_independent_numerical_status():
    result = read(report())
    assert result["profile_complete"]
    assert result["forward_sum_ticks"] == 100
    assert result["breakdown"][0]["region"] == "routed_calls"
    assert result["breakdown"][0]["ticks"] == 60
    assert result["functional_gates"]["quantized_reference"]["status"] == "pass"
    assert result["all_declared_outputs_verified"] is False
    assert result["model_quality"] == "not_established_by_this_report"
    assert result["physical_timing_qualified"] is False


def test_ordinary_full_execute_counter_field_is_consumed_without_relabeling():
    document = report()
    row = document["programs"][0]["runs"][0]
    row["cycles_functional"] = row.pop("cycles")
    assert read(document)["forward_sum_ticks"] == 100
    row["cycles"] = 101
    with pytest.raises(ValueError, match="aliases disagree"):
        read(document)


def test_missing_cache_step_cannot_be_complete():
    result = analyze(report(), expected_steps={"encode": 1, "decode": 2},
                     input_id="visible/v00", counter_kind="raw_counter_ticks")
    assert result["expected_steps"] == 3
    assert result["missing_steps"] == [{"program": "decode", "step": 0}, {"program": "decode", "step": 1}]
    assert not result["profile_complete"]


def test_dropped_calls_do_not_turn_unrecorded_device_time_into_host_overhead():
    document = report()
    document["programs"][0]["runs"][0]["group_profile"]["dropped"] = 1
    result = read(document)
    assert not result["profile_complete"]
    assert result["intervals"][0]["complete"] is False
    assert result["next_actions"][0]["kind"] == "complete_execution_profile"
    assert "between_calls" not in {row["region"] for row in result["breakdown"]}
    assert next(row for row in result["breakdown"] if row["region"] == "unattributed")["ticks"] == 40
    assert "dropped routed calls" in json.dumps(feedback(result))


@pytest.mark.parametrize("change", [
    {"group_cycles": 61}, {"group_cycles": -1}, {"calls": True}, {"dropped": -1},
])
def test_invalid_or_unreconciled_counters_refuse(change):
    document = report()
    document["programs"][0]["runs"][0]["group_profile"].update(change)
    with pytest.raises(ValueError):
        read(document)


def test_nonzero_call_time_requires_a_recorded_call():
    document = report()
    document["programs"][0]["runs"][0]["group_profile"]["calls"] = 0
    with pytest.raises(ValueError, match="without a recorded call"):
        read(document)


def test_other_input_does_not_fill_a_missing_recurrent_step():
    document = report()
    document["programs"][0]["runs"][0]["input"] = "sealed/s00"
    result = read(document)
    assert result["recorded_steps"] == 0
    assert result["forward_sum_ticks"] is None


def test_completed_execution_without_profile_is_explicitly_missing_profile():
    document = report()
    document["programs"][0]["runs"][0].pop("group_profile")
    result = read(document)
    assert result["missing_execution_steps"] == []
    assert result["missing_profile_steps"] == [{"program": "encode", "step": 0}]


def test_nested_run_cannot_claim_another_programs_profile():
    document = report()
    document["programs"][0]["runs"][0]["program"] = "decode"
    with pytest.raises(ValueError, match="different program"):
        read(document)


def test_unsupported_lowering_does_not_recommend_restoring_z3():
    document = report()
    document["programs"][0].update(runs=[], kernels=[{
        "symbol": "projection", "status": "refused", "detail": "unsupported frontend operator",
    }])
    assert read(document)["next_actions"][0]["kind"] == "inspect_failed_check"


def test_top_level_proof_runtime_error_is_preserved():
    document = report()
    document["programs"] = []
    document["failures"] = [{"detail": "ModuleNotFoundError: z3 unavailable"}]
    assert read(document)["status"] == "blocked"


def test_duplicate_steps_and_programs_refuse():
    document = report()
    document["programs"][0]["runs"] *= 2
    with pytest.raises(ValueError, match="duplicate recurrent"):
        read(document)
    document = report()
    document["programs"] *= 2
    with pytest.raises(ValueError, match="duplicate model program"):
        read(document)


def test_real_detailed_profile_ranks_calls_without_adding_gap_twice():
    document = report()
    profile = document["programs"][0]["runs"][0]["group_profile"]
    profile.update(groups=[{"index": 0, "name": "projection", "cycles": 50, "gap_before": 10},
                           {"index": 1, "name": "attention", "cycles": 10, "gap_before": 20}], tail_gap=10)
    result = read(document)
    assert result["ranked_calls"][0]["name"] == "projection"
    assert sum(row["ticks"] for row in result["breakdown"]) == 100
    profile["groups"][0]["cycles"] = 49
    with pytest.raises(ValueError, match="summary disagrees"):
        read(document)


def test_proof_environment_failure_is_not_a_numerical_failure():
    document = report()
    document["programs"][0].update(runs=[], kernels=[{
        "symbol": "projection", "status": "refused", "form": "matmul 32x64x16",
        "detail": "ModuleNotFoundError: mandatory scheduling proof dependency z3 is unavailable",
    }])
    result = read({"source_sha256": "a" * 64, "report": document})
    assert result["status"] == "blocked"
    assert result["source_sha256"] == "a" * 64
    assert result["blockers"][0]["kind"] == "proof_runtime_unavailable"
    assert result["blockers"][0]["numerical_failure_established"] is False


def test_feedback_does_not_export_private_paths_or_kernel_errors():
    document = report()
    document["submission"] = {"path": "/private/secret", "tree_sha256": "f" * 64}
    result = feedback(read(document))
    assert "private" not in json.dumps(result)
    assert "submission" not in result and "functional_gates" not in result


def test_physical_cycles_require_an_independent_measurement_workflow():
    with pytest.raises(ValueError, match="diagnostic counter"):
        analyze(report(), expected_steps={"encode": 1}, input_id="visible/v00", counter_kind="physical_cycles")


def prepare_module():
    path = repo_root() / "examples/gemmini/model-study/prepare.py"
    spec = importlib.util.spec_from_file_location("model_study_preparation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory(tmp_path):
    root = tmp_path / "compiler"
    root.mkdir()
    (root / "compiler.py").write_text("print('example')\n")
    data = (root / "compiler.py").read_bytes()
    digest = hashlib.sha256(b"compiler.py\0" + data + b"\0").hexdigest()
    document = {"tree_sha256": digest, "files": [{"source_path": "compiler.py",
                "sha256": hashlib.sha256(data).hexdigest(), "executable": False}]}
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(document))
    return root, path, digest


def test_exact_compiler_inventory_detects_added_or_changed_sources(tmp_path):
    module = prepare_module()
    root, path, digest = inventory(tmp_path)
    assert module.verify_compiler(root, path, digest)["files"] == 1
    (root / "unrecorded.py").write_text("pass\n")
    with pytest.raises(ValueError, match="membership"):
        module.verify_compiler(root, path, digest)
    (root / "unrecorded.py").unlink()
    (root / "compiler.py").write_text("changed\n")
    with pytest.raises(ValueError, match="source changed"):
        module.verify_compiler(root, path, digest)


def test_inventory_rejects_symlink_substitution(tmp_path):
    module = prepare_module()
    root, path, digest = inventory(tmp_path)
    other = tmp_path / "other.py"
    other.write_bytes((root / "compiler.py").read_bytes())
    (root / "compiler.py").unlink()
    (root / "compiler.py").symlink_to(other)
    with pytest.raises(ValueError, match="escapes"):
        module.verify_compiler(root, path, digest)


def test_unlisted_directory_symlink_cannot_change_imports(tmp_path):
    module = prepare_module()
    root, path, digest = inventory(tmp_path)
    other = tmp_path / "external"
    other.mkdir()
    (root / "extra_package").symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="unpinned symlink"):
        module.verify_compiler(root, path, digest)


def test_selected_catalog_must_match_the_complete_seal_index(tmp_path):
    module = prepare_module()
    selected = tmp_path / "catalog.json"
    selected.write_text('{}\n')
    index = tmp_path / "SHA256SUMS"
    index.write_text(hashlib.sha256(selected.read_bytes()).hexdigest() + '  catalog.json\n')
    manifest = {"seal": {"sums": "SHA256SUMS", "sums_sha256": hashlib.sha256(index.read_bytes()).hexdigest()}}
    assert len(module.sealed_files(tmp_path, manifest, {"catalog.json"})) == 2
    selected.write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="sealed bytes"):
        module.sealed_files(tmp_path, manifest, {"catalog.json"})


def test_real_runtime_probe_handles_lazy_module_versions(tmp_path):
    import os
    module = prepare_module()
    # This invokes the same required proof solver and module-origin observation as the study.
    result = module.runtime_probe(Path(sys.executable), dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), tmp_path, 30)
    assert result["returncode"] == 0
    assert result["observed"]["solver_result"] == "unsat"
    assert isinstance(result["observed"]["modules"]["xdsl"]["version"], str)


def test_routing_preflight_reports_dangling_links_without_repairing_bundle(tmp_path):
    module = prepare_module()
    base = tmp_path / "network/programs/encode"
    names = ["dev/device_signatures.json", "dev/model.prepared.mlir", "dev/quantization-policy.json",
             "ref/model.prepared.mlir", "ref/quantization-policy.json"]
    lines = []
    for name in names:
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original\n")
        lines.append(module.sha(path) + "  " + str(path.relative_to(tmp_path)) + "\n")
    index = tmp_path / "SHA256SUMS"
    index.write_text("".join(lines))
    manifest = {"seal": {"sums": "SHA256SUMS", "sums_sha256": module.sha(index)}}
    programs = [{"model": "network", "program": "encode", "interfaces": 1}]
    assert module.routing_readiness(tmp_path, manifest, programs)["ready"]
    path = base / names[0]
    path.unlink()
    path.symlink_to(tmp_path / "missing-original.json")
    result = module.routing_readiness(tmp_path, manifest, programs)
    assert not result["ready"]
    assert result["files"][0]["status"] == "dangling_symlink"
    assert path.is_symlink() and not path.exists()
    (base / names[1]).write_text("changed\n")
    assert "changed" in {row["status"] for row in module.routing_readiness(tmp_path, manifest, programs)["files"]}


@pytest.mark.parametrize("external_child", [False, True])
def test_candidate_bootstrap_refuses_ambient_compiler_modules(tmp_path, external_child):
    import os
    import subprocess
    frozen, ambient = tmp_path / "frozen", tmp_path / "ambient"
    for root in (frozen, ambient):
        (root / "merlin").mkdir(parents=True)
    (frozen / "merlin/__init__.py").write_text("from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n")
    child = ambient if external_child else frozen
    (child / "merlin/example.py").write_text("value = 7\n")
    # Raw dependency roots must not execute site startup hooks.
    marker = tmp_path / "startup-hook-ran"
    (ambient / "poison.pth").write_text("import pathlib; pathlib.Path(" + repr(str(marker)) + ").touch()\n")
    entry = frozen / "compiler.py"
    entry.write_text("import merlin.example\nassert merlin.example.value == 7\n")
    origins = tmp_path / "origins"
    origins.mkdir()
    config = tmp_path / "candidate.json"
    config.write_text(json.dumps({"entrypoint": str(entry), "compiler_roots": [str(frozen)], "origins": str(origins)}))
    bootstrap = repo_root() / "examples/gemmini/model-study/candidate_python.py"
    result = subprocess.run([sys.executable, "-B", "-S", str(bootstrap), str(config), str(entry)],
                            env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(frozen), str(ambient)])),
                            capture_output=True, timeout=20)
    assert not marker.exists()
    records = [json.loads(path.read_text()) for path in origins.glob("*.json")]
    assert len(records) == 1
    assert all(row["allowed"] for row in records[0]["modules"])
    if external_child:
        assert result.returncode != 0
        assert b"outside frozen roots" in result.stderr
    else:
        assert result.returncode == 0, result.stderr.decode()
        assert any(row["module"] == "merlin.example" for row in records[0]["modules"])
