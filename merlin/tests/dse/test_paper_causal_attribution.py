"""Evidence gates for non-agentic v2 paper causal attribution."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.compare.freeze import sha256_paths
from merlin.compare.paper_ablation_generator import (
    GENERATOR_ID,
    MEASUREMENT_TOOL_ID,
    STRUCTURAL_TOOL_ID,
    benchmark,
    explain,
    inspect as inspect_structural,
    observe,
    produce_causal_pair,
    summarize,
)
from merlin.compare import paper_ablation_generator
from merlin.compare.paper_attribution import (
    CausalEvidenceError,
    attach_causal_attribution,
    causal_record,
    expected_binding,
    freeze_causal_evidence,
)


def _raw() -> dict:
    sha = lambda char: char * 64
    return {
        "label": "frozen-study", "target": "unit-test",
        "freeze": {
            "policy_sha256": sha("a"), "compiler_source_sha256": sha("b"),
            "runtime_sha256": sha("c"),
            "capture_session_identity_sha256": {"model": {"fp32": sha("d")}},
        },
        "models": [{"name": "model", "checkpoint": "checkpoint", "fidelity": "full",
                    "session": {"kind": "continuous", "stages": ["step"]},
                    "artifacts": {"fp32": {"sha256": sha("e")}}}],
        "backends": [
            {"name": "merlin_frozen", "kind": "compiler",
             "runtime": "merlin", "quantization": "none",
             "options": {"package_sha256": sha("f")}},
            {"name": "executorch_xnnpack", "kind": "external_runtime",
             "runtime": "executorch", "quantization": "none", "options": {
                "framework_source_sha256": sha("1"),
                "packages": {"model": {"fp32": {"sha256": sha("2")}}},
            }},
        ],
        "reporting": {},
    }


def _write_evidence(root: Path, raw: dict, *, self_attested: bool = False,
                    divergent_output: bool = False) -> Path:
    evidence = root / "evidence"
    evidence.mkdir()
    probe_source = evidence / "paper_k1_board_probe.c"
    probe_source.write_bytes(Path(paper_ablation_generator.__file__).with_name(
        "paper_k1_board_probe.c").read_bytes())
    probe_source_sha = sha256_paths([probe_source])
    build_inputs: dict[str, dict[str, Path | str]] = {}
    canonical = evidence / "canonical_pair.c"
    output_sha = "3" * 64
    canonical.write_text(r'''
#include <stdio.h>
void runtime_dispatch(unsigned long value);
int main(void) {
  unsigned long value = 1;
  for (unsigned long index = 1; index < 5000000; ++index) {
    value = value * 33u + index;
    /* MERLIN_TYPED_TRANSFORM:runtime_dispatch_elimination_v1 */
  }
  puts("{\"schema_version\":1,\"kind\":\"merlin_continuous_session_completion_v1\","
       "\"status\":\"pass\",\"output_sha256\":\"OUTPUT_SHA\"}");
  return 0;
}
'''.replace("OUTPUT_SHA", output_sha), encoding="utf-8")
    package = evidence / "dispatch.c"
    package.write_text(r'''
static volatile unsigned long sink;
__attribute__((noinline)) void runtime_dispatch(unsigned long value) { sink ^= value; }
''', encoding="utf-8")
    canonical_text = canonical.read_text(encoding="utf-8")
    for arm, fragment in (
            ("control", "runtime_dispatch(value);\n  runtime_dispatch(value);"),
            ("treatment", "runtime_dispatch(value);")):
        source_dir = evidence / f"{arm}-input"
        source_dir.mkdir()
        source = source_dir / "generated.c"
        source.write_text(canonical_text.replace(
            "/* MERLIN_TYPED_TRANSFORM:runtime_dispatch_elimination_v1 */", fragment),
            encoding="utf-8")
        build_inputs[arm] = {
            "package": package, "package_sha256": sha256_paths([package]),
            "source": source, "source_sha256": sha256_paths([source]),
        }
    raw["freeze"]["compiler_source_sha256"] = build_inputs["treatment"]["source_sha256"]
    raw["backends"][0]["options"]["package_sha256"] = build_inputs["treatment"][
        "package_sha256"]
    raw["backends"][1]["options"]["framework_source_sha256"] = build_inputs["control"][
        "source_sha256"]
    raw["backends"][1]["options"]["packages"]["model"]["fp32"]["sha256"] = build_inputs[
        "control"]["package_sha256"]
    binding = expected_binding(raw, model="model", precision="fp32", core_count=1,
                               comparator="executorch_xnnpack")
    binding_sha = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    generator = evidence / "ablation_generator.py"
    generator.write_bytes(Path(paper_ablation_generator.__file__).read_bytes())
    generator_sha = sha256_paths([generator])
    pair_receipt_path: Path | None = None
    if not self_attested:
        pair_contract = evidence / "pair_contract.yaml"
        pair_contract.write_text(yaml.safe_dump({
            "schema_version": 1, "kind": "paper_causal_pair_contract_v1", "status": "ready",
            "binding_sha256": binding_sha, "target": "unit-test",
            "intervention_id": "runtime_dispatch_elimination_v1",
            "canonical_source": {"path": canonical.name,
                                 "sha256": sha256_paths([canonical])},
            "dispatch_package": {"path": package.name,
                                 "sha256": sha256_paths([package])},
            "compiler_sha256": hashlib.sha256(Path("/usr/bin/cc").read_bytes()).hexdigest(),
            "objdump_sha256": hashlib.sha256(Path("/usr/bin/objdump").read_bytes()).hexdigest(),
            "timeout_seconds": 60, "warmup_iterations": 1, "measured_iterations": 3,
        }), encoding="utf-8")
        pair_receipt_path = produce_causal_pair(pair_contract, evidence / "causal-pair")
        pair_receipt = yaml.safe_load(pair_receipt_path.read_text(encoding="utf-8"))
        for arm in ("control", "treatment"):
            row = pair_receipt["arms"][arm]
            source = pair_receipt_path.parent / row["generated_source"]["path"]
            pair_package = pair_receipt_path.parent / "inputs/dispatch.c"
            build_inputs[arm] = {
                "package": pair_package,
                "package_sha256": sha256_paths([pair_package]),
                "source": source,
                "source_sha256": sha256_paths([source]),
                "pair_artifact": pair_receipt_path.parent / row["executable"]["path"],
            }
    if self_attested:
        ablation = {
            "kind": "frozen_ablation", "status": "pass", "binding_sha256": binding_sha,
            "control_sha256": "3" * 64, "treatment_sha256": "4" * 64,
            "changed": "the frozen fusion/pass treatment",
        }
        arm_rows = {variant: {"measurement_run": {"sha256": value * 64}}
                    for variant, value in (("control", "3"), ("treatment", "4"))}
    else:
        arm_rows = {}
        build_tool = evidence / "replay_build"
        build_tool.write_text("#!/bin/sh\nexec /usr/bin/cc \"$@\"\n", encoding="utf-8")
        build_tool.chmod(0o755)
        build_tool_sha = sha256_paths([build_tool])
        for arm, payload, samples in (
                ("control", b"unfused compiler artifact", [100, 101, 102]),
                ("treatment", b"fused compiler artifact", [70, 71, 72])):
            backend = binding["comparator_backend"] if arm == "control" else binding["ours"]
            artifact = evidence / f"{arm}.bin"
            package = build_inputs[arm]["package"]
            source = build_inputs[arm]["source"]
            assert isinstance(package, Path) and isinstance(source, Path)
            pair_artifact = build_inputs[arm]["pair_artifact"]
            assert isinstance(pair_artifact, Path)
            artifact.write_bytes(pair_artifact.read_bytes())
            artifact.chmod(0o755)
            artifact_sha = sha256_paths([artifact])
            build_receipt = evidence / f"{arm}_build_receipt.yaml"
            build_receipt.write_text(yaml.safe_dump({
                "schema_version": 2, "kind": "paper_executable_build_receipt_v2",
                "status": "pass", "backend": backend["backend"],
                "package_sha256": backend["package_sha256"],
                "source_sha256": backend["source_sha256"],
                "executable_sha256": artifact_sha,
                "package": {"path": str(package.relative_to(evidence)),
                            "sha256": build_inputs[arm]["package_sha256"]},
                "source": {"path": str(source.relative_to(evidence)),
                           "sha256": build_inputs[arm]["source_sha256"]},
                "invocation": {
                    "tool": {"path": build_tool.name, "sha256": build_tool_sha},
                    "argv": ["{tool}", "-O2", "-std=c11", "-fno-lto", "-fno-inline",
                             "-fno-ident", "-Wl,--build-id=none", "{source}", "{package}",
                             "-o", "{output}"], "cwd": ".",
                    "environment": {"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
                    "timeout_seconds": 60,
                },
            }), encoding="utf-8")
            build_sha = sha256_paths([build_receipt])
            execution_argv = [str(artifact.resolve()), "--paper-session"]
            command_sha = hashlib.sha256(json.dumps(
                execution_argv, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            contract = evidence / f"{arm}_benchmark_contract.yaml"
            contract_document = {
                "schema_version": 2, "kind": "paper_ablation_benchmark_contract_v2",
                "status": "ready", "variant": arm, "binding_sha256": binding_sha,
                "target": binding["target"], "model": binding["model"],
                "precision": binding["precision"], "core_count": binding["core_count"],
                "backend": backend["backend"], "package_sha256": backend["package_sha256"],
                "source_sha256": backend["source_sha256"],
                "runtime_sha256": binding["runtime_sha256"],
                "capture_sha256": binding["capture_sha256"],
                "capture_session_identity_sha256": binding["capture_session_identity_sha256"],
                "session_protocol_sha256": binding["session_protocol_sha256"],
                "artifact_sha256": artifact_sha, "run_id": f"run-{arm}",
                "metric": "latency_ns", "direction": "lower_is_better",
                "executable": {"path": artifact.name, "sha256": artifact_sha},
                "build_receipt": {"path": build_receipt.name, "sha256": build_sha},
                "execution": {"argv": ["{executable}", "--paper-session"], "cwd": ".",
                              "environment": {}, "timeout_seconds": 60,
                              "warmup_iterations": 1, "measured_iterations": 3},
                "board_probe": {"authority": "merlin_trusted_k1_csr_sysfs_probe_v1",
                                "source": {"path": probe_source.name,
                                           "sha256": probe_source_sha},
                                "environment": {},
                                "timeout_seconds": 60},
            }
            contract.write_text(yaml.safe_dump(contract_document), encoding="utf-8")
            raw_log = evidence / f"{arm}_raw_log.yaml"
            raw_document = benchmark(
                contract_document, root=evidence, generator_source_sha256=generator_sha)
            raw_log.write_text(yaml.safe_dump(raw_document), encoding="utf-8")
            observation = evidence / f"{arm}_observation.yaml"
            observation_document = observe(raw_document, generator_source_sha256=generator_sha)
            observation.write_text(yaml.safe_dump(observation_document), encoding="utf-8")
            observation_sha = sha256_paths([observation])
            measurement_run = evidence / f"{arm}_measurement_run.yaml"
            measurement_document = {
                "schema_version": 2, "kind": "frozen_ablation_measurement_run", "status": "pass",
                **{key: observation_document[key] for key in (
                    "variant", "binding_sha256", "target", "model", "precision", "core_count",
                    "backend", "package_sha256", "source_sha256", "runtime_sha256", "capture_sha256",
                    "capture_session_identity_sha256", "session_protocol_sha256", "artifact_sha256",
                    "metric", "direction", "build_receipt_sha256",
                    "benchmark_contract_sha256")},
                "run_id": f"run-{arm}", "command_sha256": command_sha,
                "executable_sha256": artifact_sha,
                "board_receipts_sha256": observation_document["board_receipts_sha256"],
                "raw_log_sha256": sha256_paths([raw_log]), "observation_sha256": observation_sha,
                "tool": {"id": MEASUREMENT_TOOL_ID, "source_sha256": generator_sha,
                         "command": ["python3", generator.name, "observe", raw_log.name,
                                     observation.name]},
            }
            measurement_run.write_text(yaml.safe_dump(measurement_document), encoding="utf-8")
            result = evidence / f"{arm}_result.yaml"
            result.write_text(json.dumps(summarize(
                observation_document, generator_source_sha256=generator_sha),
                sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
            arm_rows[arm] = {
                "artifact": {"path": artifact.name, "sha256": artifact_sha,
                             "backend": backend["backend"], "package_sha256": backend["package_sha256"],
                             "source_sha256": backend["source_sha256"],
                             "executable": {"path": artifact.name, "sha256": artifact_sha},
                             "build_receipt": {"path": build_receipt.name, "sha256": build_sha}},
                "benchmark_contract": {"path": contract.name, "sha256": sha256_paths([contract])},
                "raw_log": {"path": raw_log.name, "sha256": sha256_paths([raw_log])},
                "measurement_run": {"path": measurement_run.name,
                                    "sha256": sha256_paths([measurement_run])},
                "observation": {"path": observation.name, "sha256": observation_sha},
                "result": {"path": result.name, "sha256": sha256_paths([result])},
            }
        if divergent_output:
            forged_pair = yaml.safe_load(pair_receipt_path.read_text(encoding="utf-8"))
            forged_pair["functional_stdout_sha256"] = "4" * 64
            pair_receipt_path.write_text(yaml.safe_dump(forged_pair), encoding="utf-8")
        ablation = {
            "schema_version": 2, "kind": "frozen_ablation", "status": "pass",
            "binding_sha256": binding_sha, "changed": "runtime_dispatch_elimination_v1",
            "intervention": {
                "id": "runtime_dispatch_elimination_v1", "scope": "compiler_lowering",
                "control": "disabled", "treatment": "enabled",
                "isolated_change": "runtime_dispatch_sites",
            },
            "metric": "latency_ns", "direction": "lower_is_better",
            "generator": {"kind": "deterministic_non_agentic", "agentic": False,
                          "id": GENERATOR_ID,
                          "source": {"path": generator.name, "sha256": generator_sha},
                          "commands": {
                              "benchmark": ["python3", generator.name, "benchmark", "{contract}", "{raw_log}"],
                              "observe": ["python3", generator.name, "observe", "{raw_log}", "{observation}"],
                              "summarize": ["python3", generator.name, "summarize", "{observation}", "{result}"],
                          }},
            **arm_rows,
            "controller_pair": {
                "path": str(pair_receipt_path.relative_to(evidence)),
                "sha256": sha256_paths([pair_receipt_path]),
            },
        }
    ablation_path = evidence / "ablation.yaml"
    ablation_path.write_text(yaml.safe_dump(ablation), encoding="utf-8")
    inspection_path = evidence / "structural_inspection.yaml"
    inspection = {
        "schema_version": 2, "kind": "frozen_structural_inspection_contract", "status": "pass",
        "binding_sha256": binding_sha, "ablation_sha256": sha256_paths([ablation_path]),
        "mechanism": "runtime_dispatch_markers",
        "intervention_id": "runtime_dispatch_elimination_v1",
        "control_artifact_sha256": arm_rows.get("control", {}).get("artifact", {}).get("sha256", "0" * 64),
        "treatment_artifact_sha256": arm_rows.get("treatment", {}).get("artifact", {}).get("sha256", "0" * 64),
        "control_package_sha256": binding["comparator_backend"]["package_sha256"],
        "treatment_package_sha256": binding["ours"]["package_sha256"],
        "control_source_sha256": binding["comparator_backend"]["source_sha256"],
        "treatment_source_sha256": binding["ours"]["source_sha256"],
        "control_measurement_run_sha256": arm_rows["control"]["measurement_run"]["sha256"],
        "treatment_measurement_run_sha256": arm_rows["treatment"]["measurement_run"]["sha256"],
        "control_build_receipt_sha256": arm_rows.get("control", {}).get("artifact", {}).get(
            "build_receipt", {}).get("sha256", "0" * 64),
        "treatment_build_receipt_sha256": arm_rows.get("treatment", {}).get("artifact", {}).get(
            "build_receipt", {}).get("sha256", "0" * 64),
        "control_artifact": {
            "path": arm_rows.get("control", {}).get("artifact", {}).get("path", "missing"),
            "sha256": arm_rows.get("control", {}).get("artifact", {}).get("sha256", "0" * 64)},
        "treatment_artifact": {
            "path": arm_rows.get("treatment", {}).get("artifact", {}).get("path", "missing"),
            "sha256": arm_rows.get("treatment", {}).get("artifact", {}).get("sha256", "0" * 64)},
    }
    inspection_path.write_text(yaml.safe_dump(inspection), encoding="utf-8")
    structural_result_path = evidence / "structural_result.yaml"
    if self_attested:
        structural_result_path.write_text("{}\n", encoding="utf-8")
    else:
        structural_result_path.write_text(json.dumps(inspect_structural(
            inspection, root=evidence, generator_source_sha256=generator_sha),
            sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    structural = {
        "schema_version": 2, "kind": "frozen_structural", "status": "pass",
        "binding_sha256": binding_sha,
        "generator": {
            "kind": "deterministic_non_agentic", "agentic": False,
            "id": STRUCTURAL_TOOL_ID,
            "source": {"path": generator.name, "sha256": generator_sha},
            "command": ["python3", generator.name, "inspect", "{inspection}", "{result}"],
        },
        "inspection": {"path": inspection_path.name,
                       "sha256": sha256_paths([inspection_path])},
        "result": {"path": structural_result_path.name,
                   "sha256": sha256_paths([structural_result_path])},
    }
    (evidence / "structural.yaml").write_text(yaml.safe_dump(structural), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "records": [{
            "model": "model", "precision": "fp32", "core_count": 1,
            "comparator": "executorch_xnnpack", "binding": binding,
            "binding_sha256": binding_sha,
            "ablation": {"path": "ablation.yaml", "sha256": sha256_paths([evidence / "ablation.yaml"])},
            "structural": {"path": "structural.yaml", "sha256": sha256_paths([evidence / "structural.yaml"])},
        }],
    }
    path = evidence / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return path


def _spec(raw: dict):
    ours = SimpleNamespace(name="merlin_frozen", kind="compiler")
    comparator = SimpleNamespace(name="executorch_xnnpack", kind="external_runtime")
    return SimpleNamespace(reporting=raw["reporting"], backends=(ours, comparator),
                           canonical_dict=lambda: copy.deepcopy(raw), sha256=lambda: "9" * 64)


def _result(backend: str) -> dict:
    return {
        "model": "model", "backend": backend, "precision": "fp32", "core_count": 1,
        "study_label": "frozen-study", "target": "unit-test", "checkpoint": "checkpoint",
        "fidelity": "full", "runtime": ("merlin" if backend == "merlin_frozen"
                                            else "executorch"),
        "quantization": "none", "session": {"kind": "continuous", "stages": ["step"]},
        "artifact_sha256": "e" * 64,
        "provenance": {"study_sha256": "9" * 64,
                       "compiler_policy_sha256": "a" * 64,
                       "compiler_source_sha256": "b" * 64, "runtime_sha256": "c" * 64},
    }


def _result_with_backend_evidence(backend: str, raw: dict) -> dict:
    result = _result(backend)
    result["provenance"]["compiler_source_sha256"] = raw["freeze"][
        "compiler_source_sha256"]
    result["provenance"]["capture_session_identity_sha256"] = "d" * 64
    if backend == "merlin_frozen":
        result["provenance"]["package_sha256"] = raw["backends"][0]["options"][
            "package_sha256"]
    else:
        options = raw["backends"][1]["options"]
        result["provenance"].update(
            framework_source_sha256=options["framework_source_sha256"],
            framework_package_sha256=options["packages"]["model"]["fp32"]["sha256"])
    return result


def test_frozen_ablation_and_structural_evidence_produce_why_how_without_timing(tmp_path):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}
    freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)

    ours = _result_with_backend_evidence("merlin_frozen", raw)
    comparator = _result_with_backend_evidence("executorch_xnnpack", raw)
    first = causal_record(_spec(raw), ours, comparator)
    # There are deliberately no timing samples in either input; causal text comes only from the
    # frozen structural artifact.  Adding arbitrary latency cannot change it.
    ours["timing"] = {"samples": [1]}
    comparator["timing"] = {"samples": [999999999]}
    second = causal_record(_spec(raw), ours, comparator)

    assert first["status"] == second["status"] == "available"
    assert first["why"] == second["why"]
    assert first["how"] == second["how"]
    assert first["evidence"]["ablation_sha256"] == sha256_paths([manifest.parent / "ablation.yaml"])


def test_ablation_requires_identical_cross_arm_functional_outputs(tmp_path):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw, divergent_output=True)
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}

    with pytest.raises(CausalEvidenceError, match="functional output|correctness"):
        freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)


def test_changed_structural_bytes_or_result_binding_is_structured_unavailable(tmp_path):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}
    freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)
    spec = _spec(raw)
    ours = _result_with_backend_evidence("merlin_frozen", raw)
    comparator = _result_with_backend_evidence("executorch_xnnpack", raw)
    assert causal_record(spec, ours, comparator)["status"] == "available"

    (manifest.parent / "structural.yaml").write_text("changed after freeze\n", encoding="utf-8")
    unavailable = causal_record(spec, ours, comparator)
    assert unavailable["status"] == "unavailable"
    assert "structural" in unavailable["reason"]


def test_frozen_evidence_is_bound_to_the_full_noncausal_study_identity(tmp_path):
    raw = _raw()
    raw["reporting"]["performance_claims"] = {"win_median_ratio_max": 0.95}
    manifest = _write_evidence(tmp_path, raw)
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}
    freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)
    raw["reporting"]["performance_claims"]["win_median_ratio_max"] = 0.50

    record = causal_record(
        _spec(raw), _result_with_backend_evidence("merlin_frozen", raw),
        _result_with_backend_evidence("executorch_xnnpack", raw))

    assert record["status"] == "unavailable"
    assert "study identity" in record["reason"]


def test_mismatched_comparator_binding_fails_during_freeze(tmp_path):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    document["records"][0]["binding"]["comparator_backend"]["package_sha256"] = "9" * 64
    manifest.write_text(yaml.safe_dump(document), encoding="utf-8")
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}

    with pytest.raises(CausalEvidenceError, match="binding differs from frozen study"):
        freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)


def test_causal_binding_rejects_a_baseline_package_without_source_identity():
    raw = _raw()
    raw["backends"][1] = {
        "name": "hand_v0", "kind": "frozen_baseline", "runtime": "merlin",
        "quantization": "none", "options": {"package_sha256": "2" * 64},
    }

    with pytest.raises(CausalEvidenceError, match="source"):
        expected_binding(
            raw, model="model", precision="fp32", core_count=1, comparator="hand_v0")


def test_causal_binding_rejects_a_manually_injected_unattested_baseline_source_digest():
    raw = _raw()
    raw["backends"][1] = {
        "name": "hand_v0", "kind": "frozen_baseline", "runtime": "merlin",
        "quantization": "none", "options": {
            "package_sha256": "2" * 64, "kernel_source_sha256": "3" * 64,
        },
    }

    with pytest.raises(CausalEvidenceError, match="source_paths"):
        expected_binding(
            raw, model="model", precision="fp32", core_count=1, comparator="hand_v0")


def test_causal_binding_recomputes_declared_baseline_source_bytes(tmp_path):
    source = tmp_path / "baseline.c"
    source.write_text("int baseline(void) { return 0; }\n", encoding="utf-8")
    raw = _raw()
    raw["backends"][1] = {
        "name": "hand_v0", "kind": "frozen_baseline", "runtime": "merlin",
        "quantization": "none", "options": {
            "package_sha256": "2" * 64, "source_paths": [str(source)],
            "kernel_source_sha256": "3" * 64,
        },
    }

    with pytest.raises(CausalEvidenceError, match="source digest differs"):
        expected_binding(
            raw, model="model", precision="fp32", core_count=1, comparator="hand_v0")


def test_missing_evidence_yields_per_comparator_structured_unavailable():
    raw = _raw()
    freeze_causal_evidence(raw, root=Path.cwd(), hasher=sha256_paths)
    ours, comparator = _result("merlin_frozen"), _result("executorch_xnnpack")
    attach_causal_attribution(_spec(raw), [ours, comparator])

    record = ours["causal_attribution"]["records"][0]
    assert record["comparator"] == "executorch_xnnpack"
    assert record["status"] == "unavailable"
    assert "no frozen causal evidence" in record["reason"]


def test_self_attested_ablation_without_retained_artifacts_is_rejected(tmp_path):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw, self_attested=True)
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}

    with pytest.raises(CausalEvidenceError, match="retained|non-agentic|closed"):
        freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)


def test_ablation_arms_must_link_artifacts_to_the_bound_backend_packages(tmp_path):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    ablation_path = manifest.parent / "ablation.yaml"
    ablation = yaml.safe_load(ablation_path.read_text(encoding="utf-8"))
    ablation["treatment"]["artifact"]["package_sha256"] = "8" * 64
    ablation_path.write_text(yaml.safe_dump(ablation), encoding="utf-8")
    _refresh_ablation_digest(manifest)
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}

    with pytest.raises(CausalEvidenceError, match="bound backend package"):
        freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)


def test_ablation_arms_require_retained_measurement_run_and_raw_log(tmp_path):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    ablation_path = manifest.parent / "ablation.yaml"
    ablation = yaml.safe_load(ablation_path.read_text(encoding="utf-8"))
    del ablation["control"]["measurement_run"]
    ablation_path.write_text(yaml.safe_dump(ablation), encoding="utf-8")
    _refresh_ablation_digest(manifest)
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}

    with pytest.raises(CausalEvidenceError, match="measurement[_ ]run"):
        freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)


def test_measurement_receipt_executable_must_be_the_retained_arm_artifact(tmp_path):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    raw_path = manifest.parent / "control_raw_log.yaml"
    raw_log = yaml.safe_load(raw_path.read_text(encoding="utf-8"))
    raw_log["executable_sha256"] = "8" * 64
    raw_path.write_text(yaml.safe_dump(raw_log), encoding="utf-8")
    run_path = manifest.parent / "control_measurement_run.yaml"
    measurement_run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
    measurement_run["executable_sha256"] = "8" * 64
    measurement_run["raw_log_sha256"] = sha256_paths([raw_path])
    run_path.write_text(yaml.safe_dump(measurement_run), encoding="utf-8")
    ablation_path = manifest.parent / "ablation.yaml"
    ablation = yaml.safe_load(ablation_path.read_text(encoding="utf-8"))
    ablation["control"]["raw_log"]["sha256"] = sha256_paths([raw_path])
    ablation["control"]["measurement_run"]["sha256"] = sha256_paths([run_path])
    ablation_path.write_text(yaml.safe_dump(ablation), encoding="utf-8")
    _refresh_ablation_digest(manifest)
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}

    with pytest.raises(CausalEvidenceError, match="executable"):
        freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)


def test_measurement_receipt_requires_observed_board_condition_endpoints(tmp_path):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    raw_path = manifest.parent / "control_raw_log.yaml"
    raw_log = yaml.safe_load(raw_path.read_text(encoding="utf-8"))
    raw_log["board_receipts"]["after"] = json.dumps({
        **json.loads(raw_log["board_receipts"]["after"]), "current_khz": 1200000})
    raw_path.write_text(yaml.safe_dump(raw_log), encoding="utf-8")
    run_path = manifest.parent / "control_measurement_run.yaml"
    measurement_run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
    measurement_run["raw_log_sha256"] = sha256_paths([raw_path])
    observation_path = manifest.parent / "control_observation.yaml"
    # Do not refresh the observation: the trusted replay must catch the altered raw board receipt.
    run_path.write_text(yaml.safe_dump(measurement_run), encoding="utf-8")
    ablation_path = manifest.parent / "ablation.yaml"
    ablation = yaml.safe_load(ablation_path.read_text(encoding="utf-8"))
    ablation["control"]["raw_log"]["sha256"] = sha256_paths([raw_path])
    ablation["control"]["measurement_run"]["sha256"] = sha256_paths([run_path])
    ablation_path.write_text(yaml.safe_dump(ablation), encoding="utf-8")
    _refresh_ablation_digest(manifest)
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}

    with pytest.raises(CausalEvidenceError, match="board|frequency"):
        freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)


def test_measurement_run_rejects_unrecognized_agentic_provenance(tmp_path):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    run_path = manifest.parent / "control_measurement_run.yaml"
    measurement_run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
    measurement_run["agentic"] = True
    run_path.write_text(yaml.safe_dump(measurement_run), encoding="utf-8")
    ablation_path = manifest.parent / "ablation.yaml"
    ablation = yaml.safe_load(ablation_path.read_text(encoding="utf-8"))
    ablation["control"]["measurement_run"]["sha256"] = sha256_paths([run_path])
    ablation_path.write_text(yaml.safe_dump(ablation), encoding="utf-8")
    _refresh_ablation_digest(manifest)
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}

    with pytest.raises(CausalEvidenceError, match="measurement run.*fields"):
        freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)


def test_raw_measurement_log_rejects_unrecognized_agentic_provenance(tmp_path):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    raw_path = manifest.parent / "control_raw_log.yaml"
    raw_log = yaml.safe_load(raw_path.read_text(encoding="utf-8"))
    raw_log["agentic"] = True
    raw_path.write_text(yaml.safe_dump(raw_log), encoding="utf-8")
    run_path = manifest.parent / "control_measurement_run.yaml"
    measurement_run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
    measurement_run["raw_log_sha256"] = sha256_paths([raw_path])
    run_path.write_text(yaml.safe_dump(measurement_run), encoding="utf-8")
    ablation_path = manifest.parent / "ablation.yaml"
    ablation = yaml.safe_load(ablation_path.read_text(encoding="utf-8"))
    ablation["control"]["raw_log"]["sha256"] = sha256_paths([raw_path])
    ablation["control"]["measurement_run"]["sha256"] = sha256_paths([run_path])
    ablation_path.write_text(yaml.safe_dump(ablation), encoding="utf-8")
    _refresh_ablation_digest(manifest)
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}

    with pytest.raises(CausalEvidenceError, match="raw.*closed|unrecognized"):
        freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)


def test_nonagentic_ablation_rejects_self_authored_aet_receipt(tmp_path):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    evidence = manifest.parent
    ablation_path = evidence / "ablation.yaml"
    ablation = yaml.safe_load(ablation_path.read_text(encoding="utf-8"))
    ablation["control"]["aet_run_record"] = {
        "path": "self-authored.json", "sha256": "0" * 64,
    }
    ablation_path.write_text(yaml.safe_dump(ablation), encoding="utf-8")
    _refresh_ablation_digest(manifest)
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}

    with pytest.raises(CausalEvidenceError, match="fields|unrecognized"):
        freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)


def test_nonagentic_ablation_rejects_authored_zero_token_stream(tmp_path):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    evidence = manifest.parent
    ablation_path = evidence / "ablation.yaml"
    ablation = yaml.safe_load(ablation_path.read_text(encoding="utf-8"))
    ablation["control"]["aet_event_stream"] = {
        "schema_version": 1, "kind": "aet_raw_event_stream_v1", "events": [],
    }
    ablation_path.write_text(yaml.safe_dump(ablation), encoding="utf-8")
    _refresh_ablation_digest(manifest)
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}

    with pytest.raises(CausalEvidenceError, match="fields|unrecognized"):
        freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)


def _refresh_ablation_digest(manifest: Path) -> None:
    document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    document["records"][0]["ablation"]["sha256"] = sha256_paths([
        manifest.parent / "ablation.yaml"])
    manifest.write_text(yaml.safe_dump(document), encoding="utf-8")


def _refresh_structural_binding(manifest: Path) -> None:
    evidence = manifest.parent
    generator_sha = sha256_paths([evidence / "ablation_generator.py"])
    ablation_path = evidence / "ablation.yaml"
    inspection_path = evidence / "structural_inspection.yaml"
    inspection = yaml.safe_load(inspection_path.read_text(encoding="utf-8"))
    inspection["ablation_sha256"] = sha256_paths([ablation_path])
    inspection_path.write_text(yaml.safe_dump(inspection), encoding="utf-8")
    result_path = evidence / "structural_result.yaml"
    result_path.write_text(yaml.safe_dump(inspect_structural(
        inspection, root=evidence, generator_source_sha256=generator_sha)), encoding="utf-8")
    structural_path = evidence / "structural.yaml"
    structural = yaml.safe_load(structural_path.read_text(encoding="utf-8"))
    structural["inspection"]["sha256"] = sha256_paths([inspection_path])
    structural["result"]["sha256"] = sha256_paths([result_path])
    structural_path.write_text(yaml.safe_dump(structural), encoding="utf-8")
    document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    document["records"][0]["structural"]["sha256"] = sha256_paths([structural_path])
    manifest.write_text(yaml.safe_dump(document), encoding="utf-8")


def test_retained_ablation_artifact_mutation_after_freeze_is_unavailable(tmp_path):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}
    freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)
    (manifest.parent / "control.bin").write_bytes(b"changed after freeze")

    record = causal_record(
        _spec(raw), _result_with_backend_evidence("merlin_frozen", raw),
        _result_with_backend_evidence("executorch_xnnpack", raw))

    assert record["status"] == "unavailable"
    assert "control.artifact digest" in record["reason"]


def test_agentic_generator_provenance_is_rejected_even_when_manifest_hash_is_refreshed(tmp_path):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    ablation_path = manifest.parent / "ablation.yaml"
    ablation = yaml.safe_load(ablation_path.read_text(encoding="utf-8"))
    ablation["generator"]["agentic"] = True
    ablation_path.write_text(yaml.safe_dump(ablation), encoding="utf-8")
    _refresh_ablation_digest(manifest)
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}

    with pytest.raises(CausalEvidenceError, match="non-agentic generator provenance"):
        freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)


def test_arbitrary_generator_source_cannot_self_attest_as_non_agentic(tmp_path):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    evidence = manifest.parent
    generator_path = evidence / "ablation_generator.py"
    generator_path.write_text(
        "# attacker-controlled program that merely claims to be non-agentic\n",
        encoding="utf-8",
    )
    generator_sha = sha256_paths([generator_path])
    ablation_path = evidence / "ablation.yaml"
    ablation = yaml.safe_load(ablation_path.read_text(encoding="utf-8"))
    ablation["generator"]["source"]["sha256"] = generator_sha
    for variant in ("control", "treatment"):
        result_path = evidence / f"{variant}_result.yaml"
        result = yaml.safe_load(result_path.read_text(encoding="utf-8"))
        result["generator_source_sha256"] = generator_sha
        result_path.write_text(yaml.safe_dump(result), encoding="utf-8")
        ablation[variant]["result"]["sha256"] = sha256_paths([result_path])
    ablation_path.write_text(yaml.safe_dump(ablation), encoding="utf-8")
    _refresh_ablation_digest(manifest)
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}

    with pytest.raises(CausalEvidenceError, match="trusted non-agentic generator"):
        freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)


def test_free_form_structural_why_how_cannot_become_claim_authority(tmp_path):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    structural_path = manifest.parent / "structural.yaml"
    structural = yaml.safe_load(structural_path.read_text(encoding="utf-8"))
    structural["facts"] = [{"fabricated": True}]
    structural["why"] = "ARBITRARY WHY"
    structural["how"] = "ARBITRARY HOW"
    structural_path.write_text(yaml.safe_dump(structural), encoding="utf-8")
    manifest_document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    manifest_document["records"][0]["structural"]["sha256"] = sha256_paths([
        structural_path])
    manifest.write_text(yaml.safe_dump(manifest_document), encoding="utf-8")
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}

    with pytest.raises(CausalEvidenceError, match="trusted structural"):
        freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)


def test_generator_invocation_cannot_hide_an_agentic_wrapper(tmp_path):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    ablation_path = manifest.parent / "ablation.yaml"
    ablation = yaml.safe_load(ablation_path.read_text(encoding="utf-8"))
    ablation["generator"]["commands"] = {
        "observe": ["agent-runner", "--then", ablation["generator"]["source"]["path"]],
        "summarize": ablation["generator"]["commands"]["summarize"],
    }
    ablation_path.write_text(yaml.safe_dump(ablation), encoding="utf-8")
    _refresh_ablation_digest(manifest)
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}

    with pytest.raises(CausalEvidenceError, match="canonical argv"):
        freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)


def test_retained_ablation_path_traversal_is_rejected(tmp_path):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    ablation_path = manifest.parent / "ablation.yaml"
    ablation = yaml.safe_load(ablation_path.read_text(encoding="utf-8"))
    ablation["control"]["artifact"]["path"] = "../outside.bin"
    ablation_path.write_text(yaml.safe_dump(ablation), encoding="utf-8")
    _refresh_ablation_digest(manifest)
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}

    with pytest.raises(CausalEvidenceError, match="relative path"):
        freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)


def test_retained_ablation_results_must_show_the_declared_improvement(tmp_path):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    raw_path = manifest.parent / "treatment_raw_log.yaml"
    raw_log = yaml.safe_load(raw_path.read_text(encoding="utf-8"))
    raw_log["samples_ns"] = [200, 201, 202]
    raw_path.write_text(yaml.safe_dump(raw_log), encoding="utf-8")
    observation_path = manifest.parent / "treatment_observation.yaml"
    generator_sha = sha256_paths([manifest.parent / "ablation_generator.py"])
    observation = observe(raw_log, generator_source_sha256=generator_sha)
    observation_path.write_text(yaml.safe_dump(observation), encoding="utf-8")
    result_path = manifest.parent / "treatment_result.yaml"
    result = summarize(observation, generator_source_sha256=generator_sha)
    result_path.write_text(yaml.safe_dump(result), encoding="utf-8")
    run_path = manifest.parent / "treatment_measurement_run.yaml"
    measurement_run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
    measurement_run["raw_log_sha256"] = sha256_paths([raw_path])
    measurement_run["observation_sha256"] = sha256_paths([observation_path])
    run_path.write_text(yaml.safe_dump(measurement_run), encoding="utf-8")
    ablation_path = manifest.parent / "ablation.yaml"
    ablation = yaml.safe_load(ablation_path.read_text(encoding="utf-8"))
    ablation["treatment"]["raw_log"]["sha256"] = sha256_paths([raw_path])
    ablation["treatment"]["measurement_run"]["sha256"] = sha256_paths([run_path])
    ablation["treatment"]["observation"]["sha256"] = sha256_paths([observation_path])
    ablation["treatment"]["result"]["sha256"] = sha256_paths([result_path])
    ablation_path.write_text(yaml.safe_dump(ablation), encoding="utf-8")
    _refresh_ablation_digest(manifest)
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}

    with pytest.raises(CausalEvidenceError, match="does not improve|canonical benchmark replay"):
        freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)


def test_forged_result_cannot_replace_the_replay_of_retained_observations(tmp_path):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    result_path = manifest.parent / "treatment_result.yaml"
    result = yaml.safe_load(result_path.read_text(encoding="utf-8"))
    result.update(samples=[1, 1, 1], median=1)
    result_path.write_text(yaml.safe_dump(result), encoding="utf-8")
    ablation_path = manifest.parent / "ablation.yaml"
    ablation = yaml.safe_load(ablation_path.read_text(encoding="utf-8"))
    ablation["treatment"]["result"]["sha256"] = sha256_paths([result_path])
    ablation_path.write_text(yaml.safe_dump(ablation), encoding="utf-8")
    _refresh_ablation_digest(manifest)
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}

    with pytest.raises(CausalEvidenceError, match="replay"):
        freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)


@pytest.mark.parametrize(("side", "mutation"), [
    ("ours", lambda row: row["provenance"].update(compiler_policy_sha256="8" * 64)),
    ("ours", lambda row: row["provenance"].update(compiler_source_sha256="8" * 64)),
    ("ours", lambda row: row["provenance"].update(runtime_sha256="8" * 64)),
    ("ours", lambda row: row.update(artifact_sha256="8" * 64)),
    ("ours", lambda row: row["provenance"].update(package_sha256="8" * 64)),
    ("ours", lambda row: row["provenance"].update(
        capture_session_identity_sha256="8" * 64)),
    ("ours", lambda row: row.update(session={"kind": "different", "stages": ["step"]})),
    ("ours", lambda row: row["provenance"].update(study_sha256="8" * 64)),
    ("ours", lambda row: row.update(runtime="different")),
    ("ours", lambda row: row.update(quantization="different")),
    ("ours", lambda row: row.update(study_label="different")),
    ("ours", lambda row: row.update(target="different")),
    ("ours", lambda row: row.update(checkpoint="different")),
    ("ours", lambda row: row.update(fidelity="different")),
    ("comparator", lambda row: row["provenance"].update(
        framework_package_sha256="8" * 64)),
    ("comparator", lambda row: row["provenance"].update(
        framework_source_sha256="8" * 64)),
    ("comparator", lambda row: row["provenance"].update(
        capture_session_identity_sha256="8" * 64)),
    ("comparator", lambda row: row.update(artifact_sha256="8" * 64)),
    ("comparator", lambda row: row.update(model="different")),
    ("comparator", lambda row: row.update(precision="w8a8")),
    ("comparator", lambda row: row.update(core_count=8)),
    ("comparator", lambda row: row.update(backend="other")),
    ("comparator", lambda row: row.update(runtime="different")),
    ("comparator", lambda row: row.update(quantization="different")),
    ("comparator", lambda row: row.update(session={"kind": "different", "stages": ["step"]})),
    ("comparator", lambda row: row["provenance"].update(study_sha256="8" * 64)),
    ("comparator", lambda row: row["provenance"].update(compiler_policy_sha256="8" * 64)),
    ("comparator", lambda row: row["provenance"].update(compiler_source_sha256="8" * 64)),
    ("comparator", lambda row: row["provenance"].update(runtime_sha256="8" * 64)),
    ("comparator", lambda row: row.update(study_label="different")),
    ("comparator", lambda row: row.update(target="different")),
    ("comparator", lambda row: row.update(checkpoint="different")),
    ("comparator", lambda row: row.update(fidelity="different")),
])
def test_result_bytes_must_match_every_frozen_causal_binding(
        tmp_path, side, mutation):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}
    freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)
    ours = _result_with_backend_evidence("merlin_frozen", raw)
    comparator = _result_with_backend_evidence("executorch_xnnpack", raw)

    mutation(ours if side == "ours" else comparator)
    record = causal_record(_spec(raw), ours, comparator)

    assert record["status"] == "unavailable"


def test_trusted_observer_rejects_self_authored_samples_and_unknown_fields():
    raw = {
        "kind": "frozen_ablation_raw_log", "status": "pass",
        "samples": [1], "agentic": True,
    }

    with pytest.raises(ValueError, match="closed|unrecognized|receipt"):
        observe(raw, generator_source_sha256="a" * 64)


def test_trusted_summarizer_rejects_unknown_agentic_fields():
    observation = {
        "kind": "frozen_ablation_observation", "status": "pass",
        "variant": "control", "binding_sha256": "a" * 64,
        "artifact_sha256": "b" * 64, "metric": "latency_ns",
        "direction": "lower_is_better", "samples": [10, 11, 12],
        "agentic": True,
    }

    with pytest.raises(ValueError, match="closed|unrecognized"):
        summarize(observation, generator_source_sha256="a" * 64)


def test_trusted_structural_analyzer_does_not_accept_authored_counts():
    inspection = {
        "kind": "frozen_structural_inspection", "status": "pass",
        "binding_sha256": "a" * 64, "ablation_sha256": "b" * 64,
        "mechanism": "dispatches", "control_value": 100,
        "treatment_value": 0,
    }

    with pytest.raises(ValueError, match="trace|artifact|contract|unrecognized"):
        explain(inspection, generator_source_sha256="a" * 64)


def test_production_benchmark_executes_exact_binary_and_board_probe(tmp_path):
    executable = tmp_path / "benchmark"
    executable.write_text(
        "#!/bin/sh\nprintf '%s\\n' '{\"schema_version\":1,\"kind\":"
        "\"merlin_continuous_session_completion_v1\",\"status\":\"pass\","
        "\"output_sha256\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}'\n",
        encoding="utf-8")
    executable.chmod(0o755)
    executable_sha = sha256_paths([executable])
    package = tmp_path / "benchmark.package"
    package.write_bytes(executable.read_bytes())
    package.chmod(0o755)
    package_sha = sha256_paths([package])
    source = tmp_path / "benchmark.source"
    source.write_text("retained benchmark source root\n", encoding="utf-8")
    source_sha = sha256_paths([source])
    build_tool = tmp_path / "build-tool"
    build_tool.write_text(
        "#!/bin/sh\nset -eu\ncp \"$1\" \"$3\"\nchmod 755 \"$3\"\n",
        encoding="utf-8")
    build_tool.chmod(0o755)
    build_receipt = tmp_path / "build.yaml"
    build_receipt.write_text(yaml.safe_dump({
        "schema_version": 2, "kind": "paper_executable_build_receipt_v2", "status": "pass",
        "backend": "backend", "package_sha256": package_sha, "source_sha256": source_sha,
        "executable_sha256": executable_sha,
        "package": {"path": package.name, "sha256": package_sha},
        "source": {"path": source.name, "sha256": source_sha},
        "invocation": {
            "tool": {"path": build_tool.name, "sha256": sha256_paths([build_tool])},
            "argv": ["{tool}", "{package}", "{source}", "{output}"], "cwd": ".",
            "environment": {}, "timeout_seconds": 10,
        },
    }), encoding="utf-8")
    identity = {
        "binding_sha256": "a" * 64, "variant": "control", "target": "unit-test",
        "model": "model", "precision": "fp32", "core_count": 1, "backend": "backend",
        "package_sha256": package_sha, "source_sha256": source_sha,
        "runtime_sha256": "d" * 64, "capture_sha256": "e" * 64,
        "capture_session_identity_sha256": "f" * 64, "session_protocol_sha256": "1" * 64,
        "artifact_sha256": executable_sha, "run_id": "aet-run", "metric": "latency_ns",
        "direction": "lower_is_better",
    }
    contract = {
        "schema_version": 2, "kind": "paper_ablation_benchmark_contract_v2", "status": "ready",
        **identity,
        "executable": {"path": executable.name, "sha256": executable_sha},
        "build_receipt": {"path": build_receipt.name,
                          "sha256": sha256_paths([build_receipt])},
        "execution": {"argv": ["{executable}"], "cwd": ".", "environment": {},
                      "timeout_seconds": 10, "warmup_iterations": 1,
                      "measured_iterations": 3},
        "board_probe": {"authority": "merlin_trusted_k1_csr_sysfs_probe_v1",
                        "source": {"path": "paper_k1_board_probe.c", "sha256": ""},
                        "environment": {},
                        "timeout_seconds": 10},
    }
    probe_source = tmp_path / "paper_k1_board_probe.c"
    probe_source.write_bytes(Path(paper_ablation_generator.__file__).with_name(
        "paper_k1_board_probe.c").read_bytes())
    contract["board_probe"]["source"]["sha256"] = sha256_paths([probe_source])
    source_sha = sha256_paths([Path(paper_ablation_generator.__file__)])

    receipt = benchmark(contract, root=tmp_path, generator_source_sha256=source_sha)
    observed = observe(receipt, generator_source_sha256=source_sha)

    assert len(observed["samples"]) == 3
    assert all(sample > 0 for sample in observed["samples"])
    assert receipt["execution_argv"] == [str(executable.resolve())]
    assert receipt["executable_sha256"] == executable_sha
    assert receipt["build_receipt_sha256"] == sha256_paths([build_receipt])


def test_shell_board_probe_forgery_is_not_authority_even_with_refreshed_hashes(tmp_path):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    evidence = manifest.parent
    probe_source = evidence / "paper_k1_board_probe.c"
    probe_source.write_text(
        "#!/bin/sh\nprintf '%s\\n' '{\"schema_version\":1,\"kind\":"
        "\"merlin_board_probe_v1\",\"identity\":\"forged\",\"vlen_bits\":256,"
        "\"vlen_source\":\"csr\",\"governor\":\"performance\","
        "\"current_khz\":1600000,\"max_khz\":1600000,"
        "\"max_thermal_millic\":1}'\n",
        encoding="utf-8")
    probe_sha = sha256_paths([probe_source])
    ablation_path = evidence / "ablation.yaml"
    ablation = yaml.safe_load(ablation_path.read_text(encoding="utf-8"))
    for arm in ("control", "treatment"):
        contract_path = evidence / f"{arm}_benchmark_contract.yaml"
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        contract["board_probe"]["source"]["sha256"] = probe_sha
        contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")
        ablation[arm]["benchmark_contract"]["sha256"] = sha256_paths([contract_path])
    ablation_path.write_text(yaml.safe_dump(ablation), encoding="utf-8")
    _refresh_ablation_digest(manifest)
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}

    with pytest.raises(CausalEvidenceError, match="separately shipped trusted K1"):
        freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)


def test_freeze_reexecutes_canonical_benchmarks_instead_of_trusting_refreshed_logs(tmp_path):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    evidence = manifest.parent
    raw_path = evidence / "treatment_raw_log.yaml"
    raw_log = yaml.safe_load(raw_path.read_text(encoding="utf-8"))
    raw_log["samples_ns"] = [1, 1, 1]
    raw_path.write_text(yaml.safe_dump(raw_log), encoding="utf-8")
    generator_sha = sha256_paths([evidence / "ablation_generator.py"])
    observation_path = evidence / "treatment_observation.yaml"
    observation = observe(raw_log, generator_source_sha256=generator_sha)
    observation_path.write_text(yaml.safe_dump(observation), encoding="utf-8")
    result_path = evidence / "treatment_result.yaml"
    result_path.write_text(yaml.safe_dump(summarize(
        observation, generator_source_sha256=generator_sha)), encoding="utf-8")
    run_path = evidence / "treatment_measurement_run.yaml"
    run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
    run["raw_log_sha256"] = sha256_paths([raw_path])
    run["observation_sha256"] = sha256_paths([observation_path])
    run_path.write_text(yaml.safe_dump(run), encoding="utf-8")
    ablation_path = evidence / "ablation.yaml"
    ablation = yaml.safe_load(ablation_path.read_text(encoding="utf-8"))
    for field, path in (("raw_log", raw_path), ("observation", observation_path),
                        ("result", result_path), ("measurement_run", run_path)):
        ablation["treatment"][field]["sha256"] = sha256_paths([path])
    ablation_path.write_text(yaml.safe_dump(ablation), encoding="utf-8")
    _refresh_ablation_digest(manifest)
    inspection_path = evidence / "structural_inspection.yaml"
    inspection = yaml.safe_load(inspection_path.read_text(encoding="utf-8"))
    inspection["ablation_sha256"] = sha256_paths([ablation_path])
    inspection_path.write_text(yaml.safe_dump(inspection), encoding="utf-8")
    result_path = evidence / "structural_result.yaml"
    result_path.write_text(yaml.safe_dump(inspect_structural(
        inspection, root=evidence, generator_source_sha256=generator_sha)), encoding="utf-8")
    structural_path = evidence / "structural.yaml"
    structural = yaml.safe_load(structural_path.read_text(encoding="utf-8"))
    structural["inspection"]["sha256"] = sha256_paths([inspection_path])
    structural["result"]["sha256"] = sha256_paths([result_path])
    structural_path.write_text(yaml.safe_dump(structural), encoding="utf-8")
    manifest_document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    manifest_document["records"][0]["structural"]["sha256"] = sha256_paths([structural_path])
    manifest.write_text(yaml.safe_dump(manifest_document), encoding="utf-8")
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}

    with pytest.raises(CausalEvidenceError, match="benchmark.*replay|canonical benchmark"):
        freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)


def test_refreshed_authored_structural_trace_cannot_create_a_causal_claim(tmp_path):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    evidence = manifest.parent
    inspection_path = evidence / "structural_inspection.yaml"
    inspection = yaml.safe_load(inspection_path.read_text(encoding="utf-8"))
    for arm, value in (("control", 1_000_000), ("treatment", 0)):
        trace_path = evidence / f"{arm}_trace.json"
        trace_path.write_text(json.dumps({
            "schema_version": 1, "kind": "merlin_structural_trace_v1",
            "artifact_sha256": inspection[f"{arm}_artifact_sha256"],
            "events": [{"kind": "materialize", "value": value}],
        }, sort_keys=True), encoding="utf-8")
        inspection[f"{arm}_trace"] = {
            "path": trace_path.name, "sha256": sha256_paths([trace_path])}
    inspection_path.write_text(yaml.safe_dump(inspection), encoding="utf-8")
    structural_path = evidence / "structural.yaml"
    structural = yaml.safe_load(structural_path.read_text(encoding="utf-8"))
    structural["inspection"]["sha256"] = sha256_paths([inspection_path])
    structural_path.write_text(yaml.safe_dump(structural), encoding="utf-8")
    manifest_document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    manifest_document["records"][0]["structural"]["sha256"] = sha256_paths([structural_path])
    manifest.write_text(yaml.safe_dump(manifest_document), encoding="utf-8")
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}

    with pytest.raises(CausalEvidenceError, match="structural.*artifact|authored trace|inspection"):
        freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)


def test_composite_true_binary_forgery_fails_canonical_production_replay(tmp_path):
    """Refreshing every authored digest cannot turn ``/bin/true`` into a benchmark/probe."""
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    evidence = manifest.parent
    generator = evidence / "ablation_generator.py"
    generator_sha = sha256_paths([generator])
    ablation_path = evidence / "ablation.yaml"
    ablation = yaml.safe_load(ablation_path.read_text(encoding="utf-8"))
    probe = json.dumps({
        "schema_version": 1, "kind": "merlin_board_probe_v1",
        "identity": "forged-board", "vlen_bits": 256, "vlen_source": "csr",
        "governor": "performance", "current_khz": 1_600_000,
        "max_khz": 1_600_000, "max_thermal_millic": 1,
    }, sort_keys=True) + "\n"
    for arm, samples in (("control", [1_000_000] * 3), ("treatment", [1] * 3)):
        artifact = evidence / f"{arm}.bin"
        artifact.write_bytes(Path("/bin/true").read_bytes())
        artifact.chmod(0o755)
        artifact_sha = sha256_paths([artifact])
        build_path = evidence / f"{arm}_build_receipt.yaml"
        build = yaml.safe_load(build_path.read_text(encoding="utf-8"))
        build["executable_sha256"] = artifact_sha
        build_path.write_text(yaml.safe_dump(build), encoding="utf-8")
        build_sha = sha256_paths([build_path])
        contract_path = evidence / f"{arm}_benchmark_contract.yaml"
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        contract["artifact_sha256"] = artifact_sha
        contract["executable"]["sha256"] = artifact_sha
        contract["build_receipt"]["sha256"] = build_sha
        contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")
        contract_sha = hashlib.sha256(json.dumps(
            contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        execution_argv = [str(artifact.resolve()), "--paper-session"]
        probe_argv = ["merlin-trusted-k1-board-probe", "--unit-test-json"]
        raw_path = evidence / f"{arm}_raw_log.yaml"
        raw_log = yaml.safe_load(raw_path.read_text(encoding="utf-8"))
        raw_log.update(
            artifact_sha256=artifact_sha, executable_sha256=artifact_sha,
            build_receipt_sha256=build_sha, benchmark_contract_sha256=contract_sha,
            execution_argv=execution_argv,
            command_sha256=hashlib.sha256(json.dumps(
                execution_argv, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            board_probe_argv=probe_argv,
            board_probe_command_sha256=hashlib.sha256(json.dumps(
                probe_argv, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            samples_ns=samples,
            board_receipts={"before": probe, "after": probe},
        )
        raw_path.write_text(yaml.safe_dump(raw_log), encoding="utf-8")
        raw_sha = sha256_paths([raw_path])
        observation_path = evidence / f"{arm}_observation.yaml"
        observation = observe(raw_log, generator_source_sha256=generator_sha)
        observation_path.write_text(yaml.safe_dump(observation), encoding="utf-8")
        observation_sha = sha256_paths([observation_path])
        result_path = evidence / f"{arm}_result.yaml"
        result_path.write_text(yaml.safe_dump(summarize(
            observation, generator_source_sha256=generator_sha)), encoding="utf-8")
        run_path = evidence / f"{arm}_measurement_run.yaml"
        run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
        run.update(
            artifact_sha256=artifact_sha, executable_sha256=artifact_sha,
            build_receipt_sha256=build_sha, benchmark_contract_sha256=contract_sha,
            command_sha256=raw_log["command_sha256"],
            board_receipts_sha256=observation["board_receipts_sha256"],
            raw_log_sha256=raw_sha, observation_sha256=observation_sha,
        )
        run_path.write_text(yaml.safe_dump(run), encoding="utf-8")
        arm_row = ablation[arm]
        arm_row["artifact"]["sha256"] = artifact_sha
        arm_row["artifact"]["executable"]["sha256"] = artifact_sha
        arm_row["artifact"]["build_receipt"]["sha256"] = build_sha
        for field, path in (
                ("benchmark_contract", contract_path),
                ("raw_log", raw_path), ("measurement_run", run_path),
                ("observation", observation_path), ("result", result_path)):
            arm_row[field]["sha256"] = sha256_paths([path])
    ablation_path.write_text(yaml.safe_dump(ablation), encoding="utf-8")
    _refresh_ablation_digest(manifest)
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}

    with pytest.raises(CausalEvidenceError, match="canonical benchmark replay"):
        freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)


@pytest.mark.parametrize(("target", "field"), [
    ("manifest", "agentic"),
    ("record", "narrative"),
    ("artifact", "agentic"),
    ("contract", "agentic"),
])
def test_every_causal_evidence_wrapper_rejects_unknown_fields(tmp_path, target, field):
    raw = _raw()
    manifest = _write_evidence(tmp_path, raw)
    if target in {"manifest", "record"}:
        document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        (document if target == "manifest" else document["records"][0])[field] = True
        manifest.write_text(yaml.safe_dump(document), encoding="utf-8")
    else:
        ablation_path = manifest.parent / "ablation.yaml"
        ablation = yaml.safe_load(ablation_path.read_text(encoding="utf-8"))
        if target == "artifact":
            ablation["control"]["artifact"][field] = True
        else:
            contract_path = manifest.parent / ablation["control"]["benchmark_contract"]["path"]
            contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
            contract[field] = True
            contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")
            ablation["control"]["benchmark_contract"]["sha256"] = sha256_paths([contract_path])
        ablation_path.write_text(yaml.safe_dump(ablation), encoding="utf-8")
        _refresh_ablation_digest(manifest)
    raw["reporting"]["causal_attribution"] = {"path": str(manifest)}

    with pytest.raises(CausalEvidenceError, match="closed|unrecognized"):
        freeze_causal_evidence(raw, root=tmp_path, hasher=sha256_paths)
