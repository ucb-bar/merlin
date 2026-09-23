"""Shared file-only fixtures for installed feedback and formal handoff tests."""

import json
import os
import shutil

from merlin.common.paths import data_path, module_source_path, python_import_roots


def inputs(tmp_path, *, host_program="", contract_root=None, schemas_root=None):
    """Create explicit synthetic package/corpus resources without running tools."""
    ws = tmp_path / "workspace"
    submission = ws / "submission"
    submission.mkdir(parents=True)
    manifest = {
        "artifact_type": "mlir_oot_target_backend",
        "target": "fixture",
        "language": "python",
        "authoring": {"mode": "hand_curated"},
        "integrity_exempt": False,
        "entrypoints": {"tool": "compiler.py"},
        "commands": {
            name: {"argv": ["{tool}", "{input_mlir}"]}
            for name in ("parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm")
        },
    }
    (submission / "manifest.yaml").write_text(json.dumps(manifest))
    (submission / "compiler.py").write_text("print('synthetic compiler')\n")
    shutil.copyfile(module_source_path("merlin_experiments.phase1.tools.simjob"), ws / "simjob.py")
    corpus = tmp_path / "public"
    cap = corpus / "A"
    cap.mkdir(parents=True)
    (cap / "capsule.yaml").write_text(
        json.dumps(
            {
                "name": "A",
                "kind": "isa",
                "source_role": "handauthored_compiler_test",
                "label": "public",
                "operation": {"op": "movement"},
                "numeric_policy": {"compare": "exact_int", "dtype": "i8"},
                "expected": {"instruction_classes": []},
                "required_oracle_tiers": ["L2"],
            }
        )
    )
    descriptor = tmp_path / "target.yaml"
    descriptor.write_text(json.dumps({"target": "fixture", "capsule_corpus": str(corpus)}))
    # Gate policy is a separate explicit work-root resource, not part of --contract schemas.
    policies = tmp_path / "merlin/contract"
    policies.mkdir(parents=True)
    contract_root = contract_root if contract_root is not None else data_path("contract")
    shutil.copyfile(contract_root / "gate_phases.yaml", policies / "gate_phases.yaml")
    shutil.copytree(schemas_root if schemas_root is not None else data_path("schemas"), tmp_path / "merlin/schemas")
    host = tmp_path / "synthetic_host.py"
    host.write_text(host_program)
    private = tmp_path / "private"
    private.mkdir()
    observer = private / "grade.jsonl"
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join(map(str, python_import_roots())),
        MERLIN_OUT_ROOT=str(tmp_path / "generated"),
        TMPDIR=str(private),
        GRADE_OBSERVER=str(observer),
    )
    for name in ("MERLIN_REQUIRED_RTL_ENGINE", "MERLIN_TARGET_EXPERIMENT", "MERLIN_REPO_ROOT"):
        env.pop(name, None)
    return ws, corpus, descriptor, host, env, observer
