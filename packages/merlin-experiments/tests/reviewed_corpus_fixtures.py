"""Real integer derivation through installed-shaped commands and reviewed preparation.

No generator/output stubs: only synthetic hardware binding/performance facts are
injected. The parent release gate separately verifies built distribution installs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from merlin.common.paths import data_path, module_source_path, python_import_roots, python_source_dir


def _member(root: Path, category: str, name: str, label: str) -> None:
    directory = root / category / name
    directory.mkdir(parents=True)
    document = {
        "name": name,
        "kind": "isa",
        "source_role": "handauthored_compiler_test",
        "label": label,
        "operation": {"op": "matmul", "attributes": {}},
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
        "expected": {"instruction_classes": [], "modes": {}},
        "required_oracle_tiers": ["L0"],
        "interface_mlir": "capsule.interface.mlir",
    }
    (directory / "capsule.yaml").write_text(yaml.safe_dump(document))
    (directory / "capsule.interface.mlir").write_text("module {}\n")
    (directory / "golden.yaml").write_text("outputs: {}\n")


def build_phase0_handoff(tmp_path, *, copy_sources=True):
    installed = tmp_path / "installed"
    extension = module_source_path("merlin_experiments").parent
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    if copy_sources:
        shutil.copytree(python_source_dir() / "merlin", installed / "merlin", ignore=ignore)
        shutil.copytree(extension.parent / "merlin", installed / "merlin", dirs_exist_ok=True, ignore=ignore)
        shutil.copytree(extension, installed / "merlin_experiments", ignore=ignore)
    workspace = tmp_path / "external-workspace"
    workspace.mkdir()
    support = workspace / "explicit-target-resources"
    for resource in ("rtl_facts", "irdl"):
        directory = support / "contracts" / resource
        directory.mkdir(parents=True)
        (directory / "FIXTURE_ONLY.txt").write_text("Synthetic resource selection; no hardware facts are qualified.\n")
    baseline = workspace / "baseline"
    _member(baseline, "isa", "generated_member", "public")
    _member(baseline, "layers", "retained_member", "public")
    _member(baseline, "hidden", "private_member_identity", "hidden")
    (baseline / "MANIFEST.yaml").write_text(
        yaml.safe_dump(
            {
                "generated": ["isa/generated_member"],
                "hand_authored": ["layers/retained_member"],
                "held_out": {"n_generated": 0, "n_hand_authored": 1},
            }
        )
    )
    experiment = workspace / "source-experiment"
    (experiment / "task").mkdir(parents=True)
    for name in ("TASK_full.md", "TASK_realistic.md"):
        (experiment / "task" / name).write_text("Synthetic external compiler task\n")
    descriptor = experiment / "target_experiment.yaml"
    descriptor.write_text(
        yaml.safe_dump(
            {
                "target": "fixture-device",
                "backend_package_dir": str(support),
                "capsule_corpus": str(baseline / "isa"),
                "grading": {
                    "expected_cohort": {"source_capsules": 2, "admitted_capsules": 2},
                    "hidden_capability_admission": {"source_capsules": 1, "admitted_capsules": 1},
                },
            }
        )
    )
    profiles = tmp_path / "external-profiles"
    profiles.mkdir()
    (profiles / "_perf.yaml").write_text("sweeps: []\n")
    (profiles / "fixture-device.yaml").write_text(
        yaml.safe_dump(
            {
                "datapath": {"required_oracle_tiers": ["L0"]},
                "capsules": [
                    {
                        "name": "generated_member",
                        "kind": "isa",
                        "cat": "isa",
                        "op": "matmul",
                        "label": "public",
                        "source_role": "handauthored_compiler_test",
                        "source_reference": "synthetic test-only integer matmul",
                        "M": 2,
                        "K": 2,
                        "N": 2,
                        "lhs": "A",
                        "weight": "W",
                        "out": "Y",
                    }
                ],
            }
        )
    )
    hooks = tmp_path / "target-facts-only"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text(
        "from merlin.targetgen import corpus_spec as CS\n"
        "from merlin_experiments.phase0 import generation\n"
        "CS.derive_binding = lambda te, dp: CS.CorpusBinding(te.target, 2, 'int8', 'i32', True, ['L0'], 'exact_int')\n"
        "generation._performance_facts = lambda target: {'sha256': '0' * 64}\n"
    )
    definition = workspace / "definition.yaml"
    definition.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "installed-derivation",
                "target": "fixture-device",
                "phases": {
                    "0": {
                        "adapter": "capsule_derivation",
                        "config": {
                            "descriptor": str(descriptor),
                            "profiles_root": str(profiles),
                        },
                    }
                },
            }
        )
    )
    env = dict(
        os.environ,
        MERLIN_REPO_ROOT=str(workspace),
        MERLIN_OUT_ROOT=str(workspace / "out"),
        MERLIN_CONTRACT_DIR=str(data_path("contract").resolve()),
        MERLIN_SCHEMAS_DIR=str(data_path("schemas").resolve()),
        MERLIN_BUNDLE_CAS="",
        PYTHONPATH=os.pathsep.join([*(map(str, (installed,) if copy_sources else python_import_roots())), str(hooks)]),
        PYTHONSAFEPATH="1",
    )
    run = workspace / "out/runs/phase0"
    release = workspace / "out/artifacts/protocols/review"

    def cli(*args):
        return subprocess.run(
            [sys.executable, "-P", "-m", "merlin_experiments", *map(str, args)],
            cwd=workspace,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )

    return {
        "installed": installed,
        "workspace": workspace,
        "baseline": baseline,
        "profiles": profiles,
        "definition": definition,
        "run": run,
        "release": release,
        "cli": cli,
        "environment": env,
        "hooks": hooks,
    }
