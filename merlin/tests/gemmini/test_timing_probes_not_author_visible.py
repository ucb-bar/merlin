"""Operator timing probes stay operator-private: never in an author grant, a release payload or a grade.

The probe data lives in the target's verification directory; the calibration stage reads it with an
explicit operator path. This checks the boundary from both sides: no root the target descriptor
grants or stages contains it, and no authoring, corpus or grading code refers to it.
"""

from __future__ import annotations

import os
import stat

import yaml

from merlin.common.paths import repo_root

ROOT = repo_root()
PROBES = ROOT / "examples/gemmini/verification/timing_probes"
DESCRIPTOR = ROOT / "examples/gemmini/target/descriptor.yaml"
#: Packages whose code shapes what an author sees, what a release stages or how a submission is graded.
AUTHOR_FACING = [
    "packages/merlin-experiments/src/merlin_experiments/phase1",
    "packages/merlin-experiments/src/merlin_experiments/phase0",
    "packages/merlin-experiments/src/merlin_experiments/corpus",
    "packages/merlin-experiments/src/merlin/targetgen",
    "src/merlin/targetgen",
    "examples/gemmini/phase1",
    "examples/gemmini/support",
]


def _granted_roots() -> list:
    descriptor = yaml.safe_load(DESCRIPTOR.read_text())
    roots = [descriptor[key] for key in ("resources_root", "task_root", "contracts_root", "capsule_corpus")]
    roots.append(str((ROOT / descriptor["numeric_profile"]).parent.relative_to(ROOT)))
    return [(ROOT / root).resolve() for root in roots]


def test_probe_data_is_outside_every_root_the_descriptor_grants_or_stages():
    probes = PROBES.resolve()
    assert probes.is_dir()
    for root in _granted_roots():
        assert root != probes and root not in probes.parents and probes not in root.parents, root


def test_no_authoring_corpus_or_grading_code_refers_to_the_probes():
    hits = []
    for relative in AUTHOR_FACING:
        for dirpath, _, names in os.walk(ROOT / relative):
            for name in names:
                if name.endswith((".py", ".md", ".yaml", ".yml", ".json", ".txt")):
                    path = os.path.join(dirpath, name)
                    with open(path, encoding="utf-8", errors="replace") as stream:
                        if "timing_probe" in stream.read():
                            hits.append(os.path.relpath(path, ROOT))
    assert hits == []


def test_probe_outputs_are_private(tmp_path):
    from merlin_experiments.timing_probes.runner import private_dir

    made = private_dir(tmp_path / "calibration" / "cache")
    assert stat.S_IMODE(made.stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "calibration").stat().st_mode) == 0o700
