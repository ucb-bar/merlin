"""Generated performance input policy without native controllers or simulator execution."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.corpus.phase_selection import generate_phase_selections
from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2 import corpus as P

from merlin.common.paths import python_import_roots


@pytest.fixture
def generated(tmp_path):
    root = tmp_path / "live"
    public = root / "public"
    public.mkdir(parents=True)
    names = ("a", "b")
    for name in names:
        source = root / "_tuning" / name
        source.mkdir(parents=True)
        descriptor = {
            "name": name,
            "label": "dev",
            "source_role": "derived_sweep",
            "performance": {"family": name.upper(), "claim": "RECOVERS"},
        }
        (source / "capsule.yaml").write_text(yaml.safe_dump(descriptor))
        (source / "input.bin").write_bytes(name.encode())
    manifest = {
        "generated": [f"_tuning/{name}" for name in names],
        "hand_authored": [],
        "performance_generation": {
            "fixture": {
                "errors": [],
                "phase": {"category": "_tuning", "label": "dev", "included_in_functional_grade": False},
            }
        },
    }
    (root / "MANIFEST.yaml").write_text(yaml.safe_dump(manifest))
    return SimpleNamespace(target="fixture", capsule_corpus=public, graded_roots=lambda: [public])


def _load(frozen):
    return P.load_frozen_performance_corpus(
        frozen.root,
        manifest_sha256=frozen.manifest_sha256,
        capsules_sha256=frozen.capsules_sha256,
        expected_target="fixture",
    )


def test_full_lifecycle_preserves_identity_schedule_and_exact_manifest(generated, tmp_path):
    discovered = P.discover_performance_corpus(generated)
    frozen = P.freeze_performance_corpus(discovered, tmp_path / "frozen")
    loaded = _load(frozen)
    P.verify_frozen_performance_corpus(loaded)
    assert loaded == frozen
    assert [member.capsule for member in loaded.capsules] == ["a", "b"]
    assert [(c.family, c.capsule, c.simulator, c.replicate) for c in P.expected_perf_cells(loaded.capsules, 2)] == [
        (name.upper(), name, simulator, f"r{replica:03d}")
        for name in ("a", "b")
        for replica in range(2)
        for simulator in ("spike", "gsim")
    ]
    assert all(not path.stat().st_mode & 0o222 for path in [frozen.root, *frozen.root.rglob("*")])
    assert frozen.capsules[0].source_dir.stat().st_ino != discovered.capsules[0].source_dir.stat().st_ino
    expected = {
        "schema_version": 1,
        "target": "fixture",
        "source": {
            "provenance_manifest": str(discovered.provenance_manifest),
            "provenance_sha256": discovered.provenance_sha256,
            "performance_generation_sha256": C.document_sha256(discovered.performance_generation),
        },
        "capsules_sha256": frozen.capsules_sha256,
        "capsules": [
            {
                "family": member.family,
                "capsule": member.capsule,
                "source_relative_path": member.source_relative_path,
                "snapshot_relative_path": f"capsules/{member.source_relative_path}",
                "snapshot_sha256": member.source_sha256,
                "n_files": member.n_files,
                "n_bytes": member.n_bytes,
                "performance": member.descriptor["performance"],
                "performance_sha256": C.document_sha256(member.descriptor["performance"]),
            }
            for member in discovered.capsules
        ],
    }
    assert frozen.manifest_path.read_bytes() == C.canonical_json(expected)


def test_selection_is_intersection_after_complete_phase_admission(generated):
    assert [m.capsule for m in P.discover_performance_corpus(generated, families="B", capsules="b").capsules] == ["b"]
    with pytest.raises(C.StageGateError, match="zero capsules"):
        P.discover_performance_corpus(generated, families="A", capsules="b")
    # Even an unselected invalid member makes the declared phase inadmissible.
    descriptor = generated.capsule_corpus.parent / "_tuning/a/capsule.yaml"
    row = yaml.safe_load(descriptor.read_text())
    row["source_role"] = "hand_authored"
    descriptor.write_text(yaml.safe_dump(row))
    with pytest.raises(C.StageGateError, match="not a generated dev member"):
        P.discover_performance_corpus(generated, capsules="b")


def test_phase0_records_distinct_functional_and_performance_selections(generated):
    manifest_path = generated.capsule_corpus.parent / "MANIFEST.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["generated"].append("public/functional")
    manifest["phase_corpora"] = {
        "fixture": generate_phase_selections(
            ["public/functional", "_tuning/a", "_tuning/b"], performance_category="_tuning"
        )
    }
    manifest_path.write_text(yaml.safe_dump(manifest))
    assert [member.capsule for member in P.discover_performance_corpus(generated).capsules] == ["a", "b"]
    roles = manifest["phase_corpora"]["fixture"]
    assert roles["phase1"]["generated_members"] == ["public/functional"]
    assert roles["phase2"]["generated_members"] == ["_tuning/a", "_tuning/b"]
    roles["phase2"]["generated_members"] = ["public/functional"]
    manifest_path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(C.StageGateError, match="phase selections are invalid"):
        P.discover_performance_corpus(generated)


@pytest.mark.parametrize("selection", ["missing", "a,a", "../a", ","])
def test_invalid_member_selection_refuses(generated, selection):
    with pytest.raises(C.StageGateError):
        P.discover_performance_corpus(generated, capsules=selection)


@pytest.mark.parametrize("mutation", ["manifest", "member", "symlink", "special", "source_after_copy"])
def test_freeze_refuses_changed_or_unsafe_input(generated, tmp_path, monkeypatch, mutation):
    discovered = P.discover_performance_corpus(generated)
    source = discovered.capsules[0].source_dir
    if mutation == "manifest":
        discovered.provenance_manifest.write_text("changed: true\n")
    elif mutation == "member":
        (source / "input.bin").write_bytes(b"changed")
    elif mutation == "symlink":
        (source / "alias").symlink_to(source / "input.bin")
    elif mutation == "special":
        import os

        os.mkfifo(source / "fifo", 0o444)
    else:
        copytree = P.shutil.copytree

        def mutate_after_copy(*args, **kwargs):
            result = copytree(*args, **kwargs)
            (source / "input.bin").write_bytes(b"changed")
            return result

        monkeypatch.setattr(P.shutil, "copytree", mutate_after_copy)
    with pytest.raises(C.StageGateError, match="changed|symlink|special file"):
        P.freeze_performance_corpus(discovered, tmp_path / "frozen")


def test_sealed_member_drift_and_wrong_target_refuse(generated, tmp_path):
    frozen = P.freeze_performance_corpus(P.discover_performance_corpus(generated), tmp_path / "frozen")
    with pytest.raises(C.StageGateError, match="identity changed"):
        P.load_frozen_performance_corpus(
            frozen.root,
            manifest_sha256=frozen.manifest_sha256,
            capsules_sha256=frozen.capsules_sha256,
            expected_target="other",
        )
    member = frozen.capsules[0].source_dir / "input.bin"
    member.chmod(0o644)
    member.write_bytes(b"changed")
    with pytest.raises(C.StageGateError, match="capsule bytes changed"):
        _load(frozen)


def test_exact_tree_hash_keeps_ephemeral_names_and_existing_byte_domain(tmp_path):
    (tmp_path / "build").mkdir()
    (tmp_path / "build/value").write_bytes(b"payload")
    assert C.exact_tree_record(tmp_path) == {
        "sha256": hashlib.sha256(b"build/value\0payload\0").hexdigest(),
        "n_files": 1,
        "n_bytes": 7,
    }
    assert C.document_sha256({"x": 1}) == hashlib.sha256(b'{"x":1}').hexdigest()


def test_outside_checkout_cold_import_and_real_frozen_load(generated, tmp_path):
    frozen = P.freeze_performance_corpus(P.discover_performance_corpus(generated), tmp_path / "frozen")
    program = """
import importlib.abc,json,sys
from pathlib import Path
sys.path[:0]=json.loads(sys.argv[1])
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,name,*args):
        if name in {'perf_agent_stage','perf_campaign','perf_prompt','_pbcommon','_common'}:
            raise AssertionError('native dependency: '+name)
sys.meta_path.insert(0,NoNative())
from merlin_experiments.phase2 import corpus
loaded=corpus.load_frozen_performance_corpus(Path(sys.argv[2]),manifest_sha256=sys.argv[3],capsules_sha256=sys.argv[4],expected_target='fixture')
corpus.verify_frozen_performance_corpus(loaded)
assert len(corpus.expected_perf_cells(loaded.capsules,2)) == 8
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            program,
            json.dumps([str(p) for p in python_import_roots()]),
            str(frozen.root),
            frozen.manifest_sha256,
            frozen.capsules_sha256,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
