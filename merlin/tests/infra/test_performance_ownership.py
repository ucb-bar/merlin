"""Performance primitives and optional experiment/scoring implementations have one owner."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from merlin.common.paths import repo_root


def test_audit_vocabulary_reexports_preserve_identity_and_fail_closed():
    import importlib

    from merlin.common import access

    answer_surfaces = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")

    for name in ("AUDIT_ADVISORY_KINDS", "AUDIT_VIOLATION_KINDS", "audit_hit_is_violation"):
        assert getattr(answer_surfaces, name) is getattr(access, name)
    for kind in access.AUDIT_ADVISORY_KINDS:
        assert not access.audit_hit_is_violation({"kind": kind})
    for hit in (None, [], "path_read", {}, {"kind": "future_kind"}, {"kind": "path_read"}, {"kind": "oracle_use"}):
        assert access.audit_hit_is_violation(hit)


@pytest.mark.parametrize(
    ("owner", "module"),
    [
        ("merlin-analysis", "merlin.perf.recovery"),
        ("merlin-experiments", "merlin.perf.source_program_pair_provider"),
    ],
)
def test_performance_extensions_do_not_replace_core_namespace(owner, module, tmp_path):
    script = """
import importlib.util
import pathlib
import sys
core, owner, module = sys.argv[1:]
sys.path.insert(0, core)
import merlin.perf
assert importlib.util.find_spec(module) is None
sys.path.insert(0, owner)
# Refresh the already-imported namespace just as a fresh interpreter would.
import importlib
import merlin
importlib.reload(merlin)
importlib.reload(merlin.perf)
assert pathlib.Path(merlin.perf.__file__).is_relative_to(core)
assert pathlib.Path(importlib.util.find_spec(module).origin).is_relative_to(owner)
assert pathlib.Path(importlib.util.find_spec('merlin.perf.envelope').origin).is_relative_to(core)
assert not any(name.startswith(('merlin_experiments', 'aet', 'torch')) for name in sys.modules)
"""
    root = repo_root()
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(root / "src"), str(root / "packages" / owner / "src"), module],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_recovery_admission_works_without_any_experiment_engine(tmp_path):
    script = """
import importlib.abc
import pathlib
import sys
core, owner, dependencies = sys.argv[1:]
sys.path[:0] = [core, owner, dependencies]
class NoExperiments(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('merlin.targetgen.sandbox', 'merlin_experiments', 'aet')):
            raise AssertionError('analysis imported experiment engine: ' + fullname)
sys.meta_path.insert(0, NoExperiments())
from merlin.perf import recovery
assert pathlib.Path(recovery.__file__).is_relative_to(owner)
assert recovery.admit_candidate(None)['status'] == 'refused_unaudited'
assert recovery.admit_candidate({'hits': []})['admitted']
assert recovery.admit_candidate({'hits': [{'kind': 'blocked_probe'}]})['admitted']
assert recovery.admit_candidate({'hits': [{'kind': 'future_kind'}]})['status'] == 'refused_answer_access'
"""
    root = repo_root()
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            script,
            str(root / "src"),
            str(root / "packages/merlin-analysis/src"),
            str(Path(yaml.__file__).parent.parent),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_recovery_command_has_exactly_one_distribution_owner():
    root = repo_root()
    manifests = [root / "pyproject.toml", *sorted((root / "packages").glob("*/pyproject.toml"))]
    owners = []
    for path in manifests:
        project = tomllib.loads(path.read_text())["project"]
        if "merlin-recovery" in project.get("scripts", {}):
            owners.append((project["name"], project["scripts"]["merlin-recovery"]))
    assert owners == [("merlin-analysis", "merlin.perf.recovery:main")]


def test_bespoke_mesh_can_refuse_unavailable_oracle_without_loading_a_grader(tmp_path):
    script = """
import importlib.abc
import sys
from types import SimpleNamespace
core, dependencies = sys.argv[1:]
sys.path[:0] = [core, dependencies]
class NoGrader(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('merlin.targetgen.capsule_runner', 'merlin_experiments', 'aet')):
            raise AssertionError('mesh imported experiment engine: ' + fullname)
sys.meta_path.insert(0, NoGrader())
from merlin.compile import mesh
from merlin.targetgen import oracle_policy
calls = []
oracle_policy.selected_sim_via = lambda target: 'fixture_engine'
oracle_policy._SIM_ORACLES['fixture_engine'] = SimpleNamespace(
    exclusive=True, available=lambda target: (calls.append(target) or False, 'unavailable'))
assert mesh._matmul_via_bespoke_sim('fixture', '', [[1]], [[2]], package='fixture', timeout=1) is None
assert calls == ['fixture']
assert 'merlin.targetgen.capsule_runner' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(repo_root() / "src"), str(Path(yaml.__file__).parent.parent)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
