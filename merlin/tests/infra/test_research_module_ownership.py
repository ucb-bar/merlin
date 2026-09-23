"""Research implementations have one owner without importing optional engines into core."""

from __future__ import annotations

import ast
import json
import subprocess
import sys

import pytest

from merlin.common.paths import repo_root


def test_lane_evidence_interpretation_is_core_only(tmp_path):
    script = """
import sys
sys.path.insert(0, sys.argv[1])
from merlin.targetgen import elf_lanes as lanes
assert lanes.PROGRAM_ARTIFACT_SUFFIX == '.program'
assert lanes.ACCELERATOR_LANE == 'on_mesh'
assert lanes.LINKED_ELF_EVIDENCE not in lanes.EXECUTED_LANE_EVIDENCE
assert lanes.DECLARED_PROGRAM_EVIDENCE not in lanes.EXECUTED_LANE_EVIDENCE
assert lanes.unjudged_lanes({'evidence': {'on_mesh': lanes.LINKED_ELF_EVIDENCE},
    'observed': ['on_mesh']}, {'require': ['on_mesh']}) == ['on_mesh']
assert set(lanes.EXECUTED_LANE_EVIDENCE) < set(lanes.negative_lane_evidence())
assert not any(name.startswith(('aet', 'merlin_experiments',
    'merlin.targetgen.capsule_runner')) for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(repo_root() / "src")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_legacy_lane_vocabulary_has_one_owner():
    from merlin.targetgen import capsule_runner, elf_lanes

    for name in ("PROGRAM_ARTIFACT_SUFFIX", "EXECUTED_LANE_EVIDENCE", "WHOLE_PROGRAM_COMPLETION_EVIDENCE"):
        assert getattr(capsule_runner, name) is getattr(elf_lanes, name)
    assert capsule_runner._ACCELERATOR_LANE is elf_lanes.ACCELERATOR_LANE


@pytest.mark.parametrize(
    ("distribution", "modules"),
    [
        (
            "merlin-analysis",
            (
                "merlin.agentreport.availability",
                "merlin.agentreport.anatomy",
                "merlin.agentreport.capsule_time",
                "merlin.agentreport.corpus_coverage",
                "merlin.agentreport.cost_curve",
                "merlin.agentreport.index",
                "merlin.agentreport.passes",
                "merlin.agentreport.phase1_tools",
                "merlin.agentreport.phase2",
                "merlin.agentreport.phase2_campaign",
                "merlin.agentreport.series",
                "merlin.agentreport.spans",
                "merlin.agentreport.tokens",
            ),
        ),
        (
            "merlin-experiments",
            (
                "merlin.targetgen.aet_bridge",
                "merlin.targetgen.experiment_tokens",
                "merlin.targetgen.group_capsules",
                "merlin.targetgen.store_probe",
                "merlin.perf.analysis_worker",
            ),
        ),
    ],
)
def test_research_imports_require_only_their_owner(distribution, modules, tmp_path):
    root = repo_root()
    owner = root / "packages" / distribution / "src"
    # Disable all site/editable packages. No sibling checkout or extension may
    # make this ownership check pass, and these readers must be import-light.
    script = """
import importlib
import pathlib
import sys
core, owner, *modules = sys.argv[1:]
sys.path[:0] = [core, owner]
for name in modules:
    module = importlib.import_module(name)
    assert pathlib.Path(module.__file__).is_relative_to(owner), module.__file__
assert not any(name.startswith(("aet", "torch", "merlin_experiments")) for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(root / "src"), str(owner), *modules],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_model_slice_recipes_are_not_shipped_by_shared_distributions():
    root = repo_root()
    for owner in (root / "src", root / "packages/merlin-experiments/src"):
        assert not (owner / "merlin/targetgen/model_slice_export.py").exists()


def test_core_does_not_ship_research_readers_or_agent_adapters(tmp_path):
    script = """
import importlib.util
import sys
sys.path.insert(0, sys.argv[1])
for module in (
    "merlin.perf.analysis_worker",
    "merlin.perf.isolated_probe_provider",
    "merlin.perf.controlled_context_provider",
    "merlin.perf.paired_context_provider",
    "merlin.perf.host_region_qualifier",
    "merlin.perf.host_physical_transition_qualifier",
    "merlin.perf.lane_migration_qualifier",
    "merlin.perf.source_contraction_preparation",
    "merlin.perf.source_convolution_preparation",
    "merlin.perf.source_program_pair",
    "merlin.perf.source_initializer_elision",
    "merlin.agentreport",
    "merlin.targetgen.aet_bridge",
    "merlin.targetgen.experiment_tokens",
    "merlin.targetgen.heavy_oracles",
    "merlin.targetgen.model_slice_export",
    "merlin.targetgen.group_capsules",
    "merlin.targetgen.store_probe",
    "merlin.verify.plots",
):
    assert importlib.util.find_spec(module) is None, module
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(repo_root() / "src")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_group_capsule_entries_are_core_only_and_compiler_uses_them(tmp_path):
    script = """
import importlib.util
import pathlib
import sys
sys.path.insert(0, sys.argv[1])
from merlin.targetgen import group_capsule_entries as entries
assert pathlib.Path(entries.__file__).is_relative_to(sys.argv[1])
assert entries.SCHEMA == 'group_capsules_v1'
assert entries.SOURCE_ROLE == 'model_derived'
assert entries._identity({'op': 'matmul', 'name': 'one', 'acc_scale': 2}) == entries._identity(
    {'op': 'matmul', 'name': 'two', 'acc_scale': 3})
assert entries._label({'op': 'matmul', 'M': 2, 'K': 4, 'N': 8}) == 'G_matmul_m2k4n8_raw'
assert importlib.util.find_spec('merlin.targetgen.group_capsules') is None
assert not any(name.startswith(('aet', 'torch', 'merlin_experiments',
    'merlin.targetgen.capsule_runner', 'merlin.targetgen.capsule_golden')) for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(repo_root() / "src")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    tree = ast.parse((repo_root() / "src/merlin/compile_cli.py").read_text())
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "targetgen"
        for alias in node.names
    }
    assert "group_capsule_entries" in names
    assert "group_capsules" not in names


def test_core_grouping_executes_without_any_optional_distribution(tmp_path):
    from im2col_conv_layer import module

    root = repo_root()
    optional_modules = set()
    for owner in (root / "packages").glob("*/src"):
        for path in owner.rglob("*.py"):
            parts = list(path.relative_to(owner).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            optional_modules.add(".".join(parts))
    script = """
import importlib.abc
import json
import pathlib
import subprocess
import sys
payload = json.load(sys.stdin)
optional = set(payload['optional_modules'])
def forbidden(name):
    return name in optional or name.split('.')[0] in {'aet', 'torch', 'torchao'}
assert not any(forbidden(name) for name in sys.modules)
class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert not forbidden(fullname), f'optional import: {fullname}'
sys.meta_path.insert(0, CoreOnly())
sys.path[:0] = sys.argv[1:]
def no_execution(*args, **kwargs):
    raise AssertionError('group derivation must not execute a backend')
subprocess.Popen = no_execution
from merlin.common import mlir_query
from merlin.targetgen import group_capsule_entries
from fake_quant_layer import Oracle
assert pathlib.Path(group_capsule_entries.__file__).is_relative_to(sys.argv[1])
result = group_capsule_entries.entries('synthetic', mlir_query.parse(payload['mlir']),
    weight_args={1, 2}, model='toy', oracle=Oracle())
assert (result['accelerator_groups'], result['stated'], result['distinct']) == (1, 1, 1)
assert len(result['entries']) == 2
fused, raw = result['entries']
assert fused['entry']['op'] == raw['entry']['op'] == 'conv2d'
assert fused['entry']['epilogue'] == ['bias_add', 'acc_scale', 'relu']
assert raw['raw_of'] == fused['name']
assert raw['entry']['epilogue'] == []
assert 'acc_scale' not in raw['entry']
for key in ('ci', 'N', 'Himg', 'Wimg', 'kh', 'kw', 'stride', 'padding'):
    assert raw['entry'][key] == fused['entry'][key], key
assert not any(forbidden(name) for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(root / "src"), str(root / "merlin/tests/fixtures")],
        cwd=tmp_path,
        input=json.dumps({"mlir": module(), "optional_modules": sorted(optional_modules)}),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_group_research_reexports_exact_core_entries_and_retains_grader_identity():
    from merlin.common import access
    from merlin.targetgen import group_capsule_entries, group_capsules, store_probe

    for name in ("SCHEMA", "SOURCE_ROLE", "_IDENTITY_DROPS", "_identity", "_label", "_element_range", "entries"):
        assert getattr(group_capsules, name) is getattr(group_capsule_entries, name)
    for module in (group_capsules, store_probe):
        assert module.__name__ in access.declared_modules("grader")
        item = next(item for item in access.MODULE_ACCESS if module.__name__ in item.modules)
        assert repo_root() / "packages/merlin-experiments/src" / (
            module.__name__.replace(".", "/") + ".py"
        ) in access.module_locations(repo_root(), item)


def test_capacity_probe_keeps_measurement_geometry_without_loading_evaluators(tmp_path):
    script = """
import sys
sys.path[:0] = sys.argv[1:]
from merlin.targetgen import store_probe as probe
assert probe.working_set_elements(2, 4, 8) == 40
assert probe.ladder(2, floor_elements=8, ceiling_elements=80) == [(2, 2, 2), (2, 4, 4), (2, 8, 8)]
empty = probe.DeclineBracket(target='fixture', dtype='i8')
assert probe.capacity_candidates(empty, {'unknown': 64}) == {'unknown': None}
assert not any(name.startswith(('merlin.targetgen.capsule_runner',
    'merlin.targetgen.capsule_golden', 'generate_corpus')) for name in sys.modules)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            script,
            str(repo_root() / "src"),
            str(repo_root() / "packages/merlin-experiments/src"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_analysis_contributes_plots_without_replacing_core_verification(tmp_path):
    script = """
import importlib.util
import pathlib
import sys
core, owner = sys.argv[1:]
sys.path[:0] = [core, owner]
import merlin.verify
assert pathlib.Path(merlin.verify.__file__).is_relative_to(core)
assert pathlib.Path(importlib.util.find_spec("merlin.verify.plots").origin).is_relative_to(owner)
assert pathlib.Path(importlib.util.find_spec("merlin.verify.refine").origin).is_relative_to(core)
assert "matplotlib" not in sys.modules
"""
    root = repo_root()
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(root / "src"), str(root / "packages/merlin-analysis/src")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
