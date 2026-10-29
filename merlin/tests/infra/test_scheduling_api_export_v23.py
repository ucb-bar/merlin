"""The prospective SDK exports pointwise views and actual parallel exploration."""

import os
import subprocess
import sys

from merlin.common.access import author_inaccessible
from merlin.common.paths import repo_root
from merlin.compile.scheduling.export import public_modules
from merlin.compile.scheduling.library_sources import library_modules
from merlin.targetgen import tool_registry as TR
from merlin.targetgen.tool_profile_shared import (
    prepare_shared_source_release,
    reviewed_shared_manifest,
    staged_module,
)


def test_v23_closed_release_runs_public_geometry_and_exploration(tmp_path):
    leaf = "merlin.compile.scheduling.pointwise_convolution"
    public_additions = {
        leaf,
        "merlin.compile.scheduling.batched_dense",
        "merlin.compile.scheduling.convolution_output_groups",
        "merlin.compile.scheduling.loaded_sum_epochs",
        "merlin.compile.tuning_knobs",
        "merlin.transforms.tuning_selection",
    }
    assert set(library_modules(23)) - set(library_modules(22)) == public_additions | {
        "merlin.compile.scheduling.affine_gather_runs"
    }
    assert set(public_modules(23)) - set(public_modules(22)) == public_additions
    assert leaf not in library_modules(22)
    import yaml

    from merlin.targetgen.tool_profile import _SHARED_SUBSTRATES

    spec = yaml.safe_load((repo_root() / "examples/gemmini/phase1/launch/fresh-campaign-next.yaml").read_text())
    assert _SHARED_SUBSTRATES[spec["inputs"]["common_substrate"]] == spec["inputs"]["library_version"]
    root = tmp_path / "sdk"
    manifest = prepare_shared_source_release(
        root,
        python_source_root=repo_root() / "src",
        license_directory=repo_root(),
        tools=TR.ARM_TOOLS["merlin_rtlchecks"],
        version=23,
        source_binding=True,
    )
    assert manifest["schema"] == "merlin.shared_scheduling_library.v17"
    assert reviewed_shared_manifest(root, version=23) == manifest
    assert not [
        p for p in (root / "merlin").rglob("*.py") if author_inaccessible(staged_module(p.relative_to(root).as_posix()))
    ]
    code = """
import pathlib, sys, importlib
from merlin.compile.scheduling.api_catalog import scheduling_capabilities
required = {'checked_cross_spatial_array_reuse', 'source_exact_convolution_readout_transfers',
            'bounded_declared_settings_cycle_selection', 'checked_disjoint_loaded_sum_epochs'}
rows = [c for c in scheduling_capabilities(23)['capabilities'] if c['name'] in required]
assert {c['name'] for c in rows} == required
for row in rows:
 for symbol in row['symbols']:
  module, name = symbol.rsplit('.', 1)
  assert callable(getattr(importlib.import_module(module), name))
from merlin.compile.scheduling.convolution_output_groups import plan_rolling_convolution_output_group
from merlin.compile.scheduling.window_transfer import ConvolutionReadoutTransferLaw, plan_convolution_readout_transfers
from merlin.compile.scheduling.dense_contracts import DenseTiledIntegerReadout
from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
from merlin.compile.scheduling.dense_panel_pipeline import DensePanelPipelineResult
assert callable(plan_rolling_convolution_output_group) and callable(plan_convolution_readout_transfers)
assert callable(schedule_dense_pipeline) and callable(DensePanelPipelineResult.emit_index_program)
readout = DenseTiledIntegerReadout.from_source_scales(0.125, columns=5, tile_columns=4)
assert readout.table_words == (0x3e000000, 0x3e000000)
from merlin.compile.scheduling.pointwise_convolution import pointwise_dense_view
from merlin.compile.scheduling.window_transfer import Window2D
from merlin.transforms.evaluated_exploration import evaluate_emitted, EvaluationRequest
from merlin.transforms.shape_policy import ShapeCaseFeedback
view = pointwise_dense_view(Window2D((2, 3, 5, 7), (1, 1)), 11, output_shape=(2, 3, 5, 11))
assert view.dimensions == (30, 7, 11)
assert view.record()['payload_copies'] == 0
assert list(evaluate_emitted([], lambda *args: None)) == []
from hashlib import sha256
from merlin.compile.tuning_knobs import Knob, KnobSettings, bounded_settings, declare
from merlin.transforms.tuning_selection import KnobTrial, unique_emissions, select_settings
from merlin.transforms.pass_exploration import EmittedProgram, MappingProposalKey
from merlin.transforms.evaluated_exploration import CompletedObservation, ExplorationIdentity
from merlin.compile.scheduling.loaded_sum_epochs import (
 LoadedSumEpochLaw, LoadedSumEpochPlan, plan_loaded_sum_epochs, emit_loaded_sum_epochs)
assert all(callable(symbol) for symbol in (LoadedSumEpochLaw, LoadedSumEpochPlan,
                                           plan_loaded_sum_epochs, emit_loaded_sum_epochs))
tree = pathlib.Path('compiler-settings')
declare(tree, {'traversal': Knob(('rows', 'columns'), 'rows')})
choices = bounded_settings(tree, knob_names=('traversal',), max_candidates=2)
assert len(choices) == 2 and choices[0].baseline and isinstance(choices[0], KnobSettings)
identity = ExplorationIdentity(*('1' * 64 for _ in range(10)), 'binary', 'modeled_roi')
domain = MappingProposalKey('1' * 64, ((4, 16), (16, 8)), ('row', 'column'),
                           ('i8', 'i8'), ('raw',), '1' * 64, '1' * 64, '1' * 64)
trials = []
for choice, data, cycles in zip(choices, (b'control', b'alternative'), (100, 80), strict=True):
 program = EmittedProgram(data, 'binary', choice.identity)
 observed = CompletedObservation(identity, sha256(data).hexdigest(), '2' * 64,
                                 'pass', 'complete', True, cycles)
 trials.append(KnobTrial(choice, program, observed))
trials = tuple(trials)
assert len(unique_emissions(trials, identity=identity)) == 2
selection = select_settings(domain, identity=identity, trials=trials)
assert selection.selected is choices[1] and selection.status == 'cycle_improvement'
assert not selection.record()['performance_qualification']
"""
    code += (
        "\nfor name, module in tuple(sys.modules.items()):\n"
        " if name == 'merlin' or name.startswith('merlin.'):\n"
        "  path = getattr(module, '__file__', None)\n"
        "  if path: assert pathlib.Path(path).is_relative_to(" + repr(str(root)) + "), (name, path)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
