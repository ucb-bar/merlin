"""V24's reviewed closure executes the checked owner without host compiler imports."""

import os
import runpy
import subprocess
import sys
from pathlib import Path

import z3

from merlin.common.access import author_inaccessible
from merlin.common.paths import repo_root
from merlin.compile.scheduling.export import public_modules
from merlin.compile.scheduling.library_sources import library_modules
from merlin.targetgen import tool_registry as TR
from merlin.targetgen.tool_profile import _SHARED_SUBSTRATES, COMMON_SUBSTRATE_V17
from merlin.targetgen.tool_profile_shared import (
    prepare_shared_source_release,
    reviewed_shared_manifest,
    staged_module,
)


def test_v24_exports_and_executes_actual_shared_rhs_owner(tmp_path):
    prefix = "merlin.compile.scheduling."
    source_public = {
        "merlin.llvmlower.late_quant_rne", "merlin.llvmlower.source_expression_interval",
        "merlin.llvmlower.source_expression_interval_llvm", "merlin.llvmlower.bounded_rne_word_cells",
        "merlin.llvmlower.scaled_integer_finite", "merlin.llvmlower.scaled_integer_finite_llvm",
        "merlin.llvmlower.repeated_pointwise", "merlin.llvmlower.weight_prepack_runtime",
        "merlin.llvmlower.contraction_row_slice",
        "merlin.transforms.dense_row_batching", "merlin.transforms.shared_rhs_views",
    }
    source_implementation = {
        "merlin.llvmlower.bounded_rne_maps", "merlin.llvmlower.closed_tensor_insert",
        "merlin.common.digest", "merlin.common.mlir_query", "merlin.common.yaml",
        "merlin.capture", "merlin.capture.safetensors", "merlin.llvmlower.quant_round_check",
        "merlin.llvmlower.segmented_matrix_view", "merlin.llvmlower.layout_conformance",
        "merlin.llvmlower.weights_pack", "merlin.llvmlower.prepared_operand_effects",
        "merlin.llvmlower.tensor_preparation_identity", "merlin.llvmlower.ordered_fma_groups",
        "merlin.transforms.head_batching", "merlin.transforms.dense_row_batching_check",
    }
    assert set(library_modules(24)) - set(library_modules(23)) == {
        prefix + name for name in ("compact_dense_group", "compact_dense_group_check", "dense_shared_rhs",
                                   "dense_shared_rhs_check", "shared_rhs_schedule_ir", "horizontal_convolution",
                                   "rolling_lookahead", "operand_slots", "projection_dense", "projection_provenance")
    } | source_public | source_implementation
    assert set(public_modules(24)) - set(public_modules(23)) == {
        prefix + "dense_shared_rhs", prefix + "horizontal_convolution", prefix + "rolling_lookahead",
        prefix + "operand_slots", prefix + "projection_dense",
    } | source_public
    assert _SHARED_SUBSTRATES[COMMON_SUBSTRATE_V17] == 24
    assert library_modules() == library_modules(6)
    root = tmp_path / "sdk"
    manifest = prepare_shared_source_release(
        root, python_source_root=repo_root() / "src", license_directory=repo_root(),
        tools=TR.ARM_TOOLS["merlin_rtlchecks"], version=24, source_binding=True,
    )
    assert manifest["schema"] == "merlin.shared_scheduling_library.v18"
    assert reviewed_shared_manifest(root, version=24) == manifest
    assert not [p for p in (root / "merlin").rglob("*.py")
                if author_inaccessible(staged_module(p.relative_to(root).as_posix()))]
    # Reuse the independent neutral effects fixture as the test driver. Every
    # compiler module, including the actual checker/emitter, must come from SDK.
    code = """
import importlib, pathlib, sys
from test_dense_shared_rhs import fixture, schedule
from merlin.compile.scheduling.api_catalog import scheduling_capabilities
rows = [r for r in scheduling_capabilities(24)['capabilities'] if r['name'] == 'shared_rhs_epochs']
assert len(rows) == 1
assert not any(r['name'] == 'shared_rhs_epochs' for r in scheduling_capabilities(23)['capabilities'])
for symbol in rows[0]['symbols']:
 module, name = symbol.rsplit('.', 1)
 assert callable(getattr(importlib.import_module(module), name))
for quantized in (False, True):
 model = fixture(quantized=quantized)
 result = schedule(model)
 assert result is not None and 'schedule.group_dispatch' in result.plan_text
 result.revalidate()
 assert schedule(model, schedule_plan=result.plan_text).kernel.digest() == result.kernel.digest()
 broken = result.plan_text.replace('shared_rhs:', 'other_rhs:', 1)
 try: schedule(model, schedule_plan=broken)
 except ValueError: pass
 else: raise AssertionError('changed source/group binding accepted')
import json, os, struct
from merlin.compile.scheduling.horizontal_convolution import (
 HorizontalConvolutionPanels, plan_horizontal_convolution_panels, emit_horizontal_convolution_panels)
from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan, Window2D
horizontal_rows = [r for r in scheduling_capabilities(24)['capabilities']
                   if r['name'] == 'bounded_horizontal_convolution_panels']
assert len(horizontal_rows) == 1
for symbol in horizontal_rows[0]['symbols']:
 module, name = symbol.rsplit('.', 1)
 assert callable(getattr(importlib.import_module(module), name))
from merlin.compile.scheduling.rolling_lookahead import (
 RollingLookaheadAdmission, materialize_lookahead_rolling_window,
 verify_lookahead_rolling_window, emit_lookahead_rolling_window)
from test_rolling_lookahead import plan as rolling_plan, IntegerBuilder
lookahead_rows = [r for r in scheduling_capabilities(24)['capabilities']
                  if r['name'] == 'disjoint_future_row_transport']
assert len(lookahead_rows) == 1
assert not any(r['name'] == 'disjoint_future_row_transport' for r in scheduling_capabilities(23)['capabilities'])
for symbol in lookahead_rows[0]['symbols']:
 module, name = symbol.rsplit('.', 1)
 assert callable(getattr(importlib.import_module(module), name))
p = rolling_plan()
lookahead = materialize_lookahead_rolling_window(p, output_rows=3)
checked_lookahead = verify_lookahead_rolling_window(lookahead,p,output_rows=3)
consumed = []
def rolling_effect(kind, fields):
 if kind == 'consume_group':
  consumed.extend((fields['image'],y) for y in range(fields['output_y'],fields['output_y']+fields['output_rows']))
 return 1
receipt = emit_lookahead_rolling_window(lookahead,p,IntegerBuilder(),rolling_effect,output_rows=3,
 admission=RollingLookaheadAdmission('test-hardware','test-effects',True,True,True,True,True))
assert receipt['actual_ir_sha256'] == checked_lookahead['actual_ir_sha256']
assert consumed == [(b,y) for b in range(p.window.input_shape[0]) for y in range(p.output_y_count)]
from test_rolling_lookahead import test_actual_scf_preserves_original_taps_and_never_overwrites_current_union
for original_runs in (1,17):
 test_actual_scf_preserves_original_taps_and_never_overwrites_current_union(p,3,original_runs)
from test_convolution_joint_storage import test_narrower_rhs_phases_free_real_accumulators_for_more_original_rows
test_narrower_rhs_phases_free_real_accumulators_for_more_original_rows()
frontier_rows = [r for r in scheduling_capabilities(24)['capabilities']
                 if r['name'] == 'last_use_rhs_phase_transport']
assert len(frontier_rows) == 1
for symbol in frontier_rows[0]['symbols']:
 module, name = symbol.rsplit('.', 1)
 assert callable(getattr(importlib.import_module(module), name))
from test_convolution_joint_storage import test_rhs_frontier_prefetch_waits_for_last_batch_and_spatial_consumer
test_rhs_frontier_prefetch_waits_for_last_batch_and_spatial_consumer(2)
from merlin.compile.scheduling.operand_slots import allocate_pipeline_operand_slots, verify_pipeline_operand_slots
from merlin.compile.scheduling.scf_commands import materialize_scf_commands
from test_operand_slots import source as slot_source, ADAPTER as slot_adapter
slot_rows = [r for r in scheduling_capabilities(24)['capabilities']
             if r['name'] == 'completion_bound_operand_slots']
assert len(slot_rows) == 1
for symbol in slot_rows[0]['symbols']:
 module, name = symbol.rsplit('.', 1)
 assert callable(getattr(importlib.import_module(module), name))
for overlap, lookahead in ((False,1),(True,1),(True,2)):
 original, operands, banks, options = slot_source(total=149,width=16,overlap=overlap,lookahead=lookahead)
 allocated = allocate_pipeline_operand_slots(original,operands,banks,slot_adapter,**options)
 assert allocated is not None
 checked = verify_pipeline_operand_slots(original,allocated.kernel,operands,banks,slot_adapter,**options)
 assert checked['operands']['invariant']['assigned_distinct_bases'] == 1
 assert checked['operands']['advancing']['assigned_distinct_bases'] == (lookahead+1 if overlap else 1)
 materialize_scf_commands(allocated.kernel).verify()
convolution = CountedConvolutionPlan(Window2D((1,6,11,3),(3,3),(2,2),padding=(1,2,0,1)),
                                   7,4,4,4,4,2048,64,0,16,0,False)
horizontal = plan_horizontal_convolution_panels(convolution,max_row_blocks=3,
                                               maximum_transfer_columns=64,private_rows_owned=True)
assert isinstance(horizontal,HorizontalConvolutionPanels)
assert horizontal.summary()['horizontal_columns'] == 9 and callable(emit_horizontal_convolution_panels)
from test_shared_rhs_views import source
from merlin.transforms.certificates import ir_text
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.weight_prepack_runtime import fixed_arguments
from merlin.llvmlower.repeated_pointwise import prepare_repeated_pointwise
from merlin.transforms.shared_rhs_views import plan_shared_rhs_row_views
from merlin.transforms.dense_row_batching import DenseRowBatchingBudgets, plan_source_dense_row_batching
from merlin.transforms.pipeline_state import PipelineContext
source_rows = [r for r in scheduling_capabilities(24)['capabilities']
               if r['name'] == 'source_proven_shared_rhs_row_views']
assert len(source_rows) == 1
for symbol in source_rows[0]['symbols']:
 module, name = symbol.rsplit('.', 1)
 assert callable(getattr(importlib.import_module(module), name))
bundle = pathlib.Path('source-bundle')
bundle.mkdir()
(bundle/'model.mlir').write_text(ir_text(source(g=2,r=3,m=1,k=3,n=5,scale='runtime',planes=2)))
manifest = {'3': {'kind':'buffer','weight':'scale','dtype':'float32','shape':[6,1,1]}}
(bundle/'weights.safetensors.manifest.json').write_text(json.dumps(manifest))
raw = struct.pack('<6f', *([2.0]*3+[3.0]*3))
header = json.dumps({'scale':{'dtype':'F32','shape':[6,1,1],'data_offsets':[0,len(raw)]}}).encode()
(bundle/'weights.safetensors').write_bytes(struct.pack('<Q',len(header))+header+raw)
assert fixed_arguments(bundle,manifest) == (3,)
os.environ['MERLIN_PREPARE_REPEATED_POINTWISE']='1'
prepared = prepare_repeated_pointwise(bundle/'model.mlir',pathlib.Path('prepared-source'),bundle_dir=bundle)
plan = plan_shared_rhs_row_views(parse_mlir_text(prepared.read_text()))
assert plan is not None
row_plan = plan_source_dense_row_batching(plan.candidate,PipelineContext(),DenseRowBatchingBudgets())
assert row_plan is not None and len(row_plan.record()['batches']) == 4
assert sum(row['members'] for row in row_plan.record()['batches']) == 12
for name, module in tuple(sys.modules.items()):
 if name == 'merlin' or name.startswith('merlin.'):
  path = getattr(module, '__file__', None)
  if path: assert pathlib.Path(path).is_relative_to(SDK), (name, path)
"""
    row_slice_source = runpy.run_path(
        str(repo_root() / "merlin/tests/ir/test_contraction_row_slice.py")
    )["source"]

    code += """
from merlin.llvmlower.contraction_row_slice import pushback_contraction_row_slices, prepare_contraction_row_slices
row_slice_entries = [r for r in scheduling_capabilities(24)['capabilities']
                     if r['name'] == 'sole_observer_contraction_rows']
assert len(row_slice_entries) == 1
for symbol in row_slice_entries[0]['symbols']:
 module, name = symbol.rsplit('.', 1)
 assert callable(getattr(importlib.import_module(module), name))
source = parse_mlir_text(ROW_SLICE_SOURCE)
smaller, row_report = pushback_contraction_row_slices(source)
assert row_report['rewrites'] == 1
assert row_report['regions'][0]['selected_m_k_n'] == [1,8,16]
os.environ['MERLIN_PASSES'] = 'contraction-row-slice'
assert prepare_contraction_row_slices(source)[1]['rewrites'] == 1
fixed_bundle = pathlib.Path('row-slice-source-bundle')
fixed_bundle.mkdir()
(fixed_bundle/'model.mlir').write_text(ROW_SLICE_RUNTIME_SOURCE)
fixed_manifest = {'2': {'kind':'buffer','weight':'coefficient','dtype':'float32','shape':[16]}}
(fixed_bundle/'weights.safetensors.manifest.json').write_text(json.dumps(fixed_manifest))
fixed_raw = struct.pack('<16f', *([0.03125]*16))
fixed_header = json.dumps({'coefficient':{'dtype':'F32','shape':[16],
                                        'data_offsets':[0,len(fixed_raw)]}}).encode()
(fixed_bundle/'weights.safetensors').write_bytes(struct.pack('<Q',len(fixed_header))+fixed_header+fixed_raw)
stored_source = parse_mlir_text(ROW_SLICE_RUNTIME_SOURCE)
_, stored_report = prepare_contraction_row_slices(stored_source,bundle_dir=fixed_bundle)
assert stored_report['rewrites'] == 1 and stored_report['immutable_generated_argument_indices'] == [2]
from test_projection_dense import setup as projection_setup, bind as projection_bind
projection_rows = [r for r in scheduling_capabilities(24)['capabilities']
                   if r['name'] == 'counted_projection_pointer_binding']
assert len(projection_rows) == 1
assert not any(r['name'] == 'counted_projection_pointer_binding'
               for r in scheduling_capabilities(23)['capabilities'])
for symbol in projection_rows[0]['symbols']:
 module, name = symbol.rsplit('.', 1)
 assert callable(getattr(importlib.import_module(module), name))
projection = projection_bind(projection_setup())
projection.revalidate()
assert not projection.report['dynamic_command_expansion']
observer_rows = [r for r in scheduling_capabilities(24)['capabilities']
                 if r['name'] == 'source_checked_integer_observers']
assert len(observer_rows) == 1
assert not any(r['name'] == 'source_checked_integer_observers'
               for r in scheduling_capabilities(23)['capabilities'])
for symbol in observer_rows[0]['symbols']:
 module, name = symbol.rsplit('.', 1)
 assert callable(getattr(importlib.import_module(module), name))
import runpy
observer_recipe = runpy.run_path(OBSERVER_RECIPE)
observed = observer_recipe['recipe']()
assert observed['source_checked_routes'] == observed['portable_round_routes'] == 1
assert observed['exact_cell_boundary_checks'] > 700 and observed['repetition'] == 3
for name, module in tuple(sys.modules.items()):
 if name == 'merlin' or name.startswith('merlin.'):
  path = getattr(module, '__file__', None)
  if path: assert pathlib.Path(path).is_relative_to(SDK), (name, path)
"""
    code = ("SDK = " + repr(str(root)) + "\nROW_SLICE_SOURCE = " + repr(row_slice_source())
            + "\nOBSERVER_RECIPE = " + repr(str(repo_root() / "packages/merlin-experiments/src/merlin_experiments/phase1/author_skills/heavy-pass-recipes/scripts/source_observer_recipe.py"))
            + "\nROW_SLICE_RUNTIME_SOURCE = " + repr(row_slice_source(runtime_scale=True)) + "\n" + code)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(root), str(repo_root() / "merlin/tests/infra"),
                                                      str(repo_root() / "merlin/tests/ir"),
                                                      str(Path(z3.__file__).parent.parent))),
           "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
