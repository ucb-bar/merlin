"""Observe shipped projection admission on fresh producer examples.

This invokes production projections as subjects, not independent semantic oracles.
No kernel, qualification, historical receipt replay, or review approval is issued.
"""
from pathlib import Path
import hashlib
import importlib
import json
import sys
import types
import traceback

from merlin.transforms.certificates import parse_ir
from merlin.verify.lean_export import export_scalar, equivalence_query, UnsupportedLeanIR
from merlin.verify.lean_state_effects import EffectScheduleContract, derive_effect_schedule

ROOT=Path(__file__).resolve().parent
snapshot=json.loads((ROOT/'inventory.json').read_text())
package=types.ModuleType('review_primary')
package.__path__=[str(Path(snapshot['verification_root'])/'src/merlin/verify')]
sys.modules[package.__name__]=package
load=lambda name: importlib.import_module('review_primary.'+name)
scalar=load('lean_graph_state_scalar_capture')
trees=load('lean_graph_loops_segment_capture')
segment=load('lean_graph_loops_segment_projection')
unroll=load('lean_graph_loops_unroll_projection')
reuse_capture=load('lean_graph_loops_reuse_capture')
reuse=load('lean_graph_loops_reuse_projection')
results=[]

def probe(case, route, action):
    folder=ROOT/'projection-probes'/case
    folder.mkdir(parents=True,exist_ok=True)
    result={'case':case,'route':route,'lean_run':False,'qualification_issued':False,'independent_correspondence_review':False}
    try:
        model=action()
        path=folder/(route+'.model.json')
        path.write_text(json.dumps(model,indent=2)+'\n')
        result.update(status='projection_emitted_not_kernel_checked',model=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    except ValueError as error:
        result.update(status='projection_refused',reason_type=type(error).__name__,reason=str(error))
    except Exception as error:
        result.update(status='setup_or_unexpected_error',reason_type=type(error).__name__,reason=str(error))
        (folder/(route+'.failure.txt')).write_text(traceback.format_exc())
    (folder/(route+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    results.append(result)

for directory in sorted((ROOT/'producer-examples').iterdir()):
    record=json.loads((directory/'capture.json').read_text())
    if record['status']!='actual_changed_producer_output': continue
    before=(directory/'before.mlir').read_text()
    after=(directory/'after.mlir').read_text()
    kind=record['kind']
    if kind in ('scalar_redundancy_elimination','scalar_loop_invariant_motion'):
        probe(directory.name,'scalar',lambda: scalar.project(parse_ir(before),parse_ir(after)))
    elif kind=='ordered_loop_segmentation':
        probe(directory.name,'segment',lambda: segment.project_trees(trees.operation_tree(parse_ir(before)),trees.operation_tree(parse_ir(after)),tile=record['options']['tile_iterations'][0],index_bits=64,max_nodes=10000))
    elif kind=='ordered_loop_unrolling':
        probe(directory.name,'unroll',lambda: unroll.project_trees(trees.operation_tree(parse_ir(before)),trees.operation_tree(parse_ir(after)),factor=record['options']['unroll_factors'][0],index_bits=64,max_nodes=10000))
    elif kind=='buffer_reuse':
        probe(directory.name,'reuse',lambda: reuse.project_exports(reuse_capture.export(parse_ir(before)),reuse_capture.export(parse_ir(after)),index_bits=64,max_nodes=10000))
    elif kind=='sequential_dependency_schedule':
        probe(directory.name,'old-scalar-export',lambda: {'query':equivalence_query(export_scalar(before,'kernel'),export_scalar(after,'kernel'))})
        probe(directory.name,'state-effects',lambda: derive_effect_schedule(before,after,entry='kernel',contract=EffectScheduleContract()).record())

report={'method':'local admission probes only; shipped projections are the subjects, not independent oracles','new_lean_qualification':False,'cases':results}
(ROOT/'projection-probes.json').write_text(json.dumps(report,indent=2)+'\n')
for result in results:
    print(result['case'],result['route'],result['status'],result.get('reason',''))
assert not any(r['status']=='setup_or_unexpected_error' for r in results)
