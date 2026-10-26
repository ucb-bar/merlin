"""Review examples only: actual pinned producer execution, no proof authority.

Run with the pinned producer's src on PYTHONPATH and this checkout's Python venv.
Outputs are new; existing qualification records and compiler sources are untouched.
"""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import traceback

from merlin.transforms.certificates import TRANSFORM_SCOPES, ir_text, parse_ir
from merlin.transforms.pipeline import PipelineOptions, apply_common_pipeline
from merlin.transforms.validation import validate_change

ROOT = Path(__file__).resolve().parent
SNAPSHOT = json.loads((ROOT / 'inventory.json').read_text())
PRODUCER = Path(SNAPSHOT['producer_root'])
PRIMARY = Path(SNAPSHOT['verification_root'])
assert subprocess.check_output(['git', '-C', str(PRODUCER), 'rev-parse', 'HEAD'], text=True).strip() == SNAPSHOT['producer_revision']
subprocess.run(['git', '-C', str(PRODUCER), 'diff', '--exit-code', 'HEAD', '--', 'src/merlin'], check=True)


def fixture_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


memory = fixture_module('review_memory_fixture', PRIMARY / 'src/merlin/verify/lean_graph_loops_reuse_cases.py')
unroll = fixture_module('review_unroll_fixture', PRIMARY / 'src/merlin/verify/lean_graph_loops_unroll_cases.py')


def scalar(body, result='%r'):
    return 'builtin.module { func.func @kernel(%a:i8, %b:i8) -> i8 {\n' + body + '\nfunc.return ' + result + ' : i8\n} }'


def loop(upper=7, dynamic=False, nested=False):
    args = '%upper:index, ' if dynamic else ''
    bound = '' if dynamic else f'%upper = arith.constant {upper} : index'
    body = '''%invariant = arith.addi %a, %b : i8
      "opaque.observe"(%invariant, %i) : (i8,index) -> ()
      %next = arith.subi %acc, %invariant : i8
      scf.yield %next : i8'''
    if nested:
        body = '''%inner = scf.for %j = %lower to %upper step %step iter_args(%state=%acc) -> (i8) {
        %invariant = arith.addi %a, %b : i8
        "opaque.observe"(%invariant, %i, %j) : (i8,index,index) -> ()
        %next = arith.subi %state, %invariant : i8
        scf.yield %next : i8
      }
      scf.yield %inner : i8'''
    return f'''builtin.module {{
  func.func @kernel({args}%a:i8, %b:i8) -> i8 {{
    %lower = arith.constant 0 : index
    {bound}
    %step = arith.constant 2 : index
    %r = scf.for %i = %lower to %upper step %step iter_args(%acc=%a) -> (i8) {{
      {body}
    }}
    func.return %r : i8
  }}
}}'''


cases = []
def add(case_id, kind, source, **options):
    cases.append((case_id, kind, source, options))

add('shape-cast', 'shape_view_normalization', '''builtin.module {
func.func @kernel(%a:memref<4xi8>) -> memref<4xi8> {
  %r = memref.cast %a : memref<4xi8> to memref<4xi8>
  func.return %r : memref<4xi8>
}}''', normalize_shapes=True)
add('shape-cast-generic', 'shape_view_normalization', '''builtin.module {
func.func @kernel(%a:memref<4xi8>) -> memref<4xi8> {
  %r = "memref.cast"(%a) : (memref<4xi8>) -> memref<4xi8>
  func.return %r : memref<4xi8>
}}''', normalize_shapes=True)
add('shape-dim', 'shape_view_normalization', '''builtin.module {
func.func @kernel(%a:memref<4xi8>) -> index {
  %z = arith.constant 0 : index
  %r = memref.dim %a, %z : memref<4xi8>
  func.return %r : index
}}''', normalize_shapes=True)
for operation in ('add', 'sub', 'mul'):
    add('integer-fold-' + operation, 'exact_integer_simplification', scalar(
        f'%x = arith.constant 127 : i8\n%y = arith.constant 3 : i8\n%r = arith.{operation}i %x, %y : i8'), simplify_integers=True)
for name, operation, constant, operands in (
    ('add-left', 'add', 0, '%c, %a'), ('add-right', 'add', 0, '%a, %c'),
    ('sub-right', 'sub', 0, '%a, %c'), ('mul-left', 'mul', 1, '%c, %a'), ('mul-right', 'mul', 1, '%a, %c')):
    add('integer-neutral-' + name, 'exact_integer_simplification', scalar(
        f'%c = arith.constant {constant} : i8\n%r = arith.{operation}i {operands} : i8'), simplify_integers=True)
add('scalar-cse', 'scalar_redundancy_elimination', scalar('''%first = arith.addi %a, %b : i8
%duplicate = arith.addi %a, %b : i8
%r = arith.muli %first, %duplicate : i8'''), eliminate_redundancy=True)
add('scalar-dce', 'scalar_redundancy_elimination', scalar('''%dead = arith.addi %a, %b : i8
%dead2 = arith.muli %dead, %b : i8''', '%a'), eliminate_redundancy=True)
for name, source in [('nonempty', loop()), ('empty', loop(upper=0)), ('dynamic', loop(dynamic=True)), ('nested', loop(nested=True))]:
    add('licm-' + name, 'scalar_loop_invariant_motion', source, hoist_invariants=True)
add('licm-dynamic-stride', 'scalar_loop_invariant_motion', loop().replace('@kernel(%a:', '@kernel(%step:index, %a:').replace('%step = arith.constant 2 : index', ''), hoist_invariants=True)
internal = scalar('''%first = arith.addi %a, %b : i8
%duplicate = arith.addi %a, %b : i8
%r = func.call @helper(%duplicate) : (i8) -> i8''')[:-1] + 'func.func @helper(%v:i8) -> i8 {func.return %v : i8}\n}'
add('scalar-internal-call', 'scalar_redundancy_elimination', internal, eliminate_redundancy=True)
for name, source, tiles in [('nonempty', loop(), (3,)), ('empty', loop(upper=0), (3,)), ('nested', loop(nested=True), (3,2))]:
    add('segment-' + name, 'ordered_loop_segmentation', source, tile_iterations=tiles)
for case in unroll.CASES:
    if case.name in ('inline_empty','inline_many','bulk_exact','bulk_tail'):
        add('unroll-' + case.name, 'ordered_loop_unrolling', unroll.source(case), unroll_factors=(case.factor,))
add('unroll-nested', 'ordered_loop_unrolling', loop(nested=True), unroll_factors=(2,2))
add('reuse-epochs', 'buffer_reuse', memory.source(memory.CASES[0]), reuse_buffers=True)
add('reuse-wide', 'buffer_reuse', memory.source(memory.Case('wide', (2,), 128, 16, 2)), reuse_buffers=True)
memory_one = memory.source(memory.Case('one', (2,), 32, 16, 2))
add('reuse-memory-space', 'buffer_reuse', memory_one.replace('memref<2xi32>', 'memref<2xi32, 3>'), reuse_buffers=True)
add('reuse-dynamic-access', 'buffer_reuse', memory_one.replace('@kernel(%seed0:', '@kernel(%address:index, %seed0:').replace('memref.load %buffer0[%zero]', 'memref.load %buffer0[%address]'), reuse_buffers=True)
add('reuse-rank-zero', 'buffer_reuse', '''builtin.module {
func.func @kernel(%seed:i32) -> i32 {
 %a = memref.alloc() : memref<i32>
 memref.store %seed, %a[] : memref<i32>
 %x = memref.load %a[] : memref<i32>
 memref.dealloc %a : memref<i32>
 %b = memref.alloc() : memref<i32>
 memref.store %seed, %b[] : memref<i32>
 %y = memref.load %b[] : memref<i32>
 memref.dealloc %b : memref<i32>
 %r = arith.addi %x, %y : i32
 func.return %r : i32
}}''', reuse_buffers=True)
pure = scalar('%x = arith.addi %a, %a : i8\n%y = arith.muli %b, %b : i8\n%r = arith.subi %x, %y : i8')
add('schedule-pure', 'sequential_dependency_schedule', pure, dependency_orders=((1,(1,0,2,3)),))
effectful = scalar('''%x = arith.addi %a, %a : i8
%y = arith.muli %b, %b : i8
"opaque.observe"(%a) : (i8) -> ()
%r = arith.subi %x, %y : i8''')
add('schedule-opaque', 'sequential_dependency_schedule', effectful, dependency_orders=((1,(1,0,2,3,4)),))
multi = pure[:-1] + 'func.func @untouched(%a:i8) -> i8 {func.return %a : i8}\n}'
add('schedule-multiple', 'sequential_dependency_schedule', multi, dependency_orders=((1,(1,0,2,3)),))

selection = sys.argv[1:]
if selection:
    cases = [case for case in cases if case[0] in selection]
    assert len(cases) == len(selection), selection
results=[]
for case_id, kind, source, options in cases:
    destination = ROOT / 'producer-examples' / case_id
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'source.mlir').write_text(source+'\n')
    record={'case':case_id,'kind':kind,'options':options,'lean_run':False,'qualified':False,'review_approved':False}
    try:
        settings=dict(index_bits=64, normalize_shapes=False, simplify_integers=False, reuse_buffers=False)
        settings.update(options)
        before=parse_ir(source)
        result=apply_common_pipeline(before,PipelineOptions(**settings),plan_only=True)
        assert len(result.changes)==1, [c.kind for c in result.changes]
        change=result.changes[0]
        assert change.kind==kind and change.before_ir!=change.after_ir
        validate_change(change)
        (destination/'before.mlir').write_text(change.before_ir)
        (destination/'after.mlir').write_text(change.after_ir)
        (destination/'change.json').write_text(json.dumps(change.to_dict(),indent=2)+'\n')
        record.update(status='actual_changed_producer_output',structural_validator='passed; same-producer reconstruction/recognition, not independent semantic proof',witness_rules=sorted({getattr(w,'rule','structural') for w in change.certificate.witnesses}),witness_count=len(change.certificate.witnesses),source_unmodified=ir_text(before)==change.before_ir)
    except Exception as error:
        record.update(status='capture_failed',error_type=type(error).__name__,error=str(error))
        (destination/'failure.txt').write_text(traceback.format_exc())
    record['files']=[{'path':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(destination.iterdir()) if p.is_file()]
    (destination/'capture.json').write_text(json.dumps(record,indent=2)+'\n')
    results.append(record)

loaded=[]
for name,module in tuple(sys.modules.items()):
    path=getattr(module,'__file__',None)
    if path and (name=='merlin' or name.startswith('merlin.')):
        path=Path(path).resolve()
        assert path.is_relative_to(PRODUCER/'src'), (name,path)
        loaded.append({'module':name,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
report={'schema':'graham.registry-eight.producer-examples.v1','producer_revision':SNAPSHOT['producer_revision'],'verification_revision':SNAPSHOT['verification_revision'],'new_lean_qualification':False,'new_lean_tests_run':False,'method':'Pinned actual common pipeline in explicit plan_only mode; private module clones; original sources and evidence unchanged','case_results':results,'loaded_producer_sources':loaded,'fixture_sources':[str(PRIMARY/'src/merlin/verify/lean_graph_loops_reuse_cases.py'),str(PRIMARY/'src/merlin/verify/lean_graph_loops_unroll_cases.py')]}
report_name = 'producer-examples' + (('-' + '-'.join(selection)) if selection else '') + '.json'
(ROOT/report_name).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'cases':len(results),'changed':sum(r['status']=='actual_changed_producer_output' for r in results),'failures':[r for r in results if r['status']=='capture_failed'],'all_eight_kinds_present':{r['kind'] for r in results}==set(TRANSFORM_SCOPES)},indent=2))
