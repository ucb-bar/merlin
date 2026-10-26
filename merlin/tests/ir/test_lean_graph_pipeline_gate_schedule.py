"""Bounded source-only tests using pinned4499 actual planning/pipeline APIs."""
from pathlib import Path
from dataclasses import asdict
import importlib.util
import json
import sys
import types
import unittest
from unittest.mock import patch

PRIMARY=REFERENCE=None
SOURCE='''builtin.module {
  func.func private @observe(i8) -> i8
  func.func private @other(i8) -> i8
  func.func @main(%a: i8, %b: i8) -> i8 {
    %x = arith.addi %a, %b : i8
    %y = arith.muli %a, %b : i8
    %first = func.call @observe(%x) : (i8) -> i8
    %z = arith.subi %a, %b : i8
    %second = func.call @other(%y) : (i8) -> i8
    %r = arith.addi %first, %second : i8
    %out = arith.addi %r, %z : i8
    func.return %out : i8
  }
  func.func @untouched(%a: i8) -> i8 { func.return %a : i8 }
}'''
ORDERS={'pure_reorder':(1,0,2,3,4,5,6,7),
        'cross_observer_keep_effect_order':(0,1,3,2,4,5,6,7),
        'unchanged_order':tuple(range(8))}


def capture(name):
    from merlin.transforms.certificates import parse_ir
    from merlin.transforms.pipeline import apply_common_pipeline,PipelineOptions
    result=apply_common_pipeline(parse_ir(SOURCE),PipelineOptions(index_bits=64,
        normalize_shapes=False,simplify_integers=False,reuse_buffers=False,
        dependency_orders=((1,ORDERS[name]),)),plan_only=True)
    if len(result.changes)!=1:raise AssertionError('one actual registered scheduling boundary required')
    return result.changes[0]


class Schedule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from merlin.verify import lean_graph_pipeline_gate_schedule as schedule
        from merlin.verify import lean_graph_pipeline_gate_schedule_requests as requests
        from merlin.verify import lean_graph_pipeline_gate_schedule_models as models
        cls.schedule,cls.requests,cls.models=schedule,requests,models

    def prepared(self,name='pure_reorder'):
        change=capture(name)
        return change,self.schedule.prepare(change.before_ir,change.after_ir,self.schedule.Order(1,ORDERS[name]))

    def test_actual_pinned_registered_producer_and_complete_entry_reuse(self):
        from merlin.transforms import planning,pipeline
        self.assertEqual(Path(planning.__file__).resolve(),REFERENCE/'src/merlin/transforms/planning.py')
        self.assertEqual(Path(pipeline.__file__).resolve(),REFERENCE/'src/merlin/transforms/pipeline.py')
        for name in ORDERS:
            with self.subTest(case=name):
                change,p=self.prepared(name)
                self.assertEqual(change.kind,'sequential_dependency_schedule')
                self.assertEqual(p.changed,name!='unchanged_order')
                data=json.loads(p.projection);self.assertEqual(len(data['functions']),4)
                self.assertIn('Merlin.GraphState.Scalar.check_sound',p.roots)
                self.assertEqual(p.roots[-1],'merlin_schedule_transform')
                self.assertIn('PreservesEntries',p.query)
                self.assertEqual(p.describe()['outcome'],'UNOBSERVED')
                matched=json.loads(p.matching)
                self.assertFalse(matched['dependency_legality_is_execution_proof'])
                self.assertFalse(matched['before']['totality_from_empty_effects'])
                self.assertEqual(matched['before']['semantic'],matched['after']['semantic'])
                q,roots,model=self.models.witness(p)
                self.assertIn('.ok ([21],9)',q)
                self.assertIn('source_second_failure',q)
                self.assertFalse(model['source_success_assumed'])
                self.assertEqual(len(roots),len(set(roots)))

    def test_compiler_valid_wrong_result_observer_reorder_and_dropped_function(self):
        from merlin.transforms.certificates import parse_ir,ir_text
        from merlin.transforms.planning import blocks_in
        from xdsl.dialects import func
        change,_=self.prepared()
        bad_result=change.after_ir.replace('"func.return"(%out)','"func.return"(%a)')
        self.assertNotEqual(bad_result,change.after_ir);parse_ir(bad_result).verify()
        with self.assertRaises(ValueError):self.schedule.prepare(change.before_ir,bad_result,self.schedule.Order(1,ORDERS['pure_reorder']))
        module=parse_ir(change.before_ir);block=blocks_in(module)[1];ops=list(block.ops)
        order=(0,1,4,3,2,5,6,7)
        for op in ops:block.detach_op(op)
        block.add_ops(ops[i] for i in order);module.verify()
        with self.assertRaises(ValueError):self.schedule.prepare(change.before_ir,ir_text(module),self.schedule.Order(1,order))
        module=parse_ir(change.after_ir)
        other=next(op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data=='untouched')
        module.body.block.erase_op(other);module.verify()
        with self.assertRaises(ValueError):self.schedule.prepare(change.before_ir,ir_text(module),self.schedule.Order(1,ORDERS['pure_reorder']))

    def test_stale_order_and_permutation_fail_without_report_authority(self):
        change,_=self.prepared()
        for block,order in ((1,ORDERS['unchanged_order']),(1,(1,1,2,3,4,5,6,7)),
                            (1,(True,0,2,3,4,5,6,7)),(2,ORDERS['pure_reorder'])):
            with self.subTest(block=block,order=order),self.assertRaises(ValueError):
                self.schedule.prepare(change.before_ir,change.after_ir,self.schedule.Order(block,order))
        changed=capture('cross_observer_keep_effect_order')
        with self.assertRaises(ValueError):self.schedule.prepare(change.before_ir,changed.after_ir,self.schedule.Order(1,ORDERS['pure_reorder']))

    def test_partial_empty_effect_arithmetic_is_not_total_or_commutative(self):
        from merlin.transforms.certificates import parse_ir,ir_text
        from merlin.transforms.planning import apply_order,blocks_in
        # divsi is region-free with empty effects, but outside the closed total
        # whitelist. It remains an opaque anchor relative to observe/other.
        source=SOURCE.replace('arith.subi %a, %b','arith.divsi %a, %b')
        module=parse_ir(source);before=ir_text(module)
        apply_order(module,1,ORDERS['cross_observer_keep_effect_order']);after=ir_text(module)
        self.assertNotEqual(before,after);module.verify()
        with self.assertRaisesRegex(ValueError,'anchor|context'):
            self.schedule.prepare(before,after,self.schedule.Order(1,ORDERS['cross_observer_keep_effect_order']))

    def test_region_and_internal_call_domains_refuse(self):
        from merlin.transforms.certificates import parse_ir,ir_text
        region='''builtin.module { func.func @main(%x:i8) -> i8 {
          %lb=arith.constant 0 : index
          %ub=arith.constant 2 : index
          %st=arith.constant 1 : index
          %r=scf.for %i=%lb to %ub step %st iter_args(%a=%x) -> (i8) { scf.yield %a : i8 }
          func.return %r : i8 } }'''
        raw=ir_text(parse_ir(region))
        with self.assertRaises(ValueError):self.schedule.prepare(raw,raw,self.schedule.Order(1,(0,1,2,3,4)))
        internal=SOURCE.replace('func.call @observe(%x)','func.call @untouched(%x)')
        raw=ir_text(parse_ir(internal))
        with self.assertRaises(ValueError):self.schedule.prepare(raw,raw,self.schedule.Order(1,ORDERS['unchanged_order']))

    def test_exact_request_wrong_domain_roots_and_held_public_route(self):
        change,p=self.prepared();order=self.schedule.Order(1,ORDERS['pure_reorder'])
        request=self.requests.request(self.schedule.PROFILE,change.before_ir,change.after_ir,order)
        self.assertEqual(len(request.project_sources),7)
        for key,value in (('domain','possible_success'),('kind','scalar_redundancy_elimination'),
                           ('final_root','identity'),('support_roots',[]),('project_sources',[])):
            forged=request.describe();forged[key]=value
            self.assertFalse(self.requests.matches(forged,self.schedule.PROFILE,change.before_ir,change.after_ir,order))
        with self.assertRaises(ValueError):self.requests.request('minimal7',change.before_ir,change.after_ir,order)
        self.assertFalse(self.schedule.compiler_eligible(request))
        self.assertFalse(self.schedule.compiler_eligible(request.describe()))
        with self.assertRaises(RuntimeError):self.schedule.public_check(request)
        with self.assertRaises(RuntimeError):self.requests.provider_arguments(request)
        from merlin.verify import lean_graph_pipeline_gate_native as two_kind
        with self.assertRaises(RuntimeError):two_kind.request(p)


def retain(destination):
    destination.mkdir(parents=True,exist_ok=False);Schedule.setUpClass();test=Schedule()
    for name in ORDERS:
        change,p=test.prepared(name);folder=destination/name;folder.mkdir()
        order=test.schedule.Order(1,ORDERS[name])
        request=test.requests.request(test.schedule.PROFILE,change.before_ir,change.after_ir,order)
        (folder/'before.mlir').write_text(change.before_ir);(folder/'after.mlir').write_text(change.after_ir)
        (folder/'CHANGE.json').write_text(json.dumps(asdict(change),indent=2)+'\n')
        (folder/'PREPARED.json').write_text(json.dumps(p.describe(),indent=2)+'\n')
        (folder/'REQUEST.json').write_text(json.dumps(request.describe(),indent=2)+'\n')
        (folder/'Query.lean').write_text(p.query)
        q,roots,model=test.models.witness(p)
        (folder/'Witness.lean').write_text(q)
        (folder/'WITNESS.json').write_text(json.dumps({**model,'roots':roots},indent=2)+'\n')
    from merlin.transforms.certificates import parse_ir,ir_text
    from merlin.transforms.planning import blocks_in
    from merlin.verify.lean_graph_state_scalar_capture import project
    change=capture('pure_reorder');module=parse_ir(change.before_ir)
    block=blocks_in(module)[1];ops=list(block.ops);order=(0,1,4,3,2,5,6,7)
    for op in ops:block.detach_op(op)
    block.add_ops(ops[i] for i in order);module.verify();bad=ir_text(module)
    folder=destination/'reordered-observer-negative';folder.mkdir()
    (folder/'before.mlir').write_text(change.before_ir);(folder/'after.mlir').write_text(bad)
    reason=None
    try:test.schedule.prepare(change.before_ir,bad,test.schedule.Order(1,order))
    except ValueError as error:reason=str(error)
    if reason is None:raise AssertionError('invalid effect reorder admitted by host')
    projection=project(parse_ir(change.before_ir),module)
    q,roots=test.models.observer_countermodel(projection)
    (folder/'Countermodel.lean').write_text(q)
    (folder/'STATUS.json').write_text(json.dumps({'parse_verify':True,'host_refusal':reason,
        'query_status':'UNCOMPILED','outcome':'UNOBSERVED','kernel_refusal_observed':False,
        'source_failure_world_proposed':8,'reordered_failure_world_proposed':9,
        'roots':roots,'projection':projection,'order':order},indent=2)+'\n')


if __name__=='__main__':
    PRIMARY=Path(sys.argv[1]).resolve();REFERENCE=Path(sys.argv[2]).resolve()
    # Source paths are explicit: actual pinned producer; owned verification and
    # primary support. No runner/shim/reimplementation substitutes production.
    for name,paths in (('merlin',[REFERENCE/'src/merlin',PRIMARY/'src/merlin']),
                       ('merlin.verify',[PRIMARY/'src/merlin/verify',REFERENCE/'src/merlin/verify']),
                       ('merlin.common',[PRIMARY/'src/merlin/common',REFERENCE/'src/merlin/common'])):
        module=types.ModuleType(name);module.__path__=[str(p) for p in paths];sys.modules[name]=module
    def held(event,args):
        if event in ('subprocess.Popen','os.system','os.exec','os.posix_spawn'):
            raise AssertionError('source-only schedule cannot launch child processes')
        if event=='open' and isinstance(args[0],(str,bytes)):
            p=str(args[0])
            if ('/.elan/' in p or '/toolchain/' in p or 'libleanshared' in p or '/original-runtime-candidate' in p
                    or p.endswith(('.olean','.ilean'))):raise AssertionError('no runtime/object access')
    sys.addaudithook(held)
    args=sys.argv[3:]
    if args and args[0]=='--retain':retain(Path(args[1]));args=args[2:]
    unittest.main(argv=[sys.argv[0],*args])
