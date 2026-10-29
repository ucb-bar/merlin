"""Bounded actual empty/full-inline source tests; runtime/object access denied."""
from dataclasses import asdict
from pathlib import Path
import json
import sys
import types
import unittest

PRIMARY=REFERENCE=None
CASES={'empty':(3,3,2,2,2,32),'inline1':(5,6,2,1,2,32),
       'inline2':(2,8,3,2,2,64),'inline3':(7,16,3,2,4,32),
       'wide_inline1':(2147483650,2147483651,2,1,2,64)}


def source(lower,upper,step,carries):
    second=', %d=%b' if carries==2 else ''
    result='(i8, i8)' if carries==2 else '(i8)'
    yielded='%next, %d : i8, i8' if carries==2 else '%next : i8'
    returned='%r#0' if carries==2 else '%r'
    second_result='%r#1' if carries==2 else '%b'
    return f'''builtin.module {{
      func.func private @step(index,i8) -> i8
      func.func private @observe(i8,i8) -> i8
      func.func private @altered(i8,i8) -> i8
      func.func @main(%a:i8,%b:i8) -> i8 {{
        %lb=arith.constant {lower} : index
        %ub=arith.constant {upper} : index
        %st=arith.constant {step} : index
        %r{' : 2' if carries==2 else ''}=scf.for %i=%lb to %ub step %st iter_args(%c=%a{second}) -> {result} {{
          %sum=arith.addi %c,%b : i8
          %next=func.call @step(%i,%sum) : (index,i8) -> i8
          scf.yield {yielded}
        }}
        %seen=func.call @observe({returned},{second_result}) : (i8,i8) -> i8
        func.return %seen : i8
      }}
      func.func @untouched(%x:i8) -> i8 {{ func.return %x : i8 }}
    }}'''


def capture(case):
    from merlin.transforms.pipeline import apply_common_pipeline,PipelineOptions
    from merlin.transforms.certificates import parse_ir
    lower,upper,step,carries,factor,bits=CASES[case]
    result=apply_common_pipeline(parse_ir(source(lower,upper,step,carries)),PipelineOptions(
        index_bits=bits,normalize_shapes=False,simplify_integers=False,reuse_buffers=False,
        unroll_factors=(factor,)),plan_only=True)
    if len(result.changes)!=1 or result.changes[0].kind!='ordered_loop_unrolling':
        raise AssertionError('one actual registered unroll change required')
    return result.changes[0]


def mutate(raw,kind):
    from merlin.transforms.certificates import parse_ir,ir_text
    from xdsl.dialects import func,arith,builtin
    from xdsl.rewriter import Rewriter
    module=parse_ir(raw)
    main=next(op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data=='main')
    block=main.body.block;ops=list(block.ops)
    steps=[op for op in ops if type(op) is func.CallOp and op.callee.root_reference.data=='step']
    observer=next(op for op in ops if type(op) is func.CallOp and op.callee.root_reference.data=='observe')
    if kind in ('lane','cutoff'):
        old=steps[-1].operands[0].owner if kind=='lane' else ops[1]
        new=arith.ConstantOp.from_int_and_width(old.value.value.data+1,builtin.IndexType())
        Rewriter.replace_op(old,new)
    elif kind=='initial_carry':
        first=next(op for op in ops if type(op) is arith.AddiOp)
        first.operands=(main.body.block.args[1],first.operands[1])
    elif kind=='lane_carry':
        adds=[op for op in ops if type(op) is arith.AddiOp]
        adds[-1].operands=(main.body.block.args[0],adds[-1].operands[1])
    elif kind=='capture':
        first=next(op for op in ops if type(op) is arith.AddiOp)
        first.operands=(first.operands[0],main.body.block.args[0])
    elif kind=='return':
        block.last_op.operands=(main.body.block.args[0],)
    elif kind=='observer':
        observer.properties['callee']=builtin.SymbolRefAttr('altered')
    elif kind=='flags':
        first=next(op for op in ops if type(op) is arith.AddiOp)
        first.properties['overflowFlags']=arith.IntegerOverflowAttr([arith.IntegerOverflowFlag.NSW])
    elif kind=='empty_body':
        call=func.CallOp('step',[ops[0].results[0],block.args[0]],[block.args[0].type])
        block.insert_op_before(call,observer)
    elif kind=='empty_carry':
        observer.operands=(block.args[1],observer.operands[1])
    elif kind=='carry_order':
        observer.operands=tuple(reversed(observer.operands))
    elif kind=='dropped_entry':
        other=next(op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data=='untouched')
        module.body.block.erase_op(other)
    else:raise ValueError(kind)
    module.verify();return ir_text(module)


MUTATIONS=('lane','cutoff','initial_carry','lane_carry','carry_order','capture','return','observer','flags','dropped_entry')


class Inline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from merlin.verify import lean_graph_pipeline_inline as inline
        from merlin.verify import lean_graph_pipeline_inline_models as models
        cls.inline,cls.models=inline,models

    def prepared(self,case='inline3'):
        change=capture(case);values=CASES[case]
        return change,self.inline.prepare(change.before_ir,change.after_ir,self.inline.Selection(values[4],values[5]))

    def test_actual_empty_and_full_inline_complete_pairs_common_roots_witnesses(self):
        from merlin.transforms import unrolling
        self.assertEqual(Path(unrolling.__file__).resolve(),REFERENCE/'src/merlin/transforms/unrolling.py')
        for case in CASES:
            with self.subTest(case=case):
                change,p=self.prepared(case);fact=json.loads(p.field_facts)[0]
                self.assertNotEqual(change.before_ir,change.after_ir)
                self.assertEqual(len(json.loads(p.projection)['functions']),5)
                self.assertEqual(fact['mode'],'empty' if case=='empty' else 'full-inline')
                self.assertEqual(len(fact['lane_constants']),fact['count'])
                self.assertIn('empty_body_never_runs' if case=='empty' else 'Inline.counted_continuation',p.query)
                self.assertEqual(p.roots[-1],'merlin_unroll_inline_transform')
                self.assertEqual(p.describe()['outcome'],'UNOBSERVED')
                q,roots,model=self.models.witness(p)
                self.assertIn('source_observer_failure',q)
                self.assertIn('source_empty_body_fault_skipped' if case=='empty' else 'source_last_body_failure',q)
                self.assertFalse(model['source_success_assumed']);self.assertEqual(len(roots),len(set(roots)))

    def test_compiler_valid_wrong_fields_captures_and_surrounding_output(self):
        change,_=self.prepared()
        for name in MUTATIONS:
            with self.subTest(name=name):
                bad=mutate(change.after_ir,name);self.assertNotEqual(bad,change.after_ir)
                with self.assertRaises(ValueError):self.inline.prepare(change.before_ir,bad,self.inline.Selection(4,32))

    def test_empty_body_must_never_execute_or_change_initial_carry(self):
        change,_=self.prepared('empty')
        for name in ('empty_body','empty_carry','return','observer'):
            with self.subTest(name=name):
                bad=mutate(change.after_ir,name)
                with self.assertRaises(ValueError):self.inline.prepare(change.before_ir,bad,self.inline.Selection(2,32))
        from merlin.transforms.certificates import parse_ir
        from merlin.verify.lean_graph_state_scalar_capture import project
        p=project(parse_ir(change.before_ir),parse_ir(mutate(change.after_ir,'empty_body')))
        q,roots=self.models.empty_countermodel(p)
        self.assertIn('observations_differ',q);self.assertEqual(len(roots),3)

    def test_wrong_profile_stale_source_domain_roots_never_authorize(self):
        change,p=self.prepared();selection=self.inline.Selection(4,32)
        request=self.inline.request(self.inline.PROFILE,change.before_ir,change.after_ir,selection)
        for key,value in (('profile','tail'),('domain','possible_success'),('final_root','rfl'),('support_roots',[]),('project_sources',[])):
            bad=dict(request);bad[key]=value
            self.assertFalse(self.inline.matches(bad,self.inline.PROFILE,change.before_ir,change.after_ir,selection))
        with self.assertRaises(ValueError):self.inline.request('minimal7',change.before_ir,change.after_ir,selection)
        with self.assertRaises(ValueError):self.inline.prepare(capture('inline2').before_ir,change.after_ir,selection)
        self.assertFalse(self.inline.compiler_eligible(p));self.assertFalse(self.inline.compiler_eligible(request))
        with self.assertRaises(RuntimeError):self.inline.provider_arguments(request)

    def test_bulk_dynamic_flags_and_internal_calls_refused(self):
        from merlin.transforms.certificates import parse_ir,ir_text
        from merlin.transforms.unrolling import unroll_ordered_loops
        from xdsl.dialects import arith
        original=source(0,5,1,1)
        for text in (original,source(0,1,1,1).replace('func.call @step(%i,%sum) : (index,i8) -> i8','func.call @untouched(%sum) : (i8) -> i8'),
                     source(0,1,1,1).replace('func.func @main(%a:i8,%b:i8)','func.func @main(%a:i8,%b:i8,%dynamic:index)').replace('%ub=arith.constant 1 : index','').replace('to %ub','to %dynamic')):
            module=parse_ir(text);before=ir_text(module)
            unroll_ordered_loops(module,(2,),index_bits=32)
            with self.assertRaises(ValueError):self.inline.prepare(before,ir_text(module),self.inline.Selection(2,32))
        module=parse_ir(source(0,1,1,1))
        addition=next(op for op in module.walk() if type(op) is arith.AddiOp)
        addition.properties['overflowFlags']=arith.IntegerOverflowAttr([arith.IntegerOverflowFlag.NSW])
        before=ir_text(module);unroll_ordered_loops(module,(2,),index_bits=32)
        with self.assertRaises(ValueError):self.inline.prepare(before,ir_text(module),self.inline.Selection(2,32))


def retain(destination):
    destination.mkdir(parents=True,exist_ok=False);Inline.setUpClass();test=Inline()
    for case,values in CASES.items():
        change,p=test.prepared(case);folder=destination/case;folder.mkdir()
        (folder/'before.mlir').write_text(change.before_ir);(folder/'after.mlir').write_text(change.after_ir)
        (folder/'CHANGE.json').write_text(json.dumps(asdict(change),indent=2)+'\n')
        (folder/'PREPARED.json').write_text(json.dumps(p.describe(),indent=2)+'\n')
        (folder/'REQUEST.json').write_text(json.dumps(test.inline.request(test.inline.PROFILE,change.before_ir,change.after_ir,test.inline.Selection(values[4],values[5])),indent=2)+'\n')
        (folder/'Query.lean').write_text(p.query)
        q,roots,model=test.models.witness(p)
        (folder/'Witness.lean').write_text(q);(folder/'WITNESS.json').write_text(json.dumps({**model,'roots':roots},indent=2)+'\n')
    folder=destination/'mutations';folder.mkdir()
    for case,names in (('inline3',MUTATIONS),('empty',('empty_body','empty_carry','observer'))):
        change,_=test.prepared(case);values=CASES[case]
        for name in names:
            bad=mutate(change.after_ir,name);label=case+'-'+name
            (folder/(label+'.mlir')).write_text(bad)
            try:test.inline.prepare(change.before_ir,bad,test.inline.Selection(values[4],values[5]))
            except ValueError as error:reason=str(error)
            else:raise AssertionError('mutation admitted')
            (folder/(label+'.json')).write_text(json.dumps({'host_refusal':reason,'parser_verify':True,
                'before':'../'+case+'/before.mlir','query_status':'UNCOMPILED','kernel_refusal_observed':False})+'\n')
            if case=='empty' and name=='empty_body':
                from merlin.transforms.certificates import parse_ir
                from merlin.verify.lean_graph_state_scalar_capture import project
                projection=project(parse_ir(change.before_ir),parse_ir(bad));q,roots=test.models.empty_countermodel(projection)
                (folder/'EmptyBodyCountermodel.lean').write_text(q)
                (folder/'EmptyBodyCountermodel.json').write_text(json.dumps({'projection':projection,'roots':roots,
                    'query_status':'UNCOMPILED','outcome':'UNOBSERVED','model_approved':False},indent=2)+'\n')


if __name__=='__main__':
    PRIMARY=Path(sys.argv[1]).resolve();REFERENCE=Path(sys.argv[2]).resolve()
    for name,paths in (('merlin',[REFERENCE/'src/merlin',PRIMARY/'src/merlin']),
                       ('merlin.verify',[PRIMARY/'src/merlin/verify',REFERENCE/'src/merlin/verify']),
                       ('merlin.common',[PRIMARY/'src/merlin/common',REFERENCE/'src/merlin/common'])):
        module=types.ModuleType(name);module.__path__=[str(p) for p in paths];sys.modules[name]=module
    def held(event,args):
        if event in ('subprocess.Popen','os.system','os.exec','os.posix_spawn'):raise AssertionError('source-only inline cannot launch children')
        if event=='open' and isinstance(args[0],(str,bytes)):
            p=str(args[0])
            if ('/.elan/' in p or '/toolchain/' in p or 'libleanshared' in p or '/original-runtime-candidate' in p
                    or p.endswith(('.olean','.ilean'))):raise AssertionError('no runtime/object access')
    sys.addaudithook(held)
    args=sys.argv[3:]
    if args and args[0]=='--retain':retain(Path(args[1]));args=args[2:]
    unittest.main(argv=[sys.argv[0],*args])
