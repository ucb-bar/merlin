"""Bounded SDK-free actual ordered-unroll tail preparation and valid mutations."""
from dataclasses import asdict
from pathlib import Path
import json
import sys
import types
import unittest

PRIMARY=REFERENCE=None


def source(lower=0,upper=5,step=1,carries=1):
    second=', %d=%b' if carries==2 else ''
    result='(i8, i8)' if carries==2 else '(i8)'
    yielded='%next, %d : i8, i8' if carries==2 else '%next : i8'
    returned='%r#0' if carries==2 else '%r'
    return f'''builtin.module {{
      func.func private @step(index, i8) -> i8
      func.func private @observe(i8) -> i8
      func.func private @altered(i8) -> i8
      func.func @main(%a:i8, %b:i8) -> i8 {{
        %lb=arith.constant {lower} : index
        %ub=arith.constant {upper} : index
        %st=arith.constant {step} : index
        %r{' : 2' if carries==2 else ''}=scf.for %i=%lb to %ub step %st iter_args(%c=%a{second}) -> {result} {{
          %next=func.call @step(%i,%c) : (index,i8) -> i8
          scf.yield {yielded}
        }}
        %seen=func.call @observe({returned}) : (i8) -> i8
        func.return %seen : i8
      }}
      func.func @untouched(%x:i8) -> i8 {{ func.return %x : i8 }}
    }}'''


def capture(lower=0,upper=5,step=1,carries=1,factor=2,bits=32):
    from merlin.transforms.pipeline import apply_common_pipeline,PipelineOptions
    from merlin.transforms.certificates import parse_ir
    result=apply_common_pipeline(parse_ir(source(lower,upper,step,carries)),PipelineOptions(
        index_bits=bits,normalize_shapes=False,simplify_integers=False,reuse_buffers=False,
        unroll_factors=(factor,)),plan_only=True)
    if len(result.changes)!=1 or result.changes[0].kind!='ordered_loop_unrolling':
        raise AssertionError('one actual registered unroll boundary required')
    return result.changes[0]


def mutate(raw,kind):
    from merlin.transforms.certificates import parse_ir,ir_text
    from xdsl.dialects import scf,arith,func,builtin
    from xdsl.rewriter import Rewriter
    m=parse_ir(raw);loops=[op for op in m.walk() if type(op) is scf.ForOp]
    bulk,tail=loops
    if kind=='missing_tail':
        Rewriter.replace_op(tail,[],list(tail.iter_args))
    elif kind=='cutoff':
        old=bulk.ub.owner
        replacement=arith.ConstantOp.from_int_and_width(old.value.value.data+1,builtin.IndexType())
        Rewriter.replace_op(old,replacement)
    elif kind=='tail_stride':
        tail.operands=(tail.lb,tail.ub,bulk.step,*tail.iter_args)
    elif kind=='tail_upper':
        tail.operands=(tail.lb,bulk.ub,tail.step,*tail.iter_args)
    elif kind=='lane_flags':
        addition=next(op for op in bulk.body.block.ops if type(op) is arith.AddiOp)
        addition.properties['overflowFlags']=arith.IntegerOverflowAttr([arith.IntegerOverflowFlag.NSW])
    elif kind=='lane_carry':
        call=next(op for op in bulk.body.block.ops if type(op) is func.CallOp)
        calls=[op for op in bulk.body.block.ops if type(op) is func.CallOp]
        calls[-1].operands=(calls[-1].operands[0],call.operands[1])
    elif kind=='tail_carry':
        tail.operands=(tail.lb,tail.ub,tail.step,*bulk.iter_args)
    elif kind=='return':
        main=next(op for op in m.body.block.ops if type(op) is func.FuncOp and op.sym_name.data=='main')
        terminal=main.body.block.last_op;terminal.operands=(main.body.block.args[0],)
    elif kind=='observer':
        call=next(op for op in m.walk() if type(op) is func.CallOp and op.callee.root_reference.data=='observe')
        call.properties['callee']=builtin.SymbolRefAttr('altered')
    elif kind=='tail_body':
        call=next(op for op in tail.body.block.ops if type(op) is func.CallOp)
        call.operands=(call.operands[0],bulk.iter_args[1] if len(bulk.iter_args)>1 else bulk.iter_args[0])
    elif kind=='dropped_function':
        other=next(op for op in m.body.block.ops if type(op) is func.FuncOp and op.sym_name.data=='untouched')
        m.body.block.erase_op(other)
    else:
        raise ValueError(kind)
    m.verify();return ir_text(m)


MUTATIONS=('missing_tail','cutoff','tail_stride','tail_upper','lane_flags','lane_carry','tail_carry','return','observer','tail_body','dropped_function')
CASES={'trip5_factor2':(0,5,1,1,2,32),'nonzero_nonunit_two':(3,18,3,2,2,64),
       'remainder_two':(1,17,2,1,3,32),'wide_index':(2147483650,2147483660,2,2,2,64)}


class Tail(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from merlin.verify import lean_graph_pipeline_tail as tail
        from merlin.verify import lean_graph_pipeline_tail_models as models
        cls.tail,cls.models=tail,models

    def prepared(self,case='trip5_factor2'):
        values=CASES[case];change=capture(*values)
        return change,self.tail.prepare(change.before_ir,change.after_ir,self.tail.Selection(values[4],values[5]))

    def test_actual_nonzero_bulk_tail_controls_signed_widths_carries_full_roster(self):
        from merlin.transforms import unrolling
        self.assertEqual(Path(unrolling.__file__).resolve(),REFERENCE/'src/merlin/transforms/unrolling.py')
        for case in CASES:
            with self.subTest(case=case):
                c,p=self.prepared(case);projection=json.loads(p.projection)
                self.assertEqual(len(projection['functions']),5)
                self.assertNotEqual(c.before_ir,c.after_ir)
                self.assertIn('Tail.ordered_bulk_tail',p.query)
                self.assertIn('Scope.check registry tailScopeL tailScopeR',p.query)
                self.assertIn('after_bulk',p.query)
                self.assertEqual(p.roots[-1],'merlin_unroll_tail_transform')
                self.assertEqual(p.describe()['outcome'],'UNOBSERVED')
                self.assertFalse(p.describe()['authority'])

    def test_valid_mutations_complete_module_refusal(self):
        change,_=self.prepared('nonzero_nonunit_two')
        for name in MUTATIONS:
            with self.subTest(name=name):
                bad=mutate(change.after_ir,name);self.assertNotEqual(bad,change.after_ir)
                with self.assertRaises(ValueError):
                    self.tail.prepare(change.before_ir,bad,self.tail.Selection(2,64))

    def test_domain_no_tail_inline_dynamic_negative_flags_refused(self):
        from merlin.transforms.certificates import parse_ir,ir_text
        from merlin.transforms.unrolling import unroll_ordered_loops
        for text in (source(0,4),source(0,2),source(-1,5),source().replace('%st=arith.constant 1 : index','%st=arith.constant 0 : index'),
                     source().replace('func.func @main(%a:i8, %b:i8)', 'func.func @main(%a:i8, %b:i8, %dynamic:index)').replace('%ub=arith.constant 5 : index','').replace('to %ub','to %dynamic')):
            with self.subTest(text=text):
                module=parse_ir(text);before=ir_text(module)
                unroll_ordered_loops(module,(2,),index_bits=32)
                with self.assertRaises(ValueError):self.tail.prepare(before,ir_text(module),self.tail.Selection(2,32))
        change,_=self.prepared()
        for selection in (self.tail.Selection(True,32),self.tail.Selection(2,16),self.tail.Selection(1,32)):
            with self.assertRaises(ValueError):self.tail.prepare(change.before_ir,change.after_ir,selection)

    def test_stale_source_profile_report_root_domain_authority_refused(self):
        change,p=self.prepared();selection=self.tail.Selection(2,32)
        request=self.tail.request(self.tail.PROFILE,change.before_ir,change.after_ir,selection)
        for key,value in (('profile','no_tail'),('domain','possible_success'),('final_root','rfl'),('support_roots',[])):
            bad=dict(request);bad[key]=value
            self.assertFalse(self.tail.matches(bad,self.tail.PROFILE,change.before_ir,change.after_ir,selection))
        with self.assertRaises(ValueError):self.tail.request('minimal7',change.before_ir,change.after_ir,selection)
        other=capture(1,6)
        with self.assertRaises(ValueError):self.tail.prepare(other.before_ir,change.after_ir,selection)
        self.assertFalse(self.tail.compiler_eligible(p));self.assertFalse(self.tail.compiler_eligible(request))
        with self.assertRaises(RuntimeError):self.tail.provider_arguments(request)

    def test_structural_actual_success_bulk_tail_failure_witness_candidates(self):
        _,p=self.prepared();query,roots,model=self.models.witness(p)
        self.assertIn('constant_carry_run',query)
        self.assertIn('fault_after_prefix',query)
        self.assertIn('source_tail_failure',query)
        self.assertFalse(model['source_success_assumed'])
        self.assertEqual(model['outcome'],'UNOBSERVED')
        self.assertEqual(len(roots),len(set(roots)))


def retain(destination):
    destination.mkdir(parents=True,exist_ok=False);Tail.setUpClass();test=Tail()
    for name,values in CASES.items():
        change,p=test.prepared(name);folder=destination/name;folder.mkdir()
        (folder/'before.mlir').write_text(change.before_ir);(folder/'after.mlir').write_text(change.after_ir)
        (folder/'CHANGE.json').write_text(json.dumps(asdict(change),indent=2)+'\n')
        (folder/'PREPARED.json').write_text(json.dumps(p.describe(),indent=2)+'\n')
        (folder/'REQUEST.json').write_text(json.dumps(test.tail.request(test.tail.PROFILE,change.before_ir,change.after_ir,test.tail.Selection(values[4],values[5])),indent=2)+'\n')
        (folder/'Query.lean').write_text(p.query)
        if name=='trip5_factor2':
            q,roots,model=test.models.witness(p)
            (folder/'Witness.lean').write_text(q)
            (folder/'WITNESS.json').write_text(json.dumps({**model,'roots':roots},indent=2)+'\n')
    change,_=test.prepared('nonzero_nonunit_two');folder=destination/'mutations';folder.mkdir()
    for name in MUTATIONS:
        bad=mutate(change.after_ir,name);(folder/(name+'.mlir')).write_text(bad)
        try:test.tail.prepare(change.before_ir,bad,test.tail.Selection(2,64))
        except ValueError as error:reason=str(error)
        else:raise AssertionError('mutation accepted')
        (folder/(name+'.json')).write_text(json.dumps({'host_refusal':reason,'parser_verify':True,
            'source':'../nonzero_nonunit_two/before.mlir','kernel_refusal_observed':False,'outcome':'UNOBSERVED'})+'\n')


if __name__=='__main__':
    PRIMARY=Path(sys.argv[1]).resolve();REFERENCE=Path(sys.argv[2]).resolve()
    for name,paths in (('merlin',[REFERENCE/'src/merlin',PRIMARY/'src/merlin']),
                       ('merlin.verify',[PRIMARY/'src/merlin/verify',REFERENCE/'src/merlin/verify']),
                       ('merlin.common',[PRIMARY/'src/merlin/common',REFERENCE/'src/merlin/common'])):
        module=types.ModuleType(name);module.__path__=[str(p) for p in paths];sys.modules[name]=module
    def held(event,args):
        if event in ('subprocess.Popen','os.system','os.exec','os.posix_spawn'):
            raise AssertionError('source-only tail cannot launch child processes')
        if event=='open' and isinstance(args[0],(str,bytes)):
            p=str(args[0])
            if ('/.elan/' in p or '/toolchain/' in p or 'libleanshared' in p or '/original-runtime-candidate' in p
                    or p.endswith(('.olean','.ilean'))):raise AssertionError('no runtime/object access')
    sys.addaudithook(held)
    args=sys.argv[3:]
    if args and args[0]=='--retain':retain(Path(args[1]));args=args[2:]
    unittest.main(argv=[sys.argv[0],*args])
