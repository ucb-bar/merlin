"""SDK-free bounded adapter checks using actual retained historical buffers.

No historical test suite, proof runner, receipt replay or qualification executes.
"""
from pathlib import Path
from dataclasses import asdict
import importlib.util
import json
import sys
import unittest
from unittest.mock import patch

PRIMARY = None
PIPELINE = "out/artifacts/verification/graph-state-pipeline/qualification-v1/suite/test_actual_complete_pipeline_0/actual"
INTEGER = "out/artifacts/verification/graph-state-integer/review/sealed-v2/qualification-v2/intake/chain-i32/producer/actual"


def buffers(kind):
    directory, count = (PIPELINE, 5) if kind == "pipeline" else (INTEGER, 6)
    return tuple((PRIMARY/directory/f"stage{i}.mlir").read_text() for i in range(count))


class Profiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from merlin.verify import lean_graph_pipeline_gate_profiles as profiles
        from merlin.verify import lean_graph_pipeline_gate_profile_prepare as preparation
        from merlin.verify import lean_graph_pipeline_gate_profile_requests as requests
        cls.profiles, cls.preparation, cls.requests = profiles, preparation, requests
        cls.settings = profiles.Settings(2, 3, 32)

    def test_actual_five_and_six_buffers_emit_frozen_complete_roots(self):
        from merlin.verify.lean_graph_state_pipeline_emit import emit as suffix_emit
        from merlin.verify.lean_graph_state_integer_emit import emit as prefix_emit
        for name, profile in (("pipeline", self.profiles.SCF), ("integer", self.profiles.INTEGER)):
            with self.subTest(profile=profile.name):
                settings=self.profiles.Settings(3,2,64) if name=='integer' else self.settings
                p = self.preparation.prepare(profile.name, buffers(name), settings)
                capture = json.loads(p.capture)
                frozen = prefix_emit(capture['prefix'], pipeline_capture=capture['pipeline']) if name=='integer' else suffix_emit(capture['pipeline'])
                self.assertTrue(p.query.startswith(frozen))
                self.assertEqual(len(p.buffers), len(profile.kinds)+1)
                self.assertTrue(all(p.buffers[i] != p.buffers[i+1] for i in range(len(profile.kinds))))
                self.assertEqual(tuple(k for k,_ in p.boundary_roots), profile.kinds)
                self.assertEqual(p.roots[-1], 'merlin_profile_transform')
                self.assertEqual(len(p.roots),len(set(p.roots)))
                self.assertNotIn('#print', p.query)
                self.assertEqual(p.describe()['outcome'],'UNOBSERVED')
                if name=='integer':
                    self.assertIn('PreservesEntriesCanonical',p.query)
                    self.assertIn('exactMiddle',p.query)
                    self.assertIn('index_model_typing',p.query)
                else:
                    self.assertIn('PreservesEntries',p.query)
                self.assertIn('Merlin.GraphGateProfileAdjacent.complete3',p.roots)
                request=self.requests.request(profile.name,buffers(name),settings)
                self.assertGreater(len(request.project_sources),7)
                self.assertFalse(request.describe()['native_authorization'])
                self.assertTrue(self.requests.matches(request.describe(),profile.name,buffers(name),settings))

    def test_stale_middle_observer_body_control_return_refuse(self):
        values=buffers('pipeline')
        variants=[]
        for index,old,new in ((2,'arith.addi','arith.subi'),(4,'@partial','@other'),
                               (3,'value = 2 : index','value = 4 : index'),
                               (4,'"func.return"(%out_3)','"func.return"(%x)')):
            text=values[index].replace(old,new)
            self.assertNotEqual(text,values[index])
            from merlin.transforms.certificates import parse_ir
            parse_ir(text).verify()
            variant=list(values);variant[index]=text;variants.append(tuple(variant))
        variants.append((values[0],values[1],values[1],values[3],values[4]))
        for variant in variants:
            with self.subTest(variant=hash(variant)),self.assertRaises(ValueError):
                self.preparation.prepare(self.profiles.SCF.name,variant,self.settings)

    def test_unsupported_settings_tail_dynamic_and_flags(self):
        values=buffers('pipeline')
        for settings in (self.profiles.Settings(1,3,32),self.profiles.Settings(2,1,32),
                         self.profiles.Settings(2,3,16),self.profiles.Settings(True,3,32)):
            with self.assertRaises(ValueError):self.preparation.prepare(self.profiles.SCF.name,values,settings)
        # Modify ALL buffers coherently so stale hashes alone cannot hide a
        # broader domain. Dynamic control comes from a real index argument.
        tail=tuple(s.replace('value = 24 : index','value = 25 : index') for s in values)
        with self.assertRaisesRegex(ValueError,'tail|bulk'):self.preparation.prepare(self.profiles.SCF.name,tail,self.settings)
        flagged=tuple(s.replace('#arith.overflow<none>','#arith.overflow<nsw>') for s in values)
        with self.assertRaisesRegex(ValueError,'flagged'):self.preparation.prepare(self.profiles.SCF.name,flagged,self.settings)
        from merlin.transforms.certificates import parse_ir,ir_text
        from xdsl.dialects import arith,builtin,func,scf
        dynamic=[]
        for text in values:
            module=parse_ir(text)
            main=next(op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data=='main')
            arg=main.body.block.insert_arg(builtin.IndexType(),2)
            sig=main.function_type
            main.properties['function_type']=builtin.FunctionType.from_lists([*sig.inputs,builtin.IndexType()],sig.outputs)
            loop=next(op for op in main.walk() if type(op) is scf.ForOp)
            operands=list(loop.operands);operands[1]=arg;loop.operands=operands
            module.verify();dynamic.append(ir_text(module))
        with self.assertRaisesRegex(ValueError,'two entry|static'):
            self.preparation.prepare(self.profiles.SCF.name,tuple(dynamic),self.settings)

    def test_wrong_profile_domain_roots_and_source_map_are_not_authority(self):
        values=buffers('pipeline');profile=self.profiles.SCF
        request=self.requests.request(profile.name,values,self.settings)
        for key,value in (('profile',self.profiles.INTEGER.name),('domain','possible_success'),
                          ('final_root','localLemma'),('support_roots',[]),('project_sources',[]),
                          ('boundary_roots',request.boundary_roots[-1:])):
            forged=request.describe();forged[key]=value
            self.assertFalse(self.requests.matches(forged,profile.name,values,self.settings))
        with self.assertRaises(ValueError):self.preparation.prepare(self.profiles.INTEGER.name,values,self.settings)
        with self.assertRaises(ValueError):self.requests.request('minimal7',values,self.settings)
        self.assertFalse(self.profiles.compiler_eligible(request))
        self.assertFalse(self.profiles.compiler_eligible(request.describe()))
        with self.assertRaises(RuntimeError):self.profiles.public_check(request)
        with self.assertRaises(RuntimeError):self.requests.provider_arguments(request)

    def test_old_guards_runners_and_minimal_provider_are_never_called(self):
        from merlin.verify import lean_graph_state_scalar_kernel as kernel
        from merlin.verify import lean_graph_state as stable
        from merlin.verify import lean_graph_pipeline_gate_provider as minimal
        with patch.object(kernel,'prepare',side_effect=AssertionError('no old runtime prepare')),patch.object(
                kernel,'guard',side_effect=AssertionError('no old guard')),patch.object(
                stable,'prepare',side_effect=AssertionError('no runtime selection')),patch.object(
                minimal,'selected',side_effect=AssertionError('no minimal provider union')):
            self.requests.request(self.profiles.SCF.name,buffers('pipeline'),self.settings)


def retain(destination):
    destination.mkdir(parents=True,exist_ok=False)
    Profiles.setUpClass()
    for name,profile in (('pipeline',Profiles.profiles.SCF),('integer',Profiles.profiles.INTEGER)):
        path=destination/name;path.mkdir()
        values=buffers(name)
        settings=Profiles.profiles.Settings(3,2,64) if name=='integer' else Profiles.settings
        p=Profiles.preparation.prepare(profile.name,values,settings)
        r=Profiles.requests.request(profile.name,values,settings)
        for i,text in enumerate(values):(path/f'stage{i}.mlir').write_text(text)
        (path/'PREPARED.json').write_text(json.dumps(p.describe(),indent=2)+'\n')
        (path/'REQUEST.json').write_text(json.dumps(r.describe(),indent=2)+'\n')
        (path/'Query.lean').write_text(p.query)
        (path/'PROVENANCE.json').write_text(json.dumps({'retained_original_directory':PIPELINE if name=='pipeline' else INTEGER,
            'original_population_member':'i32' if name=='pipeline' else 'chain-i32',
            'historical_qualification_not_transferred':True,'source_buffers_exact':True},indent=2)+'\n')


if __name__=='__main__':
    PRIMARY=Path(sys.argv[1]).resolve();producer=Path(sys.argv[2]).resolve()
    spec=importlib.util.spec_from_file_location('gate_source_tests',Path(__file__).with_name('test_lean_graph_pipeline_gate.py'))
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    old.isolated_owners(PRIMARY,producer)
    # Explicit source ownership: the frozen pure emitter's historical import
    # references primary common.run_owned, absent at the old producer revision.
    # Select the actual primary common source; do not supply a shim function.
    import types
    common=types.ModuleType('merlin.common')
    common.__path__=[str(PRIMARY/'src/merlin/common'),str(producer/'src/merlin/common')]
    sys.modules['merlin.common']=common
    def held(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            name=str(args[0])
            if ('/original-runtime-candidate' in name or '/toolchain/' in name or name.endswith(('.olean','.ilean'))):
                raise AssertionError('no selected/archived runtime objects in source-only profile tests')
    sys.addaudithook(held)
    args=sys.argv[3:]
    if args and args[0]=='--retain':retain(Path(args[1]));args=args[2:]
    unittest.main(argv=[sys.argv[0],*args])
