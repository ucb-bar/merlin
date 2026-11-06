"""Portable historical attention admission controls; original math is never rerun."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))


class FastIPTSourceIdentityTest(unittest.TestCase):
    def test_helper_is_byte_exact_admitted_source(self):
        helper = ROOT / 'tools/smol_capture/fast_ipt.py'
        self.assertEqual(hashlib.sha256(helper.read_bytes()).hexdigest(),
            '210610510f24e8d48aa6764224c6b2188cec48a2f76009147762510fa25c0ecc')


@unittest.skipUnless(os.environ.get('SMOL_ATTENTION_ADMISSION_ROOT'), 'Optional portable attention admission is not configured')
class FastAttentionAdmissionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from smol_capture import fast_attention
        cls.fast = fast_attention
        cls.root = fast_attention.original.safe_path(os.environ['SMOL_ATTENTION_ADMISSION_ROOT'])
        cls.admission = cls.fast.load_admission(data_root=cls.root)

    def temporary(self):
        root = self.fast.original.safe_path(os.environ['TMPDIR'])
        temporary = tempfile.TemporaryDirectory(prefix='attention-admission-', dir=root)
        self.addCleanup(temporary.cleanup)
        return self.fast.original.safe_path(temporary.name)

    def changed_admission(self, change):
        data = json.loads((self.root/'admission.json').read_bytes())
        change(data)
        path = self.temporary()/'admission.json'
        path.write_text(json.dumps(data))
        return path

    def portable_copy(self):
        root = self.temporary()
        catalogue = json.loads((self.root/'catalogue.json').read_bytes())
        entries = list(catalogue['files'].values())
        entries += [entry for row in catalogue['cases'] for entry in row['files'].values()]
        for entry in entries:
            source, _ = self.fast._archive_file(self.root, entry)
            target = self.fast.original.safe_path(root/entry['path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)
        (root/'catalogue.json').write_text(json.dumps(catalogue))
        return root, catalogue

    def test_complete_catalogue_and_exact_execution_ast(self):
        self.assertEqual((len(self.admission.cases), len(self.admission.kinds)), (25,17))
        archived = self.admission.historical_runner_path.read_bytes()
        current = Path(self.fast.__file__).read_bytes()
        names = ('kind','_execute_development')
        self.assertEqual(self.fast._ast_functions(archived,names),self.fast._ast_functions(current,names))

    def test_relocated_archive_loads_without_historical_imports(self):
        root,_=self.portable_copy()
        admission=self.fast.load_admission(data_root=root)
        self.assertEqual((len(admission.cases),len(admission.kinds)),(25,17))
        self.assertEqual(admission.data_root,root)
        self.assertNotIn('attention_fast_unit',sys.modules)
        self.assertNotIn('attention_unit',sys.modules)

    def test_changed_bookend_refused_after_load(self):
        root,_=self.portable_copy()
        admission=self.fast.load_admission(data_root=root)
        with (root/'catalogue.json').open('a') as stream:stream.write('\n')
        with self.assertRaisesRegex(ValueError,'bookend changed'):admission.validate_bookends()

    def test_helper_runner_and_status_refusals(self):
        for key,value in [('helper_sha256','0'*64),('runner_sha256','0'*64),('gate_of_record',True),
                          ('status','NOT_ADMITTED'),('all_full_arenas_exact',False)]:
            with self.subTest(key=key):
                path = self.changed_admission(lambda data:data.__setitem__(key,value))
                with self.assertRaises(ValueError):self.fast.load_admission(data_root=self.root,admission_path=path)

    def test_changed_execution_ast_even_with_refreshed_runner_hash(self):
        archived = self.admission.historical_runner_path.read_text()
        changed = archived.replace('timed=False','timed=True',1)
        self.assertNotEqual(changed,archived)
        root = self.temporary();runner=root/'runner.py';runner.write_text(changed)
        path=self.changed_admission(lambda data:data.__setitem__('runner_sha256',hashlib.sha256(changed.encode()).hexdigest()))
        with self.assertRaisesRegex(ValueError,'AST differs'):
            self.fast.load_admission(data_root=self.root,admission_path=path,historical_runner_path=runner)

    def test_corrupt_archived_payload_refused(self):
        root,catalogue=self.portable_copy()
        target=root/catalogue['cases'][0]['files']['expected-arena.bin']['path']
        data=bytearray(target.read_bytes());data[-1]^=1
        target.unlink();target.write_bytes(data)
        with self.assertRaisesRegex(ValueError,'payload hash mismatch'):self.fast.load_admission(data_root=root)

    def test_model_execution_port_ast_refused(self):
        root,catalogue=self.portable_copy()
        entry=catalogue['files']['model_runner'];target=root/entry['path']
        text=target.read_text().replace('max_instructions=5_000_000','max_instructions=5_000_001',1)
        target.unlink();target.write_text(text)
        entry.update(bytes=len(text.encode()),sha256=hashlib.sha256(text.encode()).hexdigest())
        (root/'catalogue.json').write_text(json.dumps(catalogue))
        with self.assertRaisesRegex(ValueError,'model execution AST differs'):self.fast.load_admission(data_root=root)

    def test_missing_kind_sample_refused(self):
        root,catalogue=self.portable_copy();catalogue['cases'].pop()
        (root/'catalogue.json').write_text(json.dumps(catalogue))
        with self.assertRaisesRegex(ValueError,'incomplete historical'):self.fast.load_admission(data_root=root)

    def test_current_generator_difference_refused(self):
        with mock.patch.object(self.fast,'assembly_v2',return_value='changed'):
            with self.assertRaisesRegex(ValueError,'recipe differs'):self.fast.load_admission(data_root=self.root)

    def test_native_and_dirty_model_refused(self):
        with mock.patch.dict(os.environ,{'NPU_MODEL_RTL_NATIVE_MXU0':'1'}):
            with self.assertRaisesRegex(ValueError,'native math'):self.fast.baseline_binding()
        check_output=self.fast.subprocess.check_output
        def output(args,**kwargs):
            return '?? injected\n' if 'status' in args else check_output(args,**kwargs)
        with mock.patch.object(self.fast.subprocess,'check_output',side_effect=output):
            with self.assertRaisesRegex(ValueError,'not clean'):self.fast.baseline_binding()

    def test_unadmitted_kind_never_executes(self):
        config=self.fast.AttentionV2Config(query_rows=96,keys=1024,query_heads=12,kv_heads=12,mask='none')
        with mock.patch.object(self.fast,'_execute_development') as execute:
            with self.assertRaisesRegex(ValueError,'not admitted'):
                self.fast.execute_admitted(config,{},b'',admission=self.admission)
            execute.assert_not_called()

    def test_factory_restored_after_execution_failure(self):
        before=self.fast.original.ArchState
        with mock.patch.object(self.fast.original,'execute',side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError,'injected'):
                self.fast._execute_development('',b'',0,0)
        self.assertIs(self.fast.original.ArchState,before)

    def test_fast_only_cached_smoke_and_explicit_provenance(self):
        report=self.fast.qualify_cached_case(self.admission,'unit-v2-smoke-smoke-0')
        self.assertTrue(report['full_arena_exact']);self.assertTrue(report['output_exact'])
        self.assertFalse(report['original_math_executed'])
        self.assertTrue(report['provenance']['math_override'])
        self.assertFalse(report['provenance']['gate_of_record'])


if __name__=='__main__':unittest.main()
