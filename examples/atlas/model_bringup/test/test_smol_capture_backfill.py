"""Resume admits only current, exact receipts with successful worker exits."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / 'tools'))
import smol_capture_audit as audit
import smol_capture_backfill as backfill
from types import SimpleNamespace


class BackfillTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.current = audit.Bindings(dict(commit=audit.PIN, numerics='rtl', root='/original'), 'runner', 'harness',
            {'/synthetic/stats.py':'stats-a','/synthetic/smol_capture_audit.py':'audit-a','/synthetic/__init__.py':'init-a'})
        def blob(name, raw):
            path = self.root / name
            path.write_bytes(raw)
            return dict(path=str(path), bytes=len(raw), sha256=audit.digest(raw))
        initial, output = b'abcd', b'xy'
        canonical = struct.pack('<I', 0x73)
        self.event = dict(index=0, kind='attention', attention_index=0, sequence=1, initial=blob('initial', initial),
                          output=blob('output', output), program=blob('program', canonical),
                          runtime_program=blob('runtime', audit.EPILOGUE), source=blob('source', b'ecall'),
                          runtime_source=blob('runtime-source', b'epilogue'), output_offset=1, output_bytes=2,
                          runtime_epilogue_sha256=audit.digest(audit.EPILOGUE),
                          expected_arena_sha256=audit.digest(b'axyd'),
                          requested_dma=dict(load_bytes=2, store_bytes=2), observed_dma=dict(load_bytes=2, store_bytes=2))
        self.frozen = json.dumps(self.event).encode()
        execution = dict(program_sha256=audit.digest(audit.EPILOGUE), program_words=6, instructions=6,
                         halt_pc=5, observed=dict(load_bytes=2, store_bytes=2))
        self.saved = dict(status='PASS', index=0, kind='attention', event_sha256=audit.digest(self.frozen),
                          runner_sha256='runner', harness_sha256='harness', model=self.current.model,
                           source_binding_schema=self.current.source_binding_schema,
                           helper_source_sha256=copy.deepcopy(self.current.helper_source_sha256),
                          original_model=True, native_backend=False, math_override=False, initial_csrs={'0xc11': 1},
                          full_arena_exact=True, output_exact=True, initial_sha256=audit.digest(initial),
                          expected_output_sha256=audit.digest(output), expected_arena_sha256=audit.digest(b'axyd'),
                          actual_arena_sha256=audit.digest(b'axyd'), runtime_program_executed=True,
                          canonical_program_sha256=audit.digest(canonical), program_sha256=audit.digest(audit.EPILOGUE),
                          execution=execution)
        self.receipt, self.completion = self.root / 'receipt.json', self.root / 'worker.json'
        self.write()

    def write(self, returncode=0):
        self.receipt.write_text(json.dumps(self.saved))
        result = dict(returncode=returncode, index=0, event_sha256=audit.digest(self.frozen),
                      receipt_sha256=audit.digest(self.receipt.read_bytes()))
        self.completion.write_text(json.dumps(result))

    def validate(self):
        return backfill.validate_completed(self.event, self.frozen, self.receipt, self.completion, self.current)

    def test_current_receipt_passes(self):
        self.assertEqual(self.validate(), self.saved)

    def test_fp8_statistics_need_independent_original_observation(self):
        self.event['fp8_stats']=dict(scale_exponent=-5,physical_values=1024,preclip_saturation_count=3)
        self.frozen=json.dumps(self.event).encode()
        self.saved['event_sha256']=audit.digest(self.frozen)
        self.write()
        with self.assertRaisesRegex(ValueError,'FP8 observation'):
            self.validate()
        self.saved['fp8_stats']=copy.deepcopy(self.event['fp8_stats'])
        self.write()
        self.assertEqual(self.validate(),self.saved)
        self.saved['fp8_stats']['preclip_saturation_count']=2
        self.write()
        with self.assertRaisesRegex(ValueError,'FP8 observation'):
            self.validate()

    def test_typed_fp8_requires_observation_even_when_both_records_omit_it(self):
        self.event.update(output_dtype='fp8',output_scale_exponent=-5)
        self.frozen=json.dumps(self.event).encode()
        self.saved['event_sha256']=audit.digest(self.frozen)
        self.write()
        with self.assertRaisesRegex(ValueError,'required FP8 observation'):
            self.validate()
        with self.assertRaisesRegex(ValueError,'required FP8 observation'):
            backfill.fp8_observation(self.event)

    def fp8_projection(self, with_stats):
        self.event.update(kind='projection',projection_index=0)
        self.saved['kind']='projection'
        if with_stats:
            self.event['fp8_stats']=dict(scale_exponent=-5,physical_values=2,preclip_saturation_count=1)
            self.saved['fp8_stats']=copy.deepcopy(self.event['fp8_stats'])
        return dict(projection_reference=dict(kernels=[dict(fusion=dict(
            output_dtype='fp8',output_scale_exponent=-5,output_bytes=2))]))

    def test_complete_manifest_rejects_dropped_type_and_observation(self):
        update=self.fp8_projection(with_stats=False)
        self.assertNotIn('output_dtype',self.event)
        self.assertNotIn('fp8_stats',self.event)
        with self.assertRaisesRegex(ValueError,'required manifest FP8 observation'):
            self.run_coordinator(0,manifest_update=update)
        report=json.loads((self.root/'backfill/gate.json').read_text())
        self.assertEqual(report['status'],'INCOMPLETE')

    def test_complete_manifest_accepts_legacy_event_with_observation(self):
        update=self.fp8_projection(with_stats=True)
        self.assertNotIn('output_dtype',self.event)
        code,report,invocations=self.run_coordinator(0,manifest_update=update)
        self.assertEqual(code,0)
        self.assertEqual(report['status'],'PASS_AUDITED_ORIGINAL_ALL_LAUNCHES')
        self.assertEqual(invocations,[])

    def test_manifest_checks_scale_extent_and_unexpected_fp8_type(self):
        manifest=self.fp8_projection(with_stats=True)
        for key,value in (('scale_exponent',-4),('physical_values',1)):
            changed=copy.deepcopy(self.event);changed['fp8_stats'][key]=value
            with self.assertRaisesRegex(ValueError,'required manifest FP8 observation'):
                backfill.validate_projection_observations([changed],manifest)
        ordinary=dict(projection_reference=dict(kernels=[{}]))
        with self.assertRaisesRegex(ValueError,'unexpected FP8 observation'):
            backfill.validate_projection_observations([self.event],ordinary)

    def test_stale_sources_or_model_rejected(self):
        for key, value in [('runner_sha256', 'old'), ('harness_sha256', 'old'),
                           ('model', dict(commit='old', numerics='rtl', root='/original')),
                           ('native_backend', True), ('math_override', True)]:
            with self.subTest(key=key):
                original = copy.deepcopy(self.saved)
                self.saved[key] = value
                self.write()
                with self.assertRaises(ValueError):
                    self.validate()
                self.saved = original

    def test_helper_only_drift_and_legacy_source_schema_rejected(self):
        for name in self.current.helper_source_sha256:
            original=copy.deepcopy(self.saved)
            self.saved['helper_source_sha256'][name]='changed-helper'
            self.write()
            with self.assertRaisesRegex(ValueError,'gate helper sources changed'):
                self.validate()
            self.saved=original
        for field in ('source_binding_schema','helper_source_sha256'):
            original=copy.deepcopy(self.saved);self.saved.pop(field);self.write()
            with self.assertRaisesRegex(ValueError,'gate helper sources changed'):
                self.validate()
            self.saved=original

    def test_helper_origin_and_hash_bind_actual_imports(self):
        runner=Path(backfill.__file__)
        sources=audit.helper_sources(runner)
        hashes=audit.helper_bindings(runner)
        self.assertEqual(set(hashes),{str(path)for path in sources.values()})
        actual_read=Path.read_bytes
        for target in sources.values():
            def altered(path):
                raw=actual_read(path)
                return raw+b'\n# helper drift\n' if path==target else raw
            with patch.object(Path,'read_bytes',altered):
                changed=audit.helper_bindings(runner)
            self.assertNotEqual(changed,hashes)
            self.assertEqual({name for name in hashes if hashes[name]!=changed[name]},{str(target)})
        module=audit.importlib.import_module('smol_capture.stats')
        with patch.object(module,'__file__','/synthetic/foreign-stats.py'):
            with self.assertRaisesRegex(ValueError,'outside selected source'):
                audit.helper_bindings(runner)

    def test_worker_binds_helpers_before_and_after_execution(self):
        event_path=self.root/'worker-event.json';event_path.write_bytes(self.frozen)
        args=SimpleNamespace(model_root=Path('/original'),harness=Path(backfill.__file__))
        execution=copy.deepcopy(self.saved['execution'])
        result=SimpleNamespace(program_sha256=execution['program_sha256'],arena=b'axyd',output=b'xy',
            observed=execution['observed'],metadata=lambda:execution)
        original_functional=lambda *args:None
        unit=SimpleNamespace(model_binding=lambda:self.current.model,_functional=original_functional,
            execute=lambda *args,**kwargs:result)
        spec=SimpleNamespace(name='synthetic_original_gate_unit',loader=SimpleNamespace(exec_module=lambda unit:None))
        for drift in (False,True):
            with self.subTest(drift=drift):
                after=replace(self.current,helper_source_sha256=dict(self.current.helper_source_sha256))
                if drift:after.helper_source_sha256['/synthetic/stats.py']='changed-at-worker-end'
                out=self.root/('drift-result.json'if drift else'stable-result.json')
                with patch.object(backfill.importlib.util,'spec_from_file_location',return_value=spec), \
                     patch.object(backfill.importlib.util,'module_from_spec',return_value=unit), \
                     patch.object(backfill.subprocess,'check_output',return_value=''), \
                     patch.object(backfill,'bindings',side_effect=[self.current,after]) as bind, \
                     patch.dict(sys.modules,{spec.name:unit}):
                    self.assertEqual(backfill.worker(event_path,out,args),not drift)
                self.assertEqual(bind.call_count,2)
                saved=json.loads(out.read_text())
                self.assertEqual(saved['helper_source_sha256'],self.current.helper_source_sha256)
                self.assertEqual(saved['source_binding_schema'],audit.SOURCE_SCHEMA)
                self.assertIs(unit._functional,original_functional)
                if drift:self.assertIn('source bindings changed during execution',saved['error']['message'])

    def test_worker_exit_required(self):
        for code in (1, -9, False, None):
            with self.subTest(code=code):
                self.write(code)
                with self.assertRaises(ValueError):
                    self.validate()
        self.completion.unlink()
        with self.assertRaises(FileNotFoundError):
            self.validate()

    def test_receipt_and_completion_hashes_bound(self):
        self.receipt.write_text(self.receipt.read_text() + ' ')
        with self.assertRaises(ValueError):
            self.validate()

    def test_full_arena_and_dma_rechecked(self):
        for field, value in [('actual_arena_sha256', 'wrong'), ('runtime_program_executed', False),
                             ('full_arena_exact', False)]:
            with self.subTest(field=field):
                original = copy.deepcopy(self.saved)
                self.saved[field] = value
                self.write()
                with self.assertRaises(ValueError):
                    self.validate()
                self.saved = original
        self.event['observed_dma']['load_bytes'] = 3
        self.frozen = json.dumps(self.event).encode()
        self.saved['event_sha256'] = audit.digest(self.frozen)
        self.write()
        with self.assertRaises(ValueError):
            self.validate()

    def run_coordinator(self, returncode, stale=False, manifest_update=None, final_bindings=None):
        capture = self.root / 'capture'
        capture.mkdir()
        ledger = capture / 'device-events.jsonl'
        ledger.write_text(json.dumps(self.event) + '\n')
        manifest_path = capture / 'manifest.json'
        manifest_path.write_text(json.dumps(dict(status='COMPLETE_REFERENCE_GENERATION',
                                                errors=[], device_events=[self.event])))
        destination = self.root / 'backfill'
        destination.mkdir()
        frozen = (json.dumps(self.event, indent=2) + '\n').encode()
        (destination / 'event-0000.json').write_bytes(frozen)
        self.saved['event_sha256'] = audit.digest(frozen)
        self.receipt = destination / 'launch-0000.json'
        self.completion = destination / 'worker-0000.json'
        self.frozen = frozen
        if stale:
            self.saved['runner_sha256'] = 'stale'
        self.write()
        invocations = []
        def execute(command, **kwargs):
            invocations.append(command)
            self.assertFalse(self.receipt.exists(), 'stale PASS must be removed before dispatch')
            self.saved['runner_sha256'] = self.current.runner_sha256
            self.receipt.write_text(json.dumps(self.saved))
            return SimpleNamespace(returncode=returncode)
        argv = ['backfill', '--ledger', str(ledger), '--out', str(destination),
                '--model-root', str(self.root), '--harness', __file__, '--workers', '1', '--expected-launches', '1']
        read_bytes = Path.read_bytes
        manifest_reads = 0
        binding_reads = 0
        def binding(*args):
            nonlocal binding_reads
            binding_reads+=1
            return final_bindings if final_bindings is not None and binding_reads>=4 else self.current
        def read(path):
            nonlocal manifest_reads
            if path == manifest_path:
                manifest_reads += 1
                if manifest_reads == 2 and manifest_update is not None:
                    manifest = json.loads(read_bytes(path))
                    manifest.update(manifest_update)
                    path.write_text(json.dumps(manifest))
            return read_bytes(path)
        with patch.object(sys, 'argv', argv), patch.object(backfill, 'bindings', side_effect=binding), \
                patch.object(backfill.subprocess, 'run', side_effect=execute), patch.object(Path, 'read_bytes', read):
            code = backfill.main()
        report = json.loads((destination / 'gate.json').read_text())
        self.assertEqual(report['manifest_sha256'], audit.digest(manifest_path.read_bytes()))
        return code, report, invocations

    def test_coordinator_reuses_only_strict_pass(self):
        code, report, invocations = self.run_coordinator(0)
        self.assertEqual(code, 0)
        self.assertEqual(invocations, [])
        self.assertEqual(report['validated'], 1)

    def test_helper_only_drift_at_final_aggregation_invalidates_pass(self):
        changed=replace(self.current,helper_source_sha256=dict(self.current.helper_source_sha256))
        changed.helper_source_sha256['/synthetic/stats.py']='changed-during-run'
        with self.assertRaisesRegex(ValueError,'source bindings changed during backfill'):
            self.run_coordinator(0,final_bindings=changed)
        report=json.loads((self.root/'backfill/gate.json').read_text())
        self.assertEqual(report['status'],'INCOMPLETE')

    def test_coordinator_invalidates_stale_pass(self):
        code, report, invocations = self.run_coordinator(0, stale=True)
        self.assertEqual(code, 0)
        self.assertEqual(len(invocations), 1)
        self.assertEqual(report['validated'], 1)

    def test_coordinator_rejects_pass_from_failed_worker(self):
        code, report, invocations = self.run_coordinator(9, stale=True)
        self.assertEqual(code, 1)
        self.assertEqual(len(invocations), 1)
        self.assertEqual(report['status'], 'INCOMPLETE')
        self.assertEqual(report['validated'], 0)

    def test_final_manifest_incomplete_rejects_current_receipt(self):
        code, report, invocations = self.run_coordinator(0, manifest_update=dict(
            status='INCOMPLETE', errors=['capture invalidated before final aggregation']))
        self.assertEqual(code, 1)
        self.assertEqual(report['status'], 'INCOMPLETE')
        self.assertEqual(report['validated'], 1)
        self.assertEqual(invocations, [])

    def test_final_manifest_errors_reject_current_receipt(self):
        code, report, invocations = self.run_coordinator(0, manifest_update=dict(errors=['late capture error']))
        self.assertEqual(code, 1)
        self.assertEqual(report['status'], 'INCOMPLETE')
        self.assertEqual(report['validated'], 1)
        self.assertEqual(invocations, [])

    def test_final_manifest_inventory_rechecked(self):
        with self.assertRaisesRegex(ValueError, 'manifest and ledger differ'):
            self.run_coordinator(0, manifest_update=dict(device_events=[]))
        report = json.loads((self.root / 'backfill/gate.json').read_text())
        self.assertEqual(report['status'], 'INCOMPLETE')

    def test_final_manifest_byte_bookend_preserved(self):
        read_bytes = Path.read_bytes
        manifest_reads = 0
        def read(path):
            nonlocal manifest_reads
            if path.name == 'manifest.json':
                manifest_reads += 1
                if manifest_reads == 3:
                    path.write_bytes(read_bytes(path) + b' ')
            return read_bytes(path)
        with patch.object(Path, 'read_bytes', read):
            with self.assertRaisesRegex(ValueError, 'capture changed during final aggregation'):
                self.run_coordinator(0)
        report = json.loads((self.root / 'backfill/gate.json').read_text())
        self.assertEqual(report['status'], 'INCOMPLETE')

    def test_coordinator_clears_pass_before_entry_binding_failure(self):
        destination = self.root / 'backfill'
        destination.mkdir()
        gate = destination / 'gate.json'
        argv = ['backfill', '--ledger', str(self.root / 'device-events.jsonl'), '--out', str(destination),
                '--model-root', str(self.root), '--harness', __file__]
        for outputs, message in [(['bad', ''], 'original model revision changed'),
                                 ([audit.PIN, ' M model.py'], 'original model is dirty')]:
            with self.subTest(outputs=outputs):
                gate.write_text(json.dumps(dict(status='PASS_AUDITED_ORIGINAL_ALL_LAUNCHES')))
                with patch.object(sys, 'argv', argv), \
                        patch.object(audit.subprocess, 'check_output', side_effect=outputs):
                    self.assertEqual(backfill.main(), 1)
                report = json.loads(gate.read_text())
                self.assertEqual(report['status'], 'INCOMPLETE')
                self.assertEqual(report['failures'][0]['type'], 'ValueError')
                self.assertEqual(report['failures'][0]['error'], message)

    def test_model_revision_and_clean_tree_checked(self):
        with patch.object(audit.subprocess, 'check_output', side_effect=['bad', '']):
            with self.assertRaises(ValueError):
                audit.bindings(self.root, __file__, __file__)
        with patch.object(audit.subprocess, 'check_output', side_effect=[audit.PIN, ' M model.py']):
            with self.assertRaises(ValueError):
                audit.bindings(self.root, __file__, __file__)
        with patch.object(audit.subprocess, 'check_output', side_effect=[audit.PIN, '']):
            current=audit.bindings(self.root, backfill.__file__, backfill.__file__)
            self.assertEqual(current.model['numerics'], 'rtl')
            self.assertEqual(len(current.helper_source_sha256),3)


if __name__ == '__main__':
    unittest.main()
