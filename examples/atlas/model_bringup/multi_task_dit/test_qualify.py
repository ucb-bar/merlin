"""Qualification accounting and authentication controls, without simulator dependencies."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).with_name('qualify.py')
spec = importlib.util.spec_from_file_location('mtdit_qualify', MODULE)
qualify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualify)


class QualificationTest(unittest.TestCase):
    def test_panels_retain_every_logical_element_and_count(self):
        record = dict(m=197, k=770, n=513, count=12)
        kinds = qualify.projection_kinds(record)
        self.assertEqual(len(kinds), 4)
        self.assertEqual(sum(p['count'] for p in kinds), 120)
        self.assertTrue(all(p['k'] == 800 and p['logical_k'] == 770 for p in kinds))
        self.assertEqual(sum(p['rows'] * p['logical_n'] * p['count'] for p in kinds), 197*513*12)
        self.assertEqual({(p['m'], p['n']) for p in kinds}, {(128,128),(128,32),(96,128),(96,32)})
        tails = [p for p in kinds if p['rows']==69 and p['logical_n']==1]
        self.assertEqual((tails[0]['row_start'],tails[0]['column_start']), (128,512))

    def test_stage_mutation_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            (folder/'program.S').write_text('ecall\n')
            (folder/'arena.bin').write_bytes(b'input')
            spec=dict(folder=str(folder),golden_sha256=None,identity=dict(source_sha256=qualify.sha(b'ecall\n'),arena_sha256=qualify.sha(b'input')))
            self.assertEqual(qualify.source_and_arena(spec)[2],b'input')
            (folder/'arena.bin').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'staged program or arena'):
                qualify.source_and_arena(spec)

    def test_receipt_rejects_changed_spec_and_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'spec.json';path.write_text('{}')
            receipt=dict(status='PASS',spec_sha256=qualify.sha(path.read_bytes()),runner_sha256=qualify.sha(MODULE.read_bytes()))
            self.assertTrue(qualify.receipt_matches(receipt,path))
            self.assertFalse(qualify.receipt_matches(dict(receipt,runner_sha256='changed'),path))
            path.write_text('{"changed":true}')
            self.assertFalse(qualify.receipt_matches(receipt,path))

    def test_pending_receipts_never_contribute_cycles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            spec=dict(folder=str(root/'launch'),label='launch',family='projection_gemm',params={},count=10,logical_source={},identity={})
            qualify.save(root/'campaign.json',dict(specs=[spec],pending=[]))
            from types import SimpleNamespace
            self.assertEqual(qualify.report(SimpleNamespace(out=root)),1)
            report=json.loads((root/'qualification.json').read_text())
            self.assertEqual((report['qualified_kinds'],report['count_weighted_cycles']), (0,0))
            self.assertEqual(report['rows'][0]['rtl']['status'],'PENDING')


if __name__=='__main__':unittest.main()
