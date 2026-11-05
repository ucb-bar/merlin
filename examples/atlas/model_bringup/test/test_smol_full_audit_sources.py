"""Static auditor pins optional recipes before their validation executes."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/host_iss')]
import check_smol_full_bundle as audit
import kernel_library


class AuditSourcesTest(unittest.TestCase):
    def run_audit(self, mode, drift):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ('smol_fusion_assets.py', 'smol_fusion_program.py', 'smol_fusion_fp8.py')
            for name in (*names, 'smol_full_plan.py'):
                path = root / kernel_library.relative_path(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# initial\n')
            bundle_path = root / 'bundle'
            bundle_path.mkdir()
            (bundle_path / 'provenance.json').write_text(json.dumps(dict(fusion_set=mode)))
            receipt = {}

            class Bundle:
                sums = {'host/host.elf': 'elf'}
                checksums_sha256 = 'sums'
                provenance = {'contract_sha256': 'contract'}
                kernel_count = 2915
                _smol_verified_programs = []

                def __init__(self, *args):
                    if drift:
                        (root / kernel_library.relative_path(drift)).write_text('# changed during bundle validation\n')

                def read(self, name): return b'elf'
                def verify_bookends(self): pass

            with patch.object(audit, 'ROOT', root), patch.object(audit, 'FullBundle', Bundle), \
                 patch.object(audit, 'code_sources', return_value={}), \
                 patch.object(audit, 'configure_model', return_value=(None, None, None, None)), \
                 patch.object(audit, 'model_binding', return_value={'pinned': True}), \
                 patch.object(audit, 'Image', return_value=None), \
                 patch.object(audit.smol_full_contract, 'validate_bundle', return_value={'projection_calls': 1421}), \
                 patch.object(audit, 'write_json', side_effect=lambda path, value: receipt.update(value)), \
                 patch.object(sys, 'argv', ['audit', '--bundle', str(bundle_path), '--model-root', str(root),
                                          '--out', str(root / 'receipt.json')]):
                status = audit.main()
            return status, receipt, {str(root / kernel_library.relative_path(name)) for name in names}

    def test_all_fused_recipes_are_entry_and_end_pinned(self):
        status, receipt, names = self.run_audit('fc1-bias-gelu-fp8', None)
        self.assertEqual(status, 0)
        self.assertTrue(names <= receipt['source_sha256'].keys())
        for name in names:
            status, receipt, _ = self.run_audit('fc1-bias-gelu-fp8', Path(name).name)
            self.assertEqual(status, 1)
            self.assertIn('source bookends changed', receipt['error']['message'])

    def test_none_profile_keeps_optional_recipes_out_of_pins(self):
        status, receipt, names = self.run_audit('none', 'smol_fusion_fp8.py')
        self.assertEqual(status, 0)
        self.assertTrue(names.isdisjoint(receipt['source_sha256']))


if __name__ == '__main__':
    unittest.main()
