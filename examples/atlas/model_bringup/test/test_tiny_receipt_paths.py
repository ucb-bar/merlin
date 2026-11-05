"""Portable receipt pins preserve hashes and reject traversal before filesystem access."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'tools/tiny_full')]
from receipt_paths import decode, encode


class ReceiptPaths(unittest.TestCase):
    def test_roundtrip_keys_values_and_nested_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roots = dict(data=root, atlas=root/'compiler', model=root/'model')
            original = {str(root/'compiler/helper.py'): 'a'*64, 'model_root': str(root/'model'),
                        'launches': [{'record': str(root/'capture/record.json')}]}
            portable = encode(original, roots)
            self.assertIn('@atlas/helper.py', portable)
            self.assertEqual(portable['model_root'], '@model/.')
            self.assertEqual(decode(portable, roots), original)

    def test_unknown_root_and_traversal_rejected(self):
        for text in ('@unknown/file', '@data/../escape', '@data//absolute'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                decode({'pin': text}, dict(data=ROOT))

    def test_legacy_relocation_preserves_digest(self):
        result = decode({'/old/helper.py': 'b'*64}, {}, [(Path('/old'), ROOT)])
        self.assertEqual(result, {str(ROOT/'helper.py'): 'b'*64})


if __name__ == '__main__':
    unittest.main()
