"""Cheap controls for Tiny block qualification and LOADMEM activation poisoning."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).absolute().parents[1] / 'tools'
for directory in (TOOLS, TOOLS / 'host_iss', TOOLS / 'chipyard'):
    sys.path.insert(0, str(directory))
import tiny_block


class TinyBlockTest(unittest.TestCase):
    def test_loadmem_poisons_only_activation_and_preserves_live_capture(self):
        activation = bytes(range(64))
        suffix = b'weights' + b'\xa5' * 256 + b'\xcd' * 64 + b'\x5a' * 256
        initial = activation + suffix
        preload = tiny_block.loadmem_image(initial, len(activation))
        self.assertEqual(preload, b'\x96' * len(activation) + suffix)
        self.assertEqual(initial, activation + suffix)
        self.assertEqual(tiny_block.loadmem_image(activation, len(activation)), b'\x96' * len(activation))

    def test_loadmem_rejects_invalid_activation_extent(self):
        for size in (True, False, 0, -1, 5, 1.0, None):
            with self.subTest(size=size), self.assertRaises(ValueError):
                tiny_block.loadmem_image(b'abcd', size)

    def test_build_rejects_unqualified_retained_and_recipe_inputs(self):
        retained_raw = b'{"outputs_sha256": {}}'
        for invalid in ('retained', 'recipe'):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                retained = root / 'retained'
                retained.mkdir()
                (retained / 'build.json').write_bytes(retained_raw)
                recipe = root / 'recipe.json'
                recipe.write_text('{}')
                args = SimpleNamespace(out=root / 'out', retained=retained, recipe=recipe, arena_base=None)
                retained_hash = (tiny_block.sha(retained_raw) if invalid == 'recipe'
                                 else tiny_block.RETAINED_SHA256)
                with (patch.object(tiny_block, 'RETAINED_SHA256', retained_hash),
                      patch.object(tiny_block, 'code_sources', return_value={}),
                      patch.object(tiny_block.subprocess, 'check_output', return_value='test\n'),
                      contextlib.redirect_stdout(io.StringIO()),
                      self.assertRaisesRegex(ValueError, invalid)):
                    tiny_block.build(args)
                receipt = json.loads((args.out / 'build.json').read_text())
                self.assertEqual(receipt['status'], 'FAIL_BUILD')
                self.assertIn('ValueError', receipt['error'])
                self.assertFalse((args.out / 'host.elf').exists())


if __name__ == '__main__':
    unittest.main()
