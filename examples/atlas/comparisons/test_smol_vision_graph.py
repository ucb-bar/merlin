"""Product vision graph layout controls, without executing attention or kernels."""
import ctypes
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from source_relocation import physical_path as safe


@unittest.skipUnless(os.environ.get('SMOL_ATTENTION_HEADERS'), 'Optional attention headers are not configured')
class VisionLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        here = safe(__file__).parent
        header = safe(os.environ['SMOL_ATTENTION_HEADERS'])
        temporary = Path(tempfile.mkdtemp(prefix='vision-layout-'))
        library = temporary / 'layout.so'
        subprocess.run(['/usr/bin/cc', '-O2', '-fPIC', '-shared', '-ffp-contract=off',
                        '-fno-fast-math', '-I'+str(header), str(here / 'smol_vision_graph.c'),
                        '-o', str(library)], check=True)
        # Only the two dependency-free layout entry points execute. The graph's
        # numerical dependencies stay unresolved and cannot qualify graph parity.
        cls.library = ctypes.CDLL(str(library), mode=os.RTLD_LAZY)
        cls.pointer = ctypes.POINTER(ctypes.c_uint16)
        for name in ('smol_vision_patch16', 'smol_vision_shuffle4'):
            getattr(cls.library, name).argtypes = [cls.pointer, cls.pointer]

    def check(self, name, incoming, expected):
        output = np.empty_like(expected)
        code = getattr(self.library, name)(incoming.ctypes.data_as(self.pointer), output.ctypes.data_as(self.pointer))
        self.assertEqual(code, 0)
        np.testing.assert_array_equal(output, expected)
        self.assertEqual(getattr(self.library, name)(incoming.ctypes.data_as(self.pointer), incoming.ctypes.data_as(self.pointer)), 1)

    def test_patch_channel_then_row_then_column(self):
        image = (np.arange(3*512*512, dtype=np.uint32) % 65536).astype(np.uint16).reshape(3, 512, 512)
        # Native Conv2d flattening CR/S at every actual16x16 patch.
        expected = torch.from_numpy(image.astype(np.int32)).unfold(1, 16, 16).unfold(2, 16, 16)
        expected = expected.permute(1, 2, 0, 3, 4).contiguous().numpy().astype(np.uint16).reshape(1024, 768)
        self.check('smol_vision_patch16', image, expected)

    def test_connector_source_pixel_shuffle_four(self):
        words = (np.arange(1024*768, dtype=np.uint32) % 65536).astype(np.uint16).reshape(1, 1024, 768)
        source = torch.from_numpy(words.astype(np.int32))
        source = source.view(1, 32, 32, 768).view(1, 32, 8, 3072)
        source = source.permute(0, 2, 1, 3).reshape(1, 8, 8, 12288)
        expected = source.permute(0, 2, 1, 3).reshape(1, 64, 12288).numpy().astype(np.uint16)
        self.check('smol_vision_shuffle4', words, expected)


if __name__ == '__main__':
    unittest.main()
