"""Product FP32 policy host math against native operator output words."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from source_relocation import physical_path as safe


class PolicyOpsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        cls.temporary = Path(tempfile.mkdtemp(prefix='policy-ops-'))
        here = Path(__file__).absolute().parent
        library = cls.temporary / 'policy.so'
        subprocess.run(['/usr/bin/cc', '-O2', '-fPIC', '-shared', '-ffp-contract=off', '-fno-fast-math',
                        str(here / 'smol_policy_ops.c'), '-lm', '-o', str(library)], check=True)
        cls.library = ctypes.CDLL(str(library))
        cls.pointer = ctypes.POINTER(ctypes.c_float)
        cls.library.smol_policy_silu_f32.argtypes = [cls.pointer, cls.pointer, ctypes.c_size_t]
        cls.word_pointer = ctypes.POINTER(ctypes.c_uint16)
        cls.library.smol_policy_cross_attention_bf16.argtypes = [cls.word_pointer]*3 + [ctypes.POINTER(ctypes.c_uint8)] + [cls.pointer]*3 + [cls.word_pointer]

    def check(self, incoming, expected):
        a = incoming.detach().numpy().copy()
        b = expected.detach().numpy().copy()
        output = np.empty_like(a)
        code = self.library.smol_policy_silu_f32(a.ctypes.data_as(self.pointer),
                                                output.ctypes.data_as(self.pointer), a.size)
        self.assertEqual(code, 0)
        mismatch = int(np.count_nonzero(output.view(np.uint32) != b.view(np.uint32)))
        print(json.dumps({'policy_silu_words': a.size, 'mismatches': mismatch}), flush=True)
        self.assertEqual(mismatch, 0)

    @unittest.skipUnless(os.environ.get('SMOL_NATIVE_CAPTURE'), 'Optional native policy fixture is not configured')
    def test_silu_actual_action_time_boundary_raw_exact(self):
        root = safe(os.environ['SMOL_NATIVE_CAPTURE'])
        receipt = json.loads(safe(root / 'receipt.json').read_bytes())

        def value(module, field):
            descriptor = next(row for row in receipt['calls'] if row['module'] == module)[field]
            if descriptor['dtype'] != 'torch.float32':
                raise ValueError('Expected native FP32 policy boundary')
            path = Path(descriptor['file'])
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Invalid captured operand path')
            raw = safe(root / path).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), descriptor['sha256'])
            return torch.from_numpy(np.frombuffer(raw, '<f4').copy().reshape(descriptor['shape']))
        self.check(value('action_time_mlp_in', 'output'), value('action_time_mlp_out', 'input'))

    def test_silu_finite_interval_and_signed_zero_raw_exact(self):
        incoming = torch.linspace(-100, 100, 65536, dtype=torch.float32)
        incoming[:2] = torch.tensor([0.0, -0.0], dtype=torch.float32)
        self.check(incoming, torch.nn.functional.silu(incoming))

    def test_odd_expert_float_probabilities_match_native_O_bridge(self):
        generator = torch.Generator().manual_seed(44)
        q = torch.randn((1, 50, 15, 64), generator=generator).to(torch.bfloat16)
        k = torch.randn((1, 177, 5, 64), generator=generator).to(torch.bfloat16)
        v = torch.randn((1, 177, 5, 64), generator=generator).to(torch.bfloat16)
        mask = torch.ones((1, 50, 177), dtype=torch.bool)
        mask[:, :, 140:176] = False
        kk = k.float().repeat_interleave(3, dim=2).transpose(1, 2)
        vv = v.float().repeat_interleave(3, dim=2).transpose(1, 2)
        scores = torch.matmul(q.float().transpose(1, 2), kk.transpose(2, 3)) * 0.125
        probabilities = torch.softmax(torch.where(mask[:, None], scores, torch.finfo(torch.float32).min), dim=-1)
        expected = torch.matmul(probabilities, vv).transpose(1, 2).contiguous().to(torch.bfloat16)
        qq, kw, vw = [item.view(torch.uint16).numpy().copy() for item in (q, k, v)]
        mm = mask.numpy().astype(np.uint8)
        scratch = [np.empty((15, 50, 177), np.float32), np.empty((15, 50, 177), np.float32), np.empty(177, np.float32)]
        output = np.empty_like(qq)
        code = self.library.smol_policy_cross_attention_bf16(*[item.ctypes.data_as(self.word_pointer) for item in (qq, kw, vw)],
            mm.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), *[item.ctypes.data_as(self.pointer) for item in scratch], output.ctypes.data_as(self.word_pointer))
        self.assertEqual(code, 0)
        mismatch = int(np.count_nonzero(output != expected.view(torch.uint16).numpy()))
        print(json.dumps({'expert_cross_O_words': output.size, 'mismatches': mismatch}), flush=True)
        self.assertEqual(mismatch, 0)


if __name__ == '__main__':
    unittest.main()
