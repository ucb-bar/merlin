"""Product vision host math against actual retained native operator boundaries."""
import ctypes
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from source_relocation import physical_path as safe


@unittest.skipUnless(os.environ.get('SMOL_NATIVE_CAPTURE') and os.environ.get('SMOL_CHECKPOINT'),
                     'Optional native vision fixtures are not configured')
class VisionOpsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        cls.temporary = Path(tempfile.mkdtemp(prefix='vision-ops-'))
        cls.library_path = cls.temporary / 'vision.so'
        here = Path(__file__).absolute().parent
        subprocess.run(['/usr/bin/cc', '-O2', '-fPIC', '-shared', '-ffp-contract=off', '-fno-fast-math',
                        str(here / 'smol_vision_ops.c'), '-lm', '-o', str(cls.library_path)], check=True)
        cls.library = ctypes.CDLL(str(cls.library_path))
        cls.pointer = ctypes.POINTER(ctypes.c_uint16)
        cls.library.smol_vision_gelu_bf16.argtypes = [cls.pointer, cls.pointer, cls.pointer, ctypes.c_size_t]
        cls.library.smol_vision_layernorm_bf16.argtypes = [cls.pointer, cls.pointer, cls.pointer,
                                                        ctypes.c_float, cls.pointer, ctypes.c_size_t]
        cls.capture = safe(os.environ['SMOL_NATIVE_CAPTURE'])
        cls.receipt = json.loads(safe(cls.capture / 'receipt.json').read_bytes())
        cls.checkpoint = safe(os.environ['SMOL_CHECKPOINT'])
        with cls.checkpoint.open('rb') as stream:
            cls.header_size = struct.unpack('<Q', stream.read(8))[0]
            if cls.header_size > 1024*1024:
                raise ValueError('Oversized checkpoint header')
            cls.header = json.loads(stream.read(cls.header_size))

    @classmethod
    def ptr(cls, array):
        return array.ctypes.data_as(cls.pointer)

    def parameter(self, suffix):
        record = self.header['model.vlm_with_expert.vlm.model.vision_model.' + suffix]
        if record['dtype'] != 'BF16':
            raise ValueError('Expected BF16 vision parameter')
        begin, end = record['data_offsets']
        with self.checkpoint.open('rb') as stream:
            stream.seek(8 + self.header_size + begin)
            raw = stream.read(end-begin)
        return torch.from_numpy(np.frombuffer(raw, '<u2').copy().reshape(record['shape'])).view(torch.bfloat16)

    def call(self, suffix, field):
        name = 'vlm_with_expert.vlm.model.vision_model.' + suffix
        item = next(row for row in self.receipt['calls'] if row['module'] == name)[field]
        path = Path(item['file'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid captured operand path')
        import hashlib
        raw = safe(self.capture / path).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), item['sha256'])
        return torch.from_numpy(np.frombuffer(raw, '<u2').copy().reshape(item['shape'])).view(torch.bfloat16)

    def test_gelu_actual_fc_boundary_raw_exact(self):
        all_words = np.arange(65536, dtype=np.uint16)
        inputs = torch.from_numpy(all_words.copy()).view(torch.bfloat16)
        table = torch.nn.functional.gelu(inputs, approximate='tanh').view(torch.uint16).numpy().copy()
        table[(all_words & 0x7f80) == 0x7f80] = 0
        incoming = self.call('encoder.layers.0.mlp.fc1', 'output').view(torch.uint16).numpy().copy()
        expected = self.call('encoder.layers.0.mlp.fc2', 'input').view(torch.uint16).numpy().copy()
        output = np.empty_like(incoming)
        code = self.library.smol_vision_gelu_bf16(self.ptr(incoming), self.ptr(table), self.ptr(output), incoming.size)
        self.assertEqual(code, 0)
        mismatches = int(np.count_nonzero(output != expected))
        print(json.dumps({'vision_gelu_words': incoming.size, 'mismatches': mismatches}), flush=True)
        self.assertEqual(mismatches, 0)

    def test_layernorm_actual_first_vision_boundary_raw_exact(self):
        patches = self.call('embeddings.patch_embedding', 'output').flatten(2).transpose(1, 2)
        position = self.parameter('embeddings.position_embedding.weight')
        incoming = (patches + position).contiguous()
        weight = self.parameter('encoder.layers.0.layer_norm1.weight')
        bias = self.parameter('encoder.layers.0.layer_norm1.bias')
        expected = self.call('encoder.layers.0.self_attn.q_proj', 'input')
        native = torch.nn.functional.layer_norm(incoming, (768,), weight, bias, 1e-6)
        self.assertTrue(torch.equal(native.view(torch.uint16), expected.view(torch.uint16)))
        words = incoming.view(torch.uint16).numpy().copy()
        ww, bb = weight.view(torch.uint16).numpy().copy(), bias.view(torch.uint16).numpy().copy()
        output = np.empty_like(words)
        code = self.library.smol_vision_layernorm_bf16(self.ptr(words), self.ptr(ww), self.ptr(bb),
                                                     ctypes.c_float(1e-6), self.ptr(output), 1024)
        self.assertEqual(code, 0)
        mismatches = int(np.count_nonzero(output != expected.view(torch.uint16).numpy()))
        print(json.dumps({'vision_layernorm_words': words.size, 'mismatches': mismatches}), flush=True)
        self.assertEqual(mismatches, 0)

    def test_layernorm_random_constant_rows_and_nonfinite_refusal(self):
        generator = torch.Generator().manual_seed(193)
        incoming = torch.randn((17, 768), generator=generator).to(torch.bfloat16)
        incoming[0].fill_(3.0)
        incoming[1].zero_()
        weight = self.parameter('encoder.layers.0.layer_norm1.weight')
        bias = self.parameter('encoder.layers.0.layer_norm1.bias')
        expected = torch.nn.functional.layer_norm(incoming, (768,), weight, bias, 1e-6)
        words = incoming.view(torch.uint16).numpy().copy()
        ww, bb = weight.view(torch.uint16).numpy().copy(), bias.view(torch.uint16).numpy().copy()
        output = np.empty_like(words)
        code = self.library.smol_vision_layernorm_bf16(self.ptr(words), self.ptr(ww), self.ptr(bb),
                                                     ctypes.c_float(1e-6), self.ptr(output), 17)
        self.assertEqual(code, 0)
        np.testing.assert_array_equal(output, expected.view(torch.uint16).numpy())
        words[0, 0] = 0x7f80
        output.fill(0x1234)
        code = self.library.smol_vision_layernorm_bf16(self.ptr(words), self.ptr(ww), self.ptr(bb),
                                                     ctypes.c_float(1e-6), self.ptr(output), 17)
        self.assertEqual(code, 2)
        self.assertTrue(np.all(output == 0x1234))


if __name__ == '__main__':
    unittest.main()
