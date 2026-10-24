"""Actual target-policy block C predecessors with explicit projection injection.

These controls execute genuine host math, not a device or the complete policy.
The sixteen prefix blocks and first complete expert traversal begin at captured
vision-prefix and timestep-MLP boundary inputs, respectively.
"""
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import numpy as np
from source_relocation import physical_path as safe


BF = np.dtype('<u2')
F32 = np.dtype('<f4')
ROLES = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
         'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
FIELDS = ('normalized', 'query', 'key', 'value', 'rotary_query', 'rotary_key',
          'joined_key', 'joined_value', 'attention', 'projected', 'residual',
          'postnorm', 'gate', 'up', 'activated', 'down', 'output',
          'bf16_probabilities', 'scores', 'float_probabilities', 'workspace',
          'cross_input', 'cross_key', 'cross_value')


class Parameters(ct.Structure):
    _fields_ = [('kind', ct.c_uint), ('input_weight', ct.c_void_p), ('post_weight', ct.c_void_p),
                ('input_epsilon', ct.c_float), ('post_epsilon', ct.c_float),
                ('cosine', ct.c_void_p), ('sine', ct.c_void_p),
                ('first_table_position', ct.c_int64), ('table_rows', ct.c_size_t)]


class Frame(ct.Structure):
    _fields_ = [(name, ct.c_void_p) for name in FIELDS]


class State(ct.Structure):
    _fields_ = [(name, ct.c_uint) for name in ('started', 'projections', 'complete', 'error')]


Callback = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_uint, ct.c_uint, ct.c_void_p,
                        ct.c_size_t, ct.c_size_t, ct.c_uint, ct.c_void_p, ct.c_size_t)


@unittest.skipUnless(all(os.environ.get(name) for name in ('SMOL_REPLAY', 'SMOL_CHECKPOINT', 'SMOL_ROPE_CONTRACT')),
                     'Optional target-policy block fixtures are not configured')
class PolicyBlockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = safe(os.environ['SMOL_REPLAY'])
        cls.receipt = json.loads(safe(cls.root / 'receipt.json').read_bytes())
        if cls.receipt['status'] != 'PASS_ATLAS_ARITHMETIC_FULL_POLICY_REPLAY':
            raise ValueError('Require completed target arithmetic reference')
        descriptor = cls.receipt['trajectory']
        raw = safe(descriptor['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != descriptor['sha256']:
            raise ValueError('Changed reference trajectory')
        with np.load(safe(descriptor['path']), allow_pickle=False) as archive:
            cls.trajectory = {name: archive[name].copy() for name in archive.files}
        checkpoint = safe(os.environ['SMOL_CHECKPOINT'])
        expected = cls.receipt['source']['weights']['sha256']
        with checkpoint.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != expected:
            raise ValueError('Changed checkpoint')
        with checkpoint.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            cls.header = json.loads(stream.read(length))
            cls.data_start = 8 + length
        cls.checkpoint = checkpoint
        contract = json.loads(safe(os.environ['SMOL_ROPE_CONTRACT']).read_bytes())
        cls.tables = []
        for name in ('cosine', 'sine'):
            item = contract['assets'][name]
            raw = safe(item['file']['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != item['file']['sha256']:
                raise ValueError('Changed source rotary table')
            cls.tables.append(np.frombuffer(raw, F32).copy().reshape(item['shape']))
        cls.first_position = contract['constant_parameters']['first_table_position']
        cls.temporary = Path(tempfile.mkdtemp(prefix='policy-block-'))
        here = safe(__file__).parent
        library = cls.temporary / 'block.so'
        stems = ('smol_policy_block', 'smol_policy_ops', 'smol_host_ops',
                 'smol_host_rope', 'smol_host_attention', 'smol_host_softmax')
        subprocess.run(['/usr/bin/cc', '-O2', '-fPIC', '-shared', '-Wall', '-Wextra', '-Werror',
                        '-ffp-contract=off', '-fno-fast-math', *[str(here / (s+'.c')) for s in stems],
                        '-lm', '-o', str(library)], check=True)
        cls.library = ct.CDLL(str(library))
        cls.library.smol_policy_block.argtypes = [ct.c_void_p, ct.c_uint, ct.c_void_p,
            ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.POINTER(Parameters), ct.POINTER(Frame),
            ct.POINTER(State), Callback, ct.c_void_p]

    def weight(self, module, suffix):
        row = self.header['model.' + module + '.' + suffix + '.weight']
        self.assertEqual(row['dtype'], 'BF16')
        begin, end = row['data_offsets']
        with self.checkpoint.open('rb') as stream:
            stream.seek(self.data_start + begin)
            raw = stream.read(end - begin)
        return np.frombuffer(raw, BF).copy().reshape(row['shape'])

    def tensor(self, row):
        path = Path(row['file'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid reference member')
        raw = safe(self.root / path).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), row['sha256'])
        dtype = BF if row['dtype'] == 'torch.bfloat16' else F32
        return np.frombuffer(raw, dtype).copy().reshape(row['shape'])

    def run_block(self, module, kind, incoming, key_cache, value_cache, positions, mask):
        input_weight = self.weight(module, 'input_layernorm')
        post_weight = self.weight(module, 'post_attention_layernorm')
        cosine, sine = self.tables
        parameters = Parameters(kind, input_weight.ctypes.data, post_weight.ctypes.data,
            1e-5, 1e-5, cosine.ctypes.data, sine.ctypes.data, self.first_position, cosine.shape[0])
        buffers = {}
        for name in FIELDS:
            size, dtype = 177*960, BF
            if name in ('key', 'value', 'rotary_key'):
                size = 177*320
            elif name in ('joined_key', 'joined_value'):
                size = 227*320
            elif name in ('gate', 'up', 'activated'):
                size = 177*2560
            elif name in ('scores', 'float_probabilities', 'bf16_probabilities'):
                size, dtype = 15*177*227, BF if name == 'bf16_probabilities' else F32
            elif name == 'workspace':
                size, dtype = 227, F32
            elif name.startswith('cross_'):
                size, dtype = 177*320, F32
            buffers[name] = np.empty(size, dtype)
        frame = Frame(*[buffers[name].ctypes.data for name in FIELDS])
        events, errors = [], []

        @Callback
        def invoke(context, role, input_dtype, address, rows, k, output_dtype, destination, n):
            try:
                self.assertEqual(role, len(events))
                name = module + '.' + ROLES[role]
                record = next(row for row in self.receipt['calls'] if row['module'] == name)
                expected = self.tensor(record['input']).reshape(rows, k)
                self.assertEqual(expected.dtype, BF if input_dtype == 0 else F32)
                pointer = ct.POINTER(ct.c_uint16 if input_dtype == 0 else ct.c_float)
                actual = np.ctypeslib.as_array(ct.cast(address, pointer), shape=(rows*k,)).reshape(rows, k)
                mismatch = int(np.count_nonzero(actual.view(np.uint16) != expected.view(np.uint16)))
                events.append({'module': name, 'mismatches': mismatch})
                self.assertEqual(mismatch, 0, name)
                output = self.tensor(record['output']).reshape(rows, n)
                self.assertEqual(output.dtype, BF if output_dtype == 0 else F32)
                ct.memmove(destination, output.ctypes.data, output.nbytes)
                return 0
            except Exception as error:
                errors.append(str(error))
                return 71
        state = State()
        code = self.library.smol_policy_block(incoming.ctypes.data, 0 if incoming.dtype == BF else 1,
            positions.ctypes.data, mask.ctypes.data, key_cache.ctypes.data, value_cache.ctypes.data,
            ct.byref(parameters), ct.byref(frame), ct.byref(state), invoke, None)
        self.assertEqual((code, state.complete, state.projections, errors), (0, 1, 7, []), events)
        rows, width = (177, 960) if kind == 0 else (50, 720)
        print(json.dumps({'block': module, 'checked_predecessors': len(events), 'mismatches': 0}), flush=True)
        return buffers['output'][:rows*width].copy().reshape(rows, width)

    def test_actual_all_prefix_caches_and_predecessors(self):
        data = self.trajectory
        pad = data['prefix_pad_mask'].reshape(177).astype(np.uint8)
        groups = np.cumsum(data['prefix_attention_mask'].reshape(177))
        mask = ((groups[None, :] <= groups[:, None]) & pad[:, None] & pad[None, :]).astype(np.uint8)
        positions = np.ascontiguousarray(np.cumsum(pad).astype(np.int64)-1)
        key, value = np.empty((177, 320), BF), np.empty((177, 320), BF)
        incoming = data['prefix_embeddings'].reshape(177, 960)
        for layer in range(16):
            incoming = self.run_block('vlm_with_expert.vlm.model.text_model.layers.'+str(layer), 0,
                incoming, key, value, positions, mask)
            np.testing.assert_array_equal(key.reshape(177, 5, 64), data['prefix_key_cache'][layer].reshape(177, 5, 64))
            np.testing.assert_array_equal(value.reshape(177, 5, 64), data['prefix_value_cache'][layer].reshape(177, 5, 64))

    def test_actual_all_step_zero_expert_predecessors(self):
        data = self.trajectory
        pad = data['prefix_pad_mask'].reshape(177).astype(np.uint8)
        record = next(row for row in self.receipt['calls'] if row['module'] == 'action_time_mlp_out')
        incoming = self.tensor(record['output']).reshape(50, 720)
        for layer in range(16):
            key = data['prefix_key_cache'][layer].reshape(177, 320).copy()
            value = data['prefix_value_cache'][layer].reshape(177, 320).copy()
            mask = np.broadcast_to(pad, (50, 177)).copy()
            if layer % 2 == 0:
                mask = np.concatenate([mask, np.tril(np.ones((50, 50), np.uint8))], axis=1)
            positions = np.arange(50, dtype=np.int64) + (int(pad.sum()) if layer % 2 == 0 else 0)
            incoming = self.run_block('vlm_with_expert.lm_expert.layers.' + str(layer),
                1 + layer % 2, incoming, key, value, positions, mask)
        norm_weight = self.weight('vlm_with_expert.lm_expert', 'norm')
        normalized = np.empty((50, 720), BF)
        norm = self.library.smol_host_rmsnorm_bf16
        norm.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_float, ct.c_void_p, ct.c_size_t, ct.c_size_t]
        self.assertEqual(norm(incoming.ctypes.data, norm_weight.ctypes.data, 1e-5,
                              normalized.ctypes.data, 50, 720), 0)
        expected = self.tensor(next(row for row in self.receipt['calls'] if row['module'] == 'action_out_proj')['input'])
        promoted = (normalized.astype(np.uint32) << 16).view(np.float32).reshape(expected.shape)
        np.testing.assert_array_equal(promoted.view(np.uint32), expected.view(np.uint32))


if __name__ == '__main__':
    unittest.main()
