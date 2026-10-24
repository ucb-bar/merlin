"""Continuous C prefix/expert/ten-update host control from vision boundary.

All projections inject arithmetic references only after exact actual C input
checks. This is a host functional control, without instruction/device claims.
"""
import ctypes as ct
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
import unittest
from source_relocation import physical_path as safe
import test_smol_policy_block as block


class Parameters(ct.Structure):
    _fields_ = [('prefix', block.Parameters*16), ('expert', block.Parameters*16),
        ('prefix_final_weight', ct.c_void_p), ('expert_final_weight', ct.c_void_p),
        ('language_embedding', ct.c_void_p), ('language_scale', ct.c_float), ('final_epsilon', ct.c_float),
        ('timestep', ct.c_void_p), ('time_embedding', ct.c_void_p)]


FIELDS = ('prefix_input', 'state_projection', 'prefix_final', 'expert_final', 'cache_key',
          'cache_value', 'prefix_mask', 'self_mask', 'cross_mask', 'groups', 'positions',
          'cross_positions', 'action_embedding', 'action_time', 'time_hidden',
          'time_activated', 'suffix_input', 'velocity')


class Frame(ct.Structure):
    _fields_ = [('block', block.Frame), *[(name, ct.c_void_p) for name in FIELDS]]


class State(ct.Structure):
    _fields_ = [(name, ct.c_uint) for name in ('prefix_ready', 'steps', 'projections', 'error')]


Callback = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_uint, ct.c_uint, ct.c_uint,
                        ct.c_uint, ct.c_void_p, ct.c_size_t, ct.c_size_t,
                        ct.c_uint, ct.c_void_p, ct.c_size_t)


@unittest.skipUnless(all(os.environ.get(name) for name in ('SMOL_REPLAY', 'SMOL_CHECKPOINT', 'SMOL_ROPE_CONTRACT')),
                     'Optional target-policy graph fixtures are not configured')
class PolicyGraphTest(unittest.TestCase):
    def test_actual_prefix_and_all_ten_expert_euler_updates(self):
        torch.set_num_threads(1)
        # Reuse this piece's focused fixture readers, without executing its tests.
        block.PolicyBlockTest.setUpClass()
        fixture = block.PolicyBlockTest()
        receipt, data = fixture.receipt, fixture.trajectory
        here = safe(__file__).parent
        path = fixture.temporary / 'graph.so'
        stems = ('smol_policy_graph', 'smol_policy_block', 'smol_policy_ops', 'smol_host_ops',
                 'smol_host_controls', 'smol_host_rope', 'smol_host_attention', 'smol_host_softmax')
        sources = [here / (stem+'.c') for stem in stems]
        sources += [here / (stem+'.h') for stem in stems] + [safe(__file__), safe(block.__file__), here / 'smol_host_bits.h']
        pins = {str(p): hashlib.sha256(safe(p).read_bytes()).hexdigest() for p in sources}
        # Bind the fixture readers' admitted in-memory values to the same raw
        # files rehashed after the complete C traversal, before parameter reads.
        receipt_path = safe(fixture.root / 'receipt.json')
        receipt_raw = receipt_path.read_bytes()
        self.assertEqual(json.loads(receipt_raw), receipt)
        pins[str(receipt_path)] = hashlib.sha256(receipt_raw).hexdigest()
        pins[str(fixture.checkpoint)] = receipt['source']['weights']['sha256']
        pins[str(safe(receipt['trajectory']['path']))] = receipt['trajectory']['sha256']
        rope_path = safe(os.environ['SMOL_ROPE_CONTRACT'])
        rope_raw = rope_path.read_bytes()
        rope = json.loads(rope_raw)
        pins[str(rope_path)] = hashlib.sha256(rope_raw).hexdigest()
        self.assertEqual(rope['constant_parameters']['first_table_position'], fixture.first_position)
        for name, table in zip(('cosine', 'sine'), fixture.tables):
            record = rope['assets'][name]
            self.assertEqual(list(table.shape), record['shape'])
            self.assertEqual(hashlib.sha256(table.tobytes()).hexdigest(), record['file']['sha256'])
            pins[str(safe(record['file']['path']))] = record['file']['sha256']
        subprocess.run(['/usr/bin/cc', '-O2', '-fPIC', '-shared', '-Wall', '-Wextra', '-Werror',
                        '-ffp-contract=off', '-fno-fast-math',
                        *[str(here / (stem+'.c')) for stem in stems], '-lm', '-o', str(path)], check=True)
        owner = ct.CDLL(str(path))
        prefix_fn, step_fn = owner.smol_policy_prefix, owner.smol_policy_step
        prefix_fn.argtypes = [ct.c_void_p]*5 + [ct.POINTER(Parameters), ct.POINTER(Frame), ct.POINTER(State), Callback, ct.c_void_p]
        step_fn.argtypes = [ct.c_void_p, ct.c_float, ct.c_void_p, ct.c_void_p,
                            ct.POINTER(Parameters), ct.POINTER(Frame), ct.POINTER(State), Callback, ct.c_void_p]
        arrays = []

        def keep(value):
            arrays.append(value)
            return value.ctypes.data

        prefixes, experts = (block.Parameters*16)(), (block.Parameters*16)()
        cosine, sine = fixture.tables
        for family, target in (('vlm.model.text_model', prefixes), ('lm_expert', experts)):
            for layer in range(16):
                module = 'vlm_with_expert.'+family+'.layers.'+str(layer)
                target[layer] = block.Parameters(0 if family.startswith('vlm') else 1+layer%2,
                    keep(fixture.weight(module, 'input_layernorm')), keep(fixture.weight(module, 'post_attention_layernorm')),
                    1e-5, 1e-5, cosine.ctypes.data, sine.ctypes.data, fixture.first_position, cosine.shape[0])
        command = receipt['command']
        observation_path = safe(command[command.index('--observation-input')+1])
        observation_digest = next(item['sha256'] for item in receipt['source_bookends'] if item['path'] == str(observation_path))
        with observation_path.open('rb') as stream:
            self.assertEqual(hashlib.file_digest(stream, 'sha256').hexdigest(), observation_digest)
        pins[str(observation_path)] = observation_digest
        with np.load(observation_path, allow_pickle=False) as archive:
            observation = {name: archive[name].copy() for name in archive.files}
        tokens = observation['language_tokens'].reshape(48)
        language = fixture.weight('vlm_with_expert.vlm.model.text_model', 'embed_tokens')[tokens].copy()
        times = np.array([data[f'step_{step:02d}_timestep'].reshape(-1)[0] for step in range(10)], np.float32)
        time_embedding = []
        # Complete source function table on the fixed ten source timesteps.
        from lerobot.policies.smolvla.modeling_smolvla import create_sinusoidal_pos_embedding
        import lerobot.policies.smolvla.modeling_smolvla as native
        native_path = safe(native.__file__)
        pins[str(native_path)] = hashlib.sha256(native_path.read_bytes()).hexdigest()
        records = [row for row in receipt['calls'] if row['module'] == 'action_time_mlp_in']
        for step, time in enumerate(times):
            function_value = create_sinusoidal_pos_embedding(torch.tensor([time]), 720, 0.004, 4.0,
                                                            device=torch.device('cpu')).float().numpy()[0]
            actual_reference = fixture.tensor(records[step]['input']).reshape(50, 1440)[:, 720:]
            np.testing.assert_array_equal(actual_reference.view(np.uint32),
                np.broadcast_to(function_value, (50, 720)).copy().view(np.uint32))
            time_embedding.append(function_value)
        time_embedding = np.array(time_embedding, np.float32)
        parameters = Parameters(prefixes, experts,
            keep(fixture.weight('vlm_with_expert.vlm.model.text_model', 'norm')),
            keep(fixture.weight('vlm_with_expert.lm_expert', 'norm')), keep(language),
            math.sqrt(960), 1e-5, keep(times), keep(time_embedding))
        buffers = {}
        for name in block.FIELDS:
            size, dtype = 177*960, np.uint16
            if name in ('key', 'value', 'rotary_key'): size = 177*320
            elif name in ('joined_key', 'joined_value'): size = 227*320
            elif name in ('gate', 'up', 'activated'): size = 177*2560
            elif name in ('scores', 'float_probabilities', 'bf16_probabilities'):
                size, dtype = 15*177*227, np.uint16 if name == 'bf16_probabilities' else np.float32
            elif name == 'workspace': size, dtype = 227, np.float32
            elif name.startswith('cross_'): size, dtype = 177*320, np.float32
            buffers['block.'+name] = np.empty(size, dtype)
        sizes = dict(prefix_input=(177*960, np.float32), state_projection=(960, np.float32),
            prefix_final=(177*960, np.uint16), expert_final=(50*720, np.uint16),
            cache_key=(16*177*320, np.uint16), cache_value=(16*177*320, np.uint16),
            prefix_mask=(177*177, np.uint8), self_mask=(50*227, np.uint8), cross_mask=(50*177, np.uint8),
            groups=(177, np.int64), positions=(177, np.int64), cross_positions=(50, np.int64),
            action_embedding=(50*720, np.float32), action_time=(50*1440, np.float32),
            time_hidden=(50*720, np.float32), time_activated=(50*720, np.float32),
            suffix_input=(50*720, np.float32), velocity=(50*32, np.float32))
        for name, (size, dtype) in sizes.items(): buffers[name] = np.empty(size, dtype)
        frame = Frame(block.Frame(*[buffers['block.'+name].ctypes.data for name in block.FIELDS]),
                      *[buffers[name].ctypes.data for name in FIELDS])
        events, errors = [], []

        @Callback
        def inject(context, site, layer, role, input_dtype, address, rows, k, output_dtype, destination, n):
            try:
                name = ('state_proj', 'action_in_proj', 'action_time_mlp_in', 'action_time_mlp_out', 'action_out_proj')[site] if site < 5 else (
                    'vlm_with_expert.' + ('vlm.model.text_model' if site == 5 else 'lm_expert')
                    + f'.layers.{layer}.' + block.ROLES[role])
                record = receipt['calls'][148 + len(events)]
                self.assertEqual(record['module'], name)
                expected = fixture.tensor(record['input']).reshape(rows, k)
                self.assertEqual(expected.dtype, np.uint16 if input_dtype == 0 else np.float32)
                pointer = ct.POINTER(ct.c_uint16 if input_dtype == 0 else ct.c_float)
                actual = np.ctypeslib.as_array(ct.cast(address, pointer), shape=(rows*k,)).reshape(rows, k)
                mismatch = int(np.count_nonzero(actual.view(np.uint16) != expected.view(np.uint16)))
                events.append({'call': record['index'], 'module': name, 'mismatches': mismatch})
                self.assertEqual(mismatch, 0, name)
                output = fixture.tensor(record['output']).reshape(rows, n)
                self.assertEqual(output.dtype, np.uint16 if output_dtype == 0 else np.float32)
                ct.memmove(destination, output.ctypes.data, output.nbytes)
                return 0
            except Exception as error:
                errors.append(str(error))
                return 71
        image_rows = data['prefix_embeddings'][0, :128].copy().view(np.uint32)
        self.assertFalse(np.any(image_rows & 65535))
        image_words = (image_rows >> 16).astype(np.uint16).reshape(2, 64, 960)
        pad, attention = [np.ascontiguousarray(data[name].reshape(177).astype(np.uint8))
                          for name in ('prefix_pad_mask', 'prefix_attention_mask')]
        state = State()
        result = prefix_fn(image_words[0].ctypes.data, image_words[1].ctypes.data, observation['state'].ctypes.data,
            pad.ctypes.data, attention.ctypes.data, ct.byref(parameters), ct.byref(frame), ct.byref(state), inject, None)
        self.assertEqual((result, state.prefix_ready, state.projections, errors), (0, 1, 113, []), events[-2:])
        np.testing.assert_array_equal(buffers['prefix_input'].reshape(data['prefix_embeddings'].shape).view(np.uint32),
                                      data['prefix_embeddings'].view(np.uint32))
        for name, reference in (('cache_key', 'prefix_key_cache'), ('cache_value', 'prefix_value_cache')):
            np.testing.assert_array_equal(buffers[name].reshape(data[reference].shape), data[reference])
        current = observation['noise'].copy().reshape(50, 32)
        for step in range(10):
            np.testing.assert_array_equal(current.view(np.uint32), data[f'step_{step:02d}_input_state'].reshape(50, 32).view(np.uint32))
            output = np.empty_like(current)
            result = step_fn(current.ctypes.data, float(times[step]), output.ctypes.data, pad.ctypes.data,
                ct.byref(parameters), ct.byref(frame), ct.byref(state), inject, None)
            self.assertEqual((result, state.steps, state.projections, errors), (0, step+1, 113+116*(step+1), []), events[-2:])
            np.testing.assert_array_equal(buffers['velocity'].view(np.uint32), data[f'step_{step:02d}_velocity'].reshape(-1).view(np.uint32))
            np.testing.assert_array_equal(output.view(np.uint32), data[f'step_{step:02d}_output_state'].reshape(50, 32).view(np.uint32))
            current = output
            print(json.dumps({'actual_C_step': step, 'checked_predecessors': len(events), 'state_exact': True}), flush=True)
        self.assertEqual(len(events), 1273)
        np.testing.assert_array_equal(current[:, :6].view(np.uint32), data['final_actions'].reshape(50, 6).view(np.uint32))
        for path, digest in pins.items():
            with safe(path).open('rb') as stream:
                self.assertEqual(hashlib.file_digest(stream, 'sha256').hexdigest(), digest)


if __name__ == '__main__':
    unittest.main()
