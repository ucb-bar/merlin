"""Fused graph replacement, emitted-byte assembly, and worker ordering controls."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from merlin.common.paths import repo_root

sys.path.insert(0, str(repo_root() / "examples/atlas/comparisons"))

from capture_tinyllama_fused import (FusedProjectionCapture, MODEL_PIN, ProjectionWorkers, assemble_fp8,
                                    down_scales)
from capture_tinyllama_native import array_identity, safe_path
from compare_tinyllama_attention import raw_array
from tiny_fp8_pack import encode
import tiny_workload_native as native


def layout_module():
    atlas_root = os.environ.get('ATLAS_MLIR_ROOT')
    if not atlas_root:
        raise unittest.SkipTest('Set ATLAS_MLIR_ROOT for the fused layout provider')
    path = safe_path(Path(atlas_root) / 'tools/tiny_full/fusion_layout.py')
    spec = importlib.util.spec_from_file_location('test_fused_layout', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeWorkers:
    """Return declared device results independently of native Linear output."""
    def __init__(self, owner):
        self.owner, self.batches = owner, []

    def run(self, requests):
        self.batches.append(requests)
        result = []
        for request in requests:
            m, n, rows, role = (request[key] for key in ('m', 'n', 'logical_rows', 'role'))
            stem = str(Path(request['out']).relative_to(self.owner.out))
            if role == 5:
                # Device emits 2.0 at fixed scale zero; extent holes are poison.
                compact = b'\x40' * (m*n)
                spans = [dict(offset=r*32*5632+request['column']*32, bytes=n*32) for r in range(m//32)]
                extent = spans[-1]['offset'] + spans[-1]['bytes']
                output = bytearray(b'\xcd' * extent)
                for span in spans:
                    output[span['offset']:span['offset']+span['bytes']] = b'\x40' * span['bytes']
                words = np.full((m, n), 0x4000, np.uint16)
            else:
                words = torch.full((m, n), float(role+1), dtype=torch.bfloat16).view(torch.uint16).numpy()
                output = self.owner.layout.pack_bf16_tiles(words)
                spans = [dict(offset=0, bytes=len(output))]
            launch = dict(model_commit=MODEL_PIN, numerics='rtl', accelerated=False, native_dispatch_enabled=False,
                          implementation_sources=[], written_spans=spans,
                          output_tiled=self.owner.save_bytes(stem + '-tiled.bin', output),
                          output_bf16=self.owner.save(stem + '-bf16.bin', words[:rows]))
            if role == 5:
                launch['output_fp8'] = self.owner.save_bytes(stem + '-fp8.bin', compact)
            result.append(launch)
        return result

    def close(self):
        pass


class TinyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(64, 5632, bias=False, dtype=torch.bfloat16)
        self.up_proj = torch.nn.Linear(64, 5632, bias=False, dtype=torch.bfloat16)
        self.down_proj = torch.nn.Linear(5632, 64, bias=False, dtype=torch.bfloat16)

    def forward(self, value):
        raise AssertionError('native activation and multiplication must be skipped')


class TinyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(self, name, torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16))
        self.forwarded = None

    def forward(self, hidden_states, **kwargs):
        self.forwarded = kwargs
        self.q_proj(hidden_states)
        self.k_proj(hidden_states)
        self.v_proj(hidden_states)
        return self.o_proj(hidden_states), None


class TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = torch.nn.Identity()
        self.post_attention_layernorm = torch.nn.Identity()
        self.self_attn, self.mlp = TinyAttention(), TinyMLP()

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                use_cache=False, position_embeddings=None, **kwargs):
        raise AssertionError('native residual operations must be skipped')


class FusedCaptureTest(unittest.TestCase):
    def test_written_spans_assemble_without_poison_holes(self):
        m, width = 64, 96
        panels = []
        for col in range(3):
            extent = bytearray(b'\xcd' * (m*width))
            spans = []
            for row in range(2):
                offset = row*32*width + col*1024
                extent[offset:offset+1024] = bytes([col+1])*1024
                spans.append(dict(offset=offset, bytes=1024))
            panels.append(dict(launch=dict(output_tiled=bytes(extent), written_spans=spans)))
        assembled = assemble_fp8(panels, lambda x: x, m, width)
        self.assertEqual(assembled, (b'\1'*1024+b'\2'*1024+b'\3'*1024)*2)
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            assemble_fp8(panels+[panels[0]], lambda x: x, m, width)
        with self.assertRaisesRegex(ValueError, 'cover'):
            assemble_fp8(panels[:-1], lambda x: x, m, width)
        panels[0]['launch']['written_spans'][0]['offset'] = -1024
        with self.assertRaisesRegex(ValueError, 'outside'):
            assemble_fp8(panels, lambda x: x, m, width)

    def test_graph_device_results_advance_and_replay_restores_methods(self):
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            owner = FusedProjectionCapture.__new__(FusedProjectionCapture)
            owner.out, owner.torch, owner.layout = Path(directory), torch, layout_module()
            def save(relative, array):
                path = owner.out / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(array.tobytes())
                return dict(file=relative, **array_identity(array), bytes=array.nbytes)
            owner.save = save
            layer = TinyLayer()
            owner.model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
            owner.sites = tuple('model.layers.0.'+site for site in native.PROJECTIONS)
            owner.scales = {(0, 'model.layers.0.mlp.down_proj'): 0}
            owner.weights, owner.saved, owner.methods, owner.context = {}, {}, [], None
            owner.calls, owner.parameters, owner.records = [], {}, []
            owner.workers = FakeWorkers(owner)
            originals = (layer.forward.__func__, layer.mlp.forward.__func__)
            handles = []
            mode = ['capture']
            for index, site in enumerate(owner.sites):
                module = layer.get_submodule(site.removeprefix('model.layers.0.'))
                def hook(module, inputs, output, site=site, index=index):
                    if site.endswith('self_attn.o_proj'):
                        words = raw_array(inputs[0][0], torch)
                        owner.context['o_input_physical'] = np.concatenate((words, np.zeros((31, 64), np.uint16)))
                    return owner.linear(site, module, raw_array(inputs[0][0], torch), 0, index, mode[0])
                handles.append(module.register_forward_hook(hook))
            marker = object()
            try:
                owner.install()
                words = torch.full((1, 1, 64), 11, dtype=torch.bfloat16)
                for label in ('capture', 'replay'):
                    mode[0] = label
                    owner.begin_session(label)
                    value = layer(words, None, position_ids=marker, use_cache=True, custom=marker)
                    self.assertTrue(torch.equal(value, torch.full_like(value, 7)))
                    self.assertIs(layer.self_attn.forwarded['position_ids'], marker)
                    self.assertIs(layer.self_attn.forwarded['custom'], marker)
                    self.assertTrue(layer.self_attn.forwarded['use_cache'])
                self.assertEqual(len(owner.calls), 7)
                self.assertEqual(len(owner.workers.batches[4]), 3)
                self.assertEqual(len(owner.workers.batches[5]), 3)
                for role, value in ((3, 11), (5, 5), (6, 4)):
                    request = owner.workers.batches[role][0]
                    operand = owner.layout.unpack_bf16_tiles(owner.read(request['operand_tiles']),
                                                              32, request['n'])
                    expected = raw_array(torch.full((1, request['n']), value, dtype=torch.bfloat16), torch)
                    self.assertTrue(np.array_equal(operand[:1], expected))
                    self.assertTrue(np.all(operand[1:] == 0))
                up_b = raw_array(layer.mlp.up_proj.weight.T, torch)
                encoded_b, _ = encode(np.ascontiguousarray(up_b[:, 2048:4096]))
                b_request = owner.workers.batches[5][1]
                expected_b = b''.join(encoded_b[r:r+32, c:c+32].tobytes()
                                      for c in range(0, 2048, 32) for r in range(0, 64, 32))
                self.assertEqual(owner.read(b_request['B_fp8']), expected_b)
                down = owner.calls[-1]
                self.assertEqual(owner.read(down['A_fp8']), b'\x40'*(32*5632))
                self.assertEqual(down['A_bf16']['shape'], [1, 5632])
                self.assertEqual([p['physical_mkn'][2] for p in owner.calls[5]['panels']], [2048, 2048, 1536])
                for panel in owner.calls[5]['panels']:
                    self.assertEqual(panel['expected_bf16']['shape'][0], 32)
                mode[0] = 'replay'
                owner.begin_session('replay')
                with self.assertRaisesRegex(ValueError, 'predecessor or residual'):
                    layer(words+1)
            finally:
                owner.restore()
                owner.restore()
                for handle in handles:
                    handle.remove()
            self.assertIs(layer.forward.__func__, originals[0])
            self.assertIs(layer.mlp.forward.__func__, originals[1])
            self.assertNotIn('forward', layer.__dict__)
            self.assertNotIn('forward', layer.mlp.__dict__)

    def test_fixed_scale_source_rejects_modified_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'contract.json'
            path.write_text(json.dumps(dict(projection_kernels=[])))
            with self.assertRaisesRegex(ValueError, 'Source hash differs'):
                down_scales(path, [])

    def test_o_scale_validates_full_physical_input_and_replay_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = FusedProjectionCapture.__new__(FusedProjectionCapture)
            owner.out, owner.torch, owner.layout = Path(directory), torch, layout_module()
            def save(relative, value):
                path = owner.out/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value.tobytes())
                return dict(file=relative, **array_identity(value), bytes=value.nbytes)
            owner.save = save
            logical = np.full((1, 64), 0x3f80, np.uint16)
            physical = np.full((32, 64), 0x4280, np.uint16)
            physical[:1] = logical
            owner.context = dict(layer=0, entry=np.zeros((1, 64), np.uint16), o_input_physical=physical)
            site = 'model.layers.0.self_attn.o_proj'
            owner.weights = {site: (0, 0, [(0, 64, {}, 0)])}
            owner.saved, owner.scales, owner.parameters, owner.calls, owner.records = {}, {}, {}, [], []
            owner.workers = FakeWorkers(owner)
            module = SimpleNamespace(out_features=64, weight=SimpleNamespace(_version=0, data_ptr=lambda: 0))
            owner.linear(site, module, logical, 0, 3, 'capture')
            self.assertEqual(owner.calls[0]['A_scale'], -2)
            encoded = owner.layout.unpack_fp8_tiles(owner.read(owner.calls[0]['A_fp8']), 32, 64)
            self.assertTrue(np.all(encoded[1:] == 0))
            self.assertTrue(np.any(encoded[:1] != 0))
            physical[31, 0] = 0x4300
            with self.assertRaisesRegex(ValueError, 'predecessor or residual'):
                owner.linear(site, module, logical, 0, 3, 'replay')
            physical[31, 0] = 0x7fc0
            with self.assertRaisesRegex(ValueError, 'finite'):
                owner.linear(site, module, logical, 1, 3, 'capture')
            self.assertEqual(len(owner.workers.batches), 1)
            owner.context.pop('o_input_physical')
            with self.assertRaisesRegex(ValueError, 'physical BF16'):
                owner.linear(site, module, logical, 1, 3, 'capture')

    def test_three_workers_dispatch_all_panels_before_reading(self):
        events = []
        class Pipe:
            def __init__(self, index, output=False):
                self.index, self.output = index, output
            def write(self, value):
                events.append(('send', self.index, json.loads(value)['panel']))
            def flush(self):
                pass
            def readline(self):
                events.append(('read', self.index))
                return json.dumps(dict(status='ok', record=self.index))
        owner = ProjectionWorkers.__new__(ProjectionWorkers)
        owner.workers = [SimpleNamespace(stdin=Pipe(i), stdout=Pipe(i, True)) for i in range(3)]
        self.assertEqual(owner.run([dict(panel=i) for i in range(5)]), [0, 1, 2, 0, 1])
        self.assertEqual(events[:4], [('send', 0, 0), ('send', 1, 1), ('send', 2, 2), ('read', 0)])


if __name__ == '__main__':
    unittest.main()
