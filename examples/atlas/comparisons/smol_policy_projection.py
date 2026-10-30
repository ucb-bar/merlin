"""Dynamic SmolVLA Atlas arithmetic replacement and per-panel kernel export."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np

from capture_tinyllama_native import identity, round_bf16_words, safe_path
from tiny_fp8_pack import encode


class AtlasProjection:
    def __init__(self, atlas_tools, out, workers):
        self.root, self.out = safe_path(atlas_tools), safe_path(out)
        self.out.mkdir(exist_ok=False)
        self.pins = [identity(__file__), identity(Path(__file__).with_name('tiny_fp8_pack.py'))]
        helpers = ('smol_policy_kernel', 'atlas_program', 'atlas_vector_arithmetic',
                   'run_model_tile', 'atlas_tile_arithmetic', 'atlas_path_guard', 'atlas_tooling', 'atlas_host_layout')
        for name in helpers:
            self.pins.append(identity(self.root / (name + '.py')))
            cached = sys.modules.get(name)
            if cached is not None and Path(getattr(cached, '__file__', '')).absolute() != self.root / (name + '.py'):
                raise ValueError('Cached Atlas helper is outside selected tools: ' + name)
        sys.path.insert(0, str(self.root))
        module = importlib.import_module('smol_policy_kernel')
        for name in helpers:
            imported = sys.modules.get(name)
            if imported is None or safe_path(imported.__file__) != self.root / (name + '.py'):
                raise ValueError('Atlas helper imported outside selected tools: ' + name)
        self.prepare, self.workers = module.prepare, workers
        self.weights, self.kernels, self.calls = {}, [], []

    @staticmethod
    def physical_payload(words):
        m, n = words.shape
        return b''.join(words[r:r+32, c:c+32].reshape(32, 2, 16).transpose(1, 0, 2).astype('<u2').tobytes()
                        for r in range(0, m, 32) for c in range(0, n, 32))

    def apply(self, name, module, incoming, dtype):
        import torch
        if isinstance(module, torch.nn.Conv2d):
            if (tuple(module.kernel_size) != (16, 16) or tuple(module.stride) != (16, 16)
                    or (module.padding != 'valid' and tuple(module.padding) != (0, 0))
                    or tuple(module.dilation) != (1, 1) or module.groups != 1):
                raise ValueError('Unadmitted vision patch convolution')
            batch, channels, height, width = incoming.shape
            if height % 16 or width % 16:
                raise ValueError('Vision patch input has a tail')
            matrix = incoming.reshape(batch, channels, height//16, 16, width//16, 16).permute(0, 2, 4, 1, 3, 5).reshape(-1, channels*256)
            weight = module.weight.reshape(module.out_channels, -1)
            shape = (batch, height//16, width//16, module.out_channels)
        else:
            matrix = incoming.reshape(-1, module.in_features)
            weight = module.weight
            shape = (*incoming.shape[:-1], module.out_features)
        a_words = round_bf16_words(matrix.detach().float().numpy())
        a_encoded, a_scale = encode(a_words)
        if name not in self.weights:
            b_words = round_bf16_words(weight.detach().float().t().contiguous().numpy())
            self.weights[name] = (*encode(b_words), hashlib.sha256(b_words.tobytes()).hexdigest())
        b_encoded, b_scale, weight_sha = self.weights[name]
        exponent = a_scale + b_scale
        outputs, panels = [], []
        for start in range(0, matrix.shape[0], 128):
            result = self.prepare(a_encoded[start:start+128], b_encoded, exponent, workers=self.workers)
            index = len(self.kernels)
            directory = self.out / f'{index:05d}'
            directory.mkdir()
            m, k, n = result['physical_mkn']
            aa, bb = result['A'], result['B']
            inputs = b''.join(aa[r:r+32, q:q+32].tobytes() for r in range(0, m, 32) for q in range(0, k, 32))
            inputs += b''.join(bb[q:q+32, c:c+32].tobytes() for c in range(0, n, 32) for q in range(0, k, 32))
            expected = self.physical_payload(result['expected'])
            initial = inputs + b'\xa5'*256 + b'\xcd'*len(expected) + b'Z'*256
            if len(initial) != result['dram_bytes']:
                raise ValueError('Projection DRAM packing extent differs')
            (directory / 'program.S').write_text(result['source'])
            (directory / 'initial-dram.bin').write_bytes(initial)
            (directory / 'expected-output.bin').write_bytes(expected)
            row = {key: result[key] for key in ('logical_mkn', 'physical_mkn', 'output_offset', 'dram_bytes', 'requested_dma', 'resident_a', 'resident_b')}
            row.update(index=index, call=len(self.calls), module=name, logical_row_start=start,
                       a_scale=a_scale, b_scale=b_scale, output_rescale_exponent=exponent,
                       weight_bf16_sha256=weight_sha, files={path.name: identity(path) for path in directory.iterdir()})
            (directory / 'kernel.json').write_text(json.dumps(row, indent=2)+'\n')
            self.kernels.append(row)
            panels.append(index)
            outputs.append(result['logical_output'])
        words = np.concatenate(outputs)
        # Source dtype is preserved after explicit Atlas BF16 result conversion.
        output = torch.from_numpy(words.copy()).view(torch.bfloat16).to(dtype).reshape(shape)
        if module.bias is not None:
            output = output + module.bias
        if isinstance(module, torch.nn.Conv2d):
            output = output.permute(0, 3, 1, 2).contiguous()
        self.calls.append({'module': name, 'panels': panels, 'input_shape': list(incoming.shape),
                           'output_shape': list(output.shape), 'dtype': str(output.dtype)})
        return output

    def finish(self):
        for pin in self.pins:
            identity(pin['path'], pin['sha256'])
        return {'kind': 'ordered_Atlas_FP8_BF16_arithmetic_reference_not_device_execution',
                'calls': self.calls, 'kernels': self.kernels, 'source_bookends': self.pins,
                'source_bookends_equal': True, 'fresh_model_executed': False, 'workers': self.workers}
