"""Atlas attention capture adapter, with a persistent separate-interpreter worker."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from capture_tinyllama_native import array_identity, identity, safe_path
from compare_tinyllama_attention import raw_array


def fixed_exponent(words):
    values = (words.astype(np.uint32) << 16).view(np.float32)
    maximum = float(np.abs(values).max())
    if not np.isfinite(values).all():
        raise ValueError('attention operands must be finite BF16')
    return math.ceil(math.log2(maximum / 448)) if maximum else 0


class AttentionCapture:
    def __init__(self, root, python, script, save, torch, records):
        self.root, self.save, self.torch, self.records = root, save, torch, records
        self.calls, self.saved, self.mode = [], {}, 'capture'
        records.append(identity(script))
        worker_env = dict(os.environ)
        worker_env.pop('PYTHONPATH', None)
        self.worker = subprocess.Popen([str(python), '-B', str(script)], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, text=True, bufsize=1, env=worker_env)

    def close(self):
        if self.worker.poll() is None:
            self.worker.stdin.write(json.dumps({'command': 'close'}) + '\n')
            self.worker.stdin.flush()
            answer = self.worker.stdout.readline()
            self.worker.wait(timeout=30)
            if not answer or json.loads(answer).get('status') != 'closed':
                raise RuntimeError('attention worker did not close cleanly')

    def apply(self, step, module, query, key, value, attention_mask, scaling):
        torch = self.torch
        layer = int(module.layer_idx)
        words = [raw_array(tensor[0], torch) for tensor in (query, key, value)]
        rows, keys = query.shape[2], key.shape[2]
        if query.shape != (1, 32, rows, 64) or key.shape != (1, 4, keys, 64) or value.shape != key.shape:
            raise ValueError('Tiny post-RoPE attention geometry changed')
        allowed = np.arange(keys)[None, :] <= np.arange(rows)[:, None] + keys - rows
        if attention_mask is not None:
            actual = attention_mask[0, 0, :, :keys].float().numpy() == 0
            if not np.array_equal(actual, allowed):
                raise ValueError('captured Tiny mask is not the declared causal mask')
        signatures = [array_identity(array) for array in words]
        key_id = (step, layer)
        if self.mode == 'replay':
            old, output = self.saved[key_id]
            if signatures != old:
                raise ValueError('replay post-RoPE Q/K/V predecessor changed')
        else:
            stem = f'calls/attention-{step:02d}-{layer:02d}'
            descriptors = {role + '_bf16': self.save(stem + '-' + role + '.bin', array)
                           for role, array in zip(('q', 'k', 'v'), words)}
            config = dict(query_rows=rows, keys=keys, query_heads=32, kv_heads=4, head_dim=64,
                          mask='causal', query_start=keys - rows, scale=scaling,
                          q_exponent=fixed_exponent(words[0]), k_exponent=fixed_exponent(words[1]),
                          v_exponent=fixed_exponent(words[2]), p_exponent=-8)
            request = dict(command='attention', root=str(self.root), out=str(self.root / stem), config=config, **descriptors)
            print(json.dumps(dict(milestone='attention_start', step=step, layer=layer, rows=rows, keys=keys)),
                  file=sys.stderr, flush=True)
            self.worker.stdin.write(json.dumps(request) + '\n')
            self.worker.stdin.flush()
            line = self.worker.stdout.readline()
            if not line:
                raise RuntimeError('attention worker terminated without a result')
            answer = json.loads(line)
            if answer['status'] != 'ok':
                raise RuntimeError('attention worker failed: ' + answer.get('error', 'unknown'))
            record = answer['record']
            print(json.dumps(dict(milestone='attention_complete', step=step, layer=layer,
                  seconds=record['execution']['elapsed_seconds'], instructions=record['execution']['instructions'])),
                  file=sys.stderr, flush=True)
            for source in record['implementation_sources']:
                if source not in self.records:
                    self.records.append(source)
            record.update(step=step, layer=layer, index=len(self.calls),
                          name=f'model.layers.{layer}.self_attn.attention', site=f'model.layers.{layer}.self_attn.attention')
            desc = record['output_bf16']
            path = safe_path(self.root / desc['file'])
            identity(path, desc['sha256'])
            output = np.fromfile(path, dtype=np.uint16).reshape(desc['shape'])
            self.saved[key_id] = (signatures, output.copy())
            self.calls.append(record)
        # Native eager attention returns [batch, rows, heads, dimension].
        return torch.from_numpy(output.copy()).view(torch.bfloat16).unsqueeze(0).transpose(1, 2).contiguous(), None
