"""Public PyTorch attention semantics for ordinary capture and compiler exploration.

Profiles vary geometry and attention semantics, never target schedules. Quantization,
placement and numerical admission belong to the ordinary selected capture pipeline.
All attention outputs and unexpanded K/V tensors remain observable.
"""

import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

PROFILE_SCHEMA = 'merlin.iteration_workload_profile.v1'
WORKLOAD_ID = 'attention_variants'
DEFAULTS = dict(kind='self', batch=1, tokens=4, memory_tokens=4,
                heads=4, kv_heads=2, head_width=8, prefix=0, causal=True)
MAX_ELEMENTS = 16_000_000


def validate_profile(profile):
    if (type(profile) is not dict or profile.get('schema') != PROFILE_SCHEMA
            or profile.get('workload_id') != WORKLOAD_ID
            or set(profile) - {'schema', 'workload_id'} - set(DEFAULTS)):
        raise ValueError('attention variants require a closed source profile')
    p = {**DEFAULTS, **profile}
    if p['kind'] not in ('self', 'cross') or type(p['causal']) is not bool:
        raise ValueError('attention kind and causal flag must be explicit source semantics')
    for key in ('batch', 'tokens', 'memory_tokens', 'heads', 'kv_heads', 'head_width', 'prefix'):
        if type(p[key]) is not int or not (0 if key == 'prefix' else 1) <= p[key] <= 8192:
            raise ValueError(f'invalid attention source extent: {key}')
    if p['heads'] > 32 or p['heads'] % p['kv_heads']:
        raise ValueError('query heads must be a multiple of KV heads')
    if p['batch'] > 16 or p['kind'] == 'cross' and p['prefix']:
        raise ValueError('cross-attention takes complete memory; prefix is for self-attention only')
    hidden, kv_hidden = p['heads'] * p['head_width'], p['kv_heads'] * p['head_width']
    length = p['tokens'] + p['prefix'] if p['kind'] == 'self' else p['memory_tokens']
    elements = 2 * hidden**2 + 2 * hidden * kv_hidden
    elements += p['batch'] * (p['tokens'] * hidden + 2 * length * kv_hidden)
    elements += p['batch'] * p['heads'] * p['tokens'] * length
    if p['kind'] == 'cross':
        elements += p['batch'] * length * hidden
    if elements > MAX_ELEMENTS:
        raise ValueError('attention source exceeds its finite logical element budget')
    return p


def values(shape, offset=0):
    count = math.prod(shape)
    return (((torch.arange(count, dtype=torch.int64) + offset) % 37 - 18).float() / 64).reshape(shape)


class Attention(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.batch, self.tokens = p['batch'], p['tokens']
        self.heads, self.kv_heads, self.width = p['heads'], p['kv_heads'], p['head_width']
        self.kind, self.prefix = p['kind'], p['prefix']
        self.source_tokens = p['tokens'] if self.kind == 'self' else p['memory_tokens']
        self.hidden = self.heads * self.width
        self.query = nn.Linear(self.hidden, self.hidden, bias=False)
        self.key = nn.Linear(self.hidden, self.kv_heads * self.width, bias=False)
        self.value = nn.Linear(self.hidden, self.kv_heads * self.width, bias=False)
        self.output = nn.Linear(self.hidden, self.hidden, bias=False)
        length = self.source_tokens + self.prefix
        positions = torch.arange(self.tokens) + self.prefix
        allowed = torch.arange(length)[None, :] <= positions[:, None]
        mask = torch.where(allowed, 0.0, float('-inf')) if p['causal'] else torch.zeros(self.tokens, length)
        self.register_buffer('mask', mask)

    def share(self, value):
        length = value.shape[2]
        return value[:, :, None].expand(self.batch, self.kv_heads,
            self.heads // self.kv_heads, length, self.width).reshape(self.batch, self.heads, length, self.width)

    def forward(self, hidden, *state):
        source = hidden if self.kind == 'self' else state[0]
        query = self.query(hidden).reshape(self.batch, self.tokens, self.heads, self.width).transpose(1, 2)
        key = self.key(source).reshape(self.batch, self.source_tokens, self.kv_heads, self.width).transpose(1, 2)
        value = self.value(source).reshape(self.batch, self.source_tokens, self.kv_heads, self.width).transpose(1, 2)
        if self.prefix:
            key = torch.cat((state[0], key), dim=2)
            value = torch.cat((state[1], value), dim=2)
        scores = (query @ self.share(key).transpose(-2, -1)) / math.sqrt(self.width)
        probabilities = torch.softmax(scores + self.mask, dim=-1)
        context = (probabilities @ self.share(value)).transpose(1, 2).reshape(hidden.shape)
        return self.output(context), key, value


def build(profile):
    p = validate_profile(profile)
    hidden = p['heads'] * p['head_width']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(83)
        model = Attention(p).eval()
        with torch.no_grad():
            for index, layer in enumerate((model.query, model.key, model.value, model.output)):
                layer.weight.copy_(values(layer.weight.shape, index * 7) / math.sqrt(hidden))
    inputs = (values((p['batch'], p['tokens'], hidden)),)
    if p['kind'] == 'cross':
        inputs += (values((p['batch'], p['memory_tokens'], hidden), 5),)
    elif p['prefix']:
        cache = (p['batch'], p['kv_heads'], p['prefix'], p['head_width'])
        inputs += (values(cache, 11), values(cache, 17))
    return model, inputs


def get_model_and_inputs():
    path = Path(__file__).with_name('profile.json')
    if path.is_symlink() or path.exists() and not path.is_file():
        raise ValueError('attention profile must be an adjacent ordinary file')
    raw = path.read_bytes() if path.exists() else None
    profile = json.loads(raw) if raw is not None else dict(schema=PROFILE_SCHEMA, workload_id=WORKLOAD_ID)
    model, inputs = build(profile)
    model.session_provenance = dict(workload_id=WORKLOAD_ID, workload_role='iteration',
        source_role='handauthored_compiler_test', input_source='seeded_synthetic',
        attention_kind=validate_profile(profile)['kind'],
        **({'profile_sha256': hashlib.sha256(raw).hexdigest()} if raw else {'default_profile': True}))
    return model, inputs
