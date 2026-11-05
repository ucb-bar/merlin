"""Attention stress sources preserve masks, grouped heads and every K/V output."""
import importlib.util
import json
from pathlib import Path

import pytest
from merlin.common.paths import repo_root
from merlin_experiments.phase0 import workload_roster

torch = pytest.importorskip('torch')
SOURCE = repo_root() / 'examples/workloads/attention_variants'


def load(root=SOURCE):
    spec = importlib.util.spec_from_file_location('attention_variant_source', Path(root) / 'loader.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile(**options):
    return dict(schema='merlin.iteration_workload_profile.v1', workload_id='attention_variants', **options)


def test_all_roster_variants_stage_export_and_match_independent_attention(tmp_path):
    rows = workload_roster.stage(SOURCE / 'phase0-workloads.yaml', tmp_path / 'staged')['performance_applications']
    assert len(rows) == 8
    for name, row in rows.items():
        module = load(row['workload_root'])
        model, inputs = module.get_model_and_inputs()
        p = module.validate_profile(json.loads((Path(row['workload_root']) / 'profile.json').read_text()))
        with torch.no_grad():
            expected = model(*inputs)
            exported = torch.export.export(model, inputs)
            actual = exported.module()(*inputs)
            source = inputs[0] if p['kind'] == 'self' else inputs[1]
            q = model.query(inputs[0]).reshape(p['batch'], p['tokens'], p['heads'], p['head_width']).transpose(1, 2)
            length = p['tokens'] if p['kind'] == 'self' else p['memory_tokens']
            k = model.key(source).reshape(p['batch'], length, p['kv_heads'], p['head_width']).transpose(1, 2)
            v = model.value(source).reshape(p['batch'], length, p['kv_heads'], p['head_width']).transpose(1, 2)
            if p['prefix']:
                k, v = torch.cat((inputs[1], k), 2), torch.cat((inputs[2], v), 2)
            # Independent attention owner: explicit repeat_interleave plus SDPA,
            # rather than the source expand/reshape + softmax implementation.
            repeats = p['heads'] // p['kv_heads']
            context = torch.nn.functional.scaled_dot_product_attention(q,
                k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1),
                attn_mask=model.mask, dropout_p=0)
            independent = model.output(context.transpose(1, 2).reshape(inputs[0].shape))
        assert len(actual) == 3 and all(torch.isfinite(x).all() for x in actual), name
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(expected[0], independent, rtol=2e-5, atol=1e-7)
        torch.testing.assert_close(expected[1], k, rtol=0, atol=0)
        torch.testing.assert_close(expected[2], v, rtol=0, atol=0)
        assert model.session_provenance['profile_sha256'] == row['profile_sha256']


def test_causal_future_mask_cache_custody_and_cross_memory_dependence():
    module = load()
    model, inputs = module.build(profile(tokens=4, prefix=3))
    saved = tuple(x.clone() for x in inputs)
    changed = inputs[0].clone()
    changed[:, -1] += torch.arange(changed.shape[-1]) / 8
    with torch.no_grad():
        before = model(*inputs)
        after = model(changed, *inputs[1:])
    torch.testing.assert_close(before[0][:, :-1], after[0][:, :-1], rtol=0, atol=0)
    for a, b in zip(inputs, saved, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(before[1][:, :, :3], inputs[1], rtol=0, atol=0)
    model, inputs = module.build(profile(kind='cross', tokens=3, memory_tokens=7, causal=False))
    with torch.no_grad():
        before, after = model(*inputs), model(inputs[0], inputs[1] + .5)
    assert not torch.equal(before[0], after[0])
    assert not torch.equal(before[1], after[1])


@pytest.mark.parametrize('changes', [dict(heads=3), dict(kv_heads=0), dict(batch=True),
    dict(kind='cross', prefix=1), dict(causal='false'), dict(heads=32, head_width=8192),
    dict(tokens=0), dict(target_schedule='private')])
def test_invalid_geometry_or_target_schedule_is_not_a_source_profile(changes):
    with pytest.raises(ValueError):
        load().validate_profile(profile(**changes))
