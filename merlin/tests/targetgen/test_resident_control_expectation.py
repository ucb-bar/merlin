"""A one-use control cannot promise repeated resident reuse."""

import pytest

@pytest.mark.parametrize('rows, required', [([16], False), ([64], True), ([16, 16], True)])
def test_reuse_requirement_follows_actual_weight_uses(monkeypatch, rows, required):
    from merlin.targetgen.corpus_spec import CorpusBinding, build_resident_reuse
    binding = CorpusBinding(target='test', tile_dim=16, operand_dtype='int8', accum_dtype='i32',
                            integer=True, tiers=['L2'], compare='exact_int')
    monkeypatch.setattr(CorpusBinding, 'classes_for', lambda self, **kw: ['MVIN', 'MVOUT'])
    capsule, source = build_resident_reuse({
        'name': 'control', 'kind': 'model_slice', 'source_role': 'derived_sweep',
        'source_reference': 'public shape control', 'matmuls': [
            {'lhs': f'A{i}', 'out': f'Y{i}', 'M': m} for i, m in enumerate(rows)]}, binding)
    assert capsule['expected']['modes']['resident_reuse'] is required
    assert source.count('merlin_iface.commit') == len(rows)

