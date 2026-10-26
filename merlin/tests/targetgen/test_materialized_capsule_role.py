"""Materialized public programs retain their producer role at native loading."""

import pytest
import yaml

from merlin.targetgen.capsule_common import load_capsule
from merlin.targetgen.contract.schemas import ContractViolation


def test_materialized_role_does_not_relax_capsule_validation(tmp_path):
    cap = dict(name='public_capture', kind='model', label='dev',
               source_role='materialized_performance_capture',
               operation={'op': 'model', 'attributes': {}},
               numeric_policy={'compare': 'tolerance_float', 'dtype': 'f32'},
               expected={'instruction_classes': []}, required_oracle_tiers=['L0', 'L1', 'L2'])
    path = tmp_path / 'capsule.yaml'
    path.write_text(yaml.safe_dump(cap))
    assert load_capsule(tmp_path)['source_role'] == cap['source_role']
    cap['source_role'] = 'unreviewed_role'
    path.write_text(yaml.safe_dump(cap))
    with pytest.raises(ContractViolation, match='source_role'):
        load_capsule(tmp_path)
    cap['source_role'] = 'materialized_performance_capture'
    cap['numeric_policy']['dtype'] = 'unregistered_format'
    path.write_text(yaml.safe_dump(cap))
    with pytest.raises(ContractViolation, match='registered numeric format'):
        load_capsule(tmp_path)
