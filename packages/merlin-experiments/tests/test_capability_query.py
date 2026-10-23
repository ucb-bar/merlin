"""The served helper runs without private modules and never dumps the catalog."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from merlin_experiments.phase1 import capability_inventory as C
from merlin_experiments.phase1 import capability_query as Q


def data():
    return {'schema': C.SCHEMA, 'total': 2,
        'cohorts': {'original_correctness': 1, 'scalability_performance': 1},
        'capability_groups': {'abcdef123456' + '0'*52: {
            'requirements': {'family': 'contraction', 'operations': ['linalg.matmul'],
                             'input_dtypes': ['i8'], 'numeric_policy': {'compare': 'exact_int'}},
            'members': ['small', 'large'], 'small_canary_candidate': 'small',
            'large_generalization_candidate': 'large', 'selection_basis': 'declared volume'}},
        'members': {'small': {'cohort': 'original_correctness', 'family': 'contraction',
            'development_engines': ['spike', 'gsim'], 'input_tensors': [{'shape': [16, 16], 'dtype': 'i8'}],
            'source_analysis': {'status': 'parsed', 'operations': {'linalg.matmul': 1},
                'graph': {'nodes': 3, 'operations': ['a', 'b', 'c'],
                          'producer_consumer_edges': [[0, 1], [1, 2]], 'shared_producers': []}}}}}


def test_compact_queries_and_explicit_pagination():
    value = data()
    groups = Q.query(value)
    assert groups['groups']['total'] == 1
    assert groups['groups']['items'][0]['group'] == 'abcdef123456'
    group = Q.query(value, group='abc', limit=1)
    assert group['members'] == {'total': 2, 'offset': 0, 'items': ['small'], 'next_offset': 1}
    assert Q.query(value, group='abc', offset=1, limit=1)['members']['items'] == ['large']
    case = Q.query(value, capsule='small')
    assert case['development_engines'] == ['spike', 'gsim']
    assert case['graph_summary']['producer_consumer_edges'] == 2
    assert 'graph' not in case['source']
    with pytest.raises(ValueError, match='exactly one'):
        Q.query(value, group='unknown')
    with pytest.raises(ValueError, match='selected public'):
        Q.query(value, capsule='hidden')
    with pytest.raises(ValueError, match='limit'):
        Q.query(value, limit=1000)


def test_standalone_helper_only_needs_adjacent_inventory(tmp_path):
    script = tmp_path / C.QUERY
    script.write_bytes(Path(Q.__file__).read_bytes())
    inventory = tmp_path / C.FILENAME
    inventory.write_text(json.dumps(data()))
    before = inventory.read_bytes()
    result = subprocess.run([sys.executable, '-I', '-S', str(script), '--capsule', 'small'],
                            text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)['capsule'] == 'small'
    assert len(result.stdout.encode()) < 1024
    assert inventory.read_bytes() == before
    result = subprocess.run([sys.executable, '-I', '-S', str(script), '--inventory', '/not/public'],
                            text=True, capture_output=True)
    assert result.returncode == 2 and not result.stdout


def test_symlink_and_oversized_inventory_refused(tmp_path):
    private = tmp_path / 'other.json'
    private.write_text(json.dumps(data()))
    path = tmp_path / C.FILENAME
    path.symlink_to(private)
    with pytest.raises(ValueError, match='ordinary'):
        Q.read_inventory(tmp_path)
    path.unlink()
    with path.open('wb') as stream:
        stream.truncate(Q.MAX_INPUT_BYTES + 1)
    with pytest.raises(ValueError, match='bounded'):
        Q.read_inventory(tmp_path)
