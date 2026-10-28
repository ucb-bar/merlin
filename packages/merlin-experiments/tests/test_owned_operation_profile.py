"""Owned execution keeps optional diagnostics separate from numerical evidence."""
import hashlib
import json

import pytest

from merlin.targetgen.owned_host_model_execution import _operation_profile


def record(tmp_path):
    path = tmp_path / 'table.json'
    path.write_text(json.dumps([
        {'id': 0, 'mlir_op': 'linalg.generic', 'body_ops': ['arith.addf']},
        {'id': 1, 'mlir_op': 'func.call', 'callee': '@selected_device'},
    ]))
    return {'build': {'inputs': {'op_profile': True}, 'output': {'op_profile_table': {
        'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }}}}


TRACE = b'PROF 0 70 1\nPROF 1 30 1\nMETRIC prof_total_ticks 100\nMETRIC prof_marks 3\nDONE\n'


def test_observations_are_compact_and_do_not_rank_functional_counters(tmp_path):
    report = _operation_profile(record(tmp_path), TRACE)
    assert report['status'] == 'observed'
    assert report['operation_counter_ticks'] == 100
    assert sum(report['category_counter_ticks'].values()) == 100
    assert report['unit'] == 'raw_counter_ticks'
    assert report['instrumented'] and not report['performance_ranking_eligible']
    assert not report['timing_qualified']
    assert 'operations' not in report and 'callee' not in str(report)


def test_missing_optional_diagnostics_are_unavailable_not_a_numeric_failure(tmp_path):
    report = _operation_profile(record(tmp_path), b'DONE\n')
    assert report['status'] == 'unavailable'
    assert not report['timing_qualified']
    assert _operation_profile({'build': {'inputs': {}, 'output': {}}}, TRACE) is None


def test_changed_profile_table_is_not_reinterpreted(tmp_path):
    selected = record(tmp_path)
    (tmp_path / 'table.json').write_text('[]')
    with pytest.raises(ValueError, match='changed'):
        _operation_profile(selected, TRACE)
