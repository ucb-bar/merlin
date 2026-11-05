"""Saved-data recipes must reject altered evidence and mismatched comparisons."""
from __future__ import annotations

import hashlib
import importlib.util
import json

import pytest

from merlin.common.paths import repo_root

SPEC = importlib.util.spec_from_file_location(
    'recompute_paper_results', repo_root() / 'build_tools/scripts/recompute_paper_results.py'
)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def measured(**changes):
    row = dict(rankable='True', median_cycles='20', cycle_samples='[10, 20, 30]',
               mismatches='0', guard_failures='0', case_id='example', hardware_id='selected',
               elf_sha256='image', input_profile='fixed', input_sha256s='inputs', reference_sha256s='reference')
    row.update(changes)
    return row


def test_mismatching_saved_median_is_rejected():
    with pytest.raises(ValueError, match='saved cycle median'):
        R.validated_cycles(measured(median_cycles='10'))


@pytest.mark.parametrize('field', ['mismatches', 'guard_failures'])
def test_rankable_flag_cannot_override_failed_checks(field):
    with pytest.raises(ValueError, match='nonzero'):
        R.validated_cycles(measured(**{field: '1'}))


def test_unrankable_case_remains_unmeasured():
    assert R.validated_cycles(measured(rankable='False')) is None


@pytest.mark.parametrize('field', ['hardware_id', 'elf_sha256', 'input_sha256s', 'reference_sha256s'])
def test_same_image_control_rejects_different_bindings(field):
    with pytest.raises(ValueError, match=field):
        R.same_image_pair(measured(), measured(**{field: 'different'}))


def test_input_checksum_prevents_silent_record_changes(tmp_path):
    original = b'{"cycles": 123}'
    selected = tmp_path / 'saved.json'
    selected.write_bytes(original)
    registry = {'inputs': {'run': {'file': 'saved.json', 'bytes': len(original),
                                   'sha256': hashlib.sha256(original).hexdigest()}}}
    source = R.Inputs(tmp_path, registry)
    assert source.json('run') == {'cycles': 123}
    selected.write_bytes(b'{"cycles": 321}')
    with pytest.raises(ValueError, match='checksum differs'):
        source.json('run')


def test_input_path_cannot_escape_selected_root(tmp_path):
    source = R.Inputs(tmp_path, {'inputs': {'run': {'file': '../outside.json'}}})
    with pytest.raises(ValueError, match='stay within'):
        source.path('run')


def test_speedup_is_geomean_of_case_ratios_not_ratio_of_means():
    assert R.gm([2, .5]) == pytest.approx(1)
    with pytest.raises(ValueError):
        R.gm([2, 0])


def supplied_fixture(rows):
    class Data:
        def csv(self, key):
            assert key == 'frozen-master_long'
            return rows
    taxonomy = {case: {'case_family': 'main16'} for case in ('a', 'b')}
    cells = [{'system': system, 'case': case, 'cycles': cycles}
             for system, cycles in [('F10-FULL-NOREF-NOCCA', 20), ('CCA-EXO-R3', 10)]
             for case in taxonomy]
    return Data(), cells, taxonomy


def test_supplied_comparison_keeps_complete_denominator_and_ratio_direction():
    rows = [{'system': 'agentic_external', 'case_id': case, 'reported_cycles': cycles}
            for case, cycles in [('a', '20'), ('b', '5')]]
    result = R.supplied_comparison(*supplied_fixture(rows))
    assert result['comparisons']['CCA-EXO-R3']['speed_geomean'] == pytest.approx(1)
    assert result['comparisons']['F10-FULL-NOREF-NOCCA']['speed_geomean'] == pytest.approx(.5)
    assert result['refinement_over_initial'] == pytest.approx(2)
    with pytest.raises(ValueError, match='declared development panel'):
        R.supplied_comparison(*supplied_fixture(rows[:1]))
    with pytest.raises(ValueError, match='conflicting'):
        R.supplied_comparison(*supplied_fixture(rows + [dict(rows[0], reported_cycles='21')]))
