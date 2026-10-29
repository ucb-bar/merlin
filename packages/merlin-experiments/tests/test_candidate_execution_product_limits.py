"""Independent bounded namespace controls; neutral bytes grant no target proof."""
import json
import os

import pytest
from merlin_experiments.phase1.feedback import candidate_observations as O
from merlin_experiments.phase1.feedback.candidate_execution_products import reopen_execution_products
from test_candidate_execution_products import retained as retained

from merlin.targetgen import executed_program as E


@pytest.mark.parametrize('damage', ['aggregate', 'record_count', 'depth', 'normalized_alias', 'members_list', 'fifo'])
def test_complete_namespace_is_admitted_before_any_payload_collector(retained, monkeypatch, damage):
    root, grade, expected = retained
    record_path = next((root / 'executed').glob('*/*/executed_program.json'))
    if damage == 'aggregate':
        # Sparse metadata-only payloads: every leaf fits64MiB but joint144MiB does not.
        for i in range(3):
            with (record_path.parent / ('unselected-' + str(i) + '.bin')).open('wb') as out:
                out.truncate(48 << 20)
    elif damage == 'record_count':
        for i in range(64):
            (record_path.parent.parent / ('unretained_' + str(i) + '.json')).write_text('{}')
    elif damage == 'depth':
        directory = record_path.parent / 'nested'
        directory.mkdir()
        (directory / 'unused').write_bytes(b'neutral')
    elif damage == 'fifo':
        os.mkfifo(record_path.parent / 'not-a-payload')
    else:
        record = json.loads(record_path.read_bytes())
        if damage == 'normalized_alias':
            record['elf']['path'] = record['elf']['path'].replace('/', '//', 1)
        else:
            record['compiler_artifacts']['members'] = ['build', 'command_buffer']
        record_path.write_text(json.dumps(record))

    def forbidden(_root):
        pytest.fail('collector reached a refused complete namespace')

    monkeypatch.setattr(E, 'collect', forbidden)
    with pytest.raises(ValueError):
        reopen_execution_products(root, grade, candidate_sha256=expected)


def test_coherent_record_cannot_misdeclare_selected_payload_length(retained):
    root, grade, expected = retained
    record_path = next((root / 'executed').glob('*/*/executed_program.json'))
    record = json.loads(record_path.read_bytes())
    record['elf']['bytes'] += 1
    record_path.write_text(json.dumps(record))
    grade['compiler_source']['executed_programs'] = E.collect(root)
    with pytest.raises(ValueError, match='changed'):
        reopen_execution_products(root, grade, candidate_sha256=expected)
    report = O.observe_generated_artifacts(root, original_grade=grade, candidate_sha256=expected)
    assert report['product_collection']['unavailable_roles'] == ['original_retained_execution_join']
    assert report['source']['status'] == 'UNKNOWN'
