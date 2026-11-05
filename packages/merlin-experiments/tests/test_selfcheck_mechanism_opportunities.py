"""Advisory discovery fixtures; imports/calls never confer execution or target laws."""
import copy
import json
from pathlib import Path

import composition_fixtures as F
from merlin_experiments.phase1.feedback import mechanism_hints as M
from merlin_experiments.phase1.feedback import next_steps as N
from merlin_experiments.phase1.feedback import performance_brief as PB

FIXTURE = json.loads(Path(__file__).with_name('library_pass_fresh4.json').read_text())


def workspace(tmp_path, ids):
    rows = [copy.deepcopy(r) for r in FIXTURE['suggestions']['all_mechanisms'] if r['id'] in ids]
    # Isolate discovery/priority from target capability admission in these fixtures.
    for row in rows:
        row['requirements'] = {k: 'supported' for k in row['requirements']}
        row['status'] = 'supported'
    (tmp_path / 'OPTIMIZATION_SUGGESTIONS.json').write_text(json.dumps({'all_mechanisms': rows}))
    (tmp_path / 'submission').mkdir()
    F.declare(tmp_path, 'layers', F.matmul('original_case'))
    names = ['original_case'] + [row['capsule'] for row in FIXTURE['scalability']['cases']]
    (tmp_path / 'PUBLIC_CAPABILITIES.json').write_text(json.dumps({
        'schema': 'merlin.public_capability_inventory.v1',
        'members': {name: {'workload_role': 'large_kernel'} for name in names}}))
    return tmp_path


def entry():
    return dict(test_case='original_case', cycles=10000, headroom_cycles=9000, bound_by='compute',
                fence_overlap=dict(fence_per_tile=True, fences=100, output_tiles=10))


def test_repeated_feedback_distinguishes_import_call_and_coverage_tuning(tmp_path):
    root = workspace(tmp_path, ('barrier_cleanup', 'copy_pipeline'))
    path = root / 'submission/lower.py'
    path.write_text('from merlin.compile.scheduling.effect_cleanup import cleanup_indices as clean\n')
    first = N.choose(entry(), N.Context(root))
    assert first['choice'] == 'barrier_cleanup'
    assert first['source_call_sites'] == [] and first['opportunity_status'] == 'missing_adapter'
    assert 'runtime use unconfirmed' in first['choice_text']
    path.write_text(path.read_text() + 'def lower(kernel):\n    return clean(kernel)\n')
    second = N.choose(entry(), N.Context(root))
    assert second['choice'] == 'copy_pipeline'  # an untried relevant alternative
    selection = json.loads((root / M.SUGGESTIONS).read_text())
    selection['all_mechanisms'] = [r for r in selection['all_mechanisms'] if r['id'] == 'barrier_cleanup']
    (root / M.SUGGESTIONS).write_text(json.dumps(selection))
    tuning = N.choose(entry(), N.Context(root))
    assert tuning['opportunity_status'] == 'already_observed'
    assert tuning['source_call_sites'] == ['lower.py:3']
    assert 'source call only' in tuning['observation_basis']
    assert 'tune' in tuning['apply'] and '--sim spike' in tuning['apply']
    assert 'runtime application unconfirmed' in tuning['observation_basis']
    # Refresh after edits, rather than retaining stale application/discovery claims.
    path.write_text('import json\n')
    assert N.choose(entry(), N.Context(root))['source_call_sites'] == []


def test_spatial_cache_and_temporary_storage_name_actual_source_and_adapter_prerequisites(tmp_path, monkeypatch):
    root = workspace(tmp_path, ('quantized_spatial_readout_cache', 'temporary_reuse'))
    selection = M.with_source_evidence(M.load_selection(root), root)
    dense = M.for_case(selection, 'original_case', root)
    offered = M.hints(dense, dominant={'name': 'accelerator.storage.accumulator.all'}, limit=64)
    assert 'quantized_spatial_readout_cache' not in {r['id'] for r in offered}
    temporary = next(r for r in offered if r['id'] == 'temporary_reuse')
    assert temporary['opportunity_status'] == 'applicability_unknown'
    assert 'StorageAdapter.project/rewrite' in ' '.join(temporary['prerequisites'])
    assert 'completion' in ' '.join(temporary['prerequisites']).lower()
    assert 'relocation' in ' '.join(temporary['prerequisites']).lower()
    from merlin_experiments.phase1.feedback import case_declarations
    conv = dict(semantic={'semantic_family': 'contraction'}, inputs=[
        dict(name='IFM', role='input', shape=[1, 8, 8, 4]),
        dict(name='W', role='weight', shape=[36, 16])], operation=dict(op='conv2d', attributes=dict(
            ifm='IFM', weight='W', layout='nhwc', kernel=[3, 3, 4, 16], stride=[1, 1], padding=[1]*4)))
    monkeypatch.setattr(case_declarations, 'load', lambda *args: conv)
    plain = M.for_case(selection, 'arbitrary_name', root)
    spatial = next(r for r in M.hints(plain, bound_by='read requests', limit=64)
                   if r['id'] == 'quantized_spatial_readout_cache')
    assert spatial['opportunity_status'] == 'applicability_unknown'
    requires = ' '.join(spatial['prerequisites'])
    assert '3x3 stride-2 pad-1' in requires and 'disjoint' in requires
    assert 'connected staged' in requires and 'ReLU' in requires and 'completion' in requires
    conv['operation']['attributes'].update(pool_size=[2, 2], pool_stride=[2, 2], pool_padding=[1]*4)
    wrong = M.for_case(selection, 'arbitrary_name', root)
    rejected = next(r for r in wrong['all_mechanisms'] if r['id'] == 'quantized_spatial_readout_cache')
    assert rejected['status'] == 'unsupported'
    conv['operation']['attributes']['pool_size'] = [3, 3]
    matched = M.for_case(selection, 'arbitrary_name', root)
    matching = next(r for r in matched['all_mechanisms'] if r['id'] == 'quantized_spatial_readout_cache')
    assert matching['consumer_geometry'] == 'supported'
    # Matching geometry still supplies no staged readout/disjoint-window adapter.
    assert M.opportunity(matching, matched)['opportunity_status'] == 'applicability_unknown'


def test_one_source_scan_per_response_even_with_multiple_cases(tmp_path, monkeypatch):
    root = workspace(tmp_path, ('barrier_cleanup', 'copy_pipeline'))
    calls = []
    original = M.with_source_evidence
    def observe(*args, **kw):
        calls.append(1)
        return original(*args, **kw)
    monkeypatch.setattr(M, 'with_source_evidence', observe)
    document = copy.deepcopy(FIXTURE['scalability'])
    assert len(document['cases']) > 1
    PB.attach(document, document['cases'], workspace=root)
    assert calls == [1]
    assert document['performance_brief'] and document['next_steps']['lines']
    assert 'steps' not in document['next_steps']  # lean reply: each step is its printed line


def test_role_manifest_controls_performance_advice_not_correctness_or_calibration(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import fence_overlap as FO
    from merlin_experiments.phase1.feedback import library_pass as LP

    root = workspace(tmp_path, ('barrier_cleanup', 'copy_pipeline'))
    roles = {'functional': 'correctness', 'calibration': 'calibration',
             'control': 'diagnostic_control', 'representative': 'connected'}
    (root / 'PUBLIC_CAPABILITIES.json').write_text(json.dumps({
        'schema': 'merlin.public_capability_inventory.v1',
        'members': {name: {'workload_role': role} for name, role in roles.items()}}))
    monkeypatch.setattr(FO, 'analyze', lambda *a, **kw: {
        'fence_per_tile': True, 'static': {'fences': 100, 'output_tiles': 10},
        'reading': 'observed completion boundaries'})
    def performance(cycles):
        return {'status': 'PROVISIONAL', 'complete_event_accounting': True,
                'estimated_whole_soc_cycles': cycles,
                'structural_roofline': {'structural_compute_floor_cycles': 10}}
    rows = [{'capsule': name, 'pass': True, 'performance': performance(100000 if name != 'representative' else 100)}
            for name in roles]
    # Model trouble on a calibration control is a diagnostic, not a tuning request.
    rows[1]['performance'] = {'status': 'UNKNOWN', 'reason': 'unsupported model mechanism'}
    for _ in range(2):
        document = {'per_capsule': copy.deepcopy(rows), 'all_pass': True}
        PB.attach(document, document['per_capsule'], workspace=root)
        brief = document['performance_brief']
        assert [e['test_case'] for e in brief['largest_headroom']] == ['representative']
        assert brief['fence_bound']['count'] == 1
        assert brief['without_evidence'] == ['calibration']
        assert brief['unknown_timing']['commands'] == [PB.GSIM_COMMAND.format(name='calibration')]
        assert 'structural_exploration' not in brief
        assert {e['test_case'] for e in brief['diagnostic_controls']} == {'functional', 'control'}
        for diagnostic in brief['diagnostic_controls']:
            assert diagnostic['optimization_status'] == 'diagnostic_only'
            assert 'suggested_mechanisms' not in diagnostic
            assert 'suggested_mechanisms' not in diagnostic['fence_overlap']
        assert document['all_pass'] is True
        for row in document['per_capsule']:
            if row['capsule'] != 'representative':
                assert 'optimization_next' not in row
    context = N.Context(root)
    for name in ('functional', 'calibration', 'control', 'absent'):
        item = {**entry(), 'test_case': name}
        assert N.choose(item, context) is None
        assert LP.block(item, selection=context.selection, suite='core', imports=context.imports,
                        workspace=root, roles=context.roles) is None


def test_missing_or_partial_roles_require_classification_without_blanket_queue(tmp_path):
    root = workspace(tmp_path, ('barrier_cleanup',))
    for inventory in (None, {'schema': 'merlin.public_capability_inventory.v1',
                              'members': {'other': {'workload_role': 'connected'}}}):
        path = root / 'PUBLIC_CAPABILITIES.json'
        if inventory is None:
            path.unlink()
        else:
            path.write_text(json.dumps(inventory))
        rows = [dict(capsule='original_case', **{'pass': True}, performance={
            'status': 'PROVISIONAL', 'complete_event_accounting': True,
            'estimated_whole_soc_cycles': 100000,
            'structural_roofline': {'structural_compute_floor_cycles': 10}}),
            dict(capsule='untimed', **{'pass': True}, performance={'status': 'UNKNOWN'})]
        document = PB.attach({'per_capsule': rows}, rows, workspace=root)
        brief = document['performance_brief']
        assert brief['largest_headroom'] == [] and not brief.get('lowest_utilization')
        assert brief['classification_needed']['cases'] == ['original_case', 'untimed']
        assert brief['diagnostic_controls'][0]['optimization_status'] == 'classification_needed'
        assert 'structural_exploration' not in brief and 'fence_bound' not in brief
        assert 'next_steps' not in document and all('optimization_next' not in r for r in rows)
        assert not brief['unknown_timing'].get('commands')
