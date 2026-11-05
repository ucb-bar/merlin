"""Selected library availability never substitutes for actual target bindings."""

import json

import pytest
from merlin_experiments.phase1.feedback import mechanism_hints as hints
from test_reference_construction import guided as guided
from test_reference_construction import workspace as workspace
from test_selfcheck_mechanism_opportunities import workspace as make_workspace


def selection(tmp_path, *, proposal=False, consumer=False):
    root = make_workspace(tmp_path, ('array_weight_retention', 'strided_tile_loads'))
    adapter = root / 'submission/tools/library_pass_adapter.py'
    adapter.parent.mkdir()
    adapter.write_text('# The ordinary apply tool independently validates these hooks.\n')
    code = ('from merlin.compile.scheduling.block_schedule import Compute\n'
            'def project(call):\n'
            '    return Compute(0, (0, 0), 1, 1, True)\n')
    if proposal:
        code += ('from merlin.compile.scheduling.dense_array_retention import propose_dense_array_retention\n'
                 'def propose(kernel, adapter, law):\n'
                 '    return propose_dense_array_retention(kernel, adapter, law)\n')
    if consumer:
        code += ('from merlin.compile.scheduling.dense_array_retention import verify_dense_array_retention\n'
                 'def check(*args, **kwargs):\n'
                 '    return verify_dense_array_retention(*args, **kwargs)\n')
    (root / 'submission/lowering.py').write_text(code)
    found = hints.with_source_evidence(hints.load_selection(root), root)
    return root, found


def row(found, name):
    return next(r for r in found['all_mechanisms'] if r['id'] == name)


def test_proposal_only_always_fresh_adapter_keeps_exact_retained_mode_gap(tmp_path):
    _, found = selection(tmp_path, proposal=True)
    observed = hints.opportunity(row(found, 'array_weight_retention'), found)
    assert observed['opportunity_status'] == 'already_observed'  # Source call, never achieved mechanism.
    contract = observed['adapter_contract']
    assert contract['consumer_binding'] == 'not_observed'
    assert contract['target_law'] == 'unverified_by_source_discovery'
    assert contract['gap']['kind'] == 'adapter_gap'
    requires = ' '.join(contract['required_bindings'])
    assert 'weight_row/weight_block=None' in requires and 'fresh_weights=False' in requires
    assert 'always-fresh projector' in requires and 'STAY/output-descriptor' in requires
    assert 'schedule.loop_transform' in requires and 'schedule.plan' in requires
    assert 'checking_kernel is proof-only' in requires
    assert 'verify_dense_array_retention' in str(contract['verification_consumers'])


def test_checked_consumer_call_remains_evidence_gap_without_emission_or_target_law(tmp_path):
    _, found = selection(tmp_path, proposal=True, consumer=True)
    contract = hints.opportunity(row(found, 'array_weight_retention'), found)['adapter_contract']
    assert contract['consumer_binding'] == 'source_call_only'
    assert contract['observed_consumer_call_sites'][0]['sites']
    assert contract['gap']['kind'] == 'evidence_gap'
    assert 'evidence.actual_emission' in contract['gap']['goals']
    assert contract['target_law'] == 'unverified_by_source_discovery'


def test_generic_adapter_presence_does_not_supply_wide_address_law_or_final_mapping(tmp_path):
    _, found = selection(tmp_path)
    observed = hints.opportunity(row(found, 'strided_tile_loads'), found)
    assert observed['opportunity_status'] == 'candidate'
    contract = observed['adapter_contract']
    assert contract['consumer_binding'] == 'not_observed' and contract['gap']['kind'] == 'adapter_gap'
    requires = ' '.join(observed['prerequisites'])
    for name in ('TileLoadAddressLaw', 'TileLoadMapping', 'source_pitch_bytes', 'local_block_stride',
                 'encode_band', 'completion epoch', 'schedule.plan'):
        assert name in requires
    assert 'packet field width is not a limit' in requires
    assert 'neither execution ordering nor completion removal' in requires
    assert any('admit_panel_tile_loads' in api for api in contract['verification_consumers'])


def test_unknown_selected_capability_stays_unknown_and_unselected_api_gets_no_route(tmp_path):
    _, found = selection(tmp_path)
    selected = row(found, 'array_weight_retention')
    selected['requirements']['compute.retained_array'] = 'unknown'
    observed = hints.opportunity(selected, found)
    assert observed['opportunity_status'] == 'applicability_unknown'
    assert 'selected array law' in observed['next_action']
    assert hints.adapter_contract(dict(api='foreign.backend.dense_array_retention'), found) is None


@pytest.mark.parametrize('mechanism', ['array_weight_retention', 'strided_tile_loads'])
def test_reference_construction_emits_actionable_binding_gap(workspace, guided, mechanism):
    from merlin_experiments.phase1 import operation_optimizations as operations
    from merlin_experiments.phase1.feedback import next_steps, reference_construction, reference_experiments

    path = workspace / hints.SUGGESTIONS
    data = json.loads(path.read_text())
    selected = next(row for row in data['all_mechanisms'] if row['id'] == mechanism)
    selected.update(status='supported', requirements={})
    path.write_text(json.dumps(data))
    (workspace / operations.FILENAME).write_text(json.dumps(dict(schema=operations.SELECTION_SCHEMA,
        entries=[dict(id='contraction', mechanisms=[dict(kind='catalog', id=mechanism)])])))
    with reference_experiments.invocation(guided['path'], guided['sha'], tasks=False):
        report = reference_construction.construction('MM_case', next_steps.Context(workspace))
    alternative, = report['alternatives']
    assert alternative['adapter_contract']['consumer_binding'] == 'not_observed'
    gap = next(g for g in report['gaps'] if g.get('mechanism') == mechanism)
    assert gap['kind'] == 'adapter_gap' and gap['api'] == alternative['api']
    assert 'schedule.consumed' in gap['goals']
    assert any('selected' in line and 'law' in line and 'schedule.plan' in line for line in report['lines'])
    assert report['mechanism_status'] == 'unknown' and report['ready_for_performance'] is False
