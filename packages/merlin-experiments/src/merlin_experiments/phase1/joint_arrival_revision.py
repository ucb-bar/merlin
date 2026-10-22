"""Narrow, provisional arrival-pipeline review; no runtime hot swap or fit grant."""
from __future__ import annotations

import copy
import json

from .feedback import joint_performance as PERF
from .joint_structural_revision import _verify_file

_HARDWARE = frozenset((
    'reader_beat_bytes', 'reader_translation_queue_entries', 'reader_tlb_queue_entries',
    'reader_beat_merger_stages', 'identity_scale_input_stages', 'identity_scale_output_stages',
    'scratchpad_singleported', 'reader_transaction_entries', 'identity_scale_queue_entries',
))
_SELECTORS = {
    'target.scheduling.scratchpad_row_loads': 1,
    'target.scheduling.scratchpad_request_admission': 1,
    'target.assumptions.scratchpad_load_translation_hits': 1,
    'target.assumptions.scratchpad_load_fifo_responses': 1,
    'target.assumptions.reader_a_ready_with_transaction_credit': 1,
}
_FETCH_FIELDS = frozenset(('beat_bytes', 'first_response_cycles', 'beat_interval_cycles',
    'line_visible_after_last_cycles', 'lookup_response_cycles', 'request_accept_cycles'))


def _model_delta(before, after, facts, selectors, fetch):
    if set(facts) != _HARDWARE or selectors != _SELECTORS:
        raise ValueError('arrival review differs from named mechanism fields')
    expected = copy.deepcopy(before)
    target = expected['target']
    if (target['scheduling'].get('version') != 5
            or target['scheduling'].get('conservative_local_store_holds') != 1
            or target['scheduling'].get('require_fenced_phases') != 1):
        raise ValueError('arrival correction requires retained provisional dependency policy')
    for key, value in facts.items():
        if key in target['hardware'] or type(value) is not int or not 0 < value <= 4096:
            raise ValueError('arrival facts must add bounded values without replacing capacities')
        target['hardware'][key] = value
    for path, value in selectors.items():
        _, section, key = path.split('.')
        values = target.setdefault(section, {})
        if key in values:
            raise ValueError('arrival correction cannot overwrite existing selectors')
        values[key] = value
    if fetch is not None:
        if (set(fetch) != _FETCH_FIELDS | {'schema', 'assumptions'}
                or fetch['schema'] != 'merlin.provisional_fetch_refill.v1'
                or fetch['assumptions'] != 'demand_only_immediate_replay_no_backpressure'
                or any(type(fetch[k]) is not int or not 0 < fetch[k] <= 4096 for k in _FETCH_FIELDS)
                or 'fetch_refill_timeline' in expected['memory']):
            raise ValueError('fetch timeline needs exact bounded provisional mechanism')
        expected['memory']['fetch_refill_timeline'] = copy.deepcopy(fetch)
    expected['domain']['feature_engine_sha256'] = after['domain']['feature_engine_sha256']
    expected['libraries'] = after['libraries']
    if expected != after:
        raise ValueError('arrival correction changes prices, capacities, dependencies or unreviewed fields')


def _accumulation_delta(before, after, audit):
    expected = copy.deepcopy(before)
    schedule = expected['target']['scheduling']
    selected = {'split_ws_accumulate_operands': 1}
    if (schedule.get('version') != 5 or any(schedule.get(k) != 1 for k in (
            'split_ws_feed_lifetimes', 'require_fenced_phases',
            'conservative_local_store_holds', 'serialized_execute_epochs',
            'serialized_output_stationary_epochs'))
            or any(k in schedule for k in selected)
            or audit.get('selected_scheduling') != selected):
        raise ValueError('accumulating WS needs retained full-result dependency policy')
    schedule.update(selected)
    expected['libraries'] = after['libraries']
    if expected != after:
        raise ValueError('accumulating WS changes prices, observer, capacities or other lifetimes')


def verify(old, new, evidence, *, previous, current, accumulating_ws=False):
    schema = ('merlin.accumulating_ws_operand_review.v1' if accumulating_ws
              else 'merlin.arrival_pipeline_review.v1')
    if (evidence.get('schema') != schema
            or evidence.get('status') != 'conditional_provisional_untuned'
            or evidence.get('previous_runtime') != previous or evidence.get('runtime') != current):
        raise ValueError('arrival correction requires exact conditional provisional review')
    before, after = json.loads(old.model_json), json.loads(new.model_json)
    for name, value in [('previous_model', before), ('model', after)]:
        if PERF._read(evidence[name]) != value:
            raise ValueError('arrival review differs from exact model')
    old_spec, new_spec = PERF._read(previous), PERF._read(current)
    mutable = {'review_id', 'source_manifest', 'libraries', 'model', 'model_sha256'}
    if not accumulating_ws:
        mutable.add('observer')
    if ({k:v for k,v in old_spec.items() if k not in mutable}
            != {k:v for k,v in new_spec.items() if k not in mutable}):
        raise ValueError('arrival correction changes execution scope or functional contract')
    for left, right in ((old_spec['libraries'], new_spec['libraries']),
                        (before['libraries'], after['libraries'])):
        if len(left) != 2 or len(right) != 2 or left[:-1] != right[:-1]:
            raise ValueError('arrival correction changes functional extension')
    for role, spec, model in [('previous', old_spec, before), ('current', new_spec, after)]:
        pin = evidence['source_manifests'][role]
        if (pin['path'] != spec['source_manifest']
                or PERF._read(pin).get('sha256') != model['domain']['feature_engine_sha256']):
            raise ValueError('arrival observer source identity differs')
    fetch = after['memory'].get('fetch_refill_timeline')
    names = ({'accumulating_ws'} if accumulating_ws else
             {'row_requests'} | ({'fetch_refill'} if fetch is not None else set()))
    if set(evidence['source_audits']) != names:
        raise ValueError('arrival source audits differ from selected mechanisms')
    audits = {name:PERF._read(evidence['source_audits'][name]) for name in names}
    elaboration = audits['accumulating_ws' if accumulating_ws else 'row_requests']['selected_elaboration']
    _verify_file(elaboration)
    for name, audit in audits.items():
        if (audit.get('schema') != 'merlin.public_arrival_mechanism_facts.v1'
                or audit.get('mechanism') != name or audit.get('selected_elaboration') != elaboration
                or not isinstance(audit.get('source_pins'), list)
                or not 1 <= len(audit['source_pins']) <= 16):
            raise ValueError('arrival source proof is missing or crosses hardware domains')
        for pin in audit['source_pins']:
            _verify_file(pin)
    if accumulating_ws:
        _accumulation_delta(before, after, audits['accumulating_ws'])
    else:
        row = audits['row_requests']
        if fetch is not None and audits['fetch_refill'].get('timeline') != fetch:
            raise ValueError('fetch parameters differ from reviewed public evidence')
        _model_delta(before, after, row['hardware'], row['selectors'], fetch)
    build = PERF._read(evidence['provider_build'])
    if (build.get('library') != new_spec['libraries'][-1]
            or build.get('library_sha256') != after['libraries'][-1]):
        raise ValueError('arrival provider build differs from selected runtime')
    _verify_file(evidence['observer'])
    if evidence['observer']['path'] != new_spec['observer']:
        raise ValueError('arrival observer binary differs from reviewed implementation')
    replays = PERF._read(evidence['replay_evidence'])
    rows = replays.get('rows')
    if (replays.get('runtime') != current or replays.get('physical_runs_added') != 0
            or replays.get('price_changes') is not False or not isinstance(rows, list)
            or not 1 <= len(rows) <= 32):
        raise ValueError('arrival review requires bounded no-fit exact runtime replays')
    for pin in rows:
        receipt = PERF._read(pin)
        report = PERF._read(receipt['report'])
        if (receipt.get('runtime_sha256') != new.sha256
                or receipt.get('model') != evidence['model']
                or receipt.get('timing_complete') is not True or receipt.get('unresolved') != []
                or receipt.get('functional_console_equal') is not True
                or receipt.get('physical_runs') != 0 or receipt.get('calibration_eligible') is not False
                or not receipt.get('runs') or any(r.get('complete_output_match') is not True
                    or type(r.get('bytes')) is not int or r['bytes'] <= 0 for r in receipt['runs'])
                or report.get('timing_complete') is not True or report.get('execution_completed') is not True
                or report.get('model_sha256') != evidence['model']['sha256']
                or report.get('elf_sha256') != receipt['original']['elf']['sha256']):
            raise ValueError('arrival replay lacks exact runtime, ELF or complete numerical evidence')
        _verify_file(receipt['original']['elf'])
    profile = PERF._read(evidence['calibration_profile'])
    denial = PERF._read(evidence['continued_denial'])
    paths = sorted(tuple(p['path']) for p in profile['parameters'])
    if (not paths or denial.get('schema') != 'merlin.native_calibration_admission_review.v1'
            or not any(r.get('model_sha256') == evidence['model']['sha256']
                and sorted(tuple(p) for p in r['price_paths']) == paths for r in denial.get('deny', []))):
        raise ValueError('arrival correction must preserve explicit calibration denial')
