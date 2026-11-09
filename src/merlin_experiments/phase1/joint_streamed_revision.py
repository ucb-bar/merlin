"""Review composed public WS and streamed readout accounting; no fit authority."""
from __future__ import annotations

import copy
import json

from .feedback import joint_performance as PERF
from .joint_structural_revision import _verify_file

_HARDWARE = frozenset((
    'accumulator_read_response_cycles', 'accumulator_scale_pipeline_stages',
    'normalizer_passthrough_queue_entries', 'write_dispatch_queue_entries',
    'write_issue_queue_entries', 'write_norm_queue_entries', 'write_scale_queue_entries',
    'writer_beat_bytes', 'writer_max_request_bytes', 'writer_tlb_queue_entries',
    'writer_transaction_entries', 'writer_translation_queue_entries', 'writer_translation_queue_pipe',
))
_SELECTORS = {
    'target.hardware.execute_command_lookahead_heads': 3,
    'target.hardware.execute_completion_pending_register': 1,
    'target.scheduling.ws_feed_queue_release': 1,
    'target.scheduling.ws_resident_mul_pre_pairing': 1,
    'target.scheduling.accumulator_readout_arrival': 1,
    'target.scheduling.accumulator_streamed_readout': 1,
    'target.assumptions.accumulator_readout_translation_hits': 1,
}


def _model_delta(before, after, facts, selectors):
    if set(facts) != _HARDWARE or selectors != _SELECTORS:
        raise ValueError('composed accounting fields differ from the reviewed mechanism roster')
    expected = copy.deepcopy(before)
    target = expected['target']
    if (target['scheduling'].get('version') != 5
            or target['scheduling'].get('split_ws_feed_lifetimes') != 1
            or target['scheduling'].get('pool_relu_shared_scale_pipeline') != 1
            or target['hardware'].get('normalizations') != 0):
        raise ValueError('composed accounting requires the existing reviewed provisional base')
    for key, row in facts.items():
        value = row['value']
        if key in target['hardware'] or type(value) is not int or not 0 < value <= 4096:
            raise ValueError('streamed accounting must add bounded source facts, never replace capacities')
        target['hardware'][key] = value
    for path, value in selectors.items():
        _, section, key = path.split('.')
        values = target.setdefault(section, {})
        if key in values:
            raise ValueError('composed accounting cannot overwrite an existing selector')
        values[key] = value
    expected['libraries'] = after['libraries']
    if expected != after:
        raise ValueError('composed accounting changes prices, dependencies, CPU, memory or unreviewed facts')


def verify(old, new, evidence, *, previous, current):
    if (evidence.get('schema') != 'merlin.composed_public_accounting_review.v1'
            or evidence.get('status') != 'conditional_provisional_untuned'
            or evidence.get('previous_runtime') != previous or evidence.get('runtime') != current):
        raise ValueError('composed accounting needs exact conditional provisional review')
    old_spec, new_spec = PERF._read(previous), PERF._read(current)
    mutable = {'review_id', 'libraries', 'model', 'model_sha256'}
    if ({k:v for k,v in old_spec.items() if k not in mutable}
            != {k:v for k,v in new_spec.items() if k not in mutable}):
        raise ValueError('composed accounting changed observer or execution scope')
    before, after = json.loads(old.model_json), json.loads(new.model_json)
    for left, right in ((old_spec['libraries'], new_spec['libraries']),
                        (before['libraries'], after['libraries'])):
        if len(left) != 2 or len(right) != 2 or left[:-1] != right[:-1]:
            raise ValueError('composed accounting changed functional libraries')
    audits = {name:PERF._read(evidence['source_audits'][name])
              for name in ('queue', 'pairing', 'stream')}
    queue, pairing, stream = (audits[k] for k in ('queue', 'pairing', 'stream'))
    if (queue.get('schema') != 'merlin.public_ws_queue_release_review.v1'
            or pairing.get('schema') != 'merlin.public_ws_resident_pairing_review.v1'
            or stream.get('schema') != 'merlin.public_streamed_readout_review.v1'
            or queue['rtl'] != stream['selected_elaboration']
            or queue['source'] != pairing['rtl_source']):
        raise ValueError('composed accounting source mechanisms or elaborations differ')
    for pin in [queue['rtl'], queue['source'], *stream['source_pins']]:
        _verify_file(pin)
    selectors = {**queue['selectors'], **pairing['selectors'], **stream['selectors'],
                 **stream['required_assumption']}
    _model_delta(before, after, stream['selected_facts'], selectors)
    if PERF._read(evidence['model']) != after:
        raise ValueError('composed accounting model differs from the reviewed model')
    build = PERF._read(evidence['provider_build'])
    if (build['library'] != new_spec['libraries'][-1]
            or build['library_sha256'] != after['libraries'][-1]):
        raise ValueError('composed accounting provider build does not match runtime')
    replay = PERF._read(evidence['replays'])
    if (replay.get('runtime') != current or replay.get('physical_runs_added') != 0
            or replay.get('price_changes') is not False or not replay.get('rows')):
        raise ValueError('composed accounting needs exact no-fit public replays')
    for row in replay['rows']:
        report = PERF._read(row['report'])
        _verify_file(row['elf']); PERF._read(row['scope'])
        if row.get('timing_complete') is not True or report.get('timing_complete') is not True:
            raise ValueError('composed accounting replay timing is incomplete')
    denial = PERF._read(evidence['continued_denial'])
    profile = PERF._read(evidence['calibration_profile'])
    paths = sorted(tuple(p['path']) for p in profile['parameters'])
    if (not paths or denial.get('schema') != 'merlin.native_calibration_admission_review.v1'
            or not any(row.get('model_sha256') == new_spec['model_sha256']
                       and sorted(tuple(p) for p in row['price_paths']) == paths
                       for row in denial.get('deny', []))):
        raise ValueError('composed accounting must retain separate calibration admission')
