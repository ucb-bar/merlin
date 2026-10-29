"""Narrow reviewed WS-lifetime/private-probe correction; grants no qualification."""
from __future__ import annotations

import copy
import json

from .feedback import joint_performance as PERF


def verify(old, new, evidence, *, previous, current):
    if (evidence.get('schema') != 'merlin.structural_accounting_review.v1'
            or evidence.get('status') != 'provisional_untuned'
            or evidence.get('previous_runtime') != previous
            or evidence.get('runtime') != current):
        raise ValueError('structural accounting needs exact provisional runtime review')
    before, after = json.loads(old.model_json), json.loads(new.model_json)
    for key, model in [('previous_model', before), ('model', after)]:
        if PERF._read(evidence[key]) != model:
            raise ValueError('structural accounting review differs from exact model')
    old_spec, new_spec = PERF._read(previous), PERF._read(current)
    replaced = {'review_id', 'observer', 'source_manifest', 'libraries', 'model', 'model_sha256'}
    if ({k:v for k,v in old_spec.items() if k not in replaced}
            != {k:v for k,v in new_spec.items() if k not in replaced}):
        raise ValueError('structural accounting changes runtime scope or execution contract')
    if (len(old_spec['libraries']) != len(new_spec['libraries'])
            or old_spec['libraries'][:-1] != new_spec['libraries'][:-1]
            or len(before['libraries']) != len(after['libraries'])
            or before['libraries'][:-1] != after['libraries'][:-1]):
        raise ValueError('structural accounting changes functional extension libraries')
    for role, spec, model in [('previous', old_spec, before), ('current', new_spec, after)]:
        pin = evidence['source_manifests'][role]
        manifest = PERF._read(pin)
        if (pin['path'] != spec['source_manifest']
                or manifest.get('sha256') != model['domain']['feature_engine_sha256']):
            raise ValueError('structural accounting observer identity is not pinned')
    facts = {name: PERF._read(evidence['source_audits'][name])
             for name in ('ws_lifetimes', 'private_probe')}
    if facts['ws_lifetimes']['selected_elaboration'] != facts['private_probe']['selected_elaboration']:
        raise ValueError('structural accounting source proofs use different elaborations')
    # Operator proof binds the exact selected source and elaboration, not just
    # a claim that the hardware is similar to another configuration.
    for audit in facts.values():
        _verify_file(audit['source'])
    _verify_file(facts['private_probe']['selected_elaboration'])
    if (facts['ws_lifetimes'].get('schema') != 'merlin.ws_feed_lifetime_facts.v1'
            or facts['private_probe'].get('schema') != 'merlin.public_private_probe_protocol_facts.v1'):
        raise ValueError('structural accounting source proof schema is unsupported')
    expected = copy.deepcopy(before)
    schedule = expected['target']['scheduling']
    if (schedule.get('version') != 4 or 'split_ws_feed_lifetimes' in schedule
            or any(schedule.get(key) != 1 for key in ('require_fenced_phases',
                'conservative_local_store_holds', 'serialized_execute_epochs',
                'serialized_output_stationary_epochs'))):
        raise ValueError('WS lifetime correction requires retained conservative v4 dependencies')
    selected = {'version':5, 'split_ws_feed_lifetimes':1}
    if facts['ws_lifetimes'].get('selected_scheduling') != selected:
        raise ValueError('WS lifetime fields differ from reviewed source semantics')
    schedule.update(selected)
    cache_name = evidence['private_cache']
    memory = expected['memory']
    if (cache_name not in memory['paths']['cpu_data'] or cache_name in memory['paths']['device']
            or memory.get('transaction_accounting') is not None):
        raise ValueError('probe correction requires an unchanged private-cache path')
    cache = memory['caches'][cache_name]
    proof = facts['private_probe']
    if cache['line_bytes'] != proof['line_bytes'] or 'coherence_probe_service' in cache:
        raise ValueError('probe correction changes existing cache service or geometry')
    fields = ('beat_bytes', 'lookup_cycles', 'dirty_enter_cycles', 'data_pipeline_cycles', 'metadata_cycles')
    service = {key:proof['field_bindings'][key]['value'] for key in fields}
    if any(type(v) is not int or v < 0 or v > 4096 for v in service.values()):
        raise ValueError('probe source facts exceed bounded integer service')
    if not service['beat_bytes'] or cache['line_bytes'] % service['beat_bytes']:
        raise ValueError('probe source width does not divide selected cache line')
    resource = evidence['added_resource']
    if (set(resource) != {'name','id','capacity','queue_capacity'}
            or resource['capacity'] != 1 or resource['queue_capacity'] != 0
            or type(resource['id']) is not int
            or resource['id'] != max(r['id'] for r in before['resources'])+1
            or not isinstance(resource['name'], str) or not resource['name']
            or any(r['name'] == resource['name'] for r in before['resources'])):
        raise ValueError('probe correction needs one new serial controller resource')
    service.update(schema='merlin.private_cache_probe_service.v1', resource=resource['name'])
    cache['coherence_probe_service'] = service
    memory['transaction_accounting'] = 'coherence_request_separate_v1'
    expected['resources'].append(copy.deepcopy(resource))
    expected['domain']['feature_engine_sha256'] = after['domain']['feature_engine_sha256']
    expected['libraries'] = after['libraries']
    if expected != after:
        raise ValueError('structural accounting changes prices, capacities, dependencies or unreviewed fields')
    replay = PERF._read(evidence['replay_evidence'])
    if (replay.get('price_changes') is not False or replay.get('physical_runs_added') != 0
            or not any(row.get('runtime') == current for row in replay.get('rows', []))):
        raise ValueError('structural accounting requires separately pinned no-fit replay evidence')
    profile = PERF._read(evidence['calibration_profile'])
    denial = PERF._read(evidence['calibration_admission_review'])
    paths = sorted(tuple(p['path']) for p in profile['parameters'])
    if (denial.get('schema') != 'merlin.native_calibration_admission_review.v1'
            or not any(row.get('model_sha256') == evidence['model']['sha256']
                       and sorted(tuple(p) for p in row['price_paths']) == paths
                       for row in denial.get('deny', []))):
        raise ValueError('structural accounting must preserve reviewed calibration denial')


def _verify_file(pin):
    from pathlib import Path
    from merlin.runtime import _native_files as F
    if F.pin(Path(pin['path']), max_bytes=256 << 20) != pin:
        raise ValueError('structural accounting source or elaboration changed')
