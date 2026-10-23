"""Reviewed opt-in observer diagnostics ancestry; no model fitting authority."""
from __future__ import annotations

import copy
import json

from .feedback import joint_performance as PERF
from .joint_structural_revision import _verify_file

SCHEMA = 'merlin.scoped_observer_revision.v1'


def verify(old, new, evidence, *, previous, current):
    """Allow only selected observer identity and bounded diagnostic enablement.

    Numerical behavior of executable code is supported by the pinned public
    neutrality probe, not proved universally by a metadata comparison. All
    existing timing parameters and runtime boundaries remain immutable.
    """
    if (evidence.get('schema') != SCHEMA or evidence.get('status') != 'diagnostics_only_provisional'
            or evidence.get('previous_runtime') != previous or evidence.get('runtime') != current
            or evidence.get('fit_authorized') is not False or evidence.get('promotion_authorized') is not False):
        raise ValueError('scoped observer requires exact diagnostics-only review')
    before, after = json.loads(old.model_json), json.loads(new.model_json)
    for key, model in [('previous_model', before), ('model', after)]:
        if PERF._read(evidence[key]) != model:
            raise ValueError('scoped observer review differs from selected model')
    old_spec, new_spec = PERF._read(previous), PERF._read(current)
    allowed = {'review_id', 'observer', 'source_manifest', 'model', 'model_sha256'}
    if ({k:v for k,v in old_spec.items() if k not in allowed}
            != {k:v for k,v in new_spec.items() if k not in allowed}):
        raise ValueError('scoped observer changes provider, functional runtime or scope')
    for prefix, spec in [('previous_', old_spec), ('', new_spec)]:
        if (evidence[prefix+'model'] != {'path':spec['model'],'sha256':spec['model_sha256']}
                or evidence[prefix+'observer']['path'] != spec['observer']):
            raise ValueError('scoped observer tool/model pin differs from runtime')
        _verify_file(evidence[prefix+'observer'])
    config = evidence.get('diagnostics')
    ids = {r['id'] for r in after['resources']}
    if (not isinstance(config, dict) or set(config) != {'enabled', 'pairs'}
            or config['enabled'] is not True or not isinstance(config['pairs'], list)
            or len(config['pairs']) > 32):
        raise ValueError('scoped observer lacks bounded explicit diagnostics')
    pairs = set()
    for pair in config['pairs']:
        if (not isinstance(pair, dict) or set(pair) != {'left', 'right'}
                or any(type(pair[k]) is not int or pair[k] not in ids for k in pair)):
            raise ValueError('scoped observer overlap references unknown resources')
        key = tuple(sorted(pair.values()))
        if key[0] == key[1] or key in pairs:
            raise ValueError('scoped observer duplicate overlap pair')
        pairs.add(key)
    expected = copy.deepcopy(before)
    diagnostics = expected.setdefault('diagnostics', {})
    if 'scoped_resources' in diagnostics:
        raise ValueError('scoped observer revision requires a previously unselected capture')
    diagnostics['scoped_resources'] = copy.deepcopy(config)
    expected['domain']['feature_engine_sha256'] = after['domain']['feature_engine_sha256']
    if expected != after:
        raise ValueError('scoped observer changes coefficients, provider, scope or model semantics')
    manifests = {}
    for role, spec, model in [('previous', old_spec, before), ('current', new_spec, after)]:
        pin = evidence['source_manifests'][role]
        manifest = PERF._read(pin)
        if pin['path'] != spec['source_manifest'] or manifest.get('sha256') != model['domain']['feature_engine_sha256']:
            raise ValueError('scoped observer source binding differs')
        manifests[role] = manifest
    prior, present = manifests['previous'], manifests['current']
    if (prior.get('provider_build') != present.get('provider_build')
            or prior.get('shared_provider_observer_headers') != present.get('shared_provider_observer_headers')):
        raise ValueError('scoped observer changes provider build or shared ABI sources')
    payload = present.get('identity_payload', {})
    if (payload.get('base_source_manifest') != evidence['source_manifests']['previous']
            or payload.get('base_selected_source_identity') != prior['sha256']):
        raise ValueError('scoped observer source lineage differs from selected predecessor')
    build = PERF._read(present['build'])
    if (build.get('source_identity') != present['sha256']
            or build.get('spike_sha256') != evidence['observer']['sha256']):
        raise ValueError('scoped observer build differs from selected source/binary')
    leaves = payload.get('candidate_compiled_leaves')
    declared = build.get('source_sha256s')
    if not isinstance(leaves, list) or not 1 <= len(leaves) <= 16 or not isinstance(declared, dict) or len(leaves) != len(declared):
        raise ValueError('scoped observer compiled source roster is missing')
    for name, digest in declared.items():
        matches = [p for p in leaves if p['path'].endswith('/'+name) and p['sha256'] == digest]
        if len(matches) != 1:
            raise ValueError('scoped observer compiled source differs')
        _verify_file(matches[0])
    _neutrality(evidence)
    _denial(evidence)


def _neutrality(evidence):
    receipt = PERF._read(evidence['neutrality'])
    if (receipt.get('schema') != 'merlin.scoped_ws_observer_acceptance.v1'
            or receipt.get('old_off_on_equal') is not True or receipt.get('qualification_issued') is not False
            or receipt.get('coefficient_changes') is not False or receipt.get('new_physical_runs') != 0):
        raise ValueError('scoped observer lacks diagnostic-only neutrality receipt')
    rows = receipt.get('results')
    if not isinstance(rows, list) or [r.get('name') for r in rows] != ['baseline','off','on']:
        raise ValueError('scoped observer neutrality run roster differs')
    sources = evidence.get('neutrality_products')
    if not isinstance(sources, dict) or set(sources) != {'baseline','off','on'}:
        raise ValueError('scoped observer lacks original neutrality products')
    reports, consoles = [], []
    for row in rows:
        product = sources[row['name']]
        report = PERF._read(product['report']);_verify_file(product['stdout'])
        if (product['report']['sha256'] != row.get('report_sha256')
                or report.get('model_sha256') != row.get('model_sha256')
                or report.get('execution_completed') is not True or report.get('timing_complete') is not True
                or report.get('unresolved') != [] or report.get('functional_status') != 'PASS'
                or report.get('raw_functional_exit_code') != 0):
            raise ValueError('scoped observer neutrality product is incomplete or changed')
        reports.append(report);consoles.append(product['stdout']['sha256'])
    for key in ('estimated_cycles','feature_counts','scope_sha256','scope_binding_sha256','elf_sha256','input_sha256','counter_windows'):
        if any(r.get(key) != reports[0].get(key) for r in reports[1:]):
            raise ValueError('scoped observer changes public probe timing or counts')
    if (len(set(consoles)) != 1 or reports[0]['model_sha256'] != evidence['previous_model']['sha256']
            or reports[2]['model_sha256'] != evidence['model']['sha256']
            or reports[2].get('scoped_resource_activity', {}).get('complete') is not True):
        raise ValueError('scoped observer neutrality is not bound to selected models/output')


def _denial(evidence):
    prior = PERF._read(evidence['previous_denial']);current = PERF._read(evidence['continued_denial'])
    profile = PERF._read(evidence['calibration_profile'])
    if (prior.get('schema') != 'merlin.native_calibration_admission_review.v1'
            or current.get('schema') != prior['schema'] or not isinstance(prior.get('deny'), list)
            or not isinstance(current.get('deny'), list) or current['deny'][:len(prior['deny'])] != prior['deny']):
        raise ValueError('scoped observer loses existing calibration denial')
    paths = sorted(tuple(p['path']) for p in profile['parameters'])
    rows = [r for r in current['deny'] if r.get('model_sha256') == evidence['model']['sha256']
            and sorted(tuple(p) for p in r.get('price_paths', [])) == paths]
    if not paths or len(rows) != 1 or not rows[0].get('reason') or not 1 <= len(rows[0].get('evidence', [])) <= 16:
        raise ValueError('scoped observer new model lacks continued parameter denial')
    for pin in rows[0]['evidence']:
        PERF._read(pin)
