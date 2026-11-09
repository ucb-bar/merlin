"""Prospective fitting scope for an explicitly reviewed fresh initial model.

This is distinct from a historical diagnostics-neutrality successor. It binds
the source model selected at campaign creation, unchanged public parameter/group
declarations and retained denials. It grants neither measurements nor a fit or
accuracy verdict: native sensitivity and all existing round gates still run.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from merlin.runtime import _native_files as F
from .feedback import joint_performance as P

ENV = "MERLIN_NATIVE_CALIBRATION_INITIAL_REVIEW"
SCHEMA = "merlin.fresh_initial_calibration_scope.v1"


def _parameter_scope(model, parameters, runtime):
    from . import joint_calibration as C
    if (not isinstance(parameters,list) or not 1 <= len(parameters) <= 2
            or math.prod(len(p.get('values',[])) for p in parameters) > 16):
        raise ValueError('initial fit scope exceeds the existing sparse parameter budget')
    mapped = set()
    for pin in [runtime['facts_alignment'], *runtime.get('additional_fact_alignments',[])]:
        mapped.update(tuple(row['model_path']) for row in P._read(pin)['mappings'])
    seen = set()
    for parameter in parameters:
        path,values=parameter.get('path'),parameter.get('values')
        if (not isinstance(path,list) or not path or any(type(k) is not str for k in path)
                or tuple(path) in seen or tuple(path) in mapped
                or not any(k in path for k in ('prices','latency_cycles','occupancy_cycles'))
                or not isinstance(values,list) or not 2 <= len(values) <= 4
                or any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in values)
                or len(set(values)) != len(values) or C._at(model,path) not in values
                or not parameter.get('mechanism') or not parameter.get('reviewed_scope')):
            raise ValueError('initial scope changes a fact or lacks reviewed finite service-price bounds')
        seen.add(tuple(path))


def verify(review_pin, *, plan, runtime, active_review):
    """Validate the new initial anchor; the original exact denial check follows."""
    return _verify(P._read(review_pin), plan=plan, runtime=runtime, active_review=active_review)


def _verify(review, *, plan, runtime, active_review):
    from . import joint_control, joint_calibration as C
    if (review.get('schema') != SCHEMA or not review.get('review_id')
            or review.get('scope') != 'fresh_campaign_initial_only'
            or review.get('strict_gates') != C.GATES
            or review.get('measurement_authorized') is not False
            or review.get('qualification_issued') is not False
            or review.get('continued_denial') != active_review):
        raise ValueError('fresh initial calibration scope lacks exact owner/gates/denial review')
    denial=P._read(review['continued_denial'])
    if (denial.get('schema') != 'merlin.native_calibration_admission_review.v1'
            or not isinstance(denial.get('deny'),list) or len(denial['deny']) > 64):
        raise ValueError('fresh scope does not retain the original bounded denial review')
    initial=P._read(review['initial_selection'])
    if initial.get('timing_epochs') or initial.get('runtime_revisions'):
        raise ValueError('fresh scope cannot claim an inherited model epoch or implementation ancestry')
    state=joint_control.selected()
    if state is not None and (state.get('epoch') is not None
            or state.get('performance_selection') != review['initial_selection']):
        raise ValueError('fresh calibration scope does not belong to this new campaign initial selection')
    selected=plan.get('selection_document')
    if selected is None and plan.get('selection'):
        selected=P._read(plan['selection'])
    if (not isinstance(selected,dict)
            or {k:v for k,v in selected.items() if k != 'timing_epochs'}
                != {k:v for k,v in initial.items() if k != 'timing_epochs'}):
        raise ValueError('fresh calibration scope source/facts/runtime selection changed')
    profile=P._read(review['parameter_profile'])
    if plan.get('parameters') != profile['parameters']:
        raise ValueError('fresh calibration scope has different reviewed parameters')
    old=P._read(profile['performance_selection'])
    for name in profile['members']:
        if (name not in initial.get('members',{})
                or initial['members'][name] != old['members'].get(name)):
            raise ValueError('fresh scope changes an original reviewed public source group/member')
    if review.get('members') != profile['members'] or review.get('facts') != initial['facts']:
        raise ValueError('fresh scope public groups or facts differ from its review')
    audit=P._read(review['source_review'])
    if (audit.get('initial_selection') != review['initial_selection']
            or audit.get('kernel_runtime') != initial['runtime']
            or audit.get('facts') != initial['facts']):
        raise ValueError('fresh scope lacks its exact current source audit')
    spec=P._read(initial['runtime'])
    source_pin=F.pin(Path(spec['source_manifest']),max_bytes=256<<20)
    if (review.get('runtime') != initial['runtime'] or review.get('source_manifest') != source_pin
            or review.get('model') != {'path':spec['model'],'sha256':spec['model_sha256']}):
        raise ValueError('fresh scope model/source pins differ')
    base=P._runtime(initial['runtime'],facts_pin=initial['facts']);base.verify()
    old_runtime=P._runtime(old['runtime'],facts_pin=old['facts']);old_runtime.verify()
    base_model=json.loads(base.model_json);old_model=json.loads(old_runtime.model_json)
    if (base_model['target']['prices'] != old_model['target']['prices']
            or any(base_model['domain'][k] != old_model['domain'][k]
                   for k in ('device_sha256','memory_regime_sha256','label_engine_sha256'))):
        raise ValueError('fresh initial review changes prices or hardware/physical measurement domain')
    _parameter_scope(base_model,profile['parameters'],spec)
    # An accepted tuned epoch may use the same source service laws at a reviewed
    # price point. It cannot change the observer, functional receiver or scope.
    current=P._read(plan['runtime'])
    allowed={'review_id','model','model_sha256'}
    if ({k:v for k,v in current.items() if k not in allowed}
            != {k:v for k,v in spec.items() if k not in allowed}):
        raise ValueError('fresh scope runtime implementation/receiver/scope changed')
    actual=P._runtime(plan['runtime'],facts_pin=initial['facts']);actual.verify()
    model=json.loads(actual.model_json);expected=copy.deepcopy(base_model)
    for parameter in profile['parameters']:
        value=C._at(model,parameter['path'])
        if value not in parameter['values']:
            raise ValueError('fresh scope price is outside its declared reviewed values')
        C._set(expected,parameter['path'],value)
    if expected != model or runtime['model_sha256'] != current['model_sha256']:
        raise ValueError('fresh scope changed model structure beyond reviewed service prices')


def prepare(*, selection_pin, parameter_profile_pin, previous_denial_pin, source_review_pin, review_id, output):
    """Host-only preparation for a NEW campaign; no historical proof is invented."""
    from . import joint_calibration as C
    selection=P._read(selection_pin);profile=P._read(parameter_profile_pin)
    spec=P._read(selection['runtime'])
    record=dict(schema=SCHEMA,review_id=review_id,scope='fresh_campaign_initial_only',
        initial_selection=selection_pin,runtime=selection['runtime'],facts=selection['facts'],
        source_manifest=F.pin(Path(spec['source_manifest']),max_bytes=256<<20),
        model={'path':spec['model'],'sha256':spec['model_sha256']},
        parameter_profile=parameter_profile_pin,members=profile['members'],source_review=source_review_pin,
        continued_denial=previous_denial_pin,strict_gates=dict(C.GATES),
        measurement_authorized=False,qualification_issued=False,
        meaning='prospective local service-parameter scope only; original rank/numerical/held gates and denials remain')
    # Verify the same exact record before publishing it. No user context or
    # process-global environment is modified for this metadata operation.
    _verify(record,plan=dict(runtime=selection['runtime'],selection_document=selection,
                            parameters=profile['parameters']),runtime=spec,active_review=previous_denial_pin)
    return C._write(Path(output),record)
