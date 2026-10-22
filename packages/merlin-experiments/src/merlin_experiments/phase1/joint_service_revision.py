"""Review selected RTL service equivalence without admitting new timing prices."""
from __future__ import annotations

import copy
import json

from .feedback import joint_performance as PERF
from .joint_structural_revision import _verify_file


def verify(old, new, evidence, *, previous, current):
    if (evidence.get('schema') != 'merlin.pool_relu_service_revision.v1'
            or evidence.get('status') != 'provisional_untuned'
            or evidence.get('previous_runtime') != previous
            or evidence.get('runtime') != current):
        raise ValueError('service equivalence needs exact provisional runtime review')
    audit = PERF._read(evidence['source_audit'])
    if (audit.get('schema') != 'merlin.public_pool_relu_service_review.v1'
            or audit.get('runtime') != current or audit.get('old_runtime') != previous
            or audit.get('private_labels_used') is not False
            or audit.get('physical_executions') != 0):
        raise ValueError('service equivalence source audit differs from reviewed runtimes')
    for pin in [audit['source'], audit['rtl'], *audit['excerpts']]:
        _verify_file(pin)
    before, after = json.loads(old.model_json), json.loads(new.model_json)
    old_spec, new_spec = PERF._read(previous), PERF._read(current)
    replaced = {'review_id', 'libraries', 'model', 'model_sha256'}
    if ({k:v for k,v in old_spec.items() if k not in replaced}
            != {k:v for k,v in new_spec.items() if k not in replaced}):
        raise ValueError('service equivalence changes observer or runtime scope')
    for a, b in [(old_spec['libraries'], new_spec['libraries']),
                 (before['libraries'], after['libraries'])]:
        if len(a) < 2 or len(a) != len(b) or a[:-1] != b[:-1]:
            raise ValueError('service equivalence changes functional libraries')
    expected = copy.deepcopy(before)
    target = expected['target']
    if (target['hardware'].get('normalizations') != 0
            or 'relu_same_accumulator_scale_pipeline' in target['hardware']
            or 'pool_relu_shared_scale_pipeline' in target['scheduling']
            or any(k.startswith('pool_store_act1') for k in target['prices'])):
        raise ValueError('service equivalence requires the original unpriced ReLU domain')
    target['hardware']['relu_same_accumulator_scale_pipeline'] = 1
    target['scheduling']['pool_relu_shared_scale_pipeline'] = 1
    expected['libraries'] = after['libraries']
    if expected != after:
        raise ValueError('service equivalence changes prices or unreviewed model fields')
    if PERF._read(audit['model']) != after:
        raise ValueError('service equivalence model differs from pinned audit')
    denial = PERF._read(audit['continued_calibration_denial'])
    profile = PERF._read(evidence['calibration_profile'])
    paths = sorted(tuple(p['path']) for p in profile['parameters'])
    if (not paths or denial.get('schema') != 'merlin.native_calibration_admission_review.v1'
            or not any(row.get('model_sha256') == new_spec['model_sha256']
                       and sorted(tuple(p) for p in row['price_paths']) == paths
                       for row in denial.get('deny', []))):
        raise ValueError('service equivalence must retain structural calibration denial')
