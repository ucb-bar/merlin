"""Bind an operator-reviewed captured constant closure, without execution credit.

The pinned review authorizes a new source declaration. Its original capture and
preparation proofs remain evidence, not an oracle or a reusable target pass.
Native source, host placement, trace and numerical checks still run normally.
"""
from copy import deepcopy
from pathlib import Path

import yaml

from ..phase2.contracts import document_sha256
from .feedback import joint_performance as PERF


def verify(evidence, *, members):
    from .joint_source_revision import _capsule

    if (set(evidence) != {'schema', 'name', 'before', 'after', 'projection', 'preparation'}
            or evidence['schema'] != 'merlin.captured_constant_closure_review.v1'):
        raise ValueError('constant closure requires an exact operator source review')
    old, old_files = _capsule(evidence['before'])
    new, new_files = _capsule(evidence['after'])
    name = evidence['name']
    projection = PERF._read(evidence['projection'])
    preparation = PERF._read(evidence['preparation'])
    mapping = projection.get('new_to_original_inputs')
    source = old.get('linalg_mlir')
    if (name not in members or old.get('name') != name or new.get('name') != name
            or old.get('kind') != 'model_slice'
            or members[name]['capsule_sha256'] != document_sha256(old)
            or source != old.get('interface_mlir') or source not in old_files
            or projection.get('schema') != 'operator.connected_boundary_projection.v1'
            or preparation.get('schema') != 'operator.prepared_source_reproduction.v1'
            or projection.get('preparation_review') != evidence['preparation']
            or projection.get('projected_sha256') != new_files.get(source)
            or preparation.get('projected', {}).get('sha256') != new_files.get(source)
            or projection.get('source_sha256') != preparation.get('new', {}).get('sha256')
            or preparation.get('attribute_only') is not True
            or any(c.get('additions_only') is not True or c.get('provenance_only') is not True
                   for c in preparation.get('changes', []))
            or projection.get('outputs_same_source_values') is not True
            or not isinstance(mapping, list) or not mapping
            or any(type(i) is not int or not 0 <= i < len(old['inputs']) for i in mapping)
            or len(set(mapping)) != len(mapping) or len(mapping) >= len(old['inputs'])):
        raise ValueError('constant closure source or captured projection identity differs')
    # Reopen the original capture record. The pinned review supplies source
    # reconstruction/literal proofs, never the compiler under evaluation.
    original = PERF._read(projection['original_record'])
    if (original.get('source_projection', {}).get('sha256') != old_files[source]
            or original.get('source') != preparation.get('old')
            or original.get('all_live_outputs_retained') is not True):
        raise ValueError('constant closure original captured region differs')
    removed = projection.get('removed_literal_inputs', [])
    if (len(removed) != len(old['inputs']) - len(mapping)
            or {r.get('original_input') for r in removed} != set(range(len(old['inputs']))) - set(mapping)
            or any(r.get('literal_bytes_equal') is not True or not r.get('array_sha256') for r in removed)):
        raise ValueError('constant closure lacks exact removed-literal review')
    expected = deepcopy(old)
    if len(new['inputs']) != len(mapping) or len({v['name'] for v in new['inputs']}) != len(mapping):
        raise ValueError('constant closure input mapping differs')
    expected['inputs'] = [dict(old['inputs'][j], name=new['inputs'][i]['name'])
                          for i, j in enumerate(mapping)]
    args = [v['name'] for v in new['inputs']] + old['operation']['attributes']['outs']
    expected['operation']['attributes']['arg_order'] = args
    expected['source_reference'] = new.get('source_reference')
    expected['software_screen'] = new.get('software_screen')
    screen = new.get('software_screen', {})
    if (expected != new or screen.get('status') != 'admitted'
            or screen.get('program_sha256') != new_files[source]):
        raise ValueError('constant closure changes another capsule obligation')
    changed = {k for k in old_files.keys() | new_files.keys() if old_files.get(k) != new_files.get(k)}
    if (changed != {source, 'golden.yaml'} or 'golden.npz' not in old_files
            or old_files['golden.npz'] != new_files['golden.npz']):
        raise ValueError('constant closure changes unreviewed source or golden payload')
    before = yaml.safe_load((Path(evidence['before']['path']).parent / 'golden.yaml').read_text())
    after = yaml.safe_load((Path(evidence['after']['path']).parent / 'golden.yaml').read_text())
    expected_golden = deepcopy(before)
    provenance = expected_golden['oracle_provenance']
    provenance['inputs'] = {new['inputs'][i]['name']: before['oracle_provenance']['inputs'][old['inputs'][j]['name']]
                            for i, j in enumerate(mapping)}
    provenance['arg_order'] = args
    provenance['source_sha256'] = projection['source_sha256']
    provenance['region_sha256'] = projection['projected_sha256']
    provenance['scope'] = after['oracle_provenance'].get('scope')
    if (expected_golden != after or projection.get('output_count') != len(before['outputs'])
            or preparation['projected'].get('inputs') != len(mapping)
            or preparation['projected'].get('outputs') != len(before['outputs'])):
        raise ValueError('constant closure changes expected values or captured input bindings')
    return name, document_sha256(new)
