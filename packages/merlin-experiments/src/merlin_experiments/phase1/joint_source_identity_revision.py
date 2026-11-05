"""Admit missing direct-IR identities without changing any typed computation."""
from __future__ import annotations

import copy
from pathlib import Path

from xdsl.dialects.builtin import ArrayAttr, StringAttr
from merlin.common import mlir_query

from ..phase2.contracts import document_sha256


def verify(evidence, *, members):
    from .joint_source_revision import _capsule

    if (set(evidence) != {'schema', 'name', 'before', 'after'}
            or evidence['schema'] != 'merlin.direct_ir_identity_review.v1'):
        raise ValueError('source identity requires an exact operator review')
    old, old_files = _capsule(evidence['before'])
    new, new_files = _capsule(evidence['after'])
    name = evidence['name']
    source = old.get('linalg_mlir')
    if (name not in members or old.get('name') != name or new.get('name') != name
            or old.get('kind') != 'model_slice' or not isinstance(source, str)
            or source != old.get('interface_mlir') or source not in old_files
            or source not in new_files
            or members[name]['capsule_sha256'] != document_sha256(old)
            or {k:v for k,v in old_files.items() if k != source}
               != {k:v for k,v in new_files.items() if k != source}):
        raise ValueError('source identity changes companion files or stale member binding')
    expected = copy.deepcopy(old)
    screen = expected.get('software_screen', {})
    if screen.get('status') != 'admitted' or screen.get('program_sha256') != old_files[source]:
        raise ValueError('source identity requires the original admitted placement screen')
    screen['program_sha256'] = new_files[source]
    if expected != new:
        raise ValueError('source identity changes numerical, placement or capsule contract')
    modules = [mlir_query.parse((Path(evidence[role]['path']).parent / source).read_text())
               for role in ('before', 'after')]
    a, b = [list(module.walk()) for module in modules]
    if len(a) != len(b):
        raise ValueError('source identity changes operation structure')
    seen = set()
    for before, after in zip(a, b, strict=True):
        region = before.attributes.get('prov.region_id')
        if region is None:
            continue
        if (not isinstance(region, StringAttr) or not region.data or region.data in seen
                or 'prov.source_node_ids' in before.attributes
                or after.attributes.get('prov.source_node_ids')
                   != ArrayAttr([StringAttr('direct_ir.' + region.data)])):
            raise ValueError('source identity is not a unique original direct-IR region')
        seen.add(region.data)
        del after.attributes['prov.source_node_ids']
    if not seen or not modules[0].is_structurally_equivalent(modules[1]):
        raise ValueError('source identity changes typed computation or unreviewed attributes')
    return name, document_sha256(new)
