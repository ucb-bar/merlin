"""Analysis-only compression of repeated structural region sequences.

The flat requests remain authoritative. A target explicitly declares attributes
that may differ; every original request is retained for exact expansion.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def structural_signature(request, *, parameter_attributes=()):
    """Ignore provenance and names, retaining types, constants, maps and wiring.

    This signature is for analysis, never a compilation cache identity.
    """
    row = deepcopy(request)
    order = dict.fromkeys(key for node in row['nodes'] if node['effect'] == 'pure' for key in node['inputs'])
    order.update(dict.fromkeys(node['id'] for node in row['nodes']))
    names = {key: f'v{i}' for i, key in enumerate(order)}
    for node in row['nodes']:
        node['id'] = names[node['id']]
        node['inputs'] = [names[key] for key in node['inputs']]
        node['attrs'] = {key: value for key, value in node['attrs'].items()
                         if not key.startswith('prov.') and key not in parameter_attributes}
    row['nodes'] = sorted(row['nodes'], key=lambda node: (node['effect'] == 'pure', node['id']))
    row['outputs'] = [names[key] for key in row['outputs']]
    for field in ('input_storages', 'physical_extents', 'boundary_layouts'):
        if field in row:
            row[field] = {names[key]: value for key, value in row[field].items()}
    if 'input_order' in row:
        row.pop('input_order')
        row['schema'] = 'structural'
    for binding in row.get('constants', []):
        binding['node_id'] = names[binding['node_id']]
    row['constants'] = sorted(row.get('constants', []), key=lambda item: item['node_id'])
    for field, empty in (('physical_extents', {}), ('boundary_layouts', {}), ('output_layouts', [])):
        row.setdefault(field, empty)
    row['schema'] = 'structural'
    row.pop('source_identity', None)
    return sha256(_bytes(row)).hexdigest()


def expand_region_loop(record):
    """Expand exact request bindings, refusing malformed counts or changed wiring."""
    offset, period, trip = (record[key] for key in ('offset', 'period', 'trip_count'))
    if any(type(v) is not int for v in (offset, period, trip)) or offset < 0 or period < 1 or trip < 2:
        raise ValueError('invalid region loop bounds')
    iterations = record['iterations']
    if len(iterations) != trip or any(len(row) != period for row in iterations):
        raise ValueError('region loop bindings differ from bounds')
    expanded = record['prefix'] + [item for row in iterations for item in row] + record['suffix']
    if len(record['prefix']) != offset or sha256(_bytes(expanded)).hexdigest() != record['flat_sha256']:
        raise ValueError('region loop expansion differs from flat requests')
    attrs = record['parameter_attributes']
    expected = [structural_signature(item, parameter_attributes=attrs) for item in iterations[0]]
    if any([structural_signature(item, parameter_attributes=attrs) for item in row] != expected
           for row in iterations[1:]):
        raise ValueError('region loop body structure differs')
    return deepcopy(expanded)


def analyze_region_loop(requests, *, parameter_attributes=()):
    """Find the largest tandem repeat, preferring its shortest body.

    Quadratic signature comparisons use linear auxiliary storage. No target or
    model identities, execution scheduling, or placement choices are inferred.
    """
    requests = list(requests)
    signatures = [structural_signature(row, parameter_attributes=parameter_attributes) for row in requests]
    size = len(signatures)
    best = None
    for period in range(1, size // 2 + 1):
        matches = 0
        for start in range(size - period - 1, -1, -1):
            matches = matches + 1 if signatures[start] == signatures[start + period] else 0
            trip = 1 + matches // period
            if trip < 2:
                continue
            candidate = (period * trip, -period, -start, start, period, trip)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    if best is None:
        return None
    _, _, _, offset, period, trip = best
    end = offset + period * trip
    record = dict(schema='merlin.region_loop.v1', mode='analysis_only', offset=offset,
                  period=period, trip_count=trip, parameter_attributes=sorted(parameter_attributes),
                  prefix=requests[:offset], suffix=requests[end:],
                  iterations=[requests[start:start + period] for start in range(offset, end, period)],
                  flat_sha256=sha256(_bytes(requests)).hexdigest())
    if expand_region_loop(record) != requests:
        raise ValueError('region loop expansion failed')
    return record
