"""Bounded offline ICache handshake attribution; no model fitting or verdicts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MAX_BYTES = 32 << 20
MAX_RECORDS = 65536


def summarize(rows):
    errors, requests, scopes = [], [], []
    pending = None
    opened = None
    last_seq = last_cycle = -1
    header = footer = None
    for row in rows:
        kind = row.get('kind')
        if kind == 'header':
            if header is not None:
                errors.append('duplicate header')
            header = row
            continue
        if kind == 'footer':
            if footer is not None:
                errors.append('duplicate footer')
            footer = row
            continue
        if footer is not None:
            errors.append('event follows footer')
        if header is None:
            errors.append('event precedes header')
        seq, cycle = row.get('seq'), row.get('reference_cycle')
        if (type(seq) is not int or type(cycle) is not int
                or seq <= last_seq or cycle < last_cycle or cycle < 0):
            errors.append('nonmonotonic or missing event identity/time')
            continue
        last_seq, last_cycle = seq, cycle
        if kind == 'icache_get':
            if pending is not None:
                errors.append('multiple outstanding local ICache requests')
            address, beat_bytes, count = row.get('line_address'), row.get('beat_bytes'), row.get('beat_count')
            valid = (type(address) is int and address >= 0
                     and type(beat_bytes) is int and beat_bytes > 0
                     and type(count) is int and count > 0
                     and type(row.get('source')) is int and row['source'] >= 0
                     and address % (beat_bytes * count) == 0
                     and row.get('handshake') is True)
            if not valid:
                errors.append('invalid request geometry or unobserved handshake')
                pending = None
                continue
            pending = dict(line_address=address, request_cycle=cycle,
                           request_seq=seq, beat_bytes=beat_bytes,
                           beat_count=count, beats=[], source=row.get('source'))
        elif kind == 'icache_d':
            if pending is None:
                errors.append('response without retained request')
                continue
            address = row.get('refill_address')
            line_bytes = pending['beat_bytes'] * pending['beat_count']
            if (row.get('handshake') is not True or type(address) is not int
                    or address // line_bytes * line_bytes != pending['line_address']
                    or row.get('source') != pending['source']
                    or row.get('beat_counter') != len(pending['beats'])):
                errors.append('response does not match exact pending request/beat')
            pending['beats'].append(cycle)
            expected_last = len(pending['beats']) == pending['beat_count']
            if row.get('last') is not expected_last:
                errors.append('response last flag disagrees with declared extent')
            if row.get('last') is True:
                pending['first_response_cycles'] = pending['beats'][0] - pending['request_cycle']
                pending['complete_response_cycles'] = cycle - pending['request_cycle']
                pending['response_span_cycles'] = cycle - pending['beats'][0]
                pending['response_intervals'] = [b-a for a,b in zip(pending['beats'], pending['beats'][1:])]
                requests.append(pending)
                pending = None
        elif kind == 'scope_marker' and row.get('scope_index') not in (None, -1):
            if row.get('wb_valid') is not True:
                errors.append('scope marker is not observed committed writeback')
            if row.get('marker') == 'begin':
                if opened is not None:
                    errors.append('overlapping scope markers')
                opened = dict(index=row['scope_index'], begin_cycle=cycle)
            elif row.get('marker') == 'end':
                if opened is None or opened['index'] != row['scope_index']:
                    errors.append('unmatched scope end')
                else:
                    scopes.append(dict(**opened, end_cycle=cycle))
                opened = None
            else:
                errors.append('unknown scope marker')
    if header is None or header.get('schema') != 'operator.frontend_refill_events.v1':
        errors.append('missing supported header')
    if (footer is None or footer.get('truncated') is not False
            or type(footer.get('dropped')) is not int or footer['dropped'] != 0):
        errors.append('missing complete footer or trace loss')
    if footer is None or footer.get('simulator_done') is not True or footer.get('exit_code') != 0:
        errors.append('normal simulator completion is unproven')
    if opened is not None:
        errors.append('unfinished scope at end of trace')
    if header and 'scope_bindings' in header:
        expected = sorted(row['scope_index'] for row in header['scope_bindings'])
        if sorted(row['index'] for row in scopes) != expected:
            errors.append('observed scopes differ from declared scope roster')
    censored = []
    if pending is not None:
        # The CPU may fetch after the final measured routine and terminate with
        # an outstanding request. Preserve that censoring; it cannot invalidate
        # a strictly earlier complete window or acquire a completion latency.
        censored.append(pending)
        if not scopes or pending['request_cycle'] <= max(s['end_cycle'] for s in scopes):
            errors.append('unfinished request may intersect a selected scope')
    for scope in scopes:
        scope['overlapping_request_indices'] = [i for i,r in enumerate(requests)
            if r['request_cycle'] <= scope['end_cycle'] and r['beats'][-1] >= scope['begin_cycle']]
    return dict(schema='operator.frontend_refill_attribution.v1',
                status='UNKNOWN' if errors else 'observed_handshakes',
                errors=list(dict.fromkeys(errors)), requests=requests, scopes=scopes,
                right_censored_requests=censored,
                qualification=False, fitting=False,
                limitations=['Reference-cycle durations require pinned clock and sampling semantics.',
                             'No directory, bank or upstream source transaction is inferred.',
                             'Refill latencies are not additive kernel stall cycles.',
                             'Writeback marker spans are not architectural counter readings.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.trace.stat().st_size > MAX_BYTES:
        raise ValueError('trace exceeds byte bound')
    lines = a.trace.read_bytes().splitlines()
    if len(lines) > MAX_RECORDS + 2:
        raise ValueError('trace exceeds record bound')
    result = summarize([json.loads(line) for line in lines])
    with a.out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], errors=result['errors'], requests=len(result['requests']))))


if __name__ == '__main__':
    main()
