"""Retain historical certificates without letting another judge suppress new work.

This changes scheduling state only. Pending records are never retired: their
original jobs must retain unambiguous completion ownership.
"""
from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path


def retire(ws, state, *, tier, instrument):
    """Called under the native tier ledger lock; archive before changing any state."""
    from . import promotion as P

    if not isinstance(instrument, str) or len(instrument) != 64 or any(
            c not in '0123456789abcdef' for c in instrument):
        return None
    retired = []
    for name in state:
        for identity, entry in P._slots_ro(state, name, tier).items():
            if (isinstance(entry, dict) and entry.get('status') in {'pass', 'fail', 'joined'}
                    and entry.get('instrument') != instrument):
                retired.append({'capsule': name, 'identity': identity, 'entry': deepcopy(entry)})
    if not retired:
        return None
    before = deepcopy(state)
    ledger_bytes = (Path(ws) / 'qa' / 'tier_state.json').read_bytes()
    if json.loads(ledger_bytes) != before:
        raise ValueError('instrument retirement must precede other ledger changes')
    document = {'schema': 'merlin.tier_instrument_retirement.v1', 'tier': tier,
                'selected_instrument': instrument, 'retired': retired,
                'previous_state': before,
                'previous_ledger_json': ledger_bytes.decode(),
                'previous_ledger_sha256': hashlib.sha256(ledger_bytes).hexdigest(),
                'authority': 'scheduling only; no new certificate'}
    raw = (json.dumps(document, sort_keys=True, indent=2) + '\n').encode()
    digest = hashlib.sha256(raw).hexdigest()
    directory = Path(ws) / 'qa' / 'tier_instrument_history'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (digest + '.json')
    try:
        with path.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != raw:
            raise ValueError('instrument history archive differs from its identity')
    for row in retired:
        name, identity, entry = row['capsule'], row['identity'], row['entry']
        P._slots(state, name, tier).pop(identity)
        if state[name].get(tier) == entry:
            state[name].pop(tier)
    return {'path': str(path), 'sha256': digest, 'retired_records': len(retired),
            'selected_instrument': instrument, 'tier': tier}


def reconcile(ws, *, context, tier='L3', expected_ledger_sha256, apply=False):
    """Reviewed operator preview/apply; derive the judge from native context.

    The optimistic byte pin prevents applying an earlier preview to a ledger
    changed by a concurrent native result. This never submits target work.
    """
    from . import promotion as P

    @P._ledger_transaction
    def locked(ws):
        ledger = Path(ws) / 'qa' / 'tier_state.json'
        raw = ledger.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_ledger_sha256:
            raise ValueError('tier ledger changed since operator review')
        state = json.loads(raw)
        instrument = P.cert_instrument(tier, context=context)
        if not isinstance(instrument, str) or len(instrument) != 64:
            raise ValueError('selected certificate instrument unavailable')
        rows = [{'capsule': name, 'identity': identity, 'instrument': entry.get('instrument'),
                 'status': entry.get('status')}
                for name in state for identity, entry in P._slots_ro(state, name, tier).items()
                if isinstance(entry, dict) and entry.get('status') in {'pass', 'fail', 'joined'}
                and entry.get('instrument') != instrument]
        result = {'schema': 'merlin.tier_instrument_reconciliation.v1', 'applied': apply,
                  'previous_ledger_sha256': expected_ledger_sha256,
                  'selected_instrument': instrument, 'tier': tier, 'retired': rows}
        if apply:
            result['archive'] = retire(ws, state, tier=tier, instrument=instrument)
            P._save_tier_state(ws, state)
            result['ledger_sha256'] = hashlib.sha256(ledger.read_bytes()).hexdigest()
        return result

    return locked(ws)
