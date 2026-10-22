"""Retain historical certificates without letting another judge suppress new work.

This changes scheduling state only. Native promotion leaves pending records
untouched. An explicit operator reconciliation can retire exact inherited
placeholders only after proving that the current epoch has no request owner;
original jobs and their old-workspace evidence retain their identities.
"""
from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path


def retire(ws, state, *, tier, instrument, inherited_pending=(), inheritance_proof=None,
           retire_terminal=True):
    """Called under the native tier ledger lock; archive before changing any state."""
    from . import promotion as P

    if not isinstance(instrument, str) or len(instrument) != 64 or any(
            c not in '0123456789abcdef' for c in instrument):
        return None
    retired = []
    for name in state:
        for identity, entry in P._slots_ro(state, name, tier).items():
            if (isinstance(entry, dict) and ((retire_terminal and entry.get('status') in {'pass', 'fail', 'joined'})
                    or (entry.get('status') == 'pending' and (name, identity) in inherited_pending))
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
    if inheritance_proof is not None:
        document['inheritance_proof'] = inheritance_proof
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


def reconcile_inherited_pending(ws, *, context, current_run, epoch_pin,
                                expected_ledger_sha256, tier='L3', apply=False):
    """Retire only exact predecessor placeholders unclaimed by this epoch.

    Old brokers own their old workspace. An inherited ledger row is not a live
    job here unless this workspace's broker has its own request for that identity.
    Age and process-ID guesses are deliberately not used.
    """
    from . import promotion as P
    import yaml

    @P._ledger_transaction
    def locked(ws):
        def pinned(pin):
            path = Path(pin['path'])
            raw = path.read_bytes()
            if path.is_symlink() or hashlib.sha256(raw).hexdigest() != pin['sha256']:
                raise ValueError('inherited ledger provenance pin changed')
            return json.loads(raw)

        epoch = pinned(epoch_pin)
        run = Path(current_run)
        state_path = run / 'joint_campaign.json'
        env_path = run / 'environment.yaml'
        if (json.loads(state_path.read_text()).get('epoch') != epoch
                or Path(yaml.safe_load(env_path.read_text())['workspace_path']).resolve() != Path(ws).resolve()):
            raise ValueError('epoch is not selected by this workspace')
        previous = pinned(epoch['tier_ledger'])
        summary_pin = {'path': str(Path(epoch['previous_run']) / 'joint_campaign_summary.json'),
                       'sha256': epoch['pins']['joint_campaign_summary.json']}
        summary = pinned(summary_pin)
        if summary.get('completion_reason') != 'library_epoch':
            raise ValueError('predecessor did not close its library epoch')
        ledger = Path(ws) / 'qa' / 'tier_state.json'
        raw = ledger.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_ledger_sha256:
            raise ValueError('tier ledger changed since operator review')
        state = json.loads(raw)
        instrument = P.cert_instrument(tier, context=context)
        if not isinstance(instrument, str) or len(instrument) != 64:
            raise ValueError('selected certificate instrument unavailable')
        claims = set()
        request_pins = []
        for path in sorted((Path(ws) / '.qa_channel').glob('simreq_*.json')):
            contents = path.read_bytes()
            if path.is_symlink():
                raise ValueError('current promotion request is not a regular owned file')
            request = json.loads(contents)  # Unreadable request means ownership is unknown.
            if not isinstance(request, dict):
                raise ValueError('current promotion request is malformed')
            identity = request.get('identity')
            if isinstance(identity, str):
                claims.add(identity)  # Conservative even across capsule names.
            request_pins.append({'path': str(path), 'sha256': hashlib.sha256(contents).hexdigest()})
        eligible = {(name, identity) for name in state
                    for identity, entry in P._slots_ro(state, name, tier).items()
                    if isinstance(entry, dict) and entry.get('status') == 'pending'
                    and entry.get('instrument') != instrument and identity not in claims
                    and P._slots_ro(previous, name, tier).get(identity) == entry}
        proof = {'epoch': epoch_pin, 'previous_ledger': epoch['tier_ledger'],
                 'previous_summary': summary_pin, 'current_run': str(run),
                 'current_requests': request_pins,
                 'scope': 'exact inherited pending records with no current request owner'}
        result = {'schema': 'merlin.inherited_tier_reconciliation.v1', 'applied': apply,
                  'previous_ledger_sha256': expected_ledger_sha256, 'selected_instrument': instrument,
                  'tier': tier, 'retired': [{'capsule': n, 'identity': i} for n, i in sorted(eligible)],
                  'inheritance_proof': proof}
        if apply:
            # Only this explicit reviewed operator route can retire pending rows.
            # Terminal retirement remains the native default.
            result['archive'] = retire(ws, state, tier=tier, instrument=instrument,
                                      inherited_pending=eligible, inheritance_proof=proof,
                                      retire_terminal=False)
            P._save_tier_state(ws, state)
            result['ledger_sha256'] = hashlib.sha256(ledger.read_bytes()).hexdigest()
        return result

    return locked(ws)
