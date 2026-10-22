"""Verify operator-selected implementation corrections without repricing models."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .feedback import joint_performance as PERF


def _fixed_model(runtime):
    model = json.loads(runtime.model_json)
    # These are implementation/ISA declaration corrections. No price, capacity,
    # ordering policy, scope, or calibration domain may change by this route.
    model.pop('libraries', None)
    model.get('cpu', {}).pop('isa', None)
    return model


def _verify_scope_extension(old, new, evidence):
    """Admit explicit new primitive classes, preserving every existing price."""
    if evidence.get('schema') != 'merlin.model_scope_extension_review.v1':
        raise ValueError('structural scope extension lacks its explicit review')
    PERF._read(evidence['source_audit'])
    before, after = _fixed_model(old), _fixed_model(new)
    prices = before.get('target', {}).get('prices', {})
    new_prices = after.get('target', {}).get('prices', {})
    if any(new_prices.get(k) != v for k, v in prices.items()):
        raise ValueError('structural scope extension changes existing prices')
    additions = {k:v for k,v in new_prices.items() if k not in prices}
    if not additions or additions != evidence.get('added_primitive_prices'):
        raise ValueError('new primitive service prices differ from reviewed assumptions')
    after['target']['prices'] = copy.deepcopy(prices)
    scheduling = before['target']['scheduling']
    for change in evidence.get('scheduling_changes', []):
        key = change['key']
        if scheduling.get(key) != change.get('before'):
            raise ValueError('structural scheduling review has stale prior value')
        scheduling[key] = change['after']
    if before != after:
        raise ValueError('structural scope extension changes unreviewed model facts')


def _verify_capability_restriction(old, new, evidence):
    """Bind reviewed disabled optional features without changing timing prices."""
    if evidence.get('schema') != 'merlin.target_capability_restriction_review.v1':
        raise ValueError('capability restriction lacks its explicit review')
    PERF._read(evidence['source_audit'])
    disabled = evidence.get('disabled_optional_features')
    if (not isinstance(disabled, list) or not disabled
            or any(not isinstance(key, str) or not key.isidentifier() for key in disabled)
            or len(disabled) != len(set(disabled))):
        raise ValueError('capability restriction needs explicit optional feature names')
    before, after = _fixed_model(old), _fixed_model(new)
    hardware = before.get('target', {}).get('hardware', {})
    new_hardware = after.get('target', {}).get('hardware', {})
    for key in disabled:
        if hardware.get(key) not in (None, 1) or new_hardware.get(key) != 0:
            raise ValueError('capability restriction must disable a reviewed optional feature')
        hardware[key] = 0
    if before != after:
        raise ValueError('capability restriction changes prices or unreviewed model semantics')


def validate(initial, current):
    from .joint_source_revision import normalized
    comparable = normalized(initial, current)
    excluded = {'runtime', 'timing_epochs', 'runtime_revisions', 'source_binding_revisions'}
    if ({k:v for k,v in initial.items() if k not in excluded}
            != {k:v for k,v in comparable.items() if k not in excluded}):
        raise ValueError('calibration profile does not cover the current non-model selection')
    prior = initial['runtime']
    revisions = current.get('runtime_revisions', [])
    if not isinstance(revisions, list) or len(revisions) > 16:
        raise ValueError('runtime corrections need a bounded reviewed chain')
    for revision in revisions:
        if (revision.get('schema') != 'merlin.runtime_implementation_revision.v1'
                or revision.get('previous') != prior or not revision.get('review_id')):
            raise ValueError('runtime correction ancestry or operator review is missing')
        evidence = PERF._read(revision['evidence'])
        old, new = PERF._runtime(prior), PERF._runtime(revision['runtime'])
        kind = revision.get('kind', 'implementation_correction')
        if kind == 'structural_scope_extension':
            _verify_scope_extension(old, new, evidence)
        elif kind == 'capability_restriction':
            _verify_capability_restriction(old, new, evidence)
        elif kind == 'structural_accounting_correction':
            from .joint_structural_revision import verify
            verify(old, new, evidence, previous=prior, current=revision['runtime'])
        elif kind != 'implementation_correction' or _fixed_model(old) != _fixed_model(new):
            raise ValueError('implementation correction changes prices or model semantics')
        prior = revision['runtime']
    if prior != current['runtime']:
        raise ValueError('base runtime changed without reviewed implementation ancestry')


def prepare(selection_pin, *, runtime_pin, evidence_pin, review_id, output,
            kind="implementation_correction"):
    """Produce a reviewable new selection; never drain or launch a native owner."""
    from . import joint_calibration as C
    original = PERF._read(selection_pin)
    if original.get('timing_epochs'):
        # Never silently replace or regress already tuned coverage. A later
        # implementation revision must explicitly rebase each tuned model.
        raise ValueError('tuned models require individual implementation rebasing')
    revised = copy.deepcopy(original)
    revised.setdefault('runtime_revisions', []).append(dict(
        schema='merlin.runtime_implementation_revision.v1', previous=original['runtime'],
        runtime=runtime_pin, evidence=evidence_pin, review_id=review_id, kind=kind))
    revised['runtime'] = runtime_pin
    baseline = copy.deepcopy(original)
    if original.get('runtime_revisions'):
        baseline['runtime'] = original['runtime_revisions'][0]['previous']
    validate(baseline, revised)
    return C._write(Path(output), revised)


def request_promotion(selection_pin, *, previous_run, review_id, output):
    """Reopen an operator-reviewed correction before requesting native drain."""
    from . import joint_epochs
    state = json.loads((Path(previous_run) / 'joint_campaign.json').read_text())
    old = PERF._read(state['performance_selection'])
    new = PERF._read(selection_pin)
    prior = old.get('runtime_revisions', [])
    revisions = new.get('runtime_revisions', [])
    if (len(revisions) <= len(prior) or revisions[:len(prior)] != prior
            or new.get('timing_epochs', []) != old.get('timing_epochs', [])):
        raise ValueError('runtime correction changed tuned epochs or lost review ancestry')
    baseline = copy.deepcopy(old)
    if prior:
        baseline['runtime'] = prior[0]['previous']
    validate(baseline, new)
    return joint_epochs.request_model_epoch(previous_run,
        selection_path=selection_pin['path'], expected_sha256=selection_pin['sha256'],
        review_id=review_id, output=output)
