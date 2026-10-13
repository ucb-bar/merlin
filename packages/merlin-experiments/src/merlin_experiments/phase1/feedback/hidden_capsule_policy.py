"""Explicit hidden-capsule corroboration policy; no corpus or answer discovery."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

MODES = ('rtl', 'functional_spike')


def record(mode):
    if mode not in MODES:
        raise ValueError('unknown hidden capsule validation mode')
    return {
        'mode': mode,
        'required_tier': 'L2' if mode == 'functional_spike' else 'L3',
        'engine': 'spike' if mode == 'functional_spike' else None,
        'claim': 'functional_corroboration' if mode == 'functional_spike' else 'rtl_certification',
        'scope': 'hidden_capsules_after_freeze',
    }


def resolve(descriptor: Path, requested=''):
    """An operator may opt in when the descriptor leaves its policy unspecified."""
    if type(requested) is not str:
        raise ValueError('operator hidden capsule mode is malformed')
    try:
        document = yaml.safe_load(Path(descriptor).read_bytes())
    except FileNotFoundError:
        if not requested:
            return record('rtl')  # Historical diagnostics never gain a cheaper qualification.
        raise ValueError('explicit hidden capsule mode requires its selected descriptor') from None
    if not isinstance(document, Mapping):
        raise ValueError('hidden capsule descriptor is malformed')
    grading = document.get('grading')
    if grading is None:
        grading = {}
    if not isinstance(grading, Mapping):
        raise ValueError('hidden capsule grading declaration is malformed')
    declared = grading.get('hidden_capsule_mode')
    if declared is not None and declared not in MODES:
        raise ValueError('descriptor hidden capsule mode is invalid')
    if requested and requested not in MODES:
        raise ValueError('operator hidden capsule mode is invalid')
    if declared is not None and requested and requested != declared:
        raise ValueError('operator hidden capsule mode conflicts with descriptor')
    return record(requested or declared or 'rtl')


def verify_record(selected, observed):
    # Historical runs with the default policy have no additive policy field.
    if observed is None and selected == record('rtl'):
        return
    if not isinstance(observed, Mapping) or dict(observed) != selected:
        raise ValueError('hidden capsule validation differs from frozen run selection')


def evidence_failures(score):
    """Require positive measured Spike evidence, never infer it from a tier name."""
    evidence = score.get('functional_hidden_evidence')
    if not isinstance(evidence, Mapping):
        return ['functional_hidden_evidence_missing_or_malformed']
    failures = []
    for key, expected in (
        ('tier', 'L2'), ('engine', 'spike'), ('fidelity', 'functional_model'),
        ('claim', 'functional_corroboration'),
    ):
        if evidence.get(key) != expected:
            failures.append(f'functional_hidden_{key}_invalid')
    for key in ('n_admitted_capsules', 'n_spike_passed', 'n_validated_refusals', 'n_higher_tier_measured'):
        if type(evidence.get(key)) is not int or evidence[key] < 0:
            failures.append(f'functional_hidden_{key}_malformed')
    if evidence.get('n_admitted_capsules') != score.get('n_capsules'):
        failures.append('functional_hidden_denominator_mismatch')
    from merlin.targetgen.expected_refusal import execution_denominator

    executable, refusal_failures = execution_denominator(score)
    failures.extend(refusal_failures)
    if evidence.get('n_spike_passed') != executable:
        failures.append('functional_hidden_spike_not_complete')
    scope = score.get('expected_refusal_scope')
    refusals = scope.get('n_validated_refusals', 0) if isinstance(scope, Mapping) else 0
    if evidence.get('n_validated_refusals') != refusals:
        failures.append('functional_hidden_refusal_count_mismatch')
    if evidence.get('n_higher_tier_measured') != 0:
        failures.append('functional_hidden_higher_fidelity_measured')
    return failures
