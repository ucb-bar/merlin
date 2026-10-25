"""Additive operator review gate for automatic fitting of known structural errors.

This never authorizes measurement, changes budgets, or blocks direct diagnostics.
A selected review can only deny matching model/price-path combinations.
"""
from __future__ import annotations

import json
import os

from .feedback import joint_performance as PERF

ENV = 'MERLIN_NATIVE_CALIBRATION_ADMISSION_REVIEW'


def _scoped_review(plan, runtime, active_review):
    """A diagnostic source hash change cannot escape its inherited fit denial."""
    if not runtime.get('model'):
        return
    model = PERF._read({'path': runtime['model'], 'sha256': runtime['model_sha256']})
    if model.get('diagnostics', {}).get('scoped_resources', {}).get('enabled') is not True:
        return
    selection = plan.get('selection_document')
    if selection is None and plan.get('selection'):
        selection = PERF._read(plan['selection'])
    if selection is None:
        current = PERF.selected()
        selection = current[1] if current else None
    if not isinstance(selection, dict):
        raise ValueError('scoped calibration requires selected reviewed runtime ancestry')
    revisions = selection.get('runtime_revisions', [])
    if not isinstance(revisions, list) or len(revisions) > 16:
        raise ValueError('scoped calibration requires bounded runtime ancestry')
    origins = [i for i, r in enumerate(revisions)
               if r.get('kind') == 'scoped_observer_diagnostics']
    if len(origins) != 1:
        raise ValueError('scoped calibration lacks its exact reviewed runtime ancestry')
    origin = revisions[origins[0]]
    evidence = PERF._read(origin['evidence'])
    if (evidence.get('schema') != 'merlin.scoped_observer_revision.v1'
            or evidence.get('runtime') != origin.get('runtime')
            or not evidence.get('continued_denial')):
        raise ValueError('scoped calibration ancestry differs from original review')
    prior = origin['runtime']
    denial = evidence['continued_denial']
    calibration_profile = evidence.get('calibration_profile')
    successors = revisions[origins[0]+1:]
    if successors:
        from .joint_provider_revision import KIND, verify
        from .joint_readout_revision import KIND as READOUT_KIND
        from .joint_readout_revision import PARTIAL_KIND
        from .joint_readout_revision import LOAD_KIND
        from .joint_readout_revision import BOOTSTRAP_KIND
        from .joint_readout_revision import verify as verify_readout
        from .joint_scoped_revision import verify as verify_origin
        verify_origin(PERF._runtime(origin['previous']), PERF._runtime(prior), evidence,
                      previous=origin['previous'], current=prior)
        for revision in successors:
            staged_kinds = (READOUT_KIND, PARTIAL_KIND, LOAD_KIND, BOOTSTRAP_KIND)
            if (revision.get('kind') not in (KIND, *staged_kinds) or revision.get('previous') != prior
                    or revision.get('schema') != 'merlin.runtime_implementation_revision.v1'
                    or not revision.get('review_id')):
                raise ValueError('scoped calibration has unreviewed provider ancestry')
            evidence = PERF._read(revision['evidence'])
            if (evidence.get('previous_denial') != denial
                    or evidence.get('calibration_profile') != calibration_profile):
                raise ValueError('scoped calibration provider loses inherited denial')
            current = revision['runtime']
            validator = verify_readout if revision['kind'] in staged_kinds else verify
            options = (dict(partial_rows=True) if revision['kind'] == PARTIAL_KIND else
                       dict(load_frontend=True) if revision['kind'] == LOAD_KIND else
                       dict(loader_bootstrap=True) if revision['kind'] == BOOTSTRAP_KIND else {})
            validator(PERF._runtime(prior), PERF._runtime(current), evidence,
                   previous=prior, current=current, **options)
            prior, denial = current, evidence['continued_denial']
    if (prior != plan['runtime'] or selection.get('runtime') != prior
            or evidence.get('model') != {'path':runtime['model'], 'sha256':runtime['model_sha256']}
            or active_review != denial):
        raise ValueError('scoped calibration active admission review differs from continued denial')
    # The normal denial check below reopens this exact current review and evidence.


def assert_allowed(plan):
    selection = os.environ.get(ENV)
    active_review = json.loads(selection) if selection else None
    runtime = PERF._read(plan['runtime'])
    _scoped_review(plan, runtime, active_review)
    if not selection:
        return
    review = PERF._read(active_review)
    if review.get('schema') != 'merlin.native_calibration_admission_review.v1':
        raise ValueError('native calibration admission review has unsupported schema')
    rows = review.get('deny')
    if not isinstance(rows, list) or len(rows) > 64:
        raise ValueError('native calibration deny review exceeds bounded roster')
    model = runtime['model_sha256']
    paths = sorted(tuple(p['path']) for p in plan['parameters'])
    for row in rows:
        if (not isinstance(row.get('model_sha256'), str)
                or len(row['model_sha256']) != 64 or not row.get('reason')):
            raise ValueError('native calibration deny row lacks model identity or reason')
        if row['model_sha256'] == model and sorted(tuple(p) for p in row['price_paths']) == paths:
            # Reopen only relevant evidence; changed/missing evidence fails closed.
            evidence = row.get('evidence')
            if not isinstance(evidence, list) or not evidence or len(evidence) > 16:
                raise ValueError('native calibration deny row lacks bounded evidence')
            for pin in evidence:
                PERF._read(pin)
            raise ValueError('blocked_structural_model_admission: ' + row['reason'])
