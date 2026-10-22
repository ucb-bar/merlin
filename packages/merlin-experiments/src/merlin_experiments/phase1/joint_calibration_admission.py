"""Additive operator review gate for automatic fitting of known structural errors.

This never authorizes measurement, changes budgets, or blocks direct diagnostics.
A selected review can only deny matching model/price-path combinations.
"""
from __future__ import annotations

import json
import os

from .feedback import joint_performance as PERF

ENV = 'MERLIN_NATIVE_CALIBRATION_ADMISSION_REVIEW'


def assert_allowed(plan):
    selection = os.environ.get(ENV)
    if not selection:
        return
    review = PERF._read(json.loads(selection))
    if review.get('schema') != 'merlin.native_calibration_admission_review.v1':
        raise ValueError('native calibration admission review has unsupported schema')
    rows = review.get('deny')
    if not isinstance(rows, list) or len(rows) > 64:
        raise ValueError('native calibration deny review exceeds bounded roster')
    model = PERF._read(plan['runtime'])['model_sha256']
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
