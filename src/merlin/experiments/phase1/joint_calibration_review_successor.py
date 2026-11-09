"""Evidence-only repair of an operator calibration denial, never fit authority.

The original native selection remains immutable. A replacement calibration
owner may select this strictly additive repair for its watcher alone.
"""
from __future__ import annotations

import copy
import json

from .feedback import joint_performance as PERF
from .joint_calibration_admission import ENV

SCHEMA = 'merlin.native_calibration_review_successor.v1'


def select(selection, successor_pin):
    """Return a watcher-only selection with missing evidence repaired.

    Existing denials, order, parameters, reasons and evidence cannot change.
    Only absent/empty evidence arrays may receive one to sixteen pinned JSON
    documents. This function starts no process and mutates no selected input.
    """
    successor = PERF._read(successor_pin)
    if (not isinstance(successor, dict) or successor.get('schema') != SCHEMA or not successor.get('review_id')
            or successor.get('fit_authorized') is not False
            or successor.get('native_continuation_authorized') is not False):
        raise ValueError('calibration review successor lacks evidence-only scope')
    previous_pin = json.loads(selection['environment'][ENV])
    if successor.get('previous_review') != previous_pin:
        raise ValueError('calibration review successor selects another original review')
    previous = PERF._read(previous_pin)
    replacement_pin = successor['review']
    replacement = PERF._read(replacement_pin)
    if (not isinstance(previous, dict) or not isinstance(replacement, dict)
            or previous.get('schema') != 'merlin.native_calibration_admission_review.v1'):
        raise ValueError('original calibration denial schema is unsupported')
    old_rows, new_rows = previous.get('deny'), replacement.get('deny')
    if (not isinstance(old_rows, list) or not 1 <= len(old_rows) <= 64
            or not isinstance(new_rows, list) or len(new_rows) != len(old_rows)
            or {k:v for k,v in previous.items() if k != 'deny'}
            != {k:v for k,v in replacement.items() if k != 'deny'}):
        raise ValueError('calibration review successor changed denial roster or authority')
    repaired = []
    for before, after in zip(old_rows, new_rows, strict=True):
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ValueError('calibration denial row is malformed')
        if before == after:
            continue
        if ({k:v for k,v in before.items() if k != 'evidence'}
                != {k:v for k,v in after.items() if k != 'evidence'}
                or before.get('evidence') not in (None, [])):
            raise ValueError('calibration review successor changed an existing denial')
        evidence = after.get('evidence')
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 16:
            raise ValueError('calibration review repair lacks bounded evidence')
        for pin in evidence:
            PERF._read(pin)
        repaired.append(after['model_sha256'])
    if not repaired:
        raise ValueError('calibration review successor repairs no missing evidence')
    result = copy.deepcopy(selection)
    result['environment'][ENV] = json.dumps(replacement_pin, sort_keys=True)
    result['calibration_review_successor'] = dict(
        pin=successor_pin, previous_review=previous_pin, review=replacement_pin,
        repaired_models=repaired, fit_authorized=False,
        native_continuation_authorized=False)
    return result
