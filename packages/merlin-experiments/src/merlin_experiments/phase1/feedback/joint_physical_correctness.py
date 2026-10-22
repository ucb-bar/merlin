"""Original-corpus correctness admission into the native certificate queue.

No calibration fit, timing qualification, or new simulator runner lives here.
"""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
from .. import joint_control
from .joint_curriculum import _passed


def roster():
    state = joint_control.selected()
    return set((state or {}).get('policy', {}).get('gsim_correctness', {}).get('capsules', ()))


def eligible(rows):
    """Only closed native structural/numerical receipts authorize a physical job."""
    required = roster()
    return {name for name, row in rows.items() if name in required and _passed(row) == 'spike_correctness'}


def selected_request(request):
    return (request.get('joint_physical_correctness') is True and request.get('promoted') is True
            and request.get('sim') == 'gsim' and request.get('tiers') == 'L3'
            and request.get('capsules') in roster())


def project(ws, state, *, source=None, instrument=None, closed_rows=None):
    """Project protected records; cross-source reuse needs a matching native L2 identity."""
    from . import promotion as P
    required = roster()
    if not required:
        return
    path = Path(ws) / 'qa/joint_physical_correctness.json'
    previous = json.loads(path.read_text()) if path.is_file() else {}
    source = source or previous.get('source_sha256')
    instrument = instrument or previous.get('instrument_sha256')
    if not source:
        return
    refusals = set(previous.get('expected_refusal_satisfied', ())) if previous.get('source_sha256') == source else set()
    refusals |= {name for name, row in (closed_rows or {}).items()
                 if name in required and _passed(row) == 'expected_refusal'}
    obligations = {}
    for name in sorted(required):
        observed = {v.get('execution_digest') for v in P._slots_ro(state, name, 'L2').values()
                    if v.get('digest') == source and v.get('status') == 'pass'
                    and P._valid_execution(v.get('execution_digest'))}
        records = [v for v in P._slots_ro(state, name, 'L3').values()
                   if v.get('instrument') == instrument and instrument
                   and (v.get('digest') == source or v.get('execution_digest') in observed)]
        status = records[-1].get('status') if records else 'pending'
        obligations[name] = {'status': 'expected_refusal_satisfied' if name in refusals else
            status if status in {'pass','fail','pending'} else 'pending',
            'physical_executed': None, 'source_sha256':source,
            'certificate_source_sha256':records[-1].get('digest') if records else None,
            'reused_execution_identity':records[-1].get('execution_digest')
                if records and records[-1].get('digest') != source else None}
    statuses = [row['status'] for row in obligations.values()]
    report = {'schema':'merlin.joint_physical_correctness.v1', 'source_sha256':source,
        'instrument_sha256':instrument, 'required':len(required), 'obligations':obligations,
        'passed':statuses.count('pass'), 'failed':statuses.count('fail'),
        'pending':statuses.count('pending'), 'expected_refusal_satisfied':sorted(refusals),
        'complete':bool(statuses) and all(s in {'pass','expected_refusal_satisfied'} for s in statuses),
        'scope':'original capsule L3 correctness; not timing calibration or full-target validation',
        'source_currentness':'exact reported source only; no claim about current author edits'}
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix='.physical-',delete=False) as stream:
        json.dump(report,stream,indent=2);stream.write('\n');temporary=stream.name
    os.replace(temporary,path)


def attach(path, verdict):
    """Keep the physical denominator visible beside cheap snapshot progress."""
    if not roster():
        return verdict
    report_path=Path(path).parent/'joint_physical_correctness.json'
    try:
        report=json.loads(report_path.read_text())
        summary={key:report[key] for key in ('required','passed','failed','pending','complete',
            'expected_refusal_satisfied','source_sha256','instrument_sha256','scope','source_currentness')}
    except (OSError,ValueError,KeyError):
        summary={'required':len(roster()),'passed':0,'failed':0,'pending':len(roster()),
                 'complete':False,'source_sha256':None,'scope':'original capsule L3 correctness pending'}
    summary['artifact']='qa/joint_physical_correctness.json'
    verdict['joint_physical_correctness']=summary
    verdict.setdefault('feedback_brief',{})['physical_correctness']=summary
    return verdict
