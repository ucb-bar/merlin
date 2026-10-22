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


def _split_l3_prerequisites(row, source_sha256):
    """Recognize a completed cheap stage, not a certificate or complete result."""
    from merlin.targetgen.trace_check import requirements_complete

    tiers = row.get('tiers') or {}
    failure = row.get('failure') or {}
    l3 = tiers.get('L3') or {}
    source = row.get('compiler_source') or {}
    numeric = row.get('numeric') or {}
    outputs = numeric.get('per_output')
    if not (row.get('status') == 'incomplete'
            and failure.get('plane') == 'oracle_unavailable'
            and failure.get('category') == 'NOT_RUN_IS_NOT_PASS'
            and failure.get('tier') == 'L3' and failure.get('tier_status') == 'unavailable'
            and l3.get('status') == 'unavailable' and l3.get('mandatory') is True
            and l3.get('reason') == ("this capsule declares required oracle tier L3 (elaborated_rtl) "
                "but this phase supplies no adapter for it; reachable tiers here: ['L2']")
            and failure.get('tier_reason') == l3['reason']
            and source.get('status') == 'stable' and source_sha256
            and source.get('candidate_sha256') == source_sha256
            and requirements_complete(row.get('trace_check'))
            and not (row.get('trace_check') or {}).get('violations')
            and not (row.get('trace_check') or {}).get('unavailable_requirements')
            and isinstance(outputs, dict) and outputs
            and numeric.get('missing_outputs') == [] and numeric.get('mismatch_count') == 0
            and all(isinstance(v, dict) and v.get('status') == 'pass'
                    and type(v.get('n_elements')) is int and v['n_elements'] > 0
                    for v in outputs.values())):
        return False
    # L0/L1 can be genuinely inapplicable to whole-program execution. The
    # shared native checker below validates their exact paired reasons and
    # complete numerical output evidence; it does not accept arbitrary skips.
    # No other unavailable/failed tier may hide behind the expected L3 gap.
    if any(t not in {'L0', 'L1', 'L3'} and record.get('mandatory') is True
            and record.get('status') != 'pass' for t, record in tiers.items()):
        return False
    # The private projection is used for scheduling only. The original durable
    # result, failure, curriculum credit and certification state stay unchanged.
    return _passed(dict(row, failure=None)) == 'spike_correctness'


def eligible(rows, *, source_root=None):
    """Only closed native structural/numerical receipts authorize a physical job."""
    required = roster()
    source_sha256 = None
    if source_root is not None and any(row.get('failure') for name, row in rows.items() if name in required):
        from merlin.common.tree_hash import hash_tree
        source_sha256 = hash_tree(Path(source_root))['sha256']
    ready = set()
    for name, row in rows.items():
        trace = row.get('trace_check') or {}
        if name not in required or trace.get('status') == 'fail' or trace.get('violations'):
            continue
        if _passed(row) == 'spike_correctness' or _split_l3_prerequisites(row, source_sha256):
            ready.add(name)
    return ready


def selected_request(request):
    return (request.get('joint_physical_correctness') is True and request.get('promoted') is True
            and request.get('sim') == 'gsim' and request.get('tiers') == 'L3'
            and request.get('capsules') in roster())


def project(ws, state, *, source=None, instrument=None, closed_rows=None):
    """Project protected records; cross-source reuse needs a matching native L2 identity."""
    from . import promotion as P
    from .tier_boundary_join import reusable_certificate
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
        loops = [v for v in P._slots_ro(state, name, 'L2').values()
                 if v.get('digest') == source and v.get('status') == 'pass']
        observed = {v.get('execution_digest') for v in P._slots_ro(state, name, 'L2').values()
                    if v.get('digest') == source and v.get('status') == 'pass'
                    and P._valid_execution(v.get('execution_digest'))}
        records, reuse = [], {}
        for value in P._slots_ro(state, name, 'L3').values():
            if not instrument or value.get('instrument') != instrument:
                continue
            boundary = next((matched for loop in loops if (matched := reusable_certificate(
                value, loop, name=name, source=source, instrument=instrument)) is not None), None)
            if value.get('digest') == source or value.get('execution_digest') in observed or boundary is not None:
                records.append(value)
                if boundary is not None:
                    reuse[value['execution_digest']] = boundary
        # A queued duplicate is not a measured counterexample to an applicable
        # certificate. A later recorded failure still takes precedence.
        if records and records[-1].get('status') in {'pending', 'joined'}:
            for record in reversed(records):
                if record.get('status') == 'fail':
                    break
                if record.get('status') == 'pass' and record.get('execution_digest') in reuse:
                    records.append(record)
                    break
        status = records[-1].get('status') if records else 'pending'
        obligations[name] = {'status': 'expected_refusal_satisfied' if name in refusals else
            status if status in {'pass','fail','pending'} else 'pending',
            'physical_executed': None, 'source_sha256':source,
            'certificate_source_sha256':records[-1].get('digest') if records else None,
            'reused_execution_identity':records[-1].get('execution_digest')
                if records and records[-1].get('digest') != source else None,
            'boundary_reuse':reuse.get(records[-1].get('execution_digest')) if records else None}
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
