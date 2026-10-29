"""Operator-only bounded evidence inventory. Never writes grading state or grants credit.

Includes ordinary raw selfchecks as well as automatic grades and issued scalability
products. Request fingerprints are not substituted for native compiler-tree hashes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .terminal_capsules import obj, full_source, project


class Reader:
    def __init__(self, max_bytes=256 << 20):
        self.remaining = max_bytes
        self.pins = {}

    def read(self, path, *, yaml=False, sha256=None):
        path = Path(path)
        if path.is_symlink():
            raise ValueError(f'indirect evidence: {path}')
        before = path.stat()
        if before.st_size > min(16 << 20, self.remaining):
            raise ValueError(f'evidence read bound exceeded: {path}')
        data = path.read_bytes()
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError(f'evidence changed during read: {path}')
        digest = hashlib.sha256(data).hexdigest()
        if sha256 is not None and digest != sha256:
            raise ValueError(f'evidence pin differs: {path}')
        self.remaining -= len(data)
        self.pins[str(path)] = dict(path=str(path), sha256=digest, bytes=len(data))
        if yaml:
            import yaml as Y
            return Y.safe_load(data)
        return json.loads(data)


def raw_paths(run):
    """Fixed-depth native paths, including the ordinary broker's private store."""
    base = Path(run) / '_qa_work'
    for pattern in ('fruns*/runs/*/*/capsule_result.json',
                    'selfchecks/*/runs/*/*/capsule_result.json'):
        yield from base.glob(pattern)


def issued_scalability_rows(workspace, reader):
    """Reopen the issued qualification, never infer numerics from compact cases."""
    channel = Path(workspace) / '.qa_channel'
    for issued in sorted(channel.glob('.resp_*.scalability/issued.json')):
        record = reader.read(issued)
        result = record['result']
        if Path(result['path']) != issued.parent / 'qualification.json':
            raise ValueError('scalability result is outside its issued directory')
        issuer = Path(record['issuer']['path'])
        if not issuer.is_relative_to(channel / 'issuers/invocations'):
            raise ValueError('scalability issuer is outside the native issuer store')
        reader.read(issuer, sha256=record['issuer']['sha256'])
        qualification = reader.read(result['path'], sha256=result['sha256'])
        if qualification.get('schema') != 'merlin.phase1.public_scalability.v1':
            raise ValueError('unexpected scalability qualification schema')
        source = qualification.get('candidate_sha256')
        if not full_source(source):
            raise ValueError('scalability qualification lacks native source identity')
        scored = {r['capsule']: r for r in qualification.get('score', {}).get('per_capsule', [])}
        for path in sorted((issued.parent / 'runs/runs').glob('*/*/capsule_result.json')):
            row = reader.read(path)
            score = scored.get(row.get('capsule'))
            if not score:
                continue
            actual = obj(row.get('compiler_source')).get('candidate_sha256')
            # Failed compilation may carry no source; its issued parent supplies it.
            if actual not in (None, source):
                raise ValueError('raw scalability source differs from issued qualification')
            if (row.get('status') != score.get('status')
                    or obj(row.get('numeric')).get('status') != score.get('numeric')):
                raise ValueError('raw scalability result differs from issued summary')
            yield row, path, source


def boundary_coverage(state, *, source, instrument):
    """Use existing native proof validators; no new cross-source union rule."""
    from ..phase1.feedback.promotion import _slots_ro
    from ..phase1.feedback.tier_boundary_join import join, reusable_certificate

    matched = {}
    for name in state:
        loops = [r for r in _slots_ro(state, name, 'L2').values()
                 if r.get('status') == 'pass'
                 and obj(obj(r.get('tier_boundary')).get('common')).get('compiler_sha256') == source]
        for loop in loops:
            boundary = loop['tier_boundary']
            digest = boundary['common'].get('submission_digest')
            if (not digest or loop.get('digest') != digest
                    or loop.get('execution_digest') != boundary.get('execution_digest')):
                continue
            for cert in _slots_ro(state, name, 'L3').values():
                if cert.get('status') != 'pass' or cert.get('instrument') != instrument or not instrument:
                    continue
                try:
                    proof = join(boundary, cert.get('tier_boundary'))
                    if (proof['common']['capsule'] != name or cert.get('digest') != digest
                            or cert.get('execution_digest') != proof['L3']['execution_digest']):
                        continue
                    matched[name] = dict(kind='same_source_native_boundary', proof=proof)
                except (KeyError, TypeError, ValueError):
                    proof = reusable_certificate(cert, loop, name=name, source=digest, instrument=instrument)
                    if proof is not None:
                        matched[name] = dict(kind='compatible_native_certificate_reuse', proof=proof)
    return matched


def audit(run, *, selected_source=None):
    run = Path(run).resolve()
    reader = Reader()
    started = datetime.now(timezone.utc).isoformat()
    env = reader.read(run / 'environment.yaml', yaml=True)
    ws = Path(env['workspace_path'])
    policy = reader.read(run / 'joint_campaign.json')['policy']
    cal = policy['native_calibration']
    calibration = reader.read(cal['path'], sha256=cal.get('sha256'))
    scale = Path(env['public_corpus_input']['public_scalability']['path']) / 'selection.json'
    scale = reader.read(scale)
    rosters = {'functional': set(policy['gsim_correctness']['capsules']),
               'calibration': set(calibration['members']), 'scalability': set(scale['cases'])}
    ledger = reader.read(ws / 'qa/tier_state.json')
    physical = reader.read(ws / 'qa/joint_physical_correctness.json')
    progress = reader.read(ws / 'qa/progress.json')
    documents, sources, failures, inventory = [], set(), [], Counter()
    paths = sorted(set(raw_paths(run)))
    if len(paths) > 2048:
        raise ValueError('native result count exceeds bounded audit')
    rows = [(reader.read(p), p, None) for p in paths]
    rows.extend(issued_scalability_rows(ws, reader))
    for row, path, issued_source in rows:
        source = full_source(obj(row.get('compiler_source')).get('candidate_sha256')) or issued_source
        if not source:
            inventory['unattributed_rows'] += 1
            continue
        sources.add(source)
        documents.append(({'graded_source_sha256': source, 'per_capsule': [row]}, str(path)))
        family = 'scalability' if '.scalability/' in str(path) else 'selfcheck' if '/selfchecks/' in str(path) else 'automatic'
        inventory[family] += 1
        failure = obj(row.get('failure'))
        if failure and failure.get('category') != 'NOT_RUN_IS_NOT_PASS':
            failures.append(dict(source=source, capsule=row.get('capsule'), failure=failure,
                                 receipt=str(path)))
    for state in ledger.values():
        for tier in ('L2', 'L3'):
            records = [obj(state).get(tier), *obj(obj(obj(state).get('<certs>')).get(tier)).values()]
            for record in records:
                source = full_source(obj(obj(obj(record).get('tier_boundary')).get('common')).get('compiler_sha256'))
                if source:
                    sources.add(source)
    if selected_source is not None:
        if not full_source(selected_source):
            raise ValueError('selected source must be an exact native compiler tree hash')
        sources.add(selected_source)
    by_source = []
    for source in sorted(sources):
        counts = project(rosters, {'graded_source_sha256': source}, ledger, physical, documents)
        joined = boundary_coverage(ledger, source=source, instrument=physical.get('instrument_sha256'))
        blockers = {r['capsule'] for r in failures if r['source'] == source}
        usable = set(joined) - blockers
        for cohort in counts['cohorts']:
            names = usable & rosters[cohort['cohort']]
            cohort['validated_native_L2_L3_join'] = len(names)
            cohort['compatible_physical_reuse'] = sum(joined[n]['kind'] == 'compatible_native_certificate_reuse' for n in names)
        by_source.append(dict(source=source, cohorts=counts['cohorts'], total=counts['total'],
            validated_native_L2_L3_join=len(usable), compatible_physical_reuse=sum(
                joined[n]['kind'] == 'compatible_native_certificate_reuse' for n in usable),
            joined_cases={n: joined[n] for n in sorted(usable)}))
    historical = {name: project({name: members}, {}, ledger, physical, documents)['historical_union']
                  for name, members in rosters.items()}
    best = max(by_source, key=lambda r: (r['total']['L2'] or 0, r['total']['L3'] or 0), default=None)
    return dict(schema='operator.current_qualification_audit.v2', run=str(run),
        started_at=started, finished_at=datetime.now(timezone.utc).isoformat(), inventory=dict(inventory),
        selected_source=selected_source, selected=next((r for r in by_source if r['source'] == selected_source), None),
        best_single_source=best, by_source=by_source, historical_union=historical,
        failure_rows=failures, native_projection_source=physical.get('source_sha256'),
        instrument=physical.get('instrument_sha256'), pins=list(reader.pins.values()),
        semantics=['Each native file read stably; live campaign observation is not globally atomic.',
                   'Historical union is not current compiler qualification.',
                   'Native boundary joins are separately reported using existing native validators; no runtime ledger write.',
                   'A selected source is an explicit observation, never inferred from stale progress or a request fingerprint.',
                   'No new compilation, simulation, grading, or acceptance credit is issued.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--source')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.run, selected_source=args.source)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(args.output, hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
