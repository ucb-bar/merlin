"""Bounded operator triage of public coverage questions; never dispatches or fits.

This reader is deliberately separate from the calibration registry selector.
A numerical pass, a mechanism match, or a row count grants no timing authority.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from merlin.runtime import _native_files as F
from .feedback import joint_performance as P

MAX_ROWS = 2048


def read_pin(pin):
    path = Path(pin['path'])
    if F.pin(path, max_bytes=16 << 20) != pin:
        raise ValueError('coverage priority input pin changed')
    return json.loads(path.read_text())


def collect(run_dirs):
    """Only explicit native run roots, with bounded published metadata reads."""
    rows = []
    for run in run_dirs:
        run = Path(run)
        paths = sorted((run/'calibration_candidate_index').glob('*.json'))
        paths += sorted((run/'calibration_registry').glob('*/entry.json'))
        if len(rows) + len(paths) > MAX_ROWS:
            raise ValueError('coverage priority input exceeds bounded roster')
        for path in paths:
            pin = F.pin(path, max_bytes=2 << 20)
            value = read_pin(pin)
            if value.get('schema') == 'merlin.joint_calibration_candidate.v1':
                row = dict(name=value['capsule_name'], source=value['capsule_sha256'],
                    compiler=value['compiler_sha256'], group=value.get('source_group_sha256'),
                    numeric=value.get('numerical_status'), report_pin=value.get('timing_report'),
                    native=value.get('native_result', {}), entry=pin, kind='broad_intake')
            else:
                cap, native = read_pin(value['capsule']), read_pin(value['result'])
                report = Path(value['generated_root'])/'joint_timing/report.json'
                row = dict(name=cap['name'], source=P.capsule_identity(cap),
                    compiler=value['compiler_sha256'], group=value.get('source_group_sha256'),
                    numeric=native.get('numeric', {}).get('status'),
                    report_pin=F.pin(report, max_bytes=16 << 20) if report.is_file() else None,
                    native=native, entry=pin, kind='profile_registry')
            report = read_pin(row['report_pin']) if row['report_pin'] else {}
            row.update(elf=report.get('elf_sha256'), model=report.get('model_sha256'),
                scope=report.get('scope_sha256'), domain=report.get('domain_sha256'),
                features=report.get('feature_counts', {}), unresolved=report.get('unresolved', []),
                timing_complete=report.get('timing_complete') is True,
                trace_status=row['native'].get('trace_check', {}).get('status'),
                retained_l3=row['native'].get('tiers', {}).get('L3', {}).get('status') == 'pass')
            row.pop('native')
            rows.append(row)
    return rows


def summarize(*, rows, profile, selection, runtime, denial, questions):
    if questions.get('schema') != 'merlin.public_model_coverage_questions.v1':
        raise ValueError('coverage question schema differs')
    qs = questions.get('questions', [])
    if len(qs) > 16 or any(q.get('evidence_domain') != 'public' for q in qs):
        raise ValueError('coverage questions must be bounded and explicitly public')
    model = runtime['model_sha256']
    paths = sorted(tuple(p['path']) for p in profile['parameters'])
    denials = [d for d in denial.get('deny', []) if d['model_sha256'] == model
               and sorted(tuple(p) for p in d['price_paths']) == paths]
    groups, observations = {}, []
    seen = set()
    for row in rows:
        member = selection.get('members', {}).get(row['name'])
        if member is None:
            continue  # No implicit admission of other or private workloads.
        current = row['source'] == member['capsule_sha256']
        key = (row['source'], row['compiler'], row['elf'], row['model'], row['scope'])
        if key in seen:
            continue
        seen.add(key)
        observed = {k: row[k] for k in ('name','source','compiler','elf','model','scope',
                    'numeric','timing_complete','trace_status','unresolved','entry','report_pin')}
        observed['source_current'] = current
        observed['model_current'] = row['model'] == model
        observed['retained_l3_reuse'] = ('requires_exact_ELF_input_scope_engine_config_join'
                                       if row['retained_l3'] else 'none_indexed')
        observed['dispatch_authorized'] = observed['fit_authorized'] = False
        observations.append(observed)
        group = profile.get('members', {}).get(row['name'], {}).get('source_group_sha256')
        if current and row['numeric'] == 'pass' and group and row['elf']:
            groups.setdefault(group, set()).add((row['compiler'], row['elf']))
    # Actual current-model failures outrank collecting variants for a quota.
    blockers = {}
    for row in observations:
        if not row['source_current'] or not row['model_current'] or row['numeric'] != 'pass':
            continue
        for reason in row['unresolved']:
            if reason == 'event timing or accounting is incomplete':
                continue  # Summary consequence, not an actionable root cause.
            blocker = blockers.setdefault(reason, dict(reason=reason, capsules=[], reports=[],
                next_action='Inspect selected RTL/provider coverage and exact retained public trace before changing prices; preserve UNKNOWN where unsupported.',
                dispatch_authorized=False, fit_authorized=False))
            blocker['capsules'].append(row['name'])
            if row['report_pin'] not in blocker['reports']: blocker['reports'].append(row['report_pin'])
    priorities = []
    for q in sorted(qs, key=lambda q: (q['priority'], q['id'])):
        related = [r for r in rows if r['name'] in selection.get('members', {}) and
                   any(r['features'].get(f, 0) for f in q.get('observed_features', []))]
        item = dict(q)
        item.update(observed_capsules=sorted({r['name'] for r in related}),
            current_model_observations=sum(r['model'] == model for r in related),
            unresolved_observations=sum(bool(r['unresolved']) for r in related),
            retained_evidence_action='reopen pinned public evidence before requesting another measurement',
            dispatch_authorized=False, fit_authorized=False)
        priorities.append(item)
    return dict(schema='merlin.operator_public_model_priority.v1',
        mode='read_only_triage_no_simulation_no_fit_no_promotion',
        model_sha256=model, automatic_fit_status='denied_structural_error' if denials else 'not_authorized_by_this_report',
        denial_reasons=[d['reason'] for d in denials],
        profile_scope=dict(parameter_paths=[p['path'] for p in profile['parameters']],
            reviewed_small_capsules=len(profile['members']),
            limitation='Only WS/PRELOAD prices are in this profile; CPU, FP, cache, divider and whole-SoC qualification do not follow from a fit.'),
        independent_variants=dict(current_source_groups=len(groups),
            source_groups_with_two_distinct_compilers_and_ELFs=sum(len({c for c,e in v}) >= 2 and len({e for c,e in v}) >= 2 for v in groups.values()),
            counts_are_not_measurement_targets=True),
        observations=observations, actual_model_blockers=list(blockers.values()), priorities=priorities,
        parallel_work=['Continue connected-model numerical/source-placement work and large-shape generalization with provisional feedback.',
                       'Do not wait for global model qualification or run gSIM merely to fill a variant quota.'],
        dispatch_authorized=False, fit_authorized=False)


def refresh(*, state, run_dir, output, denial_pin=None, questions_pin=None, now=None):
    """Owner-loop descriptive snapshot, at most once per minute and no subprocesses."""
    import time
    now = time.time() if now is None else now
    output = Path(output)
    if output.exists() and now - output.stat().st_mtime < 60:
        return None
    profile_pin = state['policy']['native_calibration']
    profile = read_pin(profile_pin)
    selection_pin = state.get('performance_selection') or profile['performance_selection']
    selection = read_pin(selection_pin)
    if selection.get('timing_epochs'):
        result = dict(schema='merlin.operator_public_model_priority.v1',
            status='requires_per_epoch_runtime_grouping',dispatch_authorized=False,fit_authorized=False)
    else:
        questions = read_pin(questions_pin) if questions_pin else {
            'schema':'merlin.public_model_coverage_questions.v1','questions':[]}
        for question in questions['questions']:
            for evidence in question.get('evidence', []): read_pin(evidence)
        result = summarize(rows=collect([run_dir]), profile=profile, selection=selection,
            runtime=read_pin(selection['runtime']), denial=read_pin(denial_pin) if denial_pin else {},
            questions=questions)
        result['inputs'] = dict(profile=profile_pin,selection=selection_pin,
                                admission_review=denial_pin,questions=questions_pin)
        result['status'] = 'current_public_evidence_reviewed'
    result['refreshed_at_epoch'] = now
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_suffix('.next.json')
    temporary.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    temporary.replace(output)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('profile','selection','admission-review','questions'):
        p.add_argument('--'+name, required=True, help='JSON {path,sha256} pin')
    p.add_argument('--run-dir', action='append', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args(argv)
    if len(a.run_dir)>16: p.error('at most16 explicit run roots')
    pins={k:json.loads(getattr(a,k.replace('-','_'))) for k in ('profile','selection','admission-review','questions')}
    docs={k:read_pin(v) for k,v in pins.items()}
    for question in docs['questions']['questions']:
        for evidence in question.get('evidence', []): read_pin(evidence)
    selection=docs['selection']
    if selection.get('timing_epochs'):
        raise ValueError('coverage triage requires explicit per-epoch runtime grouping once tuned scopes exist')
    runtime=read_pin(selection['runtime'])
    result=summarize(rows=collect(a.run_dir),profile=docs['profile'],selection=selection,
        runtime=runtime,denial=docs['admission-review'],questions=docs['questions'])
    result['inputs']=pins;result['run_roots']=list(map(str,a.run_dir))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(F.pin(a.output,max_bytes=16 << 20)))


if __name__ == '__main__': main()
