"""Operator-only completed-product registry and sparse calibration supervisor."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from merlin.benchharness import hash_tree
from merlin.runtime import _native_files as F

from . import joint_calibration as C, joint_control
from .feedback import joint_performance as PERF
from ..phase2.contracts import document_sha256


def register(row, root, *, result, compiler, capsule):
    """Called only by the native completed-row owner; no directory discovery."""
    state=joint_control.selected()
    if state is None or not state['policy'].get('native_calibration'):
        return
    profile=PERF._read(state['policy']['native_calibration'])
    # Descriptive public intake is deliberately separate from the reviewed fit
    # registry below. Failure here must not change legacy registration behavior.
    try:
        from . import joint_calibration_intake
        pin = joint_calibration_intake.register(row, root, result=result, compiler=compiler,
                                               capsule=capsule, state=state, profile=profile)
        if pin:
            row.setdefault('performance', {})['calibration_candidate_intake'] = {
                'status': 'pending_review', 'evidence_sha256': pin['sha256'],
                'dispatch_authorized': False, 'fit_authorized': False}
    except (OSError, ValueError, RuntimeError, KeyError, TypeError):
        row.setdefault('performance', {})['calibration_candidate_intake'] = 'unavailable'
    name=capsule.get('name')
    member=profile['members'].get(name)
    tier=result.get('tiers',{}).get('L2',{})
    if (member is None or result.get('numeric',{}).get('status')!='pass'
            or tier.get('status')!='pass' or tier.get('engine')!='spike'):
        return
    # Only compiler-authored products enter this registry. Merlin-only controls
    # and private operator references cannot acquire an entry via timing alone.
    own=result.get('compiler_source') or result.get('candidate_native_execution')
    if not own:
        return
    compiler=Path(compiler).resolve();root=Path(root).resolve()
    source=PERF.capsule_identity(capsule);candidate=hash_tree(compiler)['sha256']
    key=document_sha256([source,candidate,str(root)])
    registry=Path(os.environ[joint_control.ENV]).parent/'calibration_registry'
    directory=registry/key
    if (directory/'entry.json').exists():
        return
    directory.mkdir(parents=True,exist_ok=True)
    cap=C._write(directory/'capsule.json',{k:v for k,v in capsule.items() if k!='__dir__'})
    grade=C._write(directory/'result.json',result)
    performance=row.get('performance') or {}
    C._write(directory/'entry.json',dict(id=key,capsule=cap,result=grade,generated_root=str(root),
        compiler=str(compiler),compiler_sha256=candidate,source_group_sha256=member['source_group_sha256'],
        estimate_status=performance.get('status'),outside_calibrated_coverage=performance.get('timing_mode')=='untuned',
        estimated_interval=performance.get('estimated_interval')))


def step(*, profile_pin, state_path, context, output):
    """One bounded polling action; no gSIM until legal independent variants exist."""
    profile=PERF._read(profile_pin)
    selected=joint_control.selected()
    chosen = selected['policy'].get('native_calibration', {}) if selected else {}
    if any(chosen.get(key) != profile_pin.get(key) for key in ('path', 'sha256')):
        raise ValueError('calibration supervisor profile differs from the immutable campaign policy')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    registry=Path(state_path).parent/'calibration_registry'
    entries=sorted(registry.glob('*/entry.json'))[:2049]
    if len(entries)>2048:
        raise ValueError('completed calibration registry exceeds bounded review roster')
    # Completed rounds remain immutable; a failed held validation cannot become
    # training data in a retry with the same source group.
    used=set(); attempted=set()
    for completed in output.glob('round-*/outcome.json'):
        report=json.loads(completed.read_text())
        plan_pin=report.get('plan')
        if plan_pin:
            past=PERF._read(plan_pin)['cases']
            attempted.update(c['id'] for c in past)
            if report.get('status') in {'not_promoted','limited_calibrated_provisional'}:
                used.update(c['source_group_sha256'] for c in past if c['split']=='held')
    rows=[json.loads(p.read_text()) for p in entries]
    rows=[r for r in rows if r['source_group_sha256'] not in used and r['id'] not in attempted]
    # Model-only epochs retain the original policy while changing its separately
    # pinned evaluator selection. Never fit or promote against a stale incumbent.
    current_pin = selected.get('performance_selection') or profile['performance_selection']
    current = PERF._read(current_pin)
    initial = PERF._read(profile['performance_selection'])
    from .joint_runtime_revision import validate
    validate(initial, current)
    from .joint_calibration_readiness import panel
    readiness = panel(profile, current, rows,
        witnesses=selected['policy']['gsim']['witness_capsules'])
    temporary = output/'panel-readiness.next.json'
    temporary.write_text(json.dumps(readiness, indent=2)+'\n')
    temporary.replace(output/'panel-readiness.json')
    allowed = set(readiness['admissible_registry_ids'])
    rows = [row for row in rows if row['id'] in allowed]
    active = [row for row in readiness['members']
              if row['status'] not in {'not_in_current_source_selection', 'outside_physical_witness_roster'}]
    if active and all(row['status'] == 'blocked_structural_model' for row in active):
        return dict(status='blocked_structural_model_admission',
            reason='Repair selected model structure before collecting price-fit variants; no gSIM scheduled.',
            panel_readiness=str(output/'panel-readiness.json'),
            denial_reasons=sorted({row['reason'] for row in active}), registered_variants=len(rows))
    from .feedback.joint_timing_selection import choose
    by_runtime = {}
    for row in rows:
        chosen = choose(current, PERF.capsule_identity(PERF.selected_capsule(current, C._capsule(row['capsule']))))
        by_runtime.setdefault(document_sha256(chosen['runtime']), []).append(row)
    alternatives = [C.select_variant_cases(group) for group in by_runtime.values()]
    selection = next((v for v in alternatives if v['status'] == 'selected'),
                     C.select_variant_cases([]))
    if selection['status']!='selected':
        return dict(selection,registered_variants=len(rows),panel_readiness=str(output/'panel-readiness.json'),reason='need two passing compiler variants in each of four independent reviewed small source groups')
    key=document_sha256(selection['cases']);round_root=output/('round-'+key)
    round_root.mkdir(exist_ok=True)
    plan_path=round_root/'plan-input.json'
    if not plan_path.exists():
        C.freeze_round(selection_pin=current_pin,cases=selection['cases'],
            parameters=profile['parameters'],output=plan_path,question=profile['question'],stage=profile.get('stage','bootstrap'))
    plan=F.pin(plan_path,max_bytes=16<<20)
    if not profile.get('native_view'):
        return dict(status='blocked_native_runtime_view',round=str(round_root),reason='operator profile must pin the selected gSIM namespace inputs before physical measurement')
    from .joint_calibration_launcher import physical_runtime
    with physical_runtime(profile, context=context, directory=round_root) as backend:
        outcome=C.run_round(plan,context=context,backend=backend,output=round_root)
    value=PERF._read(outcome)
    if value.get('timing_epochs'):
        response=C.request_promotion(outcome,previous_run=Path(state_path).parent,
            review_id='automatic-reviewed-local-native-'+key,output=round_root/'next-epoch.json')
        return dict(status='model_epoch_requested',selection=outcome,epoch=response)
    return dict(status=value['status'],outcome=outcome)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('profile','profile-sha256','state','descriptor','repo','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--poll-seconds',type=int,default=30)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args(argv)
    if not 5<=args.poll_seconds<=60:
        parser.error('poll interval must be 5..60 seconds')
    os.environ[joint_control.ENV]=str(Path(args.state).resolve())
    from .context import load_context
    context=load_context(args.descriptor,repo=args.repo)
    pin=dict(path=str(Path(args.profile).resolve()),sha256=args.profile_sha256)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    while True:
        try:
            if joint_control.stop_reason():
                result=dict(status='native_drain_or_campaign_stop',reason=joint_control.stop_reason())
            else:
                result=step(profile_pin=pin,state_path=args.state,context=context,output=output)
        except (ValueError,OSError,RuntimeError,KeyError,TypeError) as error:
            result=dict(status='blocked',reason=str(error),exception=type(error).__name__)
        # Mutable service status is descriptive. Plans, measurements, models and
        # promotion evidence are separately immutable and digest-bound.
        temporary=output/'status.next.json';temporary.write_text(json.dumps(result,indent=2)+'\n')
        temporary.replace(output/'status.json')
        if args.once or result['status'] in {'model_epoch_requested','native_drain_or_campaign_stop'}:
            return 0
        time.sleep(args.poll_seconds)


if __name__=='__main__':
    raise SystemExit(main())
