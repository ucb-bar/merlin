"""Readable live status and evidence projections; never qualifies or promotes a compiler."""
from __future__ import annotations
import argparse
import collections
import datetime as dt
import json
from pathlib import Path
import time

import common as C
C.configure()
from merlin_experiments.phase2.roofline_priority import roofline_priorities


def report():
    manifest=json.loads((C.EVIDENCE/'capsule-manifest.json').read_text())
    stress=json.loads((C.EVIDENCE/'stress-manifest.json').read_text())
    members={x['identity']:x for x in manifest['members']+stress['members']}
    measurements=[]
    for p in (C.TASK/'out/phase2-campaign/measurements.jsonl',C.OUT/'measurements.jsonl'):
        if p.exists():
            for line in p.read_text().splitlines():
                measurements.append(json.loads(line))
    comparable=collections.defaultdict(lambda:collections.defaultdict(list))
    numerics=collections.Counter()
    for row in measurements:
        numerics[(row['engine'],'pass' if row['exact'] else 'fail')]+=1
        if row['engine']=='gsim':
            comparable[(row['member'],row['seed'],row.get('evaluator_profile','legacy_serial_b64_v1'))][(row['arm'],row['build']['context']['compiler_sha256'])].append(row)
    paired=[]
    for (member,seed,profile),arms in sorted(comparable.items()):
        base=[v for (arm,_),values in arms.items() if arm=='baseline' for v in values]
        for (arm,source),values in arms.items():
            if arm=='baseline' or not base:
                continue
            # Simulator, exact source, input tensors and window remain fixed. Cache
            # warmth remains UNKNOWN; each record is a fresh process invocation.
            good=all(x['exact'] for x in base+values)
            baseline=[x['cycles'] for x in base]
            candidate=[x['cycles'] for x in values]
            floor=members[member].get('roofline',{})
            floor_cycles=floor.get('floor_cycles') if floor.get('status') in {'derived','partial'} else None
            row={'member':member,'seed':seed,'evaluator_profile':profile,'candidate':arm,'candidate_sha256':source,
                 'baseline_cycles':baseline,'candidate_cycles':candidate,
                 'baseline_range':[min(baseline),max(baseline)],'candidate_range':[min(candidate),max(candidate)],
                 'correct':good,'improvement_worst_pair_fraction':1-max(candidate)/min(baseline),
                 'repeat_band_separated':max(candidate)<min(baseline),
                 'floor_cycles':floor_cycles,'floor_status':floor.get('status','UNKNOWN'),
                 'baseline_attainment':floor_cycles/max(baseline) if floor_cycles else None,
                 'candidate_attainment':floor_cycles/max(candidate) if floor_cycles else None,
                 'baseline_gap_cycles':max(baseline)-floor_cycles if floor_cycles else None,
                 'candidate_gap_cycles':max(candidate)-floor_cycles if floor_cycles else None,
                 'floor_limiting_resource':floor.get('limiter'),
                 'floor_unresolved':floor.get('unresolved'),
                 'qualification':'exploratory subset; no formal promotion or whole-model speedup'}
            paired.append(row)
    priorities={}
    for arm,profile in sorted({(x['candidate'],x['evaluator_profile']) for x in paired}):
        selected={}
        for row in paired:
            if (row['candidate'],row['evaluator_profile'])!=(arm,profile) or not row['correct']:
                continue
            floor=members[row['member']].get('roofline',{})
            selected[row['member']]=[{'cycles':c,'staged_roofline':floor} for c in row['candidate_cycles']]
        if selected:
            key=arm+'/'+profile
            priorities[key]=roofline_priorities(selected,attainment_min=0.8)
            priorities[key]['convergence_authority']='NONE: partial floors and unmeasured required roster'
    failures=[]
    for p in (C.TASK/'out/phase2-campaign/failures.jsonl',C.OUT/'failures.jsonl'):
        if p.exists():
            failures.extend(json.loads(x) for x in p.read_text().splitlines())
    now=dt.datetime.now(dt.timezone.utc)
    stop_path=C.EVIDENCE/'manual-stop.json'
    stop=json.loads(stop_path.read_text()) if stop_path.exists() else None
    launch_path=C.EVIDENCE/'automatic-launch-preflight.json'
    launch=json.loads(launch_path.read_text()) if launch_path.exists() else None
    value={'schema':'phase2.exploratory.results.v1','at':now.isoformat(),
           'elapsed_seconds':(now-C.START).total_seconds(),'remaining_seconds':(C.DEADLINE-now).total_seconds(),
           'timing_model_qualified':False,'formal_admission':False,
           'frozen_objective_sha256':C.digest([x['declaration'] for x in manifest['members'] if x.get('declaration')]),
           'measurement_counts':{str(k):v for k,v in numerics.items()},'paired':paired,
           'roofline_priorities':priorities,'failures':failures,
           'accepted_experiments':[], 'rejected_experiments':[],
           'incumbent':'immutable baseline; formal promotion unavailable',
           'next_priority':'lowest worst-repeat attainment among comparable measured members; missing floors remain UNKNOWN',
           'stop_state':'manual_search_stopped_by_user_workflow_correction' if stop else 'running',
           'manual_stop':stop,'automatic_launch':launch,
           'automatic_phase2_launched':False,
           'plateau':'not established: no complete admitted objective batches',
           'dependencies':C.initialize()['dependencies']}
    C.save(C.EVIDENCE/'results.json',value)
    print(f"elapsed={value['elapsed_seconds']/60:.1f}m remaining={value['remaining_seconds']/60:.1f}m measurements={len(measurements)} failures={len(failures)}")
    print('Formal admission: incomplete; Spike timing: UNKNOWN; cache warmth: UNKNOWN')
    for row in paired:
        print(f"{row['candidate']:20} {row['evaluator_profile']:30} {row['member']:50} {row['baseline_range']} -> {row['candidate_range']} gain={100*row['improvement_worst_pair_fraction']:.2f}% exact={row['correct']}")
    return value


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--watch',action='store_true')
    args=ap.parse_args()
    while True:
        report()
        if not args.watch or (C.EVIDENCE/'manual-stop.json').exists():
            break
        time.sleep(30)


if __name__=='__main__':
    main()
