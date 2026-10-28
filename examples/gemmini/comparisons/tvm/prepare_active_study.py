"""Extract a fresh ResNet-only or 16-kernel ReplayTrace owner study; no execution.

Only exact structural equality shares records. The initial allocation defaults
to 20 of 25 trials, reserving five for existing graph alternatives. Changing a
plan requires fresh extraction, with the same per-case/global budget ledger.
"""
import argparse
import json
import os
from pathlib import Path

from build_shared_chunk_elf import digest, selected_chunk
from kernel_benchmark_data import CASES


def inputs(root, kind):
    if kind=="resnet":
        catalog=json.loads((root/"catalog.json").read_text())
        rows=[r for r in catalog['chunks'] if r['model']=='resnet']
        if len(rows)!=18:raise ValueError('Require the frozen 18-chunk ResNet roster')
        for row in sorted(rows,key=lambda r:r['id']):
            _,paths=selected_chunk(root,row['id'])
            yield row['id'],paths['program'],paths['fixture']
    else:
        for case in CASES:
            folder=root/case; proof=json.loads((folder/'reference.json').read_text())
            if proof['case']!=case or digest(folder/'program.json')!=proof['program']['sha256'] or digest(folder/'fixture.npz')!=proof['fixture_sha256']:
                raise ValueError('Kernel fixture identity changed: '+case)
            for source in proof['reference_sources']:
                if digest(source['path'])!=source['sha256']:raise ValueError('Kernel reference source changed')
            yield case,folder/'program.json',folder/'fixture.npz'


def prepare(root, output, kind, trials):
    import tvm
    from program_io import import_program
    from tune_models import extract_inventory
    from tvm.relax.backend.contrib.gemmini_search import WORKLOAD_ATTR
    from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities
    from tune_firesim_chunks import measurement_sources
    import measure
    root=root.resolve(strict=True);output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    owners,representatives,rows,coverage={},{},[],[]
    for name,program,fixture in inputs(root,kind):
        prepared,tasks,inventory=extract_inventory(import_program(program),GemminiCapabilities(),fusion_level=2)
        kinds={row['task']:row['kind'] for row in inventory}
        owned=[];keys=[]
        for task in tasks:
            if kinds[task.task_name]!='contraction':continue
            mod=task.dispatched[0];key=str(mod['main'].attrs[WORKLOAD_ATTR]);keys.append(key)
            if key in owners:
                if not tvm.ir.structural_equal(representatives[key],mod):raise ValueError('Structural key collision')
            else:
                owners[key]=name;representatives[key]=mod;owned.append(key)
        if not keys:raise ValueError('No contraction tasks for selected workload: '+name)
        coverage.append(dict(id=name,keys=keys,owners=[owners[k] for k in keys],tasks=inventory,
                             program_sha256=digest(program),fixture_sha256=digest(fixture)))
        if owned:
            rows.append(dict(id=name,program=os.path.relpath(program,output),fixture=os.path.relpath(fixture,output),
                             model=kind,status='ready_for_functional_build',phase='initial',strategy='replay',
                             trials=trials,batch=1,workload_keys=owned,graph_fusion_level=2,graph_cse=False,
                             prepack_weights=False,bias_preload=False))
        print(name,len(keys),'contractions;',len(owned),'new owner keys',flush=True)
    study=dict(schema='gemmini_active_replay_study_v1',campaign_kind=kind,campaigns=rows,
               global_proposal_cap=25*len(rows),initial_trials_per_owner=trials,
               reserved_per_owner=25-trials,full_tinyllama_smolvla_excluded=True,
               slots_budget_seconds=14400 if kind=='resnet' else 86400,
               execution_admission=False,qualification='Fresh extraction only; remote proof/guest/runner checks required',
               coverage=coverage,compiler_sources=measurement_sources(tvm,measure),extractor_sha256=digest(__file__))
    (output/'study.json').write_text(json.dumps(study,indent=2)+'\n')
    print(json.dumps(dict(study=str(output/'study.json'),owners=len(rows),distinct_tasks=len(owners))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('resnet','kernel'),required=True)
    for key in ('input-root','output-dir'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--initial-trials',type=int,default=20)
    args=parser.parse_args()
    if not 1<=args.initial_trials<=25:parser.error('Use 1..25 initial trials, within the shared total of 25')
    prepare(args.input_root,args.output_dir,args.kind,args.initial_trials)


if __name__=='__main__':main()
