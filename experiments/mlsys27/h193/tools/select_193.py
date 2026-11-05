#!/usr/bin/env python3
"""select_193.py : materialise the CANONICAL case of all 193 core-ATen overloads (Merlin's corpus_193 captures: f32 linalg-on-tensors + inputs + PyTorch reference) into canon193/<n>/, and write canon193.json.
VERIFICATION INSTRUMENT: lives only under backend-verify-author/ (never in an author-visible input).  Re-uses cases/ copies where already fetched; otherwise ssh-tars from the capture dirs (read-only)."""
import csv,json,os,subprocess,sys,shutil
H=os.environ.get('H193_H', os.environ.get('H', '/scratch2/loren/backend-verify-author')); SSH=['ssh','-F','/scratch2/loren/aws/ssh/config','-o','BatchMode=yes']
sys.path.insert(0,'/scratch2/loren/aws/tooling/exercised'); import exercised as E
R=list(csv.DictReader(open('/scratch2/loren/backend-verify/results-all.csv')))
canon={r['overload']:r for r in R if r['set']=='corpus_193'}
ix={}
for f,host in (('idx-c1.jsonl','merlin-compute-1'),('idx-c2.jsonl','merlin-compute-2')):
    for l in open(f'{H}/{f}'):
        j=json.loads(l); j['host']=host; ix.setdefault(j['case_id'],j)
old={x['overload']:x for x in json.load(open(H+'/cases.json')) if x['case_kind']=='canonical'}
forms={f['overload']:f for f in json.load(open(H+'/forms93/forms.json'))}
files=['capsule.linalg.mlir','inputs.json','golden.json','request.json','capture-work/meta.json']
sel=[]
for i,(o,r) in enumerate(sorted(canon.items())):
    d=f"{H}/canon193/{i:03d}"; os.makedirs(d,exist_ok=True)
    if not os.path.exists(d+'/golden.json'):
        if o in old and os.path.exists(f"{H}/cases/{old[o]['n']}/golden.json"):
            shutil.copytree(f"{H}/cases/{old[o]['n']}",d,dirs_exist_ok=True)
        else:
            k='corpus_193:'+r['case_id']; x=ix[k]
            p=subprocess.run(SSH+[x['host'],f"cd {x['dir']} && tar cf - "+' '.join(files)],capture_output=True)
            subprocess.run(['tar','xf','-','-C',d],input=p.stdout)
    fm=forms.get(o)
    sel.append(dict(n=f"{i:03d}",overload=o,klass='DEVICE57' if o in E.ALL57 else 'HOST136',ref_placement=r['placement'],ref_policy=r['numeric_execution_policy'],ref_dev_instr=r['executed_device_instructions'],ref_status=r['pass_fail'],
        canon_dir=f"canon193/{i:03d}",form_slug=(fm['slug'] if fm and o in E.ALL57 else ''),form_lane=(fm['lane'] if fm and o in E.ALL57 else ''),form_testable=bool(fm and o in E.ALL57 and fm.get('testable')),
        form_note=(fm.get('form','') or fm.get('note','')) if fm and o in E.ALL57 else ''))
json.dump(sel,open(H+'/canon193.json','w'),indent=1)
bad=[s['overload'] for s in sel if not os.path.exists(f"{H}/{s['canon_dir']}/golden.json")]
print(len(sel),'missing',bad)
