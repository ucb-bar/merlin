"""Read retained F2 terminal logs/IR/sources only; no object or SDK reads."""
from pathlib import Path
import hashlib,json
REPO=Path.cwd(); B=REPO/'out/artifacts/verification/merlin-passes/loop-scope-20261028/NEXT-BUFFER'
F=B/'F/qualification-v2'; OUT=B/'CONTEXT/W/FINAL-REVIEW/F2-TERMINAL-OBSERVATION.json'
def read(p): return json.loads(p.read_text())
def exact(ref):
 p=Path(ref['path']);assert p.is_relative_to(REPO)
 assert p.suffix not in ('.olean','.ilean','.so','.a','.o','.pyc')
 data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==ref['sha256']
 assert len(data)==ref.get('bytes',ref.get('size'))
 return data
def ref(p):
 data=p.read_bytes();return dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
result=read(F/'RESULT.json');assert result['status']=='BOUNDED_BUFFER_ENGINEERING_BATCH_PASSED'
negative=read(F/'NEGATIVES.json');assert len(negative)==24 and all(not r['consumer_called'] for r in negative)
assert sum(r['status']=='HOST_REFUSAL' for r in negative)==23
native=[r for r in negative if r['status']=='ACTUAL_GATE_EXPECTED_OBLIGATION_REFUSAL']
assert len(native)==1 and native[0]['name']=='final_wrong_inner_cutoff'
binding=read(F/'EXACT-INPUTS.json');oracle=json.loads(exact(binding['authoritative_verifier']))
assert oracle['status']=='ok'
files=oracle['result']['files'];assert len(files)==116 and all(r['valid'] for r in files)
assert {r['parsed_bytes']['path']:r['parsed_bytes'] for r in files}=={r['path']:r for r in binding['parsed_bytes']}
for r in binding['parsed_bytes']:exact(r)
n=REPO/'out/build/loop-scope-20261028/qualification/buffer-final-v2/negatives/final_wrong_inner_cutoff'
p=n/'query-2-WholeMemory.process.json';process=read(p)
assert process['returncode']==1 and Path(process['command'][-1])==n/'WholeMemory.lean'
stdout=(n/'query-2-WholeMemory.stdout').read_text();stderr=(n/'query-2-WholeMemory.stderr').read_text()
assert stdout==str(n/'WholeMemory.lean')+':14:12: error: Tactic `decide` proved that the proposition\n  False\nis false\n' and not stderr
source=(n/'WholeMemory.lean').read_text();assert 'theorem valid : Model.Valid actual := by' in source and source.splitlines()[13].strip() == 'all_goals decide'
attacks=read(F/'artifact-controls/ARTIFACTS.json');assert len(attacks)==9
for row in attacks:
 assert row['consumer_called'] is False
 if 'before' in row:
  assert row['before']==row['restored'] and row['changed']['sha256']!=row['before']['sha256']
  # F alone owns native object binding; review other restored bytes directly.
  if Path(row['restored']['path']).suffix not in ('.olean','.ilean'):exact(row['restored'])
 else:assert row['temporary_member_removed']
r=read(F/'fresh-process-replay/RESULT.json');report=json.loads(exact(r['report']));root=Path(r['report']['path']).parent
assert r['status']=='checked' and r['incoming_ledger']==0 and r['issued_live_completions']==1
assert read(F/'REPLAY-PROCESS.json')==dict(returncode=0,timeout=False)
assert report['audit']['violations']==[] and set(report['audit']['axioms_used']) <= {'propext','Classical.choice','Quot.sound'}
expected=B/'Q/source-cut-v5/case-requests/retained_two_epochs'
assert set(report['statements'])==set(read(expected/'ROOTS.json'))
assert report['audit_modules']==[row['module'] for row in report['generated_queries']]
for i,row in enumerate(report['actual4']):assert exact(row)==(expected/f'stage{i}.mlir').read_bytes()
for row in report['generated_queries']:assert exact(row['source'])==(expected/(row['module']+'.lean')).read_bytes()
processes=[]
for row in report['process_evidence']:
 data=exact(row)
 if row['path'].endswith('.process.json'):
  proc=json.loads(data);assert proc['returncode']==0;processes.append(row)
assert len(processes)==125
assert read(root/'audit.stdout')==report['audit'] and read(root/'types.stdout')['statements']==report['statements']
out=dict(status='read_only_terminal_evidence_review_passed',result=ref(F/'RESULT.json'),
 compiler_verified_actual_buffers=116,positive_quadruples=5,valid_mutant_quadruples=24,
 host_refusals=23,actual_Lean_false_obligation_refusals=1,native_negative_source=ref(n/'WholeMemory.lean'),
 native_negative_process=ref(p),artifact_guard_controls=9,replay_report=r['report'],fresh_replay_processes=125,
 replay_roots=len(report['statements']),replay_theorems=report['audit']['theorems_audited'],std3=True,
 object_or_sdk_contents_read=False,new_native_launch=False,independent_human_approval=False,
 limitations=['Q5/F2 binds186 sources; separate191-source successor fixes five missing imported source fingerprints',
 '23 host refusals are not Lean counterexample proofs','Partial possible-success public relation, not errors/divergence/native allocator conformance',
 'Compiler parser uses allow_unregistered_dialects=True as recorded; no new compiler invocation in this review'])
with OUT.open('x') as stream:json.dump(out,stream,indent=2);stream.write('\n')
print(json.dumps(out,indent=2))
