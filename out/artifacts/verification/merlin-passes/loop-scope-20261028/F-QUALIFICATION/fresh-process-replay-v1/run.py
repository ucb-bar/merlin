"""One fresh-process receipt reconstruction; no saved completion authority."""
import json
import os
from pathlib import Path
from merlin.common.paths import repo_root
from merlin.verify import loop_transform_gate as gate
from merlin.verify.loop_transform_intake import file_ref
root=repo_root()
f=root/'out/artifacts/verification/merlin-passes/loop-scope-20261028/F-QUALIFICATION'
cut=json.loads((f/'qualification-v2/SOURCE-CUT.json').read_text())
refs=[*cut['F_sources'],cut['Q_cut'],cut['lean'],cut['compiler_python']]
refs.extend(json.loads(Path(cut['Q_cut']['path']).read_text())['sources'])
for ref in refs:
    p=Path(ref['path']);p=p if p.is_absolute() else root/p
    actual=file_ref(p)
    assert actual['sha256']==ref['sha256'] and actual['bytes']==ref.get('bytes',ref.get('size'))
old=next(x['report'] for x in json.loads((f/'qualification-v2/POSITIVES.json').read_text()) if x['name']=='factor3_carry2_nonaligned')
assert file_ref(Path(old['path']))==old
out=f/'fresh-process-replay-v1'
assert len(gate._LIVE)==0
(out/'INTENT.json').write_text(json.dumps({'pid':os.getpid(),'original':old,'initial_live_ledger_size':0,'command':'gate.replay(original REPORT, pinned4499, pinnedLean, NEW fresh-process-replay-v1 objects)','sources':refs},indent=2)+'\n')
token=gate.replay(old['path'],reference=Path('/home/jeremy/Projects/merlin-graph-proof-intake'),lean=Path(cut['lean']['path']),destination=root/'out/build/loop-scope-20261028/qualification/fresh-process-replay-v1')
assert len(gate._LIVE)==1
report=json.loads(Path(token.report).read_text())
(out/'RESULT.json').write_text(json.dumps({'status':'FRESH_PROCESS_REPLAY_CHECKED','pid':os.getpid(),'initial_live_ledger_size':0,'new_live_ledger_size':1,'original':old,'fresh_report':file_ref(Path(token.report)),'roots':len(report['statements']),'audit':report['audit'],'old_receipt_was_not_authority':True,'human_approved':False,'native_fidelity':False},indent=2)+'\n')
print(json.dumps({'status':'FRESH_PROCESS_REPLAY_CHECKED','roots':len(report['statements'])}))
