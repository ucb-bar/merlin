"""Bind retained initialized/fault witnesses to the current checked query."""
import json,os,sys
from pathlib import Path
from merlin.common.paths import repo_root
from merlin.verify.loop_transform_intake import file_ref
root=repo_root();sys.path.insert(0,str(root/'merlin/tests/ir'))
from loop_transform_qualification import run,save
f=root/'out/artifacts/verification/merlin-passes/loop-scope-20261028/F-QUALIFICATION';out=f/'witness-v2'
row=next(x for x in json.loads((f/'qualification-v2/POSITIVES.json').read_text()) if x['name']=='witness_example')
assert file_ref(Path(row['report']['path']))==row['report']
report=json.loads(Path(row['report']['path']).read_text())
for ref in report['built_artifacts']:
    assert file_ref(Path(ref['path']))==ref
old=f/'witness-v1/Witness.lean';text=old.read_text();assert text.startswith('import Query\n')
source=out/'CurrentWitness.lean';source.write_text('import LoopTransformQuery\n'+text[len('import Query\n'):])
obj=root/'out/build/loop-scope-20261028/qualification/witness-v2/lib/CurrentWitness.olean';obj.parent.mkdir(parents=True)
lean=Path(report['lean']['path']);env={**os.environ,'LEAN_PATH':str(obj.parent)+os.pathsep+str(Path(report['object']['path']).parent),'PATH':str(lean.parent)+os.pathsep+os.environ['PATH']}
roots=json.loads((f/'witness-v1/FINAL_RESULT.json').read_text())['roots']
save(out/'INTENT.json',{'original_witness':file_ref(old),'current_witness':file_ref(source),'current_query_report':row['report'],'only_source_delta':'import Query → import LoopTransformQuery','roots':roots})
proc,_=run([lean,'-j1','-o',obj,source],out/'compile',environment=env)
assert proc['returncode']==0,proc
outputs={}
for name,args in [('audit',['CurrentWitness']),('types',['--statements',','.join(roots),'CurrentWitness'])]:
    proc,stdout=run([lean,'--run',report['auditor']['path'],*args],out/name,environment=env)
    assert proc['returncode']==0,proc
    outputs[name]=json.loads(stdout.strip().splitlines()[-1])
assert outputs['audit']['violations']==[]
assert set(outputs['audit']['axioms_used'])<={'propext','Classical.choice','Quot.sound'}
assert set(outputs['types'])==set(roots)
save(out/'RESULT.json',{'status':'CURRENT_QUERY_MODEL_WITNESSES_CHECKED','source':file_ref(source),'object':file_ref(obj),'current_query_report':row['report'],'roots':roots,'audit':outputs['audit'],'statements':outputs['types'],'success':'inputs[3,5],world7 → [17],world20','failure':'IV9 external17,world15; later calls/observer suppressed','human_approved':False,'native_execution':False})
print(json.dumps({'status':'CURRENT_QUERY_MODEL_WITNESSES_CHECKED','roots':len(roots)}))
