"""Offline bind of successful native outputs; original wrapper failure retained."""
import json
from pathlib import Path
from merlin.verify.loop_transform_intake import file_ref
out=Path(__file__).resolve().parent
intent=json.loads((out/'INTENT.json').read_text())
assert file_ref(Path(intent['current_witness']['path']))==intent['current_witness']
assert file_ref(Path(intent['current_query_report']['path']))==intent['current_query_report']
for phase in ['compile','audit','types']:
    assert json.loads((out/phase/'PROCESS.json').read_text())=={'returncode':0,'timeout':False}
audit=json.loads((out/'audit/stdout.txt').read_text().strip().splitlines()[-1])
statements=json.loads((out/'types/stdout.txt').read_text().strip().splitlines()[-1])['statements']
assert set(statements)==set(intent['roots']) and all(type(v)is str and v for v in statements.values())
assert audit['violations']==[] and set(audit['axioms_used'])<={'propext','Classical.choice','Quot.sound'}
command=json.loads((out/'compile/INTENT.json').read_text())['command'];obj=Path(command[command.index('-o')+1])
result={'status':'CURRENT_QUERY_MODEL_WITNESSES_CHECKED_OFFLINE_BINDING','native_commands_repeated':False,'original_wrapper_exit':1,'source':intent['current_witness'],'object':file_ref(obj),'current_query_report':intent['current_query_report'],'roots':intent['roots'],'audit':audit,'statements':statements,'success':'inputs[3,5],world7 → [17],world20','failure':'IV9 external17,world15; later calls/observer suppressed','human_approved':False,'native_execution':False}
(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'roots':len(statements),'theorems':audit['theorems_audited'],'offline_binding':'passed'}))
