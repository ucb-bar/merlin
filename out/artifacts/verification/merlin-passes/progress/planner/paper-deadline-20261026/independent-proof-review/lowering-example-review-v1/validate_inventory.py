"""One bounded owned DATA inventory; no project imports, tools, or probes."""
import ast
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
checks = []
def check(label, condition):
    checks.append({'check': label, 'ok': bool(condition)})
def load(name):
    return json.loads((P / name).read_text())
def sha(data):
    return hashlib.sha256(data).hexdigest()
def pin(row, key='snapshot'):
    f=P/row[key]
    check('relative owned pin '+row[key], not Path(row[key]).is_absolute() and '..' not in Path(row[key]).parts and not f.is_symlink())
    b=f.read_bytes()
    check('hash '+row[key],sha(b)==row['sha256'])
    check('size '+row[key],len(b)==row['bytes'])
rows=load('SELECTION.json')['files']+load('DEPENDENCIES.json')
check('18 selected sources',len(rows)==18 and len({r['snapshot'] for r in rows})==18)
for r in rows:pin(r)
examples=load('EXAMPLE_INPUTS.json')
check('15 retained texts',len(examples['files'])==15)
for r in examples['files']:pin(r,'path')
check('exact saved OOT diagnostic pin',sha((P/'examples/oot-original16.mlir').read_bytes())==examples['declared_oot_target_sha256']=='e772ca1071a5f4558088f66e04c4427319841cd31d8a6adea41b674282a096a5')
check('small source textual reshape',(P/'examples/source-small2.mlir').read_text()==(P/'examples/source-original16.mlir').read_text().replace('16x16','2x2'))
check('11 expectations only',len(examples['mutations'])==11 and all(r['observed_status'] is None for r in examples['mutations']))
for r in examples['mutations']:
    base=(P/('examples/source-small2.mlir' if r['file'].startswith('source-') else 'examples/oot-small2.mlir')).read_text()
    for before,after in r['changes']:base=base.replace(before,after)
    check('exact mutation '+r['file'],base==(P/'examples'/r['file']).read_text())
index=load('SOURCE_INDEX.json')
for r in index:
    t=ast.parse((P/r['file']).read_text())
    check('source AST ref '+r['file']+':'+str(r['line']),any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n.name==r['name'] and n.lineno==r['line'] for n in ast.walk(t)))
state=load('REVIEW_STATE.json')
for k in ['verified','qualified','solver_ran','parser_ran','native_fidelity','model_coverage','lean_evidence_applicable']:check('false '+k,state[k] is False)
for k in ['solver_status','query_identity','translator_identity','solver_identity','retained_exact_receipt','in_tree_interface_bytes','command_buffer_bytes']:check('unobserved '+k,state[k] is None)
search=load('EVIDENCE_SEARCH.json')
check('bounded no matching receipt',search['matching_receipts']==[] and search['global_absence_claimed'] is False and search['tests_executed'] is False)
check('source findings not observed solver failures',len(load('FINDINGS.json'))==5 and all(r['observed_solver_or_native_failure'] is False for r in load('FINDINGS.json')))
# Reconstruct the selected digest formula as DATA, never import receipts.
receipt=ast.parse((P/'source/src/merlin/verify/receipts.py').read_text())
names=next(ast.literal_eval(n.value) for n in receipt.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='_SEMANTIC_FILES' for x in n.targets))
data=b''.join(n.encode()+b'\0'+(P/'source/src/merlin/verify'/n).read_bytes()+b'\0' for n in names)
data+=b'targetgen/contract/interface_emit.py\0'+(P/'source/src/merlin/targetgen/contract/interface_emit.py').read_bytes()+b'\0'
closure=load('DECLARED_VERIFIER_CLOSURE.json')
check('eight file digest formula',len(names)==8 and sha(data)==closure['selected_snapshot_digest_formula_sha256'])
check('recognizer not in digest file list','linalg_patterns.py' not in names and closure['recognizer_bound_by_formula'] is False)
watch=load('WATCHDOG_SELECTION.json')
for r in watch['files']:pin(r)
check('fixed watchdog manifest',sha((P/'watchdog-fixed/MANIFEST.json').read_bytes())==watch['manifest_sha256'])
for row in load('watchdog-fixed/MANIFEST.json')['files']:
    pin(dict(row['copy'],snapshot='watchdog-fixed/'+Path(row['copy']['path']).name))
parent=load('watchdog-fixed/WATCHDOG-REVIEW-REQUEST.json'); child=load('watchdog-fixed/REVIEW-REQUEST.json')
for request in [parent,child]:
    for r in request['source_refs']:pin(dict(r,snapshot='watchdog-fixed/'+Path(r['path']).name))
check('closed parent10 child7 sources',len(parent['source_refs'])==10 and len(child['source_refs'])==7)
check('child refs parent subset',{(Path(r['path']).name,r['sha256']) for r in child['source_refs']}<={(Path(r['path']).name,r['sha256']) for r in parent['source_refs']})
check('bound child authorization',sha((P/'watchdog-fixed/REVIEW-REQUEST.json').read_bytes())==parent['child_authorization']['sha256'])
check('exact parent budget',tuple(parent[k] for k in ['seconds','terminate_seconds','reap_seconds'])==(1800,2,3))
check('no parent authority',parent['attempts']==1 and parent['status']=='PROPOSED_NOT_AUTHORIZED' and all(parent[k] is False for k in ['native_authorized','install_authorized','old_holds_cleared']))
r=load('WATCHDOG_RESULTS.json')
check('retained 7 controls passed',r['status']=='PASS' and r['cases']==7 and all(x['ok'] for x in r['results']))
check('four actual own children reaped',r['actual_owned_python_children']==4 and r['all_actual_children_reaped'] is True)
check('no foreign signals or forbidden action',r['foreign_signal_attempts']==r['forbidden_actions']==[] and r['real_network_or_sdk_or_native_compiler'] is False)
check('unreaped branch marked mocked','MOCKED' in r['results'][-1]['evidence']['label'])
result={'status':'PASS' if all(x['ok'] for x in checks) else 'FAIL','checks':len(checks),'failed':[x for x in checks if not x['ok']],'scope':'OWNED SOURCE/DATA INVENTORY ONLY; no parser/solver/project imports/probes executed; source counts not theorem count','details':checks}
(P/'INVENTORY_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='details'}))
raise SystemExit(0 if result['status']=='PASS' else 1)
