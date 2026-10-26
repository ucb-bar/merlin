"""Stdlib-only validation of populated join refs/counts/statuses; no SDK traversal."""
import ast,collections,csv,hashlib,io,json,re,sys
from pathlib import Path
import xml.etree.ElementTree as ET
OUT=Path(__file__).resolve().parent
inputs=json.loads((OUT/'INPUTS.json').read_text())['inputs'];closed=json.loads((OUT/'LEDGER_INPUTS.json').read_text())['inputs'];cache={};errors=[];checks=0

def check(ok,msg):
 global checks
 checks+=1
 if not ok:errors.append(msg)
for label,records in [('selected',inputs),('ledger',closed)]:
 for key,r in records.items():
  p=OUT/r['path'];check(p.resolve().is_relative_to(OUT.resolve()),label+': path escapes');check(p.suffix in {'.py','.md','.json','.lean','.xml','.csv','.txt'},label+': forbidden selected type')
  if not p.is_file():check(False,label+': missing '+key);continue
  b=p.read_bytes();check((hashlib.sha256(b).hexdigest(),len(b))==(r['sha256'],r['bytes']),label+': changed '+key)
  cache.setdefault(key,b)
def ptr(v,p):
 for s in p.strip('/').split('/') if p else []:
  s=s.replace('~1','/').replace('~0','~');v=v[int(s)] if isinstance(v,list) else v[s]
 return v
def ref(r):
 key=r['input_id'];check(key in cache,'Unknown input '+key)
 if key not in cache:return None
 text=cache[key].decode()
 try:
  if 'json_pointer' in r:return ptr(json.loads(text),r['json_pointer'])
  if 'lines' in r:
   a,b=r['lines'];lines=text.splitlines();check(1<=a<=b<=len(lines),'bad source lines '+str(r));return '\n'.join(lines[a-1:b])
  return text
 except Exception as e:check(False,'selector failed '+str(r)+': '+str(e));return None
def visit(v):
 if isinstance(v,dict):
  if 'input_id' in v:ref(v)
  if 'value' in v:
   citations=v.get('refs',v.get('evidence',[]));check(bool(citations),'uncited fact')
   if v['value'] is None:check(bool(v.get('reason')),'unknown lacks reason')
   if 'refs' in v:
    for r in citations:
     if 'json_pointer' in r:check(v['value']==ref(r),'value differs from cited selected field '+str(r))
   if 'check' in v:
    c=v['check'];raw=cache[c['input_id']];op=c['operation']
    if op=='junit_tests':actual=sum(int(e.attrib.get('tests',0)) for e in ET.fromstring(raw).iter('testsuite'))
    else:
     actual=ptr(json.loads(raw),c['json_pointer'])
     if op=='length':actual=len(actual)
     elif op=='zero_if_false':actual=0 if actual is False else None
     elif op=='sum_accepted':actual=sum(r['accepted_complete_record'] for r in actual)
     else:check(op=='value','unknown metric operation')
    check(v['value']==actual,'ledger count disagreement')
  for value in v.values():visit(value)
 elif isinstance(v,list):
  for x in v:visit(x)
x=json.loads((OUT/'REGISTERED_COVERAGE.json').read_text());visit(x)
assignment=next(n for n in ast.parse(cache['registry']).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TRANSFORM_SCOPES' for t in n.targets));registry=ast.literal_eval(assignment.value)
check(x['fixed_kind_count']==8==len(x['rows'])==len(registry),'fixed8 count mismatch');check(set(r['kind'] for r in x['rows'])==set(registry),'registry membership mismatch')
actual_dispatch=set()
for n in ast.walk(ast.parse(cache['pipeline'])):
 if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'transform','run'} and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value in registry:actual_dispatch.add(n.args[0].value)
check(actual_dispatch==set(registry),'actual pipeline dispatch mismatch')
ledger=json.loads(cache['ledger']);graham=list(csv.DictReader(io.StringIO(cache['graham_coverage'].decode())));categories=collections.Counter(r['inventory_category'] for r in graham)
for row in x['rows']:
 kind=row['kind'];check(row['registry_scope_tags']['value']==list(registry[kind]),kind+': scope tags changed');check(row['scope_tags_are_variants'] is False,kind+': scope tags conflated with variants')
 expected=[r['row_id'] for r in ledger['rows'] if kind in r['registered_kinds']['value']];check([r['row_id'] for r in row['selected_claim_families']]==expected,kind+': ledger join mismatch')
 variants=[r for r in graham if r['inventory_category']=='registered_common_transformation' and r['kind']==kind];check(row['graham_variant_count']==len(variants),kind+': variant count mismatch')
 for r in row['graham_variant_rows']:
  original=graham[r['csv_record_index']];check(all(r[k]==original[k] for k in ['row_id','variant','producer_domain','graph_domain','qualification_status','gap_ids']),kind+': CSV field mismatch')
 check(row['new_gate_status'] in {'held_unqualified','unsupported_kind_source_only'},kind+': invalid new status');check(row['native_model_policy_human_approval'] is False,kind+': approval conflation')
 for family in row['selected_claim_families']:
  check(family['status']['value'] in {'historical_engineering_qualified','held_unqualified','uncompiled_unqualified'},kind+': unknown family status')
  if family['row_id'].startswith('new-'):check(family['status']['value']!='historical_engineering_qualified','new source promoted')
d=x['denominator_reconciliation'];check(d['registered_variant_rows']==34==categories['registered_common_transformation'],'34 denominator mismatch');check(d['broader_mechanism_rows']==57==sum(v for k,v in categories.items() if k!='registered_common_transformation'),'57 denominator mismatch');check(d['inventory_rows']==91==len(graham),'91 row count mismatch');check(d['category_counts']==dict(categories),'category counts mismatch');check(sum(r['graham_variant_count'] for r in x['rows'])==34,'variant partitions do not reconcile')
check(len(d['broader_rows'])==57,'missing broader rows')
for r in d['broader_rows']:
 original=graham[r['csv_record_index']];check(all(r[k]==original[k] for k in ['row_id','kind','inventory_category','claim_boundary']),'broader field mismatch')
check(all(v is False for v in x['approval_flags'].values()),'approval flags changed')
service_result=OUT/'service/RESULT.json'
if service_result.exists():
 sr=json.loads(service_result.read_text());sm=json.loads((OUT/'service/selected-packet/MANIFEST.json').read_text());sc=json.loads((OUT/'service/SELECTED_COMPARISON.json').read_text());observed=json.loads((OUT/'service/OBSERVED_SOURCE.json').read_text());raw_results=json.loads((OUT/'service/RAW_RESULTS.json').read_text())['results'];spec=json.loads((OUT/'SERVICE_CASES.json').read_text())
 check(sr['status']=='PASS' and sr['fixed_cases']==12 and sr['passed']==12,'service fixed12 status/count mismatch')
 check(len(raw_results)==12 and {r['id'] for r in raw_results}=={r['id'] for r in spec['cases']},'service case membership mismatch')
 check(sr['kernel_context_and_bundle_profile_custody']=='EXPLICITLY_MOCKED','service mocks not explicit')
 check(sr['native_invocations']==0 and sr['SDK_runtime_reads']==0 and not sr['blocked_events'],'service native/SDK boundary failure')
 check(sr['qualification'] is False and sr['native_model_human_approval'] is False,'service qualification promoted')
 check(sc['status']=='MATCH' and len(sc['comparisons'])==8 and all(r['equal'] for r in sc['comparisons']),'observed/selected service source mismatch')
 check(hashlib.sha256((OUT/'service/selected-packet/MANIFEST.json').read_bytes()).hexdigest()==sr['selected_manifest_sha256'],'service manifest association mismatch')
 for r in sm['rows']:
  p=OUT/'service/selected-packet'/r['path'];b=p.read_bytes();check((hashlib.sha256(b).hexdigest(),len(b))==(r['sha256'],r['bytes']),'selected service member changed '+r['path'])
 for r in observed['sources']:
  p=OUT/r['snapshot'];b=p.read_bytes();check((hashlib.sha256(b).hexdigest(),len(b))==(r['sha256'],r['bytes']),'observed service source changed')
 for r in raw_results:
  stored=json.loads((OUT/'service/cases'/r['id']/'RESULT.json').read_text());check(stored==r and r['passed'] is True,'service raw result mismatch '+r['id']);check(r['native_Lean_result'] is None,'service native result invented')
 check(x['service_harness_status']=='SDK_FREE_FIXED12_PASS_UNQUALIFIED','service join status not updated')
for name in ['APPENDIX_TABLE.md','OBLIGATIONS.md','service/EVIDENCE_TABLE.md']:
 p=OUT/name
 if p.exists():
  for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):check((p.parent/target.split('#')[0]).is_file(),'missing generated link '+target)
result={'schema':'merlin.registered-coverage.validation.v1','status':'PASS' if not errors else 'FAIL','checks':checks,'errors':errors,'fixed_registry_keys':8,'Graham_variant_rows':34,'Graham_broader_rows':57,'new_native_or_semantic_tests':0,'service_API_cases':12 if service_result.exists() else 0,'scope':'AST/data/ref/count/status consistency; no new qualification, native or human approval.'}
p=OUT/'VALIDATION.json'
if p.exists() and json.loads(p.read_text()).get('status')=='FAIL':
 d=OUT/'validation-attempts';d.mkdir(exist_ok=True);(d/(str(len(list(d.glob('*.json')))+1)+'.json')).write_bytes(p.read_bytes())
p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));sys.exit(bool(errors))
