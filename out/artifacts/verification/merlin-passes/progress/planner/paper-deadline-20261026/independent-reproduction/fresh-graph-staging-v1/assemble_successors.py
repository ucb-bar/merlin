"""Fixed successor source/query/metadata snapshots, never provider/source execution."""
import copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from plan_core import pin,decode,module_from_path,project_records,Reader,build
ROOT=Path.cwd();OUT=Path(__file__).resolve().parent;OLD=OUT.parent/'fresh-graph-build-plan-v1';Q=ROOT/'out/artifacts/verification/graph-pipeline-gate/development';W=ROOT/'out/artifacts/verification/graph-shape-normalization/after-binding-deadline-v1/final-memory-after-v1'
records=[];payload=OUT/'payload';payload.mkdir();allowed={'.lean','.py','.json','.mlir','.md','.txt','.patch','.xml'}
def store(raw,suffix,origin,expected=None):
 observed=pin(raw)
 if expected:
  assert observed['sha256']==expected['sha256'],origin
  if 'bytes' in expected:assert observed['bytes']==expected['bytes'],origin
 path='payload/'+observed['sha256']+suffix;target=OUT/path
 if target.exists():assert target.read_bytes()==raw
 else:target.write_bytes(raw)
 ref={'file':path,**observed};records.append({'origin':origin,'original_pin':expected,'current':observed,'ref':ref});return ref

def read(path,expected=None):
 assert path.suffix in allowed and not path.is_symlink(),str(path)
 return store(path.read_bytes(),path.suffix,str(path),expected)

def load(ref):return decode((OUT/ref['file']).read_bytes())
def stmt(query,final):
 text=query.decode();start=text.rfind('theorem '+final.rsplit('.',1)[-1]);end=text.find(':=',start);assert start>=0 and end>start;return text[start:end].rstrip()

packets={};profiles=[]
for packet,folderprefix,profileid,successor in [('schedule-profile-v1','actual-examples','Q-schedule-successor-v1','Q-service-CSE'),('unroll-tail-v1','final-actual-examples','Q-unroll-tail-successor-v1','Q-four-transform')]:
 base=Q/packet;manifest=read(base/'MANIFEST.json');files={};catalog=[]
 for row in load(manifest)['rows']:
  path=Path(row['path']);assert not path.is_absolute() and '..' not in path.parts
  if path.suffix not in allowed:continue
  ref=read(base/path,row);files[row['path']]=ref;module=module_from_path(row['path'])
  if module:catalog.append({'module':module,'source':ref,'provenance':{'fixed_packet':packet,'snapshot':row['path'],'original_pin':{k:row[k] for k in ('sha256','bytes')}}})
 cases=[];declared=None
 for relative,request in sorted(files.items()):
  if not relative.startswith(folderprefix+'/') or not relative.endswith('/REQUEST.json'):continue
  folder=relative.rsplit('/',1)[0];caseid=folder.rsplit('/',1)[1];req=load(request);prepared=files[folder+'/PREPARED.json'];prep=load(prepared);query=req['query'].encode();queryref=files[folder+'/Query.lean'];assert (OUT/queryref['file']).read_bytes()==query and prep['query']==req['query'];selected=project_records(req['project_sources']);assert declared is None or declared==selected;declared=selected
  final=req['final_root'];support=req['support_roots'];assert prep['roots']==support+[final]
  label='IDENTITY_SOURCE_CANDIDATE_NOT_CHANGED_OPTIMIZATION' if prep.get('changed') is False else 'POSITIVE_SOURCE_CANDIDATE_UNOBSERVED'
  cases.append({'id':caseid,'module':'Query','query':queryref,'request_metadata':request,'prepared_metadata':prepared,
    'bindings':{'query':{'path':['query']},'final_root':{'path':['final_root']},'support_roots':{'path':['support_roots']}},
    'project_sources_binding':{'path':['project_sources']},'final_root':final,'support_roots':support,
    'domain':prep.get('domain'),'relation':prep.get('relation'),'assumptions':prep.get('premises'),
    'exact_final_declaration':stmt(query,final),'declaration_status':'SOURCE_TEXT_NOT_KERNEL_TYPE',
    'buffers':[{'label':Path(n).name,'ref':r} for n,r in files.items() if n.startswith(folder+'/') and n.endswith('.mlir')],
    'expected_protocol':label,'query_status':'UNCOMPILED','model_metadata':{'profile':prep.get('profile'),'changed':prep.get('changed'),
        'kind':prep.get('kind'),'query_contract':req.get('query_contract'),'exact_statement':req.get('exact_statement'),
        'compiler_pass_chain':prep.get('compiler_pass_chain'),'selection':prep.get('selection'),'boundary_label':prep.get('boundary_label')}})
 profiles.append({'id':profileid,'packet':packet,'packet_manifest':manifest,'project_prefixes':['Merlin.','MerlinVerify.'],
    'catalog':[r for r in catalog if r['module'] in declared],'cases':cases,'successor_of':successor})
 packets[packet]={'manifest':manifest,'files':files}

# Reuse the exact W entries normalized plan/data, without changing old packet.
original_inputs=decode((OLD/'INPUTS.json').read_bytes());entries=copy.deepcopy(next(p for p in original_inputs['profiles'] if p['id']=='W-entries-successor-v1'))
def copy_refs(value):
 if isinstance(value,dict):
  if all(k in value for k in ('file','sha256','bytes')) and value['file'].startswith('payload/'):
   raw=(OLD/value['file']).read_bytes();ref=store(raw,Path(value['file']).suffix,'immutable previous packet/'+value['file'],value);assert ref==value
  for v in value.values():copy_refs(v)
 elif isinstance(value,list):
  for v in value:copy_refs(v)
copy_refs(entries);profiles.append(entries)

# Caller-context is now fixed/sealed. Add it as a named successor, not a mutation.
base=W/'source-context-v1';seal=read(base/'SOURCE-SEAL.json');files={};catalog=[]
for row in load(seal)['files']:
 path=Path(row['path']);assert not path.is_absolute() and '..' not in path.parts
 if path.suffix not in allowed:continue
 ref=read(base/path,row);files[row['path']]=ref;module=module_from_path(row['path'])
 if module:catalog.append({'module':module,'source':ref,'provenance':{'fixed_packet':'source-context-v1','snapshot':row['path'],'original_pin':row['sha256']}})
cases=[]
for row in load(files['QUERIES.json']):
 refs={}
 for role in ('query','report','raw_source','raw_final'):
  expected=row.get(role)
  if expected is None:continue
  matches=[ref for ref in files.values() if ref['sha256']==expected['sha256']]
  assert matches,('fixed context evidence missing',row['case'],role)
  refs[role]=matches[0]
 report=load(refs['report']);prep=report['prepared'];final=row['final_root'];support=row['support_roots'];assert prep['final_root']==final and prep['support_roots']==support
 query=(OUT/refs['query']['file']).read_bytes();statement=stmt(query,final)
 cases.append({'id':row['case'],'module':'Context','query':refs['query'],'request_metadata':refs['report'],'prepared_metadata':refs['report'],
   'bindings':{'final_root':{'path':['prepared','final_root']},'support_roots':{'path':['prepared','support_roots']}},
   'final_root':final,'support_roots':support,'buffers':[{'label':k,'ref':v} for k,v in refs.items() if k.startswith('raw_')],
   'exact_final_declaration':statement,'declaration_status':'SOURCE_TEXT_NOT_KERNEL_TYPE','domain':{'exact_statement':statement,'status':'SOURCE_BINDERS_UNCHECKED'},
   'relation':{'source_form':'caller-context EntryDomain graphOutcomes iff','status':'SOURCE_DESCRIPTION_NOT_KERNEL_TYPE'},
   'assumptions':{'exact_binders_and_relation':statement,'status':'HUMAN_AND_KERNEL_REVIEW_REQUIRED'},
   'expected_protocol':'POSITIVE_SOURCE_CANDIDATE_UNOBSERVED' if row['expected_finite_binding_acceptance'] else 'EXPECTED_BINDING_REFUSAL_UNOBSERVED',
   'query_status':'UNCOMPILED','source_case_record':row,'model_metadata':{k:prep.get(k) for k in ('entry_roster','execution_witness_roots','execution_witness_refusals','scalar_literal_model','caller_context','all_entries_proved','whole_run_equality_is_premise','local_simulation_is_premise','native_failure_preservation_claimed','divergence_preservation_claimed')}})
profiles.append({'id':'W-caller-context-successor-v1','packet':'source-context-v1','packet_manifest':seal,'project_inventory':files['PROJECT-LEAN-SOURCES.json'],
 'project_prefixes':['Merlin.','MerlinVerify.'],'catalog':catalog,'cases':cases,'successor_of':'W-entries-successor-v1'})
packets['source-context-v1']={'manifest':seal,'files':files}
refs={r['ref']['file']:r['ref'] for r in records}
inputs={'schema':'merlin.fresh-graph-plan-inputs.v1','primary_baseline':'9d7fe0846c43574ca6f83a5cccc5dc147b06ca6e','future_root':'fresh-build',
 'profiles':profiles,'packet_data_refs':[refs[k] for k in sorted(refs)],'new_runtime_context':'UNSELECTED_NEW_F_INTAKE','native_or_graph_approval':False}
(OUT/'INPUTS.json').write_text(json.dumps(inputs,indent=2)+'\n');plan=build(inputs,Reader(OUT));(OUT/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
(OUT/'SOURCE_DATA_INVENTORY.json').write_text(json.dumps({'fixed_packets':packets,'source_data_associations':records,
 'preserved_old_INPUTS':pin((OLD/'INPUTS.json').read_bytes()),'preserved_old_PLAN':pin((OLD/'PLAN.json').read_bytes()),
 'all_old_six_plans_unchanged':True,'source_only_no_provider_execution':True},indent=2)+'\n')
print(json.dumps({'profiles':[(p['id'],len(p.get('sources',{})),len(p.get('cases',[])),p.get('reason')) for p in plan['profiles']],
 'distinct_query_bytes':len({c['query']['sha256'] for p in plan['profiles'] for c in p.get('cases',[])}),'source_data_payloads':len(refs)}))
assert all(p['status']=='SOURCE_DAG_PREPARED_NOT_COMPILED' and all(c['status']=='SOURCE_QUERY_DAG_PREPARED_NOT_CHECKED' for c in p['cases']) for p in plan['profiles'])
