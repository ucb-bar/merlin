"""Pin fixed source/query/metadata as DATA; never import/execute copied contents."""
import hashlib,json
from pathlib import Path

ROOT=Path.cwd();OUT=ROOT/'out/artifacts/verification/merlin-passes/progress/planner/paper-deadline-20261026/independent-reproduction/fresh-graph-build-plan-v1';Q=ROOT/'out/artifacts/verification/graph-pipeline-gate/development';W=ROOT/'out/artifacts/verification/graph-shape-normalization/after-binding-deadline-v1/final-memory-after-v1'
ALLOWED={'.py','.lean','.json','.mlir','.md','.txt','.patch','.xml'}
def pin(raw):return {'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
records=[];unavailable=[]
def store(raw,suffix,origin,expected=None):
 observed=pin(raw)
 if expected is not None:
  assert observed['sha256']==expected['sha256'],origin
  if 'bytes' in expected:assert observed['bytes']==expected['bytes'],origin
  if 'size' in expected:assert observed['bytes']==expected['size'],origin
 path=Path('payload')/(observed['sha256']+suffix);target=OUT/path;target.parent.mkdir(exist_ok=True)
 if target.exists():assert target.read_bytes()==raw
 else:target.write_bytes(raw)
 ref={'file':path.as_posix(),**observed};records.append({'origin':origin,'expected':expected,'current_observed':observed,'ref':ref});return ref

def read(path,expected=None):
 assert path.suffix in ALLOWED and not path.is_symlink(),str(path)
 return store(path.read_bytes(),path.suffix,str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),expected)

def load(ref):return json.loads((OUT/ref['file']).read_bytes())
def module_name(path):
 marker='/lean/'
 if marker not in '/'+path:return None
 tail=('/'+path).split(marker,1)[1]
 if not tail.endswith('.lean') or not tail.startswith(('Merlin/','MerlinVerify/')):return None
 return tail[:-5].replace('/','.')

packets={}
for name in ('service-route-review-v1','chain-endpoint-v1','larger-profiles-v1'):
 base=Q/name;manifest=read(base/'MANIFEST.json');doc=load(manifest);files={};catalog=[]
 for row in doc['rows']:
  relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts
  if relative.suffix not in ALLOWED:
   unavailable.append({'packet':name,'path':row['path'],'reason':'non-source/data extension excluded without read'});continue
  ref=read(base/relative,row);files[row['path']]=ref
  mod=module_name(row.get('source',row['path'])) or module_name(row['path'])
  if mod:catalog.append({'module':mod,'source':ref,'provenance':{'packet':name,'snapshot':row['path'],'original':row.get('source'),'original_pin':{k:row[k] for k in ('sha256','bytes')}}})
 packets[name]={'manifest':manifest,'files':files,'catalog':catalog}

profiles=[]
def qcase(packet,folder,identifier,service=False):
 files=packets[packet]['files'];metadata=files[folder+'/PREPARED.json'];prepared=load(metadata)
 if service:
  request=metadata;query=prepared['query'];roots=prepared['roots'];final=roots[-1];support=roots[:-1];bindings={'final_root':{'path':['roots'],'last':True},'support_roots':{'path':['roots'],'all_but_last':True},'query':{'path':['query']}};module='GraphPipelineGateQuery'
 else:
  request=files[folder+'/REQUEST.json'];rd=load(request);query=rd['query'];final=rd['final_root'];support=rd['support_roots'];bindings={'final_root':{'path':['final_root']},'support_roots':{'path':['support_roots']},'query':{'path':['query']}};module=rd['module'];assert prepared['query']==query and prepared['roots']==support+[final]
 qr=store(query.encode(),'.lean',packet+'/'+folder+' embedded exact query')
 buffers=[{'label':Path(name).name,'ref':ref} for name,ref in files.items() if name.startswith(folder+'/') and name.endswith('.mlir')]
 return {'id':identifier,'module':module,'query':qr,'request_metadata':request,'prepared_metadata':metadata,'bindings':bindings,'query_status':'UNCOMPILED','expected_protocol':'POSITIVE_CANDIDATE_NOT_ACCEPTED','final_root':final,'support_roots':support,'buffers':buffers,'domain':prepared.get('domain'),'relation':prepared.get('relation'),'assumptions':prepared.get('premises'),'model_metadata':{'profile':prepared.get('profile'),'query_contract':prepared.get('query_contract'),'compiler_pass_chain':prepared.get('compiler_pass_chain','not_established')},'original_buffer_associations':'raw actual bytes retained; semantic joins require fresh audit'}
for packet,folder,identifier,service in [('service-route-review-v1','example','Q-service-CSE',True),('chain-endpoint-v1','final-example','Q-chain-endpoint',False),('larger-profiles-v1','final-examples/pipeline','Q-four-transform',False),('larger-profiles-v1','final-examples/integer','Q-integer-five-transform',False)]:
 profiles.append({'id':identifier,'packet':packet,'packet_manifest':packets[packet]['manifest'],'project_prefixes':['Merlin.','MerlinVerify.'],'catalog':packets[packet]['catalog'],'cases':[qcase(packet,folder,identifier,service)],'successor_of':None})

for cut,identifier,successor in [('source-whole-v1','W-whole-v1',None),('source-entries-v1','W-entries-successor-v1','W-whole-v1')]:
 base=W/cut;seal=read(base/'SOURCE-SEAL.json');files={};catalog=[]
 for row in load(seal)['files']:
  relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts
  if relative.suffix not in ALLOWED:unavailable.append({'packet':cut,'path':row['path'],'reason':'non-source/data extension excluded without read'});continue
  ref=read(base/relative,row);files[row['path']]=ref;mod=module_name(row['path'])
  if mod:catalog.append({'module':mod,'source':ref,'provenance':{'packet':cut,'snapshot':row['path'],'original_pin':{'sha256':row['sha256']}}})
 rows=load(files['QUERIES.json']);cases=[]
 for row in rows:
  case=row['case'];refs={}
  # Prefer sealed evidence copies by exact known role/path and SHA; otherwise
  # follow only explicitly pinned SOURCE/QUERY/METADATA refs, never runtime refs.
  for role in ('query','report','raw_source','raw_final'):
   reference=row.get(role)
   if reference is None:continue
   matches=[r for r in files.values() if r['sha256']==reference['sha256']]
   if matches:refs[role]=matches[0];continue
   path=Path(reference['path']);path=path if path.is_absolute() else ROOT/path
   if not path.is_file():unavailable.append({'packet':cut,'case':case,'role':role,'reason':'pinned source/data unavailable','reference':reference});continue
   assert path.is_relative_to(W) and path.suffix in ('.lean','.json','.mlir')
   refs[role]=read(path,reference)
  if 'query' not in refs or 'report' not in refs:
   cases.append({'id':case,'status':'UNAVAILABLE','reason':'missing exact query/report','input_record':row});continue
  report=load(refs['report']);prepared=report['prepared'];assert prepared['final_root']==row['final_root'] and prepared['support_roots']==row['support_roots']
  # Older whole source seal pins only query/report. Read raw bytes only when a
  # known sibling exists AND its original digest is present in pinned report.
  for role,key,filename in [('raw_source','raw_source_sha256','source.mlir'),('raw_final','raw_final_sha256','final.mlir')]:
   if role in refs:continue
   path=Path(row['query']['path']);path=path if path.is_absolute() else ROOT/path;path=path.parent/filename
   if path.is_file() and prepared.get(key):refs[role]=read(path,{'sha256':prepared[key]})
   else:unavailable.append({'packet':cut,'case':case,'role':role,'reason':'raw actual buffer not retained with this sealed request','original_sha256':prepared.get(key)})
  cases.append({'id':case,'module':Path(row['query']['path']).stem,'query':refs['query'],'request_metadata':refs['report'],'prepared_metadata':refs['report'],'bindings':{'final_root':{'path':['prepared','final_root']},'support_roots':{'path':['prepared','support_roots']}},'final_root':row['final_root'],'support_roots':row['support_roots'],'buffers':[{'label':role,'ref':ref} for role,ref in refs.items() if role in ('raw_source','raw_final')],'expected_protocol':'POSITIVE_CANDIDATE_NOT_ACCEPTED' if row.get('expected_acceptance',row.get('expected_finite_binding_acceptance')) else 'EXPECTED_BINDING_REFUSAL_NOT_OBSERVED','domain':None,'relation':None,'assumptions':None,'model_metadata':{'prepared_report_ref':refs['report'],'local_simulation_is_premise':prepared.get('local_simulation_is_premise'),'whole_run_equality_is_premise':prepared.get('whole_run_equality_is_premise'),'scalar_literal_model':prepared.get('scalar_literal_model'),'source_plan_semantics_proved':prepared.get('source_plan_semantics_proved'),'all_entries_proved':prepared.get('all_entries_proved'),'native_failure_preservation_claimed':prepared.get('native_failure_preservation_claimed'),'divergence_preservation_claimed':prepared.get('divergence_preservation_claimed')},'source_case_record':row,'query_status':'UNCOMPILED','semantic_claim_fields':'not declared by these metadata; exact statement/query and proof source retained for owner/human review'})
 profiles.append({'id':identifier,'packet':cut,'packet_manifest':seal,'project_prefixes':['Merlin.','MerlinVerify.'],'catalog':catalog,'cases':cases,'successor_of':successor})
 packets[cut]={'manifest':seal,'files':files,'catalog':catalog}

inputs={'schema':'merlin.fresh-graph-plan-inputs.v1','primary_baseline':'9d7fe0846c43574ca6f83a5cccc5dc147b06ca6e','profiles':profiles,'future_root':'fresh-build','new_runtime_context':'MISSING_SEPARATE_F_OFFICIAL_SDK_INTAKE','native_or_graph_approval':False}
(OUT/'INPUTS.json').write_text(json.dumps(inputs,indent=2)+'\n');(OUT/'SOURCE_DATA_INVENTORY.json').write_text(json.dumps({'packets':{k:{'manifest':v['manifest'],'selected_catalog_rows':len(v['catalog']),'metadata_files':v['files']} for k,v in packets.items()},'all_read_source_data_associations':records,'unavailable':unavailable,'no_runtime_or_objects_read':True},indent=2)+'\n');(OUT/'UNAVAILABLE_INPUTS.json').write_text(json.dumps(unavailable,indent=2)+'\n');print(json.dumps({'profiles':len(profiles),'cases':sum(len(p['cases']) for p in profiles),'payload_files':len(list((OUT/'payload').iterdir())),'payload_bytes':sum(p.stat().st_size for p in (OUT/'payload').iterdir()),'unavailable':len(unavailable)}))
