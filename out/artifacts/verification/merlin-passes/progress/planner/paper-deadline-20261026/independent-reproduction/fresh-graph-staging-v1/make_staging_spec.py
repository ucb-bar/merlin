"""Make a SMALL closed materialization capsule, sources/query/roots ONLY."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from plan_core import pin,decode
OUT=Path(__file__).resolve().parent;plan=decode((OUT/'PLAN.json').read_bytes());packet=OUT/'capsule';packet.mkdir();(packet/'payload').mkdir();old=OUT.parent/'fresh-graph-build-plan-v1'

def add(raw,suffix):
 identity=pin(raw);file='payload/'+identity['sha256']+suffix;path=packet/file
 if path.exists():assert path.read_bytes()==raw
 else:path.write_bytes(raw)
 return {'file':file,**identity}

profiles=[];raw_bindings=[];shared={};plans_dir=OUT/'plans';plans_dir.mkdir()
for profile in plan['profiles']:
 (plans_dir/(profile['id']+'.json')).write_text(json.dumps(profile,indent=2)+'\n')
 sources=[]
 for module,row in sorted(profile['sources'].items()):
  ref=add((OUT/row['source']['file']).read_bytes(),'.lean');assert pin((packet/ref['file']).read_bytes())=={k:row['source'][k] for k in ('sha256','bytes')}
  sources.append({'module':module,'ref':ref});shared.setdefault(module,{}).setdefault(ref['sha256'],[]).append(profile['id'])
 cases=[]
 for case in profile['cases']:
  query=add((OUT/case['query']['file']).read_bytes(),'.lean');assert query==case['query'];buffer_refs=[]
  source_record=case.get('source_case_record',{})
  for buffer in case['buffers']:
   raw=(OUT/buffer['ref']['file']).read_bytes();assert pin(raw)=={k:buffer['ref'][k] for k in ('sha256','bytes')}
   original=source_record.get(buffer['label'],{})
   original_path=original.get('path')
   original_absolute=original_path if original_path and Path(original_path).is_absolute() else None
   association={'profile':profile['id'],'case':case['id'],'label':buffer['label'],'exact_snapshot_ref':buffer['ref'],
      'original_recorded_path':original_path,'original_absolute_execution_receipt':original_absolute,
      'absolute_receipt_status':'RETAINED_PATH_STRING_ONLY_NOT_AUTHORITY' if original_absolute else 'NOT_RETAINED_OR_ONLY_RELATIVE; fixed snapshot bytes are available; no absolute execution receipt fabricated',
      'snapshot_hash_matched':True,'materialized_in_Lean_tree':False}
   raw_bindings.append(association);buffer_refs.append({'label':buffer['label'],'source_packet_ref':buffer['ref'],'materialized':False})
  sidecar={'schema':'merlin.graph-staging-roots.v1','profile':profile['id'],'case':case['id'],'module':case['module'],
    'query_pin':{k:query[k] for k in ('sha256','bytes')},'final_root':case['final_root'],'support_roots':case['support_roots'],
    'exact_final_declaration':case['exact_final_declaration'],'domain':case['domain'],'relation':case['relation'],'assumptions':case['assumptions'],
    'expected_protocol':case['expected_protocol'],'model_metadata':case.get('model_metadata'),'buffers_external_to_capsule':buffer_refs,
    'source_plan_identity':profile['source_plan_identity'],'original_source_data_provenance':{'request_metadata':case['request_metadata'],
        'prepared_metadata':case['prepared_metadata'],'refs_base':'source-data handoff packet, NOT the closed materialization capsule'},
    'native_context_request':'UNSELECTED','compile':'UNEXECUTED','audit':'UNEXECUTED','qualification':'UNEXECUTED',
    'query_status':'UNCOMPILED','native_authorization':False,'graph_approval':False,'human_review':False}
  roots=add((json.dumps(sidecar,indent=2)+'\n').encode(),'.json');cases.append({'id':case['id'],'module':case['module'],'query':query,'roots':roots})
 profiles.append({'id':profile['id'],'project_prefixes':['Merlin.','MerlinVerify.'],'sources':sources,
    'dependency_order':profile['support_library_build_order'],'cases':cases,'provenance':{'source_plan_identity':profile['source_plan_identity'],
        'successor_of':profile['successor_of'],'source_plan_ref':{'file':'plans/'+profile['id']+'.json',**pin((plans_dir/(profile['id']+'.json')).read_bytes())},
        'native_context_request':'UNSELECTED','build_qualification':'UNEXECUTED','source_union_is_semantic_theorem':False}})
spec={'schema':'merlin.graph-staging-spec.v1','primary_baseline':plan['primary_baseline'],'profiles':profiles}
(packet/'STAGING_SPEC.json').write_text(json.dumps(spec,indent=2)+'\n')
(OUT/'RAW_BINDINGS.json').write_text(json.dumps({'bindings':raw_bindings,'no_original_absolute_associations_invented':True,
    'all_actual_snapshot_bytes_available':True,'all_raw_IR_is_external_to_closed_Lean_capsule':True},indent=2)+'\n')
(OUT/'SHARED_SOURCE_CONFLICTS.json').write_text(json.dumps({'variants':[{'module':n,'variants':v} for n,v in sorted(shared.items()) if len(v)>1],
    'no_model_or_object_merging':True,'shared_identical_bytes_do_not_merge_profiles':True},indent=2)+'\n')
old_entries=next(p for p in decode((old/'PLAN.json').read_bytes())['profiles'] if p['id']=='W-entries-successor-v1');new_entries=next(p for p in plan['profiles'] if p['id']=='W-entries-successor-v1');assert old_entries==new_entries
(OUT/'PRESERVED_PLANS.json').write_text(json.dumps({'old_six_plan':{'path':'../fresh-graph-build-plan-v1/PLAN.json',**pin((old/'PLAN.json').read_bytes())},
   'old_entries_plan_reused_byte_equal_in_normalized_profile':True,'old_six_plan_mutated':False,'new_context_and_Q_profiles_are_separate_successors':True},indent=2)+'\n')
print(json.dumps({'profiles':len(profiles),'cases':sum(len(p['cases']) for p in profiles),'unique_capsule_payloads':len(list((packet/'payload').iterdir())),
 'capsule_payload_bytes':sum(p.stat().st_size for p in (packet/'payload').iterdir()),'raw_bindings':len(raw_bindings),'shared_conflicts':sum(len(v)>1 for v in shared.values())}))
