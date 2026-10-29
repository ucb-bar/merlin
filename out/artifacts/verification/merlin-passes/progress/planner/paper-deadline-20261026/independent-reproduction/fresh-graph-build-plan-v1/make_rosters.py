"""Render source-only future work and semantic-case rosters. No compiler argv."""
import hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent
plan=json.loads((OUT/'PLAN.json').read_text());inputs=json.loads((OUT/'INPUTS.json').read_text());inventory=json.loads((OUT/'SOURCE_DATA_INVENTORY.json').read_text())
def pin(raw):return {'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
folder=OUT/'plans';folder.mkdir()
profiles=[];qualification=[];output_requests=[]
for profile in plan['profiles']:
 name=profile['id'];raw=(json.dumps(profile,indent=2)+'\n').encode();(folder/(name+'.json')).write_bytes(raw);rows=[]
 for case in profile['cases']:
  rows.append({'case':case['id'],'expected_protocol':case['expected_protocol'],'query':case['query'],
               'module':case['module'],'final_root':case['final_root'],'support_roots':case['support_roots'],
               'root_count_is_not_case_count':len(case['support_roots'])+1,
               'exact_final_declaration':case['exact_final_declaration'],'domain':case['domain'],'relation':case['relation'],
               'assumptions':case['assumptions'],'actual_buffers':case['buffers'],'model_metadata':case['model_metadata'],
               'source_plan':{'file':'plans/'+name+'.json',**pin(raw)},'kernel_result':'UNOBSERVED',
               'fresh_build':'UNEXECUTED','axiom_type_audit':'UNEXECUTED','qualification':'UNEXECUTED','fresh_replay':'UNEXECUTED',
               'valid_negative_preconditions': 'UNREVIEWED; validate syntax/attribution/domain and reach exact authorized kernel; infrastructure/unsupported is not proof refusal' if case['expected_protocol'].startswith('EXPECTED_') else None,
               'semantic_counterexample_proved':False,'native_or_graph_acceptance':False})
  output_requests.append({'profile':name,'case':case['id'],'request':case['output_directory_request'],'created':False,
                          'context':'MISSING_NEW_F_OFFICIAL_SDK_INTAKE','old_absolute_workspace_reused':False})
 profiles.append({'id':name,'successor_of':profile['successor_of'],'plan':{'file':'plans/'+name+'.json',**pin(raw)},
                  'project_sources':len(profile['sources']),'cases':len(rows),'roots_per_case':[len(c['support_roots'])+1 for c in profile['cases']],
                  'library_output_request':profile['support_library_output_request'],'library_build':'UNEXECUTED',
                  'source_plan_identity':profile['source_plan_identity']})
 qualification.append({'profile':name,'successor_of':profile['successor_of'],'project_source_map':profile['sources'],
                       'deterministic_support_library_order':profile['support_library_build_order'],
                       'library_root_audit_selection':'REQUIRED_NEW_OWNER_REVIEW; no hidden default SDK/old-context policy selected',
                       'main_and_support_roles':'per-case source_build_roles_in_dependency_order; always honor full dependency order',
                       'cases':rows,'positive_valid_negative_and_replay_qualification':'UNEXECUTED',
                       'new_runtime_context':'MISSING_SEPARATE_INDEPENDENT_REVIEW','human_review':'NOT_RECORDED'})
# The old whole plan and its successor are separate immutable named files.
baseline={**plan,'profiles':[p for p in plan['profiles'] if p['id']!='W-entries-successor-v1'],
          'selection':'Q fixed profiles plus W-whole-v1 only; successor separate'}
(OUT/'BASELINE_PLAN_V1.json').write_text(json.dumps(baseline,indent=2)+'\n')
(OUT/'W_ENTRY_SUCCESSOR_PLAN_V1.json').write_text(json.dumps({'schema':plan['schema'],'profile':next(p for p in plan['profiles'] if p['id']=='W-entries-successor-v1'),
    'preserves':'plans/W-whole-v1.json','replacement_of_old_plan':False,'native_or_semantic_qualification':False},indent=2)+'\n')
(OUT/'PROFILE_INDEX.json').write_text(json.dumps({'profiles':profiles,'combined_source_build_implies_combined_theorem':False,
    'not_selected_schedule_successor':{'fixed_packet':None,'state':'UNKNOWN_NOT_NAMED_BY_THIS_ASSIGNMENT; add separately after root selection'},
    'all_new_builds_and_audits':'UNEXECUTED'},indent=2)+'\n')
(OUT/'FRESH_QUALIFICATION_ROSTER.json').write_text(json.dumps({'profiles':qualification,'semantic_cases':26,
    'distinct_query_bytes':len({c['query']['sha256'] for p in plan['profiles'] for c in p['cases']}),
    'declared_positive_candidates':sum(c['expected_protocol'].startswith('POSITIVE') for p in plan['profiles'] for c in p['cases']),
    'declared_expected_binding_refusals':sum(c['expected_protocol'].startswith('EXPECTED') for p in plan['profiles'] for c in p['cases']),
    'counting':'root counts, header controls, source-union compilation, historical tests and replay repetitions do not increase semantic case counts',
    'accepted_or_qualified_cases':0,'runtime_incident':'UNKNOWN','old_profiles':'HELD'},indent=2)+'\n')
(OUT/'OUTPUT_DIRECTORY_REQUESTS.json').write_text(json.dumps({'requests':output_requests,'creation_or_runtime_authorization':False,
    'absolute_parent':'to be selected in new independently reviewed F context; no old original absolute reference opened'},indent=2)+'\n')
# Exact shared Python source cuts are separate from conflict-free Lean headers.
logical={}
for packet,metadata in inventory['packets'].items():
 for relative,ref in metadata['metadata_files'].items():
  if not relative.endswith('.py') or '/src/merlin/' not in '/'+relative:continue
  path='src/merlin/'+('/'+relative).rsplit('/src/merlin/',1)[1]
  logical.setdefault(path,{}).setdefault(ref['sha256'],{'bytes':ref['bytes'],'packets':set(),'refs':set()})
  logical[path][ref['sha256']]['packets'].add(packet);logical[path][ref['sha256']]['refs'].add(ref['file'])
conflicts=[]
for path,variants in sorted(logical.items()):
 if len(variants)>1:conflicts.append({'logical_source':path,'variants':[{'sha256':digest,'bytes':v['bytes'],'packets':sorted(v['packets']),'files':sorted(v['refs'])} for digest,v in sorted(variants.items())],
                                  'decision':'DO_NOT_COMBINE_OR_REFRESH_PINS; owner chooses a NEW coherent host release or preserves profile isolation'})
(OUT/'SHARED_SOURCE_CONFLICTS.json').write_text(json.dumps({'Lean':plan['shared_source_conflicts'],'Python_host_variants':conflicts,
    'new_runtime_provider_interfaces':'MISSING_REVIEWED_F_OFFICIAL_SDK_CONTEXT_CUT; old absolute/private-image policy has no authority',
    'host_consumer_global_bypass_decisions':'OPEN; chosen callback gating is not ALL ordinary compilation enforcement'},indent=2)+'\n')
print(json.dumps({'separate_profile_plans':len(profiles),'semantic_candidate_cases':26,'positive_candidates':6,'expected_binding_refusals':20,
                  'accepted_qualified':0,'Lean_shared_conflicts':len(plan['shared_source_conflicts']),'Python_host_variants':len(conflicts)}))
