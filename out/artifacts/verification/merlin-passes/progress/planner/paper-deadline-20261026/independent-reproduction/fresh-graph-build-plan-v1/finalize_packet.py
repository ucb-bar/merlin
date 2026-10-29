"""Final source/data metadata checkpoint. No new test/build/native operation."""
import ast,hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent

def pin(raw):return {'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
plan=json.loads((OUT/'PLAN.json').read_text());inputs=json.loads((OUT/'INPUTS.json').read_text());validation=json.loads((OUT/'VALIDATION_BATCH.json').read_text());profile_index=json.loads((OUT/'PROFILE_INDEX.json').read_text());relocation=json.loads((OUT/'RELOCATION_CHECK.json').read_text())
assert validation['all_tiny_controls_expected'] and not validation['guard_denials'];assert len(plan['profiles'])==6
assert all(p['status']=='SOURCE_DAG_PREPARED_NOT_COMPILED' and all(c['status']=='SOURCE_QUERY_DAG_PREPARED_NOT_CHECKED' for c in p['cases']) for p in plan['profiles'])
source_inventory=[]
for name in ('build_plan.py','verify_plan.py','validate_data_batch.py','assemble_inputs.py','normalize_input_bindings.py','bind_claim_text.py','make_rosters.py','write_reports.py'):
 raw=(OUT/name).read_bytes();tree=ast.parse(raw);compile(tree,name,'exec',dont_inherit=True)
 source_inventory.append({'path':name,**pin(raw),'imports':[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))],
    'role':'qualified pure stdlib builder/verifier' if name in ('build_plan.py','verify_plan.py') else 'reviewed source/data assembly/report/control helper; no native authority'})
(OUT/'SOURCE_TOOL_INVENTORY.json').write_text(json.dumps({'tools':source_inventory,'input':pin((OUT/'INPUTS.json').read_bytes()),'plan':pin((OUT/'PLAN.json').read_bytes()),
    'call_path':'own stdlib code only; no subprocess/archive/SDK/provider/controller imports or execution by builder/verifier',
    'only_new_local_data_writes':True,'original_current_source_query_associations':'SOURCE_DATA_INVENTORY.json and ACTUAL_PLAN_VALIDATION.json'},indent=2)+'\n')
external=[{'item':'new official SDK F context and library audit/root/object policy','status':'MISSING_SEPARATE_INDEPENDENT_INTAKE; no old absolute authority'},
 {'item':'implicit/external names Init, Init.Data.List.Lemmas, Lean.Elab.Tactic.Omega','status':'NAMES_ONLY; SDK/source/object/runtime fidelity unobserved'},
 {'item':'root chosen Q/W live provider/typed Context API and same completion revalidation','status':'NEW_SOURCE_INTERFACE_REQUIRED; original fixed provider snapshots are DATA not new authorization'},
 {'item':'Q valid-negative qualification roster and per-module library audits','status':'REQUIRED_NOT_SUPPLIED_BY_TINY_DATA_CONTROLS'},
 {'item':'W EntryDomain/scalar-literal/allocator/caller/nonvacuity and valid-negative classification','status':'SOURCE/HUMAN/KERNEL_REVIEW_REQUIRED'},
 {'item':'real host/compiler import custody/callback and remaining ordinary compile bypasses','status':'OPEN; previous host-consumer prototype enforces chosen callback only'},
 {'item':'unspecified new schedule successor','status':'UNKNOWN_OPTIONAL; no fixed named input selected; preserve plan and add separate successor later'},
 {'item':'human source/model/policy/claim approval','status':'REQUIRED_NOT_RECORDED'}]
prior=OUT.parent/'host-consumer-bridge-v1/HANDOFF.md'
(OUT/'EXTERNAL_DEPENDENCIES.json').write_text(json.dumps({'requirements':external,'host_consumer_handoff_provenance':{'path':'../host-consumer-bridge-v1/HANDOFF.md',**pin(prior.read_bytes()),'open_global_bypass_decisions':True},
 'external_API_comparison':None,'official_SDK_runtime_comparison':None,'native_authority':False},indent=2)+'\n')
acceptance=[{'item':1,'status':'COMPLETE_SOURCE_QUERY_METADATA_PINS','evidence':['INPUTS.json','SOURCE_DATA_INVENTORY.json','payload/']},
 {'item':2,'status':'COMPLETE_SEPARATE_CLOSED_HEADER_DAGS','evidence':['PLAN.json','plans/','SHARED_SOURCE_CONFLICTS.json']},
 {'item':3,'status':'COMPLETE_STDLIB_BUILDER_VERIFIER','evidence':['build_plan.py','verify_plan.py','SOURCE_TOOL_INVENTORY.json']},
 {'item':4,'status':'COMPLETE_ONE_BOUNDED_DATA_BATCH','evidence':['ACTUAL_PLAN_VALIDATION.json','BOUNDED_CONTROLS.json','VALIDATION_BATCH.json']},
 {'item':5,'status':'COMPLETE_FRESH_WORK_ROSTERS_RELOCATION_RUNBOOK','evidence':['FRESH_QUALIFICATION_ROSTER.json','BUILD_AND_AUDIT_ROSTER.md','RELOCATION_CHECK.json','RUNBOOK.md']},
 {'item':6,'status':'COMPLETE_W_ENTRIES_SEPARATE_SUCCESSOR; UNSPECIFIED_SCHEDULE_UNKNOWN','evidence':['BASELINE_PLAN_V1.json','W_ENTRY_SUCCESSOR_PLAN_V1.json','OWNER_INTERFACE_TASKS.md']}]
(OUT/'ACCEPTANCE.json').write_text(json.dumps({'deliverables':acceptance,'all_future_native_steps':'UNEXECUTED','accepted_qualified_cases':0,'primary_protected_source_edits':False},indent=2)+'\n')
# Required portable verifier transfer: exactly487 deduplicated payload files,
# normalized input/plan and own two utility sources. Other reports remain local.
required=['build_plan.py','verify_plan.py','INPUTS.json','PLAN.json']+[r['file'] for r in inputs['packet_data_refs']]
transfer=[{'path':name,**pin((OUT/name).read_bytes())} for name in required]
(OUT/'TRANSFER_INDEX.json').write_text(json.dumps({'schema':'merlin.fresh-graph-source-data-transfer.v1','required_verifier_files':transfer,
 'required_bytes':sum(r['bytes'] for r in transfer),'all_payload_source_or_query_or_metadata':True,'SDK_runtime_images_or_old_objects_included':False,
 'source_data_only_no_graph_approval':True,'all_review_and_roster_files':'FILE_INDEX.json','relocation_files_are_hardlinks_not_required_second_copy':True},indent=2)+'\n')
active=OUT.parent/'ACTIVE.md';old=active.read_text();boundary=old.index('\n---\n');active.write_text('''# COMPLETE — fresh graph source-build plan v1

Latest six-item source/data queue COMPLETE, including fixed W entry-roster successor.
Handoff: fresh-graph-build-plan-v1/HANDOFF.md. Exact normalized inputs/plan: INPUTS.json/PLAN.json; six per-profile plans and BASELINE_PLAN_V1/W_ENTRY_SUCCESSOR_PLAN_V1 kept separate.
Qservice7/17roots, chain7/48, scalarSCF24/57, canonicalinteger30/82; Wwhole106/215 and Wentries111/249.26 distinctqueries =6positivecandidates+20expectedbindingrefusals, all outcomesUNOBSERVED, accepted/qualified0. Exact sources/query/root/buffer/domain/order/output requests retained. No source-union theorem inferred.
Stdlib builder/verifier validated487 source/datafiles in one bounded batch;10tinycontrols and491same-file unrelated-CWD associations pass. Source-only diagnostics, no oldQ/W suite/archivecampaign.0selected unavailableprofiles/cases;4optional historicalstdout/stderr artifacts excluded/notread with pins retained. Lean/Python shared fixed-source variants0.
Future library/main/query builds, exact kernel axiom/type audits, valid-negative qualification and freshreplay ALLUNEXECUTED; candidatequeriesUNCOMPILED. New officialSDK F context is missing separate intake, not oldabsolute authority. RuntimecauseUNKNOWN, oldprofilesHELD; host-consumer globalbypass decisionsOPEN.
No SDK/Std/runtime/archive-image reads, guards/loader/compiler/native/--deps/network/install, numerics, agents/models, primary/protected/frozen/lock/source-policy/publication edits. Primary9d7 canonical+Fprimary25 notreapplied. Prior packets immutable.
Next belongs to root/F/Q/W/human reviewers: select new context/library audit/provider/consumer interfaces, qualify exact conditional profiles, then any explicitly named separate schedule successor. Optional unspecified schedule stateUNKNOWN; no pending dependency requires polling. Root alone publishes.
''' +old[boundary:])
# Skip physical duplicate relocation links in the file index, referencing their
# already indexed counterparts and samefile report; every preserved path named.
rows=[]
for path in sorted(OUT.rglob('*')):
 if path.is_file() and path.name not in ('FILE_INDEX.json','CHECKPOINT_VALIDATION.json'):
  assert not path.is_symlink();relative=path.relative_to(OUT).as_posix()
  rows.append({'path':relative,**pin(path.read_bytes()),'relocation_hardlink':relative.startswith('relocated/')})
(OUT/'FILE_INDEX.json').write_text(json.dumps({'schema':'merlin.source-plan-packet-index.v1','files':rows,'excluded':['FILE_INDEX.json','CHECKPOINT_VALIDATION.json'],
 'integrity_only':True,'relocated_payload_is_samefile_not_second_transfer_requirement':True},indent=2)+'\n')
# Final metadata acceptance verifies held packet outputs only, not another DAG
# campaign, compiler call or proof test.
for row in transfer:assert pin((OUT/row['path']).read_bytes())=={k:row[k] for k in ('sha256','bytes')}
for row in profile_index['profiles']:assert pin((OUT/row['plan']['file']).read_bytes())=={k:row['plan'][k] for k in ('sha256','bytes')}
assert not plan['shared_source_conflicts'] and relocation['status']=='PASSED_SAME_FILE_UNRELATED_CWD'
summary={'status':'COMPLETE_FRESH_GRAPH_SOURCE_BUILD_QUEUE','profiles':6,'module_counts':[len(p['sources']) for p in plan['profiles']],
 'distinct_query_bytes':26,'positive_candidates':6,'expected_binding_refusals':20,'accepted_or_qualified':0,
 'actual_source_data_files_validated':487,'tiny_controls':10,'same_file_relocation_associations':491,
 'mandatory_transfer_files':len(transfer),'mandatory_transfer_bytes':sum(r['bytes'] for r in transfer),
 'payload_physical_files':487,'separate_W_whole_and_entries_plans':True,'unspecified_schedule':'UNKNOWN_OPTIONAL_NOT_SELECTED',
 'file_index':pin((OUT/'FILE_INDEX.json').read_bytes()),'source_tool_inventory':pin((OUT/'SOURCE_TOOL_INVENTORY.json').read_bytes()),
 'SDK_runtime_reads_or_native_launches':0,'provider_controller_Lean_execution':False,'future_build_audit_qualification_replay':'UNEXECUTED',
 'source_union_semantic_claim':False,'all_ordinary_compilation_claim':False,'old_primary_additions_reapplied':False,'prior_packets_unchanged':True}
(OUT/'CHECKPOINT_VALIDATION.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
