"""Explicit5%-quota STOP checkpoint only; no tests, research, staging or native work."""
import difflib,hashlib,json
from pathlib import Path
ROOT=Path.cwd();OUT=Path(__file__).resolve().parent;BASE=OUT.parent;SCRATCH=ROOT/'out/build/graph-paper-deadline/fresh-graph-staging-v1'
def pin(raw):return {'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
validation=json.loads((OUT/'VALIDATION_BATCH.json').read_text());spec=json.loads((OUT/'capsule/STAGING_SPEC.json').read_text());plan=json.loads((OUT/'PLAN.json').read_text());raw_bindings=json.loads((OUT/'RAW_BINDINGS.json').read_text());source_rows=[];patch=[]
for name in ('materialize_plan.py','plan_core.py','validate_staging.py','assemble_successors.py','make_staging_spec.py'):
 path=OUT/name;raw=path.read_bytes();relative=path.relative_to(ROOT).as_posix();source_rows.append({'path':relative,**pin(raw),'status':'FROZEN_SDK_FREE_DATA_VALIDATED' if name in ('materialize_plan.py','plan_core.py','validate_staging.py') else 'FROZEN_SOURCE_DATA_ASSEMBLY_EXECUTED','native_qualification':False})
 patch.append(''.join(difflib.unified_diff([],raw.decode().splitlines(True),fromfile='/dev/null',tofile='b/'+relative)))
(OUT/'OWNED_CODE.patch').write_text(''.join(patch))
(OUT/'SOURCES.json').write_text(json.dumps({'role':'Assign','latest_assignment':'fresh-graph-staging-v1','state':'STOPPED_EXPLICIT_USER_5_PERCENT','branch':'feat/pass-verification','proposal_base':'9d7fe0846c43574ca6f83a5cccc5dc147b06ca6e','owned_sources':source_rows,'owned_patch':{'path':'OWNED_CODE.patch',**pin((OUT/'OWNED_CODE.patch').read_bytes()),'applied_staged_committed':False,'check_after_stop':'UNEXECUTED'},'qualified_data_batch':validation,'unfinished':['polished per-profile staging/qualification report and final acceptance/transfer package beyond current stop handoff','optional additional owner-interface hardening; no new campaign authorized by STOP'],'all_native_compile_audit_qualification':'UNEXECUTED','live_jobs':[]},indent=2)+'\n')
# Small exact publication index: code and report paths; one deduplicated capsule,
# not duplicate491 files, staging trees or raw/source campaign copies.
publication=[]
def add(path,role,status):
 raw=path.read_bytes();publication.append({'path':path.relative_to(ROOT).as_posix(),**pin(raw),'role':role,'status':status})
for row in source_rows:publication.append({**row,'role':'latest owned SDK-free staging code'})
for path,role in [(ROOT/'src/merlin/verify/graph_artifact_bundle.py','canonical verifier'),(ROOT/'merlin/tests/infra/test_graph_artifact_bundle.py','canonical tests')]:add(path,role,'ALREADY_PUBLISHED_CANONICAL_V2_0d9b2437f')
for relative in ('sources/src/merlin/targetgen/graph_consumer_bridge.py','sources/merlin/tests/targetgen/test_graph_consumer_bridge.py','sources/merlin/tests/fixtures/graph_consumer_bridge_fixture.py','PROTOTYPE_SOURCE_FIXTURE.patch','SOURCE_PATCH_INVENTORY.json','HANDOFF.md','RUNBOOK.md'):
 add(BASE/'host-consumer-bridge-v1'/relative,'inert chosen-consumer prototype/review','FROZEN_SDK_FREE_MOCKED_ONLY')
for relative in ('build_plan.py','verify_plan.py','SOURCE_TOOL_INVENTORY.json','PROFILE_INDEX.json','FRESH_QUALIFICATION_ROSTER.json','BUILD_AND_AUDIT_ROSTER.md','OWNER_INTERFACE_TASKS.md','HANDOFF.md','RUNBOOK.md'):
 add(BASE/'fresh-graph-build-plan-v1'/relative,'preserved six-DAG26-query source plan/review','FROZEN_SOURCE_DATA_ONLY')
for relative in ('SOURCES.json','OWNED_CODE.patch','capsule/STAGING_SPEC.json','PLAN.json','PRESERVED_PLANS.json','RAW_BINDINGS.json','SHARED_SOURCE_CONFLICTS.json','ACTUAL_STAGING.json','VALIDATION_BATCH.json','BOUNDED_CONTROLS.json','RELOCATION_CHECK.json','lint.txt'):
 add(OUT/relative,'new fixed successor staging/review','FROZEN_SOURCE_DATA_ONLY')
roles={}
for profile in spec['profiles']:
 for source in profile['sources']:
  ref=source['ref'];entry=roles.setdefault(ref['file'],{'path':'capsule/'+ref['file'],**{k:ref[k] for k in ('sha256','bytes')},'role':'project_lean_source','profiles':[],'canonical_candidates':[]})
  entry['profiles'].append(profile['id']);candidate='src/merlin/verify/lean/'+source['module'].replace('.','/')+'.lean'
  if candidate not in entry['canonical_candidates']:entry['canonical_candidates'].append(candidate)
 for case in profile['cases']:
  for key,role in (('query','pure_generated_query'),('roots','exact_query_root_domain_sidecar')):
   ref=case[key];roles.setdefault(ref['file'],{'path':'capsule/'+ref['file'],**{k:ref[k] for k in ('sha256','bytes')},'role':role,'profiles':[],'canonical_candidates':[]})['profiles'].append(profile['id'])
# Preserve raw actual input/final bytes as a separate small review manifest;
# these are not copied into the Lean-only staging tree.
raw_refs={row['exact_snapshot_ref']['file']:row['exact_snapshot_ref'] for row in raw_bindings['bindings']}
(OUT/'RAW_IR_PUBLICATION_INDEX.json').write_text(json.dumps({'source_packet_relative_files':[{'path':k,**{f:r[f] for f in ('sha256','bytes')}} for k,r in sorted(raw_refs.items())],'bytes':sum(r['bytes'] for r in raw_refs.values()),'all_original_absolute_execution_receipts_reconstructed':False},indent=2)+'\n')
add(OUT/'RAW_IR_PUBLICATION_INDEX.json','small actual raw IR review manifest','FROZEN_SOURCE_DATA_ONLY')
index={'schema':'merlin.assign-publication-index.v1','role':'Assign','state':'STOPPED_EXPLICIT_USER_5_PERCENT','branch':'feat/pass-verification','base':'9d7fe0846c43574ca6f83a5cccc5dc147b06ca6e','finished_code_and_reports':publication,'latest_closed_capsule':{'root':OUT.relative_to(ROOT).as_posix()+'/capsule','specification':pin((OUT/'capsule/STAGING_SPEC.json').read_bytes()),'payloads':list(roles.values()),'payload_bytes':sum(r['bytes'] for r in roles.values()),'source_reconstruction_rule':'Use canonical candidates ONLY if exact supplied digest/size match; do not choose by module name or refresh expected pins. Copy each unique content once. Matching canonical files need not be duplicated in Git; reconstruct payload mapping for materializer.','canonical_match_checks':'NOT_PERFORMED_DURING_STOP','compiled_or_SDK_files':False},'exclude_from_git':['out/build staging/relocation/control trees','fresh-graph-build-plan-v1/relocated duplicate491 files','raw historical archive/runtime/SDK/object images','unneeded all-packet payload464 metadata/source copies'],'unfinished':['final polished staging/qualification narrative and additional owner-interface fixes stopped; current exact PLAN/root sidecars and stop HANDOFF suffice for review'],'native_context':'UNSELECTED','compile_audit_qualification':'UNEXECUTED','live_jobs':[]}
(OUT/'PUBLICATION_INDEX.json').write_text(json.dumps(index,indent=2)+'\n')
text='''# Assign — STOPPED at explicit5% quota direction

Latest assignment: fresh-graph-staging-v1. Branch feat/pass-verification; pinned proposal base9d7fe0846c43574ca6f83a5cccc5dc147b06ca6e. Root alone integrates/stages/commits/pushes. Do not autoresume after compaction; a NEW user/coordinator resume is required.

Completed/frozen: exact fixed Q schedule-profile-v1 (7modules/3queries), unroll-tail-v1 (25/4), preserved W source-entries-v1 (111/12) and fixed source-context-v1 successor (118/14).33 distinct query bytes; schedule unchanged_order is identity evidence, not a changed optimization. Old six-DAG/26-query packet and its W entries normalized profile remain unchanged. New plans/root sidecars preserve exact separate domains, roots, dependency order and actual raw bindings; no model/source-union theorem inferred.

materialize_plan.py + plan_core.py are stdlib-only. They admit only bound project .lean files, pure query .lean files and JSON sidecars into an exclusive NEW tree. Inputs are pinned before copying and rehashed immediately before each copy. Duplicate/conflicting IDs, missing/changed source, unsafe paths, cyclic DAG, extra input/output payload and existing destinations refuse. Partial first failures remain preserved; nothing is rolled back/overwritten. ContextUNSELECTED and all approval flagsfalse. No .olean/.ir/.so/SDK/provider/controller/compiled objects are materialized.

Validation complete: ONE actual4-profile materialization,337 files;10 tiny controls expected;337 same-file unrelated-CWD relocation associations pass. Exact tested materializer/core/spec pins are in VALIDATION_BATCH.json and SOURCES.json. Ruff passes. Only own utility code executes; all selected producer/provider/Lean input is DATA. Actual stage tree: out/build/graph-paper-deadline/fresh-graph-staging-v1/actual-staged. No live jobs.

Publication: PUBLICATION_INDEX.json lists exact code/report paths/hashes and the202-blob8,435,611B closed capsule. capsule/STAGING_SPEC.json is its manifest. Avoid duplicate491 older source copies/staging/relocation/raw archives in Git; canonical source alternatives may be used ONLY after matching supplied hashes/sizes. Pure query/root sidecars must remain exact. RAW_IR_PUBLICATION_INDEX.json names the small actual raw input/final review payload; RAW_BINDINGS.json explicitly leaves absent original absolute execution receiptsnull instead of fabricating them. OWNED_CODE.patch contains only this new owned code, UNAPPLIED/UNCHECKED after STOP.

Prior frozen work: canonical verifier/tests already published at0d9b2437f (24synthetic tests); host-consumer-bridge-v1 contains inert bridge +6 mocked consumer tests/3-file proposal at9d7; fresh-graph-build-plan-v1 has six separate DAGs/26queries, source-only builder/verifier, exact qualification roster/runbook and487-file data validation. Their handoffs/patches remain immutable. Chosen callback gating is not ALL ordinary compilation enforcement.

Precise SDK-free commands for a NEW reviewed operator action (examples below UNEXECUTED during STOP; do not run them automatically):

```sh
python3 -I -S -B materialize_plan.py --packet /chosen/closed-capsule --destination /chosen/existing-parent/NEW-exclusive-source-tree
python3 -I -S -B materialize_plan.py --verify-staged /chosen/source-tree
```

Required inputs: materialize_plan.py and plan_core.py together; capsule/STAGING_SPEC.json plus its exact202 payload refs; existing canonical parent and NEW destination outside original/main/private contexts. Original fixed source/data inventory and plans are retained here; original absolute path strings are provenance only. Existing destination refusal is intentional. Do not retry into preserved failed trees.

Unfinished at STOP: polished per-profile staging/qualification report/final acceptance packaging and any further owner-interface hardening. Exact PLAN.json, plans/, root sidecars, SOURCES.json, RAW_BINDINGS.json and PUBLICATION_INDEX.json already provide the current reviewable cut. No claim that the broader queue was fully finished after the explicit stop.

All new Lean queries/candidatesUNCOMPILED; no kernel/compiler execution or semantic qualification. All future fresh source builds, imported object/source association, exact final/support type/axiom audits, positive/valid-negative qualification, replay and human reviewUNEXECUTED. F alone owns new official SDK/loader/source-admission intake; old runtime causeUNKNOWN and old profilesHELD. No SDK/runtime/archive-image reads/guards/native/network/install, new agents/models/numerics, protected/frozen/other-worker edits, Git index/commit/push actions. Native contexts remainUNSELECTED. Root still must select coherent provider/release/import/callback interfaces and account for global compilation bypasses.
'''
(OUT/'HANDOFF.md').write_text(text);(OUT/'ACCOUNT_HANDOFF_DRAFT.md').write_text(text)
pointerdir=ROOT/'out/artifacts/verification/merlin-passes/progress/planner/paper-deadline-20261026/account-handoffs-20261026';pointerdir.mkdir(exist_ok=True);pointer=pointerdir/'Assign.md';pointer.write_text('''# Assign — explicit5% STOP checkpoint

[Portable handoff](../independent-reproduction/fresh-graph-staging-v1/HANDOFF.md)

[Exact publication manifest](../independent-reproduction/fresh-graph-staging-v1/PUBLICATION_INDEX.json)

[Sources and owned patch](../independent-reproduction/fresh-graph-staging-v1/SOURCES.json)

[Account handoff draft](../independent-reproduction/fresh-graph-staging-v1/ACCOUNT_HANDOFF_DRAFT.md)

Live jobs: none. Source-only staging cut validated; all new Lean/native qualificationUNCOMPILED/UNEXECUTED. No autoresume without NEW user/coordinator direction. Root alone publishes.
''')
a=BASE/'ACTIVE.md';previous=a.read_text();boundary=previous.index('\n---\n');a.write_text('''# STOPPED — explicit user5% quota safe-stop

Latest task fresh-graph-staging-v1; STOP supersedes keep-working instructions. Do NOT autoresume after compaction without NEW user/coordinator resume.
Portable HANDOFF.md, SOURCES.json, OWNED_CODE.patch, PUBLICATION_INDEX.json and ACCOUNT_HANDOFF_DRAFT.md are in fresh-graph-staging-v1/. Easy pointer: deadline/account-handoffs-20261026/Assign.md.
Validated materializer:4 separate fixed successor profiles7/25/111/118 sources and33queries; ONE actual337-file staging,10tinycontrols and337same-file unrelated-CWD relocation checks pass.202blob8.44MB closedcapsule. Prior6DAG/26query/hostconsumer/canonical packets unchanged. LivejobsNONE.
Unfinished: polished final staging/qualification narrative/acceptance packaging and any further interface hardening. All new LeanUNCOMPILED; builds/audits/qualification/replayUNEXECUTED; contextUNSELECTED, runtimeUNKNOWN/profilesHELD. No SDK/runtime/native/guards/install/network/numerics/agents/models or primary/protected/index/publication actions.
'''+previous[boundary:])
# Add the final stop documents to the exact small publication index last.
for path,role in [(OUT/'HANDOFF.md','portable STOP handoff'),(OUT/'ACCOUNT_HANDOFF_DRAFT.md','quota-stop handoff'),(pointer,'easy account pointer')]:add(path,role,'FROZEN_STOP_CHECKPOINT')
index['finished_code_and_reports']=publication;(OUT/'PUBLICATION_INDEX.json').write_text(json.dumps(index,indent=2)+'\n')
print(json.dumps({'role':'Assign','handoff':str((OUT/'HANDOFF.md').relative_to(ROOT)),'pointer':str(pointer.relative_to(ROOT)),'publication_manifest':{'path':str((OUT/'PUBLICATION_INDEX.json').relative_to(ROOT)),**pin((OUT/'PUBLICATION_INDEX.json').read_bytes())},'live_jobs':[]}))
