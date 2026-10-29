"""Source/data reports and exact owner obligations; no runtime/compiler action."""
import hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent
plan=json.loads((OUT/'PLAN.json').read_text());index=json.loads((OUT/'PROFILE_INDEX.json').read_text());inventory=json.loads((OUT/'SOURCE_DATA_INVENTORY.json').read_text())
def pin(raw):return {'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
contracts=[]
for prof in plan['profiles']:
 for case in prof['cases']:
  raw=(OUT/case['query']['file']).read_bytes();lines=raw.decode().splitlines();found=[{'line':i,'text':line.strip()} for i,line in enumerate(lines,1) if line.lstrip().startswith(('#print','#check','#eval','#reduce'))]
  contracts.append({'profile':prof['id'],'case':case['id'],'query':case['query'],'source_line_diagnostic_directives':found,
                    'separate_kernel_name_type_axiom_audit':'UNEXECUTED','kernel_stdout_contract':'not tested; selected queries contain no source-line diagnostic directives'})
(OUT/'QUERY_CONTRACT_DATA.json').write_text(json.dumps({'queries':contracts,'source_text_observation_only':True,'provider_or_Lean_execution':False},indent=2)+'\n')
table='\n'.join('| '+p['id']+' | '+str(p['project_sources'])+' | '+str(p['cases'])+' | '+','.join(str(v) for v in sorted(set(p['roots_per_case'])))+' | '+str(p['successor_of'] or 'separate original profile')+' |' for p in index['profiles'])
(OUT/'PROFILE_ROSTER.md').write_text(f'''# Fresh source-build profiles, not qualification results

Primary baseline9d7fe0846 already contains canonical integrity utility and F's primary25 admission cut. No old additions or source patches are reapplied. The packets are fixed Q service-route-review-v1, chain-endpoint-v1, larger-profiles-v1, and W source-whole-v1 plus its fixed sealed source-entries-v1 successor. All source/query/metadata is retained as DATA in a deduplicated payload. No old project objects, SDK/runtime/archive images are included or opened.

| Separate profile | Bound project modules | Distinct query cases | Final+support roots per case | Relationship |
|---|---:|---:|---:|---|
{table}

26 distinct query byte sequences:6 declared positive candidates and20 declared expected binding refusals. Accepted/qualified cases0. The source metadata expectations are not observed outcomes. Support root counts are audit work, not extra test cases. The10 tiny source/data controls and relocation are not semantic cases. Historical Python tests, old qualifications and repeated builds/replays do not increase these counts.

Exact source maps, query refs, original/final and intermediate buffer refs, model/domain/premises, final+support root lists and requested output directories are in FRESH_QUALIFICATION_ROSTER.json and plans/*.json. Q profiles preserve their raw domain/relation/premise fields. W did not supply corresponding domain labels; exact final declaration binders/types and its original prepared model metadata are retained instead, with clearly labeled source-text descriptions. No Lean type has been inferred or kernel-checked by this tool.

Q scalar/chain retain complete defined-entry returned/error/terminal-world conditional relations; the chain direct endpoint and adjacent composition declarations are distinct roots, UNCOMPILED. Four-transform scalar/SCF and canonical integer-prefix profiles have their own static/no-tail/index/word/canonical-input and local external-success typing conditions; they are not extensions of a blanket gate approval. W whole is the typed kernel-entry/CallersValid common graphOutcomes relation. W entries separately quantifies definitions through each exact EntryDomain and chosen partial scalar/literal interpretation; its native failure/divergence and numeric fidelity are not established. W all_entries_proved remains false in retained source metadata, with every new query UNCOMPILED.

plans/W-whole-v1.json and BASELINE_PLAN_V1.json preserve the old whole selection. W_ENTRY_SUCCESSOR_PLAN_V1.json and plans/W-entries-successor-v1.json name the successor separately. Neither overwrites or upgrades the old result. A further schedule cut was not named/selected in this assignment: optional stateUNKNOWN; a future fixed cut needs a separately named successor plan, never an edit to this sealed plan.
''')
(OUT/'BUILD_AND_AUDIT_ROSTER.md').write_text('''# Required fresh phases — every operation UNEXECUTED

1. **New independent intake.** F/root must select and review a NEW official SDK/toolchain/context source cut, its external Init/Lean tactic objects/runtime/loader/source association, and an independent private root. Old absolute/private-image descriptors and an old basic Init probe are not authority. This packet never opens those references. No process argv or ready-to-run intake bypass is supplied.
2. **Project source libraries.** For each profile independently, freeze exactly the listed source bytes and follow deterministic support_library_build_order. Per-query source_build_roles_in_dependency_order distinguishes direct query imports from transitive support while preserving the merged dependency order; do not compile role lists in a different order. Identical shared source bytes do not justify mixing objects across contexts/profiles. An isolated source-union build would only compile libraries; it would prove no combined semantic theorem.
3. **Library audits.** Root/F/Q/W must bind an exact per-module root/type/axiom audit policy in the new context. The packet preserves all selected query support roots and source identities but does not manufacture missing library-audit root policy. Imported declaration bodies and actual imported object/source association require fresh observation. External module names are obligations, not copied SDK files or proof of availability.
4. **Per-query kernel compilation.** Use each exact query/module and its unique source-selection-digest/case directory request, with the profile's fresh source objects. No archived/old project objects or saved completion may substitute. All26 source-line scans have no #print/#check/#eval/#reduce directives; that data observation is not a Lean syntax or stdout audit. Query compile outcomes remainUNOBSERVED. Pure declaration compilation and its separate audit must retain exactly the same checked query/context binding.
5. **Exact final/support root audit.** Independently resolve and type-check the full ordered final/support roster for each query and inspect all exact transitive axioms under the newly reviewed policy. Source-text declaration excerpts and metadata/root pin association are not kernel name/type resolution. Missing/wrong roots, parser/loader/context failures and unavailable imports must be recorded as infrastructure/unavailable, not ordinary proof refusal. No human/native/std-library approval follows from matching bytes.
6. **Positive and valid-negative qualification.** Validate raw program syntax, complete scope/entry/arity/buffer attribution, intended model/domain and source association before calling a query a valid negative. Each expected wrong binding must reach the exact authorized kernel and refuse its actual final relation, not merely fail parsing, metadata, source guard or infrastructure. W's20 expected-refusal candidates are unobserved and are not semantic counterexamples. Q profiles still require independently reviewed valid-negative rosters; the tiny DAG/hash/root controls do not fill that gap. Retain raw failures and distinguish ordinary proof refusal from unavailable/unsupported/infrastructure outcomes.
7. **Fresh replay.** After a fixed source/context/query/root qualification and human decisions, reconstruct one retained actual boundary/chain under the SAME selected model/source policy in a fresh independent workspace, recheck source/object/runtime/root associations and retain a new result. Replay is not an additional distinct semantic case. No saved report seeds a live ledger or proof authority.

OUTPUT_DIRECTORY_REQUESTS.json contains exact relative requests, not created compilation directories. The separately authorized owner chooses/reviews the new absolute parent. Each source selection digest and each case has distinct source/object/query/audit requests. No old absolute workspace is opened/reused. No compile, --deps, SDK guard, loader, native process or installation was performed.
''')
(OUT/'OWNER_INTERFACE_TASKS.md').write_text('''# Source conflicts and concrete remaining interfaces

SHARED_SOURCE_CONFLICTS.json observes0 differing Lean module variants and0 logical core Python variants among these fixed packets. This is an exact byte comparison within the selected snapshots, not approval to merge them into current mutable worker source or one common semantic model. Each profile remains separately source-bound and has its own compile/output/root roster.

| Owner | Required decision/interface | Current status |
|---|---|---|
| F/root runtime | New official SDK/context source+runtime/object/source/loader intake and exact workspace authorization | MISSING_SEPARATE_DEPENDENCY. No source/provider/controller/native execution here; runtime causeUNKNOWN and old profilesHELD. |
| Q/F provider | A NEW host adapter/context cut must accept exact project module source pins, literal headers, library-root policy and exact query/module/final/support roots, and return/revalidate SAME live typed completion/audit/context identity | UNSELECTED. Old Q provider embeds Fv5 private-image source/class/location and historical review refs; those must not become authority for a new official SDK or be silently pin-refreshed. Preserve old cuts. |
| Q larger/chain | Fresh elaboration of additive root glue under exact model/registry/buffer joins; validate static/no-tail/canonical assumptions and distinct endpoint/composition roots | All new queriesUNCOMPILED. Larger import closures still contain existing derived-instance/geometry/model source with historical failure notes; no source-only repair inferred. |
| W whole/entries | Exact EntryDomain/partial scalar-literal/callgraph and nonvacuity review, complete actual source/final attribution, per-case root/model policy and valid-negative classification | UNCOMPILED. Whole kernel-entry and entries-domain successor are different claims; no blanket error/resource/divergence/native/numeric equivalence. |
| Root source release | Choose exact coherent host/producer prerequisite and original compiler-import policy for an eventual release | No tool_registry/invocation/admission policy replacement, original profile/hash refresh or primary integration here. Existing9d7 canonical+primary admission is not reapplied. |
| Root real consumer | Supply preconstructed live binding, selected service, exact context workspace and real consume-IR callback/import custody scope; identify other compiler routes | OPEN. The host-consumer-bridge-v1 prototype gates ONLY its chosen in-process callback. JSON tool output, ordinary source-import scopes and unrelated compile/child routes do not guarantee ALL ordinary compilation is gated. |
| Human/reviewer | Exact statements, parser/projection/primitive/external/canonical/allocator/callgraph/native correspondence, policy and paper claim approval | NOT_RECORDED. Source text and byte integrity grant no semantic/human approval. |
| Future successor owner | Supply a named immutable schedule or other additional cut, source/query/root manifest and domain | UNKNOWN_OPTIONAL; append a new separately named successor, preserve this packet and old W whole plan. |

No new context/toolchain/runtime API is assumed available because Python source exists. Source/header association, fresh source builds, actual kernel root audits, live program authority and global consumer enforcement are separate obligations.
''')
(OUT/'RUNBOOK.md').write_text('''# Portable source-plan reproduction and review

The completed local batch used only Python stdlib planning code. It read SOURCE/QUERY/METADATA refs as data, with no copied provider/controller/Lean execution. Actual validation matched487 files;10 tiny controls and491 same-file relocation associations passed. No old Q/W test suite or archive campaign ran. All compile/audit/qualification/replay work isUNEXECUTED.

For a transferred packet, independently review/pin INPUTS.json, PLAN.json and the two utility sources, then arrange its payload relative paths. The plain source-only commands are:

```sh
python3 -I -S -B build_plan.py --packet /chosen/packet --output rebuilt-plan.json
python3 -I -S -B verify_plan.py --packet /chosen/packet --plan rebuilt-plan.json
```

These example command lines areUNEXECUTED here; the validated code paths ran on the same current utility bytes in validate_data_batch.py. The relocation check ran the relocated verifier entry point from an unrelated CWD with its own copied builder module. It used hard links to identical files to avoid a large duplicate payload; it does not require hard links on another host.

Builder output is a source/data plan. A verifier integrity match can coexist with a clearlyUNAVAILABLE profile/case if input metadata/source is missing; inspect unavailable lists and status fields, not only an exit code. This actual cut has0 unavailable selected profiles/cases. A changed source/query pin, conflicting module, missing source/file, cycle, unknown header, support-root swap or plan-root swap is explicitly observed in BOUNDED_CONTROLS.json. Controls are new tiny DATA fixtures and do not count as semantic tests.

The independent closed header decoder handles UTF8, nested/line comments, optional prelude, one literal ASCII dotted module import per line and explicit recognized body starts. Unknown/ambiguous headers refuse rather than guess. It is a conservative header decoder, not a Lean body parser, theorem checker or actual imported-object report. Default Init and explicit Init.Data.List.Lemmas/Lean.Elab.Tactic.Omega are external NAMES ONLY. No external Std/SDK source or object is read. Final declaration excerpts are descriptive source text; the exact kernel name/type/axiom audits remain required.

PROFILE_INDEX.json links six separate plans. BASELINE_PLAN_V1.json and plans/W-whole-v1.json stay preserved; W_ENTRY_SUCCESSOR_PLAN_V1.json names the fixed entries successor separately. Every model/domain/root roster is retained independently. Source union compilation does not establish a union theorem. The profile/case output directories are requests keyed by source-selection digest, not existing old workspace authority or permission to create a runtime context.

Read BUILD_AND_AUDIT_ROSTER.md before any later build/audit/qualification design. Root alone publishes/integrates; F alone handles runtime investigation. A NEW independently reviewed official SDK context is a separate missing input. Never reuse old absolute pins, old project objects, a basic Init success, historical failures or saved reports as fresh proof authority. No native command, --deps, guard, loader, installation or network action is authorized by this runbook.

Known exclusions:4 optional historical stdout/stderr text artifacts were not read because the assembly allowlist selected source/query/metadata extensions; their supplied pins remain in the fixed manifest/source inventory. They are not missing required DAG inputs and establish no new failure. Raw actual buffers for all selected cases are retained when bound in pinned report metadata; source-derived models/declarations remain conditional. Optional external API/portability and unspecified schedule comparisons remain null/UNKNOWN.
''')
(OUT/'HANDOFF.md').write_text('''# Completed fresh graph source-build plan v1

All six ordered source/data tasks are complete, including the available fixed W entry-roster successor. No native/runtime dependency blocked this source-planning batch.

Six separate exact plans bind Q service7 modules/17 roots, chain7/48, scalar-SCF24/57, canonical integer30/82; W whole106/215 and separately named W entries111/249.26 distinct query byte sequences comprise6 declared positive candidates and20 expected binding refusals, all outcomesUNOBSERVED and accepted/qualified cases0. Exact source/query/final+support roots, actual buffers, domains/assumptions, topological orders and distinct output requests are retained. Source-union compilation implies no combined semantic theorem.

Stdlib-only build_plan.py and verify_plan.py validate original expected pins against current packet bytes, closed literal headers, module identity/conflicts, missing dependencies/cycles, exact query/root metadata, semantic metadata associations and deterministic plan fields. The bounded data batch matched487 source/data files, passed10 tiny controls and491 same-file unrelated-CWD associations. No provider/controller/copied Python/Lean/archive content was executed. Tool lint passes; initial style failure retained.

Old W whole and baseline plans remain separate from the fixed entries successor. Unspecified schedule successor staysUNKNOWN, with a requirement for a new named fixed plan. Shared selected Lean/core-Python source byte conflicts0; this is not an integration approval. Primary baseline9d7 canonical utility+Fprimary25 is recorded and not reapplied. Old host-consumer/service/canonical/external packets remain untouched.

Start with PROFILE_ROSTER.md, PROFILE_INDEX.json, plans/, FRESH_QUALIFICATION_ROSTER.json and BUILD_AND_AUDIT_ROSTER.md. SOURCE_DATA_INVENTORY.json links exact fixed manifests and expected/current source-data associations; QUERY_CONTRACT_DATA.json and original metadata preserve query contract evidence. SOURCE_TOOL_INVENTORY.json pins the tested utilities and input/plan. TRANSFER_INDEX.json and FILE_INDEX.json describe the deduplicated packet and retained controls/relocation.

Remaining work is explicitly outside this completed batch: NEW official-SDK F context review/authorization, fresh main/support source builds and per-query kernel compile, exact imported-source/object and axiom/type audits, positive/valid-negative qualification, fresh replay and human review. Every such stepUNEXECUTED; new candidatesUNCOMPILED; runtime causeUNKNOWN, old profilesHELD. No SDK/runtime/archive-image reads, guards/loader/native/compiler/--deps/network/install, numerics, new agents/models, primary/protected/frozen/lock edits or publication occurred. Concrete host-consumer global bypass decisions remainOPEN; no blanket ordinary-compilation guarantee.
''')
print(json.dumps({'reports':'written','profiles':len(plan['profiles']),'query_contract_rows':len(contracts),'native_authority':False}))
