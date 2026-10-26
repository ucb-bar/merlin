"""Read-only consistency checks, independent of compiler/projector imports.

This checks the audit's facts, references and bytes. It is not an IR semantics
checker, proof checker, or replacement for the later correspondence audit.
"""
import ast
import collections
import csv
import datetime
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

OUT=Path(__file__).resolve().parent
INDEX=json.loads((OUT/'source-index.json').read_text())
ROOTS={'producer':Path(INDEX['producer']), 'verification':Path(INDEX['verification'])}
OLD=OUT.parent/'registry-eight'
errors=[]
checks={}
def check(name,condition,detail=None):
    checks[name]={'passed':bool(condition),'detail':detail}
    if not condition: errors.append(name)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load_csv(name): return list(csv.DictReader((OUT/name).open()))
rows=load_csv('coverage.csv'); inventory=load_csv('source-inventory.csv')
check('unique_nonempty_rows',len({r['row_id'] for r in rows})==len(rows) and all(all(r.values()) for r in rows),len(rows))
producer=ROOTS['producer']; verify=ROOTS['verification']
tree=ast.parse((producer/'src/merlin/transforms/certificates.py').read_text())
assign=next(n for n in tree.body if isinstance(n,(ast.Assign,ast.AnnAssign)) and any(isinstance(t,ast.Name) and t.id=='TRANSFORM_SCOPES' for t in (n.targets if isinstance(n,ast.Assign) else [n.target])))
registry=ast.literal_eval(assign.value)
pipeline=ast.parse((producer/'src/merlin/transforms/pipeline.py').read_text())
calls=[{'call':ast.unparse(n.func),'kind':n.args[0].value,'line':n.lineno} for n in ast.walk(pipeline) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('register','transform') and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str)]
common=[r for r in rows if r['inventory_category']=='registered_common_transformation']
check('registry_matches_actual_dispatch_and_rows',len(registry)==8 and set(registry)=={r['kind'] for r in common}=={c['kind'] for c in calls if c['kind'] in registry},sorted(registry))
excluded={c['kind'] for c in calls}-set(registry)
check('excluded_common_analysis_stages',excluded=={'storage_analysis','dependency_analysis','final_analyses','final_verification'},sorted(excluded))
check('dependency_schedule_actual_apply_order_present',any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='apply_order' for n in ast.walk(pipeline)))
policy=(producer/'src/merlin/transforms/pipeline_policy.py').read_text()
policy_calls={ast.unparse(n.func) for n in ast.walk(ast.parse(policy)) if isinstance(n,ast.Call)}
check('policy_dispatches_registered_actions_without_own_proof_authority',{'session.run','entry.action','registry_metadata','validate_pass_order'} <= policy_calls and 'supplies none of their proof' in policy, 'Caller-provided RegisteredPass mapping; no hardcoded call to apply_common_pipeline')
schedule=ast.parse((producer/'src/merlin/compile/scheduling/schedule_pipeline.py').read_text())
stages=[]
for node in ast.walk(schedule):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='PassSpec':
        key=node.args[0].value if isinstance(node.args[0],ast.Constant) else ast.unparse(node.args[0])
        mut=next((ast.literal_eval(k.value) for k in node.keywords if k.arg=='mutates_ir'),True)
        stages.append({'stage':key,'mutates_session_ir':mut,'line':node.lineno})
check('schedule_stage_roster', {s['stage'] for s in stages}=={'semantic_admission','storage','completion','emission','final_program','fusion','configuration','prefetch','post_storage.name'},stages)
check('candidate_stages_not_miscounted_by_mutation_flag',all(not s['mutates_session_ir'] for s in stages if s['stage'] in {'fusion','configuration','prefetch'}))
doc=(producer/'docs/reference/shared_compiler_passes.md').read_text()
docrefs=set(re.findall(r'src/merlin/[A-Za-z0-9_/]+\.py',doc))
check('all_document_code_refs_accounted',docrefs==set(INDEX['code_refs']) and docrefs <= {f['path'] for f in INDEX['source_files']},len(docrefs))
scope={str(p.relative_to(producer)) for folder in ('src/merlin/transforms','src/merlin/compile/scheduling') for p in (producer/folder).glob('*.py')}
check('all_direct_transform_and_scheduling_modules_accounted',scope <= {f['path'] for f in INDEX['source_files']},len(scope))
check('source_inventory_exactly_matches_index',len(inventory)==len(INDEX['source_files']) and {r['source'].split(':')[1] for r in inventory}=={f['path'] for f in INDEX['source_files']},len(inventory))
sections=[];fenced=False
for n,line in enumerate(doc.splitlines(),1):
    if line.startswith('```'):fenced=not fenced
    if not fenced and re.match(r'^#{1,6} ',line):sections.append({'line':n,'title':line.lstrip('# ').strip()})
check('document_sections_reconciled',sections==INDEX['document_sections'] and len(load_csv('document-reconciliation.csv'))==len(sections),len(sections))
hash_rows=list(INDEX['source_files'])
bad_hashes=[f['path'] for f in hash_rows if not (producer/f['path']).is_file() or sha(producer/f['path'])!=f['sha256']]
old_refs=json.loads((OLD/'reviewed-source-identities.json').read_text())['files']
refs=json.loads((OUT/'reviewed-references.json').read_text())
for f in old_refs+refs:
    base=ROOTS[f.get('root',f.get('source'))]; path=base/f['path']
    if not path.is_file() or sha(path)!=f['sha256']:bad_hashes.append(str(path))
check('reviewed_source_and_proof_reference_hashes',not bad_hashes,bad_hashes)
# All typed source locations in the matrix must resolve; no import of the audited projection.
bad_locations=[];location_count=0
for row in rows:
    for value in row.values():
        for root,path,line,symbol in re.findall(r'(producer|verification):([\w/.-]+):(\d+)(?:#([\w.]+))?',value):
            actual=ROOTS[root]/path;location_count+=1
            if not actual.is_file() or not 1<=int(line)<=len(actual.read_text().splitlines()):bad_locations.append((row['row_id'],root,path,line))
check('matrix_source_locations_exist',not bad_locations,{'locations':location_count,'invalid':bad_locations})
# Preserve previous work; all previously manifested bytes are rechecked.
old_manifest=json.loads((OLD/'audit-manifest.json').read_text())
old_bad=[f['path'] for f in old_manifest['files'] if not (OLD/f['path']).is_file() or sha(OLD/f['path'])!=f['sha256'] or (OLD/f['path']).stat().st_size!=f['bytes']]
check('registry_eight_original_manifest_intact',not old_bad,{'entries':len(old_manifest['files']),'invalid':old_bad})
captures=[];bad=[]
for row in common:
    case=OLD/'producer-examples'/row['row_id'];capture=json.loads((case/'capture.json').read_text())
    ok=capture['status']=='actual_changed_producer_output' and capture['kind']==row['kind'] and not capture['lean_run'] and not capture['qualified'] and (case/'before.mlir').read_bytes()!=(case/'after.mlir').read_bytes()
    for f in capture['files']:
        path=case/f['path'];ok &= sha(path)==f['sha256'] and path.stat().st_size==f['bytes']
    captures.append(row['row_id'])
    if not ok:bad.append(row['row_id'])
check('all_registry_rows_have_changed_unqualified_producer_bytes',not bad,{'count':len(captures),'invalid':bad})
probes=json.loads((OLD/'projection-probes.json').read_text())
bad_probes=[]
for p in probes['cases']:
    if p['lean_run'] or p['qualification_issued'] or (p.get('model') and sha(OLD/p['model'])!=p['sha256']):bad_probes.append(p['case'])
check('projection_probes_not_promoted_to_proofs',not bad_probes,dict(collections.Counter(p['status'] for p in probes['cases'])))
packet=verify/'out/artifacts/verification/merlin-passes/progress/planner/graph-first-20261025/worked-example'
manifest=json.loads((packet/'MANIFEST.json').read_text())
bad_packet=[f['path'] for f in manifest['files'] if not (packet/f['path']).is_file() or sha(packet/f['path'])!=f['sha256'] or (packet/f['path']).stat().st_size!=f['bytes']]
check('portable_worked_chain_copy_integrity',not bad_packet,{'entries':len(manifest['files']),'invalid':bad_packet,'fresh_historical_replay':False})
lookup={r['row_id']:r for r in rows}
elision=(verify/'out/artifacts/verification/graph-state-elision/active/README.md').read_text()
check('later_elision_failure_controls_status','Qualified originals/replays=0' in elision and lookup['config-elide']['qualification_status']=='unqualified_SIGSEGV_zero_qualified_originals_or_replays')
check('region_plan_is_not_claimed_as_changed_MLIR','no source MLIR rewrite' in lookup['region-plan']['whole_boundary_status'])
check('geometry_not_claimed_as_program_equivalence',lookup['ssa-allocation']['whole_boundary_status']=='geometry_only_not_complete_program_semantics')
check('support_qualification_composition_separate',all(r['support_status'] and r['qualification_status'] and r['composition_status'] for r in rows))
tests=OUT/'behavioral-tests'
current=json.loads((tests/'suite.json').read_text())
suite=ET.parse(tests/'suite.xml').getroot().findall('testsuite')
current_stats={k:sum(int(s.attrib.get(k,0)) for s in suite) for k in ('tests','failures','errors','skipped')}
current_passed=current['returncode']==0 and current_stats==current['results'] and current_stats=={'tests':530,'failures':0,'errors':0,'skipped':0} and current['verification_revision']==INDEX['verification_revision']
check('configured_behavioral_suite',current_passed,current_stats)
tools=json.loads((tests/'toolchain.json').read_text())
tool_ok=all(Path(f['path']).is_file() and sha(Path(f['path']))==f['sha256'] for f in tools['files'])
check('native_toolchain_identities',tool_ok and tools['llvm_revision']=='a47bddccec30255619bb8c37fa59700e661d4e66' and tools['downloads'] is False and tools['settings']=={k:current['environment'][k] for k in tools['settings']},'Exact locally built pinned LLVM23; original Python compiler driver; no guard changes')
evidence=json.loads((tests/'native-evidence.json').read_text())
native_ok=all((tests/f['copy']).is_file() and sha(tests/f['copy'])==f['sha256'] and (tests/f['copy']).stat().st_size==f['bytes'] for f in evidence['files'])
pair=json.loads((tests/evidence['copy_pair']).read_text())
owners=pair['owners']+[f for arm in pair['arms'].values() for field in ('calls','artifacts') for f in arm[field]]
native_ok &= all(Path(f['path']).is_file() and sha(Path(f['path']))==f['sha256'] for f in owners)
check('native_changed_pair_products_and_owners',native_ok and evidence['baseline_and_candidate_differ'] and evidence['calls_per_arm']=={'baseline':3,'candidate':3},{'copied_files':len(evidence['files']),'owner_and_product_hashes':len(owners),'Lean_qualification':False})
# Sealed graph packets: rows cite them without promoting review, gating or fresh examples.
seals={'shape':'graph-shape-normalization/standalone-mode-v1/SEAL.json','memory_scf':'graph-memory-scf/memory-scf-mode-v1/SEAL.json',
    'declared':'graph-regions-declared/qualification-v1/STATUS.json','declared_bound':'graph-regions-declared/bound-qualification-v1/STATUS.json'}
seal={k:json.loads((verify/'out/artifacts/verification'/v).read_text()) for k,v in seals.items()}
check('sealed_packets_disclaim_review_and_gating',seal['shape']['review_complete'] is False and seal['shape']['ordinary_compiler_gating'] is False and seal['shape']['adapter_proved'] is False
    and seal['declared']['engineering_qualified'] is True and seal['declared']['ordinary_compiler_gating'] is False and seal['declared_bound']['engineering_qualified'] is False,
    {k:{f:v.get(f) for f in ('review_complete','ordinary_compiler_gating','engineering_qualified','adapter_proved') if f in v} for k,v in seal.items()})
recount={}
for name,entry in json.loads((OUT/'sealed-packet-presence.json').read_text()).items():
    data=json.loads((verify/entry['manifest'].removeprefix('verification:')).read_text())
    files=data['files'] if isinstance(data,dict) else data
    files={f['path']:f for f in (files.values() if isinstance(files,dict) else files)}.values()
    local=[verify/f['path'].removeprefix('/home/jeremy/Projects/merlin/') for f in files]
    recount[name]=sum(1 for f,p in zip(files,local) if p.is_file() and sha(p)==f['sha256'])==entry['matched'] and len(local)==entry['entries']
check('sealed_packet_local_presence_recounted',all(recount.values()),recount)
sealed_rows=[r for r in rows if r['kind'] in ('shape_view_normalization','exact_integer_simplification')]
check('shape_and_integer_cite_sealed_route_not_review',len(sealed_rows)==10 and all(r['support_status'].startswith('conditional_subset_only') and 'D1-G01' not in r['gap_ids'].split(';') and 'D1-G09' in r['gap_ids'].split(';')
    and 'not one of its originals' in r['qualification_status'] and r['independent_review'].startswith('source/route review') for r in sealed_rows))
sealed_probes=json.loads((OUT/'projection-probes-sealed.json').read_text())
check('sealed_route_probes_not_promoted_to_proofs',sealed_probes['verification_revision']==INDEX['verification_revision'] and {c['case'] for c in sealed_probes['cases']}=={r['row_id'] for r in sealed_rows}
    and all(not c['lean_run'] and not c['qualification_issued'] and sha(OUT/c['model'])==c['sha256'] for c in sealed_probes['cases'] if c.get('model')),
    dict(collections.Counter(c['status'] for c in sealed_probes['cases'])))
# D1-G09: the live common-pipeline gate still admits shape changes on a statement that names no program.
checker=(producer/'src/merlin/verify/lean_checker.py').read_text()
domains=next(n for n in ast.parse(checker).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_TRANSFORM_DOMAINS' for t in n.targets))
check('live_gate_admits_shape_on_identity_statement','statements.append("∀ (α : Type) (value : α), id value = value")' in checker and 'shape_view_normalization' in {k.value for k in domains.value.keys}
    and 'accepted_pipeline_receipt' in (producer/'src/merlin/transforms/pipeline.py').read_text(),'producer lean_checker.py shape branch; D1-G09')
for name,root in ROOTS.items():
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root,text=True)
    check(name+'_pin_and_clean_tracked_tree',head==INDEX[name+'_revision'] and not status,{'head':head,'tracked_changes':status})
result={'schema':'graham.deliverable-1.audit.v1','checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':INDEX['scope'],
    'passed':not errors,'checks':checks,'errors':errors,'rows_by_category':dict(collections.Counter(r['inventory_category'] for r in rows)),
    'test_results':current_stats,'test_run':'behavioral-tests/suite.json','all_tests_passed':current_passed,
    'new_lean_qualification':False,'independent_semantic_projection_proof':False,'human_review_approval':False,
    'limits':['Missing raw qualification bundles remain missing; reports are not original-bound replay','No all-model, all-target, or transitive whole-repository coverage claim','Test totals and matrix row totals are not proof totals']}
(OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':not errors,'checks':len(checks),'errors':errors,'rows':len(rows)},indent=2))
raise SystemExit(bool(errors))
