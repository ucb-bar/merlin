"""One bounded stdlib source/ref/root/claim cross-check. No project imports,
Q/W tests, parsers, producer/native runs, SDK/runtime/image/object reads.
"""
from pathlib import Path
import json,hashlib
B=Path(__file__).resolve().parent;checks=[]
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ck(n,v):checks.append({'check':n,'ok':bool(v)})
def pin(n,p,r):
 ck(n+':exists',p.is_file())
 if p.is_file():
  ck(n+':sha',sha(p)==r['sha256'])
  size=r.get('bytes',r.get('size'))
  if size is not None:ck(n+':bytes',p.stat().st_size==size)
for r in load(B/'SELECTION.json')['files']:pin(r['snapshot'],B/r['snapshot'],r)
for r in load(B/'F_SELECTION.json')['files']:pin(r['snapshot'],B/r['snapshot'],r)
for cut in ['q-schedule','q-larger']:
 for r in load(B/cut/'MANIFEST.json')['rows']:pin(cut+':seal:'+r['path'],B/cut/r['path'],r)
 inv=load(B/cut/'SOURCE-INVENTORY.json')
 for r in inv['owned']+inv.get('dependencies',inv.get('source_dependencies',[])):pin(cut+':source:'+r['path'],B/cut/r['snapshot'],r)
for r in load(B/'w-source/SOURCE-SEAL.json')['files']:pin('Wseal:'+r['path'],B/'w-source'/r['path'],r)
for cut in ['w-source']:
 for r in load(B/cut/'OWNED-SOURCES.json'):
  p=B/cut/'source'/r['path'];pin('Wowned:'+r['path'],p,r)
f=B/'f-deadline';ck('F supplied manifest',sha(f/'MANIFEST.json')==load(B/'F_SELECTION.json')['manifest_sha256']);ck('F supplied review',sha(f/'REVIEW-REQUEST.json')==load(B/'F_SELECTION.json')['review_sha256'])
for r in load(f/'MANIFEST.json')['files']:pin('Fseal:'+Path(r['snapshot']).name,f/Path(r['snapshot']).name,r)
review=load(f/'REVIEW-REQUEST.json');request=load(f/'REQUEST.json');pin('Frequest',f/'REQUEST.json',review['request']);ck('Fproposed_only',review['status']=='PROPOSED_NOT_AUTHORIZED'and all(review[k]is False for k in ['native_authorized','install_authorized','old_holds_cleared']));ck('F source7',len(review['source_refs'])==7 and {Path(r['path']).name for r in review['source_refs']}=={'run_intake.py','intake.py','policy.py','fdio.py','http_fetch.py','zip_intake.py','outer_deadline.py'})
for r in review['source_refs']:pin('F runner source:'+Path(r['path']).name,f/Path(r['path']).name,r)
ck('Foutertimer policy',request['outer_deadline']=='POSIX_ITIMER_REAL_MAIN_THREAD_NO_EXISTING_TIMER_V1');ck('Foneattempt/noauthority',request['attempts']==1 and not any(request['authority'].values()))
for r in load(B/'q-larger/HISTORICAL-TEXT.json')['rows']:pin('history:'+r['snapshot'],B/'q-larger'/r['snapshot'],r)
for fam in ['pipeline','integer']:
 p=B/'q-larger/final-examples'/fam;prepared=load(p/'PREPARED.json');receipt=load(B/'q-larger/historical-text-evidence'/fam/'0-receipt.json');raw=[p/f'stage{i}.mlir'for i in range(len(prepared['buffers']))];ck('history raw originals:'+fam,all(sha(p)==r['sha256']and p.stat().st_size==r['bytes']for p,r in zip(raw,receipt['originals'],strict=True)));report=load(B/'q-larger/historical-text-evidence'/fam/'1-report.json');ck('historical semantic qualification false retained:'+fam,report['qualified']is False)
for name in ['pure_reorder','cross_observer_keep_effect_order','unchanged_order']:
 p=B/'q-schedule/actual-examples'/name;d=load(p/'PREPARED.json');r=load(p/'REQUEST.json');projection=json.loads(d['projection']);matching=json.loads(d['matching']);ck('schedule raw '+name,[sha(p/'before.mlir'),sha(p/'after.mlir')]==[d['before'],d['after']]);ck('schedule query '+name,d['query']==(p/'Query.lean').read_text()and sha(p/'Query.lean')==r['query_sha256']);ck('schedule root/source counts '+name,len(d['roots'])==17 and len(r['project_sources'])==7 and [r['final_root'],*r['support_roots']]==[d['roots'][-1],*d['roots'][:-1]]);ck('schedule exact statement '+name,r['exact_statement'].strip()in d['query']);ck('schedule noauthority '+name,r['authority']is False and r['native_authorization']is False and d['query_status']=='UNCOMPILED'and d['outcome']=='UNOBSERVED');ck('schedule changed classification '+name,d['changed']==(name!='unchanged_order'));ck('schedule two defined entries '+name,[x['name']for x in projection['functions']if not x['declaration']]==['main','untouched']);ck('schedule complete DAG anchors '+name,matching['before']['semantic']==matching['after']['semantic']and matching['dependency_legality_is_execution_proof']is False);w=load(p/'WITNESS.json');ck('schedule witness source success not assumed '+name,w['source_success_assumed']is False and w['state']=='UNCOMPILED');ck('schedule33 witness roots '+name,len(w['roots'])==33 if 'roots'in w else True)
for fam,rc,sc in [('pipeline',57,24),('integer',82,30)]:
 p=B/'q-larger/final-examples'/fam;d=load(p/'PREPARED.json');r=load(p/'REQUEST.json');capture=json.loads(d['capture']);stages=([capture['prefix']]if capture['prefix']else[])+capture['pipeline']['stages'];ck('larger raw '+fam,[sha(p/f'stage{i}.mlir')for i in range(len(stages)+1)]==d['buffers']);ck('larger exact query '+fam,d['query']==(p/'Query.lean').read_text()and sha(p/'Query.lean')==r['query_sha256']);ck('larger exact statement '+fam,r['exact_final_statement'].strip()in d['query']);ck('larger counts '+fam,len(d['roots'])==rc and len(r['project_sources'])==sc);ck('larger roots '+fam,len(set(d['roots']))==rc and set([r['final_root'],*r['support_roots']])==set(d['roots']));ck('larger adjacent roots '+fam,len(d['boundary_roots'])==len(d['kinds'])and all(row[1]for row in d['boundary_roots']));ck('larger all stages same registry/type/header '+fam,all(all(stage[k]==stages[0][k]for k in ['registry','types','module_header'])for stage in stages));ck('larger complete roster '+fam,all([{k:f[k]for k in ['name','header','inputs','outputs','declaration']}for f in stage['functions']]==[{k:f[k]for k in ['name','header','inputs','outputs','declaration']}for f in stages[0]['functions']]for stage in stages));ck('larger all intermediate exact joins '+fam,all(all(x['after']==y['before']for x,y in zip(a['functions'],b['functions'],strict=True))for a,b in zip(stages,stages[1:])));ck('larger notactivated '+fam,r['authority']is False and r['native_authorization']is False and d['query_status']=='UNCOMPILED'and d['outcome']=='UNOBSERVED');ck('canonical boundary type '+fam,('PreservesEntriesCanonical'in r['exact_final_statement'])==(fam=='integer'))
for q in load(B/'w-source/QUERIES.json'):
 p=B/'w-source/evidence'/q['case'];d=load(p/'PREPARATION.json');r=d['prepared']
 for k,name in [('query','Entries.lean'),('raw_source','source.mlir'),('raw_final','final.mlir'),('report','PREPARATION.json')]:pin('Wcase:'+q['case']+':'+k,p/name,q[k])
 ck('W raw prepared '+q['case'],[sha(p/'source.mlir'),sha(p/'final.mlir')]==[r['raw_source_sha256'],r['raw_final_sha256']]);ck('W query prepared '+q['case'],sha(p/'Entries.lean')==r['query_sha256']);ck('W noacceptance '+q['case'],d['kernel_outcome']=='UNOBSERVED'and r['lean_status']=='UNCOMPILED'and not r['all_entries_proved']and not r['kernel_checked']and not r['qualified']and not d['semantic_counterexample_proved']);ck('W final root '+q['case'],q['final_root']=='GraphMemoryAfterEntries.every_definition'and 'theorem every_definition'in(p/'Entries.lean').read_text())
 if q['case']=='positive':
  er=r['entry_roster'];ck('W exact3 names',[h['name']for h in er['source_headers']]==['kernel','touch','untouched']and er['source_headers']==er['final_headers']);ck('W all defaultpublic/no declarations',all(h['visibility']=='<default-public>'and h['declaration']is False for h in er['source_headers']));ck('W domain witnesses/execution only entry2',r['domain_witnesses_emitted']and len(r['execution_witness_roots'])==5 and all(x.startswith('entry2_')for x in r['execution_witness_roots']));ck('W249 prospective roots',len(q['support_roots'])+1==249)
 if q['case']=='wrapper':ck('W caller context excluded',any(x['code']=='unproved_changed_kernel_caller_context'for x in r['entry_roster']['refusals']))
 if q['case']=='declaration':ck('W declaration excluded',any(x['code']=='declaration_not_supported'for x in r['entry_roster']['refusals']))
# Lexical source navigation only; no theorem elaboration/type/namespace certification.
root_navigation=[]
for row in load(B/'CLAIM_EVIDENCE_MAP.json'):
 for raw in row['raw_buffers']:pin('claim raw:'+raw['path'],B/raw['path'],raw)
 pin('claim query:'+row['query']['path'],B/row['query']['path'],row['query']);ck('claim final has source statement:'+row['family']+row['example'],row['final_statement']is not None)
 root_navigation.append({'example':row['family']+'/'+row['example'],'requested_roots':len(row['roots']),'distinct_roots':len(set(row['roots'])),'status':'PROSPECTIVE_NAMES_ONLY_NOT_A_KERNEL_AUDIT'})
div=load(B/'DIVSI_EVIDENCE.json');ck('divsi missing raw explicit',div['before_raw_sha256']is None and div['after_raw_sha256']is None and div['semantic_or_kernel_failure_observed']is False)
dead=load(B/'DEADLINE_RESULTS.json');ck('F own12 controls',dead['status']=='PASS'and dead['cases']==12 and not dead['real_network']and dead['actual_images']==0 and dead['native_launches']==0 and dead['blocked_actions']==[]);ck('F cleanup gap retained',load(B/'deadline-cleanup-evidence/RESULT.json')['status']=='CLEANUP_NOTES_LOST_FROM_DURABLE_FIRST_FAILURE')
for name in ['Q_REVIEW.md','W_REVIEW.md','HISTORICAL_BOUNDARY.md','HUMAN_WORKSHEET.md','F_DEADLINE_REVIEW.md','FINDINGS.json','CLAIM_EVIDENCE_MAP.json','W_CASES.json']:ck('deliverable:'+name,(B/name).exists())
failed=[r for r in checks if not r['ok']];report={'kind':'ONE_BOUNDED_STDLIB_SOURCE_REF_ROOT_CLAIM_CROSSCHECK','status':'PASS'if not failed else'FAIL','data_checks':len(checks),'failures':failed,'root_navigation':root_navigation,'all_new_graph_lean':'UNCOMPILED','graph_kernel_acceptance':'UNOBSERVED','native_or_producer_or_parser_runs':0,'results':checks};(B/'CROSS_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ['results','root_navigation']}))
