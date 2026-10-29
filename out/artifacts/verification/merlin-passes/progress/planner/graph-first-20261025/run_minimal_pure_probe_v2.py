"""Offline repair of v1 trace parsing, then FIRST pure proof invocation.

V1 --deps returned zero; its raw trace is preserved and never recollected.
The parser now accounts explicitly for the main object's empty fini name and
namespace suffix. No unknown library/path is ignored or newly admitted.
"""
import hashlib,json,subprocess
from pathlib import Path
repo=Path('/home/jeremy/Projects/merlin');here=repo/'out/artifacts/verification/merlin-passes/progress/planner/graph-first-20261025'
old=here/'minimal-pure-probe-v1';out=here/'minimal-pure-probe-v2';assert not out.exists();out.mkdir()
p=json.loads((here/'pure-runtime-provenance-v1.json').read_bytes())
sdk=json.loads(Path(p['original_F_SDK_manifest']['path']).read_bytes())
refs={r['path']:r for r in [*sdk['files'].values(),*p['original_runtime_refs']]}
lean='/home/jeremy/.elan/toolchains/leanprover--lean4---v4.35.0/bin/lean'
runtime={r['path']:r for r in p['original_runtime_refs']};by_name={Path(k).name:k for k in runtime}
assert len(by_name)==len(runtime)

def pin(ref):
 path=Path(ref['path']);a=path.stat()
 with path.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
 b=path.stat();assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns,b.st_ctime_ns)
 assert a.st_size==ref['bytes'] and d==ref['sha256'],str(path)

def trace(text):
 init=[];fini=[];main=[];transfer=[];maps=[];empty_fini=0
 for line in text.splitlines():
  if 'dynamically loaded by' in line:raise RuntimeError('unexpected dynamic loading: '+line)
  if 'generating link map' in line:
   value=line.split('file=',1)[1].split(' [',1)[0].strip()
   assert '[0]' in line,line
   path=str(Path(value).resolve(strict=True)) if value.startswith('/') else by_name.get(value)
   assert path in runtime,line;maps.append(path)
  for marker,dest in [('calling init: ',init),('calling fini: ',fini),('initialize program: ',main),('transferring control: ',transfer)]:
   if marker not in line:continue
   value=line.split(marker,1)[1].strip();tokens=value.split()
   if tokens==['[0]']:
    assert marker=='calling fini: ',line;empty_fini+=1;continue
   assert len(tokens) in [1,2] and tokens[0].startswith('/'),line
   if len(tokens)==2:assert tokens[1]=='[0]',line
   path=str(Path(tokens[0]).resolve(strict=True));assert path in runtime,line;dest.append(path)
 assert main==[lean] and transfer==[lean] and empty_fini==1
 assert set(init)==set(fini)==set(runtime)-{lean}
 assert set(maps)==set(runtime)-{'/usr/lib/ld-linux-x86-64.so.2'}
 return {'init_files':sorted(set(init)),'fini_files':sorted(set(fini)),'link_maps':sorted(set(maps)),
         'main_initialized_and_transferred':lean,'empty_main_fini_namespace_zero':empty_fini,'unbound_libraries':[]}

for key in ['original_F_SDK_manifest','original_W_fixed_context']:pin(p[key])
for ref in refs.values():pin(ref)
assert not Path('/etc/ld.so.preload').is_symlink() and not Path('/etc/ld.so.preload').exists()
old_process=json.loads((old/'imports.process.json').read_bytes());assert old_process['returncode']==0
imports=[str(Path(s.strip()).resolve(strict=True)) for s in (old/'imports.stdout').read_text().splitlines() if s.strip()]
assert imports and all(s in refs for s in imports)
previous_trace=trace((old/'imports.stderr').read_text())
(out/'OFFLINE-REPAIR.json').write_text(json.dumps({'v1_native_command_returncode':0,'known_parser_error':'main fini is [0] and path fini lines carry [0] suffix','v1_raw_buffers_preserved':True,'imports_not_rerun':True,'recovered_trace':previous_trace},indent=2)+'\n')
intent=json.loads((old/'INTENT.json').read_bytes());command=list(intent['commands'][1]);command[command.index('-o')+1]=str(out/'PureProbe.olean')
source=old/'PureProbe.lean';expected=source.read_bytes()
process=subprocess.run(command,cwd=old,env=intent['environment'],capture_output=True,text=True,timeout=30)
(out/'proof.stdout').write_text(process.stdout);(out/'proof.stderr').write_text(process.stderr)
(out/'PROCESS.json').write_text(json.dumps({'command':command,'environment':intent['environment'],'cwd':str(old),'returncode':process.returncode},indent=2)+'\n')
assert process.returncode==0,'first proof invocation failed; retain buffers, no retry'
assert 'does not depend on any axioms' in process.stdout,process.stdout
observed=trace(process.stderr)
for ref in refs.values():pin(ref)
assert source.read_bytes()==expected
assert not Path('/etc/ld.so.preload').is_symlink() and not Path('/etc/ld.so.preload').exists()
summary={'status':'MINIMAL_PURE_PROBE_CHECKED','selected_original_refs_checked_pre_post':len(refs),'all_original_pins_unchanged':True,
 'actual_reported_import_paths':imports,'original_metadata_closure_module_count':sdk['module_count'],
 'observed_import_startup':previous_trace,'observed_proof_startup':observed,
 'root_theorem':'MerlinCoordinatorPureProbe.checked_kernel','root_axioms':[],
 'v1_diagnostic_trace_parse_error_retained':True,'failed_oracle_profile_rerun':False,'runtime_expected_hashes_refreshed':False,
 'formal_OS_loader_denial_proved':False,'arbitrary_future_late_loads_ruled_out':False,'graph_checker_qualification':False,
 'captured_graph_sources_compiled_here':False,'trust':['OS/kernel/vDSO','pinned loader diagnostics','pinned Lean module reporting/kernel','original SDK metadata and host transport']}
(out/'OBSERVATION.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'status':summary['status'],'original_refs':len(refs),'module_count':sdk['module_count'],'root_axioms':[],'old_failed_profile_rerun':False},indent=2))
