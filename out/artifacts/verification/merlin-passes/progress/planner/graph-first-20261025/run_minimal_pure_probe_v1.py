"""One minimal original-bound Init diagnostic, independent of held oracle profiles.

OS/kernel, loader diagnostics and Lean module reporting remain explicit trust.
This does not admit a graph checker or explain an earlier file mismatch.
"""
import hashlib,json,subprocess
from pathlib import Path

repo=Path('/home/jeremy/Projects/merlin')
here=repo/'out/artifacts/verification/merlin-passes/progress/planner/graph-first-20261025'
out=here/'minimal-pure-probe-v1';assert not out.exists();out.mkdir()
p=json.loads((here/'pure-runtime-provenance-v1.json').read_bytes())
startup=json.loads((here/'pure-loader-diagnostic-v1/OBSERVATION.json').read_bytes())
assert startup['status']=='STARTUP_PATHS_MATCH_ORIGINAL_PINS'

def check(ref):
 path=Path(ref['path']);before=path.stat()
 with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
 after=path.stat()
 assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),str(path)
 assert before.st_size==ref['bytes'] and sha==ref['sha256'],str(path)

for key in ['original_F_SDK_manifest','original_W_fixed_context']:check(p[key])
sdk=json.loads(Path(p['original_F_SDK_manifest']['path']).read_bytes())
refs={r['path']:r for r in [*sdk['files'].values(),*p['original_runtime_refs']]}
assert all('Std/Time/Format/Basic' not in x for x in refs)
for r in refs.values():check(r)
try:Path('/etc/ld.so.preload').lstat()
except FileNotFoundError:pass
else:raise RuntimeError('preload selector present')
source='''prelude
import Init
namespace MerlinCoordinatorPureProbe
theorem checked_kernel : (1 : Nat) = 1 := rfl
#print axioms checked_kernel
end MerlinCoordinatorPureProbe
'''
query=out/'PureProbe.lean';query.write_text(source)
sysroot='/home/jeremy/.elan/toolchains/leanprover--lean4---v4.35.0'
loader='/usr/lib/ld-linux-x86-64.so.2';lean=sysroot+'/bin/lean'
base=[loader,'--inhibit-cache','--glibc-hwcaps-mask','','--library-path',sysroot+'/lib:'+sysroot+'/lib/lean:/usr/lib',lean]
env={'LANG':'C','LC_ALL':'C','LEAN_SYSROOT':sysroot,'LEAN_PATH':sysroot+'/lib/lean','LD_DEBUG':'libs,files'}
(out/'INTENT.json').write_text(json.dumps({'commands':[base+['--deps',str(query)],base+['-j1','-o',str(out/'PureProbe.olean'),str(query)]],
 'environment':env,'cwd':str(out),'original_ref_count':len(refs),'scope':'minimal Init-only diagnostic, not graph/oracle qualification',
 'LD_DEBUG':'exact fixed diagnostic setting, not inherited user input','old_context_modified':False},indent=2)+'\n')
results=[]
for label,args in [('imports',['--deps',str(query)]),('proof',['-j1','-o',str(out/'PureProbe.olean'),str(query)])]:
 process=subprocess.run(base+args,cwd=out,env=env,capture_output=True,text=True,timeout=30)
 (out/(label+'.stdout')).write_text(process.stdout);(out/(label+'.stderr')).write_text(process.stderr)
 (out/(label+'.process.json')).write_text(json.dumps({'command':base+args,'environment':env,'returncode':process.returncode},indent=2)+'\n')
 assert process.returncode==0,label+' diagnostic failed: preserve first buffers; no retry'
 loaded=[]
 for line in process.stderr.splitlines():
  for marker in ['calling init: ','calling fini: ']:
   if marker in line:
    value=line.split(marker,1)[1].strip()
    assert value.startswith('/'),value
    name=str(Path(value).resolve(strict=True));assert name in refs,'unbound loaded library: '+name
    loaded.append(name)
 if label=='imports':
  imports=[str(Path(line.strip()).resolve(strict=True)) for line in process.stdout.splitlines() if line.strip()]
  assert imports and all(name in refs for name in imports),imports
 else:
  assert 'does not depend on any axioms' in process.stdout,process.stdout
 results.append({'phase':label,'returncode':0,'observed_init_fini_libraries':sorted(set(loaded))})
for r in refs.values():check(r)
try:Path('/etc/ld.so.preload').lstat()
except FileNotFoundError:pass
else:raise RuntimeError('preload selector changed')
assert query.read_text()==source
record={'status':'MINIMAL_PURE_PROBE_CHECKED','selected_original_ref_count':len(refs),'all_original_pins_unchanged_pre_post':True,
 'actual_reported_import_paths':imports,'library_observations':results,
 'raw_complete_loader_debug_retained':True,'root_theorem_axioms':[],
 'formal_OS_loader_denial_proved':False,'arbitrary_future_late_loads_ruled_out':False,
 'graph_context_or_new_capture_compilation_authorized':False,'held_oracle_or_failed_profile_rerun':False,
 'old_failure_explained_or_repaired':False,'new_expected_runtime_hashes_created':False,
 'trust':['OS/kernel/vDSO','pinned loader debug and module-dependency reporting','pinned Lean kernel/runtime','host observation/transport']}
(out/'OBSERVATION.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
