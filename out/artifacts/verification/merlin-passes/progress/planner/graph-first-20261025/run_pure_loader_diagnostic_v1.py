"""One original-pinned loader-list diagnostic; never runs a Lean query/oracle.

No proof-context admission or old-profile repair follows from startup listing.
"""
import hashlib,json,os,stat,subprocess
from pathlib import Path

repo=Path('/home/jeremy/Projects/merlin')
here=repo/'out/artifacts/verification/merlin-passes/progress/planner/graph-first-20261025'
out=here/'pure-loader-diagnostic-v1';assert not out.exists();out.mkdir()
provenance=json.loads((here/'pure-runtime-provenance-v1.json').read_bytes())

def checked(ref):
 p=Path(ref['path']);before=p.stat()
 with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
 after=p.stat()
 assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),str(p)
 assert p.resolve()==p and before.st_size==ref['bytes'] and digest==ref['sha256'],str(p)
 return {'path':str(p),'sha256':digest,'bytes':before.st_size}

def no_preload():
 try:Path('/etc/ld.so.preload').lstat()
 except FileNotFoundError:return
 raise RuntimeError('system preload selector present; diagnostic unavailable')

for key in ['original_F_SDK_manifest','original_W_fixed_context']:checked(provenance[key])
f=json.loads(Path(provenance['original_F_SDK_manifest']['path']).read_bytes())
w=json.loads(Path(provenance['original_W_fixed_context']['path']).read_bytes())
known={x['path']:x for x in [*f['files'].values(),*w['files'],*w['prior_dependencies']]}
refs=provenance['original_runtime_refs']
assert all(known.get(x['path'])==x for x in refs)
no_preload();before=[checked(x) for x in refs]
loader='/usr/lib/ld-linux-x86-64.so.2'
lean='/home/jeremy/.elan/toolchains/leanprover--lean4---v4.35.0/bin/lean'
assert loader in known and lean in known
assert not Path(loader).stat().st_mode & (stat.S_ISUID|stat.S_ISGID)
assert os.getuid()==os.geteuid() and os.getgid()==os.getegid()
argv=[loader,'--inhibit-cache','--glibc-hwcaps-mask','','--library-path',
      '/home/jeremy/.elan/toolchains/leanprover--lean4---v4.35.0/lib:/home/jeremy/.elan/toolchains/leanprover--lean4---v4.35.0/lib/lean:/usr/lib',
      '--list',lean]
env={'LANG':'C','LC_ALL':'C'}
intent={'command':argv,'environment':env,'cwd':str(repo),'original_refs':refs,
        'scope':'startup loader dependency listing only; no Lean source/query or MLIR oracle',
        'graph_or_pure_proof_execution_authorized':False}
(out/'INTENT.json').write_text(json.dumps(intent,indent=2)+'\n')
process=subprocess.run(argv,cwd=repo,env=env,capture_output=True,text=True,timeout=30)
(out/'stdout.txt').write_text(process.stdout);(out/'stderr.txt').write_text(process.stderr)
(out/'PROCESS.json').write_text(json.dumps({'returncode':process.returncode,'command':argv,'environment':env},indent=2)+'\n')
assert process.returncode==0,'Loader diagnostic failed; retain first buffers and do not retry'
no_preload();after=[checked(x) for x in refs];assert before==after
paths=[];virtual=[]
for line in process.stdout.splitlines():
 fields=line.split()
 if not fields:continue
 if fields[0]=='linux-vdso.so.1':virtual.append(line.strip());continue
 if '=>' in fields:
  value=fields[fields.index('=>')+1]
 else:value=fields[0]
 assert value.startswith('/'),'unresolved/unrecognized loader line: '+line
 path=str(Path(value).resolve(strict=True));assert path in {x['path'] for x in refs},path
 paths.append(path)
assert len(paths)==len(set(paths))
summary={'status':'STARTUP_PATHS_MATCH_ORIGINAL_PINS','mapped_file_paths':paths,
         'original_runtime_refs_checked':len(refs),'postcheck_unchanged':True,
         'virtual_kernel_objects':virtual,'system_preload_absent_pre_post':True,
         'environment_replaced_not_inherited':True,'cache_inhibited':True,
         'built_in_hwcaps_search_masked_empty':True,'default_directory_search_formally_denied':False,
         'late_load_or_import_closure_established':False,'pure_Lean_execution_authorized':False,
         'held_oracle_or_failed_profile_rerun':False,'old_failure_explained_or_repaired':False,
         'new_expected_runtime_hashes_created':False,'graph_checker_qualification':False,
         'trust':['host OS/kernel/vDSO and documented loader diagnostics','pinned loader/runtime and host observation']}
(out/'OBSERVATION.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
