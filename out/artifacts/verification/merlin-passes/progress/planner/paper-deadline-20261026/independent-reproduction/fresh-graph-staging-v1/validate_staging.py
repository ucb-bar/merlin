"""One actual staging batch; tiny refusal controls; one same-file relocation."""
import contextlib,copy,io,json,os,runpy,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=Path.cwd();SCRATCH=ROOT/'out/build/graph-paper-deadline/fresh-graph-staging-v1';sys.path.insert(0,str(OUT))
from materialize_plan import materialize,verify_materialized
from plan_core import decode,pin

def audit(event,args):
 if event.startswith(('subprocess.','socket.','ctypes.')) or event in ('os.system','os.exec','os.posix_spawn','os.fork'):raise PermissionError('source staging forbids native/process action')
 if event=='exec' and ('/payload/' in args[0].co_filename or args[0].co_filename.endswith('.lean')):raise PermissionError('payload is DATA only')
 if event=='open' and isinstance(args[0],(str,bytes)):
  name=os.fsdecode(args[0]);p=Path(name).absolute();mode=args[1] or '';flags=args[2];write=any(c in mode for c in 'wax+') or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
  allowed=p.is_relative_to(OUT) or p.is_relative_to(SCRATCH) or name.startswith('/usr/lib/python3.14/')
  if write:allowed=p.is_relative_to(OUT) or p.is_relative_to(SCRATCH)
  if not allowed:raise PermissionError('outside source staging allowlist: '+name)
sys.addaudithook(audit)
actual=SCRATCH/'actual-staged';result=materialize(OUT/'capsule',actual);(OUT/'ACTUAL_STAGING.json').write_text(json.dumps(result,indent=2)+'\n')
controls=SCRATCH/'controls';controls.mkdir();results=[]
def tiny(label):
 parent=controls/label;parent.mkdir();packet=parent/'packet';packet.mkdir();(packet/'payload').mkdir()
 def add(name,raw):
  (packet/'payload'/name).write_bytes(raw);return {'file':'payload/'+name,**pin(raw)}
 source=add('Base.lean',b'theorem base : True := by trivial\n');query=add('Query.lean',b'import Merlin.Base\ntheorem final : True := by trivial\n')
 root={'schema':'merlin.graph-staging-roots.v1','profile':'Tiny','case':'case1','module':'Query','query_pin':{k:query[k] for k in ('sha256','bytes')},'final_root':'final','support_roots':['Merlin.Base.base'],'exact_final_declaration':'theorem final : True','domain':'tiny DATA','relation':None,'assumptions':None,'native_context_request':'UNSELECTED','compile':'UNEXECUTED','audit':'UNEXECUTED','qualification':'UNEXECUTED','native_authorization':False,'graph_approval':False,'human_review':False}
 roots=add('roots.json',(json.dumps(root)+'\n').encode());spec={'schema':'merlin.graph-staging-spec.v1','primary_baseline':'tiny_DATA','profiles':[{'id':'Tiny','project_prefixes':['Merlin.'],'sources':[{'module':'Merlin.Base','ref':source}],'dependency_order':['Merlin.Base'],'cases':[{'id':'case1','module':'Query','query':query,'roots':roots}],'provenance':{}}]}
 return parent,packet,spec,add
for label in ('valid','missing','mutated','conflict','duplicate','path','extra','cycle','existing_destination','extra_output'):
 parent,packet,spec,add=tiny(label);profile=spec['profiles'][0]
 if label=='missing':profile['sources'][0]['ref']['file']='payload/missing.lean'
 elif label=='mutated':(packet/profile['sources'][0]['ref']['file']).write_bytes(b'-- changed DATA\n')
 elif label=='conflict':profile['sources'].append({'module':'Merlin.Base','ref':add('Other.lean',b'theorem other : True := by trivial\n')})
 elif label=='duplicate':profile['sources'].append(copy.deepcopy(profile['sources'][0]))
 elif label=='path':profile['sources'][0]['ref']['file']='../outside.lean'
 elif label=='extra':(packet/'payload/Extra.lean').write_text('-- extra DATA\n')
 elif label=='cycle':profile['sources'][0]['ref']=add('Cycle.lean',b'import Merlin.Base\ntheorem base : True := by trivial\n')
 (packet/'STAGING_SPEC.json').write_text(json.dumps(spec,indent=2)+'\n');target=parent/'staged'
 try:
  observed=materialize(packet,target)
  assert label in ('valid','existing_destination','extra_output'),label
  if label=='existing_destination':
   before=(target/'MATERIALIZATION.json').read_bytes()
   try:materialize(packet,target);raise AssertionError('overwrote existing destination')
   except FileExistsError:pass
   assert (target/'MATERIALIZATION.json').read_bytes()==before
   observed={'status':'EXPECTED_EXISTING_DESTINATION_REFUSAL','original_unchanged':True}
  elif label=='extra_output':
   (target/'Extra.lean').write_text('-- unlisted staged DATA\n')
   try:verify_materialized(target);raise AssertionError('extra staged payload accepted')
   except ValueError as error:observed={'status':'EXPECTED_EXTRA_OUTPUT_REFUSAL','reason':str(error)}
 except (ValueError,OSError) as error:
  assert label not in ('valid','existing_destination','extra_output'),(label,str(error))
  assert (target/'FIRST_FAILURE.json').is_file()
  observed={'status':'EXPECTED_SOURCE_DATA_REFUSAL','reason':str(error),'partial_first_failure_preserved':True}
 results.append({'control':label,'observed':observed,'proof_or_native_case':False})
(OUT/'BOUNDED_CONTROLS.json').write_text(json.dumps(results,indent=2)+'\n')
# Relocate the already staged exact files via hard links; do NOT materialize the
# actual profiles twice. Execute only our copied own utility, never payload code.
relocated=SCRATCH/'relocated-staged';relocated.mkdir();count=0
for path in actual.rglob('*'):
 if path.is_file():
  target=relocated/path.relative_to(actual);target.parent.mkdir(parents=True,exist_ok=True);os.link(path,target);assert os.path.samefile(path,target);count+=1
tools=SCRATCH/'relocation-tools';tools.mkdir()
for name in ('materialize_plan.py','plan_core.py'):(tools/name).write_bytes((OUT/name).read_bytes())
unrelated=SCRATCH/'unrelated-cwd';unrelated.mkdir();oldcwd=Path.cwd();oldargv=sys.argv[:];oldpath=sys.path[:];saved=sys.modules.pop('plan_core')
try:
 os.chdir(unrelated);sys.argv=['materialize_plan.py','--verify-staged',str(relocated)]
 with contextlib.redirect_stdout(io.StringIO()) as stdout:
  try:runpy.run_path(str(tools/'materialize_plan.py'),run_name='__main__')
  except SystemExit as stopped:assert stopped.code==0
 observed=decode(stdout.getvalue());assert observed==result
finally:os.chdir(oldcwd);sys.argv=oldargv;sys.path=oldpath;sys.modules['plan_core']=saved
relocation={'status':'PASSED_UNRELATED_CWD_SAME_FILE_RELOCATION','same_file_associations':count,'result':observed,'actual_materialization_runs':1,'native_execution':False};(OUT/'RELOCATION_CHECK.json').write_text(json.dumps(relocation,indent=2)+'\n')
summary={'status':'PASSED_SOURCE_STAGING_DATA_BATCH','actual_materialization_runs':1,'profiles':4,'query_cases':33,'files':result['files'],'tiny_controls':len(results),'all_tiny_controls_expected':True,'relocation_associations':count,'SDK_provider_controller_Lean_native_execution':False,'compile_audit_qualification':'UNEXECUTED','materializer_source':pin((OUT/'materialize_plan.py').read_bytes()),'plan_core_source':pin((OUT/'plan_core.py').read_bytes()),'spec':pin((OUT/'capsule/STAGING_SPEC.json').read_bytes())};(OUT/'VALIDATION_BATCH.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
