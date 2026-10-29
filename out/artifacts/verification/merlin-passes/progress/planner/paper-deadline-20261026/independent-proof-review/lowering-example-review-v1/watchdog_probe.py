"""Small own-child supervisor test. Only reviewed local fake Python children.
No network/SDK/compiler/archive execution, F suite imports or foreign PID signals.
"""
from pathlib import Path
import os,sys,json,time,subprocess,signal,traceback
B=Path(__file__).resolve().parent;CUT=B/'watchdog-fixed';FIX=B/'watchdog-fixtures';ROOT=B/'watchdog-probes';ROOT.mkdir(exist_ok=False);sys.path.insert(0,str(CUT))
import watchdog,fdio
assert Path(watchdog.__file__).parent==CUT
owned=set();created=[];signal_events=[];forbidden=[];originalPopen=subprocess.Popen
allowed={str(p)for p in FIX.glob('*.py')}
def own_popen(command,*args,**kwargs):
 assert type(command)is list and command[:4]==[sys.executable,'-I','-S','-B']and command[4]in allowed
 child=originalPopen(command,*args,**kwargs);owned.add(child.pid);created.append(child);return child
subprocess.Popen=own_popen
def audit(event,args):
 if event.startswith(('socket.','ctypes.'))or event in ['os.system','os.exec','os.posix_spawn']:
  forbidden.append(event);raise RuntimeError('forbidden network/native/external launch')
 if event=='os.kill':
  pid,sig=args;assert pid in owned,'foreign PID forbidden';signal_events.append({'pid':pid,'signal':sig})
sys.addaudithook(audit)
results=[]
def run(name,fn):
 row={'id':name};start=time.monotonic()
 try:row['evidence']=fn();row['ok']=True
 except BaseException as e:row.update(ok=False,error={'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()})
 row['elapsed_seconds']=time.monotonic()-start;results.append(row);print(json.dumps(row,default=str),flush=True)
def childcase(name,script):
 path=ROOT/name;before=len(created);start=time.monotonic();result=watchdog.supervise([sys.executable,'-I','-S','-B',str(FIX/script)],path,seconds=.25,terminate_seconds=.1,reap_seconds=.3);elapsed=time.monotonic()-start;assert len(created)==before+1;process=created[-1];assert process.pid==result['owned_pid']and process.poll()is not None and result['child_reaped']is True;assert result['native_authorized']is False and result['old_holds_cleared']is False and result['retry']is False
 assert all(s['owned_pid']==process.pid for s in result['signals']);assert elapsed<2
 if script=='complete.py':assert result['status']=='CHILD_COMPLETED'and result['forced_stop']is False and(path/'partial-data.bin').read_bytes()==b'four tiny bytes'
 elif script=='nonzero.py':assert result['status']=='CHILD_FAILED'and result['child_exit_code']==2 and(path/'FIRST-PARENT-FAILURE.json').exists()
 else:
  assert result['status']=='COMPLETE_INTAKE_BUDGET_EXCEEDED'and result['forced_stop']is True;assert(path/'FIRST-PARENT-FAILURE.json').exists();assert(path/'FIRST-CHILD-FAILURE.json').exists();assert(path/'partial-data.bin').read_bytes()in[b'partial fake data',b'retained before KILL'];assert b'synthetic'in(path/'child.stdout.txt').read_bytes()
  if script=='ignores_term.py':assert[s['signal']for s in result['signals']]==['SIGTERM','SIGKILL']
 return result
for name,script in [('completed','complete.py'),('blocking-cleanup','blocking_close.py'),('kill-and-reap','ignores_term.py'),('failed-child','nonzero.py')]:run(name,lambda n=name,s=script:childcase(n,s))
def existing():
 path=ROOT/'existing';path.mkdir();(path/'preserved.txt').write_text('old');before=len(created)
 try:watchdog.supervise([sys.executable,'-I','-S','-B',str(FIX/'complete.py')],path,seconds=.25,terminate_seconds=.1,reap_seconds=.3)
 except fdio.IntakeStop:pass
 else:raise AssertionError('existing workspace resumed')
 assert len(created)==before and(path/'preserved.txt').read_text()=='old';return'no child launched, old data untouched'
def command_refusal():
 before=len(created)
 try:watchdog.supervise(['/bin/true','-I','-S','-B',str(FIX/'complete.py')],ROOT/'foreign-command',seconds=.25,terminate_seconds=.1,reap_seconds=.3)
 except ValueError:pass
 else:raise AssertionError('external command admitted')
 assert len(created)==before;return'foreign executable prefix refused before Popen'
def unreaped_protocol():
 class Fake:
  pid=-1 # explicit synthetic marker; no OS signal method exists here
  def __init__(self):self.waits=[];self.signals=[]
  def terminate(self):self.signals.append('TERM')
  def kill(self):self.signals.append('KILL')
  def wait(self,timeout):self.waits.append(timeout);raise subprocess.TimeoutExpired('FAKE',timeout)
 p=Fake();result={'signals':[]};watchdog.stop_owned(p,result,.1,.3);assert result['status']=='OWNED_CHILD_UNREAPED_HOLD'and result['child_reaped']is False and result['no_retry']is True and p.waits==[.1,.3];return{'label':'MOCKED no-process reap branch only','result':result,'no_real_PID':True}
run('existing-workspace-refusal',existing);run('external-command-refusal',command_refusal);run('unreaped-hold-protocol',unreaped_protocol)
subprocess.Popen=originalPopen
assert all(p.poll()is not None for p in created)
out={'status':'PASS'if all(x['ok']for x in results)else'FAIL','cases':len(results),'actual_owned_python_children':len(created),'all_actual_children_reaped':True,'signal_events':signal_events,'foreign_signal_attempts':[],'forbidden_actions':forbidden,'real_network_or_sdk_or_native_compiler':False,'results':results};(B/'WATCHDOG_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
