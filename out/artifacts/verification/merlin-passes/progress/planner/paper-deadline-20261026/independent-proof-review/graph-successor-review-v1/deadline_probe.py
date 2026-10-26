"""Small independent POSIX timer controls: own pipes/fake backend, no network,
SDK/images, guards or native/compiler/process launches. F suites not imported.
"""
from pathlib import Path
from copy import deepcopy
from contextlib import contextmanager
import sys,os,json,time,signal,threading,math,hashlib,traceback
B=Path(__file__).resolve().parent;CUT=B/'f-deadline';ROOT=B/'deadline-probes';ROOT.mkdir(exist_ok=False);sys.path.insert(0,str(CUT))
import outer_deadline as od
import intake,fdio
for m in [od,intake,fdio]:assert Path(m.__file__).parent==CUT
blocked=[]
def audit(event,args):
 if event.startswith(('subprocess.','socket.','ctypes.','os.exec','os.spawn'))or event in ['os.system','os.fork']:
  blocked.append(event);raise RuntimeError('forbidden network/process/native action')
sys.addaudithook(audit)
initial_handler=signal.getsignal(signal.SIGALRM);initial_timer=signal.getitimer(signal.ITIMER_REAL);assert initial_timer==(0.0,0.0)
results=[]
def check(name,fn):
 row={'id':name};start=time.monotonic()
 try:row['evidence']=fn();row['ok']=True
 except BaseException as e:row.update(ok=False,error={'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()})
 row['elapsed_seconds']=time.monotonic()-start;results.append(row);print(json.dumps(row,default=str),flush=True)
def restored():assert signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0)and signal.getsignal(signal.SIGALRM)==initial_handler

def normal():
 def prior(*a):pass
 signal.signal(signal.SIGALRM,prior)
 try:
  with od.fetch_deadline(.2):assert signal.getsignal(signal.SIGALRM)is not prior
  assert signal.getsignal(signal.SIGALRM)is prior and signal.getitimer(signal.ITIMER_REAL)==(0.,0.)
 finally:signal.signal(signal.SIGALRM,initial_handler)
 return 'prior callable handler restored; timer cancelled'
def same_error():
 failure=ValueError('original body failure')
 try:
  with od.fetch_deadline(.2):raise failure
 except ValueError as e:assert e is failure
 restored();return 'same primary object propagated'
def pipe_interrupt():
 rd,wr=os.pipe();start=time.monotonic()
 try:
  try:
   with od.fetch_deadline(.03):os.read(rd,1)
  except od.FetchDeadline:pass
  else:raise AssertionError('pipe not interrupted')
 finally:os.close(rd);os.close(wr)
 elapsed=time.monotonic()-start;assert .01<=elapsed<1;restored();return {'blocked_read_interrupted':True,'elapsed':elapsed}
def existing_timer():
 signal.setitimer(signal.ITIMER_REAL,2.0)
 try:
  try:
   with od.fetch_deadline(.03):raise AssertionError('entered timer takeover')
  except ValueError as e:assert 'preexisting timer'in str(e)
  left,interval=signal.getitimer(signal.ITIMER_REAL);assert 1.0<left<=2.0 and interval==0
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 restored();return 'existing timer preserved, then cancelled by test owner'
def nonmain():
 out=[]
 def worker():
  try:
   with od.fetch_deadline(.03):out.append('BAD')
  except ValueError as e:out.append(str(e))
 t=threading.Thread(target=worker);t.start();t.join(timeout=1);assert not t.is_alive()and len(out)==1 and 'main-thread'in out[0];restored();return out

def blocked_signal():
 previous=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM})
 try:
  try:
   with od.fetch_deadline(.03):raise AssertionError('entered blocked signal scope')
  except ValueError as e:assert 'blocked/pending'in str(e)
 finally:signal.pthread_sigmask(signal.SIG_SETMASK,previous)
 restored();return 'real blocked-mask refused; original mask restored by test'
def invalid():
 for value in [True,0,-1,float('nan'),float('inf'),1801]:
  try:
   with od.fetch_deadline(value):raise AssertionError('entered invalid duration')
  except ValueError:pass
 restored();return 'bool/zero/negative/nonfinite/excess duration refused'
def mocked_pending():
 prior=od.signal.sigpending;od.signal.sigpending=lambda:{signal.SIGALRM}
 try:
  try:
   with od.fetch_deadline(.03):raise AssertionError('entered pending scope')
  except ValueError as e:assert 'blocked/pending'in str(e)
 finally:od.signal.sigpending=prior
 restored();return 'MOCKED pending-state branch refused; no signal sent'
def cleanup_failure():
 old=od.signal.setitimer;failure=ValueError('primary kept')
 def fail_after_cancel(which,seconds,interval=0):
  result=old(which,seconds,interval)
  if seconds==0:raise OSError('synthetic cleanup report')
  return result
 od.signal.setitimer=fail_after_cancel
 try:
  try:
   with od.fetch_deadline(.2):raise failure
  except ValueError as e:assert e is failure and any('cleanup'in n for n in e.__notes__)
 finally:od.signal.setitimer=old
 restored();return 'primary identity retained; timer actually cancelled before injected reporting failure'
def arm_failure():
 old=od.signal.setitimer
 def refuse_arm(which,seconds,interval=0):
  if seconds>0:raise OSError('synthetic arm failure')
  return old(which,seconds,interval)
 od.signal.setitimer=refuse_arm
 try:
  try:
   with od.fetch_deadline(.2):raise AssertionError('body entered after failed arm')
  except OSError as e:assert str(e)=='synthetic arm failure'
 finally:od.signal.setitimer=old
 restored();return 'failed arm refuses; original handler restored'
def integration(which):
 case=ROOT/which;case.mkdir();r=deepcopy(json.loads((CUT/'REQUEST.json').read_text()));raw=b'tiny fake archive';r['new_root']=str(case/'new-data');r['fetch'].update(total_deadline_seconds=1,exact_bytes=len(raw),max_read_bytes_including_sentinel=len(raw)+1,expected_sha256=hashlib.sha256(raw).hexdigest());r['environment_identity'].update(compressed_bytes=len(raw),publisher_platform_sha256=r['fetch']['expected_sha256']);r['extract']['disk_reserve_bytes']=1024**2
 rd,wr=os.pipe();calls=[]
 class Response:
  status=200;headers={'Content-Length':str(len(raw))}
  def read(self,n):calls.append('blocking-read');return os.read(rd,n)
  def close(self):calls.append('close')
 class Backend:
  def open(self,url,timeout):
   calls.append('open')
   if which=='blocking-open':os.read(rd,1)
   return Response()
 start=time.monotonic()
 try:
  try:intake.run(r,Backend())
  except fdio.IntakeStop as e:assert e.evidence['error_type']=='FetchDeadline';error=e.evidence
  else:raise AssertionError('blocked fetch accepted')
 finally:os.close(rd);os.close(wr)
 elapsed=time.monotonic()-start;assert .5<=elapsed<3;assert not(case/'new-data/RESULT.json').exists();assert(case/'new-data/FIRST-FAILURE.json').exists();restored();return {'elapsed':elapsed,'calls':calls,'first_failure':error,'backend':'FAKE, actual own-pipe blocking syscall'}
for name,fn in [('restore-normal',normal),('same-primary',same_error),('pipe-read-interrupt',pipe_interrupt),('preexisting-timer-refusal',existing_timer),('nonmain-refusal',nonmain),('blocked-mask-refusal',blocked_signal),('invalid-duration',invalid),('mocked-pending-refusal',mocked_pending),('cleanup-primary-preserved',cleanup_failure),('arm-failure-restores',arm_failure),('blocking-open',lambda:integration('blocking-open')),('blocking-read',lambda:integration('blocking-read'))]:check(name,fn)
restored();(B/'DEADLINE_RESULTS.json').write_text(json.dumps({'status':'PASS'if all(r['ok']for r in results)else'FAIL','cases':len(results),'blocked_actions':blocked,'real_network':False,'actual_images':0,'native_launches':0,'results':results},indent=2)+'\n')
