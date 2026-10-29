"""One bounded own-pipe counterexample to cleanup covered by expired timer.
A controlled helper thread releases all pipes within ~2.1s; no network/images.
"""
from pathlib import Path
from copy import deepcopy
import sys,os,signal,time,threading,json,hashlib
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'f-deadline'))
import intake,fdio
root=B/'deadline-cleanup-bound';root.mkdir();r=deepcopy(json.loads((B/'f-deadline/REQUEST.json').read_text()));r['new_root']=str(root/'data');r['fetch'].update(total_deadline_seconds=1,exact_bytes=1,max_read_bytes_including_sentinel=2,expected_sha256=hashlib.sha256(b'x').hexdigest());r['environment_identity'].update(compressed_bytes=1,publisher_platform_sha256=r['fetch']['expected_sha256']);r['extract']['disk_reserve_bytes']=1024**2
rd,wr=os.pipe();crd,cwr=os.pipe();entered=threading.Event();events=[];start=time.monotonic();handler=signal.getsignal(signal.SIGALRM)
class Response:
 status=200;headers={'Content-Length':'1'}
 def read(self,n):events.append({'read_start':time.monotonic()-start});return os.read(rd,1)
 def close(self):
  events.append({'close_entered':time.monotonic()-start,'timer_at_close':signal.getitimer(signal.ITIMER_REAL)});entered.set();os.read(crd,1);events.append({'close_returned':time.monotonic()-start})
class Backend:
 def open(self,url,timeout):return Response()
def release():
 seen=entered.wait(2);threading.Event().wait(.12)
 # Release close even if an unexpected control flow occurs; no unbounded hang.
 os.write(cwr,b'x')
 if not seen:os.write(wr,b'x')
thread=threading.Thread(target=release);thread.start()
try:
 try:intake.run(r,Backend())
 except fdio.IntakeStop as e:assert e.evidence['error_type']=='FetchDeadline';evidence=e.evidence
 else:raise AssertionError('deadline case accepted')
finally:
 thread.join(3)
 for fd in [rd,wr,crd,cwr]:os.close(fd)
elapsed=time.monotonic()-start;assert not thread.is_alive();assert signal.getitimer(signal.ITIMER_REAL)==(0.,0.)and signal.getsignal(signal.SIGALRM)==handler;assert events[1]['timer_at_close']==(0.,0.)and events[2]['close_returned']-events[1]['close_entered']>=.1;assert not(root/'data/RESULT.json').exists()
out={'status':'EXPIRED_ONESHOT_CANNOT_INTERRUPT_SUBSEQUENT_BLOCKING_RESPONSE_CLOSE','policy_seconds':1,'elapsed':elapsed,'events':events,'first_failure':evidence,'native_or_real_network':False,'claim_limit':'FetchDeadline interrupts read, but finally close blocks with no active timer; helper release guarantees bounded synthetic run.'};(root/'RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='first_failure'}))
