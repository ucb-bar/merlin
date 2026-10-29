"""One focused fake-fetch cleanup-evidence check, no network or real images."""
from pathlib import Path
from copy import deepcopy
import sys,json,signal,hashlib
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'f-deadline'))
import intake,fdio,outer_deadline
case=B/'deadline-cleanup-evidence';case.mkdir();r=deepcopy(json.loads((B/'f-deadline/REQUEST.json').read_text()));r['new_root']=str(case/'data');r['fetch'].update(total_deadline_seconds=1,exact_bytes=1,max_read_bytes_including_sentinel=2,expected_sha256=hashlib.sha256(b'x').hexdigest());r['environment_identity'].update(compressed_bytes=1,publisher_platform_sha256=r['fetch']['expected_sha256']);r['extract']['disk_reserve_bytes']=1024**2
class Response:
 status=500;headers={}
 def close(self):pass
class Backend:
 def open(self,url,timeout):return Response()
old=signal.setitimer;handler=signal.getsignal(signal.SIGALRM)
def synthetic_cancel_failure(which,seconds,interval=0):
 result=old(which,seconds,interval)
 if seconds==0:raise OSError('synthetic cancellation reporting failure')
 return result
signal.setitimer=synthetic_cancel_failure
try:
 try:intake.run(r,Backend())
 except fdio.IntakeStop as e:
  notes=getattr(e.__cause__,'__notes__',[]);durable=json.loads((case/'data/FIRST-FAILURE.json').read_text());assert notes and 'cleanup'in notes[0];assert 'synthetic cancellation reporting failure'not in json.dumps(durable);assert not(case/'data/RESULT.json').exists();result={'status':'CLEANUP_NOTES_LOST_FROM_DURABLE_FIRST_FAILURE','primary_error':durable['error'],'in_memory_cause_notes':notes,'durable':durable,'native_or_network':False}
 else:raise AssertionError('failed HTTP accepted')
finally:signal.setitimer=old;old(signal.ITIMER_REAL,0);assert signal.getsignal(signal.SIGALRM)==handler
(case/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'primary_error':result['primary_error'],'notes':result['in_memory_cause_notes']}))
