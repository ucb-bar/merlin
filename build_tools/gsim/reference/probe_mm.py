#!/usr/bin/env python3
"""Public deterministic concurrency/byte-lane probes of the production mm_t ABI."""
import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import selectors
import subprocess
import time

IN_NAMES='reset ar_valid ar_address ar_id ar_size ar_len aw_valid aw_address aw_id aw_size aw_len w_valid w_strb w_data w_last r_ready b_ready'.split()
OUT_NAMES='ar_ready aw_ready w_ready r_valid r_id r_data r_resp r_last b_valid b_id b_resp'.split()


def run(reference,candidate,selection,output,receipt_path):
    selection=json.loads(Path(selection).read_text());axi=selection['axi'];base=axi['address_base']
    receipt_path=Path(receipt_path);receipt=json.loads(receipt_path.read_text())
    for name,expected in receipt['outputs'].items():
        if hashlib.sha256((receipt_path.parent/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('adapter output differs from pinned receipt: '+name)
    for role,binary in [('verilator-mm',reference),('gsim-mm',candidate)]:
        if hashlib.sha256(Path(binary).read_bytes()).hexdigest()!=receipt['outputs'][role]:
            raise ValueError('probe binary role differs from receipt')
    if selection!=receipt['configuration']['selection']:
        raise ValueError('probe selection differs from compiled configuration')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    args=[str(base),str(axi['address_size']),str(axi['address_bits']),str(axi['id_bits'])]
    writes=[]
    for index in range(12):
        size=index%4; beats=4 if size<3 else 8
        writes.append(dict(address=base+0x10000+index*0x1000,id=index%3+1,size=size,beats=beats,index=index))
    reads=[dict(address=base+0x80000+i*0x2000,id=i%3+8,size=3,beats=8,index=i) for i in range(16)]
    readback=[dict(w) for w in writes]
    store={};pending_reads=defaultdict(deque);pending_b=defaultdict(deque)
    stage=0; ar_index=aw_index=w_index=w_beat=0;accepted_reads=completed_reads=accepted_writes=completed_writes=0
    peak_reads=peak_writes=0; r_stalled=b_stalled=w_before_aw=0
    frames=[]; completions=[]; previous=None
    err=(output/'reference.stderr').open('w')
    proc=subprocess.Popen([reference,*args],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
    selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ);deadline=time.monotonic()+60
    try:
        for cycle in range(20000):
            row=dict.fromkeys(IN_NAMES,0);row['reset']=int(cycle<8)
            row['r_ready']=int(cycle%31>=11);row['b_ready']=int(cycle%37>=17)
            available_reads=reads if stage==0 else readback
            if cycle>=16 and ar_index<len(available_reads):
                r=available_reads[ar_index]
                row.update(ar_valid=1,ar_address=r['address'],ar_id=r['id'],ar_size=r['size'],ar_len=r['beats']-1)
            if cycle>=22 and stage==0 and aw_index<len(writes):
                w=writes[aw_index]
                row.update(aw_valid=1,aw_address=w['address'],aw_id=w['id'],aw_size=w['size'],aw_len=w['beats']-1)
            if cycle>=16 and stage==0 and w_index<len(writes):
                w=writes[w_index];address=w['address']+w_beat*(1<<w['size'])
                lanes=((1<<(1<<w['size']))-1)<<(address&7)
                mask=(0x55,0xaa,0xf0,0x0f,0xff,0)[(w['index']+w_beat)%6]
                row.update(w_valid=1,w_strb=lanes&mask,w_data=(0x123456789abcdef0^(w['index']<<32)^w_beat),w_last=int(w_beat+1==w['beats']))
            proc.stdin.write(' '.join(str(row[name]) for name in IN_NAMES)+'\n');proc.stdin.flush()
            remaining=deadline-time.monotonic()
            if remaining<=0 or not selector.select(remaining):raise TimeoutError('reference wall deadline')
            line=proc.stdout.readline()
            if not line:raise RuntimeError('reference exited: '+(output/'reference.stderr').read_text()[-1500:])
            values=list(map(int,line.split()))
            if len(values)!=len(OUT_NAMES):raise ValueError('bad reference output row')
            out=dict(zip(OUT_NAMES,values));frames.append(dict(cycle=cycle,inputs=row,outputs=out))
            if previous:
                old,old_in=previous
                for chan,fields in [('r',('id','data','resp','last')),('b',('id','resp'))]:
                    if old[chan+'_valid'] and not old_in[chan+'_ready']:
                        assert out[chan+'_valid'] and all(out[chan+'_'+f]==old[chan+'_'+f] for f in fields),'response changed under backpressure'
            previous=(out,row)
            if row['reset']:continue
            if row['ar_valid'] and out['ar_ready']:
                r=available_reads[ar_index];words=[]
                for beat in range(r['beats']):
                    address=(r['address']+beat*(1<<r['size']))&~7
                    words.append(sum(store.get(address+b,0)<<(8*b) for b in range(8)))
                pending_reads[r['id']].append(dict(words=words,beat=0,request=dict(r)))
                ar_index+=1;accepted_reads+=1
            if row['aw_valid'] and out['aw_ready']:
                pending_b[row['aw_id']].append(writes[aw_index]['index']);aw_index+=1;accepted_writes+=1
            if row['w_valid'] and w_index>=aw_index:w_before_aw+=1
            if row['w_valid'] and out['w_ready']:
                assert w_index<aw_index,'W accepted before corresponding AW'
                w=writes[w_index];address=(w['address']+w_beat*(1<<w['size']))&~7
                for byte in range(8):
                    if row['w_strb']&(1<<byte):store[address+byte]=(row['w_data']>>(8*byte))&255
                w_beat+=1
                if w_beat==w['beats']:w_index+=1;w_beat=0
            if out['r_valid']:
                assert pending_reads[out['r_id']],'unknown R ID'
                current=pending_reads[out['r_id']][0]
                assert out['r_resp']==0 and out['r_data']==current['words'][current['beat']],'incorrect read data/response'
                assert out['r_last']==int(current['beat']+1==len(current['words'])),'incorrect RLAST'
                if row['r_ready']:
                    current['beat']+=1
                    if out['r_last']:
                        completions.append(dict(kind='read',cycle=cycle,request=current['request']))
                        pending_reads[out['r_id']].popleft();completed_reads+=1
                else:r_stalled+=1
            if out['b_valid']:
                assert pending_b[out['b_id']] and out['b_resp']==0,'unknown B response'
                assert pending_b[out['b_id']][0]<w_index,'B before W completion'
                if row['b_ready']:
                    completions.append(dict(kind='write',cycle=cycle,index=pending_b[out['b_id']].popleft()))
                    completed_writes+=1
                else:b_stalled+=1
            peak_reads=max(peak_reads,accepted_reads-completed_reads);peak_writes=max(peak_writes,accepted_writes-completed_writes)
            if stage==0 and completed_writes==len(writes) and completed_reads==len(reads):
                stage=1;ar_index=0
            if stage==1 and completed_reads==len(reads)+len(readback):break
        else:raise TimeoutError('probe target-cycle limit')
    finally:
        proc.stdin.close()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
        selector.close();err.close()
    if proc.returncode:raise RuntimeError('reference failed: '+(output/'reference.stderr').read_text()[-1500:])
    assert peak_reads>1 and peak_writes>1 and r_stalled and b_stalled and w_before_aw
    stimulus=''.join(' '.join(str(f['inputs'][name]) for name in IN_NAMES)+'\n' for f in frames)
    (output/'stimulus.txt').write_text(stimulus)
    result=subprocess.run([candidate,*args],input=stimulus,text=True,capture_output=True,timeout=60)
    (output/'candidate.stderr').write_text(result.stderr);result.check_returncode()
    lines=result.stdout.splitlines();assert len(lines)==len(frames)
    mismatches=[]
    for frame,line in zip(frames,lines):
        values=list(map(int,line.split()));assert len(values)==len(OUT_NAMES)
        other=dict(zip(OUT_NAMES,values));ref=frame['outputs']
        for name in OUT_NAMES:
            if name.startswith('r_') and name!='r_valid' and not ref['r_valid']:continue
            if name.startswith('b_') and name!='b_valid' and not ref['b_valid']:continue
            if ref[name]!=other[name]:mismatches.append(dict(cycle=frame['cycle'],signal=name,reference=ref[name],candidate=other[name]))
    negative=[]
    input_lines=stimulus.splitlines()
    first_write=next(i for i,f in enumerate(frames) if f['inputs']['w_valid'] and f['outputs']['w_ready'])
    first_read=next(i for i,f in enumerate(frames) if f['inputs']['ar_valid'])
    for name,at,column,value,diagnostic in [
        ('invalid_geometry',None,None,None,'geometry'),
        ('invalid_size',first_read,4,4,'size'),
        ('wrong_wlast',first_write,14,1-int(input_lines[first_write].split()[14]),'WLAST'),
        ('out_of_lane_strobe',first_write,12,255,'strobes')]:
        command=[candidate,*args]
        if at is None:
            command[1]=str(base+8);bad_input=''
        else:
            bad_lines=input_lines[:at+1];row=bad_lines[-1].split();row[column]=str(value)
            bad_lines[-1]=' '.join(row);bad_input='\n'.join(bad_lines)+'\n'
        failed=subprocess.run(command,input=bad_input,text=True,capture_output=True,timeout=10)
        if failed.returncode==0 or diagnostic not in failed.stderr:
            raise AssertionError('negative native probe did not reject '+name)
        negative.append(dict(case=name,returncode=failed.returncode,diagnostic=failed.stderr.strip()))
    report=dict(schema='merlin.fased_mm_probe.v1',target_cycles=len(frames),completed_reads=completed_reads,
        completed_writes=completed_writes,peak_outstanding_reads=peak_reads,peak_outstanding_writes=peak_writes,
        read_backpressure_cycles=r_stalled,write_backpressure_cycles=b_stalled,w_before_aw_cycles=w_before_aw,
        signal_mismatches=len(mismatches),mismatch_samples=mismatches[:20],completions=completions,negative_probes=negative,
        reference_sha256=hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256(Path(candidate).read_bytes()).hexdigest(),
        stimulus_sha256=hashlib.sha256(stimulus.encode()).hexdigest(),configuration=selection,
        integration_receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        configuration_sha256=receipt['configuration']['configuration_sha256'],
        deployment_equivalence='NOT_ESTABLISHED',whole_system_qualified=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'reference.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames))
    (output/'candidate.txt').write_text(result.stdout)
    if mismatches:raise AssertionError(str(mismatches[:3]))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','candidate','selection','output','receipt'):p.add_argument('--'+name,required=True)
    a=p.parse_args();r=run(a.reference,a.candidate,a.selection,a.output,a.receipt)
    print(json.dumps({k:v for k,v in r.items() if k not in ('completions','configuration')},indent=2))


if __name__=='__main__':main()
