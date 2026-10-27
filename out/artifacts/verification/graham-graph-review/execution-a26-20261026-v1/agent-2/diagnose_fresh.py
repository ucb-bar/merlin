"""Finite diagnostics using only the independently authenticated fresh SDK."""
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import time

ROOT=pathlib.Path('/scratch/grammy/blast/merlin-pass-verification')
EVIDENCE=ROOT/'out/artifacts/verification/graham-graph-review/execution-a26-20261026-v1/agent-2'
BUILD=ROOT/'out/build/verification-a26-20261026-v1/fresh-sdk'
SDK=BUILD/'data/lean-4.35.0-linux'
DIAGNOSTICS=BUILD/'diagnostics'
RECORDS=[]


def save(name,value):
    with (EVIDENCE/name).open('x') as output:
        json.dump(value,output,indent=2)
        output.write('\n')


def digest(path):
    data=path.read_bytes()
    return {'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def run(name,arguments,environment):
    command=[str(SDK/'bin/lean'),'-j1',*arguments]
    record={'command':command,'cwd':str(DIAGNOSTICS),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'timeout_seconds':90}
    save(name+'.intent.json',record)
    started=time.monotonic()
    with (EVIDENCE/(name+'.stdout')).open('xb') as out,(EVIDENCE/(name+'.stderr')).open('xb') as err:
        child=subprocess.Popen(command,cwd=DIAGNOSTICS,env=environment,stdout=out,stderr=err)
        try:code=child.wait(timeout=90)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:code=child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill();code=child.wait(timeout=3)
            record['timeout']=True
    record.update(returncode=code,elapsed_seconds=time.monotonic()-started,child_reaped=True)
    save(name+'.result.json',record)
    RECORDS.append(record)
    if code or record.get('timeout'):
        save('DIAGNOSTICS_FIRST_FAILURE.json',{'command':record,'stdout':(EVIDENCE/(name+'.stdout')).read_text(),'stderr':(EVIDENCE/(name+'.stderr')).read_text()})
        raise RuntimeError('first native diagnostic failure: '+name)
    return (EVIDENCE/(name+'.stdout')).read_text()


def main():
    intake=json.loads((EVIDENCE/'INTAKE_RESULT.json').read_bytes())
    if intake['phase']!='DATA_INTAKE_COMPLETE':raise ValueError('fresh archive intake required')
    manifest=json.loads((EVIDENCE/'SDK_MEMBER_MANIFEST.json').read_bytes())
    index={r['path']:r for r in manifest['members']}
    for rel in ['bin/lean','lib/lean/libInit_shared.so','lib/lean/libleanshared.so','lib/lean/libleanshared_1.so','lib/lean/libleanshared_2.so']:
        member='data/lean-4.35.0-linux/'+rel
        identity=digest(SDK/rel)
        if identity['sha256']!=index[member]['sha256'] or identity['bytes']!=index[member]['bytes']:
            raise ValueError('fresh runtime binding drift')
    DIAGNOSTICS.mkdir(mode=0o700)
    env={k:v for k,v in os.environ.items() if not k.startswith(('LEAN','LAKE','ELAN','LD_'))}
    env.update(LEAN_PATH=str(DIAGNOSTICS),LD_LIBRARY_PATH=str(SDK/'lib')+':'+str(SDK/'lib/lean'),LEAN_NUM_THREADS='1',OMP_NUM_THREADS='1')
    runtime_review={'sdk':str(SDK),'lean':digest(SDK/'bin/lean'),'runtime':digest(SDK/'lib/lean/libleanshared.so'),'environment':{k:env[k] for k in ['LEAN_PATH','LD_LIBRARY_PATH','LEAN_NUM_THREADS','OMP_NUM_THREADS']},'host_loader':'ELF interpreter /lib64/ld-linux-x86-64.so.2 and host libc/pthread/dl/m platform dependencies remain host TCB. No old SDK path is admitted.','archive_member_manifest':str(EVIDENCE/'SDK_MEMBER_MANIFEST.json'),'permission':'New user-authorized independent CLI diagnostic context; no common Context or public gate authority.'}
    save('RUNTIME_REVIEW.json',runtime_review)
    version=run('lean-version',['--version'],dict(env,LD_DEBUG='libs'))
    if 'version 4.35.0' not in version or 'eb6cafb7290e' not in version:
        raise ValueError('fresh executable version mismatch')
    prefix=run('lean-prefix',['--print-prefix'],env).strip()
    if pathlib.Path(prefix).resolve()!=SDK.resolve():raise ValueError('unexpected Lean prefix')
    cases=[]
    source_root=ROOT/'out/artifacts/verification/graph-regions-declared/runtime-forensics-v1/fresh-archive-context-candidate-v1/diagnostic-proposal-v1'
    for name in ['InitNat','InitListNat','OmegaListNat']:
        source=source_root/(name+'.lean')
        copied=DIAGNOSTICS/source.name
        copied.write_bytes(source.read_bytes())
        root='DiagnosticRepr'+name
        run(name+'-compile',['-o',str(DIAGNOSTICS/(name+'.olean')),'-i',str(DIAGNOSTICS/(name+'.ilean')),str(copied)],env)
        audit=DIAGNOSTICS/('Audit'+name+'.lean')
        audit.write_text('import '+name+'\n#check '+root+'.witness_value\n#print axioms '+root+'.witness_value\n#print axioms '+root+'.instReprPayload\n#eval repr '+root+'.witness\n')
        output=run(name+'-audit',[str(audit)],env)
        cases.append({'case':name,'source':digest(copied),'audit':digest(audit),'output':output,'objects':[digest(p) for p in sorted(DIAGNOSTICS.glob(name+'.*')) if p.suffix!='.lean']})
    save('DIAGNOSTICS_RESULT.json',{'status':'PASS','version':version.strip(),'prefix':prefix,'cases':cases,'commands':len(RECORDS),'old_holds_cleared':False,'graph_qualification':False,'context_type':'fresh independent CLI source reproduction environment'})
    print(json.dumps({'status':'PASS','commands':len(RECORDS),'sdk':str(SDK),'lean':str(SDK/'bin/lean'),'lib':str(SDK/'lib/lean'),'cases':[c['case'] for c in cases]}),flush=True)


if __name__=='__main__':main()
