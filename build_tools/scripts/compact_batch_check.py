"""Bind compact batch consoles to expected storage hashes and the actual ELF.

BATCH_HASH is FNV-1a-64 over little-endian uint64 (index, observed hash)
records. BATCH_ELF_ID is FNV-1a-64 over the embedded WANT uint64 table.
These integrity checks bind an execution receipt; they are not signatures.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def fnv64(data):
    value=0xCBF29CE484222325
    for byte in data:
        value=((value ^ byte)*0x100000001B3) & ((1<<64)-1)
    return value


def expected_ids(cases):
    words=[int(c['expected_fnv']) for c in cases]
    table=b''.join(w.to_bytes(8,'little') for w in words)
    records=b''.join(k.to_bytes(8,'little')+w.to_bytes(8,'little') for k,w in enumerate(words))
    return fnv64(records),fnv64(table)


def check(manifest, console):
    cases=manifest['cases']; seen={}; fields={}; issues=[]
    for line in console.splitlines():
        parts=line.strip().split()
        if not parts:continue
        tag=parts[0]
        if len(tag)>1 and tag[0] in 'PF' and tag[1:].isdigit():
            k=int(tag[1:])
            if not 0<=k<len(cases):issues.append('case_index_out_of_range');continue
            seen.setdefault(k,[]).append('pass' if tag[0]=='P' else 'fail')
        elif tag in {'BATCH_HASH','BATCH_ELF_ID','BATCH_DONE','PERF'}:
            fields.setdefault(tag,[]).append(parts[1:])
    rows=[]
    for k,c in enumerate(cases):
        found=seen.get(k,[])
        status='missing' if not found else 'duplicate' if len(found)>1 else found[0]
        rows.append(dict(k=k,case_id=c['case_id'],lane=c['lane'],status=status,cycles=None))
    batch_hash,elf_id=expected_ids(cases)
    def field(tag):
        values=fields.get(tag,[])
        if len(values)!=1:raise ValueError('missing_or_duplicate_'+tag)
        return values[0]
    fails=None;perf={}
    try:
        if int(field('BATCH_HASH')[0],16)!=batch_hash:issues.append('batch_result_hash_mismatch')
        if int(field('BATCH_ELF_ID')[0],16)!=elf_id:issues.append('embedded_want_digest_mismatch')
        fails=int(dict(p.split('=',1) for p in field('BATCH_DONE'))['fails'])
        perf={k:int(v) for k,v in (p.split('=',1) for p in field('PERF'))}
        if perf['n']!=len(cases):issues.append('performance_case_count_mismatch')
    except (ValueError,KeyError,IndexError) as exc:issues.append(str(exc))
    try:
        elf=Path(manifest['elf'])
        with elf.open('rb') as stream: actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=manifest['elf_sha256']:issues.append('elf_sha256_mismatch')
    except (OSError,KeyError) as exc:issues.append('elf_unavailable:'+str(exc))
    result=dict(elf_sha256=manifest.get('elf_sha256'),mode=manifest.get('mode'),cases=len(rows),
                batch_done=len(fields.get('BATCH_DONE',[]))==1,batch_fails=fails,
                status_counts=dict(Counter(r['status'] for r in rows)),
                device=dict(Counter(r['status'] for r in rows if r['lane']=='device')),
                host=dict(Counter(r['status'] for r in rows if r['lane']!='device')),
                total_case_cycles=perf.get('cyc'),max_case_cycles=perf.get('max'),
                max_case=cases[perf['k']]['case_id'] if 0<=perf.get('k',-1)<len(cases) else None,
                console_chars=len(console),issues=issues,rows=rows)
    result['pass']=not issues and fails==0 and all(r['status']=='pass' for r in rows)
    return result


def main():
    manifest,console,out=map(Path,sys.argv[1:])
    doc=check(json.loads(manifest.read_text()),console.read_text(errors='replace'))
    doc.update(manifest=str(manifest),console=str(console))
    out.write_text(json.dumps(doc,indent=1)+'\n')
    print(json.dumps({k:v for k,v in doc.items() if k!='rows'}))
    return 0 if doc['pass'] else 1


if __name__=='__main__':raise SystemExit(main())
