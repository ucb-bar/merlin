"""Prepare full Tiny reuse programs while preserving captured operand/reference bytes."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).absolute().parents[2]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'tools/host_iss')]
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked
from atlas_tooling import sha256
from atlas_program import assembly, dma_load_bytes
from kernel_library import code_sources
from full_bundle import KernelAssets
from run_model_tile import configure_model, model_binding
from run_workload_contraction import admit_program
from tiny_full_contract import contract
from types import SimpleNamespace

POLICY = 'traffic-reuse-v1'


def encode(item, assemble):
    m, k, n = item['physical_mkn']
    source = assembly(m,k,n,item['output_offset'],item['rescale'],traffic_reuse=True)
    words = list(assemble(io.StringIO(source)).assemble())
    admit_program(words)
    raw = struct.pack('<'+str(len(words))+'I',*(word & 0xffffffff for word in words))
    return source.encode(),raw,dict(load_bytes=dma_load_bytes(m,k,n,traffic_reuse=True),store_bytes=2*m*n)


def prepare(args):
    start = time.monotonic()
    source_root, out, model_root = map(checked,(args.assets,args.out,args.model_root))
    if out.exists():
        raise ValueError('fresh reuse asset destination required')
    original = KernelAssets(source_root,'tiny-full')
    if original.provenance.get('program_policy','original') != 'original':
        raise ValueError('reuse preparation requires the original captured programs')
    provenance = json.loads(original.read('provenance.json'))
    if original.provenance != dict(provenance,schema='atlas.full_kernel_assets.v1'):
        raise ValueError('original staged/full metadata differs')
    contract(SimpleNamespace(provenance=provenance,read=original.read))
    _,_,assemble,_ = configure_model(model_root)
    list(assemble(io.StringIO('ecall\n')).assemble())
    before = model_binding(model_root)
    pins = {str(ROOT/kernel_library.relative_path(name)):digest for name,digest in code_sources(traffic_reuse=True).items()}
    for path in (Path(__file__),ROOT/'tools/host_iss/full_bundle.py',ROOT/'tools/host_iss/bundle.py',
                 ROOT/'tools/host_iss/tiny_full_contract.py',ROOT/'tools/host_iss/placement.py',
                 ROOT/'tools/chipyard/arena.py',ROOT/'runtime/atlas_host_layout.h'):
        pins[str(path)] = sha256(path)
    manifest_hash = sha256(source_root/'SHA256SUMS')
    cache, variants, kernels = {}, [], []
    for item in original.kernels:
        key = (*item['physical_mkn'],item['output_offset'],item['rescale'])
        if key not in cache:
            source,raw,requested = encode(item,assemble)
            cache[key] = (source,raw,requested)
            variants.append(dict(physical_mkn=item['physical_mkn'],rescale=item['rescale'],
                                 output_offset=item['output_offset'],program_sha256=hashlib.sha256(raw).hexdigest(),
                                 assembly_sha256=hashlib.sha256(source).hexdigest(),words=len(raw)//4,
                                 requested_dma=requested))
        source,raw,requested = cache[key]
        row = dict(item,source='programs/'+hashlib.sha256(source).hexdigest()+'.S',
                   original='programs/'+hashlib.sha256(raw).hexdigest()+'.bin',requested_dma=requested)
        kernels.append(row)
    out.mkdir(parents=True)
    sums = dict(original.sums)
    for name in original.sums:
        path = checked(out/name);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(original.read(name))
    for source,raw,_ in cache.values():
        for payload,suffix in ((source,'.S'),(raw,'.bin')):
            digest = hashlib.sha256(payload).hexdigest(); name = 'programs/'+digest+suffix
            (out/name).write_bytes(payload);sums[name] = digest
    model_after = model_binding(model_root)
    if model_after != before or any(sha256(checked(path)) != digest for path,digest in pins.items()):
        raise ValueError('reuse generator or assembler changed')
    original.verify_bookends()
    if sha256(source_root/'SHA256SUMS') != manifest_hash:
        raise ValueError('original asset manifest changed')
    reuse = dict(schema='atlas.tiny_full_reuse.v1',source_assets=str(source_root),
                 source_manifest_sha256=manifest_hash,generator_source_sha256=pins,model=model_after,
                 launch_count=len(kernels),variant_count=len(cache),variants=variants,
                 captured_data_unchanged=True,instruction_execution=False,numerical_qualification=False)
    documents = {name:json.loads(original.read(name)) for name in ('contract.json','provenance.json','kernels.json')}
    for document in documents.values():
        document.update(program_policy=POLICY,kernels=kernels,reuse=reuse)
    for name in ('contract.json','provenance.json','kernels.json'):
        if name != 'contract.json':
            documents[name]['contract_sha256'] = sums['contract.json']
        raw = json.dumps(documents[name],indent=2).encode()+b'\n'
        (out/name).write_bytes(raw);sums[name] = hashlib.sha256(raw).hexdigest()
    (out/'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name,digest in sorted(sums.items())))
    prepared = KernelAssets(out,'tiny-full')
    contract(SimpleNamespace(provenance=json.loads(prepared.read('provenance.json')),read=prepared.read))
    prepared.verify_bookends()
    receipt = dict(status='PASS_PREPARED_TINY_FULL_REUSE_ASSETS_UNEXECUTED',program_policy=POLICY,
                   source_manifest_sha256=manifest_hash,output_manifest_sha256=sha256(out/'SHA256SUMS'),
                   launch_count=len(kernels),variant_count=len(cache),max_program_words=max(v['words'] for v in variants),
                   source_bookends_verified=True,model=model_after,generator_source_sha256=pins,
                   captured_data_unchanged=True,instruction_execution=False,numerical_qualification=False,
                   elapsed_seconds=time.monotonic()-start,cpu_affinity=sorted(os.sched_getaffinity(0)))
    (out/'reuse-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assets','model-root','out'):
        parser.add_argument('--'+name,type=checked,required=True)
    print(json.dumps(prepare(parser.parse_args()),indent=2))


if __name__ == '__main__':
    main()
