#!/usr/bin/env python3
"""Run two small actual original-model captures through the shipped worker."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked
from smol_capture.capture import interpreter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',type=interpreter,default=Path(sys.executable))
    parser.add_argument('--atlas-root',type=checked,default=Path(__file__).absolute().parents[1])
    parser.add_argument('--model-root',type=checked,required=True)
    parser.add_argument('--model-pin',required=True)
    parser.add_argument('--out',type=checked,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('fresh output directory required')
    args.out.mkdir(parents=True)
    import numpy as np
    from atlas_program import assembly
    from chipyard.arena import relocate_assembly
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',ATLAS_NPU_MODEL_ROOT=str(args.model_root),ATLAS_NPU_MODEL_PIN=args.model_pin)
    results=[]
    for backend in ('attention','projection'):
        folder=args.out/backend;folder.mkdir()
        command=[str(args.python),'-B',str(args.atlas_root/'tools/smol_capture/worker.py'),'--backend',backend,
                 '--workspace',str(args.out),'--atlas-root',str(args.atlas_root)]
        with (folder/'worker.log').open('w') as log:
            worker=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env=env)
            index=0
            def call(op,**kwargs):
                nonlocal index
                index+=1
                worker.stdin.write(json.dumps(dict(id=index,op=op,**kwargs))+'\n');worker.stdin.flush()
                while True:
                    line=worker.stdout.readline()
                    if not line:raise ValueError('worker exited, inspect '+str(folder/'worker.log'))
                    try:reply=json.loads(line)
                    except ValueError:continue
                    if reply.get('id')==index:break
                if reply.get('ok') is not True:raise ValueError(reply)
                return reply['result']
            try:
                if backend=='attention':
                    q=np.zeros((1,32,64),dtype=np.uint16);v=np.full_like(q,0x3f80)
                    np.savez(folder/'input.npz',q=q,k=q,v=v)
                    prepared=call('prepare_attention',input_npz=str(folder/'input.npz'),output_dir=str(folder),
                                  config=dict(query_rows=32,keys=32,query_heads=1,kv_heads=1,head_dim=64,mask='none'))
                    result=call('execute_attention',config=prepared['config'],offsets=prepared['offsets'],
                                initial_arena=prepared['arena_initial'],output_dir=str(folder))
                    expected=np.full((32,64),0x3f80,dtype='<u2').tobytes()
                else:
                    source=relocate_assembly(assembly(32,32,32,2048,0),0x100000000,0x90000000)
                    (folder/'program.S').write_text(source)
                    (folder/'initial.bin').write_bytes(bytes([0x38])*2048+bytes(2048))
                    result=call('execute_projection',program_source=str(folder/'program.S'),initial_arena=str(folder/'initial.bin'),
                                output_offset=2048,output_bytes=2048,output_dir=str(folder))
                    expected=struct.pack('<1024H',*[0x4200]*1024)
                if checked(result['output_path']).read_bytes()!=expected:raise ValueError('known constant output differs: '+backend)
                if result['provenance']['model']['numerics']!='rtl':raise ValueError('RTL numerics required')
                results.append(dict(backend=backend,known_output_exact=True,**result))
            finally:
                worker.stdin.close();worker.wait(timeout=20)
                if worker.returncode:raise ValueError('worker exit '+str(worker.returncode))
    (args.out/'result.json').write_text(json.dumps(dict(status='PASS_SMALL_ORIGINAL_MODEL_CAPTURE',results=results),indent=2)+'\n')


if __name__=='__main__':main()
