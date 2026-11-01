#!/usr/bin/env python3
"""Capture ResNet50 fused device launches from exact staged RTL-model arenas.

Frozen v1 tensors select compile-time scales only. Every v2 graph edge comes
from the new device result, with host byte gathers, maxpool, average, and bias.
"""

import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from partition_resnet import epilogue_partition

PLAN = Path('/bwrcq/C/reednicolas/ee194-sp26-chipyard/generators/sp26-atlas-acc/compiler/sessions/oct18/atlas-sota/resnet/full-host-plan/host-plan.json')
REFERENCE = PLAN.parent.parent / 'full-reference/receipt.json'
PKG = Path('/tmp/reednicolas-atlas/oct31/STAGE/pkg/resnet50')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(path):
    path = Path(path).absolute()
    for part in path.parts:
        if any(token in part.lower() for token in ('hammer', 'vlsi')):
            raise ValueError('restricted path')
    for parent in [*reversed(path.parents), path]:
        if parent.is_symlink():
            raise ValueError('source symlink')
    return path


def tiled_fp8(matrix):
    m,k = matrix.shape
    return matrix.reshape(m//32,32,k//32,32).transpose(0,2,1,3).copy().tobytes()


def bf16_tiles(matrix):
    m,n = matrix.shape
    raw = matrix.contiguous().view(torch.uint16).numpy()
    return b''.join(raw[r:r+32,c:c+16].copy().tobytes()+raw[r:r+32,c+16:c+32].copy().tobytes()
                    for r in range(0,m,32) for c in range(0,n,32))


def bf16_decode(raw,m,n):
    data = np.frombuffer(raw,dtype='<u2')
    out = np.empty((m,n),np.uint16)
    p = 0
    for r in range(0,m,32):
        for c in range(0,n,32):
            out[r:r+32,c:c+16] = data[p:p+512].reshape(32,16)
            out[r:r+32,c+16:c+32] = data[p+512:p+1024].reshape(32,16)
            p += 1024
    return (out.astype(np.uint32)<<16).view(np.float32)


def byte_gather(raw, desc, launch):
    c,h,w = desc['channels'],desc['h'],desc['w']
    m,k = launch['m'],launch['k']
    kernel,stride,pad = launch['kernel'],launch['stride'],launch['padding']
    oh,ow = (h+2*pad-kernel)//stride+1,(w+2*pad-kernel)//stride+1
    source = np.frombuffer(raw,np.uint8).reshape(-1,desc['padded_channels'])[:h*w,:c].reshape(h,w,c)
    result = np.zeros((m,k),np.uint8)
    positions = np.arange(oh*ow)
    for channel in range(c):
        for kr in range(kernel):
            for kc in range(kernel):
                ih,iw = positions//ow*stride+kr-pad,positions%ow*stride+kc-pad
                valid = (ih>=0)&(ih<h)&(iw>=0)&(iw<w)
                result[positions[valid],(channel*kernel+kr)*kernel+kc] = source[ih[valid],iw[valid],channel]
    return tiled_fp8(result)


def maxpool(raw, desc):
    h,w,c = desc['h'],desc['w'],desc['padded_channels']
    source = np.frombuffer(raw,np.uint8).reshape(-1,c)[:h*w].reshape(h,w,c)
    if np.any(source & 128):
        raise ValueError('byte maxpool requires the fused nonnegative ReLU output')
    oh,ow = (h+1)//2,(w+1)//2
    result = np.zeros((((oh*ow+31)//32)*32,c),np.uint8)
    for r in range(oh):
        for col in range(ow):
            patch = source[max(0,r*2-1):min(h,r*2+2),max(0,col*2-1):min(w,col*2+2)]
            result[r*ow+col] = patch.max(axis=(0,1))
    return result.tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--atlas-tools',type=Path,required=True)
    parser.add_argument('--model-root',type=Path,required=True)
    parser.add_argument('--plan',type=Path,default=PLAN)
    parser.add_argument('--reference',type=Path,default=REFERENCE)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path[:0] = [str(checked(args.atlas_tools)),str(checked(args.model_root))]
    from atlas_epilogue import Epilogue, Step, TileInput
    from atlas_program import assembly, dma_load_bytes
    from atlas_resnet_input import DirectInput, relocate_assembly
    from atlas_rtl_reference import RtlReference
    from atlas_tile_arithmetic import tensor_scale_exponent
    from npu_model.configs.numerics import RtlNumerics
    from npu_model.util.converter import input_to_program
    out = checked(args.out)
    out.mkdir(parents=True,exist_ok=args.resume)
    pins = {}

    def read(path, expected=None):
        path = checked(path)
        raw = path.read_bytes()
        digest = sha(raw)
        if expected is not None and digest != expected:
            raise ValueError('source digest differs: '+str(path))
        if str(path) in pins and pins[str(path)] != digest:
            raise ValueError('source changed: '+str(path))
        pins[str(path)] = digest
        return raw

    def write(path,raw):
        path = out/path
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(raw)
        return str(path)

    plan_raw = read(args.plan,'c6909725e91a2bef6139dac6782d853304d71d99db224cf2db53dbd2facb8aaa')
    plan = json.loads(plan_raw)
    old = json.loads(read(args.reference))
    nodes = plan['nodes']
    by_node = {node['node']:node for node in nodes}
    old_nodes = {node['node']:node for node in old['nodes']}
    fusion = epilogue_partition(nodes)
    old_sites = {item['site']:item for item in old['kernels']}
    position = {node['node']:i for i,node in enumerate(nodes)}
    tensors = {}
    actions = []
    input_path = write(Path('input.f32.bin'),read(PKG/'data/graph_entry.bin',plan['image']['sha256']))
    model = dict(schema='atlas.resnet50.device_fp8.v2',arena_base=0x90000000,
                 input=dict(path=input_path,shape=[3,224,224],dtype='f32_nchw',tensor='input'),
                 tensors=[],nodes=actions,launches=[],expected_logits=str(out/'logits.f32.bin'),output_tensor='fc')
    launches = {}
    for fused in fusion:
        node = by_node[fused['node']]
        baseline_index = fused['baseline_index']
        is_conv = node['operator']=='Conv2d'
        tail = by_node[fused['output_node']] if is_conv else node
        shape = tail['output_shape']
        channels,h,w = (shape[1],shape[2],shape[3]) if is_conv else (shape[1],1,1)
        m,k,n = node['padded_mkn']
        old_site = old_sites[node['target']]
        input_node = 'input' if fused['input_node']=='x' else 'avgpool' if fused['input_node']=='flatten' else fused['input_node']
        scale = 0
        if is_conv:
            saved = old_nodes[fused['output_node']]
            if 'payload' in saved:
                value = np.frombuffer(read(saved['payload']['path'],saved['payload']['sha256']),np.float32)
            else:
                payload = old_site['expected_physical_output']
                raw = read(payload['path'],payload['sha256'])
                value = bf16_decode(raw,m,n)[:node['matrix_mkn'][0],:channels].T.reshape(-1)
                bn = by_node[fused['bn_node']]['host_params']['scale_then_bias']
                coeff = np.frombuffer(read(bn['path'],bn['sha256']),np.float32)
                value = np.add(np.multiply(value.reshape(channels,-1),coeff[:channels,None],dtype=np.float32),
                               coeff[channels:,None],dtype=np.float32).reshape(-1)
                if sha(value.tobytes()) != saved['output']['sha256']:
                    raise ValueError('reconstructed BN scale source differs')
            if not np.isfinite(value).all():
                raise ValueError('nonfinite captured scale source')
            bits = torch.from_numpy(value.copy()).to(torch.bfloat16).view(torch.uint16).numpy()
            scale = tensor_scale_exponent([int((bits&32767).max())])
        output_id = fused['output_node']
        tensors[output_id] = dict(id=output_id,h=h,w=w,channels=channels,padded_channels=n,
            scale_exponent=scale,dtype='fp8_nhwc' if is_conv else 'bf16_tiles',storage_bytes=m*n*(1 if is_conv else 2),physical_rows=m)
        geometry = node.get('geometry',{})
        kernel = geometry.get('kernel_size',[1])[0]
        stride = geometry.get('stride',[1])[0]
        padding = geometry.get('padding',[0])[0]
        launch = dict(site=node['target'],baseline_index=baseline_index,input_tensor=input_node,output_tensor=output_id,
            m=m,k=k,n=n,logical_m=node['matrix_mkn'][0],logical_k=node['matrix_mkn'][1],logical_n=node['matrix_mkn'][2],
            kernel=kernel,stride=stride,padding=padding,a_offset=0,a_bytes=m*k,weight_offset=m*k,
            b_scale_exponent=old_site['b_scale_exponent'],a_scale_exponent=old_site['a_scale_exponent'],
            output_offset=m*k+k*n+256,output_bytes=tensors[output_id]['storage_bytes'],
            output_dtype='fp8' if is_conv else 'bf16',output_layout='channels_last' if is_conv else 'tiles',
            output_scale_exponent=scale,dynamic_regions=[],arena_regions=[],_fusion=fused,_node=node,
            _position=position[node['node']])
        launches[fused['node']] = launch
        dependencies = [input_node] + ([fused['residual_node']] if fused['residual_node'] else [])
        actions.append(dict(op='launch',_key=fused['node'],_deps=dependencies,_output=output_id,_position=position[node['node']]))
    stem = next(item for item in fusion if item['node']=='conv1')['output_node']
    tensors['maxpool'] = dict(id='maxpool',h=56,w=56,channels=64,padded_channels=64,
        scale_exponent=tensors[stem]['scale_exponent'],dtype='fp8_nhwc',storage_bytes=3136*64,physical_rows=3136)
    tensors['avgpool'] = dict(id='avgpool',h=1,w=1,channels=2048,padded_channels=2048,
        scale_exponent=0,dtype='f32_vector',storage_bytes=2048*4)
    tensors['fc'] = dict(id='fc',h=1,w=1,channels=1000,padded_channels=1000,
        scale_exponent=0,dtype='f32_vector',storage_bytes=4000)
    actions += [dict(op='maxpool',input=stem,output='maxpool',kernel=3,stride=2,padding=1,
                     _deps=[stem],_output='maxpool',_position=position['maxpool']),
                dict(op='avgpool',input=by_node['avgpool']['inputs'][0],output='avgpool',
                     _deps=[by_node['avgpool']['inputs'][0]],_output='avgpool',_position=position['avgpool'])]
    bias = by_node['fc']['host_bias']
    bias_path = write(Path('fc-bias.f32.bin'),read(bias['path'],bias['sha256']))
    model['fc_bias'] = bias_path
    actions.append(dict(op='fc_bias',input='fc_acc',output='fc',path=bias_path,
                        _deps=['fc_acc'],_output='fc',_position=position['fc']+1))
    remaining = list(actions)
    actions.clear()
    available = {'input'}
    while remaining:
        ready = sorted((item for item in remaining if all(dep in available for dep in item['_deps'])),key=lambda item:item['_position'])
        if not ready:
            raise ValueError('fused graph has an unresolved dependency')
        item = ready[0]
        remaining.remove(item)
        available.add(item['_output'])
        if item['op']=='launch':
            launch = launches[item['_key']]
            launch['index'] = len(model['launches'])
            model['launches'].append(launch)
            item['launch'] = launch['index']
        actions.append({k:v for k,v in item.items() if not k.startswith('_')})
    model['tensors'] = list(tensors.values())
    model['partition'] = fusion
    for module in list(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name and str(name).endswith('.py') and any(str(name).startswith(str(root)+'/') for root in (args.atlas_tools,args.model_root,Path(__file__).parent)):
            read(name)
    model['source_inputs_sha256'] = pins
    (out/'partition.json').write_text(json.dumps(fusion,indent=2)+'\n')
    values = {}
    manifest = dict(schema='atlas.resnet50.device_fp8.launches.v2',launches=[])
    engine = None if args.prepare_only else RtlReference(model['arena_base'])
    started = time.monotonic()
    try:
        for action in actions:
            op = action['op']
            if op=='maxpool':
                if not args.prepare_only:
                    values[action['output']] = maxpool(values[action['input']],tensors[action['input']])
                continue
            if op=='avgpool':
                if not args.prepare_only:
                    desc = tensors[action['input']]
                    raw = np.frombuffer(values[action['input']],np.uint8).reshape(-1,desc['padded_channels'])[:desc['h']*desc['w'],:desc['channels']]
                    decoded = RtlNumerics.from_fp8(torch.from_numpy(raw.copy()).view(torch.float8_e4m3fn),127+desc['scale_exponent']).float().numpy()
                    total = np.zeros(desc['channels'],np.float32)
                    for row in decoded:
                        total = np.add(total,row,dtype=np.float32)
                    values[action['output']] = np.divide(total,np.float32(len(decoded)),dtype=np.float32).tobytes()
                continue
            if op=='fc_bias':
                if not args.prepare_only:
                    raw = values[action['input']]
                    value = bf16_decode(raw,32,1024)[0,:1000]
                    logits = np.add(value,np.frombuffer(read(bias_path),np.float32),dtype=np.float32)
                    values['fc'] = logits.tobytes()
                    write(Path('logits.f32.bin'),logits.tobytes())
                continue
            launch = model['launches'][action['launch']]
            fused,node = launch['_fusion'],launch['_node']
            index,m,k,n = launch['index'],launch['m'],launch['k'],launch['n']
            is_conv = node['operator']=='Conv2d'
            directory = Path(f'kernel-{index:02d}')
            weight_id = node['weight']['fp8_tiled']
            weights = read(weight_id['path'],weight_id['sha256'])
            launch['weights'] = write(directory/'weight.bin',weights)
            initial = bytearray(bytes(m*k)+weights+bytes([0xA5])*256+
                                bytes([0xCD])*launch['output_bytes']+bytes([0x5A])*256)
            steps = []
            def region(raw,name):
                initial.extend(bytes((-len(initial))%64))
                offset = len(initial)
                initial.extend(raw)
                path = write(directory/name,raw)
                launch['arena_regions'].append(dict(offset=offset,path=path))
                return offset
            if is_conv:
                bn = by_node[fused['bn_node']]['host_params']['scale_then_bias']
                coefficients = torch.frombuffer(bytearray(read(bn['path'],bn['sha256'])),dtype=torch.float32).to(torch.bfloat16)
                for kind,vector in [('mul',coefficients[:n]),('add',coefficients[n:])]:
                    offset = region(bf16_tiles(vector.expand(32,n).contiguous()),'bn-'+kind+'.bin')
                    steps.append(Step(kind,TileInput(offset,'channel')))
            incoming = tensors.get(launch['input_tensor'])
            direct = None
            if launch['input_tensor']=='input':
                launch['input_mode']='stem_f32'
                launch['geometry']=[3,224,224,launch['kernel'],launch['stride'],launch['padding']]
                if not args.prepare_only:
                    image = np.frombuffer(read(input_path),np.float32).reshape(3,224,224)
                    source = torch.from_numpy(image.copy()).to(torch.bfloat16)
                    encoded = RtlNumerics.to_fp8(source,127+launch['a_scale_exponent']).view(torch.uint8).numpy()
                    desc = dict(channels=3,padded_channels=3,h=224,w=224)
                    initial[:m*k] = byte_gather(encoded.transpose(1,2,0).copy().tobytes(),desc,launch)
            elif not is_conv:
                launch['input_mode']='fc_f32'
                launch['geometry']=[2048,1,1,1,1,0]
                if not args.prepare_only:
                    value = torch.frombuffer(bytearray(values[launch['input_tensor']]),dtype=torch.float32).to(torch.bfloat16)
                    encoded = RtlNumerics.to_fp8(value,127+launch['a_scale_exponent']).view(torch.uint8).numpy()
                    matrix = np.zeros((m,k),np.uint8)
                    matrix[0,:2048] = encoded
                    initial[:m*k] = tiled_fp8(matrix)
            elif launch['kernel']==1:
                launch['input_mode']='device_gather_fp8'
                source = bytes(incoming['storage_bytes']) if args.prepare_only else values[launch['input_tensor']]
                source_offset = region(source,'source-placeholder.bin')
                # Dynamic source bytes are overwritten by the live host, never treated as constants.
                launch['arena_regions'].pop()
                zero_offset = region(bytes(32),'zero-line.bin')
                direct = DirectInput(source_offset,incoming['padded_channels'],incoming['h'],incoming['w'],launch['stride'],zero_offset)
                launch['direct_input'] = asdict(direct)
                launch['input_source_offset'] = source_offset
                launch['geometry']=[incoming['channels'],incoming['h'],incoming['w'],launch['kernel'],launch['stride'],launch['padding']]
                launch['a_scale_exponent'] = incoming['scale_exponent']
            else:
                launch['input_mode']='host_gather_fp8'
                launch['geometry']=[incoming['channels'],incoming['h'],incoming['w'],launch['kernel'],launch['stride'],launch['padding']]
                launch['a_scale_exponent'] = incoming['scale_exponent']
                if not args.prepare_only:
                    initial[:m*k] = byte_gather(values[launch['input_tensor']],incoming,launch)
            launch['input_scale_exponent'] = launch['a_scale_exponent']
            if fused['residual_node']:
                residual_desc = tensors[fused['residual_node']]
                residual = bytes(residual_desc['storage_bytes']) if args.prepare_only else values[fused['residual_node']]
                offset = region(residual,'residual-placeholder.bin')
                launch['arena_regions'].pop()
                launch['dynamic_regions'].append(dict(offset=offset,tensor=fused['residual_node']))
                steps.append(Step('add',TileInput(offset,'channels_last','fp8',residual_desc['scale_exponent'])))
            if fused['relu']:
                steps.append(Step('relu'))
            e = Epilogue(m,n,launch['output_offset'],tuple(steps),'fp8',launch['output_scale_exponent'],'channels_last') if is_conv else None
            launch['epilogue'] = asdict(e) if e else None
            launch['arena_bytes'] = len(initial)
            source = assembly(m,k,n,launch['output_offset'],launch['a_scale_exponent']+launch['b_scale_exponent'],
                              traffic_reuse=True,epilogue=e,direct_input=direct)
            source = relocate_assembly(source,model['arena_base'],arena_bytes=len(initial))
            program = input_to_program(io.StringIO(source))
            raw_program = b''.join(word.to_bytes(4,'little') for word in program.assemble())
            if len(raw_program)+20 > 131072:
                raise ValueError('patched program exceeds IMEM')
            launch['source'] = write(directory/'program.S',source.encode())
            launch['program'] = write(directory/'program.bin',raw_program)
            launch['initial'] = write(directory/'initial-dram.bin',bytes(initial))
            launch['expected'] = str(out/directory/'expected.bin')
            launch['requested_dma'] = dict(load_bytes=dma_load_bytes(m,k,n,traffic_reuse=True,epilogue=e,direct_input=direct),store_bytes=launch['output_bytes'])
            launch['program_words'] = len(raw_program)//4
            if not args.prepare_only:
                expected,stats = engine.run(source,bytes(initial),launch['output_offset'],launch['output_bytes'],
                                            launch['requested_dma'],fp8_scale=launch['output_scale_exponent'] if is_conv else None)
                write(directory/'expected.bin',expected)
                values[launch['output_tensor']] = expected
                launch['reference'] = stats
                print(json.dumps(dict(index=index,site=launch['site'],seconds=time.monotonic()-started,**stats)),flush=True)
            manifest['launches'].append(dict(name=launch['site'],source=str(directory/'program.S'),program=str(directory/'program.bin'),
                input=str(directory/'initial-dram.bin'),expected=str(directory/'expected.bin'),arena_base=model['arena_base'],
                output_offset=launch['output_offset'],requested_dma=launch['requested_dma']))
            model['arena_bytes'] = max(model.get('arena_bytes',0),len(initial))
            public = {key:val for key,val in model.items()}
            public['launches'] = [{key:val for key,val in item.items() if not key.startswith('_')} for item in model['launches']]
            public['status'] = 'PREPARED_NOT_EXECUTED' if args.prepare_only else 'CAPTURING'
            (out/'model.json').write_text(json.dumps(public,indent=2)+'\n')
        if not args.prepare_only:
            baseline = np.frombuffer(read(PKG/'data/graph_expected.bin'),np.float32)
            current = np.frombuffer(values['fc'],np.float32)
            model['comparison_v1'] = dict(images=1,v1_top1=int(baseline.argmax()),v2_top1=int(current.argmax()),
                max_abs_logit_error=float(np.abs(current-baseline).max()),l2_logit_error=float(np.linalg.norm(current.astype(np.float64)-baseline)),
                changed_logits=int((current.view(np.uint32)!=baseline.view(np.uint32)).sum()),
                v1_logits_sha256=sha(baseline.tobytes()),v2_logits_sha256=sha(current.tobytes()))
        for name,digest in pins.items():
            if sha(checked(name).read_bytes()) != digest:
                raise ValueError('capture source bookend changed')
        model['source_bookends_equal'] = True
        model['status'] = 'PREPARED_NOT_EXECUTED' if args.prepare_only else 'CAPTURED_RTL_FUNCTIONAL'
        model['launches'] = [{key:val for key,val in item.items() if not key.startswith('_')} for item in model['launches']]
        model['command'] = [sys.executable,*sys.argv]
        model['reference_mode'] = 'npu_model physical instructions, numerics=rtl, exact staged arenas; final timed QEMU gate required'
        (out/'model.json').write_text(json.dumps(model,indent=2)+'\n')
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    finally:
        if engine:
            engine.close()


if __name__=='__main__':
    main()
