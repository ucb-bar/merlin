#!/usr/bin/env python3
"""Package a captured ResNet50 device plan as a board-support target, without building."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).absolute().parent
RUNTIME = PACKAGE.parents[1]/'runtime'
sys.path.insert(0, str(PACKAGE.parent))
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked

DTYPES={'fp8_nhwc':'M2_FP8_NHWC','f32_vector':'M2_F32_VECTOR','bf16_tiles':'M2_BF16_TILES','f32_nchw':'M2_F32_NCHW'}
CMAKE='''set(_r "${CMAKE_CURRENT_SOURCE_DIR}")
file(GLOB _m2_data "${_r}/data/*.bin")
add_executable(atlas-resnet50 ${_r}/host.c ${_r}/assets.c ${_r}/images.S ${_r}/runtime/atlas_host.c)
target_include_directories(atlas-resnet50 PRIVATE ${_r} ${_r}/runtime)
target_compile_options(atlas-resnet50 PRIVATE -ffp-contract=off -fno-fast-math -fno-tree-vectorize "-Wa,-I${_r}/data")
set_source_files_properties(${_r}/images.S PROPERTIES OBJECT_DEPENDS "${_m2_data}")
target_link_libraries(atlas-resnet50 PRIVATE -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive)
target_link_libraries(atlas-resnet50 PRIVATE chip-config rocketcore)
option(RESNET50_QEMU "ResNet v2 QEMU validation console and finish" OFF)
option(RESNET50_VERIFY "ResNet v2 exact staging, output, and preservation checks" ON)
if(RESNET50_VERIFY)
 target_compile_definitions(atlas-resnet50 PRIVATE M2_VERIFY=1)
else()
 target_compile_definitions(atlas-resnet50 PRIVATE M2_VERIFY=0)
endif()
if(RESNET50_QEMU)
 target_sources(atlas-resnet50 PRIVATE ${_r}/qemu_console.c)
 target_compile_definitions(atlas-resnet50 PRIVATE RESNET50_QEMU_HARNESS)
endif()

option(RESNET50_FAST "ResNet immutable B source transport" OFF)
if(RESNET50_FAST)
 target_compile_definitions(atlas-resnet50 PRIVATE M2_FAST=1)
endif()
'''

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def integer(value,name,minimum=0):
    if type(value) is not int or value<minimum:
        raise ValueError('invalid '+name)
    return value

def round32(n):
    return (n+31)//32*32

def package(model_path,out):
    model_path=checked(model_path)
    model_raw=model_path.read_bytes()
    model=json.loads(model_raw)
    out=checked(out)
    if out.exists():
        raise ValueError('package output already exists')
    out.mkdir(parents=True)
    (out/'data').mkdir()
    assembly=['.section .rodata.m2,"a",@progbits']
    declarations=[]
    input_pins={}
    embedded={}
    symbols={}
    def resolve(path):
        return checked(path if Path(path).is_absolute() else model_path.parent/path)
    def asset(path,ctype='uint8_t'):
        if isinstance(path,dict): path=path['path']
        source=resolve(path)
        raw=source.read_bytes()
        sha=digest(raw)
        input_pins[str(source)]=sha
        key=(sha,ctype)
        if key in symbols: return symbols[key],len(raw)
        name=f'asset_{len(embedded):03d}.bin'
        (out/'data'/name).write_bytes(raw)
        symbol='m2_'+name[:-4]
        assembly.extend(['.balign 64',f'.globl {symbol}',symbol+':',f'.incbin "{name}"'])
        declarations.append(f'extern const {ctype} {symbol}[];')
        embedded[name]={'source':str(source),'sha256':sha,'bytes':len(raw)}
        symbols[key]=symbol
        return symbol,len(raw)
    tensors=[]
    ids={}
    tensor_defs=[]
    tensor_entries=[]
    def add_tensor(item,input_asset=None):
        tensor=dict(item)
        name=str(tensor['id'])
        if name in ids: raise ValueError('duplicate tensor '+name)
        dtype=tensor['dtype']
        if dtype not in DTYPES: raise ValueError('tensor dtype')
        h=integer(tensor.get('h',1),'tensor h',1)
        w=integer(tensor.get('w',1),'tensor w',1)
        channels=integer(tensor['channels'],'channels',1)
        padded=integer(tensor.get('padded_channels',channels),'padded channels',channels)
        m=integer(tensor.get('m',tensor.get('physical_rows',round32(h*w))),'tensor m',h*w)
        expected=(m*padded*(2 if dtype=='bf16_tiles' else 1) if dtype in ('fp8_nhwc','bf16_tiles')
                  else channels*4 if dtype=='f32_vector' else h*w*channels*4)
        size=integer(tensor.get('storage_bytes',expected),'tensor bytes',expected)
        scale=integer(tensor.get('scale_exponent',0)+126,'tensor scale')-126
        if scale>127: raise ValueError('tensor scale')
        index=len(tensors)
        ids[name]=index
        tensor.update(id=name,h=h,w=w,channels=channels,padded_channels=padded,m=m,storage_bytes=size)
        tensors.append(tensor)
        symbol=input_asset or f'm2_tensor_{index}'
        if input_asset is None:
            tensor_defs.append(f'static uint8_t {symbol}[{size}] __attribute__((aligned(64)));')
        tensor_entries.append(f'  {{(uint8_t *){symbol},{size},{h},{w},{channels},{padded},{m},{DTYPES[dtype]},{scale}}},')
    inp=model['input']
    shape=inp.get('shape')
    if shape:
        channels,h,w=shape[-3:]
    else:
        channels=inp['channels'];h=inp.get('height',inp.get('h'));w=inp.get('width',inp.get('w'))
    input_symbol,input_bytes=asset(inp['path'])
    if input_bytes!=channels*h*w*4: raise ValueError('input shape/bytes differ')
    add_tensor(dict(id=inp.get('tensor','input'),dtype='f32_nchw',channels=channels,h=h,w=w,storage_bytes=input_bytes),input_symbol)
    for tensor in model['tensors']:
        if str(tensor['id'])==inp.get('tensor','input'):
            continue
        add_tensor(tensor)
    bias_symbol,bias_bytes=asset(model['fc_bias'],'float')
    expected_path=model.get('expected_logits',model.get('expected'))
    expected_symbol,expected_bytes=asset(expected_path,'uint32_t')
    if expected_bytes%4 or bias_bytes<expected_bytes: raise ValueError('FC bias or logits extent')
    logits=expected_bytes//4
    launches=[]
    fast_launches=[]
    initial_symbols=[]
    dynamic_defs=[]
    maximum_arena=0
    arena_base=integer(model.get('arena_base',0x90000000),'arena base')
    if arena_base!=0x90000000: raise ValueError('ResNet package must retain the low arena')
    for index,launch in enumerate(model['launches']):
        if launch.get('index',index)!=index: raise ValueError('launches must be sequential')
        program,program_bytes=asset(launch['program'],'uint32_t')
        program_raw=resolve(launch['program']).read_bytes()
        if program_bytes%4 or program_raw[-4:]!=bytes.fromhex('73000000') or program_bytes+20>131072:
            raise ValueError('raw program terminal/capacity')
        if launch.get('source'):
            source=resolve(launch['source'])
            input_pins[str(source)]=digest(source.read_bytes())
        initial,arena_bytes=asset(launch['initial'])
        expected,out_bytes=asset(launch['expected'])
        if arena_bytes!=launch['arena_bytes'] or out_bytes!=launch['output_bytes']:
            raise ValueError('launch arena/output bytes')
        maximum_arena=max(maximum_arena,arena_bytes)
        if arena_base+arena_bytes>2**32: raise ValueError('low DMA arena overflow')
        offset=integer(launch['output_offset'],'output offset')
        if offset%32 or offset+out_bytes>arena_bytes: raise ValueError('output extent/alignment')
        src=ids[str(launch['input_tensor'])];dst=ids[str(launch['output_tensor'])]
        if out_bytes!=tensors[dst]['storage_bytes']: raise ValueError('output tensor storage extent')
        pool=launch.get('max_pool')
        m,k,n=[integer(launch[d],d,0 if pool and d=='k' else 32) for d in ('m','k','n')]
        if any(d%32 for d in (m,k,n)) or bool(pool)!=(k==0): raise ValueError('physical shape alignment')
        kernel=integer(launch.get('kernel',1),'kernel',1)
        stride=integer(launch.get('stride',1),'stride',1)
        padding=integer(launch.get('padding',0),'padding')
        logical_m=integer(launch.get('logical_m',tensors[dst]['h']*tensors[dst]['w']),'logical m',1)
        logical_k=integer(launch.get('logical_k',tensors[src]['channels']*kernel*kernel),'logical k',0 if pool else 1)
        logical_n=integer(launch.get('logical_n',tensors[dst]['channels']),'logical n',1)
        if logical_m>m or logical_k>k or logical_n>n:
            raise ValueError('logical dimensions exceed physical dimensions')
        a_offset=integer(launch.get('a_offset',0),'A offset')
        a_bytes=integer(launch.get('a_bytes',m*k),'A bytes')
        if a_offset%32 or a_offset+a_bytes>arena_bytes or a_bytes!=m*k: raise ValueError('A extent/alignment')
        direct=launch.get('direct_input')
        direct_offset=launch.get('input_source_offset',direct['offset'] if direct else pool['source_offset'] if pool else 0)
        if pool:
            # The device pools the whole producer tensor, staged like a direct input.
            source_tensor,output_tensor=tensors[src],tensors[dst]
            geometry=(source_tensor['padded_channels'],source_tensor['h'],source_tensor['w'])
            if (source_tensor['dtype']!='fp8_nhwc' or source_tensor['scale_exponent']!=output_tensor['scale_exponent']
                    or (pool['channels'],pool['height'],pool['width'])!=geometry or launch.get('dynamic_regions')
                    or (pool['source_offset'],pool['output_offset'])!=(direct_offset,offset)
                    or direct_offset+source_tensor['storage_bytes']>arena_bytes or source_tensor['storage_bytes']!=4*out_bytes):
                raise ValueError('device max pool geometry, extent, or dtype')
            direct=pool
        elif direct:
            if tensors[src]['dtype']!='fp8_nhwc' or direct_offset+tensors[src]['storage_bytes']>arena_bytes:
                raise ValueError('direct source extent/dtype')
            if (direct.get('kernel',1),direct.get('padding',0))!=(kernel,padding): raise ValueError('direct input kernel/padding')
            if direct_offset%32:
                raise ValueError('direct input alignment')
            for field,expected_value in (('channels',tensors[src]['padded_channels']),('height',tensors[src]['h']),('width',tensors[src]['w']),('stride',stride)):
                if direct[field]!=expected_value: raise ValueError('direct input geometry')
            zero=integer(direct['zero_offset'],'direct zero offset')
            if zero%32 or zero+32>arena_bytes: raise ValueError('direct zero extent/alignment')
        dynamics=launch.get('dynamic_regions',[])
        if dynamics:
            dynamic_defs.append(f'static const m2_dynamic m2_dynamic_{index}[] = {{')
            for region in dynamics:
                tensor=ids[str(region['tensor'])];region_offset=integer(region['offset'],'dynamic offset')
                if region_offset%32 or tensors[tensor]['dtype']!='fp8_nhwc' or region_offset+tensors[tensor]['storage_bytes']>arena_bytes:
                    raise ValueError('dynamic source extent/dtype')
                dynamic_defs.append(f'  {{{region_offset},{tensor}}},')
            dynamic_defs.append('};')
        dynamic_symbol=f'm2_dynamic_{index}' if dynamics else '0'
        scale=launch.get('input_scale_exponent',launch.get('input_scale',launch.get('a_scale_exponent',launch.get('a_scale',tensors[src].get('scale_exponent',0)))))
        fields=(f'{initial},{expected},{arena_bytes},{a_offset},{a_bytes},{offset},{out_bytes},'
                f'{src},{dst},{kernel},{stride},{padding},{m},{k},{n},{logical_m},{logical_k},{logical_n},{scale},'
                f'{int(bool(direct))},{direct_offset},{dynamic_symbol},{len(dynamics)}')
        launches.append(f'  {{{program},{program_bytes},{fields}}},')
        # fast.py binds the immutable-B programs and their sizes; the default program size is a placeholder.
        fast_launches.append(f'  {{m2_fast_program_{index},{program_bytes},{fields}}},')
        initial_symbols.append(initial)
    node_entries=[]
    opcodes={'launch':'M2_LAUNCH','maxpool':'M2_MAXPOOL','avgpool':'M2_AVGPOOL','fc_bias':'M2_FC_BIAS'}
    for node in model['nodes']:
        op=node['op']
        if op not in opcodes: raise ValueError('unsupported graph node')
        src=ids[str(node['input'])] if op!='launch' else 0
        dst=ids[str(node['output'])] if op!='launch' else 0
        launch=integer(node.get('launch',node.get('index',0)),'node launch')
        if op=='launch' and launch>=len(launches): raise ValueError('node launch index')
        node_entries.append(f'  {{{opcodes[op]},{src},{dst},{launch},{node.get("kernel",3)},{node.get("stride",2)},{node.get("padding",1)}}},')
    header=['#ifndef RESNET_M2_ASSETS_H','#define RESNET_M2_ASSETS_H','#include "host.h"',
            f'#define M2_LAUNCHES {len(launches)}U',f'#define M2_NODES {len(node_entries)}U',f'#define M2_LOGITS {logits}U',
            f'#define M2_ARENA_BASE 0x{arena_base:x}UL',f'#define M2_ARENA_BYTES {maximum_arena}U',f'#define M2_INPUT_FP8_BYTES {input_bytes//4}U',
            f'#define m2_fc_bias {bias_symbol}',f'#define m2_expected_logits {expected_symbol}',*declarations,
            'extern m2_tensor m2_tensors[];','extern const m2_launch m2_launches[];','extern const m2_node m2_nodes[];',
            '#ifdef M2_FAST',*(f'extern const uint32_t m2_fast_program_{i}[];' for i in range(len(launches))),'#endif','#endif']
    (out/'assets.h').write_text('\n'.join(header)+'\n')
    body=['#include "assets.h"',*tensor_defs,*dynamic_defs,'m2_tensor m2_tensors[] = {',*tensor_entries,'};',
          '#ifdef M2_FAST','const m2_launch m2_launches[] = {',*fast_launches,'};','#else',
          'const m2_launch m2_launches[] = {',*launches,'};','#endif','const m2_node m2_nodes[] = {',*node_entries,'};',
          '','#ifdef M2_FAST','const m2_fast_launch m2_fast_launches[] = {','};','#endif']
    (out/'assets.c').write_text('\n'.join(body)+'\n')
    assembly+=['','#ifdef M2_FAST']
    for i in range(len(launches)):
        assembly+=['.balign 64',f'.globl m2_fast_program_{i}',f'm2_fast_program_{i}:',f'.incbin "fast-program-{i:02}.bin"']
    (out/'images.S').write_text('\n'.join([*assembly,'#endif'])+'\n')
    for name in ('host.c','host.h','qemu_console.c'):
        shutil.copyfile(PACKAGE/'templates'/name,out/name)
    (out/'runtime').mkdir()
    for name in ('atlas_host.c','atlas_host.h','atlas_host_layout.h','atlas_numeric.h'):
        shutil.copyfile(RUNTIME/name,out/'runtime'/name)
    total_assets=sum(x['bytes'] for x in embedded.values())
    tensor_bytes=sum(t['storage_bytes'] for t in tensors if t['dtype']!='f32_nchw')
    if total_assets+tensor_bytes+2*1024*1024>=256*1024*1024:
        raise ValueError('embedded capture plus host tensors would exceed bounded host window')
    (out/'CMakeLists.txt').write_text(CMAKE)
    if model_path.read_bytes()!=model_raw: raise ValueError('model plan changed during packaging')
    for path,sha in input_pins.items():
        if digest(checked(path).read_bytes())!=sha: raise ValueError('asset changed during packaging')
    generated_sha256={name:digest((out/name).read_bytes()) for name in ('assets.h','assets.c','images.S','host.h','host.c','CMakeLists.txt','qemu_console.c')}
    receipt=dict(generated_sha256=generated_sha256,model_sha256=digest(model_raw),launches=len(launches),nodes=len(node_entries),
                 arena_base=arena_base,maximum_arena_bytes=maximum_arena,embedded_asset_bytes=total_assets,
                 tensor_bytes=tensor_bytes,initial_symbols=initial_symbols,source_sha256=input_pins,assets=embedded)
    (out/'package.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=checked,required=True,help='Captured model.json from capture_resnet_device.py')
    parser.add_argument('--out',type=checked,required=True,help='Fresh package directory')
    args=parser.parse_args()
    receipt=package(args.model,args.out)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('initial_symbols','source_sha256','assets')}))
