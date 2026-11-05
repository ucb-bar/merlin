#!/usr/bin/env python3
"""Regenerate the SmolVLA trajectory through the delivered Merlin C graph.

Reference generation only. Native libc/libm runs host control, while admitted
model workers execute every exact arena staged by the production raw adapter.
Durable prefix replies may be replayed only after exact current arena and program
identity checks; host state is always recomputed. Final ELF admission is separate.

Historical provider paths require explicit --relocate mappings. Their destinations
are independent of --bundle; original source, bookends, and command remain in the
historical provenance fields alongside the relocated view and mapping pairs.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).absolute().parent
ATLAS = PACKAGE.parents[1]
sys.path[:0] = [str(ATLAS/'tools'),str(ATLAS/'tools/host_iss')]
import numpy as np
import smol_capture
import kernel_library  # noqa: E402,F401
from atlas_program import assembly
from kernel_library import code_sources
from smol_full_plan import calls as graph_calls, reference_geometry
from smol_full_phases import segments, V2
from atlas_path_guard import checked as safe, lexical_absolute_path, map_metadata_paths
from smol_capture.fusion import FusionCapture, SETS as FUSION_SETS, fp8_diagnostic
BASE = BUNDLE = BASE_WORK = NATIVE = PROVIDERS = None
PYTHON = sys.executable
MODEL_ROOT = MODEL_PIN = None
FUSION_SET = "none"
EXECUTION_OPTIONS = {}
REPLIES = []
REPLIES_SHA256 = None
REUSE_MATCHING_REPLIES = False
SHARED_BLOBS = {}
EPILOGUE = struct.pack('<6I',0x13,0xC1102FF3,0x13,0xC10F9073,0x13,0x73)

PATH_MAP = []

def migrated(value):
    value = str(value)
    if not value.startswith('/'):return value
    lexical_absolute_path(value)
    for old, new in sorted(PATH_MAP, key=lambda pair: len(pair[0]), reverse=True):
        if value == old or value.startswith(old+'/'):
            return str(lexical_absolute_path(new+value[len(old):]))
    return value


def metadata_relocation():
    return dict(mappings=[dict(source=old,destination=new) for old,new in PATH_MAP])


def compile_bookends(compiled):
    return {migrated(name):expected for name,expected in compiled['source_sha256'].items()}


def require(condition, message):
    if not condition: raise ValueError(message)


def digest(raw): return hashlib.sha256(raw).hexdigest()

def read(path): return safe(path).read_bytes()

def sha(path): return digest(read(path))


def reply_matches(event, index, sequence, kind, offset, extent, initial, program, requested_dma):
    """An optional prior reply is reusable only for this exact whole-arena launch."""
    return (event['index']==index and event['sequence']==sequence and event['kind']==kind
        and (event['output_offset'],event['output_bytes'])==(offset,extent)
        and event['initial']['sha256']==digest(initial) and event['initial']['bytes']==len(initial)
        and event['runtime_program']['sha256']==digest(program) and event['runtime_program']['bytes']==len(program)
        and event['observed_dma']==event['requested_dma']==requested_dma)


def reuse_execution(event, folder, index, sequence, kind, offset, extent, initial, program, requested_dma,
                    *, output_dtype=None, output_scale_exponent=None):
    """Persist an exact reused reply with the same durable shape as a worker reply."""
    require(reply_matches(event,index,sequence,kind,offset,extent,initial,program,requested_dma),
        'prebound complete launch identity')
    expected=read(event['output']['path'])
    require(len(expected)==extent and digest(expected)==event['output']['sha256'],'prebound output binding')
    expected_arena=initial[:offset]+expected+initial[offset+extent:]
    require(digest(expected_arena)==event['expected_arena_sha256'],'prebound preserved arena identity')
    temporary=folder/'arena-expected.bin';blob(temporary,expected_arena)
    output_record=blob(folder/'output.bin',expected)
    result=dict(output_path=output_record['path'],arena_path=str(temporary),execution=dict(observed=event['observed_dma']),
        provenance=dict(backend='identity-bound-captured-reply',gate_of_record=False,prior_provenance=event['provenance'],
            ledger_sha256=REPLIES_SHA256,original_model_all_launch_gate=False),
        initial_sha256=digest(initial),expected_arena_sha256=digest(expected_arena))
    if 'fp8_stats' in event:result['fp8_stats']=dict(event['fp8_stats'])
    if output_dtype=='fp8':
        stats=result.get('fp8_stats')
        require(isinstance(stats,dict) and stats.get('scale_exponent')==output_scale_exponent
            and stats.get('physical_values')==extent,'prebound FP8 observation missing or differs')
    write_json(folder/'execution.json',result)
    return result


def resume_execution(event, folder):
    result=json.loads(read(folder/'execution.json'))
    require(result['provenance']==event['provenance'],'durable prefix execution provenance')
    require(result.get('fp8_stats')==event.get('fp8_stats'),'durable prefix FP8 observation differs')
    result['output_path']=event['output']['path']
    return result

_BUNDLE_SUMS=None

def bundle_read(name):
    global _BUNDLE_SUMS
    if _BUNDLE_SUMS is None:
        _BUNDLE_SUMS={}
        for line in read(BUNDLE/'SHA256SUMS').decode().splitlines():
            value,member=line.split(None,1);member=member.lstrip('*')
            require(not Path(member).is_absolute() and '..' not in Path(member).parts,'flat relative bundle member')
            _BUNDLE_SUMS[member]=value
    require(str(name) in _BUNDLE_SUMS,'unlisted delivered bundle payload '+str(name))
    raw=read(BUNDLE/name);require(digest(raw)==_BUNDLE_SUMS[str(name)],'changed delivered bundle payload '+str(name))
    return raw

def write_json(path, value):
    safe(path).write_text(json.dumps(value,indent=2)+'\n')


def blob(path, raw, *, dtype=None, shape=None):
    path = safe(path);path.parent.mkdir(parents=True,exist_ok=True)
    require(not path.exists() or path.read_bytes()==raw,'immutable capture changed: '+str(path))
    shared=SHARED_BLOBS.get((digest(raw),len(raw)))
    if path.exists() and shared is not None and (safe(shared).stat().st_dev,safe(shared).stat().st_ino)!=(path.stat().st_dev,path.stat().st_ino):
        require(read(shared)==raw,'shared existing payload identity')
        temporary=path.with_name(path.name+'.link')
        require(not temporary.exists(),'fresh shared-link temporary')
        try:os.link(safe(shared),temporary);os.replace(temporary,path)
        except OSError:
            if temporary.exists():temporary.unlink()
    if not path.exists():
        if shared is not None:
            require(read(shared)==raw,'shared immutable capture payload identity')
            try:os.link(safe(shared),path)
            except OSError:path.write_bytes(raw)
        else:path.write_bytes(raw)
    result = dict(path=str(path),file=str(path),bytes=len(raw),sha256=digest(raw))
    if dtype is not None: result['dtype']=dtype
    if shape is not None: result['shape']=shape
    return result


class Worker:
    def __init__(self, backend, out):
        env={key:value for key,value in os.environ.items() if not key.startswith(('ATLAS_NPU_MODEL_','NPU_MODEL_')) and key!='PYTHONPATH'}
        env.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(out),ATLAS_NPU_MODEL_ROOT=str(MODEL_ROOT),ATLAS_NPU_MODEL_PIN=MODEL_PIN)
        command=[PYTHON,'-B','-u',str(PACKAGE/'worker.py'),'--backend',backend,'--workspace',str(out),'--atlas-root',str(ATLAS)]
        if backend=='projection' and EXECUTION_OPTIONS.get('native_projection_model'):
            from smol_capture.admission import NATIVE_PIN
            env.update(ATLAS_NPU_MODEL_ROOT=str(EXECUTION_OPTIONS['native_projection_model']),ATLAS_NPU_MODEL_PIN=NATIVE_PIN,
                NPU_MODEL_RTL_NATIVE_MXU0='1',NPU_MODEL_RTL_NATIVE_BUILD_DIR=str(EXECUTION_OPTIONS['native_build_dir']))
            command+=['--native-admission-root',str(EXECUTION_OPTIONS['native_admission_root']),
                '--native-admission-sha256',EXECUTION_OPTIONS['native_admission_sha256'],'--original-model-root',str(MODEL_ROOT)]
        if backend=='attention' and EXECUTION_OPTIONS.get('fast_attention_admission'):
            command+=['--fast-attention-admission',str(EXECUTION_OPTIONS['fast_attention_admission'])]
        self.log=(out/(backend+'-worker.log')).open('w')
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,env=env)
        self.index=0
    def call(self, op, **args):
        self.index+=1;request=dict(id=self.index,op=op,**args)
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        while True:
            line=self.process.stdout.readline()
            require(bool(line),'model worker exited; inspect worker log')
            try: result=json.loads(line)
            except ValueError:
                self.log.write(line);self.log.flush();continue
            if result.get('id')!=self.index:
                self.log.write(line);self.log.flush();continue
            require(result.get('ok') is True,'model worker rejected '+op+': '+str(result.get('error')))
            return result['result']
    def close(self):
        code=self.process.poll()
        try:
            if code is None:
                self.process.stdin.close()
                try:code=self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.terminate();self.process.wait(timeout=10)
                    raise ValueError('model worker did not exit after all replies')
            require(type(code) is int and code==0,'model worker exit was not successful')
        finally:self.log.close()


class Buffer(ct.Structure):
    _fields_=[('address',ct.c_size_t),('size',ct.c_size_t),('access',ct.c_int)]

class Panel(ct.Structure):
    _fields_=[('program',ct.c_void_p),('weights',ct.c_void_p)]+[(x,ct.c_size_t) for x in
        ('program_words','rescale_word','physical_rows','physical_k','physical_n','row_start','logical_rows',
         'input_width','output_width','call','output_offset','dram_bytes','weight_bytes','load_bytes','store_bytes')]+[
        ('activation_scale',ct.c_int),('weight_scale',ct.c_int)]

class Call(ct.Structure):
    _fields_=[(x,ct.c_size_t) for x in ('rows','input_width','output_width','first_panel','panel_count')]+[
        ('bias',ct.c_void_p)]+[(x,ct.c_uint) for x in ('site','layer','role','image','input_dtype','output_dtype','bias_dtype')]+[
        ('activation_scale',ct.c_int)]

class Attention(ct.Structure):
    _fields_=[(x,ct.c_void_p) for x in ('program','metadata','expected_mask')]+[(x,ct.c_size_t) for x in
        ('program_words','metadata_bytes','descriptor_offset','descriptor_bytes','key_count_offset','key_count_bytes',
         'mask_offset','mask_bytes','before_call','phase','index')]+[('kind',ct.c_uint)]+[(x,ct.c_size_t) for x in
        ('rows','keys','query_heads','kv_heads','q_offset','k_offset','v_offset','output_offset','output_bytes',
         'dram_bytes','load_bytes','store_bytes')]+[(x,ct.c_int) for x in ('q_exponent','k_exponent','v_exponent')]

class Raw(ct.Structure):
    _fields_=[(x,ct.c_size_t) for x in ('host','panels','calls','panel_count','call_count','panel_cursor','call_cursor',
        'dram','activation_copy','input_copy','dram_capacity','activation_capacity','input_capacity','max_polls','cycle_delta')]+[
        ('active_buffer',Buffer),('image',ct.c_uint),('failed',ct.c_uint),('result',ct.c_int),
        ('attention_panels',ct.c_size_t),('attention_count',ct.c_size_t),('attention_cursor',ct.c_size_t)]


def wrapper_source():
    source=read(PACKAGE/'wrapper.c').decode()
    if FUSION_SET!='none':
        source='#include "smol_fusion.h"\n'+source
        source=source.replace('    if(rows*k>smol_full_raw.input_capacity)return 1;',
            '    int view=smol_fusion_input_view(&smol_full_raw,input,rows,k,scale);\n'
            '    if(view<0)return 1;\n    if(view)return 0;\n'
            '    if(rows*k>smol_full_raw.input_capacity)return 1;')
        source+='\nsize_t smol_capture_fusion_size(void){return sizeof(smol_fusion_panel); }\n'
    return source


def prepare(out):
    """Compile authenticated input tables and current production C providers."""
    out=safe(out)
    out.mkdir(parents=True,exist_ok=True);source=out/'native-source';source.mkdir(exist_ok=True)
    phase_compile=json.loads(read(BASE/'phase-canonical-v4/compile.json'))
    baseline_pins=compile_bookends(phase_compile);pins={};copies={}
    def pin(path, expected=None):
        path=safe(path);actual=sha(path)
        require(expected is None or actual==expected,'source authentication: '+str(path));pins[str(path)]=actual
        return path
    main=pin(BASE_WORK/'main.c',baseline_pins[str(BASE_WORK/'main.c')])
    assets=pin(BASE_WORK/'assets.S',baseline_pins[str(BASE_WORK/'assets.S')])
    data=json.loads(bundle_read('contract.json'))
    # Authenticate each immutable assembler input before the compiler can read it.
    baseline_sums={}
    for line in read(BASE/'host-baseline-v4/SHA256SUMS').decode().splitlines():
        value,name=line.split(None,1);baseline_sums[str(BASE/'host-baseline-v4'/name.lstrip('*'))]=value
    for name in re.findall(r'^\.incbin "([^"\n]+)"',read(assets).decode(),re.M):
        p=safe(migrated(name));require(str(p) in baseline_sums,'unlisted immutable assembler payload '+str(p))
        pin(p,baseline_sums[str(p)])
    original=read(main).decode();text=original
    assets_text=read(assets).decode()
    assets_text=re.sub(r'(?m)^\.incbin \"([^\"\n]+)\"$',lambda m: '.incbin \"'+migrated(m[1])+'\"',assets_text)
    (source/'assets.S').write_text(assets_text);assets=pin(source/'assets.S')
    text=text.replace('UINT64_C(4294967296)','UINT64_C(2415919104)')
    text=text.replace('smol_full_cycles[2715]','smol_full_cycles[2915]').replace('sizeof(smol_policy_raw_context)==160','sizeof(smol_policy_raw_context)==184')
    for name,replacement in (('smol_vision_image','phased_vision'),('smol_policy_prefix','phased_prefix'),('smol_policy_step','phased_step')):
        require(text.count(name+'(')==1,'exact generated graph invocation');text=text.replace(name+'(',replacement+'(')
    lines=text.splitlines();old_comparisons=[x for x in lines if x.startswith('if(!same(')]
    require(len(old_comparisons)==47,'exact old reference checks')
    lines=[x for x in lines if not x.startswith(('if(!same(','static int same(','for(size_t i=0;i<2715;++i)if(!smol_full_cycles'))]
    text='\n'.join(lines)+'\n'
    text=text.replace('"smol-full",2715','"smol-full-reference-generation",2915')
    text=text.replace('atlas_result_progress(&smol_result,smol_full_raw.panel_cursor,',
        'atlas_result_progress(&smol_result,smol_full_raw.panel_cursor+smol_full_raw.attention_cursor,')
    text=text.replace('atlas_workload_mailbox[2]=2715','atlas_workload_mailbox[2]=2915')
    text=text.replace('if(!atlas_result_pass(&smol_result))','smol_result.completed=2915;if(!atlas_result_pass(&smol_result))')
    fences=text.count('__asm__ volatile("fence rw,rw":::"memory")');require(fences==2,'exact native fence adaptation')
    text=text.replace('__asm__ volatile("fence rw,rw":::"memory")','__asm__ volatile("":::"memory")')
    text=wrapper_source()+text
    (source/'main.c').write_text(text);pin(source/'main.c');pin(PACKAGE/'wrapper.c')
    (source/'smol_full_phase.h').write_bytes(read(ATLAS/'runtime/smol_full_phase.h'))
    (source/'atlas_result_transport.h').write_text('''#ifndef ATLAS_RESULT_TRANSPORT_H
#define ATLAS_RESULT_TRANSPORT_H
#include "atlas_result.h"
#include "smol_native_device.h"
typedef struct { uintptr_t base; } atlas_result_uart;
static inline void atlas_result_uart_init(atlas_result_uart *uart){uart->base=0;}
static inline void atlas_result_uart_putchar(void *context,char value){(void)context;smol_native_putchar(value);}
#endif
''')
    for generated_header in ('smol_full_phase.h','atlas_result_transport.h'):
        pin(source/generated_header)
    stems=('smol_host_ops','smol_host_attention','smol_host_softmax','smol_host_rope','smol_host_controls',
        'host_matrix_fp8pack','smol_policy_ops','smol_policy_block','smol_policy_graph','smol_vision_ops',
        'smol_vision_graph','smol_policy_raw','smol_vision_attention','atlas_host','smol_attention')
    if FUSION_SET!='none':stems+=('smol_fusion',)
    queue=[stem+'.c' for stem in stems]+['smol_native_device.c','smol_native_device.h','atlas_result.h']
    done=set()
    while queue:
        name=queue.pop()
        if name in done:continue
        require(Path(name).name==name,'flat native include closure')
        choices=([ATLAS/'runtime'/name] if name.startswith(('smol_policy_raw','smol_attention','atlas_host')) else [])
        choices += [PROVIDERS/name,NATIVE/name,ATLAS/'runtime'/name]
        selected=next((safe(p) for p in choices if safe(p).is_file()),None)
        require(selected is not None,'missing flat include '+name);raw=read(pin(selected))
        target=source/name;target.write_bytes(raw);pin(target);copies[str(target)]=dict(source=str(selected),sha256=digest(raw))
        queue+=re.findall(r'^\s*#\s*include\s+"([^"\n]+)"',raw.decode(),re.M);done.add(name)
    command=['/usr/bin/gcc','-std=c11','-DSMOL_ATTENTION_RUNTIME=1',*(['-DSMOL_FUSION_RUNTIME=1'] if FUSION_SET!='none' else []),'-O2','-fPIC','-shared','-Wall','-Wextra','-Werror','-ffp-contract=off',
        '-fno-fast-math','-I'+str(source),str(source/'main.c'),*[str(source/(stem+'.c')) for stem in stems],
        str(assets),str(source/'smol_native_device.c'),'-Wl,--wrap=smol_policy_raw_vision',
        '-Wl,--wrap=smol_policy_raw_graph','-Wl,--wrap=atlas_host_direct_read32','-Wl,--wrap=atlas_host_direct_write32',
        '-Wl,-z,defs','-Wl,-z,noexecstack','-Wl,-Bsymbolic','-lm','-o',str(out/'host.so')]
    with (out/'compile.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    require(result.returncode==0,'native reference compilation failed; inspect compile.log')
    for name,expected in pins.items():require(sha(name)==expected,'source bookend '+name)
    record=dict(schema='atlas.smol_native_recapture_compile.v2',scope=__doc__,command=command,source_sha256=pins,fusion_set=FUSION_SET,
        library_sha256=sha(out/'host.so'),source_copies=copies,original_main_sha256=digest(original.encode()),
        adaptations=dict(old_reference_comparisons_removed=47,placement=0x90000000,mutable_descriptor_copies=True,
            native_fences=fences,phase_snapshots=True),reference_generation_only=True,guest_execution=False,
        device_execution=False)
    write_json(out/'compile.json',record);write_json(out/'source-copy.json',copies)
    return record



def relocate_historical_metadata(value):
    return map_metadata_paths(value,migrated)


def native_symbols(path):
    raw=read(path);header=struct.unpack_from('<16sHHIQQQIHHHHHH',raw)
    require(header[0][:7]==b'\x7fELF\x02\x01\x01' and header[1:3]==(3,62),'native AMD64 ELF')
    sections=[struct.unpack_from('<IIQQQQIIQQ',raw,header[6]+i*64) for i in range(header[12])];symbols={}
    for section in sections:
        if section[1]!=2:continue
        strings=sections[section[6]];table=raw[strings[4]:strings[4]+strings[5]]
        for offset in range(section[4],section[4]+section[5],24):
            no,info,other,index,value,size=struct.unpack_from('<IBBHQQ',raw,offset)
            if no:symbols[table[no:table.index(b'\0',no)].decode()]=dict(address=value,size=size)
    return symbols


def projection_source(row, scale):
    from chipyard.arena import relocate_assembly
    m,k,n=row['physical_mkn']
    exponent=scale+row['b_scale']
    if row['program_policy']=='traffic-reuse-v1':
        return relocate_assembly(assembly(m,k,n,row['output_offset'],exponent,traffic_reuse=True),0x100000000,0x90000000)
    source=bundle_read(row['source']).decode()
    source,count=re.subn(r'^vli.all m4, [0-9]+$',f'vli.all m4, {(exponent+127)<<7}',source,flags=re.M)
    require(count==1,'one captured projection rescale source literal')
    return source


def capture_source_paths():
    require(safe(smol_capture.__file__)==safe(PACKAGE/'__init__.py'),'capture package imported outside selected source')
    sources=[PACKAGE/name for name in ('__init__.py','capture.py','worker.py','model.py','wrapper.c','fusion.py','stats.py','admission.py','fast_attention.py','fast_ipt.py')]
    sources += [ATLAS/'tools/smol_full_plan.py',ATLAS/'tools/smol_full_phases.py']
    sources += [ATLAS/kernel_library.relative_path(name) for name in code_sources(traffic_reuse=True)]
    return sources+[ATLAS/kernel_library.relative_path(name) for name in ('smol_fusion_program.py','smol_fusion_fp8.py')]


def run(out, limit=None, resume=False):
    out=safe(out);compiled=json.loads(read(out/'compile.json'))
    require(compiled.get('fusion_set','none')==FUSION_SET,'compiled fusion profile differs')
    require(sha(out/'host.so')==compiled['library_sha256'],'compiled library bookend')
    native_pins=compile_bookends(compiled)
    for name,expected in native_pins.items():require(sha(name)==expected,'source bookend '+name)
    capture_binding={str(path):sha(path) for path in capture_source_paths()}
    run_dir=out/'run';require(not (run_dir/'manifest.json').exists(),'fresh reference capture required');run_dir.mkdir(exist_ok=True)
    ledger=run_dir/'device-events.jsonl';cached=[]
    if ledger.exists():
        require(resume,'existing capture requires --resume')
        for line in read(ledger).decode().splitlines():
            item=json.loads(line);require(item['index']==len(cached) and item['sequence']==len(cached)+1,'durable prefix order')
            for key in ('program','source','runtime_program','runtime_source','initial','output'):
                record=item[key];raw_bytes=read(record['path'])
                require(len(raw_bytes)==record['bytes'] and digest(raw_bytes)==record['sha256'],'durable prefix '+key)
            cached.append(item)
    migration=dict(schema='atlas.smol_capture_machine_move.v1',library_sha256=compiled['library_sha256'],
        original_compile_sha256=sha(out/'compile.json'),source_sha256=native_pins,
        metadata_relocation=metadata_relocation(),
        relocated_sources=[dict(previous=name,current=migrated(name),sha256=value,verified=True)
            for name,value in compiled['source_sha256'].items() if name!=migrated(name)],
        prefix_events=len(cached),prefix_ledger_sha256=sha(ledger) if cached else None,
        replay_policy='Recompute actual C host graph; compare complete staged arena and loaded program before returning durable prefix output.',
        original_model_all_launch_gate=False)
    write_json(out/'source-migration.json',migration)
    contract=json.loads(bundle_read('contract.json'));baseline_replay=json.loads(bundle_read('host/reference-replay.json'))
    historical_bookends=baseline_replay['source_bookends'];historical_source=baseline_replay['source']
    historical_command=baseline_replay['command'];baseline_replay=relocate_historical_metadata(baseline_replay)
    observation_path=baseline_replay['command'][baseline_replay['command'].index('--observation-input')+1]
    authenticated_data={record['path']:record for record in baseline_replay['source_bookends']
        if record['path'] in (observation_path,baseline_replay['source']['weights']['path'])}
    require(len(authenticated_data)==2,'complete immutable source data identity')
    migration['historical_source_data_not_revalidated']=list(authenticated_data.values())
    migration['source_data_role']='Authoritative inputs are immutable SHA-verified embedded assets, compared directly to delivered bundle host parameters; historical checkpoint and observation archive are not required or claimed revalidated.'
    write_json(out/'source-migration.json',migration)
    plans=graph_calls();ranges=[dict(x,launch_start=x.get('panel_start',x['launch_start']),launch_end=x.get('panel_end',x['launch_end'])) for x in segments(V2)];kernels=contract['kernels']
    lib=ct.CDLL(str(out/'host.so'));symbols=native_symbols(out/'host.so')
    input_provider=[]
    for name,record in contract['host_parameters'].items():
        value=bundle_read(record['file'])
        require(len(value)==record['bytes'] and digest(value)==record['sha256'],'immutable delivered host parameter '+name)
        actual=ct.string_at(ct.addressof((ct.c_ubyte*len(value)).in_dll(lib,record['symbol'])),len(value))
        require(actual==value,'embedded host parameter identity '+name)
        input_provider.append(dict(name=name,**record,path=str(BUNDLE/record['file']),embedded_bytes_equal=True))
    input_receipt=dict(schema='atlas.smol_compiled_input_provider.v1',provider='authenticated delivered FullBundle host_parameters',
        contract=dict(path=str(BUNDLE/'contract.json'),sha256=digest(bundle_read('contract.json'))),
        host_library=dict(path=str(out/'host.so'),sha256=compiled['library_sha256']),
        host_parameters=input_provider,projection_weights='Every launch compares exact staged weights to authenticated delivered bundle payload.',
        historical_raw_sources_revalidated=False,historical_raw_sources=list(authenticated_data.values()),
        references_from_provider=False,outputs='Genuine current C host trajectory with current model launch replies.')
    write_json(out/'compiled-input-provider.json',input_receipt)
    lib.smol_capture_raw_size.restype=ct.c_size_t;lib.smol_capture_attention_size.restype=ct.c_size_t
    require(lib.smol_capture_raw_size()==ct.sizeof(Raw)==184,'raw context ABI')
    require(lib.smol_capture_attention_size()==ct.sizeof(Attention),'attention descriptor ABI')
    lib.smol_capture_initialize();raw=Raw.in_dll(lib,'smol_full_raw')
    panels=(Panel*2715).in_dll(lib,'smol_capture_panels');call_table=(Call*1421).in_dll(lib,'smol_capture_calls')
    attention_table=(Attention*200).in_dll(lib,'smol_capture_attention')
    fusion=FusionCapture(lib,raw,panels,kernels,call_table,FUSION_SET) if FUSION_SET!='none' else None
    before_calls=[call['index'] for call in plans if call['site'] in (5,6,7) and call['role']==(4 if call['site']==7 else 3)]
    require(len(before_calls)==200,'complete seeded attention schedule')
    for index,before in enumerate(before_calls):
        attention_table[index].before_call=before;attention_table[index].phase=index;attention_table[index].index=index
    def address(name,size):return ct.addressof((ct.c_ubyte*size).in_dll(lib,name))
    def tensor(path,pointer,rows,width,dtype):
        result=blob(path,ct.string_at(pointer,rows*width*(2 if dtype==0 else 4)),
            dtype='torch.bfloat16' if dtype==0 else 'torch.float32',shape=[1,rows,width])
        result['file']=str(path.relative_to(run_dir));return result
    workers={name:Worker(name,out) for name in ('projection','attention')}
    calls=[];projection=[];attention=[];events=[];errors=[];buffers=[];pending={};boundaries=[];observations=[]
    live=dict(json.loads(read(BASE/'phase-canonical-v4/phase-reference.json'))['fixed_live_regions']);live['smol_full_cycles']=2915*4
    scratch={name:row['size'] for name,row in symbols.items() if name not in live and row['size'] and
        (name.startswith(('smol_v_','smol_b_','smol_g_')) or name in ('smol_full_input_copy','smol_full_activation_copy','smol_full_image'))}
    for name,size in live.items():require(symbols[name]['size']==size,'live region ABI '+name)
    def regions():
        return {name:blob(run_dir/'boundaries/blobs'/(digest(ct.string_at(address(name,size),size))+'.bin'),
            ct.string_at(address(name,size),size)) for name,size in live.items()}
    cursor=(ct.c_uint32*10).in_dll(lib,'smol_full_phase_state');cursor[:]=[1,0,0,0,0,0,0,0,0,0]
    initial=dict(next_segment=0,regions=regions())
    pred_type=ct.CFUNCTYPE(ct.c_int,ct.c_uint,ct.c_uint,ct.c_void_p,ct.c_size_t,ct.c_size_t,ct.c_uint,ct.c_uint,ct.c_uint)
    output_type=ct.CFUNCTYPE(ct.c_int,ct.c_uint,ct.c_uint,ct.c_void_p,ct.c_size_t,ct.c_size_t)
    lib.smol_capture_scale.argtypes=[ct.c_uint,ct.c_void_p,ct.c_size_t,ct.c_size_t,ct.POINTER(ct.c_int)]
    @pred_type
    def predecessor(index,dtype,pointer,rows,k,site,layer,role):
        try:
            require(index==len(calls),'projection call sequence');plan=plans[index]
            require((dtype,rows,k,site,layer,role)==tuple(plan[x] for x in ('input_dtype','rows','input_width','site','layer','role')),'projection call identity')
            view_scale=fusion.input_view(pointer,rows,k) if fusion else None
            scale=ct.c_int();require(lib.smol_capture_scale(dtype,pointer,rows,k,ct.byref(scale))==0,'actual input scale')
            if view_scale is not None:require(scale.value==view_scale,'fixed producer/consumer FP8 scale')
            call=call_table[index];call.activation_scale=scale.value
            record=dict(baseline_replay['calls'][index]);record.pop('output',None)
            if view_scale is None:
                record['input']=tensor(run_dir/'calls'/f'{index:04d}-input.bin',pointer,rows,k,dtype)
            else:
                packed=ct.string_at(pointer,rows*k)
                record['input_fp8']=blob(run_dir/'calls'/f'{index:04d}-input-fp8.bin',packed,
                    dtype='uint8',shape=[rows//32,k//32,32,32])
                record['input']=blob(run_dir/'calls'/f'{index:04d}-input.bin',fp8_diagnostic(packed,rows,k,view_scale),
                    dtype='torch.bfloat16',shape=[1,rows,k])
                record['input_source_format']='fp8_tiles'
                record['input_decode_exponent']=view_scale
                record['input_diagnostic_only']=True
            # Preserve the established capture tensor shape, including patch CHW.
            record['input']['shape']=baseline_replay['calls'][index]['input']['shape']
            if site==7 and role==0:
                words=np.frombuffer(ct.string_at(pointer,rows*k*2),dtype='<u2').reshape(32,32,3,16,16)
                record['input']=blob(run_dir/'calls'/f'{index:04d}-input-chw.bin',words.transpose(2,0,3,1,4).tobytes(),
                    dtype='torch.bfloat16',shape=[1,3,512,512])
            record['input']['file']=str(Path(record['input']['path']).relative_to(run_dir))
            if 'input_fp8' in record:record['input_fp8']['file']=str(Path(record['input_fp8']['path']).relative_to(run_dir))
            record.update(index=index,module=plan['module'],a_scale=scale.value,native_input_layout='rows,width')
            calls.append(record)
            for idx in range(call.first_panel,call.first_panel+call.panel_count):
                p=panels[idx];row=kernels[idx];folder=run_dir/'projection'/f'{idx:04d}';folder.mkdir(parents=True,exist_ok=True)
                fused=fusion.prepare(idx,row,scale.value,call,p) if fusion else None
                source=fused[0] if fused else projection_source(row,scale.value)
                source_path=folder/'program.S';blob(source_path,source.encode())
                assembled=(dict(program_bin=str(folder/'program.bin')) if any(e['projection_index']==idx for e in cached)
                    else workers['projection'].call('assemble',source_path=str(source_path),output_path=str(folder/'program.bin')))
                program=read(assembled['program_bin']);baseline=bundle_read(row['original'])
                original_words=struct.unpack('<'+'I'*(len(baseline)//4),baseline)
                actual_words=struct.unpack('<'+'I'*(len(program)//4),program)
                literal=((row['a_scale']+row['b_scale']+127)<<23)|0x25f
                locations=[i for i,word in enumerate(original_words) if word==literal]
                require(len(locations)==1,'unique original projection rescale literal')
                rescale=((scale.value+row['b_scale']+127)<<23)|0x25f
                actual_locations=[i for i,word in enumerate(actual_words) if word==rescale]
                require(len(actual_locations)==1,'unique regenerated projection rescale literal')
                p.rescale_word=actual_locations[0];p.program_words=len(program)//4
                if not fused:
                    p.load_bytes=row['requested_dma']['load_bytes'];p.store_bytes=row['requested_dma']['store_bytes']
                require(p.weight_scale==row['b_scale'],'unchanged immutable weight exponent')
                patched=bytearray(baseline)
                struct.pack_into('<I',patched,locations[0]*4,rescale)
                if not fused:require(program==patched,'regenerated projection differs beyond rescale literal')
                storage=ct.create_string_buffer(program);buffers.append(storage);p.program=ct.addressof(storage);p.activation_scale=scale.value
                pending[idx]=dict(baseline_replay['projection_reference']['kernels'][idx],**row)
                pending[idx].update(source=str(source_path),original=str(folder/'program.bin'),a_scale=scale.value,
                    output_rescale_exponent=scale.value+p.weight_scale,recipe=row['program_policy'],rescale_word=int(p.rescale_word),
                    assembly_regenerated_matches_rescale_patch=not bool(fused),baseline_program_sha256=digest(baseline))
                if fused:
                    metadata=dict(fused[1])
                    metadata['extra']=blob(folder/'fusion-extra.bin',fused[2])
                    pending[idx].update(fusion=metadata,requested_dma=metadata['requested_dma'],
                        total_dram_bytes=metadata['total_dram_bytes'],dram_bytes=metadata['dram_bytes'])
                    calls[index]['fused_operations']=metadata['operations']
            return 0
        except BaseException as exc:errors.append(dict(where='projection-predecessor',index=int(index),error=str(exc)));return 1
    @output_type
    def output(index,dtype,pointer,rows,n):
        try:
            view_scale=fusion.output_view(index) if fusion else None
            if view_scale is None:
                calls[index]['output']=tensor(run_dir/'calls'/f'{index:04d}-output.bin',pointer,rows,n,dtype)
            else:
                packed=ct.string_at(pointer,rows*n)
                calls[index]['output_fp8']=blob(run_dir/'calls'/f'{index:04d}-output-fp8.bin',packed,
                    dtype='uint8',shape=[rows//32,n//32,32,32])
                calls[index]['output']=blob(run_dir/'calls'/f'{index:04d}-output.bin',fp8_diagnostic(packed,rows,n,view_scale),
                    dtype='torch.bfloat16',shape=[1,rows,n])
                calls[index]['output_source_format']='fp8_tiles'
                calls[index]['output_decode_exponent']=view_scale
                calls[index]['output_diagnostic_only']=True
                calls[index]['output_fp8']['file']=str(Path(calls[index]['output_fp8']['path']).relative_to(run_dir))
                calls[index]['output']['file']=str(Path(calls[index]['output']['path']).relative_to(run_dir))
            calls[index]['output']['shape']=baseline_replay['calls'][index]['output']['shape']
            if plans[index]['site']==7 and plans[index]['role']==0:
                words=np.frombuffer(ct.string_at(pointer,rows*n*2),dtype='<u2').reshape(rows,n).T.reshape(1,n,32,32)
                calls[index]['output']=blob(run_dir/'calls'/f'{index:04d}-output-chw.bin',words.tobytes(),dtype='torch.bfloat16',shape=[1,n,32,32])
                calls[index]['output']['file']=str(Path(calls[index]['output']['path']).relative_to(run_dir))
            return 0
        except BaseException as exc:errors.append(dict(where='projection-output',index=int(index),error=str(exc)));return 1
    lib.smol_predecessor_configure.argtypes=[pred_type];lib.smol_predecessor_configure(predecessor)
    lib.smol_output_configure.argtypes=[output_type];lib.smol_output_configure(output)
    att_type=ct.CFUNCTYPE(ct.c_int,ct.c_void_p,ct.c_uint,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p,
        ct.c_size_t,ct.c_size_t,ct.c_size_t,ct.c_size_t,ct.c_void_p)
    lib.smol_policy_raw_attention.argtypes=list(att_type._argtypes_);lib.smol_policy_raw_attention.restype=ct.c_int
    def keep(rawbytes):
        value=ct.create_string_buffer(rawbytes);buffers.append(value);return ct.addressof(value)
    @att_type
    def prepare_attention(context,kind,q,k,v,mask,rows,keys,hq,hkv,output_pointer):
        try:
            index=raw.attention_cursor;require(index==len(attention)==len(boundaries),'attention phase sequence')
            folder=run_dir/'attention'/f'{index:03d}';folder.mkdir(parents=True,exist_ok=True)
            arrays=[];exponents=[];ranges_report={}
            for name,pointer,heads,count in (('q',q,hq,rows),('k',k,hkv,keys),('v',v,hkv,keys)):
                values=np.frombuffer(ct.string_at(pointer,count*heads*128),dtype='<u2').reshape(count,heads,64).transpose(1,0,2).copy()
                finite=(values.astype(np.uint32)<<16).view(np.float32);require(np.isfinite(finite).all(),'finite attention input')
                maximum=float(np.max(np.abs(finite)));exponent=int(np.ceil(np.log2(maximum/448))) if maximum else 0
                exponents.append(exponent);arrays.append(values)
                ranges_report[name]=dict(max_abs=maximum,decode_exponent=exponent,pack_exponent=-exponent,
                    saturation_count=int(np.count_nonzero(np.abs(finite)>448*(2.0**exponent))),source_nonfinite_count=0)
                blob(folder/(name+'.bin'),values.tobytes(),dtype='uint16',shape=list(values.shape))
            allowed=(np.frombuffer(ct.string_at(mask,rows*keys),dtype=np.uint8).reshape(rows,keys).astype(bool)
                if mask else np.ones((rows,keys),dtype=bool))
            cached_attention=next((e for e in cached if e['attention_index']==index),None)
            if cached_attention:
                require(np.array_equal(np.load(folder/'logical-mask.npy',allow_pickle=False),allowed),'recomputed cached attention mask')
            else:
                np.save(folder/'logical-mask.npy',allowed,allow_pickle=False)
                np.savez(folder/'inputs.npz',q=arrays[0],k=arrays[1],v=arrays[2],mask=allowed)
            config=dict(query_rows=rows,keys=keys,query_heads=hq,kv_heads=hkv,head_dim=64,query_start=None,scale=0.125,mask='none' if kind==0 else 'arbitrary',q_exponent=exponents[0],
                k_exponent=exponents[1],v_exponent=exponents[2],p_exponent=-8,output_fp8=False,o_exponent=0,packed_decode=True)
            if cached_attention:
                prepared=json.loads(read(folder/'prepared.json'))
                require(prepared['config']==config,'recomputed cached attention config')
                prepared['arena_initial']=cached_attention['initial']['path']
            else:
                prepared=workers['attention'].call('prepare_attention',input_npz=str(folder/'inputs.npz'),output_dir=str(folder),config=config)
            offsets=prepared['offsets'];p=attention_table[index]
            p.program=keep(read(prepared['program_bin']));p.program_words=prepared['program_words']
            metadata=read(prepared['metadata_bin']);p.metadata=keep(metadata);p.metadata_bytes=len(metadata)
            p.expected_mask=keep(allowed.astype(np.uint8).tobytes()) if kind else None
            p.descriptor_offset=offsets['descriptors'];p.descriptor_bytes=offsets['descriptor_bytes']
            p.key_count_offset=offsets['key_counts'];p.key_count_bytes=offsets['key_count_bytes']
            p.mask_offset=offsets['masks'];p.mask_bytes=offsets['mask_bytes']
            require(p.before_call==raw.call_cursor,'seeded attention call boundary')
            p.phase=index;p.index=index;p.kind=kind
            p.rows=rows;p.keys=keys;p.query_heads=hq;p.kv_heads=hkv
            p.q_offset=offsets['q'];p.k_offset=offsets['k'];p.v_offset=offsets['v']
            p.output_offset=offsets['output'];p.output_bytes=offsets['output_bytes'];p.dram_bytes=len(read(prepared['arena_initial']))
            p.load_bytes=prepared['requested_dma']['load_bytes'];p.store_bytes=prepared['requested_dma']['store_bytes']
            p.q_exponent,p.k_exponent,p.v_exponent=exponents
            entry=dict(index=index,phase=index,before_call=raw.call_cursor,kind=kind,config=prepared['config'],offsets=offsets,
                program_source=prepared['program_source'],program_bin=prepared['program_bin'],arena_initial=prepared['arena_initial'],
                output_offset=p.output_offset,output_bytes=p.output_bytes,dram_bytes=p.dram_bytes,
                requested_dma=prepared['requested_dma'],qkv_exponents=exponents,operand_ranges=ranges_report,
                p_saturation_policy=dict(count_bound=0,reason='unnormalized exponentials are in [0,1], fixed FP8 scale256'),files={})
            pending['attention']=entry;attention.append(entry)
            result=lib.smol_policy_raw_attention(context,kind,q,k,v,mask,rows,keys,hq,hkv,output_pointer)
            require(result==0,'production raw attention rejected '+str(result));return 0
        except BaseException as exc:errors.append(dict(where='attention-prepare',index=int(raw.attention_cursor),error=str(exc)));return 1
    lib.smol_attention_configure.argtypes=[att_type,ct.c_void_p];lib.smol_attention_configure(prepare_attention,ct.addressof(raw))
    reply_type=ct.CFUNCTYPE(ct.c_int,ct.c_uint32,ct.POINTER(ct.c_uint32),ct.c_size_t,ct.c_uint32,ct.c_void_p)
    @reply_type
    def reply(index,program,words,sequence,context):
        try:
            require(index==len(events) and sequence==index+1,'ordered combined native launch')
            is_attention=index==raw.panel_cursor+raw.attention_cursor and pending.get('attention') is not None
            if is_attention:
                item=pending.pop('attention');folder=run_dir/'attention'/f"{item['index']:03d}";kind='attention'
                program_source=item['program_source'];program_bin=item['program_bin'];offset=item['output_offset'];extent=item['output_bytes'];size=item['dram_bytes']
            else:
                require(raw.panel_cursor in pending,'prepared projection launch');item=pending.pop(raw.panel_cursor)
                folder=run_dir/'projection'/f'{raw.panel_cursor:04d}';kind='projection'
                program_source=item['source'];program_bin=item['original'];offset=item['output_offset'];extent=item.get('fusion',{}).get('output_bytes',2*item['physical_mkn'][0]*item['physical_mkn'][2]);size=item['total_dram_bytes']
            runtime_bytes=ct.string_at(program,words*4)
            require(runtime_bytes==read(program_bin)[:-4]+EPILOGUE,'actual loaded program matches regenerated assembly')
            runtime_source=read(program_source).decode().rstrip()
            require(runtime_source.endswith('ecall'),'terminal ecall source')
            runtime_source=runtime_source[:-5]+'nop\ncsrrs x31, 0xc11, x0\nnop\ncsrrw x0, 0xc10, x31\nnop\necall\n'
            runtime_program_record=blob(folder/'runtime-program.bin',runtime_bytes)
            runtime_source_record=blob(folder/'runtime-program.S',runtime_source.encode())
            require(raw.active_buffer.address==raw.dram and raw.active_buffer.size==size and raw.active_buffer.access==2,'raw staging ownership')
            initial_bytes=ct.string_at(raw.dram,size);initial_record=blob(folder/'initial-dram.bin',initial_bytes)
            if fusion and kind=='projection' and item.get('fusion'):
                actual_meta=fusion.verify_stage(int(raw.panel_cursor),initial_bytes,read(program_source).decode())
                item['fusion'].update(initial_sha256=actual_meta['initial_sha256'],
                    baseline_initial_sha256=actual_meta['baseline_initial_sha256'])
            cached_event=cached[index] if index<len(cached) else None
            prebound=REPLIES[index] if REPLIES else None
            if prebound and REUSE_MATCHING_REPLIES and not reply_matches(prebound,index,sequence,kind,offset,extent,
                    initial_bytes,runtime_bytes,item['requested_dma']):
                prebound=None
            if prebound:
                typed=item.get('fusion',{})
                result=reuse_execution(prebound,folder,index,sequence,kind,offset,extent,initial_bytes,runtime_bytes,
                    item['requested_dma'],output_dtype=typed.get('output_dtype'),
                    output_scale_exponent=typed.get('output_scale_exponent'))
            elif cached_event:
                require(cached_event['kind']==kind and cached_event['output_offset']==offset and cached_event['output_bytes']==extent,'durable prefix identity')
                require(initial_record==cached_event['initial'] and runtime_program_record==cached_event['runtime_program'],
                    'recomputed full prefix arena and actual loaded program')
                result=resume_execution(cached_event,folder)
            elif kind=='attention':
                require(initial_bytes==read(item['arena_initial']),'attention staging matches prepared arena')
                result=workers[kind].call('execute_attention',config=item['config'],offsets=item['offsets'],
                    initial_arena=str(folder/'initial-dram.bin'),output_dir=str(folder),expected_initial=item.get('arena_initial'))
            else:
                descriptor=dict(item.get('fusion',{}),physical_mkn=item['physical_mkn'],recipe=item['recipe'],
                    output_rescale_exponent=item['output_rescale_exponent'],output_offset=offset,output_bytes=extent)
                result=workers[kind].call('execute_projection',program_source=program_source,initial_arena=str(folder/'initial-dram.bin'),
                    output_offset=offset,output_bytes=extent,rescale_word=panels[raw.panel_cursor].rescale_word,
                    physical_mkn=item['physical_mkn'],recipe=item['recipe'],descriptor=descriptor,output_dir=str(folder),admission_timeout_seconds=1800)
            expected=read(result['output_path']);require(len(expected)==extent,'exact worker output extent')
            require(result['execution']['observed']==item['requested_dma'],'exact observed/requested DMA identity')
            expected_arena=initial_bytes[:offset]+expected+initial_bytes[offset+extent:]
            require(result['initial_sha256']==digest(initial_bytes) and result['expected_arena_sha256']==digest(expected_arena),'worker full arena preservation')
            if not cached_event:
                transient=safe(result['arena_path']);require(transient.is_relative_to(folder),'owned transient expected arena')
                require(sha(transient)==digest(expected_arena),'worker expected arena bytes');transient.unlink()
            if kind=='attention':
                if not cached_event:
                    transient=safe(item['arena_initial']);require(transient.is_relative_to(folder),'owned transient prepared arena')
                    transient.unlink()
                item.pop('arena_initial',None)
            ct.memmove(raw.dram+offset,expected,extent)
            files={'program.S':blob(Path(program_source),read(program_source)),'program.bin':blob(Path(program_bin),read(program_bin)),
                'initial-dram.bin':initial_record,'expected-output.bin':blob(folder/'expected-output.bin',expected)}
            transient=safe(result['output_path'])
            if transient!=folder/'expected-output.bin':
                require(transient.is_relative_to(folder),'owned transient worker output');transient.unlink()
            files['runtime-program.bin']=runtime_program_record;files['runtime-program.S']=runtime_source_record
            item.update(files=files,expected=str(folder/'expected-output.bin'),initial_sha256=digest(initial_bytes),expected_arena_sha256=digest(expected_arena),
                expected_output_sha256=digest(expected),execution=result['execution'],provenance=result['provenance'])
            if 'fp8_stats' in result:item['fp8_stats']=result['fp8_stats']
            if kind=='projection':
                m,k,n=item['physical_mkn'];require(initial_bytes[m*k:m*k+k*n]==bundle_read(kernels[raw.panel_cursor]['weights']),'immutable staged weight bytes')
                item['activation']=blob(folder/'activation.bin',initial_bytes[:m*k])
                item['weights']=blob(folder/'weights.bin',initial_bytes[m*k:m*k+k*n]);projection.append(item)
            else:
                item['files']['logical-mask.npy']=blob(folder/'logical-mask.npy',read(folder/'logical-mask.npy'))
                metadata=dict(config=item['config'],offsets=item['offsets'])
                blob(folder/'metadata.json',(json.dumps(metadata,indent=2)+'\n').encode())
                item['files']['metadata.json']=blob(folder/'metadata.json',read(folder/'metadata.json'))
            event=dict(index=index,sequence=sequence,kind=kind,projection_index=int(raw.panel_cursor) if kind=='projection' else None,
                attention_index=int(raw.attention_cursor) if kind=='attention' else None,phase=len(boundaries),
                program=files['program.bin'],source=files['program.S'],runtime_program=runtime_program_record,
                runtime_source=runtime_source_record,runtime_epilogue_sha256=digest(EPILOGUE),
                initial=initial_record,output=files['expected-output.bin'],
                output_offset=offset,output_bytes=extent,expected_arena_sha256=digest(expected_arena),
                observed_dma=result['execution'].get('observed'),requested_dma=item['requested_dma'],provenance=result['provenance'])
            if 'fp8_stats' in result:event['fp8_stats']=result['fp8_stats']
            if item.get('fusion',{}).get('output_dtype')=='fp8':
                event.update(output_dtype='fp8',output_scale_exponent=item['fusion']['output_scale_exponent'])
            if cached_event:
                require(event==cached_event,'reconstructed durable event identity')
                require(json.loads(read(folder/'launch.json'))==cached_event,'durable launch receipt identity')
            else:
                write_json(folder/'launch.json',event)
                with ledger.open('a') as stream:stream.write(json.dumps(event)+'\n')
            events.append(event)
            return 0
        except BaseException as exc:errors.append(dict(where='device-reply',index=int(index),error=str(exc)));return 1
    phase_type=ct.CFUNCTYPE(ct.c_int,ct.c_uint,ct.c_uint,ct.c_uint)
    @phase_type
    def observe(kind,ordinal,layer):
        try:
            index=len(boundaries);entry=ranges[index]
            require((kind,ordinal,layer)==(('vision','prefix','expert').index(entry['kind']),entry['ordinal'],entry['layer']),'phase identity')
            require(raw.panel_cursor==entry['launch_end'] and raw.call_cursor==entry['call_end'] and raw.attention_cursor==index+1,'phase drained cursors')
            start=entry['launch_start']+index;end=entry['launch_end']+index+1
            cycle_slots=(ct.c_uint32*2915).in_dll(lib,'smol_full_cycles');values=list(cycle_slots)
            require(not any(values[:start]) and not any(values[end:]) and all(values[start:end]),'phase observation slots')
            observations.append(dict(index=index,launch_start=start,launch_end=end,native_synthetic_protocol_deltas=values[start:end],measured_model_cycles=False))
            ct.memset(ct.addressof(cycle_slots),0,ct.sizeof(cycle_slots));cursor[:]=[1,index+1,kind,ordinal,layer,raw.panel_cursor,raw.call_cursor,end,0,0]
            boundary=dict(entry,projection_launch_start=entry['launch_start'],projection_launch_end=entry['launch_end'],launch_start=start,launch_end=end,regions=regions())
            boundaries.append(boundary);write_json(run_dir/'boundaries'/f'{index+1:03d}.json',boundary)
            for name,size in scratch.items():ct.memset(address(name,size),0,size)
            ct.memset(raw.dram,0,raw.dram_capacity)
            print(json.dumps(dict(phase=index,projection_calls=raw.call_cursor,projection_panels=raw.panel_cursor,attention_events=raw.attention_cursor,device_events=len(events))),flush=True)
            if limit and len(boundaries)>=limit:return 1
            return 0
        except BaseException as exc:errors.append(dict(where='phase-observe',index=len(boundaries),error=str(exc)));return 1
    lib.smol_phase_configure.argtypes=[ct.c_uint,phase_type];lib.smol_phase_configure(1,observe)
    lib.smol_native_map.argtypes=[ct.c_size_t,ct.c_size_t];require(lib.smol_native_map(raw.dram,raw.dram_capacity)==0,'native arena mapping')
    for name in ('smol_native_imem_base','smol_native_csr_base','smol_native_imem_bytes'):getattr(lib,name).restype=ct.c_size_t
    lib.smol_native_configure.argtypes=[ct.c_size_t,ct.c_size_t,ct.c_size_t,reply_type,ct.c_void_p]
    require(lib.smol_native_configure(lib.smol_native_imem_base(),lib.smol_native_csr_base(),lib.smol_native_imem_bytes(),reply,None)==0,'native MMIO configure')
    start=time.monotonic()
    try:
        lib.atlas_workload_main()
        lib.smol_native_output.restype=ct.c_void_p;lib.smol_native_output_bytes.restype=ct.c_size_t
        (run_dir/'uart.log').write_bytes(ct.string_at(lib.smol_native_output(),lib.smol_native_output_bytes()))
        write_json(run_dir/'errors.json',errors)
        actual_state=dict(projection_panels=raw.panel_cursor,projection_calls=raw.call_cursor,attention_events=raw.attention_cursor,
            failed=raw.failed,result=raw.result,starts=lib.smol_native_starts(),protocol_error=lib.smol_native_error(),
            tohost=ct.c_uint64.in_dll(lib,'tohost').value,graph=list((ct.c_uint32*4).in_dll(lib,'smol_full_graph_state')),
            vision=list((ct.c_uint32*8).in_dll(lib,'smol_full_vision_states')))
        complete=(not limit and not errors and (len(calls),len(projection),len(attention),len(events),len(boundaries))==(1421,2715,200,2915,200)
            and (raw.panel_cursor,raw.call_cursor,raw.attention_cursor,raw.failed,raw.result)==(2715,1421,200,0,0)
            and actual_state['tohost']==1 and actual_state['starts']==2915 and not actual_state['protocol_error']
            and actual_state['graph']==[1,10,1273,0] and actual_state['vision']==[1,74,1,0]*2
            and all('output' in call for call in calls))
        references=[];trajectory_arrays={};geometries=reference_geometry()
        for item in contract['reference_buffers']:
            shape,dtype=geometries[item['name']];value=ct.string_at(address(item['actual_symbol'],item['bytes']),item['bytes'])
            trajectory_arrays[item['name']]=np.frombuffer(value,dtype=np.dtype(dtype)).reshape(shape).copy()
            current=blob(run_dir/'references'/(item['name']+'.bin'),value,dtype=dtype,shape=shape)
            references.append(dict(item,**current,baseline_sha256=item['sha256'],changed_from_baseline=current['sha256']!=item['sha256']))
        np.savez(run_dir/'trajectory.npz',**trajectory_arrays)
        trajectory=blob(run_dir/'trajectory.npz',read(run_dir/'trajectory.npz'))
        source_bookends=[dict(path=name,sha256=value,bytes=safe(name).stat().st_size) for name,value in {**native_pins,**capture_binding}.items()]
        for record in input_provider:
            require(sha(record['path'])==record['sha256'],'compiled input provider bookend')
            source_bookends.append(dict(path=record['path'],sha256=record['sha256'],bytes=record['bytes']))
        for module,parameter in baseline_replay['parameters'].items():
            if 'bias' in parameter:
                value=parameter['bias'];member=Path(value['file'])
                require(not member.is_absolute() and '..' not in member.parts,'relative bias member')
                bound=contract['host_parameters']['bias.'+module];raw_bias=bundle_read(bound['file'])
                require(digest(raw_bias)==value['sha256'] and len(raw_bias)==value['bytes'],'unchanged source bias')
                blob(run_dir/member,raw_bias)
        replay=dict(baseline_replay,calls=calls,trajectory=trajectory,capture_backend='Merlin C graph',
            source_bookends=source_bookends,historical_source_bookends=historical_bookends,historical_source=historical_source,
            historical_command=historical_command,native_capture_command=[sys.executable,*sys.argv],
            compiled_input_provider=input_receipt,historical_raw_sources_revalidated=False,
            metadata_relocation=metadata_relocation(),
            status='PASS_ATLAS_ARITHMETIC_FULL_POLICY_REPLAY' if complete else 'INCOMPLETE_NATIVE_C_RECAPTURE',reference_generation_only=True)
        phase_reference=dict(status='PASS_NATIVE_PHASE_REFERENCE' if complete else 'INCOMPLETE_NATIVE_PHASE_REFERENCE',mode='phased',
            initial_boundary=initial,boundaries=boundaries,fixed_live_regions=live,excluded_scratch=scratch,
            native_protocol_observations=observations,synthetic_counter_is_timing=False,device_execution=False,guest_execution=False,
            rv64_resume_admitted=False,reference_generation_only=True)
        write_json(run_dir/'phase-reference.json',phase_reference);write_json(run_dir/'replay.json',replay)
        projection_reference=dict(kind='ordered_native_C_Atlas_FP8_BF16_arithmetic_reference_not_device_execution',calls=calls,kernels=projection,
            source_bookends=[dict(path=name,sha256=value,bytes=safe(name).stat().st_size) for name,value in {**native_pins,**capture_binding}.items()],
            source_bookends_equal=True,fresh_model_executed=False,capture_backend='Merlin C graph',original_model_all_launch_gate=False)
        replay['projection_reference']=projection_reference;write_json(run_dir/'replay.json',replay)
        # A complete publication requires successful process exits as well as all
        # replies. Close before publishing the manifest so an exit failure cannot
        # leave a COMPLETE capture authoritative.
        for name in list(workers):
            worker=workers.pop(name);worker.close()
        manifest=dict(fusion_set=FUSION_SET,execution_options={key:str(value) for key,value in EXECUTION_OPTIONS.items()},schema='atlas.smol_native_recapture.v2',status='COMPLETE_REFERENCE_GENERATION' if complete else 'INCOMPLETE',
            capture_backend='Merlin C graph',reference_generation_only=True,projection_reference=projection_reference,
            attention_reference=dict(kernels=attention),device_events=events,reference_buffers=references,
            phase_reference=str(run_dir/'phase-reference.json'),source_sha256=native_pins,
            elapsed_seconds=time.monotonic()-start,errors=errors,actual_state=actual_state,original_model_all_launch_gate=False,guest_execution=False,
            capture_source_sha256=capture_binding,python_version=sys.version,numpy_version=np.__version__)
        manifest['compiled_input_provider']=input_receipt
        manifest['reply_ledger_sha256']=REPLIES_SHA256 if REPLIES else None
        manifest['reply_reuse_policy']='exact-matching-launches' if REUSE_MATCHING_REPLIES else 'strict-complete-ledger'
        manifest['reused_reply_launches']=sum(event['provenance'].get('backend')=='identity-bound-captured-reply' for event in events)
        manifest['fresh_model_executed']=any(event['provenance'].get('backend')!='identity-bound-captured-reply' for event in events)
        for name,expected in capture_binding.items():require(sha(name)==expected,'capture source bookend '+name)
        for name,expected in native_pins.items():require(sha(name)==expected,'native source bookend '+name)
        write_json(run_dir/'manifest.json',manifest)
        if complete:compare_baseline(run_dir,manifest,phase_reference,contract,trajectory_arrays)
        require(complete or bool(limit) and len(boundaries)==limit and not errors,'native recapture incomplete: '+str(errors[:1])+str(actual_state))
        return manifest
    finally:
        lib.smol_native_unmap()
        for worker in workers.values():worker.close()



def compare_baseline(run_dir, manifest, phase_reference, contract, arrays):
    old=json.loads(read(BASE/'phase-canonical-v4/phase-reference.json'))
    before=[old['initial_boundary'],*old['boundaries']]
    after=[phase_reference['initial_boundary'],*phase_reference['boundaries']]
    boundary_diffs=[]
    for index,(prior,current) in enumerate(zip(before,after)):
        changes={}
        for name,item in current['regions'].items():
            previous=prior['regions'][name]
            changed=item['sha256']!=previous['sha256']
            entry=dict(changed=changed,baseline_sha256=previous['sha256'],current_sha256=item['sha256'],
                baseline_bytes=previous['bytes'],current_bytes=item['bytes'])
            if changed and item['bytes']==previous['bytes'] and previous.get('file') and safe(BASE/'phase-canonical-v4'/previous['file']).is_file():
                a=np.frombuffer(read(item['path']),dtype=np.uint8)
                b=np.frombuffer(read(BASE/'phase-canonical-v4'/previous['file']),dtype=np.uint8)
                entry['different_bytes']=int(np.count_nonzero(a!=b))
            changes[name]=entry
        boundary_diffs.append(dict(boundary=index,next_segment=index,regions=changes,
            changed_regions=sum(row['changed'] for row in changes.values())))
    baseline=next(x for x in contract['reference_buffers'] if x['name']=='final_actions')
    a=arrays['final_actions'].astype(np.float64);b=np.frombuffer(bundle_read(baseline['file']),dtype='<f4').reshape(a.shape).astype(np.float64)
    difference=a-b;nonzero=b!=0
    action=dict(finite=bool(np.isfinite(a).all() and np.isfinite(b).all()),elements=int(a.size),
        changed_elements=int(np.count_nonzero(a!=b)),max_abs=float(np.max(np.abs(difference))),
        mean_abs=float(np.mean(np.abs(difference))),rmse=float(np.sqrt(np.mean(difference*difference))),
        relative_l2=float(np.linalg.norm(difference)/np.linalg.norm(b)),
        max_relative_nonzero_baseline=float(np.max(np.abs(difference[nonzero]/b[nonzero]))) if nonzero.any() else None,
        baseline_zero_elements=int(np.count_nonzero(~nonzero)),policy_quality_qualified=False)
    write_json(run_dir/'baseline-differences.json',dict(action=action,boundaries=boundary_diffs,
        boundary_count=len(boundary_diffs),reference_buffers=manifest['reference_buffers']))


def interpreter(value):
    """Preserve virtualenv executable spelling, screening every symlink target first."""
    from atlas_path_guard import lexical_path
    original=lexical_path(value).absolute()
    def follow(path,depth=0):
        require(depth<32,'interpreter symlink loop')
        lexical_path(path,absolute=True)
        current=Path(path.anchor)
        for index,part in enumerate(path.parts[1:]):
            current/=part
            if current.is_symlink():
                target=Path(os.readlink(current))
                lexical_path(target)
                if not target.is_absolute():target=current.parent/target
                # Normalize parent components only after screening the target spelling.
                target=Path(os.path.normpath(target))
                return follow(target.joinpath(*path.parts[index+2:]),depth+1)
        return safe(current)
    follow(original)
    return original


def main(argv=None):
    global ATLAS, BASE, BUNDLE, BASE_WORK, NATIVE, PROVIDERS, PYTHON, MODEL_ROOT, MODEL_PIN, REPLIES, REPLIES_SHA256, FUSION_SET, EXECUTION_OPTIONS, REUSE_MATCHING_REPLIES
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlas-root',type=safe,default=ATLAS)
    parser.add_argument('--python',type=interpreter,default=Path(sys.executable),help='interpreter with NumPy and PyTorch for original-model workers')
    for name in ('model-root','bundle','baseline-root','providers','out'):
        parser.add_argument('--'+name,type=safe,required=True)
    parser.add_argument('--native-source',type=safe,default=PACKAGE,help='optional native transport override; maintained source is bundled')
    parser.add_argument('--model-pin',required=True,help='expected original npu_model commit')
    parser.add_argument('--fusion-set',choices=FUSION_SETS,default='none',help='explicit opt-in BF16 producer policy')
    parser.add_argument('--native-projection-model',type=safe,help='optional admitted native70 development model; final original47 gate remains required')
    parser.add_argument('--native-build-dir',type=safe)
    parser.add_argument('--native-admission-root',type=safe)
    parser.add_argument('--native-admission-sha256')
    parser.add_argument('--fast-attention-admission',type=safe,help='optional admitted IPT development data; final original47 gate remains required')
    parser.add_argument('--observation',type=safe,help='optional raw observation archive; SHA required when supplied; compiled bundle remains the actual host input')
    parser.add_argument('--observation-sha256')
    parser.add_argument('--relocate',action='append',default=[],metavar='OLD=NEW',help='explicit authenticated legacy source metadata prefix mapping')
    parser.add_argument('--replies',type=safe,help='optional previously admitted JSONL launch ledger; regenerates host state without a fresh model claim')
    parser.add_argument('--replies-sha256',help='required exact ledger identity')
    parser.add_argument('--reuse-matching-replies',action='store_true',help='opt in to reusing only exactly identical whole-arena launches from --replies; execute all changed launches')
    parser.add_argument('--shared-phase-reference',type=safe,help='optional immutable phase JSON whose exact matching blobs may be hardlinked')
    parser.add_argument('--shared-phase-reference-sha256')
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--execute',action='store_true')
    parser.add_argument('--phase-limit',type=int,choices=range(1,201))
    parser.add_argument('--resume',action='store_true',help='replay authenticated durable launch prefix while recomputing host state')
    args=parser.parse_args(argv)
    require(args.prepare or args.execute,'select --prepare or --execute')
    require(args.atlas_root==ATLAS,'atlas-root must be the checkout containing this package')
    ATLAS=args.atlas_root;BASE=args.baseline_root;BUNDLE=args.bundle;BASE_WORK=BASE/'host-baseline-v4-build'
    PROVIDERS=args.providers;NATIVE=args.native_source;PYTHON=str(args.python)
    MODEL_ROOT=args.model_root;MODEL_PIN=args.model_pin;FUSION_SET=args.fusion_set
    native_fields=('native_projection_model','native_build_dir','native_admission_root','native_admission_sha256')
    require(all(getattr(args,key) is not None for key in native_fields) or all(getattr(args,key) is None for key in native_fields),
        'native projection requires model, build directory, catalog, and digest')
    EXECUTION_OPTIONS={key:getattr(args,key) for key in (*native_fields,'fast_attention_admission') if getattr(args,key) is not None}
    for mapping in args.relocate:
        old,new=mapping.split('=',1);PATH_MAP.append((str(lexical_absolute_path(old)),str(lexical_absolute_path(new))))
    PATH_MAP.sort(key=lambda pair:len(pair[0]),reverse=True)
    require((args.observation is None)==(args.observation_sha256 is None),'observation requires its SHA256')
    if args.observation:require(sha(args.observation)==args.observation_sha256,'observation archive identity')
    require((args.replies is None)==(args.replies_sha256 is None),'replies requires its SHA256')
    require(not args.reuse_matching_replies or args.replies is not None,'matching reply reuse requires a hash-bound ledger')
    REUSE_MATCHING_REPLIES=args.reuse_matching_replies
    if args.replies:
        REPLIES_SHA256=sha(args.replies);require(REPLIES_SHA256==args.replies_sha256,'reply ledger identity')
        REPLIES=[json.loads(line) for line in read(args.replies).decode().splitlines()]
        require(len(REPLIES)==2915,'complete prebound reply inventory')
        for event in REPLIES:
            for key in ('initial','output','program','source','runtime_program','runtime_source'):
                item=event[key];SHARED_BLOBS[(item['sha256'],item['bytes'])]=str(safe(item['path']))
    require((args.shared_phase_reference is None)==(args.shared_phase_reference_sha256 is None),'shared phase reference requires SHA256')
    if args.shared_phase_reference:
        require(sha(args.shared_phase_reference)==args.shared_phase_reference_sha256,'shared phase reference identity')
        phases=json.loads(read(args.shared_phase_reference))
        for boundary in [phases['initial_boundary'],*phases['boundaries']]:
            for item in boundary['regions'].values():
                SHARED_BLOBS[(item['sha256'],item['bytes'])]=str(safe(item['path']))
    if args.prepare:prepare(args.out)
    if args.execute:run(args.out,args.phase_limit,args.resume)

if __name__=='__main__':main()
