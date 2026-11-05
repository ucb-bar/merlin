"""Persistent original npu_model worker with explicit RTL arithmetic.

These reference generation replies are separate from whole-ELF admission.
"""
import argparse
import contextlib
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct
import sys
import traceback

sys.dont_write_bytecode=True
ROOT=Path(__file__).absolute().parent
sys.path.insert(0,str(ROOT.parent))
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--backend',choices=('attention','projection'),required=True)
parser.add_argument('--workspace',type=checked,required=True)
parser.add_argument('--atlas-root',type=checked,required=True)
parser.add_argument('--native-admission-root',type=checked)
parser.add_argument('--native-admission-sha256')
parser.add_argument('--original-model-root',type=checked)
parser.add_argument('--fast-attention-admission',type=checked)
args=parser.parse_args()
sys.path.insert(0,str(args.atlas_root/'tools'))
import numpy as np
from smol_capture import model as unit
WORKSPACE=unit.safe_path(args.workspace)
unit.model_binding()
NATIVE_REGISTRY = FAST_ADMISSION = None
if args.native_admission_root is not None:
    if args.backend != 'projection' or not args.native_admission_sha256 or args.original_model_root is None:
        raise ValueError('native admission requires projection, catalog digest, and original model root')
    from npu_model.configs.native_systolic import backend_metadata
    from smol_capture.admission import AdmissionRegistry
    NATIVE_REGISTRY=AdmissionRegistry(args.native_admission_root,args.native_admission_sha256,
        atlas_root=args.atlas_root,original_model_root=args.original_model_root,
        native_model_root=unit.MODEL_ROOT,native_backend_metadata=backend_metadata())
elif args.native_admission_sha256 is not None or args.original_model_root is not None:
    raise ValueError('incomplete native admission configuration')
if args.fast_attention_admission is not None:
    if args.backend != 'attention' or NATIVE_REGISTRY is not None:
        raise ValueError('fast attention requires the original attention worker')
    from smol_capture.fast_attention import load_admission
    FAST_ADMISSION=load_admission(data_root=args.fast_attention_admission)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def owned(value):
    path=unit.safe_path(value)
    if not path.is_relative_to(WORKSPACE):
        raise ValueError('worker paths must belong to configured output workspace')
    return path


def write_program(source,out):
    program,digest=unit.assemble(source)
    raw=b''.join(struct.pack('<I',word) for word in program.assemble())
    out.write_bytes(raw)
    return dict(program_bin=str(out),program_words=len(program),program_sha256=digest)


def prepare_attention(request):
    if args.backend!='attention':
        raise ValueError('attention preparation requires attention backend')
    from atlas_attention_v2_program import AttentionV2Config,stage_v2,assembly_v2
    c=AttentionV2Config(**request['config'])
    out=owned(request['output_dir']);out.mkdir(parents=True,exist_ok=True)
    with np.load(owned(request['input_npz']),allow_pickle=False) as data:
        q,k,v=[data[key].copy() for key in ('q','k','v')]
        mask=data['mask'].astype(bool) if 'mask' in data else None
    table=None
    if mask is not None:
        if mask.shape!=(c.query_rows,c.keys): raise ValueError('logical mask extent differs')
        padded=np.zeros((c.query_tiles*32,c.key_tiles*32),dtype=bool)
        padded[:c.query_rows,:c.keys]=mask
        table=[[padded[i*32:(i+1)*32,j*32:(j+1)*32] for j in range(c.key_tiles)] for i in range(c.query_tiles)]
    arena,offsets=stage_v2(c,q,k,v,table)
    source=assembly_v2(c,offsets)
    (out/'program.S').write_text(source)
    (out/'arena-initial.bin').write_bytes(arena)
    (out/'metadata.bin').write_bytes(arena[offsets['descriptors']:offsets['output']-256])
    mask_file=None
    if c.mask!='none':
        if mask is None: raise ValueError('text attention requires exact logical mask')
        (out/'logical-mask.bin').write_bytes(mask.astype(np.uint8).tobytes())
        mask_file=str(out/'logical-mask.bin')
    result=dict(config=asdict(c),offsets=offsets,program_source=str(out/'program.S'),
        arena_initial=str(out/'arena-initial.bin'),metadata_bin=str(out/'metadata.bin'),
        expected_mask_bin=mask_file,requested_dma=dict(load_bytes=offsets['output']-256,store_bytes=offsets['output_bytes']))
    result.update(write_program(source,out/'program.bin'))
    (out/'prepared.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def execute(request,attention):
    out=owned(request['output_dir']);out.mkdir(parents=True,exist_ok=True)
    initial=owned(request['initial_arena']).read_bytes()
    if request.get('expected_initial') and initial!=owned(request['expected_initial']).read_bytes():
        raise ValueError('actual C staged arena differs from independent stage_v2 arena')
    if attention:
        from atlas_attention_v2_program import AttentionV2Config,assembly_v2
        c=AttentionV2Config(**request['config'])
        source=assembly_v2(c,request['offsets'])
        offset=request['offsets']['output'];extent=request['offsets']['output_bytes']
    else:
        source=owned(request['program_source']).read_text()
        offset=request['output_offset'];extent=request['output_bytes']
    binding=unit.model_binding()
    admission=None
    if NATIVE_REGISTRY is not None:
        program,_=unit.assemble(source)
        binary=b''.join(struct.pack('<I',word) for word in program.assemble())
        admission=NATIVE_REGISTRY.validate(binary,source,request['descriptor'],initial=initial)
    stats_record=None
    if attention and FAST_ADMISSION is not None:
        from smol_capture.fast_attention import execute_admitted
        result,provenance=execute_admitted(c,request['offsets'],initial,admission=FAST_ADMISSION)
    else:
        descriptor=request.get('descriptor',{})
        if descriptor.get('output_dtype')=='fp8':
            from smol_capture.stats import observe_fp8
            with observe_fp8(unit,descriptor['output_scale_exponent'],extent) as stats:
                result=unit.execute(source,initial,offset,extent)
            stats_record=stats.report(result.output)
        else:
            result=unit.execute(source,initial,offset,extent)
        provenance=dict(backend='admitted-native-mxu0' if admission else 'original-npu-model',
            gate_of_record=False,model=binding,numerics='rtl',math_override=bool(admission),
            all_launch_original_gate_required=True)
        if admission is not None:
            provenance.update(native_admission=admission,native_backend=backend_metadata())
    if unit.model_binding()!=binding:
        raise ValueError('model binding changed during execution')
    (out/'output.bin').write_bytes(result.output)
    (out/'arena-expected.bin').write_bytes(result.arena)
    record=dict(output_path=str(out/'output.bin'),arena_path=str(out/'arena-expected.bin'),
        execution=result.metadata(),provenance=provenance,transient_arena=True,initial_sha256=sha(initial),expected_arena_sha256=sha(result.arena))
    if stats_record is not None:record['fp8_stats']=stats_record
    (out/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


for line in sys.stdin:
    request=None
    try:
        request=json.loads(line)
        with contextlib.redirect_stdout(sys.stderr):
            op=request['op']
            if op=='assemble':
                target=owned(request['output_path']);target.parent.mkdir(parents=True,exist_ok=True)
                response=write_program(owned(request['source_path']).read_text(),target)
            elif op=='prepare_attention': response=prepare_attention(request)
            elif op=='execute_attention': response=execute(request,True)
            elif op=='execute_projection': response=execute(request,False)
            else: raise ValueError('unknown operation '+str(op))
        print(json.dumps(dict(id=request.get('id'),ok=True,result=response)),flush=True)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps(dict(id=request.get('id') if request else None,ok=False,
            error=dict(type=type(error).__name__,message=str(error)))),flush=True)
