"""Strict prepared-asset validation for explicit Smol producer fusion profiles."""
import hashlib
import numpy as np
from smol_full_plan import calls
import kernel_library  # noqa: E402,F401
from atlas_program import dma_load_bytes

SETS = ('none', 'vision-bias', 'vision-bias-gelu', 'fc1-bias-gelu', 'connector',
        'vision-bias-gelu-connector', 'fc1-bias-gelu-fp8')


def digest(raw): return hashlib.sha256(raw).hexdigest()


def profile(data):
    mode = data.get('fusion_set', 'none')
    if mode not in SETS: raise ValueError('Unsupported explicit Smol fusion profile')
    return mode


def operations(mode, call):
    if mode not in SETS: raise ValueError('Unsupported Smol fusion profile')
    if mode == 'none' or call['site'] != 7: return ()
    if call['role'] == 7:
        return ('connector_scale',) if mode in ('connector', 'vision-bias-gelu-connector') else ()
    if mode in ('fc1-bias-gelu', 'fc1-bias-gelu-fp8'):
        return ('bias', 'gelu_tanh') if call['role'] == 5 else ()
    if mode == 'connector': return ()
    return ('bias', 'gelu_tanh') if call['role'] == 5 and mode != 'vision-bias' else ('bias',)


def geometry(row):
    m,k,n = row['physical_mkn']
    fusion = row.get('fusion')
    output = 2*m*n
    extra = 0
    if fusion:
        ops = tuple(fusion.get('operations', ()))
        if ops not in (('bias',), ('bias','gelu_tanh'), ('connector_scale',)):
            raise ValueError('Unsupported ordered fusion operations')
        dtype = fusion.get('output_dtype')
        if dtype not in ('bf16','fp8') or fusion.get('output_layout') != 'tiles':
            raise ValueError('Unsupported fused output dtype/layout')
        if dtype == 'fp8':
            if ops != ('bias','gelu_tanh') or not row['module'].endswith('.mlp.fc1'):
                raise ValueError('Only vision fc1 bias/GELU may produce FP8')
            scale = fusion.get('output_scale_exponent')
            if type(scale) is not int or not -126 <= scale <= 127: raise ValueError('Invalid fixed FP8 output scale')
            output = m*n
        extra = n*64 if 'bias' in ops else 0
        end = row['output_offset']+output+256
        if (fusion.get('output_offset') != row['output_offset'] or fusion.get('output_bytes') != output
            or fusion.get('extra_offset') != end or fusion.get('extra_bytes') != extra
            or fusion.get('requested_dma') != row['requested_dma']):
            raise ValueError('Fusion extent, extra offset, or DMA metadata differs')
    total = row['output_offset']+output+256+(extra+256 if extra else 0)
    if (row['output_offset'] != m*k+k*n+256 or row.get('total_dram_bytes',row.get('dram_bytes')) != total
        or row['requested_dma']['store_bytes'] != output or total > 64*1024*1024):
        raise ValueError('Smol producer output/arena geometry differs')
    if fusion and (fusion.get('total_dram_bytes') != total or fusion.get('dram_bytes') != total):
        raise ValueError('Fusion total extent differs')
    return output,extra,total


def record(raw, filename, *, dtype=None, shape=None):
    result = dict(file=filename, bytes=len(raw), sha256=digest(raw))
    if dtype is not None: result['dtype']=dtype
    if shape is not None: result['shape']=shape
    return result


def read_record(item, read):
    if not isinstance(item,dict) or not isinstance(item.get('file'),str): raise ValueError('Missing prepared payload record')
    raw = read(item['file'])
    if len(raw) != item.get('bytes') or digest(raw) != item.get('sha256'):
        raise ValueError('Prepared fusion/typed payload identity differs')
    return raw


def initial_dram(row, read):
    output,extra,total=geometry(row)
    raw=read(row['activation'])+read(row['weights'])+b'\xa5'*256+b'\xcd'*output+b'Z'*256
    if row.get('fusion'):
        payload=read_record(row['fusion']['extra'],read)
        if len(payload)!=extra or digest(payload)!=row['fusion']['extra_sha256']:
            raise ValueError('Prepared fusion extras differ')
        if extra:raw+=payload+b'Z'*256
    if len(raw)!=total or digest(raw)!=row['initial_sha256']:
        raise ValueError('Prepared producer arena identity differs')
    return raw


def regenerate(row, initial, *, bias_bf16=None, weight_source=None):
    """Independently reconstruct the exact supported source and staged arena."""
    output,extra,total=geometry(row)
    from smol_fusion_program import stage_panel
    from smol_fusion_fp8 import stage_fc1_fp8
    meta=row.get('fusion')
    if not meta: raise ValueError('Explicit fusion metadata required')
    m,k,n=row['physical_mkn']
    baseline=dict(row,dram_bytes=row['output_offset']+2*m*n+256,total_dram_bytes=row['output_offset']+2*m*n+256,
        requested_dma=dict(load_bytes=dma_load_bytes(m,k,n,traffic_reuse=True),store_bytes=2*m*n),
        recipe='traffic-reuse-v1',program_policy='traffic-reuse-v1')
    arena=initial[:row['output_offset']]+b'\xcd'*(2*m*n)+b'Z'*256
    if meta['output_dtype']=='fp8':
        staged=stage_fc1_fp8(baseline,arena,bias_bf16=bias_bf16,output_scale_exponent=meta['output_scale_exponent'], weight_source=weight_source)
    else:staged=stage_panel(baseline,arena,operations=tuple(meta['operations']),bias_bf16=bias_bf16,weight_source=weight_source)
    if staged.arena!=initial or len(initial)!=total:
        raise ValueError('Actual fusion staging differs from independent producer recipe')
    if set(meta)-set(staged.metadata)-{'extra','bias'}:
        raise ValueError('Unknown producer fusion metadata')
    if any(meta.get(key)!=value for key,value in staged.metadata.items() if weight_source is None or key != "source_sha256"):
        raise ValueError('Fusion metadata differs from independently regenerated recipe')
    return staged


def validate_program(row, read):
    initial=initial_dram(row,read)
    bias=read_record(row['fusion']['bias'],read) if 'bias' in row['fusion']['operations'] else None
    staged=regenerate(row,initial,bias_bf16=bias)
    if read(row['source'])!=staged.source.encode():raise ValueError('Prepared fused source differs from exact public recipe')
    return staged


def runtime_fields(row):
    fusion=row.get('fusion')
    if not fusion:return 0,0,0,0
    _,extra,_=geometry(row)
    flags=sum({'bias':1,'gelu_tanh':2,'connector_scale':4}[op] for op in fusion['operations'])
    fp8=fusion['output_dtype']=='fp8'
    return flags|(8 if fp8 else 0),fusion['extra_offset'] if extra else 0,extra,fusion['output_scale_exponent'] if fp8 else 0


def validate_profile(data):
    mode=profile(data)
    plan=calls()
    for row in data['kernels']:
        expected=operations(mode,plan[row['call']])
        actual=tuple(row.get('fusion',{}).get('operations',()))
        if actual!=expected:raise ValueError('Prepared producer operations differ from explicit fusion profile')
        if actual:
            geometry(row)
            if (row.get('recipe')!='traffic-reuse-v1' or row['dram_base']!=0x90000000
                or (row['fusion']['output_dtype']=='fp8')!=(mode=='fc1-bias-gelu-fp8')):
                raise ValueError('Fusion recipe/output dtype differs from selected profile')
    if mode=='none' and data.get('call_views'):raise ValueError('Typed views require an explicit FP8 profile')
    return mode


def diagnostic(raw, rows, width, scale):
    values=np.frombuffer(raw,np.uint8).reshape(rows//32,width//32,32,32).transpose(0,2,1,3).reshape(rows,width)
    exponent,mantissa=(values>>3)&15,values&7
    if np.any((values&127)==127) or np.any((exponent==0)&(mantissa!=0)):
        raise ValueError('Unsupported FP8 NaN/subnormal in typed producer view')
    value=np.where(exponent!=0,np.ldexp(1+mantissa.astype(np.float32)/8,exponent.astype(np.int32)-7),0)
    value=np.where(values&128,-value,value).astype(np.float32)*np.float32(2.0**scale)
    bits=value.astype('<f4').view('<u4')
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype('<u2').tobytes()


def validate_views(data, read):
    mode=validate_profile(data)
    views=data.get('call_views',[])
    producers={row['call'] for row in data['kernels'] if row.get('fusion',{}).get('output_dtype')=='fp8'}
    expected=producers|{index+1 for index in producers}
    if len(views)!=len(expected) or {row['index'] for row in views}!=expected:
        raise ValueError('Incomplete or duplicated prepared typed-call inventory')
    if not expected:return
    if mode!='fc1-bias-gelu-fp8':raise ValueError('Unsupported typed-call profile')
    plan=calls();by_call={row['index']:row for row in views};payloads={}
    for item in views:
        call=plan[item['index']]
        if item['module']!=call['module']:raise ValueError('Typed call module differs')
        side='output' if item['index'] in producers else 'input'
        width=call['output_width'] if side=='output' else call['input_width'];rows=call['rows']
        typed=item[side+'_fp8'];scale=item.get(side+'_decode_exponent')
        if (typed.get('dtype')!='uint8' or typed.get('shape')!=[rows//32,width//32,32,32]
            or item.get(side+'_source_format')!='fp8_tiles' or item.get(side+'_diagnostic_only') is not True
            or type(scale) is not int or not -126<=scale<=127):raise ValueError('Typed view dtype/layout/scale/diagnostic marker differs')
        raw=read_record(typed,read)
        if len(raw)!=rows*width:raise ValueError('Typed view byte extent differs')
        diag=item[side+'_diagnostic']
        if diag.get('dtype')!='torch.bfloat16' or diag.get('shape')!=[1,rows,width]:raise ValueError('Diagnostic BF16 geometry differs')
        if read_record(diag,read)!=diagnostic(raw,rows,width,scale):raise ValueError('Diagnostic BF16 bytes differ from actual FP8 view')
        payloads[item['index']]=raw
    for index in producers:
        source,target=by_call[index],by_call[index+1]
        scale=source['output_decode_exponent']
        if scale!=target['input_decode_exponent'] or payloads[index]!=payloads[index+1]:
            raise ValueError('Prepared FP8 producer/consumer bytes or scales differ')
        rows=[row for row in data['kernels'] if row['call']==index]
        consumers=[row for row in data['kernels'] if row['call']==index+1]
        if any(row['fusion']['output_scale_exponent']!=scale for row in rows) or any(row['a_scale']!=scale for row in consumers):
            raise ValueError('Fixed FP8 scale differs from producer/consumer descriptors')
        if b''.join(read(row['expected']) for row in rows)!=payloads[index]:raise ValueError('Actual FP8 output panels differ from typed callback view')
        if b''.join(read(row['activation']) for row in consumers)!=payloads[index+1]:raise ValueError('Actual FP8 A panels differ from typed callback view')
