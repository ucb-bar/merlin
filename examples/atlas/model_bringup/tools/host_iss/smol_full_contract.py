"""Exact full Smol policy ELF admission and continuous live completion checks."""
import hashlib
import io
import json
import struct
import zipfile

import numpy as np

from bundle import EPILOGUE, verify_kernel_receipt
from smol_full_plan import calls, reference_geometry, validate_reference
from smol_full_phases import execution_profile
import kernel_library  # noqa: E402,F401
from atlas_program import assembly, dma_load_bytes
from chipyard.arena import DEFAULT_ARENA, relocate, relocate_assembly
from placement import placement, arena_extent

CYCLE_CAP = 200_000_000
MAGIC = 0x534D4F4C46554C4C


def require(value, message):
    if not value: raise ValueError('Smol full: '+message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def finite(raw, dtype):
    value = np.frombuffer(raw, np.dtype(dtype))
    return (not np.any((value & 0x7f80) == 0x7f80) if value.dtype == np.uint16
            else bool(np.isfinite(value).all()))


def stop_sites(image):
    # The admitted startup owns the normal WFI after atlas_workload_main returns.
    start, trap = [image.symbols[name]['address'] for name in ('_start', 'atlas_block_trap')]
    require(image.entry == start == 0x80000000 and start < trap < start+4096, 'startup/trap bounds')
    raw, offset, sites = image.bytes_at(start, trap-start), 0, {}
    while offset < len(raw):
        half = int.from_bytes(raw[offset:offset+2], 'little')
        length = 2 if half & 3 != 3 else 4
        require(offset+length <= len(raw), 'startup instruction extent')
        if length == 4 and raw[offset:offset+4] == struct.pack('<I', 0x10500073): sites[start+offset] = 4
        offset += length
    require(len(sites) == 1, 'one normal startup WFI')
    for name in ('smol_full_segment_done', 'smol_full_segment_failed'):
        if name in image.symbols:
            named = {pc:length for pc,length,word in image.instructions(name)
                     if length == 4 and word == 0x10500073}
            require(len(named) == 1 and not set(named) & set(sites), 'distinct named full-policy park: '+name)
            sites.update(named)
    return sites


def disjoint(a, b):
    return a[0]+a[1] <= b[0] or b[0]+b[1] <= a[0]


def blob(image, name, raw, *, actual=False):
    row = image.symbols[name]
    require(row['size'] == len(raw), 'symbol extent '+name)
    address = row['address']
    segments = [s for s in image.segments if s['address'] <= address
                and address+len(raw) <= s['address']+s['memory_bytes']]
    require(len(segments) == 1 and bool(segments[0]['flags'] & 2) == actual,
            'symbol memory ownership '+name)
    if not actual: require(image.bytes_at(address, len(raw)) == raw, 'embedded bytes '+name)
    return address


def audit_image(image, sections, dram_bytes, physical=None, *, fusion=False):
    physical = placement({}) if physical is None else placement(dict(placement=physical))
    arena = physical['arena_base']
    arena_extent(dict(placement=physical), dram_bytes)
    require(image.entry == 0x80000000 and type(dram_bytes) is int
            and 0 < dram_bytes <= 64*1024*1024, 'entry/staging bound')
    require(len(image.segments) == 4, 'text/readonly/writable/arena load segments')
    for index, segment in enumerate(image.segments):
        begin, size = segment['address'], segment['memory_bytes']
        require(segment['physical'] == begin
                and physical['dram_base'] <= begin < begin+size <= physical['dram_base']+physical['dram_bytes']
                and 0 <= segment['file_bytes'] <= size and segment['offset']+segment['file_bytes'] <= len(image.raw)
                and segment['flags'] in (4, 5, 6), 'bounded physical host PT_LOAD')
        require(all(disjoint((begin,size),(s['address'],s['memory_bytes'])) for s in image.segments[:index]),
                'host PT_LOAD overlap')
    arenas = [s for s in image.segments if s['address'] == arena]
    require(len(arenas) == 1 and arenas[0]['file_bytes'] == 0
            and arenas[0]['memory_bytes'] == dram_bytes and arenas[0]['flags'] == 6, 'exact arena load reservation')
    host_segments = [s for s in image.segments if s is not arenas[0]]
    require(sorted(s['flags'] for s in host_segments) == [4,5,6], 'host segment permissions')
    for name, value in (('__atlas_dram_start',arena), ('__atlas_dram_end',arena+dram_bytes)):
        require(image.symbols[name]['address'] == value, 'reserved arena '+name)
    if sections is not None:
        for name in ('.bss', '.stack', '.atlas_dram'):
            require(sections[name][1] == 8, 'NOLOAD '+name)
        require(sections['.atlas_dram'][3] == arena and sections['.atlas_dram'][5] == dram_bytes,
                'exact NOLOAD device arena')
    stack = (image.symbols['__stack_bottom']['address'], 65536)
    require(image.symbols['__stack_top']['address'] == stack[0]+stack[1], 'stack extent')
    writable = next(s for s in host_segments if s['flags'] == 6)
    zero_begin = writable['address']+writable['file_bytes']
    bss_begin, bss_end = [image.symbols[name]['address'] for name in ('__bss_start','__bss_end')]
    require(zero_begin <= bss_begin <= bss_end <= stack[0]
            and stack[0]+stack[1] <= writable['address']+writable['memory_bytes']
            and bss_begin%8 == bss_end%8 == 0, 'loader/startup BSS and stack zero fill')
    if fusion:
        state=image.symbols['state']
        require(state['size']==104 and zero_begin<=bss_begin<=state['address']<state['address']+104<=bss_end
                and writable['address']<=state['address'] and state['address']+104<=writable['address']+writable['memory_bytes']
                and disjoint((state['address'],104),(arena,dram_bytes)),
                'actual fusion registry is zero-initialized BSS control storage outside the arena')
    stop_sites(image)
    return dict(host_begin=0x80000000, host_end=max(s['address']+s['memory_bytes'] for s in host_segments),
        host_file_bytes=sum(s['file_bytes'] for s in host_segments),
        host_zero_fill_bytes=sum(s['memory_bytes']-s['file_bytes'] for s in host_segments),
        stack_bytes=65536, dram_base=arena, dram_bytes=dram_bytes, placement=physical,
        segments=image.segments, host_and_arena_disjoint=True)


def selected_kernels(bundle, data, captured, assemble=None):
    """Bind selected recipes to the unchanged captured arithmetic operands."""
    physical = placement(bundle.provenance)
    require(placement(data) == physical, 'contract/provenance placement')
    policy = data.get('program_policy')
    require(policy in ('captured-baseline-v1', 'qualified-reuse-v1')
            and bundle.provenance.get('program_policy') == policy, 'selected program policy')
    require(len(captured) == len(data['kernels']), 'selected launch count')
    mutable = {'source', 'original', 'requested_dma', 'dram_base'}
    cache = set()
    verified = getattr(bundle, '_smol_verified_programs', set())
    for before, after in zip(captured, data['kernels']):
        require(set(after) == set(before) | {'program_policy'}
                and all(after[key] == value for key, value in before.items() if key not in mutable),
                'selected recipe changed captured nonprogram data')
        require(type(after['dram_base']) is int and after['dram_base'] == physical['arena_base'],
                'selected recipe arena base')
        choice = after['program_policy']
        require(choice in ('baseline-v1', 'traffic-reuse-v1')
                and (policy == 'qualified-reuse-v1' or choice == 'baseline-v1'), 'per-launch selected recipe')
        signature = (before['source'], before['original'], after['source'], after['original'],
                     after['dram_base'], choice)
        m, k, n = before['physical_mkn']
        if after.get('fusion'):
            from smol_fusion_assets import validate_program
            require(choice=='traffic-reuse-v1' and policy=='qualified-reuse-v1' and after['dram_base']==0x90000000,
                    'explicit qualified low-arena fusion recipe')
            staged=validate_program(after,bundle.read)
            original=bundle.read(after['original'])
            binding=(digest(staged.source.encode()),digest(original))
            if assemble is None:
                require(binding in verified,'fused binary requires pinned assembly admission')
            else:
                words=list(assemble(io.StringIO(staged.source)).assemble())
                require(original==struct.pack('<'+'I'*len(words),*words),'exact regenerated producer-plus-fusion binary')
                verified.add(binding)
            require(after['requested_dma']==staged.metadata['requested_dma'],'fused producer exact DMA accounting')
            cache.add(signature)
            continue
        if choice == 'traffic-reuse-v1':
            loads = dma_load_bytes(m, k, n, traffic_reuse=True)
            if signature not in cache:
                expected_source = assembly(m, k, n, before['output_offset'],
                                           before['output_rescale_exponent'], traffic_reuse=True)
                expected_source = relocate_assembly(expected_source, DEFAULT_ARENA, after['dram_base'])
                require(bundle.read(after['source']) == expected_source.encode(), 'selected public reuse source')
                original = bundle.read(after['original'])
                binding = (digest(expected_source.encode()), digest(original))
                if assemble is None:
                    require(binding in verified, 'selected reuse binary requires pinned assembly admission')
                else:
                    words = list(assemble(io.StringIO(expected_source)).assemble())
                    expected_binary = struct.pack('<'+'I'*len(words), *words)
                    require(original == expected_binary, 'selected public reuse binary')
                    verified.add(binding)
        else:
            loads = before['requested_dma']['load_bytes']
            if signature not in cache:
                expected_source = relocate_assembly(bundle.read(before['source']).decode(),
                                                     before['dram_base'], after['dram_base'])
                require(bundle.read(after['source']) == expected_source.encode()
                        and bundle.read(after['original']) == relocate(bundle.read(before['original']),
                            before['dram_base'], after['dram_base']), 'selected captured baseline program')
        require(after['requested_dma'] == dict(load_bytes=loads, store_bytes=2*m*n), 'selected recipe DMA accounting')
        cache.add(signature)
    bundle._smol_verified_programs = verified


def parameter_geometry(replay):
    result = {'vision.position':([1024,768],'uint16'), 'gelu':([65536],'uint16'),
        'cosine':([8193,32],'float32'), 'sine':([8193,32],'float32'),
        'language.embedding':([48,960],'uint16'), 'timestep':([10],'float32'),
        'time.embedding':([10,720],'float32')}
    for layer in range(12):
        for n in (1,2):
            for field in ('weight','bias'): result[f'vision.{layer}.norm{n}.{field}'] = ([768],'uint16')
    for field in ('weight','bias'): result['vision.post.'+field] = ([768],'uint16')
    for family,width in (('prefix',960),('expert',720)):
        for layer in range(16):
            for field in ('input','post'): result[f'{family}.{layer}.{field}'] = ([width],'uint16')
        result[family+'.final'] = ([width],'uint16')
    for name, shape, dtype in (('images',[2,3,512,512],'float32'),('state',[1,32],'float32'),
        ('noise',[1,50,32],'float32'),('image_masks',[2],'bool'),('language_mask',[48],'bool'),
        ('language_tokens',[48],'int64')): result['observation.'+name] = (shape,dtype)
    for module,row in replay['parameters'].items():
        if 'bias' in row:
            item = row['bias']
            result['bias.'+module] = (item['shape'], {'torch.bfloat16':'uint16','torch.float32':'float32'}[item['dtype']])
    return result


def contract(bundle, assemble=None):
    p = bundle.provenance
    require(p['schema'] == 'atlas.full_bundle.v1' and p['workload'] == 'smol-full', 'workload schema')
    raw = bundle.read(p['contract'])
    require(digest(raw) == p['contract_sha256'], 'contract identity')
    data = json.loads(raw)
    require(data['schema'] == 'atlas.smol_full_contract.v1' and data['cycle_cap'] == CYCLE_CAP
            and type(data['kernel_count']) is int and data['kernel_count'] == bundle.projection_kernel_count == 2715
            and type(data['projection_calls']) is int and data['projection_calls'] == 1421
            and data['kernels'] == p['kernels'], 'complete policy launch inventory')
    profile = execution_profile(data)
    require(execution_profile(p) == profile and bundle.kernel_count == profile.launch_count,
            'contract/provenance execution profile')
    reference = bundle.read(data['source_reference'])
    require(digest(reference) == data['source_reference_sha256'], 'arithmetic source receipt')
    replay = json.loads(reference)
    require(data['captured_kernels'] == 'host/captured-kernels.json', 'captured launch manifest member')
    captured_raw = bundle.read(data['captured_kernels'])
    require(digest(captured_raw) == data['captured_kernels_sha256'], 'captured launch manifest identity')
    captured = json.loads(captured_raw)
    validate_reference(replay, captured)
    manifest = json.loads(bundle.read('kernels.json'))
    require(manifest['schema'] == 'atlas.full_kernel_assets.v1' and manifest['workload'] == 'smol-full'
            and manifest['kernel_count'] == 2715 and manifest['kernels'] == data['kernels']
            and manifest['program_policy'] == data['program_policy']
            and captured['replay_receipt']['sha256'] == digest(reference),
            'source kernel/reference binding')
    require(execution_profile(manifest) == execution_profile(captured) == profile, 'captured execution profile')
    from smol_fusion_assets import profile as fusion_profile
    mode=fusion_profile(data)
    require(all(fusion_profile(item)==mode for item in (p,manifest,captured)),'explicit matching fusion profile')
    if mode!='none':
        from smol_fusion_assets import validate_views
        require(data.get('fusion_symbol')==p.get('fusion_symbol')=='smol_full_fusion_panels'
                and data.get('call_views')==p.get('call_views')==manifest.get('call_views')==captured.get('call_views'),
                'unchanged actual typed-call inventory')
        validate_views(data,bundle.read)
    selected_kernels(bundle, data, captured['kernels'], assemble)
    attention = data.get('attention_kernels', [])
    if attention:
        from smol_attention_assets import event_plan, validate_attention
        require(len(attention) == 200 and data.get('attention_count') == 200
                and data.get('device_event_count') == bundle.kernel_count == 2915
                and data.get('device_events') == bundle.device_events == event_plan()
                and captured.get('attention_kernels') == attention
                and manifest.get('attention_kernels') == attention, 'complete typed attention inventory')
        for item in attention: validate_attention(item, bundle.read, assemble)
    require(data['dram_bytes'] == max(item['total_dram_bytes'] for item in [*data['kernels'], *attention]) == 12706304,
            'maximum actual staging extent')
    refs, expected = data['reference_buffers'], reference_geometry()
    require(len(refs) == len(expected) and {item['name'] for item in refs} == set(expected),
            'all47 output/cache/state references')
    for item in refs:
        shape,dtype = expected[item['name']]
        require(item['shape'] == shape and item['dtype'] == dtype
                and item['bytes'] == int(np.prod(shape))*np.dtype(dtype).itemsize
                and item['actual_symbol'] == 'smol_full_actual_'+item['name']
                and item['expected_symbol'] == 'smol_full_expected_'+item['name'], 'reference ABI geometry')
    expected = parameter_geometry(replay)
    require(set(data['host_parameters']) == set(expected), 'complete learned/input/function parameter inventory')
    for index,(name,item) in enumerate(sorted(data['host_parameters'].items())):
        shape,dtype = expected[name]
        require(item['shape'] == shape and item['dtype'] == dtype
                and item['bytes'] == int(np.prod(shape))*np.dtype(dtype).itemsize
                and item['symbol'] == f'smol_full_param_{index:03d}', 'host parameter geometry '+name)
    for key,value in (('panel_symbol','smol_full_panels'),('call_symbol','smol_full_calls'),
        ('context_symbol','smol_full_raw'),('graph_state_symbol','smol_full_graph_state'),
        ('vision_state_symbol','smol_full_vision_states'),('mailbox_symbol','atlas_workload_mailbox')):
        require(data[key] == value, 'fixed host symbol ABI')
    return data,replay


def readonly_span(image, address, size):
    require(type(address) is int and type(size) is int and size > 0
            and any(s['flags'] & 4 and not s['flags'] & 2 and s['address'] <= address
                    and address+size <= s['address']+s['file_bytes'] for s in image.segments),
            'immutable pointer lacks read-only file-backed memory')


def validate_bundle(bundle, image, *, assemble=None):
    data,replay = contract(bundle, assemble)
    build = bundle.build
    require(build['schema'] == 'atlas.smol_full_host_build.v1' and build['status'] == 'BUILT_AUDITED_ONLY'
            and type(build['actual_returncode']) is int and build['actual_returncode'] == 0
            and build['source_bookends_verified'] is True and build['ELF_executed'] is False
            and build['elf']['sha256'] == digest(image.raw) == bundle.sums['host/host.elf']
            and build['contract_sha256'] == bundle.provenance['contract_sha256']
            and bool(build['source_sha256']), 'build identity and source bookends')
    require(placement(build) == placement(bundle.provenance), 'build/contract placement')
    from smol_fusion_assets import profile as fusion_profile
    mode=fusion_profile(data)
    require(fusion_profile(build)==mode,'explicit build fusion profile')
    if mode!='none':
        commands=[command for command in build['commands'] if '-c' in command]
        require(commands and all('-DSMOL_FUSION_RUNTIME=1' in command for command in commands)
                and any(any(str(argument).endswith('/smol_fusion.c') for argument in command) for command in commands),
                'fusion runtime source and compile definitions')
    audit_image(image, None, data['dram_bytes'], placement(bundle.provenance),fusion=mode!='none')
    return validate_payloads(bundle, image, data, replay, assemble=assemble)


def validate_payloads(bundle, image, data, replay, *, assemble=None):
    """Admit exact descriptors, immutable operands, and writable results at their real addresses."""
    profile = execution_profile(data)
    immutable,actual = [],[]
    for name,size in (('tohost',8),('fromhost',8),('atlas_workload_mailbox',128),('smol_full_raw',184),
        ('smol_full_host',112),('smol_full_graph_state',16),('smol_full_vision_states',32),('smol_full_cycles',profile.cycle_bytes),
        ('smol_full_activation_copy',786432),('smol_full_input_copy',6291456)):
        address = blob(image,name,bytes(size),actual=True)
        require(all(disjoint((address,size), span) for span in immutable), 'control/input storage alias')
        immutable.append((address,size))
    if data.get('fusion_set','none')!='none':
        address=blob(image,'state',bytes(104),actual=True)
        require(all(disjoint((address,104),other) for other in immutable),'fusion registry/control storage alias')
        immutable.append((address,104))
    require(disjoint(immutable[0],immutable[1]), 'distinct HTIF words')
    for name in ('tohost','fromhost'):
        address = image.symbols[name]['address']
        require(address%64 == 0 and image.bytes_at(address,8) == bytes(8), 'loaded writable zero HTIF')
    require(image.symbols['atlas_workload_mailbox']['address']%64 == 0
            and image.bytes_at(image.symbols['atlas_workload_mailbox']['address'],128)
            == struct.pack('<16Q',MAGIC,*([0]*15)),
            'loaded initial mailbox')
    panels_address = image.symbols['smol_full_panels']['address']
    require(image.symbols['smol_full_panels']['size'] == 144*2715, 'panel array ABI')
    call_address = image.symbols['smol_full_calls']['address']
    require(image.symbols['smol_full_calls']['size'] == 80*1421, 'call array ABI')
    readonly_span(image, panels_address, 144*2715)
    readonly_span(image, call_address, 80*1421)
    immutable.extend(((panels_address,144*2715),(call_address,80*1421)))
    for index,item in enumerate(data['kernels']):
        values = struct.unpack('<17Qii',image.bytes_at(panels_address+144*index,144))
        original,weights = bundle.read(item['original']),bundle.read(item['weights'])
        readonly_span(image, values[0], len(original))
        readonly_span(image, values[1], len(weights))
        words = np.frombuffer(original,'<u4')
        positions = np.flatnonzero(words == (((item['output_rescale_exponent']+127)<<23)|0x25f)) if item.get('fusion') else np.flatnonzero((words&0x7fffff) == 0x25f)
        require(positions.size == 1 and int(words[positions[0]]) == ((item['output_rescale_exponent']+127)<<23)|0x25f,
                'actual original rescale instruction')
        m,k,n = item['physical_mkn']
        rows,lk,ln = item['logical_mkn']
        require(values[2:] == (len(original)//4,int(positions[0]),m,k,n,item['logical_row_start'],rows,lk,ln,
            item['call'],item['output_offset'],item['total_dram_bytes'],len(weights),
            item['requested_dma']['load_bytes'],item['requested_dma']['store_bytes'],item['a_scale'],item['b_scale']),
            'embedded full panel descriptor')
        require(image.bytes_at(values[0],len(original)) == original and image.bytes_at(values[1],len(weights)) == weights,
                'embedded actual program and packed weights')
        immutable.extend(((values[0],len(original)),(values[1],len(weights))))
    if data.get('fusion_set','none')!='none':
        from smol_fusion_assets import runtime_fields, read_record
        address=image.symbols['smol_full_fusion_panels']['address']
        require(image.symbols['smol_full_fusion_panels']['size']==32*2715,'fusion descriptor ABI')
        readonly_span(image,address,32*2715);immutable.append((address,32*2715))
        for index,item in enumerate(data['kernels']):
            flags,offset,count,scale=runtime_fields(item)
            values=struct.unpack('<3QIi',image.bytes_at(address+32*index,32))
            require(values[1:]==(offset,count,flags,scale),'embedded typed fusion descriptor')
            if count:
                raw=read_record(item['fusion']['extra'],bundle.read)
                readonly_span(image,values[0],count)
                require(len(raw)==count and image.bytes_at(values[0],count)==raw,'embedded actual fusion extras')
                immutable.append((values[0],count))
                parameter=data['host_parameters']['bias.'+item['module']]
                require(parameter['dtype']=='uint16' and bundle.read(parameter['file'])==read_record(item['fusion']['bias'],bundle.read),
                        'device bias matches actual host graph parameter')
            else:require(values[0]==0,'ordinary/connector fusion has no extra pointer')
    attention_address = 0
    if profile.attention_count:
        attention_address = image.symbols['smol_full_attention_panels']['address']
        require(image.symbols['smol_full_attention_panels']['size'] == 232*200, 'attention descriptor ABI')
        readonly_span(image, attention_address, 232*200)
        immutable.append((attention_address,232*200))
        from smol_attention_assets import validate_attention
        for index,item in enumerate(data.get('attention_kernels', [])):
            c,o = validate_attention(item,bundle.read,assemble)
            row = struct.unpack('<14QI4x12Q3i4x',image.bytes_at(attention_address+232*index,232))
            program,metadata = bundle.read(item['original']),bundle.read(item['metadata'])
            mask = np.load(io.BytesIO(bundle.read(item['logical_mask'])),allow_pickle=False).astype(np.uint8).tobytes()
            if index<24: require(row[2]==0,'vision attention uses a null logical-mask pointer')
            for address,raw in zip(row[:2] if index<24 else row[:3],(program,metadata) if index<24 else (program,metadata,mask)):
                readonly_span(image,address,len(raw))
                require(image.bytes_at(address,len(raw)) == raw,'embedded attention immutable bytes')
                immutable.append((address,len(raw)))
            kind = 0 if index<24 else 1 if index<40 else 2+(index-40)%2
            expected = (len(program)//4,len(metadata),o['descriptors'],o['descriptor_bytes'],o['key_counts'],
                o['key_count_bytes'],o['masks'],o['mask_bytes'],item['before_call'],index,index,kind,
                c.query_rows,c.keys,c.query_heads,c.kv_heads,o['q'],o['k'],o['v'],o['output'],o['output_bytes'],
                item['total_dram_bytes'],item['requested_dma']['load_bytes'],item['requested_dma']['store_bytes'],
                c.q_exponent,c.k_exponent,c.v_exponent)
            require(row[3:] == expected,'embedded attention typed descriptor')
    parameters = data['host_parameters']
    for call in calls():
        indices = [item['index'] for item in data['kernels'] if item['call'] == call['index']]
        row = struct.unpack('<6Q7Ii',image.bytes_at(call_address+80*call['index'],80))
        bias = parameters.get('bias.'+call['module'])
        bias_address = image.symbols[bias['symbol']]['address'] if bias else 0
        require(row == (call['rows'],call['input_width'],call['output_width'],indices[0],len(indices),bias_address,
            call['site'],call['layer'],call['role'],call['image'],call['input_dtype'],call['output_dtype'],
            call['output_dtype'],data['kernels'][indices[0]]['a_scale']), 'embedded actual call order/bias/dtype')
    context = struct.unpack('<15Q24sIIi4s3Q',image.bytes_at(image.symbols['smol_full_raw']['address'],184))
    require(context[:8] == (image.symbols['smol_full_host']['address'],panels_address,call_address,2715,1421,0,0,
                           placement(bundle.provenance)['arena_base'])
            and context[8:15] == (image.symbols['smol_full_activation_copy']['address'],
                image.symbols['smol_full_input_copy']['address'],
                data['dram_bytes'],786432,3145728,100000000,image.symbols['smol_full_cycles']['address'])
            and context[15:20] == (bytes(24),0,0,0,bytes(4))
            and context[20:] == (attention_address,profile.attention_count,0), 'selected staging and raw host context')
    for item in parameters.values():
        raw = bundle.read(item['file'])
        require(len(raw) == item['bytes'] and digest(raw) == item['sha256'] and finite(raw,item['dtype']),
                'finite host parameter bytes')
        immutable.append((blob(image,item['symbol'],raw),len(raw)))
    trajectory_raw = bundle.read('host/reference-trajectory.npz')
    require(digest(trajectory_raw) == replay['trajectory']['sha256'], 'complete arithmetic trajectory identity')
    with zipfile.ZipFile(io.BytesIO(trajectory_raw)) as archive:
        expected_names = {name+'.npy' for name in reference_geometry()}
        members = archive.infolist()
        require(len(members) == len(expected_names) and {member.filename for member in members} == expected_names
                and all(0 < member.file_size < 16*1024*1024 for member in members)
                and sum(member.file_size for member in members) < 32*1024*1024, 'bounded complete trajectory archive')
    with np.load(io.BytesIO(trajectory_raw),allow_pickle=False) as archive:
        require(set(archive.files) == set(reference_geometry()), 'complete reference archive')
        for item in data['reference_buffers']:
            raw = bundle.read(item['file'])
            require(len(raw) == item['bytes'] and digest(raw) == item['sha256'] and finite(raw,item['dtype'])
                    and raw == archive[item['name']].tobytes(), 'exact complete trajectory reference')
            expected = blob(image,item['expected_symbol'],raw)
            address = blob(image,item['actual_symbol'],raw,actual=True)
            require(all(disjoint((address,len(raw)),other) for other in actual),
                    'distinct actual output/cache/state buffers')
            actual.append((address,len(raw)))
            immutable.append((expected,len(raw)))
    require(all(disjoint(a,b) for a in actual for b in immutable), 'actual/immutable or control alias')
    return data


def verify_completion(remote, image, device, bundle, out):
    data,_ = contract(bundle)
    profile = execution_profile(data)
    count = profile.launch_count
    require(len(device.completed) == count, 'all fresh model starts')
    for index, (summary, item) in enumerate(zip(device.completed, [bundle.kernel_paths(i) for i in range(count)])):
        verify_kernel_receipt(summary, dict(item, program=bundle.read(item['original']),
            expected_sha256=digest(bundle.read(item['expected']))), index, CYCLE_CAP)
    require(remote.reg(32) in stop_sites(image), 'normal startup WFI completion')
    def actual(name,size):
        raw = remote.read(image.symbols[name]['address'],size)
        require(len(raw) == size, 'actual read extent '+name)
        return raw
    mailbox = actual('atlas_workload_mailbox',128)
    words = struct.unpack('<16Q',mailbox)
    require(words == (MAGIC,2,count,1421,1,10,1,0,0,2715,1421,count,0,0,0,0), 'full policy self-check completion')
    require(actual('tohost',8) == struct.pack('<Q',1) and actual('fromhost',8) == bytes(8), 'HTIF completion')
    require(struct.unpack('<4I',actual('smol_full_graph_state',16)) == (1,10,1273,0), 'actual policy graph state')
    require(struct.unpack('<8I',actual('smol_full_vision_states',32)) == (1,74,1,0)*2, 'both actual vision graph states')
    raw_context = actual('smol_full_raw',184)
    current = struct.unpack('<15Q24sIIi4s3Q',raw_context)
    initial = struct.unpack('<15Q24sIIi4s3Q',image.bytes_at(image.symbols['smol_full_raw']['address'],184))
    require(current[5:7] == (2715,1421) and all(current[i] == initial[i] for i in tuple(range(5))+tuple(range(7,15)))
            and current[15] == struct.pack('<QQI4s',placement(bundle.provenance)['arena_base'],
                                           data['kernels'][-1]['total_dram_bytes'],2,bytes(4))
            and current[16:20] == (1,0,0,bytes(4)) and current[20:22] == initial[20:22] and current[22] == profile.attention_count, 'actual raw traversal/context ownership')
    host = actual('smol_full_host',112)
    require(struct.unpack_from('<QQ',host,72) == (0,0) and struct.unpack_from('<6I',host,88)[1:3] == (0,0)
            and struct.unpack_from('<I',host,108)[0] == count, 'drained final host ownership')
    cycles_raw = actual('smol_full_cycles',profile.cycle_bytes)
    require(all(struct.unpack('<'+str(count)+'I',cycles_raw)), 'all actual guest cycle observations nonzero')
    results = []
    for item in data['reference_buffers']:
        raw = actual(item['actual_symbol'],item['bytes'])
        require(raw == bundle.read(item['file']), 'actual full policy output/cache/state '+item['name'])
        (out/(item['name']+'.bin')).write_bytes(raw)
        results.append(dict(name=item['name'],bytes=len(raw),sha256=digest(raw),exact=True))
    for name,raw in (('mailbox.bin',mailbox),('raw-context.bin',raw_context),('cycles.bin',cycles_raw)):
        (out/name).write_bytes(raw)
    return dict(tohost=1,fromhost=0,mailbox_words=words,expected_starts=count,actual_projection_calls=1421,
        fresh_vision_images=2,all16_prefix_KV_exact=True,denoising_steps=10,
        all_state_velocity_actions_exact=True,all_final_guest_buffers_exact=True,final_guest_buffers=results)
