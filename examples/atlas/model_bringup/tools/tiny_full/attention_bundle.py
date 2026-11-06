"""Admission for full TinyLlama attention assets and original RTL receipts.

This deliberately does not accept or modify the projection-only bundle schema.
The receipt is supplied with an external checksum pin and binds every staged
program, input, output, whole final arena, and requested DMA count.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import subprocess

import numpy as np

import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked
from bundle import ChecksumBundle

MODEL_PIN = "47b8599d8ffbe5ddd96059ffe2b5be2427a410f5"
MODEL_SOURCE_MEMBERS = frozenset(
    'npu_model/' + (directory + '/' if directory else '') + module + '.py'
    for directory, modules in {
        '': ('__init__', 'isa', 'isa_patterns', 'isa_types'),
        'configs': ('__init__', 'isa_definition', 'numerics', 'rtl_math'),
        'configs/hardware': ('__init__', 'default'),
        'hardware': ('__init__', 'arch_state', 'bank_conflict', 'config', 'core', 'dma', 'exu', 'hardware',
                     'ifu', 'lsu', 'memory_backend', 'mxu', 'sparse_memory', 'stage_data', 'vpu', 'xlu'),
        'logging': ('__init__', 'logger'),
        'software': ('__init__', 'instruction', 'program'),
        'util': ('converter', 'importer'),
    }.items() for module in modules
) | frozenset('npu_model/configs/data/' + name + '.bin.gz'
              for name in ('cos', 'cube', 'exp', 'exp2', 'log', 'rcp', 'relu', 'sin', 'sqrt', 'square', 'tanh')) \
  | {'npu_model/configs/memory_curves/ee290sim_vcs_probe.json'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def model_attention_geometry(step):
    """Logical configuration admitted for each full-model attention step."""
    return dict(query_rows=128 if step == 0 else 1, keys=128 + step,
                query_heads=32, kv_heads=4, head_dim=64, mask='causal',
                query_start=0 if step == 0 else 127 + step, scale=0.125)


def attention_before_projection(site_index):
    """The authored layer schedule inserts attention before its output projection."""
    return site_index % 7 == 3


def validate_scope(data, replay):
    require(data.get('schema') == 'atlas.tiny_attention_contract.v1'
            and data.get('workload') == 'tiny-attention', 'unsupported attention bundle')
    require(all(type(data.get(key)) is int for key in ('layer_count', 'decode_count', 'kernel_count',
            'projection_kernel_count', 'attention_kernel_count', 'projection_calls')), 'invalid attention scope counts')
    full = (data.get('layer_count') == 22 and data.get('decode_count') == 2
            and data.get('include_lm_head') is True and data.get('block_only') is False)
    block = (data.get('layer_count') == 1 and data.get('decode_count') == 0
             and data.get('include_lm_head') is False and data.get('block_only') is True)
    require(full or block, 'complete full model or complete layer-0 prefill block required')
    norm = data.get('post_o_norm', False)
    require(type(norm) is bool and (not norm or data.get('fusions') is True and data.get('attention_version') == 2),
            'post-o norm requires explicit fused attention-v2 mode')
    pre = data.get('pre_attention_norm', False)
    require(type(pre) is bool and (not pre or norm and data.get('pre_attention_norm_headroom') in (0, 1)),
            'pre-attention norm requires post-o norm and a headroom of zero or one exponent step')
    counts = dict(kernel_count=906, projection_kernel_count=840, attention_kernel_count=66,
                  projection_calls=465) if full else dict(kernel_count=13, projection_kernel_count=12,
                                                        attention_kernel_count=1, projection_calls=7)
    if norm:
        require(type(data.get('norm_kernel_count')) is int, 'invalid norm scope count type')
        counts.update(kernel_count=(1038 if full else 15) if pre else (972 if full else 14),
                      norm_kernel_count=(132 if full else 2) if pre else (66 if full else 1))
    else:
        require(not data.get('norm_kernels') and not data.get('norm_kernel_count', 0), 'untyped norm table')
    for key, count in counts.items():
        require(data.get(key) == count, 'wrong attention bundle ' + key)
    count, layers, steps = counts['kernel_count'], data['layer_count'], data['decode_count'] + 1
    launches = data['launches']
    require(replay.get('schema') == 'atlas.v2.captured_reply.v1'
            and replay.get('launches') == launches and len(launches) == count,
            'attention replay launch identity differs')
    require([row['index'] for row in launches] == list(range(count)), 'attention launch order differs')
    require(len(data['launch_plan']) == count
            and len(data['projection_kernels']) == counts['projection_kernel_count']
            and len(data['attention_kernels']) == counts['attention_kernel_count'], 'attention table extent differs')
    cursors = {'projection': 0, 'attention': 0}
    if norm:
        require(len(data['norm_kernels']) == counts['norm_kernel_count'], 'norm table extent differs')
        cursors['norm'] = 0
    for index, (item, entry) in enumerate(zip(launches, data['launch_plan'])):
        kind = item['kind']
        require(kind in cursors and entry == dict(launch_index=index, kind=kind, index=cursors[kind]),
                'interleaved attention launch plan differs')
        table = data[kind + '_kernels'][cursors[kind]]
        require(dict(table, index=index, table_index=cursors[kind]) == item,
                'attention launch and panel table differ')
        cursors[kind] += 1
    expected_cursors = dict(projection=counts['projection_kernel_count'], attention=counts['attention_kernel_count'])
    if norm:
        expected_cursors['norm'] = counts['norm_kernel_count']
    require(cursors == expected_cursors,
            'attention panel coverage differs')
    for step in range(steps):
        rows = [row for row in launches if row['step'] == step]
        attention = [row for row in rows if row['kind'] == 'attention']
        require(len(rows) == (((346 if full else 15) if pre else (324 if full else 14)) if norm else (302 if full else 13))
                and len(attention) == layers
                and [row['layer'] for row in attention] == list(range(layers)), 'attention step coverage differs')
        for row in attention:
            config = row['config']
            require(all(config[key] == value for key, value in model_attention_geometry(step).items())
                    and row['numerics'] == 'rtl' and row['model_commit'] == MODEL_PIN,
                    'attention model geometry differs')
    if block:
        expected = [('norm', None)] * pre + [('projection', role) for role in (0, 1, 2)] + [('attention', None)]
        expected += [('projection', 3)] + ([('norm', None)] if norm else [])
        expected += [('projection', role) for role in (4, 4, 4, 5, 5, 5, 6, 6)]
        require([(row['kind'], row.get('logical')) for row in launches] == expected
                and all(row['layer'] == 0 for row in launches), 'complete block projection order differs')
    required = {f'tiny_full_cache_{step:02d}_{layer:02d}_{field}'
                for step in range(steps) for layer in range(layers) for field in ('keys', 'values')}
    required |= ({f'tiny_full_logits_{step:02d}' for step in range(steps)}
                 | {'tiny_full_greedy_tokens', 'tiny_full_decode_inputs'}) if full else {'tiny_full_hidden_00'}
    references = {row['actual_symbol']: row for row in data['reference_buffers']}
    require(set(references) == required and len(data['reference_buffers']) == len(required),
            'attention final reference coverage differs')
    finals = {row['symbol']: row for row in replay['final_buffers']}
    require(set(finals) == required and len(replay['final_buffers']) == len(required),
            'attention final replay buffer coverage differs')
    tokens = data.get('greedy_tokens', [])
    require((full and len(tokens) == 3 and all(type(token) is int and 0 <= token < 32000 for token in tokens))
            or (block and tokens == []), 'invalid attention greedy tokens or block token claim')
    for name, row in references.items():
        require(finals[name] == dict(symbol=name, expected=row['file'], expected_sha256=row['sha256']),
                'attention final buffer identity differs')
        size, dtype = ((4 * 160 * 64 * 2, 'uint16') if '_cache_' in name else
                       (128 * 2048 * 2, 'uint16') if '_hidden_' in name else
                       (32000 * 4, 'float32') if '_logits_' in name else
                       (12 if name == 'tiny_full_greedy_tokens' else 8, 'uint32'))
        require(row['bytes'] == size and row['dtype'] == dtype, 'attention final buffer extent differs')
    return 'full-model' if full else 'layer0-block'


def validate_model_sources(root, sources):
    require(isinstance(root, (str, Path)) and Path(root).is_absolute(), 'original model source root missing')
    root = checked(root)
    require(root.name == 'npu-model' and isinstance(sources, dict) and len(sources) >= 45,
            'original model source root or closure differs')
    paths = [checked(name) for name in sorted(sources)]
    require(all(path.is_relative_to(root) for path in paths), 'qualified model source escaped its root')
    require(MODEL_SOURCE_MEMBERS <= {str(path.relative_to(root)) for path in paths},
            'qualified execution source or unary table closure missing')
    require(Path(subprocess.check_output(['git', '-C', str(root), 'rev-parse', '--show-toplevel'],
        text=True).strip()) == root, 'original model Git root differs')
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    require(head == MODEL_PIN, 'qualified model is not original HEAD47')
    objects = ['HEAD:' + str(path.relative_to(root)) for path in paths]
    raw = subprocess.run(['git', '-C', str(root), 'cat-file', '--batch'],
                         input=('\n'.join(objects) + '\n').encode(), stdout=subprocess.PIPE, check=True).stdout
    cursor = 0
    for path in paths:
        end = raw.index(b'\n', cursor)
        fields = raw[cursor:end].split()
        require(len(fields) == 3 and fields[1] == b'blob', 'qualified model source absent from HEAD47')
        size = int(fields[2])
        actual = path.read_bytes()
        require(actual == raw[end + 1:end + 1 + size] and digest(actual) == sources[str(path)],
                'qualified model source changed: ' + str(path))
        cursor = end + size + 2


def validate_source_pins(pins, description):
    require(isinstance(pins, dict) and pins, description + ' source bindings missing')
    for path, expected in pins.items():
        require(digest(checked(path).read_bytes()) == expected, description + ' source changed: ' + path)


def validate_gate(gate, data, replay_sha256, sums):
    count = data['kernel_count']
    block = data['block_only']
    require(count == (((15 if block else 1038) if data.get('pre_attention_norm') else (14 if block else 972))
                      if data.get('post_o_norm') else (13 if block else 906)), 'complete qualification scope required')
    d1 = gate.get('schema') == 'atlas.d1.all_launch_original.v1'
    require(gate.get('status') == 'PASS' and gate.get('model_pin') == MODEL_PIN
            and gate.get('numerics') == 'rtl' and gate.get('native_not_used') is True
            and gate.get('all_launches_original_npu_model_rtl_exact') is True
            and gate.get('launch_count') == count
            and gate.get('verified_indices') == list(range(count)) and len(gate.get('launches', [])) == count
            and gate.get('assets_replay_sha256') == replay_sha256,
            'complete checksum-bound original RTL qualification required')
    require((d1 and gate.get('accelerated') is False and gate.get('smoke_scope') is block)
            or (not d1 and not block and gate.get('attention_native_allowed') is False),
            'qualification format, native policy, or complete-block scope differs')
    model_sources, model_root, harness_sources = None, None, None
    for index, (row, item) in enumerate(zip(gate['launches'], data['launches'])):
        require(row.get('index') == index and row.get('kind') == item['kind'] and row.get('status') == 'PASS'
                and row.get('exact') is True and row.get('full_arena_exact') is True
                and row.get('accelerated') is False and row.get('native_dispatch_enabled') is False,
                'original RTL launch qualification differs')
        if d1:
            require(row.get('model_pin') == MODEL_PIN and row.get('numerics') == 'rtl'
                    and row.get('model_sources_before') == row.get('model_sources_after')
                    and row.get('harness_sources_before') == row.get('harness_sources_after'),
                    'original model or harness source bookends differ')
            sources, root = row.get('model_sources_before'), row.get('model_root')
            harness = row.get('harness_sources_before')
            require(row.get('program_sha256') == sums[item['original']]
                    and row.get('requested_dma') == item['requested_dma']
                    and row.get('output_offset') == item['output_offset']
                    and ('output_extent_bytes' not in item or row.get('output_bytes') == item['output_extent_bytes']),
                    'D1 qualified program, DMA, or extent differs')
        else:
            require(row.get('model', {}).get('commit') == MODEL_PIN and row['model'].get('numerics') == 'rtl',
                    'original model identity differs')
            sources, root = row.get('source_sha256'), row['model'].get('root')
            harness = row.get('harness_source_sha256')
        for key, member in (('initial_sha256', 'initial'), ('output_sha256', 'expected'),
                            ('final_arena_sha256', 'expected_arena'), ('program_source_sha256', 'source')):
            require(row.get(key) == sums.get(item[member]), 'qualified launch bytes differ: ' + key)
        require(row['execution']['program_sha256'] == sums[item['original']]
                and all(row['execution']['observed'].get(key) == value for key, value in item['requested_dma'].items())
                and row['execution']['observed'].get('native_mxu0_calls', 0) == 0,
                'qualified program or DMA counts differ')
        require(isinstance(sources, dict) and len(sources) >= 45 and isinstance(harness, dict) and harness,
                'qualified execution source bindings missing')
        if model_sources is None:
            model_sources, model_root, harness_sources = sources, root, harness
            validate_model_sources(root, sources)
        require(sources == model_sources and root == model_root, 'qualified model sources changed across launches')
        validate_source_pins(harness, 'qualified harness')
        if d1:
            require(harness == harness_sources and row.get('method') in
                    ('executed_original_model', 'reused_original_capture_execution', 'reused_D1_exact_program_and_arena',
                     'reused_D2_exact_program_and_arena', 'reused_F1_exact_program_and_arena',
                     *(['reused_F2_exact_program_and_arena'] if data.get('decode_device_rope') else []),
                     *(['reused_F3h0_exact_program_and_arena'] if data.get('decode_device_rope')
                       and data.get('pre_attention_norm') is True and data.get('pre_attention_norm_headroom') == 0 else [])),
                    'D1 execution method or harness differs')
            if row.get('method') == 'reused_F2_exact_program_and_arena':
                require(row.get('five_payloads_byte_identical') is True, 'F2 reuse lacks complete byte joins')
                validate_source_pins(row.get('reused_F2_harness_sources'), 'reused F2 harness')
            if row.get('method') == 'reused_F3h0_exact_program_and_arena':
                require(row.get('five_payloads_byte_identical') is True, 'F3h0 reuse lacks complete byte joins')
                validate_source_pins(row.get('reused_F3h0_harness_sources'), 'reused F3h0 harness')
            for field in ('reused_capture_harness_sources', 'reused_capture_payload_sha256'):
                if field in row:
                    validate_source_pins(row[field], 'reused original capture input')
            if row.get('reused_capture_record'):
                require(digest(checked(row['reused_capture_record']).read_bytes()) == row['reused_capture_record_sha256'],
                        'reused original capture record changed')
    if 'output_coverage' in gate:
        coverage = gate['output_coverage']
        programs = {row['execution']['program_sha256'] for row in gate['launches']}
        require(gate.get('all_program_output_spans_written') is True
                and isinstance(coverage, list) and len(coverage) == len(programs)
                and {row.get('program_sha256') for row in coverage} == programs,
                'complete unique-program output coverage required')
        for row in coverage:
            require(row.get('status') == 'PASS' and row.get('output_coverage') is True
                    and row.get('exact') is True and row.get('full_arena_exact') is True
                    and row.get('model_sources_before') == row.get('model_sources_after') == model_sources
                    and row.get('harness_sources_before') == row.get('harness_sources_after') == harness_sources,
                    'output coverage exactness or source bindings differ')
    if d1:
        require(gate.get('model_sources_before') == model_sources == gate.get('model_sources_after'),
                'aggregate original model source bookends differ')
        require(gate.get('harness_sources_before') == harness_sources == gate.get('harness_sources_after'),
                'aggregate qualification harness source bookends differ')


def validate_fused_output(item, initial, output, final, read):
    m, k, n = item['physical_mkn']
    role, column, width = item['logical'], item['columns'][0], item['width']
    require(all(type(value) is int and value > 0 and value % 32 == 0 for value in (m, k, n))
            and type(role) is int and 0 <= role <= 7
            and type(column) is int and column % 32 == 0 and type(width) is int and width > 0
            and item['columns'] == [column, column + n] and 0 <= column < width and column + n <= width,
            'fused panel geometry differs')
    extent = (m // 32 - 1) * 32 * width + (column + n) * 32 if role == 5 else 2 * m * n
    if item.get('residual_add') is False:
        require(role in (3, 6) and item.get('epilogue') is None
                and item.get('program_policy') == 'traffic-reuse-D2-host-residual-v1',
                'D2 plain producer policy differs')
    rope = 'rope_rounding' in item
    operand_bytes = (2 * m * n if role == 5 or role in (3, 6) and item.get("residual_add", True)
                     else 192 * m if rope else 0)
    offset = m * k + k * n + 256
    operand_offset = offset + extent + 256 if operand_bytes else 0
    epilogue = item.get('epilogue') or {}
    if rope or role in (3, 6) and item.get('residual_add') is True:
        phase = (item['step'] == 0 and m == 128 and item['logical_rows'] == 128
                 or rope and item['step'] in (1, 2) and m == 32 and item['logical_rows'] == 1)
        require(phase
                and epilogue.get('output_offset') == offset and epilogue.get('m') == m and epilogue.get('n') == n
                and epilogue.get('output_dtype') == 'bf16' and epilogue.get('output_layout') == 'tiles',
                'device residual and RoPE producer phase or BF16 epilogue differs')
    if rope:
        require(role in (0, 1) and item['rope_rounding'] == 'rne' and epilogue.get('steps') == []
                and item.get('program_policy') == ('traffic-reuse-F2-prefill-rope-rne-v1' if item['step'] == 0
                                                  else 'traffic-reuse-T4-decode-rope-rne-v1')
                and [epilogue.get(key) for key in ('cos_offset', 'sin_offset', 'neg_sin_offset')]
                    == [operand_offset, operand_offset + 64 * m, operand_offset + 128 * m],
                'F2 RoPE post-pass policy or table offsets differ')
    elif role in (3, 6) and item.get('residual_add') is True:
        operand = dict(offset=operand_offset, layout='tiles', dtype='bf16', scale_exponent=0)
        require(item.get('residual_rounding') == 'rne' and epilogue.get('steps') == [dict(op='add_rne', operand=operand)]
                and item.get('program_policy') == 'traffic-reuse-F2-prefill-residual-rne-v1',
                'F2 exact residual epilogue policy or operand differs')
    spans = ([dict(offset=tr * 32 * width + column * 32, bytes=n * 32) for tr in range(m // 32)]
             if role == 5 else [dict(offset=0, bytes=extent)])
    require(item['fusion_role'] == (role if role in (3, 4, 5, 6) else 0)
            and item['output_offset'] == offset and item['output_extent_bytes'] == len(output) == extent
            and item['operand_offset'] == operand_offset
            and item['total_dram_bytes'] == offset + extent + 256 + operand_bytes
            and item['written_spans'] == spans and sum(span['bytes'] for span in spans) == item['requested_dma']['store_bytes'],
            'fused output extent, operand, spans, or DMA differs')
    require(item['output_dtype'] == ('fp8' if role == 5 else 'bf16')
            and item['output_row_stride'] == (32 * 5632 if role == 5 else None)
            and item['output_column_tile'] == (column // 32 if role == 5 else 0)
            and (role != 5 or width == 5632), 'fused output layout differs')
    require(type(item['output_scale']) is int and -126 <= item['output_scale'] <= 127
            and (role == 5 or item['output_scale'] == 0), 'fused output scale differs')
    require(initial[offset - 256:offset] == b'\xa5' * 256
            and initial[offset + extent:offset + extent + 256] == b'\x5a' * 256, 'fused guards differ')
    key = 'rope_tables' if rope else 'operand'
    require(bool(item.get(key)) == bool(operand_bytes) and not (rope and item.get('operand')), 'fused operand metadata differs')
    if operand_bytes:
        operand = read(item[key])
        require(len(operand) == operand_bytes and initial[operand_offset:] == operand, 'fused staged operand differs')
    wanted = bytearray(initial)
    for span in spans:
        start, size = offset + span['offset'], span['bytes']
        wanted[start:start + size] = output[span['offset']:span['offset'] + size]
    require(bytes(wanted) == final and output == final[offset:offset + extent],
            'fused output changed an input, guard, or strided hole')


def validate_attention_v2(item, initial, output, final):
    """Independently reconstruct the static v2 tail without importing Torch."""
    c = item['config']
    keys = {'query_rows', 'keys', 'query_heads', 'kv_heads', 'head_dim', 'mask', 'query_start', 'scale',
            'q_exponent', 'k_exponent', 'v_exponent', 'p_exponent', 'output_fp8', 'o_exponent', 'packed_decode'}
    integers = keys - {'mask', 'query_start', 'scale', 'output_fp8', 'packed_decode'}
    require(set(c) == keys and all(type(c[key]) is int for key in integers)
            and type(c['output_fp8']) is bool and c['output_fp8']
            and type(c['packed_decode']) is bool and c['packed_decode']
            and type(c['scale']) in (int, float) and c['scale'] == 0.125
            and c['query_heads'] == 32 and c['kv_heads'] == 4 and c['head_dim'] == 64
            and (c['query_rows'], c['keys']) in ((128, 128), (1, 129), (1, 130))
            and c['mask'] == 'causal' and c['query_start'] in (None, c['keys']-c['query_rows'])
            and (c['query_start'] is None or type(c['query_start']) is int)
            and c['p_exponent'] == -8
            and all(-60 <= c[key] <= 60 for key in ('q_exponent', 'k_exponent', 'v_exponent', 'o_exponent')),
            'attention-v2 configuration or Tiny geometry differs')
    decode = c['query_rows'] == 1
    qt, kt = (c['query_rows']+31)//32, (c['keys']+31)//32
    qbytes = (4 if decode else qt*32)*4096
    kvbytes = kt*4*4096
    descriptors, counts, masks, unique = [], [], bytearray(), {}
    for qi in range(qt):
        visible = 0
        for ki in range(kt):
            allowed = [[(row < (8 if decode else 32) and ki*32+column < c['keys']
                         and ki*32+column <= c['keys']-c['query_rows']+qi*32+(0 if decode else row))
                        for column in range(32)] for row in range(32)]
            if not any(any(row) for row in allowed):
                descriptors.append((0, 1))
            else:
                visible = ki+1
                if all(all(row) for row in allowed):
                    descriptors.append((0, 0))
                else:
                    blob = b''.join(struct.pack('<H', 0 if allowed[row][column] else 0x7f80)
                                    for half in (0, 16) for row in range(32) for column in range(half, half+16))
                    if blob not in unique:
                        unique[blob] = 0x4000+len(masks)//4
                        masks.extend(blob)
                    descriptors.append((unique[blob], 0))
        counts.append(visible)
    table = b''.join(struct.pack('<II', *entry) for entry in descriptors)
    descriptor_bytes = (len(table)+31)//32*32
    count_blob = struct.pack('<'+'I'*len(counts), *counts)
    count_bytes = (len(count_blob)+31)//32*32
    tail = table+bytes(descriptor_bytes-len(table))+count_blob+bytes(count_bytes-len(count_blob))+masks
    offsets = dict(q=0, k=qbytes, v=qbytes+kvbytes, descriptors=qbytes+2*kvbytes,
                   descriptor_bytes=descriptor_bytes, key_counts=qbytes+2*kvbytes+descriptor_bytes,
                   key_count_bytes=count_bytes, masks=qbytes+2*kvbytes+descriptor_bytes+count_bytes,
                   mask_bytes=len(masks), unique_masks=len(unique), output=qbytes+2*kvbytes+len(tail)+256,
                   output_bytes=qt*32*2048)
    offset, extent = offsets['output'], offsets['output_bytes']
    spans = ([dict(offset=head*2048+half*1024, bytes=32) for head in range(32) for half in range(2)]
             if decode else [dict(offset=0, bytes=extent)])
    require(item.get('attention_version') == 2 and item.get('kernel_schema') == 'atlas.tiny.attention_launch.v2'
            and item['offsets'] == offsets and item['output_offset'] == offset
            and item['output_extent_bytes'] == extent == len(output)
            and item['total_dram_bytes'] == offset+extent+256
            and item['output_dtype'] == 'fp8' and item['output_layout'] == 'fp8_tiled'
            and item['output_scale'] == c['o_exponent'] and type(item['packed_decode_effective']) is bool
            and item['packed_decode_effective'] == decode
            and item['input_layout'] == ('bf16_tiled_packed_decode' if decode else 'bf16_tiled')
            and item['written_spans'] == spans
            and item['requested_dma'] == dict(load_bytes=offset-256, store_bytes=sum(span['bytes'] for span in spans)),
            'attention-v2 staging metadata, spans, or DMA differs')
    require(initial[offsets['descriptors']:offset-256] == tail
            and initial[offset-256:offset] == b'\xa5'*256
            and initial[offset:offset+extent] == bytes(extent) and initial[offset+extent:] == b'\x5a'*256,
            'attention-v2 static descriptors, masks, counts, guards, or zero output differs')
    wanted = bytearray(initial)
    for span in spans:
        begin, size = span['offset'], span['bytes']
        wanted[offset+begin:offset+begin+size] = output[begin:begin+size]
    require(bytes(wanted) == final and final[offset:offset+extent] == output,
            'attention-v2 changed an input, guard, or zero hole')


def validate_norm(item, initial, output, final, read):
    """Reconstruct canonical norm staging independently of its program generator."""
    step, layer = item['step'], item['layer']
    rows, m = (128, 128) if step == 0 else (1, 32)
    offset, size = 2*m*2048, m*2048
    mask_offset, weight_offset = 3*m*2048+256, 3*m*2048+256+65536
    require(type(step) is int and 0 <= step <= 2 and type(layer) is int and 0 <= layer < 22
            and item.get('kernel_schema') == 'atlas.tiny.norm_launch.v1' and item.get('norm_version') == 1
            and item.get('site') in (f'model.layers.{layer}.input_layernorm', f'model.layers.{layer}.post_attention_layernorm')
            and item.get('flags') == int(item['site'].endswith('input_layernorm')) and item.get('logical_rows') == rows and item.get('physical_rows') == m
            and item.get('width') == 2048 and item.get('input_layout') == 'bf16_tiled'
            and item.get('output_dtype') == 'fp8' and item.get('output_layout') == 'fp8_tiled'
            and type(item.get('output_scale')) is int and -126 <= item['output_scale'] <= 119
            and item.get('norm_scale') == item['output_scale']
            and item['output_offset'] == offset and item.get('output_extent_bytes') == size == len(output)
            and item['mask_offset'] == mask_offset and item['weight_offset'] == weight_offset
            and item['total_dram_bytes'] == weight_offset+131072+256
            and item['written_spans'] == [dict(offset=0, bytes=size)]
            and item['requested_dma'] == dict(load_bytes=2*m*2048+196608, store_bytes=size)
            and item.get('model_commit') == MODEL_PIN and item.get('numerics') == 'rtl'
            and item.get('accelerated') is False and item.get('native_dispatch_enabled') is False,
            'norm geometry, original model policy, extent, or DMA differs')
    h, gamma = read(item['h_bf16']), read(item['norm_weight'])
    require(len(h) == 2*rows*2048 and len(gamma) == 4096 and digest(h) == item['norm_h_sha256'],
            'norm logical H or gamma extent/hash differs')
    physical = h + bytes(2*(m-rows)*2048)
    h_tiles = b''.join(physical[2*((tr*32+r)*2048+tc*32+half):2*((tr*32+r)*2048+tc*32+half+16)]
        for tr in range(m//32) for tc in range(64) for half in (0,16) for r in range(32))
    masks = b''.join(struct.pack('<H', 0x3f80 if col == one else 0)
        for one in range(32) for half in (0,16) for row in range(32) for col in range(half,half+16))
    weights = b''.join(gamma[2*(tc*32+half):2*(tc*32+half+16)]*32 for tc in range(64) for half in (0,16))
    require(read(item['h_bf16_tiles']) == h_tiles and read(item['mask']) == masks
            and read(item['weight']) == weights
            and initial == h_tiles+b'\xcd'*size+b'\x5a'*256+masks+weights+b'\xa5'*256,
            'norm H packing, canonical constants, output initialization, or guards differ')
    require(final == initial[:offset]+output+initial[offset+size:], 'norm changed input, constants, or guards')
    require(all((value & 127) != 127 for value in output), 'norm output has FP8 NaN encodings')
    if rows != m:
        require(all((output[tile*1024+row*32+col] & 127) == 0
                    for tile in range(64) for row in range(1,32) for col in range(32)),
                'norm decode output padding is not signed zero')


def validate_norm_joins(data, read):
    """The post-o norm launches after o, before gate/up; the pre-attention norm opens its layer, before q/k/v."""
    layers, sites, launches = data['layer_count'], 2 if data.get('pre_attention_norm') else 1, data['launches']
    for index, norm in enumerate(data['norm_kernels']):
        step, layer = divmod(index // sites, layers)
        pre = sites == 2 and index % 2 == 0
        site = f"model.layers.{layer}.{'input_layernorm' if pre else 'post_attention_layernorm'}"
        require((norm['index'], norm['step'], norm['layer'], norm['site']) == (index, step, layer, site)
                and read(norm['norm_weight']) == read(data['host_parameters'][site + '.weight']['file']),
                'norm table order, site, or gamma differs')
        launch_index = next(i for i, row in enumerate(launches) if row['kind'] == 'norm' and row['table_index'] == index)
        previous = launches[launch_index-1] if launch_index else {}
        roles, widths = ((0, 1, 2), (2048, 256, 256)) if pre else ((4, 5), (5632, 5632))
        consumers = [row for row in launches if row['kind'] == 'projection'
            and (row['step'], row['layer']) == (step, layer) and row['logical'] in roles]
        first = min(i for i, row in enumerate(launches) if (row['step'], row['layer']) == (step, layer))
        require((launch_index == first if pre else previous.get('kind') == 'projection' and previous.get('logical') == 3
                 and (previous['step'], previous['layer']) == (step, layer))
                and [row['logical'] for row in consumers] == ([0, 1, 2] if pre else [4, 4, 4, 5, 5, 5])
                and [row['index'] for row in consumers] == list(range(launch_index+1, launch_index+1+len(consumers))),
                'norm must open its layer before q/k/v, or launch after o and directly before all gate/up shards')
        for role, width in zip(roles, widths):
            column = 0
            for row in [value for value in consumers if value['logical'] == role]:
                require(row.get('input_layout') == 'fp8_tiled' and row.get('input_norm') == index
                        and row.get('norm_scale') == row['A_scale'] == norm['output_scale']
                        and row.get('norm_h_sha256') == norm['norm_h_sha256']
                        and row['physical_mkn'][1] == 2048 and row['columns'][0] == column
                        and row['width'] == width and read(row['activation']) == read(norm['expected']),
                        'q/k/v or gate/up lost its exact FP8 norm producer, H handle, or exponent')
                column = row['columns'][1]
            require(column == width, 'norm consumer shard coverage differs')


def tiles(raw, m, n):
    return np.frombuffer(raw, dtype='<u2').reshape(m // 32, n // 32, 2, 32, 16).transpose(0, 3, 1, 2, 4).reshape(m, n)


def validate_device_joins(data, read):
    """Bind prefill device residual operands and RoPE tables to host parameters and their producers."""
    layers, launches = data['layer_count'], data['launches']
    host = {name: read(row['file']) for name, row in data['host_parameters'].items()
            if name in ('model.embed_tokens.weight', 'prefill_tokens', 'rotary_cosine', 'rotary_sine')}
    projections = {}
    for row in launches:
        if row['kind'] == 'projection' and row['step'] == 0:
            projections.setdefault((row['layer'], row['logical']), []).append(row)
    if data.get('prefill_device_rope'):
        cosine, sine = (np.frombuffer(host[name], dtype='<u2').reshape(160, 64) for name in ('rotary_cosine', 'rotary_sine'))
        require(np.array_equal(cosine[:, :32], cosine[:, 32:]) and np.array_equal(sine[:, :32], sine[:, 32:]),
                'rotate-half tables require equal halves')
        pack = lambda value: np.ascontiguousarray(value[:128, :32]).reshape(4, 32, 2, 16).transpose(0, 2, 1, 3).tobytes()
        expected = pack(cosine) + pack(sine) + pack(sine ^ 0x8000)
        for index, attention in enumerate(data['attention_kernels'][:layers]):
            arena = read(attention['initial'])
            for role, start, size in ((0, attention['offsets']['q'], 524288), (1, attention['offsets']['k'], 65536)):
                (row,) = projections[index, role]
                require(row.get('rope_rounding') == 'rne' and read(row['rope_tables']) == expected
                        and read(row['expected']) == arena[start:start + size],
                        'prefill RoPE tables or rotated q/k differ from host tables or attention inputs')
        if data.get('decode_device_rope'):
            combined = bytearray(expected)
            references = {(row['step'], row['layer'], row['field']): row for row in data['reference_buffers']
                          if row.get('field') in ('keys', 'values')}
            pack_tiles = lambda value: np.ascontiguousarray(value).reshape(
                value.shape[0] // 32, 32, value.shape[1] // 32, 2, 16).transpose(0, 2, 3, 1, 4).tobytes()
            for step in (1, 2):
                cos = np.full((32, 32), 0x3f80, dtype='<u2')
                sin = np.zeros((32, 32), dtype='<u2')
                cos[0], sin[0] = cosine[127 + step, :32], sine[127 + step, :32]
                tables = b''.join(pack_tiles(value) for value in (cos, sin, sin ^ 0x8000))
                combined.extend(tables)
                for layer in range(layers):
                    attention = data['attention_kernels'][step * layers + layer]
                    arena = read(attention['initial'])
                    q, k = ([row for row in launches if row['kind'] == 'projection'
                             and (row['step'], row['layer'], row['logical']) == (step, layer, role)]
                            for role in (0, 1))
                    require(len(q) == len(k) == 1, 'decode RoPE q/k shard coverage differs')
                    q, k = q[0], k[0]
                    require(all(row.get('rope_rounding') == 'rne' and read(row['rope_tables']) == tables
                                for row in (q, k)), 'decode RoPE coefficients differ from host position or padding')
                    packed_q = np.zeros((32, 256), dtype='<u2')
                    packed_q[:8] = tiles(read(q['expected']), 32, 2048)[0].reshape(4, 8, 64).transpose(1, 0, 2).reshape(8, 256)
                    start = attention['offsets']['q']
                    require(arena[start:attention['offsets']['k']] == pack_tiles(packed_q),
                            'decode rotated q differs from packed attention input')
                    previous = read(references[step - 1, layer, 'keys']['file'])
                    cache = tiles(previous, 160, 256).copy()
                    cache[127 + step] = tiles(read(k['expected']), 32, 256)[0]
                    rotated_cache = pack_tiles(cache)
                    require(arena[attention['offsets']['k']:attention['offsets']['v']] == rotated_cache
                            and read(references[step, layer, 'keys']['file']) == rotated_cache,
                            'decode rotated k append differs from attention cache or saved cache')
            require(read(data['rope_tables_blob']) == bytes(combined), 'combined runtime RoPE table blob differs')
    if data.get('prefill_device_residual'):
        tokens = np.frombuffer(host['prefill_tokens'], dtype='<i8')
        entry = np.frombuffer(host['model.embed_tokens.weight'], dtype='<u2').reshape(32000, 2048)[tokens]
        for layer in range(layers):
            (o,), down = projections[layer, 3], projections[layer, 6]
            h = tiles(read(o['expected']), 128, 2048)
            sites = 2 if data.get('pre_attention_norm') else 1
            norm = data['norm_kernels'][layer * sites + sites - 1] if data.get('post_o_norm') else None
            pre = data['norm_kernels'][layer * 2] if sites == 2 else None
            require(o.get('residual_add') is True and np.array_equal(tiles(read(o['operand']), 128, 2048), entry)
                    and (norm is None or read(norm['h_bf16_tiles']) == read(o['expected']))
                    and (pre is None or np.array_equal(np.frombuffer(read(pre['h_bf16']), dtype='<u2').reshape(128, 2048), entry)),
                    'prefill o residual or pre-attention norm H is not the layer input, or norm H is not the device o output')
            entry = np.zeros_like(entry)
            for row in down:
                lo, hi = row['columns']
                require(row.get('residual_add') is True
                        and np.array_equal(tiles(read(row['operand']), 128, hi - lo), h[:, lo:hi]),
                        'prefill down residual is not the device o output h')
                entry[:, lo:hi] = tiles(read(row['expected']), 128, hi - lo)
        for row in data['reference_buffers']:
            if row.get('field_kind') == 'hidden':
                require(np.array_equal(np.frombuffer(read(row['file']), dtype='<u2').reshape(128, 2048), entry),
                        'block hidden reference is not the device down output')


class AttentionBundle(ChecksumBundle):
    def __init__(self, root):
        super().__init__(root)
        self.data = json.loads(self.read('contract.json'))
        require(json.loads(self.read('kernels.json')) == self.data
                and json.loads(self.read('provenance.json')) == self.data,
                'attention metadata documents differ')
        self.replay = json.loads(self.read('replay.json'))
        self.scope = validate_scope(self.data, self.replay)
        require(type(self.data.get('fusions', False)) is bool, 'invalid fusion mode')
        version = self.data.get('attention_version', 1)
        require(type(version) is int and version in (1, 2)
                and (version == 1 or self.data.get('fusions') is True), 'invalid attention version')
        decode_rope = self.data.get('decode_device_rope', False)
        require(type(decode_rope) is bool and (not decode_rope or self.scope == 'full-model'
                and self.data.get('prefill_device_rope') is True and self.data.get('post_o_norm') is True
                and version == 2), 'decode RoPE requires complete prefill RoPE and attention-v2 mode')
        for item in self.data['projection_kernels']:
            if item['step'] and item['logical'] in (0, 1):
                require(('rope_rounding' in item) == decode_rope, 'decode RoPE capability and q/k programs differ')
        placement = self.data['placement']
        require(placement['arena_base'] == 0x90000000 and placement['dram_base'] == 0x80000000
                and type(placement['dram_bytes']) is int and placement['dram_bytes'] > 0,
                'unsupported attention placement')
        self.arena_bytes = max(item['total_dram_bytes'] for item in self.data['launches'])
        for item in self.data['launches']:
            initial, output, final, program = (self.read(item[key]) for key in
                                               ('initial', 'expected', 'expected_arena', 'original'))
            offset, size = item['output_offset'], len(output)
            require(item['dram_base'] == placement['arena_base'] and len(initial) == len(final) == item['total_dram_bytes']
                    and item['dram_base'] + len(initial) <= min(0x100000000, placement['dram_base'] + placement['dram_bytes'])
                    and 0 <= offset <= len(initial) and 0 < size <= len(initial) - offset,
                    'attention arena extent differs')
            if self.data.get('fusions') and item['kind'] == 'projection':
                validate_fused_output(item, initial, output, final, self.read)
                if version == 2 and item['logical'] == 3:
                    producer = self.data['attention_kernels'][item['input_attention']]
                    previous = self.data['launches'][item['index']-1]
                    require(item.get('input_layout') == 'fp8_tiled' and item.get('input_attention_version') == 2
                            and previous['kind'] == 'attention' and previous['table_index'] == item['input_attention']
                            and (producer['step'], producer['layer']) == (item['step'], item['layer'])
                            and item['A_scale'] == producer['output_scale']
                            and self.read(item['activation']) == self.read(producer['expected']),
                            'o projection lost its exact attention-v2 FP8 producer identity')
            elif version == 2 and item['kind'] == 'attention':
                validate_attention_v2(item, initial, output, final)
            elif self.data.get('post_o_norm') and item['kind'] == 'norm':
                validate_norm(item, initial, output, final, self.read)
            else:
                require(final == initial[:offset] + output + initial[offset + size:]
                        and item['requested_dma']['store_bytes'] == size, 'attention whole arena preservation or DMA differs')
            require(program and len(program) % 4 == 0 and program[-4:] == struct.pack('<I', 0x73)
                    and len(program) + 20 <= 16384, 'attention program does not fit IMEM')
            for key, member in (('initial_sha256', 'initial'), ('program_sha256', 'original'),
                                ('expected_sha256', 'expected')):
                require(item[key] == self.sums[item[member]], 'attention launch hash differs')
        if self.data.get('post_o_norm'):
            validate_norm_joins(self.data, self.read)
        if self.data.get('prefill_device_residual') or self.data.get('prefill_device_rope'):
            require(self.data.get('post_o_norm') is True and self.data.get('device_rounding') == 'rne'
                    and self.data.get('device_steps') == [0], 'F2 prefill device mode differs')
            validate_device_joins(self.data, self.read)
        for row in self.data['host_parameters'].values():
            self.read(row['file'])
        for row in self.data['reference_buffers']:
            raw = self.read(row['file'])
            require(len(raw) == row['bytes'] and digest(raw) == row['sha256'], 'attention final reference bytes differ')
            if row['actual_symbol'] in ('tiny_full_greedy_tokens', 'tiny_full_decode_inputs'):
                values = self.data['greedy_tokens'][:len(raw) // 4]
                require(raw == struct.pack('<' + str(len(values)) + 'I', *values), 'attention greedy feedback differs')

    def qualify(self, path, expected_sha256, *, roots=None, relocations=()):
        raw = checked(path).read_bytes()
        require(digest(raw) == expected_sha256, 'attention qualification receipt checksum differs')
        from receipt_paths import decode
        gate = decode(json.loads(raw), roots or {}, relocations)
        validate_gate(gate, self.data, self.sums['replay.json'], self.sums)
        for row, item in zip(gate['launches'], self.data['launches']):
            words = len(self.read(item['original'])) // 4
            require(row['execution']['program_words'] == words and row['execution']['halt_pc'] == words - 1,
                    'qualified program extent or ECALL differs')
            if gate.get('schema') == 'atlas.d1.all_launch_original.v1':
                require(row['output_bytes'] == len(self.read(item['expected']))
                        and row['arena_bytes'] == item['total_dram_bytes'] and row['dram_base'] == item['dram_base'],
                        'qualified D1 staged arena or output extent differs')
