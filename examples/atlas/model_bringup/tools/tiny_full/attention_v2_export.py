"""Typed Tiny attention-v2 staging and capture admission."""
from dataclasses import asdict, fields
import io
from pathlib import Path
import struct

import numpy as np

import kernel_library  # noqa: E402,F401
from atlas_attention_v2_program import AttentionV2Config, assembly_v2, stage_v2, unpack_output_v2, unpack_tiles
from atlas_program_checks import check_matrix_registers
from attention_timing_contract import metadata_contract, validate_contract
from fusion_export import original_sources

SCHEMA = 'atlas.tiny.attention_launch.v2'


def config_from_dict(values):
    if not isinstance(values, dict) or set(values) != {item.name for item in fields(AttentionV2Config)}:
        raise ValueError('complete attention-v2 configuration required')
    integer_fields = ('query_rows', 'keys', 'query_heads', 'kv_heads', 'head_dim',
                      'q_exponent', 'k_exponent', 'v_exponent', 'p_exponent', 'o_exponent')
    if (any(type(values[name]) is not int for name in integer_fields)
            or any(type(values[name]) is not bool for name in ('output_fp8', 'packed_decode'))
            or type(values['scale']) not in (int, float)
            or values['query_start'] is not None and type(values['query_start']) is not int):
        raise ValueError('attention-v2 configuration types differ')
    config = AttentionV2Config(**values)
    if (not config.output_fp8 or not config.packed_decode or config.mask != 'causal'
            or config.query_heads != 32 or config.kv_heads != 4 or config.scale != 0.125
            or (config.query_rows, config.keys) not in ((128, 128), (1, 129), (1, 130))
            or config.query_start not in (None, config.keys-config.query_rows)):
        raise ValueError('unsupported Tiny attention-v2 geometry or output policy')
    return config


def output_spans(config, extent):
    if config.decode_packed:
        return [dict(offset=head*2048+half*1024, bytes=32)
                for head in range(config.query_heads) for half in range(2)]
    return [dict(offset=0, bytes=extent)]


def logical_bf16(config, output):
    import torch
    logical = unpack_output_v2(config, output)
    return ((torch.from_numpy(logical.copy()).view(torch.float8_e4m3fn).float()
             * 2.0**config.o_exponent).to(torch.bfloat16).view(torch.uint16).numpy())


def validate_source_tiles(capture, record, name, logical, config):
    descriptor = record.get(name+'_source_tiled')
    if descriptor is None:
        return
    raw = capture.raw(descriptor)
    heads = config.query_heads if name == 'q' else config.kv_heads
    if len(raw) % (heads*128):
        raise ValueError('attention-v2 normal C source extent differs')
    rows = len(raw)//(heads*128)
    minimum = config.query_rows if name == 'q' else config.keys
    if rows % 32 or rows < minimum or rows > (config.query_tiles*32 if name == 'q' else 160):
        raise ValueError('attention-v2 normal C source geometry differs')
    decoded = unpack_tiles(raw, heads, rows)
    if not np.array_equal(decoded[:, :minimum], logical) or name != 'q' and np.any(decoded[:, minimum:]):
        raise ValueError('attention-v2 real input or future KV padding differs')


def attention_kernel_v2(capture, record, assets, assemble, arena_base):
    # The selected model and its complete source inventory remain original47.
    from attention_export import digest
    if (record.get('schema') != SCHEMA or record.get('kind') != 'attention'
            or record.get('model_commit') != '47b8599d8ffbe5ddd96059ffe2b5be2427a410f5'
            or record.get('numerics') != 'rtl' or record.get('accelerated') is not False
            or record.get('native_dispatch_enabled') is not False or record.get('dram_base') != arena_base
            or record.get('status') != 'PASS' or record.get('runtime_patched') is not False
            or record.get('mode') not in ('functional', 'timed')
            or any(record.get(key) is not True for key in ('source_and_binary_exact', 'source_bookends_verified',
                       'halt_exact', 'outside_written_spans_preserved'))
            or record.get('mode') == 'timed' and any(record.get(key) is not True
                       for key in ('engine_drain_verified', 'halt_and_drain_exact'))):
        raise ValueError('attention-v2 requires original47 RTL instruction execution')
    original_sources(capture, record)
    config = config_from_dict(record['config'])
    if (record.get('output_dtype') != 'fp8' or record.get('output_layout') != 'fp8_tiled'
            or record.get('output_scale') != config.o_exponent):
        raise ValueError('attention-v2 typed FP8 result differs')
    q, k, v = (capture.array(record[name+'_bf16']) for name in ('q', 'k', 'v'))
    for name, value, shape in zip(('q', 'k', 'v'), (q, k, v),
            ((32, config.query_rows, 64), (4, config.keys, 64), (4, config.keys, 64))):
        if value.dtype != np.uint16 or value.shape != shape or np.any((value & 0x7f80) == 0x7f80):
            raise ValueError('attention-v2 logical BF16 input differs')
        validate_source_tiles(capture, record, name, value, config)
    initial, offsets = stage_v2(config, q, k, v)
    offset, extent = offsets['output'], offsets['output_bytes']
    if (record['offsets'] != offsets or capture.raw(record['arena_initial']) != initial
            or record['arena_bytes'] != len(initial) or record['output_offset'] != offset
            or record['output_bytes'] != extent):
        raise ValueError('attention-v2 exact staging differs')
    timing_contract = metadata_contract(config, offsets, initial, arena_base)
    if 'timing_dram_contract' in record:
        validate_contract(config, offsets, initial, arena_base, record['timing_dram_contract'])
    for name, start, end in (('q', offsets['q'], offsets['k']), ('k', offsets['k'], offsets['v']),
                            ('v', offsets['v'], offsets['descriptors'])):
        if capture.raw(record['staged_'+name]) != initial[start:end]:
            raise ValueError('attention-v2 staged operand differs')
    source = assembly_v2(config, offsets, dram_base=arena_base)
    check_matrix_registers(source)
    words = list(assemble(io.StringIO(source)).assemble())
    from run_workload_contraction import admit_program
    admit_program(words)
    program = struct.pack(f'<{len(words)}I', *words)
    if source.encode() != capture.raw(record['program_source']) or program != capture.raw(record['program_binary']):
        raise ValueError('attention-v2 generated instructions differ')
    spans = output_spans(config, extent)
    requested = dict(load_bytes=offset-256, store_bytes=sum(span['bytes'] for span in spans))
    execution = record['execution']
    model = execution.get('model', {})
    expected_sources = {str(Path(record['model_root'])/name): sha
                        for name, sha in model.get('source_sha256', {}).items()}
    if (record['written_spans'] != spans or record['requested_dma'] != requested
            or execution['program_sha256'] != digest(program) or execution['program_words'] != len(words)
            or record.get('program_sha256') != digest(program)
            or execution['halt_pc'] != len(words)-1 or execution.get('instructions', 0) <= 0
            or model.get('revision') != record['model_commit'] or model.get('numerics') != 'rtl'
            or model.get('root') != record['model_root'] or expected_sources != record['model_sources_before']
            or (record['mode'] == 'timed' and (type(execution.get('cycles')) is not int or execution['cycles'] <= 0))
            or any(execution['observed'].get(key) != value for key, value in requested.items())
            or execution['observed'].get('native_mxu0_calls', 0) != 0):
        raise ValueError('attention-v2 execution, extent, or DMA differs')
    before, after = record['harness_sources_before'], record['harness_sources_after']
    if not before or before != after:
        raise ValueError('attention-v2 harness source bookends differ')
    required = {str(kernel_library.source_path(name)) for name in ('tiny_full/attention_v2_worker.py', 'atlas_execution.py',
                'atlas_path_guard.py', 'atlas_attention_v2_program.py', 'atlas_program_checks.py')}
    normalized = {}
    for path, sha in before.items():
        normalized[str(capture.pin(path, sha))] = sha
    implementations = execution.get('implementation_source_sha256', {})
    normalized_implementations = {str(capture.pin(path, sha)): sha for path, sha in implementations.items()}
    expected_implementations = {path: sha for path, sha in normalized.items()
                               if Path(path).name in ('atlas_execution.py', 'atlas_path_guard.py')}
    if (set(normalized) != required or len(normalized) != len(before)
            or normalized_implementations != expected_implementations):
        raise ValueError('attention-v2 actual helper or Executor source closure differs')
    output, final = capture.raw(record['output_tiled']), capture.raw(record['arena_expected'])
    if len(output) != extent or len(final) != len(initial):
        raise ValueError('attention-v2 full output or final arena extent differs')
    wanted = bytearray(initial)
    for span in spans:
        start, size = span['offset'], span['bytes']
        wanted[offset+start:offset+start+size] = output[start:start+size]
    compact = b''.join(output[span['offset']:span['offset']+span['bytes']] for span in spans)
    logical = logical_bf16(config, output)
    if (bytes(wanted) != final or final[offset:offset+extent] != output
            or capture.raw(record['output_fp8']) != compact
            or not np.array_equal(capture.array(record['output_bf16']), logical)):
        raise ValueError('attention-v2 changed a guard, zero hole, logical value, or stored output')
    item = dict(kind='attention', attention_version=2, kernel_schema=SCHEMA,
        step=record['step'], layer=record['layer'], site=record['site'], config=asdict(config), offsets=offsets,
        input_layout='bf16_tiled_packed_decode' if config.decode_packed else 'bf16_tiled',
        output_layout='fp8_tiled', output_dtype='fp8', output_scale=config.o_exponent,
        output_extent_bytes=extent, written_spans=spans, packed_decode_effective=bool(config.decode_packed),
        source=assets.put('programs', source.encode(), '.S'), original=assets.put('programs', program),
        initial=assets.put('arenas', initial), expected=assets.put('expected', output),
        expected_arena=assets.put('arenas', final), initial_sha256=digest(initial), program_sha256=digest(program),
        expected_sha256=digest(output), output_offset=offset, total_dram_bytes=len(initial), dram_base=arena_base,
        requested_dma=requested, model_commit=record['model_commit'], numerics='rtl', capture_execution=execution,
        timing_dram_contract=timing_contract)
    if record.get('output_quantization') is not None:
        item['output_quantization'] = record['output_quantization']
    return item, logical.transpose(1, 0, 2).reshape(config.query_rows, -1)


def companion_c(items):
    """Emit only the parallel v2 ABI; legacy panel bytes and fields are unchanged."""
    lines, rows = [], []
    for index, item in enumerate(items):
        config = config_from_dict(item['config'])
        offsets = item['offsets']
        spans = output_spans(config, offsets['output_bytes'])
        if item.get('attention_version') != 2 or item['written_spans'] != spans:
            raise ValueError('attention-v2 companion span policy differs')
        name = f'tiny_full_attention_v2_spans_{index}'
        lines.append('static const tiny_attention_span '+name+'[]={' + ','.join(
            '{'+str(span['offset'])+','+str(span['bytes'])+'}' for span in spans) + '};')
        values = (2, 2 if config.decode_packed else 1, 2, offsets['k']-offsets['q'],
                  offsets['v']-offsets['k'], offsets['key_counts'], offsets['key_count_bytes'],
                  offsets['masks'], offsets['mask_bytes'], offsets['output_bytes'],
                  config.o_exponent, int(config.decode_packed), name, len(spans))
        rows.append('{'+','.join(map(str, values))+'}')
    lines.append('const tiny_attention_v2_panel tiny_full_attention_v2_panels[TINY_FULL_ATTENTION_KERNELS]={'
                 + ',\n'.join(rows) + '};')
    return lines
