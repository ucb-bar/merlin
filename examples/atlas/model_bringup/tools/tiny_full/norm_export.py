"""Authenticate and export standalone original47 RMS2048 launches at the post-o and pre-attention sites."""
import hashlib
import io
import json
import struct

import numpy as np
from tiny_full.norm import prepare_norm, ARENA_BASE
from tiny_full.fusion_layout import pack_bf16_tiles, unpack_fp8_tiles
from fusion_export import original_sources
from run_workload_contraction import admit_program

SCHEMA = 'atlas.tiny.norm_launch.v1'
MODEL_PIN = '47b8599d8ffbe5ddd96059ffe2b5be2427a410f5'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def norm_kernel(capture, record, assets, assemble, index):
    if (record.get('schema') != SCHEMA or record.get('kind') != 'norm' or record.get('status') != 'PASS'
            or record.get('model_commit') != MODEL_PIN or record.get('numerics') != 'rtl'
            or record.get('accelerated') is not False or record.get('native_dispatch_enabled') is not False
            or record.get('source_bookends_verified') is not True or record.get('dram_base') != ARENA_BASE):
        raise ValueError('norm requires explicit original47 RTL execution')
    if any(type(record.get(key)) is not int for key in ('step', 'layer', 'm', 'logical_rows', 'norm_scale', 'version',
            'total_bytes', 'mask_offset', 'weight_offset', 'output_offset', 'output_bytes')):
        raise ValueError('norm scalar metadata must use exact integer types')
    original_sources(capture, record)
    step, layer, m, rows, scale = (record[key] for key in ('step', 'layer', 'm', 'logical_rows', 'norm_scale'))
    if (type(step) is not int or not 0 <= step <= 2 or type(layer) is not int or not 0 <= layer < 22
            or (m, rows) != ((128, 128) if step == 0 else (32, 1))):
        raise ValueError('norm logical step/layer geometry differs')
    h = capture.array(record['H_bf16'])
    weight = capture.raw(record['norm_weight'])
    site = record.get('site')
    if site not in (f'model.layers.{layer}.post_attention_layernorm', f'model.layers.{layer}.input_layernorm'):
        raise ValueError('norm site differs')
    expected_weight = capture.raw(capture.data['host_parameters'][site + '.weight'])
    if h.shape != (rows, 2048) or h.dtype != np.uint16 or weight != expected_weight:
        raise ValueError('norm H callback or captured gamma differs')
    physical = np.zeros((m, 2048), dtype=np.uint16)
    physical[:rows] = h
    h_tiles = pack_bf16_tiles(physical)
    if capture.raw(record['H_bf16_tiles']) != h_tiles:
        raise ValueError('norm physical H is not the actual callback H with zero padding')
    prepared = prepare_norm(m, h_tiles, weight, scale, rows)
    source, initial = prepared['source'], prepared['initial']
    words = list(assemble(io.StringIO(source)).assemble())
    admit_program(words)
    program = struct.pack(f'<{len(words)}I', *words)
    requested = dict(load_bytes=prepared['dma_load_bytes'], store_bytes=prepared['dma_store_bytes'])
    fields = ('version', 'total_bytes', 'mask_offset', 'weight_offset', 'output_offset', 'output_bytes',
              'output_dtype', 'output_scale', 'output_row_stride', 'output_column_tile', 'written_spans', 'extras')
    if (any(record.get(key) != prepared[key] for key in fields)
            or source.encode() != capture.raw(record['program_source'])
            or program != capture.raw(record['program_binary'])
            or initial != capture.raw(record['arena_initial']) or record['requested_dma'] != requested
            or record['program_sha256'] != digest(program)
            or json.loads(json.dumps(prepared['epilogue'])) != record['epilogue']):
        raise ValueError('norm source, binary, exact staging, or DMA differs from generator')
    execution = record['execution']
    if (execution['program_sha256'] != digest(program) or execution['program_words'] != len(words)
            or execution['halt_pc'] != len(words) - 1
            or {key: execution['observed'].get(key) for key in requested} != requested
            or execution['observed'].get('native_mxu0_calls', 0)):
        raise ValueError('norm original instruction execution or observed DMA differs')
    before = record['harness_sources_before']
    if not before or before != record['harness_sources_after']:
        raise ValueError('norm harness source bookends differ')
    for path, value in before.items():
        pinned = capture.pin(path, value['sha256'])
        if pinned.stat().st_size != value['bytes']:
            raise ValueError('norm harness source byte extent differs')
    final, output = capture.raw(record['arena_expected']), capture.raw(record['output_tiled'])
    offset, size = prepared['output_offset'], prepared['output_bytes']
    if len(output) != size or final != initial[:offset] + output + initial[offset + size:]:
        raise ValueError('norm changed H, constants, guards, or output extent')
    raw = unpack_fp8_tiles(output, m, 2048)
    if np.any(raw[rows:] & 127) or np.any((raw & 127) == 127):
        raise ValueError('norm output padding is not signed zero or output has FP8 NaNs')
    return dict(index=index, kind='norm', kernel_schema=SCHEMA, norm_version=1, step=step, layer=layer,
        site=site, logical_rows=rows, physical_rows=m, width=2048,
        norm_scale=scale, output_scale=scale, output_dtype='fp8', output_layout='fp8_tiled',
        input_layout='bf16_tiled', source=assets.put('programs', source.encode(), '.S'),
        original=assets.put('programs', program), initial=assets.put('arenas', initial),
        expected=assets.put('expected', output), expected_arena=assets.put('arenas', final),
        h_bf16=assets.put('norm-data', h.astype('<u2').tobytes()),
        h_bf16_tiles=assets.put('norm-data', h_tiles), norm_weight=assets.put('norm-data', weight),
        mask=assets.put('norm-data', initial[prepared['mask_offset']:prepared['weight_offset']]),
        weight=assets.put('norm-data', initial[prepared['weight_offset']:-256]),
        norm_h_sha256=digest(h.astype('<u2').tobytes()), initial_sha256=digest(initial),
        program_sha256=digest(program), expected_sha256=digest(output), output_offset=offset,
        output_extent_bytes=size, total_dram_bytes=len(initial), dram_base=ARENA_BASE,
        mask_offset=prepared['mask_offset'], weight_offset=prepared['weight_offset'],
        written_spans=prepared['written_spans'], requested_dma=requested, flags=int(site.endswith('input_layernorm')),
        model_commit=MODEL_PIN, numerics='rtl', accelerated=False, native_dispatch_enabled=False,
        capture_execution=execution, output_quantization=record.get('output_quantization'))
