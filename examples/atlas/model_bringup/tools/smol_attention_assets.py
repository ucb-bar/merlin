"""Typed immutable Smol attention assets, independent of projection admission."""
import hashlib
import io
import json
import struct

import numpy as np
from atlas_host_layout import ATLAS_IMEM_BYTES
import kernel_library  # noqa: E402,F401
from atlas_attention_v2_program import AttentionV2Config, assembly_v2, stage_v2, unpack_tiles, pack_output_v2, unpack_output_v2


def event_plan():
    from smol_full_plan import calls, panels
    graph, physical = calls(), panels()
    result, attention = [], 0
    by_call = {}
    for panel in physical: by_call.setdefault(panel['call'], []).append(panel)
    for call in graph:
        before = call['role'] == (4 if call['site'] == 7 else 3) and call['site'] in (5, 6, 7)
        if before:
            result.append(dict(index=len(result), kind='attention', kernel=attention, before_call=call['index']))
            attention += 1
        for panel in by_call[call['index']]:
            result.append(dict(index=len(result), kind='projection', kernel=panel['index'], call=call['index']))
    if attention != 200 or len(result) != 2915:
        raise ValueError('Incomplete Smol device-event plan')
    return result


def attention_geometry(phase):
    """Logical row/key/head geometry used by captured-asset export and admission."""
    return ((1024, 1024, 12, 12) if phase < 24 else (177, 177, 15, 5) if phase < 40
            else (50, 227 if (phase - 40) % 2 == 0 else 177, 15, 5))


def mask_tiles(config, logical):
    c = config
    value = np.asarray(logical, dtype=bool)
    if value.shape != (c.query_rows, c.keys):
        raise ValueError('Attention logical mask geometry differs')
    padded = np.zeros((c.query_tiles * 32, c.key_tiles * 32), dtype=bool)
    padded[:c.query_rows, :c.keys] = value
    return padded.reshape(c.query_tiles, 32, c.key_tiles, 32).transpose(0, 2, 1, 3)


def validate_attention(item, read, assemble=None):
    """Check complete staged bytes, immutable metadata, canonical source, and output padding."""
    required = ('source', 'original', 'initial', 'expected', 'metadata', 'logical_mask')
    if any(not isinstance(item.get(key), str) for key in required):
        raise ValueError('Missing typed attention asset')
    c = AttentionV2Config(**item['config'])
    if 'phase' in item:
        phase = item['phase']
        if type(phase) is not int or not 0 <= phase < 200 or item.get('index') != phase:
            raise ValueError('Attention semantic phase identity differs')
        geometry = attention_geometry(phase)
        planned = [event for event in event_plan() if event['kind']=='attention'][phase]
        if (c.query_rows,c.keys,c.query_heads,c.kv_heads) != geometry or item.get('before_call') != planned['before_call']:
            raise ValueError('Attention semantic geometry/order differs')
    if c.output_fp8 or c.decode_packed or item.get('dram_base') != 0x90000000:
        raise ValueError('Smol integration requires native BF16 attention in low arena')
    logical = np.load(io.BytesIO(read(item['logical_mask'])), allow_pickle=False)
    if logical.dtype != np.bool_:
        raise ValueError('Attention mask must be exact Boolean capture')
    initial = read(item['initial'])
    qbytes = c.query_tiles * c.query_heads * 4096
    kvbytes = c.key_tiles * c.kv_heads * 4096
    if len(initial) < qbytes + 2 * kvbytes:
        raise ValueError('Truncated attention operands')
    q = unpack_tiles(initial[:qbytes], c.query_heads, c.query_rows)
    k = unpack_tiles(initial[qbytes:qbytes + kvbytes], c.kv_heads, c.keys)
    v = unpack_tiles(initial[qbytes + kvbytes:qbytes + 2 * kvbytes], c.kv_heads, c.keys)
    if any(np.any((value & 0x7f80) == 0x7f80) for value in (q,k,v)):
        raise ValueError('Attention operands contain nonfinite BF16')
    canonical, offsets = stage_v2(c, q, k, v, mask_tiles(c, logical))
    metadata = canonical[offsets['descriptors']:offsets['output'] - 256]
    if (canonical != initial or item.get('offsets') != offsets or read(item['metadata']) != metadata
            or item.get('output_offset') != offsets['output'] or item.get('output_bytes') != offsets['output_bytes']
            or item.get('total_dram_bytes') != len(initial) or len(initial) > 64 * 1024 * 1024
            or hashlib.sha256(initial).hexdigest() != item.get('initial_sha256')):
        raise ValueError('Attention exact staging/metadata identity differs')
    source = assembly_v2(c, offsets, dram_base=item['dram_base']).encode()
    original = read(item['original'])
    if source != read(item['source']) or not original or len(original) % 4 or len(original) + 20 > ATLAS_IMEM_BYTES:
        raise ValueError('Attention canonical source/program extent differs')
    if assemble is not None:
        words = list(assemble(io.StringIO(source.decode())).assemble())
        if original != struct.pack('<' + 'I' * len(words), *words):
            raise ValueError('Attention assembled binary differs')
    words = struct.unpack('<' + 'I' * (len(original) // 4), original)
    if words[-1] != 0x73 or 0x73 in words[:-1] or 0x100073 in words:
        raise ValueError('Attention terminal ECALL differs')
    expected = read(item['expected'])
    if len(expected) != offsets['output_bytes'] or pack_output_v2(c, unpack_output_v2(c, expected)) != expected:
        raise ValueError('Attention output extent or zero padding differs')
    if np.any(unpack_output_v2(c, expected)[:,~logical.any(axis=1)]):
        raise ValueError('All-masked attention output must be zero')
    values = np.frombuffer(expected, '<u2')
    if np.any((values & 0x7f80) == 0x7f80):
        raise ValueError('Attention output contains nonfinite BF16')
    metadata_bytes = offsets['descriptor_bytes'] + offsets['key_count_bytes'] + offsets['mask_bytes']
    # Q loads once, K/V once per KV head, plus immutable metadata and output stores.
    loads = qbytes + 2 * kvbytes + metadata_bytes
    if item.get('requested_dma') != dict(load_bytes=loads, store_bytes=len(expected)):
        raise ValueError('Attention exact DMA accounting differs')
    return c, offsets


def export_attention(rows, load, asset, assemble):
    """Copy authenticated native capture records into typed content-addressed assets."""
    result = []
    planned = [event for event in event_plan() if event['kind'] == 'attention']
    if len(rows) != len(planned):
        raise ValueError('Require all200 captured attention events')
    for row, event in zip(rows, planned):
        if row['index'] != event['kernel'] or row['before_call'] != event['before_call']:
            raise ValueError('Captured attention source order differs')
        config = row['config']
        phase = row['index']
        geometry = attention_geometry(phase)
        if tuple(config[key] for key in ('query_rows','keys','query_heads','kv_heads')) != geometry:
            raise ValueError('Captured attention semantic geometry differs')
        payloads = {name: load(row['files'][name]) for name in
                    ('program.S', 'program.bin', 'initial-dram.bin', 'expected-output.bin', 'metadata.json', 'logical-mask.npy')}
        document = json.loads(payloads['metadata.json'])
        offsets = document.get('offsets', document)
        initial = payloads['initial-dram.bin']
        immutable = initial[offsets['descriptors']:offsets['output'] - 256]
        item = dict(index=row['index'], phase=row['index'], before_call=row['before_call'], kind='attention',
            config=row['config'], offsets=offsets, output_offset=offsets['output'], output_bytes=offsets['output_bytes'],
            total_dram_bytes=len(initial), dram_base=0x90000000, requested_dma=row['requested_dma'],
            initial_sha256=hashlib.sha256(initial).hexdigest())
        for key in ('stats', 'execution', 'provenance', 'expected_arena_sha256'):
            if key in row: item[key] = row[key]
        for key, raw, suffix in (('source', payloads['program.S'], '.S'), ('original', payloads['program.bin'], '.bin'),
            ('initial', initial, '.bin'), ('expected', payloads['expected-output.bin'], '.bin'),
            ('metadata', immutable, '.bin'), ('logical_mask', payloads['logical-mask.npy'], '.npy')):
            item[key] = asset(raw, suffix)
        lookup = {item['source']: payloads['program.S'], item['original']: payloads['program.bin'],
                  item['initial']: initial, item['expected']: payloads['expected-output.bin'],
                  item['metadata']: immutable, item['logical_mask']: payloads['logical-mask.npy']}
        validate_attention(item, lookup.__getitem__, assemble)
        result.append(item)
    return result
