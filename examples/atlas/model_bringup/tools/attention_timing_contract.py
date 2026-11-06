"""Explicit immutable DRAM input facts for attention timing verification.

The returned bytes are assumptions about the launch image, not facts inferred
from a program fingerprint. Callers must preserve these bytes during execution.
Only descriptors and counts are needed; mask element values do not affect timing.
"""
from dataclasses import asdict, is_dataclass
import struct


def metadata_contract(config, offsets, initial, dram_base=0x90000000):
    """Validate an attention-v2 metadata image and extract two exact segments."""
    c = asdict(config) if is_dataclass(config) else config
    if not isinstance(c, dict) or not isinstance(offsets, dict):
        raise ValueError('attention configuration and offsets must be mappings')
    dimensions = ('query_rows', 'keys', 'query_heads', 'kv_heads')
    if any(type(c.get(key)) is not int or c[key] <= 0 for key in dimensions):
        raise ValueError('invalid attention dimensions')
    if (c.get('head_dim') != 64 or c['query_heads'] % c['kv_heads']
            or type(c.get('packed_decode')) is not bool
            or type(c.get('output_fp8')) is not bool):
        raise ValueError('unsupported attention-v2 geometry')
    qt, kt = (c['query_rows'] + 31) // 32, (c['keys'] + 31) // 32
    if qt > 128 or kt > 64:
        raise ValueError('attention-v2 resident geometry exceeds capacity')
    packed = c['packed_decode'] and c['query_rows'] == 1
    if packed and c['query_heads'] // c['kv_heads'] > 32:
        raise ValueError('packed attention group exceeds one tile')
    qb = (c['kv_heads'] if packed else qt * c['query_heads']) * 4096
    kb = kt * c['kv_heads'] * 4096
    db, cb = (qt * kt * 8 + 31) // 32 * 32, (qt * 4 + 31) // 32 * 32
    if db > 16384 or cb > 4096:
        raise ValueError('attention-v2 metadata exceeds reserved banks')
    needed = ('q', 'k', 'v', 'descriptors', 'descriptor_bytes', 'key_counts',
              'key_count_bytes', 'masks', 'mask_bytes', 'output', 'output_bytes')
    if any(type(offsets.get(key)) is not int or offsets[key] < 0 for key in needed):
        raise ValueError('invalid attention metadata offsets')
    expected = dict(q=0, k=qb, v=qb + kb, descriptors=qb + 2 * kb,
                    descriptor_bytes=db, key_counts=qb + 2 * kb + db,
                    key_count_bytes=cb, masks=qb + 2 * kb + db + cb,
                    output_bytes=qt * c['query_heads'] * (2048 if c['output_fp8'] else 4096))
    if (any(offsets[key] != value for key, value in expected.items())
            or offsets['mask_bytes'] % 2048 or offsets['mask_bytes'] > 196608
            or offsets['output'] != offsets['masks'] + offsets['mask_bytes'] + 256):
        raise ValueError('attention metadata layout differs')
    if not isinstance(initial, (bytes, bytearray, memoryview)):
        raise ValueError('attention initial image must be bytes')
    if (len(initial) != offsets['output'] + offsets['output_bytes'] + 256
            or type(dram_base) is not int or dram_base < 0 or dram_base % 32
            or dram_base + len(initial) > 2**32):
        raise ValueError('attention DRAM image extent differs')
    descriptors = bytes(initial[offsets['descriptors']:offsets['descriptors'] + db])
    counts = bytes(initial[offsets['key_counts']:offsets['key_counts'] + cb])
    if any(descriptors[qt * kt * 8:]) or any(counts[qt * 4:]):
        raise ValueError('attention metadata padding must be zero')
    entries = list(struct.iter_unpack('<II', descriptors[:qt * kt * 8]))
    for address, skip in entries:
        if skip not in (0, 1) or skip and address:
            raise ValueError('invalid attention descriptor skip flag')
        if address:
            displacement = (address - 0x4000) * 4
            if (address < 0x4000 or displacement % 2048
                    or displacement + 2048 > offsets['mask_bytes']):
                raise ValueError('attention descriptor mask base outside reserved masks')
    for qi, (count,) in enumerate(struct.iter_unpack('<I', counts[:qt * 4])):
        visible = [ki + 1 for ki in range(kt) if entries[qi * kt + ki][1] == 0]
        if count > kt or count != (visible[-1] if visible else 0):
            raise ValueError('attention key count differs from visible descriptors')
    return dict(name='attention-metadata-v1', segments=[
        dict(address=dram_base + offsets['descriptors'], bytes_hex=descriptors.hex()),
        dict(address=dram_base + offsets['key_counts'], bytes_hex=counts.hex())])


def validate_contract(config, offsets, initial, dram_base, supplied):
    """Reject a recorded contract that differs from the validated launch bytes."""
    expected = metadata_contract(config, offsets, initial, dram_base)
    if supplied != expected:
        raise ValueError('attention timing DRAM contract differs from launch image')
    return expected
