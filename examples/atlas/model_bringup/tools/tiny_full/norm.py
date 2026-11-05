"""Standalone RMS2048 staging using the qualified resident norm post-pass."""
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / 'host_iss'))
import numpy as np
import kernel_library  # noqa: E402,F401
from atlas_norm_postpass import ResidentNormEpilogue, postpass, H_WORD
from atlas_program_checks import check_matrix_registers
from chipyard.arena import DEFAULT_ARENA, relocate_assembly, window
from tiny_full.fusion_layout import pack_bf16_tiles, unpack_bf16_tiles

ARENA_BASE = 0x90000000
WIDTH = 2048
MASK_BYTES = 65536
GAMMA_BYTES = 131072


def prepare_norm(m, H_bf16_tiles, norm_weight, norm_scale, logical_rows=None, *, arena_base=ARENA_BASE):
    """Return exact staged bytes, assembly, DMA counts, and one FP8 written span."""
    rows = (128 if m == 128 else 1) if logical_rows is None else logical_rows
    if (m, rows) not in ((128, 128), (32, 1)):
        raise ValueError('norm supports prefill128 or one decode row padded to32')
    if len(H_bf16_tiles) != 2 * m * WIDTH or len(norm_weight) != 2 * WIDTH:
        raise ValueError('norm H/gamma byte extent differs')
    h = unpack_bf16_tiles(H_bf16_tiles, m, WIDTH)
    gamma = np.frombuffer(norm_weight, dtype='<u2')
    if np.any((h & 0x7f80) == 0x7f80) or np.any((gamma & 0x7f80) == 0x7f80):
        raise ValueError('norm inputs must be finite BF16')
    if np.any(h[rows:] & 0x7fff):
        raise ValueError('decode H padding must be signed zero')
    masks = np.zeros((32, 32, 32), dtype='<u2')
    for i in range(32):
        masks[i, :, i] = 0x3f80
    mask_raw = b''.join(pack_bf16_tiles(tile) for tile in masks)
    gamma_raw = pack_bf16_tiles(np.broadcast_to(gamma, (32, WIDTH)).copy())
    output_offset, output_bytes = 2 * m * WIDTH, m * WIDTH
    mask_offset = output_offset + output_bytes + 256
    gamma_offset = mask_offset + MASK_BYTES
    e = ResidentNormEpilogue(m, WIDTH, 0, scratch_registers=(10, 11), norm_kind='rms', real_n=WIDTH,
        norm_mask_offset=mask_offset, norm_weight_offset=gamma_offset, norm_scale_exponent=norm_scale)
    initial = bytes(H_bf16_tiles) + b'\xcd' * output_bytes + b'\x5a' * 256 + mask_raw + gamma_raw + b'\xa5' * 256
    window(arena_base, len(initial))
    lines = ['li x30, 1', 'dma.config x30, ch0', f'li x15, {H_WORD}', 'li x16, 0',
        'li x17, 4096', f'li x18, {len(H_bf16_tiles) // 4096}', 'norm_h_load:',
        'dma.load x15, x16, x17, ch0', 'dma.wait ch0', 'addi x15, x15, 1024',
        'li x19, 4096', 'add x16, x16, x19', 'addi x18, x18, -1',
        'bne x18, x0, norm_h_load', 'nop']
    source = relocate_assembly('\n'.join(lines + postpass(e) + ['ecall']) + '\n', DEFAULT_ARENA, arena_base)
    check_matrix_registers(source)
    return dict(source=source, initial=initial, output_offset=output_offset, output_bytes=output_bytes,
        version=1, total_bytes=len(initial), mask_offset=mask_offset, weight_offset=gamma_offset,
        dma_load_bytes=len(H_bf16_tiles) + MASK_BYTES + GAMMA_BYTES, dma_store_bytes=output_bytes,
        epilogue=asdict(e), extras=list(e.norm_regions), written_spans=[dict(offset=0, bytes=output_bytes)],
        output_dtype='fp8', output_scale=norm_scale, output_row_stride=None, output_column_tile=0,
        m=m, n=WIDTH, logical_rows=rows, dram_base=arena_base)
