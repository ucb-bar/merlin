"""Shared Tiny projection staging for capture and immutable export."""
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "host_iss"))

import kernel_library  # noqa: E402,F401
from atlas_epilogue import Epilogue, Step, TileInput
from atlas_program import assembly, dma_load_bytes
from atlas_program_checks import check_matrix_registers
from atlas_rope_postpass import ResidentRopeEpilogue, append_postpass
from chipyard.arena import DEFAULT_ARENA, relocate_assembly

ARENA_BASE = 0x90000000


def rope_tables(cosine, sine, m=128):
    """Pack per-row-tile cos, sin, and -sin BF16 tiles from host [position, 64] rotary tables."""
    import numpy as np
    cosine, sine = (np.asarray(value, dtype='<u2') for value in (cosine, sine))
    if (cosine.shape != sine.shape or cosine.ndim != 2 or cosine.shape[1] != 64 or cosine.shape[0] < m
            or not np.array_equal(cosine[:, :32], cosine[:, 32:]) or not np.array_equal(sine[:, :32], sine[:, 32:])):
        raise ValueError('rotate-half RoPE requires [position, 64] tables with equal halves')
    return b''.join(pack_bf16_tiles(value[:m, :32]) for value in (cosine, sine, sine ^ 0x8000))


def prepare_projection(m, k, n, rescale, role, column, width, A_fp8bytes, B_fp8bytes,
                       operand_tilesbytes=None, output_scale=0, *, residual_adds=True, exact_residual=False,
                       rope=None):
    """Return the complete staged image and exact output and DMA contracts.

    exact_residual selects the RNE residual add for prefill o/down; rope is a
    rope_tables() blob that rotates q/k in a resident post-pass.
    """
    if any(type(d) is not int or d <= 0 or d % 32 for d in (m, k, n)):
        raise ValueError('physical dimensions must be positive multiples of 32')
    if type(role) is not int or not 0 <= role <= 7:
        raise ValueError('unknown Tiny projection role')
    if type(column) is not int or column < 0 or column % 32 or type(width) is not int or column + n > width:
        raise ValueError('projection column span differs from full output width')
    if len(A_fp8bytes) != m * k or len(B_fp8bytes) != k * n:
        raise ValueError('packed contraction operand extent differs')
    if (role in ((3, 5, 6) if residual_adds else (5,))) != (operand_tilesbytes is not None):
        raise ValueError('residual and up projections require exactly one BF16 tile operand')
    if operand_tilesbytes is not None and len(operand_tilesbytes) != 2 * m * n:
        raise ValueError('epilogue operand extent differs')
    if role == 5 and width != 5632:
        raise ValueError('Tiny up FP8 consumer width must be 5632')
    if role != 5 and output_scale != 0:
        raise ValueError('only up output has an FP8 scale')
    if exact_residual and (not residual_adds or role not in (3, 6) or m != 128):
        raise ValueError('exact residual adds are prefill o/down epilogues')
    if rope is not None and (role not in (0, 1) or m not in (32, 128) or len(rope) != 192 * m):
        raise ValueError('RoPE tables are q/k inputs of three BF16 tiles per row tile')
    output_offset = m * k + k * n + 256
    common = dict(m=m, n=n, output_offset=output_offset, scratch_registers=(10, 11))
    if role == 5:
        common.update(output_dtype='fp8', output_scale_exponent=output_scale,
                      output_row_stride=32 * 5632, output_column_tile=column // 32)
    preliminary = Epilogue(**common)
    extra_offset = output_offset + preliminary.output_bytes + 256
    if role in (3, 6) and residual_adds:
        steps = (Step('add_rne' if exact_residual else 'add', TileInput(extra_offset)),)
    elif role == 4:
        steps = (Step('silu'),)
    elif role == 5:
        steps = (Step('mul', TileInput(extra_offset)),)
    else:
        steps = ()
    if rope is not None:
        spec = ResidentRopeEpilogue(**common, cos_offset=extra_offset, sin_offset=extra_offset + 64 * m,
                                    neg_sin_offset=extra_offset + 128 * m)
    else:
        spec = Epilogue(**common, steps=steps) if steps else None
    extent = spec.output_bytes if spec else 2 * m * n
    store_bytes = spec.dma_store_bytes if spec else extent
    initial = bytes(A_fp8bytes) + bytes(B_fp8bytes) + b'\xa5' * 256 + b'\xcd' * extent + b'\x5a' * 256
    if operand_tilesbytes is not None:
        initial += bytes(operand_tilesbytes)
    if rope is not None:
        initial += bytes(rope)
    source = projection_source(m, k, n, output_offset, rescale, spec)
    check_matrix_registers(source)
    if role == 5:
        spans = [dict(offset=row * 32 * 5632 + column * 32, bytes=n * 32) for row in range(m // 32)]
    else:
        spans = [dict(offset=0, bytes=extent)]
    return dict(source=source, initial=initial, output_offset=output_offset, output_bytes=extent,
                dma_load_bytes=dma_load_bytes(m, k, n, traffic_reuse=True, epilogue=spec),
                dma_store_bytes=store_bytes, epilogue=asdict(spec) if spec else None,
                extras=list(spec.input_regions) if spec else [], written_spans=spans,
                output_dtype='fp8' if role == 5 else 'bf16', output_scale=output_scale,
                output_row_stride=32 * 5632 if role == 5 else None,
                output_column_tile=column // 32 if role == 5 else 0)


def projection_source(m, k, n, output_offset, rescale, spec, arena_base=ARENA_BASE):
    """Checked producer assembly, with the RoPE post-pass appended for its resident epilogue."""
    source = assembly(m, k, n, output_offset, rescale, traffic_reuse=True, epilogue=spec)
    if isinstance(spec, ResidentRopeEpilogue):
        source = append_postpass(source, spec)
    return relocate_assembly(source, DEFAULT_ARENA, arena_base)


def epilogue_from(values):
    """Rebuild a recorded asdict() epilogue, selecting RoPE by its table fields."""
    if values is None:
        return None
    values = dict(values, scratch_registers=tuple(values['scratch_registers']),
                  steps=tuple(Step(row['op'], TileInput(**row['operand']) if row['operand'] else None)
                              for row in values['steps']))
    return (ResidentRopeEpilogue if 'cos_offset' in values else Epilogue)(**values)


from tiny_full.fusion_layout import (pack_bf16_tiles, unpack_bf16_tiles, pack_fp8_tiles, unpack_fp8_tiles, pack_fp8_weights)
