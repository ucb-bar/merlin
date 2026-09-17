"""Gemmini command construction derived from the public header and RTL facts."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

from ir_ingest import InterfaceProgram, TensorSpec
from targetgen.synthesize.tiling import tile_matmul

DIM = 16
ADDR_LEN = 32
SPAD_ROWS = 262144 // DIM
ACC_ROWS = 65536 // (DIM * 4)
GARBAGE_ADDR = (1 << ADDR_LEN) - 1
ACC_BASE = 1 << 31
ACC_ACCUMULATE = 1 << 30
ACC_FULL = 1 << 29

FUNCT = {
    "CONFIG": 0, "CONFIG_EX": 0, "CONFIG_LD": 0, "CONFIG_ST": 0,
    "MVIN2": 1, "MVIN": 2, "MVOUT": 3,
    "COMPUTE_PRELOADED": 4, "COMPUTE_ACCUMULATE": 5,
    "PRELOAD": 6, "FLUSH": 7,
    "LOOP_WS": 8, "LOOP_WS_CONFIG_BOUNDS": 9,
    "LOOP_WS_CONFIG_ADDRS_AB": 10, "LOOP_WS_CONFIG_ADDRS_DC": 11,
    "LOOP_WS_CONFIG_STRIDES_AB": 12, "LOOP_WS_CONFIG_STRIDES_DC": 13,
    "LOOP_CONV_WS": 15, "LOOP_CONV_WS_CONFIG_1": 16,
    "LOOP_CONV_WS_CONFIG_2": 17, "LOOP_CONV_WS_CONFIG_3": 18,
    "LOOP_CONV_WS_CONFIG_4": 19, "LOOP_CONV_WS_CONFIG_5": 20,
    "LOOP_CONV_WS_CONFIG_6": 21,
}


@dataclass(frozen=True)
class Address:
    tensor: str
    offset: int = 0


@dataclass(frozen=True)
class Instruction:
    name: str
    rs1: int | Address | None = None
    rs2: int | Address | None = None

    @property
    def funct(self) -> int | None:
        return FUNCT.get(self.name)


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _round_up(value: int, unit: int = DIM) -> int:
    return ((value + unit - 1) // unit) * unit


def _elem_bytes(dtype: str) -> int:
    return {"i8": 1, "i16": 2, "i32": 4}.get(dtype, 1)


def _row_stride(tensor: TensorSpec) -> int:
    """Return the harness' physical byte stride for the innermost row.

    Interface shapes remain logical, but the bare-metal ABI pads the final
    dimension to whole Gemmini tiles.  DMA descriptors and pointer offsets
    must use that physical layout; tile column counts remain logical.
    """
    return _round_up(tensor.shape[-1]) * _elem_bytes(tensor.dtype)


def _tile_word(addr: int, cols: int, rows: int) -> int:
    return (rows << (ADDR_LEN + 16)) | (cols << ADDR_LEN) | addr


def _config_ld(stride: int, *, shrunk: bool = False, channel: int = 0) -> Instruction:
    rs1 = (_f32_bits(1.0) << 32) | (DIM << 16) | (1 << 8) | (channel << 3) | (int(shrunk) << 2) | 1
    return Instruction("CONFIG_LD", rs1, stride)


def _config_ex(*, transpose_b: bool = False, weight_stationary: bool = True) -> Instruction:
    # gemmini_config_ex expands through extended3_config_ex with A_stride=1
    # and C_stride=1, plus the identity systolic accumulator scale. Omitting a
    # default is not equivalent to zero: all three occupy live RTL fields.
    rs1 = (_f32_bits(1.0) << 32) | (1 << 16)
    rs1 |= (int(weight_stationary) << 2) | (int(transpose_b) << 9)
    rs2 = 1 << 48
    return Instruction("CONFIG_EX", rs1, rs2)


def _config_st(stride: int, attrs: dict[str, Any], output_dtype: str) -> Instruction:
    stages = attrs.get("epilogue", [])
    act = 1 if "relu" in stages else 0
    scale = float(attrs.get("acc_scale", 1.0))
    rs1 = (act << 2) | 2
    if "maxpool" in stages:
        h, w = attrs["pool_in_dims"]
        ph, pw = attrs["pool_size"]
        sh, sw = attrs["pool_stride"]
        pt, pl, pb, pr = attrs.get("pool_padding", [0, 0, 0, 0])
        if ph != pw or sh != sw:
            raise ValueError("Gemmini store path requires square pooling geometry")
        oh = (h + pt + pb - ph) // sh + 1
        ow = (w + pl + pr - pw) // sw + 1
        rs1 |= (w << 56) | (h << 48) | (ow << 40) | (oh << 32) | (ow << 24)
        rs1 |= (pl << 10) | (pt << 8) | (ph << 6) | (sh << 4)
    rs2 = (_f32_bits(scale) << 32) | stride
    return Instruction("CONFIG_ST", rs1, rs2)


def _matmul_trace(program: InterfaceProgram, lhs_name: str, weight_name: str, out_name: str,
                  attrs: dict[str, Any], *, transpose_b: bool = False) -> list[Instruction]:
    lhs, weight, out = program.tensors[lhs_name], program.tensors[weight_name], program.tensors[out_name]
    m, k = lhs.shape
    if transpose_b:
        n, wk = weight.shape
        if wk != k:
            raise ValueError("attention_qk contraction mismatch")
    else:
        wk, n = weight.shape
        if wk != k:
            raise ValueError("matmul contraction mismatch")
    lhs_stride = _row_stride(lhs)
    weight_stride = _row_stride(weight)
    out_stride = _row_stride(out)
    # The loop-controller configuration functs are not exposed by this target
    # contract. Use the public header's explicit weight-stationary schedule,
    # whose K -> N -> M order preserves each resident B tile across M blocks.
    i_blocks, j_blocks, k_blocks = (_round_up(x) // DIM for x in (m, n, k))
    a_base = 0
    b_base = SPAD_ROWS - k_blocks * j_blocks * DIM
    c_base = (3 << 30) | (ACC_FULL if out.dtype == "i32" else 0)
    pool = "maxpool" in attrs.get("epilogue", [])
    trace: list[Instruction] = [
        _config_ex(transpose_b=transpose_b, weight_stationary=True),
        _config_st(out_stride, attrs, out.dtype),
    ]
    if not transpose_b:
        # Mirror sp_tiled_matmul_ws exactly: K -> N -> M. The first M block
        # preloads B and uses COMPUTE_PRELOADED; later M blocks consume those
        # resident systolic weights with COMPUTE_ACCUMULATE. Re-preloading B for
        # every M block leaves the later block's pipeline empty on this RTL.
        for kk, k0 in enumerate(range(0, k, DIM)):
            td = min(DIM, k - k0)
            for nj, n0 in enumerate(range(0, n, DIM)):
                tc = min(DIM, n - n0)
                b_spad = b_base + (kk * j_blocks + nj) * DIM
                b_offset = k0 * weight_stride + n0 * _elem_bytes(weight.dtype)
                trace.extend([
                    _config_ld(weight_stride, channel=0),
                    Instruction("MVIN", Address(weight_name, b_offset),
                                _tile_word(b_spad, tc, td)),
                ])
                for mi, m0 in enumerate(range(0, m, DIM)):
                    tr = min(DIM, m - m0)
                    a_spad = a_base + (mi * k_blocks + kk) * DIM
                    a_offset = m0 * lhs_stride + k0 * _elem_bytes(lhs.dtype)
                    trace.extend([
                        _config_ld(lhs_stride, channel=0),
                        Instruction("MVIN", Address(lhs_name, a_offset),
                                    _tile_word(a_spad, td, tr)),
                    ])
                    c_index = nj * i_blocks + mi if pool else mi * j_blocks + nj
                    c_addr = c_base + c_index * DIM
                    dst = c_addr if kk else c_addr & ~ACC_ACCUMULATE
                    pre_spad = b_spad if mi == 0 else GARBAGE_ADDR
                    trace.append(Instruction("PRELOAD", _tile_word(pre_spad, tc, td),
                                             _tile_word(dst, tc, tr)))
                    compute = "COMPUTE_PRELOADED" if mi == 0 else "COMPUTE_ACCUMULATE"
                    trace.append(Instruction(compute, _tile_word(a_spad, td, tr),
                                             _tile_word(GARBAGE_ADDR, DIM, DIM)))
                    if kk == k_blocks - 1 and not pool:
                        out_offset = m0 * out_stride + n0 * _elem_bytes(out.dtype)
                        trace.append(Instruction("MVOUT", Address(out_name, out_offset),
                                                 _tile_word(c_addr, tc, tr)))
        if pool:
            # The public Gemmini store API selects its spatial max-pool walk
            # with rows=0.  Accumulator tiles are N-major here so each channel
            # block's complete flattened image is contiguous.
            for nj, n0 in enumerate(range(0, n, DIM)):
                tc = min(DIM, n - n0)
                c_addr = c_base + nj * i_blocks * DIM
                trace.append(Instruction("MVOUT",
                                         Address(out_name, n0 * _elem_bytes(out.dtype)),
                                         _tile_word(c_addr, tc, 0)))
        return trace
    if transpose_b:
        for nj, n0 in enumerate(range(0, n, DIM)):
            tc = min(DIM, n - n0)
            for kk, k0 in enumerate(range(0, k, DIM)):
                td = min(DIM, k - k0)
                b_spad = b_base + (kk * j_blocks + nj) * DIM
                b_offset = n0 * weight_stride + k0 * _elem_bytes(weight.dtype)
                b_rows, b_cols = tc, td
                trace.extend([
                    _config_ld(weight_stride, channel=0),
                    Instruction("MVIN", Address(weight_name, b_offset),
                                _tile_word(b_spad, b_cols, b_rows)),
                ])
    else:
        for kk, k0 in enumerate(range(0, k, DIM)):
            td = min(DIM, k - k0)
            b_spad = b_base + kk * j_blocks * DIM
            trace.extend([
                _config_ld(weight_stride, channel=0),
                Instruction("MVIN", Address(weight_name, k0 * weight_stride),
                            _tile_word(b_spad, n, td)),
            ])
    for mi, m0 in enumerate(range(0, m, DIM)):
        tr = min(DIM, m - m0)
        a_spad = a_base + mi * k_blocks * DIM
        trace.extend([
            _config_ld(lhs_stride, channel=0),
            Instruction("MVIN", Address(lhs_name, m0 * lhs_stride),
                        _tile_word(a_spad, k, tr)),
        ])
    for kk, k0 in enumerate(range(0, k, DIM)):
        td = min(DIM, k - k0)
        for nj, n0 in enumerate(range(0, n, DIM)):
            tc = min(DIM, n - n0)
            b_spad = b_base + (kk * j_blocks + nj) * DIM
            for mi, m0 in enumerate(range(0, m, DIM)):
                tr = min(DIM, m - m0)
                a_spad = a_base + (mi * k_blocks + kk) * DIM
                c_addr = c_base + (mi * j_blocks + nj) * DIM
                out_addr = c_addr if kk else c_addr & ~ACC_ACCUMULATE
                pre_spad = b_spad if mi == 0 else GARBAGE_ADDR
                trace.append(Instruction("PRELOAD", _tile_word(pre_spad, tc, td),
                                         _tile_word(out_addr, tc, tr)))
                compute = "COMPUTE_PRELOADED" if mi == 0 else "COMPUTE_ACCUMULATE"
                trace.append(Instruction(compute, _tile_word(a_spad, td, tr),
                                         _tile_word(GARBAGE_ADDR, DIM, DIM)))
    for mi, m0 in enumerate(range(0, m, DIM)):
        tr = min(DIM, m - m0)
        for nj, n0 in enumerate(range(0, n, DIM)):
            tc = min(DIM, n - n0)
            c_addr = c_base + (mi * j_blocks + nj) * DIM
            out_offset = m0 * out_stride + n0 * _elem_bytes(out.dtype)
            trace.append(Instruction("MVOUT", Address(out_name, out_offset),
                                     _tile_word(c_addr, tc, tr)))
    trace.append(Instruction("FENCE"))
    return trace

    # Retained explicit sequence documents the primitive-command fallback for
    # implementations which do not expose the loop controller.
    k_tiles = (k + DIM - 1) // DIM
    # The header's sp_tiled_matmul_os sequence is what preserves a systolic
    # partial sum across multiple K tiles. A single tile uses the simpler WS
    # sequence, which is also the resident-weight fast path.
    use_ws = not transpose_b
    trace: list[Instruction] = [
        _config_ex(transpose_b=transpose_b, weight_stationary=use_ws),
        _config_st(out_stride, attrs, out.dtype),
    ]
    tiles = tile_matmul(m, n, k, DIM)
    grouped: dict[tuple[int, int], list[Any]] = {}
    for tile in tiles:
        grouped.setdefault((tile.m0, tile.n0), []).append(tile)
    for (m0, n0), group in grouped.items():
        if use_ws:
            # The public WS header always retains the accumulator/full-width
            # address bits in C_sp_addr_start.  Bit 30 is cleared only for the
            # first K tile (zero-initialize), then set for accumulation and the
            # final readback.
            c_addr = ACC_BASE | (ACC_FULL if out.dtype == "i32" else 0)
        else:
            # Public sp_tiled_matmul_os: C_sp_addr_start = 3 << 30,
            # with bit 29 selecting full-width accumulator readout.
            c_addr = (3 << 30) | (ACC_FULL if out.dtype == "i32" else 0)
        for depth_index, tile in enumerate(group):
            a_offset = m0 * lhs_stride + tile.k0 * _elem_bytes(lhs.dtype)
            if transpose_b:
                b_offset = n0 * weight_stride + tile.k0 * _elem_bytes(weight.dtype)
                b_rows, b_cols = tile.cols, tile.depth
            else:
                b_offset = tile.k0 * weight_stride + n0 * _elem_bytes(weight.dtype)
                b_rows, b_cols = tile.depth, tile.cols
            trace.extend([
                _config_ld(lhs_stride, channel=0),
                Instruction("MVIN", Address(lhs_name, a_offset), _tile_word(0, tile.depth, tile.rows)),
                # The exposed target contract admits the canonical MVIN class;
                # reconfigure its row stride immediately before loading B.
                _config_ld(weight_stride, channel=0),
                Instruction("MVIN", Address(weight_name, b_offset), _tile_word(DIM, b_cols, b_rows)),
            ])
            if use_ws:
                dst = c_addr | (ACC_ACCUMULATE if depth_index else 0)
                trace.append(Instruction("PRELOAD", _tile_word(DIM, b_cols, b_rows),
                                         _tile_word(dst, tile.cols, tile.rows)))
                # WS advances through M blocks with COMPUTE_PRELOADED only for
                # the first block; later blocks use COMPUTE_ACCUMULATE to keep
                # the mesh's resident-weight control state live.  C overwrite/
                # accumulation remains independently selected by dst bit 30.
                compute = "COMPUTE_PRELOADED" if m0 == 0 else "COMPUTE_ACCUMULATE"
                trace.append(Instruction(compute, _tile_word(0, tile.depth, tile.rows),
                                         _tile_word(GARBAGE_ADDR, tile.cols, tile.rows)))
            else:
                out_addr = c_addr if depth_index == len(group) - 1 else GARBAGE_ADDR
                trace.append(Instruction("PRELOAD", _tile_word(GARBAGE_ADDR, DIM, DIM),
                                         _tile_word(out_addr, tile.cols, tile.rows)))
                compute = "COMPUTE_PRELOADED" if depth_index == 0 else "COMPUTE_ACCUMULATE"
                trace.append(Instruction(compute, _tile_word(0, tile.depth, tile.rows),
                                         _tile_word(DIM, b_cols, b_rows)))
        out_offset = m0 * out_stride + n0 * _elem_bytes(out.dtype)
        read_addr = c_addr | (ACC_ACCUMULATE if use_ws else 0)
        trace.append(Instruction("MVOUT", Address(out_name, out_offset),
                                 _tile_word(read_addr, group[-1].cols, group[-1].rows)))
        # This capacity-fit lowering intentionally reuses one A/B/C tile for
        # each output block.  Retire the current block before the following
        # DMA overwrites those physical rows.
        trace.append(Instruction("FENCE"))
    return trace


def _movement_trace(program: InterfaceProgram, command: dict[str, Any]) -> list[Instruction]:
    src_name, dst_name = command["operands"]["src"], command["operands"]["dst"]
    src, dst = program.tensors[src_name], program.tensors[dst_name]
    rows, cols = src.shape
    src_stride = _row_stride(src)
    dst_stride = _row_stride(dst)
    trace: list[Instruction] = [_config_st(dst_stride, command.get("attributes", {}), dst.dtype)]
    if dst.dtype == "i32":
        trace.insert(0, _config_ex(weight_stationary=True))
    tile_index = 0
    for m0 in range(0, rows, DIM):
        for n0 in range(0, cols, DIM):
            tr, tc = min(DIM, rows - m0), min(DIM, cols - n0)
            src_offset = m0 * src_stride + n0 * _elem_bytes(src.dtype)
            dst_offset = m0 * dst_stride + n0 * _elem_bytes(dst.dtype)
            if dst.dtype == "i32":
                spad = ACC_BASE + tile_index * DIM
                trace.append(_config_ld(src_stride, shrunk=True))
                trace.append(Instruction("MVIN", Address(src_name, src_offset),
                                         _tile_word(spad, tc, tr)))
                trace.append(Instruction("MVOUT", Address(dst_name, dst_offset),
                                         _tile_word(spad | ACC_FULL, tc, tr)))
            else:
                spad = tile_index * DIM
                trace.append(_config_ld(src_stride))
                trace.append(Instruction("MVIN", Address(src_name, src_offset), _tile_word(spad, tc, tr)))
                trace.append(Instruction("MVOUT", Address(dst_name, dst_offset), _tile_word(spad, tc, tr)))
            tile_index += 1
    return trace


def _conv_loop_trace(program: InterfaceProgram, command: dict[str, Any], weight_name: str) -> list[Instruction]:
    """Lower NHWC convolution through primitive, canonically decoded commands.

    Each kernel tap/channel slice is a K tile.  One-row MVINs gather the
    corresponding IFM pixels into an A tile, so no derived DRAM/im2col pointer
    enters the kernel ABI and no unsupported LOOP_CONV config funct is needed.
    """
    attrs, ops = command["attributes"], command["operands"]
    ifm, weight, out = program.tensors[ops["ifm"]], program.tensors[weight_name], program.tensors[ops["dst"]]
    batch, in_h, in_w, ci = ifm.shape
    kh, kw, kci, co = attrs["kernel"]
    sh, sw = attrs["stride"]
    pt, pl, pb, pr = attrs["padding"]
    dh, dw = attrs["dilation"]
    if attrs.get("layout") != "nhwc" or ci != kci:
        raise ValueError("conv primitive lowering requires matching NHWC channels")
    if weight.shape != [kh * kw * ci, co]:
        raise ValueError("conv weight must use pre-im2col [Kh*Kw*Ci, Co] layout")
    out_h = (in_h + pt + pb - (dh * (kh - 1) + 1)) // sh + 1
    out_w = (in_w + pl + pr - (dw * (kw - 1) + 1)) // sw + 1
    m = batch * out_h * out_w
    i_blocks = _round_up(m) // DIM
    j_blocks = _round_up(co) // DIM
    groups = [(kr, kc, c0, min(DIM, ci - c0))
              for kr in range(kh) for kc in range(kw)
              for c0 in range(0, ci, DIM)]
    a_rows = len(groups) * i_blocks * DIM
    b_rows = len(groups) * j_blocks * DIM
    if a_rows + b_rows > SPAD_ROWS:
        raise ValueError("conv gather footprint exceeds scratchpad capacity")
    if i_blocks * j_blocks * DIM > ACC_ROWS:
        raise ValueError("conv output footprint exceeds accumulator capacity")

    in_elem = _elem_bytes(ifm.dtype)
    weight_elem = _elem_bytes(weight.dtype)
    out_elem = _elem_bytes(out.dtype)
    ifm_stride = _round_up(ci) * in_elem
    weight_stride = _row_stride(weight)
    out_stride = _row_stride(out)
    b_base = SPAD_ROWS - b_rows
    c_base = (3 << 30) | (ACC_FULL if out.dtype == "i32" else 0)
    pool = "maxpool" in attrs.get("epilogue", [])
    trace = [_config_ex(weight_stationary=True), _config_st(out_stride, attrs, out.dtype)]

    plane = out_h * out_w
    for gi, (kr, kc, c0, depth) in enumerate(groups):
        for nj, n0 in enumerate(range(0, co, DIM)):
            tc = min(DIM, co - n0)
            b_spad = b_base + (gi * j_blocks + nj) * DIM
            weight_row = (kr * kw + kc) * ci + c0
            weight_offset = weight_row * weight_stride + n0 * weight_elem
            trace.extend([
                _config_ld(weight_stride),
                Instruction("MVIN", Address(weight_name, weight_offset),
                            _tile_word(b_spad, tc, depth)),
            ])
            for mi, m0 in enumerate(range(0, m, DIM)):
                tr = min(DIM, m - m0)
                a_spad = (gi * i_blocks + mi) * DIM
                trace.append(_config_ld(ifm_stride))
                for local_row in range(tr):
                    logical_row = m0 + local_row
                    batch_index = logical_row // plane
                    spatial = logical_row % plane
                    oh_index, ow_index = divmod(spatial, out_w)
                    ih = oh_index * sh - pt + kr * dh
                    iw = ow_index * sw - pl + kc * dw
                    if 0 <= ih < in_h and 0 <= iw < in_w:
                        pixel = (batch_index * in_h + ih) * in_w + iw
                        ifm_offset = pixel * ifm_stride + c0 * in_elem
                        trace.append(Instruction("MVIN", Address(ops["ifm"], ifm_offset),
                                                 _tile_word(a_spad + local_row, depth, 1)))
                c_index = nj * i_blocks + mi if pool else mi * j_blocks + nj
                c_addr = c_base + c_index * DIM
                dst = c_addr if gi else c_addr & ~ACC_ACCUMULATE
                pre_spad = b_spad if mi == 0 else GARBAGE_ADDR
                trace.append(Instruction("PRELOAD", _tile_word(pre_spad, tc, depth),
                                         _tile_word(dst, tc, tr)))
                compute = "COMPUTE_PRELOADED" if mi == 0 else "COMPUTE_ACCUMULATE"
                trace.append(Instruction(compute, _tile_word(a_spad, depth, tr),
                                         _tile_word(GARBAGE_ADDR, DIM, DIM)))
                if gi == len(groups) - 1 and not pool:
                    out_offset = m0 * out_stride + n0 * out_elem
                    trace.append(Instruction("MVOUT", Address(ops["dst"], out_offset),
                                             _tile_word(c_addr, tc, tr)))
    if pool:
        for nj, n0 in enumerate(range(0, co, DIM)):
            tc = min(DIM, co - n0)
            c_addr = c_base + nj * i_blocks * DIM
            trace.append(Instruction("MVOUT", Address(ops["dst"], n0 * out_elem),
                                     _tile_word(c_addr, tc, 0)))
    return trace


def build_trace(program: InterfaceProgram) -> list[Instruction]:
    if not program.is_contract_module:
        return [Instruction("FENCE"), Instruction("FLUSH", 0, 0)]
    trace: list[Instruction] = [Instruction("FENCE"), Instruction("FLUSH", 0, 0)]
    resident_sources: dict[str, str] = {}
    pending_matmuls: dict[str, tuple[str, str]] = {}
    for command in program.commands:
        opcode, ops = command["opcode"], command.get("operands", {})
        if opcode == "RES_PACK":
            resident_sources[ops["dst"]] = ops["src"]
        elif opcode == "MATMUL_RESIDENT":
            pending_matmuls[ops["dst"]] = (ops["lhs"], resident_sources[ops["rhs"]])
        elif opcode == "COMMIT":
            lhs, weight = pending_matmuls[ops["src"]]
            trace.extend(_matmul_trace(program, lhs, weight, ops["dst"], command["attributes"]))
        elif opcode == "MOVEMENT":
            trace.extend(_movement_trace(program, command))
        elif opcode == "CONV2D":
            trace.extend(_conv_loop_trace(program, command, resident_sources[ops["weight"]]))
        elif opcode == "ATTENTION_QK":
            trace.extend(_matmul_trace(program, ops["q"], ops["k"], ops["dst"],
                                       command.get("attributes", {}), transpose_b=True))
    return trace
