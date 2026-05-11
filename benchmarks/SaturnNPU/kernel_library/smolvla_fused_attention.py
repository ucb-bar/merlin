# ruff: noqa: E501, E741
# E501: the embedded MLIR-IR literal has long lines that shouldn't wrap.
# E741: kernel math uses single-letter names (l, m, O) that mirror the
# flash-attention paper — renaming would hurt readability, not help it.
"""Fused attention (SDPA) kernel — matches MLIR variant_0_12_1024_64_bf16.

MLIR source: SaturnNPU/kernels/iree_linalg_ext.attention/variant_0_12_1024_64_bf16.mlir

  Q:      [12, 1024,   64] bf16   indexing (d0, d1, d3) → [batch, q_seq, head_dim]
  K:      [12, 1024,   64] bf16   indexing (d0, d4, d3) → [batch, k_seq, head_dim]
  V:      [12,   64, 1024] bf16   indexing (d0, d2, d4) → [batch, head_dim, k_seq]
  scale:  scalar bf16
  output: [12, 1024,   64] bf16

This demo tiles to one Q-block (32 rows) and two K tiles (k_seq=64 total, 32 per tile),
with head_dim=64.  Production wraps this body in loops over all 1024 q-rows and 12 heads.

Flash attention:
    Softmax is global over the full key sequence — naive per-tile softmax gives wrong results.
    We use online softmax (flash attention), maintaining running stats across K tiles:

        m = -inf, l = 0, O = 0
        for each K tile:
            S  = Q @ K^T * scale
            m' = max(m, rowmax(S))
            α  = exp(m − m')
            O  = α * O + exp(S − m') @ V
            l  = α * l + rowsum(exp(S − m'))
            m  = m'
        output = O / l

DRAM layouts (all bf16, column-blocked so each vload gets a contiguous [32,16] chunk):

  Q_DRAM:   [32, 64] bf16 stored as [128, 16]  (4 × [32,16] blocks, cols 0:16/16:32/32:48/48:64)
  KT_DRAM:  K tile [32,64] pre-transposed → K^T[64,32] stored as [128,16]  (top/bot × left/right)
  VT_DRAM:  V_mlir tile [64,32] pre-transposed → V_std[32,64] stored as [128,16]
  OUT_DRAM: [32, 64] bf16 stored as [128, 16]

Q @ K^T (head_dim=64, two MXU passes accumulating over head_dim=32 per pass):
    Q_lo [32,32] fp8 @ K^T_top [32,32] fp8  → acc[0]  (fresh)
    Q_hi [32,32] fp8 @ K^T_bot [32,32] fp8  → acc[0]  (accumulate)
    vmatpop.bf16 writes acc[0][32,32] into registers v16 (left [32,16]) and v17 (right [32,16])

exp_s @ V (output is [32,64], two independent MXU passes):
    exp_s [32,32] fp8 @ V_left  [32,32] fp8 → acc[0]  (O left  half)
    exp_s [32,32] fp8 @ V_right [32,32] fp8 → acc[1]  (O right half)

bf16 → fp8 quantization uses the acc roundtrip (vmatpush.acc.bf16 + vmatpop.fp8).
  vmatpush.acc.bf16(vd=slot, vs1=v) reads v and v+1 as a [32,32] BF16 tile → acc[slot]
  vmatpop.fp8(vd=dst, vs1=slot) converts acc[slot][32,32] → fp8 → dst register [32,32]

-inf initialization:
  m (running row-max) is initialized to -100.0 via vli.all imm=-100.  This is encoded
  as an integer immediate (vli.all calls torch.full(shape, -100, dtype=bf16) = -100.0_bf16).
  True bf16 -inf (0xFF80) cannot be encoded directly as a vli.all integer immediate.
  -100.0_bf16 is sufficient: all real attention logits after scaling (SCALE≈0.125) will
  be much larger than -100, so the first tile always overwrites the initial m value.
"""

import math
from typing import Any

import torch
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs

from ...software import Instruction, Program

FUSED_ATTENTION_MLIR = """
// Kernel: iree_linalg_ext.attention, variant 0
// Result: tensor<12x1024x64xbf16>
// Operands: ['tensor<12x1024x64xbf16>', 'tensor<12x1024x64xbf16>', 'tensor<12x64x1024xbf16>']
// Affine maps fully resolved

%79 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> ()>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>]} ins(%70, %72, %78, %cst_231, %74 : tensor<12x1024x64xbf16>, tensor<12x1024x64xbf16>, tensor<12x64x1024xbf16>, bf16, tensor<12x1024x1024xi1>) outs(%67 : tensor<12x1024x64xbf16>) {
^bb0(%arg13: bf16):
  iree_linalg_ext.yield %arg13 : bf16
} -> tensor<12x1024x64xbf16>
"""


def fused_attention_reference(
    Q: torch.Tensor,  # [q_rows, head_dim] float
    K: torch.Tensor,  # [k_seq,  head_dim] float  (MLIR layout, not transposed)
    V_mlir: torch.Tensor,  # [head_dim, k_seq]  float  (MLIR layout, head_dim-first)
    scale: float,
) -> torch.Tensor:
    """Exact SDPA matching the MLIR affine maps. Returns [q_rows, head_dim] float."""
    scores = Q @ K.t() * scale  # [q_rows, k_seq]
    row_max = scores.max(dim=1, keepdim=True).values
    exp_s = torch.exp(scores - row_max)
    attn = exp_s / exp_s.sum(dim=1, keepdim=True)
    return attn @ V_mlir.t()  # [q_rows, head_dim]


# ── shapes ────────────────────────────────────────────────────────────────────
Q_ROWS = 32
K_SEQ = 64  # two K tiles; forces the online-softmax correction to be non-trivial
HEAD_DIM = 64  # matches MLIR head_dim
SCALE_VALUE = 1.0 / math.sqrt(float(HEAD_DIM))

torch.manual_seed(42)

# MLIR-shaped inputs (bf16, as the op receives them)
Q_RAW = (torch.randn(Q_ROWS, HEAD_DIM) * 0.5).to(torch.bfloat16)  # [32, 64]

# K tiles from MLIR K[batch, k_seq, head_dim]: tile shape [32, 64]
K0_RAW = (torch.randn(Q_ROWS, HEAD_DIM) * 0.5).to(torch.bfloat16)  # k pos  0:32
K1_RAW = (torch.randn(Q_ROWS, HEAD_DIM) * 0.5).to(torch.bfloat16)  # k pos 32:64

# V tiles from MLIR V[batch, head_dim, k_seq]: tile shape [64, 32] (head_dim-first)
V0_MLIR = (torch.randn(HEAD_DIM, Q_ROWS) * 0.5).to(torch.bfloat16)  # k pos  0:32
V1_MLIR = (torch.randn(HEAD_DIM, Q_ROWS) * 0.5).to(torch.bfloat16)  # k pos 32:64

SCALE_DATA = torch.full((Q_ROWS, HEAD_DIM // 4), SCALE_VALUE, dtype=torch.bfloat16)  # [32,16]


def _col_block(t: torch.Tensor) -> torch.Tensor:
    """Pack a [rows, cols] bf16 tensor into column-blocked DRAM layout.

    Each vload reads 1024 contiguous bytes = one [32, 16] bf16 block.
    Blocks are ordered: col 0:16, col 16:32, col 32:48, ... for the first 32 rows,
    then the same for the next 32 rows, etc.
    """
    rows, cols = t.shape
    assert rows % 32 == 0 and cols % 16 == 0
    blocks = []
    for row_start in range(0, rows, 32):
        for col_start in range(0, cols, 16):
            blocks.append(t[row_start : row_start + 32, col_start : col_start + 16].contiguous())
    return torch.cat(blocks, dim=0)


# K tiles: MLIR has K[k_seq, head_dim]; pre-transpose to K^T[head_dim, k_seq] for MXU.
KT0_RAW = K0_RAW.T.contiguous()  # [64, 32] bf16
KT1_RAW = K1_RAW.T.contiguous()  # [64, 32] bf16

# V tiles: MLIR has V[head_dim, k_seq]; pre-transpose to V_std[k_seq, head_dim] for MXU.
VT0_RAW = V0_MLIR.T.contiguous()  # [32, 64] bf16
VT1_RAW = V1_MLIR.T.contiguous()  # [32, 64] bf16

Q_DATA = _col_block(Q_RAW)  # [128, 16] bf16
KT0_DATA = _col_block(KT0_RAW)  # [128, 16] bf16
KT1_DATA = _col_block(KT1_RAW)  # [128, 16] bf16
VT0_DATA = _col_block(VT0_RAW)  # [128, 16] bf16
VT1_DATA = _col_block(VT1_RAW)  # [128, 16] bf16


def _golden(Q_raw, K0_raw, K1_raw, V0_mlir, V1_mlir, scale_val):
    """Simulate exact ISA operations so the expected output matches bit-for-bit."""
    Q_lo_fp8 = Q_raw[:, :32].to(torch.float8_e4m3fn)  # [32,32]
    Q_hi_fp8 = Q_raw[:, 32:].to(torch.float8_e4m3fn)  # [32,32]

    scale_r = torch.full((Q_ROWS, HEAD_DIM // 4), scale_val, dtype=torch.bfloat16)  # [32,16]
    m = torch.full((Q_ROWS, HEAD_DIM // 4), -100.0, dtype=torch.bfloat16)
    l = torch.zeros(Q_ROWS, HEAD_DIM // 4, dtype=torch.bfloat16)
    Oll = torch.zeros(Q_ROWS, HEAD_DIM // 4, dtype=torch.bfloat16)  # O[:, 0:16]
    Olr = torch.zeros(Q_ROWS, HEAD_DIM // 4, dtype=torch.bfloat16)  # O[:, 16:32]
    Orl = torch.zeros(Q_ROWS, HEAD_DIM // 4, dtype=torch.bfloat16)  # O[:, 32:48]
    Orr = torch.zeros(Q_ROWS, HEAD_DIM // 4, dtype=torch.bfloat16)  # O[:, 48:64]

    for k_raw, v_mlir in [(K0_raw, V0_mlir), (K1_raw, V1_mlir)]:
        KT = k_raw.T  # [64, 32]
        KT_top_fp8 = KT[:32, :].to(torch.float8_e4m3fn)  # [32,32]
        KT_bot_fp8 = KT[32:, :].to(torch.float8_e4m3fn)  # [32,32]

        V_std = v_mlir.T  # [32, 64]
        V_left_fp8 = V_std[:, :32].to(torch.float8_e4m3fn)  # [32,32]
        V_right_fp8 = V_std[:, 32:].to(torch.float8_e4m3fn)  # [32,32]

        # Q @ K^T: two MXU passes accumulating over head_dim
        scores = (
            Q_lo_fp8.to(torch.float16) @ KT_top_fp8.to(torch.float16)
            + Q_hi_fp8.to(torch.float16) @ KT_bot_fp8.to(torch.float16)
        ).to(torch.bfloat16)  # [32, 32]

        sl, sr = scores[:, :16], scores[:, 16:]
        sl = (sl * scale_r).to(torch.bfloat16)
        sr = (sr * scale_r).to(torch.bfloat16)

        rm_l = sl.max(dim=1, keepdim=True).values.expand(-1, 16).to(torch.bfloat16)
        rm_r = sr.max(dim=1, keepdim=True).values.expand(-1, 16).to(torch.bfloat16)
        tile_max = torch.maximum(rm_l, rm_r).to(torch.bfloat16)
        m_new = torch.maximum(m, tile_max).to(torch.bfloat16)

        exp_diff = torch.exp((m - m_new).to(torch.bfloat16)).to(torch.bfloat16)

        Oll = (Oll * exp_diff).to(torch.bfloat16)
        Olr = (Olr * exp_diff).to(torch.bfloat16)
        Orl = (Orl * exp_diff).to(torch.bfloat16)
        Orr = (Orr * exp_diff).to(torch.bfloat16)

        esl = torch.exp((sl - m_new).to(torch.bfloat16)).to(torch.bfloat16)
        esr = torch.exp((sr - m_new).to(torch.bfloat16)).to(torch.bfloat16)

        # quantize exp_s [32,32] to fp8 via acc roundtrip
        exp_s_fp8 = torch.cat([esl, esr], dim=1).to(torch.float8_e4m3fn)

        # exp_s @ V (two independent [32,32] matmuls)
        vc_left = (exp_s_fp8.to(torch.float16) @ V_left_fp8.to(torch.float16)).to(torch.bfloat16)
        vc_right = (exp_s_fp8.to(torch.float16) @ V_right_fp8.to(torch.float16)).to(torch.bfloat16)

        Oll = (Oll + vc_left[:, :16]).to(torch.bfloat16)
        Olr = (Olr + vc_left[:, 16:]).to(torch.bfloat16)
        Orl = (Orl + vc_right[:, :16]).to(torch.bfloat16)
        Orr = (Orr + vc_right[:, 16:]).to(torch.bfloat16)

        rs_l = esl.sum(dim=1, keepdim=True).expand(-1, 16).to(torch.bfloat16)
        rs_r = esr.sum(dim=1, keepdim=True).expand(-1, 16).to(torch.bfloat16)
        l = ((exp_diff * l).to(torch.bfloat16) + (rs_l + rs_r).to(torch.bfloat16)).to(torch.bfloat16)
        m = m_new

    inv_l = (1.0 / l).to(torch.bfloat16)
    Oll = (Oll * inv_l).to(torch.bfloat16)
    Olr = (Olr * inv_l).to(torch.bfloat16)
    Orl = (Orl * inv_l).to(torch.bfloat16)
    Orr = (Orr * inv_l).to(torch.bfloat16)

    # column-blocked DRAM layout: [128, 16] bf16
    return torch.cat([Oll, Olr, Orl, Orr], dim=0)


EXPECTED = _golden(Q_RAW, K0_RAW, K1_RAW, V0_MLIR, V1_MLIR, SCALE_VALUE)

# ── memory layout ─────────────────────────────────────────────────────────────
TILE_BYTES = Q_ROWS * HEAD_DIM * 2  # 4096 — one [32,64] bf16 tile
SCALE_BYTES = Q_ROWS * (HEAD_DIM // 4) * 2  # 1024 — one [32,16] bf16 register

DRAM_Q = 0x0000
DRAM_KT0 = 0x1000
DRAM_KT1 = 0x2000
DRAM_VT0 = 0x3000
DRAM_VT1 = 0x4000
DRAM_SCALE = 0x5000
DRAM_OUT = 0x6000

VMEM_Q = 0x8000
VMEM_KT0 = 0x9000
VMEM_KT1 = 0xA000
VMEM_VT0 = 0xB000
VMEM_VT1 = 0xC000
VMEM_SCALE = 0xD000
VMEM_OUT = 0xE000


class SmolVLAFusedAttentionProgram(Program):
    """
    Flash attention: output[b,q,h] = softmax(Q[b,q,:] @ K[b,:,:]^T * scale) @ V[b,:,h]

    One Q-block (32 rows), two K tiles (k_seq=64), head_dim=64.
    Inputs are bf16 (matching MLIR). Quantized to fp8 on-chip via acc roundtrip.

    MRF register map  (simulator: each fp8 reg = [32,32], bf16 reg = [32,16])
    ────────────────
    Persistent (survive across K tiles):
      v0   Q_lo   fp8 [32,32]  Q[:, 0:32] after bf16→fp8 roundtrip
      v1   Q_hi   fp8 [32,32]  Q[:, 32:64] after bf16→fp8 roundtrip
      v2   m_prev bf16 [32,16] running row-max; init = -100.0 (see -inf note in module docstring)
      v3   l_prev bf16 [32,16] running row-sum, init = 0
      v4   O_col0 bf16 [32,16] O[:, 0:16]
      v5   O_col1 bf16 [32,16] O[:, 16:32]
      v6   O_col2 bf16 [32,16] O[:, 32:48]
      v7   O_col3 bf16 [32,16] O[:, 48:64]
      v8   scale  bf16 [32,16]

    Per K-tile (after load+quantize):
      v9   KT_top  fp8 [32,32]  K^T[0:32,  :]  (vmatpush.acc.bf16 reads v9+v10 as pair)
      v11  KT_bot  fp8 [32,32]  K^T[32:64, :]  (vmatpush.acc.bf16 reads v11+v12 as pair)
      v12  VT_left  fp8 [32,32] V_std[:, 0:32]  (vmatpush.acc.bf16 reads v12+v13 as pair)
      v14  VT_right fp8 [32,32] V_std[:, 32:64] (vmatpush.acc.bf16 reads v14+v15 as pair)

    Temporaries (reused each iteration):
      v16,v17  scores sl,sr    bf16 [32,16]  (written as pair by vmatpop.bf16 acc[0])
      v18,v19  scaled sl,sr    bf16 [32,16]
      v20      tile_max / exp_diff bf16 [32,16]
      v21      m_new            bf16 [32,16]
      v22,v23  exp_s left,right bf16 [32,16]
      v24      exp_s_fp8        fp8  [32,32]  (acc[1] roundtrip; vmatpush reads v22+v23 as pair)
      v25,v26  vc_left  col0,col1 bf16 [32,16]  (written as pair by vmatpop.bf16 acc[0])
      v27,v28  vc_right col0,col1 bf16 [32,16]  (written as pair by vmatpop.bf16 acc[1])

    Scalar register map
    ───────────────────
    VMEM:  x1=Q x2=KT0 x3=KT1 x4=VT0 x5=VT1 x6=SCALE x7=OUT
    DRAM:  x9=KT0 x10=KT1 x11=VT0 x12=VT1 x13=SCALE x14=OUT
    Sizes: x15=4096 (tile)  x16=1024 (scale)
    """

    instructions: list[Instruction[Any]] = [
        # ── scalar setup: VMEM addresses ──────────────────────────────────────
        Instruction(mnemonic="lui", args=ScalarArgs(rd=1, imm=0x8)),  # x1  = 0x8000 VMEM_Q
        Instruction(mnemonic="lui", args=ScalarArgs(rd=2, imm=0x9)),  # x2  = 0x9000 VMEM_KT0
        Instruction(mnemonic="lui", args=ScalarArgs(rd=3, imm=0xA)),  # x3  = 0xA000 VMEM_KT1
        Instruction(mnemonic="lui", args=ScalarArgs(rd=4, imm=0xB)),  # x4  = 0xB000 VMEM_VT0
        Instruction(mnemonic="lui", args=ScalarArgs(rd=5, imm=0xC)),  # x5  = 0xC000 VMEM_VT1
        Instruction(mnemonic="lui", args=ScalarArgs(rd=6, imm=0xD)),  # x6  = 0xD000 VMEM_SCALE
        Instruction(mnemonic="lui", args=ScalarArgs(rd=7, imm=0xE)),  # x7  = 0xE000 VMEM_OUT
        # ── scalar setup: DRAM addresses ──────────────────────────────────────
        # x0 = 0 = DRAM_Q (hardwired)
        Instruction(mnemonic="lui", args=ScalarArgs(rd=9, imm=0x1)),  # x9  = 0x1000 DRAM_KT0
        Instruction(mnemonic="lui", args=ScalarArgs(rd=10, imm=0x2)),  # x10 = 0x2000 DRAM_KT1
        Instruction(mnemonic="lui", args=ScalarArgs(rd=11, imm=0x3)),  # x11 = 0x3000 DRAM_VT0
        Instruction(mnemonic="lui", args=ScalarArgs(rd=12, imm=0x4)),  # x12 = 0x4000 DRAM_VT1
        Instruction(mnemonic="lui", args=ScalarArgs(rd=13, imm=0x5)),  # x13 = 0x5000 DRAM_SCALE
        Instruction(mnemonic="lui", args=ScalarArgs(rd=14, imm=0x6)),  # x14 = 0x6000 DRAM_OUT
        Instruction(mnemonic="lui", args=ScalarArgs(rd=15, imm=0x1)),  # x15 = 0x1000 = 4096
        Instruction(mnemonic="addi", args=ScalarArgs(rd=16, rs1=0, imm=1024)),  # x16 = 1024
        # ── DMA: DRAM → VMEM ─────────────────────────────────────────────────
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.load.ch<N>", args=DmaArgs(rd=1, rs1=0, rs2=15, channel=0)),  # Q
        Instruction(mnemonic="dma.load.ch<N>", args=DmaArgs(rd=2, rs1=9, rs2=15, channel=1)),  # KT0
        Instruction(mnemonic="dma.load.ch<N>", args=DmaArgs(rd=3, rs1=10, rs2=15, channel=2)),  # KT1
        Instruction(mnemonic="dma.load.ch<N>", args=DmaArgs(rd=4, rs1=11, rs2=15, channel=3)),  # VT0
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=1)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=2)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=3)),
        Instruction(mnemonic="dma.load.ch<N>", args=DmaArgs(rd=5, rs1=12, rs2=15, channel=0)),  # VT1
        Instruction(mnemonic="dma.load.ch<N>", args=DmaArgs(rd=6, rs1=13, rs2=16, channel=1)),  # SCALE
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=1)),
        # ── Q: VMEM → MRF, then bf16 → fp8 via acc[1] roundtrip ──────────────
        # Load four [32,16] blocks: Q_col0/col1/col2/col3  (imm12 in units of 32 bytes)
        Instruction(mnemonic="vload", args=VectorArgs(vd=9, rs1=1, imm12=0)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=10, rs1=1, imm12=32)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=11, rs1=1, imm12=64)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=12, rs1=1, imm12=96)),
        # Q_lo [32,32] bf16 = (v9, v10) → v0 fp8
        Instruction(mnemonic="vmatpush.acc.bf16.mxu0", args=VectorArgs(vd=1, vs1=9)),
        Instruction(mnemonic="vmatpop.fp8.acc.mxu0", args=VectorArgs(vd=0, vs1=1)),
        # Q_hi [32,32] bf16 = (v11, v12) → v1 fp8
        Instruction(mnemonic="vmatpush.acc.bf16.mxu0", args=VectorArgs(vd=1, vs1=11)),
        Instruction(mnemonic="vmatpop.fp8.acc.mxu0", args=VectorArgs(vd=1, vs1=1)),
        # ── load scale ────────────────────────────────────────────────────────
        Instruction(mnemonic="vload", args=VectorArgs(vd=8, rs1=6, imm12=0)),
        # ── initialize online-softmax state ───────────────────────────────────
        Instruction(mnemonic="vli.all", args=VectorArgs(vd=2, imm=-100)),  # m_prev = -100 ≈ -∞
        Instruction(mnemonic="vli.all", args=VectorArgs(vd=3, imm=0)),  # l_prev = 0
        Instruction(mnemonic="vli.all", args=VectorArgs(vd=4, imm=0)),  # O_col0 = 0
        Instruction(mnemonic="vli.all", args=VectorArgs(vd=5, imm=0)),  # O_col1 = 0
        Instruction(mnemonic="vli.all", args=VectorArgs(vd=6, imm=0)),  # O_col2 = 0
        Instruction(mnemonic="vli.all", args=VectorArgs(vd=7, imm=0)),  # O_col3 = 0
        # ══════════════════════════════════════════════════════════════════════
        # K TILE 0  (k_seq 0:32)
        # ══════════════════════════════════════════════════════════════════════
        # Load KT0: K^T[64,32] bf16, column-blocked → 4 × [32,16] blocks
        Instruction(mnemonic="vload", args=VectorArgs(vd=9, rs1=2, imm12=0)),  # K^T_top left
        Instruction(mnemonic="vload", args=VectorArgs(vd=10, rs1=2, imm12=32)),  # K^T_top right
        Instruction(mnemonic="vload", args=VectorArgs(vd=11, rs1=2, imm12=64)),  # K^T_bot left
        Instruction(mnemonic="vload", args=VectorArgs(vd=12, rs1=2, imm12=96)),  # K^T_bot right
        # K^T_top (v9, v10) → v9 fp8
        Instruction(mnemonic="vmatpush.acc.bf16.mxu0", args=VectorArgs(vd=1, vs1=9)),
        Instruction(mnemonic="vmatpop.fp8.acc.mxu0", args=VectorArgs(vd=9, vs1=1)),
        # K^T_bot (v11, v12) → v11 fp8
        Instruction(mnemonic="vmatpush.acc.bf16.mxu0", args=VectorArgs(vd=1, vs1=11)),
        Instruction(mnemonic="vmatpop.fp8.acc.mxu0", args=VectorArgs(vd=11, vs1=1)),
        # Load VT0: V_std[32,64] bf16, column-blocked → 4 × [32,16] blocks
        Instruction(mnemonic="vload", args=VectorArgs(vd=12, rs1=4, imm12=0)),  # V_left col0
        Instruction(mnemonic="vload", args=VectorArgs(vd=13, rs1=4, imm12=32)),  # V_left col1
        Instruction(mnemonic="vload", args=VectorArgs(vd=14, rs1=4, imm12=64)),  # V_right col0
        Instruction(mnemonic="vload", args=VectorArgs(vd=15, rs1=4, imm12=96)),  # V_right col1
        # V_left (v12, v13) → v12 fp8
        Instruction(mnemonic="vmatpush.acc.bf16.mxu0", args=VectorArgs(vd=1, vs1=12)),
        Instruction(mnemonic="vmatpop.fp8.acc.mxu0", args=VectorArgs(vd=12, vs1=1)),
        # V_right (v14, v15) → v14 fp8
        Instruction(mnemonic="vmatpush.acc.bf16.mxu0", args=VectorArgs(vd=1, vs1=14)),
        Instruction(mnemonic="vmatpop.fp8.acc.mxu0", args=VectorArgs(vd=14, vs1=1)),
        # Q @ K^T, accumulating over head_dim (two 32-wide passes)
        Instruction(mnemonic="vmatpush.weight.mxu0", args=VectorArgs(vd=0, vs1=9)),  # WB[0] = K^T_top
        Instruction(mnemonic="delay", args=ScalarArgs(imm=17)),
        Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=0, vs2=0)),  # acc[0] = Q_lo @ K^T_top
        Instruction(mnemonic="delay", args=ScalarArgs(imm=64)),  # ≥ MT=64 cycles
        Instruction(mnemonic="vmatpush.weight.mxu0", args=VectorArgs(vd=0, vs1=11)),  # WB[0] = K^T_bot
        Instruction(mnemonic="delay", args=ScalarArgs(imm=17)),
        Instruction(mnemonic="vmatmul.acc.mxu0", args=MatrixArgs(vd=0, vs1=1, vs2=0)),  # acc[0] += Q_hi @ K^T_bot
        Instruction(mnemonic="delay", args=ScalarArgs(imm=64)),  # ≥ MT=64 cycles
        Instruction(mnemonic="vmatpop.bf16.acc.mxu0", args=VectorArgs(vd=16, vs1=0)),  # v16=sl, v17=sr
        # scale scores
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=18, vs1=16, vs2=8)),
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=19, vs1=17, vs2=8)),
        # m_new = max(m_prev, rowmax(sl), rowmax(sr))
        Instruction(mnemonic="vredmax.row.bf16", args=VectorArgs(vd=20, vs1=18)),
        Instruction(mnemonic="vredmax.row.bf16", args=VectorArgs(vd=21, vs1=19)),
        Instruction(mnemonic="vmaximum.bf16", args=VectorArgs(vd=20, vs1=20, vs2=21)),
        Instruction(mnemonic="vmaximum.bf16", args=VectorArgs(vd=21, vs1=2, vs2=20)),  # m_new
        # exp_diff = exp(m_prev - m_new)
        Instruction(mnemonic="vsub.bf16", args=VectorArgs(vd=20, vs1=2, vs2=21)),
        Instruction(mnemonic="vexp.bf16", args=VectorArgs(vd=20, vs1=20)),
        # rescale output accumulator (all 4 halves)
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=4, vs1=4, vs2=20)),
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=5, vs1=5, vs2=20)),
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=6, vs1=6, vs2=20)),
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=7, vs1=7, vs2=20)),
        # exp_s = exp(scores_sc - m_new)
        Instruction(mnemonic="vsub.bf16", args=VectorArgs(vd=22, vs1=18, vs2=21)),
        Instruction(mnemonic="vsub.bf16", args=VectorArgs(vd=23, vs1=19, vs2=21)),
        Instruction(mnemonic="vexp.bf16", args=VectorArgs(vd=22, vs1=22)),
        Instruction(mnemonic="vexp.bf16", args=VectorArgs(vd=23, vs1=23)),
        # quantize exp_s [32,32] to fp8 via acc[1] roundtrip
        Instruction(mnemonic="vmatpush.acc.bf16.mxu0", args=VectorArgs(vd=1, vs1=22)),  # reads v22,v23
        Instruction(mnemonic="vmatpop.fp8.acc.mxu0", args=VectorArgs(vd=24, vs1=1)),
        # exp_s @ V_left → acc[0]
        Instruction(mnemonic="vmatpush.weight.mxu0", args=VectorArgs(vd=0, vs1=12)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=17)),
        Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=24, vs2=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=64)),  # ≥ MT=64 cycles
        Instruction(mnemonic="vmatpop.bf16.acc.mxu0", args=VectorArgs(vd=25, vs1=0)),  # v25=vc_ll, v26=vc_lr
        # exp_s @ V_right → acc[1]
        Instruction(mnemonic="vmatpush.weight.mxu0", args=VectorArgs(vd=0, vs1=14)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=17)),
        Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=1, vs1=24, vs2=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=64)),  # ≥ MT=64 cycles
        Instruction(mnemonic="vmatpop.bf16.acc.mxu0", args=VectorArgs(vd=27, vs1=1)),  # v27=vc_rl, v28=vc_rr
        # O += V contribution
        Instruction(mnemonic="vadd.bf16", args=VectorArgs(vd=4, vs1=4, vs2=25)),
        Instruction(mnemonic="vadd.bf16", args=VectorArgs(vd=5, vs1=5, vs2=26)),
        Instruction(mnemonic="vadd.bf16", args=VectorArgs(vd=6, vs1=6, vs2=27)),
        Instruction(mnemonic="vadd.bf16", args=VectorArgs(vd=7, vs1=7, vs2=28)),
        # l = exp_diff * l + rowsum(exp_s_left) + rowsum(exp_s_right)
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=25, vs1=20, vs2=3)),
        Instruction(mnemonic="vredsum.row.bf16", args=VectorArgs(vd=26, vs1=22)),
        Instruction(mnemonic="vredsum.row.bf16", args=VectorArgs(vd=27, vs1=23)),
        Instruction(mnemonic="vadd.bf16", args=VectorArgs(vd=26, vs1=26, vs2=27)),
        Instruction(mnemonic="vadd.bf16", args=VectorArgs(vd=3, vs1=25, vs2=26)),
        # m_prev = m_new
        Instruction(mnemonic="vmov", args=VectorArgs(vd=2, vs1=21)),
        # ══════════════════════════════════════════════════════════════════════
        # K TILE 1  (k_seq 32:64) — identical body, different KT/VT tiles
        # ══════════════════════════════════════════════════════════════════════
        Instruction(mnemonic="vload", args=VectorArgs(vd=9, rs1=3, imm12=0)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=10, rs1=3, imm12=32)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=11, rs1=3, imm12=64)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=12, rs1=3, imm12=96)),
        Instruction(mnemonic="vmatpush.acc.bf16.mxu0", args=VectorArgs(vd=1, vs1=9)),
        Instruction(mnemonic="vmatpop.fp8.acc.mxu0", args=VectorArgs(vd=9, vs1=1)),
        Instruction(mnemonic="vmatpush.acc.bf16.mxu0", args=VectorArgs(vd=1, vs1=11)),
        Instruction(mnemonic="vmatpop.fp8.acc.mxu0", args=VectorArgs(vd=11, vs1=1)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=12, rs1=5, imm12=0)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=13, rs1=5, imm12=32)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=14, rs1=5, imm12=64)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=15, rs1=5, imm12=96)),
        Instruction(mnemonic="vmatpush.acc.bf16.mxu0", args=VectorArgs(vd=1, vs1=12)),
        Instruction(mnemonic="vmatpop.fp8.acc.mxu0", args=VectorArgs(vd=12, vs1=1)),
        Instruction(mnemonic="vmatpush.acc.bf16.mxu0", args=VectorArgs(vd=1, vs1=14)),
        Instruction(mnemonic="vmatpop.fp8.acc.mxu0", args=VectorArgs(vd=14, vs1=1)),
        Instruction(mnemonic="vmatpush.weight.mxu0", args=VectorArgs(vd=0, vs1=9)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=17)),
        Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=0, vs2=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=64)),  # ≥ MT=64 cycles
        Instruction(mnemonic="vmatpush.weight.mxu0", args=VectorArgs(vd=0, vs1=11)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=17)),
        Instruction(mnemonic="vmatmul.acc.mxu0", args=MatrixArgs(vd=0, vs1=1, vs2=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=64)),  # ≥ MT=64 cycles
        Instruction(mnemonic="vmatpop.bf16.acc.mxu0", args=VectorArgs(vd=16, vs1=0)),
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=18, vs1=16, vs2=8)),
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=19, vs1=17, vs2=8)),
        Instruction(mnemonic="vredmax.row.bf16", args=VectorArgs(vd=20, vs1=18)),
        Instruction(mnemonic="vredmax.row.bf16", args=VectorArgs(vd=21, vs1=19)),
        Instruction(mnemonic="vmaximum.bf16", args=VectorArgs(vd=20, vs1=20, vs2=21)),
        Instruction(mnemonic="vmaximum.bf16", args=VectorArgs(vd=21, vs1=2, vs2=20)),
        Instruction(mnemonic="vsub.bf16", args=VectorArgs(vd=20, vs1=2, vs2=21)),
        Instruction(mnemonic="vexp.bf16", args=VectorArgs(vd=20, vs1=20)),
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=4, vs1=4, vs2=20)),
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=5, vs1=5, vs2=20)),
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=6, vs1=6, vs2=20)),
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=7, vs1=7, vs2=20)),
        Instruction(mnemonic="vsub.bf16", args=VectorArgs(vd=22, vs1=18, vs2=21)),
        Instruction(mnemonic="vsub.bf16", args=VectorArgs(vd=23, vs1=19, vs2=21)),
        Instruction(mnemonic="vexp.bf16", args=VectorArgs(vd=22, vs1=22)),
        Instruction(mnemonic="vexp.bf16", args=VectorArgs(vd=23, vs1=23)),
        Instruction(mnemonic="vmatpush.acc.bf16.mxu0", args=VectorArgs(vd=1, vs1=22)),
        Instruction(mnemonic="vmatpop.fp8.acc.mxu0", args=VectorArgs(vd=24, vs1=1)),
        Instruction(mnemonic="vmatpush.weight.mxu0", args=VectorArgs(vd=0, vs1=12)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=17)),
        Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=24, vs2=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=64)),  # ≥ MT=64 cycles
        Instruction(mnemonic="vmatpop.bf16.acc.mxu0", args=VectorArgs(vd=25, vs1=0)),
        Instruction(mnemonic="vmatpush.weight.mxu0", args=VectorArgs(vd=0, vs1=14)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=17)),
        Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=1, vs1=24, vs2=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=64)),  # ≥ MT=64 cycles
        Instruction(mnemonic="vmatpop.bf16.acc.mxu0", args=VectorArgs(vd=27, vs1=1)),
        Instruction(mnemonic="vadd.bf16", args=VectorArgs(vd=4, vs1=4, vs2=25)),
        Instruction(mnemonic="vadd.bf16", args=VectorArgs(vd=5, vs1=5, vs2=26)),
        Instruction(mnemonic="vadd.bf16", args=VectorArgs(vd=6, vs1=6, vs2=27)),
        Instruction(mnemonic="vadd.bf16", args=VectorArgs(vd=7, vs1=7, vs2=28)),
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=25, vs1=20, vs2=3)),
        Instruction(mnemonic="vredsum.row.bf16", args=VectorArgs(vd=26, vs1=22)),
        Instruction(mnemonic="vredsum.row.bf16", args=VectorArgs(vd=27, vs1=23)),
        Instruction(mnemonic="vadd.bf16", args=VectorArgs(vd=26, vs1=26, vs2=27)),
        Instruction(mnemonic="vadd.bf16", args=VectorArgs(vd=3, vs1=25, vs2=26)),
        Instruction(mnemonic="vmov", args=VectorArgs(vd=2, vs1=21)),
        # ── normalize: O /= l ─────────────────────────────────────────────────
        Instruction(mnemonic="vrecip.bf16", args=VectorArgs(vd=8, vs1=3)),  # reuse v8 (scale no longer needed)
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=4, vs1=4, vs2=8)),
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=5, vs1=5, vs2=8)),
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=6, vs1=6, vs2=8)),
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=7, vs1=7, vs2=8)),
        # ── store: MRF → VMEM (4 blocks) → DRAM ──────────────────────────────
        Instruction(mnemonic="vstore", args=VectorArgs(vd=4, rs1=7, imm12=0)),
        Instruction(mnemonic="vstore", args=VectorArgs(vd=5, rs1=7, imm12=32)),
        Instruction(mnemonic="vstore", args=VectorArgs(vd=6, rs1=7, imm12=64)),
        Instruction(mnemonic="vstore", args=VectorArgs(vd=7, rs1=7, imm12=96)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=20)),
        Instruction(mnemonic="dma.store.ch<N>", args=DmaArgs(rd=14, rs1=7, rs2=15, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
    ]

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (DRAM_Q, Q_DATA),
        (DRAM_KT0, KT0_DATA),
        (DRAM_KT1, KT1_DATA),
        (DRAM_VT0, VT0_DATA),
        (DRAM_VT1, VT1_DATA),
        (DRAM_SCALE, SCALE_DATA),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUT,
        EXPECTED,  # [128, 16] bf16 — column-blocked, matches DRAM output layout
    )
