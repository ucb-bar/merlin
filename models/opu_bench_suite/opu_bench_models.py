"""OPU Benchmark Model Suite — Realistic architectures for Saturn OPU evaluation.

Four models designed to stress-test the OPU 16×16 outer-product accelerator
while using real architectural patterns from production neural networks.
All dimensions are multiples of 16 for clean OPU tile alignment.

Models:
  1. vit_block   — Vision Transformer encoder block (attention + FFN)
  2. convnet     — CNN backbone (3×3 + 1×1 convolutions, ResNet-style)
  3. hybrid      — Conv stem → Transformer blocks (MobileViT-style)
  4. large_mlp   — Dense GEMM stress test (large hidden dims)

Usage:
    python opu_bench_models.py --all           # Export all 4
    python opu_bench_models.py --model vit     # Export one
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 1. ViT Encoder Block — Multi-head Self-Attention + FFN
# =============================================================================
# Representative of: ViT, DeiT, DiT, LLM attention, SAM encoder
#
# Key OPU matmuls:
#   Q/K/V projections: [B, S, D] × [D, D] → [B, S, D]     (S=64, D=128)
#   Attention:         [B, H, S, S] (via Q @ K^T)            (H=8, S=64)
#   O projection:      [B, S, D] × [D, D] → [B, S, D]
#   FFN up:            [B, S, D] × [D, 4D] → [B, S, 4D]    (expansion=4)
#   FFN down:          [B, S, 4D] × [4D, D] → [B, S, D]
#
# Total per block: ~5 large matmuls with M=64, K/N ∈ {128, 512}


class ViTBlock(nn.Module):
    def __init__(self, dim=128, num_heads=8, seq_len=64, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        # Self-attention with residual
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        # FFN with residual
        x = x + self.mlp(self.norm2(x))
        return x


class ViTBlockExplicit(nn.Module):
    """ViT block with EXPLICIT Q/K/V linear layers (instead of
    nn.MultiheadAttention's combined in_proj_weight). This exports to
    ONNX as three separate Gemm nodes, avoiding the B=3 batch_matmul
    pattern that the IREE encoding resolver does not accelerate. Each
    QKV projection becomes a clean 2D matmul → OPU 32×32 tile."""

    def __init__(self, dim=256, num_heads=4, seq_len=256, mlp_ratio=4):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.norm1 = nn.LayerNorm(dim)
        # Separate Q / K / V projections (3 × 2D matmul, not 1 × batch_matmul).
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        B, S, C = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, C)
        x = x + self.o_proj(out)
        x = x + self.mlp(self.norm2(x))
        return x


class ViTModel(nn.Module):
    """2-block ViT encoder (enough to show OPU utilization, small enough to run)."""

    def __init__(self, dim=128, num_heads=8, seq_len=64, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([ViTBlock(dim, num_heads, seq_len) for _ in range(num_blocks)])
        self.head = nn.Linear(dim, 16)  # Classification head

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, 0])  # CLS token


class ViTAllTokens(nn.Module):
    """ViT with conv stem + transformer blocks + all-token head.

    The conv stem (3→64→dim, two 3×3 stride-2 convs) produces "direct
    conv" dispatches in the decomposition, similar to DroNet. The
    transformer blocks produce OPU 32×32 matmul dispatches. The head
    Linear applied to ALL tokens (not CLS-only) produces an encoding
    16×16 dispatch (N=output_dim < 32). This gives a rich mixed
    decomposition: direct conv + OPU 32×32 + encoding 16×16 +
    reduction/softmax + elementwise.

    Input: spatial image [B, 3, H, H] where H = sqrt(seq_len) * 4.
    Conv stem: 3→64→dim with two stride-2 convs reduces spatial to
    sqrt(seq_len) × sqrt(seq_len), then reshaped to [B, seq_len, dim].
    """

    def __init__(self, dim=1024, num_heads=8, seq_len=128, num_blocks=12, output_dim=16):
        super().__init__()
        # No conv stem — input is already a sequence [B, S, dim]. Pure
        # transformer maximizes matmul fraction of total compute, which is
        # what TinyLlama's 10.97× speedup comes from. Earlier ViT variants
        # had a conv stem + BN + LayerNorm + softmax + quant chain that
        # diluted OPU coverage (ViT v2 at dim=512 got only 1.09×).
        self.blocks = nn.ModuleList([ViTBlockExplicit(dim, num_heads, seq_len) for _ in range(num_blocks)])
        self.head = nn.Linear(dim, output_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# =============================================================================
# 2. ConvNet — Stacked 3×3 + 1×1 convolutions (ResNet-style)
# =============================================================================
# Representative of: ResNet, ConvNeXt, YOLO backbone, EfficientNet
#
# Key OPU matmuls (via im2col):
#   3×3 conv:  [H*W, C*9] × [C*9, C_out] → [H*W, C_out]
#   1×1 conv:  [H*W, C] × [C, C_out] → [H*W, C_out]    (pure GEMM)
#
# With C=64, H=W=32: M=1024, K=576 (3×3) or K=64 (1×1), N=64
# All are OPU-friendly shapes with M >> 16.


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.shortcut(x))


class ConvNet(nn.Module):
    """Small ResNet-style CNN: 3 stages × 2 blocks, channels 32→64→128."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.stage1 = nn.Sequential(ConvBlock(32, 32), ConvBlock(32, 32))
        self.stage2 = nn.Sequential(ConvBlock(32, 64, stride=2), ConvBlock(64, 64))
        self.stage3 = nn.Sequential(ConvBlock(64, 128, stride=2), ConvBlock(128, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(128, 16)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


# =============================================================================
# 3. Hybrid CNN + Transformer (MobileViT-style)
# =============================================================================
# Representative of: MobileViT, EfficientFormer, LeViT, FastViT
#
# Conv stem extracts spatial features, then transformer blocks process
# the flattened feature map as a sequence. Best of both worlds:
# - Conv layers → im2col → GEMM (OPU via im2col path)
# - Attention layers → pure GEMM (OPU via encoding path)


class HybridModel(nn.Module):
    """Conv stem (64→dim) → reshape to sequence → transformer blocks → head."""

    def __init__(self, dim=128, num_heads=4, num_blocks=2):
        super().__init__()
        # Conv stem: 3×64×64 → dim×16×16
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        # Transformer blocks on the 16×16=256 spatial tokens
        self.blocks = nn.ModuleList([ViTBlock(dim, num_heads, seq_len=256) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 16)

    def forward(self, x):
        # Conv stem
        x = self.stem(x)  # [B, 64, 16, 16]
        B, C, H, W = x.shape
        # Reshape to sequence: [B, 256, 64]
        x = x.flatten(2).transpose(1, 2)
        # Transformer
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x.mean(dim=1))


# =============================================================================
# 4. Large MLP — Dense GEMM stress test
# =============================================================================
# Representative of: FFN layers in transformers, recommendation models
#
# Key OPU matmuls:
#   fc1: [B, 128] × [128, 512] → [B, 512]   M=64, K=128, N=512
#   fc2: [B, 512] × [512, 512] → [B, 512]   M=64, K=512, N=512
#   fc3: [B, 512] × [512, 128] → [B, 128]   M=64, K=512, N=128
#
# With B=64: each matmul has M=64, K/N ≥ 128 → 4×4 OPU sub-tiling (64×64)


class LargeMLP(nn.Module):
    """Deep MLP sized to keep matmul K small enough to fit L1 on the OPU
    (working-set per 32×32 output tile ≈ K·64 B; K=512 → 32 KB, fits
    comfortably). 6 layers × 128×512×512 ≈ 200 M i8 FMAs total."""

    def __init__(self, input_dim=128, hidden_dim=512, output_dim=16, depth=6):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPFast(nn.Module):
    """MLP designed to maximize OPU matmul dominance. Modeled on the same
    principles as TinyLlama: all matmul shapes are TinyLlama-scale
    (K ≥ 1024), no per-layer non-matmul overhead beyond ReLU, so the
    matmul-% of total compute approaches 99%. Expected speedup > 2×
    (and in practice closer to 5× because the OPU 32×32 tile amortizes
    memory bandwidth well at K=1024).

    Matmul shapes produced:
      Layer 0:  128 × 1024 × 128    (input 128 → hidden 1024)
      Layer i:  128 × 1024 × 1024   (depth-2 copies)
      Last:     128 × 16   × 1024   (hidden 1024 → 16 classes; N=16 < 32
                                     so the encoding resolver picks 16×16)
    All dims are multiples of 32. Compute ≈ depth × 128 × 1024² ≈
    500M MACs for depth=4, dominated by matmul."""

    def __init__(self, input_dim=128, hidden_dim=1024, output_dim=16, depth=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Export functions
# =============================================================================

MODELS = {
    "vit": {
        "class": ViTAllTokens,
        # ViT v3 — pure transformer, no conv stem. dim=1024, 12 blocks.
        # Modeled after TinyLlama's structure (10.97× speedup): large
        # matmul dimensions with minimal non-matmul overhead so OPU
        # dominates total cycles. Combined with OPU_LLM compile flags
        # (collapse-multi-n), all 12 × ~5 matmuls per block become
        # 2D matmuls that hit OPU 32×32 via encoding resolver.
        "kwargs": {"dim": 1024, "num_heads": 8, "seq_len": 128, "num_blocks": 6, "output_dim": 16},
        "input_shape": (1, 128, 1024),  # [batch, tokens, dim]
        "input_names": ["input"],
        "output_names": ["output"],
    },
    "vit_small": {
        "class": ViTModel,
        # Small ViT for fast FireSim iteration: dim=128, 2 heads, seq=64.
        # QKV: M=64 K=128. FFN: M=64 K=512. 2 heads × 64 dim/head keeps
        # per-head attention K=64 so the batch_matmul hits OPU 32×32.
        "kwargs": {"dim": 128, "num_heads": 2, "seq_len": 64, "num_blocks": 2},
        "input_shape": (1, 64, 128),  # [batch, 8×8 patches, dim=128]
        "input_names": ["input"],
        "output_names": ["output"],
    },
    "convnet": {
        "class": ConvNet,
        "kwargs": {},
        "input_shape": (1, 3, 64, 64),  # [batch, channels, H, W]
        "input_names": ["input"],
        "output_names": ["output"],
    },
    "hybrid": {
        "class": HybridModel,
        # dim=128, 2 heads × 64 dim/head. Conv stem 3→64→128.
        # Per-head attention K=64 → lands on OPU 32×32 fast path
        # (was K=32 with 4 heads, forcing narrow-tile fallback).
        "kwargs": {"dim": 128, "num_heads": 2, "num_blocks": 2},
        "input_shape": (1, 3, 64, 64),  # [batch, channels, H, W]
        "input_names": ["input"],
        "output_names": ["output"],
    },
    "mlp_fast": {
        "class": MLPFast,
        # Fast MLP (ViT-style matmul sizes, no attention/softmax):
        # matmul K=1024 puts OPU in the compute-bound regime (same as
        # TinyLlama's transformer blocks). Only ReLU between layers.
        # Target speedup: >2× (expected 3-5× based on MNK scaling).
        "kwargs": {"input_dim": 128, "hidden_dim": 1024, "output_dim": 16, "depth": 4},
        "input_shape": (128, 128),  # [batch=128, features=128]
        "input_names": ["input"],
        "output_names": ["output"],
    },
    "large_mlp": {
        "class": LargeMLP,
        # Deep MLP: hidden_dim=512 keeps all matmul K ≤ 512 so each 32×32
        # OPU output tile's working set (≈32 KB) fits in L1. 6 layers of
        # 128×512×512 gives ~200 M FMAs total — less wall-work than the
        # old 3×(128×2048×2048) but avoids the K=2048 memory-thrash that
        # was bottlenecking OPU to 0.06× RVV.
        "kwargs": {"input_dim": 512, "hidden_dim": 512, "output_dim": 128, "depth": 6},
        "input_shape": (128, 512),  # [batch=128, features=512]
        "input_names": ["input"],
        "output_names": ["output"],
    },
}


def export_model(name, out_dir):
    spec = MODELS[name]
    model = spec["class"](**spec["kwargs"])
    model.eval()

    dummy = torch.randn(*spec["input_shape"])
    onnx_path = os.path.join(out_dir, f"opu_bench_{name}.onnx")

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=spec["input_names"],
        output_names=spec["output_names"],
        opset_version=17,
    )
    print(f"Exported {onnx_path}  input={list(spec['input_shape'])}")
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description="OPU Benchmark Model Suite")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        help="Export a single model",
    )
    parser.add_argument("--all", action="store_true", help="Export all models")
    parser.add_argument(
        "--out-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Output directory",
    )
    args = parser.parse_args()

    if args.all:
        for name in MODELS:
            export_model(name, args.out_dir)
    elif args.model:
        export_model(args.model, args.out_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
