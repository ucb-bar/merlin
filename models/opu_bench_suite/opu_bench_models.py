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
    """4-layer MLP with large hidden dims for pure GEMM throughput measurement."""

    def __init__(self, input_dim=128, hidden_dim=512, output_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Export functions
# =============================================================================

MODELS = {
    "vit": {
        "class": ViTModel,
        # ViT-like: dim=512, 8 heads of 64 each, seq=196 (14×14 patches).
        # QKV: M=196 K=512 N=512 → large GEMM. FFN: K=2048.
        # 8 heads (not 12) ensures batch_mmt4d tile compatibility with RVV.
        "kwargs": {"dim": 512, "num_heads": 8, "seq_len": 196, "num_blocks": 2},
        "input_shape": (1, 196, 512),  # [batch, 14×14 patches, dim=512]
        "input_names": ["input"],
        "output_names": ["output"],
    },
    "vit_small": {
        "class": ViTModel,
        # Small ViT for fast FireSim iteration: dim=128, 4 heads, seq=64.
        # QKV: M=64 K=128. FFN: M=64 K=512. All dims multiples of 16.
        # Finishes init+inference in ~2-3 minutes on FireSim.
        "kwargs": {"dim": 128, "num_heads": 4, "seq_len": 64, "num_blocks": 2},
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
        # dim=128 → K=128 for transformer matmuls. Conv stem 3→64→128.
        # 16×16 spatial → 256 tokens with K=128.
        "kwargs": {"dim": 128, "num_heads": 4, "num_blocks": 2},
        "input_shape": (1, 3, 64, 64),  # [batch, channels, H, W]
        "input_names": ["input"],
        "output_names": ["output"],
    },
    "large_mlp": {
        "class": LargeMLP,
        # Transformer FFN scale: hidden=2048, batch=128 → M=128.
        # fc1: [128,512]×[512,2048] K=512, N=2048
        # fc2/3: [128,2048]×[2048,2048] K=2048, N=2048 — the OPU dream case
        "kwargs": {"input_dim": 512, "hidden_dim": 2048, "output_dim": 128},
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
