"""Wider MLP variant designed for the Saturn OPU 16x16 hardware tile.

The original `mlp` model has input shape [1, 10] which produces vecmat
matmuls (M=1) that hit the OPU narrow-M code path. That path is fragile
and currently hangs on FireSim.

This variant uses [16, 16] (batch 16 × features 16) so every matmul has
M=16, K and N also multiples of 16:

  fc1: [16, 16] @ [16, 32]^T → [16, 32]   M=16  K=16  N=32
  fc2: [16, 32] @ [32, 32]^T → [16, 32]   M=16  K=32  N=32
  fc3: [16, 32] @ [32, 16]^T → [16, 16]   M=16  K=32  N=16

Every matmul lands directly on the OPU 16x16 hardware tile (PATH A) —
no narrow-M tail, no `tu, ma` vsetvli tricks, no FireSim hang.
"""

import os

import torch
import torch.nn as nn
import torch.onnx


class WideMLP(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32, output_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # Batch 16 × 16-dim input — every matmul stays on the OPU 16x16 tile.
    model = WideMLP(input_dim=16, hidden_dim=32, output_dim=16)
    model.eval()
    dummy_input = torch.randn(16, 16)
    onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlp_wide.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
    print(f"Exported {onnx_path}")
