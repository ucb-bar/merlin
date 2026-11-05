"""Capture genuine PyTorch graphs and separate float32 reference outputs."""

import argparse
import json
from pathlib import Path
import sys

import torch
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python"))
from atlas_mlir.capture import capture
from kernels import (
    CONFIG,
    MODEL_ID,
    RADIANCE_PR,
    RADIANCE_REVISION,
    AttentionTile,
    SwiGLUTile,
    Projection,
    RMSNormQKV,
    SwiGLUResidual,
    attention_inputs,
    synthetic_tensor,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--panel", choices=("positive", "signed"), default="positive")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    signed = args.panel == "signed"
    x = synthetic_tensor((32, CONFIG.hidden), signed=signed)
    # Synthetic tensors exercise model shapes; no pretrained-quality claim.
    cases = [
        (
            "attention",
            AttentionTile(),
            attention_inputs(signed=signed),
            "graph",
            "Unmasked 32-token cache window, one Qwen head, full d=128; eight matmuls and softmax",
        ),
        (
            "swiglu",
            SwiGLUTile(),
            attention_inputs(signed=signed)[:4],
            "graph",
            "32-channel interior FFN partition; three matmuls, SiLU and gating",
        ),
        (
            "projection",
            Projection(),
            (x, synthetic_tensor((CONFIG.hidden, CONFIG.hidden), signed=signed)),
            "projection",
            "Full K=1536 Q/O projection; first 32-column output tile",
        ),
        (
            "norm_qkv",
            RMSNormQKV(),
            (x,),
            "coverage",
            "Full-width RMSNorm plus biased Q/K/V projections",
        ),
        (
            "full_mlp",
            SwiGLUResidual(),
            (x,),
            "coverage",
            "Full-width RMSNorm, 8960-channel SwiGLU, down projection and residual",
        ),
    ]
    for k in (128, 256, 512):
        generator = torch.Generator().manual_seed(0)
        operands = tuple(
            synthetic_tensor((32, k), generator=generator, signed=signed)
            for _ in range(2)
        )
        cases.append(
            (
                f"projection_k{k}",
                Projection(),
                operands,
                "projection",
                f"K={k} reduction partition of a Qwen projection; 32x32 output",
            )
        )
    rows = []
    for name, model, inputs, route, description in cases:
        destination = args.out / name
        model.eval()
        with torch.no_grad():
            outputs = model(*inputs)
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        capture(model, inputs, destination / "capture")
        save_file(
            {str(i): value.detach().numpy() for i, value in enumerate(outputs)},
            str(destination / "golden.safetensors"),
        )
        row = dict(
            name=name,
            panel=args.panel,
            route=route,
            description=description,
            input_shapes=[list(t.shape) for t in inputs],
            output_shapes=[list(t.shape) for t in outputs],
        )
        (destination / "workload.json").write_text(json.dumps(row, indent=2) + "\n")
        rows.append(row)
        print(f"captured {name}", flush=True)
    (args.out / "workloads.json").write_text(
        json.dumps(
            dict(
                schema="atlas.qwen.workloads.v1",
                model=MODEL_ID,
                config=vars(CONFIG),
                radiance_pr=RADIANCE_PR,
                radiance_revision=RADIANCE_REVISION,
                tensor_source="seeded synthetic float32; not pretrained weights",
                seed=0,
                torch_version=torch.__version__,
                workloads=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
