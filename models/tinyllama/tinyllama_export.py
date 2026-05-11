#!/usr/bin/env python3
"""Export TinyLlama to ONNX for OPU benchmarking.

Uses the `optimum` library for reliable HuggingFace→ONNX export with
proper handling of KV cache, attention masks, and rotary embeddings.

Usage:
    conda run -n merlin-dev uv run python models/tinyllama/tinyllama_export.py
"""

import argparse
import os


def export_with_optimum(model_id, output_dir, seq_len=128):
    """Export using optimum-cli which handles all the tricky bits."""
    from optimum.exporters.onnx import main_export

    print(f"Exporting {model_id} to ONNX (seq_len={seq_len})...")
    print(f"Output directory: {output_dir}")

    # Export with static shapes for benchmarking
    main_export(
        model_name_or_path=model_id,
        output=output_dir,
        task="text-generation",
        opset=18,
        no_dynamic_axes=True,
        batch_size=1,
        sequence_length=seq_len,
    )
    print(f"Export complete. Files in {output_dir}")


def export_with_torch(model_id, output_path, seq_len=128):
    """Fallback: export directly with torch.onnx.export."""
    import torch
    from transformers import AutoModelForCausalLM

    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        use_cache=False,  # Disable KV cache for simpler export
    )
    model.eval()

    # Create dummy input
    dummy_input_ids = torch.ones(1, seq_len, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, seq_len, dtype=torch.long)

    print(f"Exporting to {output_path}...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            opset_version=18,
        )
    print(f"Exported to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export TinyLlama to ONNX")
    parser.add_argument(
        "--model-id",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length (default: 128)",
    )
    parser.add_argument(
        "--method",
        choices=["optimum", "torch"],
        default="torch",
        help="Export method (default: torch for simplicity)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "tinyllama.onnx")

    if args.method == "optimum":
        export_with_optimum(args.model_id, script_dir, args.seq_len)
    else:
        export_with_torch(args.model_id, output_path, args.seq_len)
