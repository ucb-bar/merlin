"""Reproduce the PyTorch .default scalar-scale capture mismatch with the pinned CPU wheel."""

import json

import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401 -- registers the decomposed operations


def main():
    assert torch.__version__ == "2.10.0+cpu", torch.__version__
    original_scale = 0.0023796888063332034
    rounded_scale = float(torch.tensor(original_scale, dtype=torch.float32).item())
    x = torch.tensor([0.11779459565877914], dtype=torch.float32)
    quantize = torch.ops.quantized_decomposed.quantize_per_tensor.default
    direct = quantize(x, original_scale, 0, -128, 127, torch.int8)
    materialized = quantize(x, rounded_scale, 0, -128, 127, torch.int8)
    try:
        quantize(x, 0.0, 0, -128, 127, torch.int8)
    except ZeroDivisionError as exc:
        zero_scale = type(exc).__name__
    else:
        raise AssertionError("PyTorch changed its zero-scale behavior")
    result = {
        "torch": torch.__version__,
        "source_overload": "quantized_decomposed.quantize_per_tensor.default",
        "original_python_scale": original_scale,
        "materialized_f32_scale": rounded_scale,
        "x_f32": float(x.item()),
        "source_default_result": int(direct.item()),
        "f32_materialized_result": int(materialized.item()),
        "zero_scale_source_behavior": zero_scale,
    }
    assert result["source_default_result"] == 50
    assert result["f32_materialized_result"] == 49
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
