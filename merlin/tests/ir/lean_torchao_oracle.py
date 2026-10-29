"""Versioned CPU TorchAO oracle, independent of Merlin and its numerical reference helpers."""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch


def tensor(bits, shape):
    return torch.from_numpy(np.array(bits, dtype=np.uint32).view(np.float32).copy()).reshape(shape)


def words(value):
    return [str(int(v)) for v in value.detach().cpu().contiguous().numpy().view(np.uint32).flat]


def require_defined_quantization(primitive, value, scale, zero):
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("affine reference requires a finite positive scale")
    precast = primitive._quantize_affine_no_dtype_cast(value, list(value.shape), scale, zero, -128, 127)
    if not torch.isfinite(precast).all():
        raise ValueError("affine reference refuses undefined float-to-byte observations")


def contraction(case, primitive, affine, plain_layout, int_scaled_matmul):
    m, k, n = case["shape"]
    if min(m, k, n) <= 0 or k * 16384 >= 2**31:
        raise ValueError("contraction reference requires nonempty shapes inside the i32 bound")
    left, right = tensor(case["left_bits"], (m, k)), tensor(case["right_bits"], (k, n))
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("contraction reference requires finite inputs")
    settings = dict(quant_min=-127, quant_max=127, eps=1e-5)
    left_q = affine.to_affine_quantized_intx(left, primitive.MappingType.SYMMETRIC, (1, k), torch.int8, **settings)
    if case["kind"] == "dynamic_contraction":
        right_q = affine.to_affine_quantized_intx(
            right.t().contiguous(),
            primitive.MappingType.SYMMETRIC,
            (1, k),
            torch.int8,
            zero_point_domain=primitive.ZeroPointDomain.NONE,
            **settings,
        )
        # TorchAO's default linear dispatch requires full-range weight metadata. The
        # declared two-reduced-range-operand recipe instead calls its scaled integer
        # matmul primitive explicitly, in the same ordered epilogue as plain layout.
        result = (
            int_scaled_matmul(
                left_q.tensor_impl.int_data,
                right_q.tensor_impl.int_data.contiguous().t(),
                left_q.tensor_impl.scale.reshape(m, 1),
            )
            * right_q.tensor_impl.scale
        )
        route = "configured_quantization_and_scaled_integer_matmul"
    else:
        scale = tensor([case["weight_scale_bits"]], ())
        zero = torch.tensor(0, dtype=torch.int64)
        require_defined_quantization(primitive, right.t().contiguous(), scale, zero)
        right_q = affine.to_affine_quantized_intx_static(
            right.t().contiguous(),
            scale,
            zero,
            (n, k),
            torch.int8,
            quant_min=-128,
            quant_max=127,
        )
        if not plain_layout._linear_int8_act_int8_weight_check(left_q, right_q, None):
            raise RuntimeError("TorchAO integer-linear dispatch refused calibrated operands")
        result = torch.nn.functional.linear(left_q, right_q)
        route = "calibrated_torchao_integer_linear_dispatch"
    return {
        "route": route,
        "linear_dispatch_accepted": plain_layout._linear_int8_act_int8_weight_check(left_q, right_q, None),
        "output_bits": words(result),
        "left_scale_bits": words(left_q.tensor_impl.scale),
        "right_scale_bits": words(right_q.tensor_impl.scale),
        "left_bytes": left_q.tensor_impl.int_data.reshape(-1).tolist(),
        "right_bytes": right_q.tensor_impl.int_data.t().contiguous().reshape(-1).tolist(),
    }


def main():
    version = sys.argv[1]
    if version not in {"0.15.0", "0.16.0"} or torch.__version__ != "2.10.0+cpu" or np.__version__ != "2.4.6":
        raise RuntimeError("reference requires TorchAO 0.15.0/0.16.0, torch 2.10.0+cpu and numpy 2.4.6")
    initialization = io.StringIO()
    with contextlib.redirect_stdout(initialization):
        import torchao
        from torchao.dtypes import affine_quantized_tensor as affine
        from torchao.dtypes.uintx import plain_layout
        from torchao.kernel import intmm
        from torchao.quantization import quant_primitives as primitive
        from torchao.quantization.quant_api import Int8DynamicActivationInt8WeightConfig

    if torchao.__version__ != version:
        raise RuntimeError(f"unexpected TorchAO version: {torchao.__version__}")
    torch.set_num_threads(1)
    results = []
    for case in json.loads(sys.stdin.read()):
        if case["kind"] in {"dynamic_contraction", "calibrated_reuse"}:
            results.append(contraction(case, primitive, affine, plain_layout, intmm.int_scaled_matmul))
        elif case["kind"] == "affine_per_tensor":
            x = tensor(case["input_bits"], case["shape"])
            scale = tensor([case["scale_bits"]], ())
            if case["zero_width"] not in (32, 64):
                raise ValueError("affine reference requires i32/i64 zero points")
            zero = torch.tensor(case["zero"], dtype=torch.int32 if case["zero_width"] == 32 else torch.int64)
            require_defined_quantization(primitive, x, scale, zero)
            quantized = primitive.quantize_affine(x, tuple(x.shape), scale, zero, torch.int8, -128, 127)
            dequantized = primitive.dequantize_affine(
                quantized, tuple(x.shape), scale, zero, torch.int8, -128, 127, output_dtype=torch.float32
            )
            results.append(
                {
                    "quantized_bytes": [str(int(v) & 255) for v in quantized.reshape(-1).tolist()],
                    "dequantized_bits": words(dequantized),
                }
            )
        else:
            raise ValueError("unknown TorchAO reference operation")
    sources = []
    for module in (primitive, plain_layout, intmm, affine, inspect.getmodule(Int8DynamicActivationInt8WeightConfig)):
        path = Path(inspect.getsourcefile(module))
        sources.append(
            {"module": module.__name__, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    print(
        json.dumps(
            {
                "scope": "versioned_cpu_torchao_fidelity_only",
                "whole_pass_verified": False,
                "torch": torch.__version__,
                "torchao": torchao.__version__,
                "numpy": np.__version__,
                "default_dynamic_config_version": Int8DynamicActivationInt8WeightConfig(
                    set_inductor_config=False
                ).version,
                "gpu_extensions_used": False,
                "initialization_stdout": initialization.getvalue(),
                "sources": sources,
                "results": results,
            }
        )
    )


if __name__ == "__main__":
    main()
