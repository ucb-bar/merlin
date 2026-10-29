"""Run the pinned PyTorch source operations in an isolated CPU interpreter (no Merlin imports)."""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401 -- registers the source operations


def main():
    if torch.__version__ != "2.10.0+cpu" or np.__version__ != "2.4.6":
        raise RuntimeError("source oracle requires torch 2.10.0+cpu and numpy 2.4.6")
    torch.set_num_threads(1)
    answers = []
    for case in json.loads(sys.stdin.read()):
        channel = case["kind"] == "per_channel"
        shape, axis = case["shape"], case["axis"]
        parameter_shape = [shape[axis]] if channel else []
        q = torch.tensor(case["q"], dtype=torch.int8).reshape(shape)
        zero_width = case.get("zero_width", 32)
        if zero_width not in (32, 64):
            raise ValueError("source oracle requires i32 or i64 zero points")
        z = torch.tensor(case["z"], dtype=torch.int32 if zero_width == 32 else torch.int64).reshape(parameter_shape)
        scales = np.array(case["scale_bits"], dtype=np.uint32).view(np.float32).copy()
        s = torch.from_numpy(scales).reshape(parameter_shape)
        if channel:
            result = torch.ops.quantized_decomposed.dequantize_per_channel.default(
                q, s, z, axis, -128, 127, torch.int8, out_dtype=torch.float32
            )
        else:
            result = torch.ops.quantized_decomposed.dequantize_per_tensor.tensor(
                q, s, z, -128, 127, torch.int8, out_dtype=torch.float32
            )
        answers.append([int(x) & 0xFFFFFFFF for x in result.contiguous().view(torch.int32).reshape(-1).tolist()])
    print(json.dumps({"torch": torch.__version__, "numpy": np.__version__, "results": answers}))


if __name__ == "__main__":
    main()
