"""Pinned captured-source quantization oracle, isolated from Merlin and its pass implementation."""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401 -- registers the source operation


def main():
    if torch.__version__ != "2.10.0+cpu" or np.__version__ != "2.4.6":
        raise RuntimeError("source oracle requires torch 2.10.0+cpu and numpy 2.4.6")
    torch.set_num_threads(1)
    answers = []
    for case in json.loads(sys.stdin.read()):
        x = torch.from_numpy(np.array(case["input_bits"], dtype=np.uint32).view(np.float32).copy())
        s = torch.from_numpy(np.array([case["scale_bits"]], dtype=np.uint32).view(np.float32).copy()).reshape(())
        width = case["zero_width"]
        if width not in (32, 64):
            raise ValueError("source oracle requires i32 or i64 zero points")
        z = torch.tensor(case["zero"], dtype=torch.int32 if width == 32 else torch.int64)
        # Exceptional conversion behavior is not a portable expected answer. Refuse before
        # running the source cast, independently of the Lean candidate's own classification.
        shifted = torch.round(x * (1.0 / s.item())) + z.item()
        if not torch.isfinite(torch.clamp(shifted, -128, 127)).all().item():
            raise ValueError("source oracle refuses undefined float-to-byte observations")
        result = torch.ops.quantized_decomposed.quantize_per_tensor.tensor(
            x.reshape(case["shape"]), s, z, -128, 127, torch.int8
        )
        answers.append([int(v) & 0xFF for v in result.reshape(-1).tolist()])
    print(json.dumps({"torch": torch.__version__, "numpy": np.__version__, "results": answers}))


if __name__ == "__main__":
    main()
