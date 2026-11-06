"""Bitwise gate for the pinned Torch 2.16 AVX2 f32 width-256 LayerNorm probe.

The shared library is built from the neighboring C++ file into the caller's
managed-task .tmp directory. The trained and holdout archives are existing
same-input Merlin host taps; this script never captures or compiles the model.
"""

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SOURCE_COMMIT = "7f24a19c0967db04f2d5ba44c1e56eca489932ec"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--trained-tap", required=True, type=Path)
    parser.add_argument("--holdout-tap", required=True, type=Path)
    parser.add_argument("--model2mlir", required=True, type=Path)
    parser.add_argument("--spectformer-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()
    if torch.version.git_version != SOURCE_COMMIT:
        raise RuntimeError("probe requires exact captured Torch source commit")
    if torch.backends.cpu.get_cpu_capability() != "AVX2":
        raise RuntimeError("probe requires source AVX2 dispatch")

    os.environ["SPECTFORMER_DIR"] = str(args.spectformer_dir)
    os.environ["SPECTFORMER_CKPT"] = str(args.checkpoint)
    sys.path[:0] = [str(args.model2mlir), str(args.model2mlir / "workloads/spectformer")]
    from loader import get_model_and_inputs
    from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization

    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    model, inputs = get_model_and_inputs()
    model = apply_quantization(
        model.eval(),
        QuantizationConfig(scheme="int8_dyn_act_int8_weight"),
        example_inputs=tuple(inputs),
    )
    norm = model.blocks[0].norm2
    gamma = np.ascontiguousarray(norm.weight.detach().numpy())
    beta = np.ascontiguousarray(norm.bias.detach().numpy())
    if gamma.shape != (256,) or beta.shape != (256,) or gamma.dtype != np.float32 or beta.dtype != np.float32:
        raise RuntimeError("probe requires f32 width-256 affine parameters")

    ptr = ctypes.POINTER(ctypes.c_float)
    library = ctypes.CDLL(str(args.library.resolve()))
    kernel = library.torch216_avx2_ln256
    kernel.argtypes = [ptr, ctypes.c_int64, ptr, ptr, ctypes.c_float, ptr, ptr, ptr]
    checks = []
    cases = (
        ("trained", np.load(args.trained_tap)["view_2"]),
        ("holdout", np.load(args.holdout_tap)["view_2"]),
        ("random", np.random.default_rng(12).standard_normal((1, 196, 256)).astype(np.float32)),
    )
    for name, values in cases:
        x = np.ascontiguousarray(values, dtype=np.float32)
        if x.shape != (1, 196, 256):
            raise RuntimeError(f"{name}: expected shape [1,196,256], got {x.shape}")
        with torch.no_grad():
            expected, expected_mean, expected_rstd = torch.ops.aten.native_layer_norm.default(
                torch.from_numpy(x), [256], norm.weight, norm.bias, norm.eps
            )
        mean = np.empty(196, dtype=np.float32)
        rstd = np.empty_like(mean)
        actual = np.empty_like(x)
        kernel(
            x.ctypes.data_as(ptr),
            196,
            gamma.ctypes.data_as(ptr),
            beta.ctypes.data_as(ptr),
            norm.eps,
            mean.ctypes.data_as(ptr),
            rstd.ctypes.data_as(ptr),
            actual.ctypes.data_as(ptr),
        )
        expected_arrays = (expected_mean.numpy().reshape(-1), expected_rstd.numpy().reshape(-1), expected.numpy())
        actual_arrays = (mean, rstd, actual)
        counts = {
            label: int(np.count_nonzero(got.view(np.uint32) == want.view(np.uint32)))
            for label, got, want in zip(("mean", "rstd", "output"), actual_arrays, expected_arrays)
        }
        checks.append({"case": name, "bitwise_equal": counts, "row_count": 196, "output_elements": int(x.size)})
        if counts != {"mean": 196, "rstd": 196, "output": int(x.size)}:
            raise AssertionError(f"{name}: native LayerNorm bitwise mismatch: {counts}")
    print(json.dumps({"torch_commit": SOURCE_COMMIT, "cpu": "AVX2", "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
