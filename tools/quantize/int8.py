#!/usr/bin/env python3
"""Generic INT8 quantization for any ONNX model.

Wraps onnxruntime.quantization.quantize_static with the exact settings
the merlin compile path expects: QDQ format, INT8 activations + weights,
symmetric quantization. The CLI in `tools/quantize/cli.py` exposes two
front-ends over this routine: direct (`<input.onnx> --shape …`) and
registry (`--model <name>` resolved against `models/models_config.json`).

Usage:
    ./merlin quantize <input.onnx> --shape 1,3,224,224
    ./merlin quantize <input.onnx> --shape 1,3,224,224 --output out.q.int8.onnx
    ./merlin quantize <input.onnx> --shape 1,3,224,224 --calibration-samples 100
    ./merlin quantize <input.onnx> --shape 1,3,224,224 1,10  # multi-input model

The output `.q.int8.onnx` can be fed directly to `./merlin compile` for
any merlin target (firesim_shuttle_gemmini, firesim_shuttle, qrb5165,
spacemit_x60, etc.).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)


class _RandomCalibrationDataReader(CalibrationDataReader):
    """Random fp32 data of the user-provided shape(s).

    For production accuracy you should swap this for a real calibration
    set (image dataset slice, recorded sensor traces, etc.). Random
    data is sufficient for getting a usable quantized model whose
    per-tensor scale/zero-point parameters cover the activation range
    seen during calibration. For deterministic int8 arithmetic on
    deterministic input the choice of calibration data does not affect
    correctness, only the activation scale factors.
    """

    def __init__(self, input_names: list[str], input_shapes: list[list[int]], n_samples: int = 50):
        self.input_names = input_names
        self.input_shapes = input_shapes
        self.n_samples = n_samples
        self._gen = self._make_gen()

    def _make_gen(self):
        rng = np.random.default_rng(seed=0xC0FFEE)
        for _ in range(self.n_samples):
            yield {
                name: rng.standard_normal(tuple(shape)).astype(np.float32)
                for name, shape in zip(self.input_names, self.input_shapes)
            }

    def get_next(self):
        return next(self._gen, None)

    def rewind(self):
        self._gen = self._make_gen()


def _parse_shape(shape_str: str) -> list[int]:
    return [int(x) for x in shape_str.split(",")]


def quantize(
    src_onnx: Path,
    dst_onnx: Path,
    input_shapes: list[list[int]],
    n_calibration_samples: int = 50,
) -> None:
    model = onnx.load(str(src_onnx))
    initializers = {init.name for init in model.graph.initializer}
    input_names = [n.name for n in model.graph.input if n.name not in initializers]

    if len(input_names) != len(input_shapes):
        raise SystemExit(
            f"model has {len(input_names)} inputs ({input_names}) "
            f"but you passed {len(input_shapes)} --shape arguments"
        )

    reader = _RandomCalibrationDataReader(
        input_names=input_names,
        input_shapes=input_shapes,
        n_samples=n_calibration_samples,
    )

    quantize_static(
        model_input=str(src_onnx),
        model_output=str(dst_onnx),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        # Symmetric quantization — what every merlin backend (Gemmini,
        # OPU, RVV ukernels, SpacemiT XSMT) expects. The compile path
        # for these targets canonicalizes QDQ to int8 matmul + scale
        # + saturate; non-symmetric activation quantization would
        # break the scale-and-saturate fold in
        # iree-global-opt-quantized-conv-to-conv and downstream
        # plugins that depend on zero_point == 0.
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_onnx", type=Path, help="Source .onnx file")
    parser.add_argument(
        "--shape",
        action="append",
        required=True,
        help=(
            "Input tensor shape as comma-separated integers (e.g. 1,3,224,224). "
            "Repeat once per input for multi-input models."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <input>.q.int8.onnx alongside the input).",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=50,
        help="Number of random calibration samples (default 50).",
    )
    args = parser.parse_args()

    if not args.input_onnx.exists():
        raise SystemExit(f"input not found: {args.input_onnx}")

    output = args.output or args.input_onnx.with_suffix(".q.int8.onnx")
    shapes = [_parse_shape(s) for s in args.shape]

    print(f"==> quantize: {args.input_onnx}")
    print(f"    output:  {output}")
    print(f"    shapes:  {shapes}")
    print(f"    samples: {args.calibration_samples}")

    quantize(args.input_onnx, output, shapes, args.calibration_samples)

    print(f"==> done: {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
