"""Run the pinned Voyager compiler on one workload and export what it produced.

Runs inside Voyager's own environment (``out/build/voyager-venv``), never merlin's: it imports the
public ``voyager_compiler`` package and nothing from merlin, so the arm is Voyager's stock toolchain.
Everything the merlin side needs crosses as files in ``--out``:

    model.txt          Voyager's protobuf IR, exactly as ``compile()`` wrote it
    model.json         the same message as JSON (field names preserved), for readers without protobuf
    layers.txt         Voyager's own per-layer table
    tensor_files/      the constants and intermediates ``compile(dump_tensors=True)`` writes
    io.pt              raw input, Voyager's preprocessed (quantized) input, and the reference outputs
    manifest.json      compiler revision, library versions, workload, config and quantization flags

The accelerator description comes in as JSON (``--config``) so the caller derives it from the target's
facts; this script never assumes a hardware value of its own.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from google.protobuf import json_format, text_format

# The accelerator repo's INT8_32 codegen flags (codegen.mk): per-tensor symmetric int8 activations
# and weights, int32 bias. The int32 bias matches an int32 accumulator; INT8 (int24 bias) is the
# repo's other int8 scheme.
QUANT_SCHEMES = {
    "int8_int32bias": {
        "input_activation": "int8,qs=per_tensor_symmetric",
        "weight": "int8,qs=per_tensor_symmetric",
        "bias": "int32",
    },
    "int8_int24bias": {
        "input_activation": "int8,qs=per_tensor_symmetric",
        "weight": "int8,qs=per_tensor_symmetric",
        "bias": "int24",
    },
}


def _voyager_root() -> Path:
    return Path(os.environ["MERLIN_EXT_VOYAGER_COMPILER"]).resolve()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()


class _Linear(torch.nn.Module):
    def __init__(self, k: int, n: int, bias: bool, relu: bool):
        super().__init__()
        self.fc = torch.nn.Linear(k, n, bias=bias)
        self.relu = relu

    def forward(self, x):
        y = self.fc(x)
        return torch.relu(y) if self.relu else y


class _Conv(torch.nn.Module):
    def __init__(self, cin: int, cout: int, k: int, stride: int, padding: int, bias: bool, relu: bool):
        super().__init__()
        self.conv = torch.nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=bias)
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        return torch.relu(y) if self.relu else y


def build_workload(spec: dict, dtype: torch.dtype):
    """(module, example input) for a workload spec. Kinds: ``linear``, ``conv``, ``torchvision``."""
    kind = spec["kind"]
    if kind == "linear":
        module = _Linear(spec["K"], spec["N"], spec.get("bias", True), spec.get("relu", False))
        example = torch.randn(spec["M"], spec["K"], dtype=dtype)
    elif kind == "conv":
        module = _Conv(
            spec["Cin"],
            spec["Cout"],
            spec["k"],
            spec.get("stride", 1),
            spec.get("padding", spec["k"] // 2),
            spec.get("bias", True),
            spec.get("relu", True),
        )
        example = torch.randn(1, spec["Cin"], spec["H"], spec["W"], dtype=dtype)
    elif kind == "resblock":
        # One torchvision residual block (BasicBlock, or Bottleneck with "bottleneck": true), with a
        # 1x1 downsample when the shape changes: the smallest graph that carries every ResNet
        # construct (bias, residual epilogue, split K). Batch-norm statistics are randomized so the
        # fused biases are not zero, then folded into the convs as Voyager's harness does.
        from torchvision.models.resnet import BasicBlock, Bottleneck
        from voyager_compiler.quantization.quantize import get_conv_bn_layers

        block = Bottleneck if spec.get("bottleneck") else BasicBlock
        cin, planes, stride = spec["Cin"], spec["planes"], spec.get("stride", 1)
        cout = planes * block.expansion
        downsample = None
        if stride != 1 or cin != cout:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(cin, cout, 1, stride=stride, bias=False), torch.nn.BatchNorm2d(cout)
            )
        module = block(cin, planes, stride=stride, downsample=downsample).eval()
        for bn in (m for m in module.modules() if isinstance(m, torch.nn.BatchNorm2d)):
            bn.weight.data.uniform_(0.5, 1.5)
            bn.bias.data.normal_(0.0, 0.5)
            bn.running_mean.normal_(0.0, 0.5)
            bn.running_var.uniform_(0.5, 1.5)
        module = torch.ao.quantization.fuse_modules(module, get_conv_bn_layers(module), inplace=True)
        example = torch.randn(1, cin, spec["H"], spec["W"], dtype=dtype)
    elif kind == "torchvision":
        from torchvision import models
        from voyager_compiler.quantization.quantize import get_conv_bn_layers

        module = models.__dict__[spec["name"]](weights=spec.get("weights")).eval()
        # Voyager's own torchvision harness folds every conv with its batch norm before export
        # (test/utils/models/torchvision_models.py); without it convs carry no bias and BN stays a
        # separate op, so the graph Voyager compiles is not the one its flow would.
        pairs = get_conv_bn_layers(module)
        if pairs:
            module = torch.ao.quantization.fuse_modules(module, pairs, inplace=True)
        example = torch.randn(*spec.get("input_shape", (1, 3, 224, 224)), dtype=dtype)
    else:
        raise ValueError(f"unknown workload kind {kind!r}")
    return module.eval().to(dtype).requires_grad_(False), example


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workload", required=True, type=Path, help="workload spec JSON")
    parser.add_argument(
        "--config", required=True, type=Path, help="AcceleratorConfig fields as JSON, derived by the caller"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--quant", default="int8_int32bias", choices=sorted(QUANT_SCHEMES))
    parser.add_argument("--calibration-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout-policy", default="systolic")
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the model in bf16, as every accelerator-repo codegen scheme does",
    )
    parser.add_argument(
        "--run-lowered", action="store_true", help="also execute the bufferized graph and save its output"
    )
    parser.add_argument(
        "--conv2d-im2col",
        action="store_true",
        help="Voyager's own replace_conv2d_with_im2col before quantization (its CI "
        "flag --conv2d_im2col): small-channel convs become linears",
    )
    parser.add_argument(
        "--extra-inputs",
        type=int,
        default=0,
        help="also run N more seeded random inputs through the quantized and the "
        "bufferized graph of the same compiled model (saved in io.pt)",
    )
    args = parser.parse_args(argv)

    root = _voyager_root()
    sys.path.insert(0, str(root / "test"))
    import test_codegen  # the repo's own fusion patterns (VECTOR_PIPELINE), as CI uses them
    from voyager_compiler import (
        compile,
        convert_pt2e,
        export_model,
        extract_input_preprocessor,
        fuse_operator,
        get_default_quantizer,
        prepare_pt2e,
        transform,
    )
    from voyager_compiler.codegen import voyager_ir_pb2
    from voyager_compiler.hardware_config import AcceleratorConfig

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)  # the pipeline re-exports while_loops; nothing may need grad
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    workload = json.loads(args.workload.read_text())
    config_fields = json.loads(args.config.read_text())
    if "pe_array_size" in config_fields:
        config_fields["pe_array_size"] = tuple(config_fields["pe_array_size"])
    config = AcceleratorConfig(**config_fields)

    module, example = build_workload(workload, dtype)
    scheme = QUANT_SCHEMES[args.quant]
    quantizer = get_default_quantizer(
        output_activation=None,
        force_scale_power_of_two=False,
        **{k: v for k, v in scheme.items() if k != "bias"},
        bias=scheme["bias"],
    )
    gm = export_model(module, (example,))
    if args.conv2d_im2col:
        from voyager_compiler import replace_conv2d_with_im2col

        replace_conv2d_with_im2col(gm)  # must precede prepare_pt2e, as in Voyager's harness
    gm = prepare_pt2e(gm, quantizer)
    for _ in range(args.calibration_steps):
        gm(torch.randn_like(example))
    convert_pt2e(gm, scheme["bias"])
    quantized_output = gm(example)
    # Extra inputs draw from their own generator, so the model, its calibration and the main example
    # are exactly what an export without them produces.
    generator = torch.Generator().manual_seed(args.seed + 1)
    extra_inputs = [torch.randn(example.shape, generator=generator).to(example.dtype) for _ in range(args.extra_inputs)]
    extra_quantized = [gm(x) for x in extra_inputs]

    patterns = test_codegen.VECTOR_PIPELINE
    transform(gm, (example,), patterns=patterns, config=config, skip_op_fusion=True, layout_policy=args.layout_policy)
    gm, preprocess = extract_input_preprocessor(gm)
    lowered_input = preprocess(example)
    fuse_operator(gm, patterns)

    args.out.mkdir(parents=True, exist_ok=True)
    compile(
        gm,
        (lowered_input,),
        config=config,
        output_dir=str(args.out),
        output_file=workload.get("name", workload["kind"]),
        dump_tensors=True,
    )
    lowered_output = gm(lowered_input) if args.run_lowered else None
    extra_lowered_inputs = [preprocess(x) for x in extra_inputs]
    extra_lowered_outputs = [gm(x) for x in extra_lowered_inputs]

    model = voyager_ir_pb2.Model()
    text_format.Parse((args.out / "model.txt").read_text(), model)
    (args.out / "model.json").write_text(
        json.dumps(json_format.MessageToDict(model, preserving_proto_field_name=True), indent=1)
    )
    torch.save(
        {
            "input": example,
            "lowered_input": lowered_input,
            "quantized_output": quantized_output,
            "lowered_output": lowered_output,
            "extra_inputs": extra_inputs,
            "extra_lowered_inputs": extra_lowered_inputs,
            "extra_quantized_outputs": extra_quantized,
            "extra_lowered_outputs": extra_lowered_outputs,
        },
        args.out / "io.pt",
    )

    import torchao
    import transformers

    manifest = {
        "compiler": "voyager",
        "voyager_commit": _git(root, "rev-parse", "HEAD"),
        "voyager_dirty_paths": [line for line in _git(root, "status", "--porcelain").splitlines() if line],
        "versions": {
            "torch": torch.__version__,
            "torchao": torchao.__version__,
            "transformers": transformers.__version__,
        },
        "workload": workload,
        "config": config_fields,
        "quant_scheme": args.quant,
        "quant_flags": scheme,
        "calibration_steps": args.calibration_steps,
        "seed": args.seed,
        "layout_policy": args.layout_policy,
        "model_dtype": str(dtype),
        "conv2d_im2col": args.conv2d_im2col,
        "lowered_output_saved": lowered_output is not None,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(json.dumps({"out": str(args.out), "voyager_commit": manifest["voyager_commit"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
