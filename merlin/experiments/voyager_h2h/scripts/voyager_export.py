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
    "int8_int32bias": {"input_activation": "int8,qs=per_tensor_symmetric",
                       "weight": "int8,qs=per_tensor_symmetric", "bias": "int32"},
    "int8_int24bias": {"input_activation": "int8,qs=per_tensor_symmetric",
                       "weight": "int8,qs=per_tensor_symmetric", "bias": "int24"},
}


def _voyager_root() -> Path:
    return Path(os.environ["MERLIN_EXT_VOYAGER_COMPILER"]).resolve()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


class _Linear(torch.nn.Module):
    def __init__(self, k: int, n: int, bias: bool, relu: bool):
        super().__init__()
        self.fc = torch.nn.Linear(k, n, bias=bias)
        self.relu = relu

    def forward(self, x):
        y = self.fc(x)
        return torch.relu(y) if self.relu else y


def build_workload(spec: dict, dtype: torch.dtype):
    """(module, example input) for a workload spec. Kinds: ``linear``, ``torchvision``."""
    kind = spec["kind"]
    if kind == "linear":
        module = _Linear(spec["K"], spec["N"], spec.get("bias", True), spec.get("relu", False))
        example = torch.randn(spec["M"], spec["K"], dtype=dtype)
    elif kind == "torchvision":
        from torchvision import models
        module = models.__dict__[spec["name"]](weights=spec.get("weights")).eval()
        example = torch.randn(*spec.get("input_shape", (1, 3, 224, 224)), dtype=dtype)
    else:
        raise ValueError(f"unknown workload kind {kind!r}")
    return module.eval().to(dtype).requires_grad_(False), example


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workload", required=True, type=Path, help="workload spec JSON")
    parser.add_argument("--config", required=True, type=Path,
                        help="AcceleratorConfig fields as JSON, derived by the caller")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--quant", default="int8_int32bias", choices=sorted(QUANT_SCHEMES))
    parser.add_argument("--calibration-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout-policy", default="systolic")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True,
                        help="run the model in bf16, as every accelerator-repo codegen scheme does")
    parser.add_argument("--run-lowered", action="store_true",
                        help="also execute the bufferized graph and save its output")
    args = parser.parse_args(argv)

    root = _voyager_root()
    sys.path.insert(0, str(root / "test"))
    import test_codegen  # the repo's own fusion patterns (VECTOR_PIPELINE), as CI uses them
    from voyager_compiler import (compile, convert_pt2e, export_model, extract_input_preprocessor,
                                  fuse_operator, get_default_quantizer, prepare_pt2e, transform)
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
    quantizer = get_default_quantizer(output_activation=None, force_scale_power_of_two=False,
                                      **{k: v for k, v in scheme.items() if k != "bias"},
                                      bias=scheme["bias"])
    gm = export_model(module, (example,))
    gm = prepare_pt2e(gm, quantizer)
    for _ in range(args.calibration_steps):
        gm(torch.randn_like(example))
    convert_pt2e(gm, scheme["bias"])
    quantized_output = gm(example)

    patterns = test_codegen.VECTOR_PIPELINE
    transform(gm, (example,), patterns=patterns, config=config, skip_op_fusion=True,
              layout_policy=args.layout_policy)
    gm, preprocess = extract_input_preprocessor(gm)
    lowered_input = preprocess(example)
    fuse_operator(gm, patterns)

    args.out.mkdir(parents=True, exist_ok=True)
    compile(gm, (lowered_input,), config=config, output_dir=str(args.out),
            output_file=workload.get("name", workload["kind"]), dump_tensors=True)
    lowered_output = gm(lowered_input) if args.run_lowered else None

    model = voyager_ir_pb2.Model()
    text_format.Parse((args.out / "model.txt").read_text(), model)
    (args.out / "model.json").write_text(json.dumps(
        json_format.MessageToDict(model, preserving_proto_field_name=True), indent=1))
    torch.save({"input": example, "lowered_input": lowered_input,
                "quantized_output": quantized_output, "lowered_output": lowered_output},
               args.out / "io.pt")

    import torchao
    import transformers
    manifest = {
        "compiler": "voyager",
        "voyager_commit": _git(root, "rev-parse", "HEAD"),
        "voyager_dirty_paths": [line for line in _git(root, "status", "--porcelain").splitlines()
                                if line],
        "versions": {"torch": torch.__version__, "torchao": torchao.__version__,
                     "transformers": transformers.__version__},
        "workload": workload, "config": config_fields, "quant_scheme": args.quant,
        "quant_flags": scheme, "calibration_steps": args.calibration_steps, "seed": args.seed,
        "layout_policy": args.layout_policy, "model_dtype": str(dtype),
        "lowered_output_saved": lowered_output is not None,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(json.dumps({"out": str(args.out), "voyager_commit": manifest["voyager_commit"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
