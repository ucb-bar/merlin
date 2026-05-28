"""`./merlin quantize` — INT8 quantization for ONNX models.

Two modes:

- **Direct**: `./merlin quantize <input.onnx> --shape 1,3,224,224 [--shape ...]`
  Quantize any `.onnx` file by passing its input shapes explicitly.

- **Registry**: `./merlin quantize --model <name>`
  Look the model up in `models/models_config.json`, which records
  `source_path` + `input_shapes` for every well-known model in the
  repo (dronet, mlp, yolov8_nano, tinyllama, opu_bench_*, …). Saves
  retyping shapes for the recurring set.

Both modes share the same QDQ INT8 symmetric pipeline; the output
`.q.int8.onnx` is consumable by `./merlin compile` for any target.
"""

from __future__ import annotations

import json
from pathlib import Path


def setup_parser(parser):
    parser.add_argument(
        "input_onnx",
        type=Path,
        nargs="?",
        help="Source .onnx file (omit when using --model).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Quantize a named model from models/models_config.json "
            "(e.g. --model dronet). Mutually exclusive with positional "
            "input_onnx + --shape."
        ),
    )
    parser.add_argument(
        "--shape",
        action="append",
        default=None,
        help=(
            "Input tensor shape as comma-separated integers (e.g. 1,3,224,224). "
            "Repeat once per input for multi-input models. Required in direct "
            "mode; ignored in --model mode."
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
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to the model registry JSON (default: " "<repo>/models/models_config.json). Only used with --model."
        ),
    )


def _resolve_registry_entry(model_name: str, config_path: Path | None) -> tuple[Path, list[list[int]]]:
    """Look up ``model_name`` in ``models/models_config.json`` and return
    ``(source_onnx_path, input_shapes)`` resolved relative to the config
    file's directory.
    """
    if config_path is None:
        # Default: <repo>/models/models_config.json
        repo_root = Path(__file__).resolve().parents[2]
        config_path = repo_root / "models" / "models_config.json"

    if not config_path.exists():
        raise SystemExit(f"model registry not found: {config_path}")

    with config_path.open() as f:
        registry = json.load(f)

    if model_name not in registry:
        known = ", ".join(sorted(registry.keys()))
        raise SystemExit(f"model '{model_name}' not in {config_path}.\nKnown models: {known}")

    entry = registry[model_name]
    source = Path(entry["source_path"])
    if not source.is_absolute():
        source = config_path.parent / source
    if not source.exists():
        raise SystemExit(f"source ONNX not found: {source}")

    shapes = [list(s) for s in entry["input_shapes"]]
    return source, shapes


def main(args) -> int:
    # Lazy-import the heavy onnxruntime dep so `./merlin --help` and
    # unrelated subcommands stay fast.
    from quantize.int8 import _parse_shape, quantize

    # Resolve input + shapes from whichever mode the user picked.
    if args.model is not None:
        if args.input_onnx is not None or args.shape:
            raise SystemExit("--model is mutually exclusive with positional input_onnx / --shape")
        input_onnx, shapes = _resolve_registry_entry(args.model, args.config)
    else:
        if args.input_onnx is None:
            raise SystemExit("either pass <input.onnx> + --shape, or use --model <name>")
        if not args.shape:
            raise SystemExit("direct mode requires at least one --shape")
        if not args.input_onnx.exists():
            print(f"input not found: {args.input_onnx}")
            return 1
        input_onnx = args.input_onnx
        shapes = [_parse_shape(s) for s in args.shape]

    output = args.output or input_onnx.with_suffix(".q.int8.onnx")

    print(f"==> quantize: {input_onnx}")
    print(f"    output:  {output}")
    print(f"    shapes:  {shapes}")
    print(f"    samples: {args.calibration_samples}")

    quantize(input_onnx, output, shapes, args.calibration_samples)

    print(f"==> done: {output} ({output.stat().st_size:,} bytes)")
    return 0
