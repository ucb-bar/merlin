"""Capture, check and replay supported Q/DQ examples with explicit source selections.

Run capture/replay-source under the pinned Torch interpreter, lower/replay-lowering
under the Merlin interpreter. See --help for the exact independent selections.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from merlin.common.paths import artifacts_dir, repo_root


def source_cut() -> dict:
    from merlin.targetgen.qdq_capture_identity import sha

    paths = (
        "build_tools/scripts/qualify_qdq_capture.py",
        "src/merlin/targetgen/qdq_capture.py",
        "src/merlin/targetgen/qdq_capture_identity.py",
        "src/merlin/targetgen/_m2m_capture_worker.py",
        "src/merlin/targetgen/_m2m_capture_artifacts.py",
        "src/merlin/llvmlower/passes_xdsl.py",
        "src/merlin/verify/gate_checks.py",
        "src/merlin/verify/lean/check.py",
        "src/merlin/verify/lean/mlir_export.py",
        "merlin/lean/SPEC_LOCK",
    )
    return {name: sha((repo_root() / name).read_bytes()) for name in paths}


def capture(args) -> dict:
    import m2m
    import torch
    import torch.ao.quantization.fx._decomposed  # noqa: F401

    from merlin.targetgen.qdq_capture import write_check
    from merlin.targetgen.qdq_capture_identity import begin, sha

    identity = begin(args.producer, args.torch_wheel)

    class Quantize(torch.nn.Module):
        def __init__(self, scale=0.125):
            super().__init__()
            self.scale = scale

        def forward(self, x):
            return torch.ops.quantized_decomposed.quantize_per_tensor.default(x, self.scale, 0, -128, 127, torch.int8)

    class Dequantize(torch.nn.Module):
        def forward(self, x):
            return torch.ops.quantized_decomposed.dequantize_per_tensor.default(x, 0.125, 0, -128, 127, torch.int8)

    class Channel(torch.nn.Module):
        def forward(self, x, scale, zero):
            return torch.ops.quantized_decomposed.dequantize_per_channel.default(
                x, scale, zero, 1, -128, 127, torch.int8
            )

    class Composition(torch.nn.Module):
        def forward(self, x):
            q = torch.ops.quantized_decomposed.quantize_per_tensor.default(x, 0.125, 0, -128, 127, torch.int8)
            d = torch.ops.quantized_decomposed.dequantize_per_tensor.default(q, 0.125, 0, -128, 127, torch.int8)
            return q, d

    x = torch.tensor([[-16.0, -0.1875, 0.0625], [0.1875, 15.875, 16.0]], dtype=torch.float32)
    q = torch.tensor([[-128, -1, 0], [1, 126, 127]], dtype=torch.int8)
    examples = {
        "quantize": (Quantize(), (x,)),
        "dequantize_tensor": (Dequantize(), (q,)),
        "dequantize_channel": (
            Channel(),
            (q, torch.tensor([0.125, 0.25, 0.5]), torch.tensor([0, -1, 1], dtype=torch.int32)),
        ),
        "composition": (Composition(), (x,)),
        "refused_scalar_precision": (
            Quantize(0.0023796888063332034),
            (torch.tensor([0.11779459565877914], dtype=torch.float32),),
        ),
    }
    results = {}
    for name, (model, inputs) in examples.items():
        output = args.output / name
        output.mkdir(parents=True, exist_ok=False)
        exported = torch.export.export(model, inputs)
        torch.export.save(exported, output / "original.pt2")
        torch.save(inputs, output / "inputs.pt")
        with torch.no_grad():
            reference = model(*inputs)
        torch.save(reference, output / "reference.pt")
        converted = m2m.convert(exported, inputs, backend="fx_importer", capture_trace=True)
        report = write_check(converted, identity, output)
        expected = "unsupported" if name.startswith("refused_") else "matched"
        if report["status"] != expected:
            raise AssertionError(f"{name}: expected {expected}: {report}")
        if expected == "unsupported" and "loses precision" not in report.get("reason", ""):
            raise AssertionError(f"precision control did not refuse the intended condition: {report}")
        raw_files = {file.name: sha(file.read_bytes()) for file in output.iterdir() if file.is_file()}
        results[name] = {"status": report["status"], "reason": report.get("reason"), "artifacts": raw_files}
        (output / "capture-files.json").write_text(json.dumps(results[name], indent=2, sort_keys=True) + "\n")
    return results


def replay_source(args) -> dict:
    import torch
    import torch.ao.quantization.fx._decomposed  # noqa: F401 -- register operators before loading PT2
    from m2m.capture.trace import snapshot_exported_program

    from merlin.targetgen.qdq_capture import QDQCaptureError, replay
    from merlin.targetgen.qdq_capture_identity import sha

    results = {}
    for output in sorted(args.output.iterdir()):
        if not output.is_dir() or not (output / "capture-files.json").is_file():
            continue
        inventory = json.loads((output / "capture-files.json").read_bytes())
        for name, expected in inventory["artifacts"].items():
            if sha((output / name).read_bytes()) != expected:
                raise AssertionError(f"capture artifact changed: {output.name}/{name}")
        exported = torch.export.load(output / "original.pt2")
        original = snapshot_exported_program(exported, stage="original")
        trace = json.loads((output / "qdq-frontend.json").read_bytes())

        def calls(graph):
            # PT2 serialization can populate previously absent output-node val
            # metadata. Output operand identities already bind every return;
            # retain full result types for placeholders and producing calls.
            return [
                {
                    k: node[k]
                    for k in ("id", "op", "target", "args", "kwargs", "results")
                    if k != "results" or node["op"] != "output"
                }
                for node in graph["nodes"]
            ]

        if calls(original) != calls(trace["graphs"]["original"]):
            raise AssertionError("retained original ExportedProgram differs from its source graph")
        inputs = torch.load(output / "inputs.pt", weights_only=True)
        expected_reference = torch.load(output / "reference.pt", weights_only=True)
        observed = exported.module()(*inputs)
        expected_flat, expected_tree = torch.utils._pytree.tree_flatten(expected_reference)
        observed_flat, observed_tree = torch.utils._pytree.tree_flatten(observed)
        if expected_tree != observed_tree or any(
            not torch.equal(a, b) for a, b in zip(expected_flat, observed_flat, strict=True)
        ):
            raise AssertionError("original source replay differs from retained native reference")
        try:
            results[output.name] = replay(output)
        except QDQCaptureError as exc:
            if inventory["status"] != "unsupported" or "loses precision" not in str(exc):
                raise
            results[output.name] = {"status": "unsupported", "reason": str(exc)}
    if len(results) != 5:
        raise AssertionError("source replay did not cover all five original examples")
    return results


def lowering(args, *, replay=False) -> dict:
    from merlin.llvmlower.passes_xdsl import lower_quant_ext, parse_mlir_text, preprocess_text_textual
    from merlin.targetgen.qdq_capture import check_source
    from merlin.verify import gate_checks
    from merlin.verify.lean import check
    from merlin.xdsl_dialects._common import text

    results = {}
    for name in ("quantize", "dequantize_tensor", "dequantize_channel", "composition"):
        source = args.output / name
        report = json.loads((source / "qdq-source-check.json").read_bytes())
        from merlin.targetgen.qdq_capture_identity import sha

        before = (source / "qdq-source.mlir").read_text()
        if report["status"] != "matched" or sha(before.encode()) != report["mlir_sha256"]:
            raise AssertionError(f"{name}: source correspondence input changed or was refused")
        trace_raw = (source / "qdq-frontend.json").read_bytes()
        if sha(trace_raw) != report["trace_sha256"]:
            raise AssertionError(f"{name}: source graph bytes changed")
        sites = check_source(json.loads(trace_raw), parse_mlir_text(before))
        if sites != report["sites"]:
            raise AssertionError(f"{name}: recomputed source correspondence differs")
        for route in ("xdsl", "textual"):
            directory = source / f"lower-{route}"
            if replay:
                if (
                    (directory / "before.mlir").read_bytes() != before.encode()
                    or (directory / "stage/before.mlir").read_bytes() != before.encode()
                    or (directory / "after.mlir").read_bytes() != (directory / "stage/after.mlir").read_bytes()
                ):
                    raise AssertionError(f"{name}/{route}: receipt boundary differs from selected capture")
                defined = directory / "defined-forward/receipt.json"
                outcome = check.compose_chain(
                    [directory / "stage"],
                    directory / "support-forward/receipt.json",
                    "forward",
                    directory / "replayed-chain.json",
                    m2m_python=args.compiler_python,
                    definedness=defined if json.loads(defined.read_bytes())["verdict"] == "verified" else None,
                )
                if outcome.get("refinement_verdict") != "verified":
                    raise AssertionError(f"{name}/{route}: {outcome}")
                results[f"{name}-{route}"] = {
                    "refinement_verdict": outcome["refinement_verdict"],
                    "verdict": outcome["verdict"],
                }
                continue
            directory.mkdir(exist_ok=False)
            stats = None
            if route == "xdsl":
                module = parse_mlir_text(before)
                count = lower_quant_ext(module)
                if count != len(report["sites"]):
                    raise AssertionError(f"{name}: unexpected lowered site count: {count}")
                after = text(module, generic=True)
            else:
                after, stats = preprocess_text_textual(before)
                if stats["quant_ext_lowered"] + stats["dequantize_lowered"] != len(report["sites"]):
                    raise AssertionError(f"{name}: unexpected textual rewrite counts: {stats}")
            (directory / "before.mlir").write_text(before)
            (directory / "after.mlir").write_text(after)
            label = (
                "preprocess_text_textual.dequant_to_generic"
                if route == "textual" and stats["dequantize_lowered"]
                else "lower_quant_ext"
            )
            detail = gate_checks.tensor_refinement(
                {
                    "directory": str(directory),
                    "label": label,
                    "before": str(directory / "before.mlir"),
                    "after": str(directory / "after.mlir"),
                    "params": {
                        "sdk_bin": str(args.sdk),
                        "compiler_python": str(args.compiler_python),
                        "index_bits": 64,
                    },
                }
            )
            results[f"{name}-{route}"] = {
                "claim": detail["claim"],
                "strengthenings": detail["strengthenings"],
                "before_sha256": sha(before.encode()),
                "after_sha256": sha(after.encode()),
                "stats": stats,
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "replay-source", "lower", "replay-lowering"))
    parser.add_argument("--output", type=Path, default=artifacts_dir() / "verification/qdq-source-binding")
    parser.add_argument("--producer", type=Path)
    parser.add_argument("--torch-wheel", type=Path)
    parser.add_argument("--compiler-python", type=Path)
    parser.add_argument("--sdk", type=Path)
    args = parser.parse_args()
    args.output = args.output.absolute()
    sources = source_cut()
    if args.mode == "capture":
        if not args.producer or not args.torch_wheel:
            parser.error("capture requires --producer and --torch-wheel")
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "source-cut.json").write_text(json.dumps(sources, sort_keys=True, indent=2) + "\n")
        result = capture(args)
    else:
        if sources != json.loads((args.output / "source-cut.json").read_bytes()):
            raise AssertionError("qualification implementation or frozen source changed; recapture explicitly")
        if args.mode == "replay-source":
            result = replay_source(args)
        else:
            if not args.compiler_python or not args.sdk:
                parser.error("lower/replay-lowering requires --compiler-python and --sdk")
            args.compiler_python = args.compiler_python.absolute()
            args.sdk = args.sdk.absolute()
            if not args.compiler_python.is_file() or not (args.sdk / "lake").is_file():
                parser.error("selected compiler Python or Lean SDK is missing")
            # The textual preprocessing helper consults the process selection.
            # Bind it to the same explicit interpreter as the receipt exporter.
            os.environ["MERLIN_COMPILER_PYTHON"] = str(args.compiler_python)
            os.environ["MERLIN_LEAN_SDK_BIN"] = str(args.sdk)
            result = lowering(args, replay=args.mode == "replay-lowering")
    if sources != source_cut():
        raise AssertionError("qualification implementation or frozen source changed during the command")
    (args.output / f"{args.mode}-summary.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
