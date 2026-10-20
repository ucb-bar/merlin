"""Public GEMM inputs and compile-only intake for a cumulative pass experiment.

No reference compiler, schedule, simulator, admission service or timing price is
provided. The selected unchanged Phase 1 compiler is invoked through its public
manifest. Successful LLVM emission is not execution or hardware qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

GEMMS_MKN = (
    (512, 512, 512),
    (12544, 256, 64),
    (12544, 64, 256),
    (3136, 512, 128),
    (3136, 128, 512),
    (784, 1024, 256),
)
CONVOLUTION_LABELS = ("4x3x56x64x64", "4x3x28x128x128", "4x3x14x256x256")
PAPERS = ("https://arxiv.org/abs/2505.18574", "https://arxiv.org/abs/2411.07211")


def gemm_source(m: int, k: int, n: int) -> str:
    """A complete ordered zero-seeded signed-i8 to i32 source contraction."""
    if any(type(value) is not int or not 0 < value < 1 << 31 for value in (m, k, n)):
        raise ValueError("GEMM extents must be explicit positive bounded integers")
    return f"""builtin.module {{
  func.func @forward(%a: tensor<{m}x{k}xi8>, %b: tensor<{k}x{n}xi8>) -> tensor<{m}x{n}xi32> {{
    %zero = arith.constant 0 : i32
    %init = tensor.splat %zero : tensor<{m}x{n}xi32>
    %out = linalg.generic {{
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    }} ins(%a, %b : tensor<{m}x{k}xi8>, tensor<{k}x{n}xi8>) outs(%init : tensor<{m}x{n}xi32>) {{
    ^bb0(%x: i8, %y: i8, %acc: i32):
      %wide_x = arith.extsi %x : i8 to i32
      %wide_y = arith.extsi %y : i8 to i32
      %product = arith.muli %wide_x, %wide_y : i32
      %sum = arith.addi %acc, %product : i32
      linalg.yield %sum : i32
    }} -> tensor<{m}x{n}xi32>
    func.return %out : tensor<{m}x{n}xi32>
  }}
}}
"""


def workload(m, k, n):
    """Explicit geometry and a proposed numerical test domain, without arrays."""
    gemm_source(m, k, n)
    return {
        "id": f"gemm_{m}_{k}_{n}",
        "dimensions": {"M": m, "K": k, "N": n},
        "inputs": [{"name": "a", "shape": [m, k], "dtype": "i8"}, {"name": "b", "shape": [k, n], "dtype": "i8"}],
        "outputs": [{"name": "out", "shape": [m, n], "dtype": "i32"}],
        "source_arithmetic": "signed_i8_products_ordered_zero_seeded_i32_sum",
        "proposed_execution_domain": {
            "operand_minimum": -2,
            "operand_maximum": 2,
            "every_partial_sum_absolute_bound": 4 * k,
            "status": "domain_proposal_only_fresh_capture_and_numerical_admission_required",
        },
        "paper_comparability": "shape_match_only_input_distribution_epilogue_runtime_and_timer_not_established",
    }


def generate(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for dimensions in GEMMS_MKN:
        row = workload(*dimensions)
        source = gemm_source(*dimensions).encode()
        filename = row["id"] + ".mlir"
        (destination / filename).write_bytes(source)
        rows.append({**row, "source": {"file": filename, "sha256": hashlib.sha256(source).hexdigest()}})
    return {
        "schema": "merlin.example.shared_pass_benchmark_inputs.v1",
        "dimension_order": "M_K_N",
        "papers": list(PAPERS),
        "gemms": rows,
        "convolutions": [
            {"label": label, "status": "awaiting_explicit_geometry_layout_stride_padding_dilation_and_epilogue"}
            for label in CONVOLUTION_LABELS
        ],
        "execution": "NOT_RUN",
        "performance": "NOT_MEASURED",
        "comparison_role": "public_development_not_held_out",
    }


def preflight(document, destination, *, compiler, compiler_sha256, python, mlir_install):
    """Invoke the original public compiler only; never launch target work."""
    import yaml

    from merlin.common.proc import run_checked
    from merlin.compile.model_execution_inputs import file_sha256, strict_tree_sha256

    compiler = Path(compiler).resolve(strict=True)
    # Keep the selected venv invocation path; resolving its interpreter symlink
    # would silently select a different Python package environment.
    python = Path(python).absolute()
    if not python.is_file():
        raise ValueError("selected compiler interpreter is unavailable")
    mlir_install = Path(mlir_install).resolve(strict=True)
    if strict_tree_sha256(compiler)["sha256"] != compiler_sha256:
        raise ValueError("original Phase 1 compiler differs from the operator-selected identity")
    manifest_bytes = (compiler / "manifest.yaml").read_bytes()
    manifest = yaml.safe_load(manifest_bytes)
    template = manifest["commands"]["emit_analysis_bundle"]["argv"]
    if type(template) is not list or not template or any(type(arg) is not str for arg in template):
        raise ValueError("selected compiler has no public analysis-emission command")
    tool = (compiler / manifest["entrypoints"]["tool"]).resolve(strict=True)
    if not tool.is_relative_to(compiler):
        raise ValueError("compiler public tool escapes its selected source tree")
    verifier = mlir_install / "bin/mlir-opt"
    translator = mlir_install / "bin/mlir-translate"

    def pin(path):
        return {"path": str(path), "resolved_path": str(path.resolve(strict=True)), "sha256": file_sha256(path)}

    tool_pins = tuple(pin(path) for path in (python, verifier, translator))
    source_pins = tuple(pin(destination / row["source"]["file"]) for row in document["gemms"])
    if any(
        actual["sha256"] != row["source"]["sha256"] for actual, row in zip(source_pins, document["gemms"], strict=True)
    ):
        raise ValueError("benchmark source changed before compiler consumption")
    product_pins = []

    def recheck():
        for expected in (*tool_pins, *source_pins, *product_pins):
            if pin(Path(expected["path"])) != expected:
                raise ValueError("selected tool, benchmark source or emitted product changed")
        if strict_tree_sha256(compiler)["sha256"] != compiler_sha256:
            raise ValueError("original compiler changed during preflight")

    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "MERLIN_MLIR_INSTALL": str(mlir_install),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    results = []
    for row in document["gemms"]:
        recheck()
        source = destination / row["source"]["file"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != row["source"]["sha256"]:
            raise ValueError("benchmark source changed before compiler consumption")
        output = destination / (row["id"] + ".target.mlir")
        command_buffer = destination / (row["id"] + ".analysis.json")
        values = {"tool": str(tool), "input_mlir": str(source), "output_json": str(command_buffer)}
        argv = [arg.format_map(values) for arg in template]
        argv[0] = str(python)
        before = time.perf_counter()
        try:
            completed = run_checked(argv, timeout=120, cwd=compiler, env=environment)
            recheck()
            if not completed.stdout.strip() or not command_buffer.is_file():
                raise RuntimeError("compiler returned success without both selected artifacts")
            json.loads(command_buffer.read_text())
            output.write_text(completed.stdout)
            (destination / (row["id"] + ".stderr")).write_text(completed.stderr)
            artifact_sha = hashlib.sha256(output.read_bytes()).hexdigest()
            product_pins.extend((pin(output), pin(command_buffer)))
            run_checked([verifier, output, "--verify-each", "-o", "-"], timeout=120, env=environment)
            recheck()
            if hashlib.sha256(output.read_bytes()).hexdigest() != artifact_sha:
                raise ValueError("target artifact changed during syntax verification")
            result = {
                "status": "EMITTED",
                "output_sha256": artifact_sha,
                "output_bytes": output.stat().st_size,
                "syntax_verified_by": tool_pins[1],
            }
        except RuntimeError as error:
            result = {"status": "REFUSED", "diagnostic": str(error)}
        if (
            strict_tree_sha256(compiler)["sha256"] != compiler_sha256
            or (compiler / "manifest.yaml").read_bytes() != manifest_bytes
        ):
            raise ValueError("original compiler changed during preflight")
        if hashlib.sha256(source.read_bytes()).hexdigest() != row["source"]["sha256"]:
            raise ValueError("benchmark source changed during compilation")
        results.append(
            {"id": row["id"], **result, "argv": argv, "compile_preflight_wall_seconds": time.perf_counter() - before}
        )
    recheck()
    return {
        "schema": "merlin.example.shared_pass_compile_preflight.v1",
        "compiler_sha256": compiler_sha256,
        "selected_tools": tool_pins,
        "source_pins": source_pins,
        "product_pins": product_pins,
        "results": results,
        "scope": "unchanged_public_Phase1_compiler_source_to_target_artifact_only",
        "device_passes_applied": [],
        "target_execution": "NOT_RUN",
        "numerical_agreement": "NOT_ESTABLISHED",
        "speedup": "NOT_MEASURED",
        "compiler_peak_memory": "UNKNOWN",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compiler", type=Path)
    parser.add_argument("--compiler-sha256")
    parser.add_argument("--python", type=Path)
    parser.add_argument("--mlir-install", type=Path)
    args = parser.parse_args()
    selection = (args.compiler, args.compiler_sha256, args.python, args.mlir_install)
    if any(value is not None for value in selection) and not all(value is not None for value in selection):
        parser.error("compiler preflight requires all four explicit compiler/toolchain arguments")
    destination = args.output_dir.resolve()
    document = generate(destination)
    (destination / "inputs.json").write_text(json.dumps(document, indent=2) + "\n")
    if args.compiler is not None:
        report = preflight(
            document,
            destination,
            compiler=args.compiler,
            compiler_sha256=args.compiler_sha256,
            python=args.python,
            mlir_install=args.mlir_install,
        )
        (destination / "compile-preflight.json").write_text(json.dumps(report, indent=2) + "\n")
        return int(any(row["status"] != "EMITTED" for row in report["results"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
