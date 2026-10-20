"""Expose the exact optimized LLVM used by device object code generation.

The ordinary selected Clang optimization level remains in both commands. The
second command skips repeated LLVM IR optimization, retaining normal backend
optimization. Object/linked instruction inventory checks remain independent.
"""

import hashlib
import json
from pathlib import Path


def _producer_output(pin):
    raw = Path(pin["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise ValueError("optimized LLVM producer invocation changed")
    outputs = json.loads(raw).get("outputs")
    if (not isinstance(outputs, list) or len(outputs) != 1
            or not isinstance(outputs[0].get("sha256"), str) or len(outputs[0]["sha256"]) != 64):
        raise ValueError("optimized LLVM producer selected another output")
    return outputs[0]


def artifacts(custody):
    """Publish original output paths even when multiple kernels share one build."""
    return {role: Path(_producer_output(custody[step])["path"])
            for role, step in (("optimized_llvm_ir", "optimization"), ("optimized_llvm_mlir", "optimized_import"))}


def commands(compile_command, translator):
    """Split an existing observed LLVM-to-object command without changing flags."""
    argv = list(compile_command)
    if len(argv) < 5 or argv[-4] != "-c" or argv[-2] != "-o":
        raise ValueError("device object command must compile one explicit LLVM input")
    source, output = Path(argv[-3]), Path(argv[-1])
    optimized = source.with_suffix(".optimized.ll")
    imported = source.with_suffix(".optimized.llvm.mlir")
    prefix = argv[:-4]
    return (
        [*prefix, "-S", "-emit-llvm", str(source), "-o", str(optimized)],
        [translator, "--import-llvm", "--mlir-print-op-generic", str(optimized), "-o", str(imported)],
        [*prefix, "-Xclang", "-disable-llvm-passes", "-c", str(optimized), "-o", str(output)],
    )


def observe(compile_command, translator, *, observer, timeout, directory, report):
    """Retain actual source→optimized LLVM→typed import invocation custody."""
    optimize, imported, backend = commands(compile_command, translator)
    result, optimization, elapsed = observer(
        optimize, timeout=timeout, directory=directory, stage="device_llvm_optimization",
        inputs=(Path(optimize[-3]),), output=Path(optimize[-1]), metrics=report,
        expected_input_sha256=report["analysis"]["sha256"],
        metrics_stage="optimization_s",
    )
    report["stages"]["optimization_s"] = elapsed
    if result.returncode:
        return None, f"clang LLVM optimization: {(result.stderr or '').strip()[:200]}"
    output = _producer_output(optimization)
    if Path(output["path"]) != Path(optimize[-1]).resolve():
        raise ValueError("optimized LLVM producer selected another output path")
    optimized_sha256 = output["sha256"]
    result, import_receipt, elapsed = observer(
        imported, timeout=timeout, directory=directory, stage="device_optimized_llvm_import",
        inputs=(Path(imported[-3]),), output=Path(imported[-1]),
        expected_input_sha256=optimized_sha256,
    )
    report["stages"]["optimized_import_s"] = elapsed
    if result.returncode:
        return None, f"mlir-translate optimized LLVM import: {(result.stderr or '').strip()[:200]}"
    return {"argv": backend, "optimization": optimization, "optimized_import": import_receipt,
            "sha256": optimized_sha256}, ""
