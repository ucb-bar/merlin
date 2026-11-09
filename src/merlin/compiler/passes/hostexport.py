"""Export a separate, explicit host compiler library for EL4 source review.

This source-only release can be selected with the existing compiler-library
contract. It never changes a frozen profile or includes a device implementation,
reference model, runtime harness, measurement record or grader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PUBLIC_MODULES = (
    "merlin.compiler.transforms.host",
    "merlin.compile.host_optimizations",
    "merlin.runtime.host_arithmetic",
    "merlin.runtime.host_fma_packet",
    "merlin.runtime.host_outward",
    "merlin.runtime.host_rounding",
    "merlin.llvmlower.host_builder",
    "merlin.llvmlower.host_scalar_builder",
    "merlin.llvmlower.host_tensor_codegen",
    "merlin.llvmlower.host_epilogue",
    "merlin.llvmlower.host_rounding",
    "merlin.llvmlower.integer_gemm",
    "merlin.llvmlower.captured_requant",
    "merlin.llvmlower.requantization_match",
    "merlin.llvmlower.requantization",
)
IMPLEMENTATION_MODULES = (
    "merlin.compiler.ir.logical_abi",
    "merlin.common.paths",
    "merlin.common.access",
    "merlin.compiler.transforms.host_contraction",
    "merlin.compiler.transforms.host_reduction",
    "merlin.compiler.transforms.host_broadcast",
    "merlin.compiler.transforms.host_integer_math",
    "merlin.llvmlower.host_tensor_rewrites",
    "merlin.llvmlower.finite_integer_math",
    "merlin.llvmlower.llvm_loop_metadata",
    "merlin.llvmlower.source_fma_batch",
    "merlin.llvmlower.late_quant_rne",
    "merlin.llvmlower.constant_float_clamp",
    "merlin.llvmlower.constant_fma_packet",
)
NAMESPACES = ("merlin", "merlin.common", "merlin.compiler", "merlin.compiler.ir", "merlin.compiler.transforms",
              "merlin.compile", "merlin.llvmlower", "merlin.runtime")


def _licenses(source_root, selected):
    if selected is not None:
        return tuple(Path(selected) / name for name in ("LICENSE", "NOTICE"))
    checkout = source_root.parent
    if all((checkout / name).is_file() for name in ("LICENSE", "NOTICE")):
        return tuple(checkout / name for name in ("LICENSE", "NOTICE"))
    from importlib.metadata import distribution

    installed = distribution("merlin")
    files = tuple(installed.files or ())
    owners = [path for path in files if str(path) == "merlin/compiler/passes/hostexport.py"]
    if len(owners) != 1 or Path(installed.locate_file(owners[0])).resolve() != Path(__file__).resolve():
        raise ValueError("host library licenses require the distribution owning the actual exporter")
    result = []
    for name in ("LICENSE", "NOTICE"):
        matches = [path for path in files if path.name == name and path.parent.name == "licenses"
                   and path.parent.parent.name.endswith(".dist-info")]
        if len(matches) != 1:
            raise ValueError(f"installed host library requires exactly one packaged {name}")
        result.append(Path(installed.locate_file(matches[0])))
    return tuple(result)


def export_host_optimizations(directory, *, license_directory=None):
    """Copy the exact declared algorithms; review/selection remains explicit."""
    directory = Path(directory).absolute()
    if any(path.is_symlink() for path in (directory, *directory.parents)):
        raise ValueError("host library export requires an unlinked directory")
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("host library export requires an empty directory")
    source_root = Path(__file__).resolve().parents[3]
    licenses = _licenses(source_root, license_directory)
    originals, members = {}, []
    modules = (*NAMESPACES, *PUBLIC_MODULES, *IMPLEMENTATION_MODULES)
    for module in modules:
        relative = module.replace(".", "/") + ("/__init__.py" if module in NAMESPACES else ".py")
        path = source_root / relative
        payload = path.read_bytes()
        originals[path] = payload
        if module in NAMESPACES:
            payload = b'"""Shared host compiler namespace; public APIs are explicit leaves."""\n'
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        members.append(dict(module=module, path=relative, sha256=hashlib.sha256(payload).hexdigest()))
    for path in licenses:
        name = path.name
        payload = path.read_bytes()
        originals[path] = payload
        (directory / name).write_bytes(payload)
        members.append(dict(module=None, path=name, sha256=hashlib.sha256(payload).hexdigest()))
    if any(path.read_bytes() != payload for path, payload in originals.items()):
        raise ValueError("host compiler sources changed during export")
    manifest = dict(
        schema="merlin.shared_host_library.v1",
        public_modules=list(PUBLIC_MODULES),
        members=members,
        admission="requires_explicit_original_compiler_library_selection_for_both_comparison_arms",
        dependencies=["xdsl; caller-selected native MLIR interpreter for tensor rewrites"],
        scope="source algorithms only; no numerical, device, runtime or timing qualification",
    )
    manifest["library_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (directory / "shared-host-library.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--license-directory", type=Path)
    args = parser.parse_args(argv)
    manifest = export_host_optimizations(args.output_dir, license_directory=args.license_directory)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
