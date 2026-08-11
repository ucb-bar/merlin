"""``merlin-compile-kernel`` — compile a ``@triton.jit`` kernel for a Merlin target.

The CLI only orchestrates APIs that are tested elsewhere; it holds no compilation logic of its own,
so what it does is exactly what the test suite covers.

Its real work is turning command-line text into a :class:`~merlin.triton.spec.TritonKernelSpec`.
That is not a formality: a Triton kernel is not self-describing — its parameters are untyped, its
pointers carry no shape, its grid lives at the call site, and which buffers it writes is visible only
inside the body. Every one of those has to be stated, and stating them wrong is a miscompile rather
than an error, which is why nothing here is inferred from the source.

    merlin-compile-kernel examples/triton/vector_add.py:vector_add --target <TARGET> \\
        --arg 'x_ptr=*fp32:1025:read' --arg 'y_ptr=*fp32:1025:read' \\
        --arg 'out_ptr=*fp32:1025:write' --arg 'n_elements=i32' \\
        --assume n_elements=1025 --constexpr BLOCK_SIZE=256 --grid 5 --emit ttir,core-mlir

``<TARGET>`` is a placeholder on purpose: naming a real one here would be a target literal inside
the frontend, which is the thing this package is not allowed to contain. Runnable invocations live
in ``examples/triton/`` and ``docs/guides/triton_kernels.md``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from .spec import GridSpec, KernelArg, KernelSpecError, TritonKernelSpec

EMIT_STAGES = ("ttir", "core-mlir", "contract", "schedule", "interface", "target", "runtime",
               "command-buffer", "report")
DEFAULT_EMIT = ("ttir", "core-mlir", "report")


def load_kernel(location: str):
    """``path/to/file.py:kernel_name`` -> the decorated function.

    The file is imported rather than parsed, because ``@triton.jit`` reads the decorated function's
    own source and only works on a real module on disk.
    """
    path_text, sep, name = location.rpartition(":")
    if not sep:
        raise SystemExit(f"expected PATH.py:KERNEL, got {location!r}")
    path = Path(path_text).resolve()
    if not path.is_file():
        raise SystemExit(f"no such file: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    fn = getattr(module, name, None)
    if fn is None:
        raise SystemExit(f"{path} defines no {name!r}")
    return fn


def parse_arg(text: str) -> KernelArg:
    """``name=*fp32:16x32:read`` (pointer) or ``name=i32`` (scalar)."""
    name, sep, rest = text.partition("=")
    if not sep or not name:
        raise SystemExit(f"--arg needs NAME=SPEC, got {text!r}")
    if not rest.startswith("*"):
        return KernelArg(name, "scalar", rest)
    fields = rest[1:].split(":")
    if len(fields) != 3:
        raise SystemExit(
            f"--arg {name}: a pointer needs *DTYPE:SHAPE:EFFECT (e.g. *fp32:16x32:read), got {rest!r}")
    dtype, shape_text, effect = fields
    try:
        shape = tuple(int(d) for d in shape_text.split("x"))
    except ValueError:
        raise SystemExit(f"--arg {name}: shape {shape_text!r} must be ints joined by 'x'") from None
    return KernelArg(name, "pointer", dtype, shape=shape, effect=effect)


def parse_binding(text: str) -> tuple[str, Any]:
    """``NAME=VALUE``; the value is read as JSON so ints stay ints and strings stay strings."""
    name, sep, value = text.partition("=")
    if not sep:
        raise SystemExit(f"expected NAME=VALUE, got {text!r}")
    try:
        return name, json.loads(value)
    except json.JSONDecodeError:
        return name, value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="merlin-compile-kernel",
        description="Compile a @triton.jit kernel for a Merlin target.")
    p.add_argument("kernel", help="PATH.py:KERNEL_NAME")
    p.add_argument("--target", help="target name (resolved from the registry)")
    p.add_argument("--target-package", help="path to a target package directory (out-of-tree)")
    p.add_argument("--arg", action="append", default=[], metavar="NAME=SPEC",
                   help="*DTYPE:SHAPE:EFFECT for a pointer, DTYPE for a scalar; repeat, in order")
    p.add_argument("--constexpr", action="append", default=[], metavar="NAME=VALUE",
                   help="a tl.constexpr value (BLOCK_*/GROUP_* are portable meta-parameters)")
    p.add_argument("--assume", action="append", default=[], metavar="NAME=VALUE",
                   help="compile-time value of a runtime scalar, e.g. a mask bound")
    p.add_argument("--grid", default="1", metavar="X[,Y[,Z]]", help="SPMD launch grid extents")
    p.add_argument("--num-warps", type=int, help="recorded as provenance; never target semantics")
    p.add_argument("--num-stages", type=int, help="recorded as provenance; never target semantics")
    p.add_argument("--emit", default=",".join(DEFAULT_EMIT),
                   help=f"comma-separated stages to write ({', '.join(EMIT_STAGES)}, or 'all')")
    p.add_argument("--out", help="write here instead of a versioned artifacts product dir")
    p.add_argument("--verify", action="store_true", help="verify every emitted stage module")
    p.add_argument("--route-only", action="store_true",
                   help="report the routing decision and stop, without compiling")
    return p


def make_spec(args) -> TritonKernelSpec:
    try:
        grid = GridSpec(dims=tuple(int(d) for d in args.grid.split(",")))
    except ValueError:
        raise SystemExit(f"--grid {args.grid!r} must be 1-3 comma-separated ints") from None
    provenance = {k: v for k, v in (("num_warps", args.num_warps),
                                    ("num_stages", args.num_stages)) if v is not None}
    try:
        return TritonKernelSpec(
            function=load_kernel(args.kernel),
            args=tuple(parse_arg(a) for a in args.arg),
            grid=grid,
            constexprs=dict(parse_binding(c) for c in args.constexpr),
            assumptions=dict(parse_binding(a) for a in args.assume),
            provenance=provenance)
    except KernelSpecError as exc:
        raise SystemExit(f"kernel spec: {exc}") from None


def resolve_emit(text: str) -> list[str]:
    if text.strip() == "all":
        return list(EMIT_STAGES)
    stages = [s.strip() for s in text.split(",") if s.strip()]
    unknown = [s for s in stages if s not in EMIT_STAGES]
    if unknown:
        raise SystemExit(f"unknown --emit stage(s) {unknown}; known: {list(EMIT_STAGES)}")
    return stages


def output_dir(args, target: str) -> Path:
    from merlin.common.artifacts import new_product

    if args.out:
        path = Path(args.out)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return Path(new_product("triton-kernel", target=target, version=0).path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.target and not args.target_package:
        raise SystemExit("one of --target / --target-package is required")

    from merlin import compile_core
    from merlin.triton import source
    from merlin.triton.bridge import BridgeError, to_linalg
    from merlin.xdsl_dialects._common import text as to_text

    spec = make_spec(args)
    package = None
    if args.target_package:
        from merlin.targetgen.registry import load_target
        package = load_target(args.target_package)
    target = package.name if package is not None else args.target

    try:
        ttir = source.make_ttir(spec)
        bridged = to_linalg(ttir, spec)
    except (source.TritonFrontendError, BridgeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    route = compile_core.choose_route(
        bridged.module, target=args.target, target_package=package)
    print(f"route: {route.kind} — {route.reason}")
    if args.route_only:
        return 0

    stages = resolve_emit(args.emit)
    out = output_dir(args, target)
    written: list[Path] = []

    def write(name: str, body: str) -> None:
        if name in stages:
            suffix = "json" if name in ("report", "command-buffer") else "mlir"
            path = out / f"{name.replace('-', '_')}.{suffix}"
            path.write_text(body, encoding="utf-8")
            written.append(path)

    write("ttir", ttir.text)
    write("core-mlir", bridged.text)

    result = compile_core.compile_core_mlir(
        bridged.module, target=args.target, target_package=package,
        workdir=out / "llvm" if route.kind == "llvm" else None)
    if result.staged is not None:
        staged = result.staged
        for name, module in (("contract", staged.contract_module),
                             ("schedule", staged.schedule_module),
                             ("interface", staged.interface_module),
                             ("target", staged.target_module),
                             ("runtime", staged.runtime_module)):
            if args.verify:
                module.verify()
            write(name, to_text(module))
        write("command-buffer", json.dumps(staged.command_buffer, indent=2, sort_keys=True))

    write("report", json.dumps({
        "kernel": spec.name,
        "target": target,
        "route": route.as_dict(),
        "triton_version": ttir.triton_version,
        "ttir_digest": ttir.digest,
        "capability": bridged.report.as_dict(),
        "provenance": dict(spec.provenance),
    }, indent=2, sort_keys=True))

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
