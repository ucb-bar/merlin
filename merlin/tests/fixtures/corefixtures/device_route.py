from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _install_logical_backend(monkeypatch, directory):
    """Select this CPU fixture's entry convention, without qualifying any hardware."""
    from merlin.llvmlower.toolchain import clang
    from merlin.runtime.backends import base
    from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe, KernelStackFramePolicy

    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "logical-abi.ld"
    script.write_text("ENTRY(main)\nSECTIONS { .text : { *(.text*) } .data : { *(.data*) } .bss : { *(.bss*) } }\n")

    def recipe():
        return HarnessBuildRecipe(
            Path(clang()).resolve(),
            (),
            (),
            script,
            0,
            cflags=("-O1", "-fPIC"),
            kernel_stack_frame=KernelStackFramePolicy("gemmini_kernel", 4096),
        )

    monkeypatch.setattr(sys.modules[__name__], "harness_build_recipe", recipe, raising=False)
    monkeypatch.setitem(
        base._REGISTRY, "gemmini", base.BackendInfo("gemmini", base.TargetClass.CPU, base.BackendKind.KERNEL, __name__)
    )


def _fused_route(capture: str, work: str, package: str, model: str = "two") -> dict:
    """The group route and device build of the capture at ``capture``, as plain data.

    The child selects the same explicit synthetic capabilities as the parent;
    no real target facts or execution support is required."""
    from pathlib import Path

    from corefixtures.logical_device_target import install
    from merlin.common import mlir_query as mq
    from merlin.llvmlower.device_build import build_device_objects
    from merlin.llvmlower.device_offload import rewrite_groups_to_device
    from merlin.xdsl_dialects._common import text as to_text

    with pytest.MonkeyPatch.context() as monkeypatch:
        install(monkeypatch)
        _install_logical_backend(monkeypatch, Path(work))
        module = mq.parse((Path(capture) / "model.mlir").read_text(encoding="utf-8"))
        rewrite = rewrite_groups_to_device(
            module, "gemmini", select=lambda _shape: True, model=model, capture=Path(capture) / "model.mlir"
        )
        built = build_device_objects(
            "gemmini",
            rewrite.signatures,
            {r.symbol: r.dtypes for r in rewrite.routed},
            package_dir=package,
            workdir=Path(work) / "device",
            operand_dtype="int8",
            accum_dtype="i32",
            codegen_target="x86",
            cflags=["-O1", "-fPIC"],
            entries=rewrite.entries,
            call_buffers=rewrite.call_buffers,
            timeout=300,
        )
        (Path(work) / "routed.mlir").write_text(to_text(module), encoding="utf-8")
        return {
            "skipped": [list(s) for s in rewrite.skipped],
            "signatures": sorted(rewrite.signatures),
            "call_buffers": rewrite.call_buffers,
            "kernels": sorted(built.kernels),
            "build_skipped": [list(s) for s in built.skipped],
            "objects": [str(o) for o in built.objects],
            "shim": str(built.shim_object.with_suffix(".c")) if built.shim_object else None,
            "ops": {sym: entry.get("op") for sym, entry in rewrite.entries.items()},
            "epilogues": {sym: list(entry.get("epilogue") or ()) for sym, entry in rewrite.entries.items()},
        }
