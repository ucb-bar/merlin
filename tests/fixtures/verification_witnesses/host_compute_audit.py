from __future__ import annotations

import pytest

_ELEMENTS = 1024


def _loop(name: str, trips: int, body: str, *, bound: str | None = None) -> str:
    """One counted loop over ``trips`` iterations running ``body`` (which may use %i, %in, %out)."""
    limit = bound or f'%n = "llvm.mlir.constant"() <{{value = {trips} : i64}}> : () -> i64'
    return f'\n  "llvm.func"() <{{sym_name = "{name}", function_type = !llvm.func<void (!llvm.ptr, !llvm.ptr, i64)>}}> ({{\n  ^entry(%in: !llvm.ptr, %out: !llvm.ptr, %dyn: i64):\n    %c0 = "llvm.mlir.constant"() <{{value = 0 : i64}}> : () -> i64\n    %c1 = "llvm.mlir.constant"() <{{value = 1 : i64}}> : () -> i64\n    %scale = "llvm.mlir.constant"() <{{value = 5.000000e-01 : f32}}> : () -> f32\n    {limit}\n    "llvm.br"(%c0)[^header] : (i64) -> ()\n  ^header(%i: i64):\n    %cond = "llvm.icmp"(%i, %n) <{{predicate = 2 : i64}}> : (i64, i64) -> i1\n    "llvm.cond_br"(%cond)[^body, ^exit] <{{operandSegmentSizes = array<i32: 1, 0, 0>}}> : (i1) -> ()\n  ^body:\n{body}\n    %next = "llvm.add"(%i, %c1) : (i64, i64) -> i64\n    "llvm.br"(%next)[^header] : (i64) -> ()\n  ^exit:\n    "llvm.return"() : () -> ()\n  }}) : () -> ()'


_REQUANT = '    %src = "llvm.getelementptr"(%in, %i) <{elem_type = i32, rawConstantIndices = array<i32: -2147483648>}> : (!llvm.ptr, i64) -> !llvm.ptr\n    %acc = "llvm.load"(%src) : (!llvm.ptr) -> i32\n    %f = "llvm.sitofp"(%acc) : (i32) -> f32\n    %s = "llvm.fmul"(%f, %scale) : (f32, f32) -> f32\n    %q = "llvm.fptosi"(%s) : (f32) -> i8\n    %dst = "llvm.getelementptr"(%out, %i) <{elem_type = i8, rawConstantIndices = array<i32: -2147483648>}> : (!llvm.ptr, i64) -> !llvm.ptr\n    "llvm.store"(%q, %dst) : (i8, !llvm.ptr) -> ()'


def _graded_run(tmp_path, name: str, artifact: str, buffer: dict):
    import json

    generated = tmp_path / name / "generated"
    generated.mkdir(parents=True)
    (generated / "command_buffer.json").write_text(json.dumps(buffer), encoding="utf-8")
    (generated / "lowered.llvm.mlir").write_text(artifact, encoding="utf-8")
    status = "fail" if name == "already_failing" else "pass"
    (tmp_path / name / "capsule_result.json").write_text(
        json.dumps({"capsule": name, "status": status}), encoding="utf-8"
    )


# Explicit fixture/reference reexports consumed by retained direct tests.
__all__ = [
    "_ELEMENTS",
    "_REQUANT",
    "_graded_run",
    "_loop",
    "pytest",
]
