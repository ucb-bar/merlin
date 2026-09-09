from __future__ import annotations

from merlin.llvmlower.ciface_inline import inline_wide_ciface_implementations


def _arguments(count: int) -> str:
    return ", ".join(f"i64 %{index}" for index in range(count))


def test_wide_ciface_implementation_is_forced_inline():
    args = _arguments(257)
    source = (
        f"define void @forward({args}) {{\n  ret void\n}}\n"
        "define void @_mlir_ciface_forward(ptr %0) {\n  ret void\n}\n"
    )
    repaired, report = inline_wide_ciface_implementations(source)

    assert ") alwaysinline {" in repaired.splitlines()[0]
    assert report == {
        "schema": "wide_ciface_inline_v1",
        "max_flattened_arguments": 256,
        "inlined": [{
            "implementation": "forward",
            "wrapper": "_mlir_ciface_forward",
            "flattened_arguments": 257,
        }],
        "count": 1,
    }


def test_small_or_unwrapped_functions_are_unchanged():
    source = (
        "define void @forward(i64 %0) {\n  ret void\n}\n"
        f"define void @helper({_arguments(300)}) {{\n  ret void\n}}\n"
        "define void @_mlir_ciface_forward(ptr %0) {\n  ret void\n}\n"
    )
    repaired, report = inline_wide_ciface_implementations(source)

    assert repaired == source
    assert report["count"] == 0


def test_existing_alwaysinline_is_idempotent():
    args = _arguments(300)
    source = (
        f"define void @forward({args}) alwaysinline {{\n  ret void\n}}\n"
        "define void @_mlir_ciface_forward(ptr %0) {\n  ret void\n}\n"
    )

    repaired, report = inline_wide_ciface_implementations(source)
    assert repaired == source
    assert report["count"] == 0


def test_aggregate_argument_commas_do_not_inflate_width():
    source = (
        "define void @forward({ ptr, ptr, i64, [2 x i64], [2 x i64] } %0) {\n  ret void\n}\n"
        "define void @_mlir_ciface_forward(ptr %0) {\n  ret void\n}\n"
    )
    repaired, report = inline_wide_ciface_implementations(
        source, max_flattened_arguments=1)
    assert repaired == source
    assert report["count"] == 0
