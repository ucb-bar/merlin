"""Exact representative discovery, complete scatter, bounded fallback and fenv."""

from __future__ import annotations

import ctypes
import subprocess

import numpy as np
import pytest

from merlin.common.cpu_pool import shared_compute_pool
from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
from merlin.llvmlower.codegen import mlir_runtime_c
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.rhs_column_reuse import FEATURE, MARKER, apply_for_test, edit_pipeline
from merlin.llvmlower.toolchain import clang


def source(m=3, n=19, k=17, *, seed="0.0", reverse=False, raw="i8"):
    a, b, c, integer = f"tensor<{m}x{k}xf32>", f"tensor<{k}x{n}xf32>", f"tensor<{m}x{n}xf32>", f"tensor<{k}x{n}x{raw}>"
    return f"""module {{func.func @forward(%a:{a},%raw:{integer})->{c} attributes {{llvm.emit_c_interface}} {{
      %be=tensor.empty():{b}
      %b=linalg.generic {{indexing_maps=[affine_map<(k,n)->(k,n)>,affine_map<(k,n)->(k,n)>],
       iterator_types=["parallel","parallel"]}} ins(%raw:{integer}) outs(%be:{b}) {{
       ^bb0(%x:{raw},%u:f32):%v=arith.sitofp %x:{raw} to f32 linalg.yield %v:f32}}->{b}
      %z=arith.constant {seed}:f32
      %ce=tensor.empty():{c}
      %c=linalg.fill ins(%z:f32) outs(%ce:{c})->{c}
      %r=linalg.generic {{indexing_maps=[affine_map<(m,n,k)->(m,k)>,
       affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],
       iterator_types=["parallel","parallel","reduction"],prov.label="original_product"}}
       ins(%a,%b:{a},{b}) outs(%c:{c}) {{^bb0(%x:f32,%y:f32,%s:f32):
        %p=arith.mulf {"%y,%x" if reverse else "%x,%y"}:f32
        %v=arith.addf {"%s,%p" if reverse else "%p,%s"}:f32
        linalg.yield %v:f32}}->{c}
      return %r:{c}
    }}}}"""


def build(root, text, selected):
    root.mkdir()
    with shared_compute_pool(timeout_s=300):
        llvm = lower_to_llvm_ir(
            text, workdir=root, features={"scalar_contraction_accumulator", *({FEATURE} if selected else set())}
        )
    path, image = root / "model.ll", root / "model.so"
    path.write_text(llvm)
    with shared_compute_pool(timeout_s=300):
        subprocess.run(
            [
                str(clang()),
                "-O3",
                "-shared",
                "-fPIC",
                "-ffp-contract=off",
                str(path),
                str(mlir_runtime_c()),
                "-lm",
                "-o",
                str(image),
            ],
            check=True,
            capture_output=True,
        )
    return HostModel.load(str(image), image_policy=PrivateHostImagePolicy(root)), llvm


@pytest.fixture
def environment(tmp_path):
    c, so = tmp_path / "fenv.c", tmp_path / "fenv.so"
    c.write_text("""#include <fenv.h>
      int mode(int i){const int m[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};return fesetround(m[i]);}
      void reset(int i){feclearexcept(FE_ALL_EXCEPT);if(i)feraiseexcept(i==1?FE_INVALID|FE_INEXACT:FE_ALL_EXCEPT);}
      int flags(void){return fetestexcept(FE_ALL_EXCEPT);}""")
    with shared_compute_pool(timeout_s=300):
        subprocess.run([str(clang()), "-shared", "-fPIC", str(c), "-lm", "-o", str(so)], check=True)
    env = ctypes.CDLL(str(so))
    yield env
    assert env.mode(0) == 0
    env.reset(0)


@pytest.mark.parametrize("n,reverse", [(19, False), (64, True), (9, False)])
def test_complete_words_flags_inputs_tails_and_unique_fallback(tmp_path, environment, n, reverse):
    text = source(n=n, reverse=reverse)
    with shared_compute_pool(timeout_s=300):
        changed, count = apply_for_test(text, directory=tmp_path / "rewrite")
    assert count == 1 and "reuse_exact_rhs_columns" in changed
    control, _ = build(tmp_path / "control", text, False)
    candidate, llvm = build(tmp_path / "candidate", text, True)
    assert "16777619" in llvm, "the supplemental numeric test must execute an admitted transform"
    assert "llvm.fma" not in llvm and "fmul contract" not in llvm
    rng = np.random.default_rng(239)
    lhs = rng.uniform(-1, 1, (3, 17)).astype("f4")
    lhs[0, :5] = np.array([0, 0x80000000, 1, 0x80000001, 0x3F800000], dtype="u4").view("f4")
    for unique in (1, min(n, 5), min(n, 17), n):
        columns = rng.integers(-128, 128, (17, unique), dtype="i1")
        rhs = np.ascontiguousarray(columns[:, np.arange(n) % unique])
        for fallback in (False, True):
            a = lhs.copy()
            if fallback:
                a[1, 0] = np.float32(2.0)
                a[1, 1] = np.array([0x7F800001], dtype="u4").view("f4")[0]
            before = (a.view("u4").copy(), rhs.copy())
            for mode in range(4):
                for flags in range(3):
                    results = []
                    for model in (control, candidate):
                        assert environment.mode(mode) == 0
                        environment.reset(flags)
                        storage = np.full((3 * n + 32,), 0xA5A5A5A5, dtype="u4")
                        out = storage[16:-16].view("f4").reshape(3, n)
                        model([(x.ctypes.data, x.shape) for x in (a, rhs, out)])
                        results.append((out.view("u4").copy(), environment.flags()))
                        assert np.all(storage[:16] == 0xA5A5A5A5) and np.all(storage[-16:] == 0xA5A5A5A5)
                    np.testing.assert_array_equal(results[0][0], results[1][0])
                    assert results[0][1] == results[1][1]
            np.testing.assert_array_equal(a.view("u4"), before[0])
            np.testing.assert_array_equal(rhs, before[1])


@pytest.mark.parametrize(
    "change",
    ["strict", "fastmath", "negative_zero", "nonzero_seed", "capacity_n", "capacity_k", "unknown_attr", "wide_integer"],
)
def test_unsupported_static_cases_remain_original(tmp_path, change):
    text = source(
        n=129 if change == "capacity_n" else 19,
        k=4097 if change == "capacity_k" else 17,
        seed="-0.0" if change == "negative_zero" else "1.0" if change == "nonzero_seed" else "0.0",
        raw="i32" if change == "wide_integer" else "i8",
    )
    if change == "strict":
        text = text.replace("llvm.emit_c_interface", "llvm.emit_c_interface,strictfp")
    if change == "fastmath":
        text = text.replace("arith.mulf %x,%y:f32", "arith.mulf %x,%y fastmath<contract>:f32")
    if change == "unknown_attr":
        text = text.replace('prov.label="original_product"', "unknown.effect=true")
    with shared_compute_pool(timeout_s=300):
        _, count = apply_for_test(text, directory=tmp_path / "rewrite")
    assert count == 0


def test_pipeline_default_and_duplicate_guard():
    original = ["canonicalize", "__merlin_scalar_contraction_accumulator__", "one-shot-bufferize"]
    assert edit_pipeline(original) == [original[0], MARKER, *original[1:]]
    with pytest.raises(ValueError):
        edit_pipeline([MARKER, *original])
    with pytest.raises(ValueError):
        edit_pipeline(["one-shot-bufferize"])


def test_nonzero_marker_stays_with_owner_after_ordinary_split_boundaries():
    from merlin.llvmlower.automatic_host_features import MARKER as automatic
    from merlin.llvmlower.automatic_host_features import edit_pipeline as edit_automatic
    from merlin.llvmlower.broadcast_math_hoist import _edit_pipeline as edit_broadcast
    from merlin.llvmlower.rhs_column_reuse import NONZERO_MARKER, edit_nonzero_pipeline
    from merlin.llvmlower.scalar_contraction import MARKER as scalar
    from merlin.llvmlower.scalar_contraction import _edit_pipeline as edit_scalar
    from merlin.llvmlower.scalar_squared_sum import edit_pipeline as edit_squared

    owner = "__merlin_scalar_contraction_guarded_nonzero_reduction__"
    stages = edit_scalar(edit_broadcast(edit_automatic(["canonicalize", "one-shot-bufferize"])))
    at = stages.index(scalar)
    stages = edit_squared([*stages[:at], owner, *stages[at:]])
    changed = edit_nonzero_pipeline(stages)
    assert changed.index(NONZERO_MARKER) == changed.index(owner) - 1
    assert NONZERO_MARKER not in changed[: changed.index(automatic)]
    assert stages.count(NONZERO_MARKER) == 0
    for invalid in (["one-shot-bufferize"], [owner, owner], [NONZERO_MARKER, owner], [MARKER, owner]):
        with pytest.raises(ValueError):
            edit_nonzero_pipeline(invalid)


def test_hash_collision_requires_complete_k_equality(tmp_path):
    rng, seen, pair = np.random.default_rng(913), {}, None
    for _ in range(10000):
        column = rng.integers(-128, 128, 17, dtype="i1")
        signature = 2166136261
        for value in column.astype("f4").view("u4"):
            signature = ((signature ^ int(value)) * 16777619) & 0xFFFFFFFF
        if signature in seen and not np.array_equal(column, seen[signature]):
            pair = [seen[signature], column]
            break
        seen[signature] = column
    assert pair is not None
    a = rng.uniform(-1, 1, (3, 17)).astype("f4")
    b = np.ascontiguousarray(np.stack([pair[n % 2] for n in range(9)], axis=1))
    values = []
    for name, selected in [("control", False), ("candidate", True)]:
        model, _ = build(tmp_path / name, source(n=9), selected)
        out = np.empty((3, 9), dtype="f4")
        model([(v.ctypes.data, v.shape) for v in (a, b, out)])
        values.append(out.view("u4").copy())
    assert np.any(values[0][:, 0] != values[0][:, 1])
    np.testing.assert_array_equal(*values)


def test_runtime_read_only_input_alias_remains_under_upstream_authority(tmp_path):
    # Both source entries may share storage. The existing result ABI separately
    # requires a distinct destination; it cannot promise output/entry overlap.
    original = np.zeros(19 * 19 + 4, dtype="u1")
    original[: 3 * 19 * 4].view("f4")[:] = np.float32(0.5)
    values = []
    for name, selected in [("control", False), ("candidate", True)]:
        model, llvm = build(tmp_path / name, source(k=19), selected)
        assert all("noalias" not in line for line in llvm.splitlines() if line.startswith("define "))
        aliased = original.copy()
        lhs = aliased[: 3 * 19 * 4].view("f4").reshape(3, 19)
        rhs = aliased[: 19 * 19].view("i1").reshape(19, 19)
        output = np.empty((3, 19), dtype="f4")
        model([(v.ctypes.data, v.shape) for v in (lhs, rhs, output)])
        values.append(output.view("u4").copy())
        np.testing.assert_array_equal(aliased, original)
    np.testing.assert_array_equal(*values)


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_support_dispatch_forwards_original_callbacks_and_restores_owner(mode):
    from merlin.llvmlower.rhs_column_reuse import EMPTY_MARKER, NONZERO_MARKER, RUNNER_PRELUDE, TAIL2_MARKER

    marker = [NONZERO_MARKER, EMPTY_MARKER, TAIL2_MARKER][mode]
    expected_options = {"fresh_result": True} if mode else {}
    if mode == 2:
        expected_options["overlap_tail"] = 2
    calls, selected, untouched = [], object(), object()
    limit, setup = object(), object()

    def original_scalarize(*args, **kwargs):
        calls.append((args, kwargs))
        return 2

    namespace = {"_ZC_ORIGINAL_SCALARIZE": original_scalarize}

    def original_stages(*args):
        assert marker not in args[2]
        return namespace["_ZC_ORIGINAL_SCALARIZE"](
            args[0],
            args[1],
            8,
            2,
            1,
            selection=[selected, untouched],
            reduction_limit=limit,
            reduction_setup=setup,
        )

    namespace["_run_stages"] = original_stages
    exec(RUNNER_PRELUDE, namespace)

    def rewrite(ctx, module, unroll, selection, reduction_limit, reduction_setup, remaining, **options):
        assert options == expected_options
        assert selection == [selected, untouched]
        assert unroll == 2 and reduction_limit is limit and reduction_setup is setup
        remaining.append(untouched)
        return 1

    namespace["_reuse_rhs_columns"] = rewrite
    pipeline = marker + ",__merlin_scalar_contraction_guarded_nonzero_reduction__"
    assert namespace["_run_stages"]("ctx", "module", pipeline, 3, (4,), (5,), (6,), (7,)) == 3
    assert namespace["_ZC_ORIGINAL_SCALARIZE"] is original_scalarize
    assert calls == [
        (
            ("ctx", "module", 8, 2, 1),
            {
                "selection": [untouched],
                "output_guard": None,
                "product_guard": None,
                "reduction_limit": limit,
                "reduction_setup": setup,
            },
        )
    ]


def test_support_dispatch_restores_on_failure_and_refuses_unknown_callback():
    from merlin.llvmlower.rhs_column_reuse import NONZERO_MARKER, RUNNER_PRELUDE

    def original(*args, **kwargs):
        return None

    namespace = {"_ZC_ORIGINAL_SCALARIZE": original}

    def unsupported(*args):
        return namespace["_ZC_ORIGINAL_SCALARIZE"](None, None, other_callback=object())

    namespace["_run_stages"] = unsupported
    exec(RUNNER_PRELUDE, namespace)
    pipeline = NONZERO_MARKER + ",__merlin_scalar_contraction_guarded_nonzero_reduction__"
    with pytest.raises(TypeError, match="other_callback"):
        namespace["_run_stages"](None, None, pipeline, 0)
    assert namespace["_ZC_ORIGINAL_SCALARIZE"] is original


def test_exact_rhs_ssa_sharing_across_consumers_and_invocation_invalidation(tmp_path, environment):
    text = source()
    out, inp = "tensor<3x19xf32>", "tensor<3x17xf32>"
    text = text.replace(f"%a:{inp},%raw", f"%a:{inp},%other:{inp},%raw", 1)
    text = text.replace(f")->{out} attributes", f")->({out},{out}) attributes", 1)
    body = text[text.index("%r=linalg.generic") : text.index("return %r")]
    second = body.replace("%r=", "%r2=").replace("ins(%a,%b:", "ins(%other,%b:")
    text = text.replace(f"return %r:{out}", second + f"return %r,%r2:{out},{out}")
    with shared_compute_pool(timeout_s=300):
        changed, count = apply_for_test(text, directory=tmp_path / "rewrite")
    assert count == 2 and changed.count("16777619 : i32") == 1
    control, _ = build(tmp_path / "control", text, False)
    candidate, _ = build(tmp_path / "candidate", text, True)
    rng = np.random.default_rng(915)
    for invocation in range(3):
        a = rng.uniform(-1, 1, (3, 17)).astype("f4")
        other = rng.uniform(-1, 1, (3, 17)).astype("f4")
        if invocation == 1:
            other[0, 0] = np.float32(2.0)
        b = rng.integers(-128, 128, (17, 5), dtype="i1")
        b = np.ascontiguousarray(b[:, np.arange(19) % 5])
        before = [x.tobytes() for x in (a, other, b)]
        for mode in range(4):
            results = []
            for model in (control, candidate):
                environment.mode(mode)
                environment.reset(1)
                outputs = [np.empty((3, 19), dtype="f4") for _ in range(2)]
                model([(x.ctypes.data, x.shape) for x in (a, other, b, *outputs)])
                results.append(([x.view("u4").copy() for x in outputs], environment.flags()))
            assert results[0][1] == results[1][1]
            for a_bits, b_bits in zip(results[0][0], results[1][0]):
                np.testing.assert_array_equal(a_bits, b_bits)
        assert before == [x.tobytes() for x in (a, other, b)]


def test_unknown_scalar_call_remains_original(tmp_path):
    text = source().replace("module {", "module {func.func private @observer()\n", 1)
    text = text.replace("%p=arith.mulf", "func.call @observer():()->()\n%p=arith.mulf")
    with shared_compute_pool(timeout_s=300):
        _, count = apply_for_test(text, directory=tmp_path / "rewrite")
    assert count == 0


def test_module_private_workspace_capacity_retains_complete_unselected_contractions(tmp_path):
    text = source(n=64)
    body = text[text.index("%r=linalg.generic") : text.index("return %r")]
    repeated = "\n".join(
        body.replace("%r=", f"%r{i}=").replace(
            'prov.label="original_product"', f'prov.label="original_product",prov.instance={i}'
        )
        for i in range(44)
    )
    result_type = "tensor<3x64xf32>"
    all_types = ",".join([result_type] * 44)
    text = text.replace(body, repeated)
    text = text.replace(f")->{result_type} attributes", f")->({all_types}) attributes", 1)
    text = text.replace(f"return %r:{result_type}", "return " + ",".join(f"%r{i}" for i in range(44)) + ":" + all_types)
    with shared_compute_pool(timeout_s=300):
        changed, count = apply_for_test(text, directory=tmp_path / "rewrite")
    assert count == 8192 // (3 * 64)
    assert changed.count('prov.label = "original_product"') >= 44


def test_fresh_result_completely_materializes_rows_and_preserves_fallback(tmp_path, environment):
    text = source(n=19, reverse=True)
    with shared_compute_pool(timeout_s=300):
        changed, count = apply_for_test(text, directory=tmp_path / "rewrite", fresh_result=True)
    assert count == 1
    control, _ = build(tmp_path / "control", text, False)
    candidate, llvm = build(tmp_path / "candidate", changed, False)
    assert "16777619" in llvm
    rng = np.random.default_rng(3729)
    for unique in (1, 5, 17, 19):
        columns = rng.integers(-128, 128, (17, unique), dtype="i1")
        rhs = np.ascontiguousarray(columns[:, np.arange(19) % unique])
        for fallback in (False, True):
            lhs = rng.uniform(-1, 1, (3, 17)).astype("f4")
            if fallback:
                lhs[1, 0] = np.float32(2)
                lhs[1, 1] = np.array([0x7F800001], dtype="u4").view("f4")[0]
            before = lhs.tobytes(), rhs.tobytes()
            for mode in range(4):
                for flags in range(3):
                    results = []
                    for model in (control, candidate):
                        assert environment.mode(mode) == 0
                        environment.reset(flags)
                        storage = np.full(3 * 19 + 32, 0xA5A5A5A5, dtype="u4")
                        out = storage[16:-16].view("f4").reshape(3, 19)
                        model([(x.ctypes.data, x.shape) for x in (lhs, rhs, out)])
                        results.append((out.view("u4").copy(), environment.flags()))
                        assert np.all(storage[:16] == 0xA5A5A5A5)
                        assert np.all(storage[-16:] == 0xA5A5A5A5)
                    np.testing.assert_array_equal(results[0][0], results[1][0])
                    assert results[0][1] == results[1][1]
            assert before == (lhs.tobytes(), rhs.tobytes())


def test_overlapping_final_group_preserves_ordered_words_and_sticky_flags(tmp_path, environment):
    width = 2
    text = source(n=19, reverse=True)
    with shared_compute_pool(timeout_s=300):
        changed, count = apply_for_test(text, directory=tmp_path / "rewrite", fresh_result=True, overlap_tail=width)
    assert count == 1
    control, _ = build(tmp_path / "control", text, False)
    candidate, llvm = build(tmp_path / "candidate", changed, False)
    assert "16777619" in llvm and "llvm.fma" not in llvm
    rng = np.random.default_rng(3731)
    lhs = rng.uniform(-1, 1, (3, 17)).astype("f4")
    lhs[0, :6] = np.array([0, 0x80000000, 1, 0x80000001, 0x3F800000, 0xBF800000], dtype="u4").view("f4")
    for unique in (1, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19):
        columns = rng.integers(-128, 128, (17, unique), dtype="i1")
        rhs = np.ascontiguousarray(columns[:, np.arange(19) % unique])
        before = lhs.tobytes(), rhs.tobytes()
        for mode in range(4):
            for flags in range(3):
                results = []
                for model in (control, candidate):
                    assert environment.mode(mode) == 0
                    environment.reset(flags)
                    storage = np.full(3 * 19 + 32, 0xA5A5A5A5, dtype="u4")
                    out = storage[16:-16].view("f4").reshape(3, 19)
                    model([(x.ctypes.data, x.shape) for x in (lhs, rhs, out)])
                    results.append((out.view("u4").copy(), environment.flags()))
                    assert np.all(storage[:16] == 0xA5A5A5A5) and np.all(storage[-16:] == 0xA5A5A5A5)
                np.testing.assert_array_equal(results[0][0], results[1][0])
                assert results[0][1] == results[1][1]
        assert before == (lhs.tobytes(), rhs.tobytes())
