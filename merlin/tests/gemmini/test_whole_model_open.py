"""An OPEN model -- host regions computing between its groups -- as one program.

The model here is a transformer's linear layer in miniature, the shape every one of SmolVLA's 302
device groups has: a per-row dynamic quantization on the host (absmax, divide, round, clamp, narrow),
an int8 x int8 contraction into an int32 accumulator, the widening cast the capture follows it with,
and a per-row x per-column dequantization back on the host. The closed build refuses such a model; the
open route states its device part, cuts the device group out of the host code without changing any
value, and has the target write each dispatch's call. The numbers are checked against numpy by hand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from merlin.common import mlir_query as mq
from merlin.llvmlower import whole_program as WP
from merlin.perf import whole_model_open as WO
from merlin.runtime import linalg_numpy as LN

pytestmark = pytest.mark.target("gemmini")

#: One real target: the open route is target-agnostic, but whether a contraction is a DEVICE group is
#: the target's own admission, so the test needs one that admits an int8 contraction.
_TARGET = "gemmini"

M, K, N = 8, 16, 12
_R = "affine_map<(d0, d1) -> (d0)>"
_C = "affine_map<(d0, d1) -> (d1)>"
_I = "affine_map<(d0, d1) -> (d0, d1)>"
_P = '["parallel", "parallel"]'

MODEL = f"""
module {{
  func.func @forward(%x: tensor<{M}x{K}xf32>, %w: tensor<{N}x{K}xi8>, %ws: tensor<{N}xf32>) -> tensor<{M}x{N}xf32> {{
    %zf = arith.constant 0.000000e+00 : f32
    %a0 = tensor.splat %zf : tensor<{M}xf32>
    %amax = linalg.generic {{indexing_maps = [{_I}, {_R}], iterator_types = ["parallel", "reduction"]}} ins(%x : tensor<{M}x{K}xf32>) outs(%a0 : tensor<{M}xf32>) {{
    ^bb0(%in: f32, %acc: f32):
      %ab = math.absf %in : f32
      %mx = arith.maximumf %acc, %ab : f32
      linalg.yield %mx : f32
    }} -> tensor<{M}xf32>
    %e1 = tensor.empty() : tensor<{M}xf32>
    %scale = linalg.generic {{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}} ins(%amax : tensor<{M}xf32>) outs(%e1 : tensor<{M}xf32>) {{
    ^bb0(%in: f32, %o: f32):
      %d = arith.constant 1.270000e+02 : f32
      %r = arith.divf %in, %d : f32
      linalg.yield %r : f32
    }} -> tensor<{M}xf32>
    %e2 = tensor.empty() : tensor<{M}x{K}xi8>
    %q = linalg.generic {{indexing_maps = [{_I}, {_R}, {_I}], iterator_types = {_P}}} ins(%x, %scale : tensor<{M}x{K}xf32>, tensor<{M}xf32>) outs(%e2 : tensor<{M}x{K}xi8>) {{
    ^bb0(%in: f32, %s: f32, %o: i8):
      %v = arith.divf %in, %s : f32
      %r = math.roundeven %v : f32
      %lo = arith.constant -1.270000e+02 : f32
      %hi = arith.constant 1.270000e+02 : f32
      %c1 = arith.maximumf %r, %lo : f32
      %c2 = arith.minimumf %c1, %hi : f32
      %n = arith.fptosi %c2 : f32 to i8
      linalg.yield %n : i8
    }} -> tensor<{M}x{K}xi8>
    %e3 = tensor.empty() : tensor<{K}x{N}xi8>
    %wt = linalg.transpose ins(%w : tensor<{N}x{K}xi8>) outs(%e3 : tensor<{K}x{N}xi8>) permutation = [1, 0]
    %zi = arith.constant 0 : i32
    %acc0 = tensor.splat %zi : tensor<{M}x{N}xi32>
    %mm = linalg.generic {{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]}} ins(%q, %wt : tensor<{M}x{K}xi8>, tensor<{K}x{N}xi8>) outs(%acc0 : tensor<{M}x{N}xi32>) {{
    ^bb0(%a: i8, %b: i8, %c: i32):
      %ae = arith.extsi %a : i8 to i32
      %be = arith.extsi %b : i8 to i32
      %p = arith.muli %ae, %be : i32
      %t = arith.addi %c, %p : i32
      linalg.yield %t : i32
    }} -> tensor<{M}x{N}xi32>
    %e4 = tensor.empty() : tensor<{M}x{N}xf32>
    %f = linalg.generic {{indexing_maps = [{_I}, {_I}], iterator_types = {_P}}} ins(%mm : tensor<{M}x{N}xi32>) outs(%e4 : tensor<{M}x{N}xf32>) {{
    ^bb0(%in: i32, %o: f32):
      %r = arith.sitofp %in : i32 to f32
      linalg.yield %r : f32
    }} -> tensor<{M}x{N}xf32>
    %e5 = tensor.empty() : tensor<{M}x{N}xf32>
    %y = linalg.generic {{indexing_maps = [{_I}, {_R}, {_C}, {_I}], iterator_types = {_P}}} ins(%f, %scale, %ws : tensor<{M}x{N}xf32>, tensor<{M}xf32>, tensor<{N}xf32>) outs(%e5 : tensor<{M}x{N}xf32>) {{
    ^bb0(%v: f32, %s: f32, %t: f32, %o: f32):
      %a1 = arith.mulf %v, %s : f32
      %a2 = arith.mulf %a1, %t : f32
      linalg.yield %a2 : f32
    }} -> tensor<{M}x{N}xf32>
    return %y : tensor<{M}x{N}xf32>
  }}
}}
"""


def _inputs():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((M, K)).astype(np.float32)
    w = rng.integers(-127, 128, size=(N, K), dtype=np.int8)
    ws = (rng.random(N).astype(np.float32) + 0.5) / 100
    return x, w, ws


def _by_hand(x, w, ws):
    scale = np.abs(x).max(axis=1) / np.float32(127)
    q = np.clip(np.rint(x / scale[:, None]), -127, 127).astype(np.int64)
    acc = q @ w.astype(np.int64).T
    return acc.astype(np.float32) * scale[:, None] * ws[None, :], q, acc


def _device_groups(module):
    from merlin.xdsl_dialects.lowering import compute_groups as CG

    return [g for g in CG.form_groups(module, _TARGET) if g.placement != CG.HOST]


def test_the_oracle_computes_the_model_by_hand() -> None:
    x, w, ws = _inputs()
    (y,) = LN.evaluate(mq.parse(MODEL), [x, w, ws])
    want, _q, _acc = _by_hand(x, w, ws)
    assert np.allclose(y, want, rtol=1e-6, atol=1e-7)


def test_the_closed_build_refuses_an_open_model_and_the_open_statement_states_its_device_part() -> None:
    with pytest.raises(WP.WholeProgramError, match="host region"):
        WP.whole_program_buffer(mq.parse(MODEL), _TARGET, weight_args={1, 2})
    buffer = WP.whole_program_buffer(mq.parse(MODEL), _TARGET, weight_args={1, 2}, open_model=True)
    route = buffer["whole_program"]
    assert route["open_model"] is True
    (row,) = route["per_group"]
    assert row["op"] == "matmul"
    lhs, rhs, dst = (row["operands"][k] for k in ("lhs", "rhs", "dst"))
    # The activation the host produced is named as such; the weight is the STORED argument behind the
    # transpose; the committed result is the accumulator, and it is an output the host reads back.
    assert lhs in route["host_produced"] and buffer["tensors"][lhs]["dtype"] == "i8"
    assert rhs == "arg1"
    assert buffer["tensors"][dst]["dtype"] == "i32" and dst in buffer["outputs"]
    assert route["host_regions"] and all(r["why"] for r in route["host_regions"])


def test_the_statement_only_reads_the_module_it_states() -> None:
    """The build shares one parse of the capture between its statement passes and its own read of the
    capture's groups; that is sound only while stating a model leaves the module as it was."""
    module = mq.parse(MODEL)
    before = str(module)
    WP.whole_program_buffer(module, _TARGET, weight_args={1, 2}, open_model=True)
    assert str(module) == before


def test_a_clone_of_the_shared_parse_is_the_module_a_fresh_parse_gives() -> None:
    """The cut normalizes a CLONE of the shared parse rather than parsing again; that is the same module."""
    from merlin.frontends.linalg_mlir import parse_mlir_text

    shared = mq.parse(MODEL)
    copy = shared.clone()
    assert str(copy) == str(parse_mlir_text(MODEL))
    WO.normalize(copy)
    assert str(shared) == str(parse_mlir_text(MODEL)), "normalizing the copy changed the shared parse"


def test_the_cut_preserves_every_value_and_states_what_crosses_it() -> None:
    x, w, ws = _inputs()
    original = mq.parse(MODEL)
    (want,) = LN.evaluate(original, [x, w, ws])
    module = mq.parse(MODEL)
    main, host, dispatches = WO.externalize_dispatches(module, _device_groups(module))
    (dispatch,) = dispatches
    assert dispatch.result == ((M, N), "i32")
    lhs_at, rhs_at = dispatch.root_operands
    assert dispatch.arguments[lhs_at] == ((M, K), "i8") and dispatch.arguments[rhs_at] == ((K, N), "i8")
    assert dispatch.zero_init and dispatch.init_argument is not None
    seen: dict[str, list] = {}

    def answer(values):
        seen["args"] = values
        return LN.evaluate(host, values, function=dispatch.host_symbol)

    (got,) = LN.evaluate(main, [x, w, ws], calls={dispatch.dispatch_symbol: answer})
    assert np.array_equal(got, want)
    _y, q, acc = _by_hand(x, w, ws)
    assert np.array_equal(np.asarray(seen["args"][lhs_at]).astype(np.int64), q)
    assert np.array_equal(LN.evaluate(host, seen["args"], function=dispatch.host_symbol)[0], acc.astype(np.int32))


def test_every_dispatch_operand_is_declared_read_only_and_the_text_keeps_it() -> None:
    """A dispatch writes a fresh result and never an operand, and says so: without the declaration the
    bufferization copies every operand (a weight, the zero accumulator) before every call to protect it.
    xDSL's custom printer drops a declaration's argument attributes, so the program's text states them in
    the generic form, and MLIR reads them back."""
    import subprocess

    from merlin.llvmlower import toolchain

    module = mq.parse(MODEL)
    main, _host, (dispatch,) = WO.externalize_dispatches(module, _device_groups(module))
    (declaration,) = [op for op in main.walk() if op.name == "func.func" and not op.body.blocks]
    assert [dict(a.data)[WO.BUFFER_ACCESS].data for a in declaration.arg_attrs.data] == ["read"] * len(
        dispatch.arguments
    )
    text = WO.module_text(main)
    assert text.count(f'{WO.BUFFER_ACCESS} = "read"') == len(dispatch.arguments)
    opt = toolchain.mlir_translate().with_name("mlir-opt")
    if opt.is_file():
        parsed = subprocess.run([str(opt)], input=text, capture_output=True, text=True, check=True).stdout
        assert f"@{dispatch.dispatch_symbol}(" in parsed and parsed.count(f'{WO.BUFFER_ACCESS} = "read"') == len(
            dispatch.arguments
        )


def test_a_group_read_outside_its_cut_is_refused() -> None:
    """A member other than the committed one that host code also reads: cutting the group out would
    drop that value, so the cut refuses it by name rather than leaving the reader dangling."""
    import dataclasses

    leaky = MODEL.replace(
        "    return %y : tensor<",
        f"    %keep = tensor.collapse_shape %q [[0, 1]] : tensor<{M}x{K}xi8> into tensor<{M * K}xi8>\n    return %y : tensor<",
    )
    module = mq.parse(leaky)
    (group,) = _device_groups(module)
    producer = group.root.operands[0].owner  # the quantize that feeds the contraction
    widened = dataclasses.replace(group, members=[producer, *group.members])
    with pytest.raises(WO.OpenModelError, match="read outside the group"):
        WO.externalize_dispatches(module, [widened])


def _dispatch_dict():
    module = mq.parse(MODEL)
    _main, _host, (dispatch,) = WO.externalize_dispatches(module, _device_groups(module))
    return {**dispatch.to_dict(), "op": "matmul"}, dispatch.group


def test_the_target_writes_each_route_and_the_local_checks() -> None:
    from merlin.runtime.backends import base as backends

    driver = backends.whole_model_driver(_TARGET)
    dispatch, group = _dispatch_dict()
    uart = driver.program.UART
    samples = {group: {"indices": [0, 5], "values": [1, -2], "bound": 1}}
    host = driver.dispatch.render_dispatch(
        [dispatch], {group: {"on": "host"}}, uart=uart, verify="local", samples=samples
    )
    assert f"_mlir_ciface_merlin_host_g{group}(a0, a1, a2, out);" in host["source"]
    assert f"GM_LOCAL {group} mismatches" in host["source"] and f"GM_HOSTIN {group}" in host["source"]
    vendor = driver.dispatch.render_dispatch(
        [dispatch], {group: {"on": "vendor", "path": "OS", "zero_init": True}}, uart=uart
    )
    assert "tiled_matmul_auto(" in vendor["source"] and ", OS);" in vendor["source"]
    assert "GM_LOCAL" not in vendor["source"]  # a timing build carries no check
    # The library's CPU matmul has no full-width result, so an accumulator commit cannot take it.
    cpu = driver.dispatch.render_dispatch(
        [dispatch], {group: {"on": "vendor", "path": "CPU", "zero_init": True}}, uart=uart
    )
    assert cpu["census"][0]["on"] == "host" and cpu["census"][0]["cause"] == "library_cpu_path_has_no_full_width_result"
    assert "tiled_matmul_auto(" not in cpu["source"]
    route = {"on": "package", "symbol": "k_g", "object": "/x.o", "args": ["lhs", "rhs", "dst"], "zero_init": True}
    package = driver.dispatch.render_dispatch([dispatch], {group: route}, uart=uart, row_padding=16)
    assert "k_g((void *)PA, (void *)PB, (void *)PC);" in package["source"] and package["objects"] == ["/x.o"]
    # THE KERNEL ABI'S PITCH: K = 16 is a whole tile edge and passes through; N = 12 is not, so the
    # weight goes in padded to 16 and the result comes back out of a 16-wide pitch.
    assert "const int8_t *PA = A;" in package["source"]
    assert (
        f"md_pitched(B, {K}, {N}, 16, 1)" in package["source"]
        and f"md_unpitch(C, PC, {M}, {N}, 16, 4)" in package["source"]
    )
    # With no derived edge, whether the rows need padding is unknown: refused to the host, never dense.
    unknown = driver.dispatch.render_dispatch([dispatch], {group: route}, uart=uart)
    assert unknown["census"][0]["on"] == "host" and unknown["census"][0]["cause"] == "kernel_row_padding_unknown"
    # A device route over an accumulator the IR does not state zero is not a device call.
    unstated = driver.dispatch.render_dispatch([dispatch], {group: {"on": "vendor", "path": "CPU"}}, uart=uart)
    assert unstated["census"][0]["on"] == "host" and unstated["census"][0]["cause"] == "no_device_call_for_dispatch"
    main = driver.dispatch.render_main(uart, host["census"])
    assert "merlin_run_multi(" in main and "OUT %d" in main


def test_the_grade_gates_on_local_checks_and_the_reference_arm_and_reports_the_rest() -> None:
    import struct

    # The grade reads the console through the verdict's parser (phase-2 port); run once it is present.
    pytest.importorskip("merlin.perf.whole_model_verdict")

    golden = [0.5, -1.0, 2.0]
    bits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in golden]
    expectations = {
        "groups": [18, 22],
        "golden": golden,
        "oracle_output": golden,
        "numeric_policy": {"atol": 0.03125, "rtol": 0.02},
        "hostin_bound": 1,
    }
    good = "\n".join(
        [
            "GM_LOCAL 18 mismatches=0 of=4 first=-1",
            "GM_LOCAL 22 mismatches=0 of=4 first=-1",
            "GM_HOSTIN 18 max_abs=1 over=0 of=64 bound=1",
            "GM_HOSTIN 22 max_abs=0 over=0 of=64 bound=1",
            "GM_OUTPUT within=3 of=3",
            "OUT 2 " + " ".join(str(b) for b in bits[:2]),
        ]
    )
    good += "\nGM_OUTPUT_DIGEST bytes=12 digest=77\n"
    reference = {"key": "k", "output_digest": {"bytes": 12, "digest": 77}, "words": {}}
    verdict = WO.grade(good, expectations, reference=reference)
    assert verdict["quotable"] and verdict["output"]["within_policy_of_oracle"] == 3
    assert verdict["output"]["printed_prefix"] == 2
    # The oracle's policy check is REPORTED: dynamic quantization's rounding makes it fail faithful runs.
    short = good.replace("GM_OUTPUT within=3 of=3", "GM_OUTPUT within=2 of=3")
    assert WO.grade(short, expectations, reference=reference)["quotable"]
    missing = good.replace("GM_LOCAL 22 mismatches=0 of=4 first=-1\n", "")
    assert WO.grade(missing, expectations, reference=reference)["local"]["disagree_or_absent"] == ["22"]
    assert not WO.grade(missing, expectations, reference=reference)["quotable"]
    # So is the chained activation check; the reference arm's digests are what gate.
    drifted = good.replace("GM_HOSTIN 18 max_abs=1 over=0", "GM_HOSTIN 18 max_abs=3 over=2")
    assert WO.grade(drifted, expectations, reference=reference)["quotable"]
    moved = good.replace("digest=77", "digest=78")
    assert not WO.grade(moved, expectations, reference=reference)["quotable"]


def test_a_stored_weight_read_without_a_dequantize_is_laid_out_as_the_contraction_reads_it() -> None:
    """An int8 capture reads the weight's integers directly, through the transpose a linear layer has:
    the device layout is that transpose, not the stored [out, in] -- which only a square matrix hides."""
    from merlin.xdsl_dialects.lowering import group_prepack as GP

    _x, w, _ws = _inputs()
    (group,) = _device_groups(mq.parse(MODEL))
    laid = GP.as_the_contraction_reads_it(group, 1, w)
    assert laid.shape == (K, N) and np.array_equal(laid, w.T)


def test_every_build_digests_each_dispatch_result_and_the_bridge_compares_them() -> None:
    from merlin.runtime.backends import base as backends

    driver = backends.whole_model_driver(_TARGET)
    dispatch, group = _dispatch_dict()
    rendered = driver.dispatch.render_dispatch(
        [dispatch], {group: {"on": "host"}}, uart=driver.program.UART, words_helper=driver.program._WORDS_HELPER
    )
    assert f"GM_WORDS {group} bytes=" in rendered["source"] and "words_digest(md_words_at[0]" in rendered["source"]
    graded = "GM_WORDS 18 bytes=384 digest=7\nGM_WORDS 22 bytes=384 digest=9\n"
    assert WO.words_bridge(graded, graded)["bridged"]
    moved = WO.words_bridge(graded, graded.replace("digest=9", "digest=10"))
    assert not moved["bridged"] and moved["differ"] == ["22"]
    assert not WO.words_bridge(graded, "GM_WORDS 18 bytes=384 digest=7\n")["bridged"]


def test_the_kernel_bench_runs_each_package_dispatch_checked_and_the_grade_reads_it() -> None:
    """The model's device part without its host code: every package dispatch once, on seeded operands,
    projection-checked and digested; a dispatch the package does not answer is not in the bench."""
    from merlin.runtime.backends import base as backends

    driver = backends.whole_model_driver(_TARGET)
    dispatch, group = _dispatch_dict()
    route = {"on": "package", "symbol": "k_g", "object": "/x.o", "args": ["rhs", "lhs", "dst"]}
    bench = driver.dispatch.render_kernel_bench(
        [dispatch],
        {group: route},
        uart=driver.program.UART,
        words_helper=driver.program._WORDS_HELPER,
        evict_bytes=4096,
        row_padding=16,
    )
    source = bench["source"]
    assert "k_g((void *)PB, (void *)PA, (void *)PC);" in source and bench["objects"] == ["/x.o"]
    assert f"md_pitched(B, {K}, {N}, 16, 1)" in source and f"md_unpitch(C, PC, {M}, {N}, 16, 4);" in source
    unknown = driver.dispatch.render_kernel_bench([dispatch], {group: route}, uart=driver.program.UART, words_helper="")
    assert unknown["census"][0]["cause"] == "kernel_row_padding_unknown" and not unknown["census"][0]["in_bench"]
    assert f"lc_project(A, B, C, {M}, {K}, {N}, {group}ULL" in source
    assert f"GM_WORDS {group} bytes=" in source and "int main(int hart)" in source
    assert bench["macs"] == M * K * N and bench["census"][0]["in_bench"]
    skipped = driver.dispatch.render_kernel_bench(
        [dispatch], {group: {"on": "host"}}, uart=driver.program.UART, words_helper=""
    )
    assert not skipped["census"][0]["in_bench"] and skipped["macs"] == 0 and "k_g(" not in skipped["source"]

    record = {"census": {"per_group": bench["census"]}}
    ran = f"GM_GROUP {group} matmul.package 1234 sum=UNKNOWN fnv1a=UNKNOWN\nGM_WORDS {group} bytes=4 digest=7\n"
    ran += f"GM_LOCAL {group} mismatches=0 of={M} first=-1\nDONE\n"
    verdict = WO.grade_kernel_bench(ran, record, reference_uart=ran)
    assert verdict["correct"] and verdict["bracketed"]["cycles"] == 1234
    assert verdict["by_regime"][f"{M}x{K}x{N}"]["macs"] == M * K * N
    assert not WO.grade_kernel_bench(ran.replace("mismatches=0", "mismatches=1"), record)["correct"]
    assert not WO.grade_kernel_bench(ran, record, reference_uart=ran.replace("digest=7", "digest=8"))["correct"]
    assert not WO.grade_kernel_bench(ran.replace("DONE\n", ""), record)["correct"]

    # Unchecked (an elaborated-RTL run): no projection in the image; correct only by its digests
    # agreeing with a checked run of the same seeds.
    unchecked = driver.dispatch.render_kernel_bench(
        [dispatch],
        {group: route},
        uart=driver.program.UART,
        words_helper=driver.program._WORDS_HELPER,
        check=False,
        row_padding=16,
    )
    assert "lc_project(A, B, C" not in unchecked["source"] and f"GM_LOCAL {group}" not in unchecked["source"]
    plain = {"census": {"per_group": unchecked["census"]}, "projection_checked": False}
    timed = ran.replace(f"GM_LOCAL {group} mismatches=0 of={M} first=-1\n", "")
    assert WO.grade_kernel_bench(timed, plain, reference_uart=ran)["correct"]
    assert not WO.grade_kernel_bench(timed, plain)["correct"]
    assert not WO.grade_kernel_bench(timed.replace("digest=7", "digest=8"), plain, reference_uart=ran)["correct"]
    # A subset of groups: the rest are listed, not run.
    none = driver.dispatch.render_kernel_bench(
        [dispatch], {group: route}, uart=driver.program.UART, words_helper="", only=[group + 1]
    )
    assert none["census"][0]["cause"] == "not_selected" and none["macs"] == 0


def _open_log(within: int, of: int) -> str:
    return "\n".join(
        [
            "MERLIN_INVOCATIONS warmup=0 measured=1",
            "GM_GROUP 18 matmul.package 1000 sum=UNKNOWN fnv1a=UNKNOWN",
            "GM_LOCAL 18 mismatches=0 of=113 first=-1",
            "FM full model cycles: 5000",
            f"GM_OUTPUT within={within} of={of}",
            "MERLIN_WINDOW end label=whole_model_open",
        ]
    )


def test_the_verdict_grades_a_tensor_output_where_a_classifier_has_a_class() -> None:
    V = pytest.importorskip("merlin.perf.whole_model_verdict")

    expectations = V.Expectations.from_record(
        {
            "groups": {"18": {"compare": "exact", "sum": 1, "fnv1a": 2, "inputs_from": []}},
            "argmax": None,
            "output": {"elements": 1600},
        }
    )
    passed = V.judge(_open_log(1600, 1600), expectations)
    assert passed["correctness"]["status"] == "pass" and passed["objective_cycles"] == 5000
    assert passed["correctness"]["argmax"]["end_result"] == "output_within_policy"
    failed = V.judge(_open_log(1599, 1600), expectations)
    assert failed["correctness"]["status"] == "fail" and failed["objective_cycles"] is None
    missing = V.judge(_open_log(1600, 1600).replace("GM_OUTPUT within=1600 of=1600\n", ""), expectations)
    assert missing["timing_status"] == V.TIMING_REFUSED and "GM_OUTPUT" in missing["refusal"]
    with pytest.raises(V.VerdictRefusal, match="no output tensor"):
        V.Expectations.from_record({"groups": {"18": {"compare": "exact", "sum": 1, "fnv1a": 2}}})


def test_an_image_states_the_memory_it_was_laid_out_for(tmp_path) -> None:
    import subprocess

    from merlin.llvmlower import toolchain
    from merlin.runtime.backends import spike_model as SM

    source = tmp_path / "a.c"
    source.write_text("int x;\n")
    obj = tmp_path / "a.o"
    subprocess.run(
        [str(toolchain.clang()), "--target=riscv64-unknown-elf", "-c", str(source), "-o", str(obj)], check=True
    )
    image = tmp_path / "a.elf"
    subprocess.run(
        [
            str(toolchain.objcopy()),
            f"--add-symbol={SM.DRAM_BASE_SYMBOL}=0x80000000,global",
            f"--add-symbol={SM.DRAM_SPAN_SYMBOL}=0x400000000,global",
            str(obj),
            str(image),
        ],
        check=True,
    )
    assert SM.declared_memory(image) == (0x80000000, 0x400000000)
    assert SM.declared_memory(obj) is None


def test_the_service_builder_routes_an_open_model_to_the_open_build(monkeypatch, tmp_path) -> None:
    # The measurement service's builder lands with the service that calls it (phase-2 port).
    B = pytest.importorskip("merlin.perf.whole_model_builder")

    seen = {}
    monkeypatch.setattr(WO, "is_open_model", lambda capsule, target: True)
    monkeypatch.setattr(WO, "service_build", lambda package, **kw: seen.update(kw) or {"elf": "x"})
    out = B.build(
        tmp_path / "pkg", target=_TARGET, out_dir=tmp_path, model_capsule="c", machine="m", header="h", verify="local"
    )
    assert out == {"elf": "x"} and seen["verify"] == "local" and seen["protocol_keys"] == B._PROTOCOL_KEYS


def test_the_service_build_threads_chunk_ops_and_surfaces_its_stage_times(monkeypatch, tmp_path) -> None:
    """The gate asks for a chunked build through the service builder and copies ``stage_times``,
    ``object_dedup`` and ``chunks`` into its build check; the service record carries them under those
    names from the open build's own ``stage_seconds`` and ``forward_chunks``."""
    import json

    from merlin.perf import whole_model_open_service as SVC

    expectations = tmp_path / "expectations.json"
    expectations.write_text(
        json.dumps(
            {
                "groups": ["0"],
                "dispatch": {"0": {"sum": 1, "fnv1a": "0"}},
                "oracle_output": [0.0],
                "numeric_policy": {"compare": "exact"},
            }
        )
    )
    seen = {}

    def fake_build(package, capsule, **kw):
        seen.update(kw)
        return {
            "elf": "e",
            "elf_sha256": "s",
            "program": {"abi_header": {"sha256": "h"}},
            "expectations": str(expectations),
            "attribution": {"per_group": [], "counts": {}},
            "stage_seconds": {"device_part": 1.0, "chunk_forward": 0.5},
            "object_dedup": {"unique": 3},
            "forward_chunks": {"chunk_ops": 64, "chunks": 7},
        }

    monkeypatch.setattr(SVC.WO, "build", fake_build)
    monkeypatch.setattr(SVC.WO, "vector_host_hart", lambda machine: None)
    record = SVC.service_build(
        tmp_path / "pkg", target=_TARGET, out_dir=tmp_path, model_capsule="c", machine="m", header="h", chunk_ops=64
    )
    assert seen["chunk_ops"] == 64
    assert record["stage_times"] == {"device_part": 1.0, "chunk_forward": 0.5}
    assert record["object_dedup"] == {"unique": 3} and record["chunks"] == {"chunk_ops": 64, "chunks": 7}


def test_a_machine_with_no_full_width_readout_is_refused_before_the_package_is_asked(monkeypatch, tmp_path) -> None:
    """Every dispatch of an open model commits its accumulator, so a machine whose header states no
    full-width readout can run none of them. The build says so before stating the device part (which
    lowers every group -- about an hour for SmolVLA), not after."""
    import types

    from merlin.perf import whole_model_build as WMB

    root = tmp_path / "harness"
    (root / "include").mkdir(parents=True)
    (root / "include" / "gemmini_params.h").write_text("#define DIM 16\n")
    monkeypatch.setattr(WMB, "load_model_capsule", lambda path: types.SimpleNamespace(name="m"))
    monkeypatch.setattr(WMB, "machine_header", lambda machine, header, sha: None)
    monkeypatch.setattr(WMB, "_with_header", lambda recipe, header, out: types.SimpleNamespace(include_roots=[root]))
    monkeypatch.setattr(WO, "forward_arguments", lambda capsule, extra=None: ([], {}))

    def _asked(*_a, **_k):
        raise AssertionError("the device part was stated for a machine that cannot run it")

    monkeypatch.setattr(WO, "_device_part", _asked)
    with pytest.raises(WO.OpenModelError, match=WO.MACHINE_CANNOT_READ_OUT):
        WO.build(tmp_path / "pkg", "c", target=_TARGET, machine="narrow", header=tmp_path / "h", out=tmp_path / "o")


def test_a_two_hart_program_hands_each_device_dispatch_to_the_units_hart() -> None:
    """The host code runs on a vector hart; only the unit's hart issues the unit's instructions. Each
    device route's body is handed to that hart through the mailbox and timed from the host's hart; a
    host route runs where the host code does, and main gives each hart its role."""
    from merlin.runtime.backends import base as backends

    driver = backends.whole_model_driver(_TARGET)
    dispatch, group = _dispatch_dict()
    uart = driver.program.UART
    device = driver.dispatch.render_dispatch(
        [dispatch], {group: {"on": "vendor", "path": "OS", "zero_init": True}}, uart=uart, unit_rpc=True
    )["source"]
    assert f"static void md_body_g{group}(void **md_v)" in device and f"md_rpc(md_body_g{group}, md_v);" in device
    assert "void merlin_unit_serve_on_own_stack(void)" in device and "tail merlin_unit_serve" in device
    wrapper = device[device.index(f"void _mlir_ciface_merlin_dispatch_g{group}(") :]
    assert "md_cycles()" in wrapper.split("}", 1)[0]
    host = driver.dispatch.render_dispatch([dispatch], {group: {"on": "host"}}, uart=uart, unit_rpc=True)["source"]
    assert f"md_body_g{group}" not in host and f"_mlir_ciface_merlin_host_g{group}(a0, a1, a2, out);" in host
    main = driver.dispatch.render_main(uart, [], host_hart=1, unit_hart=0)
    assert "if (hart == 0) merlin_unit_serve_on_own_stack();" in main and "if (hart != 1) for (;;);" in main
    assert "if (hart != 0) for (;;);" in driver.dispatch.render_main(uart, [])
    with pytest.raises(ValueError, match="unit hart"):
        driver.dispatch.render_main(uart, [], host_hart=1, unit_hart=1)


def test_the_hart_roles_come_from_the_registry_and_are_refused_when_unstated(monkeypatch) -> None:
    from merlin.perf import whole_model_headers as H

    declared = [{"hart": 0, "isa": "rv64gc", "unit": True}, {"hart": 1, "isa": "rv64gcv", "unit": False, "vlen": 128}]
    monkeypatch.setattr(H, "machine_harts", lambda machine: declared if machine == "two" else [])
    assert WO.two_harts("two", 1) == {"host": 1, "unit": 0, "host_isa": "rv64gcv", "unit_isa": "rv64gc", "count": 2}
    for machine, hart, why in (
        ("one", 1, "declares no harts"),
        ("two", 0, "the unit hart"),
        ("two", 5, "not declared"),
    ):
        with pytest.raises(WO.OpenModelError, match=why):
            WO.two_harts(machine, hart)


def test_the_registry_states_the_shuttle_boards_harts() -> None:
    from merlin.perf import whole_model_headers as H

    harts = H.machine_harts("gemmini_shuttle_opu_u250_firesim_bitstream")
    assert [(h["hart"], h["unit"]) for h in harts] == [(0, True), (1, False)]
    assert "v" not in harts[0]["isa"][4:] and "v" in harts[1]["isa"][4:]


def test_only_the_host_code_may_carry_a_vector_extension_the_units_hart_lacks(monkeypatch, tmp_path) -> None:
    from merlin.runtime.backends import spike_model as SM

    arch = {"host.o": ["rv64i", "m", "v", "zvl128b"], "kernel.o": ["rv64i", "m"], "stray.o": ["rv64i", "zve32x"]}
    monkeypatch.setattr(SM, "arch_extensions", lambda path: arch[Path(path).name])
    objs = [tmp_path / name for name in arch]
    checked = WO.check_unit_hart_code(objs[:2], host_objects=[objs[0]], unit_isa="rv64gc")
    assert checked == ["kernel.o"]
    with pytest.raises(WO.OpenModelError, match="stray.o"):
        WO.check_unit_hart_code(objs, host_objects=[objs[0]], unit_isa="rv64gc")
    assert WO.check_unit_hart_code(objs, host_objects=[], unit_isa="rv64gcv") == ["host.o", "kernel.o", "stray.o"]


def test_a_simulator_reads_an_images_isa_by_name_without_versions() -> None:
    from merlin.runtime.backends import spike_model as SM

    assert [SM._extension_name(t) for t in ("rv64i2p1", "m2p0", "zvl128b1p0", "zicsr2p0", "v1p0")] == [
        "rv64i",
        "m",
        "zvl128b",
        "zicsr",
        "v",
    ]


def test_the_service_runs_host_code_on_a_declared_vector_hart_and_nowhere_else(monkeypatch) -> None:
    from merlin.perf import whole_model_headers as H

    declared = {
        "two": [{"hart": 0, "isa": "rv64gc", "unit": True}, {"hart": 1, "isa": "rv64gcv", "unit": False}],
        "scalar_pair": [{"hart": 0, "isa": "rv64gc", "unit": True}, {"hart": 1, "isa": "rv64gc", "unit": False}],
    }
    monkeypatch.setattr(H, "machine_harts", lambda machine: declared.get(machine, []))
    assert WO.vector_host_hart("two") == 1
    assert WO.vector_host_hart("scalar_pair") is None and WO.vector_host_hart("undeclared") is None


def test_a_statement_reads_the_modules_contraction_extents_once_and_states_the_same_groups() -> None:
    """Stating every group of a model used to walk the whole module per group (quadratic: 69% of a
    1551-group statement). The table is read once and handed in; a root it does not hold is read afresh,
    so the entry is the one a per-group walk states."""
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import group_command as GC
    from merlin.xdsl_dialects.lowering.group_prepack import module_extents

    module = mq.parse(MODEL)
    (group,) = _device_groups(module)
    table = module_extents([group])
    assert id(group.root) in table
    walked = GC.program(group).entry
    assert GC.program(group, extents=table).entry == walked
    assert GC.program(group, extents={}).entry == walked
    assert CG.capsule_entry(group, extents={})["K"] == K


def test_host_code_units_compile_concurrently_and_reuse_an_unchanged_object(tmp_path, monkeypatch) -> None:
    """The host code is compiled as independent units (the seam a chunked forward plugs into): every
    unit at once, in the units' order, and an unchanged unit's object is reused by its digest."""
    import threading
    import time
    from types import SimpleNamespace

    from merlin.llvmlower import lower as LOWER

    active, peak, lowered, features_seen, layouts_seen = [0], [0], [], [], []
    lock = threading.Lock()

    def lower_model(text, work, *, targets, textual, features=None, data_layout=None):
        with lock:
            features_seen.append(features)
            layouts_seen.append(data_layout)
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            lowered.append(text)
        time.sleep(0.2)
        with lock:
            active[0] -= 1
        work.mkdir(parents=True, exist_ok=True)
        (work / "m.ll").write_text(text)
        return SimpleNamespace(ll_path=work / "m.ll")

    compiled = []

    def run(argv, *, cwd=None, timeout=None):
        compiled.append(list(map(str, argv)))
        Path(argv[argv.index("-o") + 1]).write_bytes(b"obj")

    from merlin.llvmlower import target_data_layout

    monkeypatch.setattr(LOWER, "lower_model", lower_model)
    monkeypatch.setattr(target_data_layout, "of", lambda clang, flags: f"layout-for:{' '.join(flags)}")
    monkeypatch.setattr(WO, "_run", run)
    monkeypatch.setattr(WO, "_check_host_interface", lambda path, dispatches: None)
    units = [("main", "a"), ("host", "b"), ("chunk2", "c")]
    objects = WO.compile_host_units(units, tmp_path, cross=["-O2"], dispatches=[], compile_timeout=60, jobs=3)
    assert [o.name for o in objects] == ["model_main.o", "model_host.o", "model_chunk2.o"]
    assert peak[0] == 3 and sorted(lowered) == ["a", "b", "c"]
    # Every unit is lowered with the open build's host-code features (exact rounding as intrinsics).
    assert features_seen == [frozenset(WO.HOST_LOWERING_FEATURES)] * 3
    # ...and under the cross target's own data layout, asked of the compiler with the same flags.
    assert layouts_seen == ["layout-for:-O2"] * 3
    # ...and compiled with the host-code optimizer options beside the cross flags.
    assert all(argv[1 : 2 + len(WO.HOST_CODE_OPTIONS)] == ["-O2", *WO.HOST_CODE_OPTIONS] for argv in compiled)
    lowered.clear()
    WO.compile_host_units(units, tmp_path, cross=["-O2"], dispatches=[], compile_timeout=60, jobs=3)
    assert lowered == [], "an unchanged unit is not lowered again"


_CHUNK_ID = "affine_map<(d0) -> (d0)>"


def _chunk_test_module() -> str:
    """``forward(x) = 9x^2 + 3x``, five ops chained so a value crosses several chunk boundaries:
    ``a`` (chunk 0) feeds ``d`` (chunk 3), and ``x`` (the function's own argument) feeds chunks
    0, 1 and 4 -- exactly the two shapes ``chunk_forward`` has to thread as call arguments."""

    def op(name: str, out: str, ins: str, in_types: str, body: str) -> str:
        return (
            f"    %{out}e = tensor.empty() : tensor<4xf32>\n"
            f"    %{out} = linalg.generic {{indexing_maps = [{in_types}], "
            f'iterator_types = ["parallel"]}} ins({ins}) outs(%{out}e : tensor<4xf32>) {{\n'
            f"    {body}\n"
            f"    }} -> tensor<4xf32>"
        )

    id1 = f"{_CHUNK_ID}, {_CHUNK_ID}"
    id2 = f"{_CHUNK_ID}, {_CHUNK_ID}, {_CHUNK_ID}"
    lines = [
        "module {",
        "  func.func @forward(%x: tensor<4xf32>) -> tensor<4xf32> {",
        op(
            "a",
            "a",
            "%x : tensor<4xf32>",
            id1,
            "^bb0(%p: f32, %o: f32):\n      %r = arith.addf %p, %p : f32\n      linalg.yield %r : f32",
        ),
        op(
            "b",
            "b",
            "%a, %x : tensor<4xf32>, tensor<4xf32>",
            id2,
            "^bb0(%p0: f32, %p1: f32, %o: f32):\n      %r = arith.addf %p0, %p1 : f32\n      linalg.yield %r : f32",
        ),
        op(
            "c",
            "c",
            "%b, %b : tensor<4xf32>, tensor<4xf32>",
            id2,
            "^bb0(%p0: f32, %p1: f32, %o: f32):\n      %r = arith.mulf %p0, %p1 : f32\n      linalg.yield %r : f32",
        ),
        op(
            "d",
            "d",
            "%c, %a : tensor<4xf32>, tensor<4xf32>",
            id2,
            "^bb0(%p0: f32, %p1: f32, %o: f32):\n      %r = arith.addf %p0, %p1 : f32\n      linalg.yield %r : f32",
        ),
        op(
            "f",
            "f",
            "%d, %x : tensor<4xf32>, tensor<4xf32>",
            id2,
            "^bb0(%p0: f32, %p1: f32, %o: f32):\n      %r = arith.addf %p0, %p1 : f32\n      linalg.yield %r : f32",
        ),
        "    func.return %f : tensor<4xf32>",
        "  }",
        "}",
    ]
    return "\n".join(lines)


def test_chunk_forward_computes_the_same_program_with_many_small_functions() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
    want = 9.0 * x * x + 3.0 * x

    (before,) = LN.evaluate(mq.parse(_chunk_test_module()), [x])
    np.testing.assert_allclose(before, want)

    module = mq.parse(_chunk_test_module())
    made = WO.chunk_forward(module, chunk_ops=2)
    assert made == 5  # 10 body ops (5 linalg.generic + 5 tensor.empty), chunk_ops=2 -> 5 chunks
    fn = next(op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward")
    # forward is now just calls, in order, plus the terminator.
    kinds = [op.name for op in fn.body.blocks[0].ops]
    assert kinds == ["func.call"] * 5 + ["func.return"]
    chunk_fns = sorted(op.sym_name.data for op in module.walk() if op.name == "func.func" and op is not fn)
    assert chunk_fns == [f"merlin_forward_chunk_{i}" for i in range(5)]

    (after,) = LN.evaluate(module, [x])
    np.testing.assert_allclose(after, want)  # the chunked program computes the identical result


def test_chunk_forward_is_a_no_op_when_the_block_already_fits() -> None:
    module = mq.parse(_chunk_test_module())
    assert WO.chunk_forward(module, chunk_ops=1000) == 0
    fn = next(op for op in module.walk() if op.name == "func.func")
    assert len([op for op in fn.body.blocks[0].ops if op.name == "linalg.generic"]) == 5


def test_the_forward_body_size_is_what_auto_chunking_judges() -> None:
    """The size ``chunk_ops="auto"`` resolves against: the forward's top-level ops, terminator excluded.
    Ten ops fit in one default chunk, so ``auto`` leaves this forward whole."""
    module = mq.parse(_chunk_test_module())
    assert WO.forward_body_size(module) == 10
    assert WO.resolve_chunk_ops("auto", forward_ops=WO.forward_body_size(module)) is None


def test_the_open_build_refuses_a_misspelled_chunk_size_before_building(tmp_path) -> None:
    with pytest.raises(WO.OpenModelError, match="positive op count"):
        WO.build(None, tmp_path / "capsule", target=_TARGET, machine="m", header="h", out=tmp_path, chunk_ops="1k")


def test_chunk_forward_threads_multiple_live_values_across_an_odd_boundary() -> None:
    """``chunk_ops=3`` lands its naive cuts mid-pair (``body_ops[2]`` is a ``tensor.empty``, pulled
    back to keep it with its own consuming ``linalg.generic``), so the real chunk sizes differ from
    the naive ones and more than one value still crosses a boundary."""
    x = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
    want = 9.0 * x * x + 3.0 * x
    module = mq.parse(_chunk_test_module())
    made = WO.chunk_forward(module, chunk_ops=3)
    assert made == 4  # bounds pulled back from the naive 3/6/9 to 2/6/8 -- still 4 chunks
    (after,) = LN.evaluate(module, [x])
    np.testing.assert_allclose(after, want)


def test_chunk_bounds_never_ends_a_chunk_on_a_cheap_producer() -> None:
    """A ``tensor.empty``/``arith.constant``/``tensor.splat``/``linalg.fill`` is never left as the
    last op of a chunk: one-shot-bufferize does not handle a destination-passing-style producer
    crossing a function-call boundary from its own write, and every chunk_ops tried on a real
    quantized capture (14 to 835 resulting chunks) segfaulted the upstream pipeline until bounds
    were pulled back over a trailing run of these -- independent of chunk size, which is what named
    the cut's POSITION as the cause rather than its count."""
    module = mq.parse(_chunk_test_module())
    fn = next(op for op in module.walk() if op.name == "func.func")
    body_ops = list(fn.body.blocks[0].ops)[:-1]  # drop func.return
    assert [op.name for op in body_ops].count("tensor.empty") == 5  # one before each generic
    for chunk_ops in (1, 2, 3, 4, 5, 6, 7, 1000):
        bounds = WO._chunk_bounds(body_ops, chunk_ops) + [len(body_ops)]
        for lo, hi in zip(bounds, bounds[1:], strict=False):
            if hi < len(body_ops):  # the LAST chunk has no following chunk to protect
                assert body_ops[hi - 1].name not in WO._CHEAP_PRODUCERS, (chunk_ops, lo, hi)


def _dynamic_live_out_module() -> str:
    """``forward(x) = 2x + x`` where ``x`` is also viewed as a ``tensor<?xf32>`` at the top and read
    back, element by element, inside a LATER generic's body -- the shape of SmolVLA's compaction loop,
    whose ``tensor<?xi64>`` is consumed by a ``tensor.extract`` nested in a generic several ops on."""
    ident = f"{_CHUNK_ID}, {_CHUNK_ID}"
    return "\n".join(
        [
            "module {",
            "  func.func @forward(%x: tensor<4xf32>) -> tensor<4xf32> {",
            "    %d = tensor.cast %x : tensor<4xf32> to tensor<?xf32>",
            "    %ae = tensor.empty() : tensor<4xf32>",
            f'    %a = linalg.generic {{indexing_maps = [{ident}], iterator_types = ["parallel"]}} '
            "ins(%x : tensor<4xf32>) outs(%ae : tensor<4xf32>) {",
            "    ^bb0(%p: f32, %o: f32):",
            "      %r = arith.addf %p, %p : f32",
            "      linalg.yield %r : f32",
            "    } -> tensor<4xf32>",
            "    %be = tensor.empty() : tensor<4xf32>",
            f'    %b = linalg.generic {{indexing_maps = [{ident}], iterator_types = ["parallel"]}} '
            "ins(%a : tensor<4xf32>) outs(%be : tensor<4xf32>) {",
            "    ^bb0(%p: f32, %o: f32):",
            "      %i = linalg.index 0 : index",
            "      %v = tensor.extract %d[%i] : tensor<?xf32>",
            "      %r = arith.addf %p, %v : f32",
            "      linalg.yield %r : f32",
            "    } -> tensor<4xf32>",
            "    %ce = tensor.empty() : tensor<4xf32>",
            f'    %c = linalg.generic {{indexing_maps = [{ident}], iterator_types = ["parallel"]}} '
            "ins(%b : tensor<4xf32>) outs(%ce : tensor<4xf32>) {",
            "    ^bb0(%p: f32, %o: f32):",
            "      %r = arith.mulf %p, %p : f32",
            "      linalg.yield %r : f32",
            "    } -> tensor<4xf32>",
            "    func.return %c : tensor<4xf32>",
            "  }",
            "}",
        ]
    )


def test_chunk_forward_never_returns_a_dynamically_shaped_value() -> None:
    """buffer-results-to-out-params cannot make an out-param of a ``memref<?x...>`` result, so a
    dynamically shaped value never crosses a chunk boundary: its producer and every reader (here one
    nested in a later generic's body) share a chunk, at every chunk size, and the program is unchanged."""
    from merlin.perf import whole_model_chunks as WC

    x = np.array([1.0, -2.0, 3.5, 0.25], np.float32)
    (want,) = LN.evaluate(mq.parse(_dynamic_live_out_module()), [x])
    np.testing.assert_array_equal(want, (3.0 * x) * (3.0 * x))
    for chunk_ops in range(1, 8):
        module = mq.parse(_dynamic_live_out_module())
        made = WO.chunk_forward(module, chunk_ops=chunk_ops)
        chunks = [op for op in module.walk() if op.name == "func.func" and op.sym_name.data != "forward"]
        assert len(chunks) == made
        for fn in chunks:
            ret = list(fn.body.blocks[0].ops)[-1]
            assert not any(WC._dynamically_shaped(v) for v in ret.operands), (chunk_ops, fn.sym_name.data)
        (after,) = LN.evaluate(module, [x])
        assert after.tobytes() == want.tobytes(), chunk_ops
    # The rule moved cuts: at chunk_ops=1 the naive 7 single-op chunks would hand %d across five cuts.
    body = list(
        next(op for op in mq.parse(_dynamic_live_out_module()).walk() if op.name == "func.func").body.blocks[0].ops
    )
    bounds = WO._chunk_bounds(body[:-1], 1)
    assert 1 not in bounds and 2 not in bounds and 3 not in bounds and 4 not in bounds


def test_chunk_forward_is_bit_identical_on_a_small_model() -> None:
    """A real captured model (the micro model capsule: int8 contractions, quantize/dequantize and a
    float tail), chunked at several sizes, computes exactly the bytes the unchunked program does."""
    from merlin.common.paths import repo_root

    text = (repo_root() / "merlin/contract/capsules/model/SY_micro_model/capsule.interface.mlir").read_text()
    rng = np.random.default_rng(0)
    args = [rng.integers(-128, 128, (32, 32), dtype=np.int8) for _ in range(4)]
    args.append(rng.standard_normal((32, 32)).astype(np.float32))
    reference = mq.parse(text)
    WO.normalize(reference)  # the builder chunks the normalized host module, never the raw capture
    (want,) = LN.evaluate(reference, args)
    for chunk_ops in (3, 16, 50):
        module = mq.parse(text)
        WO.normalize(module)
        assert WO.chunk_forward(module, chunk_ops=chunk_ops) > 1
        (after,) = LN.evaluate(module, args)
        assert after.dtype == want.dtype and after.tobytes() == want.tobytes(), chunk_ops


def test_every_harness_file_the_open_build_compiles_is_shipped() -> None:
    """The open build compiles and identifies its program from these harness files; one missing
    surfaces only at the end of a whole-model build (measured: after 23 minutes of SmolVLA)."""
    from merlin.runtime.backends import spike_model as SM

    missing = [name for name in WO.HARNESS_SOURCES if not (SM._harness_dir() / name).is_file()]
    assert not missing, missing


def test_independent_compiles_run_at_once_and_a_failure_is_still_raised(monkeypatch) -> None:
    import threading
    import time

    running, peak, lock = [0], [0], threading.Lock()

    def fake_run(argv, *, cwd=None, timeout=None):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.2)
        with lock:
            running[0] -= 1
        if argv[0] == "bad":
            raise WO.OpenModelError("bad failed")

    monkeypatch.setattr(WO, "_run", fake_run)
    WO.run_concurrently([["a"], ["b"], ["c"]], jobs=3)
    assert peak[0] == 3
    with pytest.raises(WO.OpenModelError, match="bad failed"):
        WO.run_concurrently([["a"], ["bad"]], jobs=2)


def _view_live_out_module() -> str:
    """``forward(x) = (2x) * (2x)`` where the row value ``s = 2x`` AND a reshape round trip of it are
    both read after the point a cut can fall -- after canonicalization the two are one value."""
    ident = f"{_CHUNK_ID}, {_CHUNK_ID}"
    pair = f"{_CHUNK_ID}, {_CHUNK_ID}, {_CHUNK_ID}"
    return "\n".join(
        [
            "module {",
            "  func.func @forward(%x: tensor<4xf32>) -> tensor<4xf32> {",
            "    %se = tensor.empty() : tensor<4xf32>",
            f'    %s = linalg.generic {{indexing_maps = [{ident}], iterator_types = ["parallel"]}} '
            "ins(%x : tensor<4xf32>) outs(%se : tensor<4xf32>) {",
            "    ^bb0(%p: f32, %o: f32):",
            "      %r = arith.addf %p, %p : f32",
            "      linalg.yield %r : f32",
            "    } -> tensor<4xf32>",
            "    %v1 = tensor.expand_shape %s [[0, 1]] output_shape [4, 1] : tensor<4xf32> into tensor<4x1xf32>",
            "    %v2 = tensor.collapse_shape %v1 [[0, 1]] : tensor<4x1xf32> into tensor<4xf32>",
            "    %c = arith.constant 0.0 : f32",
            "    %ze = tensor.splat %c : tensor<4xf32>",
            "    %pe = tensor.empty() : tensor<4xf32>",
            f'    %p = linalg.generic {{indexing_maps = [{pair}], iterator_types = ["parallel"]}} '
            "ins(%s, %v2 : tensor<4xf32>, tensor<4xf32>) outs(%pe : tensor<4xf32>) {",
            "    ^bb0(%a: f32, %b: f32, %o: f32):",
            "      %r = arith.mulf %a, %b : f32",
            "      linalg.yield %r : f32",
            "    } -> tensor<4xf32>",
            "    %qe = tensor.empty() : tensor<4xf32>",
            f'    %q = linalg.generic {{indexing_maps = [{pair}], iterator_types = ["parallel"]}} '
            "ins(%p, %ze : tensor<4xf32>, tensor<4xf32>) outs(%qe : tensor<4xf32>) {",
            "    ^bb0(%a: f32, %b: f32, %o: f32):",
            "      %r = arith.addf %a, %b : f32",
            "      linalg.yield %r : f32",
            "    } -> tensor<4xf32>",
            "    func.return %q : tensor<4xf32>",
            "  }",
            "}",
        ]
    )


def test_chunk_forward_never_returns_a_view_or_a_constant() -> None:
    """A reshape view, a constant or an empty destination never leaves a chunk as a result: a later
    reader gets the op re-created after the call. Returned, ``s`` and ``collapse(expand(s))`` fold to
    one buffer that buffer-results-to-out-params (hoisting static allocations) frees once per returned
    copy -- a segfault on SmolVLA int8full. Every chunk size computes the unchanged program."""
    x = np.array([1.0, -2.0, 3.5, 0.25], np.float32)
    (want,) = LN.evaluate(mq.parse(_view_live_out_module()), [x])
    np.testing.assert_array_equal(want, (2.0 * x) * (2.0 * x))
    for chunk_ops in range(1, 10):
        module = mq.parse(_view_live_out_module())
        WO.chunk_forward(module, chunk_ops=chunk_ops)
        for fn in (op for op in module.walk() if op.name == "func.func" and op.sym_name.data != "forward"):
            ret = list(fn.body.blocks[0].ops)[-1]
            owners = [getattr(v.owner, "name", None) for v in ret.operands]
            assert not set(owners) & {
                "tensor.expand_shape",
                "tensor.collapse_shape",
                "arith.constant",
                "tensor.splat",
                "tensor.empty",
            }, (chunk_ops, owners)
            assert len({id(v) for v in ret.operands}) == len(ret.operands)
        (after,) = LN.evaluate(module, [x])
        assert after.tobytes() == want.tobytes(), chunk_ops


def test_the_functional_arena_is_what_one_run_allocates(tmp_path, monkeypatch) -> None:
    """The bare-metal allocator never frees within a forward, so the functional map's arena is the sum of
    the host code's allocations (plus alignment and margin), not a fixed size a larger model runs past."""
    for name, body in (
        (
            "main",
            "declare ptr @malloc(i64)\n\ndefine void @forward(ptr %0) {\n  %2 = call ptr @malloc(i64 3000000000)\n  ret void\n}\n",
        ),
        (
            "host",
            "declare ptr @malloc(i64)\n\ndefine void @group_0(ptr %0) {\n  %2 = call ptr @malloc(i64 %n)\n  ret void\n}\n",
        ),
    ):
        (tmp_path / f"lower_{name}").mkdir()
        (tmp_path / f"lower_{name}" / "model.ll").write_text(body)
    monkeypatch.setattr(WO, "FUNCTIONAL_ARENA_BYTES", None)
    arena = WO.functional_arena_bytes(tmp_path, ["main"])
    assert arena["basis"] == "heap_demand"
    assert arena["bytes"] >= 3_000_000_000 + WO._ARENA_MARGIN_BYTES and arena["bytes"] % (1 << 20) == 0
    assert arena["bytes"] < 3_000_000_000 + WO._ARENA_MARGIN_BYTES + (2 << 20)
    # A run-time-sized allocation gets headroom under a STATED assumption (the largest static one)...
    both = WO.functional_arena_bytes(tmp_path, ["main", "host"])
    assert both["dynamic_headroom"]["sites"] == 1
    assert both["dynamic_headroom"]["bytes"] == 3_000_000_000 + WO._ARENA_ALIGN
    assert "assumption" in both["dynamic_headroom"]
    # ...while an allocation in a loop has no bound at all, and the build refuses unless set explicitly.
    (tmp_path / "lower_host" / "model.ll").write_text(
        "declare ptr @malloc(i64)\n\ndefine void @group_0(ptr %0) {\n  br label %l\nl:\n"
        "  %2 = call ptr @malloc(i64 64)\n  br i1 %c, label %l, label %d\nd:\n  ret void\n}\n"
    )
    with pytest.raises(WO.OpenModelError, match="no static sum bounds"):
        WO.functional_arena_bytes(tmp_path, ["main", "host"])
    monkeypatch.setattr(WO, "FUNCTIONAL_ARENA_BYTES", 24 << 30)
    assert WO.functional_arena_bytes(tmp_path, ["main", "host"]) == {
        "bytes": 24 << 30,
        "basis": "FUNCTIONAL_ARENA_BYTES",
        "demand": None,
    }


def test_the_functional_arena_also_holds_what_the_dispatch_layer_allocates(tmp_path, monkeypatch) -> None:
    """The C dispatch layer allocates each device dispatch's result and may copy its operands, contiguous
    and row-padded, outside the lowered host code. Measured: a float SigLIP layer sized on the host code
    alone ran out of arena at its 132nd allocation."""
    (tmp_path / "lower_main").mkdir()
    (tmp_path / "lower_main" / "model.ll").write_text("define void @forward() {\n  ret void\n}\n")
    monkeypatch.setattr(WO, "FUNCTIONAL_ARENA_BYTES", None)
    d = WO.Dispatch(
        group=5,
        dispatch_symbol="merlin_dispatch_g5",
        host_symbol="merlin_host_g5",
        arguments=(((1024, 768), "i8"), ((768, 10), "i8")),
        result=((1024, 10), "i32"),
        root_operands=(0, 1),
    )
    arena = WO.functional_arena_bytes(tmp_path, ["main"], dispatches=[d], row_multiple=16)
    # Rows padded to the tile edge (10 -> 16), each value twice, plus alignment.
    padded = 1024 * 768 + 768 * 16 + 1024 * 16 * 4
    assert arena["dispatch_bound"] == 2 * padded + 2 * 3 * WO._ARENA_ALIGN
    assert arena["bytes"] >= arena["dispatch_bound"] + WO._ARENA_MARGIN_BYTES
    unknown = WO.Dispatch(5, "d", "h", (((1024, -1), "i8"),), ((1024, 10), "i32"), (0,))
    with pytest.raises(WO.OpenModelError, match="dispatches"):
        WO.functional_arena_bytes(tmp_path, ["main"], dispatches=[unknown])


def test_lowering_identity_follows_the_lowering_source_and_its_switches(tmp_path):
    """A host object is reused only under the same lowering: an edited pass or prelude, or a flipped
    ``MERLIN_*`` switch the lowering reads, changes the identity its cache key carries."""
    from merlin.perf.whole_model_open import lowering_identity

    root = tmp_path / "llvmlower"
    (root / "sub").mkdir(parents=True)
    (root / "pipeline.py").write_text('import os\nGUARD = os.environ.get("MERLIN_SOME_GUARD")\n')
    (root / "sub" / "prelude.py").write_text("RUNNER_PRELUDE = 'a'\n")
    base = lowering_identity(root, {})
    assert lowering_identity(root, {}) == base
    assert lowering_identity(root, {"MERLIN_UNREAD": "1"}) == base, "a switch the lowering never names"
    assert lowering_identity(root, {"MERLIN_SOME_GUARD": "1"}) != base
    (root / "sub" / "prelude.py").write_text("RUNNER_PRELUDE = 'b'\n")
    assert lowering_identity(root, {}) != base


def test_the_host_object_key_carries_the_lowering_identity():
    import inspect

    from merlin.perf import whole_model_open as WO

    source = inspect.getsource(WO.compile_host_units)
    assert "lowering_identity()" in source and "identity," in source
