"""SpacemiT K1 adapter (merlin.rvvgen.k1) — fail-closed units + an opt-in real-board run.

Board-free units lock in the safety contract that keeps the coupled runner honest:
``available()`` is False without a configured host (so the runner records K1 ``not_run``, never a
false pass), the SpacemiT cross-toolchain resolves, and the generated Linux harness carries the
exact OUT/METRIC/DONE markers ``zephyr_model._parse_console`` consumes. The end-to-end cross-compile
+ deploy + run is gated on the board actually being reachable (``available()``), and skipped
otherwise — it requires real silicon and the SpacemiT toolchain.
"""
import os

import pytest

from merlin.rvvgen import k1


def test_available_fail_closed_without_host(monkeypatch):
    # No MERLIN_K1_HOST configured -> unavailable (never a false pass), regardless of toolchain.
    monkeypatch.setattr(k1, "K1_HOST", "")
    assert k1.available() is False


def test_toolchain_cc_resolves_or_none():
    # toolchain_cc() either resolves to a real clang/gcc file, or returns None (fail-closed) — it
    # never returns a path that does not exist.
    cc = k1.toolchain_cc()
    assert cc is None or cc.is_file()


def test_main_linux_template_has_markers():
    src = k1.main_linux_c()
    # The host parser (zephyr_model._parse_console) keys on these exact markers.
    assert 'printf("OUT %d"' in src
    assert 'printf("METRIC cycles %llu' in src
    assert 'printf("DONE' in src
    # K1-specific: vlenb probe + rdtime timing (this kernel traps userspace rdcycle), so the
    # rdcycle INSTRUCTION must not be emitted (the word may appear in an explanatory comment).
    assert "vlenb" in src
    assert "rdtime" in src
    assert "rdcycle %0" not in src


def test_dispatch_timing_default_off_is_byte_identical():
    # The per-dispatch matmul-bucket timer (dispatch_timing) is default-OFF. When off the harness
    # must carry NONE of the timing surface (byte-identical baseline path); when on it declares the
    # accessor externs + prints the two extra METRIC lines the breakdown harness reads.
    off = k1.main_linux_c()
    on = k1.main_linux_c(dispatch_timing=True)
    for tok in ("matmul_ticks", "matmul_calls", "merlin_matmul"):
        assert tok not in off, f"OFF path leaked timing token {tok!r}"
    assert "METRIC matmul_ticks" in on and "METRIC matmul_calls" in on
    assert "extern unsigned long long merlin_matmul_ticks(void);" in on


def test_dispatch_timing_requires_routed_backend():
    # The matmul-bucket timer lives in the routed GEMM shim, so dispatch_timing without a
    # kernel_backend must fail loud (never silently no-op into a bucket that is always zero).
    with pytest.raises(k1.K1Error):
        k1.build_k1_binary("output/nonexistent", "/tmp/k1_dt_guard", pkg=_DummyPkg(),
                           kernel_backend=None, dispatch_timing=True)


class _DummyPkg:
    run_id = "guard"
    is_int8 = False
    schedule_text = ""
    compiler_features = ()


def test_main_linux_is_glibc_hosted():
    # The K1 harness is glibc Linux userspace: it uses stdio, NOT the bare-metal HTIF path.
    src = k1.main_linux_c()
    assert "#include <stdio.h>" in src
    assert "merlin_run(" in src


@pytest.mark.skipif(not k1.available(),
                    reason="K1 board unreachable or SpacemiT toolchain absent (set MERLIN_K1_HOST)")
def test_real_k1_matmul_end_to_end(tmp_path):
    # REAL silicon: generate a tiny matmul, cross-compile + deploy + run on the board, and assert
    # we got real measurements back (cycles estimate + raw timebase ticks + wall ns + vlen).
    from merlin.rvvgen.registry import load_rvv_package
    from merlin.rvvgen.workloads import gen_matmul_f32

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    bundle = gen_matmul_f32(tmp_path, M=64, N=64, K=64)
    pkg = load_rvv_package(os.path.join(repo, "out/artifacts/targets", "rvv", "hand_v0"))
    res = k1.run_on_k1(bundle, tmp_path / "build", pkg, timeout=600)
    m = res["metrics"]
    assert m.get("cycles") and m["cycles"] > 0
    assert m.get("time_ticks") and m["time_ticks"] > 0
    assert m.get("wall_ns") and m["wall_ns"] > 0
    assert res["vlen"] == 256  # X60 VLEN=256 bits (vlenb=32)
