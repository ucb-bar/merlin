"""SpacemiT K1 adapter (merlin.mining.k1) — fail-closed units + an opt-in real-board run.

Board-free units lock in the safety contract that keeps the coupled runner honest:
``available()`` is False without a configured host (so the runner records K1 ``not_run``, never a
false pass), the SpacemiT cross-toolchain resolves, and the generated Linux harness carries the
exact OUT/METRIC/DONE markers ``zephyr_model._parse_console`` consumes. The end-to-end cross-compile
+ deploy + run is gated on the board actually being reachable (``available()``), and skipped
otherwise — it requires real silicon and the SpacemiT toolchain.
"""
import os
import subprocess
from contextlib import nullcontext

import numpy as np
import pytest
import yaml

from merlin.mining import k1


def test_available_fail_closed_without_host(monkeypatch):
    # No MERLIN_K1_HOST configured -> unavailable (never a false pass), regardless of toolchain.
    monkeypatch.setattr(k1, "K1_HOST", "")
    assert k1.available() is False


def test_toolchain_cc_resolves_or_none():
    # toolchain_cc() either resolves to a real clang/gcc file, or returns None (fail-closed) — it
    # never returns a path that does not exist.
    cc = k1.toolchain_cc()
    assert cc is None or cc.is_file()


def test_board_conditions_are_observed_not_inferred(monkeypatch):
    monkeypatch.setattr(k1, "_ssh", lambda *_args, **_kwargs: subprocess.CompletedProcess(
        [], 0,
        "governor=performance\ncurrent_khz=1600000\nmax_khz=1600000\n"
        "max_thermal_millic=41000\n", ""))
    assert k1.board_conditions() == {
        "governor": "performance", "current_khz": 1600000, "max_khz": 1600000,
        "max_thermal_millic": 41000,
    }


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
    assert "merlin_run_multi(" in src


def test_main_linux_times_complete_sessions_as_repeats():
    src = k1.main_linux_c()
    assert "MERLIN_SESSION_REPEATS" in src
    assert "MERLIN_SESSION_WARMUPS" in src
    assert "step < MERLIN_SESSION_STEPS" in src
    assert "merlin_one_session(validate_session && repeat == 0)" in src
    assert 'printf("METRIC iter_wall_ns %ld %llu\\n", repeat' in src


def test_main_linux_complete_session_template_is_valid_k1_c(tmp_path):
    cc = k1.toolchain_cc()
    if cc is None:
        pytest.skip("SpacemiT compiler unavailable")
    (tmp_path / "model_gen.h").write_text(
        '#include "merlin_model.h"\n'
        '#define MERLIN_N_ARGS 1\n#define MERLIN_N_OUTPUTS 1\n'
        '#define MERLIN_N_STATE_PAIRS 0\n#define MERLIN_SESSION_STEPS 3\n'
        '#define MERLIN_HAS_SESSION_CORRECTNESS 1\n'
        '#define MERLIN_HAS_SESSION_QUALITY 1\n#define MERLIN_OUT_ELEMS 1\n'
        '#define MERLIN_OUT_LASTDIM 1\n'
        'static const merlin_arg_t MERLIN_ARGS[1] = {{MERLIN_OUTPUT,0,1,{1},4}};\n',
        encoding="utf-8")
    (tmp_path / "model_io.h").write_text(
        'static float out[1]; static void *MERLIN_INPUT_PTR[1] = {0};\n'
        'static void *MERLIN_OUTPUT_PTR[1] = {out};\n'
        'static const int MERLIN_STATE_INPUT_ARGS[1] = {0};\n'
        'static const int MERLIN_STATE_OUTPUT_INDICES[1] = {0};\n'
        'static void merlin_reset_session(void) {}\n'
        'static void merlin_prepare_step(long x) {(void)x;}\n'
        'static void merlin_validate_step(long x) {(void)x;}\n'
        'static long merlin_correctness_steps(void) {return 3;}\n'
        'static long merlin_correctness_min_cos_ppm(void) {return 1000000;}\n'
        'static long merlin_correctness_max_rel_ppm(void) {return 0;}\n'
        'static long merlin_correctness_top1(void) {return 3;}\n'
        'static long merlin_quality_steps(void) {return 3;}\n'
        'static long merlin_quality_min_cos_ppm(void) {return 1000000;}\n'
        'static long merlin_quality_max_rel_ppm(void) {return 0;}\n'
        'static long merlin_quality_top1(void) {return 3;}\n', encoding="utf-8")
    source = tmp_path / "main.c"
    source.write_text(k1.main_linux_c(), encoding="utf-8")
    runtime_headers = k1.runtime_dir() / "c"
    proc = subprocess.run([
        str(cc), "--target=riscv64-unknown-linux-gnu", f"-march={k1.K1_MARCH}",
        f"-mabi={k1.K1_MABI}", f"-I{runtime_headers}", f"-I{tmp_path}",
        "-fsyntax-only", str(source),
    ], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_multi_session_template_is_valid_k1_c(tmp_path):
    cc = k1.toolchain_cc()
    if cc is None:
        pytest.skip("SpacemiT compiler unavailable")
    (tmp_path / "merlin_session.h").write_text('''
#include <stddef.h>
#define MERLIN_SESSION_N_PROGRAMS 2
const char *merlin_session_program_name(int);
long merlin_session_program_steps(int);
void merlin_session_reset(void);
int merlin_session_prepare_program(int);
int merlin_session_run_step(int,const void *,long,int);
long merlin_session_correctness_steps(void);
long merlin_session_correctness_min_cos_ppm(void);
long merlin_session_correctness_max_rel_ppm(void);
long merlin_session_correctness_top1(void);
long merlin_session_quality_steps(void);
long merlin_session_quality_min_cos_ppm(void);
long merlin_session_quality_max_rel_ppm(void);
long merlin_session_quality_top1(void);
void *merlin_session_quality_output(void);
size_t merlin_session_quality_output_elems(void);
long merlin_session_quality_output_lastdim(void);
''', encoding="utf-8")
    source = tmp_path / "main_session.c"
    source.write_text(k1.main_linux_session_c(), encoding="utf-8")
    proc = subprocess.run([
        str(cc), "--target=riscv64-unknown-linux-gnu", f"-march={k1.K1_MARCH}",
        f"-mabi={k1.K1_MABI}", f"-I{tmp_path}", "-fsyntax-only", str(source),
    ], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_session_backend_aliases_are_globally_renumbered_without_prefix_collisions():
    base = "merlin_xnn_gemm_f32"
    source = "\n".join(
        f"func.func private @{base}_{i}()\ncall @{base}_{i}()" for i in range(11))
    rewritten, count = k1._renumber_backend_aliases(source, base, 20)
    assert count == 11
    for i in range(11):
        assert f"@{base}_{20 + i}()" in rewritten
        assert f"@{base}_{i}()" not in rewritten


def test_arch_probe_records_source_and_parses_board_values(tmp_path, monkeypatch):
    source = tmp_path / "probe.c"
    source.write_text("int main(void) { return 0; }", encoding="utf-8")
    cc = tmp_path / "clang"
    cc.write_text("tool", encoding="utf-8")
    key = tmp_path / "key"
    key.write_text("key", encoding="utf-8")
    monkeypatch.setattr(k1, "toolchain_cc", lambda: cc)
    monkeypatch.setattr(k1, "K1_HOST", "root@board")
    monkeypatch.setattr(k1, "K1_SSH_KEY", str(key))
    monkeypatch.setattr(k1, "board_lock", lambda: nullcontext())
    monkeypatch.setattr(k1, "_run", lambda _args: None)

    def fake_ssh(*args, timeout=60):
        if args and str(args[0]).startswith("/tmp/merlin_k1_arch_probe_"):
            return subprocess.CompletedProcess(args, 0, "online_harts=8\nvlenb=32\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(k1, "_ssh", fake_ssh)
    result = k1.run_arch_probe(source)
    assert result["values"] == {"online_harts": 8, "vlenb": 32}
    assert len(result["source_sha256"]) == 64


@pytest.mark.skipif(not k1.available(),
                    reason="K1 board unreachable or SpacemiT toolchain absent (set MERLIN_K1_HOST)")
def test_real_k1_matmul_end_to_end(tmp_path):
    # REAL silicon: generate a tiny matmul, cross-compile + deploy + run on the board, and assert
    # we got real measurements back (cycles estimate + raw timebase ticks + wall ns + vlen).
    from merlin.mining.registry import load_rvv_package
    from merlin.mining.workloads import gen_matmul_f32

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    bundle = gen_matmul_f32(tmp_path, M=64, N=64, K=64)
    pkg = load_rvv_package(os.path.join(repo, "out/artifacts/targets", "rvv", "hand_v0"))
    res = k1.run_on_k1(bundle, tmp_path / "build", pkg, timeout=600)
    m = res["metrics"]
    assert m.get("cycles") and m["cycles"] > 0
    assert m.get("time_ticks") and m["time_ticks"] > 0
    assert m.get("wall_ns") and m["wall_ns"] > 0
    assert m.get("affinity_cpus") == 1
    assert res["core_count"] == res["requested_core_count"] == 1
    assert res["affinity_source"] == "sched_getaffinity"
    assert res["vlen"] == 256  # X60 VLEN=256 bits (vlenb=32)


@pytest.mark.skipif(not k1.available(),
                    reason="K1 board unreachable or SpacemiT toolchain absent")
@pytest.mark.parametrize("kernel_backend", [None, "xnnpack", "openblas"])
def test_real_k1_two_program_session_end_to_end(tmp_path, kernel_backend):
    """Real one-process proof: compiled matmul stage output feeds a second compiled stage."""
    from merlin.mining.registry import load_rvv_package
    from merlin.mining.workloads import gen_matmul_f32

    root = tmp_path / "session"
    stage = gen_matmul_f32(root, M=16, N=16, K=16)
    relative = stage.relative_to(root).as_posix()
    contract = {
        "version": 2, "kind": "test_session", "paper_ready": False,
        "stages": ["first", "second"],
        "stage_schedule": [
            {"name": "first", "steps": 1, "execution": "compiled", "timed": True},
            {"name": "second", "steps": 1, "execution": "compiled", "timed": True},
        ],
        "programs": [
            {"name": "first", "bundle": relative, "steps": 1},
            {"name": "second", "bundle": relative, "steps": 1},
        ],
        "bindings": [{
            "name": "intermediate", "from": {"program": "first", "output_index": 0},
            "to": {"program": "second", "input_arg": 0},
        }],
        "states": [], "streams": [],
        "quality": {"scope": "trajectory", "program": "second"},
    }
    (root / "session_contract.yaml").write_text(yaml.safe_dump(contract), encoding="utf-8")
    package = load_rvv_package(
        k1.repo_root() / "out" / "artifacts" / "targets" / "rvv" / "hand_v0")
    result = k1.run_on_k1(
        root, tmp_path / "build", package, timeout=600, session_repeats=3,
        kernel_backend=kernel_backend, fallback_policy="forbid", require_csr_vlen=True)
    with np.load(stage / "inputs.npz") as values:
        expected = (values["in0"] @ values["in1"]) @ values["in1"]
    np.testing.assert_allclose(
        result["prefix"], expected.ravel(), rtol=2e-4, atol=2e-4)
    assert len(result["iter_wall_ns"]) == 3
    assert set(result["stage_wall_ns"]) == {"first", "second"}
    assert all(len(values) == 3 for values in result["stage_wall_ns"].values())
    assert result["vlen"] == 256 and result["vlen_source"] == "csr"
    assert result["core_count"] == result["requested_core_count"] == 1
    assert result["affinity_source"] == "sched_getaffinity"
    if kernel_backend == "xnnpack":
        assert result["n_xnn_routed"] == 2
        assert result["n_xnn_eligible"] == 2
        assert result["n_xnn_candidates"] == 2
    elif kernel_backend == "openblas":
        assert result["n_openblas_routed"] == 2
        assert result["n_openblas_eligible"] == 2
        assert result["n_openblas_candidates"] == 2
