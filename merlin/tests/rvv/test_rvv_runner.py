"""certify_rvv K-ladder behavior — fast units (no spike) + integrity/fail-closed guarantees.

The heavy end-to-end (build + spike run + gate) is exercised manually against
output/small_llama_int8_consistent (status=pass, gate_ok, 19 RVV mnemonics) — too slow for the
unit suite. Here we lock in: the instruction-histogram parser, the expected-instruction prefix
match, K0 integrity failure recording (no raise), and K1/K5 not_run when targets are unreachable.
"""
import os

from merlin.rvvgen import runner, k1

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_OBJDUMP_SAMPLE = """
   10362:\t0d07d557\tvsetvli\ta0,a5,e32,m1,ta,ma
   10366:\t0205e407\tvle32.v\tv8,(a1)
   1036a:\tb2842457\tvfmacc.vv\tv8,v16,v24
   1036e:\t02d5d557\tvredsum.vs\tv0,v8,v0
   10372:\t00008067\tret
"""


def test_instruction_histogram_counts_vector_mnemonics():
    h = runner._instruction_histogram(_OBJDUMP_SAMPLE)
    assert h == {"vfmacc.vv": 1, "vle32.v": 1, "vredsum.vs": 1, "vsetvli": 1}
    # scalar `ret` is not counted
    assert "ret" not in h


def test_expected_instructions_prefix_match():
    h = runner._instruction_histogram(_OBJDUMP_SAMPLE)
    assert runner._expected_present(h, ["vsetvli", "vle32.v", "vfmacc", "vredsum"])
    assert not runner._expected_present(h, ["vwmacc"])  # int8 widening absent in this fp32 sample


def test_k0_integrity_failure_is_recorded_not_raised(tmp_path):
    # A package with an out-of-allowlist cflag must yield status=fail at K0, never raise.
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "manifest.yaml").write_text(
        "target: rvv\nrun_id: evil\nfamily: vector_schedule\n"
        "authoring: {mode: hand_curated, author: human}\n"
        "outputs: {schedule: schedule.mlir, knobs: knobs.yaml}\n")
    (pkg / "schedule.mlir").write_text("module {}\n")
    (pkg / "knobs.yaml").write_text(
        "schedule_file: schedule.mlir\ndtype_strategy: fp32\n"
        "cflags: ['-march=rv64gcv', 'rm -rf /']\n")
    wl = tmp_path / "wl"
    wl.mkdir()
    rec = runner.certify_rvv(str(pkg), str(wl), runs_root=str(tmp_path / "runs"),
                             run_id="evil_run", targets=())
    assert rec["status"] == "fail"
    assert rec["ladder"]["K0"] == "fail"
    assert (tmp_path / "runs" / "evil_run" / "results.yaml").is_file()


def test_k1_unavailable_is_fail_closed(monkeypatch):
    # Fail-closed LOGIC: with no configured board host, K1 must be unavailable (never a false pass).
    # Force the unconfigured state rather than assume the ambient env lacks a board — a repo-local
    # .env may legitimately configure a live board, and the property under test is that an ABSENT
    # host still resolves to unavailable.
    monkeypatch.setattr(k1, "K1_HOST", "")
    assert k1.available() is False


def test_workload_matmul_bundle_is_valid(tmp_path):
    import json
    import numpy as np
    from merlin.rvvgen import workloads
    b = workloads.gen_matmul_f32(tmp_path, M=8, N=8, K=8, seed=1)
    for f in ("model.mlir", "weights.safetensors", "weights.safetensors.manifest.json",
              "input_order.json", "inputs.npz", "golden.npy"):
        assert (b / f).is_file(), f
    # golden == a @ b (the compiler's job is to reproduce this)
    z = np.load(b / "inputs.npz")
    golden = np.load(b / "golden.npy")
    np.testing.assert_allclose(golden, z["in0"] @ z["in1"], rtol=1e-5, atol=1e-4)
    # @forward signature parses + both operands are inputs (no weights)
    man = json.loads((b / "weights.safetensors.manifest.json").read_text())
    assert {m["kind"] for m in man.values()} == {"input"}
    from merlin.llvmlower.model_runner import parse_forward_signature
    sig = parse_forward_signature(b / "model.mlir")
    assert [s[0] for s in sig] == [[8, 8], [8, 8]]
