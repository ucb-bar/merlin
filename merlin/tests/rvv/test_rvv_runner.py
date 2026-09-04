"""certify_rvv K-ladder behavior — fast units (no spike) + integrity/fail-closed guarantees.

The heavy end-to-end (build + spike run + gate) is exercised manually against
output/small_llama_int8_consistent (status=pass, gate_ok, 19 RVV mnemonics) — too slow for the
unit suite. Here we lock in: the instruction-histogram parser, the expected-instruction prefix
match, K0 integrity failure recording (no raise), and K1/K5 not_run when targets are unreachable.
"""
import os

from merlin.mining import runner, k1

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
    from merlin.mining import workloads
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


def test_the_wall_carries_the_conditions_and_protocol_it_was_measured_under():
    """A wall without its conditions cannot be compared to a wall measured at another time.

    `run_on_k1` already probes `board_conditions()` before AND after every run and attaches them
    (`mining/k1.py`), and `certify_rvv` was copying only cycles/ticks/wall/vlen out of that dict --
    dropping them. That is exactly how a ~2x board-condition change went unnoticed: two beam runs of
    the BYTE-IDENTICAL frozen seed (same baseline digest 631fd07f9426) measured 349,877,321 and
    175,682,867 ns, 1.9915x apart. `speedup` is internal (fork/seed) so the factor cancelled and
    looked correct, while `attainment_vs_expert` divides by an EXTERNAL wall and absorbed all of it
    (0.634 -> 1.287 on the same winning config). Nothing in either artifact could show it.

    Fetching a value and then dropping it is indistinguishable from never fetching it, so pin the
    forwarding, not just the intent.
    """
    import inspect

    from merlin.mining import runner

    src = inspect.getsource(runner.certify_rvv)
    assert '"board_conditions": kr.get("board_conditions")' in src, (
        "the conditions run_on_k1 already probed must reach the measurement entry")
    assert '"warmup": warmup, "iters": iters' in src, (
        "the protocol that produced the wall must be recorded beside it")


def test_the_beam_node_carries_conditions_from_the_entry_the_wall_came_from():
    """Recorded on the NODE, not just the certify record, and read from the same measurement entry
    the wall was picked from -- so the conditions always describe THIS number rather than some other
    substrate's run in the same record."""
    import inspect

    from merlin.mining import beam

    src = inspect.getsource(beam._score)
    assert '"board_conditions": _cond' in src
    assert '"measurement_protocol": _proto' in src
    assert 'if _m.get("wall_ns") is not None and _m.get("wall_ns") == k1_wall' in src, (
        "conditions must come from the entry the reported wall came from")


def test_score_never_invents_conditions_for_a_wall_it_could_not_pick(tmp_path):
    """Fail-closed companion: conditions describe A NUMBER, so with no citable wall there must be
    none. `_meas.pick` refuses to report a wall when the target's measurement authority is
    undeclared (no capability contract), and the conditions must refuse with it rather than being
    attached to a wall that was never reported."""
    from merlin.kernels.compare import RvvFingerprint
    from merlin.mining import beam

    conds = {"before": {"governor": "performance"}, "after": {"governor": "performance"}}
    result = {
        "target": "k1",
        "correctness": {"gate_ok": True},
        "measurement": [{"target": "k1", "cycle_accurate": False, "cycles": 10,
                         "time_ticks": 5, "wall_ns": 12345, "vlen": 256,
                         "warmup": 2, "iters": 5, "board_conditions": conds}],
    }
    curated = RvvFingerprint(key={"op": "matmul", "dtype": "int8"}, decisions={}, histogram={},
                             source="test")
    out = beam._score(result, tmp_path, curated, {"op": "matmul", "dtype": "int8"}, target="k1")
    # undeclared authority => no wall, and therefore no conditions attributed to one
    assert out["k1_wall_ns"] is None
    assert out["board_conditions"] is None
    assert out["measurement_protocol"] is None
    assert out.get("measurement_gaps"), "an undeclared authority must say so"
