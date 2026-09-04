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


def test_a_cli_expert_wall_must_declare_what_it_measured():
    """A bare --expert-wall-ns leaves ExpertBaseline.mismatches() nothing to check.

    Only DECLARED fields are compared, so an undeclared baseline can never be shown to mismatch --
    and therefore can never be refused. That is how two int8 beam runs came to be scored against
    their fp32 sibling's wall and both reported beating the expert (1.269x, 1.859x) while the one
    int8 cell carrying its own number reports 0.113. Refuse at the CLI edge instead.
    """
    import argparse
    import pytest
    from merlin.mining.beam_cli import _declared_expert

    def _args(**kw):
        d = {"expert_wall_ns": None, "expert_workload": None, "expert_dtype": None,
             "expert_substrate": "k1_spacemit", "expert_note": ""}
        d.update(kw)
        return argparse.Namespace(**d)

    # no wall at all is fine -- the beam simply reports no attainment
    assert _declared_expert(_args()) is None
    # a wall without an identity is refused, and the message says which flags are missing
    with pytest.raises(SystemExit) as e:
        _declared_expert(_args(expert_wall_ns=1234.0))
    assert "--expert-workload" in str(e.value) and "--expert-dtype" in str(e.value)
    # half-declared is still refused
    with pytest.raises(SystemExit):
        _declared_expert(_args(expert_wall_ns=1234.0, expert_workload="small_llama_int8_consistent"))
    # fully declared: the baseline carries its identity and CAN now be refused on a real mismatch
    b = _declared_expert(_args(expert_wall_ns=1234.0, expert_dtype="int8",
                               expert_workload="small_llama_int8_consistent"))
    assert b.provenance_recorded
    assert b.mismatches(workload="small_llama_int8_consistent", dtype="int8") == ()
    assert b.mismatches(workload="small_llama_int8_consistent", dtype="fp32")   # dtype guard fires
    assert b.mismatches(workload="bitvla_int8_consistent", dtype="int8")        # workload guard fires


def test_the_beam_driver_declares_its_baseline_and_its_bundle():
    """Both inert guards in the autonomous driver: the bare-float baseline and ours_bundle_id=None."""
    from merlin.common.paths import repo_root
    src = (repo_root() / "build_tools" / "scripts" / "run_autonomous_beam_experiment.py").read_text()
    assert "ExpertBaseline(wall_ns=float(ref[\"wall_ns\"])" in src
    assert 'xnn = ref["wall_ns"]' not in src, "the baseline is a bare float again"
    # both executorch_cell call sites declare the bundle ours was measured on
    assert src.count("ours_bundle_id=ours_bundle_id") == 2
    assert "executorch_cell(model, dtype, root=root)" not in src


def test_a_starved_search_says_so_in_its_summary():
    """`over_width` deferrals are proposals the search NEVER TRIED, not ones it rejected.

    Without a census they are invisible unless someone opens beam_tree.yaml and counts, so a
    budget-bounded run reads exactly like a converged one. The run this was added for deferred 41 of
    44 proposals over_width across 16 distinct families -- including both cap refinements, whose
    largest rung is worth 1.34x, and fuse_transpose_b, on a model where weight transposes cost 1.61x.
    """
    from merlin.mining import beam
    src = (beam.__file__)
    text = open(src).read()
    assert "deferral_census" in text
    # the census must reach BOTH the persisted tree and the returned dict
    assert text.count("deferral_census") >= 3
    from merlin.mining.beam_cli import __file__ as cli
    assert '"deferral_census": res.get("deferral_census")' in open(cli).read()


def test_the_census_counts_reasons_and_names_the_starved_families():
    """Shape check on real deferral records: reasons are tallied and over_width families named."""
    import collections
    deferred = [
        {"reason": "over_width", "family": "wholemodel:promote_buffers_to_stack:cap"},
        {"reason": "over_width", "family": "wholemodel:fuse_transpose_b"},
        {"reason": "over_width", "family": "wholemodel:fuse_transpose_b"},
        {"reason": "illegal_on_parent", "family": "schedule:vector_sizes"},
        {"lever": "x"},  # not forkable: no reason recorded
    ]
    by_reason = collections.Counter(str(d.get("reason") or "not_forkable") for d in deferred)
    starved = sorted({str(d.get("family") or d.get("lever") or "?")
                      for d in deferred if d.get("reason") == "over_width"})
    assert by_reason["over_width"] == 3
    assert by_reason["illegal_on_parent"] == 1
    assert by_reason["not_forkable"] == 1
    # families are DEDUPLICATED -- three deferrals, two distinct starved families
    assert starved == ["wholemodel:fuse_transpose_b", "wholemodel:promote_buffers_to_stack:cap"]


def _beam_driver():
    import importlib.util
    from merlin.common.paths import repo_root
    path = repo_root() / "build_tools" / "scripts" / "run_autonomous_beam_experiment.py"
    spec = importlib.util.spec_from_file_location("_beam_driver", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bundles_are_resolved_from_disk_not_listed_in_the_driver():
    """The hardcoded (dtype, model) -> path table had all EIGHT entries pointing at directories
    that do not exist, and a stale entry does not fail: run_cell reads it, finds nothing, and
    returns not_run. The driver reported "no bundle" for every cell it shipped configured with,
    which reads as "not captured yet" rather than as a broken map."""
    mod = _beam_driver()
    assert not hasattr(mod, "_BUNDLE"), "the hardcoded bundle table is back"
    src = open(mod.__file__).read()
    assert "_bundle_for(dtype, model)" in src
    # one resolver decides what a (model, variant) means -- this driver must not be a second one
    assert "from merlin.baselines import bundle as _bundle_mod" in src


def test_a_missing_bundle_is_reported_by_name_with_what_was_looked_for():
    mod = _beam_driver()
    audit = mod._bundle_audit(["int8:small_llama", "int8:definitely_not_a_model"])
    assert audit["int8:small_llama"]["present"] is True
    assert audit["int8:small_llama"]["bundle"].endswith("small_llama_int8_consistent")
    miss = audit["int8:definitely_not_a_model"]
    assert miss["present"] is False and miss["bundle"] is None
    # the miss says what it looked for, so a rename or typo is visible without reading the source
    assert "definitely_not_a_model_int8_full" in miss["looked_for"]


def test_the_driver_refuses_a_run_where_no_cell_has_a_bundle():
    """Spending board time on a configuration that can only produce not_run is not a search."""
    src = open(_beam_driver().__file__).read()
    assert 'if not any(i["present"] for i in audit.values()):' in src
    assert "refusing to run a search" in src
