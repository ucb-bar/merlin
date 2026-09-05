"""The ours-vs-ExecuTorch campaign: what it refuses, what a row carries, and what the summary says.

Every test here defends a property the campaign exists to hold. The two that matter most are
negative: a cell that cannot be measured must produce a REFUSAL STRING and no ratio (a bare number
whose basis cannot be shown is the failure this whole workstream is about), and a cell already
recorded — measured OR refused — must not be re-run, because re-deriving a refusal costs board time
and learns nothing.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys

import numpy as np
import pytest

from merlin.common.paths import repo_root
from merlin.compare import et_campaign as ec

INSTRUMENT = repo_root() / "build_tools" / "scripts" / "k1_int8_fair_compare.py"
DRIVER = repo_root() / "build_tools" / "scripts" / "k1_int8_et_campaign.py"


# --- fixtures: synthetic capture bundles, so nothing here needs the 130 GB recapture tree --------


def _make_bundle(root, name, *, weights=1024, extra=512, w8a8=True, mlir=True, fp32=True):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "weights.safetensors").write_bytes(b"\0" * weights)
    (d / "extra.npz").write_bytes(b"\0" * extra)
    if mlir:
        (d / "model.mlir").write_text("module {}\n", encoding="utf-8")
    if fp32:
        np.save(d / "golden.npy", np.zeros(4, dtype=np.float32))
    if w8a8:
        np.save(d / "golden_w8a8.npy", np.zeros(4, dtype=np.float32))
    return d


@pytest.fixture()
def recaps(tmp_path, monkeypatch):
    """A fake recaptures root that both bundle.resolve and layout_equivalence read."""
    root = tmp_path / "recaptures"
    root.mkdir()
    from merlin.baselines import bundle as _bundle
    from merlin.common import artifacts as _art

    monkeypatch.setattr(_bundle, "recaptures_dir", lambda: root)
    monkeypatch.setattr(_art, "recaptures_dir", lambda: root)
    return root


# --- footprint: refuse offline what cannot fit, and never refuse on a guessed margin -------------


def test_footprint_is_a_lower_bound_and_fits_is_not_a_promise(recaps):
    d = _make_bundle(recaps, "m_int8_consistent", weights=100, extra=50)
    fp = ec.bundle_footprint(d, budget_bytes=1000)
    assert fp["resident_lower_bound_bytes"] == 150
    assert fp["headroom_bytes"] == 850
    assert fp["fits"] is True
    # The number counts ONLY what is certainly resident; the row must say so rather than imply the
    # cell is known to fit.
    assert "lower bound" in fp["note"]
    assert fp["parts_bytes"]["weights.safetensors"] == 100


def test_a_bundle_larger_than_the_board_is_refused_before_any_board_time(recaps):
    _make_bundle(recaps, "big_int8_consistent", weights=5_000, extra=3_000)
    plan = ec.plan_cell("big", budget_bytes=1_000, recaptures_root=recaps)
    assert not plan.runnable
    assert any("does not fit the board" in r for r in plan.refusals)
    # The refusal must name the arithmetic, not just say no.
    assert any("budget" in r for r in plan.refusals)


def test_the_budget_records_that_it_was_declared_not_measured(recaps):
    _make_bundle(recaps, "m_int8_consistent")
    plan = ec.plan_cell("m", budget_bytes=99, budget_source="declared via --board-usable-bytes",
                        recaptures_root=recaps)
    assert plan.footprint["budget_source"] == "declared via --board-usable-bytes"


# --- preflight refusals: everything the instrument would raise on, refused for free ---------------


def test_absent_bundle_refuses_loudly_rather_than_reporting_not_run(recaps):
    plan = ec.plan_cell("never_captured", recaptures_root=recaps)
    assert not plan.runnable
    assert any("capture bundle absent" in r for r in plan.refusals)
    # It must name the path it looked at, so a stale resolution is visible instead of silent.
    assert any("never_captured_int8_consistent" in r for r in plan.refusals)


def test_missing_w8a8_golden_refuses_instead_of_grading_int8_by_fp32_rules(recaps):
    _make_bundle(recaps, "m_int8_consistent", w8a8=False)
    plan = ec.plan_cell("m", int8=True, recaptures_root=recaps)
    assert not plan.runnable
    assert any("golden_w8a8.npy" in r for r in plan.refusals)
    # ... and the same bundle is fine for a non-int8 package.
    assert ec.plan_cell("m", int8=False, recaptures_root=recaps).runnable


def test_missing_fp32_golden_refuses_because_an_ungated_wall_is_not_a_measurement(recaps):
    _make_bundle(recaps, "m_int8_consistent", fp32=False)
    plan = ec.plan_cell("m", recaptures_root=recaps)
    assert any("golden.npy" in r for r in plan.refusals)


# --- W8A8 reference provenance: an unrecorded golden decides nothing ------------------------------


def test_an_unrecorded_w8a8_golden_is_unknown_not_independent(recaps):
    d = _make_bundle(recaps, "unregistered_int8_consistent")
    ref = ec.w8a8_reference(d)
    assert ref["status"] == "unknown"
    assert ref["independent"] is None
    assert "decides nothing" in ref["note"]


def test_a_bundle_may_declare_its_own_w8a8_provenance(recaps):
    d = _make_bundle(recaps, "selfdeclared_int8_consistent")
    (d / ec.W8A8_PROVENANCE_SIDECAR).write_text(
        json.dumps({"independent": True, "source": "torch eager", "evidence": "weights match"}),
        encoding="utf-8")
    ref = ec.w8a8_reference(d)
    assert ref["status"] == "declared_by_bundle" and ref["independent"] is True
    assert ref["note"] == ""


def test_a_bundle_declaring_a_non_independent_golden_still_carries_the_warning(recaps):
    d = _make_bundle(recaps, "frozen_int8_consistent")
    (d / ec.W8A8_PROVENANCE_SIDECAR).write_text(
        json.dumps({"independent": False, "source": "our own runtime, frozen"}), encoding="utf-8")
    ref = ec.w8a8_reference(d)
    assert ref["independent"] is False
    assert "decides nothing" in ref["note"]


def test_provenance_is_inherited_across_a_layout_rewrite_only_when_the_goldens_match(recaps,
                                                                                     monkeypatch):
    src = _make_bundle(recaps, "src_int8_consistent")
    same = _make_bundle(recaps, "same_int8_consistent")
    (same / "golden_w8a8.npy").write_bytes((src / "golden_w8a8.npy").read_bytes())
    differ = _make_bundle(recaps, "differ_int8_consistent")
    np.save(differ / "golden_w8a8.npy", np.ones(4, dtype=np.float32))
    monkeypatch.setitem(ec.W8A8_GOLDEN_PROVENANCE, "src_int8_consistent",
                        {"independent": True, "source": "torchao", "evidence": "bit-for-bit"})
    got = ec.w8a8_reference(same, source_bundle_id="src_int8_consistent", recaptures_root=recaps)
    assert got["status"] == "inherited_across_layout_rewrite" and got["independent"] is True
    # A golden that is NOT the same bytes inherits nothing -- the claim would be about other numbers.
    got2 = ec.w8a8_reference(differ, source_bundle_id="src_int8_consistent", recaptures_root=recaps)
    assert got2["status"] == "unknown"


# --- layout-only rewrites are DERIVED from the artifact, never from a name rule --------------------


def _write_rewrite(d, *, name, source):
    (d / "bundle.rewrites.json").write_text(
        json.dumps({"rewrites": [{"name": name, "source_bundle": source, "soundness": "x"}]}),
        encoding="utf-8")


def test_rewritten_siblings_reads_the_record_not_the_directory_name(recaps):
    _make_bundle(recaps, "base_int8_consistent")
    good = _make_bundle(recaps, "base_int8_consistent_pretransposed")
    _write_rewrite(good, name="hoist_weight_transposes", source="base_int8_consistent")
    # Same suffix, but its record names ANOTHER source: not a derivative of this bundle.
    other = _make_bundle(recaps, "unrelated_int8_consistent_pretransposed")
    _write_rewrite(other, name="hoist_weight_transposes", source="somebody_else")
    # A rewrite that changes what is COMPUTED is never layout-only, whatever it is called.
    graph = _make_bundle(recaps, "base_int8_consistent_truncated")
    _write_rewrite(graph, name="truncate_layers", source="base_int8_consistent")
    got = ec.rewritten_siblings("base_int8_consistent", recaptures_root=recaps)
    assert [g["bundle_id"] for g in got] == ["base_int8_consistent_pretransposed"]
    assert got[0]["equivalence"]["kind"] == "layout_only"


def test_prefer_rewritten_declines_when_more_than_one_derivative_exists(recaps):
    _make_bundle(recaps, "b_int8_consistent")
    for n in ("b_int8_consistent_a", "b_int8_consistent_b"):
        _write_rewrite(_make_bundle(recaps, n), name="hoist_weight_transposes",
                       source="b_int8_consistent")
    plan = ec.plan_cell("b", prefer_rewritten=True, recaptures_root=recaps)
    assert plan.ours_bundle_id == "b_int8_consistent"     # unrewritten, not an arbitrary pick
    assert any("declined" in n for n in plan.notes)


def test_prefer_rewritten_records_the_equivalence_the_ratio_rests_on(recaps):
    _make_bundle(recaps, "c_int8_consistent")
    _write_rewrite(_make_bundle(recaps, "c_int8_consistent_pt"), name="hoist_weight_transposes",
                   source="c_int8_consistent")
    plan = ec.plan_cell("c", prefer_rewritten=True, recaptures_root=recaps)
    assert plan.ours_bundle_id == "c_int8_consistent_pt"
    assert plan.layout_equivalence["kind"] == "layout_only"
    assert any("outside the timed window" in n for n in plan.notes)


# --- the row -------------------------------------------------------------------------------------


def _plan(recaps, name="small_int8_consistent", model="small"):
    _make_bundle(recaps, name)
    return ec.plan_cell(model, recaptures_root=recaps)


def _measured_record():
    return {
        "source_digest": "abc123", "source_dirty": ["passes_quant_int.py"],
        "ours": {"min_wall_ns": 1_000_000,
                 "protocol": {"warmup": 2, "iters": 5, "launches": 3, "pick": "min-of-n"},
                 "board_conditions": [{"governor": "performance"}],
                 "gate": {"fp32_cos": 0.999, "fp32_rel": 0.001, "w8a8_cos": 0.9999,
                          "w8a8_rel": 1e-4, "tiers": ["fp32", "w8a8"], "tier_ok": True},
                 "rvv": {"compute_symbol": "forward", "compute_symbol_coverage": 0.42,
                         "coverage_overall": 0.31}},
        "executorch_qd8": {
            "recipe_requested": "pt2e_qd8", "n_lo": 1, "n_hi": 6,
            "warm_ns": 2_000_000, "cold_ns": 3_200_000, "cold_over_warm": 1.6,
            "runs": [{"n": 1, "load_ns": 26_000_000, "quant_recipe": "pt2e_qd8",
                      "bundle_id": "small_int8_consistent", "accuracy_reference": "recomputed_fp32",
                      "cos": 0.998, "rel": 0.004, "board_conditions": {}}]},
        "verdict_qd8": {"status": "measured", "ours_ns": 1_000_000,
                        "executorch_warm_ns": 2_000_000, "ours_over_executorch": 0.5,
                        "speedup_vs_executorch": 2.0, "beats_executorch": True,
                        "executorch_load_ns": 26_000_000,
                        "accuracy": {"status": "not_comparable", "reason": "different references"}},
        "session_drift": {"ratio": 1.01, "within_noise_band": True},
    }


def test_a_measured_row_carries_everything_that_makes_the_ratio_checkable(recaps):
    row = ec.campaign_row(_plan(recaps), _measured_record())
    assert row["verdict"]["status"] == "measured"
    assert row["verdict"]["speedup_vs_executorch"] == 2.0
    assert row["ours_bundle_id"] == "small_int8_consistent"
    assert row["reference_bundle_id"] == "small_int8_consistent"
    assert row["quant_recipe"]["ours"] == ec.OURS_QUANT_RECIPE
    assert row["quant_recipe"]["reference"] == "pt2e_qd8"
    assert row["protocol"]["ours"]["pick"] == "min-of-n"
    assert row["protocol"]["reference"]["n_hi"] == 6
    assert row["board_conditions"]["ours"]
    assert row["source_digest"] == "abc123"
    assert row["source_dirty"] == ["passes_quant_int.py"]
    assert row["executorch_load_ns"] == 26_000_000
    assert row["accuracy"]["ours_reference"] == ec.OURS_ACCURACY_REFERENCE
    assert row["accuracy"]["reference_reference"] == "recomputed_fp32"
    assert row["accuracy"]["comparability"] == "not_comparable"
    assert row["rvv"]["compute_symbol"] == "forward"
    assert row["rvv"]["compute_symbol_coverage"] == 0.42
    assert row["w8a8_reference"]["status"] in ("unknown", "declared_by_registry")
    assert row["footprint"]["fits"] is True


def test_a_not_measured_verdict_becomes_a_refusal_with_no_ratio_anywhere(recaps):
    rec = _measured_record()
    rec["verdict_qd8"] = {"status": "not_measured",
                          "reason": "cannot extract a warm slope: need a passing wall at BOTH N"}
    row = ec.campaign_row(_plan(recaps), rec)
    assert row["verdict"]["status"] == "refused"
    assert "warm slope" in row["verdict"]["reason"]
    for k in ("speedup_vs_executorch", "ours_over_executorch", "ours_ns", "beats_executorch"):
        assert k not in row["verdict"]


def test_a_not_comparable_verdict_keeps_the_guard_reason(recaps):
    rec = _measured_record()
    rec["verdict_qd8"] = {"status": "not_comparable",
                          "reason": "quantization recipe MISMATCH: ours ran ..."}
    row = ec.campaign_row(_plan(recaps), rec)
    assert row["verdict"]["status"] == "refused"
    assert "recipe MISMATCH" in row["verdict"]["reason"]


def test_a_preflight_refusal_still_names_the_bundle_it_refused(recaps):
    _make_bundle(recaps, "big_int8_consistent", weights=9_000)
    plan = ec.plan_cell("big", budget_bytes=100, recaptures_root=recaps)
    row = ec.campaign_row(plan, None, refusal=" | ".join(plan.refusals))
    assert row["ran"] is False
    assert row["verdict"]["status"] == "refused"
    assert row["ours_bundle_id"] == "big_int8_consistent"
    assert row["preflight_refusals"]
    assert "speedup_vs_executorch" not in row["verdict"]


def test_a_record_without_the_verdict_block_refuses_rather_than_reading_a_ratio(recaps):
    row = ec.campaign_row(_plan(recaps), {"ours": {}})
    assert row["verdict"]["status"] == "refused"
    assert "no 'verdict_qd8' block" in row["verdict"]["reason"]


# --- expectations are advisory, and go stale loudly -----------------------------------------------


def test_a_declared_blocker_that_measures_is_flagged_stale():
    blocked = sorted(ec.KNOWN_REFERENCE_BLOCKERS)[0]
    assert ec.expectation_status(blocked, "measured") == "stale_expectation"
    assert ec.expectation_status(blocked, "refused") == "expected_refusal"
    assert ec.expectation_status("no_such_model", "refused") == "unexpected_refusal"
    assert ec.expectation_status("no_such_model", "measured") == "measured"


def test_a_declared_blocker_never_skips_the_cell(recaps):
    """A blocker that has been fixed upstream can only show up as a cell that suddenly measures, so
    the blocker must annotate the plan, never gate it."""
    blocked = sorted(ec.KNOWN_REFERENCE_BLOCKERS)[0]
    _make_bundle(recaps, f"{blocked}_int8_consistent")
    plan = ec.plan_cell(blocked, recaptures_root=recaps)
    assert plan.runnable
    assert any("declared upstream blocker" in n for n in plan.notes)


# --- resumability ----------------------------------------------------------------------------------


def test_the_ledger_round_trips_and_a_refused_cell_counts_as_recorded(tmp_path, recaps):
    ledger = tmp_path / "ledger.jsonl"
    ec.append_row(ledger, ec.campaign_row(_plan(recaps), _measured_record()))
    ec.append_row(ledger, ec.campaign_row(_plan(recaps, "other_int8_consistent", "other"), None,
                                          refusal="export blocked upstream"))
    rows = ec.read_ledger(ledger)
    assert len(rows) == 2
    # A refusal is an OUTCOME: re-running it would spend board time to re-derive what is recorded.
    assert ec.recorded_models(rows) == {"small", "other"}


def test_retry_refused_reruns_only_the_refusals(tmp_path, recaps):
    """A board that went away mid-campaign refuses every remaining cell. Treating those as settled
    would record a session outage as a property of four models, forever, on every resume."""
    ledger = tmp_path / "ledger.jsonl"
    ec.append_row(ledger, ec.campaign_row(_plan(recaps), _measured_record()))
    ec.append_row(ledger, ec.campaign_row(_plan(recaps, "other_int8_consistent", "other"), None,
                                          refusal="ssh: connect to host ... No route to host"))
    rows = ec.read_ledger(ledger)
    assert ec.completed_models(rows) == {"small", "other"}
    assert ec.completed_models(rows, retry_refused=True) == {"small"}


def test_a_rerun_that_measured_settles_the_cell_even_with_retry_refused(tmp_path, recaps):
    ledger = tmp_path / "ledger.jsonl"
    ec.append_row(ledger, ec.campaign_row(_plan(recaps), None, refusal="board went away"))
    ec.append_row(ledger, ec.campaign_row(_plan(recaps), _measured_record()))
    assert ec.completed_models(ec.read_ledger(ledger), retry_refused=True) == {"small"}


def test_a_truncated_ledger_line_is_skipped_not_guessed_at(tmp_path, recaps):
    ledger = tmp_path / "ledger.jsonl"
    ec.append_row(ledger, ec.campaign_row(_plan(recaps), _measured_record()))
    with ledger.open("a", encoding="utf-8") as f:
        f.write('{"model": "half-written"\n')
    rows = ec.read_ledger(ledger)
    assert ec.recorded_models(rows) == {"small"}


# --- the summary ------------------------------------------------------------------------------------


def _row(model, status, *, beats=None, reason=""):
    v = {"status": status}
    if status == "measured":
        v.update(ours_ns=1e6, executorch_warm_ns=2e6, speedup_vs_executorch=2.0,
                 beats_executorch=beats)
    else:
        v["reason"] = reason
    return {"model": model, "verdict": v, "w8a8_reference": {"independent": None},
            "expectation": ec.expectation_status(model, status)}


def test_the_summary_makes_how_many_cells_produced_a_verdict_impossible_to_miss():
    rows = [_row("a", "measured", beats=True),
            _row("b", "refused", reason="export blocked"),
            _row("c", "refused", reason="runner cannot load"),
            _row("d", "refused", reason="does not fit the board")]
    s = ec.summarize(rows)
    assert s["cells_attempted"] == 4
    assert s["verdicts_produced"] == 1
    assert s["refused"] == 3
    assert s["wins"] == 1
    # One win out of four attempted is NOT a majority, even though it is a majority of the single
    # cell that produced a verdict. Both readings are reported so neither can be quoted alone.
    assert s["majority_of_attempted"] is False
    assert s["majority_of_measured"] is True
    assert s["refusal_reasons"]["b"] == "export blocked"
    text = ec.format_summary(s)
    assert "VERDICTS produced: 1" in text and "REFUSED: 3" in text
    assert "export blocked" in text


def test_the_summary_reports_a_real_majority_as_one():
    rows = [_row(m, "measured", beats=True) for m in ("a", "b", "c")] + [_row("d", "refused")]
    s = ec.summarize(rows)
    assert s["majority_of_attempted"] is True and s["wins"] == 3


def test_a_rerun_row_supersedes_the_earlier_one_for_the_same_model():
    rows = [_row("a", "refused", reason="transient"), _row("a", "measured", beats=True)]
    s = ec.summarize(rows)
    assert s["cells_attempted"] == 1 and s["verdicts_produced"] == 1


def test_the_summary_surfaces_a_stale_expectation():
    blocked = sorted(ec.KNOWN_REFERENCE_BLOCKERS)[0]
    s = ec.summarize([_row(blocked, "measured", beats=True)])
    assert s["stale_expectations"] == [blocked]
    assert "STALE EXPECTATIONS" in ec.format_summary(s)


# --- the campaign must stay wired to the instrument it drives ---------------------------------------


def _module_constant(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
    return None


def test_the_campaign_labels_our_recipe_exactly_as_the_instrument_does():
    """Two names for one arithmetic is how a recipe guard stops being able to fire."""
    assert _module_constant(INSTRUMENT, "OURS_QUANT_RECIPE") == ec.OURS_QUANT_RECIPE
    assert _module_constant(INSTRUMENT, "OURS_ACCURACY_REFERENCE") == ec.OURS_ACCURACY_REFERENCE


def test_every_flag_the_driver_passes_is_one_the_instrument_accepts():
    """An unrecognised child flag has hidden in a wrapper on this tree before, and presented as the
    child 'just not doing anything'. Read the instrument's real parser rather than trusting the docs."""
    tree = ast.parse(INSTRUMENT.read_text(encoding="utf-8"))
    accepted = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    accepted.add(arg.value)
    driver = ast.parse(DRIVER.read_text(encoding="utf-8"))
    passed = set()
    for node in ast.walk(driver):
        if isinstance(node, ast.FunctionDef) and node.name == "_instrument_command":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str) \
                        and sub.value.startswith("--"):
                    passed.add(sub.value)
    passed.add("--out")                      # added by the caller, not by _instrument_command
    assert passed, "the driver's command builder passed no flags at all"
    assert passed <= accepted, f"instrument does not accept {sorted(passed - accepted)}"


def test_dry_run_resolves_every_cell_spends_no_board_time_and_writes_nothing(tmp_path):
    """The gate before board time: --dry-run must exit 0, name each cell's decision, and leave no
    artifact behind. A model with no capture must REFUSE, loudly, not report not_run."""
    env = {"PYTHONPATH": str(repo_root() / "merlin" / "python"), "PATH": "/usr/bin:/bin",
           "HOME": str(tmp_path), "MERLIN_OUT_ROOT": str(tmp_path / "out")}
    got = subprocess.run([sys.executable, str(DRIVER), "--dry-run",
                          "--models", "definitely_not_a_captured_model"],
                         cwd=str(repo_root()), capture_output=True, text=True, env=env, timeout=600)
    assert got.returncode == 0, got.stderr[-2000:]
    assert "NO board time will be spent" in got.stdout
    assert "WOULD REFUSE" in got.stdout
    assert "capture bundle absent" in got.stdout
    assert "would run 0/1" in got.stdout
    assert not (tmp_path / "out").exists(), "--dry-run must not create an artifact tree"


def _load_driver():
    spec = importlib.util.spec_from_file_location("k1_int8_et_campaign", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_an_unreachable_board_fails_before_writing_any_row(tmp_path, monkeypatch, capsys):
    """The failure this prevents: an unreachable board writes a refusal on EVERY cell, and the next
    resume skips all of them as settled outcomes -- a session outage recorded as a property of four
    models. The preflight must exit non-zero having created no product dir and no ledger."""
    from merlin.mining import k1 as k1mod

    monkeypatch.setattr(k1mod, "available", lambda **kw: False)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    driver = _load_driver()
    monkeypatch.setattr(sys, "argv", ["k1_int8_et_campaign.py", "--models", "small_llama"])
    assert driver.main() == 2
    assert "Nothing recorded" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_the_board_preflight_can_only_be_skipped_deliberately():
    """It is an opt-OUT, not an opt-in: the default path must check the board."""
    driver = _load_driver()
    assert "--no-board-preflight" in DRIVER.read_text(encoding="utf-8")
    assert driver.DEFAULT_MODELS, "the campaign must ship the diverse set it is claimed over"


# --- the fit test and the gate must both be ABLE TO FAIL -----------------------------------------
#
# Both defects below shipped, and both presented as success. `smolvla_int8_w8a8_consistent` is a
# 1.83 GB session capture that priced at ZERO resident bytes and reported `fits=True`, because every
# weight lives one directory down and the pricer only stat'd the root. Its `prefix_encode` stage
# ships a golden that is constant 1.0 covering 1 of 2 results — and the covered one is the pad mask,
# not the KV cache the stage computes. A check that cannot fail must not report a pass.


def _make_session(root, name, *, programs=("a", "b"), weights=1024, extra=512, stray=0):
    """A version-2 multi-program capture: nothing at the root but a contract."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    lines = ["version: 2", "programs:"]
    for p in programs:
        c = d / "stages" / p
        c.mkdir(parents=True, exist_ok=True)
        (c / "weights.safetensors").write_bytes(b"\0" * weights)
        (c / "extra.npz").write_bytes(b"\0" * extra)
        (c / "model.mlir").write_text(
            "module {\n  func.func @forward(%0: tensor<4xf32>) -> tensor<4xf32> {\n  }\n}\n",
            encoding="utf-8")
        np.save(c / "golden.npy", np.arange(4, dtype=np.float32))
        lines += [f"  - name: {p}", f"    bundle: stages/{p}"]
    (d / "session_contract.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if stray:
        (d / "unknown_blob.bin").write_bytes(b"\0" * stray)
    return d


def test_a_session_bundle_is_priced_by_program_not_at_its_empty_root(recaps):
    d = _make_session(recaps, "s_int8_consistent", weights=1000, extra=500)
    fp = ec.bundle_footprint(d, budget_bytes=10_000)
    # Before the fix this was 0 with fits=True: the root holds only a contract.
    assert fp["program_count"] == 2
    assert fp["resident_lower_bound_bytes"] == 1500, "largest program, so fits=False stays decisive"
    assert fp["resident_all_programs_bytes"] == 3000
    assert fp["fits"] is True


def test_a_session_too_big_for_the_board_is_still_refused(recaps):
    _make_session(recaps, "big_int8_consistent", weights=9_000, extra=3_000)
    plan = ec.plan_cell("big", budget_bytes=1_000, recaptures_root=recaps)
    assert any("does not fit the board" in r for r in plan.refusals)


def test_bytes_the_pricer_does_not_recognise_make_the_verdict_unknown_not_true(recaps):
    d = _make_session(recaps, "u_int8_consistent", weights=100, extra=100, stray=50_000)
    fp = ec.bundle_footprint(d, budget_bytes=10_000)
    assert fp["unpriced_bytes"] == 50_000
    assert fp["fits"] is None, "an unexamined remainder that alone busts the budget is not a pass"
    assert "unknown_blob.bin" in fp["unpriced_examples"]
    plan = ec.plan_cell("u", budget_bytes=10_000, recaptures_root=recaps)
    assert any("footprint UNKNOWN" in r for r in plan.refusals)


def test_a_multi_program_capture_is_refused_as_a_single_cell(recaps):
    _make_session(recaps, "m_int8_consistent")
    plan = ec.plan_cell("m", recaptures_root=recaps)
    assert not plan.runnable
    why = " ".join(plan.refusals)
    assert "session capture" in why and "program selector" in why
    # And NOT the misleading root-level story: every program ships both artifacts.
    assert "ships no model.mlir" not in why
    assert "ships no golden.npy" not in why


def test_forward_result_count_does_not_split_inside_a_tensor_type(tmp_path):
    m = tmp_path / "model.mlir"
    m.write_text(
        "module {\n"
        "  func.func @forward(%0: tensor<1x2xf32>, %1: tensor<3xf32>) -> "
        "(tensor<1x113xi1>, tensor<2x16x1x113x5x64xbf16>) {\n  }\n}\n", encoding="utf-8")
    assert ec._forward_result_count(m) == 2
    m.write_text("module {\n  func.func @forward(%0: tensor<4xf32>) -> tensor<4xf32> {\n }\n}\n",
                 encoding="utf-8")
    assert ec._forward_result_count(m) == 1


def _cover_bundle(root, name, *, results, golden):
    d = _make_bundle(root, name)
    rets = ", ".join(f"tensor<{i + 1}xf32>" for i in range(results))
    sig = rets if results == 1 else f"({rets})"
    (d / "model.mlir").write_text(
        f"module {{\n  func.func @forward(%0: tensor<4xf32>) -> {sig} {{\n  }}\n}}\n",
        encoding="utf-8")
    np.save(d / "golden.npy", golden)
    return d


def test_a_partial_gate_is_declared_but_still_measurable(recaps):
    _cover_bundle(recaps, "p_int8_consistent", results=3,
                  golden=np.arange(3, dtype=np.float32))
    plan = ec.plan_cell("p", recaptures_root=recaps)
    assert plan.golden_coverage["partial"] is True
    assert plan.golden_coverage["cannot_fail"] is False
    assert plan.runnable, "grading one of three outputs is a caveat, not a reason to refuse"
    assert any("PARTIAL GATE" in n for n in plan.notes)


def test_a_gate_that_cannot_fail_is_refused(recaps):
    # 1 of 2 results graded, and that one constant -- smolvla/prefix_encode exactly.
    _cover_bundle(recaps, "v_int8_consistent", results=2,
                  golden=np.ones((1, 113), dtype=np.float32))
    plan = ec.plan_cell("v", recaptures_root=recaps)
    assert plan.golden_coverage["cannot_fail"] is True
    assert not plan.runnable
    assert any("CANNOT FAIL" in r for r in plan.refusals)


def test_a_full_nondegenerate_golden_raises_neither_flag(recaps):
    _cover_bundle(recaps, "ok_int8_consistent", results=1,
                  golden=np.arange(8, dtype=np.float32))
    plan = ec.plan_cell("ok", recaptures_root=recaps)
    assert plan.golden_coverage == {**plan.golden_coverage, "partial": False, "degenerate": False}
    assert plan.runnable
    assert not any("PARTIAL GATE" in n or "DEGENERATE" in n for n in plan.notes)


def test_a_refused_cell_still_shows_a_reference_wall_that_was_measured():
    """Board time already spent must not be discarded because the OTHER arm refused.

    tiny_llama's cell was refused because our whole-model clang outran its ceiling -- a reason with
    nothing to do with ExecuTorch, whose two-N slope had already been paid for on the board. The
    summary printed `nan`, and the 364.142 ms had to be recovered by hand out of log text.
    """
    row = {
        "model": "tiny_llama", "ours_bundle_id": "tiny_llama_int8_consistent",
        "w8a8_reference": {"independent": True},
        "protocol": {"reference": {"warm_ns": 364141594.0, "cold_ns": 391514666.0}},
        "verdict": {"status": "refused", "reason": "clang outran the compile ceiling"},
    }
    cell = ec.summarize([row])["per_model"]["tiny_llama"]
    assert cell["status"] == "refused"
    assert cell["executorch_warm_ns"] is None, "a refused cell publishes no reference verdict"
    assert cell["executorch_warm_ns_measured"] == 364141594.0, "but the measurement is kept"
    assert cell["speedup_vs_executorch"] is None, "and still no ratio"

    text = ec.format_summary(ec.summarize([row]))
    assert "364.142" in text and "MEASURED, no ratio" in text


def test_a_declared_blocker_that_was_overtaken_is_removed_not_kept():
    """lstmnetvit's declared qd8-export blocker was retired once the cell produced a verdict.

    A stale entry is worse than no entry: `expectation_status` turns every refusal of a declared
    model into `expected_refusal`, so a NEW and unrelated failure on that cell would read as the
    old known one and draw no attention.
    """
    assert "lstmnetvit" not in ec.KNOWN_REFERENCE_BLOCKERS
    assert ec.expectation_status("lstmnetvit", "refused") == "unexpected_refusal"
    # smolvla is the live one: ExecuTorch produces no .pte for it at all.
    b = ec.known_blocker("smolvla")
    assert b and b["stage"] == "executorch_export" and "u31" in b["reason"]
    assert "CAPABILITY" in b["not_a_fallback"], (
        "running a model the reference cannot export is not a speedup, and the row must say so")
    assert ec.expectation_status("smolvla", "measured") == "stale_expectation"
