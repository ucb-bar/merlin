"""``--capsules changed`` simulates only what an edit changed, and certifies nothing it did not earn.

The worker runs its real selection, classification, bound and verdict accounting over a fixture grade
that models the exact-byte certificate store: a capsule's "bytes" are its submission files, the key is
(capsule, bytes, engine), the probe pass carries a stored pass and refuses everything else with the
probe's own reasons, and only an ordinary (non-probe) grade "simulates" -- which is what is counted.
The real probe and key are tested against the real certificate in
``merlin/tests/targetgen/test_certificate_probe.py``.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.context import InvocationContext
from merlin_experiments.phase1.feedback import selfcheck, selfcheck_changed
from merlin_experiments.phase1.tools import selfcheck as shim

from merlin.common.paths import data_path
from merlin.targetgen import simulation_certificate as SC

SUITE = "fixture-suite"
TIERS = ("L0", "L1", "L2", "L3")
NAMES = ("cap_a", "cap_b", "cap_c")
SHARED = {"cap_a", "cap_c"}  # both read the submission's shared artifact


def _context(root: Path) -> InvocationContext:
    return InvocationContext(
        root, root / "target.yaml", root, "fixture", root / "runs", root / "reports", root / "bundles", ()
    )


def _adapter(engine: str):
    def run(*_args):
        raise AssertionError("fixture adapter must not run")

    run._merlin_simulator_engine = engine
    return run


def _capsule(corpus: Path, name: str, label: str = "public") -> None:
    capsule = corpus / name
    capsule.mkdir(parents=True)
    (capsule / "capsule.yaml").write_text(
        json.dumps(
            {
                "name": name,
                "kind": "isa",
                "source_role": "handauthored_compiler_test",
                "label": label,
                "operation": {"op": "movement"},
                "numeric_policy": {"compare": "exact_int", "dtype": "i8"},
                "expected": {"instruction_classes": []},
                "required_oracle_tiers": list(TIERS),
            }
        )
    )


def _tier(status, **extra):
    return {"status": status, "mandatory": True, **extra}


@pytest.fixture
def bench(tmp_path, monkeypatch):
    """A worker over three public capsules (plus one hidden) and a modelled certificate store."""
    monkeypatch.chdir(tmp_path)
    for key in ("MERLIN_SELFCHECK_PROMOTION_WS", "MERLIN_SELFCHECK_MAX_CAPSULES", SC.PROBE_ENV):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(SC.ENV, "1")
    monkeypatch.setenv("MERLIN_TIER_CERT_CACHE", str(tmp_path / "store"))
    monkeypatch.delenv("MERLIN_CAPSULE_TIMING_HISTORY", raising=False)
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "manifest.yaml").write_text("{}")
    (submission / "shared.prog").write_text("shared-v1")
    corpus = tmp_path / "public"
    for name in NAMES:
        (submission / f"{name}.prog").write_text(f"{name}-v1")
        _capsule(corpus, name)
    _capsule(corpus, "cap_hidden", label="hidden")
    monkeypatch.setattr(selfcheck, "_target_sim_via", lambda *a: ("fixture", "chipyard"))
    monkeypatch.setattr(selfcheck.CR, "suite_for", lambda *a, **k: SUITE)
    monkeypatch.setattr(selfcheck, "_log_telemetry", lambda *a: None)
    official = {"L2": _adapter("spike"), "L3": _adapter("gsim")}
    monkeypatch.setattr(selfcheck, "_adapters", lambda sim, *a, **k: (dict(official), sim))
    monkeypatch.setattr(selfcheck.CR, "oracle_adapters", lambda *a, **k: dict(official))
    state = SimpleNamespace(
        store={},
        engine="gsim-a",
        probes=[],
        simulated=[],
        grade_roots=[],
        labels=[],
        omit=set(),
        garble=set(),
        probe_env=[],
    )

    def program(sub: Path, name: str) -> str:
        own = (sub / f"{name}.prog").read_text()
        return own + ("|" + (sub / "shared.prog").read_text() if name in SHARED else "")

    def grade(sub, *, capsules_root, runs_root, labels, oracle_adapters, **_kwargs):
        assert set(oracle_adapters) == {"L2", "L3"}, "changed classifies on the grade's own selection"
        state.grade_roots.append(Path(capsules_root))
        state.labels.append(set(labels))
        probing = os.environ.get(SC.PROBE_ENV)
        state.probe_env.append(probing)
        names = sorted(
            d.name
            for d in Path(capsules_root).iterdir()
            if (d / "capsule.yaml").is_file() and json.loads((d / "capsule.yaml").read_text())["label"] in labels
        )
        (state.probes if probing else state.simulated).append(names)
        rows = []
        for name in names:
            key = (name, program(Path(sub), name), state.engine)
            tiers = {"L0": _tier("pass"), "L1": _tier("pass"), "L2": _tier("pass")}
            if "BROKEN_SCREEN" in key[1]:  # the functional screen refutes it; the ladder stops there
                tiers.update(L2=_tier("fail"), L3=_tier("skipped"))
            elif probing:
                stored = state.store.get(key)
                if stored == "pass":
                    tiers["L3"] = _tier("pass", carried={"carried": True})
                else:
                    tiers["L3"] = _tier(
                        "unavailable", reason=SC.PROBE_FAILING if stored == "fail" else SC.PROBE_CHANGED
                    )
            else:
                verdict = "fail" if "BAD_RTL" in key[1] else "pass"
                state.store[key] = verdict
                tiers["L3"] = _tier(verdict)
            status = {"pass": "pass", "fail": "fail"}.get(tiers["L3"]["status"], "incomplete")
            if tiers["L2"]["status"] == "fail":
                status = "fail"
            trace = {"status": "pass", "trace_requirement_status": "pass"}
            if "NO_TRACE" in key[1]:  # numerically right, structurally unqualified
                trace = {"status": "fail", "trace_requirement_status": "failed", "violations": ["no matmul"]}
            result = {
                "capsule": name,
                "kind": "op",
                "status": status,
                "numeric": {"status": "pass"},
                "trace_check": trace,
                "tiers": tiers,
            }
            if tiers["L2"]["status"] == "fail":
                result["failure"] = {
                    "plane": "numeric",
                    "category": "FUNCTIONAL_MISMATCH",
                    "tier": "L2",
                    "detail": "12 of 64 elements differ",
                }
            elif status == "incomplete":
                result["failure"] = {
                    "plane": "oracle_unavailable",
                    "category": "NOT_RUN_IS_NOT_PASS",
                    "tier": "L3",
                    "tier_status": "unavailable",
                    "detail": tiers["L3"]["reason"],
                }
            rows.append({"capsule": name, "status": status})
            parent = Path(runs_root) / "runs" / SUITE / name
            parent.mkdir(parents=True)
            if probing and name in state.omit:
                continue
            text = "{ truncated" if probing and name in state.garble else json.dumps(result)
            (parent / "capsule_result.json").write_text(text)
        return {"n_capsules": len(names), "per_capsule": rows}

    monkeypatch.setattr(selfcheck.CG, "grade", grade)

    def run(capsules, *argv):
        state.probes.clear(), state.simulated.clear()
        out = tmp_path / "verdict.json"
        out.unlink(missing_ok=True)
        code = selfcheck.main(
            ["--submission", str(submission), "--capsules", capsules, "--out", str(out), *argv],
            context=_context(tmp_path),
            capsules_root=corpus,
            contract=data_path("contract"),
        )
        return code, json.loads(out.read_text()) if out.exists() else None

    state.run, state.submission, state.corpus, state.tmp = run, submission, corpus, tmp_path
    return state


def _rows(report):
    return {row["capsule"]: row for row in report["per_capsule"]}


def test_unchanged_bytes_reuse_their_certificates_and_simulate_nothing(bench):
    code, first = bench.run(",".join(NAMES))
    assert code == 0 and first["all_pass"] and bench.simulated == [list(NAMES)]
    code, report = bench.run("changed")
    assert bench.simulated == [], "no capsule may be simulated when every certificate still applies"
    assert bench.probes == [list(NAMES)]
    assert code == 0 and report["all_pass"] is True and report["certified_complete"] is True
    selection = report["changed_selection"]
    assert selection["counts"] == {"unchanged-certified": 3, "unchanged-failing": 0, "failing": 0, "changed": 0}
    assert selection["simulated"] == [] and report["n_simulated"] == 0 and report["n_reused"] == 3
    rows = _rows(report)
    assert set(rows) == set(NAMES)
    assert all(
        row["certified"] and row["selection"] == "unchanged-certified" and row["reused_tiers"] == ["L3"]
        for row in rows.values()
    )
    assert report["oracle_tiers"] == ["L2", "L3"], "the broker must not enqueue a cert promotion for reuse"


def test_a_one_capsule_edit_simulates_only_that_capsule(bench):
    bench.run(",".join(NAMES))
    (bench.submission / "cap_b.prog").write_text("cap_b-v2")
    code, report = bench.run("changed")
    assert bench.simulated == [["cap_b"]]
    assert code == 0 and report["all_pass"] is True
    rows = _rows(report)
    assert rows["cap_b"]["simulated"] is True and rows["cap_b"]["certified"] is True
    assert rows["cap_a"]["selection"] == rows["cap_c"]["selection"] == "unchanged-certified"
    code, again = bench.run("changed")
    assert bench.simulated == [] and again["changed_selection"]["counts"]["unchanged-certified"] == 3


def test_an_edit_to_a_shared_artifact_simulates_every_capsule_built_from_it(bench):
    bench.run(",".join(NAMES))
    (bench.submission / "shared.prog").write_text("shared-v2")
    code, report = bench.run("changed")
    assert bench.simulated == [sorted(SHARED)]
    assert report["changed_selection"]["unchanged_certified"] == ["cap_b"]


def test_a_changed_engine_selection_is_changed_for_every_capsule(bench):
    bench.run(",".join(NAMES))
    bench.engine = "gsim-b"
    code, report = bench.run("changed")
    assert bench.simulated == [list(NAMES)]
    assert report["changed_selection"]["counts"]["changed"] == 3


def test_a_missing_or_unreadable_probe_record_is_changed_never_certified(bench):
    bench.run(",".join(NAMES))
    bench.omit, bench.garble = {"cap_a"}, {"cap_c"}
    code, report = bench.run("changed")
    assert bench.simulated == [["cap_a", "cap_c"]]
    assert report["changed_selection"]["unchanged_certified"] == ["cap_b"]
    assert all(row["certified"] for row in report["per_capsule"])  # earned by this request's own run


def test_bytes_that_already_failed_are_reported_not_resimulated(bench):
    (bench.submission / "cap_a.prog").write_text("cap_a-BAD_RTL")
    code, first = bench.run(",".join(NAMES))
    assert code == 1 and not _rows(first)["cap_a"]["pass"]
    code, report = bench.run("changed")
    assert bench.simulated == []
    assert code == 1 and report["all_pass"] is False and report["certified_complete"] is False
    row = _rows(report)["cap_a"]
    assert row["selection"] == "unchanged-failing" and row["pass"] is False and row["failed_tiers"] == ["L3"]
    assert report["changed_selection"]["unchanged_failing"] == ["cap_a"]
    (bench.submission / "cap_a.prog").write_text("cap_a-fixed")
    code, fixed = bench.run("changed")
    assert bench.simulated == [["cap_a"]] and code == 0 and fixed["all_pass"]


def test_the_bound_simulates_the_longest_changed_first_and_says_what_remains(bench, monkeypatch, tmp_path):
    history = tmp_path / "history" / "earlier"
    for name, seconds in (("cap_a", 5.0), ("cap_b", 50.0), ("cap_c", 500.0)):
        parent = history / "runs" / SUITE / name
        parent.mkdir(parents=True)
        timing = {"build_s": 1.0, "sim_active_s": seconds}
        (parent / "capsule_result.json").write_text(
            json.dumps({"capsule": name, "tiers": {"L3": {"measured_now": True, "timing": timing}}})
        )
    monkeypatch.setenv("MERLIN_CAPSULE_TIMING_HISTORY", str(tmp_path / "history"))
    monkeypatch.setenv("MERLIN_SELFCHECK_MAX_CAPSULES", "2")
    code, report = bench.run("changed")
    assert bench.simulated == [["cap_b", "cap_c"]], "the two longest by measured cost"
    selection = report["changed_selection"]
    assert selection["bound"] == 2 and selection["not_simulated"] == ["cap_a"] and selection["n_not_simulated"] == 1
    assert code == 1 and report["all_pass"] is False and report["n_unchecked"] == 1
    assert _rows(report)["cap_a"]["pass"] is False and _rows(report)["cap_a"]["simulated"] is False
    assert "exceed this request's bound of 2" in report["note"]
    code, rest = bench.run("changed")
    assert bench.simulated == [["cap_a"]] and code == 0 and rest["all_pass"]


def test_a_provided_host_capsule_is_never_probed_and_always_simulated(bench, monkeypatch):
    """Its RTL tier runs on a route with no exact-byte certificate: probing it would simulate it."""
    from merlin.targetgen import owned_host_leaf_execution as leaf

    monkeypatch.setattr(leaf, "selected_host_leaf", lambda capsule: capsule.get("name") == "cap_b")
    bench.run(",".join(NAMES))
    code, report = bench.run("changed")
    assert bench.probes == [["cap_a", "cap_c"]] and bench.simulated == [["cap_b"]]
    assert report["changed_selection"]["provided_host"] == ["cap_b"]
    assert report["changed_selection"]["n_probed"] == 2
    assert _rows(report)["cap_b"]["simulated"] is True  # its own verdict comes from this request's run


def test_hidden_capsules_are_never_probed_simulated_or_reported(bench):
    bench.run(",".join(NAMES))
    (bench.submission / "cap_b.prog").write_text("cap_b-v2")
    code, report = bench.run("changed")
    assert all(labels == {"public", "dev"} for labels in bench.labels)
    assert all(root == bench.corpus or root.parent != bench.corpus for root in bench.grade_roots)
    graded = {name for batch in bench.probes + bench.simulated for name in batch}
    assert "cap_hidden" not in graded and "cap_hidden" not in json.dumps(report)
    assert report["changed_selection"]["n_public"] == 3


def test_the_probe_environment_never_reaches_the_simulate_pass(bench):
    (bench.submission / "cap_b.prog").write_text("cap_b-v2")
    bench.run("changed")
    assert bench.probe_env == ["gsim", None], "the probe refuses exactly the RTL engine, then nothing"
    assert SC.PROBE_ENV not in os.environ


@pytest.mark.parametrize("argv", [["--sim", "spike"], ["--tiers", "L2"]])
def test_changed_classifies_only_on_the_grade_selection(bench, monkeypatch, argv):
    """A named screen or a --tiers narrowing is not the grade's selection, so nothing is classified."""
    if argv[0] == "--sim":
        monkeypatch.setattr(selfcheck, "_adapters", lambda sim, *a, **k: ({"L2": _adapter("spike")}, sim))
    code, report = bench.run("changed", *argv)
    assert code == 2 and "grade's own tier selection" in report["error"] and report["all_pass"] is False
    assert bench.probes == bench.simulated == []


def test_without_exact_byte_certificates_changed_refuses_instead_of_simulating_all(bench, monkeypatch):
    monkeypatch.delenv(SC.ENV)
    code, report = bench.run("changed")
    assert code == 2 and SC.ENV in report["error"] and bench.probes == bench.simulated == []


def test_the_probe_streams_its_classification_as_progress(bench):
    bench.run(",".join(NAMES))
    progress = bench.tmp / "progress.json"
    bench.run("changed", "--progress-out", str(progress))
    doc = json.loads(progress.read_text())
    assert doc["phase"] == "changed_probe" and doc["status"] == "probe_complete"
    assert doc["n_expected"] == doc["n_finished"] == 3 and doc["counts"] == {"unchanged-certified": 3}


def test_classification_fails_closed():
    adapters = {"L2": _adapter("spike"), "L3": _adapter("gsim")}

    def category(result):
        return selfcheck_changed.classify(result, {"name": "x"}, adapters=adapters, barrier_tier="L3", target="x")[0]

    for result in (None, {}, {"tiers": "garbage"}, {"capsule": "x", "kind": "model", "tiers": {}}):
        assert category(result) == selfcheck_changed.CHANGED, "an unreadable record is changed, never certified"
    refused = _tier("unavailable", reason=SC.PROBE_FAILING)
    assert category({"capsule": "x", "tiers": {"L2": _tier("pass"), "L3": refused}}) == "unchanged-failing"
    assert category({"capsule": "x", "tiers": {"L2": _tier("fail"), "L3": refused}}) == "failing"
    assert category({"capsule": "x", "tiers": {"L2": _tier("unavailable"), "L3": refused}}) == "changed"
    other = _tier("unavailable", reason="gsim not available")  # not the probe's refusal
    assert category({"capsule": "x", "status": "incomplete", "tiers": {"L2": _tier("pass"), "L3": other}}) == (
        "failing"
    ), "a tier the engine itself could not run fails the same way when simulated; nothing to re-run"


def test_a_screen_failure_is_reported_from_the_probe_not_resimulated(bench):
    bench.run(",".join(NAMES))
    (bench.submission / "cap_b.prog").write_text("cap_b-BROKEN_SCREEN")
    code, report = bench.run("changed")
    assert bench.simulated == [], "the screen already refuted it; the ladder would stop at the same tier"
    assert code == 1 and report["all_pass"] is False
    row = _rows(report)["cap_b"]
    assert row["selection"] == "failing" and row["pass"] is False and row["certified"] is False
    assert row["failure"]["category"] == "FUNCTIONAL_MISMATCH" and row["failure"]["tier"] == "L2"
    assert report["changed_selection"]["failing"] == ["cap_b"]


def test_a_structural_failure_on_carried_rtl_verdicts_is_not_resimulated(bench):
    (bench.submission / "cap_c.prog").write_text("cap_c-NO_TRACE")
    code, first = bench.run(",".join(NAMES))
    assert not _rows(first)["cap_c"]["certified"]
    code, report = bench.run("changed")
    assert bench.simulated == []
    row = _rows(report)["cap_c"]
    assert row["selection"] == "failing" and row["structural_qualification"]["status"] != "pass"
    assert row["tiers"]["L3"] == "pass", "the carried RTL verdict is shown, and is not a certification"


def test_the_shim_and_worker_offer_changed_and_budget_it_like_the_whole_suite():
    assert shim._request_budget_seconds(1800, capsules="changed", workers=4, suite_size=64) == (
        shim._request_budget_seconds(1800, capsules="all", workers=4, suite_size=64)
    )
    for module in (selfcheck, shim):
        tree = ast.parse(Path(module.__file__).read_text())
        call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "add_argument"
            and node.args
            and getattr(node.args[0], "value", None) == "--capsules"
        )
        help_text = ast.literal_eval(next(k.value for k in call.keywords if k.arg == "help"))
        assert "'changed'" in help_text, module.__name__
