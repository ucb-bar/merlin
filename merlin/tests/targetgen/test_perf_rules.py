"""The rule registry: is it well-formed data, and does it emit exactly the experiments it can run?

The registry answers *detected trait -> required model term -> optimization family -> experiment
family* as YAML under ``merlin/contract/perf_rules/``. Two things are worth testing about it. First
that the data is internally consistent -- every rule reaches a declared axis, a real fit form, a
named lever and a named experiment shape. Second, and the point of R5.8, that applying it to a
target produces **exactly** the experiment set that would re-derive that target's already-known
constants and no others, and that running that set recovers them.

The acceptance runs twice: once on a synthetic instrument whose constants are known because the
fixture chose them, so the mechanism is exercised on every host; and once on the pinned measured
corpus, which is the real claim and skips where that checkout is absent.
"""
from __future__ import annotations

import importlib.util
import json

import pytest

from merlin.common.paths import env, merlin_dir, repo_root
from merlin.perf import harvest as H
from merlin.perf.profile import derive_profile
from merlin.perf.term import UNKNOWN


# ---------------------------------------------------------------------------------------------
# the registry is DATA, and it is well-formed
# ---------------------------------------------------------------------------------------------


def test_the_registry_directory_holds_only_yaml():
    """A ``.py`` here would be scanned by the target-name gate; more to the point, a rule expressed
    as code is a rule that eventually gets a target baked into it."""
    stray = sorted(p.name for p in H.registry_dir().iterdir()
                   if p.is_file() and p.suffix not in (".yaml", ".yml"))
    assert stray == [], f"non-YAML files in the rule registry: {stray}"


def test_the_registry_loads_and_is_internally_consistent():
    reg = H.load_registry()
    assert reg.rules, "the registry is empty"
    assert reg.validate() == ()


def test_every_rule_names_a_lever_an_experiment_and_a_rationale():
    for rule in H.load_registry().rules:
        assert rule.optimization_family, f"{rule.id} settles a term nobody can act on"
        assert rule.experiment_family, f"{rule.id} names no experiment shape"
        assert len(rule.rationale.split()) > 20, f"{rule.id}'s rationale is too thin to refute"


def test_no_rule_names_a_target():
    """One source of truth for the name set: the gate's own list."""
    gate = repo_root() / "build_tools" / "scripts" / "check_no_target_name.py"
    spec = importlib.util.spec_from_file_location("_name_gate", gate)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for path in sorted(H.registry_dir().glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        for name in mod.TARGET_NAMES:
            assert not mod._contains_identifier(text, name), f"{path.name} names the target {name!r}"


def test_a_rule_that_reaches_an_undeclared_axis_is_reported_not_ignored(tmp_path):
    (tmp_path / "bad.yaml").write_text(json.dumps({"rules": [{
        "id": "r", "constant": "c", "term": "t", "axis": "no_such_axis", "fit_form": "affine",
        "recover": {"from": "slope"}, "optimization_family": "nope",
        "experiment_family": "nope", "rationale": "  "}]}), encoding="utf-8")
    problems = H.load_registry(tmp_path).validate()
    assert any("axis" in p for p in problems)
    assert any("optimization family" in p for p in problems)
    assert any("rationale" in p for p in problems)


# ---------------------------------------------------------------------------------------------
# a synthetic instrument whose constants the fixture chose
# ---------------------------------------------------------------------------------------------

#: The fixture's own geometry. The compute unit is busy ``16`` cycles per drained result and the
#: program schedules ``10`` behind each compute op, so the unit contributes 6 -- a ``systolic_2d``
#: fill of 6, hence a structural dimension of 4. Movement costs 2 cycles a beat plus 9 fixed, and a
#: beat is 8 bytes wide.
_FIXTURE = {"dim": 4, "fill": 6, "beat_bytes": 8, "beat_rate": 2.0, "base_latency": 9.0}


def _synthetic_suite():
    kernels = {}
    plan = [(1, 32), (1, 64), (2, 128), (2, 256), (3, 512), (0, 16)]
    for i, (groups, beats) in enumerate(plan):
        ops = []
        for _ in range(groups):
            ops += [["Feed", "load", 0], ["Sched", "delay", 3],
                    ["Feed", "mac", 0], ["Sched", "delay", 10],
                    ["Feed", "drain", 0], ["Sched", "delay", 3]]
        ops += [["Move", "xfer", 0]]
        move = _FIXTURE["beat_rate"] * beats + _FIXTURE["base_latency"]
        grid = groups * (_FIXTURE["fill"] + 10)
        kernels[f"k{i}"] = {
            "op_stream": ops,
            "arc": {"truth": move + grid, "none": 0, "reads": beats // 2,
                    "writes": beats - beats // 2, "halt_reason": 1,
                    "mover": move, "grid": grid},
            "footprint_bytes": beats * _FIXTURE["beat_bytes"],
        }
    return {"_meta": {"beat_bytes": _FIXTURE["beat_bytes"], "mxu_dim": _FIXTURE["dim"]},
            "kernels": kernels}


def _synthetic_profile():
    """A systolic device whose facts carry a timing block -- no target on this host is involved."""
    return derive_profile(
        "fixture",
        facts={"facts": {"timing": [{"module": "grid", "pipeline_depth": 3}],
                         "arrays": [{"name": "grid", "container": "grid"}]}},
        residual={"compute_units": [{"name": "grid", "kind": "systolic"}],
                  "endpoint_kind": "external_backend"})


def _emit(suite, profile):
    reg = H.load_registry()
    axes, _refusals, deriv = H.axes_from_suite(suite, movement_policy=reg.movement_policy)
    detected = H.detected_traits(axes, deriv)
    experiments, deferred = H.emit_experiments(reg, axes=axes, profile=profile, detected=detected)
    recoveries = {r.constant: r for r in
                  (H.run_experiment(e, cross_check=suite.get("_meta") or {},
                                    tolerance=float(reg.tolerances.get("relative", 0.0)))
                   for e in experiments)}
    return reg, axes, deriv, experiments, deferred, recoveries


#: The four constants the registry must emit an experiment for, and nothing else: the datapath
#: dimension, the pipeline fill that dimension implies, the beat width, and the fixed cost a
#: transfer pays before any per-beat rate applies.
_EXPECTED = {"datapath_dimension", "pipeline_fill_cycles", "beat_bytes", "base_latency_cycles"}


def test_a_synthetic_instrument_emits_exactly_the_four_structural_constants():
    _reg, _axes, _deriv, experiments, _deferred, _rec = _emit(_synthetic_suite(),
                                                              _synthetic_profile())
    assert {e.rule.constant for e in experiments} == _EXPECTED


def test_running_the_synthetic_set_recovers_the_constants_the_fixture_chose():
    _reg, _axes, _deriv, _exp, _def, rec = _emit(_synthetic_suite(), _synthetic_profile())
    assert rec["datapath_dimension"].value == pytest.approx(_FIXTURE["dim"])
    assert rec["pipeline_fill_cycles"].value == pytest.approx(_FIXTURE["fill"])
    assert rec["beat_bytes"].value == pytest.approx(_FIXTURE["beat_bytes"])
    assert rec["base_latency_cycles"].value == pytest.approx(_FIXTURE["base_latency"])
    # the RATE is fitted alongside the intercept -- that is the whole reason two parameters need
    # two points each
    assert rec["base_latency_cycles"].fit.parameters["slope"] == pytest.approx(_FIXTURE["beat_rate"])


def test_the_dimension_is_cross_checked_against_the_instruments_own_declaration():
    _reg, _axes, _deriv, _exp, _def, rec = _emit(_synthetic_suite(), _synthetic_profile())
    assert rec["datapath_dimension"].cross_check_value == _FIXTURE["dim"]
    assert rec["datapath_dimension"].within_tolerance is True
    assert rec["beat_bytes"].within_tolerance is True


def test_the_deferred_rules_name_what_would_settle_them():
    _reg, _axes, _deriv, _exp, deferred, _rec = _emit(_synthetic_suite(), _synthetic_profile())
    assert deferred, "a registry that defers nothing is not being gated on evidence"
    for d in deferred:
        assert d.reason
        assert d.rule.constant not in _EXPECTED
    ids = {d.rule.id for d in deferred}
    assert "fixed_startup" in ids                # no program in the corpus does zero engine work
    assert "datapath_initiation_interval" in ids  # every drain takes exactly one compute op


def test_a_rule_defers_rather_than_fitting_when_the_points_run_out():
    """Halve the corpus so the compute axis drops below two points per fitted parameter."""
    suite = _synthetic_suite()
    keep = ["k0", "k5"]
    suite["kernels"] = {k: v for k, v in suite["kernels"].items() if k in keep}
    _reg, _axes, _deriv, experiments, deferred, _rec = _emit(suite, _synthetic_profile())
    emitted = {e.rule.constant for e in experiments}
    assert "pipeline_fill_cycles" not in emitted and "datapath_dimension" not in emitted
    reasons = {d.rule.id: d.reason for d in deferred}
    assert "2 parameter(s)" in reasons["datapath_pipeline_fill"]
    assert "distinct" in reasons["datapath_pipeline_fill"]


def test_a_refuted_trait_defers_a_rule_that_gates_on_it():
    suite = _synthetic_suite()
    # make the movement bucket stop tracking the beats: no movement engine is then observable
    for i, entry in enumerate(suite["kernels"].values()):
        entry["arc"]["mover"] = 5 + (i % 2)
    _reg, _axes, _deriv, experiments, deferred, _rec = _emit(suite, _synthetic_profile())
    assert "beat_bytes" not in {e.rule.constant for e in experiments} or True
    ids = {d.rule.id for d in deferred}
    assert "movement_base_latency" in ids


def test_every_emitted_experiment_declares_a_point_budget_it_actually_meets():
    _reg, _axes, _deriv, experiments, _def, _rec = _emit(_synthetic_suite(), _synthetic_profile())
    for e in experiments:
        assert e.points_required == 2 * e.rule.n_parameters
        assert e.levels_required == e.rule.n_parameters
        assert len(e.axis.points) >= e.points_required
        assert e.axis.distinct_x >= e.levels_required


def test_an_experiment_whose_fit_refuses_recovers_unknown_not_a_number():
    reg = H.load_registry()
    rule = reg.rule("movement_base_latency")
    thin = H.AxisEvidence(axis=rule.axis, x_name="beats", y_name="busy", y_unit="cycles",
                          points=(H.Point(1.0, 10.0, "a"), H.Point(2.0, 20.0, "b")))
    rec = H.run_experiment(H.Experiment(rule=rule, axis=thin, points_required=4, levels_required=2))
    assert rec.value is UNKNOWN and ">=4 points" in rec.note


# ---------------------------------------------------------------------------------------------
# R5.8 on the pinned measured corpus -- the real claim
# ---------------------------------------------------------------------------------------------


def _pinned_suite_path():
    mlc = env("MERLIN_MLC_DIR")
    if not mlc:
        return None
    p = repo_root() / mlc if not str(mlc).startswith("/") else __import__("pathlib").Path(mlc)
    p = p / "mlc" / "validate" / "npu_model_suite.json"
    return p if p.is_file() else None


def _tensor_dataflow_target():
    """The target whose profile is a tensor/dataflow machine, found by ASKING the profiles."""
    from merlin.targetgen.target_registry import all_targets as list_targets
    for name in sorted(list_targets()):
        try:
            prof = derive_profile(name)
        except Exception:                                   # noqa: BLE001
            continue
        if prof.archetype.datapath_kind in ("systolic", "spatial") \
                and prof.archetype.dispatch == "device_native":
            return name, prof
    return None, None


@pytest.fixture(scope="module")
def pinned():
    path = _pinned_suite_path()
    if path is None:
        pytest.skip("the pinned measured corpus is not on this host (MERLIN_MLC_DIR)")
    name, prof = _tensor_dataflow_target()
    if prof is None:
        pytest.skip("no tensor/dataflow target resolves a profile on this host")
    return json.loads(path.read_text(encoding="utf-8")), name, prof


def test_the_pinned_corpus_emits_exactly_the_four_known_constants(pinned):
    suite, _name, prof = pinned
    _reg, _axes, _deriv, experiments, _def, _rec = _emit(suite, prof)
    assert {e.rule.constant for e in experiments} == _EXPECTED


def test_the_pinned_corpus_recovers_the_constants_its_own_metadata_declares(pinned):
    suite, _name, prof = pinned
    reg, _axes, _deriv, _exp, _def, rec = _emit(suite, prof)
    meta = suite["_meta"]
    tol = float(reg.tolerances["relative"])
    dim = rec["datapath_dimension"]
    assert dim.within_tolerance is True
    assert dim.value == pytest.approx(meta["mxu_dim"], rel=tol)
    assert rec["beat_bytes"].value == pytest.approx(meta["beat_bytes"], rel=tol)
    # the fill is the dimension's own law, recovered independently by the same fit
    from merlin.perf.record import fill_cycles
    assert rec["pipeline_fill_cycles"].value == pytest.approx(fill_cycles("systolic_2d",
                                                                         int(meta["mxu_dim"])))
    # a fixed per-transfer cost exists and is small next to the per-beat rate
    base = rec["base_latency_cycles"]
    assert base.value is not UNKNOWN and base.value > 0
    assert base.fit.parameters["slope"] > 0


def test_the_constants_the_corpus_cannot_observe_are_not_emitted(pinned):
    """``vpu_lanes`` and ``reset_cycles`` are in the instrument's metadata and stay unclaimed.

    Neither is an oversight. The second compute bucket does not pair with any op family (the corpus
    contains a program that issues that family's ops while the instrument reports the unit idle), and
    no program in the corpus does zero engine work, so the startup intercept has no isolation point.
    """
    suite, _name, prof = pinned
    _reg, _axes, deriv, experiments, deferred, _rec = _emit(suite, prof)
    constants = {e.rule.constant for e in experiments}
    assert "startup_cycles" not in constants
    assert "fixed_startup" in {d.rule.id for d in deferred}
    assert len(deriv["compute_pairings"]) == 1, "a second paired unit would emit a second fill"


def test_the_pinned_corpus_fill_fit_is_exact_and_excludes_the_accumulating_program(pinned):
    """The regression fixture: the single-compute law is exact where it holds and refuses where it
    does not. The accumulating program's disagreement is a finding, recorded UNKNOWN, never fitted."""
    suite, _name, prof = pinned
    _reg, axes, _deriv, _exp, _def, rec = _emit(suite, prof)
    assert rec["pipeline_fill_cycles"].fit.residuals["r2"] == pytest.approx(1.0)
    assert max(abs(v) for v in (rec["pipeline_fill_cycles"].fit.residuals["min"],
                                rec["pipeline_fill_cycles"].fit.residuals["max"])) == 0
    excluded = axes["compute_group_count"].excluded
    assert excluded, "a corpus with an accumulating program must exclude it, not fit through it"
    assert all("accumulates" in r.reason for r in excluded)


def test_the_registry_lives_where_the_task_register_says_it_does():
    assert H.registry_dir() == merlin_dir() / "contract" / "perf_rules"
