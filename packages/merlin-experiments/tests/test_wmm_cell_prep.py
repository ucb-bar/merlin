"""A cell run's config is composed from the measured run's own: the package arm is the loop's certifier
recipe, held-out groups are found by form (one per distinct shape), the collateral baseline is measured
on a named verified package, and the reference arm is measured once by the same path."""

from __future__ import annotations

import json

import pytest
import wmm_fixtures as FX
from merlin_experiments.phase2.whole_model_measured import cell_prep as CP
from merlin_experiments.phase2.whole_model_measured import cells as CELLS
from merlin_experiments.phase2.whole_model_measured import forms as FORMS

FORMS_A = [
    {"group": "1", "form_text": "stem", "shape": {"H": 224}},
    {"group": "2", "form_text": "1x1", "shape": {"H": 56}},
    {"group": "3", "form_text": "1x1", "shape": {"H": 56}},  # same form and shape as g2: a repeat
    {"group": "4", "form_text": "1x1", "shape": {"H": 28}},
    {"group": "5", "form_text": "add", "shape": None},
]
HELD = [
    {"group": "7", "form_text": "1x1", "shape": {"H": 14}},
    {"group": "8", "form_text": "1x1", "shape": {"H": 14}},
    {"group": "9", "form_text": "pool", "shape": None},
]
LOOP = {
    "schema": "merlin_whole_model_objective_config_v1",
    "builder": {"spec": "b:build", "sha256": None},
    "store": "/store",
    "prohibited_instruction_roles": ["loop_descriptor"],
    "instruction_policy": FX.sealed_policy(),
    "screen": {"build_options": {"machine": "board", "header": "/b.h", "prohibited_roles": ["loop_descriptor"]}},
    "certifier": {
        "build_options": {
            "machine": "emu",
            "header": "/e.h",
            "model_capsule": "/m",
            "phase0_recipe": "/r.yaml",
            "prohibited_roles": ["loop_descriptor"],
        }
    },
}


def test_a_form_cell_and_its_held_out_groups_are_one_per_distinct_shape():
    assert CP.form_cell_groups(FORMS_A, [3]) == [2, 4]
    rows = CP.held_out_rows(FORMS_A, [2, 4], {"other": ("/h", HELD)})
    assert rows == [{"label": "other", "model_capsule": "/h", "groups": [7]}]
    assert CP.held_out_rows(FORMS_A, [1], {"other": ("/h", HELD)}) == []  # nothing of the stem's form
    with pytest.raises(CELLS.CellError):
        CP.held_out_rows(FORMS_A, [99], {})


def test_the_cell_config_is_the_loops_certifier_recipe_and_its_bar_carries_no_rule():
    machine = CP.cell_machine_spec(
        target="toy",
        registry_machine="emu",
        cell_id="c",
        model_capsule="/m",
        groups=[1],
        held_out=[],
        reference_build_options=LOOP["certifier"]["build_options"],
    )
    assert machine["measurer"] == CP.CELL_MEASURER and "prohibited_roles" not in machine["reference_build_options"]
    config = CP.cell_objective_config(LOOP, machine=machine, reference="/ref.json", notice="CELL MODE")
    assert config["screen"]["build_options"] == LOOP["certifier"]["build_options"]
    assert config["store"] == "/store/cells" and "certifier" not in config
    from merlin_experiments.phase2.whole_model_measured import config as CFG

    assert CFG.check_policy(config) == ["loop_descriptor"]  # the rule it is built under and judged by agree
    assert config[CFG.SEALED_POLICY] == LOOP["instruction_policy"]  # and the sealed policy it is held to
    unsealed = {k: v for k, v in LOOP.items() if k != "instruction_policy"}
    with pytest.raises(CELLS.CellError, match="sealed Phase 0"):
        CP.cell_objective_config(unsealed, machine=machine, reference="/ref.json", notice="CELL MODE")


def test_prepare_measures_the_collateral_on_the_named_baseline_and_the_reference_once(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(FORMS, "statement_forms", lambda capsule, target: FORMS_A if capsule == "/m" else HELD)

    def baseline(spec, reps, *, baseline_package, target, out, tolerance):
        calls["baseline"] = (spec["build_options"]["phase0_recipe"], [r["group"] for r in reps], str(baseline_package))
        return {"tolerance": tolerance, "groups": [{"group": r["group"], "cycles": 10, "on": "package"} for r in reps]}

    def reference(spec, *, target, out):
        calls["reference"] = spec["reference_build_options"]
        out.mkdir(parents=True, exist_ok=True)
        return {"objective_cycles": 1}

    def own(spec, *, baseline_package, target, out):
        calls["own"] = (spec["cell"]["groups"], str(baseline_package))
        return {"groups": [{"group": 1, "cycles": 40, "on": "package"}]}

    def diagnostics(groups, *, model_capsule, target, functional_model, measured):
        calls["diagnostics"] = (list(groups), dict(measured), functional_model)
        return {"functional_model": functional_model, "rooflines": {"1": {"status": "derived"}}}

    monkeypatch.setattr(CELLS, "measure_collateral_baseline", baseline)
    monkeypatch.setattr(CELLS, "measure_cell_baseline", own)
    monkeypatch.setattr(CP, "cell_diagnostics", diagnostics)
    monkeypatch.setattr(CELLS, "reference_cell", reference)
    prepared = CP.prepare(
        LOOP,
        target="toy",
        cell_id="c",
        groups=[1],
        held_out_capsules=["/h"],
        collateral_share={"2": 5, "3": 50, "4": 1, "5": 9},
        baseline_package=tmp_path / "best",
        out=tmp_path / "cell",
        functional_model={"kind": "spike", "command": ["fm"]},
    )
    assert calls["baseline"] == ("/r.yaml", [3, 5], str(tmp_path / "best"))  # one per OTHER form, by share
    assert "prohibited_roles" not in calls["reference"]
    written = json.loads((tmp_path / "cell" / "cell_objective_config.json").read_text())
    assert written["screen"]["machine"]["cell"]["collateral"]["groups"][0]["group"] == 3
    # The cell's own groups are measured on the same verified package: they bound a candidate's runs.
    assert calls["own"] == ([1], str(tmp_path / "best"))
    assert written["screen"]["machine"]["cell"]["baseline"]["groups"] == [{"group": 1, "cycles": 40, "on": "package"}]
    assert prepared["baseline"]["groups"][0]["cycles"] == 40
    # The cell's rooflines are confronted with the baseline's own cycles, and carried in its spec.
    assert calls["diagnostics"] == ([1], {"1": [("baseline", 40)]}, {"kind": "spike", "command": ["fm"]})
    assert written["screen"]["machine"]["cell"]["diagnostics"]["rooflines"] == {"1": {"status": "derived"}}
    assert "c" in written["harness_notices"][0] and prepared["held_out"] == []
    with pytest.raises(CELLS.CellError, match="verified baseline package"):
        CP.prepare(LOOP, target="toy", cell_id="c", groups=[1], collateral_share={}, out=tmp_path / "x")


def test_a_form_capsule_screen_restricts_the_loops_own_check(tmp_path, monkeypatch):
    """The cell's screen is the loop's pre-measure check narrowed to the cell's form capsules -- the same
    runner and rule, run before any emulator time -- and a loop that declares no check has none to narrow."""
    monkeypatch.setattr(FORMS, "statement_forms", lambda capsule, target: FORMS_A)
    monkeypatch.setattr(CELLS, "reference_cell", lambda spec, *, target, out: {"objective_cycles": 1})
    monkeypatch.setattr(CP, "cell_diagnostics", lambda groups, **kw: {"rooflines": {}})
    with pytest.raises(CELLS.CellError, match="declares none"):
        CP.prepare(LOOP, target="toy", cell_id="c", groups=[1], out=tmp_path / "a", screen_capsules="cap_a")
    loop = {**LOOP, "pre_measure_check": {"argv": ["check", "{out}"], "capsules": "all", "required": False}}
    prepared = CP.prepare(
        loop, target="toy", cell_id="c", groups=[1], out=tmp_path / "b", screen_capsules="cap_a,cap_b"
    )
    check = prepared["config"]["pre_measure_check"]
    assert check["capsules"] == "cap_a,cap_b" and check["required"] is True and check["argv"] == ["check", "{out}"]
    assert (
        "pre_measure_check" not in CP.prepare(LOOP, target="toy", cell_id="c", groups=[1], out=tmp_path / "c")["config"]
    )
