"""The whole-model gate: coverage, feedback, the verdict it folds into, and the freeze it holds.

Each rule is exercised in the direction it can fail as well as pass, on synthetic build records and
gate documents, so the arithmetic and the verdict wiring are tested without a model build; the
build-and-run path is replaced at one seam, ``run``.
"""

from __future__ import annotations

import json

import pytest

from merlin.perf import whole_model_gate as G

MODEL = {"name": "M", "capsule": "c", "machine": "m", "header": "h", "coverage_floor": 1.0}


def _rows():
    return [
        {"group": 0, "op": "quantize", "on": "host", "cause": None},  # the model's host region
        {"group": 1, "op": "conv2d", "on": "vendor", "cause": "package_declined", "why": "does not fit"},
        {"group": 2, "op": "matmul", "on": "package"},
        {
            "group": 3,
            "op": "matmul",
            "on": "host",
            "cause": "library_loop_free_path",
            "declined_as": "package_declined",
        },
        {"group": 4, "op": "matmul", "on": "host", "cause": "accumulator_readout_unavailable"},  # machine limit
    ]


# ------------------------------------------------------------------------------------------ coverage


def test_coverage_counts_only_device_groups_and_every_decline():
    cov = G.coverage(_rows(), MODEL)
    assert cov["device_groups"] == 3  # g1, g2, g3: the host region and the machine limit are outside
    assert cov["package_groups"] == 1
    assert {d["group"] for d in cov["declined"]} == {"1", "3"}
    assert cov["passed"] is False


def test_full_compile_passes_the_floor():
    rows = [{"group": g, "op": "matmul", "on": "package"} for g in (1, 2, 3)] + [_rows()[0]]
    assert G.coverage(rows, MODEL)["passed"] is True


def test_a_lower_floor_is_a_priced_share(tmp_path):
    reference = tmp_path / "result.json"
    reference.write_text(
        json.dumps({"verdict": {"groups": [{"group": "1", "cycles": 10}, {"group": "2", "cycles": 90}]}})
    )
    rows = [{"group": 1, "op": "conv2d", "on": "vendor", "cause": "x"}, {"group": 2, "op": "matmul", "on": "package"}]
    cov = G.coverage(rows, {**MODEL, "coverage_floor": 0.85, "price_reference": str(reference)})
    assert cov["share"] == 0.9 and cov["passed"] is True
    assert G.coverage(rows, {**MODEL, "coverage_floor": 0.95, "price_reference": str(reference)})["passed"] is False


def test_no_device_group_is_never_a_pass():
    assert G.coverage([_rows()[0]], MODEL)["passed"] is False


# ------------------------------------------------------------------------------------------ feedback


def _document(passed=False):
    return {
        "passed": passed,
        "models": [
            {
                "model": "SY_model_x",
                "status": "pass" if passed else "fail",
                "forms": {"1": {"op": "conv2d", "taps": "7x7"}},
                "capsules_of_form": {"g1": {"capsules": ["MF_conv_7x7"]}},
                "checks": {
                    "build": {"passed": True},
                    "coverage": G.coverage(_rows(), MODEL),
                    "no_prohibited_instruction": {"passed": False, "roles": ["r"], "summary": {"X in g2": 3}},
                    "correctness": {
                        "passed": False,
                        "not_correct": [{"group": "2", "kind": "matmul", "failure": {"mismatches": 5, "of": 64}}],
                    },
                    "end_result": {"passed": False, "agrees_with_oracle": False},
                },
            }
        ],
    }


def test_feedback_names_group_form_and_failure():
    text = "\n".join(G.feedback_lines(_document()))
    assert "g1 conv2d" in text and "taps=7x7" in text and "MF_conv_7x7" in text and "NOT COMPILED" in text
    assert "g3" in text and "prohibited instruction" in text  # the loop-free fallback is named as such
    assert "X in g2" in text
    assert "g2 matmul" in text and "5 of 64 elements" in text


def test_feedback_never_carries_the_expected_output():
    """The end result is 'agrees / disagrees', never the class; no line carries an 'oracle' number."""
    doc = _document()
    doc["models"][0]["checks"]["correctness"] = {"passed": True}
    lines = G.feedback_lines(doc)
    assert any("end result" in line for line in lines)
    assert not any(ch.isdigit() for line in lines if "end result" in line for ch in line)


def test_a_build_failure_is_a_failure():
    doc = {"models": [{"model": "M", "status": "fail", "checks": {"build": {"passed": False, "error": "E: boom"}}}]}
    assert G.feedback_lines(doc) == [
        "whole_model M: FAIL build -- the model could not be built with this package: E: boom"
    ]


# ------------------------------------------------------------------------ the verdict and the freeze


@pytest.fixture()
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "program_digest_of", lambda submission: "d" * 64 if submission else None)
    return tmp_path / "run"


def _gate_doc(passed):
    return {"passed": passed, "package_program_digest": "d" * 64, "models": [], "feedback": ["line"]}


def test_attach_blocks_convergence_until_the_gate_passed_on_these_bytes(run_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(G, "run", lambda *a, **k: _gate_doc(False))
    sub = tmp_path / "sub"
    sub.mkdir()
    verdict = {"all_pass": True}
    G.attach(verdict, sub, run_dir, target="t", gate={"models": [1]}, roles=(), key="k", run_now=False)
    assert verdict["all_pass"] is False and "not yet run" in verdict["not_converged_reason"]
    verdict = {"all_pass": True}
    G.attach(verdict, sub, run_dir, target="t", gate={"models": [1]}, roles=(), key="k", run_now=True)
    assert verdict["all_pass"] is False and verdict["whole_model_gates"]["for_this_submission"] is True


def test_attach_passes_a_passing_gate_through(run_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(G, "run", lambda *a, **k: _gate_doc(True))
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "manifest.yaml").write_text("x: 1\n")
    verdict = {"all_pass": True}
    G.attach(verdict, sub, run_dir, target="t", gate={"models": [1]}, roles=(), key="k", run_now=True)
    assert verdict["all_pass"] is True and verdict["whole_model_gates"]["passed"] is True
    assert G.lookup(run_dir, "d" * 64)["passed"] is True


def test_a_recorded_program_is_not_rebuilt(run_dir, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(G, "run", lambda *a, **k: calls.append(1) or _gate_doc(True))
    sub = tmp_path / "sub"
    sub.mkdir()
    for _ in range(3):
        G.evaluate(sub, run_dir, target="t", gate={}, roles=(), key="k")
    assert len(calls) == 1


# ------------------------------------------------------------------------------ the descriptor and loop


# ------------------------------------------------------------------------------------ pipeline select


def test_a_model_declared_not_required_is_reported_and_does_not_hold_the_gate(monkeypatch, tmp_path):
    outcomes = {"A": "pass", "B": "fail"}

    def fake(package, model, **_kwargs):
        return {
            "model": model["name"],
            "required": model.get("required", True) is not False,
            "status": outcomes[model["name"]],
            "checks": {"build": {"passed": False, "error": "x"}},
        }

    monkeypatch.setattr(G, "run_model", fake)
    monkeypatch.setattr(G, "program_digest_of", lambda s: "d")
    gate = {"models": [{**MODEL, "name": "A"}, {**MODEL, "name": "B", "required": False}]}
    document = G.run(tmp_path, gate, target="t", roles=(), out=tmp_path / "o", root=tmp_path)
    assert document["passed"] is True
    assert any("B (reported, not required)" in line for line in document["feedback"])
    gate["models"][1]["required"] = True
    assert G.run(tmp_path, gate, target="t", roles=(), out=tmp_path / "o2", root=tmp_path)["passed"] is False


def test_the_written_verdict_names_the_hardware_it_came_from(monkeypatch, tmp_path):
    def fake(package, model, **_kwargs):
        return {"model": model["name"], "required": True, "status": "pass", "checks": {}}

    monkeypatch.setattr(G, "run_model", fake)
    monkeypatch.setattr(G, "program_digest_of", lambda s: "d")
    G.run(tmp_path, {"models": [{**MODEL, "name": "A"}]}, target="t", roles=(), out=tmp_path / "o", root=tmp_path)
    written = json.loads((tmp_path / "o" / G.RESULT_FILE).read_text())
    assert written["passed"] is True
    assert "hardware_pins" in written["provenance"]


# ---------------------------------------------------------------------- the machine, from the readout fact


def _attribution(*causes):
    return {"attribution": {"per_group": [{"group": i, "cause": c} for i, c in enumerate(causes)]}}


def test_the_declared_machine_is_kept_when_every_group_can_be_read_out():
    chosen = G.choose_machine(MODEL, _attribution(None, "package_declined"), target="gemmini", headers=())
    assert chosen["machine"] == "m" and chosen["header"] == "h"


def test_a_model_that_needs_a_full_width_readout_moves_to_the_machine_the_fact_derives(tmp_path, monkeypatch):
    header = tmp_path / "params.h"
    header.write_text("#define X 1\n")
    digest = G._sha256(header)
    monkeypatch.setattr(
        G,
        "full_width_machines",
        lambda target, headers=(): [{"machine": "full", "header_sha256": digest, "registry_declared": True}],
    )
    record = _attribution(G.CAUSE_READOUT_UNAVAILABLE, None)
    chosen = G.choose_machine(MODEL, record, target="t", headers=[str(header)])
    assert chosen["machine"] == "full" and chosen["header"] == str(header) and "g0" in chosen["why"]


def test_no_full_width_machine_with_a_declared_header_fails_closed(monkeypatch):
    monkeypatch.setattr(
        G,
        "full_width_machines",
        lambda target, headers=(): [{"machine": "full", "header_sha256": "0" * 64, "registry_declared": True}],
    )
    with pytest.raises(ValueError, match="refuses rather than skip"):
        G.choose_machine(MODEL, _attribution(G.CAUSE_READOUT_UNAVAILABLE), target="t", headers=[])


def test_a_model_declaring_machine_limited_groups_on_core_keeps_its_machine(monkeypatch):
    monkeypatch.setattr(G, "full_width_machines", lambda target, headers=(): [])
    model = {**MODEL, "machine_limited_groups": G.ON_CORE}
    chosen = G.choose_machine(model, _attribution(G.CAUSE_READOUT_UNAVAILABLE), target="t", headers=[])
    assert chosen["machine"] == "m" and chosen["machine_limited_groups"] == ["0"]


def test_coverage_names_what_it_leaves_outside_the_denominator():
    outside = G.coverage(_rows(), MODEL)["outside_denominator"]
    assert {row["group"] for row in outside} == {"0", "4"}


def test_the_gate_names_the_open_builders_readout_refusal_as_it_does():
    from merlin.perf import whole_model_open as WO

    assert G.MACHINE_CANNOT_READ_OUT == WO.MACHINE_CANNOT_READ_OUT


def test_a_builder_that_refuses_the_machine_moves_the_gate_to_a_full_width_one(tmp_path, monkeypatch):
    header = tmp_path / "params.h"
    header.write_text("#define X 1\n")
    monkeypatch.setattr(
        G,
        "full_width_machines",
        lambda target, headers=(): [{"machine": "full", "header_sha256": G._sha256(header), "registry_declared": True}],
    )
    built = []

    def fake_build(package, **kwargs):
        built.append(kwargs["machine"])
        if kwargs["machine"] == "m":
            raise RuntimeError(f"{G.MACHINE_CANNOT_READ_OUT}: 302 of the model's device groups commit ...")
        raise RuntimeError("stop after the choice")

    from merlin.perf import whole_model_builder as B

    monkeypatch.setattr(B, "build", fake_build)
    G._READOUT_CHOICE.clear()
    model = {**MODEL, "capsule": str(tmp_path / "cap")}
    result = G.run_model(tmp_path, model, target="t", roles=(), out=tmp_path / "o", headers=[str(header)])
    assert built == ["m", "full"]
    assert result["machine"] == "m" or result["checks"]["build"]["passed"] is False
    assert G._READOUT_CHOICE[G._choice_key(model)]["machine"] == "full"


def test_a_build_that_fails_part_way_does_not_keep_its_intermediates(tmp_path, monkeypatch):
    out = tmp_path / "o"
    lowered = out / "build" / "lower" / "g1.generated" / "lowered.target.mlir"

    def fake_build(package, **kwargs):
        lowered.parent.mkdir(parents=True)
        lowered.write_text("module {}\n")
        raise RuntimeError("PipelineError: upstream lowering failed")

    from merlin.perf import whole_model_builder as B

    monkeypatch.setattr(B, "build", fake_build)
    G._READOUT_CHOICE.clear()
    model = {**MODEL, "capsule": str(tmp_path / "cap")}
    G._READOUT_CHOICE[G._choice_key(model)] = {"machine": "m", "header": "h"}
    result = G.run_model(tmp_path, model, target="t", roles=(), out=out)
    assert result["checks"]["build"]["passed"] is False
    assert "upstream lowering failed" in result["checks"]["build"]["error"]
    assert not lowered.exists()

    lowered.parent.rmdir()
    G.run_model(tmp_path, model, target="t", roles=(), out=out, keep_build=True)
    assert lowered.exists()


def test_the_gate_hands_the_builder_its_corpus_binding_and_chunk_size(tmp_path, monkeypatch):
    """A package build refuses to guess the corpus binding, so the gate passes the experiment's
    recipe and descriptor straight to the builder, beside the chunk size it was asked for."""
    seen = {}

    def fake_build(package, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after the call")

    from merlin.perf import whole_model_builder as B

    monkeypatch.setattr(B, "build", fake_build)
    G._READOUT_CHOICE.clear()
    model = {**MODEL, "capsule": str(tmp_path / "cap")}
    G._READOUT_CHOICE[G._choice_key(model)] = {"machine": "m", "header": "h"}
    recipe, descriptor = tmp_path / "recipe.yaml", tmp_path / "descriptor.yaml"
    G.run_model(
        tmp_path,
        model,
        target="t",
        roles=(),
        out=tmp_path / "o",
        chunk_ops=64,
        phase0_recipe=recipe,
        descriptor=descriptor,
    )
    assert seen["phase0_recipe"] == str(recipe) and seen["descriptor"] == str(descriptor) and seen["chunk_ops"] == 64

    seen.clear()
    G.run_model(tmp_path, model, target="t", roles=(), out=tmp_path / "o")
    assert seen["phase0_recipe"] is None and seen["descriptor"] is None and seen["chunk_ops"] is None


def test_the_end_result_is_the_class_or_the_output_tensor_never_the_expected_value():
    assert G.end_result({"argmax": [21, 21, 1]}, {"argmax": 21}) == {
        "passed": True,
        "basis": "class",
        "agrees_with_oracle": True,
    }
    assert G.end_result({"argmax": [3, 21, 0]}, {"argmax": 21})["passed"] is False
    tensor = {"argmax": None, "output": {"elements": 1600}}
    assert G.end_result({"output": [1600, 1600]}, tensor)["passed"] is True
    assert G.end_result({"output": [1599, 1600]}, tensor)["passed"] is False
    assert G.end_result({}, tensor)["passed"] is False
    assert G.end_result({}, {"argmax": None})["passed"] is False


def test_a_reference_arm_is_prepared_on_the_machine_the_gate_would_choose(tmp_path, monkeypatch):
    """The reference arm is a property of the model and the machine, never of a package, so it is
    produced ahead of any gate -- on the full-width machine the gate itself moves to when the declared
    one cannot read the model's accumulators out. A closed model has no reference arm and is skipped."""
    from merlin.perf import whole_model_open as WO
    from merlin.perf import whole_model_reference as R

    header = tmp_path / "params.h"
    header.write_text("#define X 1\n")
    monkeypatch.setattr(
        G,
        "full_width_machines",
        lambda target, headers=(): [{"machine": "full", "header_sha256": G._sha256(header), "registry_declared": True}],
    )
    monkeypatch.setattr(WO, "is_open_model", lambda capsule, target: str(capsule).endswith("open"))
    asked = []

    def fake_prepare(capsule, **kwargs):
        asked.append(kwargs["machine"])
        if kwargs["machine"] == "m":
            raise RuntimeError(f"{G.MACHINE_CANNOT_READ_OUT}: every device group ...")
        return {"key": "k" * 64, "output_digest": {"bytes": 4, "digest": 1}, "dispatches": 2, "wall_s": 1.0}

    monkeypatch.setattr(R, "prepare", fake_prepare)
    G._READOUT_CHOICE.clear()
    gate = {
        "headers": [str(header)],
        "models": [
            {**MODEL, "name": "closed", "capsule": str(tmp_path / "closed")},
            {**MODEL, "name": "open", "capsule": str(tmp_path / "open")},
        ],
    }
    rows = G.prepare_references(gate, target="t", root=tmp_path)
    assert asked == ["m", "full"]
    assert rows[0] == {"model": "closed", "skipped": "a closed model is judged on its class"}
    assert rows[1]["machine"] == "full" and rows[1]["key"] == "k" * 64
    assert G._READOUT_CHOICE[G._choice_key({**MODEL, "capsule": str(tmp_path / "open")})]["machine"] == "full"


def test_the_gate_and_its_roles_are_read_from_the_declarations(tmp_path, monkeypatch):
    descriptor = tmp_path / "target_experiment.yaml"
    descriptor.write_text(
        "target: t\nphase1_gates:\n  whole_model:\n    models: [{name: M, capsule: c, machine: m, header: h}]\n"
    )
    monkeypatch.setenv(G.ROLES_ENV, "loop_descriptor")
    gate, roles = G.gate_for("t", descriptor=descriptor)
    assert roles == ("loop_descriptor",) and gate["models"][0]["name"] == "M"
    monkeypatch.delenv(G.ROLES_ENV)
    descriptor.write_text("target: t\n")
    assert G.gate_for("t", descriptor=descriptor) == (None, ())


def test_a_block_holding_only_another_gates_declaration_is_no_capsule_model_gate(tmp_path):
    descriptor = tmp_path / "target_experiment.yaml"
    descriptor.write_text(
        "target: t\nphase1_gates:\n  whole_model:\n    private_full_models: {required: true, programs: {}}\n"
    )
    assert G.gate_for("t", descriptor=descriptor) == (None, ())
    assert G.main(["--target", "t", "--descriptor", str(descriptor)]) == 2


def test_the_command_refuses_a_target_that_declares_no_gate(tmp_path, capsys):
    descriptor = tmp_path / "target_experiment.yaml"
    descriptor.write_text("target: t\n")
    assert G.main(["--target", "t", "--descriptor", str(descriptor)]) == 2
    assert "declares no whole-model gate" in capsys.readouterr().err


def test_a_checkout_without_the_readout_width_derivation_refuses_by_name_not_import_error(monkeypatch):
    """The derivation module may be absent on a checkout: no full-width machine is then known, and a
    model that needs one is refused with the reason, never an ImportError in the middle of the gate."""
    import importlib

    real = importlib.import_module

    def without(name, *args, **kwargs):
        if name.endswith("semantic_facts"):
            raise ImportError(f"No module named {name!r}")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", without)
    assert G.full_width_machines("t") == []
    with pytest.raises(ValueError, match="none derived"):
        G.choose_machine(MODEL, _attribution(G.CAUSE_READOUT_UNAVAILABLE), target="t", headers=[])


def test_the_full_width_machine_is_derived_from_the_targets_own_probes_with_the_declared_headers(monkeypatch):
    """The gate derives the readout-width fact itself, handing the derivation the declared headers a
    machine's header is found among by its registry digest -- no shipped facts document is read."""
    from merlin.targetgen.rtl import semantic_facts as SF

    asked = []

    def readout(target, *, headers=()):
        asked.append((target, list(headers)))
        return {
            "status": "derived",
            "value": {
                "machines": {
                    "narrow": {"status": "derived", "full_width_readout": False, "header": {"sha256": "1" * 64}},
                    "wide": {
                        "status": "derived",
                        "full_width_readout": True,
                        "header": {"sha256": "2" * 64, "registry_abi_header": {"declared_by": "wide", "agrees": True}},
                    },
                }
            },
        }

    monkeypatch.setattr(SF, "readout_machines", readout)
    rows = G.full_width_machines("t", ["/h/a.h"])
    assert rows == [{"machine": "wide", "header_sha256": "2" * 64, "registry_declared": True}]
    assert asked == [("t", ["/h/a.h"])]
