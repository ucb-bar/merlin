"""Explicit host policy reaches real scans without launching candidate code or hardware."""

from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.context import InvocationContext
from merlin_experiments.phase1.feedback import qa

from merlin.targetgen import capsule_grade as CG
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen import lowering_coverage as LC
from merlin.targetgen import package_runtime as PR


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    directory = tmp_path / "candidate"
    directory.mkdir()
    (directory / "candidate.py").write_text("# outputs_match\n")
    pkg = SimpleNamespace(directory=directory, integrity_exempt=False)

    def stop_build(_):
        raise PR.CertFailure("build", "elaboration_error", "fixture reached build")

    for module in (PR, CG):
        monkeypatch.setattr(module, "load_package", lambda *a, **kw: pkg)
        monkeypatch.setattr(module, "build_package", stop_build)
    monkeypatch.setattr(CG, "source_experiment_env", lambda _: {})
    return directory


def test_interleaved_capsule_grade_uses_real_scanner(candidate, tmp_path):
    for markers in ((), ("outputs_match",), ()):
        result = CG.grade(
            candidate,
            capsules_root=tmp_path / "capsules",
            runs_root=tmp_path / "runs",
            target="fixture",
            additional_forbidden=markers,
        )
        assert result["failure"]["plane"] == ("integrity" if markers else "build")
        assert result["functional_pass"] == 0


def test_shape_sweep_forwards_policy_through_entrypoints(candidate, monkeypatch):
    monkeypatch.setattr(
        LC,
        "_binding",
        lambda _: SimpleNamespace(
            operand_dtype="int8", accum_dtype="int32", mlir_dtype=lambda x: "i8" if x == "int8" else "i32"
        ),
    )
    monkeypatch.setattr(LC, "tile_edge", lambda _: 2)
    for markers in ((), ("outputs_match",), ()):
        result = LC.sweep(
            candidate,
            target="fixture",
            corners={"tile": (1, 1, 1)},
            tail_corners={"m_tail": (1, 0, 0)},
            additional_forbidden=markers,
        )
        assert not result["all_covered"]
        assert len(result["corners"]) == 2
        for corner in result["corners"]:
            assert corner["outcome"] == "error"
            assert corner["detail"].startswith("integrity:" if markers else "build:")


@pytest.mark.parametrize("rtl_treatment", [False, True])
def test_qa_forwards_policy_to_real_capsule_scan(candidate, tmp_path, monkeypatch, rtl_treatment):
    context = InvocationContext(
        tmp_path,
        tmp_path / "missing.yaml",
        tmp_path,
        "fixture",
        tmp_path / "runs",
        tmp_path / "reports",
        tmp_path / "bundles",
        (),
    )
    monkeypatch.setattr(CR, "qa_checkpoint_adapters", lambda *a: {})
    monkeypatch.setattr(qa, "_per_capsule_from_results", lambda *a: {})
    # Observe the real grade's result without replacing its scanner or policy handling.
    grade = CG.grade
    seen = []

    def observe(*args, **kwargs):
        result = grade(*args, **kwargs)
        seen.append(result["failure"]["plane"])
        return result

    monkeypatch.setattr(CG, "grade", observe)
    run = qa.run
    extra = {"context": context}
    if rtl_treatment:
        from merlin_experiments.phase1.feedback import rtlchecks

        monkeypatch.setattr(rtlchecks, "feedback", lambda *args, **kwargs: [])
        run = rtlchecks.run
    for markers in ((), ("outputs_match",), ()):
        run(
            str(candidate),
            str(tmp_path / "capsules"),
            tmp_path / "runs",
            {"public"},
            True,
            1,
            **extra,
            additional_forbidden=markers,
        )
    assert seen == ["build", "integrity", "build"]


def test_authoring_policy_selection_and_l3_closure(candidate, tmp_path, monkeypatch):
    """Execute the real nested L3 body with explicit closure inputs, not hardware/admission."""
    import ast
    import inspect
    import shutil

    from merlin_experiments.phase1 import authoring as A
    from merlin_experiments.phase1.feedback import loop_grading as G

    body = ast.parse(inspect.getsource(A.execute)).body[0].body
    assignment = next(
        node
        for node in body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "grading_inputs" for t in node.targets)
    )
    policy = next(kw.value for kw in assignment.value.keywords if kw.arg == "additional_forbidden")
    selection = compile(ast.Expression(policy), "authoring-policy", "eval")
    for experiment in ("realistic", "diagnostic"):
        for arm in ("merlin_assisted", "raw_baseline", "plain"):
            result = eval(selection, {"a": SimpleNamespace(experiment=experiment), "arm": arm})
            expected = ("xdsl_dialects.lowering", "xdsl_dialects/lowering", "outputs_match")
            assert result == (expected if experiment == "realistic" and arm == "merlin_assisted" else ())

    nested = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == "_verilator_grade")
    ws = tmp_path / "workspace"
    ws.mkdir()
    candidate.rename(ws / "submission")
    (ws / "submission/manifest.yaml").write_text("language: python\n")
    context = SimpleNamespace(repo=tmp_path)
    scope = dict(
        vars(A),
        ws=ws,
        run_dir=tmp_path / "run",
        context=context,
        bundle_id="fixture",
        _verify_implementation_sources=lambda: None,
        _te=lambda: SimpleNamespace(target="fixture", sim_via=""),
        _contract_root=None,
        treatment=SimpleNamespace(checkpoint_feedback=None),
        shutil=shutil,
    )
    monkeypatch.setattr(A.CERT, "_cert_tier_name", lambda **kw: "L3")
    monkeypatch.setattr(A.CERT, "_pilot_capsule_dirs", lambda **kw: {"fixture": tmp_path / "capsules"})
    monkeypatch.setattr(A.CERT, "_verilator_l3_budget", lambda *a, **kw: 1)
    monkeypatch.setattr(CR, "qa_checkpoint_adapters", lambda *a: {})
    monkeypatch.setattr(qa, "_per_capsule_from_results", lambda _: {})
    monkeypatch.setattr(
        CG, "load_package", lambda directory, **kw: SimpleNamespace(directory=directory, integrity_exempt=False)
    )
    grade = CG.grade
    seen = []

    def observe(*args, **kwargs):
        result = grade(*args, **kwargs)
        seen.append(result["failure"]["plane"])
        return result

    monkeypatch.setattr(CG, "grade", observe)
    exec(compile(ast.Module(body=[nested], type_ignores=[]), "authoring-l3", "exec"), scope)
    for attempt, markers in enumerate(((), ("outputs_match",), ())):
        scope["grading_inputs"] = G.GradingInputs(
            context, "merlin_assisted", tmp_path, (), None, additional_forbidden=markers
        )
        assert not scope["_verilator_grade"](attempt, ["fixture"], [])["all_pass"]
    assert seen == ["build", "integrity", "build"]


def test_fast_grade_forwards_policy_to_real_scan(candidate, tmp_path, monkeypatch):
    from dataclasses import replace

    from merlin_experiments.phase1.feedback import loop_grading as G

    context = InvocationContext(
        tmp_path,
        tmp_path / "missing.yaml",
        tmp_path,
        "fixture",
        tmp_path / "runs",
        tmp_path / "reports",
        tmp_path / "bundles",
        (),
    )
    ws = tmp_path / "workspace"
    ws.mkdir()
    candidate.rename(ws / "submission")
    (ws / "submission/manifest.yaml").write_text("language: python\n")
    # The load seam must point at each actual immutable scratch copy.
    monkeypatch.setattr(
        CG, "load_package", lambda directory, **kw: SimpleNamespace(directory=directory, integrity_exempt=False)
    )
    monkeypatch.setattr(G, "load_target_experiment", lambda _: SimpleNamespace(target="fixture", sim_via=""))
    monkeypatch.setattr(G, "declared_loop_tiers", lambda _: set())
    monkeypatch.setattr(G, "cert_tiers_beyond_loop", lambda **kw: (set(), set()))
    monkeypatch.setattr(G, "_record_plateau", lambda _: None)
    monkeypatch.setattr(CR, "qa_loop_adapters", lambda *a, **kw: {})
    monkeypatch.setattr(qa, "_per_capsule_from_results", lambda _: {})
    grade = CG.grade
    seen = []

    def observe(*args, **kwargs):
        result = grade(*args, **kwargs)
        seen.append(result["failure"]["plane"])
        return result

    monkeypatch.setattr(CG, "grade", observe)
    inputs = G.GradingInputs(context, "raw_baseline", tmp_path / "capsules", (), None)
    for tick, markers in enumerate(((), ("outputs_match",), ())):
        G.fast_grade(ws, tmp_path / "run", tick, 1, inputs=replace(inputs, additional_forbidden=markers))
    assert seen == ["build", "integrity", "build"]
