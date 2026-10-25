"""Complete-output and nested-site checks, with the float law kept provisional."""

import json
from pathlib import Path

import pytest
from lean_scalar_square_cases import actual, nested, scalar

from merlin.common.paths import artifacts_dir
from merlin.verify.lean.export import export_file
from merlin.verify.lean.scalar_square import census, prepare, run_checker


@pytest.fixture(scope="module")
def square_driver():
    return prepare()


def check(tmp_path, lean_tools, square_driver, before, after):
    paths = []
    for label, text in (("before", before), ("after", after)):
        source = tmp_path / f"{label}.mlir"
        source.write_text(text)
        output = tmp_path / f"{label}.json"
        export_file(source, output, backend="mlir", m2m_python=lean_tools.m2m_python)
        paths.append(output)
    result = run_checker(*paths, square_driver)
    assert result["candidate_paths"] == [s["path"] for s in census(json.loads(paths[0].read_text()))]
    assert result["verdict"] == "abstracted"
    assert result["numerical_law_status"] == "provisional_not_discharged"
    return result


@pytest.mark.parametrize("kind", ["f32", "f64"])
@pytest.mark.parametrize("flags", ["none", "contract"])
def test_actual_scalar_square(tmp_path, lean_tools, square_driver, kind, flags):
    before, after, count = actual(scalar(kind, flags))
    assert count == 1
    result = check(tmp_path, lean_tools, square_driver, before, after)
    assert result["checked"] and result["covered_sites"] == 1


@pytest.mark.parametrize("kind", ["f32", "f64"])
def test_all_nested_regions_and_captured_constant(tmp_path, lean_tools, square_driver, kind):
    before, after, count = actual(nested(kind))
    assert count == 3
    result = check(tmp_path, lean_tools, square_driver, before, after)
    assert result["checked"] and result["covered_sites"] == 3
    assert max(len(site["path"]) for site in result["sites"]) >= 10


@pytest.mark.parametrize("mutation", ["mixed", "other_square", "exponent", "flags", "unrelated", "omitted"])
@pytest.mark.parametrize("kind", ["f32", "f64"])
def test_refuses_corruptions(tmp_path, lean_tools, square_driver, mutation, kind):
    from lean_scalar_square_cases import render
    from xdsl.dialects import arith, func
    from xdsl.dialects.builtin import FloatAttr

    from merlin.frontends.linalg_mlir import parse_mlir_text

    before, after, _ = actual(scalar(kind))
    module = parse_mlir_text(after)
    multiply = next(op for op in module.walk() if isinstance(op, arith.MulfOp))
    x, y = multiply.parent.args
    if mutation == "mixed":
        multiply.operands = (x, y)
    elif mutation == "other_square":
        multiply.operands = (y, y)
    elif mutation == "exponent":
        source = parse_mlir_text(before)
        constant = next(op for op in source.walk() if isinstance(op, arith.ConstantOp))
        constant.properties["value"] = FloatAttr(3.0, constant.result.type)
        before = render(source)
    elif mutation == "flags":
        multiply.properties["fastmath"] = arith.FastMathFlagsAttr([arith.FastMathFlag.NO_NANS])
    elif mutation == "unrelated":
        next(op for op in module.walk() if isinstance(op, func.ReturnOp)).operands = (y,)
    elif mutation == "omitted":
        module = parse_mlir_text(before)
    module.verify()
    assert not check(tmp_path, lean_tools, square_driver, before, render(module))["checked"]


@pytest.mark.parametrize("kind", ["f32", "f64"])
def test_last_nested_site_cannot_be_omitted(tmp_path, lean_tools, square_driver, kind):
    from lean_scalar_square_cases import render
    from xdsl.dialects import arith

    from merlin.frontends.linalg_mlir import parse_mlir_text

    before, after, _ = actual(nested(kind))
    module = parse_mlir_text(after)
    multiplies = [op for op in module.walk() if isinstance(op, arith.MulfOp)]
    last = multiplies[-1]
    function = next(op for op in module.walk() if op.name == "func.func")
    last.operands = (function.body.block.args[2], function.body.block.args[2])
    module.verify()
    result = check(tmp_path, lean_tools, square_driver, before, render(module))
    assert result["coverage_passed"] and result["covered_sites"] == 3
    assert not result["complete_output_match"] and not result["checked"]


@pytest.mark.parametrize("exponent", ["0.0", "-2.0", "3.0"])
def test_wrong_exponent_cannot_license_square(tmp_path, lean_tools, square_driver, exponent):
    before, _, _ = actual(scalar(exponent=exponent))
    _, after, _ = actual(scalar())
    assert not check(tmp_path, lean_tools, square_driver, before, after)["checked"]


@pytest.mark.parametrize("flags", ["nnan", "ninf", "nsz", "afn", "fast"])
def test_unsupported_flags_do_not_silently_become_identity(tmp_path, lean_tools, square_driver, flags):
    before, after, _ = actual(scalar(flags=flags))
    result = check(tmp_path, lean_tools, square_driver, before, after)
    assert not result["checked"] and not result["coverage_passed"]
    assert len(result["candidate_paths"]) == 1


def test_replay_reexecutes_forged_green_report(tmp_path, lean_tools):
    from merlin.verify.lean.check import _file_ref
    from merlin.verify.lean.scalar_square import check_stage, replay

    before, after, _ = actual(scalar())
    after = after.replace('"func.return"(%result)', '"func.return"(%y)')
    stage = tmp_path / "stage"
    stage.mkdir()
    for name, text in (("before", before), ("after", after)):
        (stage / f"{name}.mlir").write_text(text)
    (stage / "meta.json").write_text(json.dumps({"boundary": "prepare-scalar-squares"}))
    destination = tmp_path / "receipt"
    receipt = check_stage(stage, destination, compiler_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "abstracted", receipt
    assert not receipt["conditional_match"]
    path = Path(receipt["report"]["path"])
    report = json.loads(path.read_text())
    report.update(checked=True, complete_output_match=True)
    path.write_text(json.dumps(report))
    receipt["report"] = _file_ref(path)
    (destination / "receipt.json").write_text(json.dumps(receipt))
    result = replay(destination / "receipt.json", tmp_path / "replay", compiler_python=lean_tools.m2m_python)
    assert result["fresh"] and not result["matches_recorded"] and not result["conditional_match"]


@pytest.mark.slow
@pytest.mark.parametrize("model", ["SY_model_smolvla", "SY_model_tiny_llama"])
def test_real_boundary_receipt_replays(tmp_path, lean_tools, model):
    from merlin.verify.lean.scalar_square import replay

    path = artifacts_dir() / "verification/merlin-passes/frontend/scalar-square" / model / "receipt.json"
    assert path.is_file(), "retain the actual model receipt before running qualification"
    result = replay(path, tmp_path / "replay", compiler_python=lean_tools.m2m_python)
    assert result["fresh"] and result["matches_recorded"] and result["conditional_match"]
