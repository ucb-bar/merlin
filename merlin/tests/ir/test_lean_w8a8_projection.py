"""Qualify actual W8A8 projection contractions and refuse altered maps, grouping and bodies."""

from __future__ import annotations

import json
import subprocess

import pytest
from lean_w8a8_projection_cases import CASES, Projection, changed_source_pair
from test_lean_w8a8_bounded_chain import check_chain, lowered, preprocessed

from merlin.common.paths import repo_root
from merlin.llvmlower.passes_quant_int import lower_contraction_int8

pytestmark = pytest.mark.usefixtures("lean_oracle")
BOUNDS = [{"function": "forward", "intervals": [None, None]}]


def component_report(path):
    proc = subprocess.run(
        [
            "lake",
            "env",
            "lean",
            "--run",
            str(repo_root() / "merlin/tests/fixtures/lean_w8a8_projection.lean"),
            str(path),
        ],
        cwd=repo_root() / "merlin/lean",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["checker"] == "MerlinVerify.Check.PolicyW8A8Projection.check"
    assert report["theorem"] == "MerlinVerify.Check.PolicyW8A8Projection.check_sound"
    return report["components"]


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_projection_contractions_match_the_conditional_reference(tmp_path, lean_tools, name, index_bits):
    import math

    case = CASES[name]
    before = case.source()
    report, documents, _ = check_chain([before, lowered(before)], BOUNDS, tmp_path, lean_tools, index_bits=index_bits)
    assert report["conditional_initialized_policy_refinement"], report
    assert component_report(documents[1]) == [
        {
            "checked": True,
            "ranges": list(case.ranges),
            "output_shape": list(case.shape(case.output)),
            "reduction_size": math.prod(
                size for size, keep in zip(case.ranges, case.parallel, strict=True) if not keep
            ),
        }
    ]


@pytest.mark.parametrize("name", ["batch", "interleaved", "two_reductions", "scalar"])
@pytest.mark.parametrize("integer_left", [False, True])
def test_mixed_bytes_and_accumulator_order_do_not_change_source_grouping(tmp_path, lean_tools, name, integer_left):
    before = CASES[name].source(integer_left=integer_left, accumulator_first=True)
    report, _, _ = check_chain([before, lowered(before)], BOUNDS, tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"], report


@pytest.mark.parametrize("name", ["batch", "interleaved", "two_reductions"])
def test_projection_policy_composes_through_real_preprocessing(tmp_path, lean_tools, monkeypatch, name):
    before = CASES[name].source()
    after = lowered(before)
    rest = preprocessed(after, tmp_path, monkeypatch)
    report, _, _ = check_chain([before, after, *rest], BOUNDS, tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"], report


@pytest.mark.parametrize("extent,accepted", [(131071, True), (131072, False), (131073, False)])
def test_actual_reduction_size_controls_the_signed_i32_bound(tmp_path, lean_tools, extent, accepted):
    case = Projection((extent, 1), (False, False), (0, 1), (1, 0))
    before = case.source()
    report, _, _ = check_chain([before, lowered(before)], BOUNDS, tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"] is accepted


def test_reduction_bound_multiplies_all_reduction_extents(tmp_path, lean_tools):
    case = Projection((512, 256), (False, False), (0, 1), (1, 0))
    before = case.source()
    report, _, _ = check_chain([before, lowered(before)], BOUNDS, tmp_path, lean_tools)
    assert not report["conditional_initialized_policy_refinement"]


@pytest.mark.parametrize("change", ["input_map", "output_map", "iterator", "group", "body", "attribute"])
def test_mutated_actual_geometry_or_quantization_cannot_match_the_reference(tmp_path, lean_tools, change):
    before = CASES["attention"].source()
    report, documents, proc = check_chain([before, lowered(before)], BOUNDS, tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"]
    document = json.loads(documents[1].read_text())
    ops = document["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
    contraction = next(
        op
        for op in ops
        if op["name"] == "linalg.generic" and op["results"][0]["type"]["element"] == {"kind": "int", "width": 32}
    )
    attrs = contraction["attrs"]
    if change in {"input_map", "output_map"}:
        maps = attrs["indexing_maps"]["elements"]
        maps[0 if change == "input_map" else 2]["results"].reverse()
    elif change == "iterator":
        attrs["iterator_types"]["elements"][0]["value"] = "reduction"
    elif change == "group":
        requant = ops[-2]
        maps = requant["attrs"]["indexing_maps"]["elements"]
        maps[1]["results"].reverse()
    elif change == "body":
        multiply = next(op for op in contraction["regions"][0]["blocks"][0]["ops"] if op["name"] == "arith.muli")
        multiply["operands"][1] = multiply["operands"][0]
    else:
        attrs["unreviewed_mode"] = {"kind": "unit"}
    documents[1].write_text(json.dumps(document))
    changed = subprocess.run(proc.args, cwd=repo_root() / "merlin/lean", capture_output=True, text=True, check=False)
    assert changed.returncode == 1, changed.stdout + changed.stderr
    assert not json.loads(changed.stdout)["conditional_initialized_policy_refinement"]


def test_nonprojection_source_is_refused_instead_of_silently_omitted(tmp_path, lean_tools):
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr
    from xdsl.ir.affine import AffineExpr, AffineMap

    before = CASES["batch"].source()
    contraction = next(op for op in before.walk() if op.name == "linalg.generic")
    maps = list(contraction.indexing_maps)
    maps[0] = AffineMapAttr(
        AffineMap(4, 0, (AffineExpr.dimension(0), AffineExpr.dimension(1), AffineExpr.dimension(3) % 4))
    )
    contraction.properties["indexing_maps"] = ArrayAttr(maps)
    assert lower_contraction_int8(before.clone()) == 0
    report, _, _ = check_chain([before, before.clone()], BOUNDS, tmp_path, lean_tools)
    assert not report["conditional_initialized_policy_refinement"]


def test_output_size_is_not_mistaken_for_reduction_length(tmp_path, lean_tools):
    case = Projection((131072, 1, 1, 1), (True, True, True, False), (0, 1, 3), (0, 3, 2))
    before = case.source()
    report, documents, _ = check_chain([before, lowered(before)], BOUNDS, tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"], report
    assert component_report(documents[1])[0]["reduction_size"] == 1


@pytest.mark.parametrize("operand", ["left", "right"])
def test_source_maps_bind_the_actual_contraction_and_scale_grouping(tmp_path, lean_tools, operand):
    first, second = changed_source_pair(operand)
    sources = [first.source(), second.source()]
    targets = [lowered(source) for source in sources]
    for source_index, source in enumerate(sources):
        for target_index, target in enumerate(targets):
            work = tmp_path / f"{source_index}-{target_index}"
            work.mkdir()
            report, _, _ = check_chain([source, target], BOUNDS, work, lean_tools)
            assert report["conditional_initialized_policy_refinement"] is (source_index == target_index)
