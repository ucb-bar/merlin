"""Ordinary source preparation, bounded route selection and immutable proof custody."""

import json
from dataclasses import replace
from math import prod
from pathlib import Path

import pytest
from test_convolution_route_model import CASES, model_source
from test_source_dense_lowering import run, selection
from xdsl.dialects import func

from merlin.common.digest import sha256_file
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import im2col_pack
from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects
from merlin.llvmlower.convolution_routes import normalize_convolution_source
from merlin.llvmlower.source_model import source_provider_builder
from merlin.transforms.certificates import ir_text


def configured(tmp_path, geometry=None, **controls):
    g = CASES[0] if geometry is None else geometry
    n, channels, _, _ = g["shape"]
    k, m = channels * prod(g["kernel"]), n * prod(g["output"])
    config, _, seen = selection(
        tmp_path,
        hardware=dict(
            tile=g["tile"], widths=(g["width"], g["width"], 4), shared=g["shared"], shape=(g["outputs"], k, m)
        ),
        max_convolution_packing_bytes=g["budget"],
        borrowed_effects=BorrowedBufferEffects("neutral-reviewed-panel-reuse-v1", True, True, True),
        **controls,
    )
    return config, seen, parse_mlir_text(model_source(g))


@pytest.mark.parametrize("geometry", CASES)
def test_ordinary_preparation_checks_actual_panel_sum_complete_source_and_padding(tmp_path, geometry):
    config, seen, module = configured(tmp_path, geometry)
    source = ir_text(module)
    result, builder = run(tmp_path, config, module)
    assert source_provider_builder(builder) is builder and len(seen) == 2
    record = json.loads((result.parent / "source_dense.json").read_text())
    (normalization,) = record["source_normalization"]
    assert normalization["rewritten"] == 1
    checkpoint = normalization["checkpoint"]
    saved = Path(checkpoint["path"])
    assert saved.is_file() and sha256_file(saved) == checkpoint["sha256"]
    selected = parse_mlir_text(saved.read_text())
    selected.verify()
    assert len([op for op in selected.walk() if type(op) is func.CallOp]) == 0
    (route,) = normalization["routes"]
    proof = route["conformance"]
    payload = sum(prod(arm["packing_shape"]) * geometry["width"] for arm in proof["arms"])
    assert payload == proof["resident_packing_bytes"] == route["realized_distinct_panel_payload_bytes"]
    assert payload <= geometry["budget"] < proof["original_packing_bytes"]
    assert payload > max(arm["packing_bytes"] for arm in proof["arms"]), "tail backing must also be budgeted"
    n, channels, height, width = geometry["shape"]
    top, left, bottom, right = geometry["padding"]
    padding_bytes = n * channels * (height + top + bottom) * (width + left + right) * geometry["width"]
    assert route["padding"]["transformation"] == "unchanged"
    assert route["padding"]["zero_fill_write_bytes"] == padding_bytes
    assert route["introduced_zero_fill_payload_bytes"] == geometry["outputs"] * n * prod(geometry["output"]) * 4
    pad_shape = (n, channels, height + top + bottom, width + left + right)
    retained_padding = [
        op
        for op in selected.walk()
        if op.name == "tensor.insert_slice"
        and tuple(op.results[0].type.get_shape()) == pad_shape
        and op.dest.owner.name == "linalg.fill"
    ]
    assert len(retained_padding) == 1
    zero = retained_padding[0].dest.owner.inputs[0].owner
    assert zero.name == "arith.constant" and zero.value.value.data == 0
    stages = normalization["pipeline"]
    assert [row["name"] for row in stages] == [
        "convolution-source",
        "convolution-routes",
        "convolution-panel-packing",
        "convolution-conformance",
    ]
    assert [row["produced"] for row in stages] == [["windows"], ["routes"], ["realized"], ["checked"]]
    assert set(stages[2]["invalidated"]) == {"windows", "routes"}
    assert stages[0]["before"]["source_sha256"] == stages[1]["after"]["source_sha256"]
    assert stages[2]["after"]["source_sha256"] == stages[3]["before"]["source_sha256"]
    for binding in seen:
        binding.compilation.revalidate()
    assert (tmp_path / "model.mlir").read_text() == source


@pytest.mark.parametrize(
    "control,value,reason",
    [
        ("max_convolution_packing_bytes", None, None),
        ("convolution_route", "full", None),
        ("max_convolution_packing_bytes", 4096, None),
        ("max_convolution_packing_bytes", 1, "no_legal_panel_within_packing_budget"),
        ("max_convolution_candidates", 1, "candidate_budget"),
        ("max_convolution_added_operations", 1, "source operation budget"),
    ],
)
def test_legal_full_materialization_fallback_retains_exact_original_source(tmp_path, control, value, reason):
    config, _, module = configured(tmp_path)
    config = replace(config, **{control: value})
    before = ir_text(module)
    selected, report = normalize_convolution_source(module, config, config.selected_contracts.revalidate)
    assert selected is module and ir_text(selected) == before
    assert report["rewritten"] == 0 and report["selected_sha256"] == report["source_sha256"]
    if reason is not None:
        assert any(reason in item for item in report["declined"])


def test_all_dense_bindings_declined_discards_panel_candidate_and_preserves_no_provider_custody(tmp_path):
    config, seen, module = configured(tmp_path, static_budget=0)
    before = ir_text(module)
    result, builder = run(tmp_path, config, module)
    assert not seen and source_provider_builder(builder) is None
    assert result.read_text() == before == (tmp_path / "model.mlir").read_text()
    (normalization,) = json.loads((result.parent / "source_dense.json").read_text())["source_normalization"]
    assert normalization["rewritten"] == 0 and normalization["pipeline"] == []
    assert normalization["selected_sha256"] == normalization["source_sha256"]
    candidate = normalization["discarded_candidate"]
    assert candidate["rewritten"] == 1 and len(candidate["pipeline"]) == 4
    assert candidate["routes"][0]["conformance"]["resident_packing_bytes"] <= config.max_convolution_packing_bytes
    assert any("no completed dense binding" in reason for reason in normalization["declined"])


@pytest.mark.parametrize(
    "controls",
    [
        dict(convolution_route="panel", max_convolution_packing_bytes=None),
        dict(convolution_route="unknown"),
        dict(max_convolution_packing_bytes=True),
        dict(max_convolution_packing_bytes=0),
        dict(max_convolution_packing_bytes=1 << 63),
        dict(max_convolution_candidates=0),
        dict(max_convolution_candidates=True),
        dict(max_convolution_added_operations=-1),
        dict(max_convolution_added_operations=1 << 21),
    ],
)
def test_route_controls_are_typed_bounded_and_do_not_grant_missing_storage(tmp_path, controls):
    config, _, _ = configured(tmp_path)
    with pytest.raises(ValueError, match="convolution routes"):
        replace(config, **controls)


@pytest.mark.parametrize("mutation", ["resource", "budget", "qualification", "facts"])
def test_normalization_rejects_changed_original_context_at_pass_boundary(tmp_path, mutation):
    config, _, module = configured(tmp_path)
    calls = 0

    def validate():
        nonlocal calls
        calls += 1
        if calls == 2:
            if mutation == "resource":
                object.__setattr__(config.storage.lhs, "capacity", config.storage.lhs.capacity + 1)
            elif mutation == "budget":
                object.__setattr__(config, "max_convolution_packing_bytes", config.max_convolution_packing_bytes + 1)
            elif mutation == "qualification":
                object.__setattr__(config.adapter, "qualification_identity", "changed-original-qualification")
            else:
                path = config.selected_contracts.facts_path
                path.write_bytes(path.read_bytes() + b"\n")
        config.selected_contracts.revalidate()

    with pytest.raises(ValueError, match="changed|replan"):
        normalize_convolution_source(module, config, validate)


@pytest.mark.parametrize("mutation", ["new_call", "external_order"])
def test_constructor_cannot_hide_unknown_effects_or_reorder_unchanged_operations(tmp_path, monkeypatch, mutation):
    config, _, module = configured(tmp_path)
    function = next(op for op in module.body.block.ops if type(op) is func.FuncOp)
    for name in ("opaque_first", "opaque_second"):
        module.body.block.add_op(func.FuncOp.external(name, [], []))
    first, second = (func.CallOp(name, [], []) for name in ("opaque_first", "opaque_second"))
    function.body.block.insert_op_before(first, function.body.block.first_op)
    function.body.block.insert_op_after(second, first)
    module.verify()
    original = im2col_pack.rewrite_bounded_match

    def faulty(match, width):
        result = original(match, width)
        if mutation == "new_call":
            loop = result.arms[0].loop
            loop.parent_block().insert_op_after(func.CallOp("opaque_first", [], []), loop)
        else:
            body = next(
                op
                for op in result.arms[0].loop.get_toplevel_object().walk()
                if type(op) is func.FuncOp and op.sym_name.data == "forward"
            ).body.block
            calls = [op for op in body.ops if type(op) is func.CallOp]
            body.detach_op(calls[0])
            body.insert_op_after(calls[0], calls[1])
        return result

    monkeypatch.setattr(im2col_pack, "rewrite_bounded_match", faulty)
    with pytest.raises(ValueError, match="membership" if mutation == "new_call" else "block order"):
        normalize_convolution_source(module, config, config.selected_contracts.revalidate)
