"""Declared compositions use source semantics and retain a usable unfused route."""

import json

import numpy as np
import pytest
from fake_quant_layer import mean_module, residual_module
from test_output_fusion import _MATMUL, _path
from test_output_fusion import _module as _legacy_module

from merlin.common.mlir_query import parse
from merlin.compile import fusion_workflow as F
from merlin.compile.composition_fusion import CompositionPlan, composition_routes
from merlin.targetgen.readout_facet import ReadoutFacet
from merlin.transforms.certificates import ir_digest
from merlin.verify.lean_contract import document_digest

SCHEME = {"readout_strategy": "f32_product_rne", "operand_sum_rounding": "single_rounding"}
FACTS = {
    "dataflows": ["WS", "OS"],
    "mesh_signed_bits": 20,
    "digit_bits": 8,
    "max_extended_k": 128,
    "accumulator_bits": 32,
}


def _module(**options):
    # xDSL's named mixed-width matmul parser builds an invalid scalar region;
    # use the explicit signed widening form so discovery verifies real typed IR.
    root = """%mm=linalg.generic {indexing_maps=[affine_map<(m,n,k)->(m,k)>,
      affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],
      iterator_types=["parallel","parallel","reduction"]}
      ins(%x,%w:tensor<4x8xi8>,tensor<8x8xi8>) outs(%init:tensor<4x8xi32>) {
      ^bb0(%a:i8,%b:i8,%c:i32):
        %aa=arith.extsi %a:i8 to i32
        %bb=arith.extsi %b:i8 to i32
        %p=arith.muli %aa,%bb:i32
        %v=arith.addi %p,%c:i32
        linalg.yield %v:i32
      } -> tensor<4x8xi32>"""
    return _legacy_module(**options).replace(_MATMUL, root)


def declaration(route, identity="arbitrary id"):
    row = {"id": identity, "producer": "contraction", "constraints": []}
    if route == "shared_input":
        row.update(
            multiplicity={"min": 2},
            constraints=["per_output_readout_independent", {"shared_input_resident": "from_facts"}],
        )
    elif route == "k_concat":
        row.update(
            extension="constant_integer_operand",
            constraints=["coefficient_integer_exact", {"k_extension_within": "from_facts"}],
        )
    elif route == "output_stationary":
        row.update(
            dataflows="from_facts",
            constraints=[{"os_dataflow_requires": "mesh_partial_sum_bound_from_constant_operands"}],
        )
    elif route == "residual_on_load":
        row.update(
            producer="operand_sum", constraints=["load_minimum_not_negated", {"operand_multiplier_in": "(0, 1]"}]
        )
    elif route == "avgpool":
        row.update(reduction_kinds=["sum"], constraints=["activation_before_scale_requires_nonnegative_multiplier"])
    elif route == "conv_pool":
        row.update(
            reduction_kinds=["max"],
            constraints=[
                {"max_requires_store_path_pooling": "from_facts"},
                "scratchpad_source_pooling_fenced",
                {"padded_max_requires": "producer_lower_bound_or_derived_pad_value"},
            ],
        )
    else:
        row.update(
            exits=["accumulator" if route == "int32_readout" else "requant"],
            constraints=["activation_before_scale_requires_nonnegative_multiplier"],
        )
    return row


def block(route, identity="arbitrary id"):
    return {
        "schema": "merlin.software_spec.compositions.v1",
        "status": "reviewed",
        "grading": "placement_and_instruction_audit",
        "chains": [declaration(route, identity)],
    }


def options(route):
    return dict(
        capability_contract={},
        numerical_policy_sha256=document_digest(SCHEME),
        compositions=block(route),
        max_region_operations=16,
    )


def realize(module, route, *, path=None, scheme=None, facts=None, **selection):
    discovery = options(route)
    scheme = SCHEME if scheme is None else scheme
    discovery["numerical_policy_sha256"] = document_digest(scheme)
    rows = F.discover(module, **discovery)["candidates"]
    row = next(row for row in rows if row["detail"].get("composition") and row["route"] == route)
    return F.realize_candidate(
        module,
        json.loads(json.dumps(row)),
        **discovery,
        composition_options=dict(
            path=_path() if path is None else path, scheme=scheme, facts=FACTS if facts is None else facts, **selection
        ),
    )


@pytest.mark.parametrize(
    "route",
    [
        "epilogue",
        "int32_readout",
        "residual_on_load",
        "conv_pool",
        "avgpool",
        "output_stationary",
        "k_concat",
        "shared_input",
    ],
)
def test_routes_register_by_semantics_not_declaration_id(route):
    assert route in F.ROUTES
    assert composition_routes(declaration(route, "first")) == (route,)
    assert composition_routes(declaration(route, "unrelated name")) == (route,)


def test_f32_epilogue_plan_and_failed_premise_emit_full_unfused_source():
    module = parse(_module(out_scale=7.0))
    before = ir_digest(module)
    product = realize(module, "epilogue")
    assert product.receipt["status"] == "applied", product.receipt
    assert isinstance(product.artifact, CompositionPlan)
    proof = product.artifact.decision["proof"]
    assert proof["definitional"] == "f32_product_rne"
    np.testing.assert_array_equal(proof["multiplier_bits"], np.full(8, np.float32(1 / 56)).view(np.uint32))
    assert proof["offsets"] == [0, 64, -64, 128, 8, -8, 0, 256]
    path = _path()
    path["exits"]["requant"] = False
    emitted = []

    def emit(result, session):
        emitted.append(result)
        return result.artifact

    results = list(
        F.evaluate_fusions(
            module,
            (
                F.FusionAlternative("declined", lambda source: realize(source, "epilogue", path=path)),
                F.FusionAlternative("fused", lambda source: realize(source, "epilogue")),
            ),
            emit,
        )
    )
    assert [r.status for r in results] == ["compiled", "declined", "compiled"]
    assert all(r.evaluation.status == "compiled" for r in results)
    assert emitted[1].receipt["status"] == "unfused"
    assert emitted[1].artifact is results[1].realization.unfused
    assert "fallback_from" in emitted[1].receipt
    assert ir_digest(module) == before
    json.dumps(product.receipt)


def test_pagination_binds_declarations_and_keeps_all_composition_rows():
    module = parse(_module())
    opts = options("epilogue")
    full = F.discover(module, **opts)["candidates"]
    page = F.discover(module, max_candidates=1, **opts)
    rows = list(page["candidates"])
    first = page["next_cursor"]
    while page["next_cursor"]:
        page = F.discover(module, max_candidates=1, cursor=page["next_cursor"], **opts)
        rows.extend(page["candidates"])
    assert rows == full
    with pytest.raises(ValueError, match="cursor differs"):
        F.discover(module, cursor=first, **dict(opts, compositions=block("epilogue", "changed")))


def test_os_runtime_weight_twin_falls_back_to_ws():
    module = parse(_module())
    product = realize(module, "output_stationary")
    assert product.receipt["status"] == "applied", product.receipt
    assert product.artifact.decision["dataflow"] == "OS"
    root = next(op for op in module.walk() if op.name == "linalg.generic")
    function = root.parent_op()
    weight = root.operands[1]
    argument = function.body.block.insert_arg(weight.type, len(function.body.block.args))
    root.operands = [root.operands[0], argument, root.operands[2]]
    function.update_function_type()
    twin = realize(module, "output_stationary")
    assert twin.receipt["status"] == "declined"
    assert "constant weights" in twin.receipt["reason"]
    assert twin.artifact is module
    outside = realize(parse(_module()), "output_stationary", facts=dict(FACTS, mesh_signed_bits=8))
    assert outside.receipt["status"] == "declined"


def test_k_concat_integer_and_fractional_source_twins():
    text = _module(residual=True)
    # residual/accumulator unit = .125/.0625 = 2, derived from scales in IR.
    integer = parse(text.replace("5.000000e-01", "2.500000e-01"))
    product = realize(integer, "k_concat")
    assert product.receipt["status"] == "applied", product.receipt
    assert product.artifact.decision["coefficient"] == 2
    assert product.artifact.decision["tile_digits"] == [2]
    fraction = parse(text.replace("5.000000e-01", "7.500000e-01"))
    twin = realize(fraction, "k_concat")
    assert twin.receipt["status"] == "declined", twin.receipt
    assert "exact integer" in twin.receipt["reason"]


def test_average_uses_source_window_and_formed_multiplier():
    module = parse(mean_module(in_scale=0.5, out_scale=0.25, count=12.0))
    product = realize(module, "avgpool")
    assert product.receipt["status"] == "applied", product.receipt
    decision = product.artifact.decision
    assert decision["window"] == 12 and decision["constant_weight"] == 1
    expected = np.asarray(np.float32(1 / 6)).view(np.uint32).item()
    assert decision["multiplier_bits"] == [expected]
    twin = realize(parse(mean_module(in_scale=0.5, out_scale=0.25, count=11.0)), "avgpool")
    assert twin.receipt["status"] == "declined"


def test_residual_identity_loads_use_existing_facet_and_scaled_twin_refuses():
    facet = ReadoutFacet(
        target="synthetic",
        unit="array",
        accumulator_kind="addressable",
        operand_sum={"operands": 2, "operand_dtype": "i8", "operand_rounding": "half_even", "operand_saturates": True},
    )
    module = parse(residual_module())
    # Select scales from the source rather than a production model identifier.
    from merlin.xdsl_dialects.lowering import compute_groups as CG

    for op in module.walk():
        stage = CG.classify(op)
        if stage and stage.kind in (CG.DEQUANTIZE, CG.QUANTIZE):
            owner = op.operands[1].owner
            from xdsl.dialects.builtin import DenseIntOrFPElementsAttr

            owner.properties["value"] = DenseIntOrFPElementsAttr.from_list(owner.result.type, [0.5])
    product = realize(module, "residual_on_load", readout_facet=facet)
    assert product.receipt["status"] == "applied", product.receipt
    twin = realize(parse(residual_module()), "residual_on_load", readout_facet=facet)
    assert twin.receipt["status"] == "declined"


def test_external_observer_preserves_full_source():
    from xdsl.dialects import func

    module = parse(_module())
    root = next(op for op in module.walk() if op.name == "linalg.generic")
    ret = next(op for op in module.walk() if isinstance(op, func.ReturnOp))
    ret.operands = [*ret.operands, root.results[0]]
    ret.parent_op().update_function_type()
    product = realize(module, "epilogue")
    assert product.receipt["status"] == "declined"
    assert product.artifact is module


def test_int32_exit_applies_no_epilogue_and_missing_exit_refuses():
    from xdsl.dialects import func

    module = parse(_module())
    root = next(op for op in module.walk() if op.name == "linalg.generic")
    ret = next(op for op in module.walk() if isinstance(op, func.ReturnOp))
    ret.operands = [root.results[0]]
    parent = root.parent
    tail = list(parent.ops)[list(parent.ops).index(root) + 1 :]
    for op in reversed(tail):
        if op is not ret:
            parent.erase_op(op)
    ret.parent_op().update_function_type()
    module.verify()
    result = realize(module, "int32_readout")
    assert result.receipt["status"] == "applied", result.receipt
    assert result.artifact.decision["proof"]["readout_stages"] == []
    path = _path()
    path["exits"]["accumulator"] = False
    assert realize(module, "int32_readout", path=path).receipt["status"] == "declined"


def test_pool_window_fields_and_padding_have_refusal_twins():
    module = parse(_module(pool=True))
    facts = dict(FACTS, pool_window_max=[2], pool_stride_max=[2])
    product = realize(module, "conv_pool", facts=facts)
    assert product.receipt["status"] == "applied", product.receipt
    assert product.artifact.decision["proof"]["pool_geometry"] == {"window": [2], "stride": [2], "padded": False}
    assert realize(module, "conv_pool").receipt["status"] == "declined"
    twin = realize(module, "conv_pool", facts=dict(facts, pool_window_max=[1]))
    assert twin.receipt["status"] == "declined"


def test_rejected_bias_and_unknown_readout_rounding_keep_source():
    # Arbitrary binary32 weight scales need not turn a floating bias into an
    # exactly representable accumulator seed.
    module = parse(_module(weight_scale=(0.3,), bias=(1, 3, 5, 7, 9, 11, 13, 15)))
    assert realize(module, "epilogue").receipt["status"] == "declined"
    path = _path()
    next(s for s in path["readout"] if s["kind"] == "requant")["rounding"] = "unknown"
    assert realize(parse(_module()), "epilogue", path=path).receipt["status"] == "declined"


def test_resumable_example_emits_provider_plan_json(tmp_path):
    import importlib.util

    from merlin.common.paths import repo_root

    spec = importlib.util.spec_from_file_location(
        "composition_example", repo_root() / "examples/compiler/fusion_exploration.py"
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.patterns = lambda: [("synthetic_chain", _module(out_scale=7.0))]

    def select(module):
        return {"compositions": block("epilogue"), "options": {"path": _path(), "scheme": SCHEME, "facts": FACTS}}

    summary = example.run(tmp_path, composition_selector=select)
    plans = [
        row
        for row in summary["patterns"][0]["artifacts"]
        if row["kind"] == "semantic_composition_plan_pending_target_emission"
    ]
    assert plans
    from pathlib import Path

    saved = json.loads(Path(plans[0]["path"]).read_text())
    assert saved["decision"]["proof"]["definitional"] == "f32_product_rne"
    assert saved["emission_obligation"]


def test_definitional_boundary_does_not_ignore_source_clamp():
    module = parse(_module(quant=(-127, 127), relu=False))
    result = realize(module, "epilogue")
    assert result.receipt["status"] == "declined"
    assert "clamp" in result.receipt["reason"]


def test_absent_composition_selection_returns_unfused_with_reason():
    module = parse(_module())
    opts = options("epilogue")
    row = next(row for row in F.discover(module, **opts)["candidates"] if row["detail"].get("composition"))
    for selection in (None, {}):
        result = F.realize_candidate(module, row, **opts, composition_options=selection)
        assert result.artifact is module and result.receipt["status"] == "declined"
        assert result.receipt["reason"]


def test_failed_unfused_emission_is_not_reported_as_a_successful_fallback():
    module = parse(_module())
    path = _path()
    path["exits"]["requant"] = False
    emitted = []

    def emit(product, session):
        emitted.append(product)
        if len(emitted) > 1:
            raise ValueError("original route emission failed")
        return product.artifact

    results = list(
        F.evaluate_fusions(
            module, (F.FusionAlternative("declined", lambda source: realize(source, "epilogue", path=path)),), emit
        )
    )
    assert [r.status for r in results] == ["compiled", "failed"]
    assert "original route emission failed" in results[-1].reason
