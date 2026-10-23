"""Native admission retains exactness gates without any profitability policy."""

import json

import pytest
import yaml
from mlir_fixtures import forward, generic

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.core_aten_admission import prepare


def software(tmp_path, excluded=()):
    spec = tmp_path / "software.yaml"
    spec.write_text(
        yaml.safe_dump(dict(numerical_semantics=dict(native_unit_contract=dict(excluded_constructions=list(excluded)))))
    )
    return spec

def source(size):
    typ = f"tensor<{size}xf32>"
    return forward(f"%x:{typ}", typ, generic([("%x", typ)], typ, "%v=arith.negf %a : f32", name="%v"), ret="%v")

def candidate(module, family="elementwise_map", numerical="F2"):
    ordinal = next(i for i, op in enumerate(module.walk()) if op.name == "linalg.generic")
    return dict(
        numerical_family=numerical, sites=[dict(status="admitted", family=family, source_operation_ordinal=ordinal)]
    )


def selected(module, *args):
    return dict(candidate(module, *args), exact_dtype_sites=1, selection=dict(identities={}))


def eager_bits(monkeypatch):
    # These sources negate floating data, which the IR floating contract sends
    # to the declared quantized model; that obligation has its own tests
    # (test_native_precision_contract). Here only construction policy and ABI
    # boundaries are exercised.
    from merlin.targetgen import native_precision_contract

    monkeypatch.setattr(native_precision_contract, "requires_quantized_model", lambda module: False)


def test_policy_uses_only_bundled_logical_boundary_shapes(tmp_path, monkeypatch):
    eager_bits(monkeypatch)
    module = source(2)
    (tmp_path / "model.mlir").write_text(str(module))
    scalar = dict(dtype="complex<f32>", shape=[])
    report = dict(
        cases=[
            dict(status="bundled", input_abi=[scalar], output_abi=[scalar]),
            dict(status="not_in_shard", input_abi=[dict(dtype="f32", shape=[99])]),
        ]
    )
    plan = prepare(tmp_path, report, lambda *a, **k: selected(module), software_spec=software(tmp_path), facts={})
    assert plan["native_admission"]["status"] == "device"
    assert report["cases"][0]["numeric_admission"] == plan
    assert "numeric_admission" not in report["cases"][1]

def movement(reverse=False):
    module = source(7)
    from xdsl.rewriter import Rewriter

    negation = next(op for op in module.walk() if op.name == "arith.negf")
    Rewriter.replace_op(negation, [], [negation.operands[0]])
    if reverse:
        from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr
        from xdsl.ir.affine import AffineDimExpr, AffineMap

        op = next(op for op in module.walk() if op.name == "linalg.generic")
        op.properties["indexing_maps"] = ArrayAttr(
            [AffineMapAttr(AffineMap(1, 0, (-AffineDimExpr(0) + 6,))), AffineMapAttr(AffineMap.identity(1))]
        )
    return module

@pytest.mark.parametrize("admitted", [False, True])
def test_policy_evidence_is_attributed_only_to_the_bundled_case(tmp_path, admitted):
    module = source(7)
    (tmp_path / "model.mlir").write_text(str(module))
    report = dict(cases=[dict(status="bundled"), dict(status="not_in_shard", case_id="unselected")])
    inactive = report["cases"][1]

    def implementation(directory, report, **kwargs):
        for row in report["cases"]:
            row["numeric_admission"] = dict(candidate(module), exact_dtype_sites=1)
        return dict(candidate(module), exact_dtype_sites=int(admitted), selection=dict(identities={}))

    prepare(tmp_path, report, implementation, software_spec=software(tmp_path), facts={})
    assert inactive is report["cases"][1]
    assert inactive == dict(status="not_in_shard", case_id="unselected")
    assert "native_admission" in report["cases"][0]["numeric_admission"]
    json.dumps(report)

def test_descriptor_fold_retains_host_while_real_scalar_tensor_work_qualifies(tmp_path):
    module = movement()
    (tmp_path / "model.mlir").write_text(str(module))
    capture = tmp_path / "capture"
    capture.mkdir()
    descriptor = dict(shape=[7], stride=[1], storage_offset=0)
    case = dict(
        overload="aten.sym_numel.default",
        arguments=dict(args=dict(kind="tuple", items=[dict(kind="tensor", **descriptor)]), kwargs={}),
    )
    proof = dict(
        authority="static_capture_input_descriptors",
        inputs=[descriptor],
        queries=[dict(operator=case["overload"], value=7)],
    )
    (capture / "capture.json").write_text(json.dumps(dict(capture_meta=dict(static_metadata_specialization=proof))))
    report = dict(cases=[dict(status="bundled", case=case, capture_directory=str(capture))])
    implementation = lambda *a, **k: selected(module, "movement", "F1")  # noqa: E731
    result = prepare(tmp_path, report, implementation, software_spec=software(tmp_path), facts={})
    assert result["refusal_code"] == "constant_metadata_fold_no_tensor_data"
    assert result["native_admission"]["descriptor_folds"] == [proof]

@pytest.mark.parametrize("missing", ["excluded_constructions"])
def test_missing_construction_policy_refuses_before_any_preparation(tmp_path, missing):
    spec = tmp_path / "software.yaml"
    spec.write_text(yaml.safe_dump(dict(numerical_semantics=dict(native_unit_contract={}))))
    with pytest.raises(ValueError, match=missing):
        prepare(tmp_path, {}, lambda *a, **k: pytest.fail("must refuse before preparing"), software_spec=spec, facts={})

@pytest.mark.parametrize("stage", ["prepared", "provider_prelude"])
def test_generated_constructions_are_screened_even_if_the_prelude_is_later_removed(tmp_path, stage, monkeypatch):
    eager_bits(monkeypatch)
    original = source(7)
    (tmp_path / "model.mlir").write_text(str(original))
    forbidden = parse_mlir_text("""module {
      func.func @bad(%a: i1,%b: i1) -> tensor<i1> {
        %e=tensor.empty() : tensor<i1>
        %r=linalg.generic {indexing_maps=[affine_map<()->()>],iterator_types=[]} outs(%e : tensor<i1>) {
        ^bb0(%unused: i1):
          %v=arith.ori %a,%b : i1
          linalg.yield %v : i1
        } -> tensor<i1>
        func.return %r : tensor<i1>
      }}""")

    def implementation(directory, report, *, construction_observer, **kwargs):
        if stage == "provider_prelude":
            construction_observer(forbidden, stage)
        else:
            (directory / "model.mlir").write_text(str(forbidden))
        return dict(candidate(original), selection=dict(identities={}), exact_dtype_sites=1)

    report = dict(cases=[dict(status="bundled")])
    plan = prepare(tmp_path, report, implementation, software_spec=software(tmp_path, ["multi_op_identity"]), facts={})
    assert plan["refusal_code"] == "non_native_unit_construction"
    assert any(e["stage"] == stage for e in plan["native_admission"]["native_unit_policy"]["evidence"])
    assert (tmp_path / "model.mlir").read_text() == str(original)


@pytest.mark.parametrize("alias_required", [False, True])
def test_native_selector_proof_also_requires_the_whole_call_alias_contract(tmp_path, alias_required):
    original = movement(reverse=True)
    original_text = str(original)
    (tmp_path / "model.mlir").write_text(original_text)
    (tmp_path / "inputs.npz").write_bytes(b"original storage")
    active = dict(status="bundled", case=dict(output_input_aliases=[[0, 0]] if alias_required else []))
    inactive = dict(status="not_in_shard", case=dict(output_input_aliases=[[0, 0]]))
    report = dict(cases=[active, inactive])

    def implementation(directory, report, **kwargs):
        (directory / "inputs.npz").write_bytes(b"materialized candidate")
        return selected(original, "movement", "F1")

    result = prepare(tmp_path, report, implementation, software_spec=software(tmp_path))
    assert report["cases"][0] is active and report["cases"][1] is inactive
    assert result["native_admission"]["status"] == ("host" if alias_required else "device")
    if alias_required:
        assert result["refusal_code"] == "native_output_alias_contract_unproved"
        assert result["exact_dtype_sites"] == 0
        assert (tmp_path / "inputs.npz").read_bytes() == b"original storage"
        assert (tmp_path / "model.mlir").read_text() == original_text
    else:
        assert (tmp_path / "inputs.npz").read_bytes() == b"materialized candidate"


def test_a_declined_candidate_cannot_leak_its_rewrites_into_the_next(tmp_path, monkeypatch):
    from xdsl.dialects.builtin import UnitAttr

    import merlin.targetgen.core_aten_components as components
    import merlin.targetgen.core_aten_exact_dtype as exact
    import merlin.targetgen.core_aten_spec_int8 as spec
    import merlin.targetgen.native_byte_activation as byte
    import merlin.targetgen.native_quantized_activation as quantized
    import merlin.targetgen.native_wide_activation as wide
    from merlin.targetgen.core_aten_numeric import _prepare_spec_int8

    (tmp_path / "model.mlir").write_text(str(source(2)))
    monkeypatch.setattr(spec, "select_spec", lambda *a: dict(spec={}))
    monkeypatch.setattr(exact, "prepare_bitwise", lambda *a: None)
    seen = []

    def declines_after_rewriting(directory, report, selection, provider, package, module, **kw):
        seen.append(str(module))
        for op in module.walk():
            op.attributes["test.declined_rewrite"] = UnitAttr()

    for owner, name in [(byte, "prepare"), (wide, "prepare"), (components, "prepare_integer_components")]:
        monkeypatch.setattr(owner, name, declines_after_rewriting)
    accepted = dict(status="admitted")
    monkeypatch.setattr(quantized, "prepare", lambda *a: seen.append(str(a[-1])) or accepted)
    kwargs = dict(target="t", facts={}, software_spec=None, host_capabilities=None)
    assert _prepare_spec_int8(tmp_path, dict(cases=[]), **kwargs) is accepted
    assert len(seen) == 4 and len(set(seen)) == 1 and "test.declined_rewrite" not in seen[0]
