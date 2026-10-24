"""Native exact-family admission requires source closure and independent bits."""

import hashlib

import pytest
from targetgen_fixtures import selector_catalog

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.core_aten_exact_dtype import bitwise_catalog_coverage
from merlin.targetgen.core_aten_host_floor import grade_native_bits
from merlin.targetgen.software_spec import admit_operation


def module_and_manifest(*, arithmetic=False):
    body = """%zero = arith.constant 0 : i64
      %input = tensor.splat %zero : tensor<2xi64>
      %e = tensor.empty() : tensor<2xi64>
      %copy = linalg.copy ins(%input : tensor<2xi64>) outs(%e : tensor<2xi64>) -> tensor<2xi64>
      func.return %copy : tensor<2xi64>"""
    module = parse_mlir_text("builtin.module { func.func @forward() -> tensor<2xi64> {" + body + "} }")
    operations = list(module.walk())
    root = next(op for op in operations if op.name == "linalg.copy")
    splat = next(op for op in operations if op.name == "tensor.splat")
    constant = next(op for op in operations if op.name == "arith.constant")
    bindings = [(operations.index(root), {"x": dict(op_ordinal=operations.index(splat), result=0)})]
    if not arithmetic:
        bindings.append((operations.index(splat), {"literal": dict(op_ordinal=operations.index(constant), result=0)}))
    return module, selector_catalog(module, "copy_bytes", *bindings)


def test_closed_bitwise_lineage_and_source_identity():
    module, manifest = module_and_manifest()
    assert len(bitwise_catalog_coverage(module, manifest)) == 2
    manifest["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different source"):
        bitwise_catalog_coverage(module, manifest)


def test_a_device_copy_does_not_admit_unbound_producer_work():
    module, manifest = module_and_manifest(arithmetic=True)
    with pytest.raises(ValueError, match="unbound tensor.splat"):
        bitwise_catalog_coverage(module, manifest)


@pytest.mark.parametrize("key,value", [("proof", "unchecked"), ("readout", "float_scale")])
def test_incomplete_kernel_proof_refuses(key, value):
    module, manifest = module_and_manifest()
    manifest["kernels"][0]["meta"][key] = value
    with pytest.raises(ValueError, match="kernel proof"):
        bitwise_catalog_coverage(module, manifest)


def test_family_constraint_is_executable_and_cannot_admit_ordinary_sites():
    declaration = dict(
        schema="merlin.software_spec.v1",
        target="test",
        status="reviewed",
        operations=[
            dict(id="bytes", families=["movement"], placement="accelerator", signature=dict(numerical_family=["F1"]))
        ],
    )
    assert admit_operation(declaration, "movement", dict(family="movement"), "accelerator")["status"] != "admitted"
    assert (
        admit_operation(declaration, "movement", dict(family="movement", numerical_family="F1"), "accelerator")[
            "status"
        ]
        == "admitted"
    )
    assert (
        admit_operation(declaration, "movement", dict(family="movement", numerical_family="F2"), "accelerator")[
            "status"
        ]
        == "unsupported"
    )


@pytest.mark.parametrize(
    "reference,changed",
    [
        ("00000000", "00000080"),  # signed zero
        ("0100c07f", "0200c07f"),  # different quiet NaN payloads
        ("ffffffffffffff7f", "0000000000000080"),  # int64 limits
        ("0100803f00000080", "0100803f00000000"),  # complex imaginary zero
    ],
)
def test_host_floor_compares_storage_bits(reference, changed):
    record = dict(
        output_indices=[0],
        native_eager_indices=dict(outputs=[0], posts=[1]),
        native_eager_boundary=dict(
            authority="untransformed_native_cpu_eager", output_bytes=[reference], post_input_bytes=["a5"]
        ),
    )
    assert grade_native_bits(record, [bytes.fromhex(reference), b"\xa5"])["status"] == "pass"
    assert grade_native_bits(record, [bytes.fromhex(changed), b"\xa5"])["status"] == "mismatch"
    assert grade_native_bits(record, [bytes.fromhex(reference), b"\xa4"])["status"] == "mismatch"
    assert grade_native_bits(record, [bytes.fromhex(reference)])["status"] == "mismatch"


def test_dense_literal_selector_keeps_nan_payload_and_signed_zero_bytes():
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr

    from merlin.targetgen.core_aten_exact_dtype import _expose_literal_movement

    source = parse_mlir_text("""builtin.module {func.func @forward() -> tensor<2xf32> {
      %literal = arith.constant dense<[0x7fc00001,0x80000000]> : tensor<2xf32>
      func.return %literal : tensor<2xf32>
    }}""")
    original = next(op for op in source.walk() if op.name == "arith.constant").value.data.data
    _expose_literal_movement(source)
    source.verify()
    literal = next(op for op in source.walk() if op.name == "arith.constant")
    assert isinstance(literal.value, DenseIntOrFPElementsAttr)
    assert literal.value.data.data == original
    assert str(literal.result.type) == "tensor<2xi32>"
    assert len([op for op in source.walk() if op.name == "linalg.generic"]) == 2
    assert len([op for op in source.walk() if op.name == "arith.bitcast"]) == 1


def test_empty_boundary_decision_precedes_provider_and_quantized_capture(tmp_path, monkeypatch):
    import merlin.targetgen.core_aten_spec_int8 as spec
    from merlin.targetgen.core_aten_numeric import prepare_spec_int8

    monkeypatch.setattr(spec, "select_spec", lambda *args: dict(identities={}))

    class Provider:
        def prepare_bundle(self, *args, **kwargs):
            raise AssertionError("empty boundary must not ask the provider for work")

    specification = tmp_path / "software.yaml"
    specification.write_text(
        "numerical_semantics: {native_unit_contract: {excluded_constructions: []}}\n"
    )
    (tmp_path / "model.mlir").write_text("module {}")
    record = dict(
        status="bundled", input_abi=[dict(dtype="f16", shape=[0, 4])], output_abi=[dict(dtype="f16", shape=[0, 4])]
    )
    plan = prepare_spec_int8(
        tmp_path,
        dict(cases=[record]),
        target="test",
        facts={},
        software_spec=specification,
        host_capabilities=None,
        provider=Provider(),
        package=None,
    )
    assert plan["refusal_code"] == "empty_tensor_no_device_work"
    assert record["numeric_admission"]["status"] == "host"


def test_dtype_bounds_allow_new_inputs_but_bind_source(tmp_path):
    from merlin.targetgen.core_aten_components import require_component_identity

    source = "builtin.module {}"
    (tmp_path / "model.mlir").write_text(source + "\n")
    (tmp_path / "inputs.npz").write_bytes(b"complete recorded payload")
    plan = dict(
        exact_dtype_sites=1,
        schedule_proof=dict(source_sha256=hashlib.sha256(source.encode()).hexdigest()),
        input_bound_proof=dict(method="full_dtype_storage_domain", absolute_bounds=[1 << 31]),
        native_semantic_proof=dict(status="proven"),
    )
    require_component_identity(tmp_path, plan)
    (tmp_path / "inputs.npz").write_bytes(b"different payload")
    require_component_identity(tmp_path, plan)
    plan["input_bound_proof"]["input_sha256"] = "legacy pin"
    with pytest.raises(ValueError, match="input-independent"):
        require_component_identity(tmp_path, plan)
    del plan["input_bound_proof"]["input_sha256"]
    (tmp_path / "model.mlir").write_text(source + " // changed\n")
    with pytest.raises(ValueError, match="source changed"):
        require_component_identity(tmp_path, plan)
