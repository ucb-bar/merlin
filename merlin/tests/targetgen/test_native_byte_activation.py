"""Actual encoded activation instructions must satisfy the raw-readout contract."""

import pytest
from mlir_fixtures import pointwise

from merlin.targetgen.native_byte_activation import audit_program


def program():
    weights = "".join("\\01" if i == j else "\\00" for i in range(16) for j in range(16))
    instructions = [
        (0, (0x3F800000 << 32) | 9, 16),
        (0, (0x3F800000 << 32) | 1, 16),
        (0, 2, (0x3F800000 << 32) | 64),
        (1, "ptrtoint (ptr @unit__relu_identity to i64)", (16 << 48) | (16 << 32)),
        (2, "%input", (1 << 48) | (16 << 32) | 32),
        (0, (0x3F800000 << 32) | (1 << 16) | 12, 1 << 48),
        (4, (1 << 48) | (16 << 32) | 32, (16 << 48) | (16 << 32) | 0xFFFFFFFF),
        (0, (0x3F800000 << 32) | (1 << 16) | 4, 1 << 48),
        (4, (1 << 48) | (16 << 32) | 64, (16 << 48) | (16 << 32) | 0xFFFFFFFF),
        (3, 1234, (1 << 48) | (16 << 32) | 0xA0000000),
    ]
    code = "\n".join(
        f'call void asm sideeffect ".insn r 0x7b, 0x3, 0x{f:x}, x0, $0, $1", "r,r,~{{memory}}"(i64 {a}, i64 {b})'
        for f, a, b in instructions
    )
    return (
        f'@unit__relu_identity = internal constant [256 x i8] c"{weights}"\ndefine void @unit(ptr %x) {{\n{code}\n}}\n'
    )


def test_zero_shift_signed_byte_relu_and_raw_widening_are_authenticated():
    proof = audit_program(program(), "unit")
    assert proof["activated"] == proof["unactivated"] == proof["readouts"] == 1
    assert proof["computes"] == 2 and proof["mesh_partial_absolute_bound"] < 2**19


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.replace("\\01", "\\02", 1),
        lambda s: s.replace(str((0x3F800000 << 32) | (1 << 16) | 12), str((0x3F800000 << 32) | (1 << 16) | 8)),
        lambda s: s.replace(str((0x3F800000 << 32) | (1 << 16) | 12), str((0x40000000 << 32) | (1 << 16) | 12)),
        lambda s: s.replace(str((16 << 48) | (16 << 32) | 0xFFFFFFFF), str((16 << 48) | (16 << 32) | 0x80000000)),
        lambda s: s.replace(str((1 << 48) | (16 << 32) | 0xA0000000), str((1 << 48) | (16 << 32) | 0x80000000)),
        lambda s: s.replace(str((1 << 48) | (16 << 32) | 32), str((1 << 48) | (17 << 32) | 32)),
        lambda s: s.replace("0x4, x0", "0x5, x0", 1),
    ],
)
def test_changed_weights_os_scales_mesh_d_or_narrow_readout_refuse(mutate):
    with pytest.raises(ValueError):
        audit_program(mutate(program()), "unit")


@pytest.mark.parametrize(
    "dtype,zero,predicate",
    [("i8", "0", "slt"), ("i64", "0", "slt"), ("f16", "0.0", "olt"), ("bf16", "0.0", "olt"), ("f32", "0.0", "olt")],
)
def test_a_native_zero_constant_needs_no_intermediate_integer_cast(dtype, zero, predicate):
    from merlin.xdsl_dialects.lowering.contraction_coverage import is_compare_select_relu

    compare = "cmpi" if dtype.startswith("i") else "cmpf"
    body = f"""%zero=arith.constant {zero} : {dtype}
      %p=arith.{compare} {predicate},%a,%zero : {dtype}
      %v=arith.select %p,%zero,%a : {dtype}"""
    module = pointwise([f"tensor<4x{dtype}>"], f"tensor<4x{dtype}>", body)
    root = next(op for op in module.walk() if op.name == "linalg.generic")
    assert is_compare_select_relu(root)
    constant = next(op for op in root.body.block.ops if op.name == "arith.constant")
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr

    element = root.body.block.args[0].type
    constant.properties["value"] = IntegerAttr(1, element) if dtype.startswith("i") else FloatAttr(1.0, element)
    assert not is_compare_select_relu(root)


def test_catalog_normalization_is_sealed_before_component_identity_check(tmp_path, monkeypatch):
    import hashlib
    import json

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.targetgen import software_spec, transfer_contracts
    from merlin.targetgen.core_aten_components import require_component_identity
    from merlin.targetgen.native_byte_activation import prepare

    text = """module {func.func @forward(%x:tensor<4xi8>) -> tensor<4xi8> {
      // Original captured comments and unnormalized spelling are not the catalog's bytes.
      %e=tensor.empty() : tensor<4xi8>
      %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
        iterator_types=["parallel"]}
        ins(%x:tensor<4xi8>) outs(%e:tensor<4xi8>) {
        ^bb0(%a:i8,%unused:i8):
        %zero=arith.constant 0 : i8
        %p=arith.cmpi slt,%a,%zero : i8
        %v=arith.select %p,%zero,%a : i8
        linalg.yield %v : i8
      } -> tensor<4xi8>
      return %r : tensor<4xi8>
    }}"""
    module = parse_mlir_text(text)
    (tmp_path / "model.mlir").write_text(text)
    monkeypatch.setattr(software_spec, "admit_operation", lambda *a, **kw: dict(status="admitted"))
    monkeypatch.setattr(transfer_contracts, "screen_transfer_contract", lambda *a, **kw: dict(status="admitted"))

    class Provider:
        def core_aten_catalog(self, path, stage, **kwargs):
            emitted = path.read_text()
            parsed = parse_mlir_text(emitted)
            ordinal = next(i for i, op in enumerate(parsed.walk()) if op.name == "linalg.generic")
            one = dict(numerator="1", denominator="1")
            zero = dict(numerator="0", denominator="1")
            proof = dict(
                operation="identity_signed_byte_then_ReLU",
                input_range=[-128, 127],
                output_range=[0, 127],
                shift=0,
                clipping_is_identity=True,
            )
            meta = dict(
                form="exact_linear_i32",
                device_relu=True,
                execute_activation_proof=proof,
                arithmetic="signed_byte_execute_relu",
                readout="full_i32",
                integer_weights=["1"],
                device_constant=False,
            )
            readout = dict(
                terms=[["acc_combined", [one] * 4]],
                constant=[zero] * 4,
                relu=False,
                rounding="half_to_even",
                zero_point=0,
                quant_min=-128,
                quant_max=127,
            )
            manifest = dict(
                source_sha256=hashlib.sha256(emitted.encode()).hexdigest(),
                bindings=[
                    dict(
                        root_ordinal=ordinal,
                        symbol="unit",
                        operands={"x": dict(func_arg=0)},
                        result={"y": dict(op_ordinal=ordinal, result=0)},
                    )
                ],
                kernels=[dict(symbol="unit", meta=meta, requantization=[readout])],
            )
            output = stage / "catalog.json"
            output.write_text(json.dumps(manifest))
            (stage / "core_aten.ll").write_text(program())
            return output, stage / "unit.o"

    # This test checks source/proof sealing; instruction contracts are checked
    # separately and the real catalog path is exercised by the Spike lane.
    import subprocess

    def simplify(argv, **kwargs):
        from pathlib import Path

        Path(argv[-1]).write_text(program())
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", simplify)
    selection = {"spec": {"numerical_semantics": {"native_unit_contract": {"mesh_mac": {"signed_bits": 20}}}}}
    plan = prepare(tmp_path, dict(cases=[dict(status="bundled")]), selection, Provider(), None, module)
    assert plan is not None
    require_component_identity(tmp_path, plan)
    assert (tmp_path / "model.mlir").read_text() == str(module) + "\n"
