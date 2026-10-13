"""Public typed transfers cannot disappear into shape-orchestration exemptions."""

import pytest
from xdsl.dialects import builtin, tensor
from xdsl.ir import Block

from merlin.common import mlir_query as mq
from merlin.targetgen import eligibility as E
from merlin.targetgen.compute_units import SemanticCapability
from merlin.xdsl_dialects.lowering import compute_groups as CG

demand = pytest.importorskip("merlin_experiments.phase1.feedback.private_source_demand")


def selected(monkeypatch, *, dtype="int8", forms=()):
    capability = SemanticCapability("movement", dtypes=(dtype,), forms=forms)
    monkeypatch.setattr(E, "capability_map_from_contract", lambda contract: {"movement": capability})


def derive(groups, *, source_operations=()):
    return demand.derive(groups, (), (), source_sha256="a" * 64, normalized_sha256="b" * 64,
        capability={"selected": "public synthetic movement"}, software={},
        facts_sha256="c" * 64, numeric_policy_sha256="d" * 64, source_operations=source_operations)


def slice_group():
    typ = builtin.TensorType(builtin.i8, [2, 3])
    argument = Block(arg_types=[typ]).args[0]
    transfer = tensor.ExtractSliceOp(argument, [0, 0], [2, 3], [1, 1], typ)
    return CG.Group(0, CG.HOST, None, members=[transfer], stages=[CG.MOVEMENT], in_dtype="int8")


def test_supported_tensor_transfer_cannot_be_waived_as_host_orchestration(monkeypatch):
    selected(monkeypatch)
    with pytest.raises(demand.SourceDemandIncomplete) as error:
        derive([slice_group()])
    assert error.value.receipt["n_unknown_host_groups"] == 1


def test_unknown_constrained_transfer_form_is_not_hardware_unsupported(monkeypatch):
    selected(monkeypatch, forms=("copy",))
    group = slice_group()
    group.refusal = "form"
    assert not demand._hardware_exclusion(group, {"selected": "public"})
    with pytest.raises(demand.SourceDemandIncomplete):
        derive([group])


def test_exact_shape_cast_remains_orchestration_without_element_conversion(monkeypatch):
    selected(monkeypatch)
    typ = builtin.TensorType(builtin.i8, [2, 3])
    argument = Block(arg_types=[typ]).args[0]
    view = tensor.CastOp(argument, typ)
    group = CG.Group(0, CG.HOST, None, members=[view], stages=[CG.VIEW], in_dtype="int8")
    assert derive([group])["host_reason_counts"] == {"shape_metadata_orchestration": 1}
    group.members = [tensor.CastOp(argument, builtin.TensorType(builtin.f32, [2, 3]))]
    with pytest.raises(demand.SourceDemandIncomplete):
        derive([group])


def test_bulk_memref_copy_outside_linalg_inventory_remains_demanded(monkeypatch):
    selected(monkeypatch, forms=("copy",))
    module = mq.parse("""module {
      func.func @copy(%src: memref<2x3xi8>, %dst: memref<2x3xi8>) {
        "memref.copy"(%src, %dst) : (memref<2x3xi8>, memref<2x3xi8>) -> ()
        func.return
      }
    }""")
    with pytest.raises(demand.SourceDemandIncomplete) as error:
        derive([], source_operations=tuple(module.walk()))
    assert error.value.receipt["n_missing_direct_groups"] == 1


def test_actual_transfer_dtype_exclusion_permits_linked_host_work(monkeypatch):
    selected(monkeypatch, dtype="fp32", forms=("copy",))
    assert derive([slice_group()])["host_reason_counts"] == {"selected_hardware:data_transfer_unsupported": 1}


def test_stored_group_precision_cannot_hide_one_actual_supported_member(monkeypatch):
    from test_source_member_hardware import _cast

    caps = {"elementwise_map": SemanticCapability("elementwise_map", dtypes=("int8",))}
    monkeypatch.setattr(E, "capability_map_from_contract", lambda contract: caps)
    floating = _cast()
    group = CG.Group(0, CG.HOST, None, members=[floating], stages=[CG.CAST], in_dtype="int8",
                     refusal="no_device_form")
    assert demand._hardware_exclusion(group, {"selected": "public"})
    assert derive([group])["host_reason_counts"] == {"selected_hardware:actual_source_members_unsupported": 1}
    group.members.append(_cast("i8", "i8", "arith.bitcast"))
    group.stages.append(CG.ELEMENTWISE)
    group.in_dtype = "fp32"
    assert not demand._hardware_exclusion(group, {"selected": "public"})
    with pytest.raises(demand.SourceDemandIncomplete):
        derive([group])
