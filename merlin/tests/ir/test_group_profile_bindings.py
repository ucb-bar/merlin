"""Source-derived void-writer profiling without guessing a renamed symbol/ABI."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_fresh_tensor_writer import contract, fixture
from xdsl.dialects import arith, func

from merlin.llvmlower.device_build import apply_post_offload_transform
from merlin.llvmlower.fresh_tensor_writer import rewrite_fresh_tensor_writers
from merlin.llvmlower.group_profile_bindings import derive_bindings, read_bindings
from merlin.runtime.whole_model_readback import ReadbackError, callees, group_names, render_group_profile
from merlin.transforms.certificates import ir_text


def pair():
    original = fixture()
    selected = original.clone()
    rewrite_fresh_tensor_writers(selected, [replace(contract(), wrapper_symbol="renamed")])
    return original, selected


def test_profile_uses_actual_borrowed_void_abi_and_original_source_route():
    original, selected = pair()
    routed = [dict(symbol="writer", group=3)]
    bindings = derive_bindings(original, selected, routed)
    assert bindings == dict(
        writer=dict(borrowed_symbol="borrowed", buffers=3, argument_ranks=[1, 1, 1], wrapper_symbol="renamed")
    )
    actual = callees(routed, dict(writer=(4, 7, 4)), {"_mlir_ciface_borrowed"}, writer_bindings=bindings)
    assert actual[0].abi == "borrowed_c_interface" and actual[0].routed_symbol == "writer"
    source = render_group_profile(actual, group_names(routed))
    assert "void __wrap__mlir_ciface_borrowed(void *arg_0, void *arg_1, void *arg_2)" in source
    assert "__real__mlir_ciface_borrowed(arg_0, arg_1, arg_2);" in source
    assert '"g3_writer"' in source and 'merlin_callee_names[] = {"writer"}' in source
    assert "void *result" not in source


@pytest.mark.parametrize("fault", ["argument", "operation", "result", "declaration", "shared"])
def test_changed_or_unknown_source_cannot_manufacture_a_profile_binding(fault):
    original, selected = pair()
    functions = {f.sym_name.data: f for f in selected.body.block.ops}
    wrapper = functions["renamed"]
    if fault == "argument":
        call = next(o for o in functions["forward"].walk() if type(o) is func.CallOp)
        a, b, c = call.operands
        call.operands = (c, b, a)
    elif fault == "operation":
        wrapper.body.block.insert_op_before(arith.ConstantOp.from_int_and_width(1, 32), wrapper.body.block.last_op)
    elif fault == "result":
        tensor = wrapper.body.block.last_op.arguments[0].owner
        tensor.operands = (wrapper.body.block.first_op.results[0],)
    elif fault == "declaration":
        del functions["borrowed"].attributes["llvm.emit_c_interface"]
    else:
        call = next(o for o in wrapper.walk() if type(o) is func.CallOp)
        wrapper.body.block.insert_op_before(call.clone(), call)
    with pytest.raises(ReadbackError):
        derive_bindings(original, selected, [dict(symbol="writer")])


@pytest.mark.parametrize("fault", ["source", "routing", "missing_snapshot"])
def test_profile_reopens_original_snapshot_and_routing_identity(tmp_path, fault):
    original, selected = pair()
    source, sidecar = tmp_path / "source.mlir", tmp_path / "sidecar.json"
    source.write_text(ir_text(original))
    sidecar.write_text(json.dumps(dict(routed=[dict(symbol="writer")])))

    def callback(path, identity, work):
        path.write_text(ir_text(selected))  # In-place callbacks must retain the original source too.
        return path

    apply_post_offload_transform(SimpleNamespace(post_offload_transform=callback), source, sidecar, tmp_path / "work")
    receipt = tmp_path / "work/post_offload_transform.json"
    assert read_bindings(receipt, source, sidecar)["writer"]["borrowed_symbol"] == "borrowed"
    record = json.loads(receipt.read_text())
    from pathlib import Path

    if fault == "source":
        Path(record["source_snapshot"]).write_text(ir_text(selected))
    elif fault == "routing":
        sidecar.write_text("{}")
    else:
        del record["source_snapshot"]
        receipt.write_text(json.dumps(record))
    with pytest.raises(ReadbackError, match="changed|retained original"):
        read_bindings(receipt, source, sidecar)


def test_mixed_original_and_retargeted_calls_cannot_leave_a_writer_unprofiled():
    original, selected = pair()
    source_forward = original.body.block.first_op
    source_call = next(o for o in source_forward.walk() if type(o) is func.CallOp)
    source_forward.body.block.insert_op_before(source_call.clone(), source_call)
    target_forward = selected.body.block.first_op
    target_call = next(o for o in target_forward.walk() if type(o) is func.CallOp)
    unchanged = target_call.clone()
    from xdsl.dialects.builtin import SymbolRefAttr

    unchanged.properties["callee"] = SymbolRefAttr("writer")
    target_forward.body.block.insert_op_before(unchanged, target_call)
    selected.body.block.add_op(original.body.block.last_op.clone())
    with pytest.raises(ReadbackError, match="ambiguous writer"):
        derive_bindings(original, selected, [dict(symbol="writer")])
    bindings = dict(writer=dict(borrowed_symbol="borrowed", buffers=3, argument_ranks=[1] * 3))
    with pytest.raises(ReadbackError, match="both remain referenced"):
        callees(
            [dict(symbol="writer")],
            dict(writer=(4, 7, 4)),
            {"writer", "_mlir_ciface_borrowed"},
            writer_bindings=bindings,
        )


def test_private_workspace_profile_carries_every_actual_void_descriptor_pointer():
    from merlin.llvmlower.fresh_tensor_writer import PrivateWorkspaceContract

    original = fixture()
    selected = original.clone()
    workspace = PrivateWorkspaceContract(37, 16, "a" * 64, True, True, True)
    rewrite_fresh_tensor_writers(
        selected, [replace(contract(), wrapper_symbol="renamed", private_workspaces=(workspace,))]
    )
    bindings = derive_bindings(original, selected, [dict(symbol="writer")])
    assert bindings["writer"]["buffers"] == 4 and bindings["writer"]["argument_ranks"] == [1] * 4
    actual = callees(
        [dict(symbol="writer")], dict(writer=(4, 7, 4)), {"_mlir_ciface_borrowed"}, writer_bindings=bindings
    )
    source = render_group_profile(actual, [("first", "writer")])
    assert "__real__mlir_ciface_borrowed(arg_0, arg_1, arg_2, arg_3);" in source
    import shutil
    import subprocess

    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native C compiler unavailable")
    subprocess.run(
        [compiler, "-fsyntax-only", "-x", "c", "-"], input=source, text=True, check=True, capture_output=True
    )


def test_binding_refuses_prepared_source_changes_during_observation(tmp_path, monkeypatch):
    from merlin.llvmlower import group_profile_bindings as owner

    original, selected = pair()
    source, sidecar = tmp_path / "source.mlir", tmp_path / "sidecar.json"
    source.write_text(ir_text(original))
    sidecar.write_text(json.dumps(dict(routed=[dict(symbol="writer")])))

    def callback(path, identity, work):
        path.write_text(ir_text(selected))
        return path

    apply_post_offload_transform(SimpleNamespace(post_offload_transform=callback), source, sidecar, tmp_path / "work")
    derive = owner.derive_bindings

    def changed(*args):
        result = derive(*args)
        source.write_text(ir_text(original))
        return result

    monkeypatch.setattr(owner, "derive_bindings", changed)
    with pytest.raises(ReadbackError, match="changed during binding"):
        owner.read_bindings(tmp_path / "work/post_offload_transform.json", source, sidecar)


@pytest.mark.parametrize("fault", ["new_function", "module_attribute", "calling_convention"])
def test_unknown_added_helpers_or_abi_effects_remain_profile_barriers(fault):
    original, selected = pair()
    from xdsl.dialects.builtin import StringAttr

    if fault == "new_function":
        selected.body.block.add_op(func.FuncOp.external("opaque_helper", [], []))
    elif fault == "module_attribute":
        selected.attributes["unknown_execution"] = StringAttr("opaque")
    else:
        external = next(f for f in selected.body.block.ops if f.sym_name.data == "borrowed")
        external.attributes["unknown_abi"] = StringAttr("opaque")
    with pytest.raises(ReadbackError, match="unrecognized function|module-level|void ranked"):
        derive_bindings(original, selected, [dict(symbol="writer")])
