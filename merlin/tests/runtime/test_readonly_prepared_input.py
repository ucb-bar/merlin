"""A provider lease requires the actual source-bound frozen member, not a hint."""

import copy
import hashlib
from dataclasses import replace

import pytest
from test_prepared_staged_kernel import selected

from merlin.compile.frozen_inputs import FrozenInput, FrozenInputs, specialize
from merlin.runtime.outlined_chain import render_outlined_chain
from merlin.runtime.prepared_staged_kernel import ReadonlyInput, readonly_bindings, reserve_states, verify_selection
from merlin.runtime.staged_dispatch import StagedRoles


def frozen_selection(tmp_path):
    plan, kernel = selected(tmp_path)
    path = tmp_path / "input.bin"
    path.write_bytes(bytes(range(32)))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    permission = FrozenInputs(
        plan.source_ir_sha256,
        (FrozenInput(1, "i32", (2, 4), path, digest),),
        4096,
        4096,
    )
    plan, blob = specialize(plan, permission, tmp_path / "freeze")
    kernel = replace(kernel, readonly_inputs=(ReadonlyInput(1, "i32", (2, 4), 32, digest, 16),))
    plan, layout = reserve_states(plan, {0: kernel}, budget_bytes=4096)
    return plan, kernel, layout, blob


def test_actual_frozen_member_is_exposed_as_weight_without_double_offset(tmp_path):
    plan, kernel, layout, blob = frozen_selection(tmp_path)
    verify_selection(plan, {0: kernel}, layout)
    binding = readonly_bindings(plan, 0, kernel)[1]
    assert blob == bytes(range(32))
    assert binding["buffer"] == plan.dispatches[0]["inputs"][1]
    assert binding["blob_offset"] == 0 and binding["runtime_argument_removed"] is True
    roles = StagedRoles((0, 1, 2), 2, 2, 2, 2, (0, 1), (0, 1))
    generated = render_outlined_chain(
        plan, ("unused",), roles=roles, prepared_kernels={0: kernel}, provider_state_layout=layout
    )
    assert "{MERLIN_WEIGHT,0,2,{2,4},4}" in generated["source"]
    assert generated["metadata"]["external_input_source_indices"] == [0]
    assert kernel.record()["readonly_inputs"] == [kernel.readonly_inputs[0].record()]


@pytest.mark.parametrize("change", ["dynamic", "content", "shape", "dtype", "span", "alignment", "operand"])
def test_invalid_immutable_requirement_refuses(tmp_path, change):
    plan, kernel, layout, _ = frozen_selection(tmp_path)
    requirement = kernel.readonly_inputs[0]
    if change == "dynamic":
        requirement = replace(requirement, index=0)
    elif change == "content":
        requirement = replace(requirement, sha256="0" * 64)
    elif change == "shape":
        requirement = replace(requirement, shape=(4, 2))
    elif change == "dtype":
        requirement = replace(requirement, dtype="f32")
    elif change == "span":
        requirement = replace(requirement, bytes=31)
    elif change == "alignment":
        requirement = replace(requirement, alignment=128)
    else:
        requirement = replace(requirement, index=2)
    with pytest.raises(ValueError):
        verify_selection(plan, {0: replace(kernel, readonly_inputs=(requirement,))}, layout)


@pytest.mark.parametrize("change", ["source", "external", "range", "workspace", "writer", "view", "schema", "root"])
def test_modified_binding_or_writer_refuses(tmp_path, change):
    plan, kernel, layout, _ = frozen_selection(tmp_path)
    frozen = copy.deepcopy(plan.frozen_inputs)
    bid = plan.dispatches[0]["inputs"][1]
    if change == "source":
        frozen["source_ir_sha256"] = "0" * 64
    elif change == "external":
        frozen["external_input_source_indices"].append(1)
    elif change == "range":
        frozen["readonly_buffers"][bid]["offset"] = 64
    elif change == "workspace":
        plan = replace(plan, buffers=[replace(b, workspace_offset=0) if b.id == bid else b for b in plan.buffers])
    elif change == "writer":
        plan = replace(plan, dispatches=[{**plan.dispatches[0], "outputs": [bid]}])
    elif change == "view":
        plan = replace(plan, trailing_views=[{"kind": "alias", "input": "other", "output": bid}])
    elif change == "schema":
        frozen["schema"] = "metadata-only-promise"
    else:
        frozen["roots"][0]["buffer"] = "other"
    plan = replace(plan, frozen_inputs=frozen)
    with pytest.raises(ValueError):
        verify_selection(plan, {0: kernel}, layout)


@pytest.mark.parametrize(
    "kwargs", [{"index": True}, {"shape": (0,)}, {"bytes": 0}, {"sha256": "x" * 64}, {"alignment": 3}]
)
def test_malformed_requirement_refuses(kwargs):
    values = dict(index=1, dtype="i8", shape=(2, 4), bytes=8, sha256="0" * 64, alignment=1)
    with pytest.raises(ValueError):
        ReadonlyInput(**(values | kwargs))


def test_default_contract_record_and_render_remain_unchanged(tmp_path):
    plan, kernel = selected(tmp_path)
    assert "readonly_inputs" not in kernel.record()
    assert readonly_bindings(plan, 0, kernel) == {}
    plan, layout = reserve_states(plan, {0: kernel}, budget_bytes=4096)
    roles = StagedRoles((0, 1, 2), 2, 2, 2, 2, (0, 1), (0, 1))
    generated = render_outlined_chain(
        plan, ("unused",), roles=roles, prepared_kernels={0: kernel}, provider_state_layout=layout
    )
    assert "MERLIN_WEIGHT" not in generated["source"]


def test_actual_nonzero_frozen_offset_is_metadata_not_descriptor_displacement(tmp_path):
    from test_prepared_staged_kernel import SOURCE

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.deployable_ir import checked_generic_text
    from merlin.llvmlower.deployable_outline import canonical_kernel, plan_deployable_chain

    _, kernel = selected(tmp_path)
    text = SOURCE.replace("@forward(", "@forward(%p:tensor<2x4xi32>,")
    text = (
        text.replace(
            "affine_map<(i,j)->(i,j)>],",
            "affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],",
        )
        .replace(
            "ins(%a,%b:tensor<2x4xi32>,tensor<2x4xi32>)",
            "ins(%a,%b,%p:tensor<2x4xi32>,tensor<2x4xi32>,tensor<2x4xi32>)",
        )
        .replace("^bb0(%x:i32,%y:i32,%o:i32):", "^bb0(%x:i32,%y:i32,%pv:i32,%o:i32):")
        .replace("linalg.yield %z:i32", "%sum=arith.addi %z,%pv:i32\n linalg.yield %sum:i32")
    )
    plan = plan_deployable_chain(
        parse_mlir_text(text), worker_slots=(0,), worker_count=2, workspace_budget_bytes=4096, alignment_bytes=64
    )
    files = [tmp_path / "p.bin", tmp_path / "b.bin"]
    for i, path in enumerate(files):
        path.write_bytes(bytes([i + 1]) * 32)
    roots = tuple(
        FrozenInput(i, "i32", (2, 4), path, hashlib.sha256(path.read_bytes()).hexdigest())
        for i, path in zip((0, 2), files, strict=True)
    )
    permission = FrozenInputs(plan.source_ir_sha256, roots, 4096, 4096)
    plan, blob = specialize(plan, permission, tmp_path / "freeze")
    source_hash = hashlib.sha256(
        checked_generic_text(canonical_kernel(plan.outlined.module, plan.dispatches[0]["symbol"])).encode()
    ).hexdigest()
    kernel = replace(
        kernel,
        source_ir_sha256=source_hash,
        readonly_inputs=(ReadonlyInput(1, "i32", (2, 4), 32, roots[1].sha256, 16),),
    )
    plan, layout = reserve_states(plan, {0: kernel}, budget_bytes=4096)
    assert readonly_bindings(plan, 0, kernel)[1]["blob_offset"] == 64
    assert blob[64:] == files[1].read_bytes()
    roles = StagedRoles((0, 1, 2), 2, 2, 2, 2, (0, 1), (0, 1))
    generated = render_outlined_chain(
        plan, ("unused",), roles=roles, prepared_kernels={0: kernel}, provider_state_layout=layout
    )
    assert "{MERLIN_WEIGHT,64,2,{2,4},4}" in generated["source"]
