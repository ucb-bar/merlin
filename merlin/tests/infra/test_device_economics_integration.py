"""Actual source rewrites and invocation custody, without hardware qualification."""

import json
from dataclasses import replace

import pytest
from test_device_offload_selection import fixture, priced
from test_offload_plan import choose
from xdsl.dialects import func

from merlin.frontends.linalg_mlir import parse_mlir_file
from merlin.llvmlower import device_offload as D
from merlin.llvmlower.device_offload_selection import bind_device_offload_source
from merlin.llvmlower.device_preparation import (
    device_route_requested,
    finish_device_offload_selection,
    freeze_device_offload_selection,
    prepare_device_route,
    require_device_offload_selection,
    require_supported_device_preparation,
    retain_device_offload_source,
)
from merlin.llvmlower.offload_plan import realize_offload_plan
from merlin.transforms.certificates import ir_text


def selected_route(tmp_path, monkeypatch, mode="explicit"):
    bound, proposal, selector, *_ = fixture(tmp_path, monkeypatch)
    if mode == "explicit":
        selected = bound.select(proposal.plan, explicit=True)
    elif mode == "opportunity":
        selected = priced(proposal)[0]
    elif mode == "unknown":
        selected = priced(proposal, unknown=True)[0]
    else:
        selected = proposal
    routing = replace(bound.routing, offload_selection=selected)
    source = tmp_path / "source.mlir"
    source.write_text(ir_text(bound.source.module))
    work = tmp_path / "work"
    work.mkdir()
    return routing, selected, selector, source, work


@pytest.mark.parametrize("mode,wanted", [("explicit", 1), ("opportunity", 1), ("unknown", 2), ("original", 2)])
def test_actual_same_shape_operations_follow_original_bound_choice(tmp_path, monkeypatch, mode, wanted):
    routing, selected, _, source, work = selected_route(tmp_path, monkeypatch, mode)
    frozen = freeze_device_offload_selection(routing)
    actual = prepare_device_route(routing, source, work, selection=frozen)
    module = parse_mlir_file(actual)
    calls = tuple(op for op in module.walk() if type(op) is func.CallOp)
    dense = tuple(op for op in module.walk() if op.name == "linalg.generic")
    assert len(calls) == wanted and len(dense) == 2 - wanted
    sidecar = json.loads((work / D.SIDECAR_NAME).read_bytes())
    record = sidecar["offload_selection"]
    assert record == selected.record()
    assert [row["source_operation_ordinal"] for row in sidecar["routed"]] == record["selected_ordinals"]
    assert set(sidecar["signatures"]) == {call.callee.root_reference.data for call in calls}
    assert require_device_offload_selection(routing, frozen) is selected
    # The complete original return list still contains both live observations.
    returned = next(op for op in module.walk() if type(op) is func.ReturnOp)
    assert len(returned.arguments) == 2
    assert [value.owner.name for value in returned.arguments] == (
        ["linalg.generic", "func.call"] if wanted == 1 else ["func.call", "func.call"]
    )


@pytest.mark.parametrize("mutation", ["remove", "equal_replacement", "sidecar", "symlink", "source"])
def test_late_selection_or_realized_sidecar_changes_refuse(tmp_path, monkeypatch, mutation):
    routing, selected, _, source, work = selected_route(tmp_path, monkeypatch)
    frozen = freeze_device_offload_selection(routing)
    prepare_device_route(routing, source, work, selection=frozen)
    if mutation == "remove":
        object.__setattr__(routing, "offload_selection", None)
    elif mutation == "equal_replacement":
        object.__setattr__(routing, "offload_selection", replace(selected))
    elif mutation == "source":
        source.write_text("module {}")  # Lost all actual calls while the realized sidecar stays unchanged.
    else:
        path = work / D.SIDECAR_NAME
        if mutation == "sidecar":
            path.write_bytes(path.read_bytes() + b"\n")
        else:
            old = path.with_suffix(".copied")
            path.rename(old)
            path.symlink_to(old)
    with pytest.raises(ValueError, match="selection changed|source or sidecar|direct regular"):
        require_device_offload_selection(routing, frozen)


def test_first_post_rewrite_selector_callback_cannot_replace_actual_source(tmp_path, monkeypatch):
    routing, _, selector, source, work = selected_route(tmp_path, monkeypatch)
    frozen = freeze_device_offload_selection(routing)
    rewrite = D.rewrite_prepared_file

    def arm_after_publication(*args, **kwargs):
        moved = rewrite(*args, **kwargs)
        selector.hook = lambda: source.write_text("module {}")
        return moved

    monkeypatch.setattr(D, "rewrite_prepared_file", arm_after_publication)
    with pytest.raises(ValueError, match="source or sidecar changed"):
        prepare_device_route(routing, source, work, selection=frozen)


def test_final_trusted_preparation_product_retains_both_source_identities(tmp_path, monkeypatch):
    routing, _, selector, source, work = selected_route(tmp_path, monkeypatch)
    frozen = freeze_device_offload_selection(routing)
    prepare_device_route(routing, source, work, selection=frozen)
    final = work / "normal-final.mlir"
    final.write_bytes(source.read_bytes())
    retain_device_offload_source(frozen, final)
    selector.hook = lambda: final.write_text("module {}")
    with pytest.raises(ValueError, match="source or sidecar changed"):
        require_device_offload_selection(routing, frozen)


def test_callback_cannot_replace_file_and_adopt_its_new_identity(tmp_path, monkeypatch):
    import hashlib

    routing, _, selector, source, work = selected_route(tmp_path, monkeypatch)
    frozen = freeze_device_offload_selection(routing)
    prepare_device_route(routing, source, work, selection=frozen)

    def replace_both():
        source.write_text("module {}")
        frozen.sources = ((source, hashlib.sha256(source.read_bytes()).hexdigest()),)

    selector.hook = replace_both
    with pytest.raises(ValueError, match="artifact pins changed"):
        require_device_offload_selection(routing, frozen)


def test_later_opaque_prepared_model_rewrite_requires_correspondence(tmp_path, monkeypatch):
    routing, _, _, _, _ = selected_route(tmp_path, monkeypatch)
    frozen = freeze_device_offload_selection(routing)
    with pytest.raises(ValueError, match="later opaque prepared-model"):
        require_supported_device_preparation(frozen, lambda source, directory: source)
    require_supported_device_preparation(frozen, None)
    require_supported_device_preparation(None, lambda source, directory: source)


@pytest.mark.parametrize(
    "mutation", ["source", "selection", "pins", "numeric", "resources", "source_context", "unchanged"]
)
def test_final_policy_cannot_leave_stale_source_or_selection(tmp_path, monkeypatch, mutation):
    from merlin.llvmlower import device_build

    routing, _, selector, source, work = selected_route(tmp_path, monkeypatch)
    frozen = freeze_device_offload_selection(routing)
    prepare_device_route(routing, source, work, selection=frozen)
    final_policy = {"ran": False}

    def selector_hook():
        assert not final_policy["ran"], "no selector/cost callback may follow final policy verification"

    def policy(*args):
        final_policy["ran"] = True
        if mutation == "source":
            source.write_text("module {}")
        elif mutation == "selection":
            object.__setattr__(routing, "offload_selection", None)
        elif mutation == "pins":
            frozen.sources = ()
        elif mutation == "numeric":
            routing.numeric_policy["comparison"] = "changed"
        elif mutation == "resources":
            budget = routing.device_ir_budget
            object.__setattr__(budget, "max_bytes", 1)
        elif mutation == "source_context":
            from xdsl.dialects.builtin import UnitAttr

            frozen.selection.owner.source.module.attributes["prov.changed"] = UnitAttr()

    selector.hook = selector_hook
    # This test isolates orchestration around the real selected source/files.
    # Actual ELF/report mutation checks have their own real-file tests.
    monkeypatch.setattr(device_build, "freeze_device_elf_custody", lambda *a, **k: object())
    monkeypatch.setattr(device_build, "revalidate_device_elf_custody", policy)
    options = dict(admission=None, result=None, elf=None, linked_sha256=None, target_configuration=None, artifacts=None)
    if mutation == "unchanged":
        finish_device_offload_selection(routing, frozen, **options)
        assert final_policy["ran"]
    else:
        with pytest.raises(ValueError, match="changed|stale"):
            finish_device_offload_selection(routing, frozen, **options)


def test_late_injection_does_not_create_original_selection(tmp_path, monkeypatch):
    routing, _, _, _, _ = selected_route(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="appeared"):
        require_device_offload_selection(routing, None)


def test_original_selector_callback_cannot_replace_frozen_selection(tmp_path, monkeypatch):
    routing, _, selector, _, _ = selected_route(tmp_path, monkeypatch)
    frozen = freeze_device_offload_selection(routing)
    selector.hook = lambda: object.__setattr__(routing, "offload_selection", None)
    with pytest.raises(ValueError, match="selection changed"):
        require_device_offload_selection(routing, frozen)


def test_missing_realized_operation_is_rejected_before_source_file_publication(tmp_path, monkeypatch):
    routing, selected, _, source, work = selected_route(tmp_path, monkeypatch)
    before = source.read_bytes()
    emit = D.emit_device_program

    def omit(*args, **kwargs):
        return emit(*args, **kwargs)[:-1]

    monkeypatch.setattr(D, "emit_device_program", omit)
    with pytest.raises(ValueError, match="none selected|complete realized"):
        prepare_device_route(routing, source, work, selection=freeze_device_offload_selection(routing))
    assert source.read_bytes() == before
    assert selected.record()["selected_ordinals"]


def test_unmodeled_post_offload_rewrite_refuses_before_callback(tmp_path, monkeypatch):
    routing, _, _, _, _ = selected_route(tmp_path, monkeypatch)

    def forbidden(*args):
        pytest.fail("an unsupported later rewrite must not execute")

    with pytest.raises(ValueError, match="later opaque"):
        freeze_device_offload_selection(replace(routing, post_offload_transform=forbidden))


def test_all_cpu_placement_cannot_claim_selected_target_ir_was_consumed(tmp_path, monkeypatch):
    from test_configuration_ir import configured

    bound, proposal, *_ = fixture(tmp_path, monkeypatch)
    directory = tmp_path / "configuration"
    directory.mkdir()
    routing = replace(bound.routing, target_configuration=configured(directory))
    bound = bind_device_offload_source(bound.source.module.clone(), routing=routing)
    plan = realize_offload_plan(
        bound.source,
        choose(bound.source, proposal.plan.device_alternatives, indices=()),
        device_alternatives=proposal.plan.device_alternatives,
        context=bound.context,
    )
    selected = bound.select(plan, explicit=True)
    routing = replace(routing, offload_selection=selected)
    source = tmp_path / "source.mlir"
    source.write_text(ir_text(bound.source.module))
    before = source.read_bytes()
    with pytest.raises(ValueError, match="all-CPU"):
        prepare_device_route(routing, source, tmp_path / "work", selection=freeze_device_offload_selection(routing))
    assert source.read_bytes() == before


def test_inert_routes_and_legacy_sidecar_keep_their_existing_shape(tmp_path):
    source = tmp_path / "unchanged.mlir"
    source.write_text("module {}")
    assert not device_route_requested(None)
    assert prepare_device_route(None, source, tmp_path) == source
    assert "offload_selection" not in D.DeviceRewrite("neutral").to_sidecar()
    assert source.read_text() == "module {}"
