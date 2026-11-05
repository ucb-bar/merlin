"""Finite original group controls do not grant grouping or placement legality."""

import hashlib
import inspect
import json
import marshal
from dataclasses import asdict, replace

import pytest
from test_source_dense_lowering import run, selection
from test_source_offload_selection import attach_plan

from merlin.common import mlir_query as mq
from merlin.llvmlower.source_dense import offload_context


def group_encode(member, operation):
    raise AssertionError("missing pointer capability must never invoke group encoding")


def group_project(call):
    raise AssertionError("missing pointer capability must never invoke group projection")


def alternate_group_encode(member, operation):
    return group_encode(member, operation)


def alternate_adapter_encode(operation):
    raise AssertionError("context mutation test must never invoke alternate encoding")


def configured(tmp_path):
    from merlin.compile.scheduling.compact_dense_group_check import DenseLHSReuseLaw
    from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects
    from merlin.llvmlower.source_dense_group import SourceDenseGrouping

    config, _, seen = selection(tmp_path)
    original = config.adapter
    law = DenseLHSReuseLaw(
        original.configuration_identity,
        original.numerical_policy_identity,
        original.qualification_identity,
        "neutral-original-group-context-test-premises",
        lhs_unchanged_by_member_operations=True,
        independent_member_interleaving=True,
        suppressed_loads_no_faults=True,
        suppressed_loads_no_extra_observations=True,
    )
    storages = tuple(
        replace(
            config.storage,
            lhs=replace(config.storage.lhs, name="lhs"),
            rhs=replace(config.storage.rhs, name=f"rhs{i}", base=config.storage.rhs.base + 12 * i),
            accumulator=replace(
                config.storage.accumulator, name=f"acc{i}", base=config.storage.accumulator.base + 12 * i
            ),
        )
        for i in range(2)
    )
    grouping = SourceDenseGrouping(storages, group_encode, group_project, law)
    return replace(
        config,
        lhs_grouping=grouping,
        borrowed_effects=BorrowedBufferEffects("neutral-original-group-context-borrowed-premises", True, True, True),
    ), seen


def test_group_context_is_closed_finite_and_binds_all_four_original_callbacks(tmp_path):
    from merlin.llvmlower.source_dense_group import grouping_record

    config, _ = configured(tmp_path)
    first = grouping_record(config)
    assert set(first) == {"schema", "storages", "retention_law", "max_members", "callbacks"}
    assert first["schema"] == "merlin.source_dense_grouping.v1"
    assert first["storages"] == [asdict(s) for s in config.lhs_grouping.storages]
    assert first["retention_law"] == asdict(config.lhs_grouping.retention_law)
    assert first["max_members"] == 2
    assert json.loads(json.dumps(first, allow_nan=False)) == first
    roles = ("adapter.encode", "adapter.project", "grouping.encode_member", "grouping.project_member")
    callbacks = (config.adapter.encode, config.adapter.project, group_encode, group_project)
    assert [row["role"] for row in first["callbacks"]] == list(roles)
    for row, callback in zip(first["callbacks"], callbacks, strict=True):
        assert set(row) == {"role", "module", "qualname", "source_path", "source_sha256", "code_sha256"}
        function = getattr(callback, "__func__", callback)
        assert row["module"] == function.__module__ and row["qualname"] == function.__qualname__
        from pathlib import Path

        source = Path(inspect.getsourcefile(function)).resolve()
        assert row["source_path"] == str(source)
        assert row["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert row["code_sha256"] == hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()
    assert grouping_record(config) == first
    assert offload_context(config) == offload_context(config)
    # Returned JSON data cannot alter the original live selection or its scope.
    first["callbacks"][0]["module"] = "changed detached report"
    assert grouping_record(config)["callbacks"][0]["module"] != first["callbacks"][0]["module"]


@pytest.mark.parametrize("control", ["storage", "law", "group_encoder", "adapter_encoder"])
def test_each_original_group_control_changes_the_bound_context(tmp_path, control):
    config, _ = configured(tmp_path)
    before = offload_context(config)
    group = config.lhs_grouping
    if control == "storage":
        storages = (group.storages[0], replace(group.storages[1], rhs=replace(group.storages[1].rhs, base=27)))
        current = replace(config, lhs_grouping=replace(group, storages=storages))
    elif control == "law":
        current = replace(
            config,
            lhs_grouping=replace(
                group,
                retention_law=replace(group.retention_law, qualification_identity="different original qualification"),
            ),
        )
    elif control == "group_encoder":
        current = replace(config, lhs_grouping=replace(group, encode_member=alternate_group_encode))
    else:
        current = replace(config, adapter=replace(config.adapter, encode=alternate_adapter_encode))
    assert offload_context(current) != before


def test_group_max_members_requires_its_original_bounded_assignment_roster(tmp_path):
    config, _ = configured(tmp_path)
    for maximum in (True, 1, 3):
        with pytest.raises(ValueError, match="grouping"):
            replace(config.lhs_grouping, max_members=maximum)


def two_contractions(config):
    from test_dense_source import contraction

    m, k, n = 3, 7, 5
    dtypes = config.selected_contracts.dense.dtypes
    lhs, rhs, acc = dtypes
    return mq.parse(f"""module {{func.func @forward(%a:tensor<{m}x{k}x{lhs}>,%b:tensor<{k}x{n}x{rhs}>)
 ->(tensor<{m}x{n}x{acc}>,tensor<{m}x{n}x{acc}>) {{
 {contraction((m, k, n), dtypes, output="first")}
 {contraction((m, k, n), dtypes, output="second")}
 return %first,%second:tensor<{m}x{n}x{acc}>,tensor<{m}x{n}x{acc}>
}}}}""")


@pytest.mark.parametrize("with_plan", [False, True])
def test_missing_group_pointer_capability_retains_actual_legal_singletons(tmp_path, with_plan):
    config, seen = configured(tmp_path)
    module = two_contractions(config)
    if with_plan:
        config = attach_plan(config, module, indices=(0, 1))
    path, _ = run(tmp_path, config, module)
    assert len(seen) == 2
    assert all(not binding.writer.result_arguments for binding in seen)
    assert all(len(binding.compilation.operations) == 1 for binding in seen)
    report = json.loads((path.parent / "source_dense.json").read_text())
    assert len(report["bindings"]) == 2
    assert any("pointer transport" in reason for reason in report["declined"])
    if with_plan:
        assert report["offload_placement"][0]["actual_route"] == "candidate"
        assert len(report["offload_placement"][0]["selected_dense_ordinals"]) == 2


def test_unchanged_none_grouping_remains_a_finite_ordinary_context(tmp_path):
    config, _, _ = selection(tmp_path)
    assert config.lhs_grouping is None
    context = offload_context(config)
    assert context == offload_context(config)
    json.dumps(asdict(context), allow_nan=False)


@pytest.mark.parametrize("case_index", [0, 1])
def test_original_native_provider_constructs_both_choices_without_compiling(tmp_path, case_index):
    from test_shared_lhs_provider import SharedLHSProvider, original_model
    from test_shared_lhs_source import CASES, reference

    case = CASES[case_index]
    facts = tmp_path / "facts"
    facts.mkdir()
    model = original_model(facts, case)
    _, _, trace, _ = reference(case)
    for grouped in (False, True):
        provider = SharedLHSProvider(tmp_path / str(grouped), model, trace, grouped=grouped)
        config = provider.selection()
        assert (config.lhs_grouping is not None) == grouped
        if grouped:
            assert config.lhs_grouping.storages == model[1]
            assert config.lhs_grouping.retention_law.admits(config.adapter)
        assert provider.records == []
        assert not list(provider.work.iterdir())


def test_changed_original_group_qualification_cannot_reuse_existing_source_plan(tmp_path):
    config, seen = configured(tmp_path)
    config = attach_plan(config, two_contractions(config), indices=(0, 1))
    group = config.lhs_grouping
    with pytest.raises(ValueError, match="context|stale|changed|replan"):
        replace(
            config,
            lhs_grouping=replace(
                group,
                retention_law=replace(group.retention_law, qualification_identity="different reviewed qualification"),
            ),
        )
    assert seen == []
