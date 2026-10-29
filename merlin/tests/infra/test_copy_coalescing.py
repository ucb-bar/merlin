"""Qualified neutral adjacent-copy mechanism, not target DMA qualification."""

import struct
from dataclasses import replace

import pytest
from test_private_copy_elimination import SPACES, command
from test_selected_storage_contracts import native  # noqa: F401

from merlin.compile.scheduling.copy_coalescing import coalesce_private_copies
from merlin.compile.scheduling.copy_coalescing_check import CopyMergeLaw, verify_coalesced_copies
from merlin.compile.scheduling.program_limits import AddressWindow, EncodedAccess
from merlin.compile.scheduling.temporary_storage import (
    StorageAdapter,
    StorageEffects,
    TemporaryObject,
)
from merlin.sched.ir import Const, Kernel, TensorArg, evaluate
from merlin.semantic_compiler.allocation_types import StorageBank


def project(call):
    if call.instr not in {"copy", "increment", "observe"}:
        return None
    args = {key: evaluate(value, {}) for key, value in call.args}
    source, destination = SPACES[args["read_bank"]], SPACES[args["write_bank"]]
    return StorageEffects(
        repr((call.instr, source, destination, args.get("mode", 0))),
        (
            EncodedAccess(source, args["source"], args["size"]),
            EncodedAccess(destination, args["destination"], args["size"], True),
        ),
        "synchronous",
        exact_copy=call.instr == "copy",
    )


def rewrite(call, accesses):
    updates = {"source": accesses[0].start, "destination": accesses[1].start, "size": accesses[0].size}
    return replace(
        call, args=tuple((key, Const(updates[key]) if key in updates else value) for key, value in call.args)
    )


def fixture(width=8, *, base=32, alignment=1, backing="memory"):
    adapter = StorageAdapter(project, rewrite, "neutral-contiguous-bytes-v1", "unchanged-bytes-v1", "neutral-C-v1")
    law = CopyMergeLaw(
        adapter.configuration_identity,
        adapter.numerical_policy_identity,
        adapter.qualification_identity,
        "original-neutral-C-bounded-copy-merge-v1",
        4 * width,
        no_faults=True,
        no_extra_observations=True,
    )
    source = Kernel(
        "merge_copies",
        (TensorArg("input", (4 * width,), "i8", "read"), TensorArg("output", (4 * width,), "i8", "write")),
        (
            command("copy", 0, base, 2 * width, 2, 0),  # Complete original private initialization.
            command("copy", width, base, width, 2, 0),
            command("copy", 2 * width, base + width, width, 2, 0),
            command("copy", base, 0, 2 * width, 0, 3),
        ),
    )
    objects = (TemporaryObject("scratch", "bytes", base, 2 * width),)
    banks = (StorageBank("bytes", backing, base + 4 * width, "bytes", alignment),)
    kwargs = dict(
        merge_law=law,
        private_ownership=True,
        external_windows=(AddressWindow("input", 0, 4 * width), AddressWindow("output", 0, 4 * width)),
    )
    return source, objects, banks, adapter, kwargs


def changed(call, **values):
    return replace(call, args=tuple((key, Const(values.get(key, evaluate(value, {})))) for key, value in call.args))


@pytest.mark.parametrize("width,alignment", [(3, 1), (8, 8), (13, 1)])
def test_actual_effects_keep_every_byte_and_remove_one_request(width, alignment):
    source, objects, banks, adapter, kwargs = fixture(width, alignment=alignment)
    result = coalesce_private_copies(source, objects, banks, adapter, **kwargs)
    assert len(result.kernel.body) == 3
    assert project(result.kernel.body[1]).accesses == (
        EncodedAccess("input", width, 2 * width),
        EncodedAccess("bytes", 32, 2 * width, True),
    )
    assert result.report == verify_coalesced_copies(source, result.kernel, objects, banks, adapter, **kwargs)
    assert result.report["merged_calls"] == 1
    assert result.report["requested_read_bytes"] == result.report["requested_write_bytes"] == 6 * width
    assert result.report["machine_emission"] == result.report["execution_qualification"] == "not_established"
    assert result.report["timing"] == "UNKNOWN"


@pytest.mark.parametrize("width,separate,word_format", [(3, False, ">BBBIII"), (8, True, "<BBBQQQ")])
def test_independent_native_byte_executor_observes_fewer_requests_same_complete_state(
    native,  # noqa: F811
    width,
    separate,
    word_format,  # noqa: F811
):
    source, objects, banks, adapter, kwargs = fixture(width)
    result = coalesce_private_copies(source, objects, banks, adapter, **kwargs)
    word, histories = struct.Struct(word_format), []
    for kernel in (source, result.kernel):
        image = bytearray()
        for call in kernel.body:
            args = {key: evaluate(value, {}) for key, value in call.args}
            image += word.pack(
                0, args["read_bank"], args["write_bank"], args["source"], args["destination"], args["size"]
            )
        state = bytes(range(4 * width))
        history = []
        for epoch in range(3):
            data = bytearray((x + 19 * epoch) % 256 for x in state)
            original = bytes(data)
            first, second, output = bytearray([173] * 128), bytearray([149] * 128), bytearray([211] * len(data))
            counts = native(
                image,
                first,
                second,
                data,
                output,
                width=word.size // 3 - 1,
                big=word_format[0] == ">",
                separate=separate,
            )
            assert output == original[width : 3 * width] + bytes([211] * (2 * width))
            assert data == original and second == bytes([149] * 128)
            assert first[:32] == bytes([173] * 32) and first[32 + 2 * width :] == bytes([173] * (96 - 2 * width))
            assert first[32 : 32 + 2 * width] == output[: 2 * width]
            assert counts[:3] == (len(kernel.body), 6 * width, 6 * width)
            history.append(bytes(output))
            state = bytes(output)
        histories.append(history)
    assert histories[0] == histories[1]


@pytest.mark.parametrize("premise", ["no_faults", "no_extra_observations"])
def test_exact_copy_does_not_supply_fault_or_nonmemory_observation_equivalence(premise):
    source, objects, banks, adapter, kwargs = fixture()
    kwargs["merge_law"] = replace(kwargs["merge_law"], **{premise: False})
    assert coalesce_private_copies(source, objects, banks, adapter, **kwargs) is None


@pytest.mark.parametrize(
    "identity", ["configuration_identity", "numerical_policy_identity", "adapter_qualification_identity"]
)
def test_different_original_selection_cannot_reuse_merge_law(identity):
    source, objects, banks, adapter, kwargs = fixture()
    kwargs["merge_law"] = replace(kwargs["merge_law"], **{identity: "another-original-selection"})
    assert coalesce_private_copies(source, objects, banks, adapter, **kwargs) is None


def test_missing_capability_and_explicit_disabled_budget_keep_source():
    source, objects, banks, adapter, kwargs = fixture()
    assert coalesce_private_copies(source, objects, banks, adapter, **(kwargs | {"merge_law": None})) is None
    assert coalesce_private_copies(source, objects, banks, adapter, max_merges=0, **kwargs) is None
    assert coalesce_private_copies(source, objects, banks, adapter, **(kwargs | {"private_ownership": False})) is None


@pytest.mark.parametrize("mutation", ["size", "source", "destination", "mode", "space", "extra", "drop"])
def test_independent_checker_rejects_corrupt_actual_output_without_a_plan(mutation):
    source, objects, banks, adapter, kwargs = fixture()
    result = coalesce_private_copies(source, objects, banks, adapter, **kwargs)
    calls = list(result.kernel.body)
    if mutation in {"size", "source", "destination"}:
        calls[1] = changed(calls[1], **{mutation: evaluate(calls[1].arg(mutation), {}) + 1})
    elif mutation == "mode":
        calls[1] = replace(calls[1], args=(*calls[1].args, ("mode", Const(1))))
    elif mutation == "space":
        calls[1] = changed(calls[1], read_bank=3)
    elif mutation == "extra":
        calls.insert(1, source.body[0])
    else:
        calls.pop()
    with pytest.raises(ValueError):
        verify_coalesced_copies(source, replace(result.kernel, body=tuple(calls)), objects, banks, adapter, **kwargs)


@pytest.mark.parametrize(
    "constraint", ["request", "source_alignment", "destination_alignment", "multiple", "capacity", "start", "window"]
)
def test_actual_resource_and_declared_window_limits_retain_original(constraint):
    source, objects, banks, adapter, kwargs = fixture()
    if constraint in {"request", "source_alignment", "destination_alignment", "multiple"}:
        field, value = {
            "request": ("max_request_bytes", 8),
            "source_alignment": ("source_alignment", 16),
            "destination_alignment": ("destination_alignment", 64),
            "multiple": ("size_multiple", 3),
        }[constraint]
        kwargs["merge_law"] = replace(kwargs["merge_law"], **{field: value})
    elif constraint == "capacity":
        banks = (replace(banks[0], capacity=40),)
    elif constraint == "start":
        banks = (replace(banks[0], start=40),)
    else:
        # Both originals fit a declared window, their union spans two windows.
        kwargs["external_windows"] = (
            AddressWindow("input", 0, 16),
            AddressWindow("input", 16, 16),
            kwargs["external_windows"][1],
        )
    assert coalesce_private_copies(source, objects, banks, adapter, **kwargs) is None


def test_union_alias_is_unsafe_even_when_each_original_copy_is_disjoint():
    source, _, banks, adapter, kwargs = fixture()
    objects = (TemporaryObject("scratch", "bytes", 0, 24),)
    source = replace(
        source,
        body=(
            command("copy", 0, 0, 24, 2, 0),
            command("copy", 0, 8, 8),
            command("copy", 8, 16, 8),
            command("copy", 0, 0, 24, 0, 3),
        ),
    )
    assert coalesce_private_copies(source, objects, banks, adapter, **kwargs) is None


@pytest.mark.parametrize("barrier", ["observer", "unknown", "inexact", "count_key", "implicit_read", "async"])
def test_observers_unknowns_and_unqualified_semantics_are_not_crossed(barrier):
    source, objects, banks, adapter, kwargs = fixture()
    if barrier in {"observer", "unknown"}:
        source = replace(
            source,
            body=(
                *source.body[:2],
                command("observe" if barrier == "observer" else "opaque", 0, 0, 1, 2, 3),
                *source.body[2:],
            ),
        )
    else:

        def decode(call):
            effect = project(call)
            if barrier == "inexact":
                return replace(effect, exact_copy=False)
            if barrier == "count_key":
                return replace(effect, semantic_key=effect.semantic_key + str(effect.accesses[0].size))
            if barrier == "implicit_read":
                return replace(effect, exact_copy=False, accesses=(*effect.accesses, EncodedAccess("input", 31, 1)))
            if call == source.body[0]:
                return replace(effect, completion="asynchronous", token="one")
            return effect

        adapter = replace(adapter, project=decode)
    if barrier == "async":
        # Undrained async is already malformed under the existing completion gate.
        with pytest.raises(ValueError, match="completion|asynchronous"):
            coalesce_private_copies(source, objects, banks, adapter, **kwargs)
    else:
        assert coalesce_private_copies(source, objects, banks, adapter, **kwargs) is None


def test_partial_initial_fragments_are_not_promoted_to_complete_initialization():
    source, objects, banks, adapter, kwargs = fixture()
    source = replace(source, body=source.body[1:])
    assert coalesce_private_copies(source, objects, banks, adapter, **kwargs) is None


def test_rewriter_must_really_change_size_not_only_addresses():
    source, objects, banks, adapter, kwargs = fixture()
    adapter = replace(adapter, rewrite=lambda call, accesses: call)
    with pytest.raises(ValueError, match="realize.*byte effects"):
        coalesce_private_copies(source, objects, banks, adapter, **kwargs)


@pytest.mark.parametrize("mutation", ["law", "bank", "adapter", "callback", "source"])
def test_callback_cannot_change_original_context_even_when_proposal_falls_back(mutation):
    source, objects, banks, adapter, kwargs = fixture()

    def corrupt(call, accesses):
        if mutation == "law":
            object.__setattr__(kwargs["merge_law"], "max_request_bytes", 17)
        elif mutation == "bank":
            object.__setattr__(banks[0], "capacity", 63)
        elif mutation == "adapter":
            object.__setattr__(adapter, "configuration_identity", "changed")
        elif mutation == "callback":
            object.__setattr__(adapter, "project", lambda call: project(call))
        else:
            object.__setattr__(source, "name", "changed")
        return rewrite(call, accesses)

    adapter = replace(adapter, rewrite=corrupt)
    with pytest.raises(ValueError, match="changed during a callback"):
        coalesce_private_copies(source, objects, banks, adapter, **kwargs)


@pytest.mark.parametrize(
    "control,value",
    [("max_calls", True), ("max_calls", 0), ("max_objects", -1), ("max_merges", -1), ("max_merges", False)],
)
def test_work_bounds_are_exact_typed_controls(control, value):
    source, objects, banks, adapter, kwargs = fixture()
    with pytest.raises(ValueError, match="bounded calls"):
        coalesce_private_copies(source, objects, banks, adapter, **kwargs, **{control: value})


def test_bounded_multicopy_group_has_no_growth_and_preserves_all_original_requests():
    source, objects, banks, adapter, kwargs = fixture()
    source = replace(
        source,
        body=(
            command("copy", 0, 32, 24, 2, 0),
            command("copy", 0, 32, 8, 2, 0),
            command("copy", 8, 40, 8, 2, 0),
            command("copy", 16, 48, 8, 2, 0),
            command("copy", 32, 0, 24, 0, 3),
        ),
    )
    objects = (replace(objects[0], size=24),)
    one = coalesce_private_copies(source, objects, banks, adapter, max_merges=1, **kwargs)
    two = coalesce_private_copies(source, objects, banks, adapter, max_merges=2, **kwargs)
    assert len(one.kernel.body) == 4 and len(two.kernel.body) == 3
    assert one.report["merged_calls"] == 1 and two.report["merged_calls"] == 2
    assert one.report["requested_read_bytes"] == two.report["requested_read_bytes"] == 72
    with pytest.raises(ValueError, match="merge budget"):
        verify_coalesced_copies(source, two.kernel, objects, banks, adapter, max_merges=1, **kwargs)


def test_adjacent_external_copies_do_not_invent_a_public_alias_contract():
    source, _, banks, adapter, kwargs = fixture()
    source = replace(source, body=(command("copy", 0, 0, 8, 2, 3), command("copy", 8, 8, 8, 2, 3)))
    assert coalesce_private_copies(source, (), banks, adapter, **kwargs) is None


def test_shared_typed_views_use_common_physical_alias_coordinates():
    source, objects, banks, adapter, kwargs = fixture()
    banks += (StorageBank("typed_view", "memory", 64, "bytes", start=16),)
    # Both original requests are individually disjoint, union is not.
    source = replace(
        source,
        body=(
            command("copy", 0, 16, 24, 2, 0),
            command("copy", 16, 24, 8, 0, 1),
            command("copy", 24, 32, 8, 0, 1),
            command("copy", 16, 0, 24, 0, 3),
        ),
    )
    objects = (TemporaryObject("scratch", "bytes", 16, 24),)
    assert coalesce_private_copies(source, objects, banks, adapter, **kwargs) is None


def test_complete_valid_async_route_retains_source_even_when_merge_group_itself_is_sync():
    source, objects, banks, adapter, kwargs = fixture()
    first = source.body[0]
    source = replace(source, body=(first, command("wait", 0, 0, 1), *source.body[1:]))

    def decode(call):
        if call.instr == "wait":
            return StorageEffects("token-one", completion="wait", waits=("one",))
        effect = project(call)
        return replace(effect, completion="asynchronous", token="one") if call == first else effect

    assert coalesce_private_copies(source, objects, banks, replace(adapter, project=decode), **kwargs) is None


@pytest.mark.parametrize(
    "field,value",
    [("max_request_bytes", 0), ("source_alignment", True), ("no_faults", 1), ("qualification_identity", "")],
)
def test_malformed_original_law_is_not_an_unavailable_capability(field, value):
    _, _, _, _, kwargs = fixture()
    with pytest.raises(ValueError):
        replace(kwargs["merge_law"], **{field: value})


def test_virtual_source_addresses_do_not_become_actual_resource_addresses():
    source, objects, banks, adapter, kwargs = fixture(base=512)
    assert coalesce_private_copies(source, objects, (replace(banks[0], capacity=64),), adapter, **kwargs) is None


def test_unrelated_virtual_backing_must_be_placed_before_actual_copy_coalescing():
    source, objects, banks, adapter, kwargs = fixture()
    source = replace(source, body=(*source.body, command("copy", 0, 256, 8, 2, 0)))
    objects += (TemporaryObject("unplaced", "bytes", 256, 8),)
    assert coalesce_private_copies(source, objects, banks, adapter, **kwargs) is None


def test_input_work_budget_fallback_does_not_construct_larger_output():
    source, objects, banks, adapter, kwargs = fixture()
    assert coalesce_private_copies(source, objects, banks, adapter, max_calls=3, **kwargs) is None


def test_verifier_catches_source_change_inside_decoding_before_accepting_actual_output():
    source, objects, banks, adapter, kwargs = fixture()
    result = coalesce_private_copies(source, objects, banks, adapter, **kwargs)

    def corrupt(call):
        object.__setattr__(kwargs["merge_law"], "no_faults", False)
        return project(call)

    with pytest.raises(ValueError, match="changed during a callback"):
        verify_coalesced_copies(source, result.kernel, objects, banks, replace(adapter, project=corrupt), **kwargs)
