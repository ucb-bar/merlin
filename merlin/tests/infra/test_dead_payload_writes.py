"""Actual omission checks and neutral native byte engines, not target laws."""

import hashlib
import json
import shutil
import struct
from dataclasses import replace
from pathlib import Path

import pytest
from test_private_copy_elimination import SPACES, command
from test_selected_storage_contracts import native  # noqa: F401

from merlin.compile.scheduling.dead_write_check import WriteDeletionLaw, verify_dead_writes
from merlin.compile.scheduling.dead_writes import eliminate_dead_writes
from merlin.compile.scheduling.program_limits import AddressWindow, EncodedAccess
from merlin.compile.scheduling.temporary_storage import StorageAdapter, StorageEffects, TemporaryObject
from merlin.sched.ir import Const, Kernel, TensorArg, evaluate
from merlin.semantic_compiler.allocation_types import StorageBank


def project(call):
    if call.instr not in {"copy", "increment", "observe", "initialize", "barrier"}:
        return None
    if call.instr == "barrier":
        return StorageEffects("opaque-observation", completion="synchronous")
    a = dict((k, evaluate(v, {})) for k, v in call.args)
    read = EncodedAccess(SPACES[a["read_bank"]], a["source"], a["size"])
    write = EncodedAccess(SPACES[a["write_bank"]], a["destination"], a["size"], True)
    return StorageEffects(
        str((call.instr, a.get("mode", 0))),
        (write,) if call.instr == "initialize" else (read, write),
        "synchronous",
        exact_copy=call.instr == "copy",
    )


def fixture(width=7, *, separate=False):
    size = 2 * width
    source = Kernel(
        "overwritten_private_payload",
        (
            TensorArg("input", (4 * width + 8,), "i8", "read"),
            TensorArg("output", (4 * width + 8,), "i8", "write"),
        ),
        (
            command("copy", 0, 32, size, 2, 0),
            command("copy", size, 32, width, 2, 1 if not separate else 0),
            command("copy", size + width, 32 + width, width, 2, 0),
            command("increment", 32, 32, size),
            command("copy", 32, 96, size, 0, 1),
            command("copy", 96, 0, size, 1, 3),
            command("copy", 0, size, size, 2, 3),
        ),
    )
    objects = (TemporaryObject("a", "bytes", 32, size), TemporaryObject("b", "typed_view", 96, size))
    banks = (
        StorageBank("bytes", "first", 192, "bytes"),
        StorageBank("typed_view", "second" if separate else "first", 192, "bytes"),
    )
    adapter = StorageAdapter(
        project,
        lambda call, accesses: pytest.fail("deletion rewrote surviving operands"),
        "original-separate-memory-v1" if separate else "original-shared-views-v1",
        "exact-byte-modulo-v1",
        "neutral-C-projection-v1",
    )
    law = WriteDeletionLaw(
        adapter.configuration_identity,
        adapter.numerical_policy_identity,
        adapter.qualification_identity,
        "independently-reviewed-neutral-delete-v1",
        (("copy", str(("copy", 0))), ("initialize", str(("initialize", 0)))),
        no_faults=True,
        no_extra_observations=True,
        write_removal_preserves_other_effects=True,
    )
    kwargs = dict(
        private_ownership=True,
        deletion_law=law,
        external_windows=(AddressWindow("input", 0, 4 * width + 8), AddressWindow("output", 0, 4 * width + 8)),
    )
    return source, objects, banks, adapter, kwargs


def changed(call, **values):
    return replace(call, args=tuple((k, Const(values.get(k, evaluate(v, {})))) for k, v in call.args))


@pytest.mark.parametrize("width,separate", [(3, False), (7, True), (13, False)])
def test_complete_overwrite_union_across_qualified_alias_views(width, separate):
    source, objects, banks, adapter, kwargs = fixture(width, separate=separate)
    result = eliminate_dead_writes(source, objects, banks, adapter, **kwargs)
    assert result.kernel == replace(source, body=source.body[1:])
    assert result.report["omitted_source_indices"] == [0]
    assert result.report == verify_dead_writes(source, result.kernel, objects, banks, adapter, **kwargs)
    assert result.report["original_requested_write_bytes"] == 12 * width
    assert result.report["requested_write_bytes"] == result.report["requested_read_bytes"] == 10 * width
    assert result.report["machine_emission"] == result.report["execution_qualification"] == "not_established"
    assert result.report["timing"] == "UNKNOWN"


def test_qualified_write_only_initialization_can_be_deleted_without_a_copy_inference():
    source, objects, banks, adapter, kwargs = fixture()
    source = replace(source, body=(replace(source.body[0], instr="initialize"), *source.body[1:]))
    result = eliminate_dead_writes(source, objects, banks, adapter, **kwargs)
    assert result.report["omitted_source_indices"] == [0]
    assert result.report["original_requested_read_bytes"] == result.report["requested_read_bytes"]


@pytest.mark.parametrize("premise", ["no_faults", "no_extra_observations", "write_removal_preserves_other_effects"])
def test_exact_copy_and_rtl_geometry_do_not_supply_deletion_permission(premise):
    source, objects, banks, adapter, kwargs = fixture()
    kwargs["deletion_law"] = replace(kwargs["deletion_law"], **{premise: False})
    assert eliminate_dead_writes(source, objects, banks, adapter, **kwargs) is None
    with pytest.raises(ValueError, match="original fault/observation/delete law"):
        verify_dead_writes(source, replace(source, body=source.body[1:]), objects, banks, adapter, **kwargs)


@pytest.mark.parametrize(
    "identity", ["configuration_identity", "numerical_policy_identity", "adapter_qualification_identity"]
)
def test_law_cannot_transfer_to_different_original_selection(identity):
    source, objects, banks, adapter, kwargs = fixture()
    kwargs["deletion_law"] = replace(kwargs["deletion_law"], **{identity: "another-original"})
    assert eliminate_dead_writes(source, objects, banks, adapter, **kwargs) is None


@pytest.mark.parametrize("barrier", ["partial", "read", "rmw", "view_read", "observer", "unknown", "async"])
def test_partial_overwrite_read_first_observers_and_unknowns_keep_original(barrier):
    source, objects, banks, adapter, kwargs = fixture()
    calls = list(source.body)
    if barrier == "partial":
        calls[2] = changed(calls[2], size=6)
    elif barrier in {"read", "view_read", "rmw"}:
        calls.insert(
            1,
            command(
                "increment" if barrier == "rmw" else "observe",
                32,
                32 if barrier == "rmw" else 0,
                14,
                1 if barrier == "view_read" else 0,
                0 if barrier == "rmw" else 3,
            ),
        )
    elif barrier in {"observer", "unknown"}:
        calls.insert(1, command("barrier" if barrier == "observer" else "opaque", 0, 0, 1))
    else:
        original = adapter.project
        adapter = replace(
            adapter,
            project=lambda call: (
                replace(original(call), completion="asynchronous", token="one")
                if call == source.body[0]
                else original(call)
            ),
        )
    source = replace(source, body=tuple(calls))
    assert eliminate_dead_writes(source, objects, banks, adapter, **kwargs) is None
    if barrier not in {"unknown", "async"}:
        with pytest.raises(ValueError):
            verify_dead_writes(source, replace(source, body=source.body[1:]), objects, banks, adapter, **kwargs)


@pytest.mark.parametrize("mutation", ["address", "partial", "mode", "read", "extra", "boundary", "trailing"])
def test_checker_reconstructs_actual_omissions_without_trusting_a_plan(mutation):
    source, objects, banks, adapter, kwargs = fixture()
    realized = eliminate_dead_writes(source, objects, banks, adapter, **kwargs).kernel
    calls = list(realized.body)
    if mutation == "address":
        calls[0] = changed(calls[0], destination=33)
    elif mutation == "partial":
        calls[0] = changed(calls[0], size=6)
    elif mutation == "mode":
        calls[0] = replace(calls[0], args=(*calls[0].args, ("mode", Const(1))))
    elif mutation == "read":
        calls.pop(1)  # Second half remains different when the actual RMW reads it.
    elif mutation == "extra":
        calls.insert(0, source.body[-1])
    elif mutation == "trailing":
        calls.pop()
    else:
        realized = replace(realized, name="another-entry")
    with pytest.raises(ValueError):
        verify_dead_writes(source, replace(realized, body=tuple(calls)), objects, banks, adapter, **kwargs)


@pytest.mark.parametrize(
    "constraint", ["no_law", "disabled", "ownership", "capacity", "view_start", "window", "virtual", "budget"]
)
def test_missing_or_unproved_physical_and_work_bounds_keep_original(constraint):
    source, objects, banks, adapter, kwargs = fixture()
    if constraint == "no_law":
        kwargs["deletion_law"] = None
    elif constraint == "disabled":
        kwargs["max_deletions"] = 0
    elif constraint == "ownership":
        kwargs["private_ownership"] = False
    elif constraint == "capacity":
        banks = (replace(banks[0], capacity=40), banks[1])
    elif constraint == "view_start":
        banks = (banks[0], replace(banks[1], start=40))
    elif constraint == "window":
        kwargs["external_windows"] = kwargs["external_windows"][:1]
    elif constraint == "virtual":
        source = replace(
            source, body=tuple(changed(c, destination=400) if i == 0 else c for i, c in enumerate(source.body))
        )
    else:
        kwargs["max_scan_steps"] = 3
    assert eliminate_dead_writes(source, objects, banks, adapter, **kwargs) is None


@pytest.mark.parametrize("mutation", ["law", "bank", "object", "callback"])
def test_original_mutation_during_projection_refuses_instead_of_using_stale_analysis(mutation):
    source, objects, banks, adapter, kwargs = fixture()
    original = adapter.project

    def mutate(call):
        result = original(call)
        if mutation == "law":
            object.__setattr__(kwargs["deletion_law"], "qualification_identity", "changed")
        elif mutation == "bank":
            object.__setattr__(banks[0], "capacity", 180)
        elif mutation == "object":
            object.__setattr__(objects[0], "alignment", 2)
        else:
            object.__setattr__(adapter, "project", project)
        return result

    adapter = replace(adapter, project=mutate)
    with pytest.raises(ValueError, match="changed during a callback"):
        eliminate_dead_writes(source, objects, banks, adapter, **kwargs)


@pytest.mark.parametrize("width,separate,word_format", [(3, False, ">BBBIII"), (13, True, "<BBBQQQ")])
def test_native_decoded_programs_preserve_full_state_inputs_and_canaries(
    tmp_path,
    native,  # noqa: F811
    width,
    separate,
    word_format,  # noqa: F811
):
    source, objects, banks, adapter, kwargs = fixture(width, separate=separate)
    result = eliminate_dead_writes(source, objects, banks, adapter, **kwargs)
    word = struct.Struct(word_format)
    arms = []
    for name, kernel in (("original", source), ("deleted", result.kernel)):
        image = b"".join(
            word.pack(
                0 if call.instr == "copy" else 1,
                *(evaluate(call.arg(key), {}) for key in ("read_bank", "write_bank", "source", "destination", "size")),
            )
            for call in kernel.body
        )
        (tmp_path / (name + ".bin")).write_bytes(image)
        # Independent actual-byte decoding is fed to the separate checker.
        calls = tuple(
            command("copy" if op == 0 else "increment", s, d, n, r, w) for op, r, w, s, d, n in word.iter_unpack(image)
        )
        actual = replace(kernel, body=calls)
        verify_dead_writes(source, actual, objects, banks, adapter, **kwargs)
        state, history, counts_rows = bytes((i * 19 + 251) % 256 for i in range(2 * width)), [], []
        raw_calls = []
        for epoch in range(3):
            raw = bytes((i * 31 + 253 + epoch) % 256 for i in range(2 * width))
            packed = raw + state + bytes([73] * 8)
            data, output = bytearray(packed), bytearray([91] * len(packed))
            first, second = bytearray([173] * 192), bytearray([149] * 192)
            counts = native(
                image,
                first,
                second,
                data,
                output,
                width=4 if word_format.endswith("I") else 8,
                big=word_format[0] == ">",
                separate=separate,
            )
            expected = bytes((x + 1) % 256 for x in state) + raw + bytes([91] * 8)
            assert bytes(output) == expected and bytes(data) == packed
            for memory, windows, guard in (
                (first, [(32, 32 + 2 * width)] + ([] if separate else [(96, 96 + 2 * width)]), 173),
                (second, [(96, 96 + 2 * width)] if separate else [], 149),
            ):
                assert all(value == guard for i, value in enumerate(memory) if not any(a <= i < b for a, b in windows))
            assert counts[:3] == (
                len(kernel.body),
                (12 if name == "original" else 10) * width,
                (12 if name == "original" else 10) * width,
            )
            raw_calls.append(
                dict(
                    epoch=epoch,
                    input_before=packed.hex(),
                    input_after=bytes(data).hex(),
                    state_before=state.hex(),
                    output=bytes(output).hex(),
                    first=bytes(first).hex(),
                    second=bytes(second).hex(),
                )
            )
            history.append(list(output))
            counts_rows.append(counts)
            state = bytes(output[: 2 * width])
        arms.append(
            dict(
                name=name,
                complete_outputs=history,
                counts=counts_rows,
                raw_calls=raw_calls,
                image_sha256=hashlib.sha256(image).hexdigest(),
            )
        )
    assert arms[0]["complete_outputs"] == arms[1]["complete_outputs"]
    compiler = Path(shutil.which("cc"))
    implementation = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (tmp_path / "machine.c", tmp_path / "machine.so")
    }
    (tmp_path / "pair.json").write_text(
        json.dumps(
            dict(
                scope="neutral byte engines; no target qualification/timing",
                implementation=implementation,
                width=width,
                separate=separate,
                word_format=word_format,
                arms=arms,
                report=result.report,
                compiler=dict(path=str(compiler), sha256=hashlib.sha256(compiler.read_bytes()).hexdigest()),
            ),
            indent=2,
        )
    )


def test_repeated_identical_initializers_use_actual_survivor_correspondence():
    source, objects, banks, adapter, kwargs = fixture()
    source = replace(source, body=(source.body[0], source.body[0], source.body[0], *source.body[1:]))
    result = eliminate_dead_writes(source, objects, banks, adapter, **kwargs)
    assert result.kernel.body == source.body[3:]
    assert result.report["deleted_calls"] == 3


def test_implicit_read_is_checked_before_all_same_instruction_writes():
    source, objects, banks, adapter, kwargs = fixture()
    original = adapter.project

    def read_before_write(call):
        effect = original(call)
        if call == source.body[1]:
            return replace(effect, exact_copy=False, accesses=(*effect.accesses, EncodedAccess("bytes", 32, 14)))
        return effect

    adapter = replace(adapter, project=read_before_write)
    assert eliminate_dead_writes(source, objects, banks, adapter, **kwargs) is None
    with pytest.raises(ValueError, match="read before complete overwrite"):
        verify_dead_writes(source, replace(source, body=source.body[1:]), objects, banks, adapter, **kwargs)


@pytest.mark.parametrize("limit", ["max_calls", "max_objects", "max_scan_steps", "max_deletions"])
def test_invalid_typed_work_budgets_refuse(limit):
    source, objects, banks, adapter, kwargs = fixture()
    with pytest.raises(ValueError, match="work bounds|deletion count"):
        eliminate_dead_writes(source, objects, banks, adapter, **(kwargs | {limit: False}))


def test_new_semantic_mode_does_not_inherit_an_instruction_spelling_law():
    source, objects, banks, adapter, kwargs = fixture()
    source = replace(
        source, body=(replace(source.body[0], args=(*source.body[0].args, ("mode", Const(1)))), *source.body[1:])
    )
    assert eliminate_dead_writes(source, objects, banks, adapter, **kwargs) is None
    with pytest.raises(ValueError, match="qualified private payload deletion law"):
        verify_dead_writes(source, replace(source, body=source.body[1:]), objects, banks, adapter, **kwargs)


def test_exhausted_source_call_ceiling_does_not_scan_or_invoke_provider():
    source, objects, banks, adapter, kwargs = fixture()
    adapter = replace(adapter, project=lambda call: pytest.fail("over-budget source invoked provider"))
    assert eliminate_dead_writes(source, objects, banks, adapter, max_calls=1, **kwargs) is None


def test_mutated_original_source_during_projection_refuses():
    source, objects, banks, adapter, kwargs = fixture()
    original = adapter.project

    def mutate(call):
        effect = original(call)
        object.__setattr__(source, "name", "changed-original-entry")
        return effect

    adapter = replace(adapter, project=mutate)
    with pytest.raises(ValueError, match="changed during a callback"):
        eliminate_dead_writes(source, objects, banks, adapter, **kwargs)


def test_exhausted_declared_storage_ceiling_keeps_source_before_provider():
    source, objects, banks, adapter, kwargs = fixture()
    adapter = replace(adapter, project=lambda call: pytest.fail("over-budget storage invoked provider"))
    assert eliminate_dead_writes(source, objects, banks, adapter, max_objects=1, **kwargs) is None
