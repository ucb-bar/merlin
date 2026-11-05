"""Realized copy removal, overwrite barriers and allocation fallback."""

from dataclasses import replace

import pytest

from merlin.compile.scheduling.program_limits import AddressWindow, EncodedAccess
from merlin.compile.scheduling.spill_conformance import verify_device_spills, verify_storage_dataflow
from merlin.compile.scheduling.temporary_storage import (
    StorageAdapter,
    StorageEffects,
    StorageOptimizationUnavailable,
    TemporaryObject,
    allocate_temporary_storage,
)
from merlin.sched.ir import Call, Const, Kernel, TensorArg, evaluate
from merlin.semantic_compiler.allocation_types import Reservation, StorageBank

SPACES = ("bytes", "typed_view", "input", "output")


def command(op, source, destination, size, read_bank=0, write_bank=0):
    return Call(
        op,
        tuple(
            (key, Const(value))
            for key, value in zip(
                ("source", "destination", "size", "read_bank", "write_bank"),
                (source, destination, size, read_bank, write_bank),
                strict=True,
            )
        ),
    )


def project(call):
    if call.instr not in {"copy", "increment"}:
        return None
    args = {key: evaluate(value, {}) for key, value in call.args}
    accesses = (
        EncodedAccess(SPACES[args["read_bank"]], args["source"], args["size"]),
        EncodedAccess(SPACES[args["write_bank"]], args["destination"], args["size"], True),
    )
    return StorageEffects(
        repr((call.instr, accesses[0].space, accesses[1].space, args["size"])),
        accesses,
        "synchronous",
        exact_copy=call.instr == "copy",
    )


def rewrite(call, accesses):
    updates = {"source": Const(accesses[0].start), "destination": Const(accesses[1].start)}
    return replace(call, args=tuple((key, updates.get(key, value)) for key, value in call.args))


ADAPTER = StorageAdapter(project, rewrite, "neutral-byte-views", "wrap-u8-v1", "synthetic-ISA-only")


def fixture(width=8, *, separate=False, capacity=None):
    kernel = Kernel(
        "private_copies",
        (TensorArg("input", (2 * width,), "i8", "read"), TensorArg("output", (2 * width,), "i8", "write")),
        (
            command("copy", 0, 32, width, 2, 0),
            command("increment", 32, 32, width),
            command("copy", 32, 128, width, 0, 1),
            command("copy", 128, 0, width, 1, 3),
        ),
    )
    objects = (TemporaryObject("a", "bytes", 32, width), TemporaryObject("b", "typed_view", 128, width))
    capacity = 2 * width if capacity is None else capacity
    banks = (
        StorageBank("bytes", "memory", capacity, "bytes"),
        StorageBank("typed_view", "other" if separate else "memory", capacity, "bytes"),
    )
    model = dict(
        private_ownership=True,
        external_windows=(AddressWindow("input", 0, 2 * width), AddressWindow("output", 0, 2 * width)),
    )
    return kernel, objects, banks, model


@pytest.mark.parametrize("width", [3, 8, 17])
def test_ordinary_allocator_removes_actual_private_copy_and_replans(width):
    kernel, objects, banks, model = fixture(width)
    baseline = allocate_temporary_storage(kernel, objects, banks, ADAPTER, eliminate_copies=False, **model)
    optimized = allocate_temporary_storage(kernel, objects, banks, ADAPTER, **model)
    proof = optimized.report["copy_elimination"]
    assert len(baseline.kernel.body) == 4
    assert len(optimized.kernel.body) == 3
    assert optimized.report["addresses"] == {"a": 0}
    assert proof["eliminated_exact_copies"] == 1
    assert proof["original_requested_read_bytes"] == proof["original_requested_write_bytes"] == 4 * width
    assert proof["requested_read_bytes"] == proof["requested_write_bytes"] == 3 * width
    assert proof["machine_emission"] == proof["execution_qualification"] == "not_established"
    assert optimized.report["repair_rounds"][0]["before_sha256"] == kernel.digest()
    assert "liveness" in optimized.report["repair_rounds"][0]["invalidated"]
    assert proof == verify_storage_dataflow(
        kernel,
        optimized.kernel,
        objects,
        banks,
        ADAPTER,
        eliminate_private_copies=True,
        max_inserted_copies=8,
        **model,
    )


def test_forwarding_can_make_an_otherwise_infeasible_allocation_fit():
    kernel, objects, banks, model = fixture(capacity=8)
    assert allocate_temporary_storage(kernel, objects, banks, ADAPTER, eliminate_copies=False, **model) is None
    assert allocate_temporary_storage(kernel, objects, banks, ADAPTER, **model) is not None


@pytest.mark.parametrize("write_target", [32, 128])
def test_source_or_destination_overwrites_keep_the_copy(write_target):
    kernel, objects, banks, model = fixture()
    bank = 0 if write_target == 32 else 1
    kernel = replace(
        kernel, body=(*kernel.body[:3], command("increment", write_target, write_target, 8, bank, bank), kernel.body[3])
    )
    result = allocate_temporary_storage(kernel, objects, banks, ADAPTER, **model)
    assert "copy_elimination" not in result.report
    assert len(result.kernel.body) == 5


def test_mutated_forwarding_across_an_overwrite_fails_actual_dataflow_check():
    kernel, objects, banks, model = fixture()
    original = replace(kernel, body=(*kernel.body[:3], command("increment", 32, 32, 8), kernel.body[3]))
    broken = replace(kernel, body=(*kernel.body[:2], command("increment", 32, 32, 8), command("copy", 32, 0, 8, 1, 3)))
    result = allocate_temporary_storage(broken, objects[:1], banks, ADAPTER, eliminate_copies=False, **model)
    with pytest.raises(ValueError, match="ordered original semantics|overwritten storage"):
        verify_storage_dataflow(
            original, result.kernel, objects, banks, ADAPTER, eliminate_private_copies=True, **model
        )


def test_write_after_the_last_forwarded_read_is_legal():
    kernel, objects, banks, model = fixture()
    kernel = replace(kernel, body=(*kernel.body, command("increment", 32, 32, 8)))
    assert (
        allocate_temporary_storage(kernel, objects, banks, ADAPTER, **model).report["copy_elimination"][
            "eliminated_exact_copies"
        ]
        == 1
    )


def test_separate_backings_do_not_infer_a_cross_space_handoff():
    kernel, objects, banks, model = fixture(separate=True)
    result = allocate_temporary_storage(kernel, objects, banks, ADAPTER, **model)
    assert len(result.kernel.body) == 4
    assert "copy_elimination" not in result.report


def test_stronger_destination_alignment_is_preserved_on_the_source_backing():
    kernel, objects, banks, model = fixture(capacity=32)
    objects = (objects[0], replace(objects[1], alignment=16))
    model["reservations"] = (Reservation("bytes", 0, 1),)
    result = allocate_temporary_storage(kernel, objects, banks, ADAPTER, **model)
    assert result.report["addresses"] == {"a": 16}


def test_failed_forwarding_allocation_selects_original_composition_and_replans():
    kernel, _, banks, model = fixture()
    banks = tuple(replace(bank, capacity=24) for bank in banks)
    objects = (
        TemporaryObject("large_source", "bytes", 0, 16),
        TemporaryObject("small_copy", "bytes", 32, 8),
        TemporaryObject("later", "bytes", 64, 16),
    )
    kernel = replace(
        kernel,
        body=(
            command("copy", 0, 0, 16, 2, 0),
            command("copy", 0, 32, 8),
            command("copy", 0, 64, 16, 2, 0),
            command("copy", 64, 0, 16, 0, 3),
            command("copy", 32, 8, 8, 0, 3),
        ),
    )
    result = allocate_temporary_storage(kernel, objects, banks, ADAPTER, **model)
    assert result is not None
    assert result.report["copy_elimination"]["status"] == "allocation_fallback"
    assert len(result.kernel.body) == 5
    assert result.report["lifetime_calls"] == {"large_source": [0, 1], "small_copy": [1, 4], "later": [2, 3]}


def test_copy_chains_are_removed_from_actual_output():
    kernel, objects, banks, model = fixture(capacity=24)
    objects += (TemporaryObject("c", "typed_view", 160, 8),)
    kernel = replace(
        kernel, body=(*kernel.body[:3], command("copy", 128, 160, 8, 1, 1), command("copy", 160, 0, 8, 1, 3))
    )
    result = allocate_temporary_storage(kernel, objects, banks, ADAPTER, **model)
    assert len(result.kernel.body) == 3
    assert result.report["copy_elimination"]["eliminated_exact_copies"] == 2


def test_subrange_forwarding_preserves_offsets_and_nonoverlapping_source_updates():
    kernel, _, banks, model = fixture(capacity=32)
    objects = (TemporaryObject("source", "bytes", 0, 16), TemporaryObject("part", "typed_view", 32, 8))
    kernel = replace(
        kernel,
        body=(
            command("copy", 0, 0, 16, 2, 0),
            command("copy", 8, 32, 8, 0, 1),
            command("increment", 0, 0, 8),
            command("copy", 35, 2, 3, 1, 3),
        ),
    )
    result = allocate_temporary_storage(kernel, objects, banks, ADAPTER, **model)
    assert len(result.kernel.body) == 3
    assert project(result.kernel.body[-1]).accesses[0].start == result.report["addresses"]["source"] + 11
    assert result.report["copy_elimination"]["eliminated_exact_copies"] == 1


def test_inexact_transforms_are_not_treated_as_byte_copies():
    kernel, objects, banks, model = fixture()

    def decode(call):
        effect = project(call)
        if all(access.space in {"bytes", "typed_view"} for access in effect.accesses):
            return replace(effect, exact_copy=False)
        return effect

    result = allocate_temporary_storage(kernel, objects, banks, replace(ADAPTER, project=decode), **model)
    assert len(result.kernel.body) == 4


@pytest.mark.parametrize("object_alignment,bank_alignment", [(8, 1), (1, 8)])
def test_destination_alignment_can_refuse_an_unaligned_source_subrange(object_alignment, bank_alignment):
    kernel, _, banks, model = fixture(capacity=32)
    objects = (
        TemporaryObject("source", "bytes", 0, 16),
        TemporaryObject("part", "typed_view", 32, 8, object_alignment),
    )
    banks = (banks[0], replace(banks[1], alignment=bank_alignment))
    kernel = replace(
        kernel,
        body=(
            command("copy", 0, 0, 16, 2, 0),
            command("copy", 1, 32, 8, 0, 1),
            command("copy", 32, 0, 8, 1, 3),
        ),
    )
    result = allocate_temporary_storage(kernel, objects, banks, ADAPTER, **model)
    assert len(result.kernel.body) == 3
    assert "copy_elimination" not in result.report


def test_corrupt_rewriter_cannot_change_numerical_semantics():
    kernel, objects, banks, model = fixture()

    def broken(call, accesses):
        return replace(rewrite(call, accesses), instr="increment")

    with pytest.raises(ValueError, match="decoded instruction semantics"):
        allocate_temporary_storage(kernel, objects, banks, replace(ADAPTER, rewrite=broken), **model)


def test_dataflow_checker_rejects_new_reservations_and_unknown_copies():
    kernel, objects, banks, model = fixture()
    result = allocate_temporary_storage(kernel, objects, banks, ADAPTER, **model)
    with pytest.raises(ValueError, match="address window"):
        verify_storage_dataflow(
            kernel,
            result.kernel,
            objects,
            banks,
            ADAPTER,
            eliminate_private_copies=True,
            reservations=(Reservation("bytes", 0, 1),),
            **model,
        )
    # Removal is selected explicitly; ordinary spill verification cannot
    # silently accept an omitted original operation.
    with pytest.raises(ValueError):
        verify_storage_dataflow(kernel, result.kernel, objects, banks, ADAPTER, **model)
    with pytest.raises(ValueError, match="spill-only"):
        verify_device_spills(kernel, result.kernel, objects, banks, ADAPTER, (), eliminate_private_copies=True, **model)


def test_symbolic_copy_extents_do_not_allocate_per_byte_state():
    width = 1 << 35
    kernel, _, banks, model = fixture(width)
    kernel = replace(
        kernel,
        body=(
            command("copy", 0, 0, width, 2, 0),
            command("copy", 0, width, width, 0, 1),
            command("copy", width, 0, width, 1, 3),
        ),
    )
    objects = (TemporaryObject("source", "bytes", 0, width), TemporaryObject("copy", "typed_view", width, width))
    result = allocate_temporary_storage(kernel, objects, banks, ADAPTER, **model)
    assert len(result.kernel.body) == 2
    assert result.report["copy_elimination"]["requested_read_bytes"] == 2 * width


@pytest.mark.parametrize(
    "mutation", ["wrong_read", "missing_arithmetic", "wrong_arithmetic", "missing_output", "wrong_output"]
)
def test_independent_checker_rejects_corrupted_realization(mutation):
    kernel, objects, banks, model = fixture()
    result = allocate_temporary_storage(kernel, objects, banks, ADAPTER, **model)
    calls = list(result.kernel.body)
    if mutation == "wrong_read":
        calls[-1] = command("copy", 8, 0, 8, 1, 3)
    elif mutation == "missing_arithmetic":
        del calls[1]
    elif mutation == "wrong_arithmetic":
        calls[1] = replace(calls[1], instr="copy")
    elif mutation == "missing_output":
        calls.pop()
    else:
        calls[-1] = command("copy", 0, 8, 8, 1, 3)
    with pytest.raises(ValueError):
        verify_storage_dataflow(
            kernel,
            replace(result.kernel, body=tuple(calls)),
            objects,
            banks,
            ADAPTER,
            eliminate_private_copies=True,
            **model,
        )


def test_unknown_effects_and_ownership_remain_barriers():
    kernel, objects, banks, model = fixture()
    with pytest.raises(StorageOptimizationUnavailable, match="ownership"):
        allocate_temporary_storage(kernel, objects, banks, ADAPTER, **{**model, "private_ownership": False})
    with pytest.raises(StorageOptimizationUnavailable, match="unknown instruction"):
        allocate_temporary_storage(
            replace(kernel, body=(Call("unknown", ()), *kernel.body)), objects, banks, ADAPTER, **model
        )


def test_valid_async_completion_keeps_copies_without_inventing_an_async_handoff():
    kernel, objects, banks, model = fixture()
    kernel = replace(kernel, body=(kernel.body[0], Call("wait", ()), *kernel.body[1:]))

    def decode(call):
        if call.instr == "wait":
            return StorageEffects("wait", completion="wait", waits=("ready",))
        effect = project(call)
        return (
            replace(effect, completion="asynchronous", token="ready") if effect.accesses[0].space == "input" else effect
        )

    result = allocate_temporary_storage(kernel, objects, banks, replace(ADAPTER, project=decode), **model)
    assert len(result.kernel.body) == 5
    assert "copy_elimination" not in result.report


@pytest.mark.parametrize("width,faulty_copy", [(3, False), (8, False), (17, False), (8, True)])
def test_actual_native_baseline_and_optimized_bytes_and_traffic(tmp_path, width, faulty_copy):
    import ctypes
    import shutil
    import subprocess

    from merlin.sched.codegen import emit_c_function
    from merlin.sched.isa import InstrDef, InstructionSet, Operand

    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("native C toolchain unavailable")
    kernel, objects, banks, model = fixture(width)
    baseline = allocate_temporary_storage(kernel, objects, banks, ADAPTER, eliminate_copies=False, **model)
    optimized = allocate_temporary_storage(kernel, objects, banks, ADAPTER, **model)
    definitions = {
        op: InstrDef(
            op,
            tuple(Operand(name, "int") for name, _ in kernel.body[0].args),
            lambda args, op=op: "do_" + op + "(" + ",".join(args) + ");",
        )
        for op in ("copy", "increment")
    }
    isa = InstructionSet("neutral-copy-ISA", definitions, {"i8": "int8_t"}, "/* synchronous */")
    source = f"""#include <stdint.h>
#include <string.h>
#include <assert.h>
static uint8_t local[{2 * width}], *input_ptr, *output_ptr;
uint64_t reads,writes,commands;
static uint8_t *space(int bank,int offset,int size) {{
 assert(offset>=0 && offset+size<={2 * width});
 return (bank<2 ? local : bank==2 ? input_ptr : output_ptr)+offset;
}}
static void do_copy(int source,int destination,int size,int read_bank,int write_bank) {{
 commands++; reads+=size; writes+=size;
 memcpy(space(write_bank,destination,size),space(read_bank,source,size),size);
 if({int(faulty_copy)} && read_bank<2 && write_bank<2) space(write_bank,destination,size)[0]++;
}}
static void do_increment(int source,int destination,int size,int read_bank,int write_bank) {{
 commands++; reads+=size; writes+=size;
 for(int i=0;i<size;i++) space(write_bank,destination,size)[i]=space(read_bank,source,size)[i]+1;
}}
"""
    for name, result in (("baseline", baseline), ("optimized", optimized)):
        source += emit_c_function(result.kernel, isa, symbol=name, index_bits=64)
        source += (
            f"void run_{name}(int8_t *input,int8_t *output) {{input_ptr=(uint8_t*)input;"
            f"output_ptr=(uint8_t*)output;reads=writes=commands=0;{name}(input,output);}}\n"
        )
    path, library = tmp_path / "copies.c", tmp_path / f"copies_{width}_{faulty_copy}.so"
    path.write_text(source)
    subprocess.run(
        [cc, "-std=c11", "-O1", "-shared", "-fPIC", str(path), "-o", str(library)], check=True, capture_output=True
    )
    compiled = ctypes.CDLL(str(library))
    data = bytes((i * 43 + 251) % 256 for i in range(2 * width))
    expected = bytes((byte + 1) % 256 for byte in data[:width]) + bytes([91] * width)
    for name, count in (("baseline", 4), ("optimized", 3)):
        inputs = (ctypes.c_uint8 * (2 * width)).from_buffer_copy(data)
        outputs = (ctypes.c_uint8 * (2 * width))(*([91] * (2 * width)))
        getattr(compiled, "run_" + name)(inputs, outputs)
        if faulty_copy and name == "baseline":
            # A lying copy declaration does not qualify its implementation.
            # Independent reference execution exposes the missing contract.
            assert bytes(outputs) != expected
        else:
            assert bytes(outputs) == expected
        assert bytes(inputs) == data
        assert ctypes.c_uint64.in_dll(compiled, "commands").value == count
        assert ctypes.c_uint64.in_dll(compiled, "reads").value == count * width
        assert ctypes.c_uint64.in_dll(compiled, "writes").value == count * width
