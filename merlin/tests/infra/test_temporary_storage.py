"""Consumed storage assignments, physical aliases and actual completion waits."""

from dataclasses import replace

import pytest

from merlin.compile.scheduling.program_limits import AddressWindow, EncodedAccess
from merlin.compile.scheduling.temporary_storage import (
    StorageAdapter,
    StorageEffects,
    StorageOptimizationUnavailable,
    TemporaryObject,
    allocate_temporary_storage,
    verify_temporary_storage,
)
from merlin.sched.ir import Call, Const, Kernel, Loop, Ptr, TensorArg, evaluate
from merlin.semantic_compiler.allocation_types import Reservation, StorageBank


def instruction(op, address=0, size=0, offset=0, bank=0, token=7):
    args = (("bank", Const(bank)), ("address", Const(address)), ("size", Const(size)))
    if op in ("load", "save", "save_async"):
        args += (("host", Ptr("input" if op == "load" else "output", Const(offset))),)
    if op in ("save_async", "wait"):
        args += (("token", Const(token)),)
    return Call(op, args)


def project(call):
    a = dict(call.args)
    n, address, bank = (evaluate(a[key], {}) for key in ("size", "address", "bank"))
    space = ("byte_view", "wide_view")[bank]
    host = a.get("host")
    accesses = []
    completion, token, waits = "synchronous", None, ()
    if call.instr == "load":
        accesses = [EncodedAccess(host.tensor, evaluate(host.offset, {}), n), EncodedAccess(space, address, n, True)]
    elif call.instr in ("save", "save_async"):
        accesses = [EncodedAccess(space, address, n), EncodedAccess(host.tensor, evaluate(host.offset, {}), n, True)]
        if call.instr == "save_async":
            completion, token = "asynchronous", str(evaluate(a["token"], {}))
    elif call.instr == "increment":
        accesses = [EncodedAccess(space, address, n), EncodedAccess(space, address, n, True)]
    elif call.instr == "wait":
        completion, waits = "wait", (str(evaluate(a["token"], {})),)
    else:
        return None
    # This independently decoded key retains every nonaddress behavior of this
    # toy ISA. It does not consult a selected allocation or retained request.
    key = repr((call.instr, tuple((key, value) for key, value in call.args if key != "address")))
    return StorageEffects(key, tuple(accesses), completion, token, waits)


def rewrite(call, accesses):
    local = [access for access in accesses if access.space in ("byte_view", "wide_view")]
    address = local[0].start
    assert all(access.start == address for access in local)
    return replace(call, args=tuple((key, Const(address) if key == "address" else value) for key, value in call.args))


ADAPTER = StorageAdapter(project, rewrite, "neutral-byte-storage", "exact-byte-transfers-v1", "synthetic-ISA-only")


def fixture(*, size=16, separate=False, asynchronous=False, overlap=False):
    banks = (
        StorageBank("byte_view", "first_memory", 128, "bytes", 4),
        StorageBank("wide_view", "second_memory" if separate else "first_memory", 128, "bytes", 8),
    )
    objects = (TemporaryObject("a", "byte_view", 16, size, 8), TemporaryObject("b", "wide_view", 64, size, 8))
    load_a, load_b = instruction("load", 16, size), instruction("load", 64, size, size, 1)
    save_a = instruction("save_async" if asynchronous else "save", 16, size)
    save_b = instruction("save", 64, size, size, 1)
    if asynchronous:
        body = (load_a, save_a, load_b, instruction("wait"), save_b)
    elif overlap:
        body = (load_a, load_b, save_a, save_b)
    else:
        # Read the first object through the other typed view of its backing.
        alias_bank = 0 if separate else 1
        body = (
            load_a,
            instruction("increment", 16, size, bank=alias_bank),
            save_a,
            load_b,
            instruction("increment", 64, size, bank=1),
            save_b,
        )
    kernel = Kernel(
        "storage",
        (TensorArg("input", (size * 2,), "i8", "read"), TensorArg("output", (size * 2,), "i8", "write")),
        body,
    )
    model = dict(
        private_ownership=True,
        external_windows=(AddressWindow("input", 0, 2 * size), AddressWindow("output", 0, 2 * size)),
        reservations=(Reservation("byte_view", 0, 8), Reservation("wide_view", 120, 8)),
    )
    return kernel, objects, banks, model


def test_addresses_are_consumed_and_alias_views_share_the_actual_backing():
    kernel, objects, banks, model = fixture()
    result = allocate_temporary_storage(kernel, objects, banks, ADAPTER, **model)
    assert result is not None
    assert result.report["addresses"] == {"a": 8, "b": 8}
    assert result.report["temporary_high_water_bytes"] == {"first_memory": 24}
    assert result.report["physical_high_water_bytes"] == {"first_memory": 128}
    assert result.report["source_virtual_high_water_bytes"] == {"first_memory": 80}
    assert all(call.arg("address") == Const(8) for call in result.kernel.body)
    assert result.report["requested_read_bytes"] == result.report["requested_write_bytes"] == 96
    assert result.report["machine_emission"] == "not_established"
    assert verify_temporary_storage(kernel, result.kernel, objects, banks, ADAPTER, **model) == result.report


def test_completion_extends_lifetimes_past_the_last_issued_read():
    kernel, objects, banks, model = fixture(asynchronous=True)
    result = allocate_temporary_storage(kernel, objects, banks, ADAPTER, **model)
    assert result.report["lifetime_calls"] == {"a": [0, 3], "b": [2, 4]}
    assert result.report["addresses"] == {"a": 8, "b": 24}
    assert result.report["temporary_high_water_bytes"] == {"first_memory": 40}
    bad = replace(kernel, body=tuple(call for call in kernel.body if call.instr != "wait"))
    with pytest.raises(ValueError, match="returns before actual completion"):
        allocate_temporary_storage(bad, objects, banks, ADAPTER, **model)
    # Metadata alone cannot replace the missing executable wait.
    body = list(bad.body)
    body[1] = replace(body[1], produces="7")
    body[-1] = replace(body[-1], awaits=("7",))
    bad = replace(bad, body=tuple(body))
    with pytest.raises(StorageOptimizationUnavailable, match="decoded execution effects"):
        allocate_temporary_storage(bad, objects, banks, ADAPTER, **model)


def test_overlapping_live_objects_and_exact_capacity():
    kernel, objects, banks, model = fixture(overlap=True)
    compact_banks = tuple(replace(bank, capacity=40) for bank in banks)
    model = {**model, "reservations": (Reservation("wide_view", 0, 8),)}
    result = allocate_temporary_storage(kernel, objects, compact_banks, ADAPTER, **model)
    assert result.report["addresses"] == {"a": 8, "b": 24}
    assert (
        allocate_temporary_storage(
            kernel, objects, tuple(replace(bank, capacity=39) for bank in banks), ADAPTER, **model
        )
        is None
    )
    # Force a consistent but overlapping emitted assignment for b. The checker
    # must reconstruct the problem, rather than accepting adapter consistency.
    corrupted = replace(
        result.kernel,
        body=tuple(
            replace(call, args=tuple((key, Const(8) if key == "address" else value) for key, value in call.args))
            if index in (1, 3)
            else call
            for index, call in enumerate(result.kernel.body)
        ),
    )
    with pytest.raises(ValueError, match="simultaneously live physical views overlap"):
        verify_temporary_storage(kernel, corrupted, objects, compact_banks, ADAPTER, **model)


@pytest.mark.parametrize("mutation", ["missing", "offset", "size", "opcode", "external", "reservation"])
def test_actual_output_mutations_are_rejected(mutation):
    kernel, objects, banks, model = fixture()
    result = allocate_temporary_storage(kernel, objects, banks, ADAPTER, **model)
    body = list(result.kernel.body)
    if mutation == "missing":
        del body[1]
    elif mutation == "opcode":
        body[0] = replace(body[0], instr="save")
    elif mutation == "external":
        body[0] = replace(
            body[0],
            args=tuple((key, Ptr("input", Const(1)) if key == "host" else value) for key, value in body[0].args),
        )
    elif mutation in ("offset", "size"):
        operand = "address" if mutation == "offset" else "size"
        body[1] = replace(
            body[1], args=tuple((key, value + 1 if key == operand else value) for key, value in body[1].args)
        )
    else:
        model = {**model, "reservations": (Reservation("wide_view", 8, 16),)}
    with pytest.raises((ValueError, KeyError)):
        verify_temporary_storage(kernel, replace(result.kernel, body=tuple(body)), objects, banks, ADAPTER, **model)


@pytest.mark.parametrize("mutation", ["ownership", "partial_init", "unknown", "loop", "alias_roots", "window", "units"])
def test_unsupported_models_preserve_the_original_route_by_diagnostic(mutation):
    kernel, objects, banks, model = fixture()
    if mutation == "ownership":
        model = {**model, "private_ownership": False}
    elif mutation == "partial_init":
        kernel = replace(kernel, body=(instruction("load", 16, 8), *kernel.body[1:]))
    elif mutation == "unknown":
        kernel = replace(kernel, body=(replace(kernel.body[0], instr="opaque"), *kernel.body[1:]))
    elif mutation == "loop":
        kernel = replace(kernel, body=(Loop("i", 2, kernel.body),))
    elif mutation == "alias_roots":
        objects = (objects[0], replace(objects[1], source_base=16))
    elif mutation == "window":
        model = {**model, "external_windows": ()}
    else:
        banks = (replace(banks[0], unit="rows"), banks[1])
    with pytest.raises(ValueError):
        allocate_temporary_storage(kernel, objects, banks, ADAPTER, **model)


def test_pending_read_prevents_a_real_physical_overwrite_and_wait_order_matters():
    kernel, objects, banks, model = fixture(asynchronous=True)
    # Reusing a root's virtual address before its asynchronous reader finishes
    # is already invalid in the source program and must not be "optimized".
    bad = replace(kernel, body=(*kernel.body[:2], instruction("increment", 16, 16), *kernel.body[2:]))
    with pytest.raises(ValueError, match="actual completion was not established"):
        allocate_temporary_storage(bad, objects, banks, ADAPTER, **model)
    bad = replace(kernel, body=(instruction("wait"), *kernel.body))
    with pytest.raises(ValueError, match="missing or already completed token"):
        allocate_temporary_storage(bad, objects, banks, ADAPTER, **model)


def test_effect_list_order_cannot_prove_read_modify_write_initialization():
    kernel, objects, banks, model = fixture()

    def read_modify_write(call):
        effect = project(call)
        if call.instr == "load":
            write = effect.accesses[-1]
            effect = replace(effect, accesses=(write, replace(write, writes=False), *effect.accesses[:-1]))
        return effect

    with pytest.raises(StorageOptimizationUnavailable, match="also reads uninitialized"):
        allocate_temporary_storage(kernel, objects, banks, replace(ADAPTER, project=read_modify_write), **model)


def test_separate_memories_and_unavailable_addressable_capacity_are_not_interchangeable():
    kernel, objects, banks, model = fixture(separate=True)
    result = allocate_temporary_storage(kernel, objects, banks, ADAPTER, **model)
    assert result.report["addresses"] == {"a": 8, "b": 0}
    assert result.report["temporary_high_water_bytes"] == {"first_memory": 24, "second_memory": 16}
    # Excess room in the first memory cannot compensate for the second one.
    small = (replace(banks[0], capacity=1024), replace(banks[1], capacity=15))
    assert allocate_temporary_storage(kernel, objects, small, ADAPTER, **{**model, "reservations": ()}) is None


def test_actual_alias_view_bounds_participate_in_selection_before_addresses_are_emitted():
    kernel, objects, banks, model = fixture(overlap=True)
    # Both objects are assigned through the larger view, but actual accesses
    # to b use the short alias. Selecting a=0 first would strand b at 16.
    objects = (objects[0], replace(objects[1], bank="byte_view"))
    banks = (banks[0], replace(banks[1], capacity=16))
    result = allocate_temporary_storage(kernel, objects, banks, ADAPTER, **{**model, "reservations": ()})
    assert result is not None
    assert result.report["addresses"] == {"a": 16, "b": 0}


def test_actual_alias_alignment_is_respected_even_with_a_less_restrictive_primary_bank():
    kernel, objects, banks, model = fixture()
    objects = tuple(replace(obj, alignment=1) for obj in objects)
    result = allocate_temporary_storage(
        kernel, objects, banks, ADAPTER, **{**model, "reservations": (Reservation("byte_view", 0, 4),)}
    )
    assert result.report["addresses"] == {"a": 8, "b": 8}


@pytest.mark.parametrize(
    "size,separate,asynchronous", [(7, False, False), (16, False, False), (24, True, False), (16, False, True)]
)
def test_actual_emitted_c_matches_baseline_complete_bytes_and_observed_effects(tmp_path, size, separate, asynchronous):
    import ctypes
    import shutil
    import subprocess

    from merlin.sched.codegen import emit_c_function
    from merlin.sched.isa import InstrDef, InstructionSet, Operand

    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("native C toolchain unavailable")
    kernel, objects, banks, model = fixture(size=size, separate=separate, asynchronous=asynchronous)
    result = allocate_temporary_storage(kernel, objects, banks, ADAPTER, **model)
    definitions = {}
    for call in kernel.body:
        names = [name for name, _ in call.args]
        definitions[call.instr] = InstrDef(
            call.instr,
            tuple(Operand(name, "ptr" if name == "host" else "int") for name in names),
            lambda args, op=call.instr: "do_" + op + "(" + ",".join(args) + ");",
        )
    iset = InstructionSet(
        "neutral-storage-ISA", definitions, {"i8": "int8_t"}, "/* actual calls establish completion */"
    )
    source = f"""#include <stdint.h>
#include <string.h>
#include <assert.h>
static unsigned char memory[2][128];
uint64_t reads, writes, commands;
static unsigned char *ptr(int bank,int address) {{ return memory[bank && {int(separate)}]+address; }}
static void do_load(int bank,int address,int size,const void *host) {{
 commands++; reads+=size; writes+=size; memcpy(ptr(bank,address),host,size);
}}
static void do_increment(int bank,int address,int size) {{
 commands++; reads+=size; writes+=size;
 for(int i=0;i<size;i++) ptr(bank,address)[i]++;
}}
static void do_save(int bank,int address,int size,void *host) {{
 commands++; reads+=size; writes+=size; memcpy(host,ptr(bank,address),size);
}}
static int pending_bank,pending_address,pending_size,pending_token,active;
static void *pending_host;
static void do_save_async(int bank,int address,int size,void *host,int token) {{
 assert(!active); commands++; reads+=size; writes+=size; active=1;
 pending_bank=bank;pending_address=address;pending_size=size;pending_host=host;pending_token=token;
}}
static void do_wait(int bank,int address,int size,int token) {{
 assert(active && token==pending_token); commands++; active=0;
 memcpy(pending_host,ptr(pending_bank,pending_address),pending_size);
}}
"""
    for arm, program in (("baseline", kernel), ("optimized", result.kernel)):
        source += emit_c_function(program, iset, symbol=arm, index_bits=64)
        source += (
            f"void run_{arm}(int8_t *input,int8_t *output) {{ "
            f"reads=writes=commands=active=0; {arm}(input,output); assert(!active); }}\n"
        )
    path = tmp_path / "storage.c"
    path.write_text(source)
    library = tmp_path / f"storage_{size}_{separate}_{asynchronous}.so"
    subprocess.run(
        [cc, "-std=c11", "-O1", "-shared", "-fPIC", str(path), "-o", str(library)], check=True, capture_output=True
    )
    compiled = ctypes.CDLL(str(library))
    source_bytes = bytes((i * 29 + 203) % 256 for i in range(size * 2))
    inp = (ctypes.c_uint8 * (size * 2)).from_buffer_copy(source_bytes)
    expected = source_bytes if asynchronous else bytes((value + 1) % 256 for value in source_bytes)
    for arm in ("baseline", "optimized"):
        out = (ctypes.c_uint8 * (size * 2))(*([111] * (size * 2)))
        function = getattr(compiled, "run_" + arm)
        function.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        function(inp, out)
        assert bytes(out) == expected
        assert ctypes.c_uint64.in_dll(compiled, "reads").value == result.report["requested_read_bytes"]
        assert ctypes.c_uint64.in_dll(compiled, "writes").value == result.report["requested_write_bytes"]
        assert ctypes.c_uint64.in_dll(compiled, "commands").value == result.report["static_commands"]
