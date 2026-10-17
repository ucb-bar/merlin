"""Actual exact transfers, bounded replanning and independent complete outputs."""

from dataclasses import replace

import pytest

from merlin.compile.scheduling.device_spills import SpillClass, verify_device_spills
from merlin.compile.scheduling.program_limits import AddressWindow, EncodedAccess
from merlin.compile.scheduling.temporary_storage import (
    StorageAdapter,
    StorageEffects,
    StorageOptimizationUnavailable,
    TemporaryObject,
    allocate_temporary_storage,
)
from merlin.sched.ir import Call, Const, Kernel, Ptr, TensorArg, evaluate
from merlin.semantic_compiler.allocation_types import Reservation, StorageBank

SPACES = ("registers", "saved")


def memory_call(operation, address, size, host_offset=0):
    arguments = (("address", Const(address)), ("size", Const(size)))
    if operation in ("input", "output"):
        arguments += (("host", Ptr(operation, Const(host_offset))),)
    return Call(operation, arguments)


def emit_copy(source, destination):
    return Call(
        "transfer",
        (
            ("source_bank", Const(SPACES.index(source.space))),
            ("source", Const(source.start)),
            ("destination_bank", Const(SPACES.index(destination.space))),
            ("destination", Const(destination.start)),
            ("size", Const(source.size)),
        ),
    )


def project(call):
    args = dict(call.args)
    size = evaluate(args["size"], {})
    if call.instr == "transfer":
        read = EncodedAccess(SPACES[evaluate(args["source_bank"], {})], evaluate(args["source"], {}), size)
        write = EncodedAccess(
            SPACES[evaluate(args["destination_bank"], {})], evaluate(args["destination"], {}), size, True
        )
        return StorageEffects(
            repr(("transfer", read.space, write.space, size)), (read, write), "synchronous", exact_copy=True
        )
    address = evaluate(args["address"], {})
    if call.instr == "input":
        host = args["host"]
        accesses = (
            EncodedAccess("input", evaluate(host.offset, {}), size),
            EncodedAccess("registers", address, size, True),
        )
    elif call.instr == "output":
        host = args["host"]
        accesses = (
            EncodedAccess("registers", address, size),
            EncodedAccess("output", evaluate(host.offset, {}), size, True),
        )
    elif call.instr == "increment":
        accesses = (EncodedAccess("registers", address, size), EncodedAccess("registers", address, size, True))
    else:
        return None
    key = repr((call.instr, tuple((key, value) for key, value in call.args if key != "address")))
    return StorageEffects(key, accesses, "synchronous", exact_copy=call.instr in ("input", "output"))


def rewrite(call, accesses):
    if call.instr == "transfer":
        changes = {"source": Const(accesses[0].start), "destination": Const(accesses[1].start)}
    else:
        local = [access for access in accesses if access.space == "registers"]
        assert all(access.start == local[0].start for access in local)
        changes = {"address": Const(local[0].start)}
    return replace(call, args=tuple((key, changes.get(key, value)) for key, value in call.args))


ADAPTER = StorageAdapter(
    project, rewrite, "neutral-exact-device-storage", "byte-exact-v1", "synthetic-transfer-ISA-only"
)


def fixture(width=8, segments=1):
    body, objects = [], []
    for segment in range(segments):
        base = segment * 3 * width
        a, b, c = (base + offset * width for offset in range(3))
        objects.extend(
            TemporaryObject(f"object_{segment}_{index}", "registers", address, width)
            for index, address in enumerate((a, b, c))
        )
        body.extend(
            (
                memory_call("input", a, width, a),
                memory_call("increment", a, width),
                memory_call("input", b, width, b),
                memory_call("increment", b, width),
                memory_call("output", b, width, b),
                memory_call("input", c, width, c),
                memory_call("output", c, width, c),
                memory_call("output", a, width, a),
            )
        )
    kernel = Kernel(
        "pressure",
        (
            TensorArg("input", (3 * width * segments,), "i8", "read"),
            TensorArg("output", (3 * width * segments,), "i8", "write"),
        ),
        tuple(body),
    )
    banks = (
        StorageBank("registers", "device-register-file", width, "bytes"),
        StorageBank("saved", "addressable-spill-memory", width, "bytes"),
    )
    capability = SpillClass("registers", "saved", width, 1, emit_copy, "synthetic-exact-copy-pair")
    model = dict(
        private_ownership=True,
        external_windows=(
            AddressWindow("input", 0, 3 * width * segments),
            AddressWindow("output", 0, 3 * width * segments),
        ),
    )
    return kernel, tuple(objects), banks, capability, model


def test_ordinary_allocation_inserts_exact_device_spills_and_recomputes_every_analysis():
    kernel, objects, banks, capability, model = fixture()
    assert allocate_temporary_storage(kernel, objects, banks, ADAPTER, **model) is None
    result = allocate_temporary_storage(kernel, objects, banks, ADAPTER, spill_classes=(capability,), **model)
    assert result is not None
    assert result.report["spills"]["inserted_exact_copies"] == 2
    assert result.report["spills"]["checked_original_operations"] == len(kernel.body)
    assert len(result.report["repair_rounds"]) == 1
    assert result.report["repair_rounds"][0]["before_sha256"] == kernel.digest()
    assert set(result.report["repair_rounds"][0]["invalidated"]) == {
        "liveness",
        "allocation",
        "completion",
        "emission",
        "program_size",
    }
    for call in result.kernel.body:
        for access in project(call).accesses:
            if access.space in SPACES:
                assert access.start == 0 and access.size == 8
    proof = verify_device_spills(kernel, result.kernel, objects, banks, ADAPTER, (capability,), **model)
    assert proof == result.report["spills"]
    assert proof["requested_read_bytes"] == proof["requested_write_bytes"] == 80


def test_slots_are_reused_and_repair_is_bounded():
    kernel, objects, banks, capability, model = fixture(segments=2)
    assert (
        allocate_temporary_storage(kernel, objects, banks, ADAPTER, spill_classes=(capability,), max_spills=1, **model)
        is None
    )
    result = allocate_temporary_storage(
        kernel, objects, banks, ADAPTER, spill_classes=(capability,), max_spills=2, **model
    )
    assert result.report["spills"]["inserted_exact_copies"] == 4
    assert len(result.report["repair_rounds"]) == 2
    saved = [repair["saved_object"] for repair in result.report["repair_rounds"]]
    assert [result.report["addresses"][name] for name in saved] == [0, 0]
    with pytest.raises(StorageOptimizationUnavailable, match="call budget"):
        allocate_temporary_storage(
            kernel, objects, banks, ADAPTER, spill_classes=(capability,), max_spills=2, max_calls=19, **model
        )


@pytest.mark.parametrize(
    "fault", ["missing_store", "missing_reload", "wrong_source", "wrong_output", "arithmetic", "extent"]
)
def test_actual_spill_mutations_do_not_inherit_a_plan_receipt(fault):
    kernel, objects, banks, capability, model = fixture()
    result = allocate_temporary_storage(kernel, objects, banks, ADAPTER, spill_classes=(capability,), **model)
    body = list(result.kernel.body)
    transfers = [index for index, call in enumerate(body) if call.instr == "transfer"]
    if fault == "missing_store":
        del body[transfers[0]]
    elif fault == "missing_reload":
        del body[transfers[1]]
    elif fault in ("wrong_source", "extent"):
        index = transfers[1]
        changed = "source" if fault == "wrong_source" else "size"
        body[index] = replace(
            body[index], args=tuple((key, value + 1 if key == changed else value) for key, value in body[index].args)
        )
    elif fault == "wrong_output":
        index = next(index for index, call in enumerate(body) if call.instr == "output")
        body[index] = replace(
            body[index],
            args=tuple((key, Ptr("output", Const(0)) if key == "host" else value) for key, value in body[index].args),
        )
    else:
        index = next(index for index, call in enumerate(body) if call.instr == "increment")
        body[index] = replace(body[index], instr="unknown-arithmetic")
    with pytest.raises(ValueError):
        verify_device_spills(
            kernel, replace(result.kernel, body=tuple(body)), objects, banks, ADAPTER, (capability,), **model
        )


def test_only_explicit_available_exact_device_classes_can_spill():
    kernel, objects, banks, capability, model = fixture()
    assert allocate_temporary_storage(kernel, objects, banks, ADAPTER, spill_classes=(), **model) is None
    assert (
        allocate_temporary_storage(kernel, objects, banks, ADAPTER, spill_classes=(capability,), max_spills=0, **model)
        is None
    )
    assert (
        allocate_temporary_storage(
            kernel, objects, (banks[0], replace(banks[1], capacity=7)), ADAPTER, spill_classes=(capability,), **model
        )
        is None
    )
    with pytest.raises(StorageOptimizationUnavailable, match="distinct addressable"):
        allocate_temporary_storage(
            kernel,
            objects,
            (banks[0], replace(banks[1], backing=banks[0].backing)),
            ADAPTER,
            spill_classes=(capability,),
            **model,
        )

    def inexact(call):
        effect = project(call)
        return replace(effect, exact_copy=False) if call.instr == "transfer" else effect

    with pytest.raises(ValueError, match="not the declared exact"):
        allocate_temporary_storage(
            kernel, objects, banks, replace(ADAPTER, project=inexact), spill_classes=(capability,), **model
        )


def test_independent_spill_check_reopens_current_reservations_and_original_alignment():
    kernel, objects, banks, capability, model = fixture()
    result = allocate_temporary_storage(kernel, objects, banks, ADAPTER, spill_classes=(capability,), **model)
    assert (
        allocate_temporary_storage(
            kernel,
            objects,
            banks,
            ADAPTER,
            spill_classes=(capability,),
            reservations=(Reservation("saved", 0, 8),),
            **model,
        )
        is None
    )
    with pytest.raises(ValueError, match="address window"):
        verify_device_spills(
            kernel,
            result.kernel,
            objects,
            banks,
            ADAPTER,
            (capability,),
            reservations=(Reservation("saved", 0, 8),),
            **model,
        )
    shifted = replace(
        result.kernel,
        body=tuple(
            replace(
                call,
                args=tuple(
                    (name, value + 1 if name in {"address", "source", "destination"} else value)
                    for name, value in call.args
                ),
            )
            for call in result.kernel.body
        ),
    )
    objects = tuple(replace(obj, alignment=2) for obj in objects)
    banks = tuple(replace(bank, capacity=16) for bank in banks)
    with pytest.raises(ValueError, match="physical alignment"):
        verify_device_spills(kernel, shifted, objects, banks, ADAPTER, (capability,), **model)


def test_spilling_cannot_make_simultaneous_operands_fit_one_device_register():
    kernel, _objects, banks, capability, _model = fixture()
    objects = (TemporaryObject("a", "registers", 0, 8), TemporaryObject("b", "registers", 8, 8))
    combine = Call("combine", (("left", Const(0)), ("right", Const(8)), ("size", Const(8))))
    kernel = replace(
        kernel, body=(memory_call("input", 0, 8), memory_call("input", 8, 8, 8), combine, memory_call("output", 8, 8))
    )

    def decode(call):
        if call.instr != "combine":
            return project(call)
        left, right = (evaluate(call.arg(name), {}) for name in ("left", "right"))
        return StorageEffects(
            "opaque-byte-combine",
            (
                EncodedAccess("registers", left, 8),
                EncodedAccess("registers", right, 8),
                EncodedAccess("registers", right, 8, True),
            ),
            "synchronous",
        )

    def rebase(call, accesses):
        if call.instr != "combine":
            return rewrite(call, accesses)
        changes = {"left": Const(accesses[0].start), "right": Const(accesses[1].start)}
        return replace(call, args=tuple((name, changes.get(name, value)) for name, value in call.args))

    adapter = replace(ADAPTER, project=decode, rewrite=rebase)
    result = allocate_temporary_storage(
        kernel,
        objects,
        banks,
        adapter,
        spill_classes=(capability,),
        private_ownership=True,
        external_windows=(AddressWindow("input", 0, 24), AddressWindow("output", 0, 24)),
    )
    assert result is None


def test_valid_asynchronous_source_requires_a_separate_spill_completion_contract():
    kernel, objects, banks, capability, model = fixture()
    wait = Call("completed_input", ())
    kernel = replace(kernel, body=(kernel.body[0], wait, *kernel.body[1:]))

    def decode(call):
        if call.instr == "completed_input":
            return StorageEffects("complete-input", completion="wait", waits=("input-ready",))
        effect = project(call)
        if call.instr == "input" and evaluate(call.arg("host").offset, {}) == 0:
            return replace(effect, completion="asynchronous", token="input-ready")
        return effect

    with pytest.raises(StorageOptimizationUnavailable, match="synchronous original completion"):
        allocate_temporary_storage(
            kernel, objects, banks, replace(ADAPTER, project=decode), spill_classes=(capability,), **model
        )


def test_exact_copy_effects_require_one_complete_bit_preserving_transfer():
    with pytest.raises(ValueError, match="equally sized"):
        StorageEffects(
            "copy",
            (EncodedAccess("registers", 0, 4), EncodedAccess("saved", 0, 3, True)),
            "synchronous",
            exact_copy=True,
        )


def test_spill_provenance_does_not_expand_large_byte_domains():
    width = 1 << 35
    kernel, objects, banks, capability, model = fixture(width)
    result = allocate_temporary_storage(kernel, objects, banks, ADAPTER, spill_classes=(capability,), **model)
    assert result.report["spills"]["inserted_exact_copies"] == 2
    assert result.report["spills"]["requested_read_bytes"] == 10 * width
    assert len(result.kernel.body) == 10


@pytest.mark.parametrize(
    "width,faulty_copy,partial",
    [(3, False, False), (8, False, False), (16, False, False), (8, True, False), (8, False, True)],
)
def test_actual_emitted_device_spill_program_matches_independent_complete_bytes(tmp_path, width, faulty_copy, partial):
    import ctypes
    import shutil
    import subprocess

    from merlin.sched.codegen import emit_c_function
    from merlin.sched.isa import InstrDef, InstructionSet, Operand

    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("native C toolchain unavailable")
    kernel, objects, banks, capability, model = fixture(width)
    if partial:
        kernel = replace(kernel, body=(kernel.body[0], memory_call("increment", 1, width - 2), *kernel.body[2:]))
    result = allocate_temporary_storage(kernel, objects, banks, ADAPTER, spill_classes=(capability,), **model)
    definitions = {}
    for call in result.kernel.body:
        names = [name for name, _ in call.args]
        definitions[call.instr] = InstrDef(
            call.instr,
            tuple(Operand(name, "ptr" if name == "host" else "int") for name in names),
            lambda args, op=call.instr: "do_" + op + "(" + ",".join(args) + ");",
        )
    iset = InstructionSet("neutral-device-spill-ISA", definitions, {"i8": "int8_t"}, "/* synchronous completion */")
    source = f"""#include <stdint.h>
#include <string.h>
#include <assert.h>
static uint8_t registers[{width}], saved[{width}];
uint64_t reads,writes,commands;
static uint8_t *space(int bank,int address,int size) {{ assert(address>=0 && address+size<={width});
 return (bank ? saved : registers)+address; }}
static void do_input(int address,int size,const void *host) {{ commands++; reads+=size; writes+=size;
 memcpy(space(0,address,size),host,size); }}
static void do_output(int address,int size,void *host) {{ commands++; reads+=size; writes+=size;
 memcpy(host,space(0,address,size),size); }}
static void do_increment(int address,int size) {{ commands++; reads+=size; writes+=size;
 uint8_t *p=space(0,address,size); for(int i=0;i<size;i++) p[i]++; }}
static void do_transfer(int source_bank,int source,int destination_bank,int destination,int size) {{
 commands++; reads+=size; writes+=size;
 memcpy(space(destination_bank,destination,size),space(source_bank,source,size),size);
 if({int(faulty_copy)} && destination_bank==1) space(destination_bank,destination,size)[0]^=1;
}}
"""
    source += emit_c_function(result.kernel, iset, symbol="optimized", index_bits=64)
    source += "void run(int8_t *input,int8_t *output) {reads=writes=commands=0;optimized(input,output);}\n"
    path = tmp_path / "spills.c"
    path.write_text(source)
    library = tmp_path / f"spills_{width}_{faulty_copy}_{partial}.so"
    subprocess.run(
        [cc, "-std=c11", "-O1", "-shared", "-fPIC", str(path), "-o", str(library)], check=True, capture_output=True
    )
    compiled = ctypes.CDLL(str(library))
    source_bytes = bytes((i * 79 + 251) % 256 for i in range(width * 3))
    expected = bytes(
        (value + (width <= index < 2 * width or (1 <= index < width - 1 if partial else index < width))) % 256
        for index, value in enumerate(source_bytes)
    )
    inp = (ctypes.c_uint8 * len(source_bytes)).from_buffer_copy(source_bytes)
    output = (ctypes.c_uint8 * len(source_bytes))(*([113] * len(source_bytes)))
    compiled.run.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    compiled.run(inp, output)
    if faulty_copy:
        assert bytes(output) != expected  # Target implementation qualification is independent of its declarations.
    else:
        assert bytes(output) == expected
    report = result.report["spills"]
    assert ctypes.c_uint64.in_dll(compiled, "reads").value == report["requested_read_bytes"]
    assert ctypes.c_uint64.in_dll(compiled, "writes").value == report["requested_write_bytes"]
    assert ctypes.c_uint64.in_dll(compiled, "commands").value == report["actual_commands"]
