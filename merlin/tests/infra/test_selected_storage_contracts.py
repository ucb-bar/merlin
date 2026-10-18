"""Selected RTL/software allocation through final bytes and independent execution.

The neutral machines establish mechanism scope, never real-target support.
"""

import ctypes
import hashlib
import json
import shutil
import struct
import subprocess
import time
from dataclasses import replace

import pytest
import yaml
from test_private_copy_elimination import ADAPTER, SPACES, command, project
from test_schedule_pipeline import encoding

from merlin.compile.scheduling.program_limits import AddressWindow, ProgramLimits
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets, compile_schedule
from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts
from merlin.compile.scheduling.temporary_storage import (
    StorageEffects,
    StorageOptimizationUnavailable,
    TemporaryObject,
    allocate_temporary_storage,
)
from merlin.sched.ir import Call, Const, Kernel, TensorArg, evaluate
from merlin.semantic_compiler.allocation_types import Reservation
from merlin.targetgen.isa_dialect_gen import decode_instruction, encode_instruction


def provider_project(call):
    if call.instr == "wait":
        return StorageEffects("wait:" + str(evaluate(call.arg("size"), {})), completion="wait",
                              waits=(str(evaluate(call.arg("size"), {})),))
    if call.instr == "async_copy":
        effect = project(replace(call, instr="copy"))
        return replace(effect, completion="asynchronous", token=str(evaluate(call.arg("size"), {})), exact_copy=False)
    return project(call)


def inputs(tmp_path, *, width=7, separate=False, word_format="<BBBQQQ", reserve=True, mutate=None):
    length = 2 * width + 3
    word = struct.Struct(word_format)
    integer_bits = 64 if word_format.endswith("Q") else 32
    widths = [8, 8, 8, integer_bits, integer_bits, integer_bits]
    byte_offsets = [0, 1, 2, 3, 3 + integer_bits // 8, 3 + integer_bits // 4]
    offsets = [b * 8 if word_format[0] == "<" else word.size * 8 - b * 8 - bits
               for b, bits in zip(byte_offsets, widths)]
    facts = dict(facts=dict(target="neutral", word_bits=word.size * 8, widths=widths, offsets=offsets,
                            codes=list(range(4)), selectors=list(range(4)), capacities=[192, 192, 2 * length + 8],
                            starts=[8, 16, 0], sizes=[184, 176, 2 * length + 8], alignments=[4, 8],
                            packing_start=32, staging_start=64, scratch_size=8))

    def ref(*parts):
        return {"fact": ["facts", *parts]}

    def arg(name):
        return {"argument": name}

    def space(name):
        return dict(argument=name, choices=[dict(value=ref("selectors", i), resource=resource)
                                            for i, resource in enumerate(SPACES)])

    memory = [dict(name=name, kind="memory", unit="bytes", capacity=ref("capacities", min(i, 2)))
              for i, name in enumerate(SPACES)]
    names = ("op", "read_bank", "write_bank", "source", "destination", "size")
    instructions, effects = [], {}
    for code, op in enumerate(("copy", "increment", "async_copy", "wait")):
        fields = [dict(offset=ref("offsets", i), width=ref("widths", i),
                       **(dict(fixed=ref("codes", code)) if i == 0 else dict(argument=name)))
                  for i, name in enumerate(names)]
        operands = [dict(name=name, type=dict(kind="integer", signed=False, bits=ref("widths", i)), unit="integer")
                    for i, name in enumerate(names) if i]
        accesses = [] if op == "wait" else [
            dict(space=space("read_bank"), start=arg("source"), size=arg("size"), writes=False),
            dict(space=space("write_bank"), start=arg("destination"), size=arg("size"), writes=True),
        ]
        effects[op] = dict(accesses=accesses,
                           completion=("asynchronous" if op == "async_copy" else
                                       "wait" if op == "wait" else "synchronous"),
                           token=arg("size") if op == "async_copy" else None,
                           waits=[arg("size")] if op == "wait" else [], exact_copy=op == "copy")
        instructions.append(dict(name=op, operands=operands, results=[], immediates=[], encoding=fields,
                                 semantics=dict(primitive="neutral." + op, bindings={name: name for name in names[1:]}),
                                 effects=dict(reads=[] if op == "wait" else list(SPACES),
                                              writes=[] if op == "wait" else list(SPACES))))
    banks = {name: dict(backing="second" if separate and i else "first", start=ref("starts", i),
                        size=ref("sizes", i), alignment=ref("alignments", i), ownership="invocation_private")
             for i, name in enumerate(SPACES[:2])}
    external = {name: dict(backing=name + "_memory", start=ref("starts", 2), size=ref("sizes", 2),
                           ownership="persistent" if name == "output" else "runtime_managed") for name in SPACES[2:]}
    reservations = ([dict(resource=SPACES[0], start=ref(key), size=ref("scratch_size"), purpose=purpose)
                     for key, purpose in (("packing_start", "packing"), ("staging_start", "dma_staging"))]
                    if reserve else [])
    software = dict(schema="merlin.software_spec.v1", target="neutral", status="reviewed", isa=dict(
        schema="merlin.isa_spec.v1", namespace="storage_test", facts_sha256="0" * 64,
        word_bits=ref("word_bits"), resources=memory, instructions=instructions), scheduling=dict(
            schema="merlin.schedule_effects.v1", numerical_policy="wrap-u8-v1",
            instruction_byteorder="little" if word_format[0] == "<" else "big", configuration={},
            storage=dict(schema="merlin.temporary_storage_contract.v1", banks=banks, external=external,
                         reservations=reservations, instructions=effects)))
    if mutate is not None:
        mutate(facts["facts"], software, software["scheduling"]["storage"])
    fact_path, software_path = tmp_path / "facts.json", tmp_path / "software.yaml"
    fact_path.write_text(json.dumps(facts))
    software["isa"]["facts_sha256"] = hashlib.sha256(fact_path.read_bytes()).hexdigest()
    software_path.write_text(yaml.safe_dump(software))
    selected = read_scheduling_contracts(fact_path, software_path, target="neutral")
    adapter = replace(ADAPTER, configuration_identity=selected.hardware_identity, project=provider_project)
    codec, _ = encoding(word_format)

    def encode(kernel):
        return b"".join(encode_instruction(selected.isa, call.instr,
                                           {key: evaluate(value, {}) for key, value in call.args})
                        .to_bytes(word.size, selected.instruction_byteorder) for call in kernel.body)

    def decode(data):
        result = []
        for offset in range(0, len(data), word.size):
            name, values = decode_instruction(selected.isa, int.from_bytes(
                data[offset:offset + word.size], selected.instruction_byteorder))
            result.append(Call(name, tuple((key, Const(value)) for key, value in values.items())))
        return tuple(result)

    codec = replace(codec, encode=encode, decode_calls=decode,
                    program_decoder=replace(codec.program_decoder, configuration_identity=selected.hardware_identity))
    # The final-size policy remains an explicit compilation budget. Resource
    # windows themselves are independently derived from the selected documents.
    windows = tuple(AddressWindow(w.resource, w.start, w.size)
                    for w in (*selected.storage.banks, *selected.storage.external))
    limits = ProgramLimits(word.size * 128, 128, 1, windows, ())
    calls, objects = [], []
    for chunk, offset in enumerate(range(0, length, width)):
        size = min(width, length - offset)
        for branch in range(2):
            bank = branch
            source = 16 + (2 * chunk + branch) * 32
            objects.append(TemporaryObject("value_" + str(len(objects)), SPACES[bank], source, size))
            alias = 1 if not separate else bank
            calls += [command("copy", branch * length + offset, source, size, 2, bank),
                      command("increment", source, source, size, alias, alias),
                      command("copy", source, branch * length + offset, size, alias, 3)]
    kernel = Kernel("two_branch_state", (TensorArg("input", (2 * length + 8,), "i8", "read"),
                                        TensorArg("output", (2 * length + 8,), "i8", "write")), tuple(calls))
    return selected, kernel, tuple(objects), adapter, codec, limits, word, length


def compile_case(model, *, optimize=True, private_ownership=True, **kwargs):
    selected, kernel, objects, adapter, codec, limits, _, _ = model
    return compile_schedule(kernel, objects, None, adapter, codec, limits,
                            budgets=ScheduleBudgets(eliminate_copies=False, optimize_storage=optimize),
                            selected_contracts=selected, private_ownership=private_ownership, **kwargs)


@pytest.fixture
def native(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("native C compiler unavailable")
    source = tmp_path / "machine.c"
    source.write_text(r'''
#include <stdint.h>
#include <string.h>
static uint64_t number(const unsigned char *p,int width,int big) {
 uint64_t x=0; for(int i=0;i<width;i++) x|=(uint64_t)p[i] << (8*(big?width-1-i:i)); return x;
}
int run(const unsigned char *code,int bytes,int width,int big,int separate,
        unsigned char *first,unsigned char *second,unsigned char *input,unsigned char *output,
        int capacity,int external,uint64_t *counts) {
 unsigned char *memory[4]={first,separate?second:first,input,output};
 int limits[4]={capacity,capacity,external,external};
 int pending=0,pr=0,pw=0; uint64_t ps=0,pd=0,pn=0;
 memset(counts,0,5*sizeof(*counts));
 int word=3+3*width;
 for(int pc=0;pc<bytes;pc+=word) {
  int op=code[pc],r=code[pc+1],w=code[pc+2];
  uint64_t s=number(code+pc+3,width,big), d=number(code+pc+3+width,width,big);
  uint64_t n=number(code+pc+3+2*width,width,big);
  if(r>3 || w>3 || pc+word>bytes) return 1;
  counts[0]++;
  if(op==3) {
   if(!pending || n!=pn) return 2;
   memmove(memory[pw]+pd,memory[pr]+ps,pn); pending=0; continue;
  }
  if(n==0 || s+n>(uint64_t)limits[r] || d+n>(uint64_t)limits[w]) return 3;
  counts[1]+=n; counts[2]+=n;
  if(r<2 && s+n>counts[3+(r&&separate)]) counts[3+(r&&separate)]=s+n;
  if(w<2 && d+n>counts[3+(w&&separate)]) counts[3+(w&&separate)]=d+n;
  if(op==2) { if(pending) return 4; pending=1; pr=r; pw=w; ps=s; pd=d; pn=n; }
  else if(op==0) memmove(memory[w]+d,memory[r]+s,n);
  else if(op==1) { for(uint64_t i=0;i<n;i++) memory[w][d+i]=(unsigned char)(memory[r][s+i]+1); }
  else return 5;
 }
 return pending?6:0;
}
''')
    library = tmp_path / "machine.so"
    subprocess.run([cc, "-std=c11", "-O1", "-shared", "-fPIC", str(source), "-o", str(library)],
                   check=True, capture_output=True)
    machine = ctypes.CDLL(str(library))
    bytep = ctypes.POINTER(ctypes.c_ubyte)
    machine.run.argtypes = [bytep, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                           bytep, bytep, bytep, bytep, ctypes.c_int, ctypes.c_int,
                           ctypes.POINTER(ctypes.c_uint64)]

    def execute(image, first, second, data, output, width=8, big=False, separate=False):
        def buffer(array):
            return (ctypes.c_ubyte * len(array)).from_buffer(array)
        counts = (ctypes.c_uint64 * 5)()
        code = bytearray(image)
        status = machine.run(buffer(code), len(code), width, big, separate, buffer(first), buffer(second),
                             buffer(data), buffer(output), len(first), len(data), counts)
        assert status == 0, status
        return tuple(counts)
    return execute


@pytest.mark.parametrize("width,separate,word_format", [(7, False, "<BBBQQQ"), (13, True, ">BBBIII")])
def test_selected_allocation_realizes_reuse_and_preserves_complete_native_state(
    tmp_path, native, record_property, width, separate, word_format
):
    model = inputs(tmp_path, width=width, separate=separate, word_format=word_format)
    selected, kernel, _, _, _, _, _, length = model
    results, compile_seconds = [], []
    for enabled in (False, True):
        started = time.perf_counter()
        results.append(compile_case(model, optimize=enabled))
        compile_seconds.append(time.perf_counter() - started)
    baseline, optimized = results
    assert baseline.kernel == kernel
    assert baseline.report["storage"]["optimization"] == "disabled_verified_original"
    assert optimized.report["context"]["library"] == baseline.report["context"]["library"]
    assert "physical_aliases" in optimized.report["selected_contracts"]["consumed_scopes"]
    assert max(optimized.report["storage"]["addresses"].values()) < min(
        baseline.report["storage"]["addresses"].values()) + 1
    arms = []
    for name, result in zip(("baseline", "optimized"), results):
        first, second = bytearray([99] * 192), bytearray([99] * 192)
        state = bytes((i * 19 + 251) % 256 for i in range(length))
        measurements = []
        for invocation in range(3):
            # Host packing is explicit and unchanged across treatments. The
            # second branch is complete carried state from the previous call.
            raw = bytes((i * 31 + 253 + invocation) % 256 for i in range(length))
            packed = raw[::-1] + state + bytes([73] * 8)
            data, output = bytearray(packed), bytearray([91] * len(packed))
            image = b"".join(result.program.segments)
            started = time.perf_counter()
            counts = native(image, first, second, data, output, width=8 if word_format.endswith("Q") else 4,
                            big=word_format[0] == ">", separate=separate)
            elapsed = time.perf_counter() - started
            expected = bytes((x + 1) % 256 for x in packed[:2 * length]) + bytes([91] * 8)
            assert bytes(output) == expected and bytes(data) == packed
            state = bytes(output[length:2 * length])
            assert first[:8] == bytes([99] * 8) and second[:16] == bytes([99] * 16)
            assert first[32:40] == first[64:72] == bytes([99] * 8)
            conformance = result.report["final_conformance"]
            assert counts[:3] == (conformance["actual_commands"], conformance["requested_read_bytes"],
                                   conformance["requested_write_bytes"])
            measurements.append(dict(invocation=invocation, actual_commands=counts[0],
                                     read_bytes=counts[1], write_bytes=counts[2],
                                     actual_memory_high_water=list(counts[3:]),
                                     host_pack_payload_bytes=2 * length, native_call_seconds=elapsed))
        arms.append(dict(name=name, encoded_bytes=len(image), measurements=measurements,
                         storage=result.report["storage"], final_state_sha256=hashlib.sha256(state).hexdigest()))
    assert arms[0]["final_state_sha256"] == arms[1]["final_state_sha256"]
    assert sum(arms[1]["measurements"][0]["actual_memory_high_water"]) < sum(
        arms[0]["measurements"][0]["actual_memory_high_water"])
    record_property("selected_storage_pair", json.dumps(dict(
        scope="complete neutral two-branch transfer/arithmetic graph with carried state, not a real target/model",
        selected_contracts=selected.record(scopes=("temporary_memory_windows", "storage_effects_and_completion")),
        library_sources=optimized.report["library_sources_sha256"], compile_seconds=compile_seconds,
        compile_peak_memory="unmeasured", accelerator_timing="unmeasured", arms=arms,
    ), sort_keys=True))


def bound(model, *, adapter=None, **kwargs):
    selected, _, _, original, *_ = model
    return selected.storage.bind(selected.isa, original if adapter is None else adapter, banks=None, reservations=(),
                                 external_windows=None, private_ownership=True, **kwargs)


def test_alias_window_lower_bounds_are_per_view_not_global_reservations(tmp_path):
    model = list(inputs(tmp_path))
    kernel = model[1]
    calls = list(kernel.body)
    # The first object's complete use remains in the wider view. The other
    # objects still use the alias whose legal window starts later.
    for i in (1, 2):
        calls[i] = replace(calls[i], args=tuple((key, Const(0) if key in ("read_bank", "write_bank") and
                                               (i == 1 or key == "read_bank") else value)
                                              for key, value in calls[i].args))
    model[1] = replace(kernel, body=tuple(calls))
    result = compile_case(model)
    assert result.report["storage"]["addresses"]["value_0"] == 8
    assert result.report["storage"]["addresses"]["value_1"] == 16
    assert result.report["storage"]["lifetime_calls"]["value_0"] == [0, 2]


def test_capacity_boundaries_and_checked_baseline_diagnostics(tmp_path):
    def capacity(value):
        def mutate(facts, software, storage):
            facts["capacities"][:2] = [value, value]
            facts["sizes"][:2] = [value - facts["starts"][i] for i in range(2)]
        return mutate
    model = inputs(tmp_path, reserve=False, mutate=capacity(23))
    assert compile_case(model).report["storage"]["addresses"]["value_0"] == 16
    # These source addresses are virtual outside the selected small physical
    # bank. The unoptimized route is therefore unavailable, with a diagnostic.
    with pytest.raises(ValueError, match="range|bank view"):
        compile_case(model, optimize=False)
    with pytest.raises(StorageOptimizationUnavailable, match="no legal temporary allocation"):
        compile_case(inputs(tmp_path, reserve=False, mutate=capacity(22)))


def test_selected_reservations_union_aliases_and_runtime_staging(tmp_path):
    model = inputs(tmp_path)
    result = compile_case(model, reservations=(Reservation("typed_view", 16, 8),
                                               Reservation("typed_view", 36, 8)))
    assert min(result.report["storage"]["addresses"].values()) == 24
    rows = result.report["resources"]["reservations"]
    assert {(r["start"], r["extent"]) for r in rows} == {(16, 8), (32, 8), (40, 4), (64, 8)}
    with pytest.raises(ValueError, match="reservation exceeds"):
        compile_case(model, reservations=(Reservation("typed_view", 4, 8),))


def test_selected_geometry_cannot_be_widened_or_relabelled(tmp_path):
    model = inputs(tmp_path)
    selected, _, _, adapter, *_ = model
    banks, reservations, external, _ = bound(model)
    for bad in (replace(banks[0], capacity=193), replace(banks[0], start=0),
                replace(banks[0], backing="hidden_alias"), replace(banks[0], alignment=1),
                replace(banks[0], unit="rows")):
        with pytest.raises(ValueError, match="disagrees"):
            selected.storage.bind(selected.isa, adapter, banks=(bad, banks[1]), reservations=(),
                                  external_windows=external, private_ownership=True)
    with pytest.raises(ValueError, match="external window exceeds"):
        compile_case(model, external_windows=(AddressWindow("input", 0, 1000),))
    assert reservations and banks[0].start == 8 and banks[1].start == 16


@pytest.mark.parametrize("fault", ["private", "external", "capacity", "runtime"])
def test_unknown_ownership_and_capacities_remain_unavailable(tmp_path, fault):
    def mutate(facts, software, storage):
        if fault == "capacity":
            software["isa"]["resources"][0].pop("capacity")
        elif fault == "private":
            storage["banks"]["bytes"]["ownership"] = "unknown"
        elif fault == "external":
            storage["external"]["input"]["ownership"] = "unknown"
    model = inputs(tmp_path, mutate=mutate)
    with pytest.raises(StorageOptimizationUnavailable, match="ownership|unknown"):
        compile_case(model, private_ownership=fault != "runtime")


@pytest.mark.parametrize("fault,diagnostic", [
    ("literal_window", "fact"), ("wrong_capacity", "RTL resource capacity"),
    ("bad_alias", "cannot alias external"), ("external_alias", "external resource aliases"),
    ("missing_semantics", "complete ISA effects"), ("missing_effects", "complete ISA effects"),
    ("extra_effects", "coverage disagrees"), ("unencoded_argument", "encoded integer"),
    ("constant_local", "relocatable"), ("selector", "duplicate encodings"),
    ("bad_wait", "tokens"), ("copy_direction", "exact copy"),
])
def test_contradictory_software_and_rtl_declarations_refuse(tmp_path, fault, diagnostic):
    def mutate(facts, software, storage):
        if fault == "literal_window":
            storage["banks"]["bytes"]["start"] = 8
        elif fault == "wrong_capacity":
            facts["sizes"][0] = 1000
        elif fault == "bad_alias":
            storage["banks"]["bytes"]["backing"] = "input_memory"
        elif fault == "external_alias":
            storage["external"]["output"]["backing"] = "input_memory"
        elif fault in ("missing_semantics", "missing_effects"):
            software["isa"]["instructions"][0].pop(fault.removeprefix("missing_"))
        elif fault == "extra_effects":
            software["isa"]["instructions"][0]["effects"]["reads"] = []
        elif fault == "unencoded_argument":
            storage["instructions"]["copy"]["accesses"][0]["size"] = {"argument": "hidden"}
        elif fault == "constant_local":
            storage["instructions"]["copy"]["accesses"][0]["start"] = {"constant": 0}
        elif fault == "selector":
            choices = storage["instructions"]["copy"]["accesses"][0]["space"]["choices"]
            choices[1]["value"] = choices[0]["value"]
        elif fault == "bad_wait":
            storage["instructions"]["wait"]["waits"] = []
        elif fault == "copy_direction":
            storage["instructions"]["copy"]["accesses"][0]["writes"] = True
            software["isa"]["instructions"][0]["effects"] = dict(reads=[], writes=list(SPACES))
    with pytest.raises(ValueError, match=diagnostic):
        inputs(tmp_path, mutate=mutate)


def test_unknown_calls_partial_initialization_and_lying_projections_refuse(tmp_path):
    model = list(inputs(tmp_path))
    original = model[1]
    model[1] = replace(original, body=(replace(original.body[0], instr="opaque"), *original.body[1:]))
    with pytest.raises(StorageOptimizationUnavailable, match="unknown instruction"):
        compile_case(model)
    call = original.body[0]
    smaller = replace(call, args=tuple((k, v + -1 if k == "size" else v) for k, v in call.args))
    model[1] = replace(original, body=(smaller, *original.body[1:]))
    with pytest.raises(StorageOptimizationUnavailable, match="complete initialization"):
        compile_case(model)
    model[1] = original
    def lies(call):
        effect = provider_project(call)
        return replace(effect, accesses=(replace(effect.accesses[0], start=effect.accesses[0].start + 1),
                                         *effect.accesses[1:]))
    model[3] = replace(model[3], project=lies)
    with pytest.raises(ValueError, match="provider storage effects disagree"):
        compile_case(model)


def test_actual_bytes_and_actual_rewritten_addresses_are_checked(tmp_path):
    model = list(inputs(tmp_path))
    original = model[4]
    def corrupt(kernel):
        call = kernel.body[-1]
        call = replace(call, args=tuple((key, value + 1 if key == "destination" else value)
                                       for key, value in call.args))
        return original.encode(replace(kernel, body=(*kernel.body[:-1], call)))
    model[4] = replace(original, encode=corrupt)
    with pytest.raises(ValueError, match="actual emitted.*effects differ"):
        compile_case(model)
    def lies(data):
        return original.decode_calls(original.encode(model[1]))
    model[4] = replace(original, encode=corrupt, decode_calls=lies)
    with pytest.raises(ValueError, match="disagrees with actual selected ISA words"):
        compile_case(model)
    model[4] = original
    def corrupt_rewrite(call, accesses):
        result = ADAPTER.rewrite(call, accesses)
        return replace(result, args=tuple((key, Const(0) if key == "destination" else value)
                                         for key, value in result.args))
    model[3] = replace(model[3], rewrite=corrupt_rewrite)
    with pytest.raises(ValueError, match="invalid|aliases|fixed external|bank view"):
        compile_case(model)


def test_async_lifetimes_use_actual_waits_and_ordinary_emission_remains_disabled(tmp_path, native):
    model = list(inputs(tmp_path, reserve=False))
    selected, kernel, objects, _, codec, _, _, length = model
    size = objects[0].size
    body = (kernel.body[0], command("async_copy", 16, 0, size, 1, 3), kernel.body[3],
            command("wait", 0, 0, size), kernel.body[5])
    model[1], model[2] = replace(kernel, body=body), objects[:2]
    with pytest.raises(StorageOptimizationUnavailable, match="synchronous completion"):
        compile_case(model)
    banks, reservations, external, adapter = bound(model)
    optimized = allocate_temporary_storage(model[1], model[2], banks, adapter, private_ownership=True,
                                           external_windows=external, reservations=reservations, eliminate_copies=False)
    assert optimized.report["lifetime_calls"] == {"value_0": [0, 3], "value_1": [2, 4]}
    assert optimized.report["addresses"] == {"value_0": 16, "value_1": 24}
    image = codec.encode(optimized.kernel)
    selected.verify_image(image, codec.decode_calls(image))
    data = bytearray(range(2 * length + 8))
    output = bytearray([91] * len(data))
    counts = native(image, bytearray([99] * 192), bytearray([99] * 192), data, output)
    expected = bytearray([91] * len(data))
    expected[:size], expected[length:length + size] = data[:size], data[length:length + size]
    assert output == expected and counts[:3] == (5, 4 * size, 4 * size)
    model[1] = replace(model[1], body=tuple(call for call in body if call.instr != "wait"))
    with pytest.raises(ValueError, match="returns before actual completion"):
        compile_case(model)


def test_false_physical_alias_qualification_is_exposed_by_native_execution(tmp_path, native):
    model = list(inputs(tmp_path, width=13, separate=True, word_format=">BBBIII"))
    kernel = model[1]
    c = kernel.body
    model[1] = replace(kernel, body=(c[0], c[3], c[1], c[2], c[4], c[5], *c[6:]))
    baseline, optimized = compile_case(model, optimize=False), compile_case(model)
    length = model[-1]
    data = bytearray((i * 31 + 253) % 256 for i in range(2 * length + 8))
    outputs = []
    for result in (baseline, optimized):
        output = bytearray([91] * len(data))
        # The software falsely declared separate backings. This independent
        # machine physically aliases them; the valid projected plan cannot
        # qualify the declaration on its behalf.
        native(b"".join(result.program.segments), bytearray([99] * 192), bytearray([99] * 192),
               data, output, width=4, big=True, separate=False)
        outputs.append(output)
    expected = bytearray((v + 1) % 256 for v in data[:2 * length]) + bytearray([91] * 8)
    assert outputs[0] == expected and outputs[1] != expected


@pytest.mark.parametrize("path", ["facts_path", "software_path"])
def test_input_and_library_changes_during_rewrite_reject(tmp_path, monkeypatch, path):
    from merlin.compile.scheduling import schedule_pipeline
    model = list(inputs(tmp_path))
    selected = model[0]
    def changed(call, accesses):
        file = getattr(selected, path)
        file.write_bytes(file.read_bytes() + b"\n")
        return ADAPTER.rewrite(call, accesses)
    model[3] = replace(model[3], rewrite=changed)
    with pytest.raises(ValueError, match="changed; reselect"):
        compile_case(model)
    model = list(inputs(tmp_path))
    sources = schedule_pipeline.shared_library_sources
    change = False
    def current():
        return {**sources(), **({"changed": "0" * 64} if change else {})}
    def changed_library(call, accesses):
        nonlocal change
        change = True
        return ADAPTER.rewrite(call, accesses)
    monkeypatch.setattr(schedule_pipeline, "shared_library_sources", current)
    model[3] = replace(model[3], rewrite=changed_library)
    with pytest.raises(ValueError, match="library changed"):
        compile_case(model)


def test_lower_bound_is_enforced_by_solver_checker_and_fixed_boundaries():
    from merlin.semantic_compiler.allocate import allocate, check_assignment
    from merlin.semantic_compiler.allocation_types import CandidateGraph, StorageBank, Value
    banks = (StorageBank("wide", "ram", 64, "bytes", start=8),
             StorageBank("narrow", "ram", 64, "bytes", alignment=8, start=32))
    graph = CandidateGraph((Value(0, "input", "wide", 16, (), "arg", "input"),
                            Value(1, "computed", "narrow", 16, (0,), None, "instruction"),
                            Value(2, "completion", "wide", 0, (0, 1), None, "event")), (1,))
    order = (1, 2)
    result = allocate(graph, order, banks)
    assert result.status == "feasible" and result.addresses == {0: 8, 1: 32, 2: 0}
    assert check_assignment(graph, order, result.addresses, banks) == (True, "")
    assert not check_assignment(graph, order, {0: 0, 1: 32, 2: 0}, banks)[0]
    assert allocate(graph, order, banks, fixed_inputs={"arg": 0}).status == "modeling_failure"
    assert allocate(graph, order, banks, fixed_outputs=(16,)).status == "modeling_failure"
    assert allocate(graph, order, banks, reservations=(Reservation("narrow", 16, 8),)).status == "unqualified_target"
    assert StorageBank.from_record(banks[1].record()) == banks[1]
    legacy = StorageBank("legacy", "other", 64, "bytes")
    assert "start" not in legacy.record() and StorageBank.from_record(legacy.record()) == legacy


def test_unknown_completion_and_actual_selector_outside_domain_are_barriers(tmp_path):
    def unknown(facts, software, storage):
        storage["instructions"]["increment"]["completion"] = "unknown"
    with pytest.raises(StorageOptimizationUnavailable, match="unknown instruction or completion"):
        compile_case(inputs(tmp_path, mutate=unknown))
    model = list(inputs(tmp_path))
    source = model[1]
    bad = replace(source.body[0], args=tuple((k, Const(9) if k == "write_bank" else v)
                                           for k, v in source.body[0].args))
    model[1] = replace(source, body=(bad, *source.body[1:]))
    with pytest.raises(ValueError, match="selector is outside"):
        compile_case(model)


def test_typed_byte_address_bindings_require_matching_resource_and_units(tmp_path):
    def typed(facts, software, storage):
        # This copy form loads only input into the first local view. Other
        # instructions retain dynamic integer-coordinate bank selectors.
        instruction = software["isa"]["instructions"][0]
        for name, resource in (("source", "input"), ("destination", "bytes")):
            argument = next(a for a in instruction["operands"] if a["name"] == name)
            argument.update(type=dict(kind="address", bits=argument["type"]["bits"], signed=False, space=resource),
                            unit="bytes")
        storage["instructions"]["copy"]["accesses"][0]["space"] = {"resource": "input"}
        storage["instructions"]["copy"]["accesses"][1]["space"] = {"resource": "bytes"}
        instruction["effects"] = dict(reads=["input"], writes=["bytes"])
    model = inputs(tmp_path, mutate=typed)
    *_, adapter = bound(model)
    assert adapter.project(model[1].body[0]).accesses == provider_project(model[1].body[0]).accesses
    def wrong(facts, software, storage):
        typed(facts, software, storage)
        argument = next(a for a in software["isa"]["instructions"][0]["operands"] if a["name"] == "source")
        argument["type"]["space"] = "output"
    with pytest.raises(ValueError, match="compatible byte address"):
        inputs(tmp_path, mutate=wrong)


def test_selected_copy_semantics_survive_relocation_and_final_byte_checks(tmp_path, native):
    from merlin.compile.scheduling.spill_conformance import verify_storage_dataflow
    model = list(inputs(tmp_path))
    selected, template, objects, original_adapter, codec, limits, _, length = model
    size = objects[0].size
    source = replace(template, body=(template.body[0], template.body[1],
                                     command("copy", 16, 48, size, 0, 1), command("copy", 48, 0, size, 1, 3)))
    objects = objects[:2]
    results = [compile_schedule(source, objects, None, original_adapter, codec, limits,
                                budgets=ScheduleBudgets(optimize_storage=enabled), selected_contracts=selected,
                                private_ownership=True) for enabled in (False, True)]
    data = bytearray((31 * i + 253) % 256 for i in range(2 * length + 8))
    for result, commands in zip(results, (4, 3)):
        output = bytearray([91] * len(data))
        counts = native(b"".join(result.program.segments), bytearray([99] * 192), bytearray([99] * 192), data, output)
        assert output == bytearray((v + 1) % 256 for v in data[:size]) + bytearray([91] * (len(data) - size))
        assert counts[:3] == (commands, commands * size, commands * size)
    assert results[1].report["storage"]["copy_elimination"]["eliminated_exact_copies"] == 1
    banks, reservations, external, adapter = bound(model)
    calls = []
    for call in results[1].kernel.body:
        effects = adapter.project(call)
        accesses = tuple(replace(a, start=8) if a.space in SPACES[:2] else a for a in effects.accesses)
        calls.append(adapter.rewrite(call, accesses))
    # All writes/reads consistently use the wrong lower address. Provenance
    # equivalence alone must not authorize an address outside the typed alias.
    with pytest.raises(ValueError, match="address|physical|bank|window"):
        verify_storage_dataflow(source, replace(source, body=tuple(calls)), objects, banks, adapter,
                                eliminate_private_copies=True, external_windows=external,
                                private_ownership=True, reservations=reservations)


def test_exact_copy_cannot_hide_additional_execution_effects(tmp_path):
    def mutate(facts, software, storage):
        software["isa"]["resources"].append(dict(name="execution_event", kind="execution", unit="events"))
        software["isa"]["instructions"][0]["effects"]["writes"].append("execution_event")
    with pytest.raises(ValueError, match="exact copy cannot discard additional"):
        inputs(tmp_path, mutate=mutate)


def test_storage_rewrite_cannot_change_the_original_source_anchor(tmp_path, monkeypatch):
    from merlin.compile.scheduling import schedule_pipeline
    model = list(inputs(tmp_path))
    source = model[1]
    original = schedule_pipeline.allocate_temporary_storage
    def mutate_source(*args, **kwargs):
        result = original(*args, **kwargs)
        # Even a valid candidate cannot excuse a change to the borrowed source
        # anchor after reconstruction but before the mutating pass returns.
        object.__setattr__(source, "name", "modified_during_storage")
        return result
    monkeypatch.setattr(schedule_pipeline, "allocate_temporary_storage", mutate_source)
    with pytest.raises(ValueError, match="source schedule changed"):
        compile_case(model)
