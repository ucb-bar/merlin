"""Selected final bounds through ordinary scheduling and independent native bytes."""

import copy
import ctypes
import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import replace

import pytest
from test_private_copy_elimination import command
from test_selected_storage_contracts import inputs as storage_inputs
from test_selected_storage_contracts import provider_project

from merlin.compile.scheduling.program_limits import (
    Continuation,
    DecodedInstruction,
    EncodedAccess,
    EncodedBranch,
    ProgramDecoder,
    finalize_program,
)
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets, compile_schedule

QUALIFICATION = "independent-neutral-launch-completion-v1"


def inputs(tmp_path, *, width=7, separate=False, word_format="<BBBQQQ", count=6, branch=False,
           configuration=False, mutate=None):
    def declare(facts, software, storage):
        word_bytes = facts["word_bits"] // 8
        facts.update(program_bytes=count * word_bytes, program_count=count, instruction_alignment=1,
                     branch_scale=word_bytes, state_fields=1, state_index=0)
        names = ("copy", "increment", "async_copy", "wait")
        if branch:
            facts["codes"].append(4)
            instruction = copy.deepcopy(software["isa"]["instructions"][3])
            instruction["name"] = "branch"
            instruction["semantics"]["primitive"] = "neutral.branch"
            instruction["encoding"][0]["fixed"] = {"fact": ["facts", "codes", 4]}
            next(a for a in instruction["operands"] if a["name"] == "source")["type"]["signed"] = True
            instruction["effects"] = dict(reads=["output"], writes=["output"])
            software["isa"]["instructions"].append(instruction)
            storage["instructions"]["branch"] = dict(
                accesses=[dict(space=dict(resource="output"), start=dict(argument="destination"),
                               size=dict(constant=1), writes=write) for write in (False, True)],
                completion="synchronous", token=None, waits=[], exact_copy=False)
            names += ("branch",)
        if configuration:
            facts["codes"].append(5)
            instruction = copy.deepcopy(software["isa"]["instructions"][3])
            instruction["name"] = "set_mode"
            instruction["semantics"]["primitive"] = "neutral.set_mode"
            instruction["encoding"][0]["fixed"] = {"fact": ["facts", "codes", len(facts["codes"]) - 1]}
            instruction["effects"] = dict(reads=[], writes=["mode"])
            software["isa"]["resources"].append(dict(name="mode", kind="state", unit="fields",
                                                       capacity={"fact": ["facts", "state_fields"]}))
            software["isa"]["instructions"].append(instruction)
            software["scheduling"]["configuration"]["set_mode"] = dict(
                reads=[], writes=[dict(resource="mode", index=dict(constant=0), value=dict(argument="size"))],
                clobbers=[], ordered=True, pure_write=True)
            names += ("set_mode",)
        windows = {name: dict(start=item["start"], size=item["size"], backing=item["backing"],
                              visibility="private" if name in storage["banks"] else "external")
                   for name, item in {**storage["banks"], **storage["external"]}.items()}
        software["scheduling"]["program"] = dict(
            schema="merlin.final_program_contract.v1",
            limits={key: {"fact": ["facts", fact]} for key, fact in (
                ("max_bytes", "program_bytes"), ("max_instructions", "program_count"),
                ("instruction_alignment", "instruction_alignment"))}, windows=windows,
            branches={"relative": dict(instruction="branch", displacement="source", origin="next_instruction",
                                         scale={"fact": ["facts", "branch_scale"]})} if branch else {},
            instructions={name: dict(completion="asynchronous" if name == "async_copy" else
                                     "drain" if name == "wait" else "synchronous", relocatable=True,
                                     control_flow="conditional_relative" if name == "branch" else "fallthrough")
                          for name in names},
            continuation=dict(sequential_completion=True, starts_at_zero=True, max_segments=16,
                              preserved_backings=[w["backing"] for w in windows.values()
                                                  if w["visibility"] == "external"],
                              preserved_configuration={}))
        if mutate:
            mutate(facts, software, software["scheduling"]["program"])
    model = storage_inputs(tmp_path, width=width, separate=separate, word_format=word_format,
                           reserve=False, mutate=declare)
    selected, kernel, objects, adapter, codec, limits, word, length = model
    operations = ("copy", "increment", "async_copy", "wait", "branch", "set_mode")
    def decode(data):
        result = []
        for offset in range(0, len(data), word.size):
            op, r, w, source, destination, size = word.unpack_from(data, offset)
            if op >= len(operations):
                raise ValueError("unknown native opcode")
            name = operations[op]
            branches = ()
            if name == "branch":
                bits = 64 if word_format.endswith("Q") else 32
                displacement = source - (1 << bits) if source & (1 << (bits - 1)) else source
                accesses = (EncodedAccess("output", destination, 1), EncodedAccess("output", destination, 1, True))
                branches = (EncodedBranch("relative", displacement),)
            elif name == "set_mode":
                accesses = ()
            else:
                accesses = provider_project(command(name, source, destination, size, r, w)).accesses
            result.append(DecodedInstruction(offset, word.size, accesses, branches,
                                              "asynchronous" if name == "async_copy" else
                                              "drain" if name == "wait" else "synchronous", True))
        return tuple(result)
    decoder = ProgramDecoder(decode, selected.hardware_identity, "neutral-independent-word-effects:" + word_format)
    return selected, kernel, objects, adapter, replace(codec, program_decoder=decoder), limits, word, length


def compile_case(model, *, legacy=False, qualification=QUALIFICATION, limits=None, budgets=None, **kwargs):
    selected, kernel, objects, adapter, codec, controls, word, _length = model
    if legacy:
        banks, reservations, external, bound = selected.storage.bind(
            selected.isa, adapter, banks=None, reservations=(), external_windows=None, private_ownership=True)
        program = json.loads(selected.program.declarations_json)
        size = program["limits"]["max_instructions"]
        policy = replace(controls, max_bytes=word.size * size, max_instructions=size)
        continuation = Continuation(QUALIFICATION, ("input", "output"), True, True, 16)
        return compile_schedule(kernel, objects, banks, bound, codec, policy, continuation=continuation,
                                reservations=reservations, external_windows=external, private_ownership=True,
                                budgets=budgets or ScheduleBudgets(eliminate_copies=False), **kwargs)
    return compile_schedule(kernel, objects, None, adapter, codec, limits, selected_contracts=selected,
                            continuation_qualification_identity=qualification, private_ownership=True,
                            budgets=budgets or ScheduleBudgets(eliminate_copies=False), **kwargs)


def finalize(model, calls, *, qualification=QUALIFICATION, **kwargs):
    selected, _kernel, _objects, _adapter, codec, _limits, _word, _length = model
    from merlin.sched.ir import Kernel
    image = codec.encode(Kernel("final", (), tuple(calls)))
    return finalize_program(image, None, codec.program_decoder, selected_contracts=selected,
                            continuation_qualification_identity=qualification, **kwargs)


@pytest.fixture
def native(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native C compiler unavailable")
    source = tmp_path / "final_machine.c"
    source.write_text(r'''
#include <stdint.h>
#include <string.h>
static uint64_t number(const unsigned char *p,int width,int big) {
 uint64_t x=0; for(int i=0;i<width;i++) x|=(uint64_t)p[i]<<(8*(big?width-1-i:i));return x;
}
int run(const unsigned char *code,int bytes,int width,int big,int separate,
 unsigned char *first,unsigned char *second,unsigned char *input,unsigned char *output,
 int capacity,int external,uint64_t *counts,uint64_t *mode) {
 unsigned char *memory[4]={first,separate?second:first,input,output};
 int bounds[4]={capacity,capacity,external,external};
 int pending=0,pr=0,pw=0;uint64_t ps=0,pd=0,pn=0;
 int word=3+3*width;int64_t pc=0;memset(counts,0,3*sizeof(*counts));
 while(pc<bytes) {
  if(pc<0 || pc+word>bytes || pc%word || ++counts[0]>10000) return 1;
  int op=code[pc],r=code[pc+1],w=code[pc+2];
  uint64_t s=number(code+pc+3,width,big),d=number(code+pc+3+width,width,big),n=number(code+pc+3+2*width,width,big);
  if(op==5){*mode=n;pc+=word;continue;}
  if(op==4){
   if(d>=(uint64_t)external)return 2;counts[1]++;
   int64_t delta=width==8?(int64_t)s:(int32_t)s;
   if(output[d]){output[d]--;counts[2]++;pc+=word+delta*word;}else pc+=word;
   continue;
  }
  if(op==3){if(!pending || n!=pn)return 3;memmove(memory[pw]+pd,memory[pr]+ps,pn);pending=0;pc+=word;continue;}
  if(r>3 || w>3 || !n || s+n>(uint64_t)bounds[r] || d+n>(uint64_t)bounds[w])return 4;
  counts[1]+=n;counts[2]+=n;
  if(op==2){if(pending)return 5;pending=1;pr=r;pw=w;ps=s;pd=d;pn=n;}
  else if(op==0)memmove(memory[w]+d,memory[r]+s,n);
  else if(op==1){for(uint64_t i=0;i<n;i++)memory[w][d+i]=(unsigned char)(memory[r][s+i]+1);}
  else return 6;
  pc+=word;
 }
 return pending?7:0;
}
''')
    library_path = tmp_path / "final_machine.so"
    subprocess.run([compiler, "-O1", "-std=c11", "-shared", "-fPIC", str(source), "-o", str(library_path)],
                   check=True, capture_output=True)
    library = ctypes.CDLL(str(library_path))
    pointer = ctypes.POINTER(ctypes.c_ubyte)
    library.run.argtypes = [pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                           pointer, pointer, pointer, pointer, ctypes.c_int, ctypes.c_int,
                           ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
    def run(segments, data, output, *, width=8, big=False, separate=False, preserve_private=False,
            preserve_configuration=False, initial_mode=0):
        first, second = bytearray(192), bytearray(192)
        mode, total = ctypes.c_uint64(initial_mode), [0, 0, 0]
        def buffer(value):
            return (ctypes.c_ubyte * len(value)).from_buffer(value)
        for segment in segments:
            if not preserve_private:
                first[:] = second[:] = bytes([167] * 192)
            if not preserve_configuration:
                mode.value = 0
            code, counts = bytearray(segment), (ctypes.c_uint64 * 3)()
            status = library.run(buffer(code), len(code), width, big, separate, buffer(first), buffer(second),
                                 buffer(data), buffer(output), 192, len(data), counts, ctypes.byref(mode))
            assert status == 0, status
            total = [a + b for a, b in zip(total, counts)]
        return tuple(total), mode.value
    return run


@pytest.mark.parametrize("width,separate,word_format", [(7, False, "<BBBQQQ"), (13, True, ">BBBIII")])
def test_ordinary_final_checks_use_selected_limits_and_preserve_complete_state(
    tmp_path, native, record_property, width, separate, word_format
):
    model = inputs(tmp_path, width=width, separate=separate, word_format=word_format)
    results, times = [], []
    for legacy in (True, False):
        started = time.perf_counter()
        results.append(compile_case(model, legacy=legacy))
        times.append(time.perf_counter() - started)
    baseline, result = results
    assert baseline.report["library_sources_sha256"] == result.report["library_sources_sha256"]
    assert b"".join(baseline.program.segments) == b"".join(result.program.segments)
    assert len(baseline.program.segments) >= len(result.program.segments)
    assert separate or len(baseline.program.segments) > len(result.program.segments)
    assert list(map(len, result.program.segments)) == [6 * model[6].size] * 3
    binding = result.report["selected_contracts"]["program"]
    assert binding["hardware_byte_capacity"] == binding["hardware_instruction_capacity"] == "checked"
    source = bytearray((i * 31 + 253) % 256 for i in range(2 * model[7] + 8))
    records = []
    for arm, compiled, elapsed in zip(("explicit_capability", "selected_contract"), results, times):
        data, output = bytearray(source), bytearray([91] * len(source))
        native_seconds = 0.0
        for ordinal, segment in enumerate(compiled.program.segments):
            (tmp_path / f"{arm}-segment-{ordinal}.bin").write_bytes(segment)
        for invocation in range(3):
            before = bytes(data)
            started = time.perf_counter()
            counts, _mode = native(compiled.program.segments, data, output,
                                   width=8 if word_format.endswith("Q") else 4,
                                   big=word_format[0] == ">", separate=separate)
            native_seconds += time.perf_counter() - started
            assert data == bytearray(before)
            assert output == bytes((v + 1) % 256 for v in data[:2 * model[7]]) + bytes([91] * 8)
            assert counts == (18, 6 * model[7], 6 * model[7])
            assert compiled.report["final_conformance"]["actual_commands"] == counts[0]
            (tmp_path / f"{arm}-input-{invocation}.bin").write_bytes(before)
            (tmp_path / f"{arm}-output-{invocation}.bin").write_bytes(bytes(output))
            data[:] = output
            data[invocation] ^= 37
        records.append(dict(arm=arm, compile_seconds=elapsed, encoded_bytes=sum(map(len, compiled.program.segments)),
                            launches=len(compiled.program.segments), commands=counts[0],
                            requested_read_bytes=counts[1], requested_write_bytes=counts[2],
                            native_host_seconds_three_invocations=native_seconds,
                            image_sha256=hashlib.sha256(b"".join(compiled.program.segments)).hexdigest(),
                            segment_bytes=list(map(len, compiled.program.segments))))
    record_property("selected_program_pair", json.dumps(dict(
        measurements=records, selected_contracts=result.program.report["selected_contracts"],
        shared_library_sources=result.report["library_sources_sha256"],
        model=dict(width=width, separate=separate, word_format=word_format),
        compile_peak_memory="unmeasured", accelerator_timing="unmeasured",
        scope="complete neutral host/state graph; actual bytes, final bounds and aliases; no timing speedup claim")))
