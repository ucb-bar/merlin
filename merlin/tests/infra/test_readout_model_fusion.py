"""Execute a complete two-layer integer graph with physical readout fusion.

Neutral instruction implementations establish the shared mechanism only. Bias
is explicitly an elementwise matrix; no column-broadcast/packing capability is
inferred. Separate addressable accumulators are a selected test capability.
"""

import ctypes
import json
import shutil
import struct
import subprocess
import time
from dataclasses import replace

import numpy as np
import pytest

from merlin.compile.scheduling.fusion import FusionCapability
from merlin.compile.scheduling.fusion_semantics import ElementwiseSemantics
from merlin.compile.scheduling.program_limits import (
    AddressWindow,
    DecodedInstruction,
    EncodedAccess,
    ProgramDecoder,
    ProgramLimits,
)
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets, ScheduleEncoding, compile_schedule
from merlin.compile.scheduling.temporary_storage import StorageAdapter, StorageEffects, TemporaryObject
from merlin.sched.ir import Call, Const, Kernel, TensorArg, evaluate
from merlin.semantic_compiler.allocation_types import StorageBank
from merlin.verify.epilogue_applicability import ReadoutCapability

SPACES = ("acc", "stage", "activation", "x", "w0", "w1", "b0", "b1", "hidden", "y")
KEYS = ("source", "destination", "m", "k", "n", "src", "dst", "weights", "bias", "width")
OPS = ("dot", "readout", "finish", "fused", "snapshot")
WORD = struct.Struct("<B" + "I" * len(KEYS))
IDENTITY = b"builtin.module {func.func @forward(%x:i32)->i32 {func.return %x:i32}}"
EPILOGUE = b"""builtin.module {func.func @forward(%x:i32,%bias:i32)->i8 {
 %sum=arith.addi %x,%bias:i32
 %low=arith.constant -128:i32
 %high=arith.constant 127:i32
 %lower=arith.maxsi %sum,%low:i32
 %bounded=arith.minsi %lower,%high:i32
 %result=arith.trunci %bounded:i32 to i8
 func.return %result:i8
}}"""


def command(op, **kwargs):
    return Call(op, tuple((key, Const(kwargs.get(key, 0))) for key in KEYS))


def params(call):
    return {key: evaluate(call.arg(key), {}) for key in KEYS}


def effects(call):
    p = params(call)
    count = p["m"] * p["n"]
    source, dest = SPACES[p["src"]], SPACES[p["dst"]]
    reads = (EncodedAccess(source, p["source"], count * (1 if call.instr == "snapshot" else 4)),)
    size, semantic = count, None
    if call.instr == "dot":
        reads = (
            EncodedAccess(source, p["source"], p["m"] * p["k"] * p["width"]),
            EncodedAccess(SPACES[p["weights"]], 0, p["k"] * p["n"] * p["width"]),
        )
        size = 4 * count
    elif call.instr == "readout":
        size = 4 * count
        semantic = ElementwiseSemantics(IDENTITY, count, ("readout",), ("little", "little"))
    elif call.instr in {"finish", "fused"}:
        reads += (EncodedAccess(SPACES[p["bias"]], 0, 4 * count),)
        semantic = ElementwiseSemantics(
            EPILOGUE,
            count,
            (("readout",) if call.instr == "fused" else ()) + ("bias", "clamp", "cast"),
            ("little",) * 3,
            "physical-finish" if call.instr == "fused" else None,
        )
    elif call.instr != "snapshot":
        return None
    return StorageEffects(
        repr((call.instr, p["m"], p["k"], p["n"], p["width"], p["src"], p["dst"], p["weights"], p["bias"])),
        (*reads, EncodedAccess(dest, p["destination"], size, True)),
        "synchronous",
        exact_copy=call.instr == "snapshot",
        elementwise=semantic,
    )


def rewrite(call, accesses):
    changes = {"source": Const(accesses[0].start), "destination": Const(accesses[-1].start)}
    return replace(call, args=tuple((key, changes.get(key, value)) for key, value in call.args))


def encoding(adapter):
    def encode(k):
        return b"".join(WORD.pack(OPS.index(c.instr), *params(c).values()) for c in k.body)

    def decode(image):
        return tuple(command(OPS[row[0]], **dict(zip(KEYS, row[1:], strict=True))) for row in WORD.iter_unpack(image))

    def instructions(image):
        return tuple(
            DecodedInstruction(
                i * WORD.size, WORD.size, effects(c).accesses, completion="synchronous", relocatable=True
            )
            for i, c in enumerate(decode(image))
        )

    return ScheduleEncoding(
        encode,
        decode,
        ProgramDecoder(instructions, adapter.configuration_identity, "synthetic"),
        adapter.numerical_policy_identity,
        "synthetic",
    )


def graph(m, k, n, p, width, separate):
    adapter = StorageAdapter(
        effects, rewrite, f"neutral-readout-{width}-{separate}", "signed-i32-wrap-clamp-i8", "synthetic"
    )
    cap = FusionCapability(
        ReadoutCapability("physical-finish", frozenset({"readout", "bias", "clamp", "cast"}), "synthetic"),
        lambda a, b: command("fused", **(params(b) | {"source": params(a)["source"], "src": params(a)["src"]})),
        adapter.configuration_identity,
        adapter.numerical_policy_identity,
        "synthetic",
    )
    args = (
        TensorArg("x", (m, k), "i8" if width == 1 else "i16", "read"),
        TensorArg("w0", (k, n), "i8" if width == 1 else "i16", "read"),
        TensorArg("w1", (n, p), "i8", "read"),
        TensorArg("b0", (m, n), "i32", "read"),
        TensorArg("b1", (m, p), "i32", "read"),
        TensorArg("hidden", (m, n), "i8", "write"),
        TensorArg("y", (m, p), "i8", "write"),
    )
    body = (
        command("dot", m=m, k=k, n=n, src=3, dst=0, weights=4, width=width, destination=1024),
        command("readout", m=m, n=n, src=0, dst=1, source=1024, destination=3072),
        command("finish", m=m, n=n, src=1, dst=2, source=3072, destination=5120, bias=6),
        command("snapshot", m=m, n=n, src=2, dst=8, source=5120),
        command("dot", m=m, k=n, n=p, src=2, dst=0, source=5120, destination=2048, weights=5, width=1),
        command("readout", m=m, n=p, src=0, dst=1, source=2048, destination=4096),
        command("finish", m=m, n=p, src=1, dst=9, source=4096, bias=7),
    )
    objects = (
        TemporaryObject("a0", "acc", 1024, 4 * m * n, 4),
        TemporaryObject("a1", "acc", 2048, 4 * m * p, 4),
        TemporaryObject("s0", "stage", 3072, 4 * m * n, 4),
        TemporaryObject("s1", "stage", 4096, 4 * m * p, 4),
        TemporaryObject("h", "activation", 5120, m * n),
    )
    banks = (
        StorageBank("acc", "accumulator", 1024, "bytes", 4),
        StorageBank("stage", "scratch", 1024, "bytes", 4),
        StorageBank("activation", "activation" if separate else "scratch", 1024, "bytes", 4),
    )
    windows = tuple(AddressWindow(a.name, 0, a.nbytes) for a in args)
    limits = ProgramLimits(
        16 * WORD.size, 16, 1, (*windows, *(AddressWindow(b.name, 0, b.capacity) for b in banks)), ()
    )
    return (
        Kernel("two_layer_integer", args, body),
        objects,
        banks,
        adapter,
        cap,
        encoding(adapter),
        limits,
        dict(private_ownership=True, external_windows=windows),
    )


_EXECUTOR = r"""
#include <stdint.h>
#include <stddef.h>
#include <string.h>
static uint32_t number(const uint8_t *p){uint32_t v=0;for(int j=0;j<4;j++)v|=(uint32_t)p[j]<<(8*j);return v;}
static uint32_t get(const uint8_t *p,int width){uint32_t v=0;for(int j=0;j<width;j++)v|=(uint32_t)p[j]<<(8*j);return v;}
static int32_t signed32(uint32_t x){return x<=INT32_MAX?(int32_t)x:(int32_t)((int64_t)x-4294967296LL);}
static int32_t signed_element(uint32_t x,int width){
 return width==1?(x<128?x:(int32_t)x-256):(x<32768?x:(int32_t)x-65536);
}
int execute(const uint8_t *image,size_t length,uint8_t **spaces,const uint64_t *capacities,uint64_t *counts,int wrong){
 uint64_t commands=0,reads=0,writes=0;uint8_t left[2048],right[2048];
 if(length%41)return 1;
 for(size_t pos=0;pos<length;pos+=41){
  unsigned op=image[pos];uint32_t a[10];for(int j=0;j<10;j++)a[j]=number(image+pos+1+4*j);
  uint32_t src=a[0],dst=a[1],m=a[2],k=a[3],n=a[4],r=a[5],w=a[6],weight=a[7],bias=a[8],width=a[9];
  uint64_t elements=(uint64_t)m*n;commands++;
  if(r>=10||w>=10||weight>=10||bias>=10||!m||!n)return 2;
  if(op==0){
   uint64_t nl=(uint64_t)m*k*width,nr=(uint64_t)k*n*width;
   if((width!=1&&width!=2)||!k||nl>2048||nr>2048||src+nl>capacities[r]||nr>capacities[weight]||
      dst+4*elements>capacities[w])return 3;
   // Explicit DMA staging: count each byte actually requested once, then
   // compute against primitive-private staged operands, not external memory.
   for(uint64_t j=0;j<nl;j++){left[j]=spaces[r][src+j];reads++;}
   for(uint64_t j=0;j<nr;j++){right[j]=spaces[weight][j];reads++;}
   for(uint32_t i=0;i<m;i++)for(uint32_t j=0;j<n;j++){
    uint32_t sum=0;
    for(uint32_t q=0;q<k;q++)sum+=(uint32_t)(signed_element(get(left+((uint64_t)i*k+q)*width,width),width)*
                                                          signed_element(get(right+((uint64_t)q*n+j)*width,width),width));
    for(int b=0;b<4;b++){spaces[w][dst+4*((uint64_t)i*n+j)+b]=(uint8_t)(sum>>(8*b));writes++;}
   }
  } else if(op==4){
   if(src+elements>capacities[r]||dst+elements>capacities[w])return 4;
   for(uint64_t j=0;j<elements;j++){spaces[w][dst+j]=spaces[r][src+j];reads++;writes++;}
  } else if(op==1||op==2||op==3){
   if(src+4*elements>capacities[r]||dst+(op==1?4:1)*elements>capacities[w]||
      (op!=1&&4*elements>capacities[bias]))return 5;
   for(uint64_t j=0;j<elements;j++){
    uint32_t value=0;for(int b=0;b<4;b++){value|=(uint32_t)spaces[r][src+4*j+b]<<(8*b);reads++;}
    if(op==1){for(int b=0;b<4;b++){spaces[w][dst+4*j+b]=(uint8_t)(value>>(8*b));writes++;}}
    else {
     uint32_t addition=0;for(int b=0;b<4;b++){addition|=(uint32_t)spaces[bias][4*j+b]<<(8*b);reads++;}
     int64_t sum=(op==3&&wrong)?(int64_t)signed32(value)+signed32(addition):signed32(value+addition);
     int32_t bounded=sum < -128 ? -128 : sum > 127 ? 127 : sum;
     spaces[w][dst+j]=(uint8_t)bounded;writes++;
    }
   }
  } else return 6;
 }
 counts[0]=commands;counts[1]=reads;counts[2]=writes;return 0;
}
"""


@pytest.fixture
def native(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("native compiler unavailable")
    source, library = tmp_path / "execute.c", tmp_path / "execute.so"
    source.write_text(_EXECUTOR)
    subprocess.run([cc, "-O2", "-shared", "-fPIC", str(source), "-o", str(library)], check=True, capture_output=True)
    fn = ctypes.CDLL(str(library)).execute
    bytep = ctypes.POINTER(ctypes.c_uint8)
    fn.argtypes = [
        bytep,
        ctypes.c_size_t,
        ctypes.POINTER(bytep),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int,
    ]

    def execute(result, inputs, separate, *, wrong=False):
        windows = {w["space"]: w["size"] for w in result.report["resources"]["external_windows"]}
        buffers = [(ctypes.c_uint8 * 1024)(*([173] * 1024)), (ctypes.c_uint8 * 1024)(*([173] * 1024))]
        buffers.append((ctypes.c_uint8 * 1024)(*([173] * 1024)) if separate else buffers[1])
        originals = {}
        for name in SPACES[3:]:
            payload = inputs[name].tobytes() if name in inputs else bytes([91] * windows[name])
            assert len(payload) == windows[name]
            originals[name] = payload + bytes([199] * 8)
            buffers.append((ctypes.c_uint8 * (len(payload) + 8)).from_buffer_copy(originals[name]))
        pointers = (bytep * 10)(*(ctypes.cast(b, bytep) for b in buffers))
        capacities = (ctypes.c_uint64 * 10)(1024, 1024, 1024, *(windows[n] for n in SPACES[3:]))
        image = b"".join(result.program.segments)
        encoded = (ctypes.c_uint8 * len(image)).from_buffer_copy(image)
        counters = (ctypes.c_uint64 * 3)()
        assert fn(encoded, len(image), pointers, capacities, counters, wrong) == 0
        for i, name in enumerate(SPACES[3:], 3):
            assert bytes(buffers[i])[windows[name] :] == bytes([199] * 8)
            if name in inputs:
                assert bytes(buffers[i]) == originals[name]
        return {name: bytes(buffers[SPACES.index(name)])[: windows[name]] for name in ("hidden", "y")}, tuple(counters)

    return execute


def inputs_and_reference(m, k, n, p, width):
    rng = np.random.default_rng(43)
    data = {
        "x": rng.integers(-32, 33, (m, k)).astype("i1" if width == 1 else "<i2"),
        "w0": rng.integers(-32, 33, (k, n)).astype("i1" if width == 1 else "<i2"),
        "w1": rng.integers(-3, 4, (n, p)).astype("i1"),
        "b0": rng.integers(-64, 65, (m, n)).astype("<i4"),
        "b1": rng.integers(-64, 65, (m, p)).astype("<i4"),
    }
    data["b0"].flat[::3] = (1 << 31) - 2
    data["b0"].flat[1::3] = -(1 << 31) + 2
    data["b1"].flat[::3] = (1 << 31) - 2

    def finish(left, right, bias):
        # Wide independent arithmetic explicitly reconstructs source wrapping
        # before signed saturation; it does not call the device implementation.
        wide = left.astype(np.int64) @ right.astype(np.int64) + bias.astype(np.int64)
        wrapped = (wide + (1 << 31)) % (1 << 32) - (1 << 31)
        return np.clip(wrapped, -128, 127).astype("i1")

    hidden = finish(data["x"], data["w0"], data["b0"])
    y = finish(hidden, data["w1"], data["b1"])
    return data, {"hidden": hidden.tobytes(), "y": y.tobytes()}


@pytest.mark.parametrize("shape,width,separate", [((3, 7, 5, 4), 1, False), ((2, 5, 11, 3), 2, True)])
def test_complete_integer_graph_realizes_readout_epilogues(shape, width, separate, native, record_property):
    kernel, objects, banks, adapter, cap, codec, limits, model = graph(*shape, width, separate)
    inputs, expected = inputs_and_reference(*shape, width)
    results = []
    measurements = []
    for fusion in ((), (cap,)):
        started = time.perf_counter()
        result = compile_schedule(kernel, objects, banks, adapter, codec, limits, fusion_capabilities=fusion, **model)
        compile_seconds = time.perf_counter() - started
        started = time.perf_counter()
        actual, counters = native(result, inputs, separate)
        execution_seconds = time.perf_counter() - started
        assert actual == expected
        final = result.report["final_conformance"]
        assert counters == (final["actual_commands"], final["requested_read_bytes"], final["requested_write_bytes"])
        assert len(b"".join(result.program.segments)) == counters[0] * WORD.size
        results.append((result, counters))
        measurements.append(
            dict(
                compile_seconds=compile_seconds,
                compiler_peak_rss="not_measured",
                encoded_bytes=len(b"".join(result.program.segments)),
                commands=counters[0],
                requested_read_bytes=counters[1],
                requested_write_bytes=counters[2],
                native_harness_wall_seconds=execution_seconds,
            )
        )
    (baseline, before), (optimized, after) = results
    m, k, n, p = shape
    assert before[0] == 7 and after[0] == 5
    assert before[1] - after[1] == before[2] - after[2] == 4 * m * (n + p)
    assert len(optimized.report["storage"]["fusion"]["checked_steps"]) == 2
    assert baseline.report["context"]["library"] == optimized.report["context"]["library"]
    assert [c.instr for c in optimized.kernel.body] == ["dot", "fused", "snapshot", "dot", "fused"]
    # A physically different readout that skips intermediate i32 wrapping
    # still agrees with plausible declarations; independent execution rejects it.
    wrong, _ = native(optimized, inputs, separate, wrong=True)
    assert wrong != expected
    record_property(
        "readout_graph_pair",
        json.dumps(
            dict(
                shape=shape,
                input_bytes=width,
                separate_memories=separate,
                source_sha256=kernel.digest(),
                library_sources_sha256=optimized.report["library_sources_sha256"],
                measurements=measurements,
                execution="neutral native instruction implementation; no hardware timing",
            )
        ),
    )


def test_missing_bias_capability_retains_both_materializations(native):
    shape = (3, 7, 5, 4)
    kernel, objects, banks, adapter, cap, codec, limits, model = graph(*shape, 1, False)
    unavailable = replace(cap, readout=replace(cap.readout, applies=frozenset({"readout", "clamp", "cast"})))
    result = compile_schedule(
        kernel, objects, banks, adapter, codec, limits, fusion_capabilities=(unavailable,), **model
    )
    data, expected = inputs_and_reference(*shape, 1)
    actual, counts = native(result, data, False)
    assert actual == expected and counts[0] == 7
    assert [c.instr for c in result.kernel.body].count("readout") == 2


def test_static_growth_budget_uses_realized_complete_graph(native):
    shape = (3, 7, 5, 4)
    kernel, objects, banks, adapter, cap, codec, limits, model = graph(*shape, 1, False)
    tight = replace(limits, max_bytes=5 * WORD.size, max_instructions=5)
    optimized = compile_schedule(kernel, objects, banks, adapter, codec, tight, fusion_capabilities=(cap,), **model)
    data, expected = inputs_and_reference(*shape, 1)
    assert native(optimized, data, False)[0] == expected
    with pytest.raises(ValueError):
        compile_schedule(
            kernel,
            objects,
            banks,
            adapter,
            codec,
            tight,
            budgets=ScheduleBudgets(max_fusions=0),
            fusion_capabilities=(cap,),
            **model,
        )
