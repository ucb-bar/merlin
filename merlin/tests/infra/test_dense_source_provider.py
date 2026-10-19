"""Private strided ABI and observed primitive support for source dense tests."""

import ctypes
import json
import shutil
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
from test_dense_source_catalog import implementation
from test_dense_source_trace import KINDS, check_trace
from test_pointwise_source_provider import Descriptor

from merlin.common.digest import sha256_file
from merlin.runtime.host_provider import HostProviderBuild, HostProviderObject, read_host_function_abi
from merlin.transforms.dense_planning import DenseResources


class DenseProvider:
    def __init__(self, work, model, trace, *, streaming=False, mutation=None):
        self.work, self.trace, self.streaming, self.mutation = work, trace, streaming, mutation
        self.selected, _, self.storage, adapter = model
        self.adapter = replace(adapter, qualification_identity="neutral-observed-strided-dense-v1")
        self.records = []
        work.mkdir()

    def selection(self):
        from merlin.llvmlower.source_dense import SourceDenseLowering

        resources = (
            DenseResources(1 << 20, 1 << 20, 1, 1, 2, partial_output_tiles=True, partial_output_blocks=True)
            if self.streaming
            else None
        )
        return SourceDenseLowering(
            selected_contracts=self.selected,
            storage=self.storage,
            adapter=self.adapter,
            execution_domain="host",
            allocation_alignment=64,
            static_budget=80,
            prepare=self.prepare,
            build=self.build,
            private_ownership=True,
            external_buffers_disjoint=True,
            resources=resources,
            max_source_elements=4096,
            verification_timeout_ms=20_000,
        )

    def prepare(self, bindings, private_directory):
        self.bindings = tuple(bindings)
        for index, binding in enumerate(bindings):
            result = binding.compilation
            lhs, rhs = result.operation.inputs
            m, k = lhs.type.get_shape()
            _, n = rhs.type.get_shape()
            problem = self.selected.dense.problem(m, k, n)
            symbol = binding.writer.symbol + "_primitive"
            bridge = "_mlir_ciface_" + binding.writer.borrowed_symbol
            directory = self.work / str(index)
            directory.mkdir()
            source = directory / "provider.c"

            def entry(name, names):
                prefix = f"r=(r+1)%{problem.m};" if self.mutation == "address" and name == "lhs_load" else ""
                args = ",".join((*names, *("0" for _ in range(8 - len(names)))))
                return prefix + f"dense_trace({KINDS.index(name)},{len(names)},{args});"

            c = implementation(
                result,
                problem,
                self.storage,
                self.selected.dense.dtypes,
                symbol,
                lose_continuation=self.mutation == "continuation",
                primitive_entry=entry,
                bridge_override=self.wrapper(index, symbol, bridge, problem),
            )
            c = c.replace(
                "static void " + symbol + "_implementation(",
                'static __attribute__((noinline,used,retain,section(".merlin_dense.'
                + symbol
                + '"))) void '
                + symbol
                + "_implementation(",
            )
            source.write_text(self.instrumentation(index, symbol) + c)
            self.records.append(
                dict(
                    index=index,
                    result=result,
                    problem=problem,
                    symbol=symbol,
                    bridge=bridge,
                    source=source,
                    section=".merlin_dense." + symbol,
                )
            )
            (directory / "scheduling.json").write_text(
                json.dumps(
                    dict(
                        source_passes=[asdict(p) for p in result.source_passes],
                        realization=result.compiled.realization,
                        scope="source schedule diagnostic, not proof of actual C implementation",
                    ),
                    indent=2,
                )
            )
        qualify(self)

    @staticmethod
    def instrumentation(index, symbol):
        return f"""
#include <stdint.h>
static int64_t trace_data[2048][10];
static uint64_t trace_used,packing_read,packing_write,dispatches;
#ifdef DENSE_TARGET_CONSOLE
extern void htif_puts(const char *);
extern void htif_putd(long);
#endif
static void dense_trace(int op,int n,int64_t a,int64_t b,int64_t c,int64_t d,int64_t e,int64_t f,int64_t g,int64_t h){{
 if(trace_used>=2048)__builtin_trap();
 int64_t values[10]={{op,n,a,b,c,d,e,f,g,h}};
 for(int i=0;i<10;i++)trace_data[trace_used][i]=values[i];trace_used++;
#ifdef DENSE_TARGET_CONSOLE
 htif_puts("MERLIN_DENSE_TRACE {index}");
 for(int i=0;i<n+2;i++){{htif_puts(" ");htif_putd((long)values[i]);}}htif_puts("\\n");
#endif
}}
uint64_t {symbol}_trace_size(void){{return trace_used;}}
int64_t {symbol}_trace(int row,int field){{
 if(row<0||row>=trace_used||field<0||field>=10)__builtin_trap();return trace_data[row][field];}}
"""

    @staticmethod
    def wrapper(index, symbol, bridge, p):
        return f"""
typedef struct {{void *allocated,*aligned;int64_t offset,sizes[2],strides[2];}} descriptor;
static void byte_copy(void *d,const void *s,int n){{
 unsigned char*a=d;const unsigned char*b=s;for(int i=0;i<n;i++)a[i]=b[i];}}
void {symbol}_observe_reset(void){{trace_used=packing_read=packing_write=dispatches=0;{symbol}_reset();}}
uint64_t {symbol}_observe_count(int i){{return i<3?{symbol}_count(i):i==3?packing_read:i==4?packing_write:dispatches;}}
void {bridge}(const descriptor *a,const descriptor *b,descriptor *out){{
 unsigned char lhs[{p.m * p.k * p.lhs_bytes + 128}] __attribute__((aligned(64)));
 unsigned char rhs[{p.k * p.n * p.rhs_bytes + 128}] __attribute__((aligned(64)));
 unsigned char output[{p.m * p.n * 4 + 128}] __attribute__((aligned(64)));
 if(a->sizes[0]!={p.m}||a->sizes[1]!={p.k}||b->sizes[0]!={p.k}||b->sizes[1]!={p.n}||
 out->sizes[0]!={p.m}||out->sizes[1]!={p.n})__builtin_trap();
 if(a->offset<0||b->offset<0||out->offset<0||a->strides[0]<=0||a->strides[1]<=0||
 b->strides[0]<=0||b->strides[1]<=0||out->strides[0]<=0||out->strides[1]<=0)__builtin_trap();
 if(!a->aligned||!b->aligned||!out->aligned||(uintptr_t)a->aligned%{p.lhs_bytes}||
    (uintptr_t)b->aligned%{p.rhs_bytes}||(uintptr_t)out->aligned%4)__builtin_trap();
 for(int i=0;i<sizeof(lhs);i++)lhs[i]=167;
 for(int i=0;i<sizeof(rhs);i++)rhs[i]=167;
 for(int i=0;i<sizeof(output);i++)output[i]=167;
 for(int i=0;i<{p.m};i++)for(int j=0;j<{p.k};j++){{
  byte_copy(lhs+64+(i*{p.k}+j)*{p.lhs_bytes},(const unsigned char*)a->aligned+
       (a->offset+i*a->strides[0]+j*a->strides[1])*{p.lhs_bytes},{p.lhs_bytes});
  packing_read+={p.lhs_bytes};packing_write+={p.lhs_bytes};
 }}
 for(int i=0;i<{p.k};i++)for(int j=0;j<{p.n};j++){{
  byte_copy(rhs+64+(i*{p.n}+j)*{p.rhs_bytes},(const unsigned char*)b->aligned+
       (b->offset+i*b->strides[0]+j*b->strides[1])*{p.rhs_bytes},{p.rhs_bytes});
  packing_read+={p.rhs_bytes};packing_write+={p.rhs_bytes};
 }}
 {symbol}(lhs+64,rhs+64,output+64);
 if(!{symbol}_guards())__builtin_trap();
 for(int i=0;i<sizeof(lhs);i++)if((i<64||i>={p.m * p.k * p.lhs_bytes + 64})&&lhs[i]!=167)__builtin_trap();
 for(int i=0;i<sizeof(rhs);i++)if((i<64||i>={p.k * p.n * p.rhs_bytes + 64})&&rhs[i]!=167)__builtin_trap();
 for(int i=0;i<sizeof(output);i++)if((i<64||i>={p.m * p.n * 4 + 64})&&output[i]!=167)__builtin_trap();
 for(int i=0;i<{p.m};i++)for(int j=0;j<{p.k};j++)for(int t=0;t<{p.lhs_bytes};t++)
  if(lhs[64+(i*{p.k}+j)*{p.lhs_bytes}+t]!=((const unsigned char*)a->aligned)
     [(a->offset+i*a->strides[0]+j*a->strides[1])*{p.lhs_bytes}+t])__builtin_trap();
 for(int i=0;i<{p.k};i++)for(int j=0;j<{p.n};j++)for(int t=0;t<{p.rhs_bytes};t++)
  if(rhs[64+(i*{p.n}+j)*{p.rhs_bytes}+t]!=((const unsigned char*)b->aligned)
     [(b->offset+i*b->strides[0]+j*b->strides[1])*{p.rhs_bytes}+t])__builtin_trap();
 for(int i=0;i<{p.m};i++)for(int j=0;j<{p.n};j++){{
  byte_copy((unsigned char*)out->aligned+4*(out->offset+i*out->strides[0]+j*out->strides[1]),
       output+64+4*(i*{p.n}+j),4);packing_read+=4;packing_write+=4;
 }}
 dispatches++;
#ifdef DENSE_TARGET_CONSOLE
 htif_puts("MERLIN_DENSE_COUNTS {index}");
 for(int i=0;i<6;i++){{htif_puts(" ");htif_putd((long){symbol}_observe_count(i));}}htif_puts("\\n");
#endif
}}
"""

    def build(self, bindings, context):
        assert tuple(bindings) == self.bindings
        witness = self.work / "native_qualification.json"
        qualified = json.loads(witness.read_text())
        assert qualified["sources"] == {str(r["source"]): sha256_file(r["source"]) for r in self.records}
        objects = []
        for record in self.records:
            result, symbol, source = record["result"], record["symbol"], record["source"]
            result.revalidate()
            llvm, obj = context.workdir / (symbol + ".ll"), context.workdir / (symbol + ".o")
            commands = []
            for flags, destination in ((["-S", "-emit-llvm"], llvm), (["-c"], obj)):
                argv = [
                    str(context.compiler),
                    *context.compiler_flags,
                    "-DDENSE_TARGET_CONSOLE",
                    *flags,
                    str(source),
                    "-o",
                    str(destination),
                ]
                context.compile(argv, inputs=(source,), output=destination)
                commands.append(tuple(argv))
            defined = [
                symbol,
                record["bridge"],
                *(
                    symbol + suffix
                    for suffix in (
                        "_count",
                        "_reset",
                        "_guards",
                        "_trace_size",
                        "_trace",
                        "_observe_count",
                        "_observe_reset",
                    )
                ),
            ]
            objects.append(
                HostProviderObject(
                    obj,
                    sha256_file(obj),
                    llvm,
                    sha256_file(llvm),
                    tuple(read_host_function_abi(llvm, s) for s in defined),
                    tuple(read_host_function_abi(llvm, s) for s in ("htif_puts", "htif_putd")),
                    tuple((p, sha256_file(p)) for p in (source, context.compiler.resolve())),
                    tuple(commands),
                    Path.cwd(),
                )
            )
        return HostProviderBuild(
            sha256_file(context.prepared_path),
            sha256_file(context.model_llvm_path),
            sha256_file(context.model_object_path),
            tuple(objects),
            sha256_file(witness),
            sha256_file(witness),
        )


def qualify(provider):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("native ABI qualifier unavailable")
    sources = [r["source"] for r in provider.records]
    library = provider.work / "qualification.so"
    argv = [cc, "-O2", "-fno-builtin", "-shared", "-fPIC", *map(str, sources), "-o", str(library)]
    subprocess.run(argv, check=True, capture_output=True)
    retained = provider.work / ("qualification-" + sha256_file(library) + ".so")
    library.replace(retained)
    native = ctypes.CDLL(str(retained))
    calls = []
    previous = {r["index"]: [0] * 6 for r in provider.records}
    for record in provider.records:
        symbol = record["symbol"]
        reset = getattr(native, symbol + "_observe_reset")
        reset()
        size = getattr(native, symbol + "_trace_size")
        size.restype = ctypes.c_uint64
        query = getattr(native, symbol + "_trace")
        query.argtypes = [ctypes.c_int] * 2
        query.restype = ctypes.c_int64
        count = getattr(native, symbol + "_observe_count")
        count.argtypes = [ctypes.c_int]
        count.restype = ctypes.c_uint64
        record["native"] = (size, query, count)
    for index, left, right, expected in provider.trace:
        record = provider.records[index]
        size, query, count = record["native"]
        start = size()
        buffers = []
        originals = []
        descriptors = []
        for array in (left, right, np.full(expected.shape, 0x71717171, dtype="<i4")):
            rows, columns = array.shape
            memory = np.full((rows + 2, 2 * columns + 5), 71, dtype=array.dtype)
            memory[1 : rows + 1, 1 : 2 * columns + 1 : 2] = array
            buffers.append(memory)
            originals.append(memory.copy())
            descriptors.append(
                Descriptor(
                    memory.ctypes.data,
                    memory.ctypes.data,
                    2 * columns + 6,
                    (ctypes.c_int64 * 2)(rows, columns),
                    (ctypes.c_int64 * 2)(2 * columns + 5, 2),
                )
            )
        fn = getattr(native, record["bridge"])
        fn.argtypes = [ctypes.POINTER(Descriptor)] * 3
        fn(*(ctypes.byref(d) for d in descriptors))
        observed = []
        for row in range(start, size()):
            length = query(row, 1)
            observed.append([query(row, field) for field in range(length + 2)])
        effects = check_trace(observed, record["problem"], provider.storage, record["result"].compiled.kernel.args)
        for got, want in zip(buffers[:2], originals[:2], strict=True):
            np.testing.assert_array_equal(got, want)
        originals[-1][1 : expected.shape[0] + 1, 1 : 2 * expected.shape[1] + 1 : 2] = expected
        np.testing.assert_array_equal(buffers[-1], originals[-1])
        counts = [count(i) for i in range(6)]
        delta = [new - old for new, old in zip(counts, previous[index], strict=True)]
        assert delta[:3] == [effects[key] for key in ("commands", "requested_read_bytes", "requested_write_bytes")]
        p = record["problem"]
        packed = p.m * p.k * p.lhs_bytes + p.k * p.n * p.rhs_bytes + p.m * p.n * 4
        assert delta[3:] == [packed, packed, 1]
        report = record["result"].compiled.realization
        assert delta[:3] == [
            report[key] for key in ("executed_instruction_calls", "requested_load_bytes", "requested_store_bytes")
        ]
        previous[index] = counts
        calls.append(dict(index=index, trace=observed, effects=effects, counters=counts))
    record = dict(
        sources={str(p): sha256_file(p) for p in sources},
        library_path=str(retained),
        library_sha256=sha256_file(retained),
        compiler=str(cc),
        compiler_sha256=sha256_file(Path(cc).resolve()),
        commands=[argv],
        calls=calls,
        scope="actual strided ABI/native primitive execution, not target qualification or generic C/ELF conformance",
        alignment="actual external base pointers checked for element alignment; private raw-kernel buffers aligned64",
        packing_scope="requested logical copy bytes; guard and logging accesses not included",
    )
    (provider.work / "native_qualification.json").write_text(json.dumps(record, indent=2))
    return record
