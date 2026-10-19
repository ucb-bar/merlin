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
    def __init__(
        self, work, model, trace, *, streaming=False, mutation=None, logical_abi=False, borrow_contiguous=True
    ):
        self.work, self.trace, self.streaming, self.mutation = work, trace, streaming, mutation
        self.selected, _, self.storage, adapter = model
        self.adapter = replace(adapter, qualification_identity="neutral-observed-strided-dense-v1")
        self.records = []
        self.logical_abi, self.borrow_contiguous = logical_abi, borrow_contiguous
        work.mkdir()

    def selection(self):
        from merlin.llvmlower.source_dense import SourceDenseLowering

        resources = (
            DenseResources(1 << 20, 1 << 20, 1, 1, 2, partial_output_tiles=True, partial_output_blocks=True)
            if self.streaming
            else None
        )
        options = {}
        if self.logical_abi:
            from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects
            from merlin.llvmlower.logical_borrowed_abi import LogicalBorrowedABI

            # Original reviewed implementation premises: the fixture primitive
            # reads logical row-major A/B, fully writes C, retains no aliases,
            # frees no supplied storage, and completes before returning.
            effects = BorrowedBufferEffects("neutral-reviewed-logical-dense-effects-v1", True, True, True)
            options = dict(
                borrowed_effects=effects,
                logical_abi=LogicalBorrowedABI(
                    declaration=self.selected.dense.logical_pointer_abi,
                    qualified_effects=effects,
                    format_qualification_identity="neutral-reviewed-row-major-dense-v1",
                    max_stack_bytes=1 << 20,
                    observe_transfers=True,
                    # This observer only updates private counters/console; it
                    # retains no aliases, frees nothing, completes on return,
                    # and does not change the logical computation or buffers.
                    observer_effects=BorrowedBufferEffects(
                        "neutral-reviewed-dense-observer-effects-v1", True, True, True
                    ),
                    observer_preserves_logical_state=True,
                ),
                borrow_contiguous=self.borrow_contiguous,
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
            **options,
        )

    def prepare(self, bindings, private_directory):
        self.bindings = tuple(bindings)
        for index, binding in enumerate(bindings):
            result = binding.compilation
            lhs, rhs = result.operation.inputs
            m, k = lhs.type.get_shape()
            _, n = rhs.type.get_shape()
            problem = self.selected.dense.problem(m, k, n)
            symbol = binding.logical_kernel_symbol if self.logical_abi else binding.writer.symbol + "_primitive"
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
                bridge_override=(
                    self.logical_wrapper(index, symbol, binding.logical_observer_symbol)
                    if self.logical_abi
                    else self.wrapper(index, symbol, bridge, problem)
                ),
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
                    **(
                        dict(logical_abi_source=binding.logical_abi_source, observer=binding.logical_observer_symbol)
                        if self.logical_abi
                        else {}
                    ),
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
    def logical_wrapper(index, symbol, observer):
        """The shared source owns the descriptor bridge; this fixture owns its primitive/observer."""
        assert observer is not None
        return f"""
static uint64_t last_transfer[3];
void {symbol}_observe_reset(void){{
 trace_used=packing_read=packing_write=dispatches=0;
 for(int i=0;i<3;i++)last_transfer[i]=0;{symbol}_reset();
}}
uint64_t {symbol}_observe_count(int i){{
 return i<3?{symbol}_count(i):i==3?packing_read:i==4?packing_write:dispatches;
}}
uint64_t {symbol}_observe_transfer(int i){{if(i<0||i>=3)__builtin_trap();return last_transfer[i];}}
void {observer}(uint64_t reads,uint64_t writes,uint64_t direct){{
 if(!{symbol}_guards())__builtin_trap();
 packing_read+=reads;packing_write+=writes;dispatches++;
 last_transfer[0]=reads;last_transfer[1]=writes;last_transfer[2]=direct;
#ifdef DENSE_TARGET_CONSOLE
 htif_puts("MERLIN_DENSE_COUNTS {index}");
 for(int i=0;i<6;i++){{htif_puts(" ");htif_putd((long){symbol}_observe_count(i));}}htif_puts("\\n");
 htif_puts("MERLIN_DENSE_TRANSFER {index}");
 for(int i=0;i<3;i++){{htif_puts(" ");htif_putd((long)last_transfer[i]);}}htif_puts("\\n");
#endif
}}
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
        assert qualified["sources"] == {str(p): sha256_file(p) for p in qualification_sources(self)}
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
                *([record["observer"], symbol + "_observe_transfer"] if self.logical_abi else [record["bridge"]]),
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


def qualification_sources(provider):
    return list(
        dict.fromkeys(
            path
            for record in provider.records
            for path in (record["source"], *([record["logical_abi_source"]] if provider.logical_abi else []))
        )
    )


def logical_descriptor(array, mode):
    """Actual guarded backing, with layout/offset and byte alignment controlled separately."""
    rows, columns = array.shape
    width = array.dtype.itemsize
    offset = 3 if mode in ("offset", "strided") else 0
    strides = (2 * columns + 5, 2) if mode == "strided" else (columns, 1)
    elements = offset + (rows - 1) * strides[0] + (columns - 1) * strides[1] + 1
    memory = np.full(elements * width + 256, 167, dtype="u1")
    base = 64 + (-memory.ctypes.data % 64) + int(mode == "misaligned")
    view = np.ndarray(
        array.shape,
        dtype=array.dtype,
        buffer=memory,
        offset=base + offset * width,
        strides=tuple(stride * width for stride in strides),
    )
    view[...] = array
    descriptor = Descriptor(
        memory.ctypes.data,
        memory.ctypes.data + base,
        offset,
        (ctypes.c_int64 * 2)(rows, columns),
        (ctypes.c_int64 * 2)(*strides),
    )
    return memory, descriptor


def expected_transfers(provider, record, descriptors):
    p = record["problem"]
    sizes = (p.m * p.k * p.lhs_bytes, p.k * p.n * p.rhs_bytes, p.m * p.n * 4)
    widths = (p.lhs_bytes, p.rhs_bytes, 4)
    mask = 0
    for i, (d, width, alignment) in enumerate(
        zip(
            descriptors,
            widths,
            provider.selected.dense.logical_pointer_abi.alignment_bytes,
            strict=True,
        )
    ):
        start = d.aligned + d.offset * width
        if provider.borrow_contiguous and tuple(d.strides) == (d.sizes[1], 1) and start % alignment == 0:
            mask |= 1 << i
    copied = sum(size for i, size in enumerate(sizes) if not mask & (1 << i))
    return [copied, copied, mask]


def qualify(provider):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("native ABI qualifier unavailable")
    sources = qualification_sources(provider)
    library = provider.work / "qualification.so"
    argv = [cc, "-O2", "-fno-builtin", "-shared", "-fPIC", *map(str, sources), "-o", str(library)]
    subprocess.run(argv, check=True, capture_output=True)
    retained = provider.work / ("qualification-" + sha256_file(library) + ".so")
    library.replace(retained)
    native = ctypes.CDLL(str(retained))
    for record in provider.records:
        symbol = record["symbol"]
        size = getattr(native, symbol + "_trace_size")
        size.restype = ctypes.c_uint64
        query = getattr(native, symbol + "_trace")
        query.argtypes = [ctypes.c_int] * 2
        query.restype = ctypes.c_int64
        count = getattr(native, symbol + "_observe_count")
        count.argtypes = [ctypes.c_int]
        count.restype = ctypes.c_uint64
        if provider.logical_abi:
            transfer = getattr(native, symbol + "_observe_transfer")
            transfer.argtypes = [ctypes.c_int]
            transfer.restype = ctypes.c_uint64
            record["transfer"] = transfer
        record["native"] = (size, query, count)
    modes = ("contiguous", "offset", "strided", "misaligned", "mixed") if provider.logical_abi else ("strided",)
    calls, transfers, additional = [], [], []
    for trial, mode in enumerate(modes):
        previous = {r["index"]: [0] * 6 for r in provider.records}
        for record in provider.records:
            getattr(native, record["symbol"] + "_observe_reset")()
        trial_calls, trial_transfers = [], []
        for index, left, right, expected in provider.trace:
            record = provider.records[index]
            size, query, count = record["native"]
            start = size()
            buffers, originals, descriptors = [], [], []
            for i, array in enumerate((left, right, np.full(expected.shape, 0x71717171, dtype="<i4"))):
                rows, columns = array.shape
                if provider.logical_abi:
                    geometry = "strided" if mode == "mixed" and i == 1 else "contiguous" if mode == "mixed" else mode
                    memory, descriptor = logical_descriptor(array, geometry)
                else:
                    memory = np.full((rows + 2, 2 * columns + 5), 71, dtype=array.dtype)
                    memory[1 : rows + 1, 1 : 2 * columns + 1 : 2] = array
                    descriptor = Descriptor(
                        memory.ctypes.data,
                        memory.ctypes.data,
                        2 * columns + 6,
                        (ctypes.c_int64 * 2)(rows, columns),
                        (ctypes.c_int64 * 2)(2 * columns + 5, 2),
                    )
                buffers.append(memory)
                originals.append(memory.copy())
                descriptors.append(descriptor)
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
            if provider.logical_abi:
                d = descriptors[-1]
                output = np.ndarray(
                    expected.shape,
                    dtype=expected.dtype,
                    buffer=originals[-1],
                    offset=d.aligned - d.allocated + d.offset * 4,
                    strides=tuple(stride * 4 for stride in d.strides),
                )
                output[...] = expected
            else:
                originals[-1][1 : expected.shape[0] + 1, 1 : 2 * expected.shape[1] + 1 : 2] = expected
            np.testing.assert_array_equal(buffers[-1], originals[-1])
            counts = [count(i) for i in range(6)]
            delta = [new - old for new, old in zip(counts, previous[index], strict=True)]
            assert delta[:3] == [effects[key] for key in ("commands", "requested_read_bytes", "requested_write_bytes")]
            p = record["problem"]
            if provider.logical_abi:
                actual_transfer = [record["transfer"](i) for i in range(3)]
                assert actual_transfer == expected_transfers(provider, record, descriptors)
                assert delta[3:] == [*actual_transfer[:2], 1]
                trial_transfers.append(
                    dict(
                        index=index,
                        mode=mode,
                        values=actual_transfer,
                        descriptor_offsets=[d.offset for d in descriptors],
                        descriptor_strides=[list(d.strides) for d in descriptors],
                    )
                )
            else:
                packed = p.m * p.k * p.lhs_bytes + p.k * p.n * p.rhs_bytes + p.m * p.n * 4
                assert delta[3:] == [packed, packed, 1]
            report = record["result"].compiled.realization
            assert delta[:3] == [
                report[key]
                for key in (
                    "executed_instruction_calls",
                    "requested_load_bytes",
                    "requested_store_bytes",
                )
            ]
            previous[index] = counts
            trial_calls.append(dict(index=index, trace=observed, effects=effects, counters=counts))
        if trial == 0:
            calls, transfers = trial_calls, trial_transfers
        else:
            additional.append(dict(mode=mode, calls=trial_calls, transfers=trial_transfers))
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
    if provider.logical_abi:
        record.update(
            transfers=transfers,
            additional_abi_trials=additional,
            alignment="actual logical pointers independently checked against selected primitive alignment",
        )
    (provider.work / "native_qualification.json").write_text(json.dumps(record, indent=2))
    return record
