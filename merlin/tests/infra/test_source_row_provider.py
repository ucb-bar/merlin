"""Private test-only ABI/qualification support for canonical source rows."""

import ctypes
import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from test_pointwise_source_provider import Descriptor
from test_row_softmax import native_program
from test_selected_row_contracts import inputs as row_inputs

from merlin.common.digest import sha256_file
from merlin.llvmlower.cpu_softmax import CPU_SOFTMAX_F32_AVX2_POLICY as POLICY
from merlin.runtime.host_provider import HostProviderBuild, HostProviderObject, read_host_function_abi


class RowProvider:
    """One source island; synchronous private state, no actual accelerator."""

    def __init__(self, work, *, columns, chunk, exp_object, faulty=False):
        self.work, self.columns, self.chunk = work, columns, chunk
        self.exp_object, self.faulty = exp_object, faulty
        work.mkdir()
        self.symbol = "_mlir_ciface_borrowed_source_rows"
        self.c_source = work / "provider.c"
        self.result = None

    def selection(self, trace=None):
        from merlin.llvmlower.source_rows import SourceRowLowering

        selected, opts = row_inputs(self.work, rows=2, columns=self.columns, separate=True, chunk=6)
        self.selected, self.opts = selected, opts

        def prepare(bindings, work):
            self.prepare_bindings(bindings, work)
            if trace is not None:
                qualify(self, trace)

        return SourceRowLowering(
            selected_contracts=selected, adapter=opts["adapter"], numerical_policy=POLICY, execution_domain="host",
            allocation_alignment=32, prepare=prepare, build=lambda bindings, context: self.build(context),
            assume_finite_inputs=True, private_ownership=True, disjoint_arguments=True,
            chunk_elements=self.chunk,
        )

    def prepare(self, source_path, output):
        from merlin.llvmlower.prepared_model_transform import apply_prepared_model_transform
        from merlin.llvmlower.source_rows import bind_source_row_lowering

        prepare, _ = bind_source_row_lowering(self.selection(), None, None)
        return apply_prepared_model_transform(source_path, output / "rows", prepare)

    def prepare_bindings(self, bindings, output):
        # This fixture supplies implementation/qualification only. Production
        # discovery, scheduling and fresh-result rewriting live in Merlin.
        assert len(bindings) == 1
        binding = bindings[0]
        self.result = binding.compilation
        self.symbol = "_mlir_ciface_" + binding.writer.borrowed_symbol
        assert self.result.source.shape == (2, self.columns)
        (self.work / "scheduling.json").write_text(json.dumps(dict(
            source_passes=[asdict(p) for p in self.result.source_passes], schedule=self.result.compiled.report,
        ), indent=2))
        # Native qualification and RISC-V model compilation independently lower
        # the exact same scalar exponential MLIR, rather than copying a poly.
        self.exp_source = self.exp_object.parent / "exponential.mlir"
        self.result.revalidate()
        c = native_program(self.result, self.opts, bad_reset=self.faulty)
        c = c.replace("#include <stdlib.h>", "").replace("#include <string.h>", "")
        c = c.replace("#include <math.h>", "")
        c = c.replace("INFINITY", "__builtin_inff()").replace("signbit(", "__builtin_signbit(")
        c = c.replace("abort()", "__builtin_trap()")
        c = c.replace("memcpy(", "row_copy(").replace("memset(", "row_set(")
        c = c.replace(
            "static void kernel(",
            'static __attribute__((noinline,used,retain,section(".merlin_row.text"))) void kernel(',
        )
        self.c_source.write_text(
            "#include <stddef.h>\n"
            "static void row_copy(void*d,const void*s,size_t n){unsigned char*a=d;const unsigned char*b=s;"
            "for(size_t i=0;i<n;i++)a[i]=b[i];}\n"
            "static void row_set(void*d,int v,size_t n){unsigned char*a=d;for(size_t i=0;i<n;i++)a[i]=v;}\n"
            + c
            + self.wrapper()
        )

    def wrapper(self):
        n = self.columns
        report = self.result.compiled.report
        expected = [report[k] for k in ("executed_commands", "requested_read_bytes", "requested_write_bytes")]
        return f"""
typedef struct {{void *allocated,*aligned;int64_t offset,sizes[2],strides[2];}} descriptor;
static uint64_t totals[6];
#ifdef ROW_TARGET_CONSOLE
extern void htif_puts(const char *);
extern void htif_putd(long);
static void row_log(void){{
 htif_puts("MERLIN_ROW_COUNTS");
 for(int i=0;i<6;i++){{htif_puts(" ");htif_putd((long)totals[i]);}}
 htif_puts("\\n");
}}
#else
static void row_log(void){{}}
#endif
uint64_t row_count(int i){{return totals[i];}}
void row_reset(void){{for(int i=0;i<6;i++)totals[i]=0;}}
void {self.symbol}(const descriptor *a,descriptor *out){{
 float input[{2 * n}],output[{2 * n}];
 if(a->sizes[0]!=2||a->sizes[1]!={n}||out->sizes[0]!=2||out->sizes[1]!={n})__builtin_trap();
 for(int i=0;i<2;i++)for(int j=0;j<{n};j++){{
  row_copy(input+i*{n}+j,(const uint8_t*)a->aligned+4*(a->offset+i*a->strides[0]+j*a->strides[1]),4);
  if(!__builtin_isfinite(input[i*{n}+j]))__builtin_trap();
  totals[3]+=4;totals[4]+=4;
 }}
 invoke(input,output);
 if(executed_commands()!={expected[0]}||requested_reads()!={expected[1]}||requested_writes()!={expected[2]})
  __builtin_trap();
 for(int i=0;i<2;i++)for(int j=0;j<{n};j++){{
  row_copy((uint8_t*)out->aligned+4*(out->offset+i*out->strides[0]+j*out->strides[1]),output+i*{n}+j,4);
  totals[3]+=4;totals[4]+=4;
 }}
 totals[0]+=executed_commands();totals[1]+=requested_reads();totals[2]+=requested_writes();totals[5]++;
 row_log();
}}
"""

    def build(self, context):
        self.result.revalidate()
        source = self.c_source
        qualification = self.work / "native_qualification.json"
        witness = json.loads(qualification.read_text())
        assert witness["source_sha256"] == sha256_file(source)
        assert witness["exponential_source_sha256"] == sha256_file(self.exp_source)
        llvm, obj = context.workdir / "provider.ll", context.workdir / "provider.o"
        commands = []
        for options, destination in ((["-S", "-emit-llvm"], llvm), (["-c"], obj)):
            argv = [
                str(context.compiler),
                *context.compiler_flags,
                "-ffp-contract=off",
                "-DROW_TARGET_CONSOLE",
                *options,
                str(source),
                "-o",
                str(destination),
            ]
            context.compile(argv, inputs=(source,), output=destination)
            commands.append(tuple(argv))
        defined = [
            self.symbol,
            "row_count",
            "row_reset",
            "invoke",
            "requested_reads",
            "requested_writes",
            "executed_commands",
            "touched_high_water",
        ]
        artifact = HostProviderObject(
            obj,
            sha256_file(obj),
            llvm,
            sha256_file(llvm),
            tuple(read_host_function_abi(llvm, symbol) for symbol in defined),
            tuple(read_host_function_abi(llvm, symbol) for symbol in ("row_exp", "htif_puts", "htif_putd")),
            tuple((path, sha256_file(path)) for path in (source, context.compiler.resolve())),
            tuple(commands),
            Path.cwd(),
        )
        # The exponential is an independently compiled provider object. Making
        # it a private, unused model helper would permit symbol DCE to remove it;
        # adding a public helper would change the model's admitted public ABI.
        from merlin.llvmlower.lower import lower_model_file

        lowered = lower_model_file(
            self.exp_source,
            context.workdir / "exponential",
            targets=(),
            textual=True,
            features=frozenset({"lower_fma_to_intrinsic"}),
        )
        exp_llvm, exp_obj = context.workdir / "exponential.ll", context.workdir / "exponential.o"
        exp_commands = []
        for options, destination in ((["-S", "-emit-llvm"], exp_llvm), (["-c"], exp_obj)):
            argv = [
                str(context.compiler),
                *context.compiler_flags,
                *options,
                str(lowered.ll_path),
                "-o",
                str(destination),
            ]
            context.compile(argv, inputs=(lowered.ll_path, self.exp_source), output=destination)
            exp_commands.append(tuple(argv))
        exponential = HostProviderObject(
            exp_obj,
            sha256_file(exp_obj),
            exp_llvm,
            sha256_file(exp_llvm),
            tuple(read_host_function_abi(exp_llvm, symbol) for symbol in ("row_exp", "_mlir_ciface_row_exp")),
            (),
            tuple((path, sha256_file(path)) for path in (self.exp_source, lowered.ll_path, context.compiler.resolve())),
            tuple(exp_commands),
            Path.cwd(),
        )
        return HostProviderBuild(
            sha256_file(context.prepared_path),
            sha256_file(context.model_llvm_path),
            sha256_file(context.model_object_path),
            (artifact, exponential),
            sha256_file(qualification),
            sha256_file(qualification),
        )


def qualify(provider, trace):
    """Execute actual source bytes and ranked ABI against independent Torch."""
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native source/ABI compiler unavailable")
    library = provider.work / "provider.so"
    argv = [
        compiler,
        "-O2",
        "-fno-builtin",
        "-ffp-contract=off",
        "-shared",
        "-fPIC",
        str(provider.c_source),
        str(provider.exp_object),
        "-lm",
        "-o",
        str(library),
    ]
    subprocess.run(argv, check=True, capture_output=True)
    content_path = provider.work / ("provider-" + sha256_file(library) + ".so")
    library.replace(content_path)
    native = ctypes.CDLL(str(content_path))
    native.row_count.argtypes, native.row_count.restype = [ctypes.c_int], ctypes.c_uint64
    native.touched_high_water.argtypes, native.touched_high_water.restype = [ctypes.c_int], ctypes.c_uint64
    native.row_reset()
    for values, expected in trace:
        m, n = values.shape
        buffers, originals, descriptors = [], [], []
        for array in (values, np.zeros_like(values)):
            memory = np.full((m + 2, n + 3), 71, dtype="<f4")
            memory[1 : m + 1, 1 : n + 1] = array
            buffers.append(memory)
            originals.append(memory.copy())
            descriptors.append(
                Descriptor(
                    memory.ctypes.data,
                    memory.ctypes.data,
                    n + 4,
                    (ctypes.c_int64 * 2)(m, n),
                    (ctypes.c_int64 * 2)(n + 3, 1),
                )
            )
        fn = getattr(native, provider.symbol)
        fn.argtypes = [ctypes.POINTER(Descriptor)] * 2
        fn(*(ctypes.byref(d) for d in descriptors))
        np.testing.assert_array_equal(buffers[0].view("<u4"), originals[0].view("<u4"))
        originals[1][1 : m + 1, 1 : n + 1] = expected
        np.testing.assert_array_equal(buffers[1].view("<u4"), originals[1].view("<u4"))
    counters = [native.row_count(i) for i in range(6)]
    report = provider.result.compiled.report
    assert counters[:3] == [
        len(trace) * report[k] for k in ("executed_commands", "requested_read_bytes", "requested_write_bytes")
    ]
    packed = len(trace) * 2 * 2 * provider.columns * 4
    assert counters[3:] == [packed, packed, len(trace)]
    spaces = tuple(dict.fromkeys(b.backing for b in provider.opts["banks"]))
    high_water = [native.touched_high_water(i) for i in range(len(spaces))]
    assert all(
        value <= min(b.capacity for b in provider.opts["banks"] if b.backing == space)
        for value, space in zip(high_water, spaces, strict=True)
    )
    record = dict(
        private_high_water_bytes=high_water,
        source_sha256=sha256_file(provider.c_source),
        library_path=content_path.name,
        library_sha256=sha256_file(content_path),
        exponential_source_sha256=sha256_file(provider.exp_source),
        exponential_object_sha256=sha256_file(provider.exp_object),
        commands=[argv],
        counters=counters,
        scope="neutral synchronous C primitives + strided ABI, not target qualification or timing",
    )
    (provider.work / "native_qualification.json").write_text(json.dumps(record, indent=2))
    return record
