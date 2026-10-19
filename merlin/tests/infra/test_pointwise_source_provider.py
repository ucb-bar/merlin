"""Explicit test-only source/ABI provider for the neutral byte machine.

This is qualification scaffolding, not a target implementation or an agent
authoring input. The production entry owns admission and program checking;
this provider owns instruction selection, interpretation and descriptor glue.
"""

import ctypes
import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from test_pointwise_source import case
from test_readout_model_fusion import _EXECUTOR

from merlin.common.digest import sha256_file
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets
from merlin.runtime.host_provider import HostProviderBuild, HostProviderObject, read_host_function_abi
from merlin.transforms.certificates import ir_text


class Provider:
    case_builder = staticmethod(case)
    executor_source = _EXECUTOR
    capability_shape = (1, 1)
    input_dtypes = ("<i4", "<i4")
    output_dtype = "i1"
    expect_command_reduction = True
    observation_count = 6

    def __init__(self, work, *, enabled, separate):
        self.work, self.enabled, self.separate = work, enabled, separate
        work.mkdir()
        self.results, self.records = [], []
        self.c_source = None

    def schedule_budgets(self):
        return ScheduleBudgets(max_fusions=8 if self.enabled else 0)

    def expected_instructions(self):
        return ["fused"] if self.enabled else ["readout", "finish"]

    def case_options(self, directory, m, n):
        return self.case_builder(directory, m=m, n=n, separate=self.separate)

    @staticmethod
    def packed_bytes(shape):
        return 3 * 9 * shape[0] * (shape[1] + shape[2])

    @staticmethod
    def qualification_reference(left, bias):
        wide = left.astype(np.int64) + bias.astype(np.int64)
        return np.clip((wide + (1 << 31)) % (1 << 32) - (1 << 31), -128, 127).astype("i1")

    def selection(self, trace):
        """Fix one capability release before looking at any model source."""
        from merlin.llvmlower.source_pointwise import SourcePointwiseLowering

        directory = self.work / "capabilities"
        directory.mkdir()
        self.capabilities = self.case_options(directory, *self.capability_shape)
        self.trace, self.lowered = trace, 0
        options = self.capabilities
        return SourcePointwiseLowering(
            selected_contracts=options["selected_contracts"],
            adapter=options["adapter"],
            encoding=options["encoding"],
            limits=options["limits"],
            execution_domain="host",
            allocation_alignment=32,
            lower=self.lower,
            prepare=self.prepare,
            build=self.build,
            private_ownership=True,
            external_buffers_disjoint=True,
            scalar_qualification_identity=options.get("fusion_qualification_identity"),
            fusion_emitters=tuple(options.get("fusion_emitters", {}).items()),
            configuration_qualification_identity=options.get("configuration_qualification_identity"),
            spill_emitters=tuple(options.get("spill_emitters", {}).items()),
            spill_qualification_identity=options.get("spill_qualification_identity"),
            budgets=self.schedule_budgets(),
            max_source_elements=1024,
        )

    def lower(self, source):
        """Target mapping sees immutable scalar semantics and current shape only."""
        from merlin.llvmlower.source_pointwise import PointwiseProgram

        if len(source.shape) != 2:
            return None
        directory = self.work / ("mapping_" + str(self.lowered))
        directory.mkdir()
        self.lowered += 1
        options = self.case_options(directory, *source.shape)
        fixed = self.capabilities
        # The small fixture factory also emits contract files. Their bytes and
        # physical/checker limits must remain invariant across all source shapes.
        for key in ("facts_bytes", "software_bytes"):
            assert getattr(options["selected_contracts"], key) == getattr(fixed["selected_contracts"], key)
        assert options["limits"] == fixed["limits"]
        for key in ("configuration_identity", "numerical_policy_identity", "qualification_identity"):
            assert getattr(options["adapter"], key) == getattr(fixed["adapter"], key)
        for key in ("numerical_policy_identity", "qualification_identity"):
            assert getattr(options["encoding"], key) == getattr(fixed["encoding"], key)
        return PointwiseProgram(options["kernel"], options["objects"], options["bindings"])

    def prepare(self, bindings, private_directory):
        """Emit/qualify checked bindings; shared code owns source IR rewriting."""
        self.bindings = tuple(bindings)
        fragments = []
        for index, binding in enumerate(bindings):
            directory = self.work / str(index)
            directory.mkdir()
            compiled = binding.compilation
            m, n = compiled.operation.results[0].type.get_shape()
            bridge = "_mlir_ciface_" + binding.writer.borrowed_symbol
            image = b"".join(compiled.images())
            (directory / "program.bin").write_bytes(image)
            (directory / "source.mlir").write_text(ir_text(compiled.operation))
            (directory / "scheduling.json").write_text(
                json.dumps(
                    dict(
                        source_passes=[asdict(record) for record in compiled.source_passes],
                        source_conformance=compiled.conformance,
                        schedule=compiled.compiled.report,
                    ),
                    indent=2,
                )
            )
            fragments.append(self._wrapper(bridge, m, n, image, self.separate))
            self.results.append(compiled)
            self.records.append(
                dict(
                    symbol=bridge,
                    shape=[m, n],
                    program_sha256=sha256_file(directory / "program.bin"),
                    program_section=".merlin_pointwise." + bridge,
                )
            )
        assert self.results
        (self.work / "writers.json").write_text(json.dumps(self.records, indent=2))
        self.c_source = self.work / "provider.c"
        self.c_source.write_text(
            self.executor_source.replace("#include <string.h>", "").replace(
                "int execute(", "static __attribute__((noinline)) int execute("
            )
            + """
typedef struct {void *allocated,*aligned;int64_t offset,sizes[2],strides[2];} descriptor;
"""
            + f"static uint64_t totals[{self.observation_count}];\n"
            + """
uint64_t pointwise_count(int index){return totals[index];}
"""
            + f"void pointwise_reset(void){{for(int i=0;i<{self.observation_count};i++)totals[i]=0;}}\n"
            + f"""
#ifdef POINTWISE_TARGET_CONSOLE
extern void htif_puts(const char *);
extern void htif_putd(long);
static void pointwise_log(void){{
 htif_puts("MERLIN_POINTWISE_COUNTS");
 for(int i=0;i<{self.observation_count};i++){{htif_puts(" ");htif_putd((long)totals[i]);}}
 htif_puts("\\n");
}}
#else
static void pointwise_log(void){{}}
#endif
"""
            + "\n".join(fragments)
        )
        for result in self.results:
            result.revalidate()
        qualified, _ = qualify(self, self.trace)
        assert len(qualified) == len(self.trace)

    @staticmethod
    def _wrapper(symbol, m, n, image, separate):
        encoded = ",".join(str(v) for v in image)
        return f"""
void {symbol}(const descriptor *a,const descriptor *b,descriptor *out) {{
 static const uint8_t image[] __attribute__((used,retain,aligned(1),section(".merlin_pointwise.{symbol}")))=
 {{{encoded}}};
 uint8_t acc[1024],stage[1024],activation[1024],x[{4 * m * n}],bias[{4 * m * n}],y[{m * n}];
 uint8_t *spaces[10]={{acc,stage,{"activation" if separate else "stage"},x,0,0,bias,0,0,y}};
 uint64_t capacities[10]={{1024,1024,1024,{4 * m * n},0,0,{4 * m * n},0,0,{m * n}}},counts[3];
 if(a->sizes[0]!={m}||a->sizes[1]!={n}||b->sizes[0]!={m}||b->sizes[1]!={n}||
    out->sizes[0]!={m}||out->sizes[1]!={n})__builtin_trap();
 for(int i=0;i<{m};i++)for(int j=0;j<{n};j++)for(int byte=0;byte<4;byte++){{
  x[4*(i*{n}+j)+byte]=((const uint8_t*)a->aligned)[4*(a->offset+i*a->strides[0]+j*a->strides[1])+byte];
  bias[4*(i*{n}+j)+byte]=((const uint8_t*)b->aligned)[4*(b->offset+i*b->strides[0]+j*b->strides[1])+byte];
  totals[3]+=2;totals[4]+=2;
 }}
 if(execute(image,sizeof(image),spaces,capacities,counts,0))__builtin_trap();
 for(int i=0;i<{m};i++)for(int j=0;j<{n};j++){{
  ((uint8_t*)out->aligned)[out->offset+i*out->strides[0]+j*out->strides[1]]=y[i*{n}+j];
  totals[3]++;totals[4]++;
 }}
 for(int i=0;i<3;i++)totals[i]+=counts[i];totals[5]++;
 pointwise_log();
}}
"""

    def build(self, bindings, context):
        assert tuple(bindings) == self.bindings
        for result in self.results:
            result.revalidate()
        source = self.c_source
        llvm, obj = context.workdir / "provider.ll", context.workdir / "provider.o"
        commands = []
        for options, destination in ((["-S", "-emit-llvm"], llvm), (["-c"], obj)):
            argv = [
                str(context.compiler),
                *context.compiler_flags,
                "-DPOINTWISE_TARGET_CONSOLE",
                *options,
                str(source),
                "-o",
                str(destination),
            ]
            context.compile(argv, inputs=(source,), output=destination)
            commands.append(tuple(argv))
        defined = [record["symbol"] for record in self.records] + ["pointwise_count", "pointwise_reset"]
        artifact = HostProviderObject(
            obj,
            sha256_file(obj),
            llvm,
            sha256_file(llvm),
            tuple(read_host_function_abi(llvm, symbol) for symbol in defined),
            tuple(read_host_function_abi(llvm, symbol) for symbol in ("htif_puts", "htif_putd")),
            tuple((path, sha256_file(path)) for path in (source, context.compiler.resolve())),
            tuple(commands),
            Path.cwd(),
        )
        qualification = self.work / "native_qualification.json"
        if not qualification.is_file() or json.loads(qualification.read_text())["source_sha256"] != sha256_file(source):
            raise ValueError("neutral provider requires independent native source/ABI execution first")
        return HostProviderBuild(
            sha256_file(context.prepared_path),
            sha256_file(context.model_llvm_path),
            sha256_file(context.model_object_path),
            (artifact,),
            sha256_file(qualification),
            sha256_file(qualification),
        )


class Descriptor(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * 2),
        ("strides", ctypes.c_int64 * 2),
    ]


def qualify(provider, arithmetic_inputs):
    """Run the exact provider C, checking complete output, guards and inputs."""
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native source/ABI qualifier unavailable")
    built_library = provider.work / "provider.so"
    argv = [compiler, "-O2", "-fno-builtin", "-shared", "-fPIC", str(provider.c_source), "-o", str(built_library)]
    subprocess.run(argv, check=True, capture_output=True)
    # dlopen can retain a prior CDLL after pytest has removed a passed test's
    # files and reused its directory name. Select by actual compiled bytes so
    # qualification cannot execute an earlier model under a new source hash.
    library = provider.work / f"provider-{sha256_file(built_library)}.so"
    built_library.replace(library)
    native = ctypes.CDLL(str(library))
    native.pointwise_count.argtypes, native.pointwise_count.restype = [ctypes.c_int], ctypes.c_uint64
    native.pointwise_reset()
    outputs = []
    for index, *inputs in arithmetic_inputs:
        record = provider.records[index]
        m, n = record["shape"]
        # Strided, offset descriptors exercise real ABI packing as well.
        buffers, descriptors, originals = [], [], []
        assert len(inputs) == len(provider.input_dtypes)
        arrays = (*inputs, np.zeros((m, n), provider.output_dtype))
        dtypes = (*provider.input_dtypes, provider.output_dtype)
        for array, dtype in zip(arrays, dtypes, strict=True):
            memory = np.full((m + 2, n + 3), 71, dtype=dtype)
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
        fn = getattr(native, record["symbol"])
        fn.argtypes = [ctypes.POINTER(Descriptor)] * len(descriptors)
        fn(*(ctypes.byref(d) for d in descriptors))
        for array, original in zip(buffers[:-1], originals[:-1], strict=True):
            np.testing.assert_array_equal(array, original)
        expected = provider.qualification_reference(*inputs)
        actual = buffers[-1][1 : m + 1, 1 : n + 1].copy()
        np.testing.assert_array_equal(actual, expected)
        originals[-1][1 : m + 1, 1 : n + 1] = expected
        np.testing.assert_array_equal(buffers[-1], originals[-1])
        outputs.append(actual)
    record = dict(
        source_sha256=sha256_file(provider.c_source),
        library_path=library.name,
        library_sha256=sha256_file(library),
        compiler=str(compiler),
        commands=[argv],
        shapes=[list(array.shape) for array in outputs],
        counters=[native.pointwise_count(i) for i in range(provider.observation_count)],
        scope="neutral native C and ranked ABI; not hardware qualification or timing",
    )
    (provider.work / "native_qualification.json").write_text(json.dumps(record, indent=2))
    return outputs, record
