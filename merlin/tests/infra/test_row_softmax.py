"""Complete-domain counted reductions, physical state and realized mutations."""

import ctypes
import hashlib
import os
import subprocess
from dataclasses import replace

import numpy as np
import pytest

from merlin.compile.scheduling.row_softmax import (
    PHASE_ROLES,
    CausalRowMask,
    RowSoftmaxAdapter,
    RowSoftmaxNumerics,
    RowSoftmaxProblem,
    RowSoftmaxResources,
    RowSoftmaxStep,
    RowSoftmaxUnavailable,
    compile_row_softmax,
)
from merlin.compile.scheduling.row_softmax_verification import verify_row_softmax
from merlin.llvmlower.cpu_softmax import CPU_SOFTMAX_F32_AVX2_POLICY, cpu_softmax_row_numerics
from merlin.sched.ir import NULL, Call, Const
from merlin.semantic_compiler.allocation_types import Reservation, StorageBank

KINDS = tuple(PHASE_ROLES)
NAMES = (
    "row",
    "column",
    "count",
    "source",
    "output",
    "space0",
    "address0",
    "space1",
    "address1",
    "numeric",
    "period",
    "offset",
)


def adapter_for(numerics, banks):
    spaces = tuple(dict.fromkeys(b.backing for b in banks))

    def encode(step):
        local = [item for space, address in step.local for item in (Const(spaces.index(space)), address)]
        local += [Const(0)] * (4 - len(local))
        operands = (
            step.row,
            step.column,
            Const(step.count),
            step.source or NULL,
            step.output or NULL,
            *local,
            Const(1),
            Const(step.mask.period if step.mask else 0),
            Const(step.mask.offset if step.mask else 0),
        )
        return Call("stage_" + str(KINDS.index(step.kind)), tuple(zip(NAMES, operands)))

    def project(call):
        assert tuple(k for k, _v in call.args) == NAMES
        values = dict(call.args)
        kind = KINDS[int(call.instr.removeprefix("stage_"))]
        if values["numeric"] != Const(1):
            raise ValueError("unknown actual numerical instruction mode")
        period, offset = values["period"].value, values["offset"].value
        if period == 0 and offset:
            raise ValueError("unknown unmasked offset")
        local = tuple((spaces[values[f"space{i}"].value], values[f"address{i}"]) for i in range(len(PHASE_ROLES[kind])))
        return RowSoftmaxStep(
            kind,
            values["row"],
            values["column"],
            values["count"].value,
            local,
            None if values["source"] is NULL else values["source"],
            None if values["output"] is NULL else values["output"],
            numerics.identity,
            CausalRowMask(period, offset) if period else None,
        )

    return RowSoftmaxAdapter(
        encode,
        project,
        "neutral-row-resources-v1",
        numerics.identity,
        "test-only-source-ordered-primitives-v1",
        True,
        True,
        True,
        64,
    )


def fixture(*, rows=5, columns=19, chunk=6, separate=False, dtype="f32", mask=None, capacity=512):
    numerics = (
        cpu_softmax_row_numerics(backend_policy=CPU_SOFTMAX_F32_AVX2_POLICY, assume_finite_inputs=True)
        if dtype == "f32"
        else RowSoftmaxNumerics("scalar-f64-source-v1", "c-libm-exp-v1", "f64", 1, 0, "divide", True, True)
    )
    banks = (
        (StorageBank("state", "private-a", capacity, "byte", 4), StorageBank("exps", "private-b", capacity, "byte", 8))
        if separate
        else (StorageBank("state", "private-a", capacity, "byte", 4),)
    )
    roles = ("state", "exps" if separate else "state", "state", "state")
    problem = RowSoftmaxProblem(rows, columns, mask)
    resources = RowSoftmaxResources(chunk, 1, True)
    adapter = adapter_for(numerics, banks)
    return dict(
        problem=problem,
        numerics=numerics,
        resources=resources,
        banks=banks,
        role_banks=roles,
        adapter=adapter,
        private_ownership=True,
        disjoint_arguments=True,
    )


def verify(result, options, kernel=None, storage=None):
    return verify_row_softmax(kernel or result.kernel, storage=storage or result.storage, **options)


def test_full_domains_and_completed_state_reuse():
    opts = fixture()
    result = compile_row_softmax(**opts)
    assert result.report["static_calls"] == 11
    assert result.report["executed_commands"] == 5 * (3 + 4 * 4)
    assert result.storage[0].base == result.storage[3].base  # maximum dies before normalization factor
    assert result.storage[2].base != result.storage[3].base  # simultaneous finish operands
    assert verify(result, opts) == {key: value for key, value in result.report.items() if key != "pipeline"}


@pytest.mark.parametrize("dtype,separate", [("f32", False), ("f32", True), ("f64", True)])
def test_distinct_models_tails_and_reservations(dtype, separate):
    opts = fixture(dtype=dtype, separate=separate, mask=CausalRowMask(3, 2))
    opts["reservations"] = (Reservation("state", 0, 24),)
    result = compile_row_softmax(**opts)
    assert result.storage[0].base * result.storage[0].unit_bytes >= 24
    assert (
        result.storage[1].backing != result.storage[0].backing
        or result.storage[1].base * result.storage[1].unit_bytes >= 24
    )
    assert verify(result, opts)["problem"]["mask"] == {"period": 3, "offset": 2}


def test_symbolic_huge_domain_has_constant_static_ir():
    opts = fixture(rows=2**25, columns=2**30, chunk=31, capacity=2**34)
    result = compile_row_softmax(**opts)
    assert len(result.kernel.text()) < 12000
    assert result.report["executed_commands"] > 2**50
    assert result.report["private_state_bytes"] == (2**30 + 10) * 4


@pytest.mark.parametrize("field", ["private_ownership", "disjoint_arguments"])
def test_unknown_invocation_ownership_refuses(field):
    opts = fixture()
    opts[field] = False
    with pytest.raises(RowSoftmaxUnavailable, match="private workspace"):
        compile_row_softmax(**opts)


@pytest.mark.parametrize("field", ["synchronous_completion", "exact_source_order", "addressable_state"])
def test_unknown_primitive_capabilities_refuse(field):
    opts = fixture()
    opts["adapter"] = replace(opts["adapter"], **{field: False})
    with pytest.raises(RowSoftmaxUnavailable, match="source order"):
        compile_row_softmax(**opts)


def test_capacity_boundary_aliases_and_overwrites():
    opts = fixture(columns=8, chunk=3, capacity=68)
    # Concurrent maximum+full exps and full exps+lanes+normalizer require 68B.
    result = compile_row_softmax(**opts)
    assert max((v.base + v.capacity) * v.unit_bytes for v in result.storage) == 68
    with pytest.raises(RowSoftmaxUnavailable, match="cannot be allocated"):
        compile_row_softmax(**{**opts, "banks": (replace(opts["banks"][0], capacity=67),)})
    aliased = list(result.storage)
    aliased[3] = replace(aliased[3], base=aliased[2].base)
    with pytest.raises(ValueError, match="ownership"):
        verify(result, opts, storage=tuple(aliased))


def test_numerical_contract_tail_budget_and_index_refusals():
    opts = fixture()
    with pytest.raises(ValueError, match="stale"):
        compile_row_softmax(**{**opts, "numerics": replace(opts["numerics"], normalization="divide")})
    with pytest.raises(RowSoftmaxUnavailable, match="legal row chunk"):
        compile_row_softmax(**{**opts, "resources": replace(opts["resources"], granularity=2, partial_chunks=False)})
    with pytest.raises(ValueError, match="static command budget"):
        compile_row_softmax(**{**opts, "resources": replace(opts["resources"], max_static_calls=6)})
    with pytest.raises(RowSoftmaxUnavailable, match="address range"):
        compile_row_softmax(**{**fixture(rows=2**30), "adapter": replace(opts["adapter"], index_bits=32)})
    with pytest.raises(ValueError, match="each lane"):
        replace(opts["numerics"], tree=(((0, 4), (2, 6)), ((1, 5), (3, 3))))
    with pytest.raises(ValueError, match="unknown masks"):
        RowSoftmaxProblem(1, 8, object())


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_tail",
        "chunk_reset",
        "early_normalize",
        "wrong_maximum",
        "wrong_mask",
        "wrong_local",
        "async",
        "extra_effect",
    ],
)
def test_actual_output_mutations_fail(mutation):
    opts = fixture(mask=CausalRowMask(3))
    result = compile_row_softmax(**opts)
    body = list(result.kernel.body[0].body)
    if mutation == "missing_tail":
        del body[2]
    elif mutation == "early_normalize":
        body[1], body[-2] = body[-2], body[1]
    elif mutation == "extra_effect":
        body.append(body[0])
    else:
        index = 6 if mutation == "chunk_reset" else 1
        loop = body[index]
        call = loop.body[0]
        values = dict(call.args)
        if mutation in ("chunk_reset", "wrong_maximum"):
            values["column"] = Const(0)
        elif mutation == "wrong_mask":
            values["offset"] = Const(1)
        elif mutation == "wrong_local":
            values["address0"] = values["address0"] + 4
        call = replace(call, args=tuple(values.items()), produces="late" if mutation == "async" else None)
        body[index] = replace(loop, body=(call,))
    kernel = replace(result.kernel, body=(replace(result.kernel.body[0], body=tuple(body)),))
    with pytest.raises(ValueError):
        verify(result, opts, kernel=kernel)


def test_alternative_chunks_use_same_source_and_verifier():
    opts = fixture(columns=113, chunk=17, capacity=1024)
    variants = [compile_row_softmax(**opts, chunk_elements=n) for n in (1, 7, 8, 13, 17)]
    assert len({r.report["numerical_identity"] for r in variants}) == 1
    assert all(verify(r, opts) == {key: value for key, value in r.report.items() if key != "pipeline"}
               for r in variants)
    assert variants[0].report["executed_commands"] > variants[-1].report["executed_commands"]


def test_ordering_freshness_and_nonpartial_fallback():
    opts = fixture(columns=18, chunk=8)
    opts["resources"] = replace(opts["resources"], granularity=2, partial_chunks=False)
    result = compile_row_softmax(**opts)
    assert result.kernel.body[0].body[1].extent == 3  # largest legal six-element chunk
    assert [p.name for p in result.passes] == ["row_semantics", "row_tiling", "row_allocation", "row_schedule"]
    assert set(result.passes[-1].invalidated) == {"row_source", "row_chunk", "row_storage"}
    assert result.passes[-1].after.source_sha256 == result.kernel.digest()
    changed = compile_row_softmax(**{**opts, "problem": replace(opts["problem"], rows=2)})
    assert result.passes[0].before.context != changed.passes[0].before.context
    wrong = ("row_semantics", "row_tiling", "row_schedule", "row_allocation")
    calls = []
    opts["adapter"] = replace(opts["adapter"], encode=lambda op: calls.append(op))
    with pytest.raises(ValueError, match="prerequisites|ordering"):
        compile_row_softmax(**opts, pass_order=wrong)
    assert not calls


def test_code_budget_selects_a_legal_smaller_loop_and_unknown_numeric_refuses():
    opts = fixture(columns=19, chunk=6)
    result = compile_row_softmax(**{**opts, "resources": replace(opts["resources"], max_static_calls=7)})
    assert result.report["static_calls"] == 7
    assert result.kernel.body[0].body[1].extent == 19
    with pytest.raises(RowSoftmaxUnavailable, match="finite inputs"):
        compile_row_softmax(**{**opts, "numerics": replace(opts["numerics"], finite_inputs=False)})
    with pytest.raises(RowSoftmaxUnavailable, match="per lane"):
        compile_row_softmax(**fixture(columns=7))
    with pytest.raises(ValueError, match="nonnegative offset"):
        CausalRowMask(3, -1)


@pytest.fixture(scope="module")
def exp_object(tmp_path_factory):
    """Compile the existing exact exponential; no copied polynomial or oracle."""
    from xdsl.dialects import builtin, func
    from xdsl.ir import Block, Region

    from merlin.llvmlower import toolchain
    from merlin.llvmlower.cpu_softmax import build_cpu_softmax_exp_nonpositive
    from merlin.llvmlower.lower import lower_model_file

    if not toolchain.available():
        pytest.skip("native MLIR toolchain unavailable")
    root = tmp_path_factory.mktemp("row-exp")
    body = Block(arg_types=[builtin.f32])
    ops, result = build_cpu_softmax_exp_nonpositive(
        body.args[0], backend_policy=CPU_SOFTMAX_F32_AVX2_POLICY, domain="nonpositive"
    )
    body.add_ops([*ops, func.ReturnOp(result)])
    module = builtin.ModuleOp([func.FuncOp("row_exp", ([builtin.f32], [builtin.f32]), Region(body))])
    module.verify()
    source = root / "exponential.mlir"
    source.write_text(str(module))
    lowered = lower_model_file(source, root / "lower", targets=(), textual=True)
    obj = root / "exponential.o"
    subprocess.run(
        [str(toolchain.clang()), "-O2", "-fPIC", "-c", str(lowered.ll_path), "-o", str(obj)],
        check=True,
        capture_output=True,
    )
    return obj


def native_program(result, opts, *, bad_reset=False):
    """An instrumented neutral implementation of the declared primitive API."""
    from merlin.sched.isa import InstrDef, InstructionSet, Operand

    model = opts["numerics"]
    width = 4 if model.dtype == "f32" else 8
    ctype = "float" if width == 4 else "double"
    kinds = ("int", "int", "int", "ptr", "ptr", "int", "int", "int", "int", "int", "int", "int")
    instrs = {
        f"stage_{i}": InstrDef(
            f"stage_{i}",
            tuple(Operand(n, k) for n, k in zip(NAMES, kinds)),
            lambda values, i=i: "stage(" + str(i) + "," + ",".join(values) + ");",
        )
        for i in range(7)
    }
    iset = InstructionSet("neutral-row-test", instrs, {model.dtype: ctype}, "/* synchronous */")
    spaces = tuple(dict.fromkeys(b.backing for b in opts["banks"]))
    capacities = [min(b.capacity for b in opts["banks"] if b.backing == space) for space in spaces]

    def tree(node):
        return f"lanes[{node}]" if type(node) is int else f"({tree(node[0])}+{tree(node[1])})"

    exp = "row_exp" if width == 4 else "exp"
    finish = "((T)1) / total" if model.normalization == "reciprocal_multiply" else "total"
    normalize = "v * factor" if model.normalization == "reciprocal_multiply" else "v / factor"
    arrays = "\n".join(f"static uint8_t mem{i}[{capacity + 32}];" for i, capacity in enumerate(capacities))
    pointers = ",".join(f"mem{i}+16" for i in range(len(spaces)))
    guards = "\n".join(
        f"for(int j=0;j<16;j++) if(mem{i}[j]!=0xa5 || mem{i}[{cap}+16+j]!=0xa5) abort();"
        for i, cap in enumerate(capacities)
    )
    dirty = "\n".join(f"memset(mem{i},0xa5,sizeof(mem{i}));" for i in range(len(spaces)))
    source = f"""
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
typedef {ctype} T;
extern float row_exp(float);
static uint64_t commands,reads,writes;
{arrays}
static uint8_t *spaces[]={{{pointers}}};
static int64_t capacities[]={{{",".join(map(str, capacities))}}};
static uint64_t high_water[{len(spaces)}];
static T readp(const void *p){{T v;memcpy(&v,p,sizeof(v));reads+=sizeof(v);return v;}}
static void writep(void *p,T v){{memcpy(p,&v,sizeof(v));writes+=sizeof(v);}}
static void *local(int64_t s,int64_t a){{
 if(s<0 || s>={len(spaces)} || a<0 || a+sizeof(T)>capacities[s]) abort();
 if(a+sizeof(T)>high_water[s]) high_water[s]=a+sizeof(T);
 return spaces[s]+a;
}}
static T readl(int64_t s,int64_t a){{return readp(local(s,a));}}
static void writel(int64_t s,int64_t a,T v){{writep(local(s,a),v);}}
static void stage(int kind,int64_t row,int64_t column,int64_t count,void *input,void *output,
 int64_t s0,int64_t a0,int64_t s1,int64_t a1,int64_t numeric,int64_t period,int64_t offset){{
 commands++;
 if(numeric!=1) abort();
 T lanes[{model.lanes}];
 if(kind==0) writel(s0,a0,-INFINITY);
 if(kind==1){{
  T maximum=readl(s0,a0);
  for(int64_t i=0;i<count;i++){{
   T v=readp((uint8_t*)input+i*sizeof(T));
   if(period && column+i>row%period+offset) v=-INFINITY;
   if(v>maximum || (v==0 && maximum==0 && !signbit(v))) maximum=v;
  }}
  writel(s0,a0,maximum);
 }}
 if(kind==2){{
  T maximum=readl(s0,a0);
  for(int64_t i=0;i<count;i++){{
   T v=readp((uint8_t*)input+i*sizeof(T));
   if(period && column+i>row%period+offset) v=-INFINITY;
   T shifted=v-maximum;
   writel(s1,a1+i*sizeof(T),{exp}(shifted));
  }}
 }}
 if(kind==3) for(int j=0;j<{model.lanes};j++) writel(s0,a0+j*sizeof(T),0);
 if(kind==4){{
  for(int j=0;j<{model.lanes};j++) lanes[j]=readl(s1,a1+j*sizeof(T));
  if({int(bad_reset)} && column>0) for(int j=0;j<{model.lanes};j++) lanes[j]=0;
  for(int64_t i=0;i<count;i++){{
   int64_t c=column+i;T v=readl(s0,a0+i*sizeof(T));
   if(c<{model.lanes}) lanes[c]=v; else lanes[c%{model.lanes}]=lanes[c%{model.lanes}]+v;
  }}
  for(int j=0;j<{model.lanes};j++) writel(s1,a1+j*sizeof(T),lanes[j]);
 }}
 if(kind==5){{
  for(int j=0;j<{model.lanes};j++) lanes[j]=readl(s0,a0+j*sizeof(T));
  T total={tree(model.tree)};T factor={finish};writel(s1,a1,factor);
 }}
 if(kind==6){{
  T factor=readl(s1,a1);
  for(int64_t i=0;i<count;i++){{T v=readl(s0,a0+i*sizeof(T));writep((uint8_t*)output+i*sizeof(T),{normalize});}}
 }}
}}
{result.emit_c(iset, symbol="kernel")}
void invoke(T *input,T *output){{commands=reads=writes=0;memset(high_water,0,sizeof(high_water));
 {dirty}kernel(input,output);{guards}}}
uint64_t requested_reads(void){{return reads;}}
uint64_t requested_writes(void){{return writes;}}
uint64_t executed_commands(void){{return commands;}}
uint64_t touched_high_water(int s){{return high_water[s];}}
"""
    return source


def execute_native(tmp_path, source, values, exp_object, name, spaces, *, compiler=None):
    if compiler is None:
        from merlin.llvmlower import toolchain

        compiler = toolchain.clang()

    # Some repository temp fixtures recycle directory names after each case;
    # dlopen caches by absolute path even after that file has been replaced.
    identity = hashlib.sha256(source.encode()).hexdigest()[:16]
    c = tmp_path / f"{name}_{identity}.c"
    lib = tmp_path / f"{name}_{identity}.so"
    c.write_text(source)
    subprocess.run(
        [
            str(compiler),
            "-O2",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            str(c),
            *([] if exp_object is None else [str(exp_object)]),
            "-lm",
            "-o",
            str(lib),
        ],
        check=True,
        capture_output=True,
    )
    library = ctypes.CDLL(str(lib))
    library.invoke.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    output_storage = np.full(values.size + 4, -421, dtype=values.dtype)
    output = output_storage[2:-2].reshape(values.shape)
    original = values.copy()
    for _ in range(2):  # Dirty reused storage cannot carry a preceding row/invocation.
        library.invoke(values.ctypes.data, output.ctypes.data)
    np.testing.assert_array_equal(values, original)
    assert np.all(output_storage[:2] == -421) and np.all(output_storage[-2:] == -421)
    counters = {}
    for field, symbol in (
        ("requested_read_bytes", "requested_reads"),
        ("requested_write_bytes", "requested_writes"),
        ("executed_commands", "executed_commands"),
    ):
        fn = getattr(library, symbol)
        fn.restype = ctypes.c_uint64
        counters[field] = fn()
    library.touched_high_water.argtypes = [ctypes.c_int]
    library.touched_high_water.restype = ctypes.c_uint64
    counters["physical_high_water_bytes"] = {space: library.touched_high_water(i) for i, space in enumerate(spaces)}
    return output.copy(), counters


@pytest.mark.parametrize(
    "columns,chunk,mask,separate,dtype",
    [
        (8, 3, None, False, "f32"),
        (19, 6, CausalRowMask(3), True, "f32"),
        (113, 13, CausalRowMask(5, 2), False, "f32"),
        (257, 7, None, False, "f32"),
        (23, 5, CausalRowMask(2), True, "f64"),
    ],
)
def test_emitted_program_complete_outputs_and_actual_payloads(
    tmp_path, exp_object, columns, chunk, mask, separate, dtype
):
    from merlin.llvmlower import toolchain

    opts = fixture(columns=columns, chunk=columns, mask=mask, separate=separate, dtype=dtype, capacity=2048)
    baseline = compile_row_softmax(**opts, chunk_elements=columns)
    tiled = compile_row_softmax(**opts, chunk_elements=chunk)
    assert baseline.passes[0].before.context.library == tiled.passes[0].before.context.library
    assert baseline.passes[0].before.context.execution == tiled.passes[0].before.context.execution
    values = np.random.default_rng(73).normal(0, 4, (5, columns)).astype(dtype.replace("f", "float"))
    values[1] = np.linspace(-120, 0, columns)
    values[2] = 1
    values[4] = np.finfo(values.dtype).min
    values[4, 0] = np.finfo(values.dtype).max
    np.save(tmp_path / "input.npy", values)
    if dtype == "f32":
        reference = """
import sys,torch,numpy as np
from pathlib import Path
if not torch.__version__.startswith('2.10.') or torch.backends.cpu.get_cpu_capability()!='AVX2':sys.exit(77)
root=Path(sys.argv[1]);x=torch.from_numpy(np.load(root/'input.npy'))
period,offset=map(int,sys.argv[2:4])
if period:
 mask=torch.arange(x.shape[1])[None,:] > torch.arange(x.shape[0])[:,None]%period+offset
 x=x.masked_fill(mask,float('-inf'))
np.save(root/'reference.npy',torch.softmax(x,-1).numpy())
"""
        run = subprocess.run(
            [
                str(toolchain.compiler_python()),
                "-c",
                reference,
                str(tmp_path),
                str(mask.period if mask else 0),
                str(mask.offset if mask else 0),
            ],
            env={**os.environ, "ATEN_CPU_CAPABILITY": "avx2"},
            text=True,
            capture_output=True,
        )
        if run.returncode == 77:
            pytest.skip("independent pinned Torch2.10 AVX2 unavailable")
        assert run.returncode == 0, run.stderr
        reference_output = np.load(tmp_path / "reference.npy")
    else:
        import math

        rows = []
        for row, values_row in enumerate(values):
            active = [
                float(x) if mask is None or col <= row % mask.period + mask.offset else -math.inf
                for col, x in enumerate(values_row)
            ]
            exponentials = [math.exp(x - max(active)) for x in active]
            total = exponentials[0]
            for x in exponentials[1:]:
                total += x
            rows.append([x / total for x in exponentials])
        reference_output = np.array(rows)
    spaces = tuple(dict.fromkeys(b.backing for b in opts["banks"]))
    before, before_counts = execute_native(
        tmp_path, native_program(baseline, opts), values, exp_object, "baseline", spaces
    )
    after, after_counts = execute_native(tmp_path, native_program(tiled, opts), values, exp_object, "tiled", spaces)
    word_type = "u4" if dtype == "f32" else "u8"
    np.testing.assert_array_equal(after.view(word_type), before.view(word_type))
    np.testing.assert_array_equal(after.view(word_type), reference_output.view(word_type))
    for report, observed in ((baseline.report, before_counts), (tiled.report, after_counts)):
        assert {key: report[key] for key in observed} == observed
    if columns == 113:
        wrong, _counts = execute_native(
            tmp_path, native_program(tiled, opts, bad_reset=True), values, exp_object, "incorrect_chunk_reset", spaces
        )
        assert not np.array_equal(wrong.view(word_type), reference_output.view(word_type))
