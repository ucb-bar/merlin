"""Actual counted-loop programs, independent arithmetic and corrupted effects."""

from dataclasses import replace

import pytest

from merlin.compile.scheduling.block_schedule import Compute, Load, Preload, Store
from merlin.compile.scheduling.compact_dense import (
    DenseLoopAdapter,
    DenseLoopStorage,
    DenseMemoryView,
    emit_compact_dense,
    schedule_compact_dense,
)
from merlin.compile.scheduling.compact_verification import verify_compact_dense
from merlin.sched.ir import Call, Loop, as_expr, evaluate, instances
from merlin.transforms.dense_planning import DenseProblem, DenseResources, dense_blocks


def encode(op):
    if isinstance(op, Load):
        names = ("a", "b", "src_row", "src_col", "rows", "cols", "dst", "step")
        values = (*op.block, op.dram_row, op.dram_col, op.rows, op.cols, op.row, op.step)
        instruction = op.role + "_load"
    elif isinstance(op, Preload):
        names = ("weight", "k", "n", "acc", "continue", "rows", "cols", "depth")
        values = (op.weight_row, *op.weight_block, op.accumulator_row, op.accumulate, op.rows, op.cols, op.weight_rows)
        instruction = "preload"
    elif isinstance(op, Compute):
        names = ("lhs", "m", "k", "rows", "depth")
        values = (op.input_row, *op.input_block, op.rows, op.cols)
        instruction = "compute"
    else:
        assert isinstance(op, Store)
        names = ("row", "col", "rows", "cols", "acc")
        values = (op.dram_row, op.dram_col, op.rows, op.cols, op.accumulator_row)
        instruction = "store"
    return Call(instruction, tuple(zip(names, map(as_expr, values))))


def project(call):
    a = dict(call.args)
    if call.instr in ("lhs_load", "weight_load"):
        return Load(
            call.instr.removesuffix("_load"),
            (a["a"], a["b"]),
            a["src_row"],
            a["src_col"],
            a["rows"],
            a["cols"],
            a["dst"],
            a["step"],
        )
    if call.instr == "preload":
        return Preload(a["weight"], (a["k"], a["n"]), a["acc"], a["continue"], a["rows"], a["cols"], a["depth"])
    if call.instr == "compute":
        return Compute(a["lhs"], (a["m"], a["k"]), a["rows"], a["depth"], True)
    if call.instr == "store":
        return Store(a["row"], a["col"], a["rows"], a["cols"], a["acc"])
    raise ValueError("unknown instruction")


ADAPTER = DenseLoopAdapter(encode, project, "neutral-test-config-v1", "source-ordered-test-policy-v1", True, True, 64)


def fixture(*, shape=(5, 14, 13), tile=(2, 3, 4), widths=(1, 1, 4), padded=False):
    p = DenseProblem(*shape, *tile, *widths)
    # Three genuinely separate memories; unequal physical row widths. Padding
    # and nonzero base exercise byte ownership rather than interchangeable rows.
    lhs_width = p.tile_k * p.lhs_bytes + (4 if padded else 0)
    rhs_width = p.tile_n * p.rhs_bytes + (6 if padded else 0)
    acc_width = p.tile_n * p.accumulator_bytes + (8 if padded else 0)
    s = DenseLoopStorage(
        DenseMemoryView("left", "left_sram", lhs_width, 2, 12),
        DenseMemoryView("right", "right_sram", rhs_width, 3, 12),
        DenseMemoryView("sums", "acc_sram", acc_width, 1, 12),
    )
    r = DenseResources(
        12 * (lhs_width + rhs_width), 12 * acc_width, 3, 3, 2, partial_output_tiles=True, partial_output_blocks=True
    )
    return p, r, s


def schedule(p, r, s, *, dtypes=("i8", "i8", "i32"), **kwargs):
    return schedule_compact_dense(
        p,
        r,
        s,
        ADAPTER,
        name="dense",
        dtypes=dtypes,
        static_budget=80,
        external_buffers_disjoint=True,
        verification_timeout_ms=20_000,
        **kwargs,
    )


@pytest.mark.parametrize("order", [("m", "n"), ("n", "m")])
@pytest.mark.parametrize("peeled", [False, True])
def test_tails_reuse_and_independent_execution(order, peeled):
    import numpy as np

    p, r, s = fixture(shape=(5, 14, 25) if peeled else (5, 14, 13))
    if peeled:
        r = replace(r, max_m_blocks=2, max_n_blocks=2)
    result = schedule(p, r, s, outer_order=order)
    assert result is not None
    lhs = (np.arange(p.m * p.k) % 7 - 3).reshape(p.m, p.k)
    rhs = (np.arange(p.k * p.n) % 5 - 2).reshape(p.k, p.n)
    output = np.full((p.m, p.n), -10000, dtype=np.int64)
    # Execute projected *actual* instruction operands. Source reduction order
    # is independently observable: each scalar multiply updates the accumulator.
    memories = {v.name: {} for v in (s.lhs, s.rhs, s.accumulator)}
    transfers = {"lhs": 0, "weight": 0}
    requested = 0
    preload = None
    for call, env in instances(result.kernel):
        concrete = replace(call, args=tuple((k, evaluate(v, env)) for k, v in call.args))
        op = project(concrete)
        if isinstance(op, Load):
            view = s.lhs if op.role == "lhs" else s.rhs
            source = lhs if op.role == "lhs" else rhs
            for i in range(op.rows):
                for j in range(op.cols):
                    memories[view.name][(op.row + i, j)] = int(source[op.dram_row + i, op.dram_col + j])
            transfers[op.role] += 1
            requested += op.rows * op.cols * (p.lhs_bytes if op.role == "lhs" else p.rhs_bytes)
        elif isinstance(op, Preload):
            preload = op
        elif isinstance(op, Compute):
            for i in range(op.rows):
                for j in range(preload.cols):
                    slot = (preload.accumulator_row + i, j)
                    value = memories[s.accumulator.name][slot] if preload.accumulate else 0
                    for k in range(op.cols):
                        value += (
                            memories[s.lhs.name][(op.input_row + i, k)]
                            * memories[s.rhs.name][(preload.weight_row + k, j)]
                        )
                    memories[s.accumulator.name][slot] = value
        else:
            for i in range(op.rows):
                for j in range(op.cols):
                    output[op.dram_row + i, op.dram_col + j] = memories[s.accumulator.name][(op.accumulator_row + i, j)]
    np.testing.assert_array_equal(output, lhs @ rhs)
    report = result.realization
    assert transfers == {"lhs": report["lhs_loads"], "weight": report["rhs_loads"]}
    assert requested == report["requested_load_bytes"]
    assert report["requested_store_bytes"] == output.size * p.accumulator_bytes
    assert report["lhs_loads"] < p.tiles[0] * p.tiles[1] * p.tiles[2]
    assert report["static_instruction_calls"] <= 80
    assert report["machine_emission"] == "not_established"


def mutate(kernel, instruction, operand=None, value=None, *, omit=False):
    changed = False

    def walk(body):
        nonlocal changed
        out = []
        for statement in body:
            if isinstance(statement, Loop):
                out.append(replace(statement, body=walk(statement.body)))
            elif not changed and statement.instr == instruction:
                changed = True
                if not omit:
                    out.append(
                        replace(
                            statement, args=tuple((k, as_expr(value) if k == operand else v) for k, v in statement.args)
                        )
                    )
            else:
                out.append(statement)
        return tuple(out)

    result = replace(kernel, body=walk(kernel.body))
    assert changed
    return result


@pytest.mark.parametrize(
    "instruction,operand,value,omit",
    [
        ("lhs_load", None, None, True),
        ("weight_load", "dst", 100, False),
        ("lhs_load", "src_col", 1, False),
        ("lhs_load", "rows", 2, False),
        ("preload", "continue", 1, False),
        ("preload", "continue", 0, False),
        ("preload", "acc", 1, False),
        ("preload", "weight", 3, False),
        ("store", "row", 1, False),
        ("store", None, None, True),
        ("compute", "k", 1, False),
        ("preload", None, None, True),
    ],
)
def test_corrupted_actual_effects_are_rejected(instruction, operand, value, omit):
    p, r, s = fixture()
    good = schedule(p, r, s).kernel
    bad = mutate(good, instruction, operand, value, omit=omit)
    with pytest.raises(ValueError):
        verify_compact_dense(bad, p, r, s, ADAPTER, static_budget=80, timeout_ms=20_000)


def test_unavailable_and_unsupported_capabilities():
    p, r, s = fixture()
    assert schedule(p, replace(r, operand_bytes=1), s) is None
    assert schedule(p, r, replace(s, lhs=replace(s.lhs, capacity=1))) is None
    with pytest.raises(ValueError, match="aliased external"):
        schedule_compact_dense(
            p,
            r,
            s,
            ADAPTER,
            name="dense",
            dtypes=("i8", "i8", "i32"),
            static_budget=80,
            external_buffers_disjoint=False,
        )
    block = dense_blocks(p, r)[0]
    for adapter in (
        replace(ADAPTER, synchronous_completion=False),
        replace(ADAPTER, exact_accumulator_continuation=False),
    ):
        with pytest.raises(ValueError, match="completion"):
            emit_compact_dense(p, block, s, adapter, name="dense", dtypes=("i8", "i8", "i32"))
    with pytest.raises(ValueError, match="zero seed"):
        emit_compact_dense(replace(p, bias=True), block, s, ADAPTER, name="dense", dtypes=("i8", "i8", "i32"))
    with pytest.raises(ValueError, match="addressable accumulator"):
        emit_compact_dense(p, block, replace(s, accumulator=None), ADAPTER, name="dense", dtypes=("i8", "i8", "i32"))


def test_physical_aliases_and_completion_tokens_are_not_erased():
    p, r, s = fixture()
    good = schedule(p, r, s).kernel
    aliased = replace(s, rhs=replace(s.rhs, backing=s.lhs.backing))
    with pytest.raises(ValueError, match="alias"):
        verify_compact_dense(good, p, r, aliased, ADAPTER, static_budget=80, timeout_ms=20_000)

    def async_project(call):
        return project(call)

    # A real token in the output is refused regardless of a synchronous label.
    serial = 0

    def annotate(body):
        nonlocal serial
        out = []
        for statement in body:
            serial += 1
            out.append(
                replace(statement, body=annotate(statement.body))
                if isinstance(statement, Loop)
                else replace(statement, produces=f"pending_{serial}")
            )
        return tuple(out)

    with pytest.raises(ValueError, match="completion tokens"):
        verify_compact_dense(
            replace(good, body=annotate(good.body)), p, r, s, replace(ADAPTER, project=async_project), static_budget=80
        )


def test_projection_cannot_promote_a_dynamic_flag_to_a_boolean_capability():
    p, r, s = fixture()
    good = schedule(p, r, s).kernel

    def bad_project(call):
        op = project(call)
        return replace(op, fresh_weights=as_expr(1)) if isinstance(op, Compute) else op

    with pytest.raises(ValueError, match="adjacent preload"):
        verify_compact_dense(good, p, r, s, replace(ADAPTER, project=bad_project), static_budget=80)


@pytest.mark.parametrize("padded", [False, True])
def test_current_resource_budgets_and_alignment(padded):
    p, r, s = fixture(padded=padded)
    result = schedule(p, r, s)
    used = result.realization["padded_high_water_bytes"]
    exact = replace(r, operand_bytes=used[s.lhs.name] + used[s.rhs.name], accumulator_bytes=used[s.accumulator.name])
    verify_compact_dense(result.kernel, p, exact, s, ADAPTER, static_budget=80, timeout_ms=20_000)
    for too_small in (
        replace(exact, operand_bytes=exact.operand_bytes - 1),
        replace(exact, accumulator_bytes=exact.accumulator_bytes - 1),
    ):
        with pytest.raises(ValueError, match="resource budget"):
            verify_compact_dense(result.kernel, p, too_small, s, ADAPTER, static_budget=80, timeout_ms=20_000)
    p, r, s = fixture(tile=(3, 3, 4))
    result = schedule(p, r, s)
    bad_alignment = replace(s, lhs=replace(s.lhs, alignment=2))
    with pytest.raises(ValueError, match="storage bounds"):
        verify_compact_dense(result.kernel, p, r, bad_alignment, ADAPTER, static_budget=80, timeout_ms=20_000)


def test_symbolic_huge_domains_have_bounded_ir():
    p, r, s = fixture(shape=(2_000_000, 30_000, 4_000_000))
    result = schedule(p, r, s)
    assert result.realization["executed_instruction_calls"] > 10**15
    assert result.realization["static_instruction_calls"] < 20
    assert len(result.kernel.text()) < 10_000


def test_physical_padding_and_separate_pools_narrow_k_before_ranking():
    p, r, s = fixture(shape=(4, 60, 8), padded=True)
    r = replace(r, max_k_blocks=20, max_m_blocks=1, max_n_blocks=1)
    result = schedule(p, r, s)
    assert result is not None
    assert result.choice.block.k <= min(s.lhs.capacity // p.tile_m, s.rhs.capacity // p.tile_k)
    assert result.choice.block.k < dense_blocks(p, r)[0].k


@pytest.mark.parametrize("failure", ["budget", "invariant"])
def test_bounded_repair_rechecks_actual_fallback_and_never_hides_a_bug(monkeypatch, failure):
    from merlin.compile.scheduling import compact_verification as checker

    original = checker.verify_compact_dense
    checked = []

    def check(kernel, *args, **kwargs):
        checked.append(kernel.digest())
        if len(checked) == 1:
            error = checker.DenseVerificationUnavailable if failure == "budget" else ValueError
            raise error("injected " + failure)
        return original(kernel, *args, **kwargs)

    monkeypatch.setattr(checker, "verify_compact_dense", check)
    p, r, s = fixture()
    if failure == "invariant":
        with pytest.raises(ValueError, match="invariant"):
            schedule(p, r, s)
        assert len(checked) == 1
    else:
        result = schedule(p, r, s)
        assert len(checked) == 2 and checked[0] != checked[1]
        assert result.choice.block.m == result.choice.block.n == result.choice.block.k == 1
        assert result.realization["repairs"] == [{"discarded_kernel_sha256": checked[0], "reason": "injected budget"}]
        assert result.realization["kernel_sha256"] == checked[1]


@pytest.mark.parametrize(
    "dtypes,widths",
    [(("i8", "i8", "i32"), (1, 1, 4)), (("i16", "i8", "i32"), (2, 1, 4)), (("f32", "f32", "f32"), (4, 4, 4))],
)
def test_existing_c_emitter_executes_actual_compact_program(tmp_path, dtypes, widths):
    import ctypes
    import shutil
    import subprocess

    import numpy as np

    from merlin.sched.codegen import emit_c_function
    from merlin.sched.isa import InstrDef, InstructionSet, Operand

    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("native C compiler unavailable")
    p, r, s = fixture(widths=widths, padded=True)
    result = schedule(p, r, s, dtypes=dtypes)
    # The baseline shares the original arithmetic and primitive, retaining only
    # one output tile; the optimized arm must actually issue fewer byte reads.
    baseline_block = next(b for b in dense_blocks(p, r) if b.m == b.n == 1)
    baseline = emit_compact_dense(p, baseline_block, s, ADAPTER, name="baseline", dtypes=dtypes)
    baseline_report = verify_compact_dense(baseline, p, r, s, ADAPTER, static_budget=80, timeout_ms=20_000)
    ct = {"i8": "int8_t", "i16": "int16_t", "i32": "int32_t", "f32": "float"}
    np_types = {"i8": np.int8, "i16": np.int16, "i32": np.int32, "f32": np.float32}
    definitions = {}
    for call, _env in instances(result.kernel):
        if call.instr in definitions:
            continue
        suffix = (
            ", lhs"
            if call.instr == "lhs_load"
            else ", rhs"
            if call.instr == "weight_load"
            else ", output"
            if call.instr == "store"
            else ""
        )
        definitions[call.instr] = InstrDef(
            call.instr,
            tuple(Operand(k, "int") for k, _v in call.args),
            render_c=lambda a, n=call.instr, end=suffix: f"do_{n}({', '.join(a)}{end});",
        )
    iset = InstructionSet("neutral-native-test", definitions, ct, "/* synchronous calls complete here */")
    source = f"""
#include <stdint.h>
#include <string.h>
static unsigned char left[{(s.lhs.base + s.lhs.capacity) * s.lhs.unit_bytes}];
static unsigned char right[{(s.rhs.base + s.rhs.capacity) * s.rhs.unit_bytes}];
static unsigned char sums[{(s.accumulator.base + s.accumulator.capacity) * s.accumulator.unit_bytes}];
uint64_t load_bytes, commands;
static int weight_row, sum_row, continuing, output_cols;
static void do_lhs_load(int a,int b,int r,int c,int rows,int cols,int dst,int step,{ct[dtypes[0]]} *src) {{
  commands++; load_bytes += rows*cols*sizeof(*src);
  for(int i=0;i<rows;i++) for(int j=0;j<cols;j++)
    memcpy(left+(dst+i)*{s.lhs.unit_bytes}+j*sizeof(*src),src+(r+i*step)*{p.k}+c+j,sizeof(*src));
}}
static void do_weight_load(int a,int b,int r,int c,int rows,int cols,int dst,int step,{ct[dtypes[1]]} *src) {{
  commands++; load_bytes += rows*cols*sizeof(*src);
  for(int i=0;i<rows;i++) for(int j=0;j<cols;j++)
    memcpy(right+(dst+i)*{s.rhs.unit_bytes}+j*sizeof(*src),src+(r+i*step)*{p.n}+c+j,sizeof(*src));
}}
static void do_preload(int w,int k,int n,int acc,int cont,int rows,int cols,int depth) {{
  commands++; weight_row=w; sum_row=acc; continuing=cont; output_cols=cols;
}}
static void do_compute(int l,int m,int k,int rows,int depth) {{
  commands++;
  for(int i=0;i<rows;i++) for(int j=0;j<output_cols;j++) {{
    {ct[dtypes[2]]} acc=0;
    if(continuing) memcpy(&acc,sums+(sum_row+i)*{s.accumulator.unit_bytes}+j*sizeof(acc),sizeof(acc));
    for(int t=0;t<depth;t++) {{
      {ct[dtypes[0]]} x; {ct[dtypes[1]]} y;
      memcpy(&x,left+(l+i)*{s.lhs.unit_bytes}+t*sizeof(x),sizeof(x));
      memcpy(&y,right+(weight_row+t)*{s.rhs.unit_bytes}+j*sizeof(y),sizeof(y));
      acc = acc + x*y;
    }}
    memcpy(sums+(sum_row+i)*{s.accumulator.unit_bytes}+j*sizeof(acc),&acc,sizeof(acc));
  }}
}}
static void do_store(int r,int c,int rows,int cols,int acc,{ct[dtypes[2]]} *dst) {{
  commands++;
  for(int i=0;i<rows;i++) for(int j=0;j<cols;j++)
    memcpy(dst+(r+i)*{p.n}+c+j,sums+(acc+i)*{s.accumulator.unit_bytes}+j*sizeof(*dst),sizeof(*dst));
}}
"""
    source += emit_c_function(baseline, iset, symbol="baseline", index_bits=ADAPTER.index_bits)
    source += result.emit_c(iset, symbol="optimized")
    assert "for (int64_t" in source
    for arm in ("baseline", "optimized"):
        source += (
            f"void run_{arm}({ct[dtypes[0]]} *lhs,{ct[dtypes[1]]} *rhs,{ct[dtypes[2]]} *out) {{ "
            f"load_bytes=commands=0; {arm}(lhs,rhs,out); }}\n"
        )
    path = tmp_path / "program.c"
    path.write_text(source)
    # The suite retains only failed temp trees; a passed parameter may release
    # its directory while dlopen still caches the library by absolute pathname.
    library = tmp_path / ("program_" + "_".join(dtypes) + ".so")
    subprocess.run(
        [cc, "-std=c11", "-O1", "-ffp-contract=off", "-shared", "-fPIC", str(path), "-o", str(library)],
        check=True,
        capture_output=True,
    )
    compiled = ctypes.CDLL(str(library))
    lhs = (np.arange(p.m * p.k) % 7 - 3).reshape(p.m, p.k).astype(np_types[dtypes[0]])
    rhs = (np.arange(p.k * p.n) % 5 - 2).reshape(p.k, p.n).astype(np_types[dtypes[1]])
    if dtypes[0] == "f32":
        lhs /= np.float32(3.7)
        rhs /= np.float32(2.3)
    expected = np.zeros((p.m, p.n), dtype=np_types[dtypes[2]])
    for k in range(p.k):
        expected += lhs[:, k : k + 1].astype(expected.dtype) * rhs[k : k + 1, :].astype(expected.dtype)
    observed_traffic = []
    for arm, report in (("baseline", baseline_report), ("optimized", result.realization)):
        output = np.full_like(expected, -111)
        fn = getattr(compiled, "run_" + arm)
        fn.argtypes = [np.ctypeslib.ndpointer(dtype=x.dtype, flags="C_CONTIGUOUS") for x in (lhs, rhs, output)]
        fn(lhs, rhs, output)
        np.testing.assert_array_equal(output, expected)
        observed_traffic.append(ctypes.c_uint64.in_dll(compiled, "load_bytes").value)
        assert observed_traffic[-1] == report["requested_load_bytes"]
        assert ctypes.c_uint64.in_dll(compiled, "commands").value == report["executed_instruction_calls"]
    assert observed_traffic[1] < observed_traffic[0]
