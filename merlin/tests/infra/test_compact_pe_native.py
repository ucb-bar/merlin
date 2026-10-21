"""Execute emitted retained-PE programs with independent physical byte storage."""

import ctypes
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_compact_dense import fixture
from test_compact_pe_weights import ADAPTER_PE

from merlin.common import invocation_record as I
from merlin.compile.scheduling import block_schedule, compact_dense, compact_verification
from merlin.compile.scheduling.compact_dense import schedule_compact_dense
from merlin.sched.ir import Call, Loop
from merlin.sched.isa import InstrDef, InstructionSet, Operand


def calls(body):
    for statement in body:
        if isinstance(statement, Loop):
            yield from calls(statement.body)
        else:
            assert isinstance(statement, Call)
            yield statement


def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


@pytest.mark.parametrize(
    "dtypes,widths,unified",
    [
        (("i8", "i8", "i32"), (1, 1, 4), False),
        (("i16", "i8", "i32"), (2, 1, 4), True),
        (("f32", "f32", "f32"), (4, 4, 4), False),
    ],
)
def test_actual_c_keeps_pe_weights_across_multiple_accumulators(tmp_path, dtypes, widths, unified):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native C compiler unavailable")
    compiler = Path(compiler).resolve()
    p, r, storage = fixture(shape=(7, 13, 25), widths=widths, padded=True)
    if unified:
        left = replace(storage.lhs, backing="shared")
        right = replace(
            storage.rhs,
            backing="shared",
            base=((left.base + left.capacity) * left.unit_bytes + storage.rhs.unit_bytes - 1) // storage.rhs.unit_bytes
            + 4,
        )
        sums = replace(
            storage.accumulator,
            backing="shared",
            base=((right.base + right.capacity) * right.unit_bytes + storage.accumulator.unit_bytes - 1)
            // storage.accumulator.unit_bytes
            + 4,
        )
        storage = replace(storage, lhs=left, rhs=right, accumulator=sums)
    results = [
        schedule_compact_dense(
            p,
            r,
            storage,
            ADAPTER_PE,
            name=name,
            dtypes=dtypes,
            static_budget=100,
            external_buffers_disjoint=True,
            verification_timeout_ms=20_000,
            reuse_pe_weights=selected,
        )
        for name, selected in (("baseline", False), ("selected", True))
    ]
    assert results[0].choice.block == results[1].choice.block
    ctype = dict(i8="int8_t", i16="int16_t", i32="int32_t", f32="float")
    ntype = dict(i8=np.int8, i16=np.int16, i32=np.int32, f32=np.float32)
    definitions = {}
    for result in results:
        for call in calls(result.kernel.body):
            suffix = (
                ",lhs"
                if call.instr == "lhs_load"
                else ",rhs"
                if call.instr == "weight_load"
                else ",output"
                if call.instr == "store"
                else ""
            )
            definitions[call.instr] = InstrDef(
                call.instr,
                tuple(Operand(k, "int") for k, _ in call.args),
                render_c=lambda a, n=call.instr, end=suffix: f"do_{n}({','.join(a)}{end});",
            )
    isa = InstructionSet("neutral-pe-machine", definitions, ctype, "/* all neutral operations synchronous */")
    views = (storage.lhs, storage.rhs, storage.accumulator)
    backings = {name: f"bytes{i}" for i, name in enumerate(sorted({v.backing for v in views}))}
    extents = {
        name: max((v.base + v.capacity) * v.unit_bytes for v in views if v.backing == name) + 16 for name in backings
    }
    declarations = "\n".join(f"static unsigned char {backings[name]}[{extent}];" for name, extent in extents.items())
    declarations += "\n" + "\n".join(
        f"#define {name} {backings[view.backing]}" for name, view in zip(("left", "right", "sums"), views)
    )
    init = "".join(f"memset({backings[name]},88,{extent});" for name, extent in extents.items())
    guard = ""
    for name, extent in extents.items():
        allowed = "||".join(
            f"(i>={v.base * v.unit_bytes}&&i<{(v.base + v.capacity) * v.unit_bytes})"
            for v in views
            if v.backing == name
        )
        guard += f"for(int i=0;i<{extent};i++)if(!({allowed})&&{backings[name]}[i]!=88)return 0;"
    acc = "uint32_t" if dtypes[2] == "i32" else "float"
    muladd = "value+=(uint32_t)(int32_t)x*(uint32_t)(int32_t)y;" if dtypes[2] == "i32" else "value=value+x*y;"
    source = f"""#include <stdint.h>
#include <string.h>
{declarations}
static {ctype[dtypes[1]]} pe[{p.tile_k * p.tile_n}];
static int64_t target_row,continuation,columns,depth_saved;
uint64_t pe_copies,pe_bytes,dram_bytes,computes;int machine_error;
void reset(void){{{init} memset(pe,0,sizeof(pe));pe_copies=pe_bytes=dram_bytes=computes=machine_error=0;}}
int guards_intact(void){{{guard}return 1;}}
static void do_lhs_load(int64_t a,int64_t b,int64_t row,int64_t col,
 int64_t rows,int64_t cols,int64_t dst,int64_t step,{ctype[dtypes[0]]}*src){{
 dram_bytes+=rows*cols*sizeof(*src);
 for(int i=0;i<rows;i++)for(int j=0;j<cols;j++)
  memcpy(left+(dst+i)*{storage.lhs.unit_bytes}+j*sizeof(*src),src+(row+i*step)*{p.k}+col+j,sizeof(*src));
}}
static void do_weight_load(int64_t a,int64_t b,int64_t row,int64_t col,
 int64_t rows,int64_t cols,int64_t dst,int64_t step,{ctype[dtypes[1]]}*src){{
 dram_bytes+=rows*cols*sizeof(*src);
 for(int i=0;i<rows;i++)for(int j=0;j<cols;j++)
  memcpy(right+(dst+i)*{storage.rhs.unit_bytes}+j*sizeof(*src),src+(row+i*step)*{p.n}+col+j,sizeof(*src));
}}
static void do_preload(int64_t row,int64_t k,int64_t n,int64_t acc,
 int64_t cont,int64_t rows,int64_t cols,int64_t depth){{
 pe_copies++;pe_bytes+=depth*cols*sizeof(pe[0]);depth_saved=depth;
 for(int i=0;i<depth;i++)for(int j=0;j<cols;j++)
  memcpy(&pe[i*{p.tile_n}+j],right+(row+i)*{storage.rhs.unit_bytes}+j*sizeof(pe[0]),sizeof(pe[0]));
 target_row=acc;continuation=cont;columns=cols;
}}
static void do_preload_keep(int64_t k,int64_t n,int64_t acc,int64_t cont,int64_t rows,int64_t cols,int64_t depth){{
 if(depth!=depth_saved)machine_error=1;target_row=acc;continuation=cont;columns=cols;
}}
static void do_compute(int64_t row,int64_t m,int64_t k,int64_t rows,int64_t depth){{
 computes++;if(depth!=depth_saved)machine_error=1;
 for(int i=0;i<rows;i++)for(int j=0;j<columns;j++){{
  {acc} value=0;
  if(continuation)memcpy(&value,sums+(target_row+i)*{storage.accumulator.unit_bytes}+j*sizeof(value),sizeof(value));
  for(int t=0;t<depth;t++){{
   {ctype[dtypes[0]]} x;{ctype[dtypes[1]]} y=pe[t*{p.tile_n}+j];
   memcpy(&x,left+(row+i)*{storage.lhs.unit_bytes}+t*sizeof(x),sizeof(x));{muladd}
  }}
  memcpy(sums+(target_row+i)*{storage.accumulator.unit_bytes}+j*sizeof(value),&value,sizeof(value));
 }}
}}
static void do_compute_keep(int64_t row,int64_t m,int64_t k,int64_t rows,int64_t depth){{
 do_compute(row,m,k,rows,depth);
}}
static void do_store(int64_t row,int64_t col,int64_t rows,int64_t cols,int64_t acc,{ctype[dtypes[2]]}*out){{
 for(int i=0;i<rows;i++)for(int j=0;j<cols;j++)
  memcpy(out+(row+i)*{p.n}+col+j,sums+(acc+i)*{storage.accumulator.unit_bytes}+j*sizeof(*out),sizeof(*out));
}}
""" + "\n".join(result.emit_c(isa, symbol=name) for result, name in zip(results, ("baseline", "selected")))
    source += "\n".join(
        f"void run_{name}({ctype[dtypes[0]]}*a,{ctype[dtypes[1]]}*b,{ctype[dtypes[2]]}*c){{{name}(a,b,c);}}"
        for name in ("baseline", "selected")
    )
    cpath = tmp_path / "program.c"
    cpath.write_text(source)
    so = tmp_path / "program.so"
    owners = [Path(module.__file__) for module in (compact_dense, compact_verification, block_schedule)]
    owners += [Path(__file__), Path(ADAPTER_PE.encode.__code__.co_filename)]
    sources = [pin(path) for path in owners]
    run = I.run(
        [str(compiler), "-std=c11", "-O2", "-fPIC", "-shared", "-ffp-contract=off", str(cpath), "-o", str(so)],
        directory=tmp_path / "compile",
        stage="neutral_pe_native_compile",
        inputs=(cpath, *owners),
        outputs=(so,),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode == 0, run.stderr
    library = ctypes.CDLL(str(so))
    arms = []
    for result, name in zip(results, ("baseline", "selected")):
        function = getattr(library, "run_" + name)
        function.argtypes = [ctypes.c_void_p] * 3
        function.restype = None
        carry = 1
        observed = []
        arrays = []
        for epoch in range(3):
            A = ((np.arange(p.m * p.k).reshape(p.m, p.k) * 3 + carry) % 7 - 3).astype(ntype[dtypes[0]])
            B = (
                (np.arange(p.k * p.n).reshape(p.k, p.n) * 7 + np.arange(p.k * p.n).reshape(p.k, p.n) // p.n * 3 + carry)
                % 5
                - 2
            ).astype(ntype[dtypes[1]])
            expected = np.empty((p.m, p.n), dtype=ntype[dtypes[2]])
            for m in range(p.m):
                for n in range(p.n):
                    value = 0
                    for k in range(p.k):
                        value += int(A[m, k]) * int(B[k, n])
                    expected[m, n] = value
            buffers = []
            payload = []
            for value in (A, B, np.zeros_like(expected)):
                raw = np.full(value.nbytes + 64, 0x5A, np.uint8)
                data = raw[32:-32].view(value.dtype).reshape(value.shape)
                data[:] = value
                buffers.append(raw)
                payload.append(data)
            before = [buffer.tobytes() for buffer in buffers[:2]]
            library.reset()
            function(*(ctypes.c_void_p(value.ctypes.data) for value in payload))
            np.testing.assert_array_equal(payload[2], expected)
            assert before == [buffer.tobytes() for buffer in buffers[:2]]
            assert all(np.all(raw[:32] == 0x5A) and np.all(raw[-32:] == 0x5A) for raw in buffers)
            assert library.guards_intact() == 1 and ctypes.c_int.in_dll(library, "machine_error").value == 0
            counts = {
                key: ctypes.c_uint64.in_dll(library, key).value
                for key in ("pe_copies", "pe_bytes", "dram_bytes", "computes")
            }
            assert counts["computes"] == p.tiles[0] * p.tiles[1] * p.tiles[2]
            carry = (int(payload[2][0, 0]) + epoch + 1) % 11
            record = tmp_path / f"{name}-{epoch}.npz"
            np.savez(
                record,
                A=payload[0],
                B=payload[1],
                C=payload[2],
                guards=np.concatenate([np.r_[raw[:32], raw[-32:]] for raw in buffers]),
            )
            arrays.append(pin(record))
            observed.append(dict(epoch=epoch, carry=carry, **counts))
        assert len({row["carry"] for row in observed}) > 1
        arms.append(dict(arm=name, arrays=arrays, actual_native=observed, checked_kernel=result.kernel.digest()))
    for before, after in zip(arms[0]["actual_native"], arms[1]["actual_native"]):
        assert after["pe_copies"] < before["pe_copies"] and after["pe_bytes"] < before["pe_bytes"]
        assert (
            after["dram_bytes"] == before["dram_bytes"]
            and after["computes"] == before["computes"]
            and after["carry"] == before["carry"]
        )
    invocations = list((tmp_path / "compile").rglob("invocation.json"))
    assert len(invocations) == 1
    I.verify(invocations[0])
    retained = [pin(cpath), pin(so), pin(invocations[0]), *[row for arm in arms for row in arm["arrays"]]]
    for row in [*sources, *retained]:
        assert pin(Path(row["path"])) == row
    (tmp_path / "pair.json").write_text(
        json.dumps(
            dict(
                arms=arms,
                sources=sources,
                retained=retained,
                scope=(
                    "Neutral native original PE-copy semantics; complete three-call carry/input/guard checks. "
                    "PE-copy reduction only; unchanged DRAM requests/computes, no hardware latency claim."
                ),
            ),
            indent=2,
        )
        + "\n"
    )
