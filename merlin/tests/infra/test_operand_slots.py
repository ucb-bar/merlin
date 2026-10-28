"""Operand-specific slots reach counted SCF/LLVM and an asynchronous neutral ISA."""

import ctypes
import io
import json
import shutil
import subprocess
from dataclasses import replace

import pytest
from xdsl.dialects import llvm, scf
from xdsl.dialects.builtin import ModuleOp, i64
from xdsl.printer import Printer

from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter
from merlin.compile.scheduling.operand_slots import (
    PipelineOperand,
    allocate_pipeline_operand_slots,
    verify_pipeline_operand_slots,
)
from merlin.compile.scheduling.program_limits import AddressWindow, EncodedAccess
from merlin.compile.scheduling.scf_commands import emit_scf_commands, materialize_scf_commands
from merlin.compile.scheduling.temporary_storage import (
    StorageAdapter,
    StorageEffects,
    StorageOptimizationUnavailable,
    TemporaryObject,
)
from merlin.llvmlower.host_builder import FnBuilder
from merlin.sched.ir import Call, Const, Expr, Kernel, Loop, Var, evaluate, select
from merlin.semantic_compiler.allocation_types import Reservation, StorageBank


def command(name, **args):
    return Call(
        "slot_" + name, tuple((key, value if isinstance(value, Expr) else Const(value)) for key, value in args.items())
    )


def project(call):
    args = {key: evaluate(value, {}) for key, value in call.args}
    instruction = call.instr.removeprefix("slot_")
    local = {"fixed": ("address",), "fetch": ("address",), "consume": ("lhs", "rhs"), "finish": ()}
    if instruction not in local:
        return None
    key = repr((call.instr, tuple((name, value) for name, value in args.items() if name not in local[instruction])))
    if instruction == "finish":
        return StorageEffects(key, completion="wait", waits=(str(args["ticket"]),))
    if instruction in ("fixed", "fetch"):
        accesses = (
            EncodedAccess("fixed" if instruction == "fixed" else "input", args["source"], args["size"]),
            EncodedAccess("memory", args["address"], args["size"], True),
        )
    else:
        accesses = (
            EncodedAccess("alias", args["lhs"], args["size"]),
            EncodedAccess("alias", args["rhs"], 1),
            EncodedAccess("output", args["destination"], args["size"], True),
        )
    if instruction == "fixed":
        return StorageEffects(key, accesses, "synchronous")
    return StorageEffects(key, accesses, "asynchronous", str(args["ticket"]))


def rewrite(call, accesses):
    if call.instr in ("slot_fixed", "slot_fetch"):
        replacements = {"address": accesses[1].start}
    else:
        assert call.instr == "slot_consume"
        replacements = {"lhs": accesses[0].start, "rhs": accesses[1].start}
    return replace(
        call,
        args=tuple((name, Const(replacements[name]) if name in replacements else value) for name, value in call.args),
    )


ADAPTER = StorageAdapter(
    project,
    rewrite,
    "neutral-byte-target-v1",
    "unsigned-byte-add-v1",
    "test-only-deferred-copy-and-read-at-completion-v1",
)


def source(total=101, width=16, *, overlap=True, lookahead=1):
    tiles = (total + width - 1) // width
    assert tiles >= 3
    tail = total - (tiles - 1) * width
    i = Var("tile")
    fixed = command("fixed", address=512, source=0, size=1)
    if overlap:
        kernel = Kernel(
            "pipeline",
            (),
            (
                fixed,
                *(
                    command("fetch", address=1024 + j * 64, source=j * width, size=width, ticket=j)
                    for j in range(lookahead)
                ),
                Loop(
                    "tile",
                    tiles - lookahead,
                    (
                        command("finish", ticket=i),
                        command(
                            "consume", lhs=1024 + i * 64, rhs=512, destination=i * width, size=width, ticket=100 + i
                        ),
                        command(
                            "fetch",
                            address=1024 + (i + lookahead) * 64,
                            source=(i + lookahead) * width,
                            size=select("tile", tiles - lookahead - 1, tail, width),
                            ticket=i + lookahead,
                        ),
                        command("finish", ticket=100 + i),
                    ),
                ),
                *(
                    call
                    for j in range(tiles - lookahead, tiles)
                    for call in (
                        command("finish", ticket=j),
                        command(
                            "consume",
                            lhs=1024 + j * 64,
                            rhs=512,
                            destination=j * width,
                            size=tail if j == tiles - 1 else width,
                            ticket=100 + j,
                        ),
                        command("finish", ticket=100 + j),
                    )
                ),
            ),
        )
    else:
        kernel = Kernel(
            "serial",
            (),
            (
                fixed,
                Loop(
                    "tile",
                    tiles,
                    (
                        command(
                            "fetch",
                            address=1024 + i * 64,
                            source=i * width,
                            size=select("tile", tiles - 1, tail, width),
                            ticket=i,
                        ),
                        command("finish", ticket=i),
                        command(
                            "consume",
                            lhs=1024 + i * 64,
                            rhs=512,
                            destination=i * width,
                            size=select("tile", tiles - 1, tail, width),
                            ticket=100 + i,
                        ),
                        command("finish", ticket=100 + i),
                    ),
                ),
            ),
        )
    operands = (
        PipelineOperand(
            "advancing",
            tuple(
                TemporaryObject(f"x{i}", "memory", 1024 + i * 64, min(width, total - i * width), 4)
                for i in range(tiles)
            ),
        ),
        PipelineOperand("invariant", (TemporaryObject("fixed", "memory", 512, 1, 4),)),
    )
    capacity = 8 + (lookahead + 1) * ((width + 3) // 4 * 4)
    banks = tuple(StorageBank(name, "same_physical_bytes", capacity, "bytes", 4) for name in ("memory", "alias"))
    model = dict(
        private_ownership=True,
        reservations=(Reservation("alias", 0, 4),),
        external_windows=(
            AddressWindow("input", 0, total),
            AddressWindow("output", 0, total),
            AddressWindow("fixed", 0, 1),
        ),
    )
    return kernel, operands, banks, model


def allocate(fixture):
    kernel, operands, banks, model = fixture
    result = allocate_pipeline_operand_slots(kernel, operands, banks, ADAPTER, **model)
    assert result is not None
    return result


def test_actual_completion_determines_each_operand_depth_and_counted_rotation():
    fixture = source()
    kernel, operands, banks, model = fixture
    result = allocate(fixture)
    rows = result.report["operands"]
    assert rows["invariant"]["required_live_generations"] == rows["invariant"]["assigned_distinct_bases"] == 1
    assert rows["advancing"]["required_live_generations"] == rows["advancing"]["assigned_distinct_bases"] == 2
    # x1 is launched before x0's asynchronous read actually completes.
    assert rows["advancing"]["lifetime_calls"]["x0"] == [1, 5]
    assert rows["advancing"]["lifetime_calls"]["x1"][0] == 4
    assert result.report["rotation"] == {"phase_extent": 2, "peeled_iterations": 0}
    assert result.report["timing"] == "UNKNOWN"
    assert verify_pipeline_operand_slots(kernel, result.kernel, operands, banks, ADAPTER, **model) == {
        key: value for key, value in result.report.items() if key != "rotation"
    }
    module = materialize_scf_commands(result.kernel)
    assert sum(type(op) is scf.ForOp for op in module.walk()) == 2
    serial = allocate(source(overlap=False))
    assert serial.report["operands"]["advancing"]["required_live_generations"] == 1
    assert serial.report["operands"]["advancing"]["assigned_distinct_bases"] == 1


def test_capacity_alias_alignment_and_premature_reuse_are_checked_on_actual_calls():
    kernel, operands, banks, model = source()
    assert (
        allocate_pipeline_operand_slots(
            kernel, operands, tuple(replace(b, capacity=36) for b in banks), ADAPTER, **model
        )
        is None
    )
    result = allocate((kernel, operands, banks, model))
    aliases = result.report["operands"]["advancing"]["addresses"]
    assert all(address % 4 == 0 and address >= 4 for address in aliases.values())
    # A changed actual alias-view capacity is binding even if the load view fits.
    assert (
        allocate_pipeline_operand_slots(kernel, operands, (banks[0], replace(banks[1], capacity=36)), ADAPTER, **model)
        is None
    )
    from merlin.compile.scheduling.operand_slots import _close

    actual = _close(result.kernel)
    body = list(actual.body)
    first = body[1].arg("address")
    body[4] = replace(
        body[4], args=tuple((name, first if name == "address" else value) for name, value in body[4].args)
    )
    with pytest.raises(ValueError, match="aliases disagree|simultaneously live|completion"):
        verify_pipeline_operand_slots(kernel, replace(actual, body=tuple(body)), operands, banks, ADAPTER, **model)


def test_unsupported_rotation_retains_the_source_and_realized_mutations_fail():
    kernel, operands, banks, model = source()
    with pytest.raises(StorageOptimizationUnavailable, match="no supported repeated"):
        allocate_pipeline_operand_slots(kernel, operands, banks, ADAPTER, max_period=1, **model)
    result = allocate((kernel, operands, banks, model))
    # A target rewrite may not silently change a source offset or remove a real
    # completion, even when the new address assignment itself still fits.
    from merlin.compile.scheduling.operand_slots import _close

    actual = _close(result.kernel)
    call = actual.body[1]
    changed = replace(call, args=tuple((name, value + 1 if name == "source" else value) for name, value in call.args))
    with pytest.raises(ValueError, match="nonaddress semantics"):
        verify_pipeline_operand_slots(
            kernel, replace(actual, body=(actual.body[0], changed, *actual.body[2:])), operands, banks, ADAPTER, **model
        )
    with pytest.raises(ValueError, match="instruction count|completion"):
        verify_pipeline_operand_slots(kernel, replace(actual, body=actual.body[:-1]), operands, banks, ADAPTER, **model)


@pytest.mark.parametrize("kind", ["budget", "nested", "carried", "unknown", "missing_wait", "mutation"])
def test_unknown_boundaries_and_incomplete_lifetimes_preserve_source(kind, monkeypatch):
    import merlin.compile.scheduling.operand_slots as slots

    kernel, operands, banks, model = source()
    digest = kernel.digest()
    if kind == "budget":
        monkeypatch.setattr(slots, "_close", lambda *_: pytest.fail("expanded before checking budget"))
        model["max_calls"] = 4
    elif kind == "nested":
        kernel = replace(kernel, body=(Loop("outer", 2, kernel.body),))
    elif kind == "carried":
        loop = replace(kernel.body[2], iter_args=(("event", "incoming"),), yields=("event",), results=("done",))
        kernel = replace(kernel, body=(*kernel.body[:2], loop, *kernel.body[3:]))
    elif kind == "unknown":
        kernel = replace(kernel, body=(replace(kernel.body[0], instr="unknown"), *kernel.body[1:]))
    elif kind == "missing_wait":
        kernel = replace(kernel, body=kernel.body[:-1])
    else:
        # A new write into a live invariant generation needs a different contract.
        kernel = replace(kernel, body=(*kernel.body, kernel.body[0]))
    with pytest.raises(ValueError):
        allocate_pipeline_operand_slots(kernel, operands, banks, ADAPTER, **model)
    assert source()[0].digest() == digest


# This neutral target keeps accesses pending until the actual finish call. It
# reads compute operands at completion, checks overlap against all pending work,
# and never treats issue/queue entry as source release. No timing model is used.
NATIVE = r"""
#include <stdint.h>
#include <string.h>
static unsigned char input[1024], output[1024], memory[2048];
typedef struct { int active, write, address, rhs, dst, n; } Job;
static Job jobs[256];
static int errors, cap, max_pending, outstanding, fetches, base[256];
static int overlaps(int a,int n,int b,int m) { return a<b+m && b<a+n; }
static void access(int address,int n,int write) {
  if(address<4 || n<=0 || address+n>cap) errors++;
  for(int i=0;i<256;i++) if(jobs[i].active) {
    Job p=jobs[i];
    if((write || p.write) && overlaps(address,n,p.address,p.n)) errors++;
    if(write && !p.write && overlaps(address,n,p.rhs,1)) errors++;
  }
}
static void launch(int t,Job j) {
  if(t<0 || t>=256 || jobs[t].active) { errors++; return; }
  jobs[t]=j; outstanding++;
  if(outstanding>max_pending) max_pending=outstanding;
}
void slot_fixed(int64_t a,int64_t s,int64_t n) {
  access(a,n,1); if(s!=0 || n!=1) errors++; memory[a]=31;
}
void slot_fetch(int64_t a,int64_t s,int64_t n,int64_t t) {
  access(a,n,1); base[fetches++]=a;
  launch(t,(Job){1,1,a,0,s,n});
}
void slot_consume(int64_t a,int64_t b,int64_t d,int64_t n,int64_t t) {
  access(a,n,0); access(b,1,0);
  launch(t,(Job){1,0,a,b,d,n});
}
void slot_finish(int64_t t) {
  if(t<0 || t>=256 || !jobs[t].active) { errors++; return; }
  Job j=jobs[t];
  if(j.write) memcpy(memory+j.address,input+j.dst,j.n);
  else for(int i=0;i<j.n;i++) output[j.dst+i]=(unsigned char)(memory[j.address+i]+memory[j.rhs]);
  jobs[t].active=0; outstanding--;
}
void setup(int n,int capacity) {
  for(int i=0;i<n;i++) input[i]=(unsigned char)(i*23+7);
  memset(output,0,sizeof(output)); memset(memory,97,sizeof(memory)); memset(jobs,0,sizeof(jobs));
  errors=max_pending=outstanding=fetches=0; cap=capacity;
}
int check(int n) {
  int e=errors+outstanding;
  for(int i=0;i<n;i++) e+=output[i]!=(unsigned char)(input[i]+31);
  for(int i=0;i<4;i++) e+=memory[i]!=97 || memory[cap+i]!=97;
  return e;
}
int pending_peak(void) { return max_pending; }
int fetch_address(int i) { return base[i]; }
"""


@pytest.mark.parametrize("total,width,lookahead", [(96, 16, 1), (101, 16, 1), (37, 9, 1), (149, 16, 2)])
def test_native_scf_llvm_consumes_real_slots_and_exact_tail_outputs(tmp_path, total, width, lookahead):
    translator = shutil.which("mlir-translate")
    clang = shutil.which("clang")
    if translator is None or clang is None:
        pytest.skip("native LLVM tools unavailable")
    functions, declarations, results = [], {}, {}
    for overlap in (False, True):
        fixture = source(total, width, overlap=overlap, lookahead=lookahead)
        result = allocate(fixture)
        results[overlap] = result
        builder = FnBuilder([])
        bridge = FnBuilderIndexAdapter(builder)

        def lower(name, args, metadata):
            declarations[name] = len(args)
            builder.add(llvm.CallOp(name, *(bridge.value(value) for value in args.values())))

        commands = materialize_scf_commands(result.kernel)
        (tmp_path / f"{result.kernel.name}.scf.mlir").write_text(str(commands))
        (tmp_path / f"{result.kernel.name}-allocation.json").write_text(json.dumps(result.report, sort_keys=True))
        emit_scf_commands(commands, bridge, lower)
        functions.append(
            llvm.FuncOp(
                result.kernel.name,
                llvm.LLVMFunctionType([], llvm.LLVMVoidType()),
                linkage=llvm.LinkageAttr("external"),
                body=builder.finish(),
            )
        )
    module = ModuleOp(
        [
            *(
                llvm.FuncOp(
                    name,
                    llvm.LLVMFunctionType([i64] * count, llvm.LLVMVoidType()),
                    linkage=llvm.LinkageAttr("external"),
                )
                for name, count in declarations.items()
            ),
            *functions,
        ]
    )
    module.verify()
    stream = io.StringIO()
    Printer(stream=stream).print_op(module)
    assert "llvm.cond_br" in stream.getvalue()
    # Successful pytest temp directories can be removed/reused while CDLL still
    # retains the old absolute pathname. Give different programs distinct names.
    mlir, ll, target, library = (
        tmp_path / name for name in ("slots.mlir", "slots.ll", "target.c", f"slots-{total}-{width}-{lookahead}.so")
    )
    mlir.write_text(stream.getvalue())
    target.write_text(NATIVE)
    subprocess.run(
        [translator, "--mlir-to-llvmir", str(mlir), "-o", str(ll)], check=True, capture_output=True, timeout=30
    )
    subprocess.run(
        [clang, "-O1", "-shared", "-fPIC", str(ll), str(target), "-o", str(library)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    native = ctypes.CDLL(str(library))
    for overlap in (False, True):
        native.setup(total, source(total, width, lookahead=lookahead)[2][0].capacity)
        getattr(native, "pipeline" if overlap else "serial")()
        assert native.check(total) == 0
        assert native.pending_peak() == (lookahead + 1 if overlap else 1)
        addresses = [native.fetch_address(i) for i in range((total + width - 1) // width)]
        expected = results[overlap].report["operands"]["advancing"]["addresses"]
        requirements = results[overlap].report["operands"]
        assert requirements["advancing"]["required_live_generations"] == (lookahead + 1 if overlap else 1)
        assert requirements["invariant"]["required_live_generations"] == 1
        assert addresses == [expected[f"x{i}"] for i in range(len(addresses))]
        assert len(set(addresses)) == (lookahead + 1 if overlap else 1)
        assert max(addresses) < 1024  # The virtual generation stride has gone.
