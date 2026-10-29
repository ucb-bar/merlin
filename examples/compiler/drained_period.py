"""Neutral counted overlap example; no target instructions or schedule recommendation.

Run with --output PROGRAM.c, then compile/run that C with an ordinary host C
compiler. Its delayed-completion runtime and scalar output comparison are
independent of the effect checker. No hardware speedup is measured.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from merlin.compile.scheduling.periodic_effects import check_drained_period, resolve_drained_period_admission
from merlin.compile.scheduling.program_limits import AddressWindow, EncodedAccess
from merlin.compile.scheduling.temporary_storage import StorageAdapter, StorageEffects, TemporaryObject
from merlin.sched.ir import Call, Const, Kernel, Loop, Var, evaluate
from merlin.sched.ir.expr import Expr
from merlin.sched.isa import InstrDef, InstructionSet, Operand
from merlin.semantic_compiler.allocation_types import StorageBank


def fixture(iterations=3):
    def call(name, slot=0, host=0, token=0):
        return Call(name, tuple((key, value if isinstance(value, Expr) else Const(value))
                               for key, value in (("slot", slot), ("host", host), ("token", token))))

    i = Var("period")
    # Asynchronous copy/outputs remain pending until actual wait commands run.
    # Both slots are freshly initialized each period; the final waits drain it.
    body = (call("fetch", 0, i * 2, 0), call("fetch", 1, i * 2 + 1, 1),
            call("wait", token=0), call("transform", 0), call("publish", 0, i * 2, 2),
            call("wait", token=1), call("transform", 1), call("publish", 1, i * 2 + 1, 3),
            call("wait", token=2), call("wait", token=3))
    kernel = Kernel("neutral_period", (), (Loop("period", iterations, body),))

    def project(command):
        v = {key: evaluate(value, {}) for key, value in command.args}
        local = EncodedAccess("local", v["slot"], 1)
        if command.instr == "fetch":
            return StorageEffects(repr(command), (EncodedAccess("input", v["host"], 1),
                                  EncodedAccess("local", v["slot"], 1, True)),
                                  "asynchronous", str(v["token"]), exact_copy=True)
        if command.instr == "publish":
            return StorageEffects(repr(command), (local, EncodedAccess("output", v["host"], 1, True)),
                                  "asynchronous", str(v["token"]), exact_copy=True)
        if command.instr == "wait":
            return StorageEffects(repr(command), completion="wait", waits=(str(v["token"]),))
        if command.instr == "transform":
            return StorageEffects(repr(command), (local, EncodedAccess("local", v["slot"], 1, True)), "synchronous")
        return None

    adapter = StorageAdapter(project, lambda command, accesses: command, "neutral-delay-runtime-v1",
                             "integer-byte-transform", "neutral-projection-assertion")
    facts = {"pending_capacity": 2}
    declaration = {"schema": "merlin.drained_period.v1", "effects": "independent_memory_and_explicit_completion",
                   "max_pending": {"fact": ["pending_capacity"]},
                   "concurrent_pairs": [["fetch", "fetch"], ["fetch", "transform"], ["fetch", "publish"],
                                        ["publish", "transform"], ["publish", "publish"]]}
    admission = resolve_drained_period_admission(declaration, facts, adapter=adapter,
                                                qualification_identity="neutral-runtime-only-assumption")
    objects = tuple(TemporaryObject("slot" + str(i), "local", i, 1) for i in range(2))
    banks = (StorageBank("local", "private", 2, "bytes", 1),)
    options = dict(max_dynamic_calls=iterations * 10, private_ownership=True,
                   external_windows=(AddressWindow("input", 0, iterations * 2),
                                     AddressWindow("output", 0, iterations * 2)))
    iset = InstructionSet("neutral", {
        name: InstrDef(name, tuple(Operand(key, "int") for key in ("slot", "host", "token")),
                       lambda args, name=name: f"op_{name}({', '.join(args)});")
        for name in ("fetch", "publish", "wait", "transform")}, {}, "assert(pending_count == 0);")
    return kernel, objects, banks, adapter, admission, options, iset


def program(iterations=3):
    kernel, objects, banks, adapter, admission, options, iset = fixture(iterations)
    checked = check_drained_period(kernel, objects, banks, adapter, admission, **options)
    emitted = checked.emit_c(iset, symbol="generated")
    runtime = f'''#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#define N {2 * iterations}
static int8_t input[N], output[N], local[2];
static int busy[2], pending_count, peak;
static struct {{ int active, kind, slot, host; }} event[4];
static void issue(int kind, int slot, int host, int token) {{
 assert(slot >= 0 && slot < 2 && host >= 0 && host < N && token >= 0 && token < 4);
 assert(!busy[slot] && !event[token].active);
 busy[slot] = 1; event[token].active = 1; event[token].kind = kind;
 event[token].slot = slot; event[token].host = host;
 pending_count++; if(pending_count > peak) peak = pending_count;
}}
static void op_fetch(int slot,int host,int token) {{ issue(1,slot,host,token); }}
static void op_publish(int slot,int host,int token) {{ issue(2,slot,host,token); }}
static void op_transform(int slot,int host,int token) {{
 (void)host; (void)token; assert(!busy[slot]); local[slot] = local[slot] * 3 + 1;
}}
static void op_wait(int slot,int host,int token) {{
 (void)slot; (void)host; assert(event[token].active);
 int s = event[token].slot, h = event[token].host;
 if(event[token].kind == 1) local[s] = input[h]; else output[h] = local[s];
 busy[s] = 0; event[token].active = 0; pending_count--;
}}
'''
    main = '''int main(void) {
 for(int i=0;i<N;i++) { input[i]=i-2; output[i]=-99; }
 generated(); assert(pending_count==0 && peak==2);
 for(int i=0;i<N;i++) assert(output[i]==input[i]*3+1);
 puts("complete outputs; peak outstanding=2; no pending work"); return 0;
}
'''
    return runtime + emitted + main, checked.report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source, report = program()
    with args.output.open("x") as output:
        output.write(source)
    print(report)
