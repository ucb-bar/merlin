#!/usr/bin/env python3
"""Prepare small BYOC guests, or run one guest under an enforced resource cap.

Run inside a <=2 GiB, no-swap, one-CPU cgroup. No toolchain or simulator is built.
The output directory retains source, IR, trace, fixture, object and ELF identities.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def budget():
    group = next(x[3:] for x in Path("/proc/self/cgroup").read_text().splitlines() if x.startswith("0::"))
    root = Path("/sys/fs/cgroup") / group.lstrip("/")
    state = {key: (root / key).read_text().strip() for key in ("memory.max", "memory.swap.max", "cpu.max", "memory.peak", "memory.events")}
    quota, period = state["cpu.max"].split()
    if state["memory.max"] == "max" or int(state["memory.max"]) > 2 << 30 or state["memory.swap.max"] != "0" or quota == "max" or int(quota) > int(period):
        raise ValueError("Requires cgroup MemoryMax<=2G, MemorySwapMax=0, CPUQuota<=100%")
    return state


def fixture(case):
    import numpy as np
    from tvm import relax

    shapes = {"tiny": ((1, 1), (1, 1)), "tails": ((3, 17), (17, 5)),
              "bmm": ((2, 2, 17), (2, 17, 3)), "conv": ((1, 2, 2, 2), (3, 2, 3, 3))}
    ash, bsh = shapes[case]
    rng = np.random.default_rng(721)
    a = rng.integers(-128, 128, ash, dtype="int8")
    b = rng.integers(-128, 128, bsh, dtype="int8")
    # Include a worst-case reduction across the PE-width boundary.
    if case == "tails":
        a.fill(-128)
        b.fill(-128)
    if case == "conv":
        padded = np.pad(a.astype("int64"), ((0, 0), (0, 0), (1, 1), (1, 1)))
        raw = np.empty((1, 3, 2, 2), "int64")
        for oc in range(3):
            for y in range(2):
                for x in range(2):
                    raw[0, oc, y, x] = np.sum(padded[0, :, y:y+3, x:x+3] * b[oc].astype("int64"))
    else:
        raw = a.astype("int64") @ b.astype("int64")
    bias = np.arange(raw.shape[-1], dtype="int32") - 7
    expected = np.maximum(raw + bias, 0).astype("int32")
    av = relax.Var("a", relax.TensorStructInfo(ash, "int8"))
    bb = relax.BlockBuilder()
    with bb.function("main", [av]):
        with bb.dataflow():
            if case == "conv":
                value = bb.emit(relax.op.nn.conv2d(av, relax.const(b), padding=(1, 1), out_dtype="int32"))
            else:
                value = bb.emit(relax.op.matmul(av, relax.const(b), out_dtype="int32"))
            value = bb.emit(relax.op.add(value, relax.const(bias)))
            value = bb.emit_output(relax.op.nn.relu(value))
        bb.emit_func_output(value)
    return bb.get(), a, b, raw.astype("int32"), expected


RUNNER = r'''
#include <stdint.h>
#include <stddef.h>
#include "fixture.h"
void *memset(void*p,int c,size_t n){uint8_t*b=p;for(size_t i=0;i<n;++i)b[i]=(uint8_t)c;return p;}
void *memcpy(void*d,const void*s,size_t n){uint8_t*a=d;const uint8_t*b=s;for(size_t i=0;i<n;++i)a[i]=b[i];return d;}
volatile uint64_t tohost __attribute__((section(".tohost"),aligned(64)));
volatile uint64_t fromhost __attribute__((section(".tohost"),aligned(64)));
static void put(const char*s){static volatile uint64_t r[4] __attribute__((aligned(8)));
  unsigned n=0;while(s[n])++n;r[0]=64;r[1]=1;r[2]=(uintptr_t)s;r[3]=n;
  asm volatile("fence rw,rw":::"memory");tohost=(uintptr_t)r;while(!fromhost){}fromhost=0;
  asm volatile("fence rw,rw":::"memory");}
int printf(const char*s,...){put("VENDOR_DIAGNOSTIC\n");return 0;}
__attribute__((noreturn)) void exit(int c){tohost=((uint64_t)(c?c:99)<<1)|1;for(;;){}}
static uint64_t cycles(void){uint64_t c;asm volatile("csrr %0,mcycle":"=r"(c));return c;}
static void metric(const char*s,uint64_t v){char line[128],digits[32];unsigned n=0,j=0;
  while(*s)line[j++]=*s++;do{digits[n++]=(char)('0'+v%10);v/=10;}while(v);
  while(n)line[j++]=digits[--n];line[j++]='\n';line[j]=0;put(line);}
extern int32_t model_run(const void*const*,void*const*,uint8_t*,uint64_t);
extern int32_t tvm_gemmini_matmul_i8_i32(const int8_t*,const int8_t*,int32_t*,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
static unsigned updates;
void __real_tvm_gemmini_tile_update_i8_i32(const int8_t*,const int8_t*,int32_t*,int64_t,int64_t,int64_t);
void __wrap_tvm_gemmini_tile_update_i8_i32(const int8_t*a,const int8_t*b,int32_t*c,int64_t sa,int64_t sb,int64_t sc){
  ++updates;__real_tvm_gemmini_tile_update_i8_i32(a,b,c,sa,sb,sc);}
static int32_t out[OUTPUT_ELEMENTS+16] __attribute__((aligned(64)));
static uint8_t workspace[WORKSPACE_BYTES+64] __attribute__((aligned(64)));
int main(void){
  const void*inputs[]={input};void*outputs[]={out};
  if(model_run(0,outputs,workspace,WORKSPACE_BYTES)!=-1||updates)return 10;
  inputs[0]=input+1;if(model_run(inputs,outputs,workspace,WORKSPACE_BYTES)!=-1||updates)return 11;
  inputs[0]=input;
#if PRACTICE
  uint64_t start=cycles();
  if(tvm_gemmini_matmul_i8_i32(input,weight,out,M,N,K,K,N,N))return 12;
  uint64_t elapsed=cycles()-start;
  for(unsigned i=0;i<OUTPUT_ELEMENTS;++i)if(out[i]!=raw_expected[i]){put("PRACTICE_FAILURE\n");return 13;}
  metric("PRACTICE_CYCLES=",elapsed);put("GEMMINI_PRACTICE_PASS\n");
#endif
  for(unsigned repeat=0;repeat<2;++repeat){
    for(unsigned i=0;i<OUTPUT_ELEMENTS+16;++i)out[i]=0x12345678;
    for(unsigned i=0;i<WORKSPACE_BYTES+64;++i)workspace[i]=0x6d;
    updates=0;uint64_t start=cycles();
    if(model_run(inputs,outputs,workspace,WORKSPACE_BYTES))return 20;
    uint64_t elapsed=cycles()-start;
    for(unsigned i=0;i<OUTPUT_ELEMENTS;++i)if(out[i]!=expected[i]+BAD_ORACLE){put("BYOC_NUMERICAL_FAILURE\n");return 22;}
    if(updates!=EXPECTED_UPDATES)return 23;
    for(unsigned i=OUTPUT_ELEMENTS;i<OUTPUT_ELEMENTS+16;++i)if(out[i]!=0x12345678)return 24;
    for(unsigned i=WORKSPACE_BYTES;i<WORKSPACE_BYTES+64;++i)if(workspace[i]!=0x6d)return 25;
    metric("BYOC_CYCLES=",elapsed);
  }
  /* Volatile loads must observe actual memory after DMA. The reference is a
   * host-computed scalar digest, so constant pooling cannot alias it to input. */
  if(BAD_INPUT)input[0]^=1;
  const volatile uint8_t*observed=(const volatile uint8_t*)input;
  uint64_t digest=UINT64_C(1469598103934665603);
  for(unsigned i=0;i<INPUT_ELEMENTS;++i)digest=(digest^observed[i])*UINT64_C(1099511628211);
  if(digest!=INPUT_EXPECTED_HASH){put("BYOC_INPUT_PRESERVATION_FAILURE\n");return 26;}
  extern uint64_t _stack_bottom[];
  if(_stack_bottom[0]!=UINT64_C(0x55aa55aa55aa55aa))return 27;
  put("TVM_BYOC_PASS repeats=2 exact=1 guards=1 input_preserved=1 counts=1\n");return 0;
}
'''


def prepare(args):
    budget()
    sys.path.insert(0, str(args.tvm_source / "apps/gemmini"))
    import numpy as np
    import tvm
    from tvm import relax, tir
    from tvm.relax.backend.contrib import gemmini_byoc as backend
    from install_backend import install
    from baremetal import export_graph
    from verify_baremetal_cpu import STARTUP, LINKER
    from verify_matmul import audit_instructions
    install(args.tvm_source, check=True)

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "preparing", "case": args.case, "order": args.order, "commands": [],
              "invocation": [sys.executable, *sys.argv],
              "source": [identity(Path(__file__)), identity(Path(backend.__file__)), identity(Path(__file__).with_name("byoc_primitives.c"))],
              "tvm_commit": subprocess.check_output(["git", "-C", args.tvm_source, "rev-parse", "HEAD"], text=True).strip(),
              "compiler": identity(args.riscv_gcc), "headers": [identity(args.headers / "include" / name) for name in ("gemmini.h", "gemmini_params.h", "gemmini_counter.h")],
              "adapter": identity(args.adapter), "performance_qualified": False}
    if report["headers"][1]["sha256"] != "3758ae967af3a179497660970201093a7fb624be00173990ce33d3f5c38da924":
        raise ValueError("Requires the documented pinned integer hardware parameter header")
    from tvm._ffi.base import _LIB
    report["tvm_library"] = identity(_LIB._name)

    def command(argv, label):
        argv = list(map(str, argv))
        result = subprocess.run(argv, capture_output=True, timeout=20)
        (out / (label + ".log")).write_bytes(result.stdout + result.stderr)
        report["commands"].append({"label": label, "argv": argv, "returncode": result.returncode})
        if result.returncode:
            raise RuntimeError(label + " failed: " + result.stderr.decode(errors="replace"))
        return result.stdout.decode()

    try:
        mod, a, b, raw, expected = fixture(args.case)
        (out / "input.relax.py").write_text(mod.script(show_meta=True))
        decomposed = backend._DecomposeBatch(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                decomposed.builder_.update_func(gv, decomposed.visit_expr(func))
        from tvm.relax.backend.contrib.gemmini import DecomposeGemminiLinear
        from tvm.relax.backend.contrib.gemmini_conv import DecomposeGemminiConv2D
        semantic = DecomposeGemminiLinear()(DecomposeGemminiConv2D()(decomposed.builder_.get()))
        partitioned = backend.partition_for_gemmini(semantic)
        (out / "partitioned.relax.py").write_text(partitioned.script(show_meta=True))
        assert "Codegen" in partitioned.script() and "Composite" in partitioned.script()
        records = []
        lowered = backend.prepare_gemmini_byoc(mod, order=args.order, records=records)
        manifest = export_graph(lowered, out / "graph", memory_limit_bytes=1 << 20)
        updates = 0
        trace_index = 0
        report["schedules"] = []
        for symbol, scheduled in dict(records).items():
            m, n, k = scheduled.shape
            invocations = sum(call["symbol"] == symbol for call in manifest["calls"])
            updates += invocations * ((m+15)//16) * ((n+7)//8) * ((k+15)//16)
            report["schedules"].append({"symbol": symbol, "shape_mnk": scheduled.shape, "invocations": invocations})
            replay = tir.Schedule(scheduled.semantic_mod, debug_mask="all")
            scheduled.trace.apply_to_schedule(replay, remove_postproc=False)
            tvm.ir.assert_structural_equal(replay.mod, scheduled.scheduled_mod)
            (out / f"semantic-{trace_index}.tir.py").write_text(scheduled.semantic_mod.script())
            (out / f"trace-{trace_index}.json").write_text(json.dumps(scheduled.trace.as_json()))
            trace_index += 1
        if not updates:
            raise AssertionError("No device calls")
        report["updates_per_inference"] = updates
        report["graph"] = manifest
        llvm = (out / "graph/operators.ll").read_text()
        assert "@tvm_gemmini_tile_update_i8_i32(" in llvm
        report["generated_object"] = identity(out / "graph/operators.o")

        def array(name, value, writable=False):
            dtype = "int8_t" if value.dtype == np.int8 else "int32_t"
            qualifier = "" if writable else "const "
            return f"static {qualifier}{dtype} {name}[] __attribute__((aligned(64))) = {{" + ",".join(map(str, value.ravel().tolist())) + "};\n"

        practice = int(args.case in ("tiny", "tails"))
        header = array("input", a, writable=True) + array("weight", b) + array("raw_expected", raw) + array("expected", expected)
        digest = 1469598103934665603
        for byte in a.tobytes():
            digest = ((digest ^ byte) * 1099511628211) & ((1 << 64) - 1)
        header += f"#define INPUT_EXPECTED_HASH UINT64_C({digest})\n"
        macros = {"INPUT_ELEMENTS": a.size, "OUTPUT_ELEMENTS": expected.size, "WORKSPACE_BYTES": manifest["workspace_bytes"],
                  "EXPECTED_UPDATES": updates, "PRACTICE": practice, "M": a.shape[-2], "K": a.shape[-1], "N": b.shape[-1]}
        header += "\n".join(f"#define {key} {value}" for key, value in macros.items())
        (out / "fixture.h").write_text(header + "\n")
        (out / "runner.c").write_text(RUNNER)
        (out / "start.S").write_text(STARTUP)
        (out / "link.ld").write_text(LINKER)
        flags = ["-std=gnu11", "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-msmall-data-limit=0", "-O2", "-ffreestanding", "-fno-builtin", "-fno-tree-loop-distribute-patterns", "-fno-stack-protector", "-DBAREMETAL=1", "-I" + str(args.headers), "-I" + str(out), "-ffunction-sections", "-fdata-sections"]
        primitive = Path(__file__).with_name("byoc_primitives.c")
        command([args.riscv_gcc, *flags, "-c", primitive, "-o", out / "primitive.o"], "primitive")
        report["elf"] = {}
        for label, wrong, corrupt in (("valid", 0, 0), ("bad-oracle", 1, 0), ("bad-input", 0, 1)):
            elf = out / (label + ".elf")
            command([args.riscv_gcc, *flags, "-nostdlib", "-nostartfiles", "-static", "-Wl,--gc-sections,--no-relax", "-Wl,-T," + str(out / "link.ld"), "-Wl,--wrap=tvm_gemmini_tile_update_i8_i32", "-DBAD_ORACLE=" + str(wrong), "-DBAD_INPUT=" + str(corrupt), out / "start.S", out / "runner.c", out / "graph/model.c", out / "graph/constants.S", out / "graph/operators.o", out / "primitive.o", args.adapter, "-o", elf], "link-" + label)
            # Source-derived forbidden instruction function codes, including
            # all loop configuration commands, from the pinned vendor header.
            import re
            definitions = dict((name, int(number)) for name, number in re.findall(r"#define\s+(k_LOOP\w*)\s+(\d+)", (args.headers / "include/gemmini.h").read_text()))
            if not definitions:
                raise ValueError("No source-derived loop definitions")
            params = (args.headers / "include/gemmini_params.h").read_text()
            custom = int(re.search(r"#define\s+XCUSTOM_ACC\s+(\d+)", params)[1])
            opcode = (0x0b, 0x2b, 0x5b, 0x7b)[custom]
            if label == "valid":
                # Prove that the audit recognizes a forbidden encoding for
                # this header's custom opcode, rather than accepting vacuously.
                probe = out / "forbidden.S"
                probe.write_text(f".text\n.insn r CUSTOM_{custom}, 3, {next(iter(definitions.values()))}, x0, x0, x0\n")
                command([args.riscv_gcc, "-march=rv64gc", "-mabi=lp64d", "-c", probe, "-o", out / "forbidden.o"], "audit-control")
                if not audit_instructions((out / "forbidden.o").read_bytes(), opcode, set(definitions.values()))["prohibited_hits"]:
                    raise AssertionError("Instruction audit failed its injected-loop control")
            audit = audit_instructions(elf.read_bytes(), opcode, set(definitions.values()))
            if audit["prohibited_hits"]:
                raise AssertionError("Hardware loop instruction in linked ELF")
            nm = str(args.riscv_gcc).removesuffix("gcc") + "nm"
            if command([nm, "-u", elf], "undefined-" + label).strip():
                raise AssertionError("Unresolved guest symbols")
            report["elf"][label] = {**identity(elf), "instruction_audit": audit, "custom_opcode": opcode, "forbidden_functs": definitions}
        report["status"] = "prepared"
    finally:
        report["resources"] = budget()
        (out / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "case": args.case, "updates": updates, "output_dir": str(out)}))


def run(args):
    before = budget()
    label = "bad-oracle" if args.bad_oracle else "bad-input" if args.bad_input else "valid"
    elf = args.bundle / (label + ".elf")
    build = json.loads((args.bundle / "build.json").read_text())
    expected_elf = build["elf"][label]
    if identity(elf)["sha256"] != expected_elf["sha256"]:
        raise ValueError("ELF differs from its prepared source binding")
    if args.spike:
        argv = [str(args.spike), "--extlib=" + str(args.plugin), "--extension=gemmini", "--isa=rv64gc", "-p1", "-m0x80000000:0x10000000", str(elf)]
    else:
        argv = [str(args.simulator), "+permissive", "+max-cycles=" + str(args.cycles), "+loadmem=" + str(elf), "+permissive-off", str(elf)]
    start = time.monotonic()
    try:
        child = subprocess.run(argv, capture_output=True, timeout=args.seconds)
        code, output = child.returncode, child.stdout + child.stderr
    except subprocess.TimeoutExpired as error:
        code, output = None, (error.stdout or b"") + (error.stderr or b"")
    text = output.decode(errors="replace")
    if args.bad_oracle or args.bad_input:
        import re
        guest_code = 22 if args.bad_oracle else 26
        marker = "BYOC_NUMERICAL_FAILURE" if args.bad_oracle else "BYOC_INPUT_PRESERVATION_FAILURE"
        assertion = rf"Assertion failed: \*\*\* FAILED \*\*\* \(exit code =\s+{guest_code}\)"
        passed = marker in text and (code == guest_code if args.spike else code == 255 and bool(re.search(assertion, text)))
    else:
        passed = code == 0 and "TVM_BYOC_PASS" in text and (bool(args.spike) or "$finish" in text)
    report = {"status": "passed" if passed else "failed", "argv": argv, "returncode": code, "elapsed_seconds": time.monotonic() - start,
              "elf": identity(elf), "build_receipt": identity(args.bundle / "build.json"), "engine": identity(args.spike or args.simulator), "plugin": identity(args.plugin) if args.spike else None,
              "negative_control": args.bad_oracle or args.bad_input, "control": label, "resource_before": before, "resource_after": budget(),
              "performance_qualified": False, "cycle_scope": "guest mcycle around graph only; Spike is functional, RTL uses local magic memory"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".log").write_bytes(output)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if not passed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    for name in ("tvm-source", "riscv-gcc", "headers", "adapter", "output-dir"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--case", choices=("tiny", "tails", "bmm", "conv"), default="tails")
    prep.add_argument("--order", choices=("ijk", "jik", "sample"), default="ijk")
    execute = sub.add_parser("run")
    execute.add_argument("--bundle", type=Path, required=True)
    engine = execute.add_mutually_exclusive_group(required=True)
    engine.add_argument("--spike", type=Path)
    engine.add_argument("--simulator", type=Path)
    execute.add_argument("--plugin", type=Path)
    execute.add_argument("--output", type=Path, required=True)
    control = execute.add_mutually_exclusive_group()
    control.add_argument("--bad-oracle", action="store_true")
    control.add_argument("--bad-input", action="store_true")
    execute.add_argument("--seconds", type=int, default=45, choices=range(1, 56))
    execute.add_argument("--cycles", type=int, default=200000, choices=range(1, 250001))
    args = parser.parse_args()
    if args.action == "run" and args.spike and not args.plugin:
        parser.error("Spike requires --plugin")
    (prepare if args.action == "prepare" else run)(args)


if __name__ == "__main__":
    main()
