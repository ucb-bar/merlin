#!/usr/bin/env python3
"""Prepare retained TVM block objects for bounded, observable RTL verification.

The graph/operator objects, constants and fixture bytes are retained unchanged.
Only the trusted one-fixture verifier/startup and graph-call instrumentation are
adapted. This is not the original four-call preservation/poisoning stress scope.
"""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def resource_state():
    line = next(line for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    group = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
    return {name: (group / name).read_text().strip() for name in
            ("memory.max", "memory.swap.max", "memory.peak", "memory.events", "cpu.max")}


def require_budget(maximum=2 << 30):
    state = resource_state()
    quota, period = state["cpu.max"].split()
    if state["memory.max"] == "max" or int(state["memory.max"]) > maximum or state["memory.swap.max"] != "0":
        raise ValueError("Requires an enforced memory cap <=2 GiB and no swap")
    if quota == "max" or int(quota) > int(period):
        raise ValueError("Requires an aggregate CPU quota <=1 CPU")
    return state


def instrument_graph(source, calls, device_symbols):
    """Match the actual exported call sequence; never infer placement by spelling."""
    lines, index = [], 0
    for line in source.splitlines():
        if line.startswith("  if (") and line.endswith(") return -3;"):
            if index >= len(calls) or not line.startswith("  if (" + calls[index]["symbol"] + "("):
                raise ValueError("Exported source and manifest call order disagree")
            lines.extend((f"  rtl_trace({index}, 0);", line, f"  rtl_trace({index}, 1);"))
            index += 1
        else:
            lines.append(line)
    if index != len(calls):
        raise ValueError("Not every exported call was instrumented")
    return ("extern void rtl_trace(unsigned, unsigned);\n" + "\n".join(lines) + "\n",
            [int(call["symbol"] in device_symbols) for call in calls])


def stack_scanner_source():
    """Same first-changed-byte watermark, with aligned words for untouched stack."""
    return r'''
static uint64_t rtl_stack_used(const uint8_t*bottom,const uint8_t*top){
  const uint64_t pattern=UINT64_C(0x55aa55aa55aa55aa);const uint8_t*first=bottom;
  if(!((uintptr_t)first%8))while(top-first>=8){uint64_t word;
    __builtin_memcpy(&word,__builtin_assume_aligned(first,8),8);
    if(word!=pattern)break;first+=8;
  }
  while(first<top&&*first==(uint8_t)(pattern>>(((uintptr_t)first-(uintptr_t)bottom)%8*8)))++first;
  return (uint64_t)(top-first);
}
'''


def make_runner(original, manifest, placement):
    """One original fixture, unchanged arithmetic/oracles, inexpensive startup."""
    prefix = original[:original.index("int main(void)")]
    helpers = r'''
static unsigned rtl_completed,rtl_cpu,rtl_gemmini;
static const uint8_t rtl_placement[] = {PLACEMENT};
static char*rtl_text(char*p,const char*s){while(*s)*p++=*s++;return p;}
static char*rtl_decimal(char*p,uint64_t n){char digits[32];unsigned i=0;
  do{digits[i++]=(char)('0'+n%10);n/=10;}while(n);while(i)*p++=digits[--i];return p;}
void rtl_trace(unsigned index,unsigned finished){
  if(index>=sizeof(rtl_placement)||index!=rtl_completed)exit(33);
  if(finished){++rtl_completed;if(rtl_placement[index])++rtl_gemmini;else ++rtl_cpu;}
  char line[96];char*p=rtl_text(line,finished?"RTL_OP_END index=":"RTL_OP_BEGIN index=");
  p=rtl_decimal(p,index);p=rtl_text(p,rtl_placement[index]?" destination=Gemmini\n":" destination=CPU\n");
  *p=0;puts_htif(line);
}
static void rtl_model_summary(void){char line[96];char*p=rtl_text(line,"RTL_MODEL_END cpu=");
  p=rtl_decimal(p,rtl_cpu);p=rtl_text(p," gemmini=");p=rtl_decimal(p,rtl_gemmini);*p++='\n';*p=0;puts_htif(line);
}
static void rtl_count_summary(void){char line[256];char*p=rtl_text(line,"RTL_COUNTS");
  const char*names[]={" loads_a="," loads_b="," computes="," stores="," admissions="};
  uint64_t values[]={loads_a,loads_b,computes,stores,admissions};
  for(unsigned i=0;i<5;++i){p=rtl_text(p,names[i]);p=rtl_decimal(p,values[i]);}*p++='\n';*p=0;puts_htif(line);
}
static void write_htif(const void*data,uint64_t bytes){
  static volatile uint64_t request[4] __attribute__((aligned(8)));
  request[0]=64;request[1]=1;request[2]=(uintptr_t)data;request[3]=bytes;
  asm volatile("fence rw,rw" ::: "memory");tohost=(uintptr_t)request;
  while(!fromhost){}fromhost=0;asm volatile("fence rw,rw" ::: "memory");
}
static int equal_i32(const void*a,const void*b,uint64_t bytes){
  if(bytes%4)return 0;
  const uint32_t*x=a,*y=b;for(uint64_t i=0;i<bytes/4;++i)if(x[i]!=y[i])return 0;
  return 1;
}
'''.replace("PLACEMENT", ",".join(map(str, placement)))
    lines = [prefix, helpers, stack_scanner_source(), "int main(void){", '  puts_htif("RTL_BLOCK_START fixture=0 repeats=1 preload_required=1\\n");',
             "  if(workspace.value[0]||workspace.value[WORKSPACE_BYTES-1])return 80;",
             "  fill(workspace.head,64,0x6d);fill(workspace.tail,64,0x6d);",
             "  const void*inputs[INPUT_COUNT];void*outputs[OUTPUT_COUNT];"]
    for i in range(len(manifest["inputs"])):
        lines.append(f"  inputs[{i}]=fixture_input_{i};")
    for i, tensor in enumerate(manifest["outputs"]):
        lines.extend((f"  outputs[{i}]=output_{i}[0].value;",
                      f"  fill(output_{i}[0].head,64,0x6d);fill(output_{i}[0].tail,64,0x6d);",
                      f"  fill(output_{i}[0].value+{tensor['bytes']},sizeof(output_{i}[0].value)-{tensor['bytes']},0x6d);"))
    lines.extend(("  if(model_run(0,outputs,workspace.value,WORKSPACE_BYTES)!=-1||computes)return 10;",
                  "  if(WORKSPACE_BYTES&&model_run(inputs,outputs,workspace.value,0)!=-1)return 10;",
                  '  puts_htif("RTL_INPUT_REJECTION_PASS\\n");',
                  '  puts_htif("RTL_MODEL_BEGIN\\n");',
                  "  if(model_run(inputs,outputs,workspace.value,WORKSPACE_BYTES))return 21;",
                  "  rtl_model_summary();"))
    for i, tensor in enumerate(manifest["outputs"]):
        lines.extend((f'  puts_htif("RTL_OUTPUT index={i} bytes={tensor["bytes"]}\\n");',
                      f"  write_htif(output_{i}[0].value,{tensor['bytes']});",
                      '  puts_htif("\\nRTL_OUTPUT_END\\n");'))
        if tensor["dtype"] == "float32":
            check = f"equal_f32(output_{i}[0].value,fixture_output_{i},{tensor['bytes']},BAD_ORACLE&&{i}==0)"
        elif tensor["dtype"] == "int32":
            check = f"equal_i32(output_{i}[0].value,fixture_output_{i},{tensor['bytes']})"
        else:
            raise ValueError("Unexpected output dtype")
        lines.extend((f'  if(!{check}){{puts_htif("RTL_NUMERICAL_FAILURE\\n");return 22;}}',
                      f"  if(!guard(output_{i}[0].head,output_{i}[0].tail))return 27;",
                      f"  if(!padding(output_{i}[0].value,{tensor['bytes']},sizeof(output_{i}[0].value),0x6d))return 23;"))
    lines.extend(("  if(!guard(workspace.head,workspace.tail))return 27;",
                  "  if(loads_a!=EXPECTED_LOADS_A||loads_b!=EXPECTED_LOADS_B||computes!=EXPECTED_COMPUTES||stores!=EXPECTED_STORES||admissions!=EXPECTED_ADMISSIONS)return 30;",
                  f"  if(rtl_completed!={len(placement)}||rtl_gemmini!={sum(placement)}||rtl_cpu!={len(placement)-sum(placement)})return 34;",
                  "  rtl_count_summary();",
                  "  extern uint8_t _stack_bottom[],_stack_top[];uint64_t used=rtl_stack_used(_stack_bottom,_stack_top);",
                  "  if(!used||used>=STACK_BYTES)return 31;",
                  '  puts_htif("STACK_HIGH_WATER_BYTES=");number(used);puts_htif("\\n");',
                  '  puts_htif("RTL_BLOCK_PASS calls=1 float_gate=1 guards=1 primitive_counts=1 original_stress_scope=0\\n");return 0;', "}"))
    return "\n".join(lines) + "\n"


def prepare(args):
    require_budget()
    src = args.input_dir.resolve(strict=True)
    out = args.output_dir.resolve()
    report = json.loads((src / "report.json").read_text())
    if report["status"] != "passed":
        raise ValueError("Requires a passing retained block verification")
    hardware = json.loads(args.hardware_receipt.read_text())
    if identity(hardware["simulator"]["path"])["sha256"] != hardware["simulator"]["sha256"]:
        raise ValueError("Selected simulator changed")
    manifest = json.loads((src / "graph/graph.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    symbols = {row["symbol"] for row in report["dispatch"]["Gemmini"]}
    model, placement = instrument_graph((src / "graph/model.c").read_text(), manifest["calls"], symbols)
    (out / "model.c").write_text(model)
    (out / "runner.c").write_text(make_runner((src / "guest/runner.c").read_text(), manifest, placement))
    start = (src / "guest/start.S").read_text()
    begin, end = start.index("  la t0, _bss_start"), start.index("2:\n") + 3
    (out / "start.S").write_text(start[:begin] + "  /* Requires +loadmem PT_LOAD zero initialization. */\n" + start[end:])
    for name in ("fixture.h", "link.ld", "errno.c"):
        (out / name).write_bytes((src / "guest" / name).read_bytes())
    retained = [src / "graph" / name for name in ("operators.o", "constants.S", "constants.bin", "graph.json")]
    retained.extend(src / "guest" / name for name in ("fixture.S", "fixture.bin"))
    retained = [path for path in retained if path.exists()]
    for assembly in (src / "graph/constants.S", src / "guest/fixture.S"):
        for line in assembly.read_text().splitlines():
            if line.startswith(".incbin "):
                retained.append(Path(json.loads(line.removeprefix(".incbin "))))
    retained = list(dict.fromkeys(retained))
    for binding in (report["runtime"]["adapter"], *report["runtime"]["math_libraries"]):
        if identity(binding["path"])["sha256"] != binding["sha256"]:
            raise ValueError("Retained runtime input changed")
        retained.append(Path(binding["path"]))
    receipt = {"status": "preparing", "source_report": identity(src / "report.json"),
               "hardware_receipt": identity(args.hardware_receipt), "simulator": hardware["simulator"],
               "source_driver": identity(__file__), "retained": [identity(path) for path in retained],
               "workload": report["workload"], "dispatch": report["dispatch"],
               "placement": placement, "original_samples": report["samples"], "samples": 1, "repeats": 1,
               "float_gate": report["float_gate"], "commands": [], "guest_adaptations": [
                   "Original first fixture, graph operations/operator object/weights unchanged; trace hooks surround existing graph calls",
                   "Input fixtures addressed directly; no guest full-weight hash, input copy, workspace poisoning or alternate-output retention stress",
                   "Preload ELF loader initializes all PT_LOAD BSS; redundant guest BSS clear omitted",
                   "Original numerical tolerance, exact INT32 outputs, output/workspace guards, primitive counters, input rejection and stack canary retained",
                   "Raw output bytes emitted through HTIF; host can compute actual-output cosine without guest hexadecimal conversion"],
               "TVM_core_changed": False, "RTL_changed": False}
    receipt["guest_adaptations"].extend([
        "Each trace/count summary uses one HTIF write instead of fragmented number/text writes; operation order and counters unchanged",
        "Stack watermark skips intact aligned words then finds the same first changed byte; exact byte-level result, bounds and rejection unchanged"])
    template = next(row["argv"] for row in report["commands"] if row["label"] == "link-valid")
    toolroot = next(Path(arg[2:]) for arg in template if arg.startswith("-I") and (Path(arg[2:]) / "verify_matmul.py").is_file())
    sys.path.insert(0, str(toolroot))
    from verify_matmul import audit_instructions
    from verify_baremetal_cpu import audit_cpu_runtime
    adapter_receipt = json.loads(Path(report["runtime"]["adapter_receipt"]["path"]).read_text())
    policy = adapter_receipt["instruction_policy"]
    forbidden = {value for name, value in policy["derived_definitions"].items() if name.startswith("k_LOOP")}

    def command(argv, name):
        result = subprocess.run(list(map(str, argv)), capture_output=True, timeout=15, check=True)
        (out / (name + ".log")).write_bytes(result.stdout + result.stderr)
        return result.stdout.decode()

    for label, wrong in (("valid", 0), ("bad-oracle", 1)):
        argv = []
        for arg in template:
            if arg.startswith("-DBAD_ORACLE="):
                arg = "-DBAD_ORACLE=" + str(wrong)
            elif arg.startswith("-Wl,-Map,"):
                arg = "-Wl,-Map," + str(out / (label + ".map"))
            elif arg.startswith("-Wl,-T,"):
                arg = "-Wl,-T," + str(out / "link.ld")
            elif arg == "-I" + str(src / "guest"):
                arg = "-I" + str(out)
            elif arg in [str(src / "guest" / name) for name in ("start.S", "runner.c", "errno.c")]:
                arg = str(out / Path(arg).name)
            elif arg == str(src / "graph/model.c"):
                arg = str(out / "model.c")
            elif arg == str(src / "guest/valid.elf"):
                arg = str(out / (label + ".elf"))
            argv.append(arg)
        process = subprocess.run(argv, capture_output=True, timeout=30)
        (out / ("link-" + label + ".log")).write_bytes(process.stdout + process.stderr)
        receipt["commands"].append({"argv": argv, "exit_code": process.returncode})
        process.check_returncode()
        elf = out / (label + ".elf")
        instructions = audit_instructions(elf.read_bytes(), policy["custom_opcode"], forbidden)
        if instructions["prohibited_hits"]:
            raise ValueError("Linked ELF contains prohibited hardware-loop instructions")
        gcc = Path(argv[0])
        tools = {name: gcc.with_name("riscv64-unknown-elf-" + name) for name in ("nm", "readelf")}
        runtime = audit_cpu_runtime(elf, out / (label + ".map"), {"archives": report["runtime"]["math_libraries"]}, tools, command)
        runtime["initialization"] = "Requires SimDRAM +loadmem zero initialization of complete PT_LOAD BSS; single-hart startup initializes FPU and stack canary; no TLS/constructors/heap"
        if command([tools["nm"], "-u", elf], label + "-undefined").strip():
            raise ValueError("Unresolved guest symbols")
        symbols = command([tools["nm"], "--defined-only", elf], label + "-callback-symbols")
        if any(line.split() and (line.split()[-1].startswith("__TVMBackend") or line.split()[-1] in ("__TVMFuncCall", "__TVMFuncGetEnv")) for line in symbols.splitlines()):
            raise ValueError("Uninitialized TVM runtime callback")
        receipt.setdefault("audits", {})[label] = {"instructions": instructions, "runtime": runtime}
    receipt.update(status="prepared", ELFs={name: identity(out / (name + ".elf")) for name in ("valid", "bad-oracle")}, resources=resource_state())
    (out / "bundle.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"bundle": str(out), "calls": len(placement), "Gemmini": sum(placement), "CPU": len(placement)-sum(placement)}))


def read_output_records(raw, *, allow_incomplete=False):
    """Parse length-delimited binary payloads without treating payload as text."""
    records, text, position = [], [], 0
    marker = b"RTL_OUTPUT index="
    while True:
        begin = raw.find(marker, position)
        if begin < 0:
            text.append(raw[position:])
            break
        text.append(raw[position:begin])
        end = raw.find(b"\n", begin)
        if end < 0:
            break
        fields = dict(part.split(b"=", 1) for part in raw[begin:end].split()[1:])
        size = int(fields[b"bytes"])
        if size < 0 or size > 16 << 20:
            raise ValueError("Unexpected output payload extent")
        tail = end + 1 + size
        if tail > len(raw):
            if allow_incomplete:
                text.append(raw[begin:])
            break
        if raw[tail:tail+16] != b"\nRTL_OUTPUT_END\n":
            if allow_incomplete:
                text.append(raw[begin:])
                break
            raise ValueError("Malformed output payload delimiter")
        records.append({"index": int(fields[b"index"]), "payload": raw[end+1:tail]})
        position = tail + 16
    return records, b"".join(text).decode("utf-8", errors="replace")


def classify(text, returncode, timed_out, negative=False):
    if timed_out:
        return "wall_timeout"
    if negative and returncode == 255 and "RTL_NUMERICAL_FAILURE" in text:
        if any("Assertion failed:" in line and "exit code =" in line and
               line.split("exit code =", 1)[1].split(")", 1)[0].strip() == "22" for line in text.splitlines()):
            return "negative_control_passed"
    if not negative and returncode == 0 and "RTL_BLOCK_PASS calls=1" in text and "Verilog $finish" in text:
        return "passed"
    if "*** FAILED ***" in text and "timeout" in text.lower():
        return "cycle_limit"
    return "failed_or_incomplete"


def complete_output_indices(records, expected):
    """Success needs every output exactly once, including an empty/truncated log."""
    return len(records) == expected and sorted(row["index"] for row in records) == list(range(expected))


def run(args):
    require_budget()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_STACK, (64 << 20, 64 << 20))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 << 20, 32 << 20))
    out = args.bundle.resolve(strict=True)
    bundle = json.loads((out / "bundle.json").read_text())
    label = "bad-oracle" if args.negative else "valid"
    elf = bundle["ELFs"][label]
    for record in (bundle["simulator"], elf, *bundle["retained"]):
        if identity(record["path"])["sha256"] != record["sha256"]:
            raise ValueError("Execution input changed: " + record["path"])
    result_path = out / (args.name + "-result.json")
    if result_path.exists():
        raise ValueError("Refusing to overwrite an execution receipt")
    argv = [bundle["simulator"]["path"], "+permissive", "+max-cycles=" + str(args.cycles),
            "+loadmem=" + elf["path"], "+permissive-off", elf["path"]]
    start = time.monotonic()
    timed_out = False
    with (out / (args.name + ".log")).open("wb") as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = child.wait(timeout=args.seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            child.terminate()
            try:
                code = child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                code = child.wait()
    raw = (out / (args.name + ".log")).read_bytes()
    parse_error = None
    try:
        records, text = read_output_records(raw, allow_incomplete=True)
    except (ValueError, KeyError) as error:
        records, text = [], raw.decode("utf-8", errors="replace")
        parse_error = str(error)
    (out / (args.name + "-text.log")).write_text(text)
    completed = [line for line in text.splitlines() if line.startswith("RTL_OP_END ")]
    begun = [line for line in text.splitlines() if line.startswith("RTL_OP_BEGIN ")]
    result = {"status": classify(text, code, timed_out, args.negative), "argv": argv,
              "simulator": bundle["simulator"], "ELF": elf, "returncode": code, "timed_out": timed_out,
              "wall_seconds": time.monotonic()-start, "cycle_limit": args.cycles,
              "CPU_calls_completed": sum("destination=CPU" in line for line in completed),
              "Gemmini_calls_completed": sum("destination=Gemmini" in line for line in completed),
              "last_call_begun": begun[-1] if begun else None, "last_call_completed": completed[-1] if completed else None,
              "output_records": len(records), "resources": resource_state()}
    if parse_error:
        result["output_parse_error"] = parse_error
        if result["status"] == "passed":
            result["status"] = "host_output_verification_failed"
    if records:
        import numpy as np
        from verify_blocks import metrics
        src = Path(bundle["source_report"]["path"]).parent
        source = json.loads((src / "report.json").read_text())
        with np.load(src / "fixture.npz", allow_pickle=False) as fixtures, np.load(src / "original-fp32.npz", allow_pickle=False) as original:
            comparisons, actual = [], {}
            for record in records:
                i = record["index"]
                reference = fixtures[f"output_{i}"][0]
                value = np.frombuffer(record["payload"], dtype=reference.dtype).reshape(reference.shape)
                actual[f"output_{i}"] = value
                if reference.dtype == np.float32:
                    comparisons.append({"output": i, "vs_quantized_torch": metrics(value, reference, np),
                                        "vs_original_fp32": metrics(value, original[f"output_{i}"][0], np),
                                        "strict_gate_passed": bool(np.allclose(value, reference, **source["float_gate"]))})
                else:
                    comparisons.append({"output": i, "integer_exact": bool(np.array_equal(value, reference))})
            result["comparisons"] = comparisons
            np.savez(out / (args.name + "-actual-outputs.npz"), **actual)
            if result["status"] == "passed" and not all(row.get("strict_gate_passed", row.get("integer_exact", False)) for row in comparisons):
                result.update(status="host_output_verification_failed", failure="Host numerical comparison disagrees with guest marker")
    if result["status"] == "passed" and not complete_output_indices(records, len(bundle["workload"]["float_output_names"])+len(bundle["workload"]["integer_outputs"])):
        result.update(status="host_output_verification_failed", failure="Missing, duplicate or unexpected output records")
    result["resources"] = resource_state()
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "returncode", "wall_seconds", "CPU_calls_completed", "Gemmini_calls_completed", "last_call_begun")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--input-dir", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--hardware-receipt", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--bundle", type=Path, required=True)
    execute.add_argument("--name", required=True)
    execute.add_argument("--seconds", type=int, default=175)
    execute.add_argument("--cycles", type=int, default=850000)
    execute.add_argument("--negative", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
