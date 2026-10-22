#!/usr/bin/env python3
"""Capture isolated W8A8 workloads, compile with TVM, and verify actual Spike outputs.

Invoke inside an externally enforced memory/CPU/time budget. Quantization quality
is reported separately from compiler correctness. No source-library admission
guards or TVM compiler sources are changed by this driver.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from block_workloads import make_workload, preflight


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def metrics(actual, expected, np):
    if actual.shape != expected.shape:
        raise ValueError("Metrics require equal tensor shapes")
    a, b = actual.astype("float64").ravel(), expected.astype("float64").ravel()
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Metrics require equal shapes and finite outputs")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cosine = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)) if na and nb else (1.0 if np.array_equal(a, b) else 0.0)
    return {"cosine": cosine, "max_absolute_error": float(np.max(np.abs(a - b))),
            "relative_l2": float(np.linalg.norm(a - b) / max(nb, 1e-30)),
            "exact_equal": bool(np.array_equal(actual, expected)), "elements": a.size}


def resource_state():
    line = next(line for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    group = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
    return {name: (group / name).read_text().strip() for name in
            ("memory.max", "memory.swap.max", "memory.peak", "memory.events", "cpu.max")}


FLOAT_CHECK = r'''
static int equal_f32(const uint8_t*a,const uint8_t*b,uint64_t n,int wrong){
  if(n%4)return 0;
  for(uint64_t i=0;i<n/4;++i){union{uint32_t u;float f;}x,y;
    x.u=((const uint32_t*)a)[i];y.u=((const uint32_t*)b)[i];
    if((x.u&0x7f800000)==0x7f800000||(y.u&0x7f800000)==0x7f800000)return 0;
    if(wrong&&i==0)y.f+=(y.f<0?-y.f:y.f)+1.0f;
    float d=x.f-y.f;if(d<0)d=-d;float r=y.f;if(r<0)r=-r;
    if(d>1e-5f+1e-4f*r)return 0;
  }return 1;
}
static void dump_f32(unsigned call,unsigned index,const uint8_t*data,uint64_t n){
  const char*digits="0123456789abcdef";
  for(uint64_t off=0;off<n/4;off+=64){char text[513];uint64_t count=n/4-off;if(count>64)count=64;
    for(uint64_t j=0;j<count;++j){uint32_t v=((const uint32_t*)data)[off+j];
      for(unsigned k=0;k<8;++k)text[j*8+k]=digits[(v>>(28-k*4))&15];}
    text[count*8]=0;puts_htif("BLOCK_OUTPUT call=");number(call);puts_htif(" index=");number(index);
    puts_htif(" offset=");number(off);puts_htif(" hex=");puts_htif(text);puts_htif("\n");
  }
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("smolvla-action", "smolvla-denoise", "smolvla-block-action", "tinyllama-head-ffn"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tvm-source", type=Path, required=True)
    parser.add_argument("--tvm-build", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-receipt", type=Path, required=True)
    parser.add_argument("--simulator-receipt", type=Path, required=True)
    parser.add_argument("--spike", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--riscv-gcc", type=Path, required=True)
    parser.add_argument("--dtc", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--action-chunk-size", type=int, choices=(1, 50), default=50,
                        help="Isolated action only: explicit one-position supplementary diagnostic; original scope remains chunk50")
    parser.add_argument("--action-hidden-size", type=int, default=720,
                        help="Isolated action checkpoint-column prefix, 1..720; reduced values are compiler diagnostics")
    parser.add_argument("--tiny-hidden-size", type=int, default=2048,
                        help="TinyLlama hidden checkpoint prefix, 64..2048; original default preserved")
    parser.add_argument("--tiny-intermediate-size", type=int, default=5632,
                        help="TinyLlama FFN checkpoint prefix, 16..5632; original default preserved")
    parser.add_argument("--samples", type=int, choices=(1, 2), default=2)
    parser.add_argument("--repeats", type=int, choices=(1, 2), default=2)
    parser.add_argument("--mode", choices=("baseline", "optimized"), default="optimized")
    parser.add_argument("--macro-tiles", choices=("largest", "one"), default="largest",
                        help="Largest supported shape/capacity-bounded grouping (default), or explicit legacy 1x1 control")
    parser.add_argument("--macro-fill-percent", type=int, default=90,
                        help="Minimum occupied micro-tile slots per macro grid; independent of unavoidable micro-tile tails")
    parser.add_argument("--gemmini-params", type=Path,
                        help="Generated parameter header matching the adapter receipt; otherwise resolve its authenticated header")
    parser.add_argument("--resume-from", type=Path, help="Reuse saved captured ONNX and independently checked fixtures; do not load checkpoint slices again")
    parser.add_argument("--specialize-static-offsets", action="store_true", help="Opt in to public TIR specialization of symbolic offsets to the static exporter's zero-offset buffer ABI")
    parser.add_argument("--localize-cpu-workspace", action="store_true", help="Opt in to legal TensorIR private-producer row placement and compact scratch allocation")
    parser.add_argument("--single-hart-cpu", action="store_true", help="Serialize CPU parallel loops for the explicit one-hart guest without a TVM thread-pool runtime")
    parser.add_argument("--host-budget-bytes", type=int, default=2 << 30)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "scope": "isolated compiler-development workload", "commands": [],
              "full_models_qualified": False, "model_quality_qualified": False, "timing_qualified": False,
              "TVM_core_changed": False, "invocation": sys.argv,
              "preflight": preflight(args.workload, args.checkpoint, **geometry_options(args))}

    def save():
        report["resources"] = resource_state()
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def command(argv, label, expected=0):
        record = {"label": label, "argv": list(map(str, argv)), "expected_exit": expected}
        report["commands"].append(record)
        save()
        print(label, "started", flush=True)
        start = time.monotonic()
        with (out / (label + ".log")).open("w") as log:
            process = subprocess.run(record["argv"], stdout=log, stderr=subprocess.STDOUT,
                                     timeout=args.timeout_seconds, check=False)
        record.update(exit_code=process.returncode, wall_seconds=time.monotonic() - start)
        save()
        if process.returncode != expected:
            raise RuntimeError(label + " failed; inspect its saved log")
        return (out / (label + ".log")).read_text()

    try:
        limits = resource_state()
        if limits["memory.max"] == "max" or int(limits["memory.max"]) > args.host_budget_bytes or limits["memory.swap.max"] != "0":
            raise ValueError("Refusing work without the declared hard memory/no-swap cap")
        quota, period = limits["cpu.max"].split()
        if quota == "max" or int(quota) > int(period):
            raise ValueError("Refusing work without an aggregate one-CPU cap")
        if report["preflight"]["host_peak_estimate_bytes"] > min(args.host_budget_bytes, int(limits["memory.max"])):
            report.update(status="skipped_resource_estimate", reason="Estimated peak exceeds declared host budget")
            save()
            return
        os.environ.update(TVM_LIBRARY_PATH=str(args.tvm_build.resolve()), TVM_FFI="ctypes")
        os.environ["PATH"] = str(args.dtc.resolve(strict=True).parent) + ":" + str(args.riscv_gcc.resolve(strict=True).parent) + ":/usr/bin:/bin"
        sys.path[:0] = [str(args.tvm_source.resolve() / "python"), str(args.tvm_source.resolve() / "apps/gemmini")]
        import numpy as np
        import onnx
        import torch
        import tvm
        from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph
        import baremetal
        import verify_exported_graph as guest
        from verify_matmul import audit_instructions
        from tinyllama_integer import export_options
        from verify_session import StageCompiler

        torch.set_num_threads(1)
        report["versions"] = {"torch": torch.__version__, "tvm": tvm.__version__, "onnx": onnx.__version__, "numpy": np.__version__}
        report["source_files"] = [identity(__file__), identity(Path(__file__).with_name("block_workloads.py")),
                                  identity(Path(__file__).with_name("tinyllama_integer.py"))]
        report["source_files"].append(identity(Path(__file__).with_name("block_tiling.py")))
        if args.workload in ("smolvla-denoise", "smolvla-block-action"):
            report["source_files"].append(identity(Path(__file__).with_name("smolvla_denoise_block.py")))
        if args.specialize_static_offsets:
            report["source_files"].append(identity(Path(__file__).with_name("static_buffer_abi.py")))
        if args.localize_cpu_workspace or args.single_hart_cpu:
            report["source_files"].append(identity(Path(__file__).with_name("cpu_row_schedule.py")))
        snapshots = out / "source"
        snapshots.mkdir()
        for record in report["source_files"]:
            source = Path(record["path"])
            (snapshots / source.name).write_bytes(source.read_bytes())
        if args.resume_from is not None:
            prior_root = args.resume_from.resolve(strict=True)
            prior = json.loads((prior_root / "report.json").read_text())
            if not prior.get("capture") or not prior["capture"][0].get("structural_import_verified") or not prior.get("independent_integer_checks"):
                raise ValueError("Resume requires a completed structural capture and independent integer checks")
            if prior["preflight"]["workload"] != args.workload or prior["workload"]["fixture_seed"] != 20261019:
                raise ValueError("Resume workload or fixture identity mismatch")
            if args.workload == "smolvla-action" and prior["workload"]["chunk_size"] != args.action_chunk_size:
                raise ValueError("Resume action chunk size differs from captured fixture")
            if args.workload == "smolvla-action" and prior["workload"]["hidden_size"] != args.action_hidden_size:
                raise ValueError("Resume action hidden size differs from captured fixture")
            if args.workload == "tinyllama-head-ffn" and (prior["workload"]["hidden_size"], prior["workload"]["intermediate_size"]) != (args.tiny_hidden_size, args.tiny_intermediate_size):
                raise ValueError("Resume TinyLlama dimensions differ from captured fixture")
            for record in prior["capture"][0]["onnx_artifacts"]:
                if identity(record["path"])["sha256"] != record["sha256"]:
                    raise ValueError("Captured ONNX artifact changed")
            with np.load(prior_root / "fixture.npz", allow_pickle=False) as data:
                arrays = {name: data[name] for name in data.files}
            if arrays["input_0"].shape[0] != args.samples:
                raise ValueError("Resume sample count differs from captured fixture")
            if args.workload == "tinyllama-head-ffn" and arrays["input_0"].shape[1:] != (1, args.tokens, args.tiny_hidden_size):
                raise ValueError("Resume TinyLlama token/hidden geometry differs from captured fixture")
            count = len([name for name in arrays if name.startswith("output_")])
            references = [[arrays[f"output_{i}"][sample] for i in range(count)] for sample in range(args.samples)]
            with np.load(prior_root / "original-fp32.npz", allow_pickle=False) as data:
                original = [[data[f"output_{i}"][sample] for i in range(3)] for sample in range(args.samples)]
            description = prior["workload"]
            for name in ("workload", "original_reference", "independent_integer_checks", "quantization_vs_fp32", "capture"):
                report[name] = prior[name]
            report["resumed_capture"] = {"report": identity(prior_root / "report.json"), "fixture": identity(prior_root / "fixture.npz"),
                                         "original_fp32": identity(prior_root / "original-fp32.npz"), "new_source_execution": False}
            np.savez(out / "fixture.npz", **arrays)
            np.savez(out / "original-fp32.npz", **{f"output_{i}": np.stack([sample[i] for sample in original]) for i in range(3)})
            from verify_session import load_onnx_graph
            from tvm.relax.frontend.onnx import from_onnx
            graph, _ = load_onnx_graph(Path(prior["capture"][0]["onnx"]["path"]), onnx)
            mod = from_onnx(graph, keep_params_in_input=False)
        else:
            mod, arrays, original, references, description = capture(args, out, report, save, np, onnx, torch, tvm, StageCompiler, export_options)
        print("ONNX to Relax passed", flush=True)
        from block_tiling import hardware_facts, select_from_relax
        choices, selections = {}, []
        facts = hardware_facts(args.adapter_receipt, args.gemmini_params)
        if args.macro_tiles == "largest":
            choices, selections = select_from_relax(mod, tvm, facts, args.macro_fill_percent)
            if not choices:
                raise ValueError("Largest-tile policy found no static INT8/INT32 matmuls")
        report["macro_tiling"] = {"policy": args.macro_tiles, "hardware": facts, "selections": selections,
                                  "supported_tiles_per_axis": [1, 2, 4],
                                  "heuristic": "Maximum supported macro area, then fewer operand-load API calls, with declared grid fill and physical row capacities",
                                  "measured_tuning": False, "TVM_source_modified": False,
                                  "graph_optimization_mode": args.mode}
        prepared = prepare_gemmini_graph(mod, optimize=args.mode == "optimized", tile_choices=choices,
                                         route="legacy" if choices else "tensorir")
        actual = {(int(f.attrs["gemmini.m"]), int(f.attrs["gemmini.n"]), int(f.attrs["gemmini.k"])):
                  (int(f.attrs["gemmini.tile_i"]), int(f.attrs["gemmini.tile_j"]))
                  for f in prepared.functions.values() if isinstance(f, tvm.tir.PrimFunc) and f.attrs and "gemmini.m" in f.attrs}
        if choices and actual != choices:
            raise ValueError("Compiled device tile attributes differ from the explicit shape-derived selection")
        report["macro_tiling"]["compiled_attributes_verified"] = True
        constraints = []
        for gv, func in prepared.functions.items():
            if not isinstance(func, tvm.tir.PrimFunc):
                continue
            for buffer in func.buffer_map.values():
                if buffer.strides or not isinstance(buffer.elem_offset, tvm.tir.IntImm) or int(buffer.elem_offset) != 0:
                    constraints.append({"function": gv.name_hint, "buffer": buffer.name,
                                        "strides": [str(value) for value in buffer.strides], "offset": str(buffer.elem_offset)})
        report["noncompact_external_buffers"] = constraints
        if constraints:
            (out / "prepared-with-abi-constraints.py").write_text(prepared.script(show_meta=False))
        if args.specialize_static_offsets:
            from static_buffer_abi import specialize_zero_offsets
            prepared, changes = specialize_zero_offsets(prepared, tvm)
            report["static_buffer_abi_specialization"] = {
                "public_API": "tir.transform.prim_func_pass plus PrimFunc.specialize",
                "precondition": "Exporter allocates compact standalone global buffers with zero implicit offsets; nonzero constant offsets, dynamic shapes and explicit strides refused",
                "changes": changes, "TVM_source_modified": False, "arithmetic_rewrite": False}
        if args.localize_cpu_workspace:
            from cpu_row_schedule import localize_private_producers
            prepared, report["CPU_row_schedule"] = localize_private_producers(prepared, tvm)
        if args.single_hart_cpu:
            from cpu_row_schedule import serialize_parallel_loops
            prepared, report["single_hart_CPU_schedule"] = serialize_parallel_loops(prepared, tvm)
        save()
        manifest = baremetal.export_graph(prepared, out / "graph", memory_limit_bytes=192 << 20)
        verify_device(args, out, report, save, command, prepared, manifest, arrays, references, original, description,
                      np, tvm, guest, audit_instructions)
    except Exception as error:
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        save()
        raise


def geometry_options(args):
    return {name: getattr(args, name) for name in
            ("tokens", "action_chunk_size", "action_hidden_size", "tiny_hidden_size", "tiny_intermediate_size")}


def capture(args, out, report, save, np, onnx, torch, tvm, StageCompiler, export_options):
        """Create references and ONNX once; later compiler modes reuse this capture."""
        fp32, inputs, fp_description = make_workload(args.workload, args.checkpoint, quantized=False, **geometry_options(args))
        samples = [inputs, tuple(value * (-0.75) + 0.125 if i == 0 else value.clone() for i, value in enumerate(inputs))][:args.samples]
        with torch.no_grad():
            original = [[value.numpy().copy() for value in fp32(*sample)[:3]] for sample in samples]
        del fp32
        gc.collect()
        model, _, description = make_workload(args.workload, args.checkpoint, quantized=True, **geometry_options(args))
        report["workload"] = description
        report["original_reference"] = fp_description
        integer_checks = []

        def check_integer(module, values, result):
            codes, _ = module.integer.integer_inputs(values[0])
            left = codes.numpy().astype(np.int64)
            right = module.integer.weight_kn.numpy().astype(np.int64)
            reference = left @ right
            actual = result[1].numpy()
            if not np.array_equal(reference, actual):
                raise AssertionError("Independent complete-K INT64 oracle disagrees with PyTorch integer contraction")
            integer_checks.append({"shape": [left.shape[0], right.shape[1], left.shape[1]], "all_elements_exact": True})

        handles = [module.register_forward_hook(check_integer) for module in model.modules() if hasattr(module, "integer")]
        with torch.no_grad():
            references = [[value.numpy().copy() for value in model(*sample)] for sample in samples]
        for handle in handles:
            handle.remove()
        report["independent_integer_checks"] = integer_checks
        report["quantization_vs_fp32"] = [[metrics(actual, expected, np) for actual, expected in zip(sample[:3], source)]
                                          for sample, source in zip(references, original)]
        arrays = {f"input_{i}": np.stack([sample[i].numpy() for sample in samples]) for i in range(len(inputs))}
        arrays.update({f"output_{i}": np.stack([sample[i] for sample in references]) for i in range(len(references[0]))})
        np.savez(out / "fixture.npz", **arrays)
        np.savez(out / "original-fp32.npz", **{f"output_{i}": np.stack([sample[i] for sample in original]) for i in range(3)})
        save()
        print("capture and independent integer checks passed", flush=True)
        compiler = StageCompiler(out, args.mode, np, torch, onnx, tvm, exporter="dynamo", opset=22,
                                 export_options=export_options(torch))
        mod = compiler(SimpleNamespace(name=args.workload, module=model), tuple(value.numpy() for value in inputs),
                       len(references[0]), import_only=True)
        report["capture"] = compiler.records
        del model
        gc.collect()
        return mod, arrays, original, references, description


def verify_device(args, out, report, save, command, prepared, manifest, arrays, references, original, description,
                  np, tvm, guest, audit_instructions):
        report["static_memory"] = {key: manifest[key] for key in ("constants_bytes", "workspace_bytes", "explicit_tensor_bytes")}
        dimensions = {}
        for gv, func in prepared.functions.items():
            if func.attrs and "gemmini.m" in func.attrs:
                dimensions[gv.name_hint] = {key: int(func.attrs["gemmini." + key]) for key in ("m", "n", "k", "tile_i", "tile_j")}
        old = json.loads(args.adapter_receipt.read_text())
        policy = old["instruction_policy"]
        forbidden = {value for name, value in policy["derived_definitions"].items() if name.startswith("k_LOOP")}
        header = args.adapter.resolve().parent / "headers/include/gemmini_params.h"
        if identity(header)["sha256"] != old["headers"]["include/gemmini_params.h"]["sha256"]:
            raise ValueError("Adapter parameter header differs from the instruction-policy receipt")
        dim = next(int(line.split()[2]) for line in header.read_text().splitlines() if line.split()[:2] == ["#define", "DIM"])
        counts = dict.fromkeys(("loads_a", "loads_b", "computes", "stores", "admissions"), 0)
        device, cpu = [], []
        for call in manifest["calls"]:
            if call["symbol"] not in dimensions:
                cpu.append(call["symbol"])
                continue
            row = dimensions[call["symbol"]]
            m, n, k, ti, tj = [row[key] for key in ("m", "n", "k", "tile_i", "tile_j")]
            mt, nt, kt = [(value + dim - 1) // dim for value in (m, n, k)]
            counts["loads_a"] += mt * ((n + tj * dim - 1) // (tj * dim)) * kt
            counts["loads_b"] += nt * ((m + ti * dim - 1) // (ti * dim)) * kt
            counts["computes"] += mt * nt * kt
            counts["stores"] += mt * nt
            counts["admissions"] += 1
            device.append({"symbol": call["symbol"], **row})
        if counts["admissions"] != description["expected_gemmini_contractions"]:
            raise AssertionError("Unexpected accelerator dispatch coverage")
        report["dispatch"] = {"Gemmini": device, "CPU": cpu, "expected_primitive_counts": counts,
                              "onnx_operators": report["capture"][0].get("onnx_inventory")}
        build = out / "guest"
        build.mkdir()
        guest.make_guest(build, manifest, arrays, args.samples, args.repeats, counts)
        runner = (build / "runner.c").read_text().replace("DECLARATIONS", "")
        position = runner.index("int main(void)")
        runner = runner[:position] + FLOAT_CHECK + runner[position:]
        for i, tensor in enumerate(manifest["outputs"]):
            if tensor["dtype"] == "float32":
                needle = f"equal(output_{i}[slot].value,fixture_output_{i}"
                if runner.count(needle) != 1:
                    raise AssertionError("Unexpected generated verifier output check")
                runner = runner.replace(needle, f"equal_f32(output_{i}[slot].value,fixture_output_{i}")
        dump = "\n".join(f"dump_f32(call,{i},output_{i}[slot].value,{tensor['bytes']});" for i, tensor in enumerate(manifest["outputs"]) if tensor["dtype"] == "float32")
        executed = "if(model_run(inputs,outputs,workspace.value,WORKSPACE_BYTES))return 21;"
        if runner.count(executed) != 1:
            raise AssertionError("Unexpected generated verifier execution boundary")
        runner = runner.replace(executed, executed + "\n" + dump)
        runner = runner.replace("TVM_EXPORTED_GRAPH_PASS", "TVM_BLOCK_GRAPH_PASS").replace("exact_bytes=1", "integer_exact=1 float_tolerance=1")
        if not description["integer_outputs"]:
            runner = runner.replace("integer_exact=1", "integer_exact=not_exposed")
        (build / "runner.c").write_text(runner)
        # Upstream newlib's default __errno pulls reentrancy/stdio/heap support.
        # This explicitly single-hart, heap-free guest needs only errno storage.
        (build / "errno.c").write_text("static int guest_errno;\nint *__errno(void) { return &guest_errno; }\n")
        gcc = args.riscv_gcc.resolve()
        flags = ["-std=gnu11", "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-msmall-data-limit=0", "-O2", "-ffreestanding", "-fno-builtin", "-fno-stack-protector", "-fno-tree-loop-distribute-patterns", "-ffp-contract=off", "-nostdlib", "-nostartfiles", "-static", "-Wl,--no-relax", "-Wl,-T," + str(build / "link.ld"), "-I" + str(args.tvm_source.resolve() / "apps/gemmini"), "-I" + str(build)]
        wrappers = ["-Wl,--wrap=tvm_gemmini_" + name for name in ("load_a", "load_b", "compute", "store", "validate_matmul_i8_i32")]
        libraries = [subprocess.check_output([str(gcc), "-march=rv64gc", "-mabi=lp64d", "-print-file-name=" + name], text=True).strip() for name in ("libm.a", "libc.a", "libgcc.a")]
        report["runtime"] = {"adapter": identity(args.adapter), "adapter_receipt": identity(args.adapter_receipt),
                             "Spike": identity(args.spike), "plugin": identity(args.plugin),
                             "dtc": identity(args.dtc),
                             "simulator_receipt": identity(args.simulator_receipt), "math_libraries": [identity(path) for path in libraries],
                             "scope": "functional Gemmini extension; no Rocket microarchitectural, RTL-equivalence or timing verdict",
                             "errno_abi": "Explicit single-hart static __errno storage; upstream math implementations unchanged; graph storage static and output through HTIF",
                             "errno_source": identity(build / "errno.c")}
        for name, wrong in (("valid", 0), ("bad-oracle", 1)):
            elf = build / (name + ".elf")
            command([gcc, *flags, *wrappers, "-DBAD_ORACLE=" + str(wrong), "-Wl,-Map," + str(build / (name + ".map")),
                     build / "start.S", build / "runner.c", build / "errno.c", build / "fixture.S", out / "graph/model.c", out / "graph/constants.S",
                     out / "graph/operators.o", args.adapter, "-Wl,--start-group", *libraries, "-Wl,--end-group", "-o", elf], "link-" + name)
            audit = audit_instructions(elf.read_bytes(), policy["custom_opcode"], forbidden)
            if audit["prohibited_hits"]:
                raise AssertionError("Hardware-loop instructions in linked executable")
            report.setdefault("instruction_audits", {})[name] = audit
            from verify_baremetal_cpu import audit_cpu_runtime
            tools = {name: gcc.with_name("riscv64-unknown-elf-" + name) for name in ("nm", "readelf")}
            report.setdefault("runtime_audits", {})[name] = audit_cpu_runtime(
                elf, build / (name + ".map"), {"archives": [identity(path) for path in libraries]}, tools, command)
            if command([tools["nm"], "-u", elf], "undefined-" + name).strip():
                raise AssertionError("Unresolved guest runtime symbols")
            symbols = command([tools["nm"], "--defined-only", elf], "runtime-callbacks-" + name)
            callbacks = [line.split()[-1] for line in symbols.splitlines() if line.split() and
                         (line.split()[-1].startswith("__TVMBackend") or line.split()[-1] in ("__TVMFuncCall", "__TVMFuncGetEnv"))]
            if callbacks:
                raise ValueError("Guest lacks initialization/runtime for TVM callbacks: " + ", ".join(callbacks))
            log = command([args.spike, "--isa=rv64gc", "-m0x80000000:0x10000000", "-p1", "--extlib=" + str(args.plugin.resolve()),
                           "--extension=gemmini", elf], "spike-" + name, expected=22 if wrong else 0)
            if wrong:
                if "EXPORTED_GRAPH_NUMERICAL_FAILURE" not in log:
                    raise AssertionError("Negative control lacks numerical failure marker")
            else:
                if f"TVM_BLOCK_GRAPH_PASS calls={args.samples * args.repeats}" not in log:
                    raise AssertionError("Missing complete acceptance marker")
                actual = {}
                for line in log.splitlines():
                    if line.startswith("BLOCK_OUTPUT "):
                        row = dict(part.split("=", 1) for part in line.split()[1:])
                        key = int(row["call"]), int(row["index"])
                        values = np.frombuffer(bytes.fromhex(row["hex"]), dtype=">u4").astype("<u4").view("<f4")
                        start = int(row["offset"])
                        target = actual.setdefault(key, np.full(references[0][key[1]].size, np.nan, dtype=np.float32))
                        target[start:start + len(values)] = values
                comparisons = []
                for call in range(args.samples * args.repeats):
                    sample = call % args.samples
                    for i in range(3):
                        value = actual[(call, i)].reshape(references[sample][i].shape)
                        np.testing.assert_allclose(value, references[sample][i], rtol=1e-4, atol=1e-5)
                        comparisons.append({"call": call, "output": i, "vs_quantized_torch": metrics(value, references[sample][i], np),
                                            "vs_original_fp32": metrics(value, original[sample][i], np)})
                report["Spike_output_comparisons"] = comparisons
                np.savez(out / "spike-actual-outputs.npz", **{f"call_{call}_output_{i}": value for (call, i), value in actual.items()})
        report.update(status="passed", float_gate={"rtol": 1e-4, "atol": 1e-5},
                      integer_gate="all declared INT32 output bytes exact" if description["integer_outputs"] else "Native contractions checked against INT64; device intermediates not exposed",
                      repeats=args.repeats, samples=args.samples, negative_control="expected code22 and numerical failure marker",
                      generated_source_adaptations="Float-output tolerance and raw-output dumping in trusted verifier; original guards, preservation, poisoning and primitive counters retained")
        save()


if __name__ == "__main__":
    main()
