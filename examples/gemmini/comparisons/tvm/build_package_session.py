#!/usr/bin/env python3
"""Link explicit TVM stage packages through Merlin's bare-metal runtime.

Consumes the existing verifier's route/fixture format. It never runs FireSim.
The resulting positive and wrong-oracle ELFs include every call/output and the
existing one-warm/one-measured protocol, with independent state reset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import shutil
import subprocess
import sys
from pathlib import Path

from package_session import WIDTHS, plan_session, render_session
from session_memory import blob_assembly, linker_suffix, plan_memory

from merlin.common.paths import repo_root
from merlin.perf.warm_profile_harness import TargetInvocationHooks, render_warm_then_measure_main


def identity(path):
    path = Path(path).resolve(strict=True)
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return {"path": str(path), "sha256": sha.hexdigest(), "bytes": path.stat().st_size}


def require_identity(record):
    current = identity(record["path"])
    if current["sha256"] != record["sha256"]:
        raise ValueError("Changed reference artifact: " + record["path"])
    return current


def reference_arrays(fixture, provenance, route, plan):
    import numpy as np

    record = json.loads(provenance.read_text())
    if (
        record.get("fixture_sha256") != identity(fixture)["sha256"]
        or not record.get("reference_contract")
        or not record.get("reference_sources")
    ):
        raise ValueError("Require independent source-bound fixture provenance")
    sources = [require_identity(item) for item in record["reference_sources"]]
    if not record.get("route") or record["route"]["sha256"] != identity(route)["sha256"]:
        raise ValueError("Reference must bind the exact session route")
    arrays = {}
    with np.load(fixture, allow_pickle=False) as data:
        for kind in ("input", "output"):
            for index, tensor in enumerate(plan[kind + "s"]):
                name = f"{kind}_{index}"
                value = data[name]
                shape = tuple(tensor["shape"])
                if value.shape == (1, *shape):
                    value = value[0]
                dtype = "uint16" if tensor["dtype"] == "bfloat16" else tensor["dtype"]
                if value.shape != shape or value.dtype.name != dtype or value.nbytes != tensor["bytes"]:
                    raise ValueError("Fixture ABI mismatch: " + name)
                if value.dtype.kind == "f" and not np.isfinite(value).all():
                    raise ValueError("Nonfinite reference tensor: " + name)
                if tensor["dtype"] == "bfloat16" and ((value & 0x7F80) == 0x7F80).any():
                    raise ValueError("Nonfinite BF16 reference tensor: " + name)
                arrays[name] = np.ascontiguousarray(value.astype(value.dtype.newbyteorder("<"), copy=False))
        if set(data.files) != set(arrays):
            raise ValueError("Fixture contains unaccounted arrays")
    return arrays, sources


def readback_source(plan):
    """Carry actual output words through Merlin's existing OUT_BIN codec."""
    lines = [
        '#include "out_bin.h"',
        "static int session_write(const void *data,size_t n) { const uint8_t *p=data;for(size_t i=0;i<n;i++)htif_putc((char)p[i]);return 1;}",
        "static int session_readback(void) { merlin_out_bin writer;",
    ]
    for index, tensor in enumerate(plan["outputs"]):
        dtype, size = tensor["dtype"], tensor["bytes"]
        width = WIDTHS[dtype]
        cols = tensor["shape"][-1] if tensor["shape"] else 1
        rows = size // width // cols
        signed = dtype.startswith("int")
        at = plan["offsets"]["output"][index]
        lines += [
            f'  printf("OUT_BIN_BEGIN v1 out{index} {rows} {cols} {width} {"s" if signed else "u"} {size}\\n");',
            "  htif_line_flush(0);",
            f"  merlin_out_bin_init(&writer,{size // width},{width},{int(signed)},session_write);",
            f"  for(uint64_t i=0;i<{size};i+={width}) {{ uint64_t word=0;",
            f"    for(unsigned b=0;b<{width};b++)word|=(uint64_t)session_storage[{at}+i+b]<<(8*b);",
        ]
        if signed and width < 8:
            lines.append(f"    if(word&(UINT64_C(1)<<{width * 8 - 1}))word|=UINT64_MAX<<{width * 8};")
        lines += [
            "    if(!merlin_out_bin_word(&writer,word))return 25; }",
            "  if(!merlin_out_bin_finish(&writer))return 25;",
            '  htif_line_flush(1);htif_puts("OUT_BIN_END v1 ");',
            "  for(int shift=60;shift>=0;shift-=4)htif_putc(\"0123456789abcdef\"[(writer.checksum>>shift)&15]);htif_putc('\\n');",
        ]
    lines += ["  return 0;", "}"]
    return "\n".join(lines) + "\n"


def build(args):
    app = args.tvm_source.resolve(strict=True) / "apps/gemmini"
    sys.path.insert(0, str(app))
    from compile_package import FLAGS, verify_package
    from qualify_package import elf_memory
    from verify_exported_graph import strict_json, validate_manifest
    from verify_matmul import audit_instructions

    spec = strict_json(args.stages)
    if set(spec) != {"version", "stages"} or spec["version"] != 1 or not isinstance(spec["stages"], dict):
        raise ValueError("Require version-one stage-name to package-path mapping")
    packages = {name: (args.stages.parent / path).resolve(strict=True) for name, path in spec["stages"].items()}
    documents, stages, bindings = {}, {}, []
    for name, package in packages.items():
        documents[name] = verify_package(package)
        stages[name] = strict_json(package / "graph/graph.json")
        validate_manifest(stages[name])
        bindings.append(identity(package / "package.json"))
    first = next(iter(packages.values()))
    for filename in ("runtime.o", "adapter.o", "libm.a", "libgcc.a"):
        if len({identity(folder / filename)["sha256"] for folder in packages.values()}) != 1:
            raise ValueError("Stages require identical runtime/adapter/math libraries: " + filename)
    if len({doc["capability_fingerprint"] for doc in documents.values()}) != 1:
        raise ValueError("Stage capability fingerprints differ")
    route = strict_json(args.route)
    plan = plan_session(stages, route)
    arrays, sources = reference_arrays(args.fixture, args.fixture_provenance, args.route, plan)
    layout = plan_memory(
        plan, stages, args.memory_bytes, args.stack_bytes, weights_base=args.weights_base, arena_base=args.arena_base
    )
    absolute = layout["kind"] == "absolute"
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    commands = []

    def command(argv, *, label):
        result = subprocess.run([str(x) for x in argv], cwd=out, text=True, capture_output=True, timeout=300)
        (out / (label + ".log")).write_text(result.stdout + result.stderr)
        commands.append({"argv": [str(x) for x in argv], "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(label + " failed: " + result.stderr[-4000:])
        return result.stdout

    gcc = args.riscv_gcc.resolve(strict=True)
    tools = {}
    for name in ("nm", "objcopy"):
        value = command([gcc, "-print-prog-name=" + name], label="tool-" + name).strip()
        tools[name] = Path(shutil.which(value) or value).resolve(strict=True)
    objects, symbols = [], {}
    # Rename all defined globals in the stage itself. Runtime and adapter stay
    # shared, so Gemmini configuration shadow state has one owner across stages.
    for index, (name, package) in enumerate(packages.items()):
        glue = package / "graph/model.o"
        constants = [package / "graph/constants.o"]
        if absolute:
            # Recompile the package's declared glue interface only. The exact
            # scheduled operator object and original constant bytes are reused.
            glue = out / f"stage_{index}.glue.o"
            address = layout["stage_constants"][name]
            command(
                [
                    gcc,
                    *FLAGS,
                    f"-DTVM_MODEL_CONSTANTS_ADDRESS=UINT64_C({address})",
                    "-c",
                    package / "graph/model.c",
                    "-o",
                    glue,
                ],
                label=f"stage-{index}-absolute-glue",
            )
            payload = package / "graph/constants.bin"
            if payload.stat().st_size != max(stages[name]["constants_bytes"], 1):
                raise ValueError("Package constant payload extent differs from its manifest")
            (out / f"stage_{index}.constants.bin").symlink_to(payload)
            constants = []
        combined = out / f"stage_{index}.original.o"
        command(
            [
                gcc,
                *FLAGS,
                "-nostdlib",
                "-nostartfiles",
                "-Wl,-r,--no-relax",
                glue,
                package / "graph/operators.o",
                *constants,
                "-o",
                combined,
            ],
            label=f"stage-{index}-combine",
        )
        rows = command(
            [tools["nm"], "--defined-only", "--extern-only", "--format=posix", combined], label=f"stage-{index}-symbols"
        )
        names = [row.split()[0] for row in rows.splitlines() if row.strip()]
        if "model_run" not in names or (not absolute and "model_constants" not in names):
            raise ValueError("Package lacks its declared stage symbols")
        symbols[name] = {symbol: f"stage_{index}_" + symbol for symbol in names}
        mapping = out / f"stage_{index}.symbols"
        mapping.write_text("".join(f"{old} {new}\n" for old, new in symbols[name].items()))
        renamed = out / f"stage_{index}.o"
        command(
            [tools["objcopy"], "--redefine-syms=" + str(mapping), combined, renamed], label=f"stage-{index}-namespace"
        )
        objects.append(renamed)

    assembly = ['.section .rodata.session_fixture,"a",@progbits']
    for name, array in arrays.items():
        array.tofile(out / (name + ".bin"))
        assembly += [".balign 64", ".global session_" + name, "session_" + name + ":", f'.incbin "{name}.bin"']
    assembly.append('.section .note.GNU-stack,"",@progbits')
    fixture_objects = [out / "fixtures.S"]
    if absolute:
        (out / "weights_blob.S").write_text(blob_assembly(layout, stages, arrays))
        command([gcc, *FLAGS, "-c", out / "weights_blob.S", "-o", out / "weights_blob.o"], label="absolute-weights")
        fixture_objects = [out / "weights_blob.o"]
    else:
        (out / "fixtures.S").write_text("\n".join(assembly) + "\n")
    profile_stages = getattr(args, "profile_stages", False)
    source = render_session(
        plan,
        rtol=args.rtol,
        atol=args.atol,
        negative_output=args.negative_output,
        memory_layout=layout,
        profile_stages=profile_stages,
    )
    source += '\n#include "htif.h"\nextern void tvm_gemmini_end(void);\n'
    source += 'static uint64_t read_cycles(void) { uint64_t v; __asm__ volatile("csrr %0,mcycle":"=r"(v)::"memory");return v; }\n'
    source += "static uint64_t session_hash(const uint8_t *p,uint64_t n) { uint64_t h=UINT64_C(14695981039346656037);for(uint64_t i=0;i<n;i++)h=(h^p[i])*UINT64_C(1099511628211);return h;}\n"
    constant_checks = []
    constant_addresses = []
    for index, (name, stage) in enumerate(stages.items()):
        address = f"stage_{index}_model_constants"
        if absolute:
            address = f"((const uint8_t *)(uintptr_t)UINT64_C({layout['stage_constants'][name]}))"
        else:
            source += f"extern const uint8_t {address}[];\n"
        constant_addresses.append(address)
        constant_checks.append(f"uint64_t before_{index}=session_hash({address},{stage['constants_bytes']});")
    unchanged = " && ".join(
        f"before_{i}==session_hash({constant_addresses[i]},{stage['constants_bytes']})"
        for i, stage in enumerate(stages.values())
    )
    marker = f"TVM_SESSION_PASS calls={len(plan['calls'])} outputs={len(plan['outputs'])} resets=2"
    source += f'static int session_measured_validate(void) {{ int rc=session_validate(BAD_ORACLE);if(!rc)printf("{marker}\\n");return rc; }}\n'
    if args.readback:
        source += readback_source(plan)
    success = []
    if profile_stages:
        success.append("session_report_stage_cycles();")
    if args.readback:
        success.append("if(session_readback())return 25;")
    source += render_warm_then_measure_main(
        prepare_input="int rc;\n" + "\n".join(constant_checks) + "\nsession_reset();",
        invocation=TargetInvocationHooks(invoke="if((rc=session_run()))return rc;", complete="tvm_gemmini_end();"),
        reset_after_warm=f"if((rc=session_validate(0)))return rc;\nif(!({unchanged}))return 24;\nsession_reset();",
        validate_outputs=f"({unchanged}) ? session_measured_validate() : 24",
        function_name="session_profile",
        success_body="\n".join(success) or None,
    )
    source += 'int main(int hart) { if(hart)for(;;){} console_init();int rc=session_profile();printf("TVM_SESSION_EXIT rc=%d\\n",rc);if(!rc)htif_puts("DONE\\n");htif_exit(rc);}\n'
    (out / "session.c").write_text(source)
    runtime = repo_root() / "merlin/runtime/baremetal/spike"
    linker = (runtime / "model_link.ld").read_text()
    linker += linker_suffix(layout)
    (out / "session.ld").write_text(linker)
    policy = json.loads((first / "build.json").read_text())["adapter"]["instruction_policy"]
    forbidden = {value for key, value in policy["derived_definitions"].items() if key.startswith("k_LOOP_")}
    elfs = {}
    for negative, name in ((0, "valid"), (1, "bad-oracle")):
        elf = out / (name + ".elf")
        placement_flags = [f"-Wl,--defsym,MERLIN_WEIGHTS_BASE={layout['weights_base']}"] if absolute else []
        argv = [
            gcc,
            *FLAGS,
            *placement_flags,
            "-msmall-data-limit=0",
            "-fno-stack-protector",
            "-fno-builtin",
            "-nostdlib",
            "-nostartfiles",
            "-static",
            "-Wl,--no-relax",
            f"-Wl,--defsym,MERLIN_STACK_BYTES={args.stack_bytes}",
            "-Wl,-T," + str(out / "session.ld"),
            "-Wl,--defsym,MERLIN_HART_COUNT=1",
            "-Wl,--defsym,MERLIN_DRAM_BASE=0x80000000",
            f"-Wl,--defsym,MERLIN_DRAM_SPAN={args.memory_bytes}",
            "-Wl,-Map," + str(out / (name + ".map")),
            "-DBAD_ORACLE=" + str(negative),
            "-I" + str(runtime),
            "-I" + str(runtime.parent),
            runtime / "crt.S",
            runtime / "htif.c",
            runtime / "printf_min.c",
            out / "session.c",
            *fixture_objects,
            *objects,
            first / "runtime.o",
            first / "adapter.o",
            "-Wl,--start-group",
            first / "libm.a",
            first / "libgcc.a",
            "-Wl,--end-group",
            "-o",
            elf,
        ]
        command(argv, label=name + "-link")
        with elf.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            memory = elf_memory(data, args.memory_bytes)
            audit = audit_instructions(data, policy["custom_opcode"], forbidden)
        memory["stack_reserved_bytes"] = args.stack_bytes
        if audit["prohibited_hits"]:
            raise ValueError("Session ELF contains prohibited hardware loop commands")
        if command([tools["nm"], "--undefined-only", elf], label=name + "-undefined").strip():
            raise ValueError("Session ELF has unresolved symbols")
        elfs[name] = {**identity(elf), "memory": memory, "instruction_audit": audit}
    bindings += sources + [
        identity(p)
        for p in (
            args.stages,
            args.route,
            args.fixture,
            args.fixture_provenance,
            gcc,
            Path(__file__),
            Path(__file__).with_name("package_session.py"),
            Path(__file__).with_name("session_memory.py"),
            runtime / "crt.S",
            runtime / "htif.c",
            runtime / "printf_min.c",
            runtime / "model_link.ld",
            runtime / "htif.h",
            runtime.parent / "out_bin.h",
            repo_root() / "src/merlin/perf/warm_profile_harness.py",
        )
    ]
    for record in bindings:
        require_identity(record)
    for package in packages.values():
        verify_package(package)
    result = {
        "schema": "tvm_merlin_package_session_v1",
        "status": "linked_not_executed",
        "plan": plan,
        "stage_symbols": symbols,
        "ELFs": elfs,
        "bindings": bindings,
        "commands": commands,
        "memory_bytes": args.memory_bytes,
        "stack_bytes": args.stack_bytes,
        "memory_layout": layout,
        "package_reuse": {
            "operator_objects": "unchanged",
            "constant_payloads": "unchanged",
            "graph_glue_recompiled_for_absolute_address": absolute,
        },
        "comparison": {"rtol": args.rtol, "atol": args.atol, "integer": "exact", "finite_required": True},
        "readback": {
            "enabled": args.readback,
            "protocol": "OUT_BIN v1",
            "order": "all invocation outputs in call/output order",
        },
        "timing": "One warm full session, reset, one measured full session including routing and completion; validation and reset outside interval",
        "stage_profiling": {
            "enabled": profile_stages,
            "scope": "Each synchronous stage model_run; routing/reset/checks excluded; print after validation",
            "aggregate_includes_counter_overhead": profile_stages,
        },
        "deployed_hardware_verified": False,
    }
    (out / "session.json").write_text(json.dumps(result, indent=2) + "\n")
    from merlin.perf.firesim_receipt import UartValidationPolicy

    policy = UartValidationPolicy(
        policy_id="tvm-session-" + identity(out / "session.json")["sha256"],
        workload=args.workload,
        success_markers=(marker,),
    )
    (out / "validation-policy.json").write_text(json.dumps(policy.to_dict(), indent=2) + "\n")
    print(out / "session.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tvm-source", "stages", "route", "fixture", "fixture-provenance", "output-dir", "riscv-gcc"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--memory-bytes", type=int, default=256 << 20)
    parser.add_argument("--stack-bytes", type=int, default=256 << 10)
    parser.add_argument(
        "--weights-base",
        type=lambda value: int(value, 0),
        help="Explicit absolute base for constants and reference/input blobs",
    )
    parser.add_argument(
        "--arena-base",
        type=lambda value: int(value, 0),
        help="Explicit absolute base for guarded session buffers/workspace",
    )
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--negative-output", type=int, default=0)
    parser.add_argument("--workload", default="tvm-static-session")
    parser.add_argument(
        "--readback", action="store_true", help="Emit all actual output words after the measured interval"
    )
    parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="Retain per-call warm/measured cycles and print after validation; adds counter overhead",
    )
    build(parser.parse_args())


if __name__ == "__main__":
    main()
