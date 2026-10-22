#!/usr/bin/env python3
"""Relink passing static block tests and stage bare-metal FireSim workloads.

This builds guest software only. A matching single-hart Gemmini FPGA image,
TSI/FESVR bridge, driver and run host must be qualified separately. No FPGA job
is submitted. The retained verifier initializes BSS itself and needs no rootfs.

Run under systemd-run --user --wait --pipe --collect -p MemoryMax=512M
-p MemorySwapMax=0 -p CPUQuota=100% -p RuntimeMaxSec=120, passing this script
--report BLOCK/report.json (repeatable), --hardware-receipt RECEIPT,
--output-dir OUTPUT and optionally --check-spike.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def resources():
    entry = next(line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    group = Path("/sys/fs/cgroup") / entry.lstrip("/")
    return {name: (group / name).read_text().strip() for name in
            ("memory.max", "memory.swap.max", "memory.peak", "memory.events", "cpu.max")}


def require_budget():
    state = resources()
    quota, period = state["cpu.max"].split()
    if state["memory.max"] == "max" or int(state["memory.max"]) > 512 << 20 or state["memory.swap.max"] != "0":
        raise ValueError("Requires an enforced aggregate memory cap <=512 MiB and no swap")
    if quota == "max" or int(quota) > int(period):
        raise ValueError("Requires an aggregate CPU quota <=1 CPU")


def elf_layout(raw, ranges):
    """Validate physical/virtual load extent and entry against selected board data."""
    if raw[:7] != b"\x7fELF\x02\x01\x01" or len(raw) < 64:
        raise ValueError("Requires little-endian ELF64")
    header = struct.unpack_from("<HHIQQQIHHHHHH", raw, 16)
    kind, machine, _, entry, offset, _, flags, _, size, count, *_ = header
    if kind != 2 or machine != 243 or flags & 6 != 4 or size != 56:
        raise ValueError("Requires an executable RISC-V LP64D ELF")
    segments = []
    loaded = hashlib.sha256(struct.pack("<Q", entry))
    for index in range(count):
        segment = struct.unpack_from("<IIQQQQQQ", raw, offset + index * size)
        typ, permissions, file_offset, virtual, physical, file_bytes, memory_bytes, _ = segment
        if typ == 3:
            raise ValueError("Dynamic interpreter is unsupported")
        if typ != 1:
            continue
        if virtual != physical or file_bytes > memory_bytes or file_offset + file_bytes > len(raw):
            raise ValueError("Invalid bare-metal load segment")
        if not any(row["base"] <= physical and physical + memory_bytes <= row["base"] + row["size"] for row in ranges):
            raise ValueError("Load segment exceeds selected enabled DRAM")
        segments.append({"base": physical, "file_bytes": file_bytes, "memory_bytes": memory_bytes, "flags": permissions})
        loaded.update(struct.pack("<QQQQ", physical, file_bytes, memory_bytes, permissions))
        loaded.update(raw[file_offset:file_offset + file_bytes])
    if not any(row["flags"] & 1 and row["base"] <= entry < row["base"] + row["file_bytes"] for row in segments):
        raise ValueError("Entry is not backed by executable loaded bytes")
    return {"entry": entry, "segments": segments, "loaded_image_sha256": loaded.hexdigest(),
            "load_span_including_BSS_and_stack": max(row["base"] + row["memory_bytes"] for row in segments) - min(row["base"] for row in segments)}


def workload_descriptor(name, variant):
    return {"benchmark_name": name, "common_bootbinary": variant + ".elf",
            "common_rootfs": None, "common_outputs": [],
            "common_simulation_outputs": ["uartlog"]}


def spike_passed(text, code, calls, negative=False):
    if negative:
        return code == 22 and "EXPORTED_GRAPH_NUMERICAL_FAILURE" in text and "TVM_BLOCK_GRAPH_PASS" not in text
    return code == 0 and f"TVM_BLOCK_GRAPH_PASS calls={calls} " in text and "EXPORTED_GRAPH_NUMERICAL_FAILURE" not in text


def prepare(args):
    require_budget()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    hardware = json.loads(args.hardware_receipt.read_text())
    if hardware["hardware"]["cpu_count"] != 1:
        raise ValueError("Retained startup requires a single-hart board")
    for binding in (hardware["hardware"]["generated_configuration"], hardware["hardware"]["memory_map"]):
        if identity(binding["path"])["sha256"] != binding["sha256"]:
            raise ValueError("Selected board metadata changed")
    ranges = hardware["hardware"]["enabled_guest_DRAM"]["ranges"]
    receipt = {"status": "building", "builder": identity(__file__),
               "local_board_reference": identity(args.hardware_receipt), "workloads": [],
               "FireSim_executed": False, "full_models_qualified": False,
               "model_quality_qualified": False, "timing_qualified": False,
               "hardware_requirements": {"harts": 1, "ISA": "RV64GC/LP64D plus matching Gemmini custom instructions",
                   "reference_DRAM_ranges": ranges, "boot_and_console": "TSI/FESVR ELF loading, HTIF syscall stdout and exit",
                   "image_and_driver_bound": False},
               "scope": "Retained checkpoint-based isolated W8A8 block tests; synthetic inputs; original guards, poisoning, repeats, input/constant/output preservation and numerical gates retained"}

    def save():
        receipt["resources"] = resources()
        (out / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def command(argv, folder, label, timeout=45, environment=None):
        start = time.monotonic()
        result = subprocess.run(list(map(str, argv)), capture_output=True, timeout=timeout, env=environment)
        log = folder / (label + ".log")
        log.write_bytes(result.stdout + result.stderr)
        return result, {"argv": list(map(str, argv)), "exit_code": result.returncode,
                        "wall_seconds": time.monotonic() - start, "log": identity(log)}

    save()
    try:
        for report_path in args.report:
            source = report_path.resolve(strict=True)
            report = json.loads(source.read_text())
            if report["status"] != "passed":
                raise ValueError("Requires a passing block report: " + str(source))
            name = source.parent.name
            folder = out / "workloads" / name
            folder.mkdir(parents=True, exist_ok=False)
            row = {"name": name, "source_report": identity(source), "workload": report["workload"],
                   "float_gate": report["float_gate"], "samples": report["samples"], "repeats": report["repeats"],
                   "dispatch": report["dispatch"], "commands": [], "ELFs": {}, "descriptors": {},
                   "retained_instruction_audits": report["instruction_audits"],
                   "retained_runtime_audits": report["runtime_audits"]}
            receipt["workloads"].append(row)
            for binding in (report["runtime"]["adapter"], *report["runtime"]["math_libraries"]):
                if identity(binding["path"])["sha256"] != binding["sha256"]:
                    raise ValueError("Retained runtime input changed")
            graph = json.loads((source.parent / "graph/graph.json").read_text())
            if identity(source.parent / "graph/constants.bin")["sha256"] != graph["constants_sha256"]:
                raise ValueError("Graph constants changed")
            for variant in ("valid", "bad-oracle"):
                template = next(item["argv"] for item in report["commands"] if item["label"] == "link-" + variant)
                argv = list(template)
                argv[argv.index("-o") + 1] = str(folder / (variant + ".elf"))
                argv = ["-Wl,-Map," + str(folder / (variant + ".map")) if item.startswith("-Wl,-Map,") else item for item in argv]
                inputs = [Path(item) for item in argv[1:] if not item.startswith("-") and Path(item).is_file()]
                row.setdefault("build_inputs", {})[variant] = [identity(path) for path in dict.fromkeys(inputs)]
                row["compiler"] = identity(argv[0])
                result, record = command(argv, folder, "link-" + variant)
                row["commands"].append(record)
                result.check_returncode()
                elf = folder / (variant + ".elf")
                original = source.parent / "guest" / (variant + ".elf")
                binding = identity(elf)
                layout = elf_layout(elf.read_bytes(), ranges)
                if layout != elf_layout(original.read_bytes(), ranges):
                    raise ValueError("Fresh load image differs from retained passing workload")
                row["ELFs"][variant] = {**binding, "retained_ELF": identity(original),
                    "byte_identical_to_retained_guest": binding["sha256"] == identity(original)["sha256"],
                    "loaded_image_identical_to_retained_guest": True, "layout": layout}
                nm = Path(argv[0]).with_name("riscv64-unknown-elf-nm")
                result, record = command([nm, "-u", elf], folder, "undefined-" + variant)
                row["commands"].append(record)
                if result.returncode or result.stdout.strip():
                    raise ValueError("Guest has unresolved symbols")
                descriptor = out / "workloads" / (name + "-" + variant + ".json")
                descriptor.write_text(json.dumps(workload_descriptor(name, variant), indent=2) + "\n")
                row["descriptors"][variant] = identity(descriptor)
                if args.check_spike:
                    runtime = report["runtime"]
                    for key in ("Spike", "plugin", "dtc"):
                        if identity(runtime[key]["path"])["sha256"] != runtime[key]["sha256"]:
                            raise ValueError("Functional simulator input changed")
                    env = dict(os.environ, PATH=str(Path(runtime["dtc"]["path"]).parent) + os.pathsep + os.environ["PATH"])
                    memory = ",".join(f"{item['base']:#x}:{item['size']:#x}" for item in ranges)
                    result, record = command([runtime["Spike"]["path"], "--isa=rv64gc", "-m" + memory, "-p1",
                        "--extlib=" + runtime["plugin"]["path"], "--extension=gemmini", elf], folder, "spike-" + variant, environment=env)
                    text = (folder / ("spike-" + variant + ".log")).read_text(errors="replace")
                    record["passed"] = spike_passed(text, result.returncode, report["samples"] * report["repeats"], variant == "bad-oracle")
                    record["qualification"] = "Functional Spike extension only; no new RTL-equivalence or timing qualification"
                    row["commands"].append(record)
                    if not record["passed"]:
                        raise ValueError("Fresh functional check failed")
                save()
        receipt["status"] = "prepared"
        save()
    except Exception as error:
        receipt.update(status="failed", failure=str(error))
        save()
        raise
    print(json.dumps({"manifest": str(out / "manifest.json"), "status": receipt["status"], "workloads": len(receipt["workloads"])}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--hardware-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-spike", action="store_true")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
