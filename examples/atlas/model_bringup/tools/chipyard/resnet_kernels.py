#!/usr/bin/env python3
"""Build, run and summarize independent EE290 SoC ResNet kernel payloads."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from run_sim import add_simulator_arguments, positive, run_simulator, safe_path

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from atlas_host_build import wrap
from atlas_host_layout import ATLAS_DRAM_BASE
import kernel_library  # noqa: E402,F401
from atlas_tooling import Admission, sha, write_json

ROOT = Path(__file__).absolute().parents[2]
LAUNCH = Path(__file__).absolute().with_name("kernel_launch.c")
PASS = re.compile(r"^ATLAS_KERNEL_PASS name=(\S+) device_cycles=([0-9]+)$", re.MULTILINE)


def kernel_name(value):
    if not re.fullmatch(r"kernel-[A-Za-z0-9_.-]+", value):
        raise ValueError("invalid kernel name")
    safe_path(Path(value))
    return value


def output_region(initial, size):
    if size <= 0 or size % 8:
        raise ValueError("nonempty 64-bit-aligned expected payload required")
    pattern = b"\xa5" * 256 + b"\xcd" * size + b"\x5a" * 256
    begin = initial.find(pattern)
    if begin < 0 or initial.find(pattern, begin + 1) >= 0 or (begin + 256) % 8:
        raise ValueError("one aligned output region with 256-byte guards required")
    return begin + 256


def build_kernel(kernel, out, compiler, *, max_polls=100000000, wall_seconds=1500):
    kernel, out, compiler = (safe_path(p) for p in (kernel, out, compiler))
    kernel_name(kernel.name)
    if out.exists():
        raise ValueError("fresh kernel output required")
    if not 0 < max_polls <= 100000000:
        raise ValueError("poll bound must be between 1 and 100000000")
    positive(wall_seconds)
    admission = Admission()
    model = kernel / "model"
    program, initial, expected = (admission.read(safe_path(model / name)) for name in
                                 ("program.bin", "initial-dram.bin", "expected-bf16-tiles.bin"))
    wrap(program)  # Admit raw marker ABI; atlas_host_load_raw performs the wrapping.
    offset = output_region(initial, len(expected))
    if len(initial) > 0xffffffff:
        raise ValueError("DRAM payload exceeds the host payload ABI")
    runtime = safe_path(ROOT / "runtime")
    for path in [LAUNCH, compiler] + [runtime / name for name in
                                    ("atlas_host.c", "atlas_host.h", "atlas_host_layout.h")]:
        admission.read(safe_path(path))
    if any(c in str(out) for c in ('"', "\\", "\n", "\r")):
        raise ValueError("unsafe assembly path")
    out.mkdir(parents=True)
    for name, raw in (("program.bin", program), ("initial-dram.bin", initial), ("expected.bin", expected)):
        (out / name).write_bytes(raw)
    (out / "payload.h").write_text(
        "#include <stdint.h>\n"
        f"#define PAYLOAD_NAME {json.dumps(kernel.name)}\n#define PAYLOAD_PROGRAM_WORDS {len(program) // 4}u\n"
        f"#define PAYLOAD_DRAM_BYTES {len(initial)}u\n#define PAYLOAD_OUTPUT_OFFSET {offset}u\n"
        f"#define PAYLOAD_OUTPUT_BYTES {len(expected)}u\n#define PAYLOAD_MAX_POLLS {max_polls}u\n"
        "extern const uint32_t atlas_program[];\nextern const uint8_t atlas_expected[];\n")
    (out / "assets.S").write_text(
        '.section .rodata,"a",@progbits\n'
        f'.balign 64\n.global atlas_program\natlas_program:\n.incbin "{out / "program.bin"}"\n'
        f'.balign 64\n.global atlas_expected\natlas_expected:\n.incbin "{out / "expected.bin"}"\n'
        f'.section .atlas_dram,"aw",@progbits\n.incbin "{out / "initial-dram.bin"}"\n')
    command = [str(compiler), "-std=gnu99", "-O2", "-Wall", "-Wextra", "-Werror", "-fno-common",
               "-fno-builtin-printf", "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany",
               "-specs=htif_nano.specs", "-static", "-T", "htif.ld",
               f"-Wl,--section-start=.atlas_dram={ATLAS_DRAM_BASE:#x}", "-I" + str(out),
               "-I" + str(runtime), str(LAUNCH), str(runtime / "atlas_host.c"), str(out / "assets.S"),
               "-Wl,-Map," + str(out / "host.map"), "-o", str(out / "host.elf")]
    with (out / "build.log").open("w") as log:
        try:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=wall_seconds,
                                  env=dict(os.environ, TMPDIR=str(out))).returncode
        except subprocess.TimeoutExpired:
            code = 124
        except OSError as error:
            log.write(str(error) + "\n")
            code = 127
    bookends = admission.recheck()
    receipt = {"kernel": kernel.name, "command": command, "returncode": code or int(not bookends),
               "source_bookends_equal": bookends, "inputs_sha256": admission.inputs,
               "program_sha256": sha(program), "initial_sha256": sha(initial),
               "expected_sha256": sha(expected), "output_offset": offset,
               "output_bytes": len(expected), "max_polls": max_polls,
               "wall_limit_seconds": wall_seconds, "full_soc_executed": False}
    if not receipt["returncode"]:
        receipt["elf_sha256"] = sha((out / "host.elf").read_bytes())
    write_json(out / "build.json", receipt)
    return receipt


def load_batch(root):
    root = safe_path(root)
    manifest = json.loads(safe_path(root / "batch.json").read_text())
    names = [kernel_name(name) for name in manifest["kernels"]]
    if not names or len(names) != len(set(names)) or manifest.get("kernel_count") != len(names):
        raise ValueError("nonempty unique batch manifest with matching kernel_count required")
    builds = {}
    for name in names:
        receipt = json.loads(safe_path(root / name / "build.json").read_text())
        if receipt.get("kernel") != name or receipt.get("returncode") != 0 or not receipt.get("source_bookends_equal"):
            raise ValueError("successful bound build required: " + name)
        if sha(safe_path(root / name / "host.elf").read_bytes()) != receipt["elf_sha256"]:
            raise ValueError("ELF changed: " + name)
        builds[name] = receipt
    return builds


def kernel_result(name, run, build):
    run = safe_path(run)
    try:
        exit_code = int(safe_path(run / "exit-status.txt").read_text().strip())
        log = safe_path(run / "sim.log").read_text(errors="replace")
        matches = PASS.findall(log)
        named = [int(cycles) for found, cycles in matches if found == name]
        metadata = json.loads(safe_path(run / "run.json").read_text())
        identity_matches = metadata.get("elf_sha256") == build["elf_sha256"]
        status_matches = metadata.get("returncode") == exit_code and metadata.get("timed_out") is False
        exact = (exit_code == 0 and len(matches) == len(named) == 1 and "ATLAS_KERNEL_FAIL" not in log
                 and identity_matches and status_matches)
        return {"kernel": name, "returncode": exit_code, "pass_count": len(matches),
                "matching_pass_count": len(named), "exact": exact,
                "elf_identity_matches": identity_matches,
                "device_cycles": named[0] if len(named) == 1 else None, "run": str(run)}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"kernel": name, "exact": False, "error": str(error), "run": str(run)}


def collect_results(build_root, runs_root):
    builds = load_batch(build_root)
    runs_root = safe_path(runs_root)
    results = [kernel_result(name, runs_root / name, build) for name, build in builds.items()]
    passed = sum(row["exact"] for row in results)
    return {"status": "PASS" if passed == len(builds) else "FAIL", "expected_kernel_count": len(builds),
            "exact_kernel_count": passed, "all_kernels_exact": passed == len(builds), "kernels": results,
            "scope": "Independent kernel output/guard and marker checks; no live full-graph equivalence claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    build = subparsers.add_parser("build", help="link independent kernel ELFs using the shared runtime in place")
    build.add_argument("--network-model", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--cc", default=os.environ.get("RISCV_CC", "riscv64-unknown-elf-gcc"))
    build.add_argument("--max-polls", type=positive, default=100000000)
    build.add_argument("--wall-seconds", type=positive, default=1500)
    build.add_argument("kernels", nargs="*", help="kernel directory names; default: all kernel-* directories")
    run = subparsers.add_parser("run", help="run all kernels from a successful batch build")
    run.add_argument("--build-root", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--jobs", type=positive, default=1)
    run.add_argument("--resume", action="store_true", help="reuse only completed exact runs")
    add_simulator_arguments(run)
    results = subparsers.add_parser("results", help="require zero exit and one matching PASS per expected kernel")
    results.add_argument("--build-root", type=Path, required=True)
    results.add_argument("--runs-root", type=Path, required=True)
    results.add_argument("--out", type=Path, required=True, help="JSON report file")
    args = parser.parse_args()
    if args.action == "build":
        model, out = safe_path(args.network_model), safe_path(args.out)
        if out.exists():
            parser.error("fresh batch output directory required")
        names = args.kernels or sorted(p.name for p in model.iterdir()
                                      if p.name.startswith("kernel-") and kernel_name(p.name) and safe_path(p).is_dir())
        names = [kernel_name(name) for name in names]
        if not names or len(set(names)) != len(names):
            parser.error("one or more unique kernel names required")
        compiler = shutil.which(args.cc)
        if compiler is None:
            parser.error("compiler not found; set --cc or RISCV_CC")
        out.mkdir(parents=True)
        write_json(out / "batch.json", {"kernels": names, "kernel_count": len(names)})
        for name in names:
            receipt = build_kernel(model / name, out / name, compiler, max_polls=args.max_polls,
                                   wall_seconds=args.wall_seconds)
            print(json.dumps(receipt, sort_keys=True))
            if receipt["returncode"]:
                return 1
        return 0
    if args.action == "run":
        if args.simulator is None:
            parser.error("--simulator or ATLAS_SOC_SIMULATOR is required")
        if args.jobs > 64:
            parser.error("--jobs must be at most 64")
        builds = load_batch(args.build_root)
        out, build_root = safe_path(args.out), safe_path(args.build_root)
        if out.exists() and not args.resume:
            parser.error("fresh output required unless --resume is set")
        out.mkdir(parents=True, exist_ok=args.resume)
        pending = []
        for name, receipt in builds.items():
            if safe_path(out / name).exists():
                if not args.resume or not kernel_result(name, out / name, receipt)["exact"]:
                    parser.error("existing run is not exact; choose a fresh output: " + name)
            else:
                pending.append(name)

        def launch(name):
            return run_simulator(args.simulator, build_root / name / "host.elf", out / name,
                                 dramsim_ini=args.dramsim_ini, max_cycles=args.max_cycles,
                                 wall_seconds=args.wall_seconds, verbose=args.verbose, extra_args=args.sim_arg)

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for status in pool.map(launch, pending):
                print(json.dumps(status, sort_keys=True))
        report = collect_results(build_root, out)
        write_json(out / "results.json", report)
    else:
        report = collect_results(args.build_root, args.runs_root)
        out = safe_path(args.out)
        if out.exists():
            parser.error("fresh results JSON path required")
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, report)
    print(json.dumps({key: report[key] for key in ("status", "expected_kernel_count", "exact_kernel_count")}))
    return 0 if report["all_kernels_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
