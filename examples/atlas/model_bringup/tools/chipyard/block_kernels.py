#!/usr/bin/env python3
"""Build and qualify independent, hash-bound raw block kernels on a prebuilt SoC."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

from run_sim import add_simulator_arguments, positive, run_simulator, safe_path

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
sys.path.insert(0, str(Path(__file__).absolute().parents[1] / 'host_iss'))
from atlas_host_build import elf_sections, wrap
import kernel_library  # noqa: E402,F401
from atlas_tooling import Admission, sha, write_json
from firesim_target import package as package_firesim
from arena import add_arguments, audit as audit_dram, relocate, window

ROOT = Path(__file__).absolute().parents[2]
LAUNCH = Path(__file__).absolute().with_name("block_kernel_launch.c")
SCHEMA = "atlas.block_soc_kernels.v1"
PASS = re.compile(r"^ATLAS_KERNEL_PASS name=(\S+) device_cycles=([0-9]+)$", re.MULTILINE)


def name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("invalid kernel ID")
    safe_path(value)
    return value


def integer(value):
    if type(value) is not int:
        raise ValueError("integer required")
    return value


def full_bundle(root, workload):
    """Use the existing checksum/image provider without importing a model."""
    directory = safe_path(ROOT / "tools/host_iss")
    sys.path.insert(0, str(directory))
    modules = {name: ROOT / kernel_library.relative_path(name + ".py") for name in
               ("atlas_host_layout", "atlas_path_guard", "atlas_tooling")}
    modules.update({name: directory / (name + ".py") for name in
                    ("full_bundle", "bundle", "block_bundle")})
    modules["run_sim"] = ROOT / "tools/chipyard/run_sim.py"

    def check_imports():
        for name, expected in modules.items():
            module = sys.modules.get(name)
            if module is not None and safe_path(module.__file__) != expected:
                raise ValueError("full bundle helper imported from another source: " + name)

    check_imports()
    import full_bundle as provider
    check_imports()
    layout = sys.modules["atlas_host_layout"]
    if (safe_path(layout.HEADER) != ROOT / "runtime/atlas_host_layout.h"
            or sha(safe_path(layout.HEADER).read_bytes()) != layout.HEADER_SHA256):
        raise ValueError("imported host layout identity differs")
    root = safe_path(root)
    if safe_path(root / "kernels.json").exists():
        return provider.KernelAssets(root, workload)
    return provider.FullBundle(root, workload)


def prepare_full(bundle_root, workload, out):
    """Adapt the existing fixture-freeze layout to full shared-asset bundles."""
    bundle_root, out = safe_path(bundle_root), safe_path(out)
    if out.exists():
        raise ValueError("fresh preparation directory required")
    sources = [Path(__file__), *(ROOT / "tools/host_iss" / filename for filename in
               ("full_bundle.py", "bundle.py", "block_bundle.py")),
               *(ROOT / kernel_library.relative_path(filename) for filename in
                 ("atlas_path_guard.py", "atlas_tooling.py", "atlas_host_layout.py")),
               ROOT / "tools/chipyard/run_sim.py", ROOT / "runtime/atlas_host_layout.h"]
    pins = {str(safe_path(path)): sha(safe_path(path).read_bytes()) for path in sources}
    bundle = full_bundle(bundle_root, workload)
    out.mkdir(parents=True)
    receipt = dict(status="FAILED", workload=workload, kernel_count=bundle.kernel_count,
                   bundle_root=str(bundle_root), source_sha256=pins,
                   source_metadata=bundle.metadata_name,
                   bundle_sha256sums_sha256=bundle.checksums_sha256,
                   hardware_execution=False, full_host_execution=False)
    try:
        rows = []
        for index in range(bundle.kernel_count):
            descriptor = bundle.kernel_paths(index)
            program = bundle.read(descriptor["original"])
            initial = bundle.initial_dram(index)
            output = bundle.read(descriptor["expected"])
            offset = descriptor["output_offset"]
            expected = bytearray(initial)
            expected[offset:offset + len(output)] = output
            item = name(f"{workload}-{index:05d}")
            directory = out / item
            directory.mkdir()
            row = dict(id=item, dram_base=descriptor["dram_base"],
                       output_ranges=[dict(offset=offset, size=len(output))],
                       source_bundle=str(bundle_root), source_index=index,
                       source_bundle_sha256sums_sha256=bundle.checksums_sha256,
                       source_descriptor=descriptor)
            for key, raw in (("program", program), ("initial_dram", initial), ("expected_dram", expected)):
                path = directory / (key + ".bin")
                path.write_bytes(raw)
                row[key + "_bin"], row[key + "_sha256"] = str(path), sha(raw)
            rows.append(row)
        bundle.verify_bookends()
        if any(sha(safe_path(path).read_bytes()) != digest for path, digest in pins.items()):
            raise ValueError("preparation source changed")
        manifest = out / "manifest.json"
        write_json(manifest, dict(schema=SCHEMA, programs=rows))
        receipt.update(status="PASS_PREPARED_INPUTS_ONLY", source_bookends_equal=True,
                       manifest=str(manifest), manifest_sha256=sha(manifest.read_bytes()))
        return receipt
    except Exception as error:
        receipt["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        write_json(out / "preparation.json", receipt)


def read_manifest(path):
    """Resolve explicitly hashed assets without executing any producer or simulator."""
    path = safe_path(path)
    admission = Admission()
    data = json.loads(admission.read(path))
    rows = data.get("programs")
    if data.get("schema") != SCHEMA or not isinstance(rows, list) or not rows:
        raise ValueError("nonempty block SoC kernel manifest required")
    names = [name(row["id"]) for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("duplicate kernel ID")
    result = []
    for row in rows:
        row = dict(row)
        for key in ("program", "initial_dram", "expected_dram"):
            asset = Path(row[key + "_bin"])
            row[key + "_bin"] = str(safe_path(asset if asset.is_absolute() else path.parent / asset))
            digest = row[key + "_sha256"]
            if not isinstance(digest, str) or not re.fullmatch("[a-f0-9]{64}", digest):
                raise ValueError("SHA256 digest required")
        row["manifest"] = str(path)
        row["manifest_sha256"] = sha(path.read_bytes())
        result.append(row)
    if not admission.recheck():
        raise ValueError("manifest changed during admission")
    return result


def admit(row, admission):
    name(row["id"])
    if sha(admission.read(safe_path(row["manifest"]))) != row["manifest_sha256"]:
        raise ValueError("manifest changed")
    program, initial, expected = (admission.read(safe_path(row[key + "_bin"]), row[key + "_sha256"])
                                  for key in ("program", "initial_dram", "expected_dram"))
    runtime_program = wrap(program)
    for word, in struct.iter_unpack("<I", program):
        if word & 127 == 0x73 and word >> 20 == 0xc01:
            kind, source = (word >> 12) & 7, (word >> 15) & 31
            if kind in (1, 5) or kind and source:
                raise ValueError("raw program writes host retirement handshake")
    base = integer(row["dram_base"])
    if (not initial or len(initial) != len(expected) or len(initial) > 0xffffffff
            or base < 0 or base % 64 or base + len(initial) > 1 << 64):
        raise ValueError("invalid aligned DRAM image")
    ranges = row["output_ranges"]
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("nonempty output ranges required")
    ranges = sorted((integer(item["offset"]), integer(item["size"])) for item in ranges)
    cursor = 0
    for offset, size in ranges:
        if offset < cursor or size <= 0 or offset + size > len(initial):
            raise ValueError("overlapping or out-of-bounds output range")
        if initial[cursor:offset] != expected[cursor:offset]:
            raise ValueError("expected image changes preserved input or guards")
        cursor = offset + size
    if initial[cursor:] != expected[cursor:]:
        raise ValueError("expected image changes preserved input or guards")
    return program, runtime_program, initial, expected, base, ranges


def audit_loadmem(raw, program, initial, expected, base):
    """Check actual ELF embedded assets and LOADMEM bytes after the linker finishes."""
    _, sections, symbols = elf_sections(raw)
    dram = sections[".atlas_dram"]
    if dram[1] != 1 or dram[3] != base or raw[dram[4]:dram[4] + dram[5]] != initial:
        raise ValueError("ELF initial DRAM differs")
    phoff, = struct.unpack_from("<Q", raw, 32)
    phsize, phcount = struct.unpack_from("<HH", raw, 54)
    loads, extents = [], []
    for index in range(phcount):
        kind, flags, offset, va, pa, filesz, memsz, align = struct.unpack_from("<IIQQQQQQ", raw, phoff + index * phsize)
        if kind == 1:
            if va != pa or filesz > memsz or offset + filesz > len(raw) or pa + memsz > 1 << 64:
                raise ValueError("invalid physical ELF load segment")
            if any(pa < start + size and start < pa + memsz for start, size in extents):
                raise ValueError("overlapping ELF load memory ranges")
            extents.append((pa, memsz))
            loads.append((pa, offset, filesz, flags))
    def loaded(address, asset, writable):
        return any(start <= address and address + len(asset) <= start + size
                   and (writable is None or bool(flags & 2) == writable)
                   and raw[offset + address - start:offset + address - start + len(asset)] == asset
                   for start, offset, size, flags in loads)
    if not loaded(base, initial, True):
        raise ValueError("ELF LOADMEM segment differs")
    shoff, = struct.unpack_from("<Q", raw, 40)
    shsize, = struct.unpack_from("<H", raw, 58)
    # The supplied HTIF linker script can put read-only sections inside a writable
    # host segment. Check section flags and loaded bytes; do not claim MMU protection.
    for label, asset in (("atlas_program", program), ("atlas_expected", expected)):
        address, _, index = symbols[label]
        section = struct.unpack_from("<IIQQQQIIQQ", raw, shoff + index * shsize)
        offset = section[4] + address - section[3]
        if section[2] & 1 or raw[offset:offset + len(asset)] != asset or not loaded(address, asset, None):
            raise ValueError("ELF immutable asset differs: " + label)


def build_kernel(row, out, compiler, *, max_polls=100000000, wall_seconds=1500, fast_check=False,
                 target='sim', arena_base=None, dram_base=0x80000000, dram_bytes=64 << 30):
    out, compiler = (safe_path(p) for p in (out, compiler))
    if target not in ('sim', 'firesim'):
        raise ValueError('unknown ELF target')
    if out.exists():
        raise ValueError("fresh kernel output required")
    if not 0 < max_polls <= 100000000:
        raise ValueError("poll bound must be between 1 and 100000000")
    positive(wall_seconds)
    if type(fast_check) is not bool:
        raise ValueError("fast_check must be boolean")
    admission = Admission()
    program, runtime_program, initial, expected, base, ranges = admit(row, admission)
    new_base = base if arena_base is None else arena_base
    window(new_base, len(initial))
    program = relocate(program, base, new_base)
    base, runtime_program = new_base, wrap(program)
    runtime = safe_path(ROOT / "runtime")
    helpers = [Path(__file__).absolute()] + [Path(sys.modules[module].__file__).absolute()
               for module in ("atlas_host_build", "atlas_tooling", "atlas_path_guard", "atlas_host_layout", "run_sim")]
    helpers.extend([Path(__file__).with_name("arena.py"), Path(sys.modules["elf_image"].__file__)])
    for path in [LAUNCH, compiler] + helpers + [runtime / filename for filename in
                                    ("atlas_host.c", "atlas_host.h", "atlas_host_layout.h")]:
        admission.read(safe_path(path))
    if target == 'firesim':
        admission.read(safe_path(Path(__file__).with_name('firesim_target.py')))
    if fast_check:
        admission.read(safe_path(runtime / "atlas_host_check.h"))
    if any(c in str(out) for c in ('"', "\\", "\n", "\r")):
        raise ValueError("unsafe assembly path")
    out.mkdir(parents=True)
    for filename, raw in (("program.bin", program), ("runtime-program.bin", runtime_program),
                          ("initial-dram.bin", initial), ("expected-dram.bin", expected)):
        (out / filename).write_bytes(raw)
    (out / "payload.h").write_text(
        "#include <stdint.h>\n"
        f"#define PAYLOAD_NAME {json.dumps(row['id'])}\n#define PAYLOAD_PROGRAM_WORDS {len(program) // 4}u\n"
        f"#define PAYLOAD_DRAM_BASE UINT64_C({base})\n#define PAYLOAD_DRAM_BYTES {len(initial)}u\n"
        f"#define PAYLOAD_OUTPUT_BYTES {sum(size for _, size in ranges)}u\n"
        f"#define PAYLOAD_OUTPUT_RANGES {len(ranges)}u\n#define PAYLOAD_MAX_POLLS {max_polls}u\n"
        "extern const uint32_t atlas_program[];\nextern const uint8_t atlas_expected[];\n"
        "static const uint32_t atlas_output_ranges[][2] = {" +
        ",".join("{%du,%du}" % item for item in ranges) + "};\n")
    (out / "assets.S").write_text(
        '.section .rodata,"a",@progbits\n'
        f'.balign 64\n.global atlas_program\natlas_program:\n.incbin "{out / "program.bin"}"\n'
        f'.balign 64\n.global atlas_expected\natlas_expected:\n.incbin "{out / "expected-dram.bin"}"\n'
        f'.section .atlas_dram,"aw",@progbits\n.incbin "{out / "initial-dram.bin"}"\n')
    command = [str(compiler), "-std=gnu99", "-O2", "-Wall", "-Wextra", "-Werror", "-fno-common",
               "-fno-builtin-printf", "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany",
               "-specs=htif_nano.specs", "-static", "-T", "htif.ld",
               f"-Wl,--section-start=.atlas_dram={base:#x}", "-I" + str(out), "-I" + str(runtime),
               str(LAUNCH), str(runtime / "atlas_host.c"), str(out / "assets.S"),
               "-Wl,-Map," + str(out / "host.map"), "-o", str(out / "host.elf")]
    if fast_check:
        command.insert(1, "-DATLAS_FAST_CHECK=1")
    error = None
    with (out / "build.log").open("w") as log:
        try:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=wall_seconds,
                                  env=dict(os.environ, TMPDIR=str(out))).returncode
            if code == 0:
                audit_loadmem((out / "host.elf").read_bytes(), program, initial, expected, base)
                audit_dram(out, dram_base, dram_bytes)
        except subprocess.TimeoutExpired:
            code = 124
        except (OSError, ValueError, KeyError, struct.error) as exc:
            code, error = 127, str(exc)
            log.write(error + "\n")
    bookends = admission.recheck()
    if code == 0 and bookends and target == 'firesim':
        try:
            package_firesim(out, row['id'], dram_base, dram_bytes)
        except (OSError, ValueError, KeyError, struct.error) as exc:
            code, error = 127, str(exc)
    receipt = dict(kernel=row["id"], command=command, returncode=code or int(not bookends),
                   source_bookends_equal=bookends, inputs_sha256=admission.inputs,
                   program_sha256=sha(program), runtime_program_sha256=sha(runtime_program),
                   initial_sha256=sha(initial), expected_sha256=sha(expected), output_ranges=ranges,
                   dram_base=base, dram_bytes=len(initial), max_polls=max_polls,
                   wall_limit_seconds=wall_seconds, full_soc_executed=False)
    receipt['target'] = target
    receipt.update(physical_dram_base=dram_base, physical_dram_bytes=dram_bytes)
    if fast_check:
        receipt.update(fast_check=True, mismatch_metric="bytes",
                       check_algorithm="exact-or-of-word-xors-with-byte-fallback")
    if error:
        receipt["error"] = error
    if not receipt["returncode"]:
        receipt["elf_sha256"] = sha((out / "host.elf").read_bytes())
    write_json(out / "build.json", receipt)
    return receipt


def load_batch(root):
    root = safe_path(root)
    manifest = json.loads(safe_path(root / "batch.json").read_text())
    names = [name(value) for value in manifest["kernels"]]
    if (manifest.get("schema") != SCHEMA or not names or len(names) != len(set(names))
            or manifest.get("kernel_count") != len(names)):
        raise ValueError("unique complete batch manifest required")
    builds = {}
    for item in names:
        receipt = json.loads(safe_path(root / item / "build.json").read_text())
        if (receipt.get("kernel") != item or receipt.get("returncode") != 0
                or receipt.get("source_bookends_equal") is not True):
            raise ValueError("successful bound build required: " + item)
        if sha(safe_path(root / item / "host.elf").read_bytes()) != receipt["elf_sha256"]:
            raise ValueError("ELF changed: " + item)
        builds[item] = receipt
    return builds


def kernel_result(item, run, build):
    run = safe_path(run)
    try:
        code = int(safe_path(run / "exit-status.txt").read_text().strip())
        log = safe_path(run / "sim.log").read_text(errors="replace")
        metadata = json.loads(safe_path(run / "run.json").read_text())
        matches = PASS.findall(log)
        phases = re.findall(r"^ATLAS_KERNEL_PHASE (.+)$", log, re.MULTILINE)
        checks = re.findall(r"^ATLAS_KERNEL_CHECK (.+)$", log, re.MULTILINE)
        fields = lambda text: dict(re.findall(r"(\w+)=([^\s]+)", text))
        phase = fields(phases[0]) if len(phases) == 1 else {}
        check = fields(checks[0]) if len(checks) == 1 else {}
        cycles = int(matches[0][1]) if len(matches) == 1 and matches[0][0] == item else None
        metric = "byte" if build.get("fast_check") is True else "unit"
        exact = (code == 0 and metadata.get("returncode") == 0 and metadata.get("timed_out") is False
                 and metadata.get("elf_sha256") == build["elf_sha256"] and cycles is not None
                 and phase.get("result") == phase.get("active") == "0" and phase.get("sequence") == "1"
                 and phase.get("device_cycles") == str(cycles)
                 and check.get("output_" + metric + "_mismatches")
                 == check.get("preserved_" + metric + "_mismatches") == "0"
                 and check.get("dbg0") == check.get("dbg1") == "1" and "ATLAS_KERNEL_FAIL" not in log)
        return dict(kernel=item, exact=exact, returncode=code, device_cycles=cycles, run=str(run))
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(kernel=item, exact=False, error=str(error), run=str(run))


def collect_results(build_root, runs_root):
    runs_root = safe_path(runs_root)
    builds = load_batch(build_root)
    rows = [kernel_result(item, runs_root / item, build) for item, build in builds.items()]
    passed = sum(row["exact"] for row in rows)
    return dict(status="PASS" if passed == len(rows) else "FAIL", expected_kernel_count=len(rows),
                exact_kernel_count=passed, all_kernels_exact=passed == len(rows), kernels=rows,
                device_cycles=sum(row["device_cycles"] for row in rows) if passed == len(rows) else None,
                scope="Independent raw kernel outputs, preserved DRAM and completion; no whole-host claim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("prepare-full")
    prepare.add_argument("--bundle-root", required=True, type=Path)
    prepare.add_argument("--workload", choices=("tiny-full", "smol-full"), required=True)
    prepare.add_argument("--out", required=True, type=Path)
    build = actions.add_parser("build")
    build.add_argument("--manifest", required=True, type=Path)
    build.add_argument("--cc", default=os.environ.get("RISCV_CC", "riscv64-unknown-elf-gcc"))
    build.add_argument("--out", required=True, type=Path)
    build.add_argument("--max-polls", type=positive, default=100000000)
    build.add_argument("--wall-seconds", type=positive, default=1500)
    build.add_argument("--fast-check", action="store_true",
                       help="opt in to exact wide comparison with byte diagnostics")
    build.add_argument('--target', choices=('sim', 'firesim'), default='sim',
                       help='export an HTIF FireSim package with the selected physical DRAM bounds')
    run = actions.add_parser("run")
    run.add_argument("--build-root", required=True, type=Path)
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--jobs", type=positive, default=1)
    run.add_argument("--resume", action="store_true")
    add_simulator_arguments(run)
    results = actions.add_parser("results")
    results.add_argument("--build-root", required=True, type=Path)
    results.add_argument("--runs-root", required=True, type=Path)
    results.add_argument("--out", required=True, type=Path)
    add_arguments(build)
    args = parser.parse_args()
    if args.action == "prepare-full":
        print(json.dumps(prepare_full(args.bundle_root, args.workload, args.out), sort_keys=True))
        return 0
    if args.action == "build":
        rows, out = read_manifest(args.manifest), safe_path(args.out)
        compiler = shutil.which(args.cc)
        if compiler is None or out.exists():
            parser.error("existing compiler and fresh output directory required")
        out.mkdir(parents=True)
        write_json(out / "batch.json", dict(schema=SCHEMA, kernels=[row["id"] for row in rows], kernel_count=len(rows)))
        for row in rows:
            receipt = build_kernel(row, out / row["id"], compiler, max_polls=args.max_polls,
                                   wall_seconds=args.wall_seconds, fast_check=args.fast_check,
                                   target=args.target, arena_base=args.arena_base,
                                   dram_base=args.dram_base, dram_bytes=args.dram_bytes)
            print(json.dumps(receipt, sort_keys=True), flush=True)
            if receipt["returncode"]:
                return 1
        return 0
    if args.action == "run":
        if args.simulator is None or args.jobs > 18:
            parser.error("prebuilt simulator required and jobs must be at most 18")
        builds = load_batch(args.build_root)
        out, build_root = safe_path(args.out), safe_path(args.build_root)
        if out.exists() and not args.resume:
            parser.error("fresh output required unless --resume is set")
        out.mkdir(parents=True, exist_ok=args.resume)
        pending = []
        for item, receipt in builds.items():
            if safe_path(out / item).exists():
                if not args.resume or not kernel_result(item, out / item, receipt)["exact"]:
                    parser.error("existing run is not exact; choose fresh output: " + item)
            else:
                pending.append(item)
        def launch(item):
            return run_simulator(args.simulator, build_root / item / "host.elf", out / item,
                                 dramsim_ini=args.dramsim_ini, max_cycles=args.max_cycles,
                                 wall_seconds=args.wall_seconds, verbose=args.verbose, extra_args=args.sim_arg)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for status in pool.map(launch, pending):
                print(json.dumps(status, sort_keys=True), flush=True)
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
