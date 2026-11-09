#!/usr/bin/env python3
"""Build and run the public single-tile Gemmini self-check on GSIM.

This is a smoke runner, not a replacement certification tier.  It executes a
compiler-emitted ``gemmini_kernel`` object inside a normal bare-metal ELF.  The
ELF checks its own mvout buffer and calls a pass/fail marker; the GSIM harness
observes that committed PC directly.  ``--cross-check-verilator`` runs the exact
same ELF on the configured Verilator oracle before the report may say that the
two simulators agreed.

The GSIM model directory is an explicit input because emitted C++ is a hardware
artifact, not a tool to rediscover by searching a developer's filesystem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

from merlin.runtime.backends import base as backends


HERE = Path(__file__).resolve().parent


class SmokeError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path | None = None, timeout: int = 600) -> tuple[str, float]:
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise SmokeError(
            f"command exited {proc.returncode}: {' '.join(command)}\n"
            f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}")
    return proc.stdout + proc.stderr, elapsed


def _symbol_table(nm: Path, elf: Path) -> dict[str, int]:
    text, _ = _run([str(nm), "-n", str(elf)])
    symbols: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        try:
            address = int(fields[0], 16)
        except ValueError:
            continue
        symbols[fields[-1]] = address
    return symbols


def _parse_record(text: str, prefix: str) -> dict[str, object]:
    lines = [line for line in text.splitlines() if line.startswith(prefix + " ")]
    if len(lines) != 1:
        raise SmokeError(f"expected one {prefix} line, found {len(lines)}")
    result: dict[str, object] = {}
    for field in lines[0].split()[1:]:
        key, separator, value = field.partition("=")
        if not separator or not key:
            raise SmokeError(f"malformed {prefix} field {field!r}")
        if value.startswith("0x"):
            result[key] = int(value, 16)
        else:
            try:
                result[key] = int(value)
            except ValueError:
                result[key] = value
    return result


def _parse_result(text: str) -> tuple[dict[str, object], dict[str, object]]:
    result = _parse_record(text, "GSIM_RESULT")
    required = {"status", "completion", "kernel_seen", "completion_cycle",
                "kernel_to_verdict_cycles", "gemmini_busy_cycles"}
    missing = sorted(required - result.keys())
    if missing:
        raise SmokeError(f"GSIM_RESULT is missing {missing}")
    axi = _parse_record(text, "GSIM_AXI")
    missing_axi = sorted({"ar", "aw", "w", "base", "size"} - axi.keys())
    if missing_axi:
        raise SmokeError(f"GSIM_AXI is missing {missing_axi}")
    if (result["status"] != "pass" or result["completion"] != 1 or
            result["kernel_seen"] != 1 or int(result["gemmini_busy_cycles"]) <= 0):
        raise SmokeError(f"GSIM run did not satisfy the completion contract: {result}")
    if any(int(axi[name]) <= 0 for name in ("ar", "aw", "w")):
        raise SmokeError(f"GSIM run did not exercise AXI read and write traffic: {axi}")
    return result, axi


def _build_selfcheck(kernel_object: Path, work: Path) -> tuple[Path, dict[str, float]]:
    recipe = backends.harness_build_recipe("gemmini")
    elf = work / "gemmini_selfcheck.elf"
    command = recipe.command(
        sources=[HERE / "gemmini_selfcheck.c", kernel_object], output=elf)
    _, elapsed = _run(command, timeout=300)

    gcc = Path(recipe.compiler)
    tool_prefix = gcc.name.removesuffix("gcc")
    objcopy = gcc.with_name(tool_prefix + "objcopy")
    boot_elf = work / "boot_dram.elf"
    boot_bin = work / "boot_dram.bin"
    _, boot_compile_s = _run([
        str(gcc), "-march=rv64gc", "-mabi=lp64d", "-nostdlib", "-nostartfiles",
        "-Wl,-Ttext=0x10000", "-Wl,--build-id=none", str(HERE / "boot_dram.S"),
        "-o", str(boot_elf)])
    _, boot_objcopy_s = _run([
        str(objcopy), "-O", "binary", "-j", ".text", str(boot_elf), str(boot_bin)])
    return elf, {"elf_build_s": elapsed, "boot_build_s": boot_compile_s + boot_objcopy_s}


def _patch_bootrom_model(model_dir: Path, boot_bin: Path, work: Path,
                         cxx: Path) -> tuple[Path, float]:
    """Recompile the emitted model with the DRAM-jump ROM word.

    GSIM lowers the combinational ROM to constant assignments inside ``step``;
    a harness-side store is overwritten before fetch.  Patch only those
    assignments in a generated work copy, leaving the source artifact intact.
    """
    raw = boot_bin.read_bytes()
    raw += b"\0" * ((-len(raw)) % 8)
    words = [struct.unpack_from("<Q", raw, offset)[0]
             for offset in range(0, len(raw), 8)]
    if not words:
        raise SmokeError("boot ROM image is empty")

    source = model_dir / "ChipTop0.cpp"
    patched = work / "ChipTop0.boot.cpp"
    prefix = "system$bootrom_domain$bootrom$rom["
    patched_count = 0
    with source.open(encoding="utf-8") as src, patched.open("w", encoding="utf-8") as dst:
        for line in src:
            position = line.find(prefix)
            if position >= 0:
                suffix = line[position + len(prefix):]
                bracket = suffix.find("]")
                try:
                    index = int(suffix[:bracket]) if bracket >= 0 else -1
                except ValueError:
                    index = -1
                if 0 <= index < len(words):
                    indentation = line[:position]
                    line = f"{indentation}{prefix}{index}] = 0x{words[index]:016x}ull;\n"
                    patched_count += 1
            dst.write(line)
    if patched_count != len(words):
        raise SmokeError(
            f"expected {len(words)} GSIM boot-ROM assignments, patched {patched_count}")

    model_object = work / "ChipTop0.boot.o"
    _, elapsed = _run([
        str(cxx), "-O1", "-std=c++2b", "-DNDEBUG", "-I", str(model_dir),
        "-c", str(patched), "-o", str(model_object)], timeout=300)
    return model_object, elapsed


def _build_gsim_runner(model_dir: Path, chipyard: Path, work: Path,
                       cxx: Path, model_object: Path) -> tuple[Path, float]:
    required = [model_dir / "ChipTop.h", model_dir / "ChipTop0.cpp",
                model_dir / "blackboxes.cpp"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SmokeError(f"GSIM model directory is missing {missing}")
    testchip_csrc = chipyard / "generators/testchipip/src/main/resources/testchipip/csrc"
    mm_cc = testchip_csrc / "mm.cc"
    fesvr_include = chipyard / ".conda-env/riscv-tools/include"
    if not mm_cc.is_file() or not fesvr_include.is_dir():
        raise SmokeError("chipyard testchipip mm.cc or fesvr headers are unavailable")

    runner = work / "gemmini_gsim_selfcheck"
    command = [
        str(cxx), "-O1", "-std=c++2b", "-DNDEBUG",
        "-I", str(model_dir), "-I", str(testchip_csrc), "-I", str(fesvr_include),
        str(model_object), str(model_dir / "blackboxes.cpp"), str(mm_cc),
        str(HERE / "axi_mem_harness.cpp"), str(HERE / "selfcheck_run_harness.cpp"),
        "-o", str(runner),
    ]
    _, elapsed = _run(command, timeout=300)
    return runner, elapsed


def _gsim_build_key(model_dir: Path, chipyard: Path, boot_bin: Path,
                    cxx: Path) -> dict[str, object]:
    testchip_csrc = chipyard / "generators/testchipip/src/main/resources/testchipip/csrc"
    inputs = [model_dir / "ChipTop0.cpp", model_dir / "ChipTop.h",
              model_dir / "blackboxes.cpp", testchip_csrc / "mm.cc",
              testchip_csrc / "mm.h", HERE / "axi_mem_harness.cpp",
              HERE / "selfcheck_run_harness.cpp", boot_bin]
    return {
        "schema_version": 1,
        "cxx": str(cxx),
        "inputs": {str(path): _sha256(path) for path in inputs},
    }


def _tool(name: str, explicit: str | None = None) -> Path:
    candidate = explicit or shutil.which(name)
    if not candidate:
        raise SmokeError(f"required tool {name!r} is unavailable")
    # Preserve an argv[0] spelling such as ``clang++``.  Resolving its symlink to
    # ``clang-21`` makes the driver select C linkage and silently drops libstdc++.
    return Path(candidate).absolute()


def run(args: argparse.Namespace) -> dict[str, object]:
    kernel_object = args.kernel_object.resolve()
    model_dir = args.gsim_model_dir.resolve()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if not kernel_object.is_file():
        raise SmokeError(f"kernel object is absent: {kernel_object}")

    gemmini = backends.get_backend("gemmini")
    chipyard = gemmini.chipyard_root().resolve()
    gcc = gemmini.gcc_path().resolve()
    nm = gcc.with_name(gcc.name.removesuffix("gcc") + "nm")
    cxx = _tool("clang++", args.cxx)

    elf, build_times = _build_selfcheck(kernel_object, work)
    symbols = _symbol_table(nm, elf)
    required_symbols = ("gemmini_kernel", "merlin_gsim_pass_marker",
                        "merlin_gsim_fail_marker")
    missing_symbols = [name for name in required_symbols if name not in symbols]
    if missing_symbols:
        raise SmokeError(f"self-check ELF lacks symbols {missing_symbols}")

    boot_bin = work / "boot_dram.bin"
    model_object = work / "ChipTop0.boot.o"
    runner = work / "gemmini_gsim_selfcheck"
    build_key_path = work / "gsim_build_inputs.json"
    build_key = _gsim_build_key(model_dir, chipyard, boot_bin, cxx)
    prior_key = None
    if build_key_path.is_file():
        try:
            prior_key = json.loads(build_key_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    reused_gsim_build = (prior_key == build_key and model_object.is_file() and runner.is_file())
    if reused_gsim_build:
        model_build_s = runner_build_s = 0.0
    else:
        model_object, model_build_s = _patch_bootrom_model(model_dir, boot_bin, work, cxx)
        runner, runner_build_s = _build_gsim_runner(
            model_dir, chipyard, work, cxx, model_object)
        build_key_path.write_text(
            json.dumps(build_key, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gsim_command = [
        str(runner), str(elf), str(boot_bin), hex(symbols["gemmini_kernel"]),
        hex(symbols["merlin_gsim_pass_marker"]),
        hex(symbols["merlin_gsim_fail_marker"]), str(args.max_cycles),
    ]
    gsim_text, gsim_wall_s = _run(gsim_command, timeout=args.timeout)
    (work / "gsim_console.log").write_text(gsim_text, encoding="utf-8")
    gsim_result, gsim_axi = _parse_result(gsim_text)

    verilator: dict[str, object] = {"requested": args.cross_check_verilator,
                                    "completed": False}
    if args.cross_check_verilator:
        started = time.perf_counter()
        console = gemmini.run_elf(elf, simulator="verilator", timeout=args.timeout)
        verilator_wall_s = time.perf_counter() - started
        (work / "verilator_console.log").write_text(console, encoding="utf-8")
        verilator = {
            "requested": True, "completed": True, "status": "pass",
            "wall_s": round(verilator_wall_s, 6),
            "simulator": str(gemmini.verilator_path().resolve()),
            "simulator_sha256": _sha256(gemmini.verilator_path().resolve()),
            "same_elf_sha256": _sha256(elf),
        }

    speedup = None
    if verilator.get("completed"):
        speedup = float(verilator["wall_s"]) / gsim_wall_s

    model_fir = model_dir / "ChipTop.fir"
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "pass" if gsim_result["status"] == "pass" else "fail",
        "claim_scope": "smoke_only",
        "claim_note": (
            "RTL-derived GSIM execution with an in-ELF numeric self-check. It is not a formal "
            "L3 replacement until the emitted GSIM model is pinned to the certifying RTL revision "
            "and its cycle interpretation is accepted by the experiment contract."),
        "network": {"policy": "unchanged_by_runner", "isolation_claimed": False},
        "kernel": {
            "object": str(kernel_object), "object_sha256": _sha256(kernel_object),
            "elf": str(elf), "elf_sha256": _sha256(elf),
            "symbols": {name: hex(symbols[name]) for name in required_symbols},
            "self_verification": "CPU bit-exact i8xi8-to-i32 golden versus Gemmini mvout",
        },
        "gsim": {
            "kind": "firrtl_to_cpp_rtl_derived", "derived_from_rtl": True,
            "model_dir": str(model_dir),
            "model_fir_sha256": _sha256(model_fir) if model_fir.is_file() else None,
            "model_object_sha256": _sha256(model_object),
            "runner_sha256": _sha256(runner), "result": gsim_result,
            "axi": gsim_axi, "wall_s": round(gsim_wall_s, 6),
        },
        "verilator_cross_check": verilator,
        "steady_state_speedup_vs_verilator": round(speedup, 3) if speedup else None,
        "timing": {**{key: round(value, 6) for key, value in build_times.items()},
                   "gsim_model_rebuild_s": round(model_build_s, 6),
                   "gsim_runner_build_s": round(runner_build_s, 6),
                   "gsim_build_reused": reused_gsim_build},
    }
    report_path = work / "gsim_smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-object", type=Path, required=True,
                        help="compiler-emitted Gemmini kernel object used in the ELF")
    parser.add_argument("--gsim-model-dir", type=Path, required=True,
                        help="directory containing ChipTop.h, ChipTop0.cpp and blackboxes.cpp")
    parser.add_argument("--workdir", type=Path, required=True,
                        help="generated build/report directory under out/")
    parser.add_argument("--cxx", help="clang++ 19+ for linking the emitted GSIM model")
    parser.add_argument("--max-cycles", type=int, default=2_000_000)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--cross-check-verilator", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (SmokeError, subprocess.TimeoutExpired) as exc:
        print(f"GSIM smoke failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
