"""Compare an existing compiler package on existing capsules, same ELF on both engines.

This does not derive capsules, modify the package, or certify untested workloads.
Build provenance and finite numerical agreement are separate retained artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.experiments.phase2 import gsim_certificate as certificate
from merlin.runtime.backends.base import get_backend
from merlin.targetgen import gsim_emulator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--verilator", type=Path, required=True)
    parser.add_argument("--verilator-firrtl", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--reference-timeout", type=int, default=1800)
    parser.add_argument(
        "--serial-timing",
        action="store_true",
        help="also compare cycles with the same serial ELF loader on both engines",
    )
    parser.add_argument(
        "--reference-observations",
        type=Path,
        action="append",
        default=[],
        help="reuse a byte-validated retained same-ELF reference console",
    )
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4:
        parser.error("qualification runs at most four reference or candidate simulators")
    root = args.build_root.resolve(strict=True)
    receipt = json.loads((root / "native/build_receipt.json").read_text())
    binary = root / "native/emulator"
    os.environ["MERLIN_GEMMINI_GSIM_EMU"] = str(binary)
    os.environ["MERLIN_GEMMINI_VERILATOR"] = str(args.verilator.resolve(strict=True))
    os.environ["MERLIN_GSIM_REQUIRE_RECEIPT"] = "1"
    resolution = gsim_emulator.resolve("gemmini", env_var="MERLIN_GEMMINI_GSIM_EMU")
    if not resolution.ok:
        raise RuntimeError(resolution.reason)
    firrtl = Path(receipt["artifacts"]["firrtl"]["path"])
    if sha256_file(firrtl) != sha256_file(args.verilator_firrtl):
        raise RuntimeError("reference and candidate FIRRTL bytes differ")
    artifacts = certificate.ArtifactPaths(
        firrtl,
        args.verilator_firrtl.resolve(strict=True),
        root / "model_manifest.json",
        binary,
        args.verilator.resolve(strict=True),
    )
    # Fail before running any simulator if the sealed build lineage is stale.
    certificate.validate_build_receipt(root / "native/build_receipt.json", pins=artifacts.pinned())
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifests = [path.resolve(strict=True) for path in args.capsule]
    names = [path.parent.name for path in manifests]
    if len(set(names)) != len(names):
        parser.error("capsule directory names must be unique")
    package = args.package.resolve(strict=True)
    backend = get_backend("gemmini")

    def run(manifest: Path) -> dict:
        case = out / manifest.parent.name
        case.mkdir(exist_ok=False)
        interface = manifest.parent / "capsule.interface.mlir"
        command_buffer = case / "command_buffer.json"
        base = [sys.executable, str(package / "gemmini_opt.py"), "--convert-iface-to-gemmini"]
        subprocess.run(
            base + ["--emit-command-buffer=" + str(command_buffer), str(interface)],
            cwd=package,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        lowered = subprocess.run(
            base + ["--emit-target-artifact", str(interface)],
            cwd=package,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        (case / "lowered.llvm.mlir").write_text(lowered.stdout)
        timings = {}

        class ObservedBackend:
            def __getattr__(self, name):
                return getattr(backend, name)

            def run_elf(self, elf, *, simulator, timeout):
                started = time.monotonic()
                console = backend.run_elf(elf, simulator=simulator, timeout=timeout)
                timings[simulator] = time.monotonic() - started
                (case / f"{simulator}.console.log").write_text(console)
                return console

        capture = certificate.capture_case(
            target="gemmini",
            capsule_manifest=manifest,
            artifact_dir=case,
            workdir=case / "execution",
            artifacts=artifacts,
            timeout=args.timeout,
            reference_timeout=args.reference_timeout,
            backend=ObservedBackend(),
        )
        (case / "capture.json").write_text(json.dumps(capture, indent=2, sort_keys=True) + "\n")
        if args.serial_timing:
            # The backend's fast GSIM path uses +loadmem, whereas its reference
            # uses serial loading. Those leave different cache states. Timing
            # agreement requires a separately observed, matched loading policy.
            reference_log = case / "verilator.console.log"
            elf = case / "execution/package_kernel.elf"
            if not reference_log.exists():
                for previous_root in args.reference_observations:
                    previous = previous_root / manifest.parent.name
                    previous_capture = previous / "capture.json"
                    previous_console = previous / "verilator.console.log"
                    if not previous_capture.is_file() or not previous_console.is_file():
                        continue
                    saved = certificate.validate_capture(previous_capture, target="gemmini", pins=artifacts.pinned())
                    if saved["elf_sha256"] == sha256_file(elf) and saved["reference"]["console_sha256"] == sha256_file(
                        previous_console
                    ):
                        shutil.copyfile(previous_console, reference_log)
                        (case / "reference_reuse.json").write_text(
                            json.dumps(
                                {
                                    "capture": str(previous_capture.resolve()),
                                    "capture_sha256": sha256_file(previous_capture),
                                    "console": str(previous_console.resolve()),
                                    "console_sha256": sha256_file(previous_console),
                                    "same_elf_sha256": sha256_file(elf),
                                },
                                indent=2,
                            )
                            + "\n"
                        )
                        break
            if not reference_log.exists():
                ObservedBackend().run_elf(elf, simulator="verilator", timeout=args.reference_timeout)
            reference_outputs, reference_metrics = backend.parse_output(reference_log.read_text())
            prepared = backend.prepare_gsim_command(
                elf,
                expected_elf_sha256=sha256_file(elf),
                expected_engine_provenance=gsim_emulator.citation("gemmini", env_var="MERLIN_GEMMINI_GSIM_EMU"),
            )
            prepared.revalidate()
            argv = list(prepared.emulator_argv)
            load_arg = "+loadmem=" + str(elf)
            if argv.count(load_arg) != 1:
                raise RuntimeError("unexpected target-owned GSIM command/loading policy")
            argv.remove(load_arg)
            started = time.monotonic()
            native = subprocess.run(
                argv, capture_output=True, text=True, timeout=args.timeout, stdin=subprocess.DEVNULL
            )
            timings["gsim_serial"] = time.monotonic() - started
            (case / "gsim_serial.console.log").write_text(native.stdout)
            (case / "gsim_serial.stderr.log").write_text(native.stderr)
            if native.returncode:
                raise RuntimeError(f"serial GSIM exited {native.returncode}: {native.stderr[-2000:]}")
            prepared.revalidate()
            if "GSIM model finished execution." not in native.stderr:
                raise RuntimeError("serial GSIM produced no native RTL/HTIF completion witness")
            observed, metrics = backend.parse_output(native.stdout)
            if observed != reference_outputs or metrics.get("cycles") != reference_metrics.get("cycles"):
                raise RuntimeError(f"serial output/cycle disagreement: GSIM={metrics}; reference={reference_metrics}")
            (case / "serial_timing.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "loader": "serial_tsi",
                        "argv": argv,
                        "same_elf_sha256": sha256_file(elf),
                        "reference_cycles": reference_metrics["cycles"],
                        "candidate_cycles": metrics["cycles"],
                        "reference_console_sha256": sha256_file(reference_log),
                        "candidate_console_sha256": sha256_file(case / "gsim_serial.console.log"),
                        "limits": ["Fast backdoor loading is not a cache-state-equivalent timing comparison."],
                    },
                    indent=2,
                )
                + "\n"
            )
        (case / "timings.json").write_text(json.dumps(timings, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"capsule": manifest.parent.name, "agreement": capture["agreement"]}), flush=True)
        return capture

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        captures = list(pool.map(run, manifests))
    summary = {
        "status": "passed",
        "cases": len(captures),
        "build_receipt_sha256": sha256_file(root / "native/build_receipt.json"),
        "same_selected_firrtl": True,
        "same_elf_per_case": True,
        "output_bytes_match": True,
        "serial_loader_cycles_match": True if args.serial_timing else None,
        "limits": [
            "Only these declared workloads are checked, not all programs or RTL configurations.",
            "The existing compiler package was exercised, not regenerated or modified.",
        ],
    }
    (out / "qualification.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
