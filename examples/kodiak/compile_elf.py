#!/usr/bin/env python3
"""Build one Kodiak Zephyr ELF and place it at a stable, user-owned path."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _result(stdout: str) -> dict:
    marker = "\n{\n"
    payload = stdout[stdout.rindex(marker) + 1 :] if marker in stdout else stdout
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("merlin-compile did not return a result object")
    return value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="saved int8 capture name, such as spectformer or tinyllama_l2_s64")
    ap.add_argument("--board", default="chipyard_kodiak", help="Zephyr board; defaults to the physical Kodiak")
    ap.add_argument("--harts", type=int, default=1)
    ap.add_argument("--package", type=Path, required=True, help="selected RVV schedule package directory")
    ap.add_argument("--capture-bundle", type=Path, help="explicit saved capture instead of its registered name")
    ap.add_argument("--output", type=Path, help="ELF filename; default is content-addressed under out/artifacts/zephyr")
    ap.add_argument("--opu", action="store_true", help="route eligible contractions to Kodiak OPU")
    ap.add_argument("--matrix-kernel", help="registered or C-file OPU kernel")
    a = ap.parse_args(argv)
    if a.harts < 1:
        ap.error("--harts must be positive")
    if a.matrix_kernel and not a.opu:
        ap.error("--matrix-kernel requires --opu")
    package = a.package.resolve()
    missing = [n for n in ("manifest.yaml", "knobs.yaml", "schedule.mlir") if not (package / n).is_file()]
    if missing:
        ap.error(f"schedule package {package} is missing {', '.join(missing)}")

    cmd = [
        sys.executable,
        "-m",
        "merlin.compile_cli",
        "--workload",
        a.model,
        "--dtype",
        "int8",
        "--target",
        "rvv",
        "--board",
        a.board,
        "--harts",
        str(a.harts),
        "--package",
        str(package),
        "--run",
        "zephyr",
        "--no-capture",
        "--no-verify",
        "--json",
    ]
    if a.capture_bundle:
        cmd.extend(("--capture-bundle", str(a.capture_bundle.resolve())))
    if a.opu:
        cmd.extend(
            (
                "--matrix-unit",
                "saturn_opu",
                "--matrix-config",
                "KodiakOPUConfig",
                "--matrix-support-target",
                "kodiak",
                "--matrix-gate",
                "build",
            )
        )
        if a.matrix_kernel:
            cmd.extend(("--matrix-kernel", a.matrix_kernel))
    print("[kodiak compile] building one Zephyr ELF; no simulator or FPGA run", flush=True)
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    lines = []
    in_result = False
    for line in proc.stdout:
        lines.append(line)
        if line == "{\n":
            in_result = True
        if not in_result:
            print(line, end="", flush=True)
    rc = proc.wait()
    if rc:
        return rc
    try:
        result = _result("".join(lines))
    except (ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"merlin-compile returned no parseable final result: {exc}") from exc
    if result.get("status") != "compiled" or not result.get("binary"):
        reason = result.get("reason") or result.get("error")
        raise SystemExit(f"expected a compiled Zephyr ELF, got {result.get('status')}: {reason}")
    source = Path(result["binary"])
    if not source.is_file():
        raise SystemExit(f"compiler result is not an ELF file: {source}")
    with source.open("rb") as f:
        magic = f.read(4)
    if magic != b"\x7fELF":
        raise SystemExit(f"compiler result is not an ELF file: {source}")
    digest = _sha256(source)
    if a.output:
        destination = a.output.absolute()
    else:
        backend = "opu" if a.opu else "rvv"
        destination = (
            REPO / "out" / "artifacts" / "zephyr" / (f"{a.model}_int8_h{a.harts}_{backend}_{a.board}_{digest[:12]}.elf")
        )
    if destination.suffix != ".elf" or destination.is_symlink():
        ap.error("--output must be an ordinary .elf filename")
    if destination.exists() and _sha256(destination) != digest:
        raise SystemExit(f"{destination} already exists with different bytes; choose another --output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        try:
            shutil.copy2(source, destination)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
    if _sha256(destination) != digest:
        raise SystemExit(f"copied ELF differs from compiler result: {destination}")
    receipt = {
        "schema": "merlin.kodiak_zephyr_elf.v1",
        "model": a.model,
        "board": a.board,
        "harts": a.harts,
        "backend": "opu" if a.opu else "rvv",
        "package": str(package),
        "package_sha256": (result.get("host_lane") or {}).get("package_sha256"),
        "capture_bundle": result.get("bundle"),
        "elf": str(destination),
        "elf_sha256": digest,
        "build_hash": result.get("build_hash"),
        "routed_contractions": (result.get("matrix") or {}).get("routed_contractions"),
        "routing_report": (result.get("matrix") or {}).get("routing_report"),
        "source_pins_ok": ((result.get("matrix") or {}).get("provenance") or {}).get("all_pins_ok"),
        "verification": "build and instruction audit only; numerical execution belongs to the user",
    }
    destination.with_suffix(".build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"[kodiak compile] ELF: {destination}")
    print(f"[kodiak compile] SHA-256: {digest}")
    print(f"[kodiak compile] receipt: {destination.with_suffix('.build.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
