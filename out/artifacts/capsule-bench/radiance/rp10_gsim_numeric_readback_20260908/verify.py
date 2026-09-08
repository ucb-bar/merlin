#!/usr/bin/env python3
from hashlib import sha256
from pathlib import Path
import json
import re

HERE = Path(__file__).resolve().parent
receipt = json.loads((HERE / "receipt.json").read_text())

def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()

checks = {
    HERE / "readback_carrier.c": receipt["instrument"]["readback_carrier_sha256"],
    HERE / "build_readback.py": receipt["instrument"]["build_script_sha256"],
    HERE / "build/readback_main.c": receipt["instrument"]["generated_readback_main_sha256"],
    HERE / "build/rp10.readback.radiance.elf": receipt["binaries"]["muon_elf_sha256"],
    HERE / "build/rp10.readback.soc.elf": receipt["binaries"]["soc_elf_sha256"],
    HERE / "build/readback_symbols.txt": receipt["binaries"]["symbol_manifest_sha256"],
    HERE / "gsim_bounded_console.log": receipt["run"]["console_sha256"],
    Path(receipt["emulator"]["path"]): receipt["emulator"]["sha256"],
}
for path, expected in checks.items():
    actual = digest(path)
    assert actual == expected, f"hash mismatch: {path}: {actual} != {expected}"

symbols = (HERE / "build/readback_symbols.txt").read_text()
assert "rp10_numeric_pass 0x0000000080000086" in symbols
assert "rp10_numeric_fail 0x00000000800000c6" in symbols

log = (HERE / "gsim_bounded_console.log").read_text(errors="replace")
final = re.search(r"\[gsim-probe final\] rocket_pc=(0x[0-9a-f]+)", log)
assert final and final.group(1) == "0x80000086", final.group(1) if final else "no final PC"
assert "rocket_pc=0x800000c6" not in log
assert "FINISHED: cycles=120000" in log
assert "dram_aw=1 dram_w=4" in log
assert "last_aw=0x110010060" in log
assert "Exit status: 0" in log
print("PASS: RP10 32/32 tolerance-bounded readback reached rp10_numeric_pass")
