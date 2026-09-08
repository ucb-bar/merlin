"""Declared streaming-result ABI for Muon kernels carried by an rv64 SoC ELF.

The Muon console aperture is not connected in the elaborated SoC.  Numeric L3
grading therefore uses a small coherent mailbox: the runner-owned Muon harness
streams at most 32 result words at a time and does not reuse the mailbox until a
generated Rocket carrier acknowledges that sequence.  The carrier compares the
stream with the post-submission expected result and exposes its verdict as one
of two retained PC loops, which the GSIM wrapper reports independently of the
broken Muon console path.

Addresses are never assumed here.  Muon addresses come from the linked ELF;
the SoC address is that symbol plus the offset parsed from the fuse helper that
performs the mapping.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import struct
import subprocess
from typing import Any


RESULT_READY = 0x4D525231   # software ABI token, "MRR1"
RESULT_ACK = 0x4D524131     # software ABI token, "MRA1"
STATUS_SYMBOL = "merlin_result_status"
MAILBOX_SYMBOL = "merlin_result_mailbox"
MAILBOX_WORDS = 32
PASS_SYMBOL = "merlin_numeric_pass"
FAIL_SYMBOL = "merlin_numeric_fail"


def _sequence_token(base: int, sequence: int) -> int:
    """The changing 32-bit publication word for one mailbox transaction."""
    return (int(base) ^ int(sequence)) & 0xFFFFFFFF


def ready_token(sequence: int) -> int:
    return _sequence_token(RESULT_READY, sequence)


def ack_token(sequence: int) -> int:
    return _sequence_token(RESULT_ACK, sequence)


def result_specs(outputs: list[Any]) -> list[dict[str, Any]]:
    """Stable result declarations corresponding to harness output arguments."""
    return [{"name": out.name, "elements": int(out.rows) * int(out.cols), "dtype": out.dtype}
            for out in outputs]


def symbol_addresses(elf: str | Path, names: tuple[str, ...]) -> dict[str, int]:
    """Resolve named ELF symbols structurally with the host ``readelf``."""
    readelf = shutil.which("readelf")
    if readelf is None:
        raise RuntimeError("readelf is required to resolve the declared result page")
    text = subprocess.run([readelf, "-Ws", str(elf)], check=True, capture_output=True,
                          text=True).stdout
    wanted = set(names)
    found: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[-1] in wanted:
            found[fields[-1]] = int(fields[1], 16)
    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"ELF lacks declared result symbol(s): {sorted(missing)}")
    return found


def manifest_from_elf(elf: str | Path, outputs: list[Any], *, soc_offset: int) -> dict[str, Any]:
    """Bind the fixed mailbox, not the full outputs, to fused-SoC addresses."""
    specs = result_specs(outputs)
    names = (STATUS_SYMBOL, MAILBOX_SYMBOL)
    addresses = symbol_addresses(elf, names)
    status_local = addresses[STATUS_SYMBOL]
    mailbox_local = addresses[MAILBOX_SYMBOL]
    return {
        "schema": "merlin.muon-result-mailbox.v2",
        "status": {"symbol": STATUS_SYMBOL, "muon_address": status_local,
                   "soc_address": status_local + int(soc_offset)},
        "mailbox": {"symbol": MAILBOX_SYMBOL, "muon_address": mailbox_local,
                    "soc_address": mailbox_local + int(soc_offset),
                    "words": MAILBOX_WORDS},
        # Output declarations carry shape/type only.  Their backing buffers are
        # private to the Muon harness and are never read by another bus master.
        "outputs": [{k: spec[k] for k in ("name", "elements", "dtype")} for spec in specs],
        "soc_fuse_offset": int(soc_offset),
    }


def _flat(value: Any) -> list[Any]:
    if isinstance(value, dict) and "values" in value:
        return _flat(value["values"])
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for item in value:
            out.extend(_flat(item))
        return out
    return [value]


def _f32_bits(value: float) -> int:
    v = float(value)
    try:
        return struct.unpack("<I", struct.pack("<f", v))[0]
    except OverflowError:
        return 0x7F800000 if v > 0 else 0xFF800000


def _hex_words(words: list[int], *, indent: str = "  ") -> str:
    rows = []
    for start in range(0, len(words), 8):
        rows.append(indent + ", ".join(f"0x{x:08x}u" for x in words[start:start + 8]) + ",")
    return "\n".join(rows)


def render_carrier(manifest: dict[str, Any], expected: dict[str, Any], policy: dict | None) -> str:
    """Generate the integer-only Rocket carrier for a declared result manifest.

    Floating comparisons are converted to inclusive IEEE-f32 interval bounds on
    the host.  Rocket compares monotonic integer keys, avoiding scalar floating
    instructions unsupported by some SoC carrier configurations.
    """
    compare = str((policy or {}).get("compare", "exact_int"))
    atol = float((policy or {}).get("atol", 1e-3))
    rtol = float((policy or {}).get("rtol", 0.0))
    outputs = list(manifest.get("outputs") or [])
    if not outputs:
        raise ValueError("result manifest declares no outputs")

    arrays: list[str] = []
    checks: list[str] = []
    total = 0
    for index, spec in enumerate(outputs):
        name = str(spec["name"])
        if name not in expected:
            raise ValueError(f"expected result has no declared output {name!r}")
        values = _flat(expected[name])
        count = int(spec["elements"])
        if len(values) != count:
            raise ValueError(
                f"expected output {name!r} has {len(values)} elements, manifest declares {count}")
        total += count
        offset = total - count
        if compare in ("exact_int", "exact") and str(spec.get("dtype")) == "i32":
            words = [int(v) & 0xFFFFFFFF for v in values]
            arrays.append(f"static const uint32_t expected_{index}[{count}] = {{\n"
                          f"{_hex_words(words)}\n}};")
            checks.append(
                f"  if (index >= {offset}u && index < {offset + count}u)\n"
                f"    return got != expected_{index}[index - {offset}u];")
            continue

        # Float outputs (including exact float) are graded as an interval.  Exact
        # means a zero-width interval after rounding the expected value to f32.
        lower: list[int] = []
        upper: list[int] = []
        for value in values:
            want = float(value)
            tol = 0.0 if compare in ("exact_int", "exact") else atol + rtol * abs(want)
            lower.append(_f32_bits(want - tol))
            upper.append(_f32_bits(want + tol))
        arrays.append(f"static const uint32_t lower_{index}[{count}] = {{\n"
                      f"{_hex_words(lower)}\n}};")
        arrays.append(f"static const uint32_t upper_{index}[{count}] = {{\n"
                      f"{_hex_words(upper)}\n}};")
        checks.append(
            f"  if (index >= {offset}u && index < {offset + count}u) {{\n"
            f"    uint32_t key = ordered_f32(got);\n"
            f"    uint32_t nan = ((got & 0x7f800000u) == 0x7f800000u) && (got & 0x007fffffu);\n"
            f"    return nan || key < ordered_f32(lower_{index}[index - {offset}u]) "
            f"|| key > ordered_f32(upper_{index}[index - {offset}u]);\n  }}")

    status = int((manifest.get("status") or {})["soc_address"])
    mailbox = manifest.get("mailbox") or {}
    if int(mailbox.get("words", 0)) != MAILBOX_WORDS:
        raise ValueError(f"result manifest must declare a {MAILBOX_WORDS}-word mailbox")
    mailbox_address = int(mailbox["soc_address"])
    return f"""/* Generated from merlin.muon-result-mailbox.v2; do not hand-edit. */
#include <stdint.h>
#define STATUS ((volatile uint32_t *)0x{status:x}ULL)
#define MAILBOX ((volatile uint32_t *)0x{mailbox_address:x}ULL)
#define MERLIN_RESULT_READY(sequence) (0x{RESULT_READY:08x}u ^ (sequence))
#define MERLIN_RESULT_ACK(sequence) (0x{RESULT_ACK:08x}u ^ (sequence))
#define MERLIN_MAILBOX_WORDS {MAILBOX_WORDS}u
{chr(10).join(arrays)}

static uint32_t ordered_f32(uint32_t bits) {{
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}}

static uint32_t mismatch(uint32_t index, uint32_t got) {{
{chr(10).join(checks)}
  return 1u;
}}

__attribute__((noreturn, noinline, aligned(64))) static void pass_loop(void) {{
  __asm__ volatile(".globl {PASS_SYMBOL}\\n{PASS_SYMBOL}:\\nwfi\\nj {PASS_SYMBOL}");
  __builtin_unreachable();
}}
__attribute__((noreturn, noinline, aligned(64))) static void fail_loop(void) {{
  __asm__ volatile(".globl {FAIL_SYMBOL}\\n{FAIL_SYMBOL}:\\nwfi\\nj {FAIL_SYMBOL}");
  __builtin_unreachable();
}}

int main(void) {{
  uint32_t bad = 0u;
  uint32_t checksum = 2166136261u;
  uint32_t received = 0u;
  uint32_t sequence = 1u;
  while (received < {total}u) {{
    while (STATUS[0] != MERLIN_RESULT_READY(sequence))
      __asm__ volatile("fence r,r" ::: "memory");
    /* Acquire the count and mailbox payload published before READY(sequence). */
    __asm__ volatile("fence r,rw" ::: "memory");
    uint32_t count = STATUS[1];
    if (count == 0u || count > MERLIN_MAILBOX_WORDS || count > {total}u - received) {{
      /* Order the observed payload before returning ownership to Muon. */
      __asm__ volatile("fence rw,rw" ::: "memory");
      STATUS[2] = MERLIN_RESULT_ACK(sequence);
      fail_loop();
    }}
    for (uint32_t i = 0; i < count; ++i) {{
      uint32_t got = MAILBOX[i];
      checksum = (checksum ^ got) * 16777619u;
      bad += mismatch(received + i, got);
    }}
    received += count;
    /* All mailbox reads happen before the changing ACK publication word. */
    __asm__ volatile("fence rw,rw" ::: "memory");
    STATUS[2] = MERLIN_RESULT_ACK(sequence);
    sequence++;
  }}
  STATUS[5] = bad;
  STATUS[6] = checksum;
  __asm__ volatile("fence rw,rw" ::: "memory");
  if (bad == 0) pass_loop();
  fail_loop();
}}
"""


def _final_rocket_pc(console: str) -> int | None:
    marker = "[gsim-probe final] rocket_pc="
    for line in reversed(console.splitlines()):
        if marker in line:
            token = line.split(marker, 1)[1].split(maxsplit=1)[0]
            try:
                return int(token, 16)
            except ValueError:
                return None
    return None


def outcome_from_console(console: str, symbols: dict[str, int]) -> str | None:
    """Return ``pass``/``fail`` only for an exact final-PC outcome witness."""
    pc = _final_rocket_pc(console)
    if pc is None:
        return None
    if pc == int(symbols[PASS_SYMBOL]):
        return "pass"
    if pc == int(symbols[FAIL_SYMBOL]):
        return "fail"
    return None
