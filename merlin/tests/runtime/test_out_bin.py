"""Lossless binary readback keeps arbitrary output bytes out of text decoding."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from merlin.common.paths import runtime_dir
from merlin.runtime.out_bin import OutBinError, parse_binary_console, restore_onlcr_console


def _frame(payload: bytes, *, name: str = "out") -> bytes:
    checksum = 0xCBF29CE484222325
    for byte in payload:
        checksum = ((checksum ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    return (
        f"METRIC cycles 7\nOUT_BIN_BEGIN v1 {name} 1 {len(payload)} 1 u {len(payload)}\n".encode()
        + payload
        + f"OUT_BIN_END v1 {checksum:016x}\nDONE\n".encode()
    )


def test_raw_frame_roundtrips_nul_newline_magic_and_non_utf8() -> None:
    payload = b"\x00\xff\nOUT_BIN_END\x00DONE\n"
    outputs, metrics = parse_binary_console(_frame(payload))
    assert outputs == {"out": [list(payload)]}
    assert metrics == {"cycles": 7}


def test_explicit_terminal_restoration_preserves_all_bytes_and_original_crlf():
    payload = bytes(range(256)) + b"\r\n\r\r\n\nOUT_BIN_END\x00"
    original = _frame(payload)
    transcript = original.replace(b"\n", b"\r\n")
    with pytest.raises(OutBinError):
        parse_binary_console(transcript)
    assert restore_onlcr_console(transcript) == original
    assert parse_binary_console(restore_onlcr_console(transcript))[0]['out'] == [list(payload)]


def test_terminal_restoration_cannot_repair_a_damaged_payload():
    transcript = _frame(b"original\r\npayload").replace(b"\n", b"\r\n")
    with pytest.raises(OutBinError, match="checksum"):
        restore_onlcr_console(transcript.replace(b"payload", b"damaged"))


def test_real_freestanding_packer_roundtrips_all_container_widths(tmp_path) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("host C compiler unavailable")
    source = tmp_path / "out_bin.c"
    source.write_text(
        """
#include <stdint.h>
#include <stdio.h>
#include "out_bin.h"
static int sink(const void *data, size_t length) {
  return fwrite(data, 1, length, stdout) == length;
}
static uint64_t mutable_values[1025];
static unsigned flushes;
static int mutating_sink(const void *data, size_t length) {
  (void)data;
  if (length != 1024 || ++flushes != 1) return 0;
  mutable_values[1024] = 256; /* A real callback may change not-yet-read source bytes. */
  return 1;
}
static int output(const char *name, const uint64_t *words, unsigned count,
                  unsigned width, int signed_words) {
  merlin_out_bin out;
  printf("OUT_BIN_BEGIN v1 %s 1 %u %u %c %u\\n", name, count, width,
         signed_words ? 's' : 'u', count * width);
  merlin_out_bin_init(&out, count, width, signed_words, sink);
  for (unsigned i = 0; i < count; ++i)
    if (!merlin_out_bin_word(&out, words[i])) return 0;
  if (!merlin_out_bin_finish(&out)) return 0;
  printf("OUT_BIN_END v1 %016llx\\n", (unsigned long long)out.checksum);
  return 1;
}
int main(void) {
  uint64_t a[] = {0, 255, 10, 0};
  uint64_t b[] = {(uint64_t)(int64_t)-129, 0, 32767};
  uint64_t c[] = {UINT32_C(0x80000000), UINT32_C(0x7fc12345)};
  uint64_t d[] = {UINT64_MAX, 0};
  if (!output("A", a, 4, 1, 0) || !output("B", b, 3, 2, 1) ||
      !output("C", c, 2, 4, 0) || !output("D", d, 2, 8, 0)) return 2;
  merlin_out_bin bad;
  merlin_out_bin_init(&bad, 1, 1, 0, sink);
  if (merlin_out_bin_word(&bad, 256) || merlin_out_bin_finish(&bad)) return 3;
  merlin_out_bin_init(&bad, 2, 1, 0, sink);
  if (!merlin_out_bin_word(&bad, 1) || merlin_out_bin_finish(&bad)) return 4;
  merlin_out_bin_init(&bad, 1025, 1, 0, mutating_sink);
  for (unsigned i = 0; i < 1025; ++i) {
    if (!merlin_out_bin_word(&bad, mutable_values[i])) {
      if (i != 1024 || bad.seen_words != 1024 || bad.raw_len != 0 ||
          flushes != 1 || merlin_out_bin_finish(&bad)) return 5;
      break;
    }
    if (i == 1024) return 6; /* The mutated actual value must not be narrowed. */
  }
  puts("DONE");
  return 0;
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "out_bin"
    subprocess.run(
        [
            compiler,
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(runtime_dir() / "baremetal"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )
    raw = subprocess.run([str(executable)], check=True, capture_output=True).stdout
    outputs, metrics = parse_binary_console(raw)
    assert metrics == {}
    assert outputs == {
        "A": [[0, 255, 10, 0]],
        "B": [[-129, 0, 32767]],
        "C": [[0x80000000, 0x7FC12345]],
        "D": [[(1 << 64) - 1, 0]],
    }


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda raw: raw[:-5], "DONE"),
        (lambda raw: raw.replace(b"\xff", b"\xfe", 1), "checksum"),
        (lambda raw: raw.replace(b" 1 u ", b" 2 u ", 1), "count"),
        (lambda raw: b"DONE\n" + raw, "DONE"),
        (lambda raw: raw + _frame(b"x"), "DONE|duplicate"),
        (lambda raw: raw[:-5] + _frame(b"x"), "duplicate"),
        (lambda raw: raw.replace(b"OUT_BIN_END v1 ", b"OUT_BIN_END v2 ", 1), "END"),
        (lambda raw: raw.replace(b"OUT_BIN_BEGIN v1 ", b"OUT_BIN_BEGIN v2 ", 1), "BEGIN"),
        (lambda raw: b"OUTSUM out 1 1 0000000000000000\n" + raw, "mixed"),
        (lambda raw: b"OUT out 1 1 0\n" + raw, "mixed"),
        (lambda raw: b"OUT_ND out 1 1 0\n" + raw, "mixed"),
    ],
)
def test_raw_frame_refuses_damage(damage, message) -> None:
    raw = _frame(b"\x00\xff\nOUT_BIN_END\x00DONE\n")
    with pytest.raises(ValueError, match=message):
        parse_binary_console(damage(raw))
