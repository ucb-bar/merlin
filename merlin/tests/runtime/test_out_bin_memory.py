"""A bounded memory sink retains complete, byte-exact binary output frames."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from merlin.common.paths import runtime_dir
from merlin.runtime.out_bin import parse_binary_console


def test_freestanding_memory_sink_preserves_binary_payload_and_seals_last(tmp_path) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("host C compiler unavailable")
    source = tmp_path / "out_bin_memory.c"
    source.write_text(
        r"""
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "out_bin_memory.h"

int main(void) {
  unsigned char arena[160];
  uint64_t used = UINT64_MAX;
  merlin_out_bin_memory sink;
  const unsigned char raw[] = {0, 255, '\n', 'D', 'O', 'N', 'E', 0};
  const char prefix[] = "OUT_BIN_BEGIN v1 X 1 8 1 u 8\n";
  const char suffix[] = "OUT_BIN_END v1 ";
  const char ending[] = "\nDONE\n";
  uint64_t checksum = UINT64_C(0xcbf29ce484222325);
  for (unsigned i = 0; i < sizeof(raw); ++i)
    checksum = (checksum ^ raw[i]) * UINT64_C(0x100000001b3);
  memset(arena, 0xa5, sizeof(arena));
  merlin_out_bin_memory_init(&sink, arena, sizeof(arena), &used);
  if (used != 0 || !merlin_out_bin_memory_cstr(&sink, prefix) ||
      !merlin_out_bin_memory_append(&sink, raw, sizeof(raw)) ||
      !merlin_out_bin_memory_cstr(&sink, suffix) ||
      !merlin_out_bin_memory_hex16(&sink, checksum) ||
      !merlin_out_bin_memory_cstr(&sink, ending) || used != 0) return 1;
  size_t expected = sizeof(prefix) - 1 + sizeof(raw) + sizeof(suffix) - 1 +
                    16 + sizeof(ending) - 1;
  if (sink.used != expected || !merlin_out_bin_memory_finish(&sink) ||
      used != expected || memcmp(arena, prefix, sizeof(prefix) - 1) ||
      memcmp(arena + sizeof(prefix) - 1, raw, sizeof(raw)) ||
      memcmp(arena + sizeof(prefix) - 1 + sizeof(raw), suffix, sizeof(suffix) - 1) ||
      arena[expected] != 0xa5) return 2;
  if (fwrite(arena, 1, used, stdout) != used) return 3;
  if (merlin_out_bin_memory_finish(&sink) || used != 0) return 4;

  merlin_out_bin_memory_init(&sink, arena, sizeof(arena), &used);
  if (!merlin_out_bin_memory_cstr(&sink, "X") ||
      !merlin_out_bin_memory_finish(&sink) ||
      merlin_out_bin_memory_append(&sink, "Y", 1) || used != 0) return 5;

  merlin_out_bin_memory_init(&sink, arena, sizeof(arena), &used);
  if (merlin_out_bin_memory_finish(&sink) || used != 0) return 6;

  unsigned char tiny[4] = {0xa5, 0xa5, 0xa5, 0xa5};
  merlin_out_bin_memory_init(&sink, tiny, sizeof(tiny), &used);
  if (used != 0 || !merlin_out_bin_memory_append(&sink, "ABCD", 4) ||
      merlin_out_bin_memory_append(&sink, "E", 1) ||
      merlin_out_bin_memory_finish(&sink) || used != 0 ||
      memcmp(tiny, "ABCD", 4)) return 7;

  merlin_out_bin_memory_init(&sink, tiny, sizeof(tiny), &used);
  if (merlin_out_bin_memory_cstr(&sink, "ABCDE") ||
      merlin_out_bin_memory_finish(&sink) || used != 0 || tiny[0] != 'A') return 8;

  merlin_out_bin_memory_init(&sink, tiny, sizeof(tiny), &used);
  if (!merlin_out_bin_memory_append(&sink, "AB", 2) ||
      merlin_out_bin_memory_append(&sink, tiny + 1, 2) ||
      merlin_out_bin_memory_finish(&sink) || used != 0) return 9;

  unsigned char numbers[37];
  merlin_out_bin_memory_init(&sink, numbers, sizeof(numbers), &used);
  if (!merlin_out_bin_memory_decimal(&sink, UINT64_MAX) ||
      !merlin_out_bin_memory_cstr(&sink, " ") ||
      !merlin_out_bin_memory_hex16(&sink, UINT64_MAX) ||
      !merlin_out_bin_memory_finish(&sink) || used != 37 ||
      memcmp(numbers, "18446744073709551615 ffffffffffffffff", 37)) return 10;

  merlin_out_bin_memory_init(&sink, tiny, sizeof(tiny), &used);
  if (merlin_out_bin_memory_append(&sink, 0, 1) ||
      merlin_out_bin_memory_finish(&sink) || used != 0) return 11;

  merlin_out_bin_memory_init(&sink, tiny, sizeof(tiny), &used);
  sink.used = sizeof(tiny) + 1;
  if (merlin_out_bin_memory_finish(&sink) || used != 0) return 12;
  merlin_out_bin_memory_init(&sink, tiny, sizeof(tiny), &used);
  sink.used = sizeof(tiny) + 1;
  if (merlin_out_bin_memory_cstr(&sink, "X") || used != 0) return 13;

  /* A hoisted copy cursor must never overwrite its mutable control object. */
  merlin_out_bin_memory_init(&sink, (unsigned char *)&sink, sizeof(sink), &used);
  if (merlin_out_bin_memory_append(&sink, "X", 1) || used != 0 ||
      merlin_out_bin_memory_finish(&sink)) return 14;
  merlin_out_bin_memory_init(&sink, (unsigned char *)&used, sizeof(used), &used);
  if (merlin_out_bin_memory_append(&sink, "X", 1) || used != 0 ||
      merlin_out_bin_memory_finish(&sink)) return 15;

  /* Reading control bytes is safe when the destination is disjoint. */
  merlin_out_bin_memory_init(&sink, arena, sizeof(arena), &used);
  if (!merlin_out_bin_memory_cstr(&sink, "AB") ||
      !merlin_out_bin_memory_append(&sink, &sink.used, sizeof(sink.used)) ||
      !merlin_out_bin_memory_finish(&sink) || used != 2 + sizeof(sink.used)) return 16;
  size_t original_used;
  memcpy(&original_used, arena + 2, sizeof(original_used));
  if (original_used != 2) return 17;
  return 0;
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "out_bin_memory"
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
    assert outputs == {"X": [[0, 255, 10, 68, 79, 78, 69, 0]]}
    assert metrics == {}
