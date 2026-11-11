"""The diagnostic formatter matches the selected libc's binary64 fixed-six ABI."""

import ctypes
import os
import shutil
import subprocess

import numpy as np
import pytest

from merlin.common.paths import runtime_dir


def test_binary64_fixed6_matches_independent_libc_on_rounding_extrema_and_words(tmp_path):
    cc = os.environ.get("MERLIN_CLANG") or shutil.which("clang")
    if not cc: pytest.skip("existing C compiler required")
    source, library = tmp_path / "format.c", tmp_path / "format.so"
    source.write_text('#include "fixed_decimal.h"\nsize_t format(uint64_t word,char *text) {return merlin_binary64_fixed6(text,word);}\n')
    built = subprocess.run([cc, "-O2", "-shared", "-fPIC", "-I", str(runtime_dir() / "c"), str(source), "-o", str(library)],
                           capture_output=True, timeout=30)
    assert built.returncode == 0, built.stderr.decode()
    actual = ctypes.CDLL(str(library)).format
    actual.argtypes, actual.restype = [ctypes.c_uint64, ctypes.c_void_p], ctypes.c_size_t
    reference = ctypes.CDLL(None).snprintf
    reference.argtypes, reference.restype = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p], ctypes.c_int
    words = [sign | exponent << 52 | fraction for sign in (0, 1 << 63) for exponent in range(2048)
             for fraction in (0, 1, (1 << 52) - 1)]
    words.extend(int(word) for word in np.random.default_rng(17).integers(0, 1 << 64, size=1024, dtype=np.uint64))
    words.extend(int(word) for word in (np.arange(-1023, 1024, 2, dtype=np.float64) / 128).view(np.uint64))
    for word in words:
        observed, expected = ctypes.create_string_buffer(320), ctypes.create_string_buffer(320)
        count = actual(word, observed)
        value = ctypes.c_double.from_buffer_copy(word.to_bytes(8, "little"))
        reference(expected, 320, b"%f", value)
        assert observed.value == expected.value, hex(word)
        assert count == len(expected.value)
