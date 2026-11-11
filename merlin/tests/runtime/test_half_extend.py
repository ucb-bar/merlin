"""The compiler-rt half extension keeps all source words and the typed ABI."""

import ctypes
import os
import shutil
import subprocess

import numpy as np
import pytest

from merlin.common.paths import runtime_dir


def test_every_binary16_word_matches_the_compiler_runtime_widening_contract(tmp_path):
    cc = os.environ.get("MERLIN_CLANG") or shutil.which("clang")
    if not cc: pytest.skip("existing C compiler required")
    source, library = tmp_path / "extend.c", tmp_path / "extend.so"
    source.write_text('''#include "half_extend.h"
      uint32_t word(uint16_t input) {return merlin_half_extend_word(input);}
      uint32_t typed(uint16_t input) {_Float16 value;__builtin_memcpy(&value,&input,2);
        float output=__extendhfsf2(value);uint32_t result;__builtin_memcpy(&result,&output,4);return result;}''')
    built = subprocess.run([cc, "-O2", "-fno-inline", "-shared", "-fPIC", "-I", str(runtime_dir() / "abi"),
                            str(source), "-o", str(library)], capture_output=True, timeout=30)
    assert built.returncode == 0, built.stderr.decode()
    loaded = ctypes.CDLL(str(library))
    inputs = np.arange(65536, dtype=np.uint16)
    with np.errstate(invalid="ignore"):
        expected = inputs.view(np.float16).astype(np.float32).view(np.uint32)
    # compiler-rt preserves the original NaN payload/signaling word on this
    # widening; NumPy's arithmetic conversion may quiet it.
    nan = ((inputs & 0x7c00) == 0x7c00) & ((inputs & 0x03ff) != 0)
    expected[nan] = ((inputs[nan].astype(np.uint32) & 0x8000) << 16) | 0x7f800000 | ((inputs[nan].astype(np.uint32) & 0x03ff) << 13)
    for function in (loaded.word, loaded.typed):
        function.argtypes, function.restype = [ctypes.c_uint16], ctypes.c_uint32
        actual = np.fromiter((function(int(word)) for word in inputs), dtype=np.uint32, count=65536)
        np.testing.assert_array_equal(actual, expected)
