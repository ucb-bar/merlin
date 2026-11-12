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


def test_binary16_narrowing_checks_every_rounding_boundary_and_typed_abi(tmp_path):
    cc = os.environ.get("MERLIN_CLANG") or shutil.which("clang")
    if not cc: pytest.skip("existing C compiler required")
    source, library = tmp_path / "narrow.c", tmp_path / "narrow.so"
    source.write_text('''#include "half_extend.h"
      uint16_t word(uint32_t input) {return merlin_half_trunc_word(input);}
      uint16_t typed(uint32_t input) {float value;__builtin_memcpy(&value,&input,4);
        _Float16 output=__truncsfhf2(value);uint16_t result;__builtin_memcpy(&result,&output,2);return result;}''')
    built = subprocess.run([cc, "-O2", "-fno-inline", "-shared", "-fPIC", "-I", str(runtime_dir() / "abi"),
                            str(source), "-o", str(library)], capture_output=True, timeout=30)
    assert built.returncode == 0, built.stderr.decode()
    loaded = ctypes.CDLL(str(library))
    # Adjacent positive half values, extended exactly to binary32. Each midpoint
    # and its immediately adjacent binary32 words select opposite/even half words.
    lower = np.arange(0x7c00, dtype=np.uint16)
    left = lower.view(np.float16).astype(np.float32)
    right = (lower + np.uint16(1)).view(np.float16).astype(np.float32)
    right[-1] = 65536  # The virtual next binade supplies the overflow midpoint.
    middle = ((left + right) / np.float32(2)).view(np.uint32)
    inputs = np.stack((middle - 1, middle, middle + 1), axis=1).ravel()
    expected = np.stack((lower, lower + (lower & 1), lower + 1), axis=1).ravel().astype(np.uint16)
    inputs = np.concatenate((inputs, inputs | np.uint32(0x80000000)))
    expected = np.concatenate((expected, expected | np.uint16(0x8000)))
    originals = np.arange(65536, dtype=np.uint16)
    exact = originals.view(np.float16).astype(np.float32).view(np.uint32)
    nan = (originals & 0x7c00 == 0x7c00) & (originals & 0x03ff != 0)
    exact[nan] = ((originals[nan].astype(np.uint32) & 0x8000) << 16) | 0x7f800000 | ((originals[nan].astype(np.uint32) & 0x03ff) << 13)
    roundtrip = originals.copy()
    roundtrip[nan] |= np.uint16(0x0200)
    inputs = np.concatenate((inputs, exact, np.array([1, 0x80000001, 0x7f800001, 0xff800001], np.uint32)))
    expected = np.concatenate((expected, roundtrip, np.array([0, 0x8000, 0x7e00, 0xfe00], np.uint16)))
    for function in (loaded.word, loaded.typed):
        function.argtypes, function.restype = [ctypes.c_uint32], ctypes.c_uint16
        actual = np.fromiter((function(int(word)) for word in inputs), dtype=np.uint16, count=len(inputs))
        np.testing.assert_array_equal(actual, expected)
