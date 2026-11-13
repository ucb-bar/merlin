"""Compiled exact span admission retains residuals lost by float subtraction."""
import ctypes
from fractions import Fraction
import random
import shutil
import struct
import subprocess

from merlin.compile.c_softmax_mask import render_bf16_row_span_preflight


def test_compiled_encoded_row_span_edges_and_exact_residual(tmp_path):
    helper = render_bf16_row_span_preflight(1, 3, 'check', maximum_lower_bound=-32,
                                           maximum_upper_bound=48, maximum_span=80, masked=True)
    source = tmp_path / 'span.c'
    library = tmp_path / 'span.so'
    source.write_text('#include <stdint.h>\n#include <stddef.h>\n' + helper +
                      '\nint check(const uint8_t *p,const uint8_t *k){return check_row_span(p,k);}\n')
    subprocess.run([shutil.which('cc'), '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-shared', '-fPIC', str(source), '-o', str(library)], check=True, capture_output=True)
    call = ctypes.CDLL(str(library)).check
    call.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    def evaluate(values, active=(1, 1, 1)):
        words = [struct.unpack('<I', struct.pack('<f', value))[0] >> 16 for value in values]
        encoded = struct.pack('<3H', *words)
        keep = struct.pack('<3H', *(0x3f80 if flag else 0 for flag in active))
        decoded = [struct.unpack('<f', struct.pack('<I', code << 16))[0]
                   for code, flag in zip(words, active) if flag]
        expected = bool(decoded) and -32 <= max(decoded) <= 48 and Fraction(max(decoded))-Fraction(min(decoded)) <= 80
        assert (call(encoded, keep) == 0) == expected

    for edge in ([-44.5, 4.53125, -32], [-112, -32, -32], [-112.5, -32, -32],
                 [-32, 48, 0], [-32.25, 48, 0], [-80, 2.0**-119, 0], [-80, -2.0**-119, 0],
                 [48.25, 0, 0], [-32.25]*3, [-0.0, 0, 0]):
        evaluate(edge)
    evaluate([-128, 48, -32], (0, 1, 0))
    evaluate([-32]*3, (0, 0, 0))
    rng = random.Random(17)
    for _ in range(1000):
        words = [rng.randrange(0x400, 0x4301) | (rng.randrange(2) << 15) for _ in range(3)]
        values = [struct.unpack('<f', struct.pack('<I', word << 16))[0] for word in words]
        evaluate(values)
