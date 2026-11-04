"""Fixture-independent ctypes declarations for Smol host test controls."""
import ctypes as ct
import subprocess


FIELDS = ('normalized', 'query', 'key', 'value', 'rotary_query', 'rotary_key',
          'joined_key', 'joined_value', 'attention', 'projected', 'residual',
          'postnorm', 'gate', 'up', 'activated', 'down', 'output',
          'bf16_probabilities', 'scores', 'float_probabilities', 'workspace',
          'cross_input', 'cross_key', 'cross_value')


class Parameters(ct.Structure):
    _fields_ = [('kind', ct.c_uint), ('input_weight', ct.c_void_p), ('post_weight', ct.c_void_p),
                ('input_epsilon', ct.c_float), ('post_epsilon', ct.c_float),
                ('cosine', ct.c_void_p), ('sine', ct.c_void_p),
                ('first_table_position', ct.c_int64), ('table_rows', ct.c_size_t)]


class Frame(ct.Structure):
    _fields_ = [(name, ct.c_void_p) for name in FIELDS]


class State(ct.Structure):
    _fields_ = [(name, ct.c_uint) for name in ('started', 'projections', 'complete', 'error')]


Callback = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_uint, ct.c_uint, ct.c_void_p,
                        ct.c_size_t, ct.c_size_t, ct.c_uint, ct.c_void_p, ct.c_size_t)


class VisionLayer(ct.Structure):
    _fields_ = [(name, ct.c_void_p) for name in ('norm1_weight', 'norm1_bias', 'norm2_weight', 'norm2_bias')]


class VisionParameters(ct.Structure):
    _fields_ = [('position', ct.c_void_p), ('layers', VisionLayer * 12),
                ('post_weight', ct.c_void_p), ('post_bias', ct.c_void_p), ('gelu', ct.c_void_p),
                ('epsilon', ct.c_float)]


class VisionFrame(ct.Structure):
    _fields_ = [(name, ct.c_void_p) for name in ('tokens', 'normalized', 'query', 'key', 'value',
                'attended', 'projected', 'large', 'activated', 'shuffle', 'output')]


VisionCallback = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_uint, ct.c_uint,
                             ct.c_void_p, ct.c_size_t, ct.c_size_t, ct.c_void_p, ct.c_size_t)


def load_shared(output, sources, *, flags=(), compiler='/usr/bin/cc', math=True, mode=ct.DEFAULT_MODE, **options):
    subprocess.run([compiler, '-O2', '-fPIC', '-shared', '-ffp-contract=off', '-fno-fast-math',
                    *flags, *map(str, sources), *(['-lm'] if math else []), '-o', str(output)],
                   check=True, **options)
    return ct.CDLL(str(output), mode=mode)
