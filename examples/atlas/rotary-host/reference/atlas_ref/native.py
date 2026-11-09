"""Optional independent C99 MXU acceleration; NumPy remains the default engine.

The library is compiled only into this job's tmp directory and loaded via
ctypes, so Python 3.12 and 3.14 share the same ordinary C ABI. No model or torch
code participates in native computation. Callers must explicitly select this
engine after qualifying it against the authoritative NumPy implementation.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


_LIBRARY = None
_METADATA = None
_SOURCE = Path(__file__).with_suffix('.c')
_TMP = Path(__file__).resolve().parent.parent / 'tmp'


def _load():
    global _LIBRARY, _METADATA
    if _LIBRARY is not None:
        return _LIBRARY
    source_bytes = _SOURCE.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    # Include compiler identity in the cache key. All subprocesses inherit the
    # job TMPDIR, and the compiler itself runs at nice level 10.
    environment = dict(os.environ, TMPDIR=str(_TMP))
    _TMP.mkdir(parents=True, exist_ok=True)
    compiler = '/usr/bin/cc'
    compiler_version = subprocess.check_output([compiler, '--version'], env=environment, text=True).splitlines()[0]
    options = ['-O3', '-std=c99', '-fPIC', '-shared', '-fopenmp']
    build_identity = json.dumps({'source_sha256': source_hash, 'compiler': compiler_version,
                                 'options': options}, sort_keys=True).encode()
    key = hashlib.sha256(build_identity).hexdigest()
    library_path = _TMP / ('atlas-native-' + key[:24] + '.so')
    if not library_path.exists():
        fd, temporary = tempfile.mkstemp(prefix='atlas-native-build-', suffix='.so', dir=_TMP)
        os.close(fd)
        try:
            result = subprocess.run(['nice', '-n', '10', compiler, *options, str(_SOURCE), '-o', temporary],
                                    env=environment, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError('native C99 build failed: ' + result.stderr)
            os.replace(temporary, library_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    library = ctypes.CDLL(str(library_path))
    u8p, u16p = ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint16)
    library.atlas_matmul_codes.argtypes = [u8p, u8p, u16p, u16p,
                                         ctypes.c_size_t, ctypes.c_size_t,
                                         ctypes.c_size_t, ctypes.c_size_t,
                                         ctypes.c_int, ctypes.c_int]
    library.atlas_matmul_codes.restype = ctypes.c_int
    library.atlas_native_openmp.restype = ctypes.c_int
    _METADATA = {'source_sha256': source_hash, 'compiler': compiler_version,
                 'options': options, 'library_sha256': hashlib.sha256(library_path.read_bytes()).hexdigest(),
                 'openmp': bool(library.atlas_native_openmp()), 'default_threads': 1,
                 'maximum_threads': 8}
    _LIBRARY = library
    return library


def backend_metadata():
    """Build/load the local library and report source/compiler/library identity."""
    _load()
    return dict(_METADATA)


def _raw(value, width):
    value = np.asarray(value)
    if value.dtype.kind not in 'ui' or np.any(value < 0) or np.any(value > (1 << width) - 1):
        raise ValueError(f'expected unsigned {width}-bit integer encodings')
    return value.astype(np.uint8 if width == 8 else np.uint16, copy=False)


def matmul_codes(a, b, unit='mxu0', *, initial=None, chunk_size=32, threads=1):
    """Native version of arithmetic.matmul_codes, explicitly opt in per call.

    Operands are [..., M, K] and [..., K, N]. All FP8 bytes and raw BF16
    initial encodings follow the custom MXU integer datapaths. Output has raw
    uint16 BF16 bits. Threads parallelize independent output cells only.
    """
    if unit not in ('mxu0', 'mxu1'):
        raise ValueError('unit must be mxu0 or mxu1')
    if chunk_size != 32:
        raise ValueError('only the pinned 32-lane instruction schedule is supported')
    threads = operator.index(threads)
    if not 1 <= threads <= 8:
        raise ValueError('native threads must lie in [1, 8]')
    a, b = _raw(a, 8), _raw(b, 8)
    if a.ndim < 2 or b.ndim < 2 or a.shape[-1] != b.shape[-2]:
        raise ValueError('matmul_codes requires [..., M, K] and [..., K, N]')
    batch = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
    batches = math.prod(batch)
    if max(batches * m * n, batches * m * k, batches * n * k) > np.iinfo(np.intp).max:
        raise ValueError('native operand shape exceeds addressable size')
    aa = np.ascontiguousarray(np.broadcast_to(a, batch + (m, k)))
    # Weight K becomes contiguous, retaining logical K order and code identity.
    bb = np.ascontiguousarray(np.swapaxes(np.broadcast_to(b, batch + (k, n)), -1, -2))
    shape = batch + (m, n)
    output = np.empty(shape, dtype=np.uint16)
    incoming = None if initial is None else np.require(
        np.broadcast_to(_raw(initial, 16), shape), requirements=['C', 'A'])
    library = _load()
    u8p, u16p = ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint16)
    error = library.atlas_matmul_codes(aa.ctypes.data_as(u8p), bb.ctypes.data_as(u8p),
                                       None if incoming is None else incoming.ctypes.data_as(u16p),
                                       output.ctypes.data_as(u16p), batches, m, k, n,
                                       0 if unit == 'mxu0' else 1, threads)
    if error:
        raise RuntimeError(f'native integer contraction failed: {error}')
    return output
