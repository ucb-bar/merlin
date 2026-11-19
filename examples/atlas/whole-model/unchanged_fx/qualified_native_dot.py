"""Explicitly qualified host execution of a table-driven integer dot profile.

The caller supplies multiplication semantics, the original math implementation,
and authenticated source bindings. No target provider or reference engine is
imported here. Only the selected dot method changes; the proxy delegates every
other method to its original object. This is an alternate arithmetic execution
implementation, not evidence of unchanged execution implementation or hardware.
"""
from __future__ import annotations

import ctypes
from hashlib import sha256
import inspect
import json
import operator
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

import numpy as np

PROFILE = "table-seeded-anchor-int32-bf16-rne-v1"
BACKEND = "qualified_native"
_OPTIONS = ("-O3", "-std=c11", "-fPIC", "-shared", "-Wall", "-Wextra", "-Werror")


def _file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def derive_product_table(product):
    """Call the original callback once for each ordered pair of raw bytes.

    Entries are sign bit16, exponent bits8..12, and fraction bits0..6. The
    table is numerical metadata; no operand or stored parameter is transformed.
    """
    table = np.empty(65536, np.uint32)
    for left in range(256):
        for right in range(256):
            fields = product(left, right)
            if not isinstance(fields, tuple) or len(fields) != 3:
                raise ValueError("product callback must return three integer fields")
            sign, exponent, fraction = (operator.index(value) for value in fields)
            if sign not in (0, 1) or not 0 <= exponent <= 31 or not 0 <= fraction <= 127:
                raise ValueError("product callback leaves its selected sign/exponent/fraction format")
            table[(left << 8) | right] = (sign << 16) | (exponent << 8) | fraction
    table.flags.writeable = False
    return table


class _Candidate:
    """Raw numerical candidate, exposed only for independent qualification."""
    def __init__(self, table, cache_root, compiler="cc"):
        self.table = table
        if not isinstance(table, np.ndarray) or table.shape != (65536,) or table.dtype != np.uint32 or table.flags.writeable:
            raise ValueError("candidate requires the immutable complete product table")
        cache = Path(cache_root).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        compiler_path = Path(shutil.which(compiler) or compiler).resolve(strict=True)
        version = subprocess.check_output([str(compiler_path), "--version"], text=True).splitlines()[0]
        c_path = Path(__file__).with_suffix(".c")
        build = {"c_source_sha256": _file_hash(c_path), "compiler_sha256": _file_hash(compiler_path),
                 "compiler_version": version, "options": list(_OPTIONS)}
        build_key = sha256(json.dumps(build, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        library_path = cache / ("seeded-dot-" + build_key + ".so")
        if not library_path.exists():
            descriptor, temporary = tempfile.mkstemp(dir=cache, prefix="seeded-dot-build-", suffix=".so")
            os.close(descriptor)
            try:
                subprocess.run([str(compiler_path), *_OPTIONS, str(c_path), "-o", temporary],
                               check=True, capture_output=True)
                Path(temporary).replace(library_path)
            finally:
                if Path(temporary).exists():
                    Path(temporary).unlink()
        library = ctypes.CDLL(str(library_path))
        u8p = ctypes.POINTER(ctypes.c_uint8)
        u16p = ctypes.POINTER(ctypes.c_uint16)
        u32p = ctypes.POINTER(ctypes.c_uint32)
        library.merlin_seeded_anchor_dot.argtypes = [u32p, u8p, u8p, u16p, u16p,
                                                   ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
        library.merlin_seeded_anchor_dot.restype = ctypes.c_int
        library.merlin_product_table_lookup.argtypes = [u32p, u8p, u8p, u32p, ctypes.c_size_t]
        library.merlin_product_table_lookup.restype = ctypes.c_int
        self._library = library
        self._table_pointer = table.ctypes.data_as(u32p)
        self.build_identity = {**build, "library_sha256": _file_hash(library_path)}

    def matmul_codes(self, a, b, seed):
        """Qualification kernel accepts arbitrary positive M,K,N, including wrap cases."""
        a, b, seed = (np.asarray(value) for value in (a, b, seed))
        if (a.dtype != np.uint8 or b.dtype != np.uint8 or seed.dtype != np.uint16
                or a.ndim != 2 or b.ndim != 2 or seed.ndim != 2 or a.shape[1] != b.shape[0]
                or seed.shape != (a.shape[0], b.shape[1]) or any(extent < 1 for extent in (*a.shape, *b.shape))):
            raise ValueError("raw dot requires u8 [M,K], u8 [K,N], and BF16 bits [M,N]")
        if max(a.size, b.size, seed.size) > np.iinfo(np.intp).max // 2:
            raise ValueError("raw dot exceeds the host index domain")
        a, b, seed = (np.ascontiguousarray(value) for value in (a, b, seed))
        output = np.empty(seed.shape, np.uint16)
        u8p, u16p = ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint16)
        code = self._library.merlin_seeded_anchor_dot(self._table_pointer, a.ctypes.data_as(u8p), b.ctypes.data_as(u8p),
                                                    seed.ctypes.data_as(u16p), output.ctypes.data_as(u16p),
                                                    a.shape[0], a.shape[1], b.shape[1])
        if code:
            raise ValueError("native seeded dot refused its raw ABI")
        return output

    def lookup(self, a, b):
        a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
        if a.dtype != np.uint8 or b.dtype != np.uint8 or a.shape != b.shape:
            raise ValueError("product lookup needs equal raw byte arrays")
        output = np.empty(a.shape, np.uint32)
        u8p, u32p = ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint32)
        code = self._library.merlin_product_table_lookup(self._table_pointer, a.ctypes.data_as(u8p),
                                                       b.ctypes.data_as(u8p), output.ctypes.data_as(u32p), a.size)
        if code:
            raise ValueError("native product lookup refused its ABI")
        return output


def candidate_for_qualification(product, cache_root, compiler="cc"):
    """Build a candidate without granting runtime proxy authority."""
    return _Candidate(derive_product_table(product), cache_root, compiler)


def backend_identity(candidate, original_math, product, source_files):
    """Bind every input used to select this optional execution implementation."""
    import torch
    if not source_files or any(not isinstance(key, str) or not key for key in source_files):
        raise ValueError("explicit original arithmetic source bindings are required")
    sources = {key: _file_hash(path) for key, path in sorted(source_files.items())}
    original_file = Path(inspect.getsourcefile(original_math.inner_product_matmul)).resolve()
    product_file = Path(inspect.getsourcefile(product)).resolve()
    allowed = {Path(path).resolve() for path in source_files.values()}
    if original_file not in allowed or product_file not in allowed:
        raise ValueError("original numerical methods are not bound by declared source files")
    return {"backend": BACKEND, "profile": PROFILE,
            "adapter_sha256": _file_hash(__file__), "build": candidate.build_identity,
            "product_table_sha256": sha256(candidate.table.astype("<u4", copy=False).tobytes()).hexdigest(),
            "model_sources": sources, "original_method": original_math.inner_product_matmul.__qualname__,
            "product_function": product.__qualname__, "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__, "byteorder": sys.byteorder,
            "production_k": 32, "original_arithmetic_implementation_unchanged": False,
            "replaced_method": "inner_product_matmul", "other_methods": "delegated to original object",
            "instruction_control_or_memory_overrides": False}


class _QualifiedNativeMath:
    """A qualified, explicitly named math proxy; no reference-engine import."""
    backend_name = BACKEND

    def __init__(self, original_math, candidate, identity, qualification_sha256):
        self._original_math, self._candidate = original_math, candidate
        self._identity = identity
        self.qualification_sha256 = qualification_sha256
        self.calls = 0

    @property
    def identity(self):
        return json.loads(json.dumps(self._identity))

    def execution_metadata(self):
        return {**self.identity, "qualification_sha256": self.qualification_sha256, "calls": self.calls}

    def __getattr__(self, name):
        return getattr(self._original_math, name)

    def inner_product_matmul(self, a, b, c):
        import torch
        if (a.device.type != "cpu" or b.device.type != "cpu" or c.device.type != "cpu"
                or a.ndim != 2 or b.ndim != 2 or c.ndim != 2 or a.shape[1] != 32 or b.shape[0] != 32
                or a.element_size() != 1 or b.element_size() != 1 or a.dtype == torch.bool or b.dtype == torch.bool
                or c.dtype != torch.bfloat16 or tuple(c.shape) != (a.shape[0], b.shape[1])):
            raise ValueError("qualified_native requires CPU byte-code matrices, K=32, and a BF16 seed")
        av = a.contiguous().view(torch.uint8).numpy()
        bv = b.contiguous().view(torch.uint8).numpy()
        seed = c.contiguous().view(torch.uint16).numpy()
        result = self._candidate.matmul_codes(av, bv, seed)
        self.calls += 1
        return torch.from_numpy(result).view(torch.bfloat16)


def load_qualified_math(original_math, product, source_files, qualification, cache_root, compiler="cc"):
    """Refuse missing, incomplete, stale, or differently bound qualifications."""
    if qualification is None:
        raise ValueError("qualified_native requires its matching passing qualification")
    raw = Path(qualification).read_bytes()
    record = json.loads(raw)
    required_groups = {"product_table", "all_seed_codes", "directed", "random_matrices", "continuation", "non_model_shapes"}
    if (record.get("schema") != "merlin.qualified_native_dot.v1" or record.get("status") != "PASS"
            or record.get("mismatches") != 0 or record.get("original_python_comparison") is not True
            or not required_groups <= set(record.get("case_groups", ()))):
        raise ValueError("qualified_native qualification is incomplete or not passing")
    candidate = candidate_for_qualification(product, cache_root, compiler)
    identity = backend_identity(candidate, original_math, product, source_files)
    if record.get("backend_identity") != identity:
        raise ValueError("qualified_native qualification does not match adapter, library, table, or original sources")
    return _QualifiedNativeMath(original_math, candidate, identity, sha256(raw).hexdigest())
