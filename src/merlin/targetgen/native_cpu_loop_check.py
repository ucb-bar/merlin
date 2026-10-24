"""Observe CPU vector widths and the closed TensorIterator loop from installed headers."""

import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path


def tokens(text):
    """Small C++ lexical reader for a closed loop; no text/regex IR matching."""
    result, i = [], 0
    while i < len(text):
        if text[i].isspace():
            i += 1
        elif text.startswith("//", i):
            end = text.find("\n", i)
            i = len(text) if end < 0 else end + 1
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                raise ValueError("unterminated source comment")
            i = end + 2
        elif text[i].isalnum() or text[i] == "_":
            end = i + 1
            while end < len(text) and (text[end].isalnum() or text[end] == "_"):
                end += 1
            result.append(text[i:end])
            i = end
        elif text[i] in {'"', "'"}:
            quote, end = text[i], i + 1
            while end < len(text) and text[end] != quote:
                end += 2 if text[end] == "\\" else 1
            if end >= len(text):
                raise ValueError("unterminated source string")
            result.append(text[i : end + 1])
            i = end + 1
        else:
            pair = text[i : i + 2]
            if pair in {"::", "+=", "<=", "-=", ">=", "==", "!=", "&&", "||", "++", "--"}:
                result.append(pair)
                i += 2
            else:
                result.append(text[i])
                i += 1
    return result


def observe(torch):
    root = Path(torch.__file__).parent / "include"
    loops = root / "ATen/native/cpu/Loops.h"
    source = tokens(loops.read_text())
    start = source.index("vectorized_loop")
    loop = source.index("for", source.index("Vec", start))
    # Skip the local data-pointer initialization loop.
    while source[loop : loop + 7] != ["for", "(", ";", "i", "<=", "n", "-"]:
        loop = source.index("for", loop + 1)
    factor = source[loop + 7]
    if not factor.isdigit():
        raise ValueError("CPU vector-loop unroll is not a source literal")
    tail = ["*", "Vec", "::", "size", "(", ")", ";", "i", "+=", factor, "*", "Vec", "::", "size", "(", ")", ")"]
    if source[loop + 8 : loop + 8 + len(tail)] != tail:
        raise ValueError("CPU vector-loop boundary/update form changed")
    text = """#include <ATen/cpu/vec/vec.h>
#include <ATen/TensorIterator.h>
extern "C" long half_size() {return at::vec::Vectorized<c10::Half>::size();}
extern "C" long float_size() {return at::vec::Vectorized<float>::size();}
extern "C" long grain_size() {return at::internal::GRAIN_SIZE;}
"""
    with tempfile.TemporaryDirectory(prefix="cpu-loop-proof-") as temporary:
        src, library = Path(temporary) / "observe.cpp", Path(temporary) / "observe.so"
        src.write_text(text)
        subprocess.run(
            [
                os.environ.get("CXX", "c++"),
                "-std=c++17",
                "-O2",
                "-shared",
                "-fPIC",
                "-mavx2",
                "-mfma",
                "-mf16c",
                "-DCPU_CAPABILITY=AVX2",
                "-DCPU_CAPABILITY_AVX2",
                "-I",
                str(root),
                str(src),
                "-o",
                str(library),
            ],
            check=True,
        )
        handle = ctypes.CDLL(str(library))
        half, wide, grain = (int(getattr(handle, name)()) for name in ("half_size", "float_size", "grain_size"))
    paths = [loops, root / "ATen/TensorIterator.h", *(root / "ATen/cpu/vec").rglob("*.h")]
    return dict(
        reduced_vector_elements=half * int(factor),
        float_copy_vector_elements=wide,
        parallel_grain=grain,
        loop_unroll=int(factor),
        significand_storage_vector_elements=half,
        observer_sha256=hashlib.sha256(text.encode()).hexdigest(),
        source_files={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},
    )
