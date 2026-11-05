"""Check the deployed C K-window packer against independent element indexing."""

from __future__ import annotations

import ctypes
import random
import shutil
import subprocess

import pytest

from merlin.common.paths import repo_root


@pytest.fixture(scope="module")
def packer(tmp_path_factory: pytest.TempPathFactory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for the deployed packer test")
    build = tmp_path_factory.mktemp("row-major-k-window")
    source = build / "wrapper.c"
    library = build / "packer.so"
    source.write_text("""
        #include "row_major_k_window.h"
        int pack(const uint8_t *src, size_t src_bytes,
                 uint8_t *dst, size_t dst_bytes, size_t rows,
                 size_t cols, size_t start, size_t width, size_t elem) {
          return merlin_pack_row_major_k_window(
              src, src_bytes, dst, dst_bytes, rows, cols, start, width, elem);
        }
    """)
    subprocess.run(
        [compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
         "-I", str(repo_root() / "merlin/runtime/c"), str(source), "-o", str(library)],
        check=True, capture_output=True, text=True,
    )
    function = ctypes.CDLL(str(library)).pack
    function.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                         ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                         ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
    function.restype = ctypes.c_int
    return function


def test_pack_every_small_exact_window_and_preserve_canary(packer) -> None:
    rng = random.Random(0xA71A5)
    for element_bytes in (1, 2, 4):
        for rows in (1, 2, 5):
            for source_columns in range(1, 8):
                source = bytes(rng.randrange(256) for _ in range(rows * source_columns * element_bytes))
                source_buffer = ctypes.create_string_buffer(source)
                for tile_columns in range(1, source_columns + 1):
                    for start in range(source_columns - tile_columns + 1):
                        tile_bytes = rows * tile_columns * element_bytes
                        destination = (ctypes.c_uint8 * (tile_bytes + 2))(*([0xA5] * (tile_bytes + 2)))
                        result = packer(
                            source_buffer, len(source), ctypes.byref(destination, 1), tile_bytes,
                            rows, source_columns, start, tile_columns, element_bytes,
                        )
                        assert result == 0
                        expected = bytes(
                            source[(row * source_columns + col) * element_bytes + byte]
                            for row in range(rows)
                            for col in range(start, start + tile_columns)
                            for byte in range(element_bytes)
                        )
                        assert bytes(destination[1:-1]) == expected
                        assert destination[0] == destination[-1] == 0xA5
                        assert bytes(source_buffer.raw[:len(source)]) == source


def test_invalid_geometry_lengths_and_overlap_leave_output_untouched(packer) -> None:
    source = ctypes.create_string_buffer(bytes(range(24)))
    destination = (ctypes.c_uint8 * 9)(*([0xA5] * 9))
    valid = (source, 24, destination, 6, 3, 8, 2, 2, 1)
    for changed_index, changed_value in ((1, 23), (3, 5), (4, 0),
                                         (5, 0), (6, 7), (7, 3), (8, 0)):
        arguments = list(valid)
        arguments[changed_index] = changed_value
        assert packer(*arguments) == 1
        assert bytes(destination) == bytes([0xA5] * 9)
    assert packer(source, 24, ctypes.byref(source, 1), 6, 3, 8, 2, 2, 1) == 1
    assert bytes(source.raw[:24]) == bytes(range(24))
