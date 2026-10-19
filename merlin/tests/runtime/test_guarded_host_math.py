"""Actual linker routing, bit guards and explicit provider/archive selection."""

import ctypes
import shutil
import subprocess
from dataclasses import replace

import pytest

from merlin.runtime.host_math import GuardedF32Call, build_host_math, host_math_recipe


def contract():
    return GuardedF32Call("original", "alternate", 0x42A00000, "a" * 64, ((0xFF800000, 0),))


def test_complete_provider_contract_roundtrips_without_implicit_policy():
    guard = contract()
    assert GuardedF32Call.from_dict(guard.to_dict()) == guard
    with pytest.raises(ValueError, match="explicit guarded_f32"):
        host_math_recipe("native", guarded_math=guard)
    with pytest.raises(ValueError, match="typed"):
        host_math_recipe("guarded_f32")
    data = guard.to_dict()
    del data["provider_obligation"]
    with pytest.raises(ValueError, match="complete"):
        GuardedF32Call.from_dict(data)


@pytest.mark.parametrize(
    "changed",
    [
        {"source_symbol": "a;b"},
        {"source_symbol": "alternate"},
        {"body_symbol": "__real_original"},
        {"max_magnitude_bits": True},
        {"max_magnitude_bits": 0x7F800000},
        {"max_magnitude_bits": -1},
        {"library_sha256": ""},
        {"exact_points": ((0, 0), (0, 1))},
        {"exact_points": ((0, 2**32),)},
    ],
)
def test_unknown_symbols_nonfinite_bounds_and_ambiguous_points_refuse(changed):
    with pytest.raises(ValueError):
        host_math_recipe("guarded_f32", guarded_math=replace(contract(), **changed))


def test_actual_compiled_guard_routes_all_bf16_words_and_boundary_neighbors(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("C compiler unavailable")

    def run(command):
        subprocess.run(list(map(str, command)), check=True, capture_output=True)

    objects, flags = build_host_math(
        "guarded_f32",
        tmp_path / "guard",
        cc,
        ["-O2", "-fPIC", "-Wall", "-Wextra", "-Werror"],
        run,
        guarded_math=contract(),
    )
    source = tmp_path / "endpoints.c"
    source.write_text("""#include <stdint.h>
      extern float original(float);
      unsigned original_calls, alternate_calls;
      float original(float x){++original_calls;(void)x;return 11.0f;}
      float alternate(float x){++alternate_calls;(void)x;return 19.0f;}
    """)
    caller = tmp_path / "caller.c"
    caller.write_text("""#include <stdint.h>
      extern float original(float);
      uint32_t evaluate(uint32_t bits) {
        union { float f; uint32_t u; } value={.u=bits};
        value.f=original(value.f);return value.u;
      }
    """)
    for path in (source, caller):
        run([cc, "-O2", "-fno-builtin", "-fPIC", "-c", path, "-o", path.with_suffix(".o")])
    output = tmp_path / "guard.so"
    run(
        [
            cc,
            "-shared",
            "-Wl,-Bsymbolic",
            source.with_suffix(".o"),
            caller.with_suffix(".o"),
            *objects,
            *flags,
            "-o",
            output,
        ]
    )
    lib = ctypes.CDLL(str(output))
    lib.evaluate.argtypes = [ctypes.c_uint32]
    lib.evaluate.restype = ctypes.c_uint32
    before = ctypes.c_uint.in_dll(lib, "original_calls")
    after = ctypes.c_uint.in_dll(lib, "alternate_calls")
    expected_before = expected_after = 0
    words = [word << 16 for word in range(65536)]
    words += [0x42A00000 + offset for offset in (-1, 0, 1)]
    words += [0xC2A00000 + offset for offset in (-1, 0, 1)]
    words += [1, 0x80000001, 0x7FC01234, 0x7FA01234, 0xFFFFFFFF]
    for word in words:
        if word == 0xFF800000:
            expected = 0
        elif word & 0x7FFFFFFF <= 0x42A00000:
            expected_after += 1
            expected = 0x41980000
        else:
            expected_before += 1
            expected = 0x41300000
        assert lib.evaluate(word) == expected
    assert before.value == expected_before and after.value == expected_after
