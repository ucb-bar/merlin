"""Independent CPU packet contracts, target compilation and portable oracle."""

import ctypes
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.bounded_rne_lanes import emit_bounded_rne_lanes
from merlin.llvmlower.toolchain import clang


@pytest.mark.parametrize("lanes", [1, 2, 3, 4, 7, 8])
@pytest.mark.parametrize("bits", [8, 16])
def test_portable_lanes_match_independent_rounding_and_target_compiles(tmp_path, lanes, bits):
    calls = ", ".join(f"x[{i}]" for i in range(lanes))
    libs = []
    for isa in ("portable", "rv64gc"):
        src = tmp_path / (isa + ".c")
        src.write_text(
            "#include <stdint.h>\n"
            + emit_bounded_rne_lanes("packet", bits=bits, lanes=lanes, host_isa=isa)
            + f"void run(const float*x,int32_t*out){{packet({calls},out);}}\n"
        )
        if isa == "portable":
            out = tmp_path / f"portable_{bits}_{lanes}.so"
            subprocess.run(
                [clang(), "-O2", "-shared", "-fPIC", str(src), "-lm", "-o", str(out)], check=True, capture_output=True
            )
            libs.append(ctypes.CDLL(str(out)))
        else:
            subprocess.run(
                [
                    clang(),
                    "--target=riscv64-unknown-elf",
                    "-march=rv64gc",
                    "-O2",
                    "-c",
                    str(src),
                    "-o",
                    str(tmp_path / "target.o"),
                ],
                check=True,
                capture_output=True,
            )
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    half = np.arange(lo - 2, hi + 3, dtype=np.float32) + np.float32(0.5)
    values = np.concatenate(
        [
            half,
            np.nextafter(half, np.float32(np.inf)),
            np.nextafter(half, np.float32(-np.inf)),
            np.array([-np.inf, np.inf, 0.0, -0.0, 1e-40, -1e-40], dtype=np.float32),
        ]
    )
    fn = libs[0].run
    ptr = ctypes.c_void_p
    fn.argtypes = [ptr, ptr]
    for start in range(0, len(values), lanes):
        a = np.resize(values[start : start + lanes], lanes).astype(np.float32)
        out = np.full(lanes + 2, 77123, dtype=np.int32)
        fn(a.ctypes.data, out.ctypes.data + 4)
        np.testing.assert_array_equal(out[1:-1], np.rint(np.clip(a, lo, hi)).astype(np.int32))
        assert out[0] == out[-1] == 77123


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(bits=25),
        dict(bits=True),
        dict(lanes=0),
        dict(lanes=9),
        dict(lanes=True),
        dict(host_isa=None),
        dict(host_isa="unknown"),
    ],
)
def test_invalid_contract_refuses(kwargs):
    options = dict(bits=8, lanes=4, host_isa="rv64gc")
    options.update(kwargs)
    with pytest.raises(ValueError):
        emit_bounded_rne_lanes("packet", **options)
