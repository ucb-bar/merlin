"""The portable transcendentals: one erf/exp/tanh/log, bit for bit, in the golden and on the device.

``merlin.llvmlower.portable_math`` binds the lowered model's ``erff``/``expf``/``tanhf``/``logf`` to
Merlin's own functions (``merlin/runtime/abi/portable_math.h``), linked with the MLIR runtime into the
host build that writes a golden and into every device image. What this pins:

* the binding renames exactly the bound global symbols of LLVM IR -- not strings, comments or
  look-alike names -- reports what still reaches a C library, and is the identity under ``libm``;
* the policy is a named optional pass, selectable both ways, and recorded with its implementation digest;
* the functions compute the SAME bits whatever compiles them: the host compiler at the host build's
  flags and at -O3 -march=native -ffp-contract=fast, and -- on spike -- the RISC-V GCC that builds a
  Zephyr image's runtime and the clang that builds a bare-metal one, over a dense sweep;
* they are accurate: within 0.5 ulp of the true function over a dense sweep (the exhaustive
  measurement is in docs/design/portable_transcendentals.md), with defined special values.
"""

from __future__ import annotations

import ctypes
import math
import os
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest

from merlin.common.paths import merlin_dir, repo_root
from merlin.llvmlower import optional_passes as OP
from merlin.llvmlower import portable_math as PM

ABI = merlin_dir() / "runtime" / "abi"
RUNTIME_C = ABI / "mlir_runtime.c"
SWEEP_C = merlin_dir() / "tests" / "data" / "portable_math_sweep.c"
FUNCTIONS = ("erf", "exp", "tanh", "log")


@pytest.fixture(autouse=True)
def _default_policy(monkeypatch):
    monkeypatch.delenv(OP.ENV, raising=False)
    monkeypatch.delenv(PM.SWITCH, raising=False)


# ---- the binding --------------------------------------------------------------------------------

_IR = """\
@msg = private constant [12 x i8] c"call @erff\\00"   ; a string that names @erff is not a reference
@erff_table = global [2 x float] zeroinitializer
declare float @erff(float) #0
declare float @expf(float)
declare float @sinf(float)
declare <4 x float> @llvm.exp.v4f32(<4 x float>)
define float @f(float %x) {
  %a = call float @erff(float %x)  ; @expf in a comment is not a reference either
  %b = call float @expf(float %a)
  %c = call float @sinf(float %b)
  %d = load float, ptr @erff_table
  ret float %c
}
"""


def test_bind_renames_only_the_bound_symbols():
    text, report = PM.bind(_IR, environ={})
    assert "declare float @merlin_erff(float) #0" in text
    assert "call float @merlin_erff(float %x)" in text
    assert "call float @merlin_expf(float %a)" in text
    # look-alikes, strings and comments are untouched
    assert 'c"call @erff\\00"' in text
    assert "@erff_table" in text and "@merlin_erff_table" not in text
    assert "; @expf in a comment" in text
    assert report["policy"] == PM.PORTABLE
    assert report["rebound"] == {"erff": 2, "expf": 2}
    # what still depends on a C library's rounding is said, not hidden
    assert report["left_on_target_libm"] == ["llvm.exp.v4f32", "sinf"]
    assert report["implementation_sha256"] == PM.implementation_sha256()


@pytest.mark.parametrize(
    "environ", [{PM.SWITCH: "0"}, {OP.ENV: "-portable-transcendentals"}, {PM.SWITCH: "off"}], ids=str
)
def test_libm_policy_leaves_the_ir_byte_identical(environ):
    text, report = PM.bind(_IR, environ=environ)
    assert text == _IR
    assert report["policy"] == PM.LIBM
    assert report["rebound"] == {}
    assert "erff" in report["left_on_target_libm"] and "expf" in report["left_on_target_libm"]


def test_selection_wins_over_the_switch():
    assert PM.enabled({PM.SWITCH: "0", OP.ENV: "portable-transcendentals"})
    assert not PM.enabled({PM.SWITCH: "1", OP.ENV: "-portable-transcendentals"})
    assert PM.enabled({})  # the default


def test_policy_is_a_listed_numerics_changing_pass_on_by_default():
    entry = OP.get(PM.PASS_NAME)
    assert entry.exactness == OP.NUMERICS_CHANGING
    assert entry.on_by_default and entry.default == "on"
    assert entry.switch == PM.SWITCH
    assert PM.PASS_NAME in OP.table()


def test_record_and_cache_token_follow_the_policy():
    portable, libm = PM.record({}), PM.record({PM.SWITCH: "0"})
    assert portable["policy"] == "portable" and portable["functions"] == PM.BINDINGS
    assert libm == {"name": PM.PASS_NAME, "policy": "libm"}
    assert PM.cache_token({}) != PM.cache_token({PM.SWITCH: "0"})


def test_a_golden_from_the_other_policy_is_named_as_such():
    """A golden that predates the policy was computed with each host's libm."""
    assert PM.reference_disagreement(None, environ={}) is not None
    assert PM.reference_disagreement({}, environ={PM.SWITCH: "0"}) is None
    current = {"transcendental_math": PM.record({})}
    assert PM.reference_disagreement(current, environ={}) is None
    assert PM.reference_disagreement(current, environ={PM.SWITCH: "0"}) is not None
    stale = {"transcendental_math": {**PM.record({}), "implementation_sha256": "0" * 64}}
    assert "revision" in PM.reference_disagreement(stale, environ={})


def test_the_runtime_defines_every_bound_symbol():
    """mlir_runtime.c is what every link path carries; it must define what the binding calls."""
    source = RUNTIME_C.read_text(encoding="utf-8")
    assert '#include "portable_math.h"' in source and "MERLIN_PORTABLE_MATH_IMPLEMENTATION" in source
    header = (ABI / "portable_math.h").read_text(encoding="utf-8")
    for symbol in PM.BINDINGS.values():
        assert f"float {symbol}(float xf)" in header


def test_coefficient_tables_regenerate_byte_identically(tmp_path):
    script = repo_root() / "build_tools" / "scripts" / "gen_portable_math_tables.py"
    try:
        import mpmath  # noqa: F401

        python = sys.executable
    except ImportError:
        from merlin.llvmlower import toolchain

        python = str(toolchain.m2m_python())
        probe = subprocess.run([python, "-c", "import mpmath"], capture_output=True) if os.path.isfile(python) else None
        if probe is None or probe.returncode != 0:
            pytest.skip("mpmath is not available to regenerate the tables")
    done = subprocess.run([python, str(script), "--check"], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


# ---- the functions, host -------------------------------------------------------------------------


def _cc() -> str:
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("no host C compiler")
    return cc


def _host_runtime_lib(tmp_path) -> ctypes.CDLL:
    """The runtime exactly as the host build compiles it (llvmlower.codegen.build_host_shared)."""
    lib = tmp_path / "libmlir_runtime.so"
    subprocess.run([_cc(), "-O2", "-fPIC", "-shared", str(RUNTIME_C), "-lm", "-o", str(lib)], check=True)
    handle = ctypes.CDLL(str(lib))
    for name in FUNCTIONS:
        fn = getattr(handle, f"merlin_{name}f")
        fn.restype, fn.argtypes = ctypes.c_float, [ctypes.c_float]
    return handle


def _bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def test_special_values(tmp_path):
    lib = _host_runtime_lib(tmp_path)
    erf, exp, tanh, log = (getattr(lib, f"merlin_{n}f") for n in FUNCTIONS)
    inf, nan_bits = float("inf"), 0x7FC00000
    for fn in (erf, exp, tanh, log):  # a NaN in -> the canonical quiet NaN out, payload or sign aside
        for b in (0x7FC00000, 0xFFC00001, 0x7F800001, 0xFFBFFFFF):
            assert _bits(fn(_f32(b))) == nan_bits
    assert erf(inf) == 1.0 and erf(-inf) == -1.0 and erf(4.0) == 1.0
    assert _bits(erf(-0.0)) == 0x80000000 and _bits(erf(0.0)) == 0
    assert exp(inf) == inf and exp(-inf) == 0.0 and exp(0.0) == 1.0 and exp(89.0) == inf and exp(-104.0) == 0.0
    assert exp(-103.0) > 0.0  # a subnormal result is produced, not flushed
    assert tanh(inf) == 1.0 and tanh(-inf) == -1.0 and _bits(tanh(-0.0)) == 0x80000000
    assert log(0.0) == -inf and log(-0.0) == -inf and log(inf) == inf and log(1.0) == 0.0
    assert _bits(log(-1.0)) == nan_bits and _bits(log(-inf)) == nan_bits
    assert log(_f32(1)) == pytest.approx(-103.27893, rel=1e-6)  # the smallest subnormal


def _sample(name: str) -> np.ndarray:
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2**32, size=40_000, dtype=np.uint64).astype(np.uint32)
    x = np.concatenate([bits.view(np.float32), rng.uniform(-8, 8, 40_000).astype(np.float32)])
    x = x[np.isfinite(x)]
    if name == "log":
        x = np.abs(x)
        x = x[x > 0]
    return x


@pytest.mark.parametrize("name", FUNCTIONS)
def test_accuracy_within_half_an_ulp(name, tmp_path):
    """Against the double-precision function (error < 1e-16 relative): the portable float result is
    the true value rounded to nearest, so no sampled input may be off by more than half an ulp (plus
    the reference's own error). Exhaustively over every float, see the design doc."""
    lib = _host_runtime_lib(tmp_path)
    fn = getattr(lib, f"merlin_{name}f")
    ref = {"erf": math.erf, "exp": math.exp, "tanh": math.tanh, "log": math.log}[name]
    worst = 0.0
    for x in _sample(name).tolist():
        try:
            want = ref(x)
        except OverflowError:
            continue
        got = fn(x)
        r = np.float32(want) if abs(want) < 3.4e38 else np.float32(np.inf)
        if not np.isfinite(r):
            assert got == float(r)
            continue
        a = abs(r)
        ulp = float(np.nextafter(a, np.float32(np.inf)) - a) if a < np.finfo(np.float32).max else 2.0**104
        worst = max(worst, abs(got - want) / ulp)
    assert worst <= 0.5 + 1e-6, f"{name}: {worst} ulp"


# ---- the functions, one sweep per build ---------------------------------------------------------


def _parse_hashes(console: str) -> dict[str, int]:
    out = {}
    for line in console.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "HASH":
            out[parts[1]] = int(parts[2], 16)
    assert set(out) == set(FUNCTIONS), console
    return out


def _host_sweep(tmp_path, tag: str, flags: list[str]) -> dict[str, int]:
    exe = tmp_path / f"sweep_{tag}"
    subprocess.run([_cc(), *flags, "-DSWEEP_HOST", str(SWEEP_C), str(RUNTIME_C), "-lm", "-o", str(exe)], check=True)
    return _parse_hashes(subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout)


def test_host_builds_agree_bit_for_bit(tmp_path):
    """The golden's build (``cc -O2 -fPIC``) and builds that contract, vectorize and use FMA freely."""
    golden = _host_sweep(tmp_path, "golden", ["-O2", "-fPIC"])
    assert _host_sweep(tmp_path, "o0", ["-O0"]) == golden
    assert _host_sweep(tmp_path, "native", ["-O3", "-march=native", "-ffp-contract=fast", "-funroll-loops"]) == golden


def _spike_or_skip():
    from merlin.runtime.backends import spike

    if not spike.available():
        pytest.skip("chipyard toolchain/spike not available")
    return spike


def _device_sweep(tmp_path, tag: str, runtime_object_cmd) -> dict[str, int]:
    """The sweep as a bare-metal spike image, the runtime compiled by ``runtime_object_cmd``."""
    from merlin.common.paths import runtime_dir
    from merlin.runtime.backends.spike_model import RVV_CFLAGS

    spike = _spike_or_skip()
    gcc = spike.gcc_path()
    harness = runtime_dir() / "baremetal" / "spike"
    work = tmp_path / tag
    work.mkdir()
    flags = [*RVV_CFLAGS, "-ffp-contract=fast"]
    objects = []
    for name, src in (
        ("sweep.o", SWEEP_C),
        ("crt.o", harness / "crt.S"),
        ("htif.o", harness / "htif.c"),
        ("libc.o", harness / "libc_min.c"),
    ):
        subprocess.run([str(gcc), *flags, "-I", str(harness), "-c", str(src), "-o", str(work / name)], check=True)
        objects.append(work / name)
    rt = work / "mlir_rt.o"
    subprocess.run([*runtime_object_cmd, "-c", str(RUNTIME_C), "-o", str(rt)], check=True)
    elf = work / "sweep.elf"
    subprocess.run(
        [
            str(gcc),
            *flags,
            "-nostdlib",
            "-nostartfiles",
            "-T",
            str(harness / "link.ld"),
            *map(str, objects),
            str(rt),
            "-lm",
            "-lgcc",
            "-o",
            str(elf),
        ],
        check=True,
    )
    return _parse_hashes(spike.run_elf(elf, harts=1, isa="rv64gcv", timeout=900))


def test_device_runtime_built_by_gcc_matches_the_golden_build(tmp_path):
    """RISC-V GCC (the compiler a Zephyr image's runtime goes through) at -ffp-contract=fast."""
    from merlin.runtime.backends.spike_model import RVV_CFLAGS

    spike = _spike_or_skip()
    device = _device_sweep(tmp_path, "gcc", [str(spike.gcc_path()), *RVV_CFLAGS, "-ffp-contract=fast"])
    assert device == _host_sweep(tmp_path, "golden", ["-O2", "-fPIC"])


def test_device_runtime_built_by_clang_matches_the_golden_build(tmp_path):
    """clang (the compiler the bare-metal spike images build their runtime with)."""
    from merlin.llvmlower import toolchain
    from merlin.runtime.backends.spike_model import RVV_CFLAGS, _mlir_runtime_compiler

    spike = _spike_or_skip()
    if not toolchain.clang().is_file():
        pytest.skip("clang not available")
    cmd = _mlir_runtime_compiler(toolchain.clang(), spike.gcc_path(), [*RVV_CFLAGS, "-ffp-contract=fast"])
    device = _device_sweep(tmp_path, "clang", cmd)
    assert device == _host_sweep(tmp_path, "golden", ["-O2", "-fPIC"])
