"""Explicit precision policy changes only selected libm calls and linked identity."""

import ctypes
import decimal
import math
import shutil
import struct
import subprocess

import pytest

from merlin.runtime.backends import spike
from merlin.runtime.backends.spike_model import _supplemental_object_digest
from merlin.runtime.host_math import build_host_math, host_math_recipe


def test_default_is_emission_neutral_without_compiler(tmp_path):
    def forbidden(_):
        raise AssertionError("default must not invoke a compiler")

    work = tmp_path / "absent"
    assert build_host_math("native", work, "absent-compiler", [], forbidden) == ([], ())
    assert not work.exists()


@pytest.mark.parametrize("policy", [None, True, "double", "expf_via_double "])
def test_unknown_policy_refuses_before_io(tmp_path, policy):
    with pytest.raises(ValueError, match="host_math_policy"):
        build_host_math(policy, tmp_path / "absent", "absent", [], lambda _: None)
    assert not (tmp_path / "absent").exists()


def test_missing_compiler_output_refuses(tmp_path):
    with pytest.raises(RuntimeError, match="did not produce"):
        build_host_math("expf_via_double", tmp_path, "absent", [], lambda _: None)


def test_rsqrtf_policy_links_without_native_rsqrtf(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("requires C compiler and libm")

    def run(cmd):
        return subprocess.run(list(map(str, cmd)), check=True, capture_output=True)

    caller = tmp_path / "caller.c"
    caller.write_text("extern float rsqrtf(float); float evaluate(float x){return rsqrtf(x);}\n")
    run([cc, "-fno-builtin", "-fPIC", "-c", caller, "-o", tmp_path / "caller.o"])
    objects, flags = build_host_math("rsqrtf_via_sqrtf", tmp_path / "policy", cc, ["-fPIC"], run)
    assert len(objects) == 1
    assert flags == ("-Wl,--wrap=rsqrtf",)
    assert "__wrap_rsqrtf" in host_math_recipe("rsqrtf_via_sqrtf").source
    run([cc, "-shared", tmp_path / "caller.o", *objects, *flags, "-lm", "-o", tmp_path / "selected.so"])
    lib = ctypes.CDLL(str(tmp_path / "selected.so"))
    lib.evaluate.argtypes = [ctypes.c_float]
    lib.evaluate.restype = ctypes.c_float
    assert struct.pack("<f", lib.evaluate(4.0)) == struct.pack("<f", 0.5)
    assert math.isinf(lib.evaluate(0.0)) and lib.evaluate(0.0) > 0
    assert math.isinf(lib.evaluate(-0.0)) and lib.evaluate(-0.0) < 0
    assert math.isnan(lib.evaluate(-1.0))
    assert lib.evaluate(float("inf")) == 0.0


def test_rsqrtf_policy_links_with_selected_baremetal_toolchain(tmp_path):
    gcc = spike.gcc_path()
    nm = gcc.with_name("riscv64-unknown-elf-nm")
    if not gcc.is_file() or not nm.is_file():
        pytest.skip("requires selected RV64 bare-metal GCC and nm")

    def run(cmd):
        return subprocess.run(list(map(str, cmd)), check=True, capture_output=True, text=True)

    flags = ["-march=rv64gcv", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding"]
    objects, link_flags = build_host_math("rsqrtf_via_sqrtf", tmp_path / "policy", gcc, flags, run)
    source = tmp_path / "caller.c"
    source.write_text("extern float rsqrtf(float); int main(void) { volatile float x=4.0f; "
                      "return rsqrtf(x)==0.5f ? 0:1; }\n")
    run([gcc, *flags, "-fno-builtin", "-c", source, "-o", tmp_path / "caller.o"])
    elf = tmp_path / "rsqrt.elf"
    run([gcc, *flags, "-nostartfiles", "-Wl,-e,main", tmp_path / "caller.o", *objects,
         *link_flags, "-lm", "-lc", "-lgcc", "-o", elf])
    assert elf.is_file()
    assert not run([nm, "-u", elf]).stdout.strip()


def test_real_link_selection_special_values_and_precision(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("requires C compiler and libm")

    def run(cmd):
        return subprocess.run(list(map(str, cmd)), check=True, capture_output=True)

    # A deliberately distinguishable original implementation proves linker interception.
    src = tmp_path / "caller.c"
    src.write_text("extern float expf(float); float evaluate(float x){return expf(x);}\n")
    stub = tmp_path / "native.c"
    stub.write_text("float expf(float x){return x-17.0f;}\n")
    for source in (src, stub):
        run([cc, "-O2", "-fPIC", "-fno-builtin", "-c", source, "-o", source.with_suffix(".o")])
    common = [src.with_suffix(".o"), stub.with_suffix(".o")]
    native, native_flags = build_host_math("native", tmp_path / "native", cc, ["-O2", "-fPIC"], run)
    selected, flags = build_host_math("expf_via_double", tmp_path / "selected", cc, ["-O2", "-fPIC"], run)
    for name, objects, extra in [("baseline", [], ()), ("native", native, native_flags), ("selected", selected, flags)]:
        run([cc, "-shared", "-Wl,-Bsymbolic", *common, *objects, *extra, "-lm", "-o", tmp_path / (name + ".so")])
    assert (tmp_path / "baseline.so").read_bytes() == (tmp_path / "native.so").read_bytes()
    assert _supplemental_object_digest(native) != _supplemental_object_digest(selected)
    libs = [ctypes.CDLL(str(tmp_path / (n + ".so"))) for n in ("native", "selected")]
    for lib in libs:
        lib.evaluate.argtypes = [ctypes.c_float]
        lib.evaluate.restype = ctypes.c_float
    assert libs[0].evaluate(0) == -17
    assert libs[1].evaluate(0) == 1
    assert libs[1].evaluate(-0.0) == 1
    assert libs[1].evaluate(float("-inf")) == 0
    assert libs[1].evaluate(float("inf")) == float("inf")
    assert math.isnan(libs[1].evaluate(float("nan")))

    def bits(x):
        return struct.pack("<f", x)

    for value in [-100.0, -90.0, -16.125, -1.0, -0.03125, 0.5, 1.0, 10.125, 80.0, 88.0]:
        x = ctypes.c_float(value).value
        with decimal.localcontext() as context:
            context.prec = 100
            expected = ctypes.c_float(float(decimal.Decimal(x).exp())).value
        assert bits(libs[1].evaluate(x)) == bits(expected)
