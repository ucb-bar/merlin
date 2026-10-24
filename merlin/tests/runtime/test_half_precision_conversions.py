"""Exact soft conversions, including every binary16 boundary and source NaNs."""

import ctypes
import subprocess

import numpy as np
import pytest

from merlin.common.paths import runtime_dir
from merlin.llvmlower import toolchain
from merlin.llvmlower.c_runtime import _bf16_bits


@pytest.fixture(scope="module")
def conversions(tmp_path_factory):
    if not toolchain.clang().is_file():
        pytest.skip("LLVM compiler unavailable")
    work = tmp_path_factory.mktemp("half-conversions")
    source = work / "wrappers.c"
    source.write_text(
        f'#include "{runtime_dir() / "abi/mlir_runtime.c"}"\n'
        "uint16_t half_from_float(float x) { return merlin_f16_bits(__truncsfhf2(x)); }\n"
        "uint16_t half_from_double(double x) { return merlin_f16_bits(__truncdfhf2(x)); }\n"
        "float half_to_float(uint16_t x) { return __extendhfsf2(merlin_f16_from_bits(x)); }\n"
        "uint16_t brain_from_float(float x) { return merlin_bf16_bits(__truncsfbf2(x)); }\n"
        "float brain_to_float(uint16_t x) { return __extendbfsf2(merlin_bf16_from_bits(x)); }\n"
        "uint16_t brain_from_double(double x) { return merlin_bf16_bits(__truncdfbf2(x)); }\n"
    )
    library = work / "conversions.so"
    subprocess.run(
        [str(toolchain.clang()), "-O2", "-shared", "-fPIC", "-fno-builtin", str(source), "-lm", "-o", str(library)],
        check=True,
        capture_output=True,
    )
    loaded = ctypes.CDLL(str(library))
    for name, argument, result in [
        ("half_from_float", ctypes.c_float, ctypes.c_uint16),
        ("half_from_double", ctypes.c_double, ctypes.c_uint16),
        ("half_to_float", ctypes.c_uint16, ctypes.c_float),
        ("brain_from_float", ctypes.c_float, ctypes.c_uint16),
        ("brain_to_float", ctypes.c_uint16, ctypes.c_float),
        ("brain_from_double", ctypes.c_double, ctypes.c_uint16),
    ]:
        function = getattr(loaded, name)
        function.argtypes = [argument]
        function.restype = result
    return loaded


def test_every_half_and_bfloat_storage_word_widens_exactly(conversions):
    words = np.arange(65536, dtype=np.uint16)
    actual_half = np.array([conversions.half_to_float(int(x)) for x in words], dtype=np.float32)
    reference_half = words.view(np.float16).astype(np.float32)
    np.testing.assert_array_equal(actual_half, reference_half)
    actual_brain = np.array([conversions.brain_to_float(int(x)) for x in words], dtype=np.float32)
    reference_brain = (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
    np.testing.assert_array_equal(actual_brain, reference_brain)
    finite = np.isfinite(reference_half)
    np.testing.assert_array_equal(actual_half[finite].view(np.uint32), reference_half[finite].view(np.uint32))
    finite = np.isfinite(reference_brain)
    np.testing.assert_array_equal(actual_brain[finite].view(np.uint32), reference_brain[finite].view(np.uint32))


def test_every_half_midpoint_and_neighbors_round_even(conversions):
    values = np.arange(0x7C00, dtype=np.uint16).view(np.float16).astype(np.float32)
    midpoints = (values[:-1] + values[1:]) * np.float32(0.5)
    # Include zero/subnormal and maximum-finite/overflow midpoints.
    midpoints = np.concatenate([midpoints, np.array([65520], dtype=np.float32)])
    inputs = np.concatenate(
        [
            np.nextafter(midpoints, np.float32(-np.inf)),
            midpoints,
            np.nextafter(midpoints, np.float32(np.inf)),
        ]
    )
    inputs = np.concatenate([inputs, -inputs])
    with np.errstate(over="ignore"):
        expected = inputs.astype(np.float16).view(np.uint16)
    actual = np.array([conversions.half_from_float(float(x)) for x in inputs], dtype=np.uint16)
    np.testing.assert_array_equal(actual, expected)
    # A binary64 neighbor must not double-round through the binary32 midpoint.
    midpoint = 1.0 + 2.0**-11
    assert conversions.half_from_double(np.nextafter(midpoint, np.inf)) == 0x3C01
    assert conversions.half_from_double(midpoint) == 0x3C00
    midpoint = 1.0 + 2.0**-8
    assert conversions.brain_from_double(np.nextafter(midpoint, np.inf)) == 0x3F81
    assert conversions.brain_from_double(midpoint) == 0x3F80


def test_random_binary32_and_special_values(conversions):
    rng = np.random.default_rng(834)
    words = np.concatenate(
        [
            rng.integers(0, 2**32, 100000, dtype=np.uint32),
            np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7F800001, 0xFFFFFFFF], dtype=np.uint32),
        ]
    )
    values = words.view(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        expected_half = values.astype(np.float16).view(np.uint16)
    actual_half = np.array([conversions.half_from_float(float(x)) for x in values], dtype=np.uint16)
    finite_or_inf = ~np.isnan(values)
    np.testing.assert_array_equal(actual_half[finite_or_inf], expected_half[finite_or_inf])
    actual_brain = np.array([conversions.brain_from_float(float(x)) for x in values], dtype=np.uint16)
    np.testing.assert_array_equal(actual_brain[finite_or_inf], _bf16_bits(values)[finite_or_inf])
    for words16, fraction_mask, exponent_mask in [(actual_half, 0x3FF, 0x7C00), (actual_brain, 0x7F, 0x7F80)]:
        assert np.all((words16[~finite_or_inf] & exponent_mask) == exponent_mask)
        assert np.all((words16[~finite_or_inf] & fraction_mask) != 0)


def test_bfloat_nan_cannot_round_to_infinity_or_wrap_sign():
    words = np.array([0x7F800001, 0x7FFFFFFF, 0xFF800001, 0xFFFFFFFF], dtype=np.uint32)
    encoded = _bf16_bits(words.view(np.float32))
    assert np.all((encoded & 0x7F80) == 0x7F80)
    assert np.all((encoded & 0x7F) != 0)
    np.testing.assert_array_equal(encoded >> 15, words >> 31)


def test_selected_scalar_isa_executes_compiler_generated_half_conversions(tmp_path):
    from merlin.runtime.backends import spike, spike_model

    if not all(path.is_file() for path in (spike.gcc_path(), toolchain.clang(), spike.spike_path())):
        pytest.skip("bare-metal toolchain unavailable")
    plan = spike_model.selected_model_compiler_plan(backend="scalar", cflags_override=None, vlen=None, features=None)
    flags = plan["model_cflags"] + ["-ffunction-sections", "-fdata-sections"]
    gcc = spike.gcc_path()
    compiler = spike_model._mlir_runtime_compiler(toolchain.clang(), gcc, flags)
    harness = runtime_dir() / "baremetal/spike"
    inputs = np.array(
        [0, 0x80000000, 0x33800000, 0x33000000, 0x387FC000, 0x3F801000, 0x3F803000, 0x477FF000, 0x7F800000, 0xFF800000],
        np.uint32,
    )
    with np.errstate(over="ignore"):
        expected = inputs.view(np.float32).astype(np.float16).view(np.uint16)
    source = tmp_path / "caller.c"
    source.write_text(
        '#include <stdint.h>\n#include "htif.h"\n'
        "volatile uint32_t inputs[]={" + ",".join(str(int(x)) + "u" for x in inputs) + "};\n"
        "int main(long hart) { if(hart) htif_exit(0); "
        "for(unsigned i=0;i<sizeof(inputs)/sizeof(inputs[0]);i++) { "
        "uint32_t word=inputs[i]; float x; __builtin_memcpy(&x,&word,4); "
        "_Float16 h=(_Float16)x; uint16_t bits; __builtin_memcpy(&bits,&h,2); "
        "htif_putd(bits); htif_putc(32); x=(float)h; __builtin_memcpy(&word,&x,4); "
        "htif_putd(word); htif_putc(32); } htif_putc(10); htif_exit(0); }\n"
    )

    def run(command):
        return subprocess.run(list(map(str, command)), check=True, capture_output=True, text=True, timeout=60)

    objects = []
    for name, path, prefix in [
        ("caller", source, compiler),
        ("runtime", runtime_dir() / "abi/mlir_runtime.c", compiler),
        *[
            (name, harness / (name + suffix), [gcc, *plan["gcc_cflags"]])
            for name, suffix in [("crt", ".S"), ("htif", ".c"), ("libc_min", ".c")]
        ],
    ]:
        obj = tmp_path / (name + ".o")
        run([*prefix, "-I", harness, "-c", path, "-o", obj])
        objects.append(obj)
    elf = tmp_path / "conversions.elf"
    run(
        [
            gcc,
            *plan["gcc_cflags"],
            "-nostdlib",
            "-nostartfiles",
            "-Wl,--gc-sections",
            "-T",
            harness / "link.ld",
            *objects,
            "-lm",
            "-lgcc",
            "-o",
            elf,
        ]
    )
    march = next(flag.removeprefix("-march=") for flag in flags if flag.startswith("-march="))
    result = run([spike.spike_path(), "--isa=" + march, "-m128", elf])
    observed = np.array([int(x) for x in result.stdout.split()], np.uint32).reshape(-1, 2)
    np.testing.assert_array_equal(observed[:, 0], expected)
    np.testing.assert_array_equal(observed[:, 1], expected.view(np.float16).astype(np.float32).view(np.uint32))


def test_soft_conversions_match_eager_cpu_torch(conversions):
    from merlin.integrations.model2mlir import capture_python

    python = capture_python()
    if not python.is_file():
        pytest.skip("capture interpreter unavailable")
    script = r"""
import ctypes
import numpy as np
import torch
lib=ctypes.CDLL(LIBRARY)
for dtype, encode in [(torch.float16, lib.half_from_float), (torch.bfloat16, lib.brain_from_float)]:
    encode.argtypes=[ctypes.c_float]; encode.restype=ctypes.c_uint16
    bits=np.arange(65536,dtype=np.uint16)
    if dtype == torch.float16:
        values=bits.view(np.float16).astype(np.float32)
    else:
        values=(bits.astype(np.uint32)<<np.uint32(16)).view(np.float32)
    finite=values[np.isfinite(values)]
    positive=np.sort(finite[finite>=0])
    midpoints=(positive[:-1].astype(np.float64)+positive[1:].astype(np.float64))/2
    midpoints=midpoints.astype(np.float32)
    values=np.concatenate([values, midpoints, np.nextafter(midpoints,np.float32(np.inf)),
                           np.nextafter(midpoints,np.float32(-np.inf))])
    expected=torch.from_numpy(values).to(dtype).view(torch.uint16).numpy()
    observed=np.array([encode(float(x)) for x in values],np.uint16)
    finite_or_inf=~np.isnan(values)
    np.testing.assert_array_equal(observed[finite_or_inf],expected[finite_or_inf])
    # Each tensor operation returns 16-bit storage; internal scalar coefficients
    # retain f32 precision. Separately called operations must round separately.
    x=torch.tensor([0.7, 1.5, -2.5],dtype=dtype)
    coefficient=1.0004 if dtype==torch.float16 else 1.0039
    eager=torch.ops.aten.mul.Scalar(x,coefficient)
    wide=x.float()*coefficient
    np.testing.assert_array_equal(eager.view(torch.uint16).numpy(),wide.to(dtype).view(torch.uint16).numpy())
    assert not torch.equal(eager, x*torch.tensor(coefficient,dtype=dtype))
print(torch.__version__)
""".replace("LIBRARY", repr(conversions._name))
    subprocess.run([str(python), "-c", script], check=True, capture_output=True, text=True)


@pytest.mark.parametrize(
    "extension,native_half,native_brain", [(None, False, False), ("zfhmin", True, False), ("zfbfmin", False, True)]
)
def test_selected_isa_chooses_native_or_soft_conversions(tmp_path, extension, native_half, native_brain):
    from merlin.runtime.backends import spike_model

    if not toolchain.clang().is_file():
        pytest.skip("LLVM compiler unavailable")
    plan = spike_model.selected_model_compiler_plan(backend="scalar", cflags_override=None, vlen=None, features=None)
    flags = list(plan["model_cflags"])
    if extension:
        flags = [flag + "_" + extension if flag.startswith("-march=") else flag for flag in flags]
    source = tmp_path / "conversions.ll"
    source.write_text(
        "define half @half_narrow(float %x) { %y = fptrunc float %x to half\n ret half %y }\n"
        "define bfloat @brain_narrow(float %x) { %y = fptrunc float %x to bfloat\n ret bfloat %y }\n"
    )
    assembly = tmp_path / "conversions.s"
    subprocess.run(
        [str(toolchain.clang()), spike_model.CLANG_TARGET, *flags, "-S", str(source), "-o", str(assembly)],
        check=True,
        capture_output=True,
        text=True,
    )
    emitted = assembly.read_text()
    assert ("__truncsfhf2" not in emitted) == native_half
    assert ("__truncsfbf2" not in emitted) == native_brain
