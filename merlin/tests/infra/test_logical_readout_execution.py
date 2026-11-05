"""Native byte/descriptor evidence for an explicitly qualified narrow output ABI.

The primitive retains modular i32 accumulation and writes only final i8 values.
The shared bridge owns descriptor checks and packing/scatter. This tests that
conditional neutral implementation, not hardware readout support or latency.
"""

import ctypes
import importlib.util
import json
import shutil
import signal
import subprocess
import sys
from dataclasses import replace

import pytest

from merlin.common.paths import merlin_dir
from merlin.llvmlower.logical_borrowed_abi import emit_borrowed_dense_translation_unit, scratch_bytes

# Reuse the public neutral ABI test harness rather than duplicate its memory
# engine, descriptor layout, guarded views and original effect premises.
_HELPERS = merlin_dir() / "tests" / "ir" / "test_logical_borrowed_abi.py"
_SPEC = importlib.util.spec_from_file_location("neutral_logical_abi_execution_helpers", _HELPERS)
BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(BASE)
FORMAT_QUALIFICATION = "reviewed-neutral-exact-wrap32-clamp0-127-i8-format-v1"


def capability():
    return replace(BASE.capability(), readout_format_qualification_identity=FORMAT_QUALIFICATION)


def buffers(width):
    lhs, rhs, output = BASE.buffers(width)
    return lhs, rhs, replace(output, dtype="i8")


def primitive(width):
    source = BASE.primitive(width)
    old = "for(int t=0;t<4;t++)((unsigned char*)out)[4*(i*5+j)+t]=(unsigned char)(acc>>(8*t));"
    assert source.count(old) == 1
    source = source.replace("static int forbid_kernel;", "static int forbid_kernel, corrupt_after_write;")
    return (
        source.replace(
            old,
            """
  int64_t signed_acc=acc<=INT32_MAX?(int64_t)acc:(int64_t)acc-((int64_t)1<<32);
  int64_t clipped=signed_acc<0?0:signed_acc>127?127:signed_acc;
  ((unsigned char*)out)[i*5+j]=(unsigned char)clipped;
  if(corrupt_after_write && i==2 && j==4)((unsigned char*)out)[15]^=1;
""",
        )
        + "\nvoid set_corrupt_output(int value){corrupt_after_write=value;}\n"
    )


@pytest.fixture(scope="module", params=[(width, borrow) for width in (1, 2) for borrow in (False, True)])
def compiled(request, tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None or ctypes.sizeof(ctypes.c_void_p) != ctypes.sizeof(ctypes.c_int64):
        pytest.skip("native compiler and selected 64-bit ranked caller required")
    width, borrow = request.param
    root = tmp_path_factory.mktemp("logical-readout")
    unit = emit_borrowed_dense_translation_unit(
        capability(),
        buffers(width),
        symbol="bridge",
        kernel="kernel",
        borrow_contiguous=borrow,
        observer_symbol="observe",
    )
    assert "span(d2,1u" in unit.text and "int8_t raw2[" in unit.text
    source, library = root / "bridge.c", root / "bridge.so"
    source.write_text(unit.text + primitive(width))
    command = [
        compiler,
        "-std=c11",
        "-O2",
        "-shared",
        "-fPIC",
        "-ffreestanding",
        "-fno-builtin",
        str(source),
        "-o",
        str(library),
    ]
    subprocess.run(command, capture_output=True, text=True, check=True)
    return width, borrow, library


def inputs(width, overflow=False):
    if overflow:
        value = (1 << (width * 8 - 1)) - 1
        return [value] * 21, [value] * 35
    lhs = [1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    rhs = [-1, 0, 1, 126, 127, 1, 1, 127, 2, 1] + [0] * 25
    return lhs, rhs


def reference(lhs, rhs):
    raw = [sum(lhs[i * 7 + k] * rhs[k * 5 + j] for k in range(7)) for i in range(3) for j in range(5)]
    wrapped = [(value + (1 << 31)) % (1 << 32) - (1 << 31) for value in raw]
    return bytes(max(0, min(127, value)) for value in wrapped), raw, wrapped


def assert_complete_output(backing, descriptor, before, expected):
    after = bytearray(before)
    for address, value in zip(BASE.addresses(descriptor, 1), expected, strict=True):
        assert ctypes.string_at(address, 1) == bytes([value])
        after[address - ctypes.addressof(backing)] = value
    assert bytes(backing) == bytes(after), "write outside the exact i8 result payload"


@pytest.mark.parametrize("mode", ["contiguous", "offset", "strided", "mixed", "output_misaligned"])
@pytest.mark.parametrize("overflow", [False, True])
def test_narrow_bridge_preserves_every_logical_and_guard_byte(compiled, mode, overflow):
    width, borrow, path = compiled
    native = ctypes.CDLL(str(path))
    native.count.restype = ctypes.c_uint64
    previous = [native.count(i) for i in range(4)]
    shapes, widths, alignments = ((3, 7), (7, 5), (3, 5)), (width, width, 1), (4, 8, 16)
    views = [
        BASE.view(
            shape,
            item_width,
            strided=mode == "strided" or mode == "mixed" and index == 1,
            offset=alignment // item_width if mode == "offset" else 0,
            misaligned=mode == "output_misaligned" and index == 2,
        )
        for index, (shape, item_width, alignment) in enumerate(zip(shapes, widths, alignments, strict=True))
    ]
    lhs, rhs = inputs(width, overflow)
    for (_, descriptor), values in zip(views[:2], (lhs, rhs), strict=True):
        for address, value in zip(BASE.addresses(descriptor, width), values, strict=True):
            ctypes.memmove(address, value.to_bytes(width, "little", signed=True), width)
    before = [bytes(backing) for backing, _ in views]
    expected, raw, wrapped = reference(lhs, rhs)
    if width == 2 and overflow:
        assert all(value > 0 for value in raw) and all(value < 0 for value in wrapped) and expected == bytes(15)
    if not overflow:
        assert {0, 1, 126, 127}.issubset(set(expected)) and 128 in raw
    native.bridge(*(ctypes.byref(descriptor) for _, descriptor in views))
    assert [bytes(backing) for backing, _ in views[:2]] == before[:2]
    assert_complete_output(*views[-1], before[-1], expected)
    eligible = [
        borrow
        and descriptor.strides[1] == 1
        and descriptor.strides[0] == descriptor.sizes[1]
        and (descriptor.aligned + descriptor.offset * item_width) % alignment == 0
        for (_, descriptor), item_width, alignment in zip(views, widths, alignments, strict=True)
    ]
    copied = sum(
        shape[0] * shape[1] * item_width
        for shape, item_width, direct in zip(shapes, widths, eligible, strict=True)
        if not direct
    )
    now = [native.count(i) for i in range(4)]
    assert [now[i] - previous[i] for i in (0, 1, 3)] == [copied, copied, 1]
    assert now[2] == sum(int(direct) << i for i, direct in enumerate(eligible))
    (path.parent / f"result-{mode}-{overflow}.json").write_text(
        json.dumps(
            dict(
                input_bytes=width,
                output_bytes=1,
                accumulator_bytes=4,
                borrowing=borrow,
                descriptor_mode=mode,
                full_output_hex=expected.hex(),
                requested_copy_bytes_each_direction=copied,
                direct_mask=now[2],
                readout_format_qualification_identity=FORMAT_QUALIFICATION,
                scope="independent exact native bytes/guards and logical ABI; no accelerator support or timing",
            ),
            indent=2,
        )
    )


@pytest.mark.parametrize("mode", range(9))
def test_invalid_or_aliasing_narrow_descriptor_refuses_before_primitive(compiled, mode):
    _width, _borrow, path = compiled
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ctypes,sys; ctypes.CDLL(sys.argv[1]).invoke_bad(int(sys.argv[2]))",
            str(path),
            str(mode),
        ],
        capture_output=True,
    )
    assert result.returncode == -signal.SIGILL, (mode, result.returncode, result.stderr)


def test_full_buffer_oracle_detects_a_one_byte_primitive_overwrite(compiled):
    width, _borrow, path = compiled
    native = ctypes.CDLL(str(path))
    lhs, rhs = inputs(width)
    raw_inputs = [b"".join(value.to_bytes(width, "little", signed=True) for value in values) for values in (lhs, rhs)]
    arrays = [ctypes.create_string_buffer(raw) for raw in raw_inputs]
    output = ctypes.create_string_buffer(31)
    ctypes.memset(output, 167, 31)
    descriptor = BASE.Descriptor(ctypes.addressof(output), ctypes.addressof(output), 0, (3, 5), (5, 1))
    before = bytes(output)
    expected, _, _ = reference(lhs, rhs)
    native.set_corrupt_output(1)
    try:
        native.kernel(arrays[0], arrays[1], output)
    finally:
        native.set_corrupt_output(0)
    assert bytes(output)[:15] == expected
    with pytest.raises(AssertionError, match="outside the exact i8 result payload"):
        assert_complete_output(output, descriptor, before, expected)


@pytest.mark.parametrize("width,budget", [(1, 96), (2, 150)])
def test_narrow_scratch_respects_exact_external_byte_width(width, budget):
    cap, views = capability(), buffers(width)
    assert scratch_bytes(views, cap) == budget
    emit_borrowed_dense_translation_unit(
        replace(cap, max_stack_bytes=budget), views, symbol="bridge", kernel="kernel", observer_symbol="observe"
    )
    with pytest.raises(ValueError, match="scratch budget"):
        emit_borrowed_dense_translation_unit(
            replace(cap, max_stack_bytes=budget - 1), views, symbol="bridge", kernel="kernel", observer_symbol="observe"
        )
