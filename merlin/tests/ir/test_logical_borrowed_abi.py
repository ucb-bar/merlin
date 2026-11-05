"""Execute actual shared bridges under explicit neutral logical-pointer premises."""

import ctypes
import shutil
import signal
import subprocess
import sys

import pytest

from merlin.compile.scheduling.dense_contracts import DenseLogicalPointerABI
from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects
from merlin.llvmlower.device_shim import LogicalBuffer
from merlin.llvmlower.logical_borrowed_abi import (
    LogicalBorrowedABI,
    emit_borrowed_dense_translation_unit,
    scratch_bytes,
)


class Descriptor(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * 2),
        ("strides", ctypes.c_int64 * 2),
    ]


def capability():
    return LogicalBorrowedABI(
        DenseLogicalPointerABI(
            "dense_row_major",
            (4, 8, 16),
            ("read_only", "read_only", "fully_written"),
            "logical_elements",
            "synchronous",
        ),
        BorrowedBufferEffects("reviewed-neutral-byte-access-v1", True, True, True),
        "reviewed-neutral-logical-pointer-format-v1",
        4096,
        True,
        BorrowedBufferEffects("reviewed-neutral-private-counter-observer-v1", True, True, True),
        True,
    )


def buffers(width):
    return (
        LogicalBuffer("a", "input", "lhs", (3, 7), "i" + str(width * 8)),
        LogicalBuffer("b", "input", "rhs", (7, 5), "i" + str(width * 8)),
        LogicalBuffer("out", "output", "out", (3, 5), "i32"),
    )


def primitive(width):
    """Independent byte-indexed arithmetic; no planner or generated bridge used."""
    return r"""
#include <stdlib.h>
static uint64_t observed[4];
static int forbid_kernel;
void observe(uint64_t r,uint64_t w,uint64_t mask) {
 observed[0]+=r;observed[1]+=w;observed[2]=mask;observed[3]++;
}
uint64_t count(int i){return observed[i];}
static uint32_t read_word(const unsigned char*p,int n){
 uint32_t v=0;for(int b=0;b<n;b++)v|=(uint32_t)p[b]<<(8*b);
 if(n<4 && (v&(1u<<(8*n-1))))v|=~((1u<<(8*n))-1);
 return v;
}
void kernel(void *a,void *b,void *out){
 if(forbid_kernel)_Exit(77);
 for(int i=0;i<3;i++)for(int j=0;j<5;j++){
  uint32_t acc=0;
  for(int k=0;k<7;k++)acc+=read_word((unsigned char*)a+(i*7+k)*WIDTH,WIDTH)*
                            read_word((unsigned char*)b+(k*5+j)*WIDTH,WIDTH);
  for(int t=0;t<4;t++)((unsigned char*)out)[4*(i*5+j)+t]=(unsigned char)(acc>>(8*t));
 }
}
void invoke_bad(int mode){
 unsigned char a[512]={0},b[512]={0},c[512]={0};
 merlin_lmemref_2d x={a,a,0,{3,7},{7,1}},y={b,b,0,{7,5},{5,1}},z={c,c,0,{3,5},{5,1}};
 forbid_kernel=1;
 if(mode==0)x.aligned=0;
 if(mode==1)x.offset=-1;
 if(mode==2)x.sizes[0]=4;
 if(mode==3)x.strides[0]=0;
 if(mode==4)x.strides[0]=INTPTR_MAX;
 if(mode==5)x.aligned=(void*)(UINTPTR_MAX-2);
 if(mode==6)z.aligned=a;
 if(mode==7)z.strides[0]=1;
 if(mode==8)x.offset=INTPTR_MAX;
 bridge(&x,&y,&z);
}
""".replace("WIDTH", str(width))


@pytest.fixture(scope="module", params=[(w, borrow) for w in (1, 2, 4) for borrow in (False, True)])
def compiled(request, tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None or ctypes.sizeof(ctypes.c_void_p) != ctypes.sizeof(ctypes.c_int64):
        pytest.skip("selected native compiler and 64-bit ranked caller required")
    width, borrow = request.param
    root = tmp_path_factory.mktemp("logical-abi")
    unit = emit_borrowed_dense_translation_unit(
        capability(),
        buffers(width),
        symbol="bridge",
        kernel="kernel",
        borrow_contiguous=borrow,
        observer_symbol="observe",
    )
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


def view(shape, width, *, strided=False, offset=0, misaligned=False):
    strides = (shape[1] * 3 + 2, 3) if strided else (shape[1], 1)
    end = offset + (shape[0] - 1) * strides[0] + (shape[1] - 1) * strides[1] + 1
    backing = ctypes.create_string_buffer(end * width + 512)
    ctypes.memset(backing, 167, ctypes.sizeof(backing))
    begin = ctypes.addressof(backing)
    aligned = (begin + 127) // 64 * 64 + int(misaligned)
    descriptor = Descriptor(begin, aligned, offset, shape, strides)
    return backing, descriptor


def addresses(descriptor, width):
    return [
        descriptor.aligned + (descriptor.offset + i * descriptor.strides[0] + j * descriptor.strides[1]) * width
        for i in range(descriptor.sizes[0])
        for j in range(descriptor.sizes[1])
    ]


@pytest.mark.parametrize("mode", ["contiguous", "offset", "strided", "misaligned"])
def test_actual_borrow_or_fallback_preserves_every_byte(compiled, mode):
    width, borrowing, path = compiled
    # Counter deltas isolate calls sharing the same private library image.
    library = ctypes.CDLL(str(path))
    library.count.restype = ctypes.c_uint64
    previous = [library.count(i) for i in range(4)]
    shapes, widths = ((3, 7), (7, 5), (3, 5)), (width, width, 4)
    values = [
        view(
            shape,
            w,
            strided=mode == "strided",
            offset=(alignment // w if mode == "offset" else 0),
            misaligned=mode == "misaligned" and index == 0,
        )
        for index, (shape, w, alignment) in enumerate(zip(shapes, widths, (4, 8, 16), strict=True))
    ]
    expected_inputs = []
    for index, (backing, descriptor) in enumerate(values[:2]):
        logical = []
        for k, address in enumerate(addresses(descriptor, width)):
            value = ((k * 131 + index * 17) % (1 << (width * 8))) - (1 << (width * 8 - 1))
            raw = value.to_bytes(width, "little", signed=True)
            ctypes.memmove(address, raw, width)
            logical.append(value)
        expected_inputs.append(logical)
    before = [bytes(backing) for backing, _ in values]
    a, b = expected_inputs
    reference = [sum(a[i * 7 + k] * b[k * 5 + j] for k in range(7)) % (1 << 32) for i in range(3) for j in range(5)]
    library.bridge(*(ctypes.byref(descriptor) for _, descriptor in values))
    assert [bytes(backing) for backing, _ in values[:2]] == before[:2]
    output, descriptor = values[2]
    after = bytearray(before[2])
    for address, value in zip(addresses(descriptor, 4), reference, strict=True):
        assert ctypes.string_at(address, 4) == value.to_bytes(4, "little")
        start = address - ctypes.addressof(output)
        after[start : start + 4] = value.to_bytes(4, "little")
    assert bytes(output) == bytes(after)
    eligible = [
        borrowing
        and descriptor.strides[1] == 1
        and descriptor.strides[0] == descriptor.sizes[1]
        and (descriptor.aligned + descriptor.offset * w) % alignment == 0
        for (_, descriptor), w, alignment in zip(values, widths, (4, 8, 16), strict=True)
    ]
    copied = sum(
        shape[0] * shape[1] * w for shape, w, direct in zip(shapes, widths, eligible, strict=True) if not direct
    )
    now = [library.count(i) for i in range(4)]
    assert [now[i] - previous[i] for i in (0, 1, 3)] == [copied, copied, 1]
    assert now[2] == sum(int(direct) << i for i, direct in enumerate(eligible))


@pytest.mark.parametrize("mode", range(9))
def test_invalid_descriptor_never_reaches_primitive(compiled, mode):
    _width, _borrowing, path = compiled
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


def test_explicit_budget_and_unknown_or_changed_capabilities_refuse():
    from dataclasses import replace

    cap = capability()
    for changed in (
        replace(cap.declaration, layout="unknown"),
        replace(cap.declaration, access=("read_write", "read_only", "fully_written")),
        replace(cap.declaration, bounds="unknown"),
        replace(cap.declaration, completion="asynchronous"),
    ):
        with pytest.raises(ValueError):
            replace(cap, declaration=changed)
    with pytest.raises(ValueError, match="scratch budget"):
        emit_borrowed_dense_translation_unit(
            replace(cap, max_stack_bytes=1), buffers(2), symbol="bridge", kernel="k", observer_symbol="observe"
        )
    for changes in (dict(borrow_contiguous=1), dict(symbol="malformed();"), dict(kernel="bridge")):
        parameters = dict(symbol="bridge", kernel="kernel", borrow_contiguous=True, observer_symbol="observe")
        parameters.update(changes)
        with pytest.raises(ValueError):
            emit_borrowed_dense_translation_unit(cap, buffers(1), **parameters)


@pytest.mark.parametrize("width,budget", [(1, 138), (2, 192), (4, 300)])
def test_complete_fallback_array_storage_respects_exact_budget(width, budget):
    from dataclasses import replace

    cap, views = capability(), buffers(width)
    # Complete arrays: 21 lhs elements, 35 rhs elements, 15 i32 outputs,
    # plus typed padding for the independently selected 4/8/16-byte alignments.
    assert scratch_bytes(views, cap) == budget
    emit_borrowed_dense_translation_unit(
        replace(cap, max_stack_bytes=budget), views, symbol="bridge", kernel="kernel", observer_symbol="observe"
    )
    with pytest.raises(ValueError, match="scratch budget"):
        emit_borrowed_dense_translation_unit(
            replace(cap, max_stack_bytes=budget - 1), views, symbol="bridge", kernel="kernel", observer_symbol="observe"
        )
    with pytest.raises(ValueError, match="complete contraction"):
        scratch_bytes((views[0], replace(views[1], shape=(6, 5)), views[2]), cap)


@pytest.mark.parametrize(
    "changes",
    [
        {"observer_effects": None},
        {"observer_preserves_logical_state": False},
        {"observer_preserves_logical_state": 1},
        {"observe_transfers": False},
        {"qualified_effects": None},
        {"format_qualification_identity": ""},
    ],
)
def test_observation_does_not_inherit_primitive_qualification(changes):
    from dataclasses import replace

    with pytest.raises(ValueError):
        replace(capability(), **changes)


def test_read_only_input_aliases_are_safe_in_actual_bridge(compiled):
    width, _borrowing, path = compiled
    library = ctypes.CDLL(str(path))
    shared, rhs = view((7, 5), width)
    lhs = Descriptor(rhs.allocated, rhs.aligned, 0, (3, 7), (7, 1))
    output, out = view((3, 5), 4)
    values = [i - 17 for i in range(35)]
    for address, value in zip(addresses(rhs, width), values, strict=True):
        ctypes.memmove(address, value.to_bytes(width, "little", signed=True), width)
    before = bytes(shared)
    library.bridge(ctypes.byref(lhs), ctypes.byref(rhs), ctypes.byref(out))
    assert bytes(shared) == before
    actual = [int.from_bytes(ctypes.string_at(address, 4), "little") for address in addresses(out, 4)]
    assert actual == [
        sum(values[i * 7 + k] * values[k * 5 + j] for k in range(7)) % (1 << 32) for i in range(3) for j in range(5)
    ]


@pytest.mark.parametrize("borrow", [False, True])
def test_default_unobserved_bridge_executes_without_callback(tmp_path, borrow):
    from dataclasses import replace

    compiler = shutil.which("cc")
    if compiler is None or ctypes.sizeof(ctypes.c_void_p) != ctypes.sizeof(ctypes.c_int64):
        pytest.skip("selected native compiler and 64-bit ranked caller required")
    cap = replace(capability(), observe_transfers=False, observer_effects=None, observer_preserves_logical_state=False)
    unit = emit_borrowed_dense_translation_unit(
        cap, buffers(2), symbol="bridge", kernel="kernel", borrow_contiguous=borrow
    )
    source, output = tmp_path / "bridge.c", tmp_path / "bridge.so"
    source.write_text(unit.text + primitive(2))
    subprocess.run(
        [compiler, "-O2", "-shared", "-fPIC", "-fno-builtin", str(source), "-o", str(output)],
        capture_output=True,
        text=True,
        check=True,
    )
    native = ctypes.CDLL(str(output))
    values = [
        view(shape, width, strided=index == 1)
        for index, (shape, width) in enumerate((((3, 7), 2), ((7, 5), 2), ((3, 5), 4)))
    ]
    for index, (_, descriptor) in enumerate(values[:2]):
        for address in addresses(descriptor, 2):
            ctypes.memmove(address, (index + 1).to_bytes(2, "little", signed=True), 2)
    before = [bytes(backing) for backing, _ in values]
    native.bridge(*(ctypes.byref(descriptor) for _, descriptor in values))
    assert [bytes(backing) for backing, _ in values[:2]] == before[:2]
    backing, descriptor = values[-1]
    expected = bytearray(before[-1])
    for address in addresses(descriptor, 4):
        start = address - ctypes.addressof(backing)
        expected[start : start + 4] = (14).to_bytes(4, "little")
    assert bytes(backing) == bytes(expected)
    native.count.restype = ctypes.c_uint64
    assert native.count(3) == 0
