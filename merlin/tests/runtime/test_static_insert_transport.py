"""Fresh insertion copies complete immutable initial bytes and bounded spans."""

import itertools
import math
import shutil
import subprocess
from dataclasses import replace

import pytest
from xdsl.dialects.builtin import DenseArrayBase, StringAttr, i64

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.deployable_outline import ChainBuffer
from merlin.llvmlower.deployable_packing import static_insert_slice
from merlin.runtime.outlined_chain import _views
from merlin.xdsl_dialects.lowering.dispatch_program import Node


def proof(source_shape, initial_shape, offsets, strides, *, dtype="i8", duplicate=False, sizes=None):
    sizes = source_shape if sizes is None else sizes
    width = {"i1": 1, "i8": 1, "i16": 2, "i32": 4, "f32": 4}[dtype]

    def fmt(shape):
        return "tensor<" + "x".join([*map(str, shape), dtype]) + ">"

    names = ["a", "a"] if duplicate else ["a", "b"]
    args = {names[0]: source_shape, names[1]: initial_shape}

    def arrays(values):
        return "array<i64:" + ",".join(map(str, values)) + ">"

    text = "module {func.func @forward(" + ",".join("%" + n + ":" + fmt(s) for n, s in args.items())
    text += ")->" + fmt(initial_shape) + '{\n%r="tensor.insert_slice"('
    text += ",".join("%" + name for name in names)
    text += ") <{static_offsets=" + arrays(offsets) + ",static_sizes=" + arrays(sizes)
    text += ",static_strides=" + arrays(strides) + ",operandSegmentSizes=array<i32:1,1,0,0,0>}>:("
    text += fmt(source_shape) + "," + fmt(initial_shape) + ")->" + fmt(initial_shape)
    text += "\nfunc.return %r:" + fmt(initial_shape) + "}}"
    module = parse_mlir_text(text)
    module.verify()
    op = next(op for op in module.walk() if op.name == "tensor.insert_slice")
    specs = {n: ChainBuffer(n, s, dtype, math.prod(s) * width, i, None) for i, (n, s) in enumerate(args.items())}
    specs["out"] = ChainBuffer("out", initial_shape, dtype, math.prod(initial_shape) * width, None, None)
    return op, Node("view", "tensor.insert_slice", names, ["out"]), specs


def coordinates(shape, index):
    flat = 0
    for extent, position in zip(shape, index, strict=True):
        flat = flat * extent + position
    return flat


@pytest.mark.parametrize(
    "source_shape,initial_shape,offsets,strides,dtype,duplicate",
    [
        ((2, 3), (5, 7), (1, 2), (2, 1), "i8", False),
        ((2, 3), (5, 9), (1, 2), (2, 2), "i16", False),
        ((3,), (9,), (1,), (2,), "i32", False),
        ((3,), (9,), (2,), (1,), "i8", False),
        ((2, 3), (2, 3), (0, 0), (1, 1), "i8", True),
        ((1, 2, 3), (3, 4, 7), (1, 1, 2), (1, 2, 1), "f32", False),
        ((1, 4, 32, 64), (1, 4, 40, 64), (0, 0, 0, 0), (1, 1, 1, 1), "f32", False),
    ],
)
def test_every_output_byte_source_and_guards(tmp_path, source_shape, initial_shape, offsets, strides, dtype, duplicate):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler unavailable")
    op, node, specs = proof(source_shape, initial_shape, offsets, strides, dtype=dtype, duplicate=duplicate)
    record = static_insert_slice(op, node, specs)
    positions = {name: i for i, name in enumerate(specs)}
    data = {
        name: bytes((i * 17 + j * 41 + 5) % 256 for i in range(specs[name].bytes))
        for j, name in enumerate(dict.fromkeys(node.inputs))
    }
    source, initial = (data[name] for name in node.inputs)
    expected = bytearray(initial)
    width = record["element_bytes"]
    for index in itertools.product(*(range(n) for n in source_shape)):
        read = coordinates(source_shape, index) * width
        dest = (
            coordinates(initial_shape, tuple(o + i * s for o, i, s in zip(offsets, index, strides, strict=True)))
            * width
        )
        expected[dest : dest + width] = source[read : read + width]
    setup = []
    checks = []
    for i, (name, raw) in enumerate(data.items()):
        setup += [
            "unsigned char input" + str(i) + "[]={" + ",".join(map(str, raw)) + "};",
            "unsigned char saved" + str(i) + "[sizeof input" + str(i) + "];",
            "memcpy(saved" + str(i) + ",input" + str(i) + ",sizeof input" + str(i) + ");",
            "c->buffers[" + str(positions[name]) + "]=input" + str(i) + ";",
        ]
        checks.append("assert(!memcmp(saved" + str(i) + ",input" + str(i) + ",sizeof input" + str(i) + "));")
    n = len(expected)
    program = "#include <string.h>\n#include <stddef.h>\n#include <assert.h>\nint main(void){"
    program += "struct{void*buffers[" + str(len(specs)) + "];}context={0},*c=&context;" + "\n".join(setup)
    program += (
        "unsigned char out["
        + str(n + 128)
        + "];memset(out,0x5a,sizeof out);c->buffers["
        + str(positions["out"])
        + "]=out+64;"
    )
    program += "\n".join(_views([record], positions))
    program += (
        "const unsigned char expected[]={"
        + ",".join(map(str, expected))
        + "};assert(!memcmp(out+64,expected,sizeof expected));"
    )
    program += (
        "for(unsigned i=0;i<64;i++)assert(out[i]==0x5a&&out[64+sizeof expected+i]==0x5a);"
        + "\n".join(checks)
        + "return 0;}"
    )
    path = tmp_path / "copy.c"
    path.write_text(program)
    subprocess.run(
        [cc, "-std=c11", "-Wall", "-Wextra", "-Werror", str(path), "-o", str(tmp_path / "copy")],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run([str(tmp_path / "copy")], check=True, capture_output=True, timeout=10)
    assert record["logical_bytes_written"] == len(initial) + len(source)


@pytest.mark.parametrize(
    "change", ["offset", "stride", "sizes", "segment", "unknown", "quant_inner", "borrowed", "dtype"]
)
def test_missing_bounds_types_ownership_or_effect_proof_refuses(change):
    op, node, specs = proof((2, 3), (5, 7), (1, 2), (2, 1))
    if change == "offset":
        op.properties["static_offsets"] = DenseArrayBase.from_list(i64, [5, 2])
    elif change == "stride":
        op.properties["static_strides"] = DenseArrayBase.from_list(i64, [0, 1])
    elif change == "sizes":
        op.properties["static_sizes"] = DenseArrayBase.from_list(i64, [3, 3])
    elif change == "segment":
        from xdsl.dialects.builtin import i32

        op.properties["operandSegmentSizes"] = DenseArrayBase.from_list(i32, [1, 1, 1, 0, 0])
    elif change == "borrowed":
        specs["out"] = replace(specs["out"], arg_index=0)
    elif change == "dtype":
        specs["a"] = replace(specs["a"], dtype="i16")
    else:
        op.attributes["unknown.effect" if change == "unknown" else "prov.quant_inner"] = StringAttr("unproved")
    with pytest.raises(ValueError):
        static_insert_slice(op, node, specs)


def test_rank_reduction_and_unproved_boolean_storage_refuse():
    op, node, specs = proof((2, 3), (1, 2, 3), (0, 0, 0), (1, 1, 1), sizes=(1, 2, 3))
    with pytest.raises(ValueError, match="same-rank"):
        static_insert_slice(op, node, specs)
    op, node, specs = proof((2, 3), (5, 7), (1, 2), (2, 1), dtype="i1")
    with pytest.raises(ValueError, match="nonboolean"):
        static_insert_slice(op, node, specs)


def test_partially_overlapping_readonly_inputs_are_not_mutated(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler unavailable")
    op, node, specs = proof((2, 3), (4, 6), (1, 2), (1, 1))
    record = static_insert_slice(op, node, specs)
    source = tmp_path / "overlap.c"
    original = bytes(range(32))
    expected = bytearray(original[:24])
    expected[8:11] = original[4:7]
    expected[14:17] = original[7:10]
    source.write_text(
        "#include <string.h>\n#include <stddef.h>\n#include <assert.h>\n"
        "int main(void){struct{void*buffers[3];}context={0},*c=&context;"
        "unsigned char input[32],saved[32],dest[152];"
        "for(unsigned i=0;i<32;i++)input[i]=(unsigned char)i;"
        "memcpy(saved,input,32);memset(dest,0x5a,152);"
        "c->buffers[0]=input+4;c->buffers[1]=input;c->buffers[2]=dest+64;"
        + "\n".join(_views([record], {name: i for i, name in enumerate(specs)}))
        + "const unsigned char expected[]={"
        + ",".join(map(str, expected))
        + "};"
        "assert(!memcmp(dest+64,expected,24));assert(!memcmp(input,saved,32));"
        "for(unsigned i=0;i<64;i++)assert(dest[i]==0x5a&&dest[88+i]==0x5a);return 0;}"
    )
    exe = tmp_path / "overlap"
    subprocess.run(
        [cc, "-std=c11", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(exe)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run([str(exe)], check=True, capture_output=True, timeout=10)


def test_rank_zero_or_unknown_result_encoding_never_grants_raw_transport():
    from xdsl.dialects.builtin import TensorType, i8

    for typ in (TensorType(i8, []), TensorType(i8, [5, 7], StringAttr("unknown"))):
        op, node, specs = proof((2, 3), (5, 7), (1, 2), (2, 1))
        op.results[0]._type = typ
        with pytest.raises(ValueError):
            static_insert_slice(op, node, specs)
