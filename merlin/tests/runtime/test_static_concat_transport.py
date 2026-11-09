"""Static concat copies dense spans, retaining aliasing inputs and fresh output."""

import math
import shutil
import subprocess

import pytest
from xdsl.dialects.builtin import StringAttr

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.deployable_outline import ChainBuffer
from merlin.llvmlower.deployable_packing import static_concat
from merlin.runtime.outlined_chain import _views
from merlin.xdsl_dialects.lowering.dispatch_program import Node


def proof(shapes, axis, *, dtype="i8", duplicate=False):
    width = {"i8": 1, "i16": 2, "i32": 4, "f32": 4, "i1": 1}[dtype]
    dims = list(shapes[0])
    dims[axis] = sum(shape[axis] for shape in shapes)
    outshape = tuple(dims)

    def fmt(shape):
        return "tensor<" + "x".join([*map(str, shape), dtype]) + ">"

    names = ["a"] * len(shapes) if duplicate else ["a" + str(i) for i in range(len(shapes))]
    args = {name: shape for name, shape in zip(names, shapes, strict=True)}
    source = (
        "module {func.func @forward("
        + ",".join("%" + name + ":" + fmt(shape) for name, shape in args.items())
        + ")->"
        + fmt(outshape)
        + '{\n%cat="tensor.concat"('
        + ",".join("%" + name for name in names)
        + ") <{dim="
        + str(axis)
        + ":i64}> : ("
        + ",".join(fmt(shape) for shape in shapes)
        + ")->"
        + fmt(outshape)
        + "\nfunc.return %cat:"
        + fmt(outshape)
        + "}}"
    )
    module = parse_mlir_text(source)
    module.verify()
    op = next(op for op in module.walk() if op.name == "tensor.concat")
    specs = {
        name: ChainBuffer(name, shape, dtype, math.prod(shape) * width, i, None)
        for i, (name, shape) in enumerate(args.items())
    }
    specs["out"] = ChainBuffer("out", outshape, dtype, math.prod(outshape) * width, None, None)
    node = Node("view", "tensor.concat", names, ["out"])
    return op, node, specs


@pytest.mark.parametrize(
    "shapes,axis,dtype,duplicate",
    [
        ([(1, 1, 32), (1, 1, 32)], 2, "f32", True),
        ([(2, 3), (2, 5), (2, 1)], 1, "i8", False),
        ([(3, 2), (1, 2)], 0, "i32", False),
        ([(2, 1, 3), (2, 2, 3)], 1, "i16", False),
        ([(1, 2, 1, 3, 1), (1, 2, 1, 5, 1)], 3, "i8", False),
    ],
)
def test_all_static_spans_execute_exactly_and_preserve_guards(tmp_path, shapes, axis, dtype, duplicate):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler unavailable")
    op, node, specs = proof(shapes, axis, dtype=dtype, duplicate=duplicate)
    selected = static_concat(op, node, specs)
    positions = {key: i for i, key in enumerate(specs)}
    emitted = "\n".join(_views([selected], positions))
    n = specs["out"].bytes
    width = selected["element_bytes"]
    # Independent lexicographic coordinate oracle, including every byte/tail.
    named_data = {
        name: bytes((i * 29 + j * 11 + 3) % 256 for i in range(specs[name].bytes))
        for j, name in enumerate(dict.fromkeys(node.inputs))
    }
    inputs = [named_data[name] for name in node.inputs]
    expected = bytearray()
    outshape = specs["out"].shape
    for outer in range(math.prod(outshape[:axis])):
        for shape, data in zip(shapes, inputs, strict=True):
            span = math.prod(shape[axis:]) * width
            expected.extend(data[outer * span : (outer + 1) * span])
    declarations = []
    checks = []
    for i, (name, data) in enumerate(named_data.items()):
        declarations.append("unsigned char input" + str(i) + "[]={" + ",".join(map(str, data)) + "};")
        declarations.append("unsigned char saved" + str(i) + "[sizeof input" + str(i) + "];")
        declarations.append("memcpy(saved" + str(i) + ",input" + str(i) + ",sizeof input" + str(i) + ");")
        declarations.append("c->buffers[" + str(positions[name]) + "]=input" + str(i) + ";")
        checks.append("assert(!memcmp(saved" + str(i) + ",input" + str(i) + ",sizeof input" + str(i) + "));")
    source = tmp_path / "copy.c"
    source.write_text(
        "#include <string.h>\n#include <stddef.h>\n#include <assert.h>\n"
        + "typedef struct {void*buffers["
        + str(len(specs))
        + "];} C;\nint main(void){C context={0},*c=&context;\n"
        + "\n".join(declarations)
        + "\nunsigned char dest["
        + str(n + 128)
        + "];memset(dest,0x5a,sizeof dest);\n"
        + "c->buffers["
        + str(positions["out"])
        + "]=dest+64;\n"
        + emitted
        + "\nconst unsigned char expected[]={"
        + ",".join(map(str, expected))
        + "};"
        + "assert(!memcmp(dest+64,expected,sizeof expected));"
        + "for(unsigned i=0;i<64;i++)assert(dest[i]==0x5a&&dest[64+sizeof expected+i]==0x5a);"
        + "\n".join(checks)
        + "return 0;}"
    )
    exe = tmp_path / "copy"
    subprocess.run(
        [cc, "-std=c11", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(exe)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run([str(exe)], check=True, capture_output=True, timeout=10)


def test_same_ssa_duplicate_operands_are_preserved_in_order():
    op, node, specs = proof([(1, 1, 32), (1, 1, 32)], 2, dtype="f32", duplicate=True)
    selected = static_concat(op, node, specs)
    assert selected["inputs"] == ["a", "a"]
    assert selected["input_block_bytes"] == [128, 128]
    assert selected["output_block_bytes"] == 256


@pytest.mark.parametrize("change", ["axis", "shape", "unknown", "quant_inner", "borrowed_destination"])
def test_unknown_metadata_bounds_or_destination_ownership_refuse(change):
    from dataclasses import replace

    from xdsl.dialects.builtin import IntegerAttr, i64

    op, node, specs = proof([(2, 3), (2, 5)], 1)
    if change == "axis":
        op.properties["dim"] = IntegerAttr(-1, i64)
    elif change == "shape":
        specs["a0"] = replace(specs["a0"], shape=(2, 4))
    elif change == "borrowed_destination":
        specs["out"] = replace(specs["out"], arg_index=0)
    else:
        op.attributes["test.unknown" if change == "unknown" else "prov.quant_inner"] = StringAttr("unproved")
    with pytest.raises(ValueError):
        static_concat(op, node, specs)


def test_boolean_concat_needs_its_own_storage_permission():
    op, node, specs = proof([(2, 3), (2, 5)], 1, dtype="i1")
    with pytest.raises(ValueError, match="nonboolean"):
        static_concat(op, node, specs)


def test_partially_overlapping_readonly_inputs_keep_every_source_byte(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler unavailable")
    op, node, specs = proof([(2, 3), (2, 2)], 1)
    selected = static_concat(op, node, specs)
    positions = {key: i for i, key in enumerate(specs)}
    source = tmp_path / "overlap.c"
    source.write_text(
        "#include <string.h>\n#include <stddef.h>\n#include <assert.h>\n"
        "int main(void){struct{void*buffers[3];}context={0},*c=&context;"
        "unsigned char input[16],saved[16],dest[138];"
        "for(unsigned i=0;i<16;i++)input[i]=(unsigned char)i;"
        "memcpy(saved,input,16);memset(dest,0x5a,138);"
        "c->buffers[0]=input+2;c->buffers[1]=input+4;c->buffers[2]=dest+64;"
        + "\n".join(_views([selected], positions))
        + "const unsigned char expected[]={2,3,4,4,5,5,6,7,6,7};"
        "assert(!memcmp(dest+64,expected,10));assert(!memcmp(input,saved,16));"
        "for(unsigned i=0;i<64;i++)assert(dest[i]==0x5a&&dest[74+i]==0x5a);return 0;}"
    )
    exe = tmp_path / "overlap"
    subprocess.run(
        [cc, "-std=c11", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(exe)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run([str(exe)], check=True, capture_output=True, timeout=10)


def test_rank_zero_and_unknown_encoding_do_not_acquire_transport_permission():
    from xdsl.dialects.builtin import TensorType, i8

    op, node, specs = proof([(2, 3), (2, 5)], 1)
    op.results[0]._type = TensorType(i8, [])
    with pytest.raises(ValueError, match="ranked"):
        static_concat(op, node, specs)
    op, node, specs = proof([(2, 3), (2, 5)], 1)
    op.results[0]._type = TensorType(i8, [2, 8], StringAttr("unknown"))
    with pytest.raises(ValueError, match="encoding"):
        static_concat(op, node, specs)
