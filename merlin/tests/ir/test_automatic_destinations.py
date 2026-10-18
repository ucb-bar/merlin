"""Ordinary destination reuse: actual native copies, complete results and state."""

from __future__ import annotations

import ctypes
import json
import subprocess

import numpy as np
import pytest
from xdsl.dialects import arith, builtin, tensor
from xdsl.utils.exceptions import VerifyException

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.automatic_destinations import prepare_automatic_destinations
from merlin.llvmlower.destination_check import verify_destination_rewrite
from merlin.llvmlower.insert_slice_destination import rewrite_module
from merlin.llvmlower.optional_passes import Selection, applied


def fixture():
    return (repo_root() / "merlin/tests/ir/insert_destination_fixture.mlir").read_text()


def test_checks_reconstruct_ir_without_mutating_source_or_uses():
    source = fixture()
    original = parse_mlir_text(source)
    current = original.clone()
    assert len(rewrite_module(current, initialize_border_only=True)) == 1
    uses = [tuple(v.uses) for op in current.walk() for v in op.results]
    assert len(verify_destination_rewrite(original, current)) == 1
    assert [tuple(v.uses) for op in current.walk() for v in op.results] == uses
    output, report = prepare_automatic_destinations(source, serial=True)
    assert str(original) == str(parse_mlir_text(source))
    assert report["rewrites"] == 1
    assert [row["name"] for row in report["checks"]] == [
        "destination_source_legality", "destination_forwarding", "destination_conformance",
    ]
    assert tuple(report["checks"][1]["invalidated"]) == ("source",)
    assert prepare_automatic_destinations(output, serial=True)[1]["rewrites"] == 0


@pytest.mark.parametrize("options", [
    {"serial": False}, {"max_added_operations": 0}, {"max_metadata_items": 0}, {"index_bits": 5},
])
def test_unsupported_family_capacity_and_growth_retain_exact_source(options):
    source = fixture()
    output, report = prepare_automatic_destinations(source, **({"serial": True} | options))
    assert output == source and report["rewrites"] == 0


def test_exact_growth_boundaries_and_selected_index_width():
    source = fixture()
    for ops, metadata, expected in [(13, 120, 0), (14, 119, 0), (14, 120, 1)]:
        assert prepare_automatic_destinations(source, serial=True, index_bits=6,
            max_added_operations=ops, max_metadata_items=metadata)[1]["rewrites"] == expected
    with applied(Selection.parse("-automatic-host-destinations")):
        assert prepare_automatic_destinations(source, serial=True)[0] == source
    full = fixture().replace("tensor<4x5xi8>", "tensor<2x3xi8>").replace(
        "static_offsets=array<i64:1,1>", "static_offsets=array<i64:0,0>")
    assert prepare_automatic_destinations(full, serial=True)[0] == full


@pytest.mark.parametrize("typ", ['tensor<?x3xf32>', 'tensor<2x3xf32, "opaque">'])
def test_dynamic_or_encoded_inputs_do_not_imply_a_complete_writable_domain(typ, monkeypatch):
    from merlin.llvmlower import insert_slice_destination

    source = fixture().replace('tensor<2x3xf32>', typ)
    before = parse_mlir_text(source)
    before.verify()
    output, report = prepare_automatic_destinations(source, serial=True)
    assert output == source and report["rewrites"] == 0
    # A permissive emitter still cannot qualify an unknown input domain.
    monkeypatch.setattr(insert_slice_destination, "_safe_source", lambda *args: True)
    after = before.clone()
    assert len(insert_slice_destination.rewrite_module(after, initialize_border_only=True)) == 1
    with pytest.raises(ValueError, match="positive static unencoded"):
        verify_destination_rewrite(before, after)


@pytest.mark.parametrize("change", ["accounting", "checker_mutation", "library"])
def test_checked_facts_and_library_cannot_change_before_publication(tmp_path, monkeypatch, change):
    from merlin.llvmlower import destination_check, insert_slice_destination

    rewrite = insert_slice_destination.rewrite_module
    if change == "library":
        owner = tmp_path / "insert_slice_destination.py"
        owner.write_text("selected helper bytes")
        monkeypatch.setattr(insert_slice_destination, "__file__", str(owner))

        def changed(*args, **kwargs):
            result = rewrite(*args, **kwargs)
            owner.write_text("changed helper bytes")
            return result

        monkeypatch.setattr(insert_slice_destination, "rewrite_module", changed)
    elif change == "accounting":
        def changed(*args, **kwargs):
            rewrite(*args, **kwargs)
            return []

        monkeypatch.setattr(insert_slice_destination, "rewrite_module", changed)
    else:
        verify = destination_check.verify_destination_rewrite

        def changed(before, after):
            result = verify(before, after)
            after.attributes["unaccounted"] = builtin.UnitAttr()
            return result

        monkeypatch.setattr(destination_check, "verify_destination_rewrite", changed)
    with pytest.raises(ValueError, match="accounting|analysis pass.*changed|library changed"):
        prepare_automatic_destinations(fixture(), serial=True)


@pytest.mark.parametrize("change", ["effect", "index", "live", "attribute", "output_read"])
def test_unknown_effects_indexing_observers_and_output_reads_are_barriers(change):
    source = fixture()
    if change == "effect":
        source = source.replace("func.func @padded", "func.func private @touch()\nfunc.func @padded")
        source = source.replace(
            ' %result = "tensor.insert_slice"', ' func.call @touch() : () -> ()\n %result = "tensor.insert_slice"')
    elif change == "index":
        source = source.replace("%y = arith.fptosi", "%index = linalg.index 0 : index\n %y = arith.fptosi")
    elif change == "live":
        source = source.replace("-> tensor<4x5xi8> {", "-> (tensor<4x5xi8>,tensor<2x3xi8>) {")
        source = source.replace("return %result : tensor<4x5xi8>", "return %result,%q : tensor<4x5xi8>,tensor<2x3xi8>")
    elif change == "attribute":
        source = source.replace("-> tensor<4x5xi8> {", "-> tensor<4x5xi8> attributes {unknown_semantics = true} {")
    else:
        source = source.replace("linalg.yield %y : i8", "linalg.yield %unused : i8")
    parse_mlir_text(source).verify()
    output, report = prepare_automatic_destinations(source, serial=True)
    assert output == source and report["rewrites"] == 0


@pytest.mark.parametrize("mutation", ["missing_border", "overlap", "fill", "interior", "arithmetic", "escape"])
def test_actual_geometry_arithmetic_fill_and_ownership_mutations_refuse(mutation):
    before = parse_mlir_text(fixture())
    after = before.clone()
    assert len(rewrite_module(after, initialize_border_only=True)) == 1
    inserts = [op for op in after.walk() if isinstance(op, tensor.InsertSliceOp)]
    if mutation == "missing_border":
        first = inserts[0]
        first.result.replace_all_uses_with(first.dest)
        first.parent.erase_op(first)
    elif mutation == "overlap":
        second = inserts[1]
        second.properties["static_offsets"] = inserts[0].static_offsets
        view = second.source.owner.outputs[0].owner
        view.properties["static_offsets"] = inserts[0].static_offsets
    elif mutation == "fill":
        fill = inserts[0].source.owner
        value = arith.ConstantOp(builtin.IntegerAttr(9, builtin.i8))
        fill.parent.insert_op_before(value, fill)
        fill.operands = [value.result, *fill.outputs]
    elif mutation == "interior":
        inserts[-1].properties["static_offsets"] = builtin.DenseArrayBase.from_list(builtin.i64, [0, 1])
    elif mutation == "arithmetic":
        body = inserts[-1].source.owner.body.block
        body.last_op.operands = [body.args[-1]]
    else:
        parent = inserts[-1].parent
        returned = parent.last_op
        returned.operands = [inserts[0].result]
    with pytest.raises((ValueError, VerifyException), match="destination|verify|operand|matching|unaccounted"):
        verify_destination_rewrite(before, after)


def program(shape, padded, offsets, element, pad_literal=None):
    rank = len(shape)
    dims = ",".join(f"d{i}" for i in range(rank))
    identity = f"affine_map<({dims})->({dims})>"
    maps = f"[{identity},{identity},{identity}]"
    iterators = "[" + ",".join('"parallel"' for _ in shape) + "]"
    small = "tensor<" + "x".join(map(str, shape)) + "x" + element + ">"
    large = "tensor<" + "x".join(map(str, padded)) + "x" + element + ">"
    floating = element.startswith("f")
    minus, one = ("-7.0", "1.0") if floating else ("-7", "1")
    if pad_literal is not None:
        minus = pad_literal
    add = "arith.addf" if floating else "arith.addi"
    def arrays(numbers):
        return "array<i64:" + ",".join(map(str, numbers)) + ">"
    return f'''module {{
      func.func @forward(%x:{small},%state:{small}) -> ({large},{large},{small}) {{
        %v = arith.constant {minus} : {element}
        %empty = tensor.empty() : {small}
        %pad = tensor.splat %v : {large}
        %q = linalg.generic {{indexing_maps={maps},iterator_types={iterators}}}
          ins(%x,%state : {small},{small}) outs(%empty : {small}) {{
          ^bb0(%a:{element},%b:{element},%unused:{element}):
            %sum = {add} %a,%b : {element}
            linalg.yield %sum : {element}
        }} -> {small}
        %result = "tensor.insert_slice"(%q,%pad) <{{static_offsets={arrays(offsets)},
          static_sizes={arrays(shape)},static_strides={arrays([1]*rank)},
          operandSegmentSizes=array<i32:1,1,0,0,0>}}> : ({small},{large}) -> {large}
        %next = linalg.generic {{indexing_maps=[{identity},{identity}],iterator_types={iterators}}}
          ins(%state : {small}) outs(%empty : {small}) {{
          ^bb0(%old:{element},%unused:{element}):
            %one = arith.constant {one} : {element}
            %sum = {add} %old,%one : {element}
            linalg.yield %sum : {element}
        }} -> {small}
        return %result,%pad,%next : {large},{large},{small}
      }}
    }}'''


@pytest.mark.parametrize("shape,padded,offsets,element,pad_literal,transpose", [
    ((5,), (10,), (2,), "i8", None, False), ((2, 3), (4, 5), (1, 1), "i16", None, False),
    ((5, 7), (7, 11), (1, 2), "i32", None, False), ((2, 3, 4), (4, 6, 7), (1, 2, 1), "f32", None, False),
    ((2, 3), (5, 7), (2, 2), "f32", "-0.0", False), ((2, 3), (4, 5), (1, 1), "i16", None, True),
])
def test_ordinary_native_outputs_live_padding_aliases_and_carried_state(
    tmp_path, shape, padded, offsets, element, pad_literal, transpose,
):
    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.toolchain import clang

    source = program(shape, padded, offsets, element, pad_literal)
    if transpose:
        source = source.replace("%x:tensor<2x3xi16>", "%x:tensor<3x2xi16>").replace(
            "ins(%x,%state : tensor<2x3xi16>", "ins(%x,%state : tensor<3x2xi16>").replace(
            "indexing_maps=[affine_map<(d0,d1)->(d0,d1)>",
            "indexing_maps=[affine_map<(d0,d1)->(d1,d0)>", 1)
    pad_value = -7 if pad_literal is None else float(pad_literal)
    dtype = {"i8": np.int8, "i16": np.int16, "i32": np.int32, "f32": np.float32}[element]
    support = tmp_path / "copy_counter.c"
    support.write_text('''#include <stdint.h>
#include <stddef.h>
struct unranked { int64_t rank; void *descriptor; };
struct ranked { void *allocated, *aligned; int64_t offset; int64_t sizes[]; };
uint64_t observed_copy_calls, observed_copy_bytes;
uint64_t observed_malloc_calls, observed_malloc_bytes;
extern void *__real_malloc(size_t);
void *__wrap_malloc(size_t bytes) {
  observed_malloc_calls++; observed_malloc_bytes += bytes;
  return __real_malloc(bytes);
}
extern void __real_memrefCopy(int64_t, struct unranked *, struct unranked *);
void __wrap_memrefCopy(int64_t width, struct unranked *src, struct unranked *dst) {
  struct ranked *desc = src->descriptor;
  uint64_t count = 1;
  for (int64_t i = 0; i < src->rank; ++i) count *= desc->sizes[i];
  observed_copy_calls++; observed_copy_bytes += count * width;
  __real_memrefCopy(width, src, dst);
}
''')
    measurements = {}
    for arm, switch in [("baseline", "-automatic-host-destinations"), ("optimized", "+automatic-host-destinations")]:
        with applied(Selection.parse(switch)):
            result = lower_model(source, tmp_path / arm, targets=(), vectorize=False)
        assert result.stats["automatic_host_destinations"]["rewrites"] == (arm == "optimized")
        library_path = tmp_path / (arm + ".so")
        subprocess.run([str(clang()), "-O2", "-fPIC", "-shared", str(result.ll_path), str(mlir_runtime_c()),
                        str(support), "-Wl,--wrap=memrefCopy", "-Wl,--wrap=malloc", "-o", str(library_path)],
                       check=True, capture_output=True)
        invoke = HostModel.load(str(library_path), image_policy=PrivateHostImagePolicy(tmp_path.resolve()))
        library = invoke.lib
        assert invoke.image_sha256 is not None
        n = int(np.prod(shape))
        storage = np.full(n + 33, 67, dtype=dtype)
        base = storage[16:16 + n + 1]
        base[:] = (np.arange(n + 1) * 17 - 129).astype(dtype)
        x = base[:-1].reshape(tuple(reversed(shape)) if transpose else shape)
        state = base[1:].reshape(shape)
        original = storage.copy()
        for _ in range(3):
            backing = [np.full(int(np.prod(dims)) + 32, 99, dtype=dtype) for dims in (padded, padded, shape)]
            outputs = [buf[16:-16].reshape(dims) for buf, dims in zip(backing, (padded, padded, shape), strict=True)]
            old_state = state.base.copy()
            expected = np.full(padded, pad_value, dtype=dtype)
            expected[tuple(slice(o, o + n) for o, n in zip(offsets, shape, strict=True))] = (
                x.T if transpose else x) + state
            next_state = (state + np.array(1, dtype=dtype)).astype(dtype)
            invoke([(a.ctypes.data, a.shape) for a in [x, state, *outputs]])
            np.testing.assert_array_equal(outputs[0], expected)
            np.testing.assert_array_equal(outputs[1], np.full(padded, pad_value, dtype=dtype))
            if element.startswith("f"):
                np.testing.assert_array_equal(outputs[0].view(np.uint8), expected.view(np.uint8))
                pad_bits = np.full(padded, pad_value, dtype=dtype).view(np.uint8)
                np.testing.assert_array_equal(outputs[1].view(np.uint8), pad_bits)
            np.testing.assert_array_equal(outputs[2], next_state)
            np.testing.assert_array_equal(storage, original)
            np.testing.assert_array_equal(state.base, old_state)
            for buf in backing:
                np.testing.assert_array_equal(buf[:16], np.full(16, 99, dtype=dtype))
                np.testing.assert_array_equal(buf[-16:], np.full(16, 99, dtype=dtype))
            state = outputs[2]
        measurements[arm] = {"copy_calls": ctypes.c_uint64.in_dll(library, "observed_copy_calls").value,
                             "copy_bytes": ctypes.c_uint64.in_dll(library, "observed_copy_bytes").value,
                             "loaded_image_sha256": invoke.image_sha256,
                             "malloc_calls": ctypes.c_uint64.in_dll(library, "observed_malloc_calls").value,
                             "malloc_requested_bytes": ctypes.c_uint64.in_dll(library, "observed_malloc_bytes").value,
                             "llvm_copy_sites": sum("call " in line and (
                                 "@llvm.memcpy" in line or "@memrefCopy" in line)
                                 for line in result.ll_path.read_text().splitlines())}
    if measurements["baseline"]["copy_calls"]:
        assert measurements["optimized"]["copy_calls"] < measurements["baseline"]["copy_calls"]
        assert measurements["optimized"]["copy_bytes"] < measurements["baseline"]["copy_bytes"]
    else:
        # A contiguous copy can lower inline; the runtime counter then has no
        # observation of its traffic. Check actual LLVM sites separately.
        assert measurements["optimized"]["llvm_copy_sites"] < measurements["baseline"]["llvm_copy_sites"]
    (tmp_path / "measurements.json").write_text(json.dumps(measurements, indent=2))


def test_mutated_emitter_cannot_move_arithmetic_past_an_opaque_call(monkeypatch):
    from merlin.llvmlower import insert_slice_destination

    source = fixture().replace("func.func @padded", "func.func private @touch()\nfunc.func @padded")
    source = source.replace(
        ' %result = "tensor.insert_slice"', ' func.call @touch() : () -> ()\n %result = "tensor.insert_slice"')
    before = parse_mlir_text(source)
    after = before.clone()
    monkeypatch.setattr(insert_slice_destination, "_safe_source", lambda *args: True)
    assert len(insert_slice_destination.rewrite_module(after, initialize_border_only=True)) == 1
    with pytest.raises(ValueError, match="execution barrier"):
        verify_destination_rewrite(before, after)


def test_negative_zero_fill_mutation_is_not_hidden_by_numeric_attribute_equality():
    before = parse_mlir_text(program((2, 3), (4, 5), (1, 1), "f32", "-0.0"))
    after = before.clone()
    assert len(rewrite_module(after, initialize_border_only=True)) == 1
    constant = next(op for op in after.walk()
                    if isinstance(op, arith.ConstantOp) and isinstance(op.value, builtin.FloatAttr)
                    and op.value.value.data == 0)
    constant.properties["value"] = builtin.FloatAttr(0.0, builtin.f32)
    with pytest.raises(ValueError, match="unmodeled"):
        verify_destination_rewrite(before, after)
