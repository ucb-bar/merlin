"""Direct permuted pointwise writes retain complete arithmetic and state."""

from __future__ import annotations

import ctypes
import json
import subprocess
from math import prod

import numpy as np
import pytest
from xdsl.dialects import builtin
from xdsl.ir.affine import AffineDimExpr, AffineMap

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.automatic_destinations import prepare_automatic_destinations
from merlin.llvmlower.destination_check import verify_permutation_rewrite
from merlin.llvmlower.insert_slice_destination import rewrite_permuted_outputs
from merlin.llvmlower.optional_passes import Selection, applied


def program(shape=(2, 3), permutation=(1, 0), element="i16"):
    result = tuple(shape[i] for i in permutation)

    def typ(extents):
        return "tensor<" + "x".join([*(str(i) for i in extents), element]) + ">"

    source, output = typ(shape), typ(result)
    dims = ",".join(f"d{i}" for i in range(len(shape)))
    mapping = f"affine_map<({dims})->({dims})>"
    iters = ",".join('"parallel"' for _ in shape)
    literal = "2.0" if element.startswith("f") else "2"
    add = "arith.addf" if element.startswith("f") else "arith.addi"
    return f"""module {{func.func @forward(%x:{source},%state:{source}) -> ({output},{source}) {{
      %init=tensor.empty():{source}
      %p=linalg.generic {{indexing_maps=[{mapping},{mapping},{mapping}],iterator_types=[{iters}]}}
        ins(%x,%state:{source},{source}) outs(%init:{source}) {{
        ^bb0(%a:{element},%s:{element},%unused:{element}):
          %v={add} %a,%s:{element}
          linalg.yield %v:{element}
      }} -> {source}
      %out=tensor.empty():{output}
      %t=linalg.transpose ins(%p:{source}) outs(%out:{output}) permutation=[{",".join(map(str, permutation))}]
      %si=tensor.empty():{source}
      %next=linalg.generic {{indexing_maps=[{mapping},{mapping}],iterator_types=[{iters}]}}
        ins(%state:{source}) outs(%si:{source}) {{
        ^bb0(%s:{element},%unused:{element}):
          %two=arith.constant {literal}:{element}
          %v={add} %s,%two:{element}
          linalg.yield %v:{element}
      }} -> {source}
      return %t,%next:{output},{source}
    }}}}"""


def prepare(source, **kwargs):
    return prepare_automatic_destinations(source, **({"serial": True, "permuted_outputs": True} | kwargs))


def test_actual_coordinates_body_order_and_complete_source_are_checked():
    source = program((2, 3, 5), (1, 2, 0))
    before = parse_mlir_text(source)
    out, report = prepare(source)
    after = parse_mlir_text(out)
    assert report["rewrites"] == 1
    assert report["realized_regions"][0]["eliminated_tensor_bytes"] == 60
    assert [row["name"] for row in report["checks"]] == [
        "permutation_source_legality",
        "permutation_forwarding",
        "permutation_conformance",
    ]
    assert tuple(report["checks"][1]["invalidated"]) == ("source",)
    assert len(verify_permutation_rewrite(before, after, index_bits=64)) == 1
    assert not any(op.name == "linalg.transpose" for op in after.walk())
    assert prepare(out)[1]["rewrites"] == 0
    assert str(before) == str(parse_mlir_text(source))
    original = next(op for op in before.walk() if op.name == "linalg.generic")
    actual = next(op for op in after.walk() if op.name == "linalg.generic")
    assert original.body.is_structurally_equivalent(actual.body)
    assert original.indexing_maps.data[:-1] == actual.indexing_maps.data[:-1]
    assert original.iterator_types == actual.iterator_types


def test_chained_transposes_do_not_rewrite_a_generated_writer_again():
    source = (
        program((2, 3), (0, 1))
        .replace("-> (tensor<2x3xi16>,tensor<2x3xi16>)", "-> (tensor<3x2xi16>,tensor<2x3xi16>)")
        .replace(
            "%si=tensor.empty()",
            """%e2=tensor.empty():tensor<3x2xi16>
        %t2=linalg.transpose ins(%t:tensor<2x3xi16>) outs(%e2:tensor<3x2xi16>) permutation=[1,0]
        %si=tensor.empty()""",
        )
        .replace("return %t,%next:tensor<2x3xi16>", "return %t2,%next:tensor<3x2xi16>")
    )
    output, report = prepare(source)
    assert report["rewrites"] == 1
    assert len(verify_permutation_rewrite(parse_mlir_text(source), parse_mlir_text(output), index_bits=64)) == 1


@pytest.mark.parametrize(
    "options",
    [
        {"serial": False},
        {"max_added_operations": 1},
        {"max_metadata_items": 9},
        {"index_bits": 4},
    ],
)
def test_unavailable_family_and_exact_capacity_boundaries_retain_source(options):
    source = program()
    assert prepare(source, **options)[0] == source
    assert prepare(source, max_added_operations=2, max_metadata_items=10, index_bits=5)[1]["rewrites"] == 1
    with applied(Selection.parse("-automatic-host-permutations")):
        assert prepare(source)[0] == source


@pytest.mark.parametrize(
    "mutation", ["arithmetic", "permutation", "input", "iterator", "attribute", "extra", "initializer"]
)
def test_independent_checker_rejects_actual_output_mutations(mutation):
    source = program((2, 2, 2), (1, 2, 0))
    before = parse_mlir_text(source)
    after = before.clone()
    rewrite_permuted_outputs(after, index_bits=64, max_added_operations=2, max_metadata_items=20)
    writer = next(op for op in after.walk() if op.name == "linalg.generic")
    if mutation == "arithmetic":
        writer.body.block.last_op.operands = [writer.body.block.args[0]]
    elif mutation == "permutation":
        maps = list(writer.indexing_maps.data)
        maps[-1] = builtin.AffineMapAttr(AffineMap(3, 0, tuple(AffineDimExpr(i) for i in (2, 0, 1))))
        writer.properties["indexing_maps"] = builtin.ArrayAttr(maps)
    elif mutation == "input":
        writer.operands = [writer.inputs[1], writer.inputs[0], writer.outputs[0]]
    elif mutation == "iterator":
        maps = [
            builtin.AffineMapAttr(AffineMap(3, 0, tuple(AffineDimExpr(i) for i in order)))
            for order in ((1, 0, 2), (1, 0, 2), (0, 2, 1))
        ]
        writer.properties["indexing_maps"] = builtin.ArrayAttr(maps)
    elif mutation == "attribute":
        writer.attributes["unknown_execution"] = builtin.UnitAttr()
    elif mutation == "initializer":
        empty = next(op for op in after.walk() if op.name == "tensor.empty")
        writer.operands = [*writer.inputs, empty.tensor]
    else:
        writer.parent.insert_op_before(builtin.UnregisteredOp.with_name("opaque.touch").create(), writer)
    with pytest.raises(ValueError, match="permuted destination"):
        verify_permutation_rewrite(before, after, index_bits=64)


@pytest.mark.parametrize("mutation", ["effect", "live", "output_read", "attribute", "capture"])
def test_unknown_effects_live_observers_and_initialized_outputs_are_barriers(mutation):
    source = program()
    if mutation == "effect":
        source = source.replace("func.func @forward", "func.func private @touch()\nfunc.func @forward")
        source = source.replace("%out=tensor.empty()", "func.call @touch():()->()\n%out=tensor.empty()")
    elif mutation == "live":
        source = source.replace(
            "-> (tensor<3x2xi16>,tensor<2x3xi16>)", "-> (tensor<3x2xi16>,tensor<2x3xi16>,tensor<2x3xi16>)"
        )
        source = source.replace(
            "return %t,%next:tensor<3x2xi16>,tensor<2x3xi16>",
            "return %t,%next,%p:tensor<3x2xi16>,tensor<2x3xi16>,tensor<2x3xi16>",
        )
    elif mutation == "output_read":
        source = source.replace("%v=arith.addi %a,%s", "%v=arith.addi %a,%unused", 1)
    elif mutation == "attribute":
        source = source.replace("module {", "module attributes {unknown_execution} {", 1)
    else:
        source = source.replace(
            "%init=tensor.empty()",
            "%one=arith.constant 1:i16\n%external=arith.addi %one,%one:i16\n%init=tensor.empty()",
            1,
        )
        source = source.replace("%v=arith.addi %a,%s", "%v=arith.addi %a,%external", 1)
    assert prepare(source)[0] == source


@pytest.mark.parametrize("mutation", ["count", "library", "checker"])
def test_emitter_accounting_library_and_checker_mutations_refuse(tmp_path, monkeypatch, mutation):
    from merlin.llvmlower import destination_check, insert_slice_destination

    original = insert_slice_destination.rewrite_permuted_outputs
    if mutation == "checker":
        verify = destination_check.verify_permutation_rewrite

        def changed(before, after, **kwargs):
            result = verify(before, after, **kwargs)
            after.attributes["unknown"] = builtin.UnitAttr()
            return result

        monkeypatch.setattr(destination_check, "verify_permutation_rewrite", changed)
    else:
        owner = tmp_path / "selected.py"
        if mutation == "library":
            owner.write_text("original")
            monkeypatch.setattr(insert_slice_destination, "__file__", str(owner))

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            if mutation == "library":
                owner.write_text("mutated")
            return [] if mutation == "count" else result

        monkeypatch.setattr(insert_slice_destination, "rewrite_permuted_outputs", changed)
    with pytest.raises(ValueError, match="accounting|library changed|analysis pass.*changed"):
        prepare(program())


def test_immutable_outer_constants_keep_exact_bindings_and_words():
    source = (
        program(element="f32")
        .replace("%init=tensor.empty()", "%c=arith.constant -0.0:f32\n%init=tensor.empty()", 1)
        .replace("%v=arith.addf %a,%s", "%v=arith.addf %a,%c", 1)
    )
    before = parse_mlir_text(source)
    out, report = prepare(source)
    assert report["rewrites"] == 1
    after = parse_mlir_text(out)
    constant = next(op for op in after.walk() if op.name == "arith.constant")
    constant.properties["value"] = builtin.FloatAttr(0.0, builtin.f32)
    with pytest.raises(ValueError, match="permuted destination"):
        verify_permutation_rewrite(before, after, index_bits=64)


@pytest.mark.parametrize("typ", ["tensor<?x3xi16>", 'tensor<2x3xi16, "unknown">', "tensor<2x3xi24>"])
def test_unknown_layout_dynamic_shape_and_unqualified_storage_format_fall_back(typ):
    source = (
        program()
        .replace("%x:tensor<2x3xi16>", "%x:" + typ)
        .replace("ins(%x,%state:tensor<2x3xi16>", "ins(%x,%state:" + typ)
    )
    if "i24" in typ:
        source = source.replace("i16", "i24")
    assert prepare(source)[0] == source


def test_complete_reduction_keeps_its_original_route():
    source = """module {func.func @forward(%x:tensor<3x5xf32>,%seed:tensor<3xf32>) -> tensor<3xf32> {
    %r=linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i)>],
      iterator_types=["parallel","reduction"]} ins(%x:tensor<3x5xf32>) outs(%seed:tensor<3xf32>) {
        ^bb0(%x:f32,%acc:f32): %v=arith.addf %x,%acc:f32
        linalg.yield %v:f32
    } -> tensor<3xf32>
    %empty=tensor.empty():tensor<3xf32>
    %t=linalg.transpose ins(%r:tensor<3xf32>) outs(%empty:tensor<3xf32>) permutation=[0]
    return %t:tensor<3xf32>}}"""
    assert prepare(source)[0] == source


def test_checked_rounding_runs_before_output_permutation(tmp_path):
    from merlin.llvmlower.lower import lower_model

    source = program(element="f32")
    # The standard intrinsic is retained exactly; other opaque scalar calls
    # remain outside the accepted scalar-effect domain.
    source = source.replace(
        "%v=arith.addf %a,%s:f32",
        """%sum=arith.addf %a,%s:f32
      %v="llvm.intr.roundeven"(%sum):(f32)->f32""",
        1,
    )
    result = lower_model(source, tmp_path / "build", targets=(), ir_audit=True)
    assert result.stats["automatic_host_permutations"]["rewrites"] == 1
    assert (
        result.stats["automatic_host_permutations"]["source_file_sha256"]
        == (result.stats["automatic_host_rounding"]["output_file_sha256"])
    )
    audit = json.loads(result.audit_index.read_text())
    stages = [row["name"] for row in audit["stages"]]
    assert stages.index("host-permutation-forwarded") < stages.index("xdsl-parsed")


@pytest.mark.parametrize(
    "shape,permutation,element",
    [
        ((3, 5), (1, 0), "i8"),
        ((5, 7), (1, 0), "i16"),
        ((2, 3, 5), (1, 2, 0), "f32"),
        ((1, 3, 7), (2, 0, 1), "f64"),
    ],
)
@pytest.mark.parametrize("observe", [False, True])
def test_ordinary_native_complete_outputs_aliases_and_carried_state(tmp_path, shape, permutation, element, observe):
    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.toolchain import clang

    source = program(shape, permutation, element)
    dtype = {"i8": np.int8, "i16": np.int16, "f32": np.float32, "f64": np.float64}[element]
    support = tmp_path / "memory_observer.c"
    support.write_text("""#include <stddef.h>
#include <stdint.h>
uint64_t load_bytes, store_bytes, loads, stores;
#define OBSERVE(N) \
void __sanitizer_cov_load##N(void *p) { loads++; load_bytes += N; } \
void __sanitizer_cov_store##N(void *p) { stores++; store_bytes += N; }
OBSERVE(1) OBSERVE(2) OBSERVE(4) OBSERVE(8) OBSERVE(16)
void __sanitizer_cov_trace_pc_guard(uint32_t *p) {}
void __sanitizer_cov_trace_pc_guard_init(uint32_t *a, uint32_t *b) {}
""")
    observations = {}
    for arm, selection in [("baseline", "-automatic-host-permutations"), ("optimized", "automatic-host-permutations")]:
        with applied(Selection.parse(selection)):
            result = lower_model(source, workdir=tmp_path / arm, targets=(), vectorize=False, parallel=False)
        assert result.stats["automatic_host_permutations"]["rewrites"] == (arm == "optimized")
        image = tmp_path / (arm + ".so")
        obj = tmp_path / (arm + ".o")
        instrumentation = ["-fsanitize-coverage=trace-loads,trace-stores,trace-pc-guard"] if observe else []
        subprocess.run(
            [
                str(clang()),
                "-O0" if observe else "-O2",
                "-fPIC",
                *instrumentation,
                "-c",
                str(result.ll_path),
                "-o",
                str(obj),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                str(clang()),
                "-shared",
                "-fPIC",
                str(obj),
                str(mlir_runtime_c()),
                str(support),
                "-lm",
                "-o",
                str(image),
            ],
            check=True,
            capture_output=True,
        )
        model = HostModel.load(str(image), image_policy=PrivateHostImagePolicy(tmp_path.resolve()))
        backing = np.full(prod(shape) + 32, 99, dtype=dtype)
        values = (np.arange(prod(shape)) * 17 - 129).astype(dtype).reshape(shape)
        if element.startswith("f"):
            values /= 4
            values.flat[0] = -0.0
        state = np.zeros(shape, dtype=dtype)
        for step in range(3):
            backing[16:-16] = values.ravel()
            x = backing[16:-16].reshape(shape)
            if step == 1:
                state = x  # Read-only caller aliasing is preserved.
            old_x, old_state = x.copy(), state.copy()
            expected = np.transpose(np.add(old_x, old_state), permutation).copy()
            expected_state = np.add(old_state, np.array(2, dtype=dtype))
            destinations = [
                np.full(prod(dims) + 32, 99, dtype=dtype) for dims in (tuple(shape[i] for i in permutation), shape)
            ]
            outputs = [
                buf[16:-16].reshape(dims)
                for buf, dims in zip(destinations, (tuple(shape[i] for i in permutation), shape), strict=True)
            ]
            model([(a.ctypes.data, a.shape) for a in (x, state, *outputs)])
            for got, want in zip(outputs, [expected, expected_state], strict=True):
                np.testing.assert_array_equal(got.view(np.uint8), want.view(np.uint8))
            np.testing.assert_array_equal(x.view(np.uint8), old_x.view(np.uint8))
            np.testing.assert_array_equal(state.view(np.uint8), old_state.view(np.uint8))
            np.testing.assert_array_equal(backing[:16], np.full(16, 99, dtype=dtype))
            np.testing.assert_array_equal(backing[-16:], np.full(16, 99, dtype=dtype))
            for buf in destinations:
                np.testing.assert_array_equal(buf[:16], np.full(16, 99, dtype=dtype))
                np.testing.assert_array_equal(buf[-16:], np.full(16, 99, dtype=dtype))
            state = outputs[1].copy()
        observations[arm] = {
            "loads": ctypes.c_uint64.in_dll(model.lib, "loads").value,
            "stores": ctypes.c_uint64.in_dll(model.lib, "stores").value,
            "load_bytes": ctypes.c_uint64.in_dll(model.lib, "load_bytes").value,
            "store_bytes": ctypes.c_uint64.in_dll(model.lib, "store_bytes").value,
            "llvm_bytes": result.ll_path.stat().st_size,
        }
    if observe:
        for kind in ("load_bytes", "store_bytes"):
            assert observations["baseline"][kind] - observations["optimized"][kind] == (
                3 * prod(shape) * np.dtype(dtype).itemsize
            )
    (tmp_path / "measurements.json").write_text(json.dumps(observations, indent=2))
