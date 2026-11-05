"""Actual upstream wrapper data, conditional on explicit original declarations.

Reading prior immutable public synthetic products exercises the structural
checker only. Fresh invocation custody and byte layout are separate controls.
"""

import hashlib
import os
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.llvmlower.descriptor_contract import (
    DescriptorLimits,
    OriginalDescriptorSource,
    TensorStorageDeclaration,
)
from merlin.llvmlower.descriptor_wrapper import inspect_descriptor_wrapper
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor
from merlin.targetgen.contract.linalg_iface import make_linalg_context

LIMITS = DescriptorLimits(100000, 100000, 64, 128, 8, 10000, 100000)


def original(tmp_path, case):
    from xdsl.dialects import func
    from xdsl.parser import Parser

    selected = os.environ.get("MERLIN_TEST_LAYOUT_SOURCE_ROOT")
    if not selected:
        pytest.skip("explicit original public synthetic upstream source products are required")
    root = Path(selected).absolute() / case / "ordinary"
    raw = (root / "model.mlir").read_bytes()
    module = Parser(make_linalg_context(), raw.decode()).parse_module()
    function = next(op for op in module.body.block.ops if type(op) is func.FuncOp)
    tensors, storage = [], []
    for ordinal, typ in enumerate((*function.function_type.inputs, *function.function_type.outputs)):
        role = "input" if ordinal < len(function.function_type.inputs) else "output"
        tensor = CompileOnlyTensor(f"slot_{ordinal}", tuple(typ.get_shape()), str(typ.element_type))
        stride, strides = 1, []
        for extent in reversed(tensor.shape):
            strides.append(stride)
            stride *= extent
        tensors.append(tensor)
        # Explicit test caller choice, not a production contiguity default.
        storage.append(TensorStorageDeclaration(tensor.name, role, 0, tuple(reversed(strides)), stride, 16))
    inputs = len(function.function_type.inputs)
    source = tmp_path / "original.mlir"
    source.write_bytes(raw)
    declaration = OriginalDescriptorSource(
        source,
        hashlib.sha256(raw).hexdigest(),
        function.sym_name.data,
        "_mlir_ciface_" + function.sym_name.data,
        CompileOnlySourceAbi(tuple(tensors[:inputs]), tuple(tensors[inputs:])),
        "mlir_ranked_memref_ciface_v1",
        tuple(storage),
        "disjoint",
    )
    modules = tuple(root.glob("llvm-dialect-*/module.mlir"))
    assert len(modules) == 1
    return declaration, modules[0].read_bytes()


@pytest.mark.parametrize("case", ("scalar_integer", "round_tail", "integer_rectangle"))
def test_actual_rank_zero_tail_and_rectangle_wrappers_preserve_complete_fields(tmp_path, case):
    source, emitted = original(tmp_path, case)
    result = inspect_descriptor_wrapper(source=source, llvm_dialect=emitted, limits=LIMITS)
    assert len(result["slots"]) == 2
    assert [row["name"] for row in result["slots"]] == [slot.name for slot in source.storage]
    for row in result["slots"]:
        rank = len(row["shape"])
        assert len(row["fields"]) == 3 + 2 * rank
        assert [field["role"] for field in row["fields"]] == (
            ["allocated", "aligned", "offset"]
            + [f"size_{axis}" for axis in range(rank)]
            + [f"stride_{axis}" for axis in range(rank)]
        )
    assert "implementation storage/index/effects remain unproved" in result["scope"]


@pytest.mark.parametrize("defect", ("call_swap", "offset_type", "extra_load", "output_omission", "tail", "source"))
def test_actual_order_type_full_source_and_output_mutations_refuse(tmp_path, defect):
    source, raw = original(tmp_path, "integer_rectangle")
    text = raw.decode()
    if defect == "call_swap":
        text = text.replace('"llvm.call"(%1, %2,', '"llvm.call"(%2, %1,')
    elif defect == "offset_type":
        text = text.replace("array<2 x i64>", "array<2 x i32>")
    elif defect == "extra_load":
        text = text.replace(
            '"llvm.return"() : () -> ()\n  }) {llvm.emit_c_interface} : () -> ()\n})',
            '%extra = "llvm.load"(%arg0) <{ordering = 0 : i64}> : (!llvm.ptr) -> i64\n'
            '    "llvm.return"() : () -> ()\n  }) {llvm.emit_c_interface} : () -> ()\n})',
        )
    elif defect == "output_omission":
        source = replace(
            source, original_abi=CompileOnlySourceAbi(source.original_abi.inputs, ()), storage=source.storage[:1]
        )
    elif defect == "tail":
        slot = source.original_abi.outputs[0]
        source = replace(
            source, original_abi=CompileOnlySourceAbi(source.original_abi.inputs, (replace(slot, shape=(3, 3)),))
        )
    else:
        source.path.write_bytes(source.path.read_bytes() + b"\n// changed original\n")
    assert defect not in {"call_swap", "offset_type", "extra_load"} or text != raw.decode()
    with pytest.raises(ValueError):
        inspect_descriptor_wrapper(source=source, llvm_dialect=text.encode(), limits=LIMITS)


@pytest.mark.parametrize("defect", ("capacity", "stride", "offset", "alignment", "missing", "boolean", "abi"))
def test_missing_or_malformed_original_storage_has_no_inferred_defaults(tmp_path, defect):
    source, emitted = original(tmp_path, "integer_rectangle")
    slot = source.storage[1]
    updates = {
        "capacity": {"capacity_elements": 11},
        "stride": {"element_strides": (1, 1)},
        "offset": {"offset_elements": 1},
        "alignment": {"alignment": 3},
        "boolean": {"capacity_elements": True},
    }
    if defect == "missing":
        source = replace(source, storage=source.storage[:1])
    elif defect == "abi":
        source = replace(source, descriptor_abi="unselected")
    else:
        source = replace(source, storage=(source.storage[0], replace(slot, **updates[defect])))
    with pytest.raises(ValueError):
        inspect_descriptor_wrapper(source=source, llvm_dialect=emitted, limits=LIMITS)


def test_reordered_extraction_definitions_keep_actual_call_semantics(tmp_path):
    source, raw = original(tmp_path, "scalar_integer")
    lines = raw.decode().splitlines()
    first = next(i for i, line in enumerate(lines) if '%1 = "llvm.extractvalue"' in line)
    lines[first], lines[first + 1] = lines[first + 1], lines[first]
    result = inspect_descriptor_wrapper(source=source, llvm_dialect="\n".join(lines).encode(), limits=LIMITS)
    assert result["slots"][0]["fields"][0]["role"] == "allocated"


def test_bounded_lexical_admission_precedes_aggregate_parser(tmp_path):
    source, emitted = original(tmp_path, "scalar_integer")
    with pytest.raises(ValueError, match="byte bound"):
        inspect_descriptor_wrapper(source=source, llvm_dialect=emitted, limits=replace(LIMITS, source_bytes=100))
    # Dense payloads are not an admitted descriptor wrapper representation.
    with pytest.raises(ValueError, match="aggregate literals"):
        inspect_descriptor_wrapper(
            source=source, llvm_dialect=b"builtin.module {dense<0> : tensor<999999xi64>}", limits=LIMITS
        )
