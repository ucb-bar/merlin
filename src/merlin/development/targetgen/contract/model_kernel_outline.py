"""Outline exact integer contractions from a captured model as standalone interface kernels.

This is a model-to-kernel *slice*, not a whole-model lowering: producer SSA values
become explicit kernel arguments, and no host/device transfer or dispatch is implied.
The selected target's OOT compiler must still lower and execute each interface.
"""

from __future__ import annotations

import hashlib
from typing import Any

import yaml

from merlin.common import mlir_query as mq
from merlin.development.targetgen.application_inventory import exact_int_mm_generic_operation
from merlin.development.targetgen.contract.interface_emit import emit_interface_mlir
from merlin.development.targetgen.contract.model_stitching import stitching_inventory
from merlin.development.targetgen.software_spec import admit_operation, validate_software_spec

SCHEMA = "merlin.model_kernel_outline.v1"
_REQUIRED_CLASS_FEATURES = frozenset({"resident_packed_tensor", "accumulator_commit", "command_buffer"})


def _zero_initializer(value: Any) -> bool:
    """The interface MATMUL starts at zero; reject unknown accumulation inputs."""
    splat = getattr(value, "owner", None)
    if splat is None or mq.op_name(splat) != "tensor.splat" or len(splat.operands) != 1:
        return False
    constant = getattr(splat.operands[0], "owner", None)
    if constant is None or mq.op_name(constant) != "arith.constant":
        return False
    literal = constant.properties.get("value", constant.attributes.get("value"))
    scalar = getattr(getattr(literal, "value", None), "data", None)
    return type(scalar) is int and scalar == 0


def outline_integer_matmuls(model: bytes, *, target: str, software_spec: bytes, capability_contract: bytes) -> dict:
    """Return diagnostic kernel MLIR and typed SSA bindings for exact i8×i8→i32 ops.

    A selected same-target SW declaration and resident command-buffer class are required.
    Unknown SW admission remains an inspectable candidate, never a support verdict.
    """
    if not target or not target.isidentifier():
        raise ValueError("target must be a nonempty identifier")
    if not isinstance(software_spec, bytes) or not isinstance(capability_contract, bytes):
        raise ValueError("selected software spec and capability contract must be exact bytes")
    spec = validate_software_spec(yaml.safe_load(software_spec), target=target)
    contract = yaml.safe_load(capability_contract)
    if not isinstance(contract, dict) or contract.get("name") != target:
        raise ValueError("selected capability contract must match the target")
    features = contract.get("features")
    if not isinstance(features, list) or any(not isinstance(feature, str) for feature in features):
        raise ValueError("selected capability contract requires a features list")
    class_missing = sorted(_REQUIRED_CLASS_FEATURES - set(features))
    if contract.get("family") != "tensor_resident":
        class_missing.insert(0, "tensor_resident family")
    digest = hashlib.sha256(model).hexdigest()
    module = mq.parse(model.decode("utf-8"))
    operations = list(mq.walk(module))
    identities = {id(op): f"mlir:{digest}:{ordinal}" for ordinal, op in enumerate(operations)}
    result_owners = {
        value: {
            "producer_operation_id": identities[id(op)],
            "result_index": index,
            "source_value_id": f"{identities[id(op)]}:result:{index}",
        }
        for op in operations
        for index, value in enumerate(op.results)
    }
    candidates, refused = [], []
    for ordinal, op in enumerate(operations):
        if mq.op_name(op) != "linalg.generic" or mq.attr_str(op, "prov.op") != "int_matmul":
            continue
        operation_id = identities[id(op)]
        if not exact_int_mm_generic_operation(op):
            refused.append({"operation_id": operation_id, "reason": "not the exact signed i8×i8→i32 reduction"})
            continue
        if mq.op_name(op.parent_op()) != "func.func":
            refused.append({"operation_id": operation_id, "reason": "contraction is nested in control flow"})
            continue
        if not _zero_initializer(op.operands[2]):
            refused.append({"operation_id": operation_id, "reason": "accumulator initializer is not proven zero"})
            continue
        left, right = (mq.type_shape_dtype(value.type)[0] for value in op.operands[:2])
        output = mq.type_shape_dtype(op.results[0].type)[0]
        if output != [left[0], right[1]]:
            refused.append({"operation_id": operation_id, "reason": "result shape disagrees with contraction"})
            continue
        if class_missing:
            refused.append(
                {"operation_id": operation_id, "reason": "selected contract lacks " + ", ".join(class_missing)}
            )
            continue
        # Ranked tensor SSA establishes logical value shape and dtype, not the
        # bufferization result's strides, output overlap, or the target kernel's
        # valid-window tail behavior.  Do not fill layouts/tails/aliasing with
        # guesses here: those are separate Phase 1 compiler/shim obligations,
        # and this Phase 0 SW screen must stay unknown until reviewed evidence
        # closes them.  ``staged_admission`` records a development plan without
        # changing this admission decision.
        admission = admit_operation(
            spec,
            "linalg.generic",
            {
                "family": "contraction",
                "operand_dtype": "i8",
                "accum_dtype": "i32",
                "readout_dtype": "i32",
                "rank": 2,
                "broadcasting": "none",
                "epilogues": [],
                "dimensions": {"M": left[0], "K": left[1], "N": right[1]},
            },
            "accelerator",
        )
        if admission["status"] == "unsupported":
            refused.append({"operation_id": operation_id, "reason": admission["reason"]})
            continue
        command_buffer = {
            "abi_version": "0.1",
            "target": target,
            "tensors": {
                # The resident-matmul pointer ABI is weight, lhs, output.  The
                # OOT artifact takes external tensors in interface declaration
                # order, so this order must agree with the host device shim.
                "B": {"shape": right, "dtype": "i8", "role": "input"},
                "A": {"shape": left, "dtype": "i8", "role": "input"},
            },
            "commands": [
                {
                    "opcode": "RES_PACK",
                    "operands": {"src": "B", "dst": "B_res"},
                    "attributes": {"layout": "packed_rhs"},
                },
                {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A", "rhs": "B_res", "dst": "acc"}},
                {
                    "opcode": "COMMIT",
                    "operands": {"src": "acc", "dst": "Y"},
                    "attributes": {"output_dtype": "i32", "epilogue": []},
                },
            ],
        }
        interface = emit_interface_mlir(command_buffer)

        def source(value: Any) -> dict:
            if value in result_owners:
                return result_owners[value]
            if hasattr(value, "index") and hasattr(value, "owner") and value.owner is op.parent:
                return {"source": "function_argument", "argument_index": value.index}
            return {"source": "unresolved"}

        bindings = [source(value) for value in op.operands[:2]]
        if any(binding.get("source") == "unresolved" for binding in bindings):
            refused.append(
                {
                    "operation_id": operation_id,
                    "reason": "operand source is not a direct SSA result or function argument",
                }
            )
            continue
        candidates.append(
            {
                "operation_id": operation_id,
                "ordinal": ordinal,
                "operand_bindings": bindings,
                "output_result_index": 0,
                "output_type": str(op.results[0].type),
                "software_admission": admission,
                "compiler_support": "not_evaluated",
                "interface_sha256": hashlib.sha256(interface.encode()).hexdigest(),
                "interface_mlir": interface,
            }
        )
    return {
        "schema": SCHEMA,
        "target": target,
        "model_sha256": digest,
        "software_spec_sha256": hashlib.sha256(software_spec).hexdigest(),
        "capability_contract_sha256": hashlib.sha256(capability_contract).hexdigest(),
        "interface_class": {
            "family": "tensor_resident",
            "required_features": sorted(_REQUIRED_CLASS_FEATURES),
            "status": "declared" if not class_missing else "unsupported",
            "missing": class_missing,
        },
        "candidates": candidates,
        "refused": refused,
        "stitching": stitching_inventory(module, digest, candidates),
        "qualification": (
            "diagnostic isolated integer kernels; SW admission may be unknown; "
            "no target compilation, executable model stitching, or numerical proof"
        ),
    }
