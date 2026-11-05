"""Independent source/ABI inverse for candidate zero-initializer grouping.

This private host check grants no target effects. Its caller must independently
check actual zero accumulation, products, ownership, complete outputs and issue
state. Candidate-owned transformation source is never imported here.
"""

from __future__ import annotations

import copy
import hashlib

from merlin.compile.scheduling.zero_initializer_groups import find_zero_initializer_groups
from merlin.targetgen.oot_starterkit.plan import validate_mixed_program_plan


def _parse_complete_source(text):
    from xdsl.context import Context
    from xdsl.dialects import arith, builtin, cf, func, linalg, math, memref, scf, tensor
    from xdsl.dialects.linalg.ops import GenericOp
    from xdsl.ir import Dialect

    from merlin.common.dense_float_ir import DenseFloatParser
    from merlin.common.ir_lock import IR_LOCK
    from merlin.frontends.linalg_mlir import strip_paren_results

    class CompleteParser(DenseFloatParser):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.generic_attributes = []

        def parse_optional_attr_dict(self, *args, **kwargs):
            result = super().parse_optional_attr_dict(*args, **kwargs)
            if self.generic_attributes and self.generic_attributes[-1] is None:
                self.generic_attributes[-1] = dict(result)
            return result

    class CompleteGeneric(GenericOp):
        @classmethod
        def parse(cls, parser):
            # Upstream consumes but drops unknown attributes from its first
            # custom-syntax dictionary. Retain them, including on nested ops.
            parser.generic_attributes.append(None)
            try:
                operation = super().parse(parser)
                attributes = parser.generic_attributes[-1]
                known = {"indexing_maps", "iterator_types", "doc", "library_call"}
                operation.attributes.update({name: value for name, value in attributes.items() if name not in known})
                return operation
            finally:
                parser.generic_attributes.pop()

    context = Context(allow_unregistered=True)
    complete_linalg = Dialect(
        linalg.Linalg.name,
        [CompleteGeneric if operation is GenericOp else operation for operation in linalg.Linalg.operations],
        list(linalg.Linalg.attributes),
    )
    for dialect in (
        builtin.Builtin,
        func.Func,
        arith.Arith,
        complete_linalg,
        tensor.Tensor,
        scf.Scf,
        math.Math,
        cf.Cf,
        memref.MemRef,
    ):
        context.load_dialect(dialect)
    with IR_LOCK:
        return CompleteParser(context, strip_paren_results(text)).parse_module()


def _logical(command_buffer):
    result = copy.deepcopy(command_buffer)
    plan = result["params"]["global_program_plan"]
    for name in ("schedule_instruction_count", "prologue_instruction_range", "epilogue_instruction_range"):
        plan.pop(name, None)
    for task in plan["tasks"]:
        task.pop("instruction_start", None)
        task.pop("instruction_end", None)
    for row in result["params"].get("lane_placement", ()):
        # Explanatory prose is not effect or ownership authority.
        row.pop("reason", None)
    return result


def check_zero_initializer_grouping(source_text, before, after):
    """Invert only exact current-source private zero seed virtualizations.

    Task instruction spans may change, with their actual coverage independently
    checked by the final program verifier. All source operations, commands,
    tensor types, required operands and complete ordered outputs are retained.
    Policy may select a subset of legal source groups. Unrelated changes refuse.
    """
    if type(source_text) is not str or any(type(cb) is not dict for cb in (before, after)):
        raise ValueError("initializer inverse requires original source text and complete command buffers")
    for cb in (before, after):
        if validate_mixed_program_plan(source_text, cb).get("ok") is not True:
            raise ValueError("initializer inverse lacks complete current-source plan ownership")
    legal = {
        group.source_op_indices: group for group in find_zero_initializer_groups(_parse_complete_source(source_text))
    }
    old = before["params"]["global_program_plan"]
    new = after["params"]["global_program_plan"]
    if any(len(task["source_op_indices"]) != 1 for task in old["tasks"]):
        raise ValueError("initializer inverse needs the original separate source-operation tasks")
    chosen = {}
    for task in new["tasks"]:
        indices = tuple(task["source_op_indices"])
        if len(indices) > 1:
            if indices not in legal:
                raise ValueError("candidate grouped an unproved, shared, returned or nonzero initializer")
            chosen[indices] = legal[indices]
    virtual_positions = {position for group in chosen.values() for position in group.virtual_result_positions}
    virtual_names = {
        row["tensor"] for row in old["source_values"] if (row["op_index"], row["result_index"]) in virtual_positions
    }
    expected = copy.deepcopy(before)
    expected["tensors"] = {name: value for name, value in expected["tensors"].items() if name not in virtual_names}
    abi = expected["kernel_abi"]
    abi["args"] = [row for row in abi["args"] if row["tensor"] not in virtual_names]
    if "compiler_pointer_abi" in expected:
        arguments = expected["compiler_pointer_abi"]["arguments"]
        expected["compiler_pointer_abi"]["arguments"] = [name for name in arguments if name not in virtual_names]
    plan = expected["params"]["global_program_plan"]
    plan["source_values"] = [
        row for row in plan["source_values"] if (row["op_index"], row["result_index"]) not in virtual_positions
    ]
    absorbed = {
        index for group in chosen.values() for index in group.source_op_indices if index != group.contraction_ordinal
    }
    contractions = {group.contraction_ordinal: group for group in chosen.values()}
    original_tasks = {task["source_op_indices"][0]: task for task in plan["tasks"]}
    tasks = []
    for task in plan["tasks"]:
        ordinal = task["source_op_indices"][0]
        if ordinal in absorbed:
            continue
        if ordinal in contractions:
            group = contractions[ordinal]
            task["source_op_indices"] = list(group.source_op_indices)
            task["reads"] = [name for name in task["reads"] if name not in virtual_names]
            if "source_region_ids" in task:
                task["source_region_ids"] = list(
                    dict.fromkeys(
                        name
                        for index in group.source_op_indices
                        for name in original_tasks[index].get("source_region_ids", ())
                    )
                )
        task["task_index"] = len(tasks)
        tasks.append(task)
    plan["tasks"] = tasks
    grouped_indices = {index for group in chosen.values() for index in group.source_op_indices}
    for row in expected["params"].get("lane_placement", ()):
        if row["source_op_index"] in grouped_indices:
            row["lane"] = "accelerator"
    if _logical(expected) != _logical(after):
        raise ValueError("initializer grouping changed unrelated source work, types, ABI, tasks or complete outputs")
    return {
        "schema": "merlin.zero_initializer_program_inverse.v1",
        "status": "verified_source_initializer_grouping",
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "groups": [group.record() for group in chosen.values()],
        "virtual_tensor_names": sorted(virtual_names),
        "logical_initializer_bytes_removed": sum(group.initializer_bytes for group in chosen.values()),
        "scope": "source/task/ABI inverse only; actual target effects and complete outputs required independently",
        "timing_authority": "NONE",
    }
