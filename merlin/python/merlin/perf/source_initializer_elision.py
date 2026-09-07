"""Separate source-to-command-buffer proof for an implicit integer zero seed.

This does not repair a candidate plan, prove machine instructions, or qualify
numerical execution. Declared source coverage remains an independent fact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from xdsl.dialects.builtin import IntegerAttr
from xdsl.ir import Operation

from merlin.common.paths import merlin_dir
from merlin.frontends.linalg_mlir import parse_mlir_text
from .source_contraction_witness import _contract
from .source_program_pair import document_digest, source_owners, text_digest


def prove_implicit_zero_initializer(*, source_text: str, command_buffer: dict,
                                    source_op_index: int) -> dict:
    """Recognize only a closed constant/fill/MAC source with a fresh CB result.

    No caller-provided list of waived source indices is accepted. The missing
    indices are derived from SSA uses, source order and the declared task plan.
    """
    result = {"schema": "implicit_zero_initializer_evidence_v1", "status": "UNKNOWN",
        "source_sha256": text_digest(source_text), "command_buffer_sha256": document_digest(command_buffer),
        "declared_plan_modified": False, "emitted_machine_initialization": "UNPROVEN",
        "numeric_equivalence": "UNPROVEN", "scope": "source initializer to abstract command-buffer zero seed only"}
    try:
        if len(source_text.encode()) > 256*1024 or type(source_op_index) is not int:
            raise ValueError("implicit initializer proof needs a bounded explicit source operation")
        module = parse_mlir_text(source_text)
        module.verify()
        functions = list(module.body.block.ops)
        if len(functions) != 1 or functions[0].name != "func.func" or len(functions[0].body.blocks) != 1:
            raise ValueError("initializer proof requires one closed single-block short source")
        block = functions[0].body.block
        ops = [op for op in block.ops if op.name != "func.return"]
        if len(ops) != 3 or not 0 <= source_op_index < len(ops):
            raise ValueError("only constant/fill/contraction source is supported")
        selected = ops[source_op_index]
        _, (m, k, n) = _contract(selected)
        if m*k*n > 20000:
            raise ValueError("initializer admission is only for a bounded short contraction")
        fill = selected.operands[2].owner
        if (not isinstance(fill, Operation) or fill.name != "linalg.fill" or fill not in ops
                or len(fill.operands) != 2 or len(fill.results) != 1
                or fill.results[0] != selected.operands[2]
                or fill.operands[1].type != selected.results[0].type):
            raise ValueError("source contraction has no matching complete fill initializer")
        scalar = list(fill.regions[0].block.ops)
        if (len(fill.regions) != 1 or len(fill.regions[0].blocks) != 1 or len(scalar) != 1
                or scalar[0].name != "linalg.yield" or len(scalar[0].operands) != 1
                or scalar[0].operands[0] != fill.regions[0].block.args[0]):
            raise ValueError("fill scalar semantics are not a complete overwrite")
        zero = fill.operands[0].owner
        value = zero.properties.get("value") if isinstance(zero, Operation) else None
        if (zero not in ops or zero.name != "arith.constant" or not isinstance(value, IntegerAttr)
                or value.value.data != 0 or str(value.type) != str(selected.results[0].type.get_element_type())):
            raise ValueError("initializer is not an exact same-width integer zero constant")
        if (set(ops) != {zero, fill, selected}
                or len(tuple(fill.results[0].uses)) != 1
                or any(use.operation is not selected or use.index != 2 for use in fill.results[0].uses)
                or len(tuple(zero.results[0].uses)) != 1
                or any(use.operation is not fill or use.index != 0 for use in zero.results[0].uses)):
            raise ValueError("zero initializer has another consumer or source work")
        returns = [op for op in block.ops if op.name == "func.return"]
        if len(returns) != 1 or list(returns[0].operands) != [selected.results[0]]:
            raise ValueError("source exposes a different result or an initializer observer")
        if any(value not in block.args for value in selected.operands[:2]):
            raise ValueError("contraction input SSA roots are not explicit short-source boundary operands")
        plan = command_buffer["params"]["global_program_plan"]
        if plan["source_sha256"] != text_digest(source_text) or plan["source_op_count"] != len(ops):
            raise ValueError("source plan is not bound to exact initializer source")
        owners = source_owners(plan)
        task = owners.get(source_op_index)
        missing = sorted(set(range(len(ops)))-set(owners))
        implicit = sorted((ops.index(zero), ops.index(fill)))
        if (missing != implicit or len(plan["tasks"]) != 1 or task is None
                or task.get("kind") != "contraction" or task["source_op_indices"] != [source_op_index]):
            raise ValueError("missing ownership is not exclusively this exact initializer chain")
        entries = plan["entry_bindings"]
        abi = command_buffer["kernel_abi"]
        if (len(entries) != len(block.args) or len(set(entries)) != len(entries)
                or abi["kind"] != "whole_program"):
            raise ValueError("source boundary lacks an unambiguous whole-program ABI mapping")
        lhs, rhs = (entries[list(block.args).index(v)] for v in selected.operands[:2])
        values = [row for row in plan["source_values"]
                  if row["op_index"] == source_op_index and row["result_index"] == 0]
        if (len(values) != 1 or plan["output_bindings"] != [values[0]["tensor"]]
                or abi["outputs"] != plan["output_bindings"] or task["reads"] != [lhs, rhs]):
            raise ValueError("source operand/result identity differs from its selected task")
        destination = values[0]["tensor"]
        commands = command_buffer["commands"]
        if len(commands) != 3 or [c["opcode"] for c in commands] != ["RES_PACK", "MATMUL_RESIDENT", "COMMIT"]:
            raise ValueError("unsupported command chain or extra accumulator users")
        pack, matmul, commit = commands
        pack_ops, mm_ops, commit_ops = (c["operands"] for c in commands)
        if (set(pack_ops) != {"src", "dst"} or pack_ops["src"] != rhs
                or pack.get("attributes", {}) != {"layout": "packed_rhs"}
                or set(mm_ops) != {"lhs", "rhs", "dst"} or mm_ops["lhs"] != lhs
                or mm_ops["rhs"] != pack_ops["dst"] or matmul.get("attributes", {})
                or mm_ops["dst"] in set(entries) | set(command_buffer["tensors"]) | {pack_ops["dst"]}
                or set(commit_ops) != {"src", "dst"} or commit_ops["src"] != mm_ops["dst"]):
            raise ValueError("command contraction does not create a fresh, uninitialized-by-input accumulator")
        attrs = commit.get("attributes", {})
        if (set(attrs) != {"epilogue", "output_dtype"} or attrs["epilogue"] != []
                or commit_ops["dst"] not in task["writes"]):
            raise ValueError("unsupported command accumulator readout or epilogue")
        temporary = task.get("accumulator_temporary")
        if commit_ops["dst"] != destination and (commit_ops["dst"] != temporary
                or not any(row.get("tensor") == temporary and row.get("purpose") == "accumulator_readout"
                           and row.get("source_op_index") == source_op_index for row in plan.get("compiler_temporaries", []))):
            raise ValueError("command output is not the declared source result/readout temporary")
        # The shared tensor implementation defines MATMUL as a newly initialized
        # dot product; this records the contract implementation, never executes it.
        semantics = merlin_dir()/"python/merlin/runtime/tensor.py"
        result.update(status="proved", implicit_source_op_indices=implicit,
            owning_source_op_index=source_op_index, owning_task_index=task["task_index"],
            source_inputs={"lhs": lhs, "rhs": rhs}, command_indices=[0, 1, 2],
            fresh_accumulator=mm_ops["dst"], command_semantics="fresh MATMUL_RESIDENT dot product, no prior accumulator operand",
            shared_semantics_source={"path": str(semantics), "sha256": hashlib.sha256(semantics.read_bytes()).hexdigest()},
            declared_source_plan_coverage="still incomplete; this is separate host-derived implicit ownership")
    except (ValueError, KeyError, TypeError, AttributeError, IndexError, StopIteration) as error:
        result.update(reason=type(error).__name__+": "+str(error))
    return result


def numerical_probe_admission(*, source_text: str, lowered_text: str, command_buffer: dict,
                              declared_proof: dict, candidate_sha256: str, source_op_index: int) -> dict:
    """A separate, short-only numerical probe admission; declared plan stays refused."""
    from .compiler_plan_evidence import verify_compiler_global_plan
    # Proofs are JSON documents. Normalize task-index object keys before hashing
    # so a receipt round trip cannot change their sort order (e.g. -2 and -1).
    declared_proof = json.loads(json.dumps(declared_proof, allow_nan=False))
    evidence = prove_implicit_zero_initializer(source_text=source_text, command_buffer=command_buffer,
                                               source_op_index=source_op_index)
    result = {"schema": "short_initializer_execution_admission_v1", "status": "UNKNOWN",
        "source_sha256": text_digest(source_text), "lowered_sha256": text_digest(lowered_text),
        "command_buffer_sha256": document_digest(command_buffer), "candidate_sha256": candidate_sha256,
        "declared_proof_sha256": document_digest(declared_proof), "initializer_closure": evidence,
        "declared_plan_modified": False, "declared_plan_verified": False,
        "target_route_verified": False, "numeric_equivalence": "UNPROVEN", "global_cost_calibration": False,
        "scope": "permission to independently test one bounded complete source program; not a plan or numerical pass"}
    if evidence["status"] != "proved":
        result["reason"] = "initializer source-to-command closure was not proved"
        return result
    if len(lowered_text.encode()) > 2*1024*1024:
        result["reason"] = "short LLVM exceeds bounded verifier input"
        return result
    actual = verify_compiler_global_plan(source_text=source_text, lowered_text=lowered_text,
        command_buffer=command_buffer, candidate_sha256=candidate_sha256,
        command_buffer_sha256=document_digest(command_buffer))
    actual = json.loads(json.dumps(actual, allow_nan=False))
    expected_problem = "source operations are not fully covered: " + str(evidence["implicit_source_op_indices"])
    if (document_digest(actual) != document_digest(declared_proof) or actual.get("status") != "refused"
            or actual.get("problems") != [expected_problem]
            or actual.get("control_flow", {}).get("status") != "verified"):
        result["reason"] = "declared refusal has other defects, unverified CFG/ABI, or stale artifact binding"
        return result
    result.update(status="source_bound_numerical_probe",
        structural_scope="all existing short ABI/task/CFG checks retained; sole missing zero initializer source indices proved separately",
        runtime_requirements=["bounded complete source, no full model", "exact source inputs including initializer",
            "one warm and one measured invocation with appropriate input restoration",
            "independent typed output check and exact engine/ELF binding"])
    return result
