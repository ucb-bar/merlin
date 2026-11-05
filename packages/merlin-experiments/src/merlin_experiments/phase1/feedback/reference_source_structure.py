"""Observe ordinary source IR without inventing its command-buffer lowering."""

from __future__ import annotations


def source_structure(raw, *, pin, reduction_lanes=None):
    from xdsl.dialects.builtin import TensorType
    from xdsl.utils.exceptions import ParseError, VerifyException

    from merlin.common.ir_lock import IR_LOCK
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.kernels.kernel_evidence import semantic_region_from_ir
    from merlin.kernels.mechanism_inference import source_mechanisms
    from merlin.perf.candidate_source import source_observation

    try:
        with IR_LOCK:
            module = parse_mlir_text(raw.decode())
            module.verify()
    except (ParseError, VerifyException) as error:
        raise ValueError("analysis_source_parse_failed") from error
    if sum(1 for _ in module.walk()) > 100000:
        raise ValueError("analysis_source_operation_budget_exhausted")

    def shape(value):
        return dict(shape=list(value.get_shape()), dtype=str(value.get_element_type())) if isinstance(
            value, TensorType) else dict(type=str(value))

    functions = [op for op in module.body.block.ops if op.name == "func.func" and op.body.blocks]
    if not functions:
        raise ValueError("analysis_source_defined_function_missing")
    try:
        region = semantic_region_from_ir(module, region_id="source-" + pin["sha256"][:24], complete=False).to_dict()
        region_status = dict(status="observed_source_structure", complete=False)
    except ValueError as error:
        region = None
        region_status = dict(status="unknown", reason=str(error)[:256])
    return dict(status="observed_source_structure", input=pin, groups=[], products=None, outputs={},
        argument_order=None, observation=source_observation(raw.decode()),
        functions=[dict(symbol=op.sym_name.data,
                        inputs=[shape(value) for value in op.function_type.inputs],
                        outputs=[shape(value) for value in op.function_type.outputs]) for op in functions],
        semantic_region=region, semantic_region_status=region_status, mechanism_observations=source_mechanisms(
            module, evidence_id="source:" + pin["sha256"], lanes=reduction_lanes),
        source_semantics="parsed_original_ir; command-buffer correspondence unknown",
        command_buffer_correspondence="unknown", numerical_equivalence="unknown", emitted_mechanism="unknown")
