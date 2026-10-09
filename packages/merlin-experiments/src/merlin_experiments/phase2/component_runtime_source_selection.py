"""Explicit source observations on prepared diagnostics, without qualification.

Selected source readers retain complete actual original ABI/LLVM products. They
cannot change the original numeric policy, substitute a private helper default
or issue source, effect, stage or runtime roles from returned labels.
"""

from pathlib import Path

from merlin.targetgen.contract import source_observation

from . import component_runtime_controls as controls
from . import component_runtime_copy_controls as copy_controls
from . import component_runtime_copy_support as copy_support
from .contracts import StageGateError, mapping_file, sha256_file


def selection(context):
    service = context.source_observation
    if service is None:
        return None
    if type(service) is not source_observation.ExplicitSourceObservation:
        raise StageGateError("runtime source observation needs an exact explicitly selected transport")
    if context.copy_control_support is not None:
        raise StageGateError("runtime source observations cannot substitute the fixed copy-helper control selection")
    if service.target != context.build_service.target:
        raise StageGateError("runtime source observation differs from the original target")
    pins = {(Path(path), digest) for path, digest in service.source_pins}
    if not pins <= set(context.source_pins):
        raise StageGateError("runtime source observation omits original reader/source membership")
    return service.verify()


def evaluate(context, source, lowered, fixture):
    """Keep fixed controls and original numeric policy; retain selected data only."""
    try:
        diagnostic = selection(context)
        entry = context.build_service.recipe.require_kernel_stack_frame().entry_symbol
        # A selected reader observes ordinary external diagnostics. It cannot
        # replace the fixed original qualifier's source/policy controls.
        if diagnostic is not None and fixture is None:
            command_path = lowered.parent / "command_buffer.bound.json"
            proof = context.source_observation.observe(
                source=source,
                lowered_mlir=lowered,
                command_buffer=mapping_file(command_path),
                command_buffer_path=command_path,
                entry_symbol=entry,
                evidence_root=lowered.parent,
            )
            status = "diagnostic_observed"
        else:
            copy = (
                context.copy_control_support
                if fixture is not None and fixture.case_id.partition(".")[0] in copy_support.MECHANISMS
                else None
            )
            if copy is None:
                proof = controls.verify_primitive_llvm(source.read_text(), lowered.read_text(), entry_symbol=entry)
            else:
                program = copy_controls.parse_copy(source.read_text())
                if program != copy_controls.CopyProgram(copy.shape, copy.dtype, 2, (0, 1)):
                    raise ValueError("copy control changes the original selected source domain")
                proof = copy_controls.verify_copy_llvm(
                    source.read_text(), lowered.read_text(), entry_symbol=entry, callee_symbol=copy.callee_symbol
                )
                proof["helper_source"] = {"path": str(copy.helper_source), "sha256": sha256_file(copy.helper_source)}
            status = "accepted"
        capsule = mapping_file(source.parent / "capsule.yaml", yaml_file=True)
        if fixture is not None and capsule["numeric_policy"] != mapping_file(
            fixture.evidence_root / "original_policy.json"
        ):
            raise ValueError("original numeric policy was weakened")
        return {"status": status, "proof": proof}
    except ValueError as error:
        return {"status": "refused", "actual_reason": str(error)}
