"""Retain explicit program-local static refusals without skipping sibling checks.

Only typed semantic refusals may continue. A changed input, route, recipe or
product, an unexpected exception and an interrupted process keep their original
failure path. No failed program gains linked, numerical or full-model credit.
"""

from pathlib import Path
from time import monotonic

from merlin.compile.model_execution_inputs import file_sha256
from merlin.verify.host_compute_audit import HostComputeVeto

VERIFIED = "capture_lower_codegen_link_verified"
FAILED = "program_static_checks_failed"


class ProgramCheckFailure(ValueError):
    """One explicitly identified semantic checker refused this program."""

    def __init__(self, message, *, check):
        super().__init__(message)
        self.check = check


def verify(checker, receipt, *, revalidate, **arguments):
    """Run the ordinary checker; continue only after independent identity rechecks."""
    started = monotonic()
    try:
        result = checker(receipt, **arguments)
    except (ProgramCheckFailure, HostComputeVeto) as error:
        # Identity/route exceptions are intentionally not caught. A semantic
        # refusal also cannot mask simultaneous shared-input/product mutation.
        revalidate()
        from .private_compilation_inputs import verify as verify_compilation

        output = receipt.get("output") or {}
        elf = Path(str(output.get("elf") or ""))
        if elf.is_symlink() or not elf.is_file() or file_sha256(elf) != output.get("elf_sha256"):
            raise ValueError("linked program changed during semantic verification") from error
        verify_compilation(output, elf, required=arguments.get("require_compilation_recipe", True))
        return {
            "program": arguments["program"],
            "status": FAILED,
            "linked_device_groups": 0,
            "check_stage": "linked_program_checks",
            "check": error.check if isinstance(error, ProgramCheckFailure) else "host_compute",
            "reason": f"{type(error).__name__}: {error}",
            "observed_elf": {"path": str(elf), "sha256": output["elf_sha256"]},
            "check_elapsed_s": monotonic() - started,
            "scope": "program-local static refusal; no linked or numerical credit",
        }
    return {**result, "check_stage": "linked_program_checks", "check_elapsed_s": monotonic() - started}


def status(programs):
    return VERIFIED if programs and all(program.get("status") == VERIFIED for program in programs) else FAILED


def require_complete(programs):
    if status(programs) != VERIFIED:
        failed = [program.get("program") for program in programs if program.get("status") != VERIFIED]
        raise ProgramCheckFailure(f"independent program checks failed: {failed}", check="program_roster")


def progress_stage(program):
    return "program_receipt_verified" if program.get("status") == VERIFIED else "program_checks_failed"
