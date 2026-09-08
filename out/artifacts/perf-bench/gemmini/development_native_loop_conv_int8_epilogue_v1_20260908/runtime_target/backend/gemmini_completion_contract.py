"""Host edge for the completion ABI already used by the curated target harness.

This contract identifies an existing synchronization operation; it does not re-prove the hardware
implementation or strengthen the candidate's ordering. Numerical/performance qualification remains
separate from the shared relative IR transformation proof.
"""
from __future__ import annotations

import hashlib

from merlin.perf.completion_delta import CompletionContract
from merlin.runtime.backends.base import get_backend


def derive_completion_contract() -> CompletionContract:
    backend = get_backend("gemmini")
    recipe = backend.harness_build_recipe()
    headers = [root / "include/gemmini.h" for root in recipe.include_roots
               if (root / "include/gemmini.h").is_file()]
    if len(set(headers)) != 1:
        raise ValueError("target completion ABI header is absent or ambiguous")
    header = headers[0]
    payload = header.read_bytes()
    matches = [(index + 1, line) for index, line in enumerate(payload.decode().splitlines())
               if line.startswith("#define gemmini_fence() ")]
    if len(matches) != 1:
        raise ValueError("target completion ABI macro is absent or ambiguous")
    line_number, macro = matches[0]
    body = macro.partition("#define gemmini_fence() ")[2].strip()
    if not body.startswith('asm volatile("') or not body.endswith('")'):
        raise ValueError("target completion ABI is not an explicit volatile assembly operation")
    assembly = body[len('asm volatile("'):-len('")')]
    if not assembly or '"' in assembly:
        raise ValueError("target completion ABI assembly is unsupported")

    def decoded(row):
        return row.get("class") == "FENCE"

    def operation(op, row):
        return (decoded(row) and op.name == "llvm.inline_asm"
                and op.asm_string.data == assembly and op.has_side_effects is not None
                and "~{memory}" in op.constraints.data.split(",")
                and not op.operands and not op.results)

    return CompletionContract(
        identity="curated_gemmini_completion_abi_v1",
        provenance={"header": str(header.resolve()), "header_sha256": hashlib.sha256(payload).hexdigest(),
                    "line": line_number, "macro": macro,
                    "scope": "existing device completion ABI plus explicit emitted host-memory clobber",
                    "hardware_implementation": "assumed existing target contract; not requalified here"},
        recognizes_decoded=decoded, recognizes_operation=operation)
