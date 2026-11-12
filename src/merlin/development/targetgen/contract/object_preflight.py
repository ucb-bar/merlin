"""Selected emitted-module ABI and measured entry-frame preflight."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path


def _module_target_abi(llvm_text: str) -> str | None:
    """Read the LLVM module's target-abi flag, without treating IR as ABI authority."""
    references: list[str] = []
    definitions: dict[str, str] = {}
    for line in llvm_text.splitlines():
        line = line.strip()
        if line.startswith("!llvm.module.flags = !{"):
            if references:
                raise ValueError("LLVM module has duplicate module-flag tables")
            payload = line.partition("!{")[2].removesuffix("}")
            references = [item.strip() for item in payload.split(",")]
        elif line.startswith("!") and " = !{" in line:
            name, _, payload = line.partition(" = !{")
            definitions[name] = payload.removesuffix("}")
    found: list[str] = []
    for reference in references:
        operands = [part.strip() for part in definitions.get(reference, "").split(",")]
        if len(operands) > 1 and operands[1] == '!"target-abi"':
            if len(operands) != 3 or not operands[2].startswith('!"') or not operands[2].endswith('"'):
                raise ValueError("LLVM module has malformed target-abi flag")
            found.append(operands[2][2:-1])
    if len(found) > 1:
        raise ValueError("LLVM module has ambiguous target-abi flags")
    return found[0] if found else None


def _abi_receipt(workdir: Path, obj: Path, abi: str) -> None:
    """Bind the object to the ABI selected for its harness in a paired build."""
    (workdir / "kernel.abi.json").write_text(
        json.dumps(
            {
                "schema": "merlin_kernel_abi_v1",
                "abi": abi,
                "object_sha256": hashlib.sha256(obj.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _repair_oversized_frame(
    llvm_path, object_path, *, recipe, policy, extra, remaining: Callable[[], float | None] | None = None
):
    """Seat the entry frame's static temporaries in one arena and RE-MEASURE.

    Returns ``(object, measurement, llvm_path, report_path, report)``, or ``None`` when the repair
    is unavailable or did not actually fit. The measurement is taken on the REBUILT object; a repair
    that is believed rather than measured is how an over-budget frame reaches a device.

    On ``None`` the caller's original refusal stands, and it is deliberately the diagnostic the
    caller sees: the emitted program is what was asked about, and a second diagnostic describing a
    repaired variant would name a program nobody requested.
    """
    from merlin.compiler.lower.model import codegen as _codegen
    from merlin.compiler.lower.model.stack_arena import StackArenaError, bind_stack_arena
    from merlin.development.targetgen.contract.stack_usage import StackFramePreflightError, measure_entrypoint

    llvm_path, object_path = Path(llvm_path), Path(object_path)
    try:
        rewritten, report = bind_stack_arena(llvm_path.read_text(encoding="utf-8"), entry_symbol=policy.entry_symbol)
    except StackArenaError:
        return None
    if not report.n_bound:
        return None
    arena_llvm = llvm_path.with_suffix(".arena.ll")
    arena_object = object_path.with_suffix(".arena.o")
    arena_su = arena_object.with_suffix(".su")
    arena_llvm.write_text(rewritten, encoding="utf-8")
    for stale in (arena_object, arena_su, arena_object.with_suffix(".measured.su")):
        stale.unlink(missing_ok=True)
    compile_kwargs = {"extra_flags": (*extra, "-fstack-usage")}
    if remaining is not None:
        compile_kwargs["timeout_s"] = remaining()
    rebuilt = Path(_codegen.compile_ll(arena_llvm, arena_object, "riscv", **compile_kwargs))
    try:
        measurement = measure_entrypoint(
            arena_su, llvm_path=arena_llvm, entry_symbol=policy.entry_symbol, max_static_bytes=policy.max_static_bytes
        )
    except StackFramePreflightError:
        return None
    # The arena's bytes are NOT free: they are `.bss` in the same image, and a large one narrows the
    # PC-relative window a medany build has to work in. Reported through the receipt, never absorbed.
    return rebuilt, measurement, arena_llvm, arena_su, report.to_dict()


# Preserve historical callable and record pickle lookup identities.
_module_target_abi.__module__ = "merlin.development.targetgen.contract.compile"
_abi_receipt.__module__ = "merlin.development.targetgen.contract.compile"
_repair_oversized_frame.__module__ = "merlin.development.targetgen.contract.compile"
