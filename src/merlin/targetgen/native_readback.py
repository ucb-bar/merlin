"""Revalidate a native adapter's selected full-value build without granting a tier."""

from __future__ import annotations

from pathlib import Path

from merlin.targetgen.contract import readback_policy as RB
from merlin.targetgen.contract.build_service import BuildOnlyService


def verify_build(
    *, target: str, policy: RB.ReadbackPolicy, service: BuildOnlyService,
    output: Path, cb: dict, elf: Path, expected: tuple | None = None,
    expected_receipt: dict | None = None, stage: str | None = None,
) -> tuple[dict, list[dict], dict]:
    """Preserve the adapter's initial and post-execution input/receipt checks.

    This binds selected bytes only. It does not authenticate a reference,
    execution, hardware selection or numeric agreement. The caller retains the
    surrounding source/engine/ELF checks at their original ordering boundaries.
    """
    from .native_model_execution import NativeModelExecutionError

    policy = RB.selected(policy)
    if policy is None:
        raise ValueError("native full-value build revalidation requires an explicit policy")
    recipe, pins = RB.selected_build_inputs(target, service.recipe.with_effective_abi(), service, policy=policy)
    if expected is not None and (recipe, pins) != expected:
        raise NativeModelExecutionError(f"full-value build inputs changed during {stage} execution")
    receipt = RB.require_current_build_receipt(
        cb=cb, target=target, workdir=output / "build", elf_path=elf,
        policy=policy, build_service=service,
    )
    if expected_receipt is not None and receipt != expected_receipt:
        raise NativeModelExecutionError(f"full-value build receipt changed during {stage} execution")
    return recipe, pins, receipt
