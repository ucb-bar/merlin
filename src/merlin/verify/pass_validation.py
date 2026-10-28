"""Check selected compiler passes against a Lean proof, beside the compile that runs them.

A validated pass reports the exact IR before and after it ran; the pair goes to a
:class:`~merlin.verify.speculative_gate.SpeculativeGate`, the compile carries on, and a check that
does not accept stops the compile with its evidence. A pass that changes nothing submits nothing.

``lower_model`` takes a :class:`PassValidation`, or builds one from the environment so every entry
point that lowers (``merlin-compile --validate-passes``, the whole-model backends) can turn it on:

* ``MERLIN_VALIDATE_PASSES`` -- ``all`` or a comma-separated subset of :data:`CHECKS`.
* ``MERLIN_LEAN_TOOLCHAIN_MANIFEST`` -- the ``merlin.lean_toolchain.v1`` manifest the checks use.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Validated pass -> the gate check that proves its before/after pair.
CHECKS = {
    "prune_dead_pure_tensor_ops": "pure_graph",
    "share_exact_tensor_preparations": "pure_graph",
}
ENV_PASSES = "MERLIN_VALIDATE_PASSES"
ENV_TOOLCHAIN = "MERLIN_LEAN_TOOLCHAIN_MANIFEST"


@dataclass(frozen=True)
class PassValidation:
    gate: Any
    toolchain_manifest: Path
    passes: frozenset[str] = frozenset(CHECKS)

    def __post_init__(self) -> None:
        from .speculative_gate import SpeculativeGate

        if type(self.gate) is not SpeculativeGate:
            raise ValueError("pass validation requires the host's SpeculativeGate")
        if type(self.passes) is not frozenset or not self.passes or not self.passes <= set(CHECKS):
            raise ValueError(f"validated passes must be a nonempty subset of {sorted(CHECKS)}")
        if not Path(self.toolchain_manifest).is_file():
            raise ValueError(f"Lean toolchain manifest not found: {self.toolchain_manifest}")


_ACTIVE: ContextVar[PassValidation | None] = ContextVar("merlin_pass_validation", default=None)


def from_environment(workdir: Path) -> PassValidation | None:
    """The validation the environment asks for, or None. A request without a toolchain is an error."""
    requested = os.environ.get(ENV_PASSES, "").strip()
    if not requested:
        return None
    passes = frozenset(CHECKS) if requested == "all" else frozenset(p.strip() for p in requested.split(","))
    manifest = os.environ.get(ENV_TOOLCHAIN)
    if not manifest:
        raise ValueError(f"{ENV_PASSES} is set but {ENV_TOOLCHAIN} does not select a Lean toolchain")
    from .speculative_gate import SpeculativeGate

    return PassValidation(SpeculativeGate(Path(workdir) / "pass-validation", interrupt=True), Path(manifest), passes)


@contextmanager
def validating(validation: PassValidation) -> Iterator[PassValidation]:
    """Make ``validation`` active for this compile, inside its gate's guard."""
    with validation.gate.guard():
        token = _ACTIVE.set(validation)
        try:
            yield validation
        finally:
            _ACTIVE.reset(token)


def selected(pass_name: str) -> bool:
    validation = _ACTIVE.get()
    return validation is not None and pass_name in validation.passes


def submit(pass_name: str, before: str, after: str) -> bool:
    """Queue the pass's exact before/after pair; returns whether a check was submitted."""
    validation = _ACTIVE.get()
    if validation is None or pass_name not in validation.passes or before == after:
        return False
    validation.gate.submit(
        CHECKS[pass_name],
        before=before,
        after=after,
        params={"toolchain_manifest": str(validation.toolchain_manifest)},
        label=pass_name,
    )
    return True
