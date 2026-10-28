"""Check selected compiler passes against Merlin's Lean proofs, beside the compile that runs them.

A validated pass reports the exact IR before and after it ran; the pair goes to a
:class:`~merlin.verify.speculative_gate.SpeculativeGate`, the compile carries on, and a check that
does not accept stops the compile with its evidence. A pass that changes nothing submits nothing.

``lower_model`` takes a :class:`PassValidation`, or builds one from the environment so every entry
point that lowers (``merlin-compile --validate-passes``, the whole-model backends) can turn it on:

* ``MERLIN_VALIDATE_PASSES`` -- ``all`` or a comma-separated subset of :data:`CHECKS`.
* ``MERLIN_LEAN_SDK_BIN`` -- the Lean SDK ``bin`` directory (``lean``, ``lake``) the checker was built with.
* ``MERLIN_COMPILER_PYTHON`` -- the interpreter of the authoritative MLIR exporter (``torch_mlir``).

The ``merlin-lean`` checker must already be built from ``merlin/lean`` (``lake build``).
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
CHECKS = {"prune_dead_pure_tensor_ops": "tensor_refinement"}
ENV_PASSES = "MERLIN_VALIDATE_PASSES"
ENV_SDK_BIN = "MERLIN_LEAN_SDK_BIN"
ENV_COMPILER_PYTHON = "MERLIN_COMPILER_PYTHON"


@dataclass(frozen=True)
class CheckerTools:
    """The exact tools a check uses; every one must exist, or validation refuses to start."""

    sdk_bin: Path
    compiler_python: Path

    def __post_init__(self) -> None:
        from .lean import check

        missing = [
            f"{what} ({path})"
            for what, path in (
                ("Lean SDK lake", Path(self.sdk_bin) / "lake"),
                ("compiler Python", Path(self.compiler_python)),
                ("merlin-lean checker (run lake build in merlin/lean)", check.lean_exe()),
            )
            if not Path(path).is_file()
        ]
        if missing:
            raise ValueError("pass validation tools are missing: " + ", ".join(missing))
        self._require_current_checker()

    def _require_current_checker(self) -> None:
        """A checker built from other proof sources must not certify anything."""
        import subprocess

        from .lean import check

        environment = {**os.environ, "PATH": f"{self.sdk_bin}{os.pathsep}{os.environ.get('PATH', '')}"}
        result = subprocess.run(
            [str(Path(self.sdk_bin) / "lake"), "build", "--no-build", "merlin-lean", "merlin-lean-audit"],
            cwd=check.lean_root(),
            env=environment,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if result.returncode != 0:
            raise ValueError(
                "the merlin-lean checker is out of date with merlin/lean (or lake failed); "
                f"run `lake build` there: {(result.stdout + result.stderr).strip()[-300:]}"
            )


@dataclass(frozen=True)
class PassValidation:
    gate: Any
    tools: CheckerTools
    passes: frozenset[str] = frozenset(CHECKS)
    index_bits: int = 64

    def __post_init__(self) -> None:
        from .speculative_gate import SpeculativeGate

        if type(self.gate) is not SpeculativeGate:
            raise ValueError("pass validation requires the host's SpeculativeGate")
        if type(self.tools) is not CheckerTools:
            raise ValueError("pass validation requires explicit CheckerTools")
        if type(self.passes) is not frozenset or not self.passes or not self.passes <= set(CHECKS):
            raise ValueError(f"validated passes must be a nonempty subset of {sorted(CHECKS)}")
        if self.index_bits not in (32, 64):
            raise ValueError("the typed boundary check supports 32- and 64-bit index widths")


_ACTIVE: ContextVar[PassValidation | None] = ContextVar("merlin_pass_validation", default=None)


def from_environment(workdir: Path, *, index_bits: int | None = None) -> PassValidation | None:
    """The validation the environment asks for, or None. A request with a missing tool is an error."""
    requested = os.environ.get(ENV_PASSES, "").strip()
    if not requested:
        return None
    passes = frozenset(CHECKS) if requested == "all" else frozenset(p.strip() for p in requested.split(","))
    sdk_bin = os.environ.get(ENV_SDK_BIN)
    if not sdk_bin:
        raise ValueError(f"{ENV_PASSES} is set but {ENV_SDK_BIN} does not select a Lean SDK")
    from merlin.llvmlower.toolchain import compiler_python

    from .speculative_gate import SpeculativeGate

    tools = CheckerTools(Path(sdk_bin), compiler_python())
    gate = SpeculativeGate(Path(workdir) / "pass-validation", interrupt=True)
    return PassValidation(gate, tools, passes, 64 if index_bits is None else index_bits)


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
        params={
            "sdk_bin": str(validation.tools.sdk_bin),
            "compiler_python": str(validation.tools.compiler_python),
            "index_bits": validation.index_bits,
        },
        label=pass_name,
    )
    return True
