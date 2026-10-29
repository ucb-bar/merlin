"""Check selected compiler passes against Merlin's Lean proofs, beside the compile that runs them.

A validated pass reports the exact IR before and after it ran; the pair goes to a
:class:`~merlin.verify.speculative_gate.SpeculativeGate`, the compile carries on, and a check that
does not accept stops the compile with its evidence. A pass that changes nothing submits nothing.

``lower_model`` takes a :class:`PassValidation`, or builds one from the environment so every entry
point that lowers (``merlin-compile --validate-passes``, the whole-model backends) can turn it on:

* ``MERLIN_VALIDATE_PASSES`` -- ``all`` (:data:`DEFAULT_CHECKS`) or an explicit subset of :data:`CHECKS`.
* ``MERLIN_LEAN_SDK_BIN`` -- the Lean SDK ``bin`` directory (``lean``, ``lake``) the checker was built with.
* ``MERLIN_COMPILER_PYTHON`` -- the interpreter of the authoritative MLIR exporter (``torch_mlir``).
* ``MERLIN_VALIDATE_WORKERS`` -- how many checks run at once (default 1; each is one niced process).

The ``merlin-lean`` checker must already be built from ``merlin/lean`` (``lake build``).
"""

from __future__ import annotations

import functools
import inspect
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Validated pass -> the gate check that proves its before/after pair.
CHECKS = {
    "prune_dead_pure_tensor_ops": "tensor_refinement",
    # The pipeline's leading tensor-level cse, run by merlin.llvmlower.leading_cse.
    "cse": "tensor_refinement",
    "ordered_loop_unrolling": "compiler_loop_refinement",
    "ordered_loop_segmentation": "compiler_loop_refinement",
    "buffer_reuse": "compiler_buffer_refinement",
    "bf16_f32acc_draft_contract": "compiler_macc_contract",
    "lower_quant_ext": "compiler_qdq_refinement",
    "configuration_write_elision": "compiler_configuration_refinement",
}
# Numerical draft conformance requires explicit selection. It is neither an
# admitted numerical policy nor an original-source refinement check.
DEFAULT_CHECKS = frozenset(CHECKS) - {
    "bf16_f32acc_draft_contract",
    "lower_quant_ext",
    "configuration_write_elision",
}
ENV_PASSES = "MERLIN_VALIDATE_PASSES"
ENV_SDK_BIN = "MERLIN_LEAN_SDK_BIN"
ENV_COMPILER_PYTHON = "MERLIN_COMPILER_PYTHON"
ENV_WORKERS = "MERLIN_VALIDATE_WORKERS"


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
        """Require current executables and proof objects, including audit-only modules."""
        import subprocess

        from .lean import check

        environment = {**os.environ, "PATH": f"{self.sdk_bin}{os.pathsep}{os.environ.get('PATH', '')}"}
        result = subprocess.run(
            [
                str(Path(self.sdk_bin) / "lake"),
                "build",
                "--no-build",
                "MerlinVerify",
                "merlin-lean",
                "merlin-lean-audit",
            ],
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
    passes: frozenset[str] = DEFAULT_CHECKS
    index_bits: int = 64
    qdq_source: Any | None = None

    def __post_init__(self) -> None:
        from .speculative_gate import SpeculativeGate

        if type(self.gate) is not SpeculativeGate:
            raise ValueError("pass validation requires the host's SpeculativeGate")
        if type(self.tools) is not CheckerTools:
            raise ValueError("pass validation requires explicit CheckerTools")
        if type(self.passes) is not frozenset or not self.passes or not self.passes <= set(CHECKS):
            raise ValueError(f"validated passes must be a nonempty subset of {sorted(CHECKS)}")
        for name in self.passes:
            require_canonical_handler(self, CHECKS[name])
        if self.qdq_source is not None:
            from .compiler_qdq import SELECTOR, SourceCapture

            if type(self.qdq_source) is not SourceCapture or SELECTOR not in self.passes:
                raise ValueError("Q/DQ source capture requires explicit SourceCapture and lower_quant_ext selection")
        if self.index_bits != 64:
            # merlin/lean/LEDGER.md S6: at index32 the Lean model and the pinned native pipeline
            # disagree on a retained overflow example; a refinement claim there is not established.
            raise ValueError("pass validation supports the 64-bit index width only (LEDGER S6)")


_ACTIVE: ContextVar[PassValidation | None] = ContextVar("merlin_pass_validation", default=None)
_OBSERVING: ContextVar[tuple] = ContextVar("merlin_pass_observation", default=())


def require_canonical_handler(validation, check_name):
    """Formal compiler selections cannot be rerouted through custom callbacks."""
    from .gate_worker import BUILTIN_CHECKS

    gate = validation.gate
    if check_name not in BUILTIN_CHECKS or gate._worker_path or gate._aliases.get(check_name, check_name) != check_name:
        raise ValueError("formal pass validation requires its canonical handler and worker import roots")


def from_environment(workdir: Path, *, index_bits: int | None = None) -> PassValidation | None:
    """The validation the environment asks for, or None. A request with a missing tool is an error."""
    requested = os.environ.get(ENV_PASSES, "").strip()
    if not requested:
        return None
    passes = DEFAULT_CHECKS if requested == "all" else frozenset(p.strip() for p in requested.split(","))
    if not passes or not passes <= set(CHECKS):
        raise ValueError(f"validated passes must be a nonempty subset of {sorted(CHECKS)}")
    sdk_bin = os.environ.get(ENV_SDK_BIN)
    if not sdk_bin:
        raise ValueError(f"{ENV_PASSES} is set but {ENV_SDK_BIN} does not select a Lean SDK")
    from merlin.llvmlower.toolchain import compiler_python

    from .speculative_gate import SpeculativeGate

    tools = CheckerTools(Path(sdk_bin), compiler_python())
    workers = os.environ.get(ENV_WORKERS, "1").strip()
    if not workers.isdigit() or int(workers) < 1:
        raise ValueError(f"{ENV_WORKERS} must be a positive integer, not {workers!r}")
    gate = SpeculativeGate(Path(workdir) / "pass-validation", interrupt=True, workers=int(workers))
    from .compiler_qdq import SourceCapture

    return PassValidation(
        gate, tools, passes, 64 if index_bits is None else index_bits, SourceCapture.from_environment()
    )


@contextmanager
def validating(validation: PassValidation) -> Iterator[PassValidation]:
    """Make ``validation`` active for this compile, inside its gate's guard."""
    with validation.gate.guard():
        token = _ACTIVE.set(validation)
        try:
            yield validation
        finally:
            _ACTIVE.reset(token)


def current() -> PassValidation | None:
    """The validation active for this compile, if any."""
    return _ACTIVE.get()


def validated(workdir_argument: str, *, index_bits_argument: str | None = None, on_failure=None):
    """Run the decorated compile step under pass validation, from its ``pass_validation`` argument
    or the environment.

    Nesting: inside an active validation the step joins it (one gate, one guard, owned by the outer
    step); an explicit ``pass_validation`` that is not the active one is refused. ``on_failure(work)``
    runs when a check fails, before the failure propagates.
    """

    def decorate(function: Callable) -> Callable:
        signature = inspect.signature(function)

        @functools.wraps(function)
        def invoke(*args, **kwargs):
            arguments = signature.bind(*args, **kwargs).arguments
            explicit = arguments.get("pass_validation")
            active = _ACTIVE.get()
            bits = arguments.get(index_bits_argument) if index_bits_argument else None
            if active is not None:
                if explicit is not None and explicit is not active:
                    raise ValueError("a different pass validation is already active for this compile")
                if bits is not None and bits != active.index_bits:
                    raise ValueError("lowering index width differs from the active pass validation")
                return function(*args, **kwargs)
            work = Path(arguments[workdir_argument])
            validation = explicit or from_environment(work, index_bits=bits)
            if validation is None:
                return function(*args, **kwargs)
            if bits is not None and bits != validation.index_bits:
                raise ValueError("lowering index width differs from the selected pass validation")
            from .speculative_gate import GateFailure

            try:
                with validating(validation):
                    return function(*args, **kwargs)
            except GateFailure:
                if on_failure is not None:
                    on_failure(work)
                raise

        return invoke

    return decorate


def observe(pass_name: str, module, transform: Callable):
    """Run ``transform(module)`` and, when ``pass_name`` is validated, submit its exact before/after IR."""
    if not selected(pass_name) or any(
        name == pass_name and (actual is module or actual is None) for name, actual in _OBSERVING.get()
    ):
        return transform(module)
    from merlin.xdsl_dialects._common import text

    before = text(module, generic=True)
    token = _OBSERVING.set((*_OBSERVING.get(), (pass_name, module)))
    try:
        result = transform(module)
    finally:
        _OBSERVING.reset(token)
    submit(pass_name, before, text(module, generic=True))
    return result


def observed(pass_name: str):
    """Capture an existing pass wherever it runs inside a validation scope."""

    def decorate(transform):
        @functools.wraps(transform)
        def invoke(module, *args, **kwargs):
            if not selected(pass_name):
                return transform(module, *args, **kwargs)
            return observe(pass_name, module, lambda actual: transform(actual, *args, **kwargs))

        return invoke

    return decorate


def selected(pass_name: str) -> bool:
    validation = _ACTIVE.get()
    return validation is not None and pass_name in validation.passes


def submit(pass_name: str, before: str, after: str, *, label: str | None = None) -> bool:
    """Queue the pass's exact before/after pair; returns whether a check was submitted. ``label``
    names the check in reports (default: the pass)."""
    validation = _ACTIVE.get()
    if validation is None or pass_name not in validation.passes or before == after:
        return False
    require_canonical_handler(validation, CHECKS[pass_name])
    params = {
        "sdk_bin": str(validation.tools.sdk_bin),
        "compiler_python": str(validation.tools.compiler_python),
        "index_bits": validation.index_bits,
    }
    if pass_name == "lower_quant_ext" and validation.qdq_source is not None:
        params["source_capture"] = validation.qdq_source.describe()
    validation.gate.submit(
        CHECKS[pass_name],
        before=before,
        after=after,
        params=params,
        label=label or pass_name,
    )
    return True
