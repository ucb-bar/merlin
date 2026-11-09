"""Export an ``smt``-dialect module to SMT-LIB and solve it.

The whole chain is upstream infrastructure, which is the point: our dialect semantics is a
compilation pass to MLIR's own ``smt`` dialect, and the SMT-LIB comes out of upstream's
``ExportSMTLIB`` translation rather than an encoder of ours.

    xDSL smt module  ->  mlir-translate --export-smtlib  ->  z3

One trap, learned by measurement and guarded by a test: the exporter emits a trailing ``(reset)``
per solver scope. Handed to z3 verbatim the query still answers ``sat``, but the model comes back
EMPTY — the reset has already discarded it, so the counterexample silently disappears. It is
stripped here. An empty assignment set can still be legitimate if the outputs differ for every
input, so consumers must keep the solver status and model object together.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from merlin.verify import HAS_XDSL, HAS_Z3

from .tools import find_mlir_tool

#: The exporter's per-scope terminator. Stripping it is what keeps a counterexample readable.
_RESET = "(reset)"
_MAX_INPUT_BYTES = 16 * 1024 * 1024
_MAX_OUTPUT_BYTES = 64 * 1024 * 1024


class SmtUnavailable(RuntimeError):
    """Raised when the SMT chain cannot run. Never degrades into a verdict."""


@dataclass(frozen=True)
class Verdict:
    """The outcome of one query. Three states, never two.

    ``unsat``   the refinement obligation holds (this is the PASS for a verification query)
    ``sat``     refuted, with a counterexample in ``model``
    ``unknown`` the solver gave up (timeout / incompleteness) — NOT a pass
    """

    status: str
    model: str | None = None
    smt2: str = field(default="", repr=False)
    #: The counterexample as structured data: declared constant name -> signed integer value.
    #: Kept alongside the printed model because a counterexample is only actionable if it can be
    #: turned back into a concrete input tensor, and re-parsing a solver's pretty-printed model is
    #: exactly the brittle spelling-match this repo forbids.
    model_values: dict = field(default_factory=dict, repr=False)
    reason: str | None = None

    @property
    def verified(self) -> bool:
        return self.status == "unsat"

    @property
    def refuted(self) -> bool:
        return self.status == "sat"


def module_text(module) -> str:
    """Print an xDSL module. Generic form is what the upstream exporter parses."""
    if not HAS_XDSL:
        raise SmtUnavailable("xDSL is not installed")
    import io

    from xdsl.printer import Printer

    buf = io.StringIO()
    Printer(stream=buf, print_generic_format=True).print_op(module)
    return buf.getvalue()


def bounded_tool_output(
    argv: list[str], *, input_text: str | None = None, timeout_s: int = 120, max_output_bytes: int = _MAX_OUTPUT_BYTES
) -> tuple[int, str, str]:
    """Run an explicit executable with bounded time, input, captured output, and no shell."""
    if input_text is not None and len(input_text.encode("utf-8")) > _MAX_INPUT_BYTES:
        raise SmtUnavailable(f"translator input exceeds {_MAX_INPUT_BYTES} bytes")

    # The child writes to temporary files so its output never accumulates in process memory.
    # RLIMIT_FSIZE also bounds the files themselves on the supported Unix toolchain.
    def limit_files() -> None:
        import resource

        resource.setrlimit(resource.RLIMIT_FSIZE, (max_output_bytes, max_output_bytes))

    try:
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            result = subprocess.run(
                argv,
                input=input_text.encode("utf-8") if input_text is not None else None,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout_s,
                preexec_fn=limit_files,
                check=False,
            )
            stdout.seek(0)
            stderr.seek(0)
            out = stdout.read(max_output_bytes + 1)
            err = stderr.read(max_output_bytes + 1)
    except subprocess.TimeoutExpired as exc:
        raise SmtUnavailable(f"tool timed out after {timeout_s}s: {argv[0]!r}") from exc
    except OSError as exc:
        raise SmtUnavailable(f"tool cannot run at {argv[0]!r}: {exc}") from exc
    if len(out) > max_output_bytes or len(err) > max_output_bytes:
        raise SmtUnavailable(f"tool output exceeds {max_output_bytes} bytes: {argv[0]!r}")
    return result.returncode, out.decode("utf-8", errors="replace"), err.decode("utf-8", errors="replace")


def to_smtlib(module, *, translator: str | Path | None = None) -> str:
    """Export an ``smt``-dialect module to SMT-LIB text via upstream ``mlir-translate``."""
    tool = str(translator) if translator is not None else find_mlir_tool("mlir-translate")
    if tool is None:
        raise SmtUnavailable("mlir-translate not found; cannot export SMT-LIB")
    text = module_text(module)
    returncode, stdout, stderr = bounded_tool_output([tool, "--export-smtlib"], input_text=text)
    if returncode != 0:
        raise SmtUnavailable(f"mlir-translate --export-smtlib failed (exit {returncode}):\n{stderr[:4000]}")
    return stdout


def strip_reset(smt2: str) -> str:
    """Drop the exporter's per-scope ``(reset)``, which would discard the model."""
    return "\n".join(l for l in smt2.splitlines() if l.strip() != _RESET)


def solve(smt2: str, *, timeout_ms: int = 30_000) -> Verdict:
    """Solve SMT-LIB text with z3.

    ``timeout_ms`` bounds one query; a timeout is reported as ``unknown``, which is explicitly not a
    pass — an abstracted or timed-out obligation must never be counted as verified.
    """
    if not HAS_Z3:
        raise SmtUnavailable("z3 is not installed (pip install 'merlin[verify]')")
    import z3

    body = strip_reset(smt2)
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    solver.from_string(body)
    res = solver.check()
    if res == z3.sat:
        m = solver.model()
        values = {}
        for decl in m.decls():
            val = m[decl]
            try:
                # as_signed_long() gives the two's-complement reading of a bitvector, which is the
                # element value the program actually computes with.
                values[decl.name()] = val.as_signed_long()
            except Exception:
                values[decl.name()] = str(val)
        return Verdict("sat", model=str(m), smt2=body, model_values=values)
    if res == z3.unsat:
        return Verdict("unsat", smt2=body)
    return Verdict("unknown", smt2=body, reason=solver.reason_unknown())


def check_module(module, *, timeout_ms: int = 30_000, translator: str | Path | None = None) -> Verdict:
    """Export and solve in one step."""
    return solve(to_smtlib(module, translator=translator), timeout_ms=timeout_ms)


# Preserve historical object lookup; implementation and completion state live here.
for _compat_definition in (
    SmtUnavailable,
    Verdict,
    module_text,
    bounded_tool_output,
    to_smtlib,
    strip_reset,
    solve,
    check_module,
):
    _compat_definition.__module__ = "merlin.verify.smt_export"
del _compat_definition
