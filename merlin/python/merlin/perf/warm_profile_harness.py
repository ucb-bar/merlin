"""Target-neutral C assembly for one warm-then-measured compute-cycle invocation.

The target owns how work is launched and how completion becomes visible to the
host.  This module owns only the measurement ordering around those two hooks.
Keeping them separate prevents a generic deployment wrapper from spelling an
accelerator-specific fence while still making it impossible to close the cycle
window before the target has completed.

The successful runtime protocol intentionally contains one performance metric in the repository's
existing parser format: ``METRIC cycles ...``.  Correctness diagnostics may be printed by the
validation hook, but validation itself runs after the cycle window closes and a
failed validation emits no performance metric.
"""
from __future__ import annotations

from dataclasses import dataclass
import textwrap

from .execution_policy import WarmProfileContract


class WarmProfileHarnessError(ValueError):
    """The requested wrapper cannot prove the warm measurement contract."""


_RESERVED_SOURCE_TOKENS = (
    "MERLIN_INVOCATIONS",
    "MERLIN_PROFILE",
    "METRIC cycles",
    "merlin_profile_",
)


def _is_c_identifier(value: object) -> bool:
    return (isinstance(value, str) and bool(value) and value.isascii()
            and (value[0].isalpha() or value[0] == "_")
            and all(character.isalnum() or character == "_" for character in value[1:]))


def _block(value: str, *, role: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WarmProfileHarnessError(f"{role} must be a nonempty C source block")
    source = value.strip()
    if "\x00" in source:
        raise WarmProfileHarnessError(f"{role} contains a NUL byte")
    present = [token for token in _RESERVED_SOURCE_TOKENS if token in source]
    if present:
        raise WarmProfileHarnessError(
            f"{role} may not emit or shadow reserved profile tokens {present}")
    return source


def _expression(value: str, *, role: str) -> str:
    source = _block(value, role=role)
    if "\n" in source or ";" in source or "{" in source or "}" in source:
        raise WarmProfileHarnessError(
            f"{role} must be one C expression returning a status")
    return source


@dataclass(frozen=True)
class TargetInvocationHooks:
    """Target-owned launch and completion statements for one invocation.

    ``complete`` is deliberately mandatory.  A synchronous target can supply
    its explicit completion acknowledgement, while an asynchronous target can
    supply its fence/wait primitive.  The generic renderer never guesses that
    returning from ``invoke`` means the target is complete.
    """

    invoke: str
    complete: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "invoke", _block(self.invoke, role="target invocation"))
        object.__setattr__(self, "complete", _block(self.complete, role="target completion"))


def render_warm_then_measure_main(
        *, prepare_input: str, invocation: TargetInvocationHooks,
        validate_outputs: str, contract: WarmProfileContract = WarmProfileContract(),
        cycle_reader: str = "read_cycles", function_name: str = "main",
        reset_after_warm: str | None = None,
        success_body: str | None = None) -> str:
    """Render a C entrypoint with a completed warm run and one measured run.

    Input preparation occurs once before any model invocation.  Each
    warm invocation is followed by the target completion hook.  The measured
    cycle window opens immediately before one target invocation and closes only
    after its completion hook.  Output validation follows the closing cycle
    read, and only a successful validation publishes the single cycles metric.
    A mutating caller may provide ``reset_after_warm``; it is emitted after
    every completed warm invocation and before the measured window.
    ``success_body`` is emitted after the metric, allowing target-owned result
    readback to remain outside the compute window without putting a large UART
    dump in front of the primary measurement.
    """
    if (not isinstance(contract.warmup_runs, int)
            or isinstance(contract.warmup_runs, bool)
            or contract.warmup_runs < 1):
        raise WarmProfileHarnessError(
            "warm profile harness requires at least one unmeasured warm invocation")
    if contract.measured_runs != 1:
        raise WarmProfileHarnessError(
            "warm profile harness supports exactly one measured invocation")
    if contract.captured_metrics != frozenset({"total_compute_cycles"}):
        raise WarmProfileHarnessError(
            "warm profile harness emits only total_compute_cycles")
    for role, symbol in (("cycle reader", cycle_reader), ("function name", function_name)):
        if not _is_c_identifier(symbol):
            raise WarmProfileHarnessError(f"{role} must be one plain C identifier")

    prepare = _block(prepare_input, role="input preparation")
    validate = _expression(validate_outputs, role="output validation")
    reset = (_block(reset_after_warm, role="post-warm reset")
             if reset_after_warm is not None else None)
    success = (_block(success_body, role="post-profile success body")
               if success_body is not None else None)
    invoke = textwrap.indent(invocation.invoke, "  ")
    complete = textwrap.indent(invocation.complete, "  ")

    lines = [
        f"int {function_name}(void) {{",
        (f'  printf("MERLIN_INVOCATIONS warmup={contract.warmup_runs} '
         f'measured=1\\n");'),
        textwrap.indent(prepare, "  "),
        '  printf("MERLIN_PROFILE warmup begin\\n");',
    ]
    for index in range(contract.warmup_runs):
        lines.extend([
            f"  /* unmeasured warm invocation {index + 1}/{contract.warmup_runs} */",
            invoke,
            complete,
        ])
        if reset is not None:
            lines.append(textwrap.indent(reset, "  "))
    lines.extend([
        '  printf("MERLIN_PROFILE warmup end rc=0\\n");',
        '  printf("MERLIN_PROFILE measured begin\\n");',
        f"  const uint64_t merlin_profile_cycle_start = {cycle_reader}();",
        invoke,
        complete,
        f"  const uint64_t merlin_profile_cycle_end = {cycle_reader}();",
        f"  const int merlin_profile_validation_rc = ({validate});",
        "  if (merlin_profile_validation_rc != 0) {",
        ('    printf("MERLIN_PROFILE measured end rc=%d\\n", '
         "merlin_profile_validation_rc);"),
        "    return merlin_profile_validation_rc;",
        "  }",
        ('  printf("METRIC cycles %llu\\n", (unsigned long long)'
         "(merlin_profile_cycle_end - merlin_profile_cycle_start));"),
        '  printf("MERLIN_PROFILE measured end rc=0\\n");',
    ])
    if success is not None:
        lines.append(textwrap.indent(success, "  "))
    lines.extend(["  return 0;", "}"])
    return "\n".join(lines) + "\n"


def require_strict_final_warm_profile(contract: object) -> WarmProfileContract:
    """Validate the cycle-only profile used by final comparison artifacts."""
    if type(contract) is not WarmProfileContract:
        raise WarmProfileHarnessError(
            "strict final warm profile requires an exact WarmProfileContract")
    if (type(contract.warmup_runs) is not int or type(contract.measured_runs) is not int
            or contract.warmup_runs != 1 or contract.measured_runs != 1):
        raise WarmProfileHarnessError(
            "strict final warm profile requires exactly one warm and one measured invocation")
    if contract.captured_metrics != frozenset({"total_compute_cycles"}):
        raise WarmProfileHarnessError(
            "strict final warm profile permits exactly one METRIC cycles value")
    return contract


def render_target_warm_then_measure_main(
        *, target: str, arguments: str, prepare_input: str,
        validate_outputs: str, contract: WarmProfileContract = WarmProfileContract(),
        cycle_reader: str = "read_cycles", function_name: str = "main") -> str:
    """Resolve launch/completion hooks from ``target`` and render its profile main.

    This is the entrypoint for bundle generators.  The target contract supplies
    both symbols through its harness ABI; callers cannot accidentally copy one
    accelerator's fence into another target's wrapper.
    """
    if not isinstance(target, str) or not target.strip():
        raise WarmProfileHarnessError("target must name a registered target contract")
    from merlin.targetgen.contract.harness_abi import for_target
    invocation = for_target(target).warm_profile_invocation(arguments)
    return render_warm_then_measure_main(
        prepare_input=prepare_input,
        invocation=invocation,
        validate_outputs=validate_outputs,
        contract=contract,
        cycle_reader=cycle_reader,
        function_name=function_name,
    )
