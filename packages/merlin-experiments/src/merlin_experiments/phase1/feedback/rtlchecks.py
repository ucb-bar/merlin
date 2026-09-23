"""Invocation-local advisory RTL feedback over the actual graded corpus.

This adds structural observations, never changes numerical pass/fail. No native
controller, corpus registry or machine-specific toolchain is selected at import.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from merlin.targetgen import rtl_check_runner as RUN

from ..context import InvocationContext
from ..treatments import Treatment
from . import qa as _base


def _option_values(argv: list[str], option: str) -> list[str | None]:
    values: list[str | None] = []
    for index, token in enumerate(argv):
        if token == option:
            values.append(argv[index + 1] if index + 1 < len(argv) and not argv[index + 1].startswith("--") else None)
        elif token.startswith(option + "="):
            values.append(token.split("=", 1)[1])
    return values


def prepare_arguments(
    argv: list[str],
    *,
    bundle_manifest: Path | None = None,
    bundles: Path | None = None,
    default_bundle: str | None = None,
) -> list[str]:
    """Validate the authored RTL treatment before creating execution callbacks.

    Only a native CLI edge supplies legacy bundle defaults. Installed callers name
    both the bundle and manifest explicitly; duplicate or malformed selectors refuse.
    """
    argv = list(argv)
    requested = _option_values(argv, "--bundle")
    if len(requested) > 1 or any(value is None for value in requested):
        raise ValueError(f"Arm-4 requires at most one well-formed --bundle; received {requested!r}")
    selected = requested[0] if requested else default_bundle
    if not selected or Path(selected).name != selected or not selected.startswith("merlin_assisted_rtlchecks_"):
        raise ValueError(f"invalid Arm-4 bundle identity {selected!r}")
    if bundle_manifest is None:
        if bundles is None:
            raise ValueError("installed RTLchecks requires an explicit bundle manifest")
        bundle_manifest = bundles / selected / "input_bundle_manifest.yaml"
    try:
        manifest = yaml.safe_load(bundle_manifest.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"cannot read Arm-4 bundle manifest {bundle_manifest}: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("bundle_id") != selected
        or manifest.get("arm") != "merlin_rtlchecks"
    ):
        raise ValueError(f"{selected!r} is not a generated merlin_rtlchecks bundle")
    arms = _option_values(argv, "--arm")
    if any(value != "merlin_assisted" for value in arms):
        raise ValueError("the Arm-4 RTL-checks wrapper requires --arm merlin_assisted")
    if not arms:
        argv += ["--arm", "merlin_assisted"]
    if not requested:
        argv += ["--bundle", selected]
    return argv


def _first_line(s: str | None) -> str | None:
    return s.splitlines()[0] if s else None


def _redact_rtl(r: dict) -> dict:
    """Answer-free per-capsule rtl_checks entry: verdict + FileCheck pass/diag + failing checks only."""
    fc = {k: {"ok": v.get("ok"), "diag": _first_line(v.get("diag"))} for k, v in (r.get("filecheck") or {}).items()}
    screen = r.get("screen") or {}
    fails = [
        {
            "id": c.get("id"),
            "severity": c.get("severity"),
            "message": c.get("message"),
            "expected": c.get("expected"),
            "got": c.get("got"),
            "ratio": c.get("ratio"),
            "fix_hint": c.get("fix_hint"),
        }
        for c in screen.get("checks", [])
        if c.get("status") == "fail"
    ]
    # SKIPPED checks are surfaced too, with their reason: a check that could not run must never be
    # indistinguishable from one that passed. The reasons are answer-free (they name a missing artifact
    # or an undecidable declaration, never a value).
    not_run = [{"id": k.get("id"), "reason": k.get("reason")} for k in (screen.get("skipped") or [])]
    return {
        "capsule": r.get("capsule"),
        "verdict": r.get("verdict"),
        "filecheck": fc,
        "screen_verdict": screen.get("verdict"),
        "findings": fails,
        "not_run": not_run,
    }


def feedback(
    runs_root,
    *,
    context: InvocationContext,
    capsule_roots: tuple[Path, ...],
    filecheck_candidates: tuple[Path, ...] = (),
) -> list[dict]:
    # RTL facts are the regenerated CIRCT artifact now (RUN._FACTS was retired in the facts-as-artifact
    # refactor); load_facts regenerates/reads it on demand. Same full-record shape screen_run expects.
    # Invocation-target-parameterized and ENDPOINT-routed: a RoCC target
    # (gemmini, endpoint inline_asm_insn) gets the dialect+trace FileCheck over its RoCC stream; a
    # self-hosted-ISA target (atlas, endpoint external_backend) gets the kernel opcode-LEGALITY FileCheck
    # over its emitted kernel.S. compile_checks picks by the DERIVED endpoint_kind, never by
    # funct_decode_table presence (the mlc extractor synthesises one for a self-hosted decoder too). Never
    # gemmini facts/ops applied to another target.
    target = context.target
    facts = RUN.load_facts(target)
    index = RUN.capsule_index(capsule_roots)
    fc = RUN.find_filecheck(filecheck_candidates)
    bench = Path(runs_root) / "runs" / f"{target}-capsule-bench"
    out = []
    if bench.is_dir():
        # Discover per-capsule run dirs by EITHER RTL-check input: a RoCC target emits
        # generated/instruction_trace.json; a self-hosted-ISA (external_backend, e.g. atlas) emits
        # generated/kernel.S (screen_run picks the right check by endpoint). Globbing only the RoCC trace
        # silently skipped every external_backend run — so the kernel opcode-legality check never fired.
        dirs = sorted(
            {p.parent.parent for g in ("instruction_trace.json", "kernel.S") for p in bench.glob(f"*/generated/{g}")}
        )
        for d in dirs:
            r = RUN.screen_run(d, facts, index, fc, write=True, target=target)  # writes rtl_checks.json
            if r:
                out.append(_redact_rtl(r))
    return out


def run(
    submission: str,
    capsules_root: str,
    runs_root,
    labels,
    no_oracle: bool,
    timeout: int,
    *,
    context: InvocationContext,
    filecheck_candidates: tuple[Path, ...] = (),
    contract: Path | None = None,
    additional_forbidden: tuple[str, ...] = (),
) -> dict:
    verdict = _base.run(
        submission,
        capsules_root,
        runs_root,
        labels,
        no_oracle,
        timeout,
        context=context,
        contract=contract,
        additional_forbidden=additional_forbidden,
    )
    try:
        verdict["rtl_checks"] = feedback(
            runs_root, context=context, capsule_roots=(Path(capsules_root),), filecheck_candidates=filecheck_candidates
        )
        verdict["rtl_checks_note"] = (
            "ADVISORY RTL-derived checks (FileCheck over emitted MLIR + decoded RoCC trace; bounds from "
            "CIRCT-extracted hardware facts). Does NOT gate pass/fail. Fix encoding/tile/capacity findings "
            "before the RTL oracle would; a clean result means ISA structure is hardware-legal, not that "
            "numerics are correct. Four of the findings are STRUCTURAL and worth reading first, because "
            "each one is a defect the numeric plane can only report as a value error: "
            "T0.output_store_coverage (a declared output with no covering store, or a store past its "
            "declared extent), T0.extent_tile_legalization (a declared extent the RTL-derived array edge "
            "does not divide, with no tail sequence legalizing it), T0.conv_lowering (a declared "
            "convolution whose Kh*Kw window was never folded into the contraction), and "
            "T0.encoded_field_intent (does the DRAM base pointer of each emitted move carry the tensor "
            "the kernel ABI puts in that argument slot? The argument order is RESOLVED from "
            "kernel_abi.arg_order_by_command_shape for YOUR buffer's own command shape -- there is one "
            "order per shape and it is NOT the capsule's or the interface's declaration order, which "
            "coincides with it only for buffers that happen to declare the tensors in that same order. "
            "A buffer resolving against no contract shape, or against two, is reported UNKNOWN rather "
            "than screened against a guess. The check also compares each store's readout dtype, the "
            "store activation and accumulator scale, and each move's column extent against what your own "
            "declaration derives. Both command-buffer planes read the DECLARATION, so an encoding whose "
            "pointer binding disagrees with it passes them and diverges only on the oracle). Every bound "
            "is derived from your capsule's DECLARED shapes plus the array geometry extracted from the "
            "RTL, and every finding is computed from YOUR OWN emitted artifact. A field for which no "
            "intent is derivable produces NO finding and is listed under fields_not_derivable in the "
            "check's evidence instead; a check listed under not_run did NOT run and is not a pass."
        )
    except Exception as e:  # advisory must never break the gate
        verdict["rtl_checks_error"] = repr(e)
    return verdict


def treatment(context: InvocationContext, *, filecheck_candidates: tuple[Path, ...] = ()) -> Treatment:
    """Create source-attributed callbacks bound to one explicit target invocation."""

    def qa_runner(
        submission, capsules_root, runs_root, labels, no_oracle, timeout, *, contract=None, additional_forbidden=()
    ):
        return run(
            submission,
            capsules_root,
            runs_root,
            labels,
            no_oracle,
            timeout,
            context=context,
            filecheck_candidates=filecheck_candidates,
            contract=contract,
            additional_forbidden=additional_forbidden,
        )

    def checkpoint_feedback(runs_root, *, capsule_roots):
        return feedback(
            runs_root, context=context, capsule_roots=capsule_roots, filecheck_candidates=filecheck_candidates
        )

    return Treatment(name="rtlchecks", qa_runner=qa_runner, checkpoint_feedback=checkpoint_feedback)
