"""In-sandbox self-check SHIM — staged as <ws>/agent_selfcheck.py under --sandbox bwrap.

Same CLI as the real agent_selfcheck.py, but imports NOTHING from merlin (the oracle is masked in the
sandbox). It forwards the request to the driver-side broker via <ws>/.qa_channel and prints the redacted
verdict the broker returns. From the agent's view this is identical to running the self-check directly —
it gets pass/fail + its own artifacts (in ./selfcheck_out/) — but the oracle/goldens never enter the box.

A request names the tree it checks: ``submission/`` (the default, and the only tree graded officially) or
an author-side alternative at ``variants/<name>/``. A short ``--wait-budget`` returns a request id at once
so the author keeps working while the check runs, and ``--attach <id>`` collects it later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import sys
import time
from pathlib import Path, PurePosixPath

if __package__:
    from .qa_selfcheck_protocol import (
        _CALIBRATION_CAP,
        _DIGEST_IGNORES,
        _GSIM_QUEUE_STATUS,
        _LIST_COMPLETED,
        _NAME_CHARS,
        _NAME_FIRST,
        _PANEL,
        _PANEL_CASES,
        _SIMS,
        _SUBMISSION,
        _SUITE_SIZE_FALLBACK,
        _VARIANT_NAME_MAX,
        _VARIANTS,
        PROTOCOL_VERSION,
        _atomic_write,
        _capacity_note,
        _default_sim,
        _gsim_queue,
        _gsim_queue_note,
        _list_requests,
        _read_json,
        _reply_is_usable,
        _request_budget_seconds,
        _request_id,
        _requester_identity,
        _required_rtl_engine,
        _sim_policy_error,
        _source_label,
        _submission_digest,
        _valid_variant_name,
        _variant_error,
        _verdict,
    )
    from .qa_selfcheck_sweeps import (
        _EXPERIMENT_SCHEMA,
        _MODEL_KEYS,
        _SWEEP_SCHEMA,
        _case_groups,
        _run_sweep,
        _sweep_candidates,
        _sweep_cases,
        _sweep_passed,
        _sweep_rank,
        _sweep_rows,
    )
else:
    from qa_selfcheck_protocol import (
        _CALIBRATION_CAP,
        _DIGEST_IGNORES,
        _GSIM_QUEUE_STATUS,
        _LIST_COMPLETED,
        _NAME_CHARS,
        _NAME_FIRST,
        _PANEL,
        _PANEL_CASES,
        _SIMS,
        _SUBMISSION,
        _SUITE_SIZE_FALLBACK,
        _VARIANT_NAME_MAX,
        _VARIANTS,
        PROTOCOL_VERSION,
        _atomic_write,
        _capacity_note,
        _default_sim,
        _gsim_queue,
        _gsim_queue_note,
        _list_requests,
        _read_json,
        _reply_is_usable,
        _request_budget_seconds,
        _request_id,
        _requester_identity,
        _required_rtl_engine,
        _sim_policy_error,
        _source_label,
        _submission_digest,
        _valid_variant_name,
        _variant_error,
        _verdict,
    )
    from qa_selfcheck_sweeps import (
        _EXPERIMENT_SCHEMA,
        _MODEL_KEYS,
        _SWEEP_SCHEMA,
        _case_groups,
        _run_sweep,
        _sweep_candidates,
        _sweep_cases,
        _sweep_passed,
        _sweep_rank,
        _sweep_rows,
    )


def _case_rows(progress) -> list:
    rows = progress.get("per_capsule") if isinstance(progress, dict) else None
    return [row for row in rows or () if isinstance(row, dict) and isinstance(row.get("capsule"), str)]


def _case_line(row: dict) -> str:
    """One case's correctness as the grader has decided it so far."""
    name, status = row["capsule"], str(row.get("status"))
    if status == "running":
        return (
            f"[self-check] {name}: numerics {row.get('numerics')} (provisional: later checks still "
            "running and can still fail it)"
        )
    why = "/".join(str(row[key]) for key in ("failure_plane", "failure_category") if row.get(key))
    numerics = f"; numerics {row['numerics']}" if row.get("numerics") else ""
    return f"[self-check] {name}: {status}" + (f" ({why})" if why else "") + numerics


def _new_case_lines(seen: dict, progress) -> list:
    """Lines for cases whose row is new or changed since ``seen`` (updated in place).

    Each case is reported as soon as its numerics are decided and again when it finishes, so a large
    suite never hides its decided cases behind its slowest one."""
    lines = []
    for row in _case_rows(progress):
        state = tuple(row.get(key) for key in ("status", "numerics", "failure_plane", "failure_category"))
        if seen.get(row["capsule"]) != state:
            seen[row["capsule"]] = state
            lines.append(_case_line(row))
    return lines


_WORKFLOW = """\
keep working while checks run:
  python agent_selfcheck.py --capsules A,B,C --wait-budget 5   # returns a request id (exit 3: running)
  ...keep editing; each check grades a snapshot taken when it starts...
  python agent_selfcheck.py --attach ID                         # collect the verdict
  python agent_selfcheck.py --list                              # requests in flight

Operator-selected connected model sources use `--suite programs --sim spike --tiers L0,L1,L2`.
The original model owner checks all outputs; programs never acquire core or RTL credit.

Several requests may run at once. Put every capsule an edit affects in ONE request: its capsules run
in parallel across the CPU pool, and per-request setup dominates small checks. `--capsules failing`
re-checks what did not pass in your most recent completed check.

Before changing a default that governs many shapes: `python agent_selfcheck.py --panel dense
--wait-budget 5` checks one case per shape class (square, tall, wide, thin M=1, ragged, a model
layer) on the Spike cycle model and compares each with its own best earlier cycles of the same
evidence kind. Performance is measured on the Spike cycle model by default; add --sim gsim for RTL cycles.

variants: copy submission/ to variants/NAME/, change it, and check it with --submission variants/NAME.
A variant result is labelled with its name and digest and is a self-check only: only submission/ is
graded officially or counts toward convergence, so copy the winner into submission/.

Time alternatives together: --sweep variants/a,variants/b --capsules A,B [--knob NAME=false,true].
This submits parallel scalability requests, including submission/ as the baseline, and returns ranked
model estimates. --bundle REPRESENTATIVE selects a bundle in qa/idle_tuning.json. --attach SWEEP_ID
collects the same sweep later; full results and immutable source identities are retained.

When reference guidance is selected, pin completed source/command/emission views without a broker call:
  python agent_selfcheck.py --write-reference-views artifacts/views.json artifacts/source.mlir \\
    artifacts/commands.json artifacts/emitted.mlir
  python agent_selfcheck.py --reference-analysis candidate --reference-path artifacts/views.json
The source is the original interface contract; emitted LLVM is not schedule.plan. Inspect an actual
schedule separately with --reference-format schedule_mlir. Compare owned before/after disassemblies:
  python agent_selfcheck.py --reference-analysis compare --reference-path artifacts/before.dis \\
    --reference-symbol kernel_entry --candidate-path artifacts/after.dis --candidate-symbol kernel_entry
Verify your own change from ordinary selfcheck outputs (CCA bank arm): traffic, storage, path, sweep:
  python agent_selfcheck.py --reference-analysis verify --before selfcheck_out/requests/OLD \\
    --after selfcheck_out/requests/NEW [--mechanism issue_overlap] [--sweep selfcheck_out/requests/NEW2] \\
    [--candidate-symbol kernel_entry]
Stream facts come from each build's package_kernel.elf, else one *.o you put in the case directory (for
example a -mno-relax rebuild), else its lowered.llvm.mlir compiled host-side with -mno-relax.
Recovered ISA operands appear under observation_comparison with explicit missingness. To inspect
one field, repeat --reference-observation-field Instruction.field (at most eight); use field names
from the selected ISA/decoded observation roster. Full evidence remains in the immutable detail.
Pins and static comparisons do not qualify source equivalence, scheduling realization or performance.
"""


#: Return a resumable request before the calling shell reaches its 600 s limit.
_DEFAULT_WAIT_BUDGET_S = 540


#: The reply's ranked next steps (``feedback.next_steps``): printed after the reply, bounded.
_NEXT_STEPS_KEY = "next_steps"
_NEXT_STEPS_LINES = 8
_NEXT_STEPS_BYTES = 6000


def _next_steps_text(verdict) -> str:
    """The reply's ``next_steps`` lines as one bounded block, or '' when it carries none."""
    block = verdict.get(_NEXT_STEPS_KEY) if isinstance(verdict, dict) else None
    lines = block.get("lines") if isinstance(block, dict) else None
    if not isinstance(lines, list):
        return ""
    kept, size = [], 0
    for line in lines[:_NEXT_STEPS_LINES]:
        if not isinstance(line, str) or not line.strip():
            continue
        text = " ".join(line.split())
        if size + len(text.encode()) + 1 > _NEXT_STEPS_BYTES:
            break
        kept.append(text)
        size += len(text.encode()) + 1
    return "\n".join(kept) if len(kept) > 1 else ""


def _print_next_steps(verdict) -> None:
    """Print the next steps on stderr: shown in a terminal even when stdout is piped to a JSON reader,
    and never appended to the JSON a script parses from stdout."""
    text = _next_steps_text(verdict)
    if text:
        print("\n[self-check] " + text, file=sys.stderr, flush=True)


def _write_reference_views(workspace, arguments, *, entry_symbol="kernel_entry"):
    """Pin existing author-owned artifacts; never execute or qualify their code."""
    workspace = Path(workspace).resolve()

    def owned(value):
        relative = PurePosixPath(value)
        if (
            relative.is_absolute()
            or str(relative) != value
            or "\\" in value
            or any(part in (".", "..") for part in relative.parts)
            or not relative.parts
            or relative.parts[0] not in ("submission", "variants", "selfcheck_out", "artifacts")
        ):
            raise ValueError("use workspace-relative submission/, variants/, selfcheck_out/ or artifacts/ paths")
        path = workspace / relative
        if any(p.is_symlink() for p in (path, *path.parents)) or not path.resolve().is_relative_to(workspace):
            raise ValueError("artifact views require ordinary author-owned paths")
        return path, relative.as_posix()

    if not entry_symbol or len(entry_symbol) > 512:
        raise ValueError("entry symbol must be nonempty and bounded")
    destination, label = owned(arguments[0])
    bundle = dict(schema="merlin.reference_artifact_views.v1", entry_symbol=entry_symbol)
    for role, value in zip(("source", "command_buffer", "emitted"), arguments[1:], strict=True):
        path, relative = owned(value)
        before = path.stat()
        if not path.is_file() or not 0 < before.st_size <= 8 << 20 or path == destination:
            raise ValueError("each completed artifact must be an ordinary nonempty file <=8MiB")
        raw = path.read_bytes()
        after = path.stat()
        if (
            any(
                getattr(before, key) != getattr(after, key)
                for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            )
            or len(raw) != before.st_size
        ):
            raise ValueError("artifact changed while pinning; finish compilation and retry")
        bundle[role] = dict(path=relative, sha256=hashlib.sha256(raw).hexdigest())
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(destination, json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return dict(
        status="artifact_views_written",
        manifest=label,
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        qualification_issued=False,
        next_argv=[
            sys.executable,
            Path(__file__).name,
            "--reference-analysis",
            "candidate",
            "--reference-path",
            label,
            "--reference-format",
            "artifact_views",
        ],
        note="Pins identify supplied bytes only. Source binding, emitted mechanism, numerics and timing "
        "require the selected analysis and original executable checks.",
    )


def _analysis_summary_line(text, rid):
    """``status``, report kind, unit/region counts and the detail file of one reference-analysis reply."""
    try:
        reply = json.loads(text)
    except ValueError:
        return f"[reference-analysis] reply .qa_channel/resp_{rid}.json is not JSON"
    report = reply.get("report") if isinstance(reply.get("report"), dict) else {}
    parts = [f"status={reply.get('status')}", f"kind={report.get('kind')}"]
    for key in ("total", "next_offset"):
        if report.get(key) is not None:
            parts.append(f"{key}={report[key]}")
    units = report.get("units") if isinstance(report.get("units"), list) else []
    for row in units[:4]:
        mechanisms = row.get("mechanisms") or {}
        known = sorted(
            name
            for name, fact in mechanisms.items()
            if isinstance(fact, dict) and fact.get("status") not in (None, "unknown")
        )
        parts.append(f"{str(row.get('unit_id'))[:20]}: observed {','.join(known) or 'none'}")
    detail = reply.get("detail")
    if isinstance(detail, dict):
        detail = detail.get("path")
    parts.append(f"full reply .qa_channel/resp_{rid}.json" + (f", detail {detail}" if detail else ""))
    return "[reference-analysis] " + "; ".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Agent self-check (redacted; routed to the driver-side broker).",
        epilog=_WORKFLOW,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--submission",
        default=_SUBMISSION,
        help="tree to check: submission (default; the only tree graded officially) or variants/NAME "
        "(a variant runs on the Spike cycle model unless --sim names another engine)",
    )
    ap.add_argument(
        "--suite",
        choices=("core", "scalability", "programs"),
        default="core",
        help="core conformance, required scalability, or operator-selected public programs (Spike only)",
    )
    # Default to the declared certifying engine. Explicit screen-only checks cannot certify.
    ap.add_argument("--sim", choices=list(_SIMS), default=_default_sim())
    # Forward tier selection unchanged to the broker.
    ap.add_argument("--tiers", default="")
    ap.add_argument(
        "--capsules",
        default="all",
        help="'all', 'failing', 'changed' (build and screen every public capsule, reuse exact-byte RTL "
        "certificates, simulate only those whose bytes are neither certified nor known to fail), or names",
    )
    ap.add_argument(
        "--panel",
        default="",
        metavar="FAMILY",
        help="check one representative per shape class (square, tall, wide, thin M=1, ragged, a "
        "model layer) of FAMILY (dense, conv, gemv, ...) and compare each with its best earlier cycles "
        "of the same evidence kind; the Spike cycle model by default (--sim gsim for RTL cycles)",
    )
    ap.add_argument("--workers", type=int, default=0, help="0 uses the host's selected grading worker count")
    # Bound the complete check. Native gSIM execution retains its separate 1800 s cap.
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--out", default="")
    # Keep diagnostic options available inside the sandbox.
    ap.add_argument("--shape-coverage", action="store_true")
    ap.add_argument("--offload-census", action="store_true")
    ap.add_argument("--model-layers", action="store_true")
    ap.add_argument(
        "--candidate-diagnostics", action="store_true", help="inspect original public emission without grading"
    )
    ap.add_argument(
        "--compile-only",
        action="store_true",
        help="compile public scale/frontier tensor sources without executing a probe",
    )
    ap.add_argument(
        "--scale-probe",
        action="store_true",
        help="with --compile-only, inspect public scale/tail emission before object compilation",
    )
    ap.add_argument(
        "--dynamic-census", action="store_true", help="append bounded own-ELF activity after functional screening"
    )
    ap.add_argument(
        "--coverage-sweep",
        default=None,
        metavar="SHAPES",
        help="with --suite scalability and one public --capsules template: compile only the shapes you name "
        "in its family, e.g. m=24,m=64:k=1024 (name=integer joined by ':'; shapes by ','), and report "
        "per shape admitted/refused library consumers, seed fallback, transfer widths and fences",
    )
    ap.add_argument(
        "--coverage-sweep-numerical",
        action="store_true",
        help="with --coverage-sweep: also run the functional Spike check of each named shape",
    )
    ap.add_argument(
        "--caller-layout",
        default="",
        help="inspect an emitted submission-relative command buffer's physical caller layout without grading",
    )
    # Return the request identity and progress before the caller shell expires.
    # --attach resumes the same request without submitting it again.
    ap.add_argument(
        "--wait-budget",
        type=int,
        default=_DEFAULT_WAIT_BUDGET_S,
        help="seconds to block before returning in_progress with a resumable request id (0 = no bound); "
        "5 submits and returns at once so you can keep working",
    )
    ap.add_argument(
        "--attach",
        default="",
        help="collect (or keep waiting on) a request id instead of submitting a new one",
    )
    ap.add_argument("--list", action="store_true", help="list in-flight and recent requests, then exit")
    ap.add_argument(
        "--sweep",
        nargs="?",
        const="",
        default=None,
        metavar="VARIANTS",
        help="time comma-separated variants together with submission/ and return model rankings",
    )
    ap.add_argument(
        "--knob",
        action="append",
        default=[],
        metavar="NAME=V1,V2",
        help="add independent declared knob values (JSON scalars) to --sweep",
    )
    ap.add_argument("--bundle", default="", metavar="REPRESENTATIVE", help="sweep a retained performance bundle")
    ap.add_argument(
        "--tuning-overrides",
        default="",
        metavar="JSON",
        help="check a private copy with declared knob values changed; submission itself is unchanged",
    )
    ap.add_argument(
        "--reference-request",
        default="",
        metavar="ID",
        help="associate this check with an operator-selected reference experiment request",
    )
    ap.add_argument(
        "--reference-analysis",
        choices=("inventory", "lift", "candidate", "compare", "applicability", "verify"),
        help="selected guidance only: inspect visible references or candidate artifacts; no grading",
    )
    ap.add_argument("--before", default="", help="verify only: owned before build (selfcheck_out/ case or request dir)")
    ap.add_argument("--after", default="", help="verify only: owned after build (selfcheck_out/ case or request dir)")
    ap.add_argument("--mechanism", default="", help="verify only: derived-effect name the change targets")
    ap.add_argument("--baseline", default="", help="verify only: owned seed build of the same capsules")
    ap.add_argument(
        "--verdicts",
        action="append",
        default=[],
        help="verify only: owned saved selfcheck response JSON (repeatable) for numerical verdicts",
    )
    ap.add_argument(
        "--write-reference-views",
        nargs=4,
        metavar=("MANIFEST", "SOURCE", "COMMAND_BUFFER", "EMITTED"),
        help="write pinned artifact_views from completed owned files; no request or execution. "
        "Use artifacts/ paths; --reference-symbol selects the emitted entry (default kernel_entry).",
    )
    ap.add_argument(
        "--reference-path",
        default="",
        help="workspace-relative reference-kernels/ or owned artifact; "
        "compare accepts owned before/after files on both sides",
    )
    ap.add_argument(
        "--reference-format",
        default="auto",
        choices=(
            "auto",
            "c_source",
            "python_schedule",
            "listing",
            "disassembly",
            "elf",
            "schedule_mlir",
            "semantic_region",
            "interface_mlir",
            "source_mlir",
            "emitted_mlir",
            "command_buffer",
            "artifact_views",
            "candidate_build",
        ),
        help="schedule_mlir is an actual schedule.plan; for emitted LLVM use a pinned artifact_views "
        "manifest created with --write-reference-views; candidate_build selects an original "
        "owned linked view",
    )
    ap.add_argument("--reference-member", default="", help="member of an already visible ZIP; no extraction")
    ap.add_argument(
        "--reference-entry",
        default="",
        help="pinned reference-map entry ID, or a CCA bank unit ID listed by inventory; selects its file/locator",
    )
    ap.add_argument("--reference-symbol", default="", help="exact source or disassembly function symbol")
    ap.add_argument("--reference-region", default="", help="stable machine region identity returned by lift/inventory")
    ap.add_argument(
        "--reference-observation-field",
        action="append",
        default=[],
        help="compare only: repeat up to eight ISA Instruction.field selectors",
    )
    ap.add_argument(
        "--reference-instruction-offset",
        type=int,
        default=0,
        help="first instruction in the selected region's bounded instruction page",
    )
    ap.add_argument("--reference-offset", type=int, default=0)
    ap.add_argument("--reference-limit", type=int, default=8)
    ap.add_argument(
        "--candidate-path",
        default="",
        help="compare only: owned submission/, variants/, selfcheck_out/ or artifacts/ file",
    )
    ap.add_argument(
        "--candidate-schedule",
        default="",
        help="compare/applicability: owned artifacts/ or "
        "variants/ stream document from python -m merlin.analysis.kernels.schedule_effects; with "
        "--candidate-path the reply attributes each difference to the lowering or to emission",
    )
    ap.add_argument(
        "--candidate-format",
        default="auto",
        choices=(
            "auto",
            "c_source",
            "python_schedule",
            "listing",
            "disassembly",
            "elf",
            "schedule_mlir",
            "semantic_region",
            "interface_mlir",
            "source_mlir",
            "emitted_mlir",
            "command_buffer",
            "artifact_views",
            "candidate_build",
        ),
    )
    ap.add_argument(
        "--candidate-symbol",
        default="",
        help="compare/applicability: exact emitted function "
        "symbol; verify: the entry to step (default: the contract's declared kernel entry)",
    )
    ap.add_argument("--candidate-region", default="", help="compare only: stable emitted machine region identity")
    ap.add_argument(
        "--candidate-instruction-offset",
        type=int,
        default=0,
        help="compare only: first instruction in the candidate region's instruction page",
    )
    ap.add_argument("--candidate-offset", type=int, default=0)
    ap.add_argument(
        "--candidate-shape",
        default="",
        metavar="DIM=N,...",
        help="applicability only: the candidate's source shape, m=,k=,n= (dense) or "
        "kh=,kw=,c=,oc= (convolution); omitted dimensions keep admission unknown",
    )
    ap.add_argument(
        "--mechanism-question",
        default="",
        help="propose a public model question to the trusted diagnostic owner; does not grade or grant gSIM",
    )
    ap.add_argument("--mechanism-discriminator", default="", help="what the proposed witnesses distinguish")
    ap.add_argument(
        "--mechanism-purpose", choices=("unresolved_cost", "structural_validation"), default="unresolved_cost"
    )
    a = ap.parse_args(argv)
    if a.write_reference_views:
        if (
            a.reference_analysis
            or a.attach
            or a.list
            or a.sweep is not None
            or a.panel
            or a.mechanism_question
            or a.tuning_overrides
            or a.reference_request
            or a.reference_path
            or a.reference_entry
            or a.reference_member
            or a.reference_region
            or a.candidate_region
            or a.reference_observation_field
            or a.reference_instruction_offset
            or a.candidate_instruction_offset
            or a.reference_format != "auto"
            or a.candidate_path
            or a.candidate_format != "auto"
            or a.candidate_symbol
            or a.candidate_offset
            or a.suite != "core"
            or a.capsules != "all"
            or a.knob
            or a.bundle
            or a.reference_offset
            or a.reference_limit != 8
            or a.submission != _SUBMISSION
            or any(
                getattr(a, key)
                for key in (
                    "candidate_diagnostics",
                    "compile_only",
                    "caller_layout",
                    "shape_coverage",
                    "offload_census",
                    "model_layers",
                    "dynamic_census",
                    "scale_probe",
                )
            )
        ):
            ap.error("--write-reference-views is a separate local artifact-pinning action")
        try:
            result = _write_reference_views(
                Path.cwd(), a.write_reference_views, entry_symbol=a.reference_symbol or "kernel_entry"
            )
        except (OSError, ValueError) as error:
            ap.error(str(error))
        print(json.dumps(result, indent=2))
        return 0
    verify_sweep = []
    if a.reference_analysis == "verify":
        if not a.before or not a.after or a.reference_path or a.reference_entry or a.candidate_path:
            ap.error("--reference-analysis verify requires --before and --after (owned paths) only")
        verify_sweep = [item for item in (a.sweep or "").split(",") if item]
        a.sweep, a.reference_path = None, a.before
    elif a.before or a.after or a.mechanism or a.verdicts or a.baseline:
        ap.error("--before/--after/--baseline/--mechanism/--verdicts require --reference-analysis verify")
    if a.reference_analysis:
        if (
            a.attach
            or a.list
            or a.sweep is not None
            or a.panel
            or a.mechanism_question
            or a.tuning_overrides
            or a.reference_request
            or a.suite != "core"
            or a.capsules != "all"
            or any(
                getattr(a, key)
                for key in (
                    "candidate_diagnostics",
                    "compile_only",
                    "caller_layout",
                    "shape_coverage",
                    "offload_census",
                    "model_layers",
                    "dynamic_census",
                    "scale_probe",
                )
            )
        ):
            ap.error("--reference-analysis is a separate read-only diagnostic action")
        if (
            not 0 <= a.reference_offset
            or not 0 <= a.candidate_offset
            or not 1 <= a.reference_limit <= 16
            or a.reference_instruction_offset < 0
            or a.candidate_instruction_offset < 0
        ):
            ap.error("reference analysis requires a nonnegative offset and limit1..16")
        a.reference_path = a.reference_path or ("reference-kernels" if a.reference_analysis == "inventory" else "")
        if not a.reference_path and not a.reference_entry:
            ap.error("reference analysis requires --reference-path or --reference-entry")
        if a.reference_entry and a.reference_analysis not in ("lift", "compare", "applicability"):
            ap.error("--reference-entry selects a reference for lift/compare")
        if a.reference_analysis == "compare":
            if len(a.reference_observation_field) > 8 or any(
                not name or len(name) > 512 for name in a.reference_observation_field
            ):
                ap.error("reference comparison accepts at most eight bounded observation fields")
            if not a.candidate_path and not a.candidate_schedule:
                ap.error("--reference-analysis compare requires --candidate-path or --candidate-schedule")
            if a.candidate_schedule and a.reference_observation_field:
                ap.error("--candidate-schedule takes no observation fields")
        elif a.reference_analysis == "applicability":
            if not (a.candidate_path or a.candidate_schedule) or a.reference_observation_field:
                ap.error(
                    "--reference-analysis applicability requires --candidate-path or --candidate-schedule "
                    "and no observation fields"
                )
            try:
                pairs = [item.partition("=") for item in a.candidate_shape.split(",") if item]
                a.candidate_shape = {key.strip(): int(number) for key, _, number in pairs}
            except ValueError:
                ap.error("--candidate-shape takes DIM=N pairs separated by commas")
        elif (
            a.candidate_path
            or a.candidate_format != "auto"
            or a.candidate_offset
            or (a.candidate_symbol and a.reference_analysis != "verify")
            or a.candidate_region
            or a.candidate_instruction_offset
            or a.reference_observation_field
            or a.candidate_shape
            or a.candidate_schedule
        ):
            ap.error("candidate comparison options require --reference-analysis compare")
        if a.candidate_schedule and not (
            a.candidate_schedule.split("/", 1)[0] == "artifacts"
            and "/" in a.candidate_schedule
            or a.candidate_schedule.count("/") >= 2
            and a.candidate_schedule.startswith("variants/")
        ):
            ap.error("--candidate-schedule must be an owned artifacts/ or variants/NAME/ file")
    elif (
        a.reference_path
        or a.reference_member
        or a.reference_format != "auto"
        or a.reference_offset
        or a.reference_entry
        or a.reference_symbol
        or a.candidate_path
        or a.candidate_format != "auto"
        or a.candidate_symbol
        or a.candidate_offset
        or a.reference_region
        or a.candidate_region
        or a.reference_instruction_offset
        or a.candidate_instruction_offset
        or a.reference_observation_field
        or a.candidate_shape
        or a.candidate_schedule
    ):
        ap.error("reference input options require --reference-analysis")
    if a.sweep is not None:
        if (
            a.submission != _SUBMISSION
            or a.panel
            or a.attach
            or a.list
            or a.mechanism_question
            or a.suite == "programs"
            or a.tuning_overrides
            or a.reference_request
        ):
            ap.error("--sweep uses submission/ as baseline and cannot combine with another action")
        a.suite = "scalability"
    elif (a.knob or a.bundle) and not a.attach:
        ap.error("--knob and --bundle require --sweep")
    if (a.tuning_overrides or a.reference_request) and (a.attach or a.list):
        ap.error("tuning/reference options submit a new check; omit them with --attach or --list")
    overrides = None
    if a.tuning_overrides:
        try:
            if len(a.tuning_overrides.encode()) > 8192:
                raise ValueError("request exceeds 8192 bytes")
            overrides = json.loads(a.tuning_overrides)
            if not isinstance(overrides, dict) or not 1 <= len(overrides) <= 4:
                raise ValueError("expected one to four knob values")
            for key, value in overrides.items():
                if not 0 < len(key) <= 64 or not (
                    value is None
                    or type(value) in (bool, int)
                    or (type(value) is float and math.isfinite(value))
                    or (type(value) is str and len(value) <= 64)
                ):
                    raise ValueError("expected bounded knob names and finite JSON scalar values")
        except (ValueError, RecursionError) as exc:
            ap.error(f"--tuning-overrides: {exc}")
    if a.reference_request and (
        len(a.reference_request) > 160
        or a.reference_request[0] not in _NAME_FIRST
        or not all(c in _NAME_CHARS for c in a.reference_request)
    ):
        ap.error("--reference-request takes a bounded request ID, not a path")
    if a.panel:
        family = a.panel.strip().lower()
        if not family or not all(c in _NAME_CHARS for c in family) or len(family) > _VARIANT_NAME_MAX:
            ap.error("--panel takes a family name such as dense, conv or gemv")
        if a.capsules != "all" or a.suite != "core" or a.mechanism_question:
            ap.error("--panel selects its own cases; drop --capsules/--suite/--mechanism-question")
        a.capsules = _PANEL + family
        # Performance on the selected Spike cycle model by default (model cycles, compared like with
        # like); `--sim gsim` still asks for RTL cycles explicitly.
        if not any(v == "--sim" or v.startswith("--sim=") for v in (argv or sys.argv[1:])):
            a.sim = "spike"
    elif (
        (a.submission != _SUBMISSION or a.tuning_overrides)
        and not a.attach
        and not any(v == "--sim" or v.startswith("--sim=") for v in (argv or sys.argv[1:]))
    ):
        # A variant or tuning A/B is a performance comparison: the Spike cycle model by default.
        a.sim = "spike"
    if a.mechanism_question:
        if (
            not a.mechanism_discriminator.strip()
            or len(a.mechanism_question) + len(a.mechanism_discriminator) > 2000
            or a.suite != "core"
            or a.capsules in ("all", "changed", "failing")
        ):
            ap.error("mechanism questions need named public capsules, a bounded discriminator, and the core suite")
        names = a.capsules.split(",")
        if not 1 <= len(names) <= 4 or len(set(names)) != len(names):
            ap.error("mechanism questions accept one to four distinct public witnesses")
    elif a.mechanism_discriminator:
        ap.error("--mechanism-discriminator requires --mechanism-question")
    if a.suite in {"scalability", "programs"}:
        # This shim is standalone inside the author sandbox; the host repeats this gate.
        explicit_sim = any(v == "--sim" or v.startswith("--sim=") for v in (argv or sys.argv[1:]))
        if explicit_sim and a.sim != "spike":
            ap.error("public Spike suites accept only --sim spike")
        a.sim = "spike"
        tiers = [v.strip() for v in a.tiers.split(",") if v.strip()]
        source_inspection = (
            a.suite == "scalability"
            and a.sweep is None
            and a.compile_only != a.candidate_diagnostics
            and not any(
                getattr(a, key, False)
                for key in (
                    "caller_layout",
                    "shape_coverage",
                    "offload_census",
                    "model_layers",
                    "dynamic_census",
                    "scale_probe",
                )
            )
            and not (a.tuning_overrides or a.reference_request or a.mechanism_question)
        )
        probing = any(
            getattr(a, k, False)
            for k in (
                "compile_only",
                "candidate_diagnostics",
                "caller_layout",
                "shape_coverage",
                "offload_census",
                "model_layers",
                "dynamic_census",
                "scale_probe",
            )
        )
        if (tiers and tiers != ["L0", "L1", "L2"]) or probing and not source_inspection:
            ap.error("public Spike suites require L0,L1,L2 and refuse inspection/RTL requests")
    if a.coverage_sweep_numerical and a.coverage_sweep is None:
        ap.error("--coverage-sweep-numerical requires --coverage-sweep")
    if a.coverage_sweep is not None and (
        a.suite != "scalability" or len(a.capsules.split(",")) != 1 or a.capsules in ("all", "failing", "changed")
    ):
        ap.error("--coverage-sweep needs --suite scalability and exactly one public --capsules template case")
    if a.timeout is None:
        a.timeout = 3600 if a.sim == "gsim" else 1800
    if a.scale_probe and not a.compile_only:
        ap.error("--scale-probe requires --compile-only")

    policy_error = (
        None
        if a.caller_layout or a.candidate_diagnostics or a.compile_only or a.reference_analysis
        else (_sim_policy_error(a.sim) if a.suite == "core" else None)
    )
    if policy_error:
        txt = json.dumps(
            {"error": policy_error, "all_pass": False, "sim": a.sim, "required_rtl_engine": _required_rtl_engine()}
        )
        print(txt)
        if a.out:
            Path(a.out).write_text(txt)
        return 2

    ws = Path(__file__).resolve().parent  # the shim lives at <ws>/agent_selfcheck.py
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True, exist_ok=True)
    if a.list:
        print(json.dumps(_list_requests(ch), indent=2))
        return 0
    manifest = _read_json(ch / f"{a.attach}.json") if a.attach and Path(a.attach).name == a.attach else None
    if a.sweep is not None or (isinstance(manifest, dict) and manifest.get("schema") == _SWEEP_SCHEMA):
        try:
            return _run_sweep(a, ws, ch, manifest)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(json.dumps({"error": str(exc), "all_pass": False}))
            return 2
    rid = _request_id()
    requested_at_ns = time.time_ns()
    label, source_error = _source_label(ws, a.submission)
    original = None
    if a.attach:
        rid = str(a.attach)
        if Path(rid).name != rid or rid in {".", ".."}:
            source_error = f"--attach takes a request id as printed by a previous call, not a path: {rid!r}"
        else:
            original = _read_json(ch / f"req_{rid}.json")
        if isinstance(original, dict):
            # The request names its own tree and bytes; --submission on a resume is not consulted.
            label, source_error = _source_label(ws, original.get("submission", _SUBMISSION))
        elif source_error is None and not any(
            (ch / f"{prefix}{rid}{suffix}").exists()
            for prefix, suffix in (("progress_", ".json"), ("claimed_", ""), ("resp_", ".json"), ("done_", ""))
        ):
            source_error = f"no self-check request {rid!r} in this workspace; --list shows the ones in flight"
    elif label is not None and label != _SUBMISSION:
        source_error = _variant_error(ws, label)
    if source_error:
        txt = json.dumps({"error": source_error, "all_pass": False, "submission": a.submission})
        print(txt)
        if a.out:
            Path(a.out).write_text(txt)
        return 2
    if isinstance(original, dict) and isinstance(original.get("submission_sha256"), str):
        submission_sha256 = original["submission_sha256"]  # what was asked about, not what is there now
    else:
        submission_sha256 = _submission_digest(ws / label)
    suite_size = None
    if a.capsules in ("all", "changed"):
        public_root = ws / "contract" / "capsules"
        discovered = sum(1 for _ in public_root.rglob("capsule.yaml")) if public_root.is_dir() else 0
        suite_size = discovered or _SUITE_SIZE_FALLBACK
    deadline = time.time() + _request_budget_seconds(
        a.timeout, capsules=a.capsules, workers=a.workers, suite_size=suite_size
    )
    if isinstance(original, dict) and isinstance(original.get("deadline_unix_ns"), int):
        deadline = original["deadline_unix_ns"] / 1_000_000_000
    # Resuming (--attach): the request is already on the channel and the broker is already working it.
    # Writing a second req_ for the same work would run a duplicate beside the first and double its cost.
    if not a.attach:
        # Caller identity is accounting metadata, never authorization.
        caller_thread = os.environ.get("CODEX_THREAD_ID", "").strip()[:128]
        _atomic_write(
            ch / f"req_{rid}.json",
            json.dumps(
                {
                    "protocol": PROTOCOL_VERSION,
                    "request_id": rid,
                    "requester": _requester_identity(),
                    **({"caller_thread_id": caller_thread} if caller_thread else {}),
                    "requested_at_unix_ns": requested_at_ns,
                    "submission": label,
                    "submission_sha256": submission_sha256,
                    **({"tuning_overrides": overrides} if overrides else {}),
                    **({"reference_experiment_request_id": a.reference_request} if a.reference_request else {}),
                    **(
                        dict(
                            action="reference-analysis",
                            reference_analysis=dict(
                                operation=a.reference_analysis,
                                path=a.reference_path,
                                format=a.reference_format,
                                member=a.reference_member,
                                offset=a.reference_offset,
                                limit=a.reference_limit,
                                symbol=a.reference_symbol,
                                entry=a.reference_entry,
                                **(
                                    dict(observation_fields=a.reference_observation_field)
                                    if a.reference_observation_field
                                    else {}
                                ),
                                **(dict(region=a.reference_region) if a.reference_region else {}),
                                **(
                                    dict(instruction_offset=a.reference_instruction_offset)
                                    if a.reference_instruction_offset
                                    else {}
                                ),
                                **(
                                    dict(
                                        candidate=dict(
                                            path=a.candidate_path,
                                            format=a.candidate_format,
                                            symbol=a.candidate_symbol,
                                            offset=a.candidate_offset,
                                            **(dict(region=a.candidate_region) if a.candidate_region else {}),
                                            **(
                                                dict(instruction_offset=a.candidate_instruction_offset)
                                                if a.candidate_instruction_offset
                                                else {}
                                            ),
                                        )
                                    )
                                    if a.reference_analysis in ("compare", "applicability") and a.candidate_path
                                    else {}
                                ),
                                **(
                                    dict(candidate_schedule=a.candidate_schedule)
                                    if a.reference_analysis in ("compare", "applicability") and a.candidate_schedule
                                    else {}
                                ),
                                **(
                                    dict(shape=a.candidate_shape)
                                    if a.reference_analysis == "applicability" and a.candidate_shape
                                    else {}
                                ),
                                **(
                                    dict(
                                        candidate=dict(
                                            path=a.after,
                                            **({"symbol": a.candidate_symbol} if a.candidate_symbol else {}),
                                        ),
                                        sweep=verify_sweep,
                                        verdicts=a.verdicts,
                                        **(dict(mechanism=a.mechanism) if a.mechanism else {}),
                                        **(dict(baseline=a.baseline) if a.baseline else {}),
                                    )
                                    if a.reference_analysis == "verify"
                                    else {}
                                ),
                            ),
                        )
                        if a.reference_analysis
                        else {}
                    ),
                    "deadline_unix_ns": int(deadline * 1_000_000_000),
                    # Omitted entirely when the caller named none: an ABSENT key lets the driver
                    # resolve the engine, where a present one is a choice the broker must honour.
                    **({"sim": a.sim} if a.sim and not a.caller_layout else {}),
                    **({"tiers": a.tiers} if a.tiers else {}),
                    **({"suite": a.suite} if a.suite != "core" else {}),
                    "capsules": a.capsules,
                    "workers": a.workers,
                    "timeout": a.timeout,
                    "shape_coverage": bool(a.shape_coverage),
                    "offload_census": bool(a.offload_census),
                    "model_layers": bool(a.model_layers),
                    "candidate_diagnostics": bool(a.candidate_diagnostics),
                    "compile_only": bool(a.compile_only),
                    "scale_probe": bool(a.scale_probe),
                    "dynamic_census": bool(a.dynamic_census),
                    "caller_layout": a.caller_layout,
                    **(
                        {
                            "coverage_sweep": a.coverage_sweep,
                            "coverage_sweep_numerical": bool(a.coverage_sweep_numerical),
                        }
                        if a.coverage_sweep is not None
                        else {}
                    ),
                    **(
                        {
                            "action": "mechanism-question",
                            "mechanism_question": {
                                "schema": "merlin.mechanism_question_proposal.v1",
                                "question": a.mechanism_question,
                                "discriminator": a.mechanism_discriminator,
                                "capsules": a.capsules.split(","),
                                "purpose": a.mechanism_purpose,
                            },
                        }
                        if a.mechanism_question
                        else {}
                    ),
                }
            ),
        )
    resp, done = ch / f"resp_{rid}.json", ch / f"done_{rid}"
    # Printed before any wait: a caller whose own wait ends first still knows where the reply lands.
    print(
        f"[self-check] request {rid}: reply will be written to .qa_channel/resp_{rid}.json "
        f"(read it there, or collect it with: python agent_selfcheck.py --attach {rid})",
        file=sys.stderr,
        flush=True,
    )
    progress = ch / f"progress_{rid}.json"
    claimed = ch / f"claimed_{rid}"
    last_progress_finished = -1
    reported_cases: dict = {}
    last_queue_note = None
    announced_claim = False
    waiting_since = time.time()
    last_waiting_note = 0.0
    # The bound that makes this survivable by a 600s-capped caller. `deadline` still governs how long
    # the REQUEST may live broker-side; this only governs how long THIS invocation blocks.
    soft_deadline = time.time() + a.wait_budget if a.wait_budget > 0 else deadline
    while time.time() < min(deadline, soft_deadline):
        # Distinguish a queued request from a missing or failed broker.
        if not announced_claim and claimed.exists():
            announced_claim = True
            try:
                behind = json.loads(claimed.read_text(encoding="utf-8")).get("queued_behind_this")
            except (OSError, ValueError, TypeError):
                behind = None
            print(
                f"[self-check] claimed by the grader after {time.time() - waiting_since:.0f}s"
                + (f"; {behind} request(s) were queued behind it" if isinstance(behind, int) else ""),
                file=sys.stderr,
                flush=True,
            )
        if not announced_claim and time.time() - waiting_since >= 15 and time.time() - last_waiting_note > 60:
            last_waiting_note = time.time()
            print(
                f"[self-check] queued, not yet claimed ({time.time() - waiting_since:.0f}s); {_capacity_note(ch)}",
                file=sys.stderr,
                flush=True,
            )
        if progress.exists():
            try:
                p = json.loads(progress.read_text(encoding="utf-8"))
                queue = _gsim_queue(p)
                if queue is not None:
                    # Waiting for a busy gSIM slot is a queue, not a fault and not a refusal.
                    if _gsim_queue_note(queue) != last_queue_note:
                        last_queue_note = _gsim_queue_note(queue)
                        print(f"[self-check] {last_queue_note}", file=sys.stderr, flush=True)
                finished = int(p.get("n_finished", 0)) if queue is None else last_progress_finished
                if finished != last_progress_finished:
                    # stderr keeps stdout machine-readable while making a long foreground check visibly
                    # alive in an interactive terminal.
                    print(
                        f"[self-check] {finished}/{p.get('n_expected', '?')} finished; counts={p.get('counts', {})}",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_progress_finished = finished
                for line in _new_case_lines(reported_cases, p) if queue is None else ():
                    print(line, file=sys.stderr, flush=True)
            except (OSError, ValueError, TypeError):
                pass
        if done.exists() and resp.exists():
            txt = resp.read_text()
            if a.reference_analysis:
                # stderr carries a one-line summary even when stdout goes to a file or is truncated.
                print(_analysis_summary_line(txt, rid), file=sys.stderr, flush=True)
            print(txt)
            if a.out:
                Path(a.out).write_text(txt)
            _v = _verdict(txt)
            if _v is None:
                # An unreadable reply cannot be reported as a pass.
                print(
                    json.dumps(
                        {"error": "self-check reply was not parseable as JSON — treating as FAILED, not as clean"}
                    )
                )
                return 2
            _ok, _why = _reply_is_usable(_v, rid, submission_sha256, label=label)
            if not _ok:
                print(json.dumps({"error": f"self-check reply {_why} — treating as FAILED, not as stale feedback"}))
                return 2
            if _why:
                print(json.dumps({"note": _why}))
            graded = _v.get("submission_sha256")
            if isinstance(graded, str) and graded and _submission_digest(ws / label) != graded:
                # The check graded a snapshot; say so when the tree has moved on since, which is the
                # normal case for a check collected with --attach after more editing.
                print(json.dumps({"note": f"{label}/ has changed since this check's snapshot ({graded[:12]})"}))
            if label != _SUBMISSION:
                print(
                    f"[self-check] {label} ({str(graded)[:12]}): variant self-check only -- not graded "
                    "officially and not counted toward convergence; copy the winner into submission/",
                    file=sys.stderr,
                    flush=True,
                )
            _print_next_steps(_v)
            # the shape-coverage report has no `all_pass`; its verdict is `all_covered`
            if a.shape_coverage:
                return 0 if _v.get("all_covered") else 1
            if a.candidate_diagnostics:
                return 0 if _v.get("schema") == "merlin.public_candidate_diagnostics.v1" else 2
            if a.compile_only:
                return 0 if _v.get("schema") == "merlin.public_compile_only_feedback.v1" and _v.get("cases") else 2
            if a.coverage_sweep is not None:
                return 0 if _v.get("schema") == "merlin.phase1.coverage_sweep.v1" and _v.get("shapes") else 2
            if _v.get("status") == "layout_only":
                supported = {"public_caller_layout_receipt_v1", "public_caller_layout_receipt_v2"}
                return 0 if _v.get("schema") in supported else 2
            if a.mechanism_question or (original or {}).get("action") == "mechanism-question":
                return 0 if _v.get("status") == "queued_for_trusted_diagnostic_review" else 1
            if a.reference_analysis or (original or {}).get("action") == "reference-analysis":
                return 0 if _v.get("schema") == "merlin.reference_analysis.v1" and _v.get("status") == "analyzed" else 2
            return 0 if _v.get("all_pass") else 1
        time.sleep(0.4)
    if a.wait_budget > 0 and time.time() < deadline:
        # Budget expiry is resumable progress, not a verdict or a failure.
        snapshot = {}
        if progress.exists():
            try:
                snapshot = json.loads(progress.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                snapshot = {}
        queue = _gsim_queue(snapshot)
        try:
            # Returning on purpose: the request now lives on its deadline, not on this process, so the
            # broker neither drops it as abandoned nor aborts it when this process exits.
            _atomic_write(ch / f"detached_{rid}", json.dumps({"detached_at_unix_ns": time.time_ns()}))
        except OSError:
            pass
        print(
            json.dumps(
                {
                    "status": "in_progress",
                    "request_id": rid,
                    "submission": label,
                    "waited_s": round(time.time() - waiting_since, 1),
                    "claimed": claimed.exists(),
                    "n_finished": snapshot.get("n_finished"),
                    "n_expected": snapshot.get("n_expected"),
                    "n_remaining": snapshot.get("n_remaining"),
                    "counts": snapshot.get("counts", {}),
                    # Every case decided so far: finished rows, and rows whose numerics are decided
                    # while later checks run ("status": "running", "final": false).
                    "n_numerics_decided": snapshot.get("n_numerics_decided"),
                    "numerics_counts": snapshot.get("numerics_counts", {}),
                    "cases": _case_rows(snapshot),
                    **({"gsim_slot_queue": queue} if queue is not None else {}),
                    "resume_with": f"python agent_selfcheck.py --attach {rid}" + (f" --out {a.out}" if a.out else ""),
                    "reply_path": f".qa_channel/resp_{rid}.json",
                    "note": (
                        (
                            f"{_gsim_queue_note(queue)}: the check waits for a free gSIM slot and then runs; "
                            if queue is not None
                            else ""
                        )
                        + "the check is STILL RUNNING broker-side; this is not a pass, a failure or a "
                        "timeout. Keep working and collect it with the command above -- do NOT write a "
                        "sleep/poll loop, and do not resubmit this same check (that only repeats it); "
                        "checks of other capsules or variants may run alongside it."
                    ),
                },
                indent=2,
            )
        )
        return 3
    print(json.dumps({"error": "self-check broker did not respond (timeout) — tell the operator"}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
