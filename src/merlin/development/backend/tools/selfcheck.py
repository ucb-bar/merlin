"""Standalone self-check client, staged as agent_selfcheck.py in the workspace.

The copied client uses the standard library and declared qa_selfcheck_* sidecars.
It exchanges requests through .qa_channel; the broker owns grading and private
answers. Variant checks are advisory; only submission/ is graded officially.
A short --wait-budget returns a request ID, and --attach collects its result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

if __package__:
    from merlin.development.backend.tools import qa_selfcheck_protocol as protocol
    from merlin.development.backend.tools import qa_selfcheck_sweeps as sweeps
else:
    import qa_selfcheck_protocol as protocol
    import qa_selfcheck_sweeps as sweeps


def _arguments(argv=None, *, client_path):
    """Return an early CLI status or the validated arguments and request options."""
    ap = argparse.ArgumentParser(
        description="Agent self-check (redacted; routed to the driver-side broker).",
        epilog=protocol._WORKFLOW,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--submission",
        default=protocol._SUBMISSION,
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
    ap.add_argument("--sim", choices=list(protocol._SIMS), default=protocol._default_sim())
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
        default=protocol._DEFAULT_WAIT_BUDGET_S,
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
        "variants/ stream document from python -m merlin.analysis.kernels.effects.schedule_replay; with "
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
            or a.submission != protocol._SUBMISSION
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
            result = protocol._write_reference_views(
                Path.cwd(),
                a.write_reference_views,
                entry_symbol=a.reference_symbol or "kernel_entry",
                client_name=Path(client_path).name,
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
            a.submission != protocol._SUBMISSION
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
        or a.reference_request[0] not in protocol._NAME_FIRST
        or not all(c in protocol._NAME_CHARS for c in a.reference_request)
    ):
        ap.error("--reference-request takes a bounded request ID, not a path")
    if a.panel:
        family = a.panel.strip().lower()
        if not family or not all(c in protocol._NAME_CHARS for c in family) or len(family) > protocol._VARIANT_NAME_MAX:
            ap.error("--panel takes a family name such as dense, conv or gemv")
        if a.capsules != "all" or a.suite != "core" or a.mechanism_question:
            ap.error("--panel selects its own cases; drop --capsules/--suite/--mechanism-question")
        a.capsules = protocol._PANEL + family
        # Performance on the selected Spike cycle model by default (model cycles, compared like with
        # like); `--sim gsim` still asks for RTL cycles explicitly.
        if not any(v == "--sim" or v.startswith("--sim=") for v in (argv or sys.argv[1:])):
            a.sim = "spike"
    elif (
        (a.submission != protocol._SUBMISSION or a.tuning_overrides)
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
        else (protocol._sim_policy_error(a.sim) if a.suite == "core" else None)
    )
    if policy_error:
        txt = json.dumps(
            {
                "error": policy_error,
                "all_pass": False,
                "sim": a.sim,
                "required_rtl_engine": protocol._required_rtl_engine(),
            }
        )
        print(txt)
        if a.out:
            Path(a.out).write_text(txt)
        return 2

    return a, overrides, verify_sweep


def main(argv=None):
    prepared = _arguments(argv, client_path=__file__)
    if isinstance(prepared, int):
        return prepared
    a, overrides, verify_sweep = prepared

    ws = Path(__file__).resolve().parent  # the shim lives at <ws>/agent_selfcheck.py
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True, exist_ok=True)
    if a.list:
        print(json.dumps(protocol._list_requests(ch), indent=2))
        return 0
    manifest = protocol._read_json(ch / f"{a.attach}.json") if a.attach and Path(a.attach).name == a.attach else None
    if a.sweep is not None or (isinstance(manifest, dict) and manifest.get("schema") == sweeps._SWEEP_SCHEMA):
        try:
            return sweeps._run_sweep(a, ws, ch, manifest)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(json.dumps({"error": str(exc), "all_pass": False}))
            return 2
    rid = protocol._request_id()
    requested_at_ns = time.time_ns()
    label, source_error = protocol._source_label(ws, a.submission)
    original = None
    if a.attach:
        rid = str(a.attach)
        if Path(rid).name != rid or rid in {".", ".."}:
            source_error = f"--attach takes a request id as printed by a previous call, not a path: {rid!r}"
        else:
            original = protocol._read_json(ch / f"req_{rid}.json")
        if isinstance(original, dict):
            # The request names its own tree and bytes; --submission on a resume is not consulted.
            label, source_error = protocol._source_label(ws, original.get("submission", protocol._SUBMISSION))
        elif source_error is None and not any(
            (ch / f"{prefix}{rid}{suffix}").exists()
            for prefix, suffix in (("progress_", ".json"), ("claimed_", ""), ("resp_", ".json"), ("done_", ""))
        ):
            source_error = f"no self-check request {rid!r} in this workspace; --list shows the ones in flight"
    elif label is not None and label != protocol._SUBMISSION:
        source_error = protocol._variant_error(ws, label)
    if source_error:
        txt = json.dumps({"error": source_error, "all_pass": False, "submission": a.submission})
        print(txt)
        if a.out:
            Path(a.out).write_text(txt)
        return 2
    if isinstance(original, dict) and isinstance(original.get("submission_sha256"), str):
        submission_sha256 = original["submission_sha256"]  # what was asked about, not what is there now
    else:
        submission_sha256 = protocol._submission_digest(ws / label)
    suite_size = None
    if a.capsules in ("all", "changed"):
        public_root = ws / "contract" / "capsules"
        discovered = sum(1 for _ in public_root.rglob("capsule.yaml")) if public_root.is_dir() else 0
        suite_size = discovered or protocol._SUITE_SIZE_FALLBACK
    deadline = time.time() + protocol._request_budget_seconds(
        a.timeout, capsules=a.capsules, workers=a.workers, suite_size=suite_size
    )
    if isinstance(original, dict) and isinstance(original.get("deadline_unix_ns"), int):
        deadline = original["deadline_unix_ns"] / 1_000_000_000
    # Resuming (--attach): the request is already on the channel and the broker is already working it.
    # Writing a second req_ for the same work would run a duplicate beside the first and double its cost.
    if not a.attach:
        # Caller identity is accounting metadata, never authorization.
        caller_thread = os.environ.get("CODEX_THREAD_ID", "").strip()[:128]
        protocol._atomic_write(
            ch / f"req_{rid}.json",
            json.dumps(
                {
                    "protocol": protocol.PROTOCOL_VERSION,
                    "request_id": rid,
                    "requester": protocol._requester_identity(),
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
                f"[self-check] queued, not yet claimed ({time.time() - waiting_since:.0f}s); {protocol._capacity_note(ch)}",
                file=sys.stderr,
                flush=True,
            )
        if progress.exists():
            try:
                p = json.loads(progress.read_text(encoding="utf-8"))
                queue = protocol._gsim_queue(p)
                if queue is not None:
                    # Waiting for a busy gSIM slot is a queue, not a fault and not a refusal.
                    if protocol._gsim_queue_note(queue) != last_queue_note:
                        last_queue_note = protocol._gsim_queue_note(queue)
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
                for line in protocol._new_case_lines(reported_cases, p) if queue is None else ():
                    print(line, file=sys.stderr, flush=True)
            except (OSError, ValueError, TypeError):
                pass
        if done.exists() and resp.exists():
            txt = resp.read_text()
            if a.reference_analysis:
                # stderr carries a one-line summary even when stdout goes to a file or is truncated.
                print(protocol._analysis_summary_line(txt, rid), file=sys.stderr, flush=True)
            print(txt)
            if a.out:
                Path(a.out).write_text(txt)
            _v = protocol._verdict(txt)
            if _v is None:
                # An unreadable reply cannot be reported as a pass.
                print(
                    json.dumps(
                        {"error": "self-check reply was not parseable as JSON — treating as FAILED, not as clean"}
                    )
                )
                return 2
            _ok, _why = protocol._reply_is_usable(_v, rid, submission_sha256, label=label)
            if not _ok:
                print(json.dumps({"error": f"self-check reply {_why} — treating as FAILED, not as stale feedback"}))
                return 2
            if _why:
                print(json.dumps({"note": _why}))
            graded = _v.get("submission_sha256")
            if isinstance(graded, str) and graded and protocol._submission_digest(ws / label) != graded:
                # The check graded a snapshot; say so when the tree has moved on since, which is the
                # normal case for a check collected with --attach after more editing.
                print(json.dumps({"note": f"{label}/ has changed since this check's snapshot ({graded[:12]})"}))
            if label != protocol._SUBMISSION:
                print(
                    f"[self-check] {label} ({str(graded)[:12]}): variant self-check only -- not graded "
                    "officially and not counted toward convergence; copy the winner into submission/",
                    file=sys.stderr,
                    flush=True,
                )
            protocol._print_next_steps(_v)
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
        queue = protocol._gsim_queue(snapshot)
        try:
            # Returning on purpose: the request now lives on its deadline, not on this process, so the
            # broker neither drops it as abandoned nor aborts it when this process exits.
            protocol._atomic_write(ch / f"detached_{rid}", json.dumps({"detached_at_unix_ns": time.time_ns()}))
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
                    "cases": protocol._case_rows(snapshot),
                    **({"gsim_slot_queue": queue} if queue is not None else {}),
                    "resume_with": f"python agent_selfcheck.py --attach {rid}" + (f" --out {a.out}" if a.out else ""),
                    "reply_path": f".qa_channel/resp_{rid}.json",
                    "note": (
                        (
                            f"{protocol._gsim_queue_note(queue)}: the check waits for a free gSIM slot and then runs; "
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
