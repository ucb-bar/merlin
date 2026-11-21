"""Agent-invokable SELF-CHECK runner — the realistic dev test loop (given to ALL arms, unlimited use).

This is the agent's own test harness: it builds the agent's package (its 4 CLI entrypoints), emits the
command buffer for each PUBLIC capsule, runs it on a chosen simulator, compares the output to the
harness's internally-computed golden, and prints a REDACTED verdict — pass/fail + mismatch_count + the
failing tier/plane, **never the golden values**. The agent may call this as many times as it wants, on
any subset, in parallel.

By default (no ``--sim``) one request runs every tier the official grade selects -- the same
adapters, cheapest first with fail-fast -- so each capsule comes back certified
(every tier it declares mandatory passed) or not, exactly as the grade would score those bytes. A
single engine name is a legacy narrower selection: ``--sim spike`` is a fast functional screen that
can refute but never certify. Cycles are NOT a criterion here — only functional/numerical correctness.

  # certify the capsules an edit touched (every mandatory tier, as the grade runs them):
  python agent_selfcheck.py --capsules B3_conv2d_i8,A1_mvin_mvout
  # an uncertified functional screen (seconds/capsule):
  python agent_selfcheck.py --sim spike --capsules B3_conv2d_i8

What it does NOT reveal: golden output tensors, private model weights, hidden capsules, or the reference
oracle. It tells you
WHETHER and roughly WHERE you are wrong (mismatch_count, failing plane) — not the answer.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

from merlin.development.backend.authoring.library.compiler_library import selected_child
from merlin.development.backend.campaign.control import selfcheck_policy
from merlin.development.backend.context import (
    InvocationContext,
    add_context_arguments,
    context_argv,
    readback_kwargs,
    resolve_context,
)
from merlin.development.backend.feedback.candidates.observations import _shape_coverage, diagnostics_cli
from merlin.development.backend.feedback.certification.promotion_admission import REQUIRED_SIM
from merlin.development.backend.feedback.grading import capsule_feedback as _merlin_capsule_feedback_3
from merlin.development.backend.feedback.grading import (
    package_rejection,
)
from merlin.development.backend.feedback.grading import qa as _qc
from merlin.development.backend.feedback.grading import (
    worker_resources as scratch_retention,
)
from merlin.development.backend.feedback.performance import brief as performance_brief
from merlin.development.backend.feedback.selfcheck import options as request_options
from merlin.development.backend.feedback.selfcheck import results as SP
from merlin.development.backend.feedback.selfcheck import results as selfcheck_notes
from merlin.development.backend.feedback.selfcheck.diagnostic_adapters import (
    _adapters,
    _atomic_json,
    _model_layers,
    _public_capsules,
    _request_artifact_root,
    _required_rtl_engine,
    _sim_policy_error,
    _streaming_completion,
    _strip_build_state,
    _suite_size,
    _target,
    _target_sim_via,
    worker_command,
)
from merlin.development.backend.feedback.selfcheck.results import (
    _candidate_selfcheck_row,
    _gated_without_result_rows,
    _log_telemetry,
    _offload_census,
    _operator_feedback_state,
    _operator_structural_feedback,
    _public_candidate_emission_failure,
    _selfcheck_counts,
    _successful_expected_refusal,
    compact_document,
)
from merlin.development.backend.feedback.selfcheck.selection import (
    _previously_failing,
    capsule_barrier,
    grade_l3_selection,
    select_capsules,
    select_tiers,
    selection_note,
    tier_engines,
)
from merlin.development.backend.session.inputs.oracle_timing import barrier_measurements
from merlin.development.targetgen.corpus import capsule_io as capsule_capsule_common
from merlin.development.targetgen.grading import model_checks as CG  # build+run+compare
from merlin.development.targetgen.grading import submission as _merlin_submission_2
from merlin.evaluation.capsule import eligibility as capsule_eligibility
from merlin.evaluation.capsule.evidence import certificates as capsule_adapters
from merlin.evaluation.capsule.evidence import certificates as capsule_availability

# ---- selfcheck ---------------------------------------------------------------------------------


_progress_snapshot, _progress_publisher = SP.progress_snapshot, SP.progress_publisher


SIM_TIER = {"spike": "L2", "verilator": "L3", "gsim": "L3", "vcs": "L4"}


SIM_CHOICES = (REQUIRED_SIM, *SIM_TIER)


@selected_child
@selfcheck_policy
@scratch_retention.releases_selfcheck_scratch
def main(argv=None, *, context=None, capsules_root: Path | None = None, contract: Path | None = None):
    ap = request_options.parser()
    a = ap.parse_args(argv)
    request_deadline = time.monotonic() + a.timeout  # the broker stops this check 240 s after its timeout
    if a.artifact_root:
        try:
            a.artifact_root = _request_artifact_root(a.artifact_root, progress=a.progress_out, output=a.out)
        except ValueError as error:
            ap.error(str(error))
    if a.scale_probe and not a.compile_only:
        ap.error("--scale-probe requires --compile-only")
    if a.rtl_facts is not None:
        from merlin.development.backend.session.inputs.run_record import select

        select(Path.cwd(), a.rtl_facts)
    context = resolve_context(a, ap, context)
    from merlin.development.backend.feedback.disclosure import private_cli, restricted, unavailable

    if restricted() and (a.shape_coverage or a.caller_layout or a.capsules.startswith("panel:")):
        return unavailable(a.out)
    if a.capsules.startswith("panel:") and a.suite == "core":  # one request per shape class (shape_panel)
        from merlin.development.backend.feedback.selfcheck.shape_panel import cli as panel

        return panel(argv, a, context=context, run=main, root=a.capsules_root or capsules_root, contract=contract)
    if a.suite == "programs":
        from merlin.development.backend.feedback.certification.model_checks import cli

        return private_cli(a, lambda request: cli(request, context=context))
    if a.suite == "scalability":
        from merlin.development.backend.feedback.scalability.public_scalability import cli

        return private_cli(a, lambda request: cli(request, context=context))
    if a.compile_only:
        from merlin.development.backend.feedback.execution.profile_runtime import selfcheck_compile_only

        return private_cli(a, lambda request: selfcheck_compile_only(request, context=context))
    if a.caller_layout:
        try:
            if a.rtl_facts is None:
                raise ValueError("selected frozen RTL facts are required")
            from merlin.development.backend.feedback.candidates.caller_layout import inspect_caller_layout

            result = inspect_caller_layout(
                submission=Path(a.submission),
                command_buffer_member=a.caller_layout,
                target=context.target,
                facts_path=a.rtl_facts,
            )
        except Exception as exc:  # noqa: BLE001 -- never print private host paths or provider text
            result = {"status": "unavailable", "error": f"caller layout refused: {type(exc).__name__}"}
        txt = json.dumps(result, indent=2, sort_keys=True)
        print(txt)
        if a.out:
            Path(a.out).write_text(txt)
        return 0 if result.get("status") == "layout_only" else 2
    if a.sim is None and not a.candidate_diagnostics:
        a.sim = REQUIRED_SIM
    contract = a.contract if a.contract is not None else contract
    capsules_root = a.capsules_root if a.capsules_root is not None else capsules_root
    if capsules_root is None:
        capsules_root = _public_capsules(context)
    capsules_root = Path(capsules_root)

    if a.candidate_diagnostics or a.shape_coverage or a.model_layers or a.offload_census:
        from merlin.development.backend.feedback.execution.profile_runtime import selfcheck_diagnostics

        def diagnostics(request):
            with selfcheck_diagnostics(request.compiler_bundle_manifest, context=context, arguments=request):
                if request.candidate_diagnostics:
                    return diagnostics_cli(request, context=context, capsules_root=capsules_root, contract=contract)
                if request.shape_coverage:
                    return _shape_coverage(Path(request.submission), request.out, context=context, contract=contract)
                if request.model_layers:
                    return _model_layers(
                        Path(request.submission),
                        request.out,
                        timeout=request.timeout,
                        workers=request.workers,
                        capsules=request.capsules,
                        context=context,
                        contract=contract,
                        artifact_root=request.artifact_root or None,
                    )
                if request.offload_census:
                    return _offload_census(
                        Path(request.submission),
                        request.out,
                        context=context,
                        capsules_root=capsules_root,
                        contract=contract,
                        timeout=request.timeout,
                        workers=request.workers,
                    )

        return private_cli(a, diagnostics)

    sub = Path(a.submission)
    if not (sub / "manifest.yaml").exists():
        print(json.dumps({"error": f"no {sub}/manifest.yaml — build your package first"}))
        return 2
    if not capsules_root.is_dir():
        print(json.dumps({"error": "public capsule set not found"}))
        return 2

    # FAITHFUL FEEDBACK: grade an ISOLATED COPY of the submission (no bench_contract / repo files sitting
    # next to it), exactly like the real driver grade (cand_00). Otherwise the self-check runs the agent's
    # entrypoints from a workspace where bench_contract IS present, so an agent reading a package-relative
    # repo path (e.g. src/merlin/resources/contracts/schemas/...) PASSES self-check but the standalone grade CRASHES — the
    # self-check would lie. EXCLUDE build/ exactly like the driver grade (a copied CMakeCache has absolute
    # paths to the original location and would break) -> a from-scratch build, same as the grade (fast: the
    # LLVM/MLIR is prebuilt, the OOT project is small).
    import tempfile as _tf2

    _isodir = Path(_tf2.mkdtemp(prefix="selfcheck_iso_"))
    scratch_retention.temporary(_isodir)
    shutil.copytree(sub, _isodir / "submission", ignore=shutil.ignore_patterns("build", "__pycache__", ".git"))
    sub = _isodir / "submission"
    _strip_build_state(sub)  # every grade builds from scratch in its OWN path — see helper
    from merlin.development.backend.brokers.selfcheck_dispatch import _submission_digest

    original_compiler_sha = _submission_digest(sub)

    _tgt, _sim_via = _target_sim_via(context)
    try:
        adapters, sim = _adapters(a.sim, _tgt, _sim_via, **readback_kwargs(context))
    except ValueError as exc:
        out = {"error": str(exc), "all_pass": False, "sim": a.sim, "required_rtl_engine": _required_rtl_engine()}
        txt = json.dumps(compact_document(out, workspace=Path.cwd(), namespace="selfcheck"), indent=2)
        print(txt)
        if a.out:
            Path(a.out).write_text(txt)
        _log_telemetry(out, a.capsules)
        return 2
    grade_tiers = set(adapters) if sim == REQUIRED_SIM else None
    if a.tiers:
        # Validate the REQUEST against everything the endpoint can reach, not against the cheap loop
        # ladder -- that conflation is exactly what select_tiers documents, and it is why a cert tier
        # (derived as oracle_adapters - qa_loop_adapters) could never be asked for.
        _full = dict(capsule_adapters.oracle_adapters(_tgt, _sim_via, **readback_kwargs(context)))
        _full.update(adapters)  # a --sim-selected tier is reachable by definition
        adapters, _tier_err = select_tiers(_full, adapters, a.tiers)
        if _tier_err:
            # Fail CLOSED and leave the reason on disk. Exiting without writing --out is what produced
            # the broker's contentless "no verdict produced".
            out = {"error": _tier_err, "all_pass": False, "sim": sim, "tiers_requested": a.tiers}
            txt = json.dumps(compact_document(out, workspace=Path.cwd(), namespace="selfcheck"), indent=2)
            print(txt)
            if a.out:
                Path(a.out).write_text(txt)
            _log_telemetry(out, a.capsules)
            return 2
    # The grade's whole selection, unnarrowed by --tiers, judges each capsule exactly as the grade does:
    # per-capsule mandatory barrier, and a tier that could not run is never a pass. A narrowed request
    # (a promotion's `--tiers L3`) keeps the screen accounting its consumers read.
    combined = grade_tiers is not None and set(adapters) == grade_tiers
    # barrier = the deepest resolved RTL tier: chipyard maps a named engine from --sim; the grade's own
    # selection and any other target's contract-derived tiers are read from the adapters.
    barrier_tier = SIM_TIER[sim] if _sim_via == "chipyard" and sim in SIM_TIER else max(adapters) if adapters else "L3"
    # subset selection (operator-side capsule dirs)
    selection = select_capsules(a, sub, context, capsules_root, contract, adapters, combined, _tgt, barrier_tier)
    if selection.answered is not None:
        return selection.answered
    want = selection.want
    import tempfile

    from merlin.development.backend.feedback.selfcheck.results import create_runs_root

    runs_root = create_runs_root()
    scratch_retention.raw_runs(runs_root)

    # Subset filtering that ACTUALLY limits the grade: CG.grade runs EVERY capsule under capsules_root, so
    # the `want` output-filter below alone would still grade all 20 (a "1-capsule verilator" check = ~1hr).
    # Build a temp capsules_root symlinking only the requested capsule dirs -> grade runs just the subset.
    caps_root = capsules_root
    if want:
        caps_root = Path(tempfile.mkdtemp(prefix="selfcheck_caps_"))
        scratch_retention.temporary(caps_root)
        missing = [n for n in sorted(want) if not (capsules_root / n / "capsule.yaml").exists()]
        if missing:
            print(json.dumps({"error": f"unknown capsule(s): {missing}"}))
            return 2
        for n in sorted(want):  # copy (not symlink): rglob won't recurse symlinked dirs
            shutil.copytree(capsules_root / n, caps_root / n)

    # build + run + compare (parallel); CG.grade handles the agent's 4 entrypoints + the tier ladder.
    # Publish completed rows independently of the final verdict: a few cycle-cap cases must not hide all
    # fast feedback. The final response remains a separate atomic document owned by the broker.
    cb_root = runs_root / "runs" / capsule_eligibility.suite_for(_tgt)
    _progress_stop = threading.Event()
    _progress_thread = None
    if a.progress_out:
        _progress_thread = threading.Thread(
            target=_progress_publisher,
            args=(Path(a.progress_out), cb_root),
            kwargs={
                "expected": len(want) if want else _suite_size(capsules_root),
                "started_ns": time.time_ns(),
                "stop": _progress_stop,
            },
            name="selfcheck-progress",
            daemon=True,
        )
        _progress_thread.start()
    try:
        completion = _streaming_completion(
            context=context,
            capsules_root=caps_root,
            contract=contract,
            source=sub,
            runs=runs_root,
            adapters=adapters,
        )
        model_continuations = []
        if a.model_gate_voucher:
            from merlin.development.backend.feedback.certification.model_gate_vouchers import verify
            from merlin.development.backend.feedback.certification.promotion_ledger import (
                _submission_digest as promotion_digest,
            )

            if not want or set(adapters) != {"L2", "L3"}:
                raise ValueError("model continuation requires an explicit L2/L3 model selection")
            model_continuations = [
                verify(
                    path,
                    source_digest=promotion_digest(sub.parent),
                    capsules=want,
                    context=context,
                    cert_tier="L3",
                    contract=contract,
                )
                for path in a.model_gate_voucher
            ]
        from merlin.development.backend.feedback.execution.profile_runtime import selfcheck_execution

        with selfcheck_execution(
            a.compiler_bundle_manifest, context=context, submission=sub, work_root=_isodir / "profile-package-work"
        ):
            from merlin.development.backend.feedback.certification.rolling_model_gate import (
                prepare as prepare_rolling_gate,
            )

            rolling, rolling_complete = prepare_rolling_gate(
                context=context, contract=contract, source=sub, runs=runs_root, graded_root=capsules_root
            )
            if rolling is not None:
                if want is None or set(want) & set(rolling.gate.models):
                    rolling.load()
                previous_completion = completion

                def completed_with_prerequisites(row):
                    rolling_complete(row)
                    if previous_completion is not None:
                        previous_completion(row)

                completion = completed_with_prerequisites
            from contextlib import nullcontext

            with rolling.observing() if rolling is not None else nullcontext():
                _score = _merlin_submission_2.grade(
                    str(sub),
                    capsules_root=str(caps_root),
                    runs_root=str(runs_root),
                    labels={"public", "dev"},
                    contract=str(contract if contract is not None else context.repo / "src/merlin/resources/contracts"),
                    oracle_adapters=adapters,
                    timeout=a.timeout,
                    max_workers=a.workers,
                    target=_tgt,
                    model_gate_continuations=tuple(model_continuations),
                    model_gate_prerequisites=rolling.gate if rolling is not None else None,
                    **({"on_capsule_complete": completion} if completion is not None else {}),
                )
    except Exception as e:
        print(json.dumps({"error": f"grade failed: {str(e)[:300]}"}))
        return 1
    finally:
        if _progress_thread is not None:
            _progress_stop.set()
            _progress_thread.join(timeout=2)

    # SURFACE a build / integrity failure that prevented ANY capsule from running. Without this the agent
    # only sees n_capsules=0 with no reason and (as happened) misreads it as a "stubbed grader". The build
    # stderr is compiler/cmake output — it contains no golden tensors, so it is safe to show.
    if isinstance(_score, dict) and _score.get("failure"):
        f = _score["failure"]
        out = {
            "sim": sim,
            "barrier_tier": barrier_tier,
            "n_passed": 0,
            "n_capsules": 0,
            "all_pass": False,
            "per_capsule": [],
            "build_failed": True,
            "failure_plane": f.get("plane"),
            "failure_category": f.get("category"),
            "detail": f.get("detail"),
            **package_rejection.namespace_hint(f),
            "note": selfcheck_notes.package_failure_note(f),
        }
        txt = json.dumps(compact_document(out, workspace=Path.cwd(), namespace="selfcheck"), indent=2)
        print(txt)
        if a.out:
            Path(a.out).write_text(txt)
        _log_telemetry(out, a.capsules)
        return 1

    # Ordinary operator capsules retain full developer visibility into their own command buffer,
    # trace, console and redacted numeric detail. Candidate whole-model capsules instead use QA's
    # closed submitted-artifact projection: the runner-owned host graph and candidate ELF are distinct
    # programs, so its legacy trace, timing, console or failure must not be shown as candidate evidence.
    out_dir = Path(a.artifact_root or "selfcheck_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Read results from the TARGET'S OWN suite dir. run_capsule writes under cfg.suite
    # (e.g. <target>-capsule-bench); globbing one target's SUITE literal here made every other target's
    # self-check return n_capsules:0 with per_capsule:[] — the agent's feedback loop went blind while
    # the driver's in-memory grade was correct (the atlas 0/11 blind-loop bug).
    rows, npass, ncert, nscreened = [], 0, 0, 0
    _results = sorted(cb_root.glob("*/capsule_result.json")) if cb_root.exists() else []
    closed_rows = _merlin_capsule_feedback_3._per_capsule_from_results(runs_root)
    try:
        selected = capsule_capsule_common.discover_capsules(
            caps_root,
            labels={"public", "dev"},
            contract=contract if contract is not None else context.repo / "src/merlin/resources/contracts",
        )
        original_capsules = {cap["name"]: cap for cap in selected}
        if len(original_capsules) != len(selected):
            raise ValueError("selected public capsule names are ambiguous")
    except (OSError, ValueError, KeyError, TypeError):
        selected, original_capsules = None, {}
    # Does the declared barrier EVER produce a verdict for this target? Two very different situations
    # look identical per-capsule, and only this whole-corpus view separates them:
    #   • the tier ran for some capsule but not this one -> this capsule genuinely fell short of the
    #     barrier and MUST fail; falling back would pass it on weaker evidence than the bar demands.
    #   • the tier ran for NO capsule -> the declared barrier is one the target cannot produce at all
    #     (a descriptor/adapter mismatch), and scoring against it fails everything regardless of merit.
    # Only the second case licenses a fallback.
    _declared_ran = False
    for _cr in _results:
        try:
            _d = json.loads(_cr.read_text())
        except Exception:
            continue
        _summary = _merlin_capsule_feedback_3._candidate_native_feedback(_d)
        if _summary is not None:
            _closed = closed_rows.get(_d.get("capsule", _cr.parent.name)) or {}
            _candidate = _closed.get("candidate_native_verification") or {}
            _s = (_candidate.get("tiers") or {}).get(barrier_tier)
            ran = _s in ("pass", "fail")
        else:
            _s = ((_d.get("tiers") or {}).get(barrier_tier) or {}).get("status")
            ran = _s not in (None, "skipped")
        if ran:
            _declared_ran = True
            break
    census_deadline = time.monotonic() + min(30, a.timeout)
    observations = []  # one timing observation per passing row, each on its own budget (below)
    for cr in _results:
        try:
            d = json.loads(cr.read_text())
        except Exception:
            continue
        name = d.get("capsule", cr.parent.name)
        if want and name not in want:
            continue
        if _merlin_capsule_feedback_3._candidate_native_feedback(d) is not None:
            row, certified = _candidate_selfcheck_row(d, closed_rows.get(name), name=name, barrier_tier=barrier_tier)
            if a.dynamic_census:
                row["dynamic_census"] = {"status": "UNKNOWN", "reason": "distinct own model ELF owner required"}
            rows.append(row)
            npass += int(row["pass"])
            ncert += int(certified)
            nscreened += int(row.get("screened_only") is True)
            continue
        _tier_results = d.get("tiers") or {}
        tiers = {t: (v or {}).get("status") for t, v in _tier_results.items()}
        bar_used = capsule_barrier(_tier_results, adapters, barrier_tier) if combined else barrier_tier
        bar = tiers.get(bar_used)
        if bar is None:
            # A mixed suite can legitimately have different tier shapes. Whole-model capsules, for
            # example, do not produce the per-op command-buffer screen tier, but can produce a DEEPER
            # required model/RTL verdict. A corpus-wide `_declared_ran` must not let an unrelated op's
            # L2 result veto that stronger per-capsule L3 evidence. Derive this exception entirely from
            # the capsule result: the substitute must be mandatory, must actually have run, and must be
            # deeper than the missing barrier. Thus an op that stopped at L0/L1 still fails closed, and
            # an advisory tier cannot certify it.
            _ran_required = [
                t
                for t, v in _tier_results.items()
                if bool((v or {}).get("mandatory")) and (v or {}).get("status") not in (None, "skipped", "unavailable")
            ]
            _deeper_required = [t for t in _ran_required if t > barrier_tier]
            if _deeper_required:
                bar_used = max(_deeper_required)
                bar = tiers.get(bar_used)
            elif not _declared_ran:
                # The declared barrier never ran for ANY capsule, so scoring against it marks EVERY
                # capsule failed no matter what the grade said. Measured on atlas: the barrier resolved
                # to L4 while its results carry only {L0 skipped, L1 skipped, L2 pass}. Fall back to the
                # deepest tier that actually produced a verdict. `status == "pass"` is still required
                # below, and a skipped tier is never treated as a barrier.
                ran = [k for k, v in tiers.items() if v not in (None, "skipped")]
                if ran:
                    bar_used = max(ran)
                    bar = tiers.get(bar_used)
        # CLASSIFY UNDER THE ORACLE SELECTION IN FORCE.
        #
        # A capsule declares the tiers it requires; an oracle selection supplies adapters for some subset.
        # A required tier with no adapter here did not RUN, and a tier that did not run is absence of
        # evidence about the backend, not evidence against it. `NOT_RUN_IS_NOT_PASS` is the right rule for
        # a CERTIFYING grade -- it stops an unrun tier being read as a pass -- but applying it as a FAILURE
        # inside a screen scores the submission for the engine the caller picked.
        #
        # Measured: a submission whose capsule graded {L0 pass, L1 pass, L2 pass, L3 unavailable} scored
        # 0, and a run spent 325 consecutive self-checks at zero while the certifying grade of the same
        # submission was 22/25. An agent told it fails everything rewrites working code.
        #
        # So: `passed` = passed everything this selection could measure; `certified` = also cleared every
        # mandatory tier and applicable original structural obligation. `all_pass` keys on certification -- it may only
        # eliminate. Derived from the capsule's own declared tiers and the adapters present; no engine or
        # target is named here.
        # Use the same original discovery/schema/name validation as grading,
        # including its capsule-local source identity and inherited declarations.
        _original = original_capsules.get(name)
        certified, screened, refused_as_required, _structural = _operator_feedback_state(
            # Under the grade's selection the barrier is the capsule's own deepest mandatory tier; one
            # that could not run has no verdict, so it is accounted as absent (never as a pass).
            d,
            _original,
            target=_tgt,
            barrier_status=None if combined and bar == "unavailable" else bar,
        )
        # Under the grade's own selection a tier that could not run is not a property of the engine
        # the caller picked, so a screened row is NOT a pass: only certification passes.
        passed = certified if combined else certified or screened
        npass += int(passed)
        ncert += int(certified)
        nscreened += int(screened)
        # numeric diagnostics — keep ALL stats; redact ONLY the golden expected value
        num = dict(d.get("numeric") or {})
        fm = num.get("first_mismatch")
        if isinstance(fm, dict):
            num["first_mismatch"] = {
                "output": fm.get("output"),
                "index": fm.get("index"),
                "your_value": fm.get("observed"),
            }  # 'expected' (golden) withheld
        # the agent's OWN emitted artifacts -> copy out for inspection + summarize the trace
        gen = cr.parent / "generated"
        own = {}
        for art in ("command_buffer.json", "instruction_trace.json", "lowered.llvm.mlir"):
            src = gen / art
            if src.exists():
                dst = out_dir / name / art
                from merlin.development.backend.feedback.disclosure import copy_candidate_artifact, restricted

                if restricted():
                    if not copy_candidate_artifact(src, dst):
                        continue
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(src, dst)
                own[art] = str(dst)
        linked_view = None
        if a.compiler_bundle_manifest is not None and not restricted():
            from merlin.development.backend.feedback.candidates.candidate_build_views import (
                publish_feedback,
                registry_root,
            )

            linked_view = publish_feedback(
                workspace=Path.cwd(),
                public_root=(out_dir / name).absolute(),
                private_root=registry_root(a.compiler_bundle_manifest),
                generated=gen.absolute(),
                compiler=sub.absolute(),
                compiler_snapshot_sha256=original_compiler_sha,
            )
        tr = {}
        try:
            t = json.loads((gen / "instruction_trace.json").read_text())
            tr = t.get("summary") or {}
        except Exception:
            pass
        row = {
            "capsule": name,
            "pass": passed,
            "structural_qualification": _structural,
            "numeric_consistency_passed": (d.get("numeric") or {}).get("status") == "pass",
            "numerical_execution_passed": any(tiers.get(tier) == "pass" for tier in adapters),
            "execution_digest": _qc._execution_digest_from_result(cr),
            "barrier_tier": bar_used,
            "barrier_declared": barrier_tier,
            "barrier_status": bar,
        }
        if linked_view is not None:
            row["candidate_build_view"] = linked_view
        carried_from = (
            (d.get("tier_reuse") or {}).get("carried_from") if isinstance(d.get("tier_reuse"), dict) else None
        )
        if isinstance(carried_from, str) and carried_from.startswith("carried from ") and len(carried_from) <= 160:
            row["carried_from"] = (
                carried_from  # a seeded grade's pass, carried in full (merlin.development.targetgen.grading.seeded_grade)
            )
        from merlin.development.backend.feedback.campaign import correctness as joint_physical_correctness
        from merlin.development.backend.feedback.certification import tier_evidence as tier_boundary_join

        if name in joint_physical_correctness.roster() and bar_used in {"L2", "L3"}:
            row["tier_boundary"] = tier_boundary_join.optional_observation(cr, tier=bar_used)
        if combined:
            # The evidence behind a certified verdict is small and is the point of this selection.
            row["certified"] = certified
            row["tiers"] = tiers
            if screened:
                row["screened_only"] = True
        if refused_as_required:
            row["expected_refusal"] = dict(d["expected_refusal"])
            row["qualification"] = "expected_refusal"
            row["native_executed"] = False
        # A post-search physical stage needs a per-capsule observation bound, not one corpus-wide
        # default. Preserve the cycle count from the tier that actually supplied this row's barrier
        # verdict. This is measurement metadata, not answer-bearing data, and is sealed with the score.
        _barrier_record = (d.get("tiers") or {}).get(bar_used) or {}
        row.update(barrier_measurements(_barrier_record, cert_tier=bar_used == "L3"))
        # The L2 cycle count is meaningful only with the exact selected simulator bytes and timing
        # configuration that produced it.  The selected OOT adapter writes this generic sidecar around
        # its successful invocation; absent/malformed provenance stays absent so the downstream seal
        # rejects the row. Historical cyclotron_engine_binding.json files remain inspectable as old
        # artifacts, but cannot silently qualify a newly scored run.
        if bar_used == "L2":
            try:
                _binding = json.loads((gen / "l2_engine_binding.json").read_text())
            except (OSError, json.JSONDecodeError):
                _binding = None
            if isinstance(_binding, dict):
                row["barrier_engine_binding"] = _binding
        # A STATED DECLINE IS THE MOST ACTIONABLE THING THIS REPORT CAN CARRY, so it rides the row
        # whether or not the capsule passed, and ahead of the numeric block. Without it a declined
        # capsule reads exactly like one whose arithmetic is wrong -- an output of zeros and a
        # "does not compute the declared operation" -- and the fix for those two is not the same.
        if d.get("declined"):
            row["declined"] = d["declined"]
            row["declined_note"] = (
                ("declined as required by this capsule; no native execution is claimed")
                if refused_as_required
                else (
                    "your backend DECLINED this capsule -- it never emitted a program. "
                    "This is a coverage gap (a shape/op you do not lower), not a "
                    "numeric bug; do not debug the arithmetic."
                )
            )
        if not passed:
            # Only the selected barrier tier can supply this diagnostic. A lower-tier DONE log
            # is not evidence that an unmeasured higher tier reached DONE. The grader records
            # the console basename on the tier itself; malformed or absent names stay null.
            _tier_log = _barrier_record.get("console_log") if isinstance(_barrier_record, dict) else None
            _artifacts = cr.parent / "artifacts"
            console_tail = console_tail_unreadable = None
            if (
                isinstance(_tier_log, str)
                and _tier_log.endswith("_console.log")
                and Path(_tier_log).name == _tier_log
                and not _artifacts.is_symlink()
            ):
                _console = _artifacts / _tier_log
                if _console.is_file() and not _console.is_symlink():
                    try:
                        # Large public readouts can be many megabytes. Decode only the bounded
                        # suffix needed for the agent-facing tail, never the whole console.
                        with _console.open("rb") as _stream:
                            _stream.seek(0, os.SEEK_END)
                            _stream.seek(max(0, _stream.tell() - 4096))
                            console_tail = _stream.read(4096).decode("utf-8", errors="replace")[-800:]
                    except (OSError, UnicodeError) as exc:  # reported, never shown as an absent console
                        console_tail_unreadable = f"{type(exc).__name__}: {exc}"
            # FULL debug detail ONLY for a FAILING capsule (the one you are working). A passing capsule's
            # diff stats / trace dump / console tail are noise that re-inflates the agent's context every
            # round (the self_check output is re-fed each turn) — the pass flag is all that's needed for it.
            row.update(
                {
                    "tiers": tiers,
                    "numeric": num,  # full diff stats, reference value redacted
                    "trace_summary": tr,  # YOUR instruction counts (mvin/mvout/compute/…)
                    "trace_check": d.get("trace_check"),
                    "failure": d.get("failure") or _public_candidate_emission_failure(d),  # bounded compiler detail
                    "your_artifacts": own,  # copied to ./selfcheck_out/<capsule>/
                    "sim_console_tail": console_tail,
                    **({"sim_console_tail_unreadable": console_tail_unreadable} if console_tail_unreadable else {}),
                }
            )
        if a.dynamic_census:
            from merlin.development.backend.feedback.candidates.observations import attach_candidate_census

            attach_candidate_census(
                row,
                gen,
                result=d,
                context=context,
                compiler=sub,
                remaining_s=max(0, census_deadline - time.monotonic()),
            )
        observations.append(dict(row=row, root=gen, result=d, compiler=sub, capsule=_original))
        rows.append(row)
    from merlin.development.backend.feedback.campaign.timing import attach_all

    attach_all(observations, context=context, request_deadline=request_deadline)
    n_result = len(rows)
    try:
        if selected is None:
            raise ValueError("original public cohort could not be validated")
        requested_names = [cap["name"] for cap in selected]
        if len(set(requested_names)) != len(requested_names):
            raise ValueError("selected public capsule names are ambiguous")
        requested = set(requested_names)
        models = {cap["name"] for cap in selected if cap.get("kind") == "model"}
        if want is not None and requested != want:
            raise ValueError("selected public capsule names differ from the request")
        gated = _gated_without_result_rows(
            _score, requested=requested, models=models, represented={row["capsule"] for row in rows}
        )
        suite_size = _suite_size(capsules_root)
        scope = "all" if want is None else "subset"
        counts = (
            _selfcheck_counts(
                result_rows=n_result,
                gated_rows=len(gated),
                passed=npass,
                certified=ncert,
                requested_size=len(requested),
                suite_size=suite_size,
                scope=scope,
            )
            if n_result
            else None
        )
    except (OSError, ValueError, KeyError, TypeError):
        out = {
            "sim": sim,
            "barrier_tier": barrier_tier,
            "all_pass": False,
            "score_integrity_fault": True,
            "error": "self-check score does not match the selected public capsules and result rows",
        }
        txt = json.dumps(compact_document(out, workspace=Path.cwd(), namespace="selfcheck"), indent=2)
        print(txt)
        if a.out:
            Path(a.out).write_text(txt)
        _log_telemetry(out, a.capsules)
        return 1
    # A model-only request has no operation evidence to clear the capstone gate.
    # The grade records that deferral in memory, not as a capsule_result.json.
    # Name it before the genuinely unexpected no-results path below.
    if n_result == 0:
        if gated and len(gated) == len(_score["per_capsule"]):
            out = {
                "sim": sim,
                "barrier_tier": barrier_tier,
                "n_passed": 0,
                "n_capsules": len(gated),
                "all_pass": False,
                "per_capsule": gated,
                "model_gate": True,
                "scope": "subset" if want else "all",
                "note": "MODEL GATE: these capstones were not executed because this request supplied no "
                "passing operation suite. Run the admitted operation suite first; use --model-layers "
                "for a separate functional layer diagnostic. A gated model is not a compiler pass.",
            }
            txt = json.dumps(compact_document(out, workspace=Path.cwd(), namespace="selfcheck"), indent=2)
            print(txt)
            if a.out:
                Path(a.out).write_text(txt)
            _log_telemetry(out, a.capsules)
            return 1
    # Otherwise no top-level build failure and no capsule_result is a harness/path
    # problem. Never read it as an empty-but-clean verdict.
    if n_result == 0:
        out = {
            "sim": sim,
            "barrier_tier": barrier_tier,
            "n_passed": 0,
            "n_capsules": 0,
            "all_pass": False,
            "per_capsule": [],
            "no_results": True,
            "caps_discovered": sum(1 for _ in capsules_root.glob("*/capsule.yaml")),
            "results_root": str(cb_root),
            "note": "HARNESS ISSUE: the build did not fail, but ZERO capsule results were found at "
            "results_root. This is NOT 'all clear' and NOT a stubbed grader — it means the "
            "grade wrote results somewhere this reader did not look. Report this; it is not "
            "something your dialect can fix.",
        }
        txt = json.dumps(compact_document(out, workspace=Path.cwd(), namespace="selfcheck"), indent=2)
        print(txt)
        if a.out:
            Path(a.out).write_text(txt)
        _log_telemetry(out, a.capsules)
        return 1
    rows.extend(gated)
    n_declined = sum(1 for r in rows if r.get("declined"))
    # A capsule can fail here for a reason the agent does not control: it declares a mandatory cert tier
    # that the SIM CHOSEN FOR THIS RUN supplies no adapter for. That is a screen being asked to certify,
    # not a defect in the submission, and it must say so -- being told to fix an oracle it cannot reach is
    # how a conformant run decays into ten rounds of effort against nothing.
    _unreached = sorted(
        {
            (r.get("failure") or {}).get("tier")
            for r in rows
            if (r.get("failure") or {}).get("category") == "NOT_RUN_IS_NOT_PASS"
            and (r.get("failure") or {}).get("tier_status") == "unavailable"
        }
        - {None}
    )
    # `all_pass` covers the complete requested cohort, not the entire suite for
    # subsets. A gated model has no result and cannot pass that cohort; the
    # separate `certified_complete` still requires the whole public suite.
    has_candidate_rows = any("candidate_native_verification" in row for row in rows)
    if counts is None:  # the no-results branches returned above
        raise RuntimeError("self-check result accounting is incomplete")
    out = {
        "sim": sim,
        "barrier_tier": barrier_tier,
        "oracle_tiers": sorted(adapters),
        "tier_engines": tier_engines(adapters),
        **counts,
        "scope": scope,
        "suite_size": suite_size,
        # `n_passed` is what this oracle selection could measure; `n_certified` is what cleared every
        # mandatory tier and applicable structural requirement. A missing tier or structural evidence
        # leaves a numerical/native pass as a screen. `all_pass` keys on certification.
        "n_screened_only": nscreened,
        "n_structural_unqualified": sum(
            row.get("structural_qualification", {}).get("status") not in (None, "pass") for row in rows
        ),
        "per_capsule": rows,
        "n_declined": n_declined,
        "note": (
            (
                f"⚠ PARTIAL: {n_result} of {suite_size} public capsule(s) returned results; "
                f"{len(gated)} requested model(s) were gated without execution, and "
                f"{counts['n_unknown']} have no status in this request. "
                f"`all_pass` below covers only what you asked for — `certified_complete` "
                f"is the one that means done. A fix can also BREAK a capsule you did not check, so "
                f"re-run with --capsules all before believing you are finished. "
                if scope == "subset" and suite_size
                else ""
            )
            + (
                f" {len(gated)} requested whole-model capsule(s) were GATED and NOT EXECUTED; "
                "their rows are non-verdicts, not passes. "
                if gated
                else ""
            )
            + selection_note(
                combined=combined, required=sim == REQUIRED_SIM, sim=sim, adapters=adapters, barrier_tier=barrier_tier
            )
            + (
                "Candidate whole-program rows show only submitted-artifact numeric status, "
                "required-tier status and closed verification codes; the separate runner-owned "
                "graph's trace, console, timing and failure are withheld. An incomplete candidate "
                "is not a passed screen. Other rows retain their ordinary diagnostic detail."
                if has_candidate_rows
                else "You see EVERYTHING your dialect produced — command buffer, decoded trace + instruction "
                "counts, sim console, and your artifacts copied to ./selfcheck_out/. The diff stats "
                "(mismatch_count, magnitudes) are YOUR output measured against the operation's own "
                "definition, which you can reproduce from the declared inputs — there is no answer key; "
                "the reference output values are withheld so you debug from your own intent, as in real "
                "bring-up. 'done' = all public pass on verilator/VCS; cycles are not a criterion. "
                "(Movement ops legitimately have 0 matmuls.)"
            )
            + (
                f" {n_declined} capsule(s) were DECLINED by your backend -- see 'declined' on those "
                f"rows. A decline is a shape/op you never lowered, NOT wrong arithmetic."
                if n_declined
                else ""
            )
            + (
                f" ⚠ {len(_unreached)} mandatory tier(s) {_unreached} could NOT RUN although the grade's "
                f"own selection was requested. {nscreened} capsule(s) passed every tier that did run; "
                f"they are NOT passes and NOT certified. A selected tier that cannot run is an "
                f"operator/environment condition (see each row's failure detail), not a defect in your "
                f"backend; report it."
                if _unreached and combined
                else f" ⚠ {len(_unreached)} mandatory cert tier(s) {_unreached} were NOT REACHABLE with "
                f"--sim {sim}. {nscreened} capsule(s) passed EVERY tier this sim can run and are "
                f"reported as screened, not certified: they are counted in n_passed and NOT in "
                f"n_certified, and they are not failures — the tier could not run here, which is a "
                f"property of the sim you selected, not of your backend. Re-run without --sim (the "
                f"default certifies) to convert them; all_pass requires certification."
                if _unreached
                else ""
            )
        ),
    }
    if scope == "all" and selected is not None:
        # The complete grade has already emitted the public operator buffers and
        # lowered host artifacts. Read those exact source-bound outputs once for
        # offload accounting instead of running every compiler entrypoint again.
        # Missing artifacts remain named errors; model placement has its own
        # submitted-artifact audit and is never inferred from an operator buffer.
        from merlin.development.targetgen.conformance import offload_census as OC

        try:
            out["offload_census"] = OC.census_from_graded_artifacts(
                target=_tgt,
                capsules=selected,
                suite_runs=cb_root,
            )
        except Exception as exc:  # noqa: BLE001 -- the grade remains authoritative
            out["offload_census"] = {
                "schema": OC.SCHEMA,
                "execution": {"mode": "same_grade_artifacts"},
                "all_accounted": False,
                "status": "unavailable",
                "reason": f"graded-artifact census failed: {type(exc).__name__}: {str(exc)[:200]}",
            }
    out = performance_brief.attach(selection.report(out), rows, artifact_root=a.artifact_root or None)
    txt = json.dumps(compact_document(out, workspace=Path.cwd(), namespace="selfcheck"), indent=2)
    print(txt)
    if a.out:
        _atomic_json(Path(a.out), out)
    _log_telemetry(out, a.capsules)  # one operator-side dev-trajectory line (all verdicts, incl. degenerate)
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
