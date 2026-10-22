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

from merlin.targetgen import capsule_grade as CG  # build+run+compare
from merlin.targetgen import capsule_runner as CR  # tier adapters
from merlin_experiments.phase1.context import (
    InvocationContext,
    add_context_arguments,
    context_argv,
    readback_kwargs,
    resolve_context,
)
from merlin_experiments.phase1.feedback import qa as _qc
from merlin_experiments.phase1.feedback.candidate_observations import _shape_coverage, diagnostics_cli
from merlin_experiments.phase1.feedback.dispatch import REQUIRED_SIM
from merlin_experiments.phase1.feedback.selfcheck_results import (
    _candidate_selfcheck_row as _candidate_selfcheck_row,
)
from merlin_experiments.phase1.feedback.selfcheck_results import (
    _gated_without_result_rows as _gated_without_result_rows,
)
from merlin_experiments.phase1.feedback.selfcheck_results import (
    _log_telemetry as _log_telemetry,
)
from merlin_experiments.phase1.feedback.selfcheck_results import (
    _offload_census as _offload_census,
)
from merlin_experiments.phase1.feedback.selfcheck_results import (
    _operator_feedback_state as _operator_feedback_state,
)
from merlin_experiments.phase1.feedback.selfcheck_results import (
    _operator_structural_feedback as _operator_structural_feedback,
)
from merlin_experiments.phase1.feedback.selfcheck_results import (
    _public_candidate_emission_failure as _public_candidate_emission_failure,
)
from merlin_experiments.phase1.feedback.selfcheck_results import (
    _selfcheck_counts as _selfcheck_counts,
)
from merlin_experiments.phase1.feedback.selfcheck_results import (
    _successful_expected_refusal as _successful_expected_refusal,
)
from merlin_experiments.phase1.feedback.selfcheck_selection import (
    capsule_barrier,
    grade_l3_selection,
    selection_note,
    tier_engines,
)
from merlin_experiments.phase1.feedback.selfcheck_selectors import (
    _previously_failing as _previously_failing,
)
from merlin_experiments.phase1.feedback.selfcheck_selectors import select_capsules
from merlin_experiments.phase1.timing import barrier_measurements

from ..compiler_library import selected_child


# Restored during branch integration: the merge took this file whole from the other side, which
# dropped select_tiers and left the six tests that pin it failing on a missing attribute.
def select_tiers(full: dict, default: dict, requested: str) -> tuple[dict, str | None]:
    """Which tiers this self-check runs: the target's cheap LOOP ladder by default, or exactly the tiers
    ``--tiers`` names, validated against everything the endpoint can REACH.

    The two maps are a deliberate distinction and were being conflated. Resolving the DEFAULT from the
    loop ladder is what keeps a sweep cheap -- a cert tier is minutes per capsule against seconds for the
    functional one, and paying it on every capsule of every sweep was 80% of an agent round. Validating
    the REQUEST against that same shrunken map is a different question, and answering it with the loop
    map made the flag useless for the one job it exists to do: on this repo's SIMT target the loop ladder
    is {L2}, so ``--tiers L2,L3`` was refused as "unreachable" by an endpoint whose full map is
    {L2, L3}.

    The same conflation silently disabled automatic promotion. ``tier_promote.resolve_tiers`` derives its
    cert tier as ``oracle_adapters - qa_loop_adapters`` -- i.e. precisely a tier the loop map never
    contains -- and the broker forwards it as ``--tiers <cert>``. So every promoted job asked for a tier
    this validation could only refuse. Promotion was not merely untested in production; it could not fire
    even in principle.

    Returns ``(adapters, error)``. An unreachable tier is still NAMED, never silently dropped: a
    self-check that quietly grades fewer tiers than it was asked for reads as a pass.
    """
    want = {t.strip().upper() for t in requested.split(",") if t.strip()}
    if not want:
        return default, None
    missing = sorted(want - set(full))
    if missing:
        return {}, (f"--tiers names {missing}, which this endpoint does not reach; reachable: {sorted(full)}")
    return {t: ad for t, ad in full.items() if t in want}, None


def _strip_build_state(root: Path) -> None:
    """Delete ALL cmake/ninja build state under `root` so a graded copy configures from scratch in its own
    absolute path. `ignore_patterns("build")` only drops a dir literally named 'build'; a stale CMakeCache /
    ninja state in any other location pins the ORIGINAL temp source dir and makes cmake error
    ('source does not match cache used to generate cache'). Stripping these guarantees a clean, relocatable
    build (the abc9 baseline 'L3 0/20 build' bug)."""
    for p in (
        list(root.rglob("CMakeCache.txt"))
        + list(root.rglob("CMakeFiles"))
        + list(root.rglob("build.ninja"))
        + list(root.rglob(".ninja_deps"))
        + list(root.rglob(".ninja_log"))
        + list(root.rglob("cmake_install.cmake"))
    ):
        try:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        except Exception:
            pass


# operator-side modules (this tool is NOT part of the graded submission; the submission must stay
# integrity-clean and never import merlin — but this harness may, exactly like a vendor's test rig)


# the PUBLIC capsule set (with goldens — operator-side; output is redacted before the agent sees it),
# DERIVED per-target from the descriptor's capsule_corpus (e.g. fp8/L3, i8/L2) — no committed set
# leaking another target's capsules. Falls back to the legacy committed set if the descriptor can't be resolved.
def _public_capsules(context: InvocationContext) -> Path:
    try:
        from merlin.targetgen.target_experiment import load_target_experiment
        from merlin_experiments.corpus.admission import public_capsules_for

        return public_capsules_for(load_target_experiment(context.descriptor))
    except Exception as error:  # noqa: BLE001 — keep the self-check usable without a resolvable descriptor
        if context.harness is not None:
            # Said out loud: the legacy set is older than the corpus, so a silent fallback reads as the
            # package failing a capsule that is merely missing from it ("unknown capsule(s)").
            print(
                f"[selfcheck] public capsule set not derivable from {context.descriptor} "
                f"({type(error).__name__}: {error}); falling back to the legacy committed set "
                f"{context.harness / 'full_public_capsules'}",
                file=sys.stderr,
            )
            return context.harness / "full_public_capsules"
        raise


def worker_command(
    context: InvocationContext,
    capsules_root: Path,
    contract: Path | None = None,
    *,
    rtl_facts: Path | None = None,
    compiler_bundle_manifest: Path | None = None,
) -> list[str]:
    """Fixed trusted worker; candidate request fields never select executable or resources."""
    command = [sys.executable, "-m", "merlin_experiments.phase1.feedback.selfcheck"]
    command += context_argv(context) + ["--capsules-root", str(capsules_root)]
    if contract is not None:
        command += ["--contract", str(contract)]
    if rtl_facts is not None:
        command += ["--rtl-facts", str(rtl_facts)]
    if compiler_bundle_manifest is not None:
        command += ["--compiler-bundle-manifest", str(compiler_bundle_manifest)]
    from ..compiler_library import child_arguments

    return [*command, *child_arguments()]


def _atomic_json(path: Path, doc: dict) -> None:
    """Publish an agent-visible diagnostic without exposing a half-written JSON document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _progress_snapshot(cb_root: Path, *, expected: int, started_ns: int) -> dict:
    """A deliberately small, golden-free view of durable per-capsule results.

    ``run_capsule`` writes ``capsule_result.json`` as soon as one worker finishes. A suite may still
    spend another half hour waiting for a few cycle-cap cases, so hiding those durable results until
    ``grade`` returns makes the compiler loop needlessly blind. This snapshot exposes only public
    capsule names and their already-agent-visible status; numeric reference values never enter it.
    """
    rows: list[dict] = []
    for result_path in sorted(Path(cb_root).glob("*/capsule_result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # writer still publishing; the next snapshot will see it
        if not isinstance(result, dict):
            continue
        row = {
            "capsule": str(result.get("capsule") or result_path.parent.name),
            "status": str(result.get("status") or "unknown"),
        }
        # Raw whole-model graph tiers do not qualify the submitted compiler's
        # artifact. Model progress keeps its legacy status-only projection.
        model = result.get("kind") == "model" or any(
            key in result
            for key in ("model_execution_check", "candidate_native_model_check", "candidate_native_execution")
        )
        recorded = result.get("tiers")
        if not model and isinstance(recorded, dict):
            tiers = {
                tier: record["status"]
                for tier, record in recorded.items()
                if tier in {"L0", "L1", "L2", "L3"}
                and isinstance(record, dict)
                and isinstance(record.get("status"), str)
                and record.get("status") in {"pass", "fail", "unverified", "unavailable", "incomplete", "skipped"}
            }
            if tiers:
                row["tiers"] = tiers
        rows.append(row)
    counts: dict[str, int] = {}
    for row in rows:
        status = row["status"]
        counts[status] = counts.get(status, 0) + 1
    finished = len(rows)
    return {
        "status": "running",
        "started_at_unix_ns": int(started_ns),
        "updated_at_unix_ns": time.time_ns(),
        "n_expected": int(expected),
        "n_finished": finished,
        "n_remaining": max(0, int(expected) - finished),
        "counts": counts,
        "per_capsule": rows,
        "note": (
            "Incremental self-check progress from completed capsule records. This is not a final "
            "verdict and contains no golden/reference values. Reported tiers are observed public "
            "operator screens, not full-case certification; raw model tiers are omitted."
        ),
    }


def _progress_publisher(
    path: Path, cb_root: Path, *, expected: int, started_ns: int, stop: threading.Event, poll_s: float = 0.5
) -> None:
    """Continuously publish changing progress until the grade's final roll-up is available."""
    last_rows = None
    while not stop.is_set():
        doc = _progress_snapshot(cb_root, expected=expected, started_ns=started_ns)
        if doc["per_capsule"] != last_rows:
            _atomic_json(path, doc)
            last_rows = doc["per_capsule"]
        stop.wait(poll_s)
    doc = _progress_snapshot(cb_root, expected=expected, started_ns=started_ns)
    doc["status"] = "grading_complete"
    _atomic_json(path, doc)


def _suite_size(capsules_root: Path) -> int:
    """How many public capsules exist, so a partial check can say what it did NOT check.

    Counted the way the grader DISCOVERS them (:func:`merlin.targetgen.capsule_common.discover_capsules`:
    every ``capsule.yaml`` below the root, one per directory). A flat listing undercounts a root whose
    capsules sit in category subdirectories, and the shortfall would read as capsules the check skipped.
    """
    try:
        return len({cy.parent for cy in capsules_root.rglob("capsule.yaml")})
    except OSError:
        return 0


def _target_sim_via(context: InvocationContext) -> tuple[str, str | None]:
    """The (target, sim_via) being self-checked, from the descriptor (so CG.grade builds ITS RunnerConfig
    — e.g. an external_backend target — not a hardcoded default, and the oracle tiers resolve from the
    target's contract). Falls back to the active target (from _common) with an unknown sim_via if the
    descriptor can't be parsed."""

    try:
        from merlin.targetgen.target_experiment import load_target_experiment

        te = load_target_experiment(context.descriptor)
        return te.target, te.sim_via
    except Exception:  # noqa: BLE001
        return context.target, None


def _target(context: InvocationContext) -> str:
    return _target_sim_via(context)[0]


# gsim shares verilator's rung: both are elaborated-RTL, cycle-accurate engines over the SAME
# design (GemminiGsimMonitorsConfig is GemminiRocketConfig plus a harness-level clock
# instantiator, so the DUT is identical). It is the tier that differs by ENGINE, never by name.
SIM_TIER = {"spike": "L2", "verilator": "L3", "gsim": "L3", "vcs": "L4"}


def _required_rtl_engine() -> str | None:
    """The exact RTL engine imposed by the enclosing experiment, if any."""
    return os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip() or None


#: Every ``--sim`` value the self-check accepts. ``REQUIRED_SIM`` is the default and is not an engine.
SIM_CHOICES = (REQUIRED_SIM, *SIM_TIER)


def _sim_policy_error(sim: str, sim_via: str | None) -> str | None:
    """Refuse an RTL request that differs from the experiment-wide engine pin.

    Spike remains available as a correctness-only screen.  Every elaborated-RTL request must name the
    required engine exactly; inheriting the pin while directly constructing a Verilator adapter used to
    let a self-check run the wrong simulator even though formal grading was GSIM-only.
    """
    if sim_via != "chipyard" or sim == "spike":
        return None
    required = _required_rtl_engine()
    if required is None:
        return None
    if required not in SIM_TIER or required == "spike":
        return f"MERLIN_REQUIRED_RTL_ENGINE={required!r} is not a registered elaborated-RTL self-check engine"
    if sim != required:
        return (
            f"--sim {sim!r} conflicts with MERLIN_REQUIRED_RTL_ENGINE={required!r}; "
            f"use --sim 'spike' for a correctness-only screen or --sim {required!r} for RTL"
        )
    return None


def _streaming_completion(*, context, capsules_root, contract, source, runs, adapters):
    """Use only the trusted broker's frozen source, never public progress JSON."""
    from ..brokers.selfcheck import _submission_digest
    from . import loop_grading, promotion

    ws, root, digest = (
        os.environ.get(key)
        for key in (
            "MERLIN_SELFCHECK_PROMOTION_WS",
            "MERLIN_SELFCHECK_PROMOTION_ROOT",
            "MERLIN_SELFCHECK_PROMOTION_SHA256",
        )
    )
    if not any((ws, root, digest)):
        return None
    if not all((ws, root, digest)):
        raise ValueError("streaming selfcheck promotion has incomplete trusted context")
    if _submission_digest(Path(source)) != digest:
        raise ValueError("streaming selfcheck candidate differs from frozen broker identity")
    loop, cert, cover = promotion.resolve_tiers(Path(ws), context=context, capsules_root=Path(root))
    if not loop or not cert or loop not in adapters or cert in adapters:
        return None
    inputs = loop_grading.GradingInputs(
        context=context,
        arm="selfcheck",
        public_root=Path(capsules_root),
        policy_roots=(Path(root),),
        contract=contract,
        promotion_root=Path(root),
    )
    completed = loop_grading._completed_loop_promoter(
        Path(ws), Path(source).parent, Path(runs), loop, cert, cover, inputs=inputs
    )

    def callback(row):
        if _submission_digest(Path(source)) != digest:
            raise ValueError("streaming selfcheck frozen candidate changed during grading")
        completed(row)

    return callback


def _adapters(sim: str, target: str, sim_via: str | None, *, readback_policy=None) -> tuple[dict, str]:
    """Resolve the self-check oracle tiers from the TARGET's contract (target-agnostic, mirrors the driver
    grade). A chipyard target exposes the spike/verilator/vcs ladder selectable via --sim; any
    other target grades on its OWN contract-derived RTL tier (external_backend -> the program oracle;
    an arc target -> the RTL-derived arc cosim), where --sim is not applicable. Routing a self-hosted
    ISA here through the hardcoded spike/verilator adapters ran the RoCC/RVV lowering and crashed (AW4).

    ``REQUIRED_SIM`` returns the official grade's own adapter set (``qa.run`` grades with
    ``qa_checkpoint_adapters``): the same tiers, engine selection record and readback policy. One request
    then runs each capsule's whole mandatory ladder -- the functional tier, the elaborated-RTL tier and a
    provided-host tier alike -- so it reaches the grade's verdict on the same bytes instead of a screen
    that reports the other tiers ``unavailable``. Sharing the adapters also shares the exact-byte
    simulator certificate instrument, which binds the adapter's selection record."""
    kwargs = {} if readback_policy is None else {"readback_policy": readback_policy}
    if sim == REQUIRED_SIM:
        # The selection honours MERLIN_REQUIRED_RTL_ENGINE itself (`chipyard_l3_selection`), so there
        # is no named engine here for the pin to refuse.
        return dict(CR.qa_checkpoint_adapters(target, sim_via, **kwargs)), sim
    policy_error = _sim_policy_error(sim, sim_via)
    if policy_error:
        raise ValueError(policy_error)
    if sim == "vcs" and readback_policy is not None:
        raise ValueError("VCS self-check does not support the selected invocation readback policy")
    if sim_via == "chipyard":
        ad = {"L2": CR.simulator_adapter("spike", target, **kwargs)}
        if sim in ("verilator", "gsim"):
            # Build L3 from the engine the caller NAMED. This used to hardcode "verilator", which made
            # --sim gsim silently certify on verilator -- a result attributed to the wrong engine. The
            # adapter factory is engine-generic (it forwards simulator=sim and gates on
            # backend.available(sim)), so an absent engine fails closed here rather than substituting.
            selection = grade_l3_selection(target, sim)
            ad["L3"] = CR.simulator_adapter(sim, target, **kwargs, **({"selection": selection} if selection else {}))
        if sim == "vcs":
            # In the unpinned legacy ladder VCS is an L4 addition above Verilator L3.  An experiment-wide
            # VCS pin, however, means *no other RTL engine may run*, so do not build that Verilator rung.
            # If VCS is unavailable the pinned run must refuse instead of silently substituting.
            required = _required_rtl_engine()
            if required is None:
                ad["L3"] = CR.simulator_adapter("verilator", target)
            try:
                from merlin.targetgen import vcs_adapter as VA  # optional, config-gated

                ad["L4"] = VA.adapter()
            except Exception as exc:
                if required is not None:
                    raise ValueError(
                        f"required RTL engine {required!r} is unavailable: {type(exc).__name__}: {exc}"
                    ) from exc
                print("  (vcs adapter unavailable in this environment — falling back to verilator L3)", file=sys.stderr)
                sim = "verilator"
        return ad, sim
    # non-chipyard target: its contract-resolved tiers (arc / program oracle); --sim does not apply.
    return CR.oracle_adapters(target, sim_via, **kwargs), sim


def _model_layers(
    sub: Path,
    out_path: str,
    *,
    timeout: int,
    workers: int,
    capsules: str = "all",
    context: InvocationContext,
    contract: Path | None = None,
    artifact_root: Path | None = None,
) -> int:
    """The agent-facing model-layer report (see ``--model-layers``).

    Grades the package, at the functional tier only, against capsules generated from the layers real
    models form on this target: the model's own extents and the stage combinations the compiler will
    try to fuse. They are not part of the graded suite. Output goes through the same redaction as the
    ordinary self-check, so no golden value reaches the agent.

    ``capsules`` narrows it to named layers, as the ordinary self-check's ``--capsules`` does. These
    are the corpus's biggest programs, so grading all ten to learn about one is the difference
    between a check an agent runs while it works and one it waits on: measured on the first campaign
    to use this probe, a full pass held the single self-check slot for 76 minutes and the agent's
    request for exactly one layer was widened back to all ten without a word. A name that matches
    nothing is refused, because a typo that silently grades everything is the same defect again.
    """

    from merlin.targetgen import tier_policy
    from merlin.targetgen.target_experiment import load_target_experiment

    experiment = load_target_experiment(context.descriptor)
    roots = experiment.model_layer_roots()
    if not roots:
        print(json.dumps({"error": "this target ships no model-layer capsules", "n_capsules": 0}))
        return 2
    root = Path(roots[0])
    wanted = [name.strip() for name in str(capsules or "all").split(",") if name.strip()]
    if wanted and [w.lower() for w in wanted] != ["all"]:
        held = {d.name for d in root.iterdir() if (d / "capsule.yaml").is_file()}
        unknown = sorted(set(wanted) - held)
        if unknown:
            print(json.dumps({"error": f"no model-layer capsule named {unknown}", "known": sorted(held)}))
            return 2
        # A corpus root IS its set of capsules, so a subset is a root of its own. Each capsule is a
        # real directory of symlinked FILES: the bytes graded are the tracked ones, nothing is
        # copied per request, and discovery (`rglob`) finds them, which it does not do through a
        # symlinked directory.
        narrowed = (artifact_root or Path("selfcheck_out")) / "model_layers" / "_subset"
        if narrowed.exists():
            shutil.rmtree(narrowed)
        for name in wanted:
            (narrowed / name).mkdir(parents=True)
            for member in sorted((root / name).iterdir()):
                if member.is_file():
                    (narrowed / name / member.name).symlink_to(member.resolve())
        root = narrowed
    adapters = CR.oracle_adapters(context.target, experiment.sim_via, **readback_kwargs(context)) or {}
    # The functional tier only: an elaborated-RTL simulator spends about twenty minutes on ONE such
    # layer. Every row names the tiers withheld, so a pass here reads as the screen it is.
    ladder = tier_policy.tier_depth_order(adapters)
    screen = {tier: adapters[tier] for tier in ladder[:1]}
    runs_root = (artifact_root or Path("selfcheck_out")) / "model_layers"
    try:
        score = CG.grade(
            str(sub),
            capsules_root=str(root),
            runs_root=str(runs_root),
            labels={"public", "dev"},
            contract=str(contract if contract is not None else context.repo / "merlin/contract"),
            oracle_adapters=screen,
            timeout=timeout,
            max_workers=workers,
            target=context.target,
        )
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": f"model-layer grade failed: {type(e).__name__}: {str(e)[:300]}"}))
        return 2
    # Built from each capsule's own result, and only from fields that carry no golden value: the
    # statuses, the failure's class, its detail through the redactor every agent-facing channel
    # uses, and how MANY elements disagreed. Never which value was expected.
    rows = []
    # ONLY WHAT THIS RUN GRADED. The runs directory is reused across invocations, so a glob over it
    # returns every capsule any earlier invocation left behind: asking for one layer reported four,
    # three of them from a previous request against different bytes. Stale feedback that reads as
    # fresh is the failure this harness exists to prevent, so the rows are restricted to the corpus
    # root this invocation actually graded.
    graded_now = {d.name for d in root.iterdir() if (d / "capsule.yaml").is_file()}
    closed_rows = _qc._per_capsule_from_results(runs_root)
    for result_path in sorted((runs_root / "runs" / CR.suite_for(context.target)).glob("*/capsule_result.json")):
        if result_path.parent.name not in graded_now:
            continue
        try:
            result = json.loads(result_path.read_text())
        except Exception:  # noqa: BLE001
            continue
        name = result.get("capsule", result_path.parent.name)
        if _qc._candidate_native_feedback(result) is not None:
            candidate_row, certified = _candidate_selfcheck_row(
                result, closed_rows.get(name), name=name, barrier_tier=ladder[0] if ladder else "L3"
            )
            summary = candidate_row["candidate_native_verification"]
            failure = candidate_row.get("failure") or {}
            rows.append(
                {
                    "capsule": name,
                    "status": result.get("status"),
                    "tiers": summary.get("tiers", {}),
                    "failure_plane": failure.get("plane"),
                    "failure_category": failure.get("category"),
                    "failure_detail": failure.get("detail"),
                    "mismatch_count": (candidate_row.get("numeric") or {}).get("mismatch_count"),
                    "candidate_native_verification": summary,
                    "candidate_verified": certified,
                }
            )
            continue
        failure, numeric = result.get("failure") or {}, result.get("numeric") or {}
        rows.append(
            {
                "capsule": result.get("capsule", result_path.parent.name),
                "status": result.get("status"),
                "tiers": {t: (v or {}).get("status") for t, v in (result.get("tiers") or {}).items()},
                "failure_plane": failure.get("plane"),
                "failure_category": failure.get("category"),
                "failure_detail": _qc._redact_detail(failure.get("detail")),
                "mismatch_count": numeric.get("mismatch_count"),
            }
        )
    # A TIER WE WITHHELD IS NOT A FAILURE OF THE BACKEND. This counted every tier in the row, and the
    # deeper adapters are deliberately not supplied here, so a layer that passed every tier that RAN
    # still read as not passed: `all_pass` could never be true and the probe could never say a model
    # layer works -- while the errata tells the agent a round is not converged until one does. A row
    # passes when nothing that ran failed and something did run.
    withheld = set(ladder[1:])

    def _passed_here(row: dict) -> bool:
        if "candidate_native_verification" in row:
            return row["candidate_verified"]
        ran = {tier: status for tier, status in row["tiers"].items() if tier not in withheld}
        return (
            bool(ran)
            and any(v == "pass" for v in ran.values())
            and not any(v not in ("pass", "skipped") for v in ran.values())
        )

    screened = [row for row in rows if _passed_here(row)]
    report = {
        "n_capsules": len(rows),
        "n_passed_functional_tier": len(screened),
        "all_pass": bool(rows) and len(screened) == len(rows),
        "per_capsule": rows,
        "grade_failure": (
            (score or {}).get("failure")
            if isinstance(score, dict) and not any("candidate_native_verification" in row for row in rows)
            else None
        ),
    }
    report["tiers_withheld"] = ladder[1:]
    report["graded"] = sorted(row["capsule"] for row in rows)
    report["passed"] = sorted(row["capsule"] for row in rows if _passed_here(row))
    report["note"] = (
        "These capsules are the layers real models form on this target, at the MODEL'S extents: "
        "far past the few-tile shapes of the graded suite, with the stage combinations the compiler "
        "fuses (bias, scale, activation, pool on one contraction or convolution) and, beside each, "
        "the same contraction with a bare accumulator readout. A backend that passes the suite and "
        "fails here handles toy shapes only: look for a weight larger than the operand store, a "
        "partial last tile, a scratchpad address past its bank. They are graded at the functional "
        "tier only and do not count toward your score; they decide whether a model can run."
    )
    if any("candidate_native_verification" in row for row in rows):
        report["note"] += (
            " Candidate whole-program rows use only submitted-artifact verification; "
            "the runner-owned functional layer screen cannot pass them."
        )
    txt = json.dumps(report, indent=2)
    print(txt)
    if out_path:
        Path(out_path).write_text(txt)
    return 0 if report.get("all_pass") else 1


def _request_artifact_root(value: str, *, progress: str, output: str) -> Path:
    """A broker output root belongs only to its exact original request channel."""
    workspace = Path.cwd().resolve()
    record = Path(progress)
    if record.parent.resolve() != workspace / ".qa_channel" or not (
        record.name.startswith("progress_") and record.name.endswith(".json")
    ):
        raise ValueError("artifact root requires the original request progress record")
    rid = record.name[len("progress_") : -len(".json")]
    if not rid or rid in {".", ".."}:
        raise ValueError("artifact root requires a valid request identity")
    staged = Path(output)
    if staged.parent.resolve() != record.parent.resolve() or not (
        staged.name.startswith(f".resp_{rid}.") and staged.name.endswith(".tmp")
    ):
        raise ValueError("artifact root differs from the original response identity")
    expected = workspace / "selfcheck_out" / "requests" / rid
    if Path(value).absolute() != expected or any(
        path.is_symlink() for path in (workspace / "selfcheck_out", workspace / "selfcheck_out" / "requests", expected)
    ):
        raise ValueError("artifact root must be the original per-request output directory")
    return expected


from ..joint_control import selfcheck_policy


@selected_child
@selfcheck_policy
def main(argv=None, *, context=None, capsules_root: Path | None = None, contract: Path | None = None):
    ap = argparse.ArgumentParser(description="Agent self-check runner (redacted; spike/verilator/vcs).")
    add_context_arguments(ap)
    ap.add_argument("--suite", choices=("core", "scalability"), default="core")
    ap.add_argument("--capsules-root", type=Path, help="explicit public corpus, never rediscovered when supplied")
    ap.add_argument("--contract", type=Path, help="explicit grading contract root")
    ap.add_argument("--rtl-facts", type=Path, help="Selected facts in the verified frozen input snapshot")
    ap.add_argument("--submission", default="submission", help="path to your package (with manifest.yaml)")
    # DEFAULTS TO THE GRADE'S OWN SELECTION, not to one engine. The capsules declare their mandatory
    # tiers (a functional screen, an elaborated-RTL cert tier, a provided-host tier), and the ladder
    # runs cheapest-measured-first with fail-fast, so the screen still refutes a broken submission at
    # screen cost -- what changes is that a capsule which PASSES the screen goes on to certify in the
    # same request. A single engine name ("spike") is a legitimate fast screen, but it CANNOT certify:
    # each mandatory tier it does not run reports unavailable and the capsule is not a pass.
    # Measured on one run: 166 of 217 self-check capsule results were incomplete because each request
    # ran a single engine, and pass/fail was learned only from the background grade.
    ap.add_argument(
        "--sim",
        choices=list(SIM_CHOICES),
        default=None,
        help=f"default {REQUIRED_SIM!r}: every tier the official grade runs; one engine name is a narrower screen",
    )
    ap.add_argument(
        "--capsules",
        default="all",
        help="'all', 'failing' (only what did not pass last time — fast iteration), 'changed' (build and "
        "screen every public capsule, reuse exact-byte RTL certificates, simulate only those whose bytes are "
        "neither certified nor known to fail), or comma-separated capsule names",
    )
    # 0 = derive from this host (see capsule_grade.default_grade_workers): the per-capsule builds and
    # sims are independent, and a literal 8 left most of a 48-core machine idle through every sweep.
    ap.add_argument(
        "--workers", type=int, default=0, help="parallel per-capsule workers; 0 (default) sizes them to this host"
    )
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="", help="optional: also write the redacted JSON here")
    ap.add_argument("--progress-out", default="", help=argparse.SUPPRESS)
    ap.add_argument("--artifact-root", default="", help=argparse.SUPPRESS)
    ap.add_argument("--model-gate-voucher", action="append", default=[], help=argparse.SUPPRESS)
    ap.add_argument(
        "--caller-layout",
        default="",
        help="INSTEAD of grading, inspect a submission-relative command buffer's selected physical pointer layout",
    )
    # THE FLAG select_tiers WAS WRITTEN FOR. It existed as a function with six tests pinning it and no
    # way to reach it: argparse never accepted --tiers, so the broker's promotion jobs -- which always
    # forward `--tiers <cert_tier>` -- died on "unrecognized arguments: --tiers L3" before doing any
    # work, and the broker reported that as "no verdict produced". Measured on
    # merlincirct_arm4_func_20260901_v4: 19 promotion jobs, every one killed by argparse.
    ap.add_argument(
        "--tiers",
        default="",
        help="comma-separated tiers to grade (default: this target's cheap loop ladder), e.g. L3 for a cert run",
    )
    ap.add_argument(
        "--shape-coverage",
        action="store_true",
        help="INSTEAD of the capsule suite, probe whether your backend LOWERS the same "
        "contraction at one tile and at two tiles in each of M, K and N. Costs no "
        "simulator and no oracle (it only runs your emit path), so it is cheap to run "
        "often. The public capsules are a fixed set of shapes; passing all of them "
        "says nothing about whether you generalize past them, and a backend that "
        "hardcodes those shapes scores 100%% here and fails the held-out grade.",
    )
    ap.add_argument(
        "--offload-census",
        action="store_true",
        help="INSTEAD of the capsule suite, report what your package puts on the "
        "accelerator: per public program, the work that reached a unit, what was "
        "placed on the host, what was declined, and what was emitted with no "
        "statement at all. Runs only your emit path, so it is cheap. A passing "
        "program whose work all ran on the host is not what this target is for.",
    )
    ap.add_argument(
        "--model-layers",
        action="store_true",
        help="INSTEAD of the capsule suite, grade your package at the functional tier against "
        "capsules generated from the layers REAL MODELS form on this target, at the models' own "
        "extents. Not part of your score. A backend that passes the suite and fails these runs "
        "toy shapes only. Honors --capsules: name one layer to check one layer.",
    )
    ap.add_argument(
        "--candidate-diagnostics",
        action="store_true",
        help="INSTEAD of grading, inspect own public source/emission/host work; no simulator or baseline",
    )
    ap.add_argument(
        "--compile-only",
        action="store_true",
        help="compile original public scale/frontier tensor sources and link a no-data probe; no execution",
    )
    ap.add_argument(
        "--scale-probe",
        action="store_true",
        help="with --compile-only, inspect public scale/tail LLVM emission before object compilation",
    )
    ap.add_argument("--compiler-bundle-manifest", type=Path, help=argparse.SUPPRESS)
    ap.add_argument(
        "--dynamic-census", action="store_true", help="append bounded own-ELF activity after functional screen"
    )
    a = ap.parse_args(argv)
    if a.artifact_root:
        try:
            a.artifact_root = _request_artifact_root(a.artifact_root, progress=a.progress_out, output=a.out)
        except ValueError as error:
            ap.error(str(error))
    if a.scale_probe and not a.compile_only:
        ap.error("--scale-probe requires --compile-only")
    if a.rtl_facts is not None:
        from ..frozen_facts import select

        select(Path.cwd(), a.rtl_facts)
    context = resolve_context(a, ap, context)
    if a.suite == "scalability":
        from .public_scalability import cli

        return cli(a, context=context)
    if a.compile_only:
        from .compiler_profile_runtime import selfcheck_compile_only

        return selfcheck_compile_only(a, context=context)
    if a.caller_layout:
        try:
            if a.rtl_facts is None:
                raise ValueError("selected frozen RTL facts are required")
            from .caller_layout import inspect_caller_layout

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
        from .compiler_profile_runtime import selfcheck_diagnostics

        with selfcheck_diagnostics(a.compiler_bundle_manifest, context=context, arguments=a):
            if a.candidate_diagnostics:
                return diagnostics_cli(a, context=context, capsules_root=capsules_root, contract=contract)

            if a.shape_coverage:
                return _shape_coverage(Path(a.submission), a.out, context=context, contract=contract)
            if a.model_layers:
                return _model_layers(
                    Path(a.submission),
                    a.out,
                    timeout=a.timeout,
                    workers=a.workers,
                    capsules=a.capsules,
                    context=context,
                    contract=contract,
                    artifact_root=a.artifact_root or None,
                )
            if a.offload_census:
                return _offload_census(
                    Path(a.submission),
                    a.out,
                    context=context,
                    capsules_root=capsules_root,
                    contract=contract,
                    timeout=a.timeout,
                    workers=a.workers,
                )

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
    # repo path (e.g. merlin/contract/schemas/...) PASSES self-check but the standalone grade CRASHES — the
    # self-check would lie. EXCLUDE build/ exactly like the driver grade (a copied CMakeCache has absolute
    # paths to the original location and would break) -> a from-scratch build, same as the grade (fast: the
    # LLVM/MLIR is prebuilt, the OOT project is small).
    import tempfile as _tf2

    _isodir = Path(_tf2.mkdtemp(prefix="selfcheck_iso_"))
    shutil.copytree(sub, _isodir / "submission", ignore=shutil.ignore_patterns("build", "__pycache__", ".git"))
    sub = _isodir / "submission"
    _strip_build_state(sub)  # every grade builds from scratch in its OWN path — see helper

    _tgt, _sim_via = _target_sim_via(context)
    try:
        adapters, sim = _adapters(a.sim, _tgt, _sim_via, **readback_kwargs(context))
    except ValueError as exc:
        out = {"error": str(exc), "all_pass": False, "sim": a.sim, "required_rtl_engine": _required_rtl_engine()}
        txt = json.dumps(out, indent=2)
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
        _full = dict(CR.oracle_adapters(_tgt, _sim_via, **readback_kwargs(context)))
        _full.update(adapters)  # a --sim-selected tier is reachable by definition
        adapters, _tier_err = select_tiers(_full, adapters, a.tiers)
        if _tier_err:
            # Fail CLOSED and leave the reason on disk. Exiting without writing --out is what produced
            # the broker's contentless "no verdict produced".
            out = {"error": _tier_err, "all_pass": False, "sim": sim, "tiers_requested": a.tiers}
            txt = json.dumps(out, indent=2)
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

    from .raw_selfcheck_evidence import create_runs_root

    runs_root = create_runs_root()

    # Subset filtering that ACTUALLY limits the grade: CG.grade runs EVERY capsule under capsules_root, so
    # the `want` output-filter below alone would still grade all 20 (a "1-capsule verilator" check = ~1hr).
    # Build a temp capsules_root symlinking only the requested capsule dirs -> grade runs just the subset.
    caps_root = capsules_root
    if want:
        caps_root = Path(tempfile.mkdtemp(prefix="selfcheck_caps_"))
        missing = [n for n in sorted(want) if not (capsules_root / n / "capsule.yaml").exists()]
        if missing:
            print(json.dumps({"error": f"unknown capsule(s): {missing}"}))
            return 2
        for n in sorted(want):  # copy (not symlink): rglob won't recurse symlinked dirs
            shutil.copytree(capsules_root / n, caps_root / n)

    # build + run + compare (parallel); CG.grade handles the agent's 4 entrypoints + the tier ladder.
    # Publish completed rows independently of the final verdict: a few cycle-cap cases must not hide all
    # fast feedback. The final response remains a separate atomic document owned by the broker.
    cb_root = runs_root / "runs" / CR.suite_for(_tgt)
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
            from .model_gate_vouchers import verify
            from .promotion import _submission_digest as promotion_digest

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
        from .compiler_profile_runtime import selfcheck_execution

        with selfcheck_execution(
            a.compiler_bundle_manifest, context=context, submission=sub, work_root=_isodir / "profile-package-work"
        ):
            _score = CG.grade(
                str(sub),
                capsules_root=str(caps_root),
                runs_root=str(runs_root),
                labels={"public", "dev"},
                contract=str(contract if contract is not None else context.repo / "merlin/contract"),
                oracle_adapters=adapters,
                timeout=a.timeout,
                max_workers=a.workers,
                target=_tgt,
                model_gate_continuations=tuple(model_continuations),
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
            "note": "Your package did NOT build/scan, so ZERO capsules ran — this is NOT a stubbed "
            "grader. Fix the build first (see 'detail' for the cmake/compiler stderr). NOTE: "
            "the grade builds an ISOLATED COPY of your package WITHOUT the build/ tree, so your "
            "manifest 'build' must support a CLEAN configure (provide a 'configure' step, e.g. "
            "cmake -S <src> -B <build>, not only 'cmake --build'); build commands run from the "
            "package root, so relative paths and $MLIR_DIR/$LLVM_DIR/$CM env vars are honored.",
        }
        txt = json.dumps(out, indent=2)
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
    closed_rows = _qc._per_capsule_from_results(runs_root)
    try:
        selected = CR.discover_capsules(
            caps_root,
            labels={"public", "dev"},
            contract=contract if contract is not None else context.repo / "merlin/contract",
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
        _summary = _qc._candidate_native_feedback(_d)
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
    from . import joint_performance

    performance_deadline = joint_performance.deadline(a.timeout)
    for cr in _results:
        try:
            d = json.loads(cr.read_text())
        except Exception:
            continue
        name = d.get("capsule", cr.parent.name)
        if want and name not in want:
            continue
        if _qc._candidate_native_feedback(d) is not None:
            row, certified = _candidate_selfcheck_row(d, closed_rows.get(name), name=name, barrier_tier=barrier_tier)
            if a.dynamic_census:
                row["dynamic_census"] = {"status": "UNKNOWN", "reason": "distinct own model ELF owner required"}
            rows.append(row)
            npass += int(row["pass"])
            ncert += int(certified)
            nscreened += int(row.get("screened_only", False))
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
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)
                own[art] = str(dst)
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
        from . import joint_physical_correctness, tier_boundary_join
        if name in joint_physical_correctness.roster() and bar_used in {'L2', 'L3'}:
            row['tier_boundary'] = tier_boundary_join.optional_observation(cr, tier=bar_used)
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
            from .candidate_census import attach_candidate_census

            attach_candidate_census(
                row,
                gen,
                result=d,
                context=context,
                compiler=sub,
                remaining_s=max(0, census_deadline - time.monotonic()),
            )
        joint_performance.attach(
            row, gen, result=d, context=context, compiler=sub, capsule=_original,
            remaining_s=max(0, performance_deadline - time.monotonic()),
        )
        rows.append(row)
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
        txt = json.dumps(out, indent=2)
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
            txt = json.dumps(out, indent=2)
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
        txt = json.dumps(out, indent=2)
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
        from merlin.targetgen import offload_census as OC

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
    out = selection.report(out)
    txt = json.dumps(out, indent=2)
    print(txt)
    if a.out:
        _atomic_json(Path(a.out), out)
    _log_telemetry(out, a.capsules)  # one operator-side dev-trajectory line (all verdicts, incl. degenerate)
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
