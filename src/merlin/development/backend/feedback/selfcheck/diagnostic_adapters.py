"""Public diagnostic adapters and simulator-policy resolution for selfcheck."""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from pathlib import Path

from merlin.development.backend.context import (
    InvocationContext,
    context_argv,
    readback_kwargs,
)
from merlin.development.backend.feedback.certification.promotion_admission import REQUIRED_SIM
from merlin.development.backend.feedback.grading import capsule_feedback as _merlin_capsule_feedback_3
from merlin.development.backend.feedback.grading import qa as _qc
from merlin.development.backend.feedback.selfcheck.results import _candidate_selfcheck_row, compact_document
from merlin.development.backend.feedback.selfcheck.selection import (
    grade_l3_selection,
)
from merlin.development.targetgen.grading import submission as _merlin_submission_2
from merlin.evaluation.capsule import eligibility as capsule_eligibility
from merlin.evaluation.capsule.evidence import certificates as capsule_adapters
from merlin.evaluation.capsule.evidence import certificates as capsule_availability


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


def _public_capsules(context: InvocationContext) -> Path:
    try:
        from merlin.experiments.corpus.admission import public_capsules_for
        from merlin.target.registry.descriptor_loading import load_target_experiment

        return public_capsules_for(load_target_experiment(context.descriptor))
    except Exception as error:  # noqa: BLE001 — retain an explicitly selected historical source owner
        if context.harness is not None and any((context.harness / "full_public_capsules").glob("*/capsule.yaml")):
            # Said out loud: the legacy set is older than the corpus, so a silent fallback reads as the
            # package failing a capsule that is merely missing from it ("unknown capsule(s)").
            print(
                f"[selfcheck] public capsule set not derivable from {context.descriptor} "
                f"({type(error).__name__}: {error}); using the selected historical harness corpus "
                f"{context.harness / 'full_public_capsules'}",
                file=sys.stderr,
            )
            return context.harness / "full_public_capsules"
        raise ValueError(
            "selfcheck requires an explicitly selected reviewed corpus release; "
            "no historical harness corpus was supplied"
        ) from error


def worker_command(
    context: InvocationContext,
    capsules_root: Path,
    contract: Path | None = None,
    *,
    rtl_facts: Path | None = None,
    compiler_bundle_manifest: Path | None = None,
) -> list[str]:
    """Fixed trusted worker; candidate request fields never select executable or resources."""
    command = [sys.executable, "-m", "merlin.development.backend.feedback.selfcheck.adapters"]
    command += context_argv(context) + ["--capsules-root", str(capsules_root)]
    if contract is not None:
        command += ["--contract", str(contract)]
    if rtl_facts is not None:
        command += ["--rtl-facts", str(rtl_facts)]
    if compiler_bundle_manifest is not None:
        command += ["--compiler-bundle-manifest", str(compiler_bundle_manifest)]
    from merlin.development.backend.authoring.library.compiler_library import child_arguments

    return [*command, *child_arguments()]


def _atomic_json(path: Path, doc: dict) -> None:
    """Publish an agent-visible diagnostic without exposing a half-written JSON document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(
            json.dumps(compact_document(doc, workspace=Path.cwd(), namespace="selfcheck"), sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _suite_size(capsules_root: Path) -> int:
    """How many public capsules exist, so a partial check can say what it did NOT check.

    Counted the way the grader DISCOVERS them (:func:`merlin.development.targetgen.corpus.capsule_io.discover_capsules`:
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
        from merlin.target.registry.descriptor_loading import load_target_experiment

        te = load_target_experiment(context.descriptor)
        return te.target, te.sim_via
    except Exception:  # noqa: BLE001
        return context.target, None


def _target(context: InvocationContext) -> str:
    return _target_sim_via(context)[0]


def _required_rtl_engine() -> str | None:
    """The exact RTL engine imposed by the enclosing experiment, if any."""
    return os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip() or None


def _sim_policy_error(sim: str, sim_via: str | None) -> str | None:
    """Refuse an RTL request that differs from the experiment-wide engine pin.

    Spike remains available as a correctness-only screen.  Every elaborated-RTL request must name the
    required engine exactly; inheriting the pin while directly constructing a Verilator adapter used to
    let a self-check run the wrong simulator even though formal grading was GSIM-only.
    """
    from merlin.development.backend.feedback.selfcheck.adapters import SIM_TIER

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
    from merlin.development.backend.brokers.selfcheck_dispatch import _submission_digest
    from merlin.development.backend.feedback.certification import promotion
    from merlin.development.backend.feedback.grading import fast_grade as _merlin_fast_grade_1
    from merlin.development.backend.feedback.grading import loop_grade as loop_grading

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
    completed = _merlin_fast_grade_1._completed_loop_promoter(
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
        return dict(capsule_availability.qa_checkpoint_adapters(target, sim_via, **kwargs)), sim
    policy_error = _sim_policy_error(sim, sim_via)
    if policy_error:
        raise ValueError(policy_error)
    if sim == "vcs" and readback_policy is not None:
        raise ValueError("VCS self-check does not support the selected invocation readback policy")
    if sim_via == "chipyard":
        ad = {"L2": capsule_adapters.simulator_adapter("spike", target, **kwargs)}
        if sim in ("verilator", "gsim"):
            # Build L3 from the engine the caller NAMED. This used to hardcode "verilator", which made
            # --sim gsim silently certify on verilator -- a result attributed to the wrong engine. The
            # adapter factory is engine-generic (it forwards simulator=sim and gates on
            # backend.available(sim)), so an absent engine fails closed here rather than substituting.
            selection = grade_l3_selection(target, sim)
            ad["L3"] = capsule_adapters.simulator_adapter(
                sim, target, **kwargs, **({"selection": selection} if selection else {})
            )
        if sim == "vcs":
            # In the unpinned legacy ladder VCS is an L4 addition above Verilator L3.  An experiment-wide
            # VCS pin, however, means *no other RTL engine may run*, so do not build that Verilator rung.
            # If VCS is unavailable the pinned run must refuse instead of silently substituting.
            required = _required_rtl_engine()
            if required is None:
                ad["L3"] = capsule_adapters.simulator_adapter("verilator", target)
            try:
                from merlin.development.targetgen import vcs_adapter as VA  # optional, config-gated

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
    return capsule_adapters.oracle_adapters(target, sim_via, **kwargs), sim


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

    from merlin.development.targetgen.oracles import tier_policy
    from merlin.target.registry.descriptor_loading import load_target_experiment

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
    adapters = capsule_adapters.oracle_adapters(context.target, experiment.sim_via, **readback_kwargs(context)) or {}
    # The functional tier only: an elaborated-RTL simulator spends about twenty minutes on ONE such
    # layer. Every row names the tiers withheld, so a pass here reads as the screen it is.
    ladder = tier_policy.tier_depth_order(adapters)
    screen = {tier: adapters[tier] for tier in ladder[:1]}
    runs_root = (artifact_root or Path("selfcheck_out")) / "model_layers"
    try:
        score = _merlin_submission_2.grade(
            str(sub),
            capsules_root=str(root),
            runs_root=str(runs_root),
            labels={"public", "dev"},
            contract=str(contract if contract is not None else context.repo / "src/merlin/resources/contracts"),
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
    closed_rows = _merlin_capsule_feedback_3._per_capsule_from_results(runs_root)
    for result_path in sorted(
        (runs_root / "runs" / capsule_eligibility.suite_for(context.target)).glob("*/capsule_result.json")
    ):
        if result_path.parent.name not in graded_now:
            continue
        try:
            result = json.loads(result_path.read_text())
        except Exception:  # noqa: BLE001
            continue
        name = result.get("capsule", result_path.parent.name)
        if _merlin_capsule_feedback_3._candidate_native_feedback(result) is not None:
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
    txt = json.dumps(compact_document(report, workspace=Path.cwd(), namespace="selfcheck"), indent=2)
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
