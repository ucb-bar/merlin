"""Host-side grading of immutable candidate copies during Phase 1 authoring.

Owns snapshot, grade, evidence, promotion and redacted publication ordering.
Numerical grading remains in feedback.qa and the independent capsule evaluators.
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import yaml

from merlin.targetgen.target_experiment import load_target_experiment
from merlin_experiments.phase1 import run_inputs as RI
from merlin_experiments.phase1 import treatments as T
from merlin_experiments.phase1.context import InvocationContext
from merlin_experiments.phase1.feedback import lifecycle as FL


@dataclass(frozen=True)
class GradingInputs:
    """One invocation's public view, descriptor policy and grading contract.

    Public and policy roots remain distinct for fullsuite treatments.
    promotion_root retains the existing promotion API's explicit frozen-view
    override; None preserves its descriptor-owned legacy multi-root selection.
    Frozen runs pass an admitted public Path. Legacy diagnostic runs may supply
    a resolver: it is invoked only after the manifest/language gate, preserving
    the historical no-submission path without materializing a corpus.
    No construction/import discovers a target or publishes a corpus.
    """

    context: InvocationContext
    arm: str
    public_root: Path | Callable[[], Path]
    policy_roots: tuple[Path, ...]
    contract: Path | None
    promotion_root: Path | None = None
    additional_forbidden: tuple[str, ...] = ()


def declared_loop_tiers(policy_roots: tuple[Path, ...]) -> set:
    """The oracle tiers THIS target's graded corpus declares in ``required_oracle_tiers``.

    Read from the corpus roots (not the materialized subset, which is derived FROM this) and handed to
    ``qa_loop_adapters`` so the per-round loop tier is one the capsules actually declared, rather than
    whichever endpoint tier happens to be fastest."""
    from merlin.targetgen.contract.materialize import declared_oracle_tiers

    try:
        return declared_oracle_tiers(*policy_roots)
    except Exception:  # noqa: BLE001 — no resolvable corpus -> legacy fastest-tier behavior
        return set()


def cert_tiers_beyond_loop(
    *,
    context: InvocationContext,
    policy_roots: tuple[Path, ...],
    public_roots: Callable[[], Path | tuple[Path, ...] | list[Path]],
) -> tuple[set, set]:
    """``(cert tiers held back from the fast loop, the subset the pilot corpus makes MANDATORY)``.

    One derivation, shared by the checkpoint gate and by the fast first grade, so the two can never
    disagree about which tier a verdict is missing. Target-agnostic: the ladders come from
    ``qa_checkpoint_adapters`` / ``qa_loop_adapters`` for THIS target and the mandatory set from the
    capsules' own ``required_oracle_tiers`` — no sim name is written down here.

    Resolve public roots only after finding an additional cert tier. Native
    diagnostic callers may lazily materialize their legacy corpus at that point;
    admitted grading callers return their already-resolved immutable public view.
    """
    from merlin.targetgen import capsule_runner as _CR

    try:
        te = load_target_experiment(context.descriptor)
        ck = _CR.qa_checkpoint_adapters(te.target, te.sim_via)
        loop = _CR.qa_loop_adapters(te.target, te.sim_via, declared_tiers=declared_loop_tiers(policy_roots))
    except Exception:  # noqa: BLE001 — no resolvable checkpoint oracle -> nothing to gate on
        ck, loop = {}, {}
    cert_tiers = set(ck) - set(loop)  # the cycle-accurate cert tiers held back from the fast loop
    if not cert_tiers:
        return set(), set()
    mandatory: set = set()
    _roots = public_roots()
    for _r in _roots if isinstance(_roots, (list, tuple)) else [_roots]:
        for cf in Path(_r).rglob("capsule.yaml"):
            try:
                doc = yaml.safe_load(cf.read_text()) or {}
            except Exception:  # noqa: BLE001
                continue
            mandatory |= set(doc.get("required_oracle_tiers") or [])
    return cert_tiers, cert_tiers & mandatory


def submission_language_ok(submission_dir, arm: str) -> tuple[bool, str]:
    """ENFORCE the arm language contract on an EMITTED submission: a merlin arm (arm-3/arm-4) must build
    its dialect with the xDSL/Python kit — NOT a hand C++/TableGen backend. Returns (ok, reason). Reads
    the submission's manifest.yaml: for a merlin arm it fails if ``language: cpp`` or a build block that
    compiles a C++ tool (cmake / mlir-tblgen / a *-opt binary) is present. The C++ arms are exempt (that
    is their mandated method). Pure + target-agnostic — the grader/driver calls this to reject a
    non-compliant round with an actionable reason instead of grading a forbidden backend."""
    from pathlib import Path

    import yaml

    if "merlin_assisted" not in arm:  # only the merlin (xDSL) arms are constrained
        return True, "not a merlin arm (no xDSL mandate)"
    mpath = Path(submission_dir) / "mlir_oot" / "manifest.yaml"
    if not mpath.is_file():
        mpath = next(iter(sorted(Path(submission_dir).rglob("manifest.yaml"))), None)
    if not mpath or not mpath.is_file():
        return False, "no manifest.yaml in submission"
    m = yaml.safe_load(mpath.read_text()) or {}
    lang = str(m.get("language", "")).strip().lower()
    if lang in ("cpp", "c++", "cxx"):
        return False, f"merlin arm must use xDSL/Python, manifest declares language: {lang}"
    build = m.get("build") or {}
    blob = " ".join(str(v) for v in (build.values() if isinstance(build, dict) else [build])).lower()
    for marker in ("cmake", "mlir-tblgen", "tblgen", "clang++", "g++"):
        if marker in blob:
            return False, f"merlin arm must use xDSL/Python, build block invokes {marker!r} (C++ toolchain)"
    return True, f"xDSL/Python (language={lang or 'python'})"


def _language_ok(submission_dir: Path, arm: str) -> tuple[bool, str]:
    """Enforce the current arm's language mandate on the emitted submission (merlin arms => xDSL/Python,
    not a hand C++ backend). Uses the submission language policy; degrades to OK if the
    check is unavailable so it never spuriously blocks a run."""
    try:
        return submission_language_ok(submission_dir, arm)
    except Exception:  # noqa: BLE001 — never let the compliance check itself break grading
        return True, "language check unavailable"


def grade(
    ws: Path,
    run_dir: Path,
    rnd: int,
    no_oracle: bool,
    timeout: int,
    label: str = "round",
    *,
    inputs: GradingInputs,
    scratch_key: str | None = None,
    previous_scratch_key: str | None = None,
    qa_runner: T.QARunner | None = None,
) -> dict:
    """Copy the agent's submission to an operator-only scratch, grade it, return + persist the
    redacted verdict (into ws/qa/verdict.json for the next round, and archived per round).

    ``label`` names the archive namespace. It stays ``"round"`` — byte-identical file names — for every
    round grade, so the per-round trajectory readers (`gen_trajectory_v2`, `abc_status`, which glob
    ``verdict_round_*.json``) keep seeing exactly the rounds. A grade taken WHILE a turn is still running
    is not a round and is filed under its own label, so it cannot be mistaken for one."""
    # In-turn tick numbers restart for every agent round. Keying scratch only by ``rnd`` therefore
    # reuses the previous round's sealed, read-only ``cand_901/submission`` and fails before grading
    # with PermissionError. Keep the numeric id for logs, but let the caller supply a run-wide scratch
    # identity that includes both the agent round and its tick.
    _key = scratch_key or (f"{rnd:02d}" if label == "round" else str(rnd))
    cand = run_dir / "_qa_work" / f"cand_{_key}" / "submission"
    if cand.exists():
        shutil.rmtree(cand.parent)
    _lok, _lwhy = _language_ok(ws / "submission", inputs.arm)
    if not (ws / "submission" / "manifest.yaml").exists():
        # ZERO, NOT A PLACEHOLDER. These two stubs are "nothing was graded", and they used to say
        # `n_capsules: 4`, which reads as a real cohort of four: `0/4` appears in the round log
        # identically to a genuine grade, and `per_capsule: []` is the only thing distinguishing them.
        # Measured cost of that: a reader tracked a live run for an hour believing its graded
        # denominator had collapsed from 96 to 4. A count of nothing graded is 0.
        verdict = {
            "all_pass": False,
            "n_passed": 0,
            "n_capsules": 0,
            "package_failure": {"plane": "schema", "detail": "no submission/manifest.yaml"},
            "per_capsule": [],
            "note": "write submission/manifest.yaml first.",
        }
    elif not _lok:
        # ENFORCE the arm language mandate: a merlin arm must build with the xDSL kit, not a C++ tool.
        # A non-compliant submission gets a failing verdict with the fix reason (never silently graded).
        verdict = {
            "all_pass": False,
            "n_passed": 0,
            "n_capsules": 0,
            "package_failure": {"plane": "language", "detail": _lwhy},
            "per_capsule": [],
            "note": f"language mandate: {_lwhy}. Use the xDSL/Python kit "
            "(oot_starterkit + xdsl_dialects), declare language: python, no C++/cmake build.",
        }
    else:
        shutil.copytree(ws / "submission", cand, ignore=shutil.ignore_patterns("build", "__pycache__", ".git"))
        RI.strip_build_state(cand)  # clean, relocatable build per grade (abc9 L3-build bug)
        public_root = inputs.public_root() if callable(inputs.public_root) else inputs.public_root
        out = run_dir / "qa_history" / f"verdict_{label}_{_key}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        argv = [
            "--submission",
            str(cand),
            "--capsules-root",
            str(public_root),
            "--out",
            str(out),
            "--runs-root",
            str(run_dir / "_qa_work" / f"runs_{_key}"),
            "--timeout",
            str(timeout),
        ]
        if no_oracle:
            argv.append("--no-oracle")
        from merlin_experiments.phase1.feedback import qa

        verdict = (qa_runner or partial(qa.run, context=inputs.context))(
            str(cand),
            str(public_root),
            run_dir / "_qa_work" / f"runs_{_key}",
            {"public", "dev"},
            no_oracle,
            timeout,
            **({"contract": inputs.contract} if inputs.contract is not None else {}),
            **({"additional_forbidden": inputs.additional_forbidden} if inputs.additional_forbidden else {}),
        )
        verdict = write_verdict(out, verdict)
        _write_stage_ledger(
            run_dir,
            rnd,
            cand,
            run_dir / "_qa_work" / f"runs_{_key}",
            verdict,
            artifact_key=scratch_key,
            previous_artifact_key=previous_scratch_key,
        )
        _attach_shape_generalization(
            verdict,
            cand,
            run_dir,
            rnd,
            timeout=timeout,
            artifact_key=scratch_key,
            context=inputs.context,
            contract=inputs.contract,
            additional_forbidden=inputs.additional_forbidden,
        )
        _record_plateau(run_dir)  # operator-side; deliberately not in the agent's verdict
        FL.record_channel_health(ws, run_dir)
    # PROMOTE off the round grade too. Promotion is hooked into both BROKERS, but a broker only sees a
    # verdict the agent ASKED for -- and a converged agent stops asking. Measured on the run that
    # motivated this: 24 self-checks in round 0, then ZERO in rounds 1 and 2 once it reached the corpus
    # ceiling, so the only verdict produced in those rounds was this one and promotion had nothing to fire
    # on. Three paths produce a verdict; all three must consider promotion, or the deeper tier is only
    # ever reached while the agent is still struggling -- which is exactly backwards, since a converged
    # submission is the one worth certifying.
    # Runs AFTER _attach_shape_generalization so promotion considers the completed verdict.
    try:
        import sys as _sys

        from merlin_experiments.phase1.feedback.promotion import promote as _promote
        from merlin_experiments.phase1.feedback.promotion import resolve_tiers as _resolve

        _loop, _cert, _cover = _resolve(ws, context=inputs.context, capsules_root=inputs.promotion_root)
        if _loop and _cert and isinstance(verdict, dict) and verdict.get("per_capsule"):
            _p = _promote(
                ws,
                ws / ".qa_channel",
                verdict,
                _loop,
                _cert,
                _cover,
                _sys.stderr,
                source_ws=cand.parent,
                context=inputs.context,
                capsules_root=inputs.promotion_root,
            )
            if _p:
                print(f"  [promote] round grade -> {_cert}: {_p}", flush=True)
    except Exception as _pe:  # noqa: BLE001 -- promotion is an optimisation, never a gate
        print(f"  [promote] skipped: {type(_pe).__name__}: {_pe}", flush=True)

    # hand the redacted verdict to the agent for the next round
    qa_dir = ws / "qa"
    qa_dir.mkdir(exist_ok=True)
    write_verdict(qa_dir / "verdict.json", verdict)
    return verdict


# --- the agent must never run a whole turn BLIND -------------------------------------------------
# MEASURED on merlincirct_arm4_func_20260901_codex1: the agent ran its ENTIRE 6184-second turn with no
# `qa/` directory at all, and said so itself in its closing message ("qa/verdict.json was not present").
# The round's verdict is produced AFTER the turn, and the mandatory ladder that produces it includes the
# cycle-accurate cert tier — one capsule (GM0) alone cost 1818s of Verilator — so under a long turn
# (`--schedule continuous --round-timeout 43200`) the first feedback an agent can read is hours away, or
# never. A blind agent is not a measurement of the agent.
#
# The fix is to make the FIRST grade fast and cheap: the LOOP tier only, published within minutes, then
# the expensive full-ladder grade on the normal --grade-interval. What the fast grade must never do is
# claim more than it ran: the cert tiers it withheld are NAMED in `tiers_not_run`, and `all_pass` is
# `null` (UNKNOWN) whenever a MANDATORY tier was withheld — never `true`. A check that could not run
# reading as SUCCESS is a recurring bug in this repo; this is not another instance of it.


def _fast_loop_verdict_doc(red: dict, tiers_graded, tiers_withheld) -> dict:
    """Build the fast first verdict from the per-capsule readback. PURE, so its honesty is testable.

    `n_passed` counts capsules that passed THE TIERS THAT RAN, and the document says so. `all_pass` is a
    bool only when nothing mandatory was withheld; otherwise it is None — the JSON null that reads as
    UNKNOWN to every consumer and is falsy to every gate, so no downstream check can mistake a partial
    ladder for a converged run.
    """
    rows: list = []
    npass = 0
    for name, info in sorted(red.items()):
        info = info or {}
        npass += int(info.get("status") == "pass")
        rows.append(
            {
                "capsule": name,
                "status": info.get("status"),
                "tiers": info.get("tiers") or {},
                "numeric_status": info.get("numeric_status"),
                "mismatch_count": info.get("mismatch_count"),
                "trace_status": info.get("trace_status"),
                "trace_violations": info.get("trace_violations") or [],
                "failure_plane": info.get("failure_plane"),
                "failure_category": info.get("failure_category"),
                "failure_detail": info.get("failure_detail"),
            }
        )
    graded, withheld = sorted(tiers_graded), sorted(tiers_withheld)
    complete = not withheld
    nc = len(rows)
    return {
        "qa_gate": "capsule_bench_v0_pilot",
        "stage": "first_grade_loop_tier",
        "gradeable": True,
        "tiers_graded": graded,
        "tiers_not_run": withheld,
        "mandatory_tiers_complete": complete,
        # UNKNOWN, never success: a mandatory tier that did not run may not be reported as passed.
        "all_pass": (bool(nc > 0 and npass == nc) if complete else None),
        "n_passed": npass,
        "n_capsules": nc,
        "per_capsule": rows,
        "note": (
            "FAST FIRST GRADE — the LOOP tier only, published early so you are not working blind. "
            f"Tiers graded: {graded or 'none'}. Tiers NOT run: {withheld or 'none'} — those are "
            "UNKNOWN, not passed; `all_pass` is null while any mandatory tier is outstanding. The "
            "full mandatory-ladder grade follows on the normal interval and is the one that "
            "converges the run. Contains NO reference output values."
        ),
    }


def fast_grade(
    ws: Path,
    run_dir: Path,
    tick: int,
    timeout: int,
    *,
    inputs: GradingInputs,
) -> dict:
    """Grade the current submission at the LOOP tier ONLY and publish it as the agent's verdict.

    Cheap by construction: `qa_loop_adapters` is the fast functional tier the corpus itself declares, so
    this costs seconds-to-minutes where the checkpoint ladder costs tens of minutes per capsule. Same
    snapshot-copy discipline as `qa_grade`, so it is safe to run while the agent is editing.
    """
    from merlin.targetgen import capsule_grade as _CG
    from merlin.targetgen import capsule_runner as _CR
    from merlin_experiments.phase1.feedback import qa as _qc

    if not (ws / "submission" / "manifest.yaml").is_file():
        raise RuntimeError("no submission/manifest.yaml to grade yet")
    cand = run_dir / "_qa_work" / f"fcand_{tick}" / "submission"
    if cand.parent.exists():
        shutil.rmtree(cand.parent)
    shutil.copytree(ws / "submission", cand, ignore=shutil.ignore_patterns("build", "__pycache__", ".git"))
    RI.strip_build_state(cand)
    fruns = run_dir / "_qa_work" / f"fruns_{tick}"
    te = load_target_experiment(inputs.context.descriptor)
    adapters = _CR.qa_loop_adapters(te.target, te.sim_via, declared_tiers=declared_loop_tiers(inputs.policy_roots))
    _, _withheld = cert_tiers_beyond_loop(
        context=inputs.context,
        policy_roots=inputs.policy_roots,
        public_roots=lambda: inputs.public_root() if callable(inputs.public_root) else inputs.public_root,
    )
    try:
        _CG.grade(
            str(cand),
            capsules_root=str(inputs.public_root() if callable(inputs.public_root) else inputs.public_root),
            runs_root=str(fruns),
            labels={"public", "dev"},
            contract=str(inputs.contract if inputs.contract is not None else inputs.context.repo / "merlin/contract"),
            oracle_adapters=adapters,
            timeout=timeout,
            max_workers=_CG.default_grade_workers(),
            target=te.target,
            additional_forbidden=inputs.additional_forbidden,
        )
    except Exception as e:  # noqa: BLE001 — a partial readback is still feedback, and every tier the
        # grade did not reach stays UNKNOWN in the document below rather than reading as clean.
        print(f"[first-grade] grade error: {type(e).__name__}: {str(e)[:200]}", flush=True)
    verdict = _fast_loop_verdict_doc(_qc._per_capsule_from_results(fruns), sorted(adapters), sorted(_withheld))
    out = run_dir / "qa_history" / f"verdict_fast_{tick}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    verdict = write_verdict(out, verdict)
    (ws / "qa").mkdir(exist_ok=True)
    write_verdict(ws / "qa" / "verdict.json", verdict)
    _record_plateau(run_dir)  # operator-side; deliberately not in the agent's verdict
    FL.record_channel_health(ws, run_dir)
    return verdict


# The public suite is the set of shapes an agent can see, so a backend that keys on those shapes passes
# it by construction and the loop has nothing left to say. Measured: a submission converged at 14/26 with
# every failure a shape it had never implemented, its self-check clean on everything it HAD, and no round
# feedback able to point at the gap -- the first instrument to see it was a post-freeze holdout.
#
# These probes are derived from the target's own declared capability closure and its DERIVED tile edge
# (never the corpus), so they leak nothing an agent could not compute itself, and they run at the cheap
# loop tier. Restricted to the multi-tile corners: the question is "does this backend generalize past ONE
# tile, and along WHICH axis" -- per-axis, because a backend that loops over K and N but not M passes two
# of the three, and only naming the axis makes the result actionable.
def _attach_shape_generalization(
    verdict: dict,
    cand,
    run_dir,
    rnd: int,
    *,
    timeout: int,
    artifact_key: str | None = None,
    context: InvocationContext,
    contract: Path | None = None,
    additional_forbidden: tuple[str, ...] = (),
) -> None:
    """Probe whether this round's candidate LOWERS shapes past a single tile, and fold it into the gate.

    Structural, not numerical: it runs only the emit half of the contract and compares the size of the
    emitted artifact across shapes, so it costs no oracle, needs no golden, and works on an operand
    format that has no CPU reference. See :mod:`merlin.targetgen.lowering_coverage` for the invariant
    ("a program for a bigger problem cannot be smaller") and why it is per-axis.

    Failure to run is RECORDED, never treated as clean -- a probe that did not run reading as a pass is
    the same class of bug as an unavailable oracle scoring as one.
    """
    key = artifact_key or f"{rnd:02d}"
    out = run_dir / "qa_history" / f"shape_coverage_round_{key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        from merlin.targetgen import lowering_coverage as LC

        cov = LC.sweep(
            cand,
            target=context.target,
            contract=str(contract if contract is not None else context.repo / "merlin/contract"),
            timeout=min(timeout, 300),
            additional_forbidden=additional_forbidden,
        )
        out.write_text(json.dumps(cov, indent=2))
    except Exception as e:  # noqa: BLE001 -- record it; never let it read as clean
        verdict["shape_coverage"] = {
            "ran": False,
            "error": f"{type(e).__name__}: {e}",
            "note": "the shape-coverage probes did NOT run this round; this is NOT a pass.",
        }
        return

    verdict["shape_coverage"] = {
        "ran": True,
        "tile_edge": cov.get("tile_edge"),
        "baseline_tile_lowered": cov.get("baseline_tile_lowered"),
        "per_corner": {c["corner"]: c["outcome"] for c in cov.get("corners", [])},
        "emitted_work": cov.get("emitted_work"),
        "multi_tile_axes_uncovered": cov.get("multi_tile_axes_uncovered") or [],
        "all_covered": bool(cov.get("all_covered")),
        "unmeasured": cov.get("unmeasured"),
        "detail": {c["corner"]: c.get("detail") for c in cov.get("corners", []) if c.get("detail")},
        "note": (
            "DERIVED shape probes, not corpus capsules: the SAME contraction at one tile and at two "
            "tiles in each of M, K and N, at this target's derived tile edge. `emitted_work` is the "
            "size of the program you emitted for each -- a bigger problem cannot need a SMALLER "
            "program, so a corner marked `collapsed` is a shape you silently refused. "
            "`multi_tile_axes_uncovered` names the axis your lowering does not loop over: fix the "
            "loop, not the arithmetic. If you genuinely cannot lower a shape, DECLARE it "
            "(`declined` on the command buffer) instead of emitting a terminator."
        ),
    }
    # THE GATE. Passing every public capsule while lowering only the shapes they happen to use is exactly
    # the state that shipped at 14/26, so the loop must not call that converged.
    if verdict.get("all_pass") and not verdict["shape_coverage"]["all_covered"]:
        verdict["all_pass"] = False
        verdict["not_converged_reason"] = (
            "every public capsule passes, but the derived shape probes show the backend does not lower "
            + (
                f"past one tile on axis/axes {verdict['shape_coverage']['multi_tile_axes_uncovered']}"
                if verdict["shape_coverage"]["multi_tile_axes_uncovered"]
                else "the baseline tile itself (nothing about shape can be concluded yet)"
            )
        )


def write_verdict(path: Path, verdict: dict) -> dict:
    """Write a verdict WITH the time it was produced, and return it.

    Verdicts carried no timestamp of any kind. Under the continuous schedule the grader refreshes
    them while the agent works, so the sequence of verdicts IS the run's progress record -- and with
    no stamp inside, the only clock a reader has is the file's mtime. An mtime does not survive a
    copy, and several of these run trees have been copied between worktrees, so every progress curve
    drawn from them rests on filesystem metadata rather than on anything the run recorded.

    Stamping costs one field. `graded_at` is UTC and ISO-8601, matching the spelling every other
    record in the run tree uses. Existing verdicts are unaffected: readers fall back to the mtime and
    say so.
    """
    verdict = dict(verdict)
    verdict.setdefault("graded_at", _dt.datetime.now(_dt.UTC).isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdict, indent=2))
    return verdict


def _record_plateau(run_dir: Path) -> None:
    """Assess this run's progress across GRADES and record it OPERATOR-SIDE. Never raises.

    OUT OF BAND, for the same reason `_write_stage_ledger` is: telling the agent it has plateaued, or
    naming the capsules that have never passed, is FEEDBACK -- and feedback defines an arm. Handing it
    over would change the treatment and make the run incomparable with every earlier one. This is for
    whoever decides whether to keep paying for the run.

    It exists because the detector that was here could not fire. `--plateau-rounds` counts consecutive
    ROUNDS with no progress, and the default schedule is `continuous` -- one long session, so exactly
    one round. Its threshold was unreachable in the mode every real run uses, and opt-in besides.
    Measured on merlincirct_g4p1_biasabi_20260906: 92/96 reached at 2.19 h, then 3.91 h more -- 64% of
    the run -- with the score never moving and nothing saying so. A grade is the unit the continuous
    schedule actually produces, so that is the unit here.
    """
    try:
        from merlin.targetgen import plateau as _PL

        hist = sorted((run_dir / "qa_history").glob("verdict_*.json"), key=lambda q: q.stat().st_mtime)
        grades = []
        for q in hist:
            try:
                grades.append(json.loads(q.read_text()))
            except Exception:  # noqa: BLE001 -- one unreadable grade must not blind the assessment
                continue
        got = _PL.assess(grades)
        doc = {
            "stuck": got.stuck,
            "reason": got.reason,
            "stalled_grades": got.stalled_grades,
            "n_grades": got.n_grades,
            "best_passed": got.best_passed,
            "latest_passed": got.latest_passed,
            "n_capsules": got.n_capsules,
            "never_passed": list(got.never_passed),
            "regressed": list(got.regressed),
            "sentence": got.sentence(),
            "note": (
                "operator-side only: this is never handed to the agent, because new feedback "
                "would change the arm's treatment"
            ),
        }
        (run_dir / "plateau.json").write_text(json.dumps(doc, indent=2))
        if got.stuck or got.regressed:
            print(f"[plateau] {got.sentence()}", flush=True)
    except Exception:  # noqa: BLE001 -- diagnostics may never fail a grade
        return


def _write_stage_ledger(
    run_dir,
    rnd: int,
    cand,
    runs_root,
    verdict,
    *,
    artifact_key: str | None = None,
    previous_artifact_key: str | None = None,
) -> None:
    """Record per-round artifact fingerprints beside the verdict — OUT OF BAND.

    Answers "did the agent's edit reach what was graded?" from bytes, so a plateau is one line instead of
    an investigation. Written into the run dir (a sandbox-DENIED path) and never into the agent's
    ``qa/verdict.json``: telling the agent its edit was inert is feedback, and feedback defines an arm.

    Never raises. This is diagnostics — it must not be able to fail a round that otherwise graded fine.
    """
    try:
        from merlin.targetgen import stage_ledger as SL

        led_dir = run_dir / "rounds"
        led_dir.mkdir(parents=True, exist_ok=True)
        key = artifact_key or f"{rnd:02d}"
        if artifact_key is None:
            previous_key = f"{rnd - 1:02d}" if rnd else None
        else:
            previous_key = previous_artifact_key
        prev_p = led_dir / f"round_{previous_key}.stage_ledger.json" if previous_key is not None else None
        prev = json.loads(prev_p.read_text()) if prev_p is not None and prev_p.is_file() else None

        # Per-capsule emit dirs, found by SHAPE at any depth: a dir named for the emit output whose
        # PARENT is a graded capsule dir (it holds the capsule's own result/manifest). Depth and the
        # intervening dir names are never assumed -- no target, no suite, and no generated-root literal --
        # so a target that lays its run tree out differently is still discovered, and one that lays it out
        # unrecognizably simply yields none rather than a wrong answer.
        marks = ("capsule_result.json", "run_manifest.yaml")
        roots = {
            d.parent.name: d
            for d in Path(runs_root).rglob("generated")
            if d.is_dir() and any((d.parent / m).is_file() for m in marks)
        }
        led = SL.build(submission_dir=cand, emitted_roots=roots, previous=prev)
        led["round"] = rnd
        led["failing_and_frozen"] = SL.failing_and_frozen(led, verdict)
        (led_dir / f"round_{key}.stage_ledger.json").write_text(json.dumps(led, indent=2))
        print(f"  {SL.summarize(led)}", flush=True)
        if led["failing_and_frozen"]:
            print(
                f"  stage_ledger: FAILING AND FROZEN ({len(led['failing_and_frozen'])}): "
                f"{', '.join(led['failing_and_frozen'][:10])}"
                f"{' ...' if len(led['failing_and_frozen']) > 10 else ''}",
                flush=True,
            )
    except Exception as e:  # noqa: BLE001 - diagnostics never gate
        print(f"  stage_ledger: unavailable ({type(e).__name__}: {e})", flush=True)
