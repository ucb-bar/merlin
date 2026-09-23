"""Cert-tier promotion, admission and evidence policy for Phase 1 authoring.

These helpers consume explicit invocation/grading inputs. They do not run or own
agent rounds: ordinary authoring and L3 repair rounds share the authoring engine's
single continuation state.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import yaml

from merlin.targetgen.target_experiment import load_target_experiment
from merlin_experiments.phase1.feedback import loop_grading as LG

#: Safety multiple applied to a FITTED per-capsule L3 cost — the same "generous 2x" the measured-T_obs
#: path below already applies. A margin, not a hardware fact.
L3_BUDGET_SAFETY = 2.0
#: Hang bound. A size-scaled budget is never granted more than this multiple of the unscaled bound: the
#: budget's OTHER job is to stop a kernel that never reaches its stop condition, and a fit extrapolated
#: far past its observations must not turn that bound off. Not a measurement — raise it deliberately.
L3_BUDGET_MAX_MULT = 4


def _public_capsules(inputs: LG.GradingInputs):
    return inputs.public_root() if callable(inputs.public_root) else inputs.public_root


def _l3_barrier_decision(l3_all_pass: bool, rnd: int, max_rounds: int) -> str:
    """The NON-TERMINAL barrier decision (extracted so it is unit-testable — this is the abc7 regression).
    Returns: 'done' (L3 passed → the ONLY success exit), 'budget' (round budget exhausted → stop honestly,
    reason=max_rounds), or 'iterate' (L3 failed but rounds remain → feed back + fix round, do NOT end)."""
    if l3_all_pass:
        return "done"
    if rnd >= max_rounds:
        return "budget"
    return "iterate"


# --- PER-CAPSULE promotion into the cycle-accurate checkpoint ------------------------------------
# The checkpoint used to be ALL-OR-NOTHING: it ran only when ``verdict["all_pass"]`` was true, i.e. only
# when EVERY capsule had cleared the loop tier. MEASURED on the live run
# merlincirct_arm4_func_20260901_codex1: 19 of 32 capsules passed the loop tier and 13 did not, so
# ``all_pass`` was False, the whole checkpoint phase never ran, and the 19 capsules that had EARNED a
# cycle-accurate cert got neither the cert nor any of its VERILATOR_ATTEMPTS fix rounds — because of the
# other 13. Promotion is a PER-CAPSULE fact (a capsule that clears the loop tier is certifiable on its
# own), so it is decided per capsule here.
#
# The capsules still failing the loop tier are NOT submitted: their loop-tier bug is the thing to fix
# first, and paying minutes of cycle-accurate sim to re-learn it is waste. They are also NOT dropped —
# each is recorded ``not_promoted`` with the reason it was held back, and a tier that DID NOT RUN is
# never counted as a pass (see :func:`_l3_attempt_tally`). Whole-corpus completion is unaffected: the
# attempt's ``all_pass`` is still ``n_passed == n_capsules`` over the WHOLE corpus, held-back capsules
# included, so a partially-promoted run can never read as certified.
def _l3_promotion(verdict: dict, loop_tier: str | None) -> tuple[list, list]:
    """Split a loop verdict into ``(capsules eligible for the cert tier, not-promoted records)``.

    Eligibility is read from the capsule's own loop-tier status when the verdict carries one, and from
    its overall ``status`` otherwise. A row we cannot read either way is held back with that reason
    rather than promoted — an unknown loop-tier verdict is not a pass.
    """
    eligible: list = []
    held: list = []
    for row in verdict.get("per_capsule") or []:
        row = row if isinstance(row, dict) else {}
        name = str(row.get("capsule") or "")
        tiers = row.get("tiers") or {}
        if loop_tier and loop_tier in tiers:
            status, basis = tiers.get(loop_tier), loop_tier
        else:
            status, basis = row.get("status"), "status"
        if not name:
            held.append(
                {
                    "capsule": name,
                    "reason": "verdict row carries no capsule name",
                    "loop_tier": loop_tier,
                    "loop_tier_status": status,
                }
            )
            continue
        if status == "pass":
            eligible.append(name)
            continue
        held.append(
            {
                "capsule": name,
                "reason": f"did not pass the loop tier ({basis}={status!r}); fix the loop-tier failure "
                f"first — it was NOT submitted to the cycle-accurate cert",
                "loop_tier": loop_tier,
                "loop_tier_status": status,
                "failure_plane": row.get("failure_plane"),
                "failure_tier": row.get("failure_tier"),
            }
        )
    return eligible, held


def _l3_checkpoint_should_run(run_l3: bool, workflow_conformant: bool, eligible) -> bool:
    """PER-CAPSULE gate: the cycle-accurate checkpoint runs when AT LEAST ONE capsule cleared the loop
    tier — never "when all of them did". With nothing eligible there is nothing to certify, so it is
    skipped and said out loud (an empty submission must not read as a clean checkpoint)."""
    return bool(run_l3 and workflow_conformant and eligible)


def _l3_attempt_tally(red: dict, not_promoted: list, cert_tier: str) -> tuple[list, int, int]:
    """``(rows, n_passed, n_capsules)`` for ONE checkpoint attempt, over the WHOLE corpus.

    ``red`` is the per-capsule readback for the capsules that WERE submitted; ``not_promoted`` are the
    ones that were not. Only an explicit ``pass`` AT THE CERT TIER counts toward ``n_passed``: a missing
    tier entry (the sim never ran, timed out, or the capsule was never submitted) reads as
    ``not_promoted`` / ``None``, never as success. The denominator is the whole corpus, so an attempt
    that certified every promoted capsule while capsules were held back still cannot reach
    ``n_passed == n_capsules``.
    """
    rows: list = []
    npass = 0
    for name, info in sorted(red.items()):
        status = ((info or {}).get("tiers") or {}).get(cert_tier)
        npass += int(status == "pass")
        rows.append(
            {
                "capsule": name,
                "l3_status": status,
                "cert_tier": cert_tier,
                "failure_plane": (info or {}).get("failure_plane"),
            }
        )
    graded = {r["capsule"] for r in rows}
    for held in not_promoted:
        held = held if isinstance(held, dict) else {}
        name = str(held.get("capsule") or "")
        if name in graded:
            continue  # it was submitted after all; its graded row is the authority
        rows.append(
            {
                "capsule": name,
                "l3_status": "not_promoted",
                "cert_tier": cert_tier,
                "reason": held.get("reason"),
                "loop_tier": held.get("loop_tier"),
                "loop_tier_status": held.get("loop_tier_status"),
                "failure_plane": held.get("failure_plane"),
            }
        )
    rows.sort(key=lambda r: r["capsule"])
    return rows, npass, len(rows)


def _l3_fix_verdict(vv: dict, attempt: int) -> dict:
    """Redacted L3 feedback for the next fix round, including Arm4's answer-free RTL readback."""
    out = {
        "stage": "verilator_checkpoint",
        "attempt": attempt,
        "all_pass": False,
        "n_passed": vv["n_passed"],
        "n_capsules": vv["n_capsules"],
        "note": "CYCLE-ACCURATE (verilator/L3) results — fix the capsules failing at L3, then "
        "re-run agent_selfcheck (spike) clean before declaring READY again.",
        "per_capsule": vv["per_capsule"],
    }
    # Say WHY a capsule has no cert result. Rows marked `not_promoted` were never submitted to the
    # cycle-accurate tier because they had not passed the loop tier yet; that is a loop-tier bug to fix,
    # not a cert failure, and it is emphatically not a pass.
    _held = [r for r in (vv.get("per_capsule") or []) if r.get("l3_status") == "not_promoted"]
    if _held:
        out["n_not_promoted"] = len(_held)
        out["not_promoted_note"] = (
            f"{len(_held)} capsule(s) were NOT submitted to the cycle-accurate tier because they have "
            f"not passed the loop tier yet (rows with l3_status='not_promoted'). Their cert result is "
            f"UNKNOWN, not a pass. Fix their loop-tier failure first; they are promoted automatically "
            f"as soon as the loop tier passes."
        )
    if vv.get("rtl_checks") is not None:
        out["rtl_checks"] = vv["rtl_checks"]
        out["rtl_checks_note"] = (
            "Answer-free RTL structural feedback regenerated from this L3 attempt's emitted "
            "lowering/trace. Read and address it during this fix round."
        )
    return out


def _authoring_completion(numeric_all_pass: bool, workflow_conformant: bool) -> bool:
    """The agent may leave the authoring loop only when both in-sandbox gates are evidenced."""
    return bool(numeric_all_pass and workflow_conformant)


def _formal_completion(
    numeric_all_pass: bool, workflow_conformant: bool, official_grade_complete: bool, telemetry_complete: bool
) -> bool:
    """Formal success requires the outer grade AND a complete, sealed process record."""
    return bool(numeric_all_pass and workflow_conformant and official_grade_complete and telemetry_complete)


def _official_grade_result(returncode: int, run_dir: Path, *, required_tier: str = "L3") -> dict:
    """Validate the official grader's exit status *and* its claim-bearing manifest.

    The subprocess return code is necessary but not sufficient: this rejects a stale/malformed manifest,
    a vacuous 0/0 phase, and a phase whose status says complete without the matching L3/RTL evidence.
    """
    failures: list[str] = []
    if returncode != 0:
        failures.append(f"grader_exit_nonzero:{returncode}")
    manifest_path = run_dir / "run_manifest.yaml"
    try:
        manifest = yaml.safe_load(manifest_path.read_text()) or {}
    except Exception as exc:  # noqa: BLE001 — malformed/missing evidence is a refusal, not a crash
        manifest = {}
        failures.append(f"manifest_unreadable:{type(exc).__name__}")
    if not isinstance(manifest, Mapping):
        manifest = {}
        failures.append("manifest_malformed")

    completion = manifest.get("completion")
    if not isinstance(completion, Mapping):
        failures.append("completion_status_missing")
    else:
        if completion.get("formal_grade_complete") is not True:
            failures.append("grader_reported_incomplete")
        if completion.get("required_tier") != required_tier:
            failures.append("required_tier_mismatch")

    for phase_name in ("public_dev", "hidden"):
        phase = manifest.get(phase_name)
        if not isinstance(phase, Mapping):
            failures.append(f"{phase_name}:status_missing")
            continue
        n_capsules, n_passed = phase.get("n_capsules"), phase.get("n_passed")
        if not isinstance(n_capsules, int) or isinstance(n_capsules, bool) or n_capsules <= 0:
            failures.append(f"{phase_name}:capsule_set_empty_or_malformed")
        if (
            not isinstance(n_passed, int)
            or isinstance(n_passed, bool)
            or not isinstance(n_capsules, int)
            or isinstance(n_capsules, bool)
            or n_passed != n_capsules
        ):
            failures.append(f"{phase_name}:not_all_capsules_passed")
        if phase.get("formal_complete") is not True:
            failures.append(f"{phase_name}:formal_complete_not_true")
        if phase.get("gradeable") is not True:
            failures.append(f"{phase_name}:not_gradeable")
        if phase.get("integrity_status") != "clean":
            failures.append(f"{phase_name}:integrity_not_clean")
        if phase.get("numeric_all_exact") is not True:
            failures.append(f"{phase_name}:numeric_exactness_not_complete")
        if phase.get("trace_all_pass") is not True:
            failures.append(f"{phase_name}:trace_conformance_not_complete")
        unmeasured = phase.get("unmeasured_counts")
        for field in (
            "n_not_graded_ineligible",
            "n_gated_deferred",
            "n_screened_only",
            "n_budget_exhausted",
            "n_incomplete",
            "n_not_gradeable_no_oracle",
        ):
            if not isinstance(unmeasured, Mapping) or unmeasured.get(field) != 0:
                failures.append(f"{phase_name}:{field}_not_zero")
        tiers = phase.get("tier_reached")
        if not isinstance(tiers, Mapping) or tiers.get(required_tier) != n_capsules:
            failures.append(f"{phase_name}:not_all_capsules_reached_{required_tier}")
        evidence = phase.get("pass_evidence")
        if not isinstance(evidence, Mapping) or evidence.get("rtl_backed") != n_capsules:
            failures.append(f"{phase_name}:not_all_capsules_rtl_backed")
    return {
        "complete": not failures,
        "grader_returncode": int(returncode),
        "manifest": str(manifest_path),
        "failures": failures,
    }


def _l3_instruction_counts(run_dir: Path) -> dict[str, int]:
    """``{capsule: instruction count}`` for the capsules this run has already emitted an artifact for.

    The count is the length of the capsule's OWN decoded ``instruction_trace.json`` — the submission's
    emitted RoCC stream, which is what the RTL sim actually executes and therefore what its cost scales
    with. Read from the most recently written trace per capsule (the newest submission wins). A capsule
    with no trace is simply absent: unsized, never assumed small.
    """
    out: dict[str, tuple[float, int]] = {}
    work = run_dir / "_qa_work"
    if not work.is_dir():
        return {}
    for tp in work.glob("*/runs/*/*/generated/instruction_trace.json"):
        name = tp.parent.parent.name
        try:
            n = len(json.loads(tp.read_text()).get("instructions") or [])
            mt = tp.stat().st_mtime
        except Exception:  # noqa: BLE001 — an unreadable trace sizes nothing
            continue
        if n > 0 and (name not in out or mt > out[name][0]):
            out[name] = (mt, n)
    return {k: v[1] for k, v in out.items()}


def _l3_cost_observations(run_dir: Path, cert_tier: str) -> list[tuple[int, float]]:
    """``[(instruction count, adapter wall seconds)]`` for cert-tier runs that COMPLETED.

    Only ``pass`` observations are used: a passing tier ran the artifact to its end, so its wall is a
    true cost. A failed/crashed/timed-out tier's wall is bounded by whatever budget it was given, and
    fitting a cost model to a truncated measurement is how a timeout becomes self-confirming.
    """
    obs: list[tuple[int, float]] = []
    work = run_dir / "_qa_work"
    if not work.is_dir():
        return obs
    for rp in work.glob("*/runs/*/*/capsule_result.json"):
        try:
            tier = (json.loads(rp.read_text()).get("tiers") or {}).get(cert_tier) or {}
        except Exception:  # noqa: BLE001
            continue
        wall = (tier.get("timing") or {}).get("adapter_wall_s")
        if tier.get("status") != "pass" or not wall:
            continue
        tp = rp.parent / "generated" / "instruction_trace.json"
        try:
            n = len(json.loads(tp.read_text()).get("instructions") or [])
        except Exception:  # noqa: BLE001
            continue
        if n > 0:
            obs.append((n, float(wall)))
    return obs


def _l3_cost_fit(obs: list[tuple[int, float]]) -> tuple[float, float, str] | None:
    """``(seconds_per_instruction, fixed_seconds, basis)`` from the run's own cert-tier observations.

    An RTL sim's wall is affine in the length of the stream it executes: a fixed build/elaborate/boot
    term plus a per-instruction term. Two or more distinct instruction counts separate the two by least
    squares; a single count can only give a through-origin rate, which is stated as such. Fewer than one
    observation -> ``None``, and the caller keeps the unscaled bound rather than inventing a rate.
    """
    if not obs:
        return None
    ns = [n for n, _ in obs]
    if len(set(ns)) < 2:
        rate = max(w / n for n, w in obs)
        return (
            rate,
            0.0,
            (
                f"{len(obs)} observation(s) at a single instruction count ({ns[0]}); "
                f"through-origin rate {rate:.4f} s/instruction (no fixed term separable)"
            ),
        )
    n_ = len(obs)
    mx = sum(ns) / n_
    my = sum(w for _, w in obs) / n_
    sxx = sum((n - mx) ** 2 for n in ns)
    slope = sum((n - mx) * (w - my) for n, w in obs) / sxx
    intercept = my - slope * mx
    slope, intercept = max(slope, 0.0), max(intercept, 0.0)
    return (
        slope,
        intercept,
        (
            f"least squares over {n_} passing {sorted(set(ns))[0]}..{sorted(ns)[-1]}"
            f"-instruction observations: {intercept:.0f} s fixed + "
            f"{slope:.4f} s/instruction"
        ),
    )


def _verilator_per_capsule_timeout(context) -> int:
    """The UNSCALED per-capsule L3 (verilator RTL cert) bound, from a T_obs that is POSITIVELY confirmed
    to be THIS
    target's sim (generous 2x, min 900s). The readiness gate writes scripts/.oracle_timing.json — but that
    path is a symlink shared across targets, so a radiance run must NOT inherit a GemminiRocketConfig T_obs
    measured for a different, far lighter RTL (which would floor to 900s and mass-timeout every L3 capsule,
    the abc9 "L3 0/N = timeout not skill" trap). A measurement is trusted only when it is target-scoped:
    the file ``.oracle_timing.<target>.json`` OR a legacy ``.oracle_timing.json`` whose ``target`` field
    matches. An unconfirmed / foreign-config measurement is ignored in favor of a conservative bound and a
    loud log — never a silent under-time.

    This is the FLOOR, not the budget the grade runs with: :func:`_verilator_l3_budget` scales it by the
    promoted capsules' own emitted size, because a flat bound fails a large capsule for being large."""
    import math

    tgt = context.target
    for p in (
        context.experiment / "scripts" / f".oracle_timing.{tgt}.json",
        context.experiment / "scripts" / ".oracle_timing.json",
    ):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if p.name.endswith(f".{tgt}.json") or d.get("target") == tgt:
            return max(900, int(math.ceil(2 * float(d["verilator_per_capsule_s"]))))
        print(
            f"[timeout] ignoring {p.name}: T_obs measured on config={d.get('config')!r} "
            f"target={d.get('target')!r}, not {tgt!r} — needs a target-scoped measurement",
            file=sys.stderr,
        )
    print(
        f"[timeout] no target-confirmed L3 timing for {tgt!r}; using conservative 2400s. Run readiness "
        f"or the L3 measurement to record scripts/.oracle_timing.{tgt}.json",
        file=sys.stderr,
    )
    return 2400


def _verilator_l3_budget(run_dir: Path | None, capsules: list | None, cert_tier: str = "L3", *, context) -> int:
    """The per-capsule cert-tier budget, SCALED BY CAPSULE SIZE instead of one flat number.

    Why: a flat bound fails a capsule for being large. Measured on this corpus — ``GM0`` (2310 emitted
    instructions) completed its RTL cert in 1818 s against a 2400 s flat bound, and ``GM1`` (3084
    instructions, 1.335x) projected ~2427 s and was recorded a ``tool_crash``. Its numerics passed the
    functional tier; nothing about the submission was wrong. A timeout attributed to the submission is
    worse than no result, because it gets counted.

    Derivation, all from THIS run's own measurements (never a literal):

    1. the unscaled bound from :func:`_verilator_per_capsule_timeout` (a target-confirmed T_obs, else
       the conservative fallback) — this is the FLOOR, so the budget can only ever grow;
    2. the cost model ``fixed + rate * instructions`` least-squares fitted to the cert-tier walls this
       run already observed on capsules that PASSED (:func:`_l3_cost_fit`);
    3. the largest promoted capsule's own emitted instruction count, priced through that model and given
       the same generous 2x margin the measured-T_obs path applies;
    4. capped at ``L3_BUDGET_MAX_MULT`` x the unscaled bound, so the budget still bounds a hang.

    Fails closed to today's behavior: with no fit (no passing cert-tier observation yet) or no sized
    capsule, the unscaled bound is returned AND the reason is printed. A capsule whose size is unknown
    is excluded from the pricing rather than assumed small — the floor still covers it. The whole
    derivation is written to ``<run_dir>/l3_timeout_derivation.json`` so a budget can be audited.
    """
    flat = _verilator_per_capsule_timeout(context)
    rec: dict = {
        "cert_tier": cert_tier,
        "unscaled_bound_s": flat,
        "budget_s": flat,
        "safety": L3_BUDGET_SAFETY,
        "cap_multiple": L3_BUDGET_MAX_MULT,
    }
    if run_dir is None:
        return flat
    sizes = _l3_instruction_counts(Path(run_dir))
    names = [str(c) for c in (capsules or sizes.keys())]
    sized = {n: sizes[n] for n in names if n in sizes}
    rec["n_capsules"] = len(names)
    rec["n_sized"] = len(sized)
    rec["unsized"] = sorted(n for n in names if n not in sized)
    obs = _l3_cost_observations(Path(run_dir), cert_tier)
    fit = _l3_cost_fit(obs)
    rec["n_observations"] = len(obs)
    if not sized or fit is None:
        rec["basis"] = "NOT SCALED: " + (
            "no promoted capsule has an emitted instruction trace yet"
            if not sized
            else f"no passing {cert_tier} wall observed yet, so no cost model"
        )
        print(
            f"[timeout] {cert_tier} budget NOT size-scaled ({rec['basis']}); using the unscaled {flat}s bound",
            file=sys.stderr,
        )
        _write_l3_budget_record(run_dir, rec)
        return flat
    rate, fixed, basis = fit
    biggest, n_instr = max(sized.items(), key=lambda kv: kv[1])
    predicted = fixed + rate * n_instr
    scaled = int(-(-(L3_BUDGET_SAFETY * predicted) // 1))  # ceil, no math import needed
    budget = max(flat, min(scaled, flat * L3_BUDGET_MAX_MULT))
    rec.update(
        {
            "cost_model": {"fixed_s": fixed, "s_per_instruction": rate, "basis": basis},
            "largest_capsule": biggest,
            "largest_instructions": n_instr,
            "predicted_s": predicted,
            "scaled_s": scaled,
            "budget_s": budget,
            "capped": scaled > flat * L3_BUDGET_MAX_MULT,
            "basis": (
                f"largest promoted capsule {biggest!r} emits {n_instr} instructions; "
                f"{basis}; predicted {predicted:.0f}s x{L3_BUDGET_SAFETY} margin "
                f"-> {scaled}s, floored at the {flat}s unscaled bound and capped at "
                f"{flat * L3_BUDGET_MAX_MULT}s"
            ),
        }
    )
    print(f"[timeout] {cert_tier} per-capsule budget {budget}s (was a flat {flat}s): {rec['basis']}", flush=True)
    _write_l3_budget_record(run_dir, rec)
    return budget


def _write_l3_budget_record(run_dir, rec: dict) -> None:
    """Persist the budget derivation next to the run. A budget that decides a verdict has to be
    auditable; a printed line scrolls away."""
    if not Path(run_dir).is_dir():
        return
    try:
        p = Path(run_dir) / "l3_timeout_derivation.json"
        hist = []
        if p.exists():
            try:
                hist = json.loads(p.read_text())
                hist = hist if isinstance(hist, list) else [hist]
            except Exception:  # noqa: BLE001
                hist = []
        hist.append({"at": datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ"), **rec})
        p.write_text(json.dumps(hist, indent=2))
    except Exception as e:  # noqa: BLE001 — recording must never break a grade
        print(f"[timeout] could not record the L3 budget derivation: {e}", file=sys.stderr)


def _cycle_accurate_checkpoint_enabled(*, inputs: LG.GradingInputs) -> tuple[bool, str]:
    """Whether to run the cycle-accurate RTL-cert (verilator/L3) barrier for THIS target.

    Target-agnostic + DERIVED, never a per-target literal: the barrier's cycle-accurate cert tiers are the
    checkpoint ladder (:func:`qa_checkpoint_adapters`) MINUS the fast loop tier (:func:`qa_loop_adapters`) —
    e.g. verilator L3 (and VCS L4) once spike/cyclotron L2 is the loop gate. The barrier is a pass-gate ONLY
    when at least one of those cert tiers is a MANDATORY tier of the pilot corpus (listed in a public
    capsule's ``required_oracle_tiers``). A target whose corpus makes every cert tier OPTIONAL — a prototype
    / not-RTL-certified accelerator graded on its functional oracle (its required tier is the fast
    functional sim, e.g. cyclotron L2) — SKIPS the barrier, so a normal run is never blocked on a
    slow/hanging RTL sim (the atlas-0/N `oracle_unavailable` trap, here on the driver side). An explicit
    ``MERLIN_CAPSULE_L3_CHECKPOINT`` env var forces it on (opt-in RTL cert when the sim is available and
    there is time) or off, overriding the derivation."""
    env = os.environ.get("MERLIN_CAPSULE_L3_CHECKPOINT")
    if env is not None:
        on = env.strip().lower() in ("1", "true", "yes", "on")
        return on, f"MERLIN_CAPSULE_L3_CHECKPOINT={env!r} (explicit opt-{'in' if on else 'out'})"
    cert_tiers, hit = LG.cert_tiers_beyond_loop(
        context=inputs.context,
        policy_roots=inputs.policy_roots,
        public_roots=lambda: _public_capsules(inputs),
    )
    if not cert_tiers:
        return False, "no cycle-accurate cert tier beyond the fast loop oracle for this target"
    if hit:
        return True, f"cert tier(s) {sorted(hit)} are mandatory in the pilot corpus"
    return False, (
        f"cert tier(s) {sorted(cert_tiers)} are OPTIONAL for the pilot corpus (functional-tier "
        f"pass bar) — skipping the cycle-accurate RTL barrier; set "
        f"MERLIN_CAPSULE_L3_CHECKPOINT=1 to opt in"
    )


def _loop_tier_name(inputs: LG.GradingInputs) -> str | None:
    """The fast loop tier THIS target grades on, derived from its own adapter map (``None`` when the
    ladder is unresolvable — promotion then falls back to the capsule's overall status)."""
    from merlin.targetgen import capsule_runner as _CR

    try:
        te = load_target_experiment(inputs.context.descriptor)
        loop = _CR.qa_loop_adapters(te.target, te.sim_via, declared_tiers=LG.declared_loop_tiers(inputs.policy_roots))
    except Exception:  # noqa: BLE001 — unresolvable ladder: no tier name, never a guessed one
        return None
    return sorted(loop)[0] if loop else None


def _cert_tier_name(*, inputs: LG.GradingInputs) -> str:
    """The DEEPEST cert tier the checkpoint adds above the loop tier, derived per target.

    ``"L3"`` only as the fallback for an unresolvable ladder, so an environment that cannot resolve its
    adapters behaves exactly as this code did before — it is a tier LABEL, never a target or sim name.
    """
    cert, _ = LG.cert_tiers_beyond_loop(
        context=inputs.context,
        policy_roots=inputs.policy_roots,
        public_roots=lambda: _public_capsules(inputs),
    )
    return sorted(cert)[-1] if cert else "L3"


def _pilot_capsule_dirs(*, inputs: LG.GradingInputs) -> dict:
    """``{capsule name: directory}`` for the pilot corpus, read straight off each ``capsule.yaml``.

    Needed because per-capsule promotion submits a SUBSET to the cert tier, and the grader selects
    capsules by root, not by name. ``capsules_root`` already accepts several roots and discovers
    ``capsule.yaml`` under each, so the eligible capsule DIRECTORIES are the roots — no staged copy and
    no symlink farm (``rglob`` does not descend into symlinked directories).
    """
    out: dict = {}
    _roots = _public_capsules(inputs)
    for _r in _roots if isinstance(_roots, (list, tuple)) else [_roots]:
        for cf in sorted(Path(_r).rglob("capsule.yaml")):
            try:
                doc = yaml.safe_load(cf.read_text()) or {}
            except Exception:  # noqa: BLE001 — unreadable capsule: fall back to its directory name
                doc = {}
            out[str((doc or {}).get("name") or cf.parent.name)] = cf.parent
    return out
