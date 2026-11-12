"""Capsule evaluation: result."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aet.core.run_paths import RunPaths

import datetime as _dt
from pathlib import Path

import yaml

from merlin.common.jsonio import readable_json
from merlin.development.backend.feedback import setup_timing as _host_timing
from merlin.development.targetgen import seeded_grade as _seeded_grade
from merlin.development.targetgen import tier_cache as _TC
from merlin.development.targetgen.contract import schemas
from merlin.development.targetgen.elf_lanes import PROGRAM_ARTIFACT_SUFFIX
from merlin.evaluation.capsule.identity import CONTRACT_VERSION
from merlin.evaluation.capsule.instructions import apply_executed_stream_rule
from merlin.evaluation.capsule.instructions import (
    apply_prohibited_instruction_rule as apply_prohibited_instruction_rule,
)
from merlin.evaluation.capsule.instructions import prohibited_instruction_report as prohibited_instruction_report
from merlin.evaluation.capsule.lanes import (
    _tier_field,
    credit_executed_required_lanes,
    graded_execution_identity,
    operator_path_lane_execution,
)


@_host_timing.measured("result_processing")
def _finalize_capsule_result(
    *,
    name: str,
    capsule: dict,
    status: str,
    failure: dict | None,
    tiers: dict,
    trace_check_res: dict,
    numeric: dict,
    required,
    no_oracle: bool,
    eff_target: str,
    paths,
    run_id: str,
    cfg,
    contract=None,
    executability: dict | None = None,
    epilogue_applicability: dict | None = None,
    declined: dict | None = None,
    extra: dict | None = None,
    submission: dict | None = None,
    decoded_trace: dict | None = None,
    execution_tiers=None,
    result_filename: str = "capsule_result.json",
) -> dict:
    """Turn a graded capsule's parts into its result row, and write it.

    Lifted VERBATIM out of :func:`run_capsule`, whose tail this was, for two reasons.

    The MODEL capsule path finalizes separately, so every rule enforced here -- a mandatory tier that
    did not run cannot yield a pass, a screened capsule is not a pass, a capsule with no runnable
    required tier does not pass on our own command-buffer interpretation alone -- applied to ISA
    capsules and silently did not apply to model ones.

    And the behaviour becomes testable directly, on the row this returns, instead of only through a
    full graded run. Pinning it to a source line inside `run_capsule` is testing where code lives
    rather than what it does, which is how that assertion broke when the block last moved.

    Pure with respect to grading: it decides `status`/`failure` from what it is handed and never
    re-runs an oracle. Its side effects are the artifacts the row implies -- capsule_result.json and
    the run manifest -- plus a self-validation warning."""
    # Imported here, as run_capsule did: this module's import of .provenance is deferred to call time.
    from merlin.development.targetgen.provenance import toolchain_shas

    # not_run_is_not_pass: a mandatory tier that did not pass closed (unavailable/skipped/absent) makes
    # the capsule incomplete — never a silent pass. A tier that is honestly N/A for this capsule's
    # datatype (``not_applicable``; the integer L0/L1 floor on a float datapath) is the ONE exception —
    # a legitimate skip like a dropped RoCC gate, not a missing oracle. An unavailable/absent RTL oracle
    # is never not_applicable, so it still fails closed here.
    # A DECLARED LANE CONTRACT THAT NOTHING EVALUATED IS NOT A SATISFIED ONE. `lane_report` runs only on
    # the whole-model path, which owns a routing plan and an execution record; the op tier ladder owns
    # neither, so a capsule of any other kind that declared `lanes` had its assertion silently ignored.
    # Measured: the corpus's only NEGATIVE lane assertion -- `forbid: [on_mesh]` on a host-only slice --
    # was never once evaluated, which made ACCELERATED_A_FORBIDDEN_LANE unreachable for it. A submission
    # could accelerate the family the target's manifest does not admit and still pass the capsule
    # written to catch exactly that.
    #
    # The operator path CAN prove the negative half, from the artifact it already produces: the linked
    # ELF is the complete set of instructions the program can execute, so a scan of its executable
    # sections for this target's DERIVED accelerator opcode settles "no work reached the accelerator"
    # whatever spelling the backend used -- including the `.word`/`.insn` datum the IR-level decoder
    # reads as silence. It settles nothing in the positive direction (an instruction present in a
    # binary is not one that ran). See `merlin.development.targetgen.elf_lanes`.
    #
    # It can prove the POSITIVE half too, but from execution rather than from the artifact, and only by
    # holding both halves of the claim at once: an oracle tier RAN this program on an instrument and
    # passed -- freshly, or on a certificate attributed to these exact bytes -- AND the decoded stream
    # it ran carries instructions of this target's own accelerator classes. See
    # `operator_path_lane_execution` for why neither half alone says anything.
    # Without both, nothing is credited and the capsule stays incomplete -- which is what the four
    # conv-epilogue capsules (`lanes.require: [on_mesh]`, the only ones asserting that a conv's
    # requant/bias/relu rides the mesh store path) hit on every grade before this existed.
    #
    # `incomplete`, never `fail`, for what is left unmeasured: nothing about the submission was
    # disproved. We simply did not measure the one thing this capsule exists to assert, and an
    # unmeasured assertion must not read as a pass.
    _decl_lanes = capsule.get("lanes") or {}
    if _decl_lanes.get("require") or _decl_lanes.get("forbid"):
        from merlin.development.targetgen import elf_lanes as _EL

        _lane_rep = (extra or {}).get("lane_report")
        # A report the CALLER supplied comes from a path that owns evidence this one cannot see (the
        # whole-model dispatch ledger), and its quality is judged where that evidence lives
        # (`capsule_grade`). This rule remains about the ABSENCE of an evaluation. Only the report built
        # HERE is held to `unjudged_lanes`, because here we know exactly how thin the evidence is.
        _from_caller = _lane_rep is not None
        if _lane_rep is None:
            try:
                _lane_rep = _EL.lane_report_from_elf(capsule, paths.generated / _EL.PACKAGE_ELF_NAME, target=eff_target)
            except Exception as _le:  # noqa: BLE001 -- our scanner's limit is not the submission's defect
                _lane_rep = None
                import sys as _sys

                _sys.stderr.write(f"WARNING: linked-ELF lane scan failed (unmeasured): {_le}\n")
            # A self-hosted target executes a descriptor-declared word stream rather than a linked ELF.
            # If the ELF path could not settle the negative assertion, use the exact ``.program`` bytes
            # the oracle recorded. Exactly one declaration is required; ambiguity stays unmeasured.
            if _EL.unjudged_lanes(_lane_rep, _decl_lanes):
                _programs = sorted(Path(paths.generated).glob(f"*{PROGRAM_ARTIFACT_SUFFIX}"))
                if len(_programs) == 1:
                    try:
                        _program_rep = _EL.lane_report_from_declared_program(capsule, _programs[0], target=eff_target)
                        if len(_EL.unjudged_lanes(_program_rep, _decl_lanes)) < len(
                            _EL.unjudged_lanes(_lane_rep, _decl_lanes)
                        ):
                            _lane_rep = _program_rep
                    except Exception as _pe:  # noqa: BLE001 -- scanner limits are not submission defects
                        import sys as _sys

                        _sys.stderr.write(f"WARNING: declared-program lane scan failed (unmeasured): {_pe}\n")
            # The positive half. Folded in AFTER the artifact scans so the negative half is whatever
            # they concluded, unweakened -- this only ever adds evidence for a REQUIRED lane, and only
            # on a rung `lane_report` itself counts as executed.
            _exec_tiers = execution_tiers if execution_tiers is not None else getattr(cfg, "oracle_tiers", ())
            # Recomputed from the bytes on disk, and ONLY when some tier record actually claims to have
            # been carried -- forming it hashes every executable, and a run in which nothing was carried
            # has nothing to attribute. `None` here refuses every carried verdict, which is the
            # fail-closed direction.
            _graded_identity = None
            if any(_tier_field(tiers.get(t), "carried") for t in _exec_tiers or ()):
                _graded_identity = graded_execution_identity(
                    target=eff_target, generated=paths.generated, shas=toolchain_shas(eff_target)
                )
            _lane_rep = credit_executed_required_lanes(
                _lane_rep,
                capsule,
                operator_path_lane_execution(
                    capsule,
                    tiers=tiers,
                    execution_tiers=_exec_tiers,
                    decoded_trace=decoded_trace,
                    execution_identity=_graded_identity,
                ),
            )
            if _lane_rep is not None:
                extra = {**(extra or {}), "lane_report": _lane_rep}
        _unjudged = [] if _from_caller else _EL.unjudged_lanes(_lane_rep, _decl_lanes)
        if _lane_rep is not None and _lane_rep.get("violated"):
            # DISPROVED, so this is a fail and not an `incomplete`: the emitted binary carries
            # instructions for a lane the capsule forbids. Same category name the whole-model path uses
            # for the same defect, so one grep finds both.
            _scan = _lane_rep.get("elf_scan") or _lane_rep.get("program_scan") or {}
            _artifact_kind = "linked executable" if _lane_rep.get("elf_scan") is not None else "oracle-declared program"
            _identity = _scan.get("opcode_source") or "the target's discovered mesh-compute ISA role"
            status = "fail"
            failure = {
                "plane": "lanes",
                "category": "ACCELERATED_A_FORBIDDEN_LANE",
                "detail": (
                    f"capsule forbids lane(s) {_lane_rep['violated']}, and the "
                    f"{_artifact_kind} carries {_scan.get('n_hits')} instruction(s) "
                    f"identified by {_identity}. This "
                    f"target's capability manifest does not admit this capsule's family, "
                    f"so the compiler must leave it on the host lane."
                ),
            }
        elif status == "pass" and (_lane_rep is None or _unjudged):
            status = "incomplete"
            if failure is None:
                failure = {
                    "plane": "lanes",
                    "category": "LANE_CONTRACT_NOT_EVALUATED",
                    "detail": (
                        f"capsule declares lanes {dict(_decl_lanes)}; lane(s) {_unjudged} "
                        f"were not measured on this path. A required lane needs evidence "
                        f"that something RAN, which only the whole-model path's dispatch "
                        f"ledger carries; a forbidden lane also accepts a complete linked-ELF "
                        f"or oracle-declared program scan. Grade it as kind=model, or provide "
                        f"one of those complete executable artifacts"
                    ),
                }

    # THE INSTRUCTION RULE, on every graded capsule's linked program (``applies_to:
    # phase1_capsule_elfs`` in the sealed policy). Before the screened/not-run rules below, so a capsule
    # that carries a prohibited instruction fails whatever else is true of it, and one whose program
    # could not be scanned never reads as a pass.
    _isa = prohibited_instruction_report(capsule, paths.generated, target=eff_target)
    if _isa is not None:
        extra = {**(extra or {}), "isa_prohibition": _isa}
        if no_oracle and _isa.get("clean") is None and status == "pass":
            # A structure-only smoke built no executable to scan: not gradeable, never a pass, and not
            # a "fix this" signal the submission could act on.
            status = "not_gradeable_no_oracle"
            failure = failure or {
                "plane": "not_gradeable_no_oracle",
                "category": "NOT_GRADEABLE_NO_ORACLE",
                "detail": (
                    "--no-oracle built no linked program, so the prohibited-role scan did not run: "
                    f"{_isa.get('detail')}"
                ),
            }
        else:
            status, failure = apply_prohibited_instruction_rule(status, failure, _isa)
    # The same rule on the executed command stream the trace gate decoded; UNKNOWN never passes.
    status, failure = apply_executed_stream_rule(status, failure, trace_check_res)

    if status == "pass" and any(getattr(t, "budget_deferred", False) for t in tiers.values()):
        # SCREENED, NOT CERTIFIED. Distinct from `incomplete` (something that should have run did not)
        # and from `pass` (it certified). The capsule cleared the cheap screen and the expensive tier was
        # deliberately not bought, because a capsule in the covering set already certifies the axes this
        # one exercises. Excluded from the pass/fail denominators and reported by name — the one thing it
        # must never do is read as a pass.
        _def = sorted(t for t, r in tiers.items() if getattr(r, "budget_deferred", False))
        status = "screened_only"
        failure = {
            "plane": "budget",
            "category": "SCREENED_NOT_CERTIFIED",
            "tier": _def[0] if _def else None,
            "detail": (
                f"passed the screen tier; certify tier(s) {', '.join(_def)} not purchased "
                f"(outside the derived covering set, certify budget exhausted). This is "
                f"NOT a verdict on this capsule."
            ),
        }
    if status == "pass":
        for tier in required:
            tr = tiers.get(tier)
            if tr is not None and getattr(tr, "not_applicable", False):
                continue
            if tr is None or tr.status in ("unavailable", "skipped"):
                if no_oracle:
                    # Explicit --no-oracle STRUCTURE-ONLY smoke: NO numeric oracle was requested this run,
                    # so a mandatory numeric tier that did not run is NOT a fixable failure — it is
                    # honestly NOT GRADEABLE. Record a DISTINCT status/plane so the agent is not handed a
                    # phantom `oracle_unavailable` "fix this" signal it can never satisfy. INTEGRITY: this
                    # is NEVER reported as a pass — the numeric verdict is simply withheld. The
                    # not_run_is_not_pass gate for GRADED (oracle-present) runs is the `else` branch below,
                    # unchanged; keep no_oracle False for every real grade.
                    status = "not_gradeable_no_oracle"
                    if failure is None:
                        failure = {
                            "plane": "not_gradeable_no_oracle",
                            "category": "NOT_GRADEABLE_NO_ORACLE",
                            "detail": f"numeric oracle unavailable this run (--no-oracle): "
                            f"mandatory tier {tier} not graded — structural tiers "
                            f"(L0/L1/trace) only",
                        }
                else:
                    status = "incomplete"
                    if failure is None:
                        # The tier's OWN reason is the only actionable half of this message, and it was
                        # being dropped. Measured on a self-hosted-ISA target: every capsule carried
                        # "program did not halt within N instructions" while the agent was shown only
                        # "mandatory tier L# did not run (unavailable)". Ten rounds of a fully conformant
                        # model went into feedback naming nothing it could fix; effort decayed each round.
                        # An oracle that is ABSENT and a program that RAN AND HUNG are different events;
                        # when the tier reported a reason, that reason leads.
                        _why = (getattr(tr, "reason", None) or "").strip() if tr else ""
                        _st = tr.status if tr else "absent"
                        failure = {
                            "plane": "oracle_unavailable",
                            "category": "NOT_RUN_IS_NOT_PASS",
                            "tier": tier,
                            "tier_status": _st,
                            "tier_reason": _why or None,
                            "detail": (
                                f"{_why} (mandatory tier {tier}, status {_st})"
                                if _why
                                else f"mandatory tier {tier} did not run ({_st})"
                            ),
                        }
                break

    # Fail-open guard (complements the loop above, which never fires for an EMPTY/all-N/A required set):
    # a capsule that declares no runnable oracle requirement must NOT grade 'pass' on the L0/L1 command-
    # buffer interpretation alone — that is our own engine, not an independent oracle. A real grade
    # (not --no-oracle) requires at least one runnable, non-N/A required tier to have certified it.
    if status == "pass" and not no_oracle:
        ran_required = [
            t
            for t in required
            if tiers.get(t) is not None and not getattr(tiers[t], "not_applicable", False) and tiers[t].status == "pass"
        ]
        if not ran_required:
            status = "incomplete"
            if failure is None:
                # When the phase capped a DECLARED tier away, name it: "this phase cannot reach the tier
                # your capsule requires" is a configuration fact the operator can fix, and it is a
                # different statement from "the oracle is missing". The materializer records the dropped
                # declared tiers on the capsule precisely so this message can be specific.
                _unreach = capsule.get("unreachable_required_oracle_tiers") or []
                _ceil = capsule.get("oracle_tier_ceiling")
                failure = {
                    "plane": "declared_tier_unreachable" if _unreach else "oracle_unavailable",
                    "category": "NOT_RUN_IS_NOT_PASS",
                    "unreachable_required_oracle_tiers": sorted(_unreach) or None,
                    "detail": (
                        f"this capsule declares required oracle tier(s) {sorted(_unreach)} that "
                        f"this phase cannot reach (ceiling {_ceil}); the tiers it can reach "
                        f"({sorted(required)}) are not applicable to this capsule's datapath, so "
                        f"nothing certified it. Refusing to substitute a tier the capsule never "
                        f"declared."
                        if _unreach
                        else f"no runnable required oracle tier certified this capsule "
                        f"(required={sorted(required)}) — refusing to pass on the L0/L1 "
                        f"command-buffer interpretation alone"
                    ),
                }

    # STAMP THE SUBMISSION ON EVERY TIER RECORD before it is serialised. Stated here, once, rather
    # than left for a reader to infer from the file's path: cycles are a property of the submission,
    # and a path heuristic that mis-keys two submissions of one capsule pools an 8.2x spread into a
    # single "latency". A record that already carries one keeps it.
    if submission:
        for _r in tiers.values():
            if getattr(_r, "submission", None) is None:
                _r.submission = dict(submission)
    # A MUST-REFUSE CAPSULE INVERTS THE GRADE, after every other rule has spoken: its only correct
    # answer is the backend's stated decline, and an emitted program -- right-looking or not -- fails.
    from merlin.development.targetgen import expected_refusal as _ER

    status, failure, _refusal_record = _ER.verdict(capsule, status, failure, declined)
    result = {
        "capsule": name,
        "kind": capsule.get("kind"),
        "label": capsule.get("label"),
        "status": status,
        "contract_version": CONTRACT_VERSION,
        "tiers": {t: r.to_dict() for t, r in tiers.items()},
        "trace_check": trace_check_res,
        "numeric": numeric,
        "failure": failure,
        "toolchain_shas": toolchain_shas(eff_target),
    }
    # WHICH TIERS THIS RUN ACTUALLY EXECUTED, and which it carried from an earlier certification of the
    # same executable on the same instrument. Emitted on EVERY result, including one where nothing was
    # carried: a saving and a lie differ only in whether the reuse is stated, and a reader must never
    # have to infer "freshly measured" from the absence of a block. See merlin.development.targetgen.tier_cache.
    result["tier_reuse"] = _TC.reuse_block(result["tiers"])
    if _seeded_grade.seed_run() and (_from := _seeded_grade.label(result, getattr(cfg, "oracle_tiers", None))):
        result["tier_reuse"]["carried_from"] = _from  # a seeded grade's pass, carried in full
    # Advisory RTL-executability smoke (never a gate): record it as its own field when one ran, so a
    # reader sees the RTL-legality backstop verdict without it ever touching the pass/fail status.
    if executability:
        result["executability"] = executability
    # Whether the target's readout applied every epilogue stage the program declared. Recorded even
    # when the answer is "not checked": an absent key reads as "this axis does not apply", which is
    # the same shape as the defect -- a declared activation that silently never happened.
    if epilogue_applicability:
        result["epilogue_applicability"] = epilogue_applicability
    # The refusal rides the result BY NAME AND SHAPE, so the round feedback can quote what was declined
    # rather than reporting a numeric mismatch on a program that was never emitted.
    if declined:
        result["declined"] = declined
    if _refusal_record is not None:
        result["expected_refusal"] = _refusal_record
    # Caller-supplied evidence (the whole-model path attaches its routing plan and mesh counters).
    # Merged AFTER status and failure are decided, and with setdefault so it can never overwrite an
    # authoritative field: a routing plan is something a reader interprets, never an input to the
    # verdict it is reported alongside.
    for _k, _v in (extra or {}).items():
        result.setdefault(_k, _v)
    if result_filename not in ("capsule_result.json", ".operator_child_result.json"):
        raise ValueError("capsule result publication requires its original fixed filename")
    (paths.run_path / result_filename).write_text(readable_json(result), encoding="utf-8")
    _write_run_manifest(
        paths,
        run_id,
        name,
        status,
        tiers,
        capsule,
        target=cfg.target,
        suite=cfg.suite,
        filename=".operator_child_manifest.yaml" if result_filename != "capsule_result.json" else "run_manifest.yaml",
    )
    try:
        schemas.validate(result, "capsule_result", contract=contract)
    except schemas.ContractViolation as e:
        import sys

        sys.stderr.write(f"WARNING: capsule_result self-validation failed: {e}\n")
    return result


def _write_run_manifest(
    paths: RunPaths,
    run_id: str,
    name: str,
    status: str,
    tiers: dict,
    capsule: dict,
    *,
    target: str,
    suite: str | None = None,
    filename: str = "run_manifest.yaml",
) -> None:
    manifest = {
        "schema_version": "1.0",
        "project": "merlin",
        "suite": suite or f"{target}-capsule-bench",
        "method": run_id,
        "run_id": run_id,
        "target": target,
        "benchmark": name,
        "status": status,
        "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "codegen_backend": "oot_package",
        "metadata": {
            "kind": capsule.get("kind"),
            "label": capsule.get("label"),
            "required_oracle_tiers": capsule.get("required_oracle_tiers", []),
            "tier_status": {t: r.status for t, r in tiers.items()},
            "cycles": {t: r.cycles for t, r in tiers.items() if r.cycles is not None},
        },
    }
    if filename not in ("run_manifest.yaml", ".operator_child_manifest.yaml"):
        raise ValueError("run manifest publication requires its original fixed filename")
    (paths.run_path / filename).write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
