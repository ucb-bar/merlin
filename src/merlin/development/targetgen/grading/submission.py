"""Grade a submitted mlir_oot_target_backend package through the capsule_bench_v0 fidelity suite.

This turns capsule_bench from a self-test into a reusable GRADER: given any contract-satisfying
package directory, run the public/dev capsule suite (and, post-freeze, the hidden suite) through the
shared command-buffer/reference/oracle ladder, run the integrity scan, decode + check the RoCC trace,
and emit a single machine-readable ``score_capsule.json``.

Run this OUTSIDE the agent sandbox (it needs spike/verilator + the hidden capsules). The package is
still only invoked via its 4 CLI entrypoints (subprocess) — never imported.

Usage:
    python -m merlin.development.targetgen.grading.model_checks --package <pkg_dir> --target <target>         --runs-root <out> [--hidden] [--labels public,dev] [--score <out.json>] [--no-oracle]

Do NOT pass ``--capsules experiments/corpora``: that is the common parent of every target's corpus
and grades the package against all of them. Omit the flag and the target's own graded roots are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from merlin.analysis.optimization.candidates import gate_phase as _GATE_PHASE
from merlin.analysis.optimization.candidates import lowering_coverage as _LOWERING_COVERAGE
from merlin.analysis.perf.contract import cost_plane as _COST_PLANE
from merlin.analysis.perf.measurement import comparand as _comparand
from merlin.development.targetgen.corpus import capsule_io as capsule_capsule_common
from merlin.development.targetgen.corpus.capsule_io import NOT_MEASURED_STATUSES
from merlin.development.targetgen.corpus.capsule_io import tier_field as _tier_field
from merlin.development.targetgen.corpus.capsule_io import tier_status as _tier_status
from merlin.development.targetgen.grading import coverage_report as CV
from merlin.development.targetgen.grading import elf_lanes as capsule_elf_lanes
from merlin.development.targetgen.grading.elf_lanes import native_host_execution
from merlin.development.targetgen.grading.perf_ledger import _activity_source
from merlin.development.targetgen.grading.perf_ledger import emit_perf_ledger as _emit_perf_ledger
from merlin.development.targetgen.grading.trace_check import (
    requirements_complete as _trace_requirements_complete,
)
from merlin.development.targetgen.oracles import tier_policy as _TIER_INTEGRITY
from merlin.development.targetgen.packaging.loading import INFRASTRUCTURE_PLANE as OOT_INFRASTRUCTURE_PLANE
from merlin.development.targetgen.packaging.loading import (
    CertFailure,
    build_package,
    integrity_scan,
    load_package,
)
from merlin.evaluation.capsule import eligibility as capsule_eligibility
from merlin.evaluation.capsule import runner as CR
from merlin.evaluation.capsule.evidence import certificates as capsule_adapters
from merlin.evaluation.capsule.evidence import tiers as capsule_model_tiers
from merlin.target.registry.descriptor_loading import source_experiment_env

# ---- capsule_grade -----------------------------------------------------------------------------


def grade(
    package_dir: str | Path,
    *,
    capsules_root: str | Path,
    runs_root: str | Path,
    labels: set[str] | None = None,
    contract: str | Path | None = None,
    oracle_adapters: dict | None = None,
    timeout: int = 900,
    max_workers: int = 0,
    target: str,
    no_oracle: bool = False,
    capability_admission: bool = False,
    target_experiment=None,
    additional_forbidden: tuple[str, ...] = (),
    model_snapshot_root: str | Path | None = None,
    on_capsule_complete=None,
    on_cohort_ready=None,
    model_gate_continuations=(),
    model_gate_prerequisites=None,
    functional_hidden_tier: str | None = None,
    functional_public_tier: str | None = None,
    defer_pending=None,
    defer_running=True,
) -> dict:
    """Run the capsule suite over a submitted package; return a score dict (also schema-checkable).

    ``max_workers`` fans the per-capsule oracle runs out in parallel (verilator/VCS/cyclotron instances);
    ``<= 0`` (the default) derives a host-scaled count via :func:`default_grade_workers` so grading a
    multi-capsule suite never serializes the RTL sims. Any positive value is honored as an explicit cap.

    ``no_oracle`` marks an EXPLICIT structure-only smoke (typically paired with ``oracle_adapters={}``):
    a capsule that clears the structural tiers but whose mandatory numeric tier was deliberately not run
    is recorded ``not_gradeable_no_oracle`` (a withheld numeric verdict), NOT the fixable
    ``oracle_unavailable`` plane — so the report is honest and never claims a numeric pass. Graded runs
    (``no_oracle=False``) keep the ``not_run_is_not_pass`` behavior byte-for-byte."""
    from merlin.development.targetgen.grading.enforcement import (
        _bounded_grade_workers,
        _candidate_score_view,
        _qualified_model_check,
        _score_numeric_status,
        cycles_by_tier,
        default_grade_workers,
        enforce_model_execution_check,
    )
    from merlin.development.targetgen.grading.model_checks import _materialized_cohort_admission, _name_set_sha256

    labels = labels or {"public", "dev"}
    if functional_public_tier is not None and (
        functional_public_tier != "L2"
        or functional_hidden_tier is not None
        or not labels <= {"public", "dev"}
        or no_oracle
        or set(oracle_adapters or {}) != {"L2"}
        or getattr((oracle_adapters or {}).get("L2"), "_merlin_simulator_engine", None) != "spike"
    ):
        raise ValueError("functional public grading requires public/dev-only exact Spike L2 evaluation")
    if functional_hidden_tier is not None and (
        functional_hidden_tier != "L2"
        or labels != {"hidden"}
        or not capability_admission
        or no_oracle
        or set(oracle_adapters or {}) != {"L2"}
    ):
        raise ValueError("functional hidden grading requires hidden-only admitted L2 oracle evaluation")
    # Source the target's own tooling paths before resolving any adapter, exactly as the harness does
    # for an arm. Without this a grade run outside the harness can lose a target's certifying-tier sim
    # and report the whole suite `incomplete` -- fail-closed, but describing the environment rather than
    # the submission. Process env wins, so an exported var is untouched.
    _sourced = source_experiment_env(target)
    if _sourced:
        print(
            f"  sourced {len(_sourced)} tooling path(s) from the target's experiment.env: "
            f"{', '.join(sorted(_sourced))}",
            flush=True,
        )
    # absolute before any threading, for the same reason run_suite does it: this function also reads the
    # per-capsule traces back out of this root, and a root that moved under a sibling thread's chdir
    # silently yields an empty coverage dict rather than an error.
    runs_root = str(Path(runs_root).resolve())
    pkg_dir = Path(package_dir).resolve()
    if contract is not None and Path(contract).exists():
        contract = str(Path(contract).resolve())
    capsules_root = (
        [str(Path(r).resolve()) for r in capsules_root]
        if isinstance(capsules_root, (list, tuple))
        else str(Path(capsules_root).resolve())
    )
    # HOLD the staged cohort for the duration of this grade. `.resolve()` above is deliberate (a root that
    # moved under a sibling thread's chdir yields an empty coverage dict instead of an error) but it also
    # turns the per-target cohort SYMLINK into one concrete `.<target>.build.<pid>.<hex>` path that is then
    # read for the whole grade -- 10-20 min functional, ~30-40 min per capsule on the cert tier. The
    # materializer's collector used to drop any build older than 15 minutes, and it dropped one a live
    # grade was mid-way through: 31 of 33 capsules were then recorded as structurally invalid SUBMISSIONS.
    # The lease makes "still being read" a fact the collector can check rather than infer from age. It is
    # advisory and best-effort -- if it cannot be taken, the grade proceeds exactly as before.
    try:
        from merlin.development.targetgen.corpus.materialization import pin_cohort_builds

        pin_cohort_builds(*(capsules_root if isinstance(capsules_root, list) else [capsules_root]))
    except Exception:  # noqa: BLE001 -- pinning is an optimisation for the collector, never a gate
        pass

    score: dict = {
        "task": f"{target}-mlir-oot-capsule",
        "package": str(pkg_dir),
        "integrity_exempt": None,
        "integrity_status": None,
        "labels_graded": sorted(labels),
        "functional_pass": 0,
        "n_capsules": 0,
        "n_passed": 0,
        "public_passed": None,
        "hidden_passed": None,
        "headline": None,
        "per_capsule": [],
        "tier_reached": {},
        "first_failure_planes": {},
        "numeric_all_exact": None,
        "trace_all_pass": None,
        "model_execution_all_pass": None,
        "structural_evidence_all_pass": None,
        "structural_evidence_scope": None,
        "cycles_diagnostic": {},
        "cycles_provenance": {},
        "highest_tier": None,
        "timing_diagnostic": {},
        "timing_rollup": {},
    }

    # Preparation leases end before build/callback/child owners select their own CPU.
    from merlin.common.process.cpu_pool import shared_compute_pool

    # K0/K1: load + integrity scan + build (fail-closed, recorded honestly)
    package_precheck_stage = "load_package"
    try:
        with shared_compute_pool(timeout_s=timeout):
            pkg = load_package(pkg_dir, contract=contract)
            score["integrity_exempt"] = pkg.integrity_exempt
            package_precheck_stage = "integrity_scan"
            integrity_scan(pkg, **({"additional_forbidden": additional_forbidden} if additional_forbidden else {}))
            score["integrity_status"] = "clean" if not pkg.integrity_exempt else "exempt"
        package_precheck_stage = None  # Arbitrary build execution is NOT a shared host-check refusal.
        build_package(pkg)
        # A manifest can land before its source entrypoint during authoring.
        # Check once, after a declared build has had the chance to produce it,
        # rather than issuing a doomed original-process request per capsule.
        if not pkg.tool.is_file():
            score["package_readiness"] = {"status": "not_ready", "reason": "declared_tool_missing"}
            raise CertFailure("build", "elaboration_error", f"declared tool is not a file: {pkg.tool}")
    except CertFailure as cf:
        score["integrity_status"] = f"FAIL[{cf.plane}]: {cf.detail[:200]}"
        score["functional_pass"] = 0
        score["failure"] = {"plane": cf.plane, "category": str(cf.category), "detail": cf.detail}
        if package_precheck_stage is not None:
            score["package_precheck_refusal"] = {
                "schema": "merlin.package_precheck_refusal.v1",
                "scope": "selected_suite",
                "stage": package_precheck_stage,
                "candidate_execution_started": False,
                "suppression": "native_pre_fanout_package_gate",
                "reuse_authority": False,
            }
        return score

    with shared_compute_pool(timeout_s=timeout):
        source_caps = capsule_capsule_common.discover_capsules(capsules_root, labels=labels, contract=contract)
    caps = source_caps
    admission_policy = "all_discovered"
    capability_excluded: list[dict] = []
    if capability_admission:
        # The hidden source pool is sealed, so its target-ineligible members cannot be named in the
        # public descriptor as the public exclusions are.  Derive its formal cohort here, outside the
        # agent sandbox and after freeze, using the same fail-open hardware predicate run_suite uses:
        # only an operand dtype absent from EVERY frozen target capability may be excluded.  Models stay
        # admitted because their host/accelerator partition can transform the device-facing dtype.
        op_caps = [c for c in source_caps if c.get("kind") != "model"]
        _admitted_ops, capability_excluded = capsule_eligibility._split_ineligible(op_caps, target)
        excluded_names = {str(r.get("capsule")) for r in capability_excluded}
        caps = [c for c in source_caps if str(c.get("name")) not in excluded_names]
        admission_policy = "frozen_target_capability_operand_dtype"
    materialized_admission = (
        None if capability_admission else _materialized_cohort_admission(capsules_root, source_caps)
    )
    if materialized_admission is not None:
        # Public cohort admission is a decision made under exact descriptor bytes.  Resolve through the
        # active MERLIN_TARGET_EXPERIMENT-aware path unless the caller already holds the frozen parsed
        # experiment, then require the materializer record to name those same bytes.  Hidden grading has
        # no public admission record and keeps its name-secrecy-preserving capability boundary below.
        experiment = target_experiment
        if experiment is None:
            from merlin.target.registry.descriptor_loading import descriptor_path, load_target_experiment

            descriptor = descriptor_path(target)
            if not descriptor.is_file():
                raise ValueError(f"target {target!r} has no experiment descriptor at {descriptor}")
            experiment = load_target_experiment(descriptor)
        if experiment.target != target:
            raise ValueError(f"experiment descriptor names {experiment.target!r}, not grade target {target!r}")
        if materialized_admission.get("descriptor_sha256") != experiment.descriptor_sha256:
            raise ValueError(
                "materialized cohort descriptor digest does not match the target descriptor used for grade"
            )
    if materialized_admission is not None:
        score["cohort_admission"] = materialized_admission
    else:
        excluded_names_sorted = sorted(str(r.get("capsule")) for r in capability_excluded)
        score["cohort_admission"] = {
            "version": 1,
            "policy": admission_policy,
            "n_source_capsules": len(source_caps),
            "n_admitted_capsules": len(caps),
            "n_capability_excluded": len(capability_excluded),
            "n_resource_excluded": 0,
            # Seal the exact boundary without publishing held-out capsule names into a reusable report.
            "excluded_name_set_sha256": _name_set_sha256(excluded_names_sorted),
            "admitted_name_set_sha256": _name_set_sha256(c.get("name") for c in caps),
        }
    workers = max_workers if max_workers and max_workers > 0 else default_grade_workers(len(caps))
    if functional_hidden_tier is not None or functional_public_tier is not None:
        from merlin.development.targetgen.corpus.materialization import _cap_required, _cap_tiers
        from merlin.development.targetgen.grading.boundary import expects_refusal

        selected = []
        for capsule in caps:
            if expects_refusal(capsule):
                selected.append(capsule)  # A declared refusal has no invented execution obligation.
                continue
            required = capsule.get("required_oracle_tiers")
            if (
                type(required) is not list
                or "L2" not in required
                or any(tier not in _cap_tiers("L5") for tier in required)
            ):
                raise ValueError("functional capsule has no declared L2 oracle obligation")
            kept, _ = _cap_required(required, set(_cap_tiers("L2")))
            selected.append({**capsule, "required_oracle_tiers": kept, "oracle_tier_ceiling": "L2"})
        caps = selected
    workers = _bounded_grade_workers(workers, oracle_adapters)
    if on_cohort_ready is not None:
        on_cohort_ready(len(caps))  # Actual selected roster, never completed/measured rows.
    import time as _time

    def _enforce_and_persist_models(results: list[dict]) -> None:
        # This callback executes before run_suite releases its model source
        # snapshot. Trusted callers additionally retain that closure under an
        # explicit host-private root for later independent pin inspection.
        _caps_by_name = {str(cap.get("name")): cap for cap in caps}
        for result in results:
            if result.get("kind") != "model" or result.get("status") == "gated":
                continue
            # Reopen under the same trusted adapter selection as the model
            # child. Candidate result records cannot choose their own codec.
            readback_policy = capsule_adapters._adapter_readback_policy(oracle_adapters or {})
            enforce_model_execution_check(
                result,
                _caps_by_name.get(str(result.get("capsule"))),
                target=target,
                **({"readback_policy": readback_policy} if readback_policy is not None else {}),
            )
            # The runner persisted the model result before this suite-level cross-record proof existed.
            # Keep the durable row in sync because QA/selfcheck/checkpoint consumers deliberately re-glob
            # it rather than trusting this function's in-memory score.
            result_path = (
                Path(runs_root)
                / "runs"
                / capsule_eligibility.suite_for(target)
                / str(result.get("capsule"))
                / "capsule_result.json"
            )
            if result_path.is_file():
                result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    _suite_t0 = _time.perf_counter()
    _model_audit_ran = False

    def _post_suite(results: list[dict]) -> None:
        nonlocal _model_audit_ran
        _enforce_and_persist_models(results)
        _model_audit_ran = True

    def _completed(result: dict) -> None:
        # Completion is serialized by the scheduler, after the durable result was
        # written and while the frozen model source scope is still owned here.
        _enforce_and_persist_models([result])
        if on_capsule_complete is not None:
            on_capsule_complete(result)

    # A per-capsule budget below what this run already measured a capsule needs can only ever leave
    # that capsule cert_not_measured. Raise it from the measurement, bounded and recorded.
    from merlin.development.targetgen.grading.scheduling import selected_history
    from merlin.development.targetgen.oracles.schedule import measured_capsule_budget, timed_out

    _budget = measured_capsule_budget(caps, selected_history(), timeout=timeout)
    if _budget["raised"]:
        print(f"  per-capsule budget: {_budget['basis']}", flush=True)
    timeout = _budget["budget_s"]
    results = CR.run_suite(
        caps,
        pkg_dir,
        runs_root=runs_root,
        contract=contract,
        oracle_adapters=oracle_adapters,
        timeout=timeout,
        max_workers=workers,
        target=target,
        no_oracle=no_oracle,
        model_snapshot_root=model_snapshot_root,
        post_suite=_post_suite,
        model_gate_continuations=model_gate_continuations,
        model_gate_prerequisites=model_gate_prerequisites,
        **({"defer_pending": defer_pending, "defer_running": defer_running} if defer_pending is not None else {}),
        **({"on_capsule_complete": _completed} if on_capsule_complete is not None else {}),
    )
    _suite_wall = _time.perf_counter() - _suite_t0
    if not _model_audit_ran:
        # Synthetic graders may replace run_suite with a result fixture. Keep
        # the same enforcement on that path; production run_suite always calls
        # the hook while the frozen model sources are still available.
        _enforce_and_persist_models(results)

    # collect decoded traces for coverage — read from the TARGET's own suite dir (run_capsule writes
    # under cfg.suite, e.g. <target>-capsule-bench), not one target's SUITE literal (which left every
    # other target's coverage dict silently empty; same root cause as the self-check n_capsules:0 blind loop).
    traces: dict[str, dict] = {}
    rr = Path(runs_root) / "runs" / capsule_eligibility.suite_for(target)
    for cap in caps:
        if cap.get("kind") == "model":
            # generated/ is the separate legacy host-graph path, not the
            # submitted whole-program ELF audited under candidate_native/.
            continue
        tp = rr / cap["name"] / "generated" / "instruction_trace.json"
        if tp.exists():
            try:
                traces[cap["name"]] = json.loads(tp.read_text())
            except Exception:
                pass

    # A PASS THAT DOES THE UNIT'S WORK ON THE HOST IS NOT A PASS. Applied here, before anything is
    # counted, from the artifacts each run already left under `rr`; see `apply_host_compute_veto`.
    from merlin.development.targetgen.conformance import offload_census as _offload_census

    score["host_compute_vetoed"] = sorted(
        str(row.get("program"))
        for row in _offload_census.apply_host_compute_veto(results, rr, capsules=caps, target=target)
    )

    # A NUMERICALLY PERFECT PASS THAT ASKED THE UNIT FOR ONLY THE CONTRACTION. The veto above catches
    # arithmetic the host ran inside an accelerator group; this catches the readout the accelerator
    # was derived to apply and nothing asked it to -- the bias, the requantization, the activation and
    # the pool that are most of a model's wall clock. Applied here for the same reason, before
    # anything is counted, and PHASED: its phase is declared in `src/merlin/resources/contracts/gate_phases.yaml`,
    # so at `report` it records both numbers and blocks nothing.
    from merlin.development.targetgen.grading import boundary as _store_path

    score["epilogue_store_path"] = _store_path.score_rows(_store_path.apply_gate(results, caps, rr, target=target))

    # Everything below is a score projection. The historical model runner can
    # still leave passing host-graph tiers, cycles and timing on a failed native
    # candidate row; none describe the submitted ELF. Keep the durable legacy
    # diagnostic, but never roll its evidence into candidate-native claims.
    results = [_candidate_score_view(row) for row in results]

    # A capsule withheld as outside the target's declared capability is in NEITHER the numerator nor the
    # denominator. Counting it as a failure is what made all_pass unreachable and disabled the loop's
    # early exit; counting it as a pass would be a phantom certification. It stays in `results` so the
    # skip is auditable, and its count is reported separately.
    # A capsule DEFERRED by its own gate (a whole-model capstone waiting on the op suite) is likewise in
    # neither bucket. It never ran a tier, so scoring it as a failure is not a measurement of the
    # submission -- and because the loop's ONLY early exit is a genuine all_pass, a permanently-deferred
    # capsule makes all_pass unreachable and forces every run to buy its entire round budget. Measured:
    # a 28-capsule grade of {pass 14, fail 12, not_graded 1, gated 2} could never reach all_pass no
    # matter what the agent did. The gate fraction it is waiting on is reported separately, as OP
    # COVERAGE -- it is not a verdict on the model.
    ungraded = [r for r in results if r.get("status") == "not_graded"]
    deferred = [r for r in results if r.get("status") == "gated"]
    screened = [r for r in results if r.get("status") == "screened_only"]
    graded = [r for r in results if r.get("status") not in NOT_MEASURED_STATUSES]
    from merlin.development.targetgen.grading import boundary as _ER

    _declared_refusals = {str(cap.get("name")): cap for cap in caps if _ER.expects_refusal(cap)}
    _validated_refusals = sum(
        sum(row.get("capsule") == name for row in results) == 1
        and any(_ER.validated_result(cap, row) for row in results if row.get("capsule") == name)
        for name, cap in _declared_refusals.items()
    )
    if _declared_refusals:
        score["expected_refusal_scope"] = {
            "binding": "original_capsule_declaration_and_canonical_result_v1",
            "n_admitted_capsules": len(caps),
            "n_declared_refusals": len(_declared_refusals),
            "n_validated_refusals": _validated_refusals,
            "n_executable_capsules": len(caps) - len(_declared_refusals),
        }
    _execution_rows = [row for row in graded if str(row.get("capsule")) not in _declared_refusals]
    n_pass = sum(1 for r in graded if r["status"] == "pass")
    score["n_capsules"] = len(graded)
    score["n_passed"] = n_pass
    score["n_not_graded_ineligible"] = len(ungraded)
    score["n_gated_deferred"] = len(deferred)
    if ungraded:
        score["not_graded_ineligible"] = sorted(r.get("capsule") for r in ungraded)
    if deferred:
        score["gated_deferred"] = sorted(r.get("capsule") for r in deferred)
    # EVERY not-measured row, BY NAME AND STATUS, in one place. The per-status lists above and below
    # (`not_graded_ineligible`, `gated_deferred`, `budget_exhausted`, `screened_only`) each name their
    # own bucket, so the union has to be reassembled by anyone who wants "what is missing from this
    # denominator" -- and `infrastructure_fault` only ever named a TRUNCATED first twelve. That
    # reassembly is what a CROSS-RUN comparison needs, and re-deriving it from `per_capsule` puts a
    # second copy of the exclusion rule in the reader.
    #
    # Why it matters: `n_capsules` above is the MEASURED denominator (correct -- a deferred row is not
    # a verdict), so two runs that deferred different rows have ratios over different cohorts.
    # Measured on a 97-capsule gemmini batch: the runs whose op coverage opened the whole-model gate
    # RAN and FAILED two model capsules inside a denominator of 97 (93/97); the run whose coverage
    # left the gate shut had those same two DEFERRED out of its denominator (75/95) -- so the weakest
    # run was scored over a cohort with the two hardest rows removed. Naming the dropped rows here is
    # what lets an aggregator intersect the cohorts instead of dividing by whichever denominator it
    # happened to read. It does NOT change `n_capsules`/`n_passed`: those stay this run's own
    # measured figures.
    _unmeasured = [r for r in results if r.get("status") in NOT_MEASURED_STATUSES]
    score["n_not_measured"] = len(_unmeasured)
    if _unmeasured:
        score["not_measured"] = sorted(str(r.get("capsule")) for r in _unmeasured)
        score["not_measured_status"] = {
            str(r.get("capsule")): str(r.get("status"))
            for r in sorted(_unmeasured, key=lambda r: str(r.get("capsule")))
        }
        score["not_measured_note"] = (
            "these capsules produced NO verdict, so they are in neither n_passed nor n_capsules. "
            "That is right for this run in isolation and WRONG to divide by when comparing runs: "
            "another run that reached them carries them in ITS denominator. Compare runs on the "
            "intersection of what both measured, and quote these names beside the ratio."
        )
    # STOPPED BY ITS OWN CLOCK, listed by name. A whole-model capsule under a wall-clock ceiling that
    # ran out produced no verdict: it is not a failure of the submission (nothing was measured) and
    # certainly not a pass. Excluded like the other unmeasured states, and reported separately so a
    # headline can never quietly rest on a capstone that never finished.
    _budget_out = [r for r in results if r.get("status") == "budget_exhausted"]
    score["n_budget_exhausted"] = len(_budget_out)
    if _budget_out:
        score["budget_exhausted"] = sorted(r.get("capsule") for r in _budget_out)
        score["budget_exhausted_note"] = (
            "these capsules STARTED and their wall-clock ceiling (MERLIN_MODEL_BUDGET_S) expired "
            "before they finished. Neither numerator nor denominator: nothing about the submission "
            "was measured. Clear the ceiling to grade them -- an operator certification run has none."
        )
    # ATTEMPTED THE CERT TIER AND GOT NO VERDICT, listed by name. Distinct from `screened_only`, where
    # the tier was deliberately not bought: here it RAN (or was killed running) and produced nothing, so
    # the cheaper tiers that passed are a screen and the capsule is uncertified. Excluded from both the
    # numerator and the denominator for the same reason every other not-measured state is -- and named,
    # because a headline resting on rows whose cert tier never answered is only honest beside them.
    _cert_unmeasured = [r for r in results if r.get("status") == _TIER_INTEGRITY.CERT_NOT_MEASURED]
    score["n_cert_not_measured"] = len(_cert_unmeasured)
    if _cert_unmeasured:
        score["cert_not_measured"] = sorted(str(r.get("capsule")) for r in _cert_unmeasured)
        score["cert_not_measured_reason"] = {
            str(r.get("capsule")): (r.get("failure") or {}).get("tier_reason")
            for r in sorted(_cert_unmeasured, key=lambda r: str(r.get("capsule")))
        }
        score["cert_not_measured_note"] = (
            "these capsules passed every cheaper tier and their elaborated-RTL tier was ATTEMPTED and "
            "produced no verdict (no completion witness, no engine, or killed on its wall clock). A "
            "screen may eliminate; it may never certify -- so they are in neither n_passed nor "
            "n_capsules, and nothing here is a verdict on them. Each reason is quoted above."
        )
    # WHICH BUDGET STOPPED IT, said with the number. The tier reason quotes only the clipped tail of
    # the killed command, which named neither the budget nor the option that sets it.
    score["per_capsule_budget"] = _budget
    _stopped = timed_out(results, _budget)
    if _stopped:
        score["timed_out"] = _stopped
        for _name, _row in sorted(_stopped.items()):
            print(
                f"  [timeout] {_name} {_row['tier']}: stopped by the {_row['budget_s']} s per-capsule wall "
                f"budget (requested {_row['requested_s']} s); unmeasured, not failed -- {_row['remedy']}",
                flush=True,
            )
    # SCREENED BUT NOT CERTIFIED, listed by name. A headline of "14/14" over a suite where nine more
    # capsules were screened and never certified is only honest if the nine are visible next to it.
    score["n_screened_only"] = len(screened)
    if screened:
        score["screened_only"] = sorted(r.get("capsule") for r in screened)
        score["screened_only_note"] = (
            "these capsules PASSED the cheap screen tier and were deliberately not measured against the "
            "certifying tier: they fall outside the derived covering set and the certify budget was "
            "exhausted. They are in neither the numerator nor the denominator. The covering set "
            "guarantees every axis they exercise was certified by some capsule -- it does NOT certify "
            "these capsules."
        )
    # DECLINED, listed by name and shape. Unlike `screened_only` these ARE in the denominator: the
    # backend was asked and said no, so the capsule is uncertified and counts against the suite. What
    # changes is that it is no longer reported as wrong arithmetic. A run whose deficit is twelve
    # declines is a backend that does not cover the shape space; a run whose deficit is twelve numeric
    # mismatches is a backend whose math is wrong. Those need opposite repairs, and the score used to
    # spell them the same way.
    _declined = [r for r in graded if r.get("status") == "declined"]
    score["n_declined"] = len(_declined)
    if _declined:
        score["declined"] = sorted(
            ({"capsule": r.get("capsule"), **(r.get("declined") or {})} for r in _declined),
            key=lambda d: str(d.get("capsule")),
        )
        score["declined_note"] = (
            "the backend STATED it does not lower these capsules rather than emitting a program that "
            "writes nothing. They are in the denominator and are NOT passes. Read them as coverage "
            "gaps (a shape/op the backend never implemented), not as numeric defects."
        )
    # BOTH NUMBERS, ALWAYS. When the runtime drove a loop nest over a backend's declared single tile,
    # the resulting pass is evidence about the runtime AND the backend; the backend's own coverage is the
    # smaller number. Reporting only the larger one is precisely how "our runtime covered for them" gets
    # cited as "their compiler generalizes", so the two travel together or the guardrail is not real.
    _runtime_tiled = [
        r
        for r in graded
        if (
            ((r.get("contract_obligations") or {}).get("capacity_fit") or {}).get("tiled_by")
            == "declared_primitive_tile"
        )
    ]
    if _runtime_tiled:
        _names = sorted(str(r.get("capsule")) for r in _runtime_tiled)
        _unblocked = n_pass - sum(1 for r in _runtime_tiled if r.get("status") == "pass")
        score["backend_coverage"] = {
            "with_runtime_loop": f"{n_pass}/{len(graded)}",
            "unblocked": f"{_unblocked}/{len(graded)}",
            "runtime_tiled_capsules": _names,
            "note": (
                "`unblocked` is what the BACKEND lowered on its own. `with_runtime_loop` adds the "
                "capsules that only passed because the runtime drove the M/N/K loop over the single "
                "tile the backend declared. Quote `unblocked` for any claim about the backend "
                "generalizing over shape; `with_runtime_loop` is a statement about the pair."
            ),
        }
    score["functional_pass"] = int(n_pass == len(graded) and len(graded) > 0)
    # Structure-only smoke bookkeeping (honest, never a numeric pass): a capsule is structurally clean
    # when it did not FAIL a structural tier — status `pass` OR `not_gradeable_no_oracle` (numeric verdict
    # withheld under --no-oracle). `gradeable` says whether this run had a numeric oracle at all.
    n_not_gradeable = sum(1 for r in graded if r["status"] == "not_gradeable_no_oracle")
    # Fail-closed on an empty suite: if NO capsule matched the requested labels at this root, nothing was
    # graded. `all([])` is vacuously True, so numeric_all_exact / trace_all_pass would read as a phantom
    # pass and `gradeable` as True — the exact vacuous-pass trap that made a mis-rooted hidden phase
    # (n_capsules:0) look green. Report the boolean flags as null and gradeable False, never a pass.
    _empty = len(graded) == 0
    # A capsule is INCOMPLETE when a MANDATORY tier came back `unavailable`: the oracle never ran, so
    # this score says nothing about the submission for it. That is a third thing, distinct from both
    # neighbours it keeps being confused with — a FAIL means the submission was wrong, and
    # `not_gradeable_no_oracle` means --no-oracle deliberately withheld the verdict. Here the run
    # BELIEVED it was grading and the tool was simply missing.
    #
    # Measured: re-grading a frozen submission whose own run scored 33/36 produced `n_passed: 6` with 29
    # capsules incomplete ("no derived ISA encoding fact for target 'radiance'") — the six that "passed"
    # were the MX fixtures, which need no oracle. Nothing in the rollup said so: n_not_gradeable_no_oracle
    # stayed 0 and a reader would quote 6/36 as a capability collapse. It is an ENVIRONMENT gap, and the
    # scorecard has to say which of the two it is or the number is worse than no number.
    _incomplete = [r for r in graded if r.get("status") == "incomplete"]
    score["n_incomplete"] = len(_incomplete)
    if _incomplete:
        _why = sorted(
            {
                (r.get("failure") or {}).get("tier_reason") or (r.get("failure") or {}).get("detail") or "unknown"
                for r in _incomplete
            }
        )
        score["measurement_incomplete"] = {
            "n": len(_incomplete),
            "of": len(graded),
            "capsules": sorted(r["capsule"] for r in _incomplete)[:12],
            "reasons": _why[:4],
            "detail": (
                f"{len(_incomplete)} of {len(graded)} capsules had a MANDATORY tier come back "
                f"unavailable, so the oracle never ran on them. n_passed is NOT a capability "
                f"measurement for this run — fix the environment and re-grade before quoting it."
            ),
        }
    # THE HARNESS BROKE, NOT THE SUBMISSION. A capsule whose staged inputs were not on disk measured
    # nothing at all, so it is excluded from the pass/fail denominators (NOT_MEASURED_STATUSES) — but
    # exclusion ALONE would be the more dangerous of the two lies. Measured: 31 of 33 capsules lost their
    # staging dir to a sibling materialization's collector; excluding them without this block leaves
    # `2/2` and `all_pass` reachable, turning a total staging failure into a clean sweep. So the run is
    # declared NOT gradeable and the fact is reported by name, the same fail-closed posture an empty
    # suite and a missing oracle already get. A check that could not run must never read as success.
    _infra = [r for r in results if r.get("status") == "infrastructure_fault"]
    score["n_infrastructure_fault"] = len(_infra)
    if _infra:
        _paths = sorted({(r.get("failure") or {}).get("detail") or "unknown" for r in _infra})
        score["infrastructure_fault"] = {
            "n": len(_infra),
            "of": len(results),
            "plane": OOT_INFRASTRUCTURE_PLANE,
            "capsules": sorted(r["capsule"] for r in _infra)[:12],
            "reasons": _paths[:2],
            "detail": (
                f"{len(_infra)} of {len(results)} capsules could not be graded because their "
                f"STAGED INPUTS were missing — the capsule cohort was not materialized, or its "
                f"staging directory was removed while this grade was running. This is a HARNESS "
                f"fault: it is NOT a verdict on the submission, and n_passed/n_capsules is not a "
                f"measurement of it. Re-materialize the cohort and re-grade before quoting any "
                f"number from this run."
            ),
        }
    # `gradeable` means this run had a working numeric oracle. It did not, for those capsules, so it is
    # False even though an oracle was requested — the same fail-closed posture the empty suite gets.
    score["gradeable"] = (not no_oracle) and not _empty and not _incomplete and not _infra
    score["n_not_gradeable_no_oracle"] = n_not_gradeable
    score["n_structural_pass"] = n_pass + n_not_gradeable
    score["structural_pass"] = bool(not _empty and not _infra and (n_pass + n_not_gradeable) == len(graded))
    score["numeric_all_exact"] = None if _empty else all(_score_numeric_status(r) == "pass" for r in _execution_rows)
    from merlin.development.targetgen.execution.host import leaf_execution as _OH

    _caps_by_name = {str(cap.get("name")): cap for cap in caps}
    _host_rows = [r for r in _execution_rows if _OH.selected_host_leaf(_caps_by_name.get(r.get("capsule"), {}))]
    _host_names = {r.get("capsule") for r in _host_rows}
    _trace_rows = [r for r in _execution_rows if r.get("kind") != "model" and r.get("capsule") not in _host_names]
    _model_rows = [r for r in _execution_rows if r.get("kind") == "model"]
    score["trace_all_pass"] = (
        None if not _trace_rows else all(_trace_requirements_complete(r.get("trace_check")) for r in _trace_rows)
    )
    _trace_requirements = Counter(
        (r.get("trace_check") or {}).get("trace_requirement_status")
        or ({"pass": "pass", "fail": "failed"}.get((r.get("trace_check") or {}).get("status"), "unavailable_record"))
        for r in _trace_rows
    )
    score["trace_requirement_summary"] = dict(sorted(_trace_requirements.items()))
    score["model_execution_all_pass"] = (
        None
        if not _model_rows
        else all(r.get("model_execution_check", {}).get("status") == "pass" for r in _model_rows)
    )
    score["provided_host_execution_all_pass"] = (
        None
        if not _host_rows
        else all(_OH.qualified_host_result(_caps_by_name[r["capsule"]], r, target=target) for r in _host_rows)
    )
    score["structural_evidence_scope"] = {
        "n_instruction_trace_capsules": len(_trace_rows),
        "n_model_execution_capsules": len(_model_rows),
        "n_provided_host_capsules": len(_host_rows),
    }
    score["structural_evidence_all_pass"] = bool(
        not _empty
        and (not _trace_rows or score["trace_all_pass"] is True)
        and (not _model_rows or score["model_execution_all_pass"] is True)
        and (not _host_rows or score["provided_host_execution_all_pass"] is True)
    )
    if _empty:
        score["note"] = (
            "no capsules matched the requested labels at this root — nothing graded; "
            "flags are null (not a pass). Check the capsules root / labels."
        )

    pub = [r for r in graded if r.get("label") in ("public", "dev")]
    hid = [r for r in graded if r.get("label") == "hidden"]
    if pub:
        score["public_passed"] = f"{sum(1 for r in pub if r['status'] == 'pass')}/{len(pub)}"
    if hid:
        score["hidden_passed"] = f"{sum(1 for r in hid if r['status'] == 'pass')}/{len(hid)}"

    tiers = ["L0", "L1", "L2", "L3", "L4", "L5"]
    for t in tiers:
        # over GRADED only: a withheld capsule has no tiers, so counting it in the denominator would
        # make tier_reached never reach the total and silently force highest_tier to None.
        score["tier_reached"][t] = sum(1 for r in graded if _tier_status((r.get("tiers") or {}).get(t)) == "pass")
    for t in reversed(tiers):
        if score["tier_reached"][t] == len(graded) and len(graded) > 0:
            score["highest_tier"] = t
            break

    # WHAT KIND OF EVIDENCE EACH PASS RESTS ON, stated next to the count rather than left to be
    # reconstructed from `tier_reached`. A capsule can report `pass` while a hardware tier beside it
    # reports `fail` -- when that tier was advisory at the time -- and the flattering half is the half
    # that gets quoted. Measured: one gemmini submission shipped as "20/20" whose Verilator tier passed
    # exactly 1 of 20, next to three sibling submissions whose 20/20 was RTL-clean on all 20. Both are
    # "20/20"; they are not the same result, and nothing in the headline said so.
    #
    # RTL-ness is DERIVED, never a tier-name literal: each tier record carries `derived_from_rtl`, set
    # from the target's own `cfg.rtl_tiers`, so a target whose RTL tier is L4 (or L2) is described
    # correctly without this code knowing which one it is.
    # A model capsule records its tier as a bare string, which carries no `derived_from_rtl` flag, so
    # keying only on the flag counted a whole model that PASSED the RTL tier as cheap-tier-only. Fall
    # back to the target's own declared rtl_tiers, matched by tier NAME -- still derived (the manifest
    # says which tiers are RTL for this target), never a literal.
    from merlin.evaluation.capsule.evidence.tiers import _rtl_tiers_of

    _rtl_names = _rtl_tiers_of(target)

    # AN EXPLICIT DENIAL MUST WIN OVER THE NAME. `derived_from_rtl` has three states, and collapsing the
    # last two is what let a model count as RTL: True (the oracle says it ran RTL), False (the oracle says
    # it did NOT -- e.g. an arc cosim, an RTL-DERIVED model, landing on the tier named L3), and None (the
    # record is the bare-string form a model capsule writes, which states nothing). Only None may fall
    # back to the tier name. Reading `flag or name in _rtl_names` credited an oracle that had explicitly
    # denied being RTL, purely because its tier shared a name with another target's Verilator tier.
    def _is_rtl_pass(row, name, rec) -> bool:
        if _tier_status(rec) != "pass":
            return False
        flag = _tier_field(rec, "derived_from_rtl")
        if flag is None and _qualified_model_check(row) is not None:
            return False  # no tier-name fallback for a submitted whole-program candidate
        return bool(name in _rtl_names) if flag is None else bool(flag)

    def _tier_is_rtl(row, name, rec) -> bool:
        """Same three-state rule as :func:`_is_rtl_pass`, without the pass requirement."""
        flag = _tier_field(rec, "derived_from_rtl")
        if flag is None and _qualified_model_check(row) is not None:
            return False
        return bool(name in _rtl_names) if flag is None else bool(flag)

    _passed = [r for r in graded if r.get("status") == "pass"]
    if functional_hidden_tier is not None or functional_public_tier is not None:
        evidence_key = "functional_hidden_evidence" if functional_hidden_tier else "functional_public_evidence"
        score[evidence_key] = {
            "tier": "L2",
            "engine": "spike",
            "fidelity": "functional_model",
            "claim": "functional_corroboration",
            "n_admitted_capsules": len(caps),
            "n_validated_refusals": _validated_refusals,
            "n_spike_passed": sum(
                row.get("status") == "pass"
                and _tier_status((row.get("tiers") or {}).get("L2")) == "pass"
                and _tier_field((row.get("tiers") or {}).get("L2"), "engine") == "spike"
                and _tier_field((row.get("tiers") or {}).get("L2"), "fidelity") == "functional_model"
                and _tier_field((row.get("tiers") or {}).get("L2"), "derived_from_rtl") is False
                for row in results
            ),
            "n_higher_tier_measured": sum(
                _tier_status(value) in {"pass", "fail"}
                for row in results
                for tier, value in (row.get("tiers") or {}).items()
                if tier in {"L3", "L4", "L5"}
            ),
        }
    _rtl_backed = [r for r in _passed if any(_is_rtl_pass(r, n, t) for n, t in (r.get("tiers") or {}).items())]
    # A pass that cleared a tier the ORACLE ITSELF called a model (fidelity != elaborated_rtl) and no RTL
    # tier beside it. Named separately because "not RTL-backed" reads as "did not run on hardware-grade
    # evidence at all", which is not what an RTL-derived cosim pass is.
    _model_certified = [
        r
        for r in _passed
        if r not in _rtl_backed
        and any(_tier_status(t) == "pass" and _tier_field(t, "fidelity") for t in (r.get("tiers") or {}).values())
    ]
    score["pass_evidence"] = {
        "n_passed": len(_passed),
        "rtl_backed": len(_rtl_backed),
        "model_certified": len(_model_certified),
        "cheap_tier_only": len(_passed) - len(_rtl_backed),
        "rtl_tiers_seen": sorted(
            {n for r in graded for n, t in (r.get("tiers") or {}).items() if _tier_is_rtl(r, n, t)}
        ),
        "fidelity_seen": sorted(
            {f for r in graded for t in (r.get("tiers") or {}).values() if (f := _tier_field(t, "fidelity"))}
        ),
        "note": (
            "`rtl_backed` counts passes that cleared a tier the ORACLE reported as elaborated RTL "
            "(`derived_from_rtl`), falling back to the target's declared rtl_tiers only when the "
            "oracle states nothing. A tier NAME is not evidence: one target's L3 is Verilator and "
            "another's is an RTL-derived model, and counting by name credited the model. When "
            "`rtl_backed` is below `n_passed`, the headline score is NOT an RTL result."
        ),
    }

    # THE CITABLE FORM OF THE SCORE. `public_passed` stays a bare "20/20" because consumers parse it as
    # a fraction (agg_agentic_results._frac does int(total)), and a bare fraction is exactly what gets
    # quoted -- one gemmini submission travelled as "20/20" while its Verilator tier passed 1 of 20,
    # indistinguishable in the headline from three siblings that were RTL-clean on all 20. So the
    # qualification is built ONCE, here, next to the evidence that justifies it, and the renderers print
    # THIS rather than reassembling a bare fraction each time. Quote `headline`; parse `public_passed`.
    score["headline"] = _headline(score)

    # THE ARRAY THE COST FLOOR IS DIVIDED BY, derived once from this target's own RTL facts. Never a
    # literal: `silicon_facts` fails closed to None for a target whose facts are not derivable here,
    # and the cost plane then records an explicit UNKNOWN floor rather than dividing by a default
    # array that would produce a plausible, precise, wrong number.
    from merlin.verification.liveness import silicon_facts as _silicon_facts

    _sf = _silicon_facts(target)
    _array_rows, _array_cols = _sf.mesh_rows, _sf.mesh_cols

    # THE COST PLANE'S ENFORCEMENT CALL SITE. `assess` computes a verdict and returns it; `apply_gate`
    # computes the same verdict, attaches it to the row, and -- at a BLOCKING phase -- turns a DECIDED
    # failure into `status: fail` with a `failure` block, exactly as `epilogue_store_path.apply_gate`
    # does above. The loop below used to call `assess` directly and record the verdict in the score
    # entry, which is why `gate_phases.yaml` reading `cost_plane: fail` changed what the artifact SAID
    # and not which rows failed: nothing read `blocking`. That file's own comment names this move as
    # what makes the flip reach a submission's status.
    #
    # WHY IT IS SAFE TO MOVE NOW. The old comment here declined the flip because it would change which
    # rows sit in the denominator mid-experiment, and two arms of a live A/B must be scored by the same
    # rule. That constraint was real and is now spent: phase 1 and phase 2 are being regenerated, so
    # there is no live A/B to disturb, and a gate that never blocks is the defect the regeneration
    # exists to remove.
    #
    # Phase still decides. At `report` this attaches the verdict and fails nothing, so the call is
    # correct at either phase and the one-line `gate_phases.yaml` edit stays the whole control surface.
    _COST_PLANE.apply_gate(results, caps, array_rows=_array_rows, array_cols=_array_cols)

    _agg = {"build_s": 0.0, "sim_active_s": 0.0, "oracle_wait_s": 0.0}
    for r in results:
        _cyc_by_tier = cycles_by_tier(r.get("tiers"), ladder=tiers)
        if _cyc_by_tier:
            score["cycles_diagnostic"][r["capsule"]] = _cyc_by_tier
        # WHICH PROGRAM produced each of those cycle counts. A block-scaled capsule is graded on the
        # harness's own reference kernel rather than the submission, so a cycle number and a fixture's
        # cycle number are indistinguishable in `cycles_diagnostic` alone -- the shape of the recorded
        # run that reported 40/40 where 9 passes were the fixture. Attribution rides ALONGSIDE rather
        # than inside, so `cycles_diagnostic` keeps the integer shape its pinned test and the audit
        # renderer expect.
        # COMPARE LIKE WITH LIKE. The tier record states the submission it ran, RESOLVED
        # (`capsule_runner.submission_identity`), while `score["package"]` is whatever string the
        # caller passed. Comparing the two directly makes an attributed count read as OTHER_PROGRAM --
        # a false claim about a different program, strictly worse than UNATTRIBUTED. Prefer the tier
        # record's own resolved package and fall back to the score's only when it states none.
        _sub = (
            (r.get("submission") or {}).get("package") if isinstance(r.get("submission"), dict) else None
        ) or score.get("package")
        _prov = _comparand.cycles_provenance(r.get("tiers"), submission=_sub, ladder=tiers)
        if _prov:
            score.setdefault("cycles_provenance", {})[r["capsule"]] = _prov
        # active-vs-waiting timing: sum across every tier that actually ran an oracle for this capsule
        cap_tm = {"build_s": 0.0, "sim_active_s": 0.0, "oracle_wait_s": 0.0, "by_tier": {}}
        for t in ("L2", "L3", "L4", "L5"):
            tm = _tier_field((r.get("tiers") or {}).get(t), "timing")
            if tm:
                cap_tm["by_tier"][t] = tm
                for k in ("build_s", "sim_active_s", "oracle_wait_s"):
                    v = tm.get(k) or 0.0
                    cap_tm[k] += v
                    _agg[k] += v
        if cap_tm["by_tier"]:
            # ⚠️ THE SUM ACROSS TIERS IS NOT A SIMULATION COST, and only the sum was ever recorded.
            # `sim_active_s` here adds a functional oracle and a cycle-accurate one together, and they
            # differ by orders of magnitude on the same capsule: measured on PC00_k64, L2 (spike) took
            # 0.009s while L3 (verilator) took 698.2s. Anything fitting a cost model over the sum is
            # fitting whichever tiers happened to run, so a budget derived from it cannot say what a
            # CERTIFICATION costs -- which is the one question a capsule's size has to be sized against.
            # The per-tier breakdown was already computed here and discarded; keep it, and carry the
            # two flags that say what KIND of oracle produced each number so a consumer can select the
            # cycle-accurate one rather than assuming.
            entry_tm = {k: round(cap_tm[k], 3) for k in ("build_s", "sim_active_s", "oracle_wait_s")}
            per_tier = {}
            for _t, _tm in cap_tm["by_tier"].items():
                _rec = (r.get("tiers") or {}).get(_t) or {}
                per_tier[_t] = {
                    **{k: round(_tm.get(k) or 0.0, 3) for k in ("build_s", "sim_active_s", "oracle_wait_s")},
                    "derived_from_rtl": _tier_field(_rec, "derived_from_rtl"),
                    "cycle_accurate": _tier_field(_rec, "cycle_accurate"),
                    "evidence": _tier_field(_rec, "evidence"),
                    # The seconds in this block are the cost of ONE named engine. Carrying the name
                    # beside them is what lets cert_cost fit a budget per engine instead of over a
                    # mixture of two that differ by more than an order of magnitude.
                    "engine": _tier_field(_rec, "engine"),
                    "console_log": _tier_field(_rec, "console_log"),
                }
            entry_tm["by_tier"] = per_tier
            score["timing_diagnostic"][r["capsule"]] = entry_tm
        if r.get("failure"):
            p = r["failure"]["plane"]
            score["first_failure_planes"][p] = score["first_failure_planes"].get(p, 0) + 1
        entry = {
            "capsule": r["capsule"],
            "label": r.get("label"),
            "status": r["status"],
            "numeric": _score_numeric_status(r),
            "trace": r.get("trace_check", {}).get("status"),
            "tiers": {t: _tier_status((r.get("tiers") or {}).get(t)) for t in tiers if t in (r.get("tiers") or {})},
        }
        if isinstance(r.get("background_deferred"), str):
            entry["background_deferred"] = r["background_deferred"]
        _original_refusal = _declared_refusals.get(str(r.get("capsule")))
        if _original_refusal is not None and _ER.validated_result(_original_refusal, r):
            entry.update(
                validated_refusal=True,
                qualification="expected_refusal",
                native_executed=False,
                expected_refusal=r["expected_refusal"],
            )
        # WHY EACH TIER DID NOT PASS. The projection above renders a tier record as its bare status
        # STRING, and that projection is where the diagnosis died: the ladder writes a good sentence
        # onto every non-passing tier ("the console carries no completion witness", "injected operand
        # of dtype 'f8E4M3FN' has no decoder", "timed out after 1200 seconds"), the durable
        # capsule_result.json keeps it, and the score file -- the artifact that gets cited -- carried
        # `L3: "unavailable"` with `failure_detail: null` on 24 of 29 rows. An `unavailable` tier with
        # no reason is uninterpretable and indistinguishable from a tier nobody tried. Never null:
        # `tier_integrity.not_certified` synthesises an explicit UNKNOWN sentence when a record
        # states nothing, so "the tier said nothing" is itself reported rather than rendered as
        # silence. Ordered by the target's own ladder.
        # Omitted when every tier passed, so a clean row keeps its exact shape (a gate pins that: an op
        # capsule must not be padded with fields that do not apply to it).
        _why_not = _TIER_INTEGRITY.not_certified(r.get("tiers"), ladder=tiers)
        if _why_not:
            entry["tier_reasons"] = _why_not
        # WHETHER THE CERT TIER ACTUALLY CERTIFIED THIS ROW. SURFACED, NOT GATED -- and the distinction
        # is deliberate rather than timid. `tier_integrity.qualify` is the rule that a row whose
        # elaborated-RTL tier RAN and did not certify may not read `pass`; wiring it into the runner's
        # status changes which rows sit in the denominator (measured: 6 gemmini capsules per run, 24 of
        # 29 on one SIMT run), and two arms of a live A/B must be scored by the same rule. So the rule
        # runs here on every row and REPORTS, without moving `status`, until the status flip lands
        # together for every arm. A reader of this file can see today which passes are uncertified;
        # nothing silently counts them differently mid-experiment.
        _cert_verdict = _TIER_INTEGRITY.qualify(r)
        if _cert_verdict is not None:
            entry["cert_verdict"] = {
                "would_be_status": _cert_verdict["status"],
                **{
                    k: v
                    for k, v in _cert_verdict["failure"].items()
                    if k in ("plane", "category", "tier", "tier_status", "tier_reason")
                },
                "note": (
                    "this row is reported `pass` because a cheaper tier passed; its elaborated-RTL "
                    "tier did not certify it. A screen may eliminate, it may never certify."
                ),
            }
        # WHAT THE PROGRAM COST, on the timing axis the capsule's own acceptance block declares.
        # Nothing in this grade has ever consumed a cycle count: `cycles_diagnostic` records them and
        # no rule reads them, so a submission that emitted a program a hundred times slower than the
        # machine needs is graded exactly like one that did not. The plane runs on every row and
        # REPORTS -- same reasoning as `cert_verdict` above: turning it into a status would change
        # which rows sit in the denominator mid-experiment, and two arms of a live A/B must be scored
        # by the same rule. Its floor is derived from the array the target's own facts declare and
        # the work this capsule's own command buffer counted, so it needs no slack constant; it can
        # therefore refute a count that is below what the array must spend, and says explicitly that
        # it cannot decide "fast enough" without a measured slack nobody has.
        # Read the verdict `apply_gate` already attached to the row, rather than recomputing it. Two
        # calls to `assess` on one row could disagree the moment either argument is threaded
        # differently, and the row's own copy is the one that decided its status.
        _cost = r.get(_COST_PLANE.PLANE) or {}
        # Only for a row that is ON this axis. A capsule whose acceptance block names no timing rung
        # owes no cycle count, so a verdict saying "there is nothing here to bound" is not evidence
        # about it -- it is padding, and a gate pins that a row must not carry fields that do not
        # apply to it.
        if _cost.get("timing_tier"):
            entry["cost_plane"] = _cost
        # CARRY THE QUALIFIERS WITH THE VERDICT. The roll-up kept six fields, and every one of them
        # flatters: status, tier, numeric, trace. Everything that says what the pass RESTS ON --
        # which contract obligations someone else discharged on the backend's behalf, how many of the
        # model's own layers reached the accelerator versus fell back to the host, which declared tiers
        # never ran -- stayed behind in the per-capsule result, several directories down. The score file
        # is the artifact that gets cited, so a qualifier that does not travel with it does not exist.
        if r.get("status") == "gated":
            # WHY it was deferred. "gated" with no reason is uninterpretable in the artifact -- it reads
            # as "skipped" when what it means is "the op suite did not earn the right to run this".
            entry["gate_reason"] = (r.get("failure") or {}).get("detail")
        if r.get("kind") == "model":
            entry["kind"] = "model"
            entry["model_execution_check"] = r.get("model_execution_check")
            me = r.get("mesh_execution") or {}
            if me:
                entry["mesh_execution"] = {
                    k: me.get(k) for k in ("matmul_layers_on_mesh", "matmul_layers_host_fallback", "status") if k in me
                }
            for k in ("contract_obligations", "tiers_unexercised"):
                if r.get(k):
                    entry[k] = r[k]
        score["per_capsule"].append(entry)

    # active-vs-waiting rollup: wall is the suite wall-clock (overlapped under parallelism); the sum of
    # active_sim across capsules can exceed wall (that ratio IS the parallel speedup). oracle_wait_s is
    # time blocked on a queue/FPGA slot (≈0 for local spike/verilator).
    _active = _agg["build_s"] + _agg["sim_active_s"]
    score["timing_rollup"] = {
        "suite_wall_s": round(_suite_wall, 3),
        "build_s": round(_agg["build_s"], 3),
        "sim_active_s": round(_agg["sim_active_s"], 3),
        "active_total_s": round(_active, 3),
        "oracle_wait_s": round(_agg["oracle_wait_s"], 3),
        "max_workers": workers,
        "parallel_speedup": round(_active / _suite_wall, 2) if _suite_wall > 0 else None,
    }

    # WHAT THIS GRADE ACTUALLY PAID FOR. `timing_rollup` above is the cost of the tiers this run
    # EXECUTED, and once a tier can be carried from an earlier certification of the same executable on
    # the same instrument (merlin.development.targetgen.oracles.tier_cache) a falling wall clock has two possible causes --
    # a faster grade, or a grade that measured less. Stated here so the two are never confused: a saving
    # that is not reported as reuse is indistinguishable from a claim that this run measured everything.
    # Emitted even when nothing was carried, so "no reuse" and "no accounting" cannot look the same.
    _reuse = {"executed": {}, "carried": {}, "capsules_with_a_carried_tier": 0}
    for r in results:
        blk = r.get("tier_reuse") if isinstance(r.get("tier_reuse"), dict) else {}
        for _t in blk.get("executed") or []:
            _reuse["executed"][_t] = _reuse["executed"].get(_t, 0) + 1
        for _t in blk.get("carried") or []:
            _reuse["carried"][_t] = _reuse["carried"].get(_t, 0) + 1
        if blk.get("carried"):
            _reuse["capsules_with_a_carried_tier"] += 1
    score["tier_reuse"] = _reuse

    # coverage aggregate (written alongside)
    cov = CV.aggregate(results, capsules=caps, traces=traces, target=target)
    score["coverage"] = {
        "by_tier_reached": cov["by_tier_reached"],
        "instruction_class_coverage": cov["instruction_class_coverage"],
        "mode_coverage": cov["mode_coverage"],
        "unavailable": cov["unavailable"],
        "acceleratable_coverage": cov["acceleratable_coverage"],
    }

    # PROVENANCE. This score asserts a hardware result -- N of M capsules passed on named oracle tiers --
    # so it must record WHICH hardware revision produced it. A result attributed to the wrong device is
    # worse than no result, because it gets cited. Pins for these targets already existed; nothing emitted
    # the block, and the gate could not see the score to complain, so every bench verdict on every target
    # was unattributable. Never fatal: a grade that ran must still report its numbers, with the
    # provenance gap visible rather than the whole grade lost.
    # THE PERF LEDGER. Written by the run that took the numbers, because that is the only moment the
    # submission and the concurrency are known for certain. Never fatal: a grade that ran must still
    # report its verdict, with the ledger's gap visible rather than the whole grade lost.
    try:
        score["perf_ledger"] = emit_perf_ledger(results, target=target, runs_root=runs_root)
    except Exception as _e:  # noqa: BLE001
        score["perf_ledger"] = {"unavailable": f"{type(_e).__name__}: {_e}"}

    try:
        from merlin.common import provenance as _P

        _pins = {}
        for _name in _P.load_pins():
            try:
                _pins[_name] = _P.verify(_name)
            except Exception:  # noqa: BLE001 - checkout absent is a gap
                continue
        score["provenance"] = _P.record(
            pins=_pins or None,
            extra={"target": target, "n_capsules": score.get("n_capsules"), "n_passed": score.get("n_passed")},
        )
    except Exception as _e:  # noqa: BLE001
        score["provenance"] = {"unavailable": f"{type(_e).__name__}: {_e}"}
    return score


def emit_perf_ledger(results: list, *, target: str, runs_root: str | Path, version: int = 1) -> dict:
    """Publish using the current grader callbacks, preserving legacy overrides."""
    return _emit_perf_ledger(
        results,
        target=target,
        runs_root=runs_root,
        version=version,
        activity_source=_activity_source,
        suite_for=capsule_eligibility.suite_for,
    )


def _headline(score: dict) -> str:
    """The score as it may be QUOTED: never a bare fraction, always with what the passes rest on.

    Reports the RTL-backed share whenever it is below the pass count, and the highest tier every graded
    capsule reached (``None`` -> stated as such rather than omitted, because an omitted tier reads as a
    high one). A reader who copies this string cannot accidentally drop the qualification, which is the
    whole failure this exists to prevent.
    """
    ev = score.get("pass_evidence") or {}
    n_passed, n_caps = score.get("n_passed", 0), score.get("n_capsules", 0)
    head = score.get("public_passed") or f"{n_passed}/{n_caps}"
    tier = score.get("highest_tier")
    bits = [f"tier {tier}" if tier else "no tier cleared by every capsule"]
    rtl, cheap = ev.get("rtl_backed"), ev.get("cheap_tier_only")
    # SAY WHAT THE NON-RTL PASSES ACTUALLY RESTED ON. "passed on cheap tiers only" is right for a spike
    # pass and wrong for an RTL-DERIVED cosim pass, and the two now arrive distinguished.
    modelc = ev.get("model_certified") or 0
    if rtl is not None and cheap:
        rest = (
            f"the other {cheap} on a model oracle only ({modelc} RTL-derived)"
            if modelc
            else f"the other {cheap} passed on cheap tiers only"
        )
        bits.append(f"RTL-backed {rtl}/{n_passed} \u2014 {rest}")
    elif rtl is not None and n_passed:
        bits.append(f"RTL-backed {rtl}/{n_passed}")
    if score.get("hidden_passed"):
        bits.append(f"hidden {score['hidden_passed']}")
    # CAPSULES THAT WERE SCREENED AND NEVER CERTIFIED BELONG IN THE QUOTABLE STRING. Under a certify
    # budget the denominator is what was certified, so a headline of "17/18" can sit over a suite where
    # seven more capsules passed the cheap screen and nobody ever paid the RTL tier for them. Leaving
    # that to the JSON is the same mistake as leaving the tier to the JSON: the string is what gets
    # copied. The covering set guarantees the AXES were certified; it certifies nothing about these
    # capsules, and the wording has to keep those two apart.
    n_screened = score.get("n_screened_only") or 0
    if n_screened:
        bits.append(f"{n_screened} more screened only, NOT certified (outside the covering set)")
    # A DEFICIT MADE OF DECLINES IS A DIFFERENT RESULT FROM ONE MADE OF WRONG ANSWERS, and the headline
    # is what gets quoted. "14/26" reads as a compiler that computes twelve things incorrectly; if the
    # twelve were declined it is a compiler that never implemented them, which is a different repair.
    n_declined = score.get("n_declined") or 0
    if n_declined:
        bits.append(f"{n_declined} DECLINED by the backend (never lowered, not wrong answers)")
    # HOW MUCH OF THE MODEL THE UNIT WAS ASKED TO DO, not just whether the numbers came out right. A
    # whole-model pass that requests no readout on any command is numerically indistinguishable from
    # one that requests all of it, and 70x apart in cycles; the number that separates them belongs in
    # the string people quote, not only in the JSON. It is stated as a share of what the target is
    # DERIVED to apply, and when nothing could be compared it says that instead of saying zero.
    sp = score.get("epilogue_store_path") or {}
    if sp.get("n_model_rows"):
        if sp.get("share") is None:
            bits.append(f"store-path readout not measured on {sp['n_incomplete']} model capsule(s)")
        else:
            silent = sp.get("silent_fallbacks") or 0
            measured = f"store-path readout {sp['share']:.0%} of what the target applies"
            bits.append(f"{measured}, {silent} site(s) silently on the host" if silent else measured)
    return f"{head} ({'; '.join(bits)})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Grade a backend package through capsule_bench_v0")
    ap.add_argument("--package", required=True)
    ap.add_argument(
        "--capsules",
        default=None,
        help="capsules ROOT override (a single directory); default = the TARGET's own graded "
        "roots. Prefer the default: a hand-passed root is how a grade ends up scoring "
        "one target's package against another target's corpus.",
    )
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--target", required=True, help="target being graded (its config/oracle are derived)")
    ap.add_argument("--contract", default=None)
    ap.add_argument("--labels", default="public,dev")
    ap.add_argument("--hidden", action="store_true", help="grade ONLY hidden capsules (post-freeze)")
    ap.add_argument("--no-oracle", action="store_true", help="L0/L1/trace only (skip spike/verilator)")
    ap.add_argument("--score", default=None)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="parallel oracle instances (verilator/VCS/cyclotron); 0 = host-scaled default",
    )
    a = ap.parse_args(argv)
    # Resolve the roots ABSOLUTELY before anything runs. run_capsule chdirs into the build tree, so a
    # relative contract/capsules root stops resolving mid-grade and surfaces as a FileNotFoundError
    # from inside a worker thread -- which reads as a broken package rather than a bad path. The
    # defaults also came from string literals; the repo's rule is to ask paths for its roots.
    from merlin.common.paths import data_path

    a.contract = Path(a.contract).resolve() if a.contract else data_path("contract")
    # DEFAULT TO THE TARGET'S OWN GRADED ROOTS, not the corpus parent. The parent holds every target's
    # capsules at once, so it graded the gemmini package against 173 capsules from seven targets, called
    # 89 of them "outside this target's declared capability", and printed `1/84`. That number reads as a
    # catastrophic regression and means nothing; the target's real suite is 36. The usage line in this
    # module's own docstring recommended that parent, while discover_capsules two modules over documents
    # why it is wrong -- so following the documentation produced the bad grade.
    from merlin.target.registry.descriptor_loading import graded_capsule_roots

    a.capsules = [Path(a.capsules).resolve()] if a.capsules else graded_capsule_roots(a.target, hidden=a.hidden)
    if not a.capsules:
        ap.error("no descriptor-selected capsule roots; pass --capsules explicitly or select a reviewed descriptor")

    labels = {"hidden"} if a.hidden else set(a.labels.split(","))
    adapters = {} if a.no_oracle else None
    score = grade(
        a.package,
        capsules_root=a.capsules,
        runs_root=a.runs_root,
        labels=labels,
        contract=a.contract,
        oracle_adapters=adapters,
        timeout=a.timeout,
        max_workers=a.workers,
        target=a.target,
        no_oracle=a.no_oracle,
    )
    out = Path(a.score) if a.score else Path(a.runs_root) / "score_capsule.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(score, indent=2), encoding="utf-8")
    print(
        f"wrote {out}: functional_pass={score['functional_pass']} "
        f"score={score.get('headline')} integrity={score.get('integrity_status')}"
    )
    return 0 if score["functional_pass"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
