"""Grade a submitted mlir_oot_target_backend package through the capsule_bench_v0 fidelity suite.

This turns capsule_bench from a self-test into a reusable GRADER: given any contract-satisfying
package directory, run the public/dev capsule suite (and, post-freeze, the hidden suite) through the
shared command-buffer/reference/oracle ladder, run the integrity scan, decode + check the RoCC trace,
and emit a single machine-readable ``score_capsule.json``.

Run this OUTSIDE the agent sandbox (it needs spike/verilator + the hidden capsules). The package is
still only invoked via its 4 CLI entrypoints (subprocess) — never imported.

Usage:
    python -m merlin.targetgen.capsule_grade --package <pkg_dir> --target <target> \
        --runs-root <out> [--hidden] [--labels public,dev] [--score <out.json>] [--no-oracle]

Do NOT pass ``--capsules merlin/contract/capsules``: that is the common parent of every target's corpus
and grades the package against all of them. Omit the flag and the target's own graded roots are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from . import capsule_runner as CR
from merlin.perf import comparand as _comparand
from .capsule_common import NOT_MEASURED_STATUSES
from .capsule_common import tier_field as _tier_field
from .capsule_common import tier_status as _tier_status
from . import coverage_report as CV
from .corpora import source_experiment_env
from .oot_runner import INFRASTRUCTURE_PLANE as OOT_INFRASTRUCTURE_PLANE
from .oot_runner import CertFailure, build_package, integrity_scan, load_package


def _name_set_sha256(names) -> str:
    return hashlib.sha256(
        json.dumps(sorted(str(x) for x in names), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _materialized_cohort_admission(capsules_root, discovered: list[dict], *,
                                   expected_descriptor_sha256: str | None = None) -> dict | None:
    """Load and validate a materializer-sealed public cohort boundary, when present."""
    roots = capsules_root if isinstance(capsules_root, (list, tuple)) else [capsules_root]
    records = [Path(root) / ".cohort_admission.json" for root in roots
               if (Path(root) / ".cohort_admission.json").is_file()]
    if not records:
        return None
    if len(records) != 1:
        raise ValueError(f"expected one materialized cohort admission record, found {records}")
    record = json.loads(records[0].read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("version") != 1:
        raise ValueError(f"malformed materialized cohort admission record at {records[0]}")
    ints = ("n_source_capsules", "n_admitted_capsules", "n_capability_excluded",
            "n_resource_excluded")
    if any(not isinstance(record.get(k), int) or isinstance(record.get(k), bool)
           or record[k] < 0 for k in ints):
        raise ValueError("materialized cohort admission has malformed counts")
    if record["n_source_capsules"] != (record["n_admitted_capsules"]
                                        + record["n_capability_excluded"]
                                        + record["n_resource_excluded"]):
        raise ValueError("materialized cohort admission arithmetic is inconsistent")
    names = [str(cap.get("name")) for cap in discovered]
    if record["n_admitted_capsules"] != len(names):
        raise ValueError("materialized cohort admitted count does not match discovered capsules")
    if record.get("admitted_name_set_sha256") != _name_set_sha256(names):
        raise ValueError("materialized cohort admitted-name digest does not match its contents")
    for field in ("excluded_name_set_sha256", "admitted_name_set_sha256", "descriptor_sha256"):
        digest = record.get(field)
        if (not isinstance(digest, str) or len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest)):
            raise ValueError(f"materialized cohort {field} is malformed")
    if (expected_descriptor_sha256 is not None
            and record.get("descriptor_sha256") != expected_descriptor_sha256):
        raise ValueError(
            "materialized cohort descriptor digest does not match the target descriptor used for grade")
    if record.get("policy") == "descriptor_capability_and_resource_v1":
        if record.get("resource_policy") != "representative_l3_capstones_v1":
            raise ValueError("materialized cohort has an unapproved resource-exclusion policy")
        required = record.get("required_admitted_models")
        if not isinstance(required, list) or not required:
            raise ValueError("materialized cohort does not name its required admitted model capstones")
        if not set(str(x) for x in required).issubset(names):
            raise ValueError("materialized cohort is missing a required admitted model capstone")
    return record


def model_execution_check(result: dict, capsule: dict | None = None) -> dict:
    """Return the model capsule's honest structural/effect evidence.

    Operator capsules have one emitted kernel, so their decoded instruction trace is the appropriate
    structural check. A model capsule is a composition of many accelerator kernels and host islands,
    so its equivalent proof is an ordered dynamic dispatch ledger. Every accelerator entry must bind
    the exact decoded trace, RTL fidelity and content-addressed compiler artifacts; aggregate counters,
    independently synthesized tiles and the declared host/accelerator seam must all agree with it.
    Missing/malformed evidence fails closed.
    """
    violations: list[str] = []
    execution = result.get("mesh_execution")
    tiles = result.get("mesh_tile_verification")
    requested_engine = execution.get("simulator_requested") if isinstance(execution, dict) else None
    if requested_engine is not None and (
            not isinstance(requested_engine, str) or not requested_engine):
        violations.append("model_requested_oracle_engine_malformed")
        requested_engine = None
    required_engine = os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip() or None
    if required_engine is not None and requested_engine is None:
        violations.append("model_requested_oracle_engine_missing")
    if required_engine is not None and requested_engine != required_engine:
        violations.append("model_requested_oracle_engine_differs_from_required_engine")

    def _count(record, field: str, *, minimum: int = 0) -> int | None:
        value = record.get(field) if isinstance(record, dict) else None
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            violations.append(f"{field}_missing_or_malformed")
            return None
        return value

    def _valid_digest(value) -> bool:
        return (isinstance(value, str) and len(value) == 64 and value == value.lower()
                and all(ch in "0123456789abcdef" for ch in value))

    required_artifacts = {"input_interface", "command_buffer", "lowered_llvm", "kernel_object",
                          "package_kernel_elf", "instruction_trace"}

    def _check_exact_cert(record: dict, prefix: str) -> None:
        trace = record.get("trace_check") if isinstance(record, dict) else None
        if (not isinstance(trace, dict) or trace.get("status") != "pass"
                or trace.get("drives_accelerator") is not True
                or not isinstance(trace.get("n_instructions"), int)
                or isinstance(trace.get("n_instructions"), bool)
                or trace.get("n_instructions", 0) < 1):
            violations.append(f"{prefix}_accelerator_trace_missing_or_invalid")
        oracle = record.get("oracle_evidence") if prefix == "model_call" else {
            "result": record.get("oracle_result"),
            "engine": record.get("oracle_engine"),
            "derived_from_rtl": record.get("derived_from_rtl"),
            "cycle_accurate": record.get("cycle_accurate"),
        }
        if (not isinstance(oracle, dict) or oracle.get("result") != "pass"
                or oracle.get("derived_from_rtl") is not True
                or oracle.get("cycle_accurate") is not True):
            violations.append(f"{prefix}_oracle_fidelity_missing_or_invalid")
        engine = oracle.get("engine") if isinstance(oracle, dict) else None
        expected_engine = required_engine or requested_engine
        if expected_engine is not None and (not isinstance(engine, str) or not engine):
            violations.append(f"{prefix}_oracle_engine_missing_or_invalid")
        elif expected_engine is not None and engine != expected_engine:
            violations.append(f"{prefix}_oracle_engine_mismatch")
        identity = record.get("artifact_identity") if isinstance(record, dict) else None
        artifacts = identity.get("artifacts") if isinstance(identity, dict) else None
        if (not isinstance(identity, dict) or identity.get("version") != 1
                or not isinstance(identity.get("run_id"), str) or not identity.get("run_id")
                or not isinstance(artifacts, dict)
                or set(artifacts) < required_artifacts):
            violations.append(f"{prefix}_artifact_identity_missing_or_invalid")
            return
        for name in required_artifacts:
            item = artifacts.get(name)
            if (not isinstance(item, dict) or not _valid_digest(item.get("sha256"))
                    or not isinstance(item.get("size_bytes"), int)
                    or isinstance(item.get("size_bytes"), bool) or item["size_bytes"] < 1):
                violations.append(f"{prefix}_artifact_digest_missing_or_invalid")
                return
        canonical = json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode()
        if identity.get("content_sha256") != hashlib.sha256(canonical).hexdigest():
            violations.append(f"{prefix}_artifact_identity_digest_mismatch")
        if isinstance(trace, dict) and trace.get("artifact_sha256") != artifacts[
                "instruction_trace"]["sha256"]:
            violations.append(f"{prefix}_trace_artifact_digest_mismatch")
        if prefix == "model_call" and record.get("cert_run_id") != identity.get("run_id"):
            violations.append("model_call_cert_run_identity_mismatch")

    routed = _count(execution, "matmul_layers_routed", minimum=1)
    on_mesh = _count(execution, "matmul_layers_on_mesh", minimum=1)
    fallback = _count(execution, "matmul_layers_host_fallback")
    unavailable = _count(execution, "matmul_layers_oracle_unavailable")
    unrouted = _count(execution, "matmul_layers_unrouted")
    # ``routed`` counts static kernel symbols while ``on_mesh`` counts dynamic calls, so equality is
    # not a valid invariant for loops/reuse.  Nonzero on both sides plus zero fallback/unavailable/
    # unrouted is the unit-correct obligation for the captured model execution.
    if fallback not in (None, 0):
        violations.append("eligible_model_layer_fell_back_to_host")
    if unavailable not in (None, 0):
        violations.append("model_layer_oracle_unavailable")
    if unrouted not in (None, 0):
        violations.append("model_contraction_layer_unrouted")

    ledger = execution.get("dispatch_ledger") if isinstance(execution, dict) else None
    mesh_entries: list[dict] = []
    fallback_entries: list[dict] = []
    unavailable_entries: list[dict] = []
    scalar_entries: list[dict] = []
    if not isinstance(ledger, list) or not ledger:
        violations.append("dynamic_dispatch_ledger_missing_or_malformed")
        ledger = []
    else:
        allowed = {"on_mesh", "scalar_rvv_lane", "host_fallback", "mesh_unavailable"}
        cert_run_contents: dict[str, str] = {}
        for ordinal, entry in enumerate(ledger):
            if (not isinstance(entry, dict) or entry.get("ordinal") != ordinal
                    or not isinstance(entry.get("symbol"), str) or not entry.get("symbol")
                    or entry.get("lane") not in allowed or entry.get("status") != "pass"):
                violations.append("dynamic_dispatch_ledger_entry_missing_or_malformed")
                continue
            lane = entry["lane"]
            if lane == "on_mesh":
                mesh_entries.append(entry)
                _check_exact_cert(entry, "model_call")
                # A run id names the durable artifact directory.  Reusing it is safe only for the same
                # exact content; a repeated shape with different operands otherwise overwrites call A
                # with call B while both stale identities remain in the ledger.
                run_id = entry.get("cert_run_id")
                identity = entry.get("artifact_identity")
                content = identity.get("content_sha256") if isinstance(identity, dict) else None
                if isinstance(run_id, str) and isinstance(content, str):
                    previous = cert_run_contents.setdefault(run_id, content)
                    if previous != content:
                        violations.append("model_call_run_id_content_collision")
            elif lane == "host_fallback":
                fallback_entries.append(entry)
            elif lane == "mesh_unavailable":
                unavailable_entries.append(entry)
            else:
                scalar_entries.append(entry)
    if on_mesh is not None and len(mesh_entries) != on_mesh:
        violations.append("model_mesh_counter_ledger_mismatch")
    if fallback is not None and len(fallback_entries) != fallback:
        violations.append("model_fallback_counter_ledger_mismatch")
    if unavailable is not None and len(unavailable_entries) != unavailable:
        violations.append("model_unavailable_counter_ledger_mismatch")
    route_symbols = execution.get("mesh_route_symbols") if isinstance(execution, dict) else None
    if (not isinstance(route_symbols, list) or not route_symbols
            or any(not isinstance(x, str) or not x for x in route_symbols)
            or len(set(route_symbols)) != len(route_symbols)
            or (routed is not None and len(route_symbols) != routed)):
        violations.append("mesh_route_symbol_evidence_missing_or_malformed")
    else:
        attempted = {e["symbol"] for e in mesh_entries + fallback_entries + unavailable_entries}
        if attempted != set(route_symbols):
            violations.append("static_dynamic_mesh_route_mismatch")

    n_tiles = _count(tiles, "n_tiles", minimum=1)
    n_passed = _count(tiles, "n_passed")
    n_failed = _count(tiles, "n_failed")
    n_unavailable = _count(tiles, "n_unavailable")
    n_unsynth = _count(tiles, "n_unsynthesizable")
    if n_tiles is not None and n_passed is not None and n_passed != n_tiles:
        violations.append("not_all_model_tiles_certified")
    if n_failed not in (None, 0):
        violations.append("model_tile_failed")
    if n_unavailable not in (None, 0):
        violations.append("model_tile_oracle_unavailable")
    if n_unsynth not in (None, 0):
        violations.append("model_tile_unsynthesizable")
    # THE SCREEN RUNG IS EVIDENCE, AND ITS ABSENCE IS A HOLE. A model declares
    # `required_oracle_tiers: [L0, L1, L2, L3]`, so a cert-tier pass with no screen record means the
    # capsule reached the cert tier without earning the tier below it -- the ordering every operator
    # capsule obeys. Absent must never read as satisfied, so an unrecorded screen is a violation and
    # not a silent omission.
    n_screened = _count(tiles, "n_screened", minimum=1)   # None -> `_count` already recorded the hole
    if n_screened is not None:
        if _count(tiles, "n_screen_passed") != n_screened:
            violations.append("not_all_model_tiles_screened")
        if _count(tiles, "n_screen_failed") not in (None, 0):
            violations.append("model_tile_screen_failed")
        if _count(tiles, "n_screen_unavailable") not in (None, 0):
            violations.append("model_tile_screen_oracle_unavailable")
    per_tile = tiles.get("per_tile") if isinstance(tiles, dict) else None
    if (not isinstance(per_tile, list) or n_tiles is None or len(per_tile) != n_tiles):
        violations.append("model_tile_evidence_missing_or_malformed")
    else:
        for tile in per_tile:
            if not isinstance(tile, dict) or tile.get("status") != "pass":
                violations.append("model_tile_record_not_pass")
                break
            if tile.get("derived_from_rtl") is not True:
                violations.append("model_tile_not_rtl_derived")
                break
            if tile.get("cycle_accurate") is not True:
                violations.append("model_tile_not_cycle_accurate")
                break
            _check_exact_cert(tile, "model_tile")

    lane = result.get("lane_report")
    required_lanes = [str(x) for x in (((capsule or {}).get("lanes") or {}).get("require") or [])]
    observed_lanes = ({"on_mesh"} if mesh_entries else set()) | (
        {"scalar_rvv_lane"} if scalar_entries else set())
    missing_required = sorted(set(required_lanes) - observed_lanes)
    if missing_required:
        violations.append("required_model_lane_unexercised")
    if required_lanes and lane is None:
        violations.append("required_lane_report_missing")
    elif lane is not None:
        # `evidence` is a MAPPING of lane -> the rung that lane was judged on; a single label for the
        # whole report had to lie about at least one lane, which is why the producer reports per lane.
        # This end compared that mapping to a bare string, so the condition was true for every report
        # ever produced and every lane-declaring whole-model capsule failed as "malformed" -- blaming
        # the submission for a report the harness itself builds. Judge the SHAPE here, and the evidence
        # QUALITY below, against the producer's own exported vocabulary.
        _ev = lane.get("evidence")
        if (not isinstance(lane, dict) or not isinstance(lane.get("unexercised"), list)
                or not isinstance(_ev, dict)):
            violations.append("lane_report_missing_or_malformed")
        elif (set(lane.get("observed") or []) != observed_lanes
              or sorted(lane["unexercised"]) != missing_required):
            violations.append("lane_report_disagrees_with_dispatch_ledger")
        else:
            # A required lane reported as exercised must rest on evidence that something RAN. The
            # remaining rung, "routing_plan", is the router's INTENT: measured on one submission, 15
            # matmuls were assigned to the mesh and 15 fell back to the host at run time. Accepting it
            # here would let a plan satisfy an execution obligation.
            _plan_only = sorted(ln for ln in required_lanes
                                if ln not in missing_required
                                and _ev.get(ln) not in CR.EXECUTED_LANE_EVIDENCE)
            if _plan_only:
                violations.append("required_lane_evidenced_by_plan_only")

    expected_boundary = result.get("boundary_expectation")
    actual_boundary = result.get("boundary_execution")
    if (not isinstance(expected_boundary, dict)
            or expected_boundary.get("boundary") in (None, "UNKNOWN")
            or expected_boundary.get("n_unresolved") != 0):
        violations.append("model_boundary_expectation_missing_or_unresolved")
    if (not isinstance(actual_boundary, dict) or actual_boundary.get("status") != "pass"
            or actual_boundary.get("boundary") in (None, "UNKNOWN")
            or actual_boundary.get("n_unresolved") != 0):
        violations.append("model_dynamic_boundary_missing_or_unresolved")
    elif isinstance(expected_boundary, dict):
        required_boundary = expected_boundary.get("boundary")
        actual_patterns = set(actual_boundary.get("contains") or [])
        if required_boundary != actual_boundary.get("boundary") and required_boundary not in actual_patterns:
            violations.append("model_dynamic_boundary_does_not_cover_declaration")
        # Explicitly preserve the important island seam even when the expectation's strongest label is
        # ``routing``; that label necessarily contains A->H->A and must not erase the concrete obligation.
        if "A->H->A" in set(expected_boundary.get("contains") or []) | {required_boundary} \
                and "A->H->A" not in actual_patterns:
            violations.append("model_a_h_a_seam_not_exercised")

    ledger_digest = (hashlib.sha256(json.dumps(ledger, sort_keys=True, separators=(",", ":"),
                                                  default=str).encode()).hexdigest()
                     if ledger else None)

    return {
        "status": "pass" if not violations else "fail",
        "kind": "model_accelerator_execution",
        "matmul_layers_routed": routed,
        "matmul_layers_on_mesh": on_mesh,
        "matmul_layers_unrouted": unrouted,
        "n_tiles": n_tiles,
        "n_tiles_certified": n_passed,
        "simulator_requested": requested_engine,
        "required_rtl_engine": required_engine,
        "dispatch_ledger_sha256": ledger_digest,
        "observed_lanes": sorted(observed_lanes),
        "dynamic_boundary": (actual_boundary or {}).get("boundary") if isinstance(
            actual_boundary, dict) else None,
        "violations": list(dict.fromkeys(violations)),
        "note": ("model-level counterpart to an operator instruction trace: evidence that this model's "
                 "accelerator regions executed through the submitted package and its required lanes "
                 "were exercised; it is not represented as a decoded single-kernel trace"),
    }


def enforce_model_execution_check(result: dict, capsule: dict | None, *, target: str) -> dict:
    """Attach and enforce a whole-model execution proof on the capsule verdict itself.

    The score-level structural flag is not enough: QA and checkpoint readers intentionally re-open each
    durable ``capsule_result.json``.  Before this enforcement a Verilator-backed model remained
    ``status=pass, L3=pass`` under a GSIM pin; only the in-memory score carried a failing auxiliary check,
    so those durable readers false-accepted it.  A wrong/missing required engine is ``incomplete`` (the
    requested evidence did not run), while malformed evidence from the requested engine is a real fail.
    """
    check = model_execution_check(result, capsule)
    result["model_execution_check"] = check
    if check.get("status") == "pass":
        return result

    violations = [str(v) for v in (check.get("violations") or [])]
    engine_unmeasured = any(
        v.startswith("model_requested_oracle_engine_") or v.endswith("_oracle_engine_mismatch")
        or v.endswith("_oracle_engine_missing_or_invalid")
        for v in violations
    )
    status = "unavailable" if engine_unmeasured else "fail"
    detail = "whole-model execution proof failed: " + ", ".join(violations)
    for tier in CR._rtl_tiers_of(target):
        record = (result.get("tiers") or {}).get(tier)
        if isinstance(record, dict) and record.get("status") == "pass":
            record.update(status=status, reason=detail, cycles=None,
                          derived_from_rtl=False, cycle_accurate=False)

    # Preserve a pre-existing stronger failure.  The dangerous case is the flattering pass that escaped
    # into durable QA; convert that to an honest no-measurement or protocol verdict.
    if result.get("status") == "pass":
        result["status"] = "incomplete" if engine_unmeasured else "fail"
        result["failure"] = {
            "plane": "required_rtl_engine" if engine_unmeasured else "model_execution",
            "category": "NOT_RUN_IS_NOT_PASS" if engine_unmeasured else "PROTOCOL_VIOLATION",
            "detail": detail,
        }
    return result


def cycles_by_tier(tiers: dict | None, *, ladder: list[str] | tuple[str, ...] = ()) -> dict[str, int]:
    """Every tier record that REPORTS a cycle count, keyed by the tier that reported it.

    Which tier holds a capsule's cycle count is a property of the run, not a constant. The ladder runs
    a target's cheapest oracle first and that ordering differs per target, so a capsule can carry a
    count under one tier and nothing under another. Harvesting a single fixed tier therefore drops
    counts silently and asymmetrically: measured on one graded run, 11 of 12 FAILING capsules carried
    an elaborated-RTL record with no entry at the tier the diagnostic read, so the cycle count of
    nearly every failure — exactly the capsules a diagnostic exists to explain — never reached the
    summary, while the passes (which cleared every tier) were all present. That reads as "failures
    have no cycles", which is the opposite of true.

    Keyed by tier rather than flattened, so two tiers' counts for one capsule sit side by side and can
    be compared. ``ladder`` orders the result (tiers outside it are appended in encounter order); a
    tier record in the bare-string form (``"pass"``) reports no fields and contributes nothing.
    """
    found: dict[str, int] = {}
    for name, record in (tiers or {}).items():
        cycles = _tier_field(record, "cycles")
        if cycles is not None:
            found[name] = cycles
    ordered = {t: found[t] for t in ladder if t in found}
    ordered.update({t: c for t, c in found.items() if t not in ordered})
    return ordered


def default_grade_workers(n_capsules: int | None = None) -> int:
    """How many per-capsule oracle instances (verilator/VCS/cyclotron) to fan out in parallel — the
    per-capsule sim runs are independent, so grading N capsules serially wastes wall-clock. Derived from
    the host (leave 2 cores headroom, cap at 16), never more than the number of capsules; overridable via
    ``MERLIN_GRADE_WORKERS``. Applies to every arm's grade (the shared suite grader)."""
    env = os.environ.get("MERLIN_GRADE_WORKERS")
    if env and env.isdigit() and int(env) > 0:
        w = int(env)
    else:
        w = max(1, min((os.cpu_count() or 4) - 2, 16))
    return max(1, min(w, n_capsules)) if n_capsules else w


def grade(package_dir: str | Path, *, capsules_root: str | Path, runs_root: str | Path,
          labels: set[str] | None = None, contract: str | Path | None = None,
          oracle_adapters: dict | None = None, timeout: int = 900,
          max_workers: int = 0, target: str, no_oracle: bool = False,
          capability_admission: bool = False, target_experiment=None) -> dict:
    """Run the capsule suite over a submitted package; return a score dict (also schema-checkable).

    ``max_workers`` fans the per-capsule oracle runs out in parallel (verilator/VCS/cyclotron instances);
    ``<= 0`` (the default) derives a host-scaled count via :func:`default_grade_workers` so grading a
    multi-capsule suite never serializes the RTL sims. Any positive value is honored as an explicit cap.

    ``no_oracle`` marks an EXPLICIT structure-only smoke (typically paired with ``oracle_adapters={}``):
    a capsule that clears the structural tiers but whose mandatory numeric tier was deliberately not run
    is recorded ``not_gradeable_no_oracle`` (a withheld numeric verdict), NOT the fixable
    ``oracle_unavailable`` plane — so the report is honest and never claims a numeric pass. Graded runs
    (``no_oracle=False``) keep the ``not_run_is_not_pass`` behavior byte-for-byte."""
    labels = labels or {"public", "dev"}
    # Source the target's own tooling paths before resolving any adapter, exactly as the harness does
    # for an arm. Without this a grade run outside the harness can lose a target's certifying-tier sim
    # and report the whole suite `incomplete` -- fail-closed, but describing the environment rather than
    # the submission. Process env wins, so an exported var is untouched.
    _sourced = source_experiment_env(target)
    if _sourced:
        print(f"  sourced {len(_sourced)} tooling path(s) from the target's experiment.env: "
              f"{', '.join(sorted(_sourced))}", flush=True)
    # absolute before any threading, for the same reason run_suite does it: this function also reads the
    # per-capsule traces back out of this root, and a root that moved under a sibling thread's chdir
    # silently yields an empty coverage dict rather than an error.
    runs_root = str(Path(runs_root).resolve())
    pkg_dir = Path(package_dir).resolve()
    if contract is not None and Path(contract).exists():
        contract = str(Path(contract).resolve())
    capsules_root = ([str(Path(r).resolve()) for r in capsules_root]
                     if isinstance(capsules_root, (list, tuple))
                     else str(Path(capsules_root).resolve()))
    # HOLD the staged cohort for the duration of this grade. `.resolve()` above is deliberate (a root that
    # moved under a sibling thread's chdir yields an empty coverage dict instead of an error) but it also
    # turns the per-target cohort SYMLINK into one concrete `.<target>.build.<pid>.<hex>` path that is then
    # read for the whole grade -- 10-20 min functional, ~30-40 min per capsule on the cert tier. The
    # materializer's collector used to drop any build older than 15 minutes, and it dropped one a live
    # grade was mid-way through: 31 of 33 capsules were then recorded as structurally invalid SUBMISSIONS.
    # The lease makes "still being read" a fact the collector can check rather than infer from age. It is
    # advisory and best-effort -- if it cannot be taken, the grade proceeds exactly as before.
    try:
        from .contract.materialize import pin_cohort_builds
        pin_cohort_builds(*(capsules_root if isinstance(capsules_root, list) else [capsules_root]))
    except Exception:  # noqa: BLE001 -- pinning is an optimisation for the collector, never a gate
        pass

    score: dict = {
        "task": f"{target}-mlir-oot-capsule", "package": str(pkg_dir),
        "integrity_exempt": None, "integrity_status": None,
        "labels_graded": sorted(labels),
        "functional_pass": 0, "n_capsules": 0, "n_passed": 0,
        "public_passed": None, "hidden_passed": None, "headline": None,
        "per_capsule": [], "tier_reached": {}, "first_failure_planes": {},
        "numeric_all_exact": None, "trace_all_pass": None,
        "model_execution_all_pass": None, "structural_evidence_all_pass": None,
        "structural_evidence_scope": None,
        "cycles_diagnostic": {}, "cycles_provenance": {}, "highest_tier": None,
        "timing_diagnostic": {}, "timing_rollup": {},
    }

    # K0/K1: load + integrity scan + build (fail-closed, recorded honestly)
    try:
        pkg = load_package(pkg_dir, contract=contract)
        score["integrity_exempt"] = pkg.integrity_exempt
        integrity_scan(pkg)
        score["integrity_status"] = "clean" if not pkg.integrity_exempt else "exempt"
        build_package(pkg)
    except CertFailure as cf:
        score["integrity_status"] = f"FAIL[{cf.plane}]: {cf.detail[:200]}"
        score["functional_pass"] = 0
        score["failure"] = {"plane": cf.plane, "category": str(cf.category), "detail": cf.detail}
        return score

    source_caps = CR.discover_capsules(capsules_root, labels=labels, contract=contract)
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
        _admitted_ops, capability_excluded = CR._split_ineligible(op_caps, target)
        excluded_names = {str(r.get("capsule")) for r in capability_excluded}
        caps = [c for c in source_caps if str(c.get("name")) not in excluded_names]
        admission_policy = "frozen_target_capability_operand_dtype"
    materialized_admission = (None if capability_admission else
                              _materialized_cohort_admission(capsules_root, source_caps))
    if materialized_admission is not None:
        # Public cohort admission is a decision made under exact descriptor bytes.  Resolve through the
        # active MERLIN_TARGET_EXPERIMENT-aware path unless the caller already holds the frozen parsed
        # experiment, then require the materializer record to name those same bytes.  Hidden grading has
        # no public admission record and keeps its name-secrecy-preserving capability boundary below.
        experiment = target_experiment
        if experiment is None:
            from .corpora import descriptor_path
            from .target_experiment import load_target_experiment

            descriptor = descriptor_path(target)
            if not descriptor.is_file():
                raise ValueError(
                    f"target {target!r} has no experiment descriptor at {descriptor}")
            experiment = load_target_experiment(descriptor)
        if experiment.target != target:
            raise ValueError(
                f"experiment descriptor names {experiment.target!r}, not grade target {target!r}")
        if materialized_admission.get("descriptor_sha256") != experiment.descriptor_sha256:
            raise ValueError(
                "materialized cohort descriptor digest does not match the target descriptor used for grade")
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
    import time as _time
    _suite_t0 = _time.perf_counter()
    results = CR.run_suite(caps, pkg_dir, runs_root=runs_root, contract=contract,
                           oracle_adapters=oracle_adapters, timeout=timeout,
                           max_workers=workers, target=target, no_oracle=no_oracle)
    _suite_wall = _time.perf_counter() - _suite_t0

    # A whole model has no single model-level instruction stream.  Attach its distinct execution
    # obligation before roll-up; never synthesize a trace_check=pass for an artifact that did not exist.
    _caps_by_name = {str(cap.get("name")): cap for cap in caps}
    for result in results:
        if result.get("kind") == "model":
            enforce_model_execution_check(
                result, _caps_by_name.get(str(result.get("capsule"))), target=target)
            # The runner persisted the model result before this suite-level cross-record proof existed.
            # Keep the durable row in sync because QA/selfcheck/checkpoint consumers deliberately re-glob
            # it rather than trusting this function's in-memory score.
            result_path = (Path(runs_root) / "runs" / CR.suite_for(target)
                           / str(result.get("capsule")) / "capsule_result.json")
            if result_path.is_file():
                result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # collect decoded traces for coverage — read from the TARGET's own suite dir (run_capsule writes
    # under cfg.suite, e.g. atlas-capsule-bench), not the gemmini SUITE literal (which left the atlas
    # coverage dict silently empty; same root cause as the self-check n_capsules:0 blind loop).
    traces: dict[str, dict] = {}
    rr = Path(runs_root) / "runs" / CR.suite_for(target)
    for cap in caps:
        tp = rr / cap["name"] / "generated" / "instruction_trace.json"
        if tp.exists():
            try:
                traces[cap["name"]] = json.loads(tp.read_text())
            except Exception:
                pass

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
    graded = [r for r in results
              if r.get("status") not in NOT_MEASURED_STATUSES]
    n_pass = sum(1 for r in graded if r["status"] == "pass")
    score["n_capsules"] = len(graded)
    score["n_passed"] = n_pass
    score["n_not_graded_ineligible"] = len(ungraded)
    score["n_gated_deferred"] = len(deferred)
    if ungraded:
        score["not_graded_ineligible"] = sorted(r.get("capsule") for r in ungraded)
    if deferred:
        score["gated_deferred"] = sorted(r.get("capsule") for r in deferred)
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
            "was measured. Clear the ceiling to grade them -- an operator certification run has none.")
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
            "these capsules.")
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
            key=lambda d: str(d.get("capsule")))
        score["declined_note"] = (
            "the backend STATED it does not lower these capsules rather than emitting a program that "
            "writes nothing. They are in the denominator and are NOT passes. Read them as coverage "
            "gaps (a shape/op the backend never implemented), not as numeric defects.")
    # BOTH NUMBERS, ALWAYS. When the runtime drove a loop nest over a backend's declared single tile,
    # the resulting pass is evidence about the runtime AND the backend; the backend's own coverage is the
    # smaller number. Reporting only the larger one is precisely how "our runtime covered for them" gets
    # cited as "their compiler generalizes", so the two travel together or the guardrail is not real.
    _runtime_tiled = [r for r in graded
                      if (((r.get("contract_obligations") or {}).get("capacity_fit") or {}).get("tiled_by")
                          == "declared_primitive_tile")]
    if _runtime_tiled:
        _names = sorted(str(r.get("capsule")) for r in _runtime_tiled)
        _unblocked = n_pass - sum(1 for r in _runtime_tiled if r.get("status") == "pass")
        score["backend_coverage"] = {
            "with_runtime_loop": f"{n_pass}/{len(graded)}",
            "unblocked": f"{_unblocked}/{len(graded)}",
            "runtime_tiled_capsules": _names,
            "note": ("`unblocked` is what the BACKEND lowered on its own. `with_runtime_loop` adds the "
                     "capsules that only passed because the runtime drove the M/N/K loop over the single "
                     "tile the backend declared. Quote `unblocked` for any claim about the backend "
                     "generalizing over shape; `with_runtime_loop` is a statement about the pair."),
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
        _why = sorted({(r.get("failure") or {}).get("tier_reason")
                       or (r.get("failure") or {}).get("detail") or "unknown"
                       for r in _incomplete})
        score["measurement_incomplete"] = {
            "n": len(_incomplete), "of": len(graded),
            "capsules": sorted(r["capsule"] for r in _incomplete)[:12],
            "reasons": _why[:4],
            "detail": (f"{len(_incomplete)} of {len(graded)} capsules had a MANDATORY tier come back "
                       f"unavailable, so the oracle never ran on them. n_passed is NOT a capability "
                       f"measurement for this run — fix the environment and re-grade before quoting it."),
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
            "n": len(_infra), "of": len(results),
            "plane": OOT_INFRASTRUCTURE_PLANE,
            "capsules": sorted(r["capsule"] for r in _infra)[:12],
            "reasons": _paths[:2],
            "detail": (f"{len(_infra)} of {len(results)} capsules could not be graded because their "
                       f"STAGED INPUTS were missing — the capsule cohort was not materialized, or its "
                       f"staging directory was removed while this grade was running. This is a HARNESS "
                       f"fault: it is NOT a verdict on the submission, and n_passed/n_capsules is not a "
                       f"measurement of it. Re-materialize the cohort and re-grade before quoting any "
                       f"number from this run."),
        }
    # `gradeable` means this run had a working numeric oracle. It did not, for those capsules, so it is
    # False even though an oracle was requested — the same fail-closed posture the empty suite gets.
    score["gradeable"] = (not no_oracle) and not _empty and not _incomplete and not _infra
    score["n_not_gradeable_no_oracle"] = n_not_gradeable
    score["n_structural_pass"] = n_pass + n_not_gradeable
    score["structural_pass"] = bool(not _empty and not _infra
                                    and (n_pass + n_not_gradeable) == len(graded))
    score["numeric_all_exact"] = None if _empty else all(
        r.get("numeric", {}).get("status") == "pass" for r in graded)
    _trace_rows = [r for r in graded if r.get("kind") != "model"]
    _model_rows = [r for r in graded if r.get("kind") == "model"]
    score["trace_all_pass"] = (None if not _trace_rows else all(
        r.get("trace_check", {}).get("status") == "pass" for r in _trace_rows))
    score["model_execution_all_pass"] = (None if not _model_rows else all(
        r.get("model_execution_check", {}).get("status") == "pass" for r in _model_rows))
    score["structural_evidence_scope"] = {
        "n_instruction_trace_capsules": len(_trace_rows),
        "n_model_execution_capsules": len(_model_rows),
    }
    score["structural_evidence_all_pass"] = bool(
        not _empty
        and (not _trace_rows or score["trace_all_pass"] is True)
        and (not _model_rows or score["model_execution_all_pass"] is True))
    if _empty:
        score["note"] = ("no capsules matched the requested labels at this root — nothing graded; "
                         "flags are null (not a pass). Check the capsules root / labels.")

    pub = [r for r in graded if r.get("label") in ("public", "dev")]
    hid = [r for r in graded if r.get("label") == "hidden"]
    if pub:
        score["public_passed"] = f"{sum(1 for r in pub if r['status']=='pass')}/{len(pub)}"
    if hid:
        score["hidden_passed"] = f"{sum(1 for r in hid if r['status']=='pass')}/{len(hid)}"

    tiers = ["L0", "L1", "L2", "L3", "L4", "L5"]
    for t in tiers:
        # over GRADED only: a withheld capsule has no tiers, so counting it in the denominator would
        # make tier_reached never reach the total and silently force highest_tier to None.
        score["tier_reached"][t] = sum(
            1 for r in graded if _tier_status((r.get("tiers") or {}).get(t)) == "pass")
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
    from .capsule_runner import _rtl_tiers_of
    _rtl_names = _rtl_tiers_of(target)

    # AN EXPLICIT DENIAL MUST WIN OVER THE NAME. `derived_from_rtl` has three states, and collapsing the
    # last two is what let a model count as RTL: True (the oracle says it ran RTL), False (the oracle says
    # it did NOT -- e.g. an arc cosim, an RTL-DERIVED model, landing on the tier named L3), and None (the
    # record is the bare-string form a model capsule writes, which states nothing). Only None may fall
    # back to the tier name. Reading `flag or name in _rtl_names` credited an oracle that had explicitly
    # denied being RTL, purely because its tier shared a name with another target's Verilator tier.
    def _is_rtl_pass(name, rec) -> bool:
        if _tier_status(rec) != "pass":
            return False
        flag = _tier_field(rec, "derived_from_rtl")
        return bool(name in _rtl_names) if flag is None else bool(flag)

    def _tier_is_rtl(name, rec) -> bool:
        """Same three-state rule as :func:`_is_rtl_pass`, without the pass requirement."""
        flag = _tier_field(rec, "derived_from_rtl")
        return bool(name in _rtl_names) if flag is None else bool(flag)

    _passed = [r for r in graded if r.get("status") == "pass"]
    _rtl_backed = [r for r in _passed
                   if any(_is_rtl_pass(n, t) for n, t in (r.get("tiers") or {}).items())]
    # A pass that cleared a tier the ORACLE ITSELF called a model (fidelity != elaborated_rtl) and no RTL
    # tier beside it. Named separately because "not RTL-backed" reads as "did not run on hardware-grade
    # evidence at all", which is not what an RTL-derived cosim pass is.
    _model_certified = [r for r in _passed
                        if r not in _rtl_backed
                        and any(_tier_status(t) == "pass" and _tier_field(t, "fidelity")
                                for t in (r.get("tiers") or {}).values())]
    score["pass_evidence"] = {
        "n_passed": len(_passed),
        "rtl_backed": len(_rtl_backed),
        "model_certified": len(_model_certified),
        "cheap_tier_only": len(_passed) - len(_rtl_backed),
        "rtl_tiers_seen": sorted({n for r in graded for n, t in (r.get("tiers") or {}).items()
                                  if _tier_is_rtl(n, t)}),
        "fidelity_seen": sorted({f for r in graded for t in (r.get("tiers") or {}).values()
                                 if (f := _tier_field(t, "fidelity"))}),
        "note": ("`rtl_backed` counts passes that cleared a tier the ORACLE reported as elaborated RTL "
                 "(`derived_from_rtl`), falling back to the target's declared rtl_tiers only when the "
                 "oracle states nothing. A tier NAME is not evidence: one target's L3 is Verilator and "
                 "another's is an RTL-derived model, and counting by name credited the model. When "
                 "`rtl_backed` is below `n_passed`, the headline score is NOT an RTL result."),
    }

    # THE CITABLE FORM OF THE SCORE. `public_passed` stays a bare "20/20" because consumers parse it as
    # a fraction (agg_agentic_results._frac does int(total)), and a bare fraction is exactly what gets
    # quoted -- one gemmini submission travelled as "20/20" while its Verilator tier passed 1 of 20,
    # indistinguishable in the headline from three siblings that were RTL-clean on all 20. So the
    # qualification is built ONCE, here, next to the evidence that justifies it, and the renderers print
    # THIS rather than reassembling a bare fraction each time. Quote `headline`; parse `public_passed`.
    score["headline"] = _headline(score)

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
        _prov = _comparand.cycles_provenance(r.get("tiers"), submission=score.get("package"),
                                             ladder=tiers)
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
            entry_tm = {k: round(cap_tm[k], 3)
                        for k in ("build_s", "sim_active_s", "oracle_wait_s")}
            per_tier = {}
            for _t, _tm in cap_tm["by_tier"].items():
                _rec = (r.get("tiers") or {}).get(_t) or {}
                per_tier[_t] = {
                    **{k: round(_tm.get(k) or 0.0, 3)
                       for k in ("build_s", "sim_active_s", "oracle_wait_s")},
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
            "capsule": r["capsule"], "label": r.get("label"), "status": r["status"],
            "numeric": r.get("numeric", {}).get("status"),
            "trace": r.get("trace_check", {}).get("status"),
            "tiers": {t: _tier_status((r.get("tiers") or {}).get(t)) for t in tiers
                      if t in (r.get("tiers") or {})},
        }
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
                entry["mesh_execution"] = {k: me.get(k) for k in (
                    "matmul_layers_on_mesh", "matmul_layers_host_fallback", "status") if k in me}
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

    # coverage aggregate (written alongside)
    cov = CV.aggregate(results, capsules=caps, traces=traces, target=target)
    score["coverage"] = {"by_tier_reached": cov["by_tier_reached"],
                         "instruction_class_coverage": cov["instruction_class_coverage"],
                         "mode_coverage": cov["mode_coverage"], "unavailable": cov["unavailable"],
                         "acceleratable_coverage": cov["acceleratable_coverage"]}

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
    except Exception as _e:                                   # noqa: BLE001
        score["perf_ledger"] = {"unavailable": f"{type(_e).__name__}: {_e}"}

    try:
        from merlin.common import provenance as _P
        _pins = {}
        for _name in _P.load_pins():
            try:
                _pins[_name] = _P.verify(_name)
            except Exception:                                 # noqa: BLE001 - checkout absent is a gap
                continue
        score["provenance"] = _P.record(pins=_pins or None,
                                        extra={"target": target, "n_capsules": score.get("n_capsules"),
                                               "n_passed": score.get("n_passed")})
    except Exception as _e:                                   # noqa: BLE001
        score["provenance"] = {"unavailable": f"{type(_e).__name__}: {_e}"}
    return score




# =================================================================================================
# THE PERF LEDGER — what this grade MEASURED, kept separately from what it VERDICTED.
#
# Three perf products exist and they are three different concerns, so they get three topics and are
# never merged:
#   * ``perf-records``  — a pinned measured cycle suite (merlin.perf.record): a designed experiment.
#   * ``perf-harvest``  — retro-mined from runs already on disk (merlin.perf.harvest): archaeology.
#   * ``perf-ledger``   — THIS: what one grade run observed, written by the run that observed it.
#
# Every number here is ``trace_derived`` and stays that way. A per-unit busy count taken from a
# running program is exactly as contended as a total: other engines were moving data inside the same
# window, so the isolated cost of that unit is no larger than the number and nothing here may promote
# it to a constant (``merlin.perf.harvest.promote`` raises, permanently, and is not called).
# =================================================================================================


def _activity_source(workload: str, block, *, total_cycles: int, provenance: str):
    """An :class:`~merlin.perf.decompose.ActivitySource` from one validated timing block, or the
    reason there is none. Returns ``(source, refusal)`` with exactly one of them set.

    The KINDS come from the producer. They are not inferred from a unit's spelling and they are not
    defaulted: a unit whose kind the producer did not state is a hole in the input, and reading a
    role out of a name is how a local register load once became "DMA".
    """
    from merlin.perf.decompose import ResourceKind, activity_from_busy
    from merlin.perf.observations import IDLE_QUANTITY

    busy = block.busy_by_unit()
    if not busy:
        return None, f"{workload}: the block carries no per-unit busy count"
    declared = block.kinds()
    missing = sorted(set(busy) - set(declared))
    if missing:
        return None, (f"{workload}: the producer stated no kind for unit(s) {missing}. A role read "
                      f"out of a unit's NAME is not a derivation, so the source is refused")
    kinds: dict = {}
    for unit, kind in declared.items():
        try:
            kinds[unit] = ResourceKind(kind)
        except ValueError:
            return None, (f"{workload}: unit {unit!r} declares kind {kind!r}, which is not a "
                          f"resource kind this layer knows; refusing to map it onto one")
    idle = block.quantity(IDLE_QUANTITY)
    if idle is not None:
        # Cycles charged to NO engine are an intercept, not a rate, and they are first-class here:
        # on the measured corpus this is the single largest bucket, and an occupancy-only schema
        # cannot see it at all.
        busy = dict(busy)
        busy[IDLE_QUANTITY] = int(idle)
        kinds[IDLE_QUANTITY] = ResourceKind.FIXED
    try:
        src = activity_from_busy(workload, int(total_cycles), busy, kinds,
                                 # NOT a partition, asserted by the producer and re-asserted here.
                                 # If this ever becomes True the overlap reading is refused rather
                                 # than silently reported as zero.
                                 partitioned=bool(block.partitioned),
                                 completion_observable=None, provenance=provenance)
    except ValueError as exc:
        return None, f"{workload}: {exc}"
    return src, None


def emit_perf_ledger(results: list, *, target: str, runs_root: str | Path, version: int = 1) -> dict:
    """Write this grade's ``perf-ledger`` product and return a pointer + summary for the score.

    Products: ``ledger.json`` (per-tier rows plus the corpus-level composition question),
    ``observations.jsonl`` (one recovered observation per line) and ``refusals.json`` (everything
    that could have contributed and did not, with the reason). Per-cycle occupancy CSVs are NOT
    products -- they are ~34 B/cycle of replay-regenerable detail and live under
    ``cache_dir("occupancy")``, which is purgeable.

    Never fatal: a grade that ran must still report its numbers, with the ledger's gap visible.
    """
    from merlin.common.artifacts import cache_dir, new_product
    from merlin.kernels.measurement import authority_for
    from merlin.perf import harvest as HV
    from merlin.perf import observations as OBS
    from merlin.perf.headroom import composition_operator
    from merlin.perf.decompose import Unavailable

    auth = authority_for(target)
    rr = Path(runs_root) / "runs" / CR.suite_for(target)
    rows: list[dict] = []
    obs_rows: list[dict] = []
    refusals: list[dict] = []
    sources: list = []
    overlaps: dict[str, int] = {}
    no_capability: list[str] = []

    for r in results:
        capsule = str(r.get("capsule") or "")
        path = rr / capsule / "capsule_result.json"
        tiers = r.get("tiers") or {}
        saw_block = False
        for tier, rec in sorted(tiers.items()):
            if not isinstance(rec, dict):
                continue
            # Recomposed from its two halves: the observation list and the instrument's own
            # capability record, which the tier keeps beside it rather than inside it.
            block = OBS.block_from_tier_record(rec)
            row = {
                "capsule": capsule, "tier": tier, "status": rec.get("status"),
                "cycles": rec.get("cycles"), "fidelity": rec.get("fidelity"),
                "evidence": rec.get("evidence"),
                # WHICH ENGINE produced these seconds and cycles, which BUILD of it, and where its
                # console went. A perf ledger without the engine is a set of numbers from an unknown
                # instrument: GSIM and Verilator run the same design at the same fidelity and cost
                # ~20-30x apart, so a fit over rows that mix them silently averages two instruments.
                # Until now the only trace of the engine was inside the `evidence` FILENAME, which
                # every consumer here would have had to string-parse to recover.
                "engine": rec.get("engine"),
                "engine_selection": rec.get("engine_selection"),
                "sim_provenance": rec.get("sim_provenance"),
                "console_log": rec.get("console_log"),
                "console_bytes": rec.get("console_bytes"),
                # STATED, not inferred from the file's path. See TierResult.submission.
                "submission": rec.get("submission"),
                # A run taken before the stamp existed is MARKED, never retro-labelled: the
                # concurrency it ran at is not recoverable and pretending otherwise is worse.
                "concurrency": OBS.concurrency_of(rec),
                "timing": rec.get("timing"),
                "timing_capability": rec.get("timing_capability"),
                "has_timing_block": block is not None,
            }
            rows.append(row)
            if block is None:
                continue
            saw_block = True
            for why in block.refusals:
                refusals.append({"what": f"{capsule}@{tier}", "reason": why, "where": str(path)})
            if block.alias_collisions is None:
                refusals.append({
                    "what": f"{capsule}@{tier} alias accounting",
                    "reason": ("the producer carried no alias-collision count, so whether this run's "
                               "addresses stayed inside one window page is UNKNOWN, not zero"),
                    "where": str(path)})
            elif block.alias_collisions > 0:
                refusals.append({
                    "what": f"{capsule}@{tier}",
                    "reason": (f"{block.alias_collisions} access(es) collided inside the wrapping "
                               "memory window: two addresses a window apart became one byte, so "
                               "these numbers are not about the program that was submitted"),
                    "where": str(path)})
            total = block.quantity(OBS.SAMPLED_QUANTITY)
            if total is None:
                total = rec.get("cycles")
            overlap = block.overlap_cycles()
            key = f"{capsule}@{tier}"
            if total is None:
                refusals.append({"what": key, "reason": "no cycle total to place the buckets "
                                 "against", "where": str(path)})
                continue
            src, why = _activity_source(key, block, total_cycles=int(total),
                                        provenance=str(rec.get("evidence") or tier))
            if src is None:
                refusals.append({"what": key, "reason": why, "where": str(path)})
                continue
            sources.append(src)
            if overlap is None:
                refusals.append({
                    "what": key,
                    "reason": ("the block licenses no overlap reading (it asserts a partitioned "
                               "bucket set, or carries no joint-occupancy entry), so this workload "
                               "cannot contribute to the composition operator"),
                    "where": str(path)})
            else:
                overlaps[key] = int(overlap)
        if not saw_block and tiers:
            no_capability.append(capsule)
        if path.is_file():
            # Reuse the harvest path rather than a second reader: same citability gate, same
            # contended-provenance rules. The submission is passed EXPLICITLY so the path heuristic
            # in ``_submission_of`` is only a fallback, never the key.
            _sub = next((str((t.get("submission") or {}).get("package"))
                         for t in tiers.values()
                         if isinstance(t, dict) and (t.get("submission") or {}).get("package")), None)
            o, rf = HV.harvest_capsule_result(path, authority=auth, submission=_sub)
            obs_rows.extend(x.to_dict() for x in o)
            refusals.extend(x.to_dict() for x in rf)

    comp = composition_operator(sources, observed_overlap_cycles=overlaps or None)
    if isinstance(comp, Unavailable):
        # NAMED, not a bare "unknown": a reader has to be able to go and buy the missing evidence.
        composition = {"resolved": False, "what": comp.what, "missing": list(comp.missing),
                       "detail": comp.detail}
    else:
        op, eta = comp
        composition = {"resolved": True, "operator": op.value, "eta": round(float(eta), 4),
                       "note": ("derived from a JOINT occupancy observation independent of the "
                               "activity buckets -- never from a partitioned bucket set, which "
                               "reports zero overlap by construction")}

    occ = cache_dir("occupancy") / target
    pd = new_product("perf-ledger", version=version, target=target,
                     sources=[str(rr)],
                     notes="what ONE graded run measured: per-tier cycle counts with the submission "
                           "and the concurrency they were taken at, plus whatever per-unit "
                           "decomposition the oracle could see. Every value is trace_derived and "
                           "contended; none of it may become a calibrated constant. Distinct from "
                           "perf-records (a designed suite) and perf-harvest (retro-mined).")
    (pd.path / "observations.jsonl").write_text(
        "".join(json.dumps(o, sort_keys=True) + "\n" for o in obs_rows), encoding="utf-8")
    (pd.path / "refusals.json").write_text(
        json.dumps({"n": len(refusals), "refusals": refusals,
                    "capsules_with_no_timing_capability": sorted(no_capability),
                    "note": ("a capsule listed here ran on an oracle that reports no per-unit "
                             "timing. That is UNMEASURED, not zero -- the tier emitted no block at "
                             "all, which is the correct output for an instrument without the "
                             "capability.")}, indent=2) + "\n", encoding="utf-8")
    (pd.path / "ledger.json").write_text(json.dumps({
        "target": target,
        "authority": {"citable_tier": auth.citable_tier, "declared": auth.declared,
                      "gaps": list(auth.gaps())},
        "n_rows": len(rows), "n_observations": len(obs_rows), "n_refusals": len(refusals),
        "rows": rows,
        "composition_operator": composition,
        "per_cycle_occupancy": {
            "dir": str(occ),
            "purgeable": True,
            "note": ("per-cycle occupancy CSVs are ~34 B/cycle and are regenerable by replaying the "
                     "same program, so they are a cache, never a product. Written only when "
                     "MERLIN_PERF_PER_CYCLE=1 was set for the run."),
            "present": sorted(x.name for x in occ.glob("*.csv")) if occ.is_dir() else [],
        },
    }, indent=2, default=str) + "\n", encoding="utf-8")
    pd.write_manifest()
    return {"product": str(pd.path), "n_rows": len(rows), "n_observations": len(obs_rows),
            "n_refusals": len(refusals),
            "capsules_with_no_timing_capability": len(no_capability),
            "composition_operator": composition}


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
        rest = (f"the other {cheap} on a model oracle only ({modelc} RTL-derived)" if modelc
                else f"the other {cheap} passed on cheap tiers only")
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
    return f"{head} ({'; '.join(bits)})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Grade a backend package through capsule_bench_v0")
    ap.add_argument("--package", required=True)
    ap.add_argument("--capsules", default=None,
                    help="capsules ROOT override (a single directory); default = the TARGET's own graded "
                         "roots. Prefer the default: a hand-passed root is how a grade ends up scoring "
                         "one target's package against another target's corpus.")
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--target", required=True, help="target being graded (its config/oracle are derived)")
    ap.add_argument("--contract", default=None)
    ap.add_argument("--labels", default="public,dev")
    ap.add_argument("--hidden", action="store_true", help="grade ONLY hidden capsules (post-freeze)")
    ap.add_argument("--no-oracle", action="store_true", help="L0/L1/trace only (skip spike/verilator)")
    ap.add_argument("--score", default=None)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel oracle instances (verilator/VCS/cyclotron); 0 = host-scaled default")
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
    from .corpora import graded_capsule_roots
    a.capsules = ([Path(a.capsules).resolve()] if a.capsules
                  else graded_capsule_roots(a.target, hidden=a.hidden))

    labels = {"hidden"} if a.hidden else set(a.labels.split(","))
    adapters = {} if a.no_oracle else None
    score = grade(a.package, capsules_root=a.capsules, runs_root=a.runs_root, labels=labels,
                  contract=a.contract, oracle_adapters=adapters, timeout=a.timeout,
                  max_workers=a.workers, target=a.target, no_oracle=a.no_oracle)
    out = Path(a.score) if a.score else Path(a.runs_root) / "score_capsule.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(score, indent=2), encoding="utf-8")
    print(f"wrote {out}: functional_pass={score['functional_pass']} "
          f"score={score.get('headline')} integrity={score.get('integrity_status')}")
    return 0 if score["functional_pass"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
