"""Pure, fail-closed statistics for an arms-grade Phase-P experiment.

This module launches nothing and searches for nothing.  It consumes one predeclared exact matrix and
the rows claimed for that matrix.  The declaration remains the denominator: missing, duplicate, or
inadmissible cells refuse the claim instead of disappearing from the statistic.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SCHEMA = "merlin.phase-p-performance-statistics.v4"
TARGET_SELECTED_SCHEMA = "merlin.phase-p-performance-statistics.v5"
FAILURE_POLICY = "refuse_claim_on_any_missing_duplicate_or_failed_declared_cell"
SUBJECTS = ("baseline", "candidate")
PERFORMANCE_ORACLES = {"gsim": "rtl_gsim"}
DEFAULT_PERFORMANCE_SIMULATOR = "gsim"
MIN_AGENT_TRIALS = 3
MIN_PAIRED_REPLICATES = 2


class EvidenceError(ValueError):
    """The experiment declaration itself is not an arms-grade predeclaration."""


@dataclass(frozen=True, order=True)
class MatrixIdentity:
    trial: str
    subject: str
    family: str
    capsule: str
    simulator: str
    replicate: str

    @classmethod
    def from_mapping(cls, value: Any) -> MatrixIdentity:
        if not isinstance(value, Mapping):
            raise EvidenceError("matrix identity is not a mapping")
        fields = ("trial", "subject", "family", "capsule", "simulator", "replicate")
        if set(value) != set(fields):
            raise EvidenceError("matrix identity must contain exactly trial/subject/family/capsule/simulator/replicate")
        values = []
        for field in fields:
            item = value.get(field)
            if not isinstance(item, str) or not item.strip():
                raise EvidenceError(f"matrix identity {field} must be a non-empty string")
            values.append(item)
        return cls(*values)

    def to_dict(self) -> dict[str, str]:
        return {
            "trial": self.trial,
            "subject": self.subject,
            "family": self.family,
            "capsule": self.capsule,
            "simulator": self.simulator,
            "replicate": self.replicate,
        }


def canonical_json(value: Any) -> str:
    """Canonical JSON used for matrix identities and evidence products."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _rtl_simulator(value: object) -> bool:
    """A concrete simulator identity, suitable for a derived RTL oracle kind."""
    return (
        isinstance(value, str)
        and bool(value)
        and value[0].isalpha()
        and value.isascii()
        and all(char.isalnum() or char == "_" for char in value)
        and value not in {"spike", "unknown", "elaborated_rtl"}
    )


_ORACLE_BINDING_FIELDS = frozenset(
    {"simulator", "simulator_binary_sha256", "elaborated_firrtl_sha256", "certificate_sha256"}
)


def _valid_oracle_binding(binding: object, simulator: object) -> bool:
    return (
        isinstance(binding, Mapping)
        and set(binding) == _ORACLE_BINDING_FIELDS
        and binding.get("simulator") == simulator
        and all(_is_digest(binding.get(field)) for field in _ORACLE_BINDING_FIELDS - {"simulator"})
    )


def predeclare(
    *,
    trials: Sequence[Mapping[str, Any]],
    capsules: Sequence[Mapping[str, str]],
    replicates: Sequence[str],
    primary_simulator: str = DEFAULT_PERFORMANCE_SIMULATOR,
    schema: str | None = None,
    oracle_binding: Mapping[str, str] | None = None,
    replicate_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact deterministic elaborated-RTL matrix before measurements exist.

    ``replicate_policy`` (optional) records which members get ONE gSIM observation cited by both of
    their replicate identities: the matrix keeps both identities, and the policy says which of them
    carry one observation, so an analysis never reads zero dispersion there as measured.

    The historical v4 declaration retains GSIM as its sole timing authority.  V5
    binds an explicitly selected RTL simulator before measurements exist.
    """
    selected_schema = schema or (
        SCHEMA if primary_simulator == DEFAULT_PERFORMANCE_SIMULATOR else TARGET_SELECTED_SCHEMA
    )
    if selected_schema not in (SCHEMA, TARGET_SELECTED_SCHEMA):
        raise EvidenceError("unsupported performance statistics declaration schema")
    if selected_schema == SCHEMA and primary_simulator != DEFAULT_PERFORMANCE_SIMULATOR:
        raise EvidenceError("historical v4 primary performance simulator must be GSIM")
    if selected_schema == TARGET_SELECTED_SCHEMA and not _rtl_simulator(primary_simulator):
        raise EvidenceError("target-selected primary performance simulator must name a concrete RTL engine")
    if selected_schema == TARGET_SELECTED_SCHEMA and not _valid_oracle_binding(oracle_binding, primary_simulator):
        raise EvidenceError("target-selected timing requires a pinned certificate/binary/FIRRTL binding")
    if selected_schema == SCHEMA and oracle_binding is not None:
        raise EvidenceError("historical v4 declaration cannot add a target-selected oracle binding")
    if any(not isinstance(row, Mapping) for row in trials):
        raise EvidenceError("trial records must be mappings")
    trial_rows = sorted((dict(row) for row in trials), key=lambda row: str(row.get("trial", "")))
    if len(trial_rows) < MIN_AGENT_TRIALS:
        raise EvidenceError(f"at least {MIN_AGENT_TRIALS} independent agent trials are required")
    trial_names: list[str] = []
    agent_runs: set[str] = set()
    for row in trial_rows:
        if set(row) != {"trial", "agent_run_id"}:
            raise EvidenceError("trial record must contain exactly trial/agent_run_id")
        trial, agent_run = row["trial"], row["agent_run_id"]
        if not isinstance(trial, str) or not trial or trial in trial_names:
            raise EvidenceError("trial identities must be non-empty and unique")
        if not isinstance(agent_run, str) or not agent_run or agent_run in agent_runs:
            raise EvidenceError("independent trials require distinct non-empty agent run ids")
        trial_names.append(trial)
        agent_runs.add(agent_run)

    if any(not isinstance(item, str) or not item for item in replicates):
        raise EvidenceError(f"at least {MIN_PAIRED_REPLICATES} unique paired replicate ids are required")
    replicate_ids = sorted(replicates)
    if len(replicate_ids) < MIN_PAIRED_REPLICATES or len(set(replicate_ids)) != len(replicate_ids):
        raise EvidenceError(f"at least {MIN_PAIRED_REPLICATES} unique paired replicate ids are required")
    if any(not isinstance(row, Mapping) for row in capsules):
        raise EvidenceError("capsule declarations must be mappings")
    capsule_rows = sorted(
        (dict(row) for row in capsules), key=lambda row: (str(row.get("family", "")), str(row.get("capsule", "")))
    )
    capsule_keys = []
    for row in capsule_rows:
        if set(row) != {"family", "capsule"}:
            raise EvidenceError("capsule declaration must contain exactly family/capsule")
        key = (row["family"], row["capsule"])
        if any(not isinstance(item, str) or not item for item in key) or key in capsule_keys:
            raise EvidenceError("family/capsule identities must be non-empty and unique")
        capsule_keys.append(key)
    if not capsule_keys:
        raise EvidenceError("the declared performance capsule cohort must be non-empty")

    matrix = [
        MatrixIdentity(trial, subject, family, capsule, primary_simulator, replicate).to_dict()
        for trial in sorted(trial_names)
        for subject in SUBJECTS
        for family, capsule in capsule_keys
        for replicate in replicate_ids
    ]
    declaration = {
        "schema": selected_schema,
        "failure_policy": FAILURE_POLICY,
        "primary_simulator": primary_simulator,
        "trials": trial_rows,
        "matrix": matrix,
    }
    if selected_schema == TARGET_SELECTED_SCHEMA:
        declaration["oracle_binding"] = dict(oracle_binding)
    if replicate_policy is not None:
        problem = _replicate_policy_issue(replicate_policy, set(capsule_keys))
        if problem:
            raise EvidenceError(problem)
        declaration["replicate_policy"] = json.loads(json.dumps(dict(replicate_policy), sort_keys=True))
    declaration["matrix_sha256"] = _sha256(matrix)
    declaration["declaration_sha256"] = _sha256(declaration)
    return declaration


_REPLICATE_POLICY_FIELDS = frozenset({"rule", "threshold_roofline_cycles", "estimate", "single_observation"})


def _replicate_policy_issue(policy: Any, capsule_keys: set[tuple[str, str]]) -> str | None:
    if not isinstance(policy, Mapping) or set(policy) != _REPLICATE_POLICY_FIELDS:
        return "replicate policy must declare exactly rule/threshold_roofline_cycles/estimate/single_observation"
    threshold = policy["threshold_roofline_cycles"]
    if type(threshold) is not int or threshold <= 0:
        return "replicate policy threshold must be a positive cycle count"
    rows = policy["single_observation"]
    if not isinstance(rows, list):
        return "replicate policy single_observation must be a list"
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"family", "capsule", "roofline_floor_cycles"}:
            return "replicate policy rows must name family/capsule/roofline_floor_cycles"
        if (row["family"], row["capsule"]) not in capsule_keys:
            return f"replicate policy names an undeclared member {row['family']}/{row['capsule']}"
        floor = row["roofline_floor_cycles"]
        if type(floor) is not int or floor <= threshold:
            return f"replicate policy member {row['capsule']} is not above the declared threshold"
    return None


def _declaration(declaration: Any) -> tuple[list[MatrixIdentity], list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    if not isinstance(declaration, Mapping):
        return [], [], ["declaration is not a mapping"]
    schema = declaration.get("schema")
    if schema not in (SCHEMA, TARGET_SELECTED_SCHEMA):
        issues.append("declaration schema is absent or unsupported")
    if declaration.get("failure_policy") != FAILURE_POLICY:
        issues.append("declared failure policy does not refuse incomplete cells")
    raw_matrix = declaration.get("matrix")
    identities: list[MatrixIdentity] = []
    if not isinstance(raw_matrix, list):
        issues.append("declaration has no exact matrix")
    else:
        for index, raw in enumerate(raw_matrix):
            try:
                identities.append(MatrixIdentity.from_mapping(raw))
            except EvidenceError as exc:
                issues.append(f"matrix[{index}]: {exc}")
        if len(set(identities)) != len(identities):
            issues.append("declaration matrix contains duplicate identities")
        if declaration.get("matrix_sha256") != _sha256(raw_matrix):
            issues.append("declaration matrix digest does not match its exact identities")
    if any(identity.subject not in SUBJECTS for identity in identities):
        issues.append("matrix subject must be exactly baseline or candidate")
    primary_simulator = declaration.get("primary_simulator")
    if schema == SCHEMA and primary_simulator != DEFAULT_PERFORMANCE_SIMULATOR:
        issues.append("historical v4 declaration must use GSIM as its primary timing authority")
    if schema == TARGET_SELECTED_SCHEMA and not _rtl_simulator(primary_simulator):
        issues.append("target-selected declaration lacks a concrete RTL timing simulator")
    if schema == TARGET_SELECTED_SCHEMA and not _valid_oracle_binding(
        declaration.get("oracle_binding"), primary_simulator
    ):
        issues.append("target-selected declaration lacks pinned certificate/binary/FIRRTL identities")
    if any(identity.simulator != primary_simulator for identity in identities):
        issues.append("performance matrix may contain only its predeclared primary simulator")
    trial_shapes: dict[str, set[tuple[str, str, str, str, str]]] = {}
    for identity in identities:
        trial_shapes.setdefault(identity.trial, set()).add(
            (identity.subject, identity.family, identity.capsule, identity.simulator, identity.replicate)
        )
    shapes = list(trial_shapes.values())
    if shapes and any(shape != shapes[0] for shape in shapes[1:]):
        issues.append("every independent agent trial must declare the identical performance matrix")

    raw_trials = declaration.get("trials")
    trials = [dict(row) for row in raw_trials if isinstance(row, Mapping)] if isinstance(raw_trials, list) else []
    if len(trials) < MIN_AGENT_TRIALS or len(trials) != len(raw_trials or []):
        issues.append(f"declaration needs at least {MIN_AGENT_TRIALS} well-formed agent trials")
    declared_trials = {identity.trial for identity in identities}
    evidence_trials = {row.get("trial") for row in trials}
    if declared_trials != evidence_trials:
        issues.append("trial evidence and matrix trial identities differ")
    agent_runs = [row.get("agent_run_id") for row in trials]
    if any(set(row) != {"trial", "agent_run_id"} for row in trials):
        issues.append("trial evidence has fields outside the predeclared identity schema")
    if any(not isinstance(value, str) or not value for value in agent_runs) or len(set(agent_runs)) != len(agent_runs):
        issues.append("agent trials are not independently identified")
    groups: dict[tuple[str, str, str], dict[str, set[str]]] = {}
    for identity in identities:
        key = (identity.trial, identity.family, identity.capsule)
        groups.setdefault(key, {}).setdefault(identity.subject, set()).add(identity.replicate)
    for key, subjects in sorted(groups.items()):
        if set(subjects) != set(SUBJECTS):
            issues.append(f"L3 cell {key} does not declare both paired subjects")
            continue
        if subjects["baseline"] != subjects["candidate"]:
            issues.append(f"L3 cell {key} baseline/candidate replicate sets differ")
        if len(subjects["baseline"]) < MIN_PAIRED_REPLICATES:
            issues.append(f"L3 cell {key} has fewer than {MIN_PAIRED_REPLICATES} paired replicates")
    if "replicate_policy" in declaration:
        problem = _replicate_policy_issue(
            declaration["replicate_policy"], {(identity.family, identity.capsule) for identity in identities}
        )
        if problem:
            issues.append(problem)
    if declaration.get("declaration_sha256") != _sha256(
        {key: value for key, value in declaration.items() if key != "declaration_sha256"}
    ):
        issues.append("declaration digest does not match its content")
    return sorted(identities), trials, issues


def _row_identity(row: Any) -> MatrixIdentity | None:
    if not isinstance(row, Mapping):
        return None
    raw = row.get("identity")
    try:
        return MatrixIdentity.from_mapping(raw)
    except EvidenceError:
        return None


def _row_problem(
    row: Mapping[str, Any], identity: MatrixIdentity, *, schema: object, oracle_binding: object
) -> str | None:
    if row.get("correct") is not True:
        return "correct is not true"
    if row.get("tier") != "L3":
        return "row is not an L3 measurement"
    if row.get("cycle_accurate") is not True:
        return "row lacks explicit cycle-accurate standing"
    oracle = row.get("oracle")
    expected_oracle = (
        f"rtl_{identity.simulator}"
        if schema == TARGET_SELECTED_SCHEMA and _rtl_simulator(identity.simulator)
        else PERFORMANCE_ORACLES.get(identity.simulator)
    )
    if (
        not isinstance(oracle, Mapping)
        or oracle.get("derived_from_rtl") is not True
        or oracle.get("kind") != expected_oracle
    ):
        return f"row is not from the predeclared RTL-derived cycle-accurate {identity.simulator} oracle"
    if schema == TARGET_SELECTED_SCHEMA:
        if not _valid_oracle_binding(oracle_binding, identity.simulator):
            return "target-selected declaration has no pinned oracle binding"
        if any(
            oracle.get(field) != oracle_binding[field]
            for field in ("simulator_binary_sha256", "elaborated_firrtl_sha256")
        ) or not _is_digest(oracle.get("program_elf_sha256")):
            return "row lacks selected binary/FIRRTL/ELF identity"
        qualification = row.get("qualification")
        if (
            not isinstance(qualification, Mapping)
            or qualification.get("admitted") is not True
            or qualification.get("certificate_sha256") != oracle_binding["certificate_sha256"]
        ):
            return "row lacks qualification against the pinned timing certificate"
    cycles = row.get("cycles")
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
        return "cycles is not a positive integer"
    return None


def _geometric_mean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise EvidenceError("geometric mean requires finite positive values")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def evaluate(
    declaration: Any, rows: Sequence[Any], *, trial_evidence: Sequence[Mapping[str, Any]] | None = None
) -> dict[str, Any]:
    """Admit the exact matrix and calculate paired all-trial statistics, or refuse."""
    identities, trials, issues = _declaration(declaration)
    schema = declaration.get("schema") if isinstance(declaration, Mapping) else None
    attached = list(trial_evidence or ())
    if len(attached) != len(trials) or any(not isinstance(row, Mapping) for row in attached):
        issues.append("every predeclared trial must attach one post-run evidence record")
    else:
        expected_trials = {(row.get("trial"), row.get("agent_run_id")) for row in trials}
        observed_trials = {(row.get("trial"), row.get("agent_run_id")) for row in attached}
        if any(set(row) != {"trial", "agent_run_id", "agent_evidence_sha256"} for row in attached):
            issues.append("post-run trial evidence has fields outside its exact schema")
        if observed_trials != expected_trials:
            issues.append("post-run evidence does not exactly cover the predeclared agent trials")
        evidence_hashes = [row.get("agent_evidence_sha256") for row in attached]
        if any(not _is_digest(value) for value in evidence_hashes) or len(set(evidence_hashes)) != len(evidence_hashes):
            issues.append("agent trials lack distinct content-addressed post-run evidence")
    expected = set(identities)
    observed: dict[MatrixIdentity, list[Mapping[str, Any]]] = {}
    excluded_spike = 0
    malformed = 0
    undeclared = 0
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        issues.append("result rows are not a sequence")
        rows = ()
    for row in rows:
        identity = _row_identity(row)
        if schema == SCHEMA and identity is not None and identity.simulator == "spike":
            excluded_spike += 1
            continue
        if identity is None or not isinstance(row, Mapping):
            malformed += 1
            continue
        if identity not in expected:
            undeclared += 1
            continue
        observed.setdefault(identity, []).append(row)
    if malformed:
        issues.append(f"{malformed} result row(s) have malformed identities")
    if undeclared:
        issues.append(f"{undeclared} non-Spike result row(s) are outside the predeclared matrix")

    ledger = []
    admitted: dict[MatrixIdentity, Mapping[str, Any]] = {}
    missing = duplicate = failed = 0
    for identity in identities:
        matches = observed.get(identity, [])
        if not matches:
            status, reason = "missing", "no result for declared identity"
            missing += 1
        elif len(matches) != 1:
            status, reason = "duplicate", f"{len(matches)} results for one declared identity"
            duplicate += 1
        else:
            problem = _row_problem(
                matches[0],
                identity,
                schema=schema,
                oracle_binding=declaration.get("oracle_binding") if isinstance(declaration, Mapping) else None,
            )
            if problem:
                status, reason = "failed", problem
                failed += 1
            else:
                status, reason = "admitted", None
                admitted[identity] = matches[0]
        ledger.append({"identity": identity.to_dict(), "status": status, "reason": reason})
    if missing:
        issues.append(f"{missing} declared matrix cell(s) are missing")
    if duplicate:
        issues.append(f"{duplicate} declared matrix cell(s) are duplicated")
    if failed:
        issues.append(f"{failed} declared matrix cell(s) failed admission")

    incomplete_trials = sorted({entry["identity"]["trial"] for entry in ledger if entry["status"] != "admitted"})
    declared_trial_ids = {row.get("trial") for row in trials}
    incomplete_declared_trials = sorted(trial for trial in incomplete_trials if trial in declared_trial_ids)

    accounting = {
        "declared_cells": len(identities),
        "admitted_cells": len(admitted),
        "missing_cells": missing,
        "duplicate_cells": duplicate,
        "failed_cells": failed,
        "undeclared_non_spike_rows": undeclared,
        "malformed_rows": malformed,
        "excluded_spike_rows": excluded_spike,
        "declared_agent_trials": len(trials),
        "attached_agent_evidence": len(attached),
        "complete_agent_trials": len(trials) - len(incomplete_declared_trials),
        "incomplete_agent_trials": len(incomplete_declared_trials),
        "incomplete_trial_identities": incomplete_declared_trials,
    }
    if issues:
        result = {
            "schema": schema if schema in (SCHEMA, TARGET_SELECTED_SCHEMA) else SCHEMA,
            "status": "refused",
            "failure_policy": FAILURE_POLICY,
            "matrix_sha256": declaration.get("matrix_sha256") if isinstance(declaration, Mapping) else None,
            "accounting": accounting,
            "issues": sorted(set(issues)),
            "cell_ledger": ledger,
            "per_trial": [],
            "aggregate": None,
        }
        result["result_sha256"] = _sha256(result)
        return result

    paired: dict[str, list[dict[str, Any]]] = {str(row["trial"]): [] for row in trials}
    pair_keys = sorted(
        {
            (identity.trial, identity.family, identity.capsule, identity.simulator, identity.replicate)
            for identity in identities
        }
    )
    for trial, family, capsule, simulator, replicate in pair_keys:
        common = {"trial": trial, "family": family, "capsule": capsule, "simulator": simulator, "replicate": replicate}
        baseline = admitted[MatrixIdentity(subject="baseline", **common)]
        candidate = admitted[MatrixIdentity(subject="candidate", **common)]
        speedup = baseline["cycles"] / candidate["cycles"]
        paired[trial].append(
            {
                "family": family,
                "capsule": capsule,
                "simulator": simulator,
                "replicate": replicate,
                "baseline_cycles": baseline["cycles"],
                "candidate_cycles": candidate["cycles"],
                "speedup": speedup,
            }
        )

    per_trial = []
    for trial in sorted(paired):
        pairs = paired[trial]
        family_names = sorted({pair["family"] for pair in pairs})
        family_gm = {
            family: _geometric_mean([pair["speedup"] for pair in pairs if pair["family"] == family])
            for family in family_names
        }
        per_trial.append(
            {
                "trial": trial,
                "paired_cells": len(pairs),
                "geometric_mean_speedup": _geometric_mean([pair["speedup"] for pair in pairs]),
                "family_geometric_mean_speedup": family_gm,
                "pairs": pairs,
            }
        )
    trial_values = [row["geometric_mean_speedup"] for row in per_trial]
    median = statistics.median(trial_values)
    mad = statistics.median(abs(value - median) for value in trial_values)
    # Do not let a gain on the public-shaped/tuning family conceal a loss on
    # an unseen-shape cohort.  Families are predeclared matrix identities, so
    # these summaries cannot be invented or selected after measurements.
    family_names = sorted({identity.family for identity in identities})
    family_aggregate = {}
    for family in family_names:
        values = [row["family_geometric_mean_speedup"][family] for row in per_trial]
        family_median = statistics.median(values)
        family_aggregate[family] = {
            "method": "median_of_independent_trial_family_paired_geometric_means",
            "median_speedup": family_median,
            "all_trial_speedups": values,
            "uncertainty": {
                "minimum": min(values),
                "maximum": max(values),
                "median_absolute_deviation": statistics.median(abs(value - family_median) for value in values),
                "n_independent_trials": len(values),
            },
        }
    aggregate = {
        "method": "median_of_independent_trial_paired_geometric_means",
        "median_speedup": median,
        "all_trial_speedups": trial_values,
        "uncertainty": {
            "method": "across_trial_min_max_and_median_absolute_deviation",
            "minimum": min(trial_values),
            "maximum": max(trial_values),
            "median_absolute_deviation": mad,
            "n_independent_trials": len(trial_values),
        },
        "selection": "all_predeclared_trials_no_best_of_selection",
        "family_aggregate": family_aggregate,
        "generalization_policy": (
            "every predeclared family is reported separately; aggregate gains do not suppress "
            "a family-specific regression"
        ),
    }
    result = {
        "schema": schema,
        "status": "admitted",
        "failure_policy": FAILURE_POLICY,
        "matrix_sha256": declaration["matrix_sha256"],
        "accounting": accounting,
        "issues": [],
        "cell_ledger": ledger,
        "per_trial": per_trial,
        "aggregate": aggregate,
    }
    result["result_sha256"] = _sha256(result)
    return result
