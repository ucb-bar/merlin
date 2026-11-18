"""Grade stable passing functional cases through a rotating covering sample instead of every grade.

Functional cases that pass and keep passing consumed most grading executions: on one joint run 1915
of 3110 executions re-ran the original correctness cohort, and after a case had passed twice in a row
no case of that run ever regressed. Every grade's bounded admission window was spent re-proving
those cases while failing and large cases waited. Here, inside the rolling correctness curriculum
(:mod:`.joint_curriculum`), a case is *carried* -- not re-run this grade -- when all of these hold:

* it belongs to a sampled cohort (policy ``cohorts``; default the original correctness cohort);
* its last :data:`stable grades <DEFAULTS>` completed results passed in a row (``stable_grades``; by
  default derived from this run: one more than the longest passing streak any regression in this run
  ended, never fewer than ``min_stable_grades``);
* its dependency closure is unchanged since that streak: the case's own source bytes, the sources of
  the selected cases it ``extends``, and the frozen grading context except the compiler under test
  (descriptor, runtime configuration, policy, performance selection);
* its newest completed result in this run's runs roots is a pass (a failing or missing result is
  never carried), it was actually graded within the last full-sweep period, and it awaits no
  certification decision (its certificate ledger entry, when it has one, holds a passing L3 record);
* it is not one of this grade's hinted cases (the cases the author named in
  ``submission/joint-test-scope.json``, else the representative cases of the declared families).

Of the carried-eligible cases each grade still runs a covering sample: the fewest cases whose declared
axes (:func:`merlin.development.targetgen.oracles.tier_policy.capsule_axes`) span every axis value the eligible cases
declare, ties broken toward the case graded least recently, so the sample rotates through the cohort.
The compiler under test is excluded from the closure on purpose: the sample is the regression detector
for compiler edits, and a regression it finds makes the next grade a full sweep. A full sweep -- no case
carried -- also runs every ``full_sweep_every`` grades (default derived: eligible cases / sample size, so
a sweep comes as often as the rotation could visit every case once), during the campaign's final window
(``final_window_s``, default the policy's ``drain_s``), and in every milestone or final grade, which never
uses the curriculum at all.

A carried case keeps its newest completed result, attributed to the grade and source that produced it;
the grade lists it as ``carried (sampled)`` (its row status is ``carried``), it stays pending for the
current source, and it never counts as a pass of this grade, so ``all_pass``, promotion and
certification are unchanged. Disable with ``regression_sampling: {"enabled": false}``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from merlin.common.digest import hash_tree
from merlin.development.targetgen.corpus import capsule_io as capsule_capsule_common

# ---- regression_sampling -----------------------------------------------------------------------

SECTION = "regression_sampling"

DEFAULTS = {
    "enabled": True,
    # Performance cohorts (joint_test_selection lanes) whose stable cases may be carried.
    "cohorts": ["correctness_regression"],
    # Consecutive passing grades before a case may be carried; None derives it from this run's regressions.
    "stable_grades": None,
    # Floor of the derived stable_grades.
    "min_stable_grades": 2,
    # Grades between two full sweeps; None derives eligible cases / covering-sample size.
    "full_sweep_every": None,
    # A regression observed in a grade makes the next grade a full sweep.
    "sweep_after_regression": True,
    # Seconds before the campaign's drain boundary in which every grade is a full sweep; None = drain_s.
    "final_window_s": None,
}

LANES = ("correctness_regression", "calibration_diagnostics", "other")

STATUS = "carried"

REASON = "sampled"

LABEL = "carried (sampled)"

_KEEP_DECISIONS = 64


def validate(section) -> None:
    if section is None:
        return
    if not isinstance(section, dict) or set(section) - set(DEFAULTS):
        raise ValueError("regression sampling policy must be an object of known knobs")
    for key in ("enabled", "sweep_after_regression"):
        if key in section and type(section[key]) is not bool:
            raise ValueError(f"invalid regression sampling {key}")
    cohorts = section.get("cohorts", DEFAULTS["cohorts"])
    if not isinstance(cohorts, list) or not cohorts or any(c not in LANES for c in cohorts):
        raise ValueError("invalid regression sampling cohorts")
    for key, low, high in (
        ("stable_grades", 1, 10000),
        ("min_stable_grades", 1, 10000),
        ("full_sweep_every", 1, 10000),
        ("final_window_s", 0, 86400),
    ):
        value = section.get(key)
        if value is not None and (type(value) is not int or not low <= value <= high):
            raise ValueError(f"invalid regression sampling {key}")


def settings(policy: dict) -> dict:
    """The effective knobs; the run-time derivations are reported per grade in the curriculum report."""
    section = {**DEFAULTS, **((policy or {}).get(SECTION) or {})}
    final = section["final_window_s"]
    if final is None:
        final = (policy or {}).get("drain_s") if type((policy or {}).get("drain_s")) is int else 0
    return {
        "enabled": bool(section["enabled"]),
        "cohorts": list(section["cohorts"]),
        "stable_grades": section["stable_grades"],
        "min_stable_grades": section["min_stable_grades"],
        "full_sweep_every": section["full_sweep_every"],
        "sweep_after_regression": bool(section["sweep_after_regression"]),
        "final_window_s": final,
        "basis": {
            "stable_grades": "policy"
            if section["stable_grades"]
            else "max(min_stable_grades, 1 + longest passing streak a regression in this run ended)",
            "full_sweep_every": "policy"
            if section["full_sweep_every"]
            else "ceil(carried-eligible cases / covering-sample size)",
            "final_window_s": "policy" if section["final_window_s"] is not None else "policy drain_s",
        },
        "certification_authority": False,
    }


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def closures(source_digests: dict, docs: dict, context: dict) -> dict:
    """``{case: digest}`` of each case's dependency closure (see the module docstring)."""
    from merlin.development.targetgen.grading.scheduling import dependencies

    names = sorted(source_digests)
    try:
        deps = dependencies([{**(docs.get(n) or {}), "name": n} for n in names])
    except ValueError:
        deps = {n: set() for n in names}
    context = {k: v for k, v in (context or {}).items() if k != "compiler_sha256"}

    def ancestry(name, seen):
        for parent in sorted(deps.get(name, ())):
            if parent not in seen:
                seen.add(parent)
                ancestry(parent, seen)
        return seen

    return {
        n: _digest(
            {
                "context": context,
                "source": source_digests[n],
                "extends": {p: source_digests.get(p) for p in sorted(ancestry(n, set()))},
            }
        )
        for n in names
    }


def load_docs(paths: dict) -> dict:
    """``{case: capsule.yaml document}``; an unreadable declaration declares no axes."""
    import yaml

    docs = {}
    for name, path in paths.items():
        try:
            document = yaml.safe_load((Path(path) / "capsule.yaml").read_text())
        except (OSError, ValueError, yaml.YAMLError):
            document = None
        docs[name] = document if isinstance(document, dict) else {}
    return docs


def lanes(names, policy: dict) -> dict:
    """``{case: lane}`` by the same operator rosters :func:`.joint_test_scope.select` uses."""
    original = set((policy.get("correctness_curriculum") or {}).get("original_capsules") or ())
    witnesses = set((policy.get("gsim") or {}).get("witness_capsules") or ())
    return {
        n: "correctness_regression" if n in original else "calibration_diagnostics" if n in witnesses else "other"
        for n in names
    }


def _cert_settled(entry) -> bool:
    """No certification decision is waiting: no ledger entry, or a passing newest L3 record."""
    if not isinstance(entry, dict) or "L3" not in entry:
        return True
    record = entry.get("L3")
    return (record.get("status") if isinstance(record, dict) else record) in ("pass", "joined")


def rotating_cover(docs: dict, eligible, recency: dict) -> list[str]:
    """Greedy cover of the eligible cases' axis values, ties toward the least recently graded case."""
    from merlin.development.targetgen.oracles.tier_policy import capsule_axes

    axes = {n: capsule_axes(docs.get(n) or {}) for n in sorted(eligible)}
    uncovered = set().union(*axes.values()) if axes else set()
    # A case absent from this scheduler history has never been selected (grade zero).
    order = sorted(axes, key=lambda n: (-recency[n] if n in recency else 0, n))
    chosen: list[str] = []
    while uncovered:
        best = max(order, key=lambda n: len(axes[n] & uncovered))  # first maximum: least recent wins ties
        gain = axes[best] & uncovered
        if not gain:
            break
        chosen.append(best)
        uncovered -= gain
    if not chosen and order:
        chosen.append(order[0])  # cases declaring no axes still rotate one at a time
    return sorted(chosen)


def _state(ledger: dict) -> dict:
    state = ledger.setdefault("sampling", {})
    for key, default in (
        ("grade", 0),
        ("last_full_sweep_grade", 0),
        ("regression_grade", None),
        ("longest_streak_before_regression", 0),
        ("decisions", []),
    ):
        state.setdefault(key, default)
    return state


def _member(ledger: dict, name: str) -> dict:
    entry = ledger["members"].setdefault(name, {"attempts": 0, "passes": {}})
    return entry.setdefault(
        "sampling", {"streak": 0, "last_outcome": None, "closure": None, "last_completed_grade": None}
    )


def decide(
    ledger: dict,
    *,
    names,
    policy: dict,
    closures_now: dict,
    docs: dict,
    tier_state: dict | None,
    outcomes: dict | None,
    remaining_s,
    hint=(),
) -> tuple[dict, dict]:
    """Advance the grade counter and choose this grade's carried cases (``{case: info}``, report)."""
    knobs = settings(policy)
    state = _state(ledger)
    state["grade"] += 1
    grade = state["grade"]
    stable = knobs["stable_grades"] or max(
        knobs["min_stable_grades"], 1 + int(state["longest_streak_before_regression"])
    )
    lane = lanes(names, policy)
    tier_state = tier_state if isinstance(tier_state, dict) else {}
    outcomes = outcomes or {}
    hinted = set(hint or ())
    base, recency = [], {}
    for name in sorted(names):
        member = (ledger["members"].get(name) or {}).get("sampling") or {}
        newest = outcomes.get(name)
        if (
            lane[name] in knobs["cohorts"]
            and name not in hinted
            and member.get("last_outcome") == "pass"
            and int(member.get("streak") or 0) >= stable
            and member.get("closure") is not None
            and member.get("closure") == closures_now.get(name)
            and type(member.get("last_completed_grade")) is int
            and (newest is None or newest.get("passed") is True)
            and _cert_settled(tier_state.get(name))
        ):
            base.append(name)
            recency[name] = grade - member["last_completed_grade"]
    cover = rotating_cover(docs, base, recency) if base else []
    sweep_every = knobs["full_sweep_every"] or (math.ceil(len(base) / len(cover)) if cover else 1)
    # Stale: not actually graded within one sweep period (e.g. a sweep's admission window closed first).
    eligible = [n for n in base if recency[n] <= sweep_every]
    if len(eligible) != len(base):
        cover = rotating_cover(docs, eligible, recency) if eligible else []
    reasons = []
    if not knobs["enabled"]:
        reasons.append("disabled by policy")
    if grade - int(state["last_full_sweep_grade"] or 0) >= sweep_every:
        reasons.append(f"periodic full sweep (every {sweep_every} grades)")
    regression = state.get("regression_grade")
    if knobs["sweep_after_regression"] and type(regression) is int and regression >= grade - 1:
        reasons.append("a regression was observed in the previous grade")
    if type(remaining_s) in (int, float) and remaining_s <= knobs["final_window_s"]:
        reasons.append("final campaign window")
    if not eligible and not reasons:
        reasons.append("no case is stable yet")
    carried = (
        {}
        if reasons
        else {
            name: {
                "reason": REASON,
                "streak": ledger["members"][name]["sampling"]["streak"],
                "grades_since_graded": recency[name],
            }
            for name in eligible
            if name not in set(cover)
        }
    )
    if reasons:
        state["last_full_sweep_grade"] = grade
    decision = {
        "grade": grade,
        "eligible": len(eligible),
        "sample": len(cover) if not reasons else len(eligible),
        "carried": len(carried),
        "full_sweep": bool(reasons),
        "full_sweep_reasons": reasons,
    }
    state["decisions"] = (state["decisions"] + [decision])[-_KEEP_DECISIONS:]
    report = {
        "schema": "merlin.regression_sampling.v1",
        **decision,
        "stable_grades": stable,
        "full_sweep_every": sweep_every,
        "covering_sample": [] if reasons else cover,
        "cohorts": knobs["cohorts"],
        "status_label": LABEL,
        "scope": "carried cases keep their newest completed result from an earlier grade; they remain "
        "pending for the current source and never count as a pass of this grade",
        "certification_authority": False,
    }
    return carried, report


def record_completion(ledger: dict, name: str, *, passed: bool, closure, grade: int | None) -> None:
    """Advance one member's passing streak from an actually completed result (carried grades do not)."""
    member = _member(ledger, name)
    state = _state(ledger)
    if passed:
        member["streak"] = int(member.get("streak") or 0) + 1
        member["closure"] = closure
    else:
        if member.get("last_outcome") == "pass" and int(member.get("streak") or 0) > 0:
            state["longest_streak_before_regression"] = max(
                int(state["longest_streak_before_regression"]), int(member["streak"])
            )
            # The newest grade index, not the completing snapshot's: two snapshots may overlap and the
            # next decision must see the regression whichever of them reported it.
            state["regression_grade"] = state["grade"]
        member["streak"] = 0
        member["closure"] = None
    member["last_outcome"] = "pass" if passed else "fail"
    member["last_completed_grade"] = grade if type(grade) is int else state["grade"]


def carried_rows(selection) -> dict:
    """``{case: verdict row}`` for this grade's carried cases (no tiers: a carried row states no result)."""
    report = ((selection or {}).get("curriculum") or {}).get("sampling") or {}
    carried = ((selection or {}).get("curriculum") or {}).get("carried_sampled") or {}
    return {
        name: {
            "status": STATUS,
            "tiers": {},
            "carried": {
                "reason": REASON,
                "label": LABEL,
                "sampling_grade": report.get("grade"),
                **{k: v for k, v in (info or {}).items() if k != "reason"},
            },
        }
        for name, info in sorted(carried.items())
    }


# ---- grade_reuse -------------------------------------------------------------------------------
# Reuse one completed host grade inside its original authoring invocation.
#
# This is not a certificate store or a public-channel reader. The host retains an
# in-memory receipt over its own canonical artifacts and rechecks its original
# admission, source and cohort before omitting a redundant grade.


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("grade artifact is absent or a symlink")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _artifacts(root: Path) -> tuple[tuple[Path, str], ...]:
    files, size = [], 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("retained grade has linked artifacts")
        if path.is_file():
            size += path.stat().st_size
            if size > 256 << 20 or len(files) >= 8192:
                raise ValueError("retained grade exceeds bounded artifact inventory")
            files.append((path, _sha(path)))
    return tuple(files)


def _source(path: Path) -> str:
    from merlin.development.backend.optimization.search.retained_revisions import source_inventory

    return source_inventory(path.resolve(), max_bytes=32 << 20, max_files=8192, exclude_generated=True)["sha256"]


def _retained_program(artifacts) -> bool:
    from merlin.development.targetgen.grading.elf_lanes import PROGRAM_ARTIFACT_SUFFIX

    for path, _ in artifacts:
        if path.name.endswith(PROGRAM_ARTIFACT_SUFFIX):
            return True
        with path.open("rb") as stream:
            header = stream.read(20)
        if len(header) >= 18 and header[:4] == b"\x7fELF" and header[5] in (1, 2):
            endian = "little" if header[5] == 1 else "big"
            if int.from_bytes(header[16:18], endian) in (2, 3):
                return True
    return False


def _binding(inputs, run_dir: Path, public: Path) -> dict:
    from merlin.common.provenance import source_digest
    from merlin.development.backend.feedback.execution.profile_runtime import compiler_profile_binding
    from merlin.development.targetgen.oracles import tier_cache

    if inputs.compiler_profile_service is None or inputs.verify_grade_inputs is None:
        raise ValueError("no original host admission")
    inputs.verify_grade_inputs()
    owner = compiler_profile_binding(inputs.compiler_profile_service, run_dir=run_dir)
    files = tier_cache.grading_path(inputs.context.target)
    if not owner or not files:
        raise ValueError("original owner or grading instrument is unavailable")
    return {
        "owner": owner,
        "descriptor": _sha(Path(inputs.context.descriptor)),
        "public": hash_tree(public),
        "contract": hash_tree(Path(inputs.contract)) if inputs.contract is not None else None,
        "instrument": source_digest([str(p) for p in files]),
        # Simulator/protocol selection remains the original host's, including
        # environment controls which do not appear in a source-only digest.
        "selection": {key: value for key, value in os.environ.items() if key.startswith("MERLIN_")},
        "target": inputs.context.target,
        "forbidden": inputs.additional_forbidden,
        "policy_roots": [hash_tree(Path(p)) for p in getattr(inputs, "policy_roots", ())],
        "arm": getattr(inputs, "arm", None),
        "readback_policy": getattr(inputs.context, "readback_policy", None),
        "public_build_selection": json.loads(json.dumps(getattr(inputs, "public_build_selection", None))),
        "public_build_budget_s": getattr(inputs, "public_build_budget_s", 0),
        "rtl_facts": _sha(Path(inputs.rtl_facts)) if getattr(inputs, "rtl_facts", None) is not None else None,
    }


@dataclass(frozen=True)
class DriverGradeReceipt:
    candidate: Path
    runs: Path
    public: Path
    source_sha256: str
    binding: dict
    artifacts: tuple[tuple[Path, str], ...]
    rows: dict
    verdict_path: Path | None = None
    verdict_json: str | None = None
    oot_commit: str | None = None

    def current(self, *, submission: Path, inputs, run_dir: Path) -> bool:
        try:
            return (
                _source(submission) == self.source_sha256
                and _source(self.candidate) == self.source_sha256
                and _binding(inputs, run_dir, self.public) == self.binding
                and all(_sha(path) == digest for path, digest in self.artifacts)
            )
        except Exception:  # noqa: BLE001 -- unavailable reuse always falls back to grading
            return False


def retain_driver_grade(
    *, candidate, runs, public, verdict, inputs, run_dir, verdict_path=None, oot_commit=None, evidence_paths=()
):
    """Retain a complete ordinary grade, never a screen or a redacted claim."""
    from merlin.development.backend.feedback.execution.compiler_profile_child import _original_executor
    from merlin.development.targetgen.grading import boundary as ER
    from merlin.development.targetgen.grading import enforcement as _merlin_enforcement_1
    from merlin.development.targetgen.grading import model_checks as CG

    try:
        if verdict.get("all_pass") is not True or verdict.get("gradeable") is not True:
            return None
        candidate, runs, public, run_dir = map(Path, (candidate, runs, public, run_dir))
        # These paths are selected by LG.grade, not supplied by the author.
        if not candidate.resolve().is_relative_to(run_dir.resolve() / "_qa_work"):
            return None
        if not runs.resolve().is_relative_to(run_dir.resolve() / "_qa_work"):
            return None
        binding = _binding(inputs, run_dir, public)
        source = _source(candidate)
        if not source:
            return None
        executor = _original_executor()
        if (
            executor is None
            or executor.owner is not inputs.compiler_profile_service
            or executor.source_sha256 != source
        ):
            return None
        caps = capsule_capsule_common.discover_capsules(public, labels={"public", "dev"}, contract=inputs.contract)
        cohort = {cap["name"]: cap for cap in caps}
        if not cohort or len(cohort) != len(caps):
            return None
        paths = sorted((runs / "runs").glob("*/*/capsule_result.json"))
        artifacts, rows = [], {}
        for path in paths:
            raw = json.loads(path.read_text())
            name = raw.get("capsule")
            if name not in cohort or name in rows:
                return None
            cap = cohort[name]
            if ER.expects_refusal(cap):
                if not ER.validated_result(cap, raw):
                    return None
                row = {
                    "status": "pass",
                    "tiers": {},
                    "validated_refusal": True,
                    "qualification": "expected_refusal",
                    "native_executed": False,
                    "expected_refusal": raw["expected_refusal"],
                }
            else:
                child = raw.get("original_compiler_child")
                if child is not None and (
                    not isinstance(child, dict)
                    or child.get("schema") != "merlin.original_model_child.v1"
                    or child.get("candidate_sha256") != source
                    or any(child.get(key) != binding["owner"][key] for key in ("profile_sha256", "snapshot_sha256"))
                ):
                    return None
                view = _merlin_enforcement_1._candidate_score_view(raw)
                if raw.get("kind") == "model":
                    check = _merlin_enforcement_1._qualified_model_check(raw)
                    if not isinstance(check, dict) or check.get("status") != "pass":
                        return None
                required = set(cap.get("required_oracle_tiers") or [])
                tiers = {
                    tier: record.get("status")
                    for tier, record in (view.get("tiers") or {}).items()
                    if isinstance(record, dict)
                }
                if (
                    view.get("status") != "pass"
                    or view.get("failure")
                    or not required
                    or any(tiers.get(tier) != "pass" for tier in required)
                ):
                    return None
                row = {"status": "pass", "tiers": tiers}
            retained = _artifacts(path.parent)
            if not ER.expects_refusal(cap) and not _retained_program(retained):
                return None
            artifacts.extend(retained)
            rows[name] = row
        if set(rows) != set(cohort):
            return None
        # The completed archive is written by the original grader after all
        # shape/resource checks, separately from its pre-probe snapshot archive.
        payload, completed = None, None
        if verdict_path is not None:
            completed = Path(verdict_path)
            payload = completed.read_text()
            if json.loads(payload) != json.loads(json.dumps(verdict, default=str)):
                return None
        for path in (*evidence_paths, *((completed,) if completed is not None else ())):
            path = Path(path)
            if path.resolve() != path or not path.is_relative_to(run_dir.resolve()):
                return None
            artifacts.extend(_artifacts(path) if path.is_dir() else ((path, _sha(path)),))
        return DriverGradeReceipt(
            candidate, runs, public, source, binding, tuple(artifacts), rows, completed, payload, oot_commit
        )
    except Exception:  # noqa: BLE001 -- optional evidence retention is never a grade gate
        return None


def reuse_checkpoint(receipt, *, submission, inputs, run_dir, eligible, held, cert_tier, attempt):
    """Supply the checkpoint's tally from the same retained full-grade evidence."""
    from merlin.development.backend.feedback.certification.tier_policy import _l3_attempt_tally

    if (
        type(receipt) is not DriverGradeReceipt
        or held
        or set(eligible) != set(receipt.rows)
        or len(eligible) != len(receipt.rows)
        or not receipt.current(submission=Path(submission), inputs=inputs, run_dir=Path(run_dir))
    ):
        return None
    rows, passed, count = _l3_attempt_tally(receipt.rows, [], cert_tier)
    if passed != count or not count:
        return None
    return {
        "attempt": attempt,
        "all_pass": True,
        "n_passed": passed,
        "n_capsules": count,
        "n_rtl_passed": sum(row.get("l3_status") == "pass" for row in rows),
        "n_expected_refusal_passed": sum(row.get("validated_refusal") is True for row in rows),
        "n_not_promoted": 0,
        "cert_tier": cert_tier,
        "per_capsule": rows,
        "rtl_checks": None,
        "evidence_reuse": {
            "stage": "completed_driver_full_grade",
            "measured_now": False,
            "runs_root": str(receipt.runs),
            "source_sha256": receipt.source_sha256,
        },
    }


def reuse_post_turn(receipt, *, ws, run_dir, rnd, inputs):
    """Reopen a completed original full grade, preserving its measured evidence.

    Only the host's in-memory receipt can select this path. Author verdict files,
    fast screens, partial grades and missing final archives cannot authorize it.
    Formal/public/hidden certification remains a separate unchanged consumer.
    """
    from merlin.common import oot_repository as O
    from merlin.development.backend.feedback.grading import fast_grade as _merlin_fast_grade_2
    from merlin.development.backend.feedback.grading import loop_grade as LG
    from merlin.development.backend.session import workspaces as OH

    try:
        if type(receipt) is not DriverGradeReceipt or receipt.verdict_json is None:
            return None
        ws, run_dir = Path(ws), Path(run_dir)
        verdict = json.loads(receipt.verdict_json)
        if (
            verdict.get("all_pass") is not True
            or verdict.get("gradeable") is not True
            or verdict.get("snapshot_current") is not True
            or verdict.get("current_source_eligible") is not True
            or not receipt.current(submission=ws / "submission", inputs=inputs, run_dir=run_dir)
        ):
            return None
        repo = OH.repo_path(run_dir)
        if repo.exists() and (
            receipt.oot_commit is None
            or O.resolve(repo, "HEAD") != receipt.oot_commit
            or O.tree_digest(repo, receipt.oot_commit) != O.package_digest(receipt.candidate)
        ):
            return None
        # A round still records its exact source under the ordinary OOT owner.
        # This checkpoint is not a new simulator measurement.
        key = f"{rnd:02d}"
        committed = OH.commit_graded(run_dir, receipt.candidate, label="round", key=key, sandbox_roots=(ws.parent,))
        if repo.exists() and committed is None:
            return None
        verdict["evidence_reuse"] = {
            "stage": "completed_driver_full_grade",
            "measured_now": False,
            "runs_root": str(receipt.runs),
            "source_sha256": receipt.source_sha256,
            "completed_verdict": {"path": str(receipt.verdict_path), "sha256": _sha(receipt.verdict_path)},
        }
        if not receipt.current(submission=ws / "submission", inputs=inputs, run_dir=run_dir):
            return None
    except Exception:  # noqa: BLE001 -- changed/unavailable evidence takes ordinary grading
        return None
    # Publication failures propagate; never hide a half-published round by
    # starting another expensive grade against the same owned evidence.
    _merlin_fast_grade_2.write_verdict(run_dir / "qa_history" / f"verdict_round_{key}.json", verdict)
    verdict = _merlin_fast_grade_2.write_verdict(ws / "qa" / "verdict.json", verdict)
    OH.record_round(run_dir, committed, label="round", key=key, verdict=verdict)
    print(f"[round {rnd}] reused completed current-source full grade; no numerical jobs started", flush=True)
    return {**verdict, **({"oot_commit": committed.commit} if committed is not None else {})}


def post_turn_grade(receipt, ws, run_dir, rnd, no_oracle, timeout, *, inputs, grade_callback):
    """Called only after the background owner has drained; no parallel grade."""
    verdict = None if no_oracle else reuse_post_turn(receipt, ws=ws, run_dir=run_dir, rnd=rnd, inputs=inputs)
    return verdict if verdict is not None else grade_callback(ws, run_dir, rnd, no_oracle, timeout)


__all__ = [
    "DEFAULTS",
    "LABEL",
    "SECTION",
    "STATUS",
    "carried_rows",
    "closures",
    "decide",
    "load_docs",
    "record_completion",
    "rotating_cover",
    "settings",
    "validate",
]
