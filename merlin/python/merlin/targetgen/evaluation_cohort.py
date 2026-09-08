"""Materialize and grade a descriptor-declared post-search evaluation cohort.

Search and evaluation answer different questions.  The paid search cohort may provide iterative
feedback at a fast oracle tier; this module creates a fresh, immutable view for a named evaluation
stage and makes that stage's oracle tier mandatory in the copied capsule contracts.  It therefore
cannot silently turn an optional GSIM observation into a certified result.

The source capsules are never changed.  The emitted manifest records the descriptor, source and
materialized tree digests, stage ordering, and the exact engine preflight used for the run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml

from merlin.common.paths import repo_root
from .contract.materialize import materialize_public_capsules
from .target_experiment import TargetExperiment, descriptor_for, load_target_experiment


def _tree_sha256(
    root: Path, *, exclude: frozenset[str] = frozenset(),
    exclude_python_cache: bool = False,
) -> str:
    """Content-and-relative-path digest for a symlink-free capsule/cohort tree."""
    rows: list[bytes] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"evaluation input contains a symlink: {path}")
        if relative.as_posix() in exclude:
            continue
        # Importing a frozen Python package creates interpreter caches.  Those are neither compiler
        # source nor a stable part of an evaluation artifact, and including them makes validation fail
        # merely because grading executed the candidate.  Only these well-defined Python byproducts are
        # ignored; every authored/source/config/binary byte remains in the digest.
        if exclude_python_cache and (
            "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        if not path.is_file():
            continue
        rel = relative.as_posix().encode("utf-8")
        rows.append(rel + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return hashlib.sha256(b"\n".join(rows)).hexdigest()


def _candidate_tree_sha256(root: Path) -> str:
    return _tree_sha256(root, exclude_python_cache=True)


def _file_sha256(path: Path, *, what: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{what} is not a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_source_identity(path: Path) -> dict[str, str]:
    """Best-effort source citation; executable/config byte hashes remain authoritative."""
    try:
        root = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"path": str(path.resolve())}
    return {"path": str(path.resolve()), "git_root": str(Path(root).resolve()),
            "git_commit": commit}


def configured_executable_binding(
    *, engine: str, binary: str | Path, source: str | Path,
    config: str | Path, config_tree: str | Path,
) -> dict[str, Any]:
    """Content-address one executable and every configuration byte it may consume.

    This is target-neutral provenance machinery.  The backend owns which paths constitute its engine;
    this helper merely refuses aliases/symlinks and binds regular bytes, relative names, and (when
    resolvable) the source commit that produced the executable.
    """
    binary, source, config, config_tree = (
        Path(binary), Path(source), Path(config), Path(config_tree))
    if not engine:
        raise ValueError("configured executable binding requires an engine name")
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"engine source is not a regular directory: {source}")
    if config_tree.is_symlink() or not config_tree.is_dir():
        raise ValueError(f"engine config tree is not a regular directory: {config_tree}")
    binding: dict[str, Any] = {
        "schema": "configured_executable_binding_v1",
        "engine": engine,
        "binary": {
            "path": str(binary.resolve()),
            "sha256": _file_sha256(binary, what=f"{engine} executable"),
        },
        "source": _git_source_identity(source),
        "config": {
            "path": str(config.resolve()),
            "sha256": _file_sha256(config, what=f"{engine} config"),
            "tree": str(config_tree.resolve()),
            "tree_sha256": _tree_sha256(config_tree),
        },
    }
    binding["binding_sha256"] = _canonical_json_sha256(binding)
    return binding


def cyclotron_l2_engine_binding(target: str) -> dict[str, Any]:
    """Resolve the exact Cyclotron executable/config identity used by the Muon L2 adapter."""
    from merlin.runtime.backends.base import get_backend

    # Cyclotron is the ENGINE here; the target being EVALUATED is the `target` parameter, recorded
    # as binding["target"] and checked by _validate_l2_engine_binding.
    backend = get_backend("muon")  # target-ok: cyclotron is the muon backend's simulator, and the evaluated target is the `target` parameter
    binary = backend.cyclotron_path().resolve()
    # An explicit executable override may come from a different checkout than the timing tree.  Cite
    # both honestly: the binary's source tree is inferred from its canonical Cargo output layout while
    # config_tree is the directory the runtime actually links into each work directory.
    source = binary.parent.parent.parent if binary.parent.name == "release" else backend.cyclotron_root()
    binding = configured_executable_binding(
        engine="cyclotron",
        binary=binary,
        source=source,
        config=backend.config_path(),
        config_tree=backend.cyclotron_root() / "config",
    )
    binding["target"] = target
    binding["binding_sha256"] = _canonical_json_sha256(
        {key: value for key, value in binding.items() if key != "binding_sha256"})
    return binding


def _validate_l2_engine_binding(binding: Any, *, target: str) -> dict[str, Any]:
    if (not isinstance(binding, dict)
            or binding.get("schema") != "configured_executable_binding_v1"
            or binding.get("engine") != "cyclotron"
            or binding.get("target") != target):
        raise ValueError("search score has no valid Cyclotron L2 engine binding")
    expected_sha = binding.get("binding_sha256")
    unsigned = {key: value for key, value in binding.items() if key != "binding_sha256"}
    if _canonical_json_sha256(unsigned) != expected_sha:
        raise ValueError("Cyclotron L2 engine binding record digest mismatch")
    binary = binding.get("binary")
    config = binding.get("config")
    if not isinstance(binary, dict) or not isinstance(config, dict):
        raise ValueError("Cyclotron L2 engine binding is malformed")
    if _file_sha256(Path(str(binary.get("path", ""))), what="bound Cyclotron executable") \
            != binary.get("sha256"):
        raise ValueError("bound Cyclotron executable content digest mismatch")
    if _file_sha256(Path(str(config.get("path", ""))), what="bound Cyclotron config") \
            != config.get("sha256"):
        raise ValueError("bound Cyclotron config content digest mismatch")
    tree = Path(str(config.get("tree", "")))
    if tree.is_symlink() or not tree.is_dir() or _tree_sha256(tree) != config.get("tree_sha256"):
        raise ValueError("bound Cyclotron config tree content digest mismatch")
    return binding


def _source_capsules(te: TargetExperiment) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for root in te.graded_roots():
        for path in sorted(root.glob("*/capsule.yaml")):
            name = path.parent.name
            if name in out:
                raise ValueError(f"duplicate capsule {name!r} across {te.graded_roots()}")
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if doc.get("label") in {"public", "dev"}:
                out[name] = path.parent
    return out


def _search_source_records(te: TargetExperiment) -> list[dict[str, str]]:
    """Identity of the exact application-derived trees admitted to the paid search loop."""
    sources = _source_capsules(te)
    include = tuple(te.graded_include)
    if not include:
        raise ValueError(f"target {te.target!r} declares no search cohort")
    missing = sorted(set(include) - set(sources))
    if missing:
        raise ValueError(f"search cohort names absent capsules: {missing}")
    records: list[dict[str, str]] = []
    for name in include:
        source = sources[name]
        doc = yaml.safe_load((source / "capsule.yaml").read_text(encoding="utf-8")) or {}
        role = str(doc.get("source_role") or "")
        if role != "model_derived":
            raise ValueError(
                f"search cohort is not exclusively model-derived: {name} has source_role={role!r}")
        records.append({"name": name, "source_role": role,
                        "source_tree_sha256": _tree_sha256(source)})
    return sorted(records, key=lambda row: row["name"])


def _search_score_problems(score: Any, expected_names: list[str], *, target: str) -> list[str]:
    """Accept only an exact, non-vacuous L2 numeric pass from the self-check score schema.

    Search deliberately stops at Cyclotron L2.  Successful self-check rows are compact: ``pass`` and
    ``barrier_status`` are the numeric verdict, while ``execution_digest`` binds the executed artifact.
    Failed rows retain the detailed ``numeric`` block.  Requiring a field that successful rows omit
    would make a genuine exact all-pass impossible to seal, so this validator follows that schema
    explicitly.  The expected count is derived from the declared cohort rather than duplicated here.
    """
    if not isinstance(score, dict):
        return ["search score is not a JSON object"]
    expected_n = len(expected_names)
    problems: list[str] = []
    if score.get("n_capsules") != expected_n or score.get("n_passed") != expected_n:
        problems.append(
            f"score is not an exact all-pass ({score.get('n_passed')}/{score.get('n_capsules')}, "
            f"expected {expected_n}/{expected_n})")
    if score.get("all_pass") is not True:
        problems.append("score all_pass is not true")
    if score.get("n_certified") != expected_n:
        problems.append("score does not certify every admitted search capsule")
    if score.get("n_unchecked") not in (None, 0):
        problems.append("score contains unchecked search capsules")
    rows = score.get("per_capsule")
    if not isinstance(rows, list):
        problems.append("score has no per_capsule evidence")
        rows = []
    row_names = [row.get("capsule") for row in rows if isinstance(row, dict)]
    if len(row_names) != len(set(row_names)) or sorted(row_names) != expected_names:
        problems.append("score per_capsule names do not exactly cover the search cohort")
    engine_bindings: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            problems.append("score contains a malformed per_capsule row")
            continue
        name = row.get("capsule", "<unnamed>")
        if (row.get("pass") is not True or row.get("barrier_tier") != "L2"
                or row.get("barrier_status") != "pass"):
            problems.append(f"{name} is not a recorded L2 numeric pass")
        digest = row.get("execution_digest")
        try:
            valid_digest = isinstance(digest, str) and len(digest) == 64 and int(digest, 16) >= 0
        except ValueError:
            valid_digest = False
        if not valid_digest:
            problems.append(f"{name} has no execution digest")
        cycles = row.get("barrier_cycles")
        if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles <= 0:
            problems.append(f"{name} has no positive measured L2 barrier cycle count")
        binding = row.get("barrier_engine_binding")
        try:
            engine_bindings.append(_validate_l2_engine_binding(binding, target=target))
        except ValueError as exc:
            problems.append(f"{name} has invalid L2 engine provenance: {exc}")
    if engine_bindings and any(binding != engine_bindings[0] for binding in engine_bindings[1:]):
        problems.append("search capsules were measured by different L2 engine/config identities")
    return problems


def _sealed_predecessor_cycle_policy(stage: dict[str, Any]) -> dict[str, Any]:
    """Validated descriptor policy for turning one sealed predecessor count into an L3 cap."""
    policy = stage.get("cycle_budget")
    if not isinstance(policy, dict) or policy.get("source") != "sealed_predecessor_tier":
        raise ValueError("post-search evaluation requires cycle_budget.source=sealed_predecessor_tier")
    out = {"source": "sealed_predecessor_tier"}
    for key in ("multiplier", "minimum_cycles", "compute_floor_multiplier"):
        value = policy.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"post-search evaluation cycle_budget.{key} must be a positive integer")
        out[key] = value
    return out


def create_search_pass_seal(
    dest: str | Path, te: TargetExperiment, candidate: str | Path, score: str | Path,
) -> dict[str, Any]:
    """Seal one exact all-pass L2 result with the candidate and source-tree identities.

    The seal lives outside the candidate so creating it cannot change the digest it freezes.  It is a
    gate for this admitted covering set only; it is deliberately not an end-to-end readiness claim.
    """
    dest, candidate, score_path = Path(dest), Path(candidate), Path(score)
    if dest.exists():
        raise ValueError(f"search pass seal destination already exists: {dest}")
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError(f"search candidate is not a regular directory: {candidate}")
    if not any(path.is_file() for path in candidate.rglob("*")):
        raise ValueError(f"search candidate has no files: {candidate}")
    candidate_resolved = candidate.resolve()
    seal_resolved = dest.resolve()
    if seal_resolved == candidate_resolved or candidate_resolved in seal_resolved.parents:
        raise ValueError("search pass seal must live outside the candidate it freezes")
    score_sha256 = _file_sha256(score_path, what="search score")
    score_doc = json.loads(score_path.read_text(encoding="utf-8"))
    source_records = _search_source_records(te)
    names = [row["name"] for row in source_records]
    problems = _search_score_problems(score_doc, names, target=te.target)
    if problems:
        raise ValueError("search pass evidence rejected: " + "; ".join(problems))
    score_binding = score_doc["per_capsule"][0]["barrier_engine_binding"]
    if score_binding != cyclotron_l2_engine_binding(te.target):
        raise ValueError(
            "search pass evidence rejected: measured Cyclotron executable/config identity differs "
            "from the current L2 engine")
    record = {
        "schema": "descriptor_search_pass_v3",
        "claim_scope": "admitted_search_covering_set_not_e2e_readiness",
        "target": te.target,
        "policy": te.graded_cohort_policy,
        "descriptor": str(te.path.relative_to(repo_root())),
        "descriptor_sha256": te.descriptor_sha256,
        "required_oracle_tier": "L2",
        "candidate": str(candidate_resolved),
        "candidate_tree_sha256": _candidate_tree_sha256(candidate),
        "score": str(score_path.resolve()),
        "score_sha256": score_sha256,
        "capsules": source_records,
        "l2_cycles": {str(row["capsule"]): int(row["barrier_cycles"])
                      for row in score_doc["per_capsule"]},
        "l2_engine_binding": score_binding,
        "n_capsules": len(names),
        "n_passed": len(names),
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def validate_search_pass_seal(
    path: str | Path, te: TargetExperiment, candidate: str | Path,
) -> dict[str, Any]:
    path, candidate = Path(path), Path(candidate)
    seal_sha256 = _file_sha256(path, what="search pass seal")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != "descriptor_search_pass_v3":
        raise ValueError(f"unsupported search pass seal: {path}")
    if (record.get("target") != te.target or record.get("descriptor_sha256") != te.descriptor_sha256
            or record.get("policy") != te.graded_cohort_policy):
        raise ValueError("search pass seal does not belong to the loaded target descriptor")
    if record.get("claim_scope") != "admitted_search_covering_set_not_e2e_readiness":
        raise ValueError("search pass seal overstates or omits its admitted-covering-set scope")
    if record.get("required_oracle_tier") != "L2":
        raise ValueError("search pass seal is not an L2 pass")
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError(f"frozen search candidate is unavailable: {candidate}")
    if Path(str(record.get("candidate", ""))).resolve() != candidate.resolve():
        raise ValueError("search pass seal names a different candidate")
    candidate_digest = _candidate_tree_sha256(candidate)
    if candidate_digest != record.get("candidate_tree_sha256"):
        raise ValueError("frozen search candidate content digest mismatch")
    expected_records = _search_source_records(te)
    if record.get("capsules") != expected_records:
        raise ValueError("search pass seal capsule source trees differ from the current descriptor")
    expected_n = len(expected_records)
    if record.get("n_capsules") != expected_n or record.get("n_passed") != expected_n:
        raise ValueError("search pass seal is not an exact all-pass")
    cycle_names = record.get("l2_cycles")
    if (not isinstance(cycle_names, dict) or sorted(cycle_names) !=
            sorted(row["name"] for row in expected_records)
            or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
                   for value in cycle_names.values())):
        raise ValueError("search pass seal lacks exact positive per-capsule L2 cycles")
    score_path = Path(str(record.get("score", "")))
    if _file_sha256(score_path, what="sealed search score") != record.get("score_sha256"):
        raise ValueError("search score evidence digest mismatch")
    score_doc = json.loads(score_path.read_text(encoding="utf-8"))
    problems = _search_score_problems(
        score_doc, [row["name"] for row in expected_records], target=te.target)
    if problems:
        raise ValueError("sealed search score no longer proves the pass: " + "; ".join(problems))
    score_cycles = {str(row["capsule"]): int(row["barrier_cycles"])
                    for row in score_doc["per_capsule"]}
    if record.get("l2_cycles") != score_cycles:
        raise ValueError("search pass seal L2 cycle map differs from its hash-bound score evidence")
    score_binding = score_doc["per_capsule"][0]["barrier_engine_binding"]
    if record.get("l2_engine_binding") != score_binding:
        raise ValueError("search pass seal L2 engine differs from its hash-bound score evidence")
    current_binding = cyclotron_l2_engine_binding(te.target)
    if score_binding != current_binding:
        raise ValueError("sealed Cyclotron executable/config identity differs from the current engine")
    return {**record, "seal": str(path.resolve()), "seal_sha256": seal_sha256}


def engine_preflight(te: TargetExperiment, stage_name: str) -> dict[str, Any]:
    """Resolve the declared evaluation engine and reject unattributed executable bytes."""
    from .capsule_runner import describe_l3_engine

    stage = te.evaluation_cohort(stage_name)
    requested = str(stage["oracle_engine"])
    tier = str(stage["oracle_tier"])
    if tier != "L3":
        # L3 is the only post-loop engine selector currently exposed by the target-neutral runner.
        # Refuse instead of pretending the L3 selection says anything about another tier.
        raise ValueError(f"evaluation engine preflight currently supports L3, got {tier}")
    selected = describe_l3_engine(te.target, te.sim_via)
    reason = str(selected.get("reason") or "")
    problems = []
    if selected.get("available") is not True:
        problems.append("declared evaluation engine is unavailable")
    if selected.get("engine") != requested:
        problems.append(
            f"descriptor requests {requested!r}, resolver selected {selected.get('engine')!r}")
    if selected.get("fidelity") != "elaborated_rtl":
        problems.append("selected evaluation engine is not classified as elaborated RTL")
    if "UNRECORDED" in reason or "no build receipt" in reason.lower():
        problems.append("selected engine executable is not tied to an RTL build receipt")
    binding = None
    if requested == "gsim":
        from . import gsim_emulator

        # Evaluation is deliberately bound to the installed canonical engine, never a target-specific
        # environment override.  This keeps the workflow target-neutral and makes the recorded path the
        # same path future validation re-hashes.
        citation = gsim_emulator.citation(te.target)
        canonical_binary = (gsim_emulator.gsim_home(te.target) / gsim_emulator.BINARY_NAME).resolve()
        binary_path = Path(str(citation.get("path") or ""))
        receipt = citation.get("receipt")
        receipt_path = Path(str((receipt or {}).get("receipt_path") or ""))
        if citation.get("available") is not True or citation.get("engine") != "gsim":
            problems.append("canonical GSIM resolver does not report an available GSIM binary")
        if citation.get("reason") != selected.get("reason"):
            problems.append("engine policy and canonical GSIM resolver selected different bytes")
        if citation.get("receipt_status") != "bound" or not isinstance(receipt, dict):
            problems.append("canonical GSIM binary lacks a bound build receipt")
        if binary_path.resolve() != canonical_binary:
            problems.append(
                f"GSIM selection is not the canonical install ({binary_path} != {canonical_binary})")
        try:
            binary_sha256 = _file_sha256(binary_path, what="canonical GSIM binary")
            receipt_sha256 = _file_sha256(receipt_path, what="canonical GSIM receipt")
        except ValueError as exc:
            problems.append(str(exc))
        else:
            if binary_sha256 != citation.get("binary_sha256"):
                problems.append("canonical GSIM binary digest differs from its resolver citation")
            if binary_sha256 != (receipt or {}).get("binary_sha256"):
                problems.append("canonical GSIM receipt does not bind the selected binary digest")
            if receipt_sha256 != (receipt or {}).get("receipt_sha256"):
                problems.append("canonical GSIM receipt digest differs from its resolver citation")
            binding = {
                "engine": "gsim",
                "binary": str(binary_path.resolve()),
                "binary_sha256": binary_sha256,
                "receipt": str(receipt_path.resolve()),
                "receipt_sha256": receipt_sha256,
                "receipt_status": citation.get("receipt_status"),
                "receipt_identity": receipt,
            }
            binding["binding_sha256"] = _canonical_json_sha256(binding)
    return {
        "stage": stage_name,
        "required_tier": tier,
        "requested_engine": requested,
        "selected": selected,
        "engine_binding": binding,
        "ok": not problems,
        "problems": problems,
    }


def _validate_engine_binding(
    preflight: Any, *, expected_engine: str, target: str,
) -> dict[str, Any]:
    if not isinstance(preflight, dict) or preflight.get("ok") is not True:
        raise ValueError("evaluation engine preflight is not clean")
    binding = preflight.get("engine_binding")
    if not isinstance(binding, dict) or binding.get("engine") != expected_engine:
        raise ValueError("evaluation manifest has no binding for its declared engine")
    if expected_engine == "gsim":
        from . import gsim_emulator

        canonical = (gsim_emulator.gsim_home(target) / gsim_emulator.BINARY_NAME).resolve()
        if Path(str(binding.get("binary", ""))).resolve() != canonical:
            raise ValueError("evaluation engine binding is not the canonical GSIM install")
    expected_binding_sha = binding.get("binding_sha256")
    unsigned = {key: value for key, value in binding.items() if key != "binding_sha256"}
    if _canonical_json_sha256(unsigned) != expected_binding_sha:
        raise ValueError("evaluation engine binding record digest mismatch")
    if _file_sha256(Path(str(binding.get("binary", ""))), what="bound engine binary") \
            != binding.get("binary_sha256"):
        raise ValueError("bound engine binary content digest mismatch")
    if _file_sha256(Path(str(binding.get("receipt", ""))), what="bound engine receipt") \
            != binding.get("receipt_sha256"):
        raise ValueError("bound engine receipt content digest mismatch")
    receipt = binding.get("receipt_identity")
    if (not isinstance(receipt, dict) or binding.get("receipt_status") != "bound"
            or receipt.get("binary_sha256") != binding.get("binary_sha256")
            or receipt.get("receipt_sha256") != binding.get("receipt_sha256")
            or Path(str(receipt.get("receipt_path", ""))).resolve()
            != Path(str(binding.get("receipt", ""))).resolve()):
        raise ValueError("bound engine receipt identity does not match its executable")
    return binding


def _predecessor_pass_evidence(
    te: TargetExperiment,
    stage_name: str,
    candidate: Path,
    predecessor_cohort: str | Path | None,
    predecessor_score: str | Path | None,
) -> dict[str, Any] | None:
    """Validate the physical pass that gates a dependent evaluation stage.

    A descriptor's ``after: <stage>_pass`` is an executable dependency, not documentation.  The
    predecessor score must cover exactly the predecessor cohort, clear its mandatory physical tier,
    and name the same byte-frozen package.  Both evidence files remain content-addressed dependencies
    of the new cohort so editing a result after materialization is detected.
    """
    after = str(te.evaluation_cohort(stage_name)["after"])
    if after == "search_l2_pass":
        raise ValueError(
            f"evaluation stage {stage_name!r} requires a search-pass seal, not evaluation-stage "
            "predecessor evidence")
    if not after.endswith("_pass"):
        raise ValueError(f"evaluation stage {stage_name!r} has unsupported dependency {after!r}")
    predecessor_stage = after.removesuffix("_pass")
    if predecessor_cohort is None or predecessor_score is None:
        raise ValueError(
            f"evaluation stage {stage_name!r} requires --predecessor-cohort and "
            f"--predecessor-score proving {predecessor_stage!r} passed")

    cohort_root = Path(predecessor_cohort)
    score_path = Path(predecessor_score)
    if cohort_root.is_symlink() or not cohort_root.is_dir():
        raise ValueError(f"predecessor cohort is not a regular directory: {cohort_root}")
    if score_path.is_symlink() or not score_path.is_file():
        raise ValueError(f"predecessor score is not a regular file: {score_path}")
    prior = validate_evaluation_cohort(cohort_root, te, candidate)
    if prior.get("stage") != predecessor_stage:
        raise ValueError(
            f"predecessor cohort is stage {prior.get('stage')!r}, expected {predecessor_stage!r}")

    score = json.loads(score_path.read_text(encoding="utf-8"))
    expected_names = sorted(row["name"] for row in prior["capsules"])
    expected_n = len(expected_names)
    problems: list[str] = []
    package = score.get("package")
    if not isinstance(package, str) or Path(package).resolve() != candidate.resolve():
        problems.append("score package does not resolve to the frozen candidate")
    if score.get("integrity_status") != "clean":
        problems.append("score integrity_status is not clean")
    if score.get("gradeable") is not True:
        problems.append("score is not gradeable")
    if score.get("n_capsules") != expected_n or score.get("n_passed") != expected_n or expected_n == 0:
        problems.append(
            f"score is not an exact all-pass ({score.get('n_passed')}/{score.get('n_capsules')}, "
            f"expected {expected_n}/{expected_n})")

    rows = score.get("per_capsule")
    if not isinstance(rows, list):
        problems.append("score has no per_capsule evidence")
        rows = []
    row_names = [row.get("capsule") for row in rows if isinstance(row, dict)]
    if len(row_names) != len(set(row_names)) or sorted(row_names) != expected_names:
        problems.append("score per_capsule names do not exactly cover the predecessor cohort")
    required_tier = str(prior["required_oracle_tier"])
    for row in rows:
        if not isinstance(row, dict):
            problems.append("score contains a malformed per_capsule row")
            continue
        tier_record = (row.get("tiers") or {}).get(required_tier)
        tier_status = tier_record.get("status") if isinstance(tier_record, dict) else tier_record
        if (row.get("status") != "pass" or tier_status != "pass"
                or (row.get("numeric") or {}).get("status") != "pass"):
            problems.append(
                f"{row.get('capsule', '<unnamed>')} did not pass numeric grading and required tier "
                f"{required_tier}")
    if (score.get("pass_evidence") or {}).get("rtl_backed") != expected_n:
        problems.append("not every predecessor pass is backed by elaborated-RTL evidence")
    if problems:
        raise ValueError("predecessor pass evidence rejected: " + "; ".join(problems))

    return {
        "stage": predecessor_stage,
        "cohort": str(cohort_root.resolve()),
        "cohort_manifest_sha256": hashlib.sha256(
            (cohort_root / ".evaluation_cohort.json").read_bytes()).hexdigest(),
        "score": str(score_path.resolve()),
        "score_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
        "candidate_tree_sha256": prior["candidate_tree_sha256"],
        "materialized_tree_sha256": prior["materialized_tree_sha256"],
        "engine_binding_sha256": prior["engine_preflight"]["engine_binding"]["binding_sha256"],
        "search_pass_seal_sha256": (
            (prior.get("search_pass_evidence") or {}).get("seal_sha256")),
        "required_oracle_tier": required_tier,
        "n_capsules": expected_n,
        "n_passed": expected_n,
        "rtl_backed": expected_n,
    }


def materialize_evaluation_cohort(
    dest: str | Path,
    te: TargetExperiment,
    stage_name: str,
    candidate: str | Path,
    *,
    search_pass_seal: str | Path | None = None,
    predecessor_cohort: str | Path | None = None,
    predecessor_score: str | Path | None = None,
) -> dict[str, Any]:
    """Create one descriptor-declared frozen-candidate cohort at ``dest``.

    ``dest`` must be absent or empty.  Refusing an existing populated directory avoids combining two
    descriptor revisions or leaving stale capsules in the denominator.
    """
    stage = te.evaluation_cohort(stage_name)
    dest = Path(dest)
    candidate = Path(candidate)
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError(f"frozen evaluation candidate is not a regular directory: {candidate}")
    if not any(path.is_file() for path in candidate.rglob("*")):
        raise ValueError(f"frozen evaluation candidate has no files: {candidate}")
    candidate_digest = _candidate_tree_sha256(candidate)
    after = str(stage["after"])
    search_evidence = None
    predecessor = None
    if after == "search_l2_pass":
        if predecessor_cohort is not None or predecessor_score is not None:
            raise ValueError(
                f"evaluation stage {stage_name!r} follows the sealed search pass, not another "
                "evaluation cohort")
        if search_pass_seal is None:
            raise ValueError(
                f"evaluation stage {stage_name!r} requires --search-pass-seal proving the exact "
                "admitted cohort passed L2")
        search_evidence = validate_search_pass_seal(search_pass_seal, te, candidate)
        if tuple(stage["include_capsules"]) != tuple(te.graded_include):
            raise ValueError(
                f"evaluation stage {stage_name!r} does not contain exactly the search cohort")
        cycle_policy = _sealed_predecessor_cycle_policy(stage)
    else:
        if search_pass_seal is not None:
            raise ValueError(
                f"evaluation stage {stage_name!r} does not directly consume a search-pass seal")
        predecessor = _predecessor_pass_evidence(
            te, stage_name, candidate, predecessor_cohort, predecessor_score)
        cycle_policy = None
    preflight = engine_preflight(te, stage_name)
    if preflight.get("ok") is not True:
        raise ValueError(
            "evaluation engine preflight rejected materialization: "
            + "; ".join(preflight.get("problems") or ["unknown engine preflight failure"]))
    current_binding = _validate_engine_binding(
        preflight, expected_engine=str(stage["oracle_engine"]), target=te.target)
    if predecessor is not None and predecessor.get("engine_binding_sha256") \
            != current_binding.get("binding_sha256"):
        raise ValueError("evaluation engine binding changed since the predecessor GSIM pass")
    if dest.exists() and any(dest.iterdir()):
        raise ValueError(f"evaluation destination is not empty: {dest}")
    dest.mkdir(parents=True, exist_ok=True)

    sources = _source_capsules(te)
    include = tuple(stage["include_capsules"])
    missing = sorted(set(include) - set(sources))
    if missing:
        raise ValueError(f"evaluation cohort {stage_name!r} names absent capsules: {missing}")
    required_roles = set(stage.get("require_source_roles") or ())
    source_records = []
    for name in include:
        src = sources[name]
        doc = yaml.safe_load((src / "capsule.yaml").read_text(encoding="utf-8")) or {}
        role = str(doc.get("source_role") or "")
        if required_roles and role not in required_roles:
            raise ValueError(
                f"evaluation cohort {stage_name!r} requires source role(s) "
                f"{sorted(required_roles)}, but {name} is {role!r}")
        source_records.append({"name": name, "source_role": role,
                               "source_tree_sha256": _tree_sha256(src)})
    source_records = sorted(source_records, key=lambda row: row["name"])
    if search_evidence is not None and source_records != search_evidence["capsules"]:
        raise ValueError(
            "derived GSIM source trees/shapes differ from the sealed search cohort")

    excluded = set(sources) - set(include)
    written = materialize_public_capsules(
        dest, tier_ceiling=str(stage["oracle_tier"]), corpus_roots=te.graded_roots(),
        exclude=excluded,
    )
    if set(written) != set(include):
        raise ValueError(
            f"materialized cohort differs from declaration: wrote={written}, declared={sorted(include)}")

    # This is a POST-SEARCH certification contract.  The source capsule intentionally leaves L3
    # optional so cheap search can use its sibling; in this isolated copy the stage's tier is mandatory.
    required_tier = str(stage["oracle_tier"])
    for name in written:
        path = dest / name / "capsule.yaml"
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        tiers = [str(t) for t in (doc.get("required_oracle_tiers") or [])]
        if required_tier not in tiers:
            tiers.append(required_tier)
        doc["required_oracle_tiers"] = sorted(
            set(tiers), key=lambda t: int(t[1:]) if t.startswith("L") and t[1:].isdigit() else 999)
        doc.pop("unreachable_required_oracle_tiers", None)
        doc.pop("oracle_tier_ceiling", None)
        # Application-derived search capsules normally cap themselves at L2 and point at a reduced
        # `_l3` sibling.  A frozen direct-evaluation copy deliberately lifts that search-time cap: it
        # is the same full workload, now required to execute at the stage's declared physical tier.
        doc.pop("max_oracle_tier", None)
        doc["evaluation_stage"] = {
            "name": stage_name,
            "policy": stage["policy"],
            "after": stage["after"],
            "required_oracle_tier": required_tier,
            "oracle_engine": stage["oracle_engine"],
        }
        if search_evidence is not None:
            doc["evaluation_stage"]["predecessor_l2_cycles"] = int(
                search_evidence["l2_cycles"][name])
            doc["evaluation_stage"]["cycle_budget"] = cycle_policy
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")

    record = {
        "schema": "descriptor_evaluation_cohort_v1",
        "target": te.target,
        "stage": stage_name,
        "policy": stage["policy"],
        "after": stage["after"],
        "required_oracle_tier": required_tier,
        "oracle_engine": stage["oracle_engine"],
        "descriptor": str(te.path.relative_to(repo_root())),
        "descriptor_sha256": te.descriptor_sha256,
        "candidate": str(candidate.resolve()),
        "candidate_tree_sha256": candidate_digest,
        "capsules": source_records,
        "n_capsules": len(written),
        "engine_preflight": preflight,
    }
    if cycle_policy is not None:
        record["cycle_budget"] = cycle_policy
    if search_evidence is not None:
        record["search_pass_evidence"] = {
            "seal": search_evidence["seal"],
            "seal_sha256": search_evidence["seal_sha256"],
            "score": search_evidence["score"],
            "score_sha256": search_evidence["score_sha256"],
            "candidate_tree_sha256": search_evidence["candidate_tree_sha256"],
            "capsules": search_evidence["capsules"],
            "n_capsules": search_evidence["n_capsules"],
            "n_passed": search_evidence["n_passed"],
            "required_oracle_tier": search_evidence["required_oracle_tier"],
            "claim_scope": search_evidence["claim_scope"],
            "l2_cycles": search_evidence["l2_cycles"],
            "l2_engine_binding": search_evidence["l2_engine_binding"],
        }
    if predecessor is not None:
        record["predecessor_pass_evidence"] = predecessor
    record["materialized_tree_sha256"] = _tree_sha256(dest)
    (dest / ".evaluation_cohort.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def validate_evaluation_cohort(
    root: str | Path, te: TargetExperiment, candidate: str | Path,
) -> dict[str, Any]:
    root = Path(root)
    record_path = root / ".evaluation_cohort.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("schema") != "descriptor_evaluation_cohort_v1":
        raise ValueError(f"unsupported evaluation cohort record: {record_path}")
    if record.get("target") != te.target or record.get("descriptor_sha256") != te.descriptor_sha256:
        raise ValueError("evaluation cohort does not belong to the loaded target descriptor")
    candidate = Path(candidate)
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError(f"frozen evaluation candidate is unavailable: {candidate}")
    if _candidate_tree_sha256(candidate) != record.get("candidate_tree_sha256"):
        raise ValueError("frozen evaluation candidate content digest mismatch")
    if Path(str(record.get("candidate", ""))).resolve() != candidate.resolve():
        raise ValueError("evaluation cohort names a different frozen candidate")
    stage = te.evaluation_cohort(str(record.get("stage")))
    names = sorted(path.name for path in root.iterdir() if path.is_dir())
    if names != sorted(stage["include_capsules"]):
        raise ValueError("evaluation cohort contents differ from its descriptor stage")
    for name in names:
        doc = yaml.safe_load((root / name / "capsule.yaml").read_text(encoding="utf-8")) or {}
        if stage["oracle_tier"] not in (doc.get("required_oracle_tiers") or []):
            raise ValueError(f"{name} does not require evaluation tier {stage['oracle_tier']}")
        if doc.get("max_oracle_tier") is not None:
            raise ValueError(f"{name} still carries a search-time maximum oracle tier")
        marker = doc.get("evaluation_stage") or {}
        if marker.get("name") != record["stage"] or marker.get("oracle_engine") != stage["oracle_engine"]:
            raise ValueError(f"{name} carries the wrong evaluation-stage marker")
        if str(stage["after"]) == "search_l2_pass":
            expected_cycles = ((record.get("search_pass_evidence") or {}).get("l2_cycles") or {}).get(name)
            if (not isinstance(expected_cycles, int) or isinstance(expected_cycles, bool)
                    or expected_cycles <= 0
                    or marker.get("predecessor_l2_cycles") != expected_cycles):
                raise ValueError(f"{name} carries no exact sealed predecessor L2 cycle count")
            if marker.get("cycle_budget") != _sealed_predecessor_cycle_policy(stage) \
                    or record.get("cycle_budget") != _sealed_predecessor_cycle_policy(stage):
                raise ValueError(f"{name} carries the wrong sealed-cycle observation policy")
    expected_digest = record.get("materialized_tree_sha256")
    # The manifest itself was deliberately absent when its digest was calculated.
    actual_digest = _tree_sha256(root, exclude=frozenset({record_path.name}))
    if actual_digest != expected_digest:
        raise ValueError("evaluation cohort content digest mismatch")
    _validate_engine_binding(
        record.get("engine_preflight"), expected_engine=str(stage["oracle_engine"]),
        target=te.target)
    source_records = []
    sources = _source_capsules(te)
    for name in stage["include_capsules"]:
        source = sources[name]
        doc = yaml.safe_load((source / "capsule.yaml").read_text(encoding="utf-8")) or {}
        source_records.append({
            "name": name,
            "source_role": str(doc.get("source_role") or ""),
            "source_tree_sha256": _tree_sha256(source),
        })
    if sorted(source_records, key=lambda row: row["name"]) != record.get("capsules"):
        raise ValueError("evaluation source capsule tree digest mismatch")
    search_evidence = record.get("search_pass_evidence")
    if str(stage["after"]) == "search_l2_pass":
        if not isinstance(search_evidence, dict):
            raise ValueError("derived GSIM cohort has no search-pass dependency evidence")
        seal_path = Path(str(search_evidence.get("seal", "")))
        if _file_sha256(seal_path, what="search pass seal") != search_evidence.get("seal_sha256"):
            raise ValueError("search pass seal evidence digest mismatch")
        sealed = validate_search_pass_seal(seal_path, te, candidate)
        for field in ("seal_sha256", "score", "score_sha256", "candidate_tree_sha256",
                      "capsules", "n_capsules", "n_passed", "required_oracle_tier", "claim_scope",
                      "l2_cycles", "l2_engine_binding"):
            if search_evidence.get(field) != sealed.get(field):
                raise ValueError(f"search pass evidence field changed: {field}")
    elif search_evidence is not None:
        raise ValueError("non-search-dependent evaluation cohort carries direct search evidence")
    predecessor = record.get("predecessor_pass_evidence")
    if predecessor is not None:
        manifest_path = Path(str(predecessor.get("cohort", ""))) / ".evaluation_cohort.json"
        score_path = Path(str(predecessor.get("score", "")))
        if (
            not manifest_path.is_file()
            or hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            != predecessor.get("cohort_manifest_sha256")
        ):
            raise ValueError("predecessor cohort evidence digest mismatch")
        if (
            not score_path.is_file()
            or hashlib.sha256(score_path.read_bytes()).hexdigest()
            != predecessor.get("score_sha256")
        ):
            raise ValueError("predecessor score evidence digest mismatch")
        prior = validate_evaluation_cohort(manifest_path.parent, te, candidate)
        if prior.get("materialized_tree_sha256") != predecessor.get("materialized_tree_sha256"):
            raise ValueError("predecessor materialized-tree hash chain mismatch")
        prior_binding = ((prior.get("engine_preflight") or {}).get("engine_binding") or {})
        if prior_binding.get("binding_sha256") != predecessor.get("engine_binding_sha256"):
            raise ValueError("predecessor engine-binding hash chain mismatch")
        prior_search = prior.get("search_pass_evidence") or {}
        if prior_search.get("seal_sha256") != predecessor.get("search_pass_seal_sha256"):
            raise ValueError("predecessor search-pass hash chain mismatch")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seal search convergence or materialize a frozen post-search evaluation cohort")
    parser.add_argument("--target", required=True)
    parser.add_argument("--stage")
    parser.add_argument("--dest")
    parser.add_argument("--candidate", required=True,
                        help="frozen compiler package evaluated by this stage")
    parser.add_argument("--create-search-pass-seal",
                        help="write a digest-bound exact-search-pass seal at this absent path")
    parser.add_argument("--search-score",
                        help="exact all-pass L2 self-check score used to create a search-pass seal")
    parser.add_argument("--search-pass-seal",
                        help="sealed exact all-pass L2 evidence required by the derived GSIM stage")
    parser.add_argument("--predecessor-cohort",
                        help="materialized predecessor cohort required by an after: *_pass stage")
    parser.add_argument("--predecessor-score",
                        help="grader score proving the predecessor cohort passed its physical tier")
    parser.add_argument("--replace-empty", action="store_true",
                        help="remove an existing empty destination before materializing")
    args = parser.parse_args(argv)
    descriptor = descriptor_for(args.target)
    if descriptor is None:
        raise SystemExit(f"no target experiment descriptor for {args.target!r}")
    te = load_target_experiment(descriptor)
    if args.create_search_pass_seal:
        if args.stage or args.dest or args.search_pass_seal \
                or args.predecessor_cohort or args.predecessor_score:
            parser.error("search-pass sealing cannot be combined with evaluation-stage arguments")
        if not args.search_score:
            parser.error("--create-search-pass-seal requires --search-score")
        record = create_search_pass_seal(
            args.create_search_pass_seal, te, args.candidate, args.search_score)
        validate_search_pass_seal(args.create_search_pass_seal, te, args.candidate)
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    if args.search_score:
        parser.error("--search-score is only valid with --create-search-pass-seal")
    if not args.stage or not args.dest:
        parser.error("evaluation materialization requires --stage and --dest")
    dest = Path(args.dest)
    if args.replace_empty and dest.is_dir() and not any(dest.iterdir()):
        dest.rmdir()
    record = materialize_evaluation_cohort(
        dest, te, args.stage, args.candidate,
        search_pass_seal=args.search_pass_seal,
        predecessor_cohort=args.predecessor_cohort,
        predecessor_score=args.predecessor_score,
    )
    validate_evaluation_cohort(dest, te, args.candidate)
    print(json.dumps(record, indent=2, sort_keys=True))
    if not record["engine_preflight"]["ok"]:
        print("evaluation cohort materialized, but engine preflight failed; refusing to call it runnable")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
