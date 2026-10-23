"""Auditable exact set cover for captured PyTorch Core ATen provenance.

This module deliberately separates three claims:

* the denominator is every overload tagged :class:`torch.Tag.core` in one recorded PyTorch build;
* a capsule covers only exact overload strings found in its captured ``prov.aten`` attributes;
* minimality is over the supplied candidate pool, never over every program one could write.

The implementation is target-agnostic.  It does not ask what a target accelerates: a hardware target
may route an operator to an accelerator, a scalar/vector unit, or a host fallback.  This layer only
constructs the framework-level test denominator and proves the smallest captured capsule subset that
contains it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class CoverageFailure(RuntimeError):
    """A coverage claim could not be established without guessing."""


class SolverFailure(CoverageFailure):
    """Z3 was unavailable or did not return a conclusive answer."""


def overload_digest(overloads: Sequence[str]) -> str:
    """Stable digest of a sorted, unique overload list.

    The final newline makes the byte representation unambiguous and convenient to reproduce with
    ordinary command-line tools.
    """

    normalized = sorted(set(str(op) for op in overloads))
    return hashlib.sha256(("\n".join(normalized) + "\n").encode()).hexdigest()


def denominator_document(opset: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an enumerator result and project the durable denominator artifact."""

    overloads = [str(op) for op in opset.get("ops") or ()]
    if not overloads:
        raise CoverageFailure("Core ATen enumeration returned no overloads")
    if overloads != sorted(set(overloads)):
        raise CoverageFailure("Core ATen overloads are not sorted and unique")
    if int(opset.get("n_core", -1)) != len(overloads):
        raise CoverageFailure("Core ATen count does not match its overload list")
    if not all(op.startswith("aten.") and op.count(".") >= 2 for op in overloads):
        raise CoverageFailure("Core ATen enumeration contains a malformed overload name")
    torch_version = str(opset.get("torch") or "").strip()
    if not torch_version:
        raise CoverageFailure("Core ATen enumeration did not record the PyTorch version")
    digest = overload_digest(overloads)
    emitted_digest = opset.get("sha256")
    if emitted_digest is not None and str(emitted_digest) != digest:
        raise CoverageFailure("Core ATen enumerator digest does not match its overload list")
    return {
        "schema_version": 1,
        "scope": "operator overloads tagged torch.Tag.core",
        "enumeration_source": str(
            opset.get("enumeration_source")
            or "torch dispatcher registry filtered by torch.Tag.core (legacy enumerator without source field)"
        ),
        "pytorch_version": torch_version,
        "overload_count": len(overloads),
        "overload_sha256": digest,
        "digest_encoding": "UTF-8 sorted overload names, one per line, with final newline",
        "overloads": overloads,
    }


def _provenance_ops(path: Path) -> set[str]:
    """Read exact ``prov.aten`` strings, refusing a truncated attribute."""

    text = path.read_text(encoding="utf-8")
    marker = 'prov.aten = "'
    out: set[str] = set()
    position = 0
    while True:
        index = text.find(marker, position)
        if index < 0:
            break
        start = index + len(marker)
        end = text.find('"', start)
        if end < 0:
            raise CoverageFailure(f"unterminated prov.aten attribute in {path}")
        value = text[start:end]
        if not value:
            raise CoverageFailure(f"empty prov.aten attribute in {path}")
        out.add(value)
        position = end + 1
    return out


def capsule_operator_matrix(capsule_roots: Sequence[str | Path], denominator: Sequence[str]) -> dict[str, Any]:
    """Extract a deterministic capsule-to-Core-ATen matrix from captured MLIR.

    Every directory containing ``capsule.yaml`` is inventoried.  A directory is solver-eligible only
    when at least one MLIR file contains provenance.  If two captured files in one capsule disagree on
    their provenance set, the function fails closed instead of choosing the more favorable one.
    """

    core = set(str(op) for op in denominator)
    if not core:
        raise CoverageFailure("cannot build a candidate matrix for an empty denominator")
    roots = [Path(root).resolve() for root in capsule_roots]
    if not roots:
        raise CoverageFailure("no capsule roots were supplied")

    capsules: dict[str, list[str]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []
    for root_index, root in enumerate(roots):
        if not root.is_dir():
            raise CoverageFailure(f"capsule root does not exist: {root}")
        prefix = "" if len(roots) == 1 else f"root{root_index}/"
        markers = sorted({*root.rglob("capsule.yaml"), *root.rglob("capture.json")})
        directories = sorted({marker.parent for marker in markers})
        for directory in directories:
            identity = prefix + directory.relative_to(root).as_posix()
            if identity in capsules:
                raise CoverageFailure(f"duplicate capsule identity: {identity}")
            tagged: list[tuple[Path, set[str]]] = []
            unreadable: list[str] = []
            for mlir in sorted(directory.glob("*.mlir")):
                try:
                    observed = _provenance_ops(mlir)
                except (OSError, UnicodeError, CoverageFailure) as exc:
                    unreadable.append(f"{mlir.name}: {type(exc).__name__}: {exc}")
                    continue
                if observed:
                    tagged.append((mlir, observed))
            if unreadable:
                raise CoverageFailure(f"capture provenance is unreadable for {identity}: {unreadable}")
            if not tagged:
                capsules[identity] = []
                provenance[identity] = {
                    "status": "ineligible_no_provenance",
                    "source_files": [],
                    "reason": "no MLIR file in the capsule contains prov.aten",
                }
                continue
            reference = tagged[0][1]
            disagreements = [path.name for path, values in tagged[1:] if values != reference]
            if disagreements:
                files = [tagged[0][0].name, *disagreements]
                raise CoverageFailure(f"ambiguous prov.aten sets for {identity}: {files}")
            exact_core = sorted(reference & core)
            capsules[identity] = exact_core
            eligible.append(identity)
            provenance[identity] = {
                "status": "eligible",
                "source_files": [path.name for path, _values in tagged],
                "observed_aten": sorted(reference),
                "non_core_observed": sorted(reference - core),
            }

    return {
        "schema_version": 1,
        "scope": "exact Core ATen overload strings extracted from captured prov.aten attributes",
        "capsule_roots": [str(root) for root in roots],
        "discovered_capsule_count": len(capsules),
        "candidate_count": len(eligible),
        "excluded_no_provenance_count": len(capsules) - len(eligible),
        "eligible_candidates": sorted(eligible),
        "capsules": dict(sorted(capsules.items())),
        "provenance": dict(sorted(provenance.items())),
    }


def _load_z3():
    try:
        import z3
    except ImportError as exc:
        raise SolverFailure("z3 is not installed (install the merlin verify extra)") from exc
    return z3


def _check(solver, z3, context: str):
    try:
        status = solver.check()
    except Exception as exc:  # noqa: BLE001 -- a solver error invalidates the proof
        raise SolverFailure(f"Z3 failed while {context}: {type(exc).__name__}: {exc}") from exc
    if status == z3.unknown:
        reason = solver.reason_unknown() if hasattr(solver, "reason_unknown") else "not reported"
        raise SolverFailure(f"Z3 returned unknown while {context}: {reason}")
    if status not in (z3.sat, z3.unsat):
        raise SolverFailure(f"Z3 returned unexpected status {status!r} while {context}")
    return status


@dataclass(frozen=True)
class CoverResult:
    status: str
    solver: str
    solver_version: str
    denominator_count: int
    coverable_count: int
    candidate_count: int
    selected_count: int
    objective_value: int
    selected_capsules: list[str]
    uncovered_overloads: list[str]
    operator_to_selected_capsule: dict[str, str]
    selected_union: list[str]
    selected_union_matches_coverable: bool
    selected_union_matches_denominator: bool
    minimum_cardinality_lower_bound: int | None
    minimum_cardinality_upper_bound: int | None
    objective_bounds_closed: bool
    minimality_scope: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def exact_minimum_cover(
    denominator: Sequence[str],
    candidates: Mapping[str, Sequence[str]],
    *,
    allow_partial: bool = False,
) -> CoverResult:
    """Prove a minimum-cardinality capsule cover with deterministic name tie-breaking.

    Z3 first minimizes the pseudo-Boolean cardinality.  At that fixed primary optimum, a unique
    binary positional objective prefers earlier sorted names, making the chosen optimum independent
    of input ordering without one solver call per candidate.
    """

    universe = tuple(sorted(set(str(op) for op in denominator)))
    normalized = {
        str(name): set(str(op) for op in operators) & set(universe) for name, operators in sorted(candidates.items())
    }
    providers = {op: sorted(name for name, covered in normalized.items() if op in covered) for op in universe}
    uncovered = sorted(op for op, names in providers.items() if not names)
    if uncovered and not allow_partial:
        return CoverResult(
            status="uncovered",
            solver="z3",
            solver_version="not_run",
            denominator_count=len(universe),
            coverable_count=len(universe) - len(uncovered),
            candidate_count=len(normalized),
            selected_count=0,
            objective_value=0,
            selected_capsules=[],
            uncovered_overloads=uncovered,
            operator_to_selected_capsule={},
            selected_union=[],
            selected_union_matches_coverable=False,
            selected_union_matches_denominator=False,
            minimum_cardinality_lower_bound=None,
            minimum_cardinality_upper_bound=None,
            objective_bounds_closed=False,
            minimality_scope="optimization not run because at least one denominator overload has no candidate",
        )

    target = tuple(op for op in universe if op not in set(uncovered))
    z3 = _load_z3()
    names = tuple(sorted(normalized))
    variables = {name: z3.Bool(f"capsule_{index:06d}") for index, name in enumerate(names)}

    def constraints(solver) -> None:
        for op in target:
            solver.add(z3.Or(*(variables[name] for name in providers[op])))

    canonical = z3.Optimize()
    canonical.set(timeout=60_000)
    constraints(canonical)
    cardinality = z3.Sum(*(z3.If(variables[name], 1, 0) for name in names)) if names else z3.IntVal(0)
    cardinality_handle = canonical.minimize(cardinality)
    # At a fixed cardinality every subset has a unique binary positional value.  Maximizing it
    # implements the historical "prefer the earliest name" lexicographic tie-break exactly.
    preference = (
        z3.Sum(*(z3.If(variables[name], 1 << (len(names) - index - 1), 0) for index, name in enumerate(names)))
        if names
        else z3.IntVal(0)
    )
    preference_handle = canonical.maximize(preference)
    if _check(canonical, z3, "optimizing cardinality and deterministic tie-break") != z3.sat:
        raise SolverFailure("Z3 found no cover for an allegedly coverable universe")
    model = canonical.model()
    selected = sorted(name for name in names if z3.is_true(model.eval(variables[name], model_completion=True)))
    optimum = cardinality_handle.lower().as_long()
    optimum_upper = cardinality_handle.upper().as_long()
    if optimum_upper != optimum:
        raise SolverFailure("Z3 did not close the minimum-cardinality objective bound")
    if len(selected) != optimum:
        raise CoverageFailure(f"solver model selected {len(selected)} capsules for proven objective {optimum}")
    preference_value = model.eval(preference, model_completion=True).as_long()
    if (
        preference_handle.lower().as_long() != preference_value
        or preference_handle.upper().as_long() != preference_value
    ):
        raise SolverFailure("Z3 did not close the deterministic tie-break objective bound")
    selected_union = set().union(*(normalized[name] for name in selected)) if selected else set()
    if selected_union != set(target):
        missing = sorted(set(target) - selected_union)
        extra = sorted(selected_union - set(target))
        raise CoverageFailure(f"independent selected-union validation failed: missing={missing}, extra={extra}")
    witnesses = {op: next(name for name in selected if op in normalized[name]) for op in target}
    full = selected_union == set(universe)
    return CoverResult(
        status="optimal" if full else "partial_optimal",
        solver="z3 Optimize with closed exact cardinality and unique lexicographic objective bounds",
        solver_version=str(z3.get_version_string()),
        denominator_count=len(universe),
        coverable_count=len(target),
        candidate_count=len(names),
        selected_count=len(selected),
        objective_value=optimum,
        selected_capsules=selected,
        uncovered_overloads=uncovered,
        operator_to_selected_capsule=witnesses,
        selected_union=sorted(selected_union),
        selected_union_matches_coverable=True,
        selected_union_matches_denominator=full,
        minimum_cardinality_lower_bound=optimum,
        minimum_cardinality_upper_bound=optimum_upper,
        objective_bounds_closed=True,
        minimality_scope="exact minimum over the supplied captured candidate pool",
    )


def json_bytes(document: Mapping[str, Any]) -> bytes:
    """Canonical human-readable JSON bytes used by artifacts and determinism checks."""

    return (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def markdown_summary(
    denominator: Mapping[str, Any],
    matrix: Mapping[str, Any],
    cover: CoverResult,
    *,
    case_corpus: Mapping[str, Any] | None = None,
    gap_reasons: Mapping[str, str] | None = None,
) -> str:
    """Render the concise point-in-time report requested by the coverage workflow."""

    total = cover.denominator_count
    covered = cover.coverable_count
    percentage = 100.0 * covered / total if total else 0.0
    reasons = gap_reasons or {}
    gaps = "\n".join(
        f"- `{op}` — {reasons.get(op, 'no captured candidate contains this exact provenance overload')}"
        for op in cover.uncovered_overloads
    )
    selected = "\n".join(f"- `{name}`" for name in cover.selected_capsules) or "- None"
    if not gaps:
        gaps = "- None"
    capture_headline = (
        f"{covered}/{total} Core ATen overloads ({percentage:.2f}%) are covered by "
        f"{cover.selected_count}/{cover.candidate_count} provenance-bearing candidate capsules."
    )
    case_count = int((case_corpus or {}).get("case_count", 0))
    suite_headline = (
        f"{case_count}/{total} Core ATen overloads have executable, target-agnostic eager cases. "
        if case_corpus is not None
        else ""
    )
    qualification = (
        "The denominator is overloads tagged `torch.Tag.core` in the pinned PyTorch version above. "
        "Overloads remain distinct. Coverage is credited only from exact `prov.aten` strings in captured MLIR. "
        "The selected set is a proven minimum-cardinality subset of the available provenance-bearing candidate "
        "pool (or of its coverable portion when gaps remain); it is not a claim of global minimality over every "
        "program that could be authored. This report is framework- and target-agnostic and does not claim that "
        "any particular hardware target accelerates the listed operators."
    )
    coverable_union_status = "PASS" if cover.selected_union_matches_coverable else "FAIL"
    full_union_status = "PASS" if cover.selected_union_matches_denominator else "FAIL (partial result)"
    return f"""# Core ATen capsule coverage

{suite_headline}{capture_headline}

- PyTorch version: `{denominator["pytorch_version"]}`
- Denominator digest (SHA-256): `{denominator["overload_sha256"]}`
- Source-level executable cases: {case_count if case_corpus is not None else "not generated"}/{total}
- Solver: {cover.solver}, version `{cover.solver_version}`
- Solver status: `{cover.status}`
- Discovered capsule directories: {matrix["discovered_capsule_count"]}
- Capsules excluded because they carry no `prov.aten`: {matrix["excluded_no_provenance_count"]}
- Independent selected-union validation against coverable set: {coverable_union_status}
- Selected union equals the full denominator: {full_union_status}

## Remaining gaps

{gaps}

## Selected capsules

{selected}

## Scope and minimality

{qualification}
"""
