"""Select the fewest probe requests that make every fit parameter identifiable over the domain.

Requirements are derived, not listed:

* ``presence:<class>`` -- some label exercises each price class the space requires;
* ``variant:<class>/<instruction>`` -- every instruction the provider pools into a class appears in
  some label, so pooling is tested rather than assumed (an unexercised decode path is a gap even when
  its class is covered by another instruction);
* ``points:<column>`` -- two distinct nonzero values per column (the fit's two-points gate);
* ``identifiable:<column>`` -- the column's unit vector lies in the labels' row space, i.e. no
  combination of the other columns reproduces it (exact rank test on the scaled design);
* ``range_min/range_max:<quantity>`` -- the labels' whole-call quantities span the domain's range,
  clipped by an explicit simulation budget, so a fitted screen never extrapolates over the domain.

Candidate requests are generated from the unmet requirements and realized by the target's probe
backend into command streams, which the same decoder turns into predicted feature rows. A greedy
cover then adds, at each step, the candidate that satisfies the most unmet requirements (cheapest
first on ties) until none is left or no candidate helps. The final design is checked with the fit's
own gates (:func:`merlin_timing.fitting.design_check`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from merlin_timing.fitting import design_check

from . import domain as D
from .decode import Command, Decoded, decode
from .space import TimingSpace, canonical_sha, parameter_explanations

SCHEMA = "merlin.timing_probe_selection.v1"
TOLERANCE = 1e-9
MAXIMUM_CONDITION = 1e6


class Unrealizable(ValueError):
    """The target backend cannot emit a program for this request; the reason is reported."""


@dataclass
class Realized:
    commands: list[Command]
    cost: float
    description: str
    program: Any = None


@dataclass
class Row:
    id: str
    source: str
    decoded: Decoded
    features: dict[str, float | None]
    aggregates: dict[str, float | None]
    request: dict | None = None
    realized: Realized | None = None
    covers: list[str] = field(default_factory=list)


def make_row(space: TimingSpace, row_id: str, source: str, decoded: Decoded, quantities) -> Row:
    return Row(row_id, source, decoded, decoded.features(space), {q: decoded.aggregate(space, q) for q in quantities})


def design_columns(space: TimingSpace) -> list[str]:
    """Columns whose parameters must be identified: the intercept and every required class's columns."""
    required = {c.name for c in space.classes if c.status == "required"}
    return [c.name for c in space.columns if c.kind == "fixed" or (c.kind == "class" and c.price_class in required)]


def _identifiable(rows: list[list[float]], scales: list[float]) -> tuple[int, set[int]]:
    """``(rank, columns whose unit vector is in the row space)`` by a reduced row echelon test."""
    if not rows:
        return 0, set()
    width = len(scales)
    matrix = [[value / scale for value, scale in zip(row, scales)] for row in rows]
    pivots: list[int] = []
    rank = 0
    for column in range(width):
        if rank == len(matrix):
            break
        best = max(range(rank, len(matrix)), key=lambda i: abs(matrix[i][column]))
        if abs(matrix[best][column]) <= TOLERANCE:
            continue
        matrix[rank], matrix[best] = matrix[best], matrix[rank]
        pivot = matrix[rank][column]
        matrix[rank] = [value / pivot for value in matrix[rank]]
        for i in range(len(matrix)):
            if i != rank and abs(matrix[i][column]) > 0.0:
                factor = matrix[i][column]
                matrix[i] = [a - factor * b for a, b in zip(matrix[i], matrix[rank])]
        pivots.append(column)
        rank += 1
    free = [j for j in range(width) if j not in set(pivots)]
    return rank, {column for r, column in enumerate(pivots) if all(abs(matrix[r][j]) <= TOLERANCE for j in free)}


def _basis(rows: list[list[float]], scales: list[float]) -> list[list[float]]:
    """A row basis (unscaled) spanning ``rows``: identifiability depends only on the row space."""
    basis: list[list[float]] = []
    reduced: list[tuple[int, list[float]]] = []  # forward-reduced: each is zero at every earlier lead
    for row in rows:
        vector = [value / scale for value, scale in zip(row, scales)]
        for lead, base in reduced:
            factor = vector[lead]
            if factor:
                vector = [a - factor * b for a, b in zip(vector, base)]
        lead = max(range(len(vector)), key=lambda j: abs(vector[j]))
        if abs(vector[lead]) <= TOLERANCE:
            continue
        pivot = vector[lead]
        reduced.append((lead, [value / pivot for value in vector]))
        basis.append(row)
    return basis


class Requirements:
    """The derived requirement roster and an incremental evaluator over label rows."""

    def __init__(self, space: TimingSpace, targets: dict[str, dict]):
        self.space = space
        self.columns = design_columns(space)
        self.targets = targets
        self.ids: list[str] = []
        self.text: dict[str, str] = {}
        for cls in space.classes:
            if cls.status != "required":
                continue
            self._add(f"presence:{cls.name}", f"a label exercising price class {cls.name}")
            if len(cls.instructions) > 1:
                for instruction in cls.instructions:
                    self._add(
                        f"variant:{cls.name}/{instruction}",
                        f"a label exercising {instruction} while the state selects {cls.name}",
                    )
        for column in self.columns:
            if column != "fixed":
                self._add(f"points:{column}", f"two distinct nonzero values of {column}")
            self._add(f"identifiable:{column}", f"{column} not reproducible from the other columns")
        for quantity, bounds in targets.items():
            self._add(f"range_min:{quantity}", f"a label with whole-call {quantity} <= {bounds['min']}")
            self._add(f"range_max:{quantity}", f"a label with whole-call {quantity} >= {bounds['max']}")
        self._add(
            "design:fit_gates",
            "the design passes the fit's own gates (two distinct points per parameter, full rank, bounded condition)",
        )

    def _add(self, key: str, text: str) -> None:
        self.ids.append(key)
        self.text[key] = text

    def evaluate(
        self, rows: list[Row], scales: list[float], basis: list[list[float]], rank_rows: list[Row] | None = None
    ) -> set[str]:
        return self.evaluate_with_rank(rows, scales, basis, rank_rows)[0]

    def evaluate_with_rank(
        self, rows: list[Row], scales: list[float], basis: list[list[float]], rank_rows: list[Row] | None = None
    ) -> tuple[set[str], int]:
        """Requirements ``rows`` meet, and the design rank. ``basis`` spans reduced rows; ``rank_rows`` the rest."""
        met: set[str] = set()
        for row in rows:
            for name in row.decoded.counts:
                met.add(f"presence:{name}")
            for name, instruction in row.decoded.variants:
                met.add(f"variant:{name}/{instruction}")
        for column in self.columns:
            values = {row.features.get(column) for row in rows}
            if len({v for v in values if v}) >= 2:
                met.add(f"points:{column}")
        complete = [
            [row.features[c] for c in self.columns]
            for row in (rows if rank_rows is None else rank_rows)
            if all(row.features.get(c) is not None for c in self.columns)
        ]
        rank, identifiable = _identifiable(basis + complete, scales)
        for index in identifiable:
            met.add(f"identifiable:{self.columns[index]}")
        for quantity, bounds in self.targets.items():
            values = [row.aggregates.get(quantity) for row in rows]
            values = [v for v in values if v is not None and v > 0]
            if values and min(values) <= bounds["min"]:
                met.add(f"range_min:{quantity}")
            if values and max(values) >= bounds["max"]:
                met.add(f"range_max:{quantity}")
        # The global gate is the fit's own check; it is only worth computing once every column is.
        if all(f"identifiable:{c}" in met for c in self.columns):
            matrix = [
                [row.features[c] for c in self.columns]
                for row in rows
                if all(row.features.get(c) is not None for c in self.columns)
            ]
            if matrix and design_check(matrix, maximum_condition=MAXIMUM_CONDITION)["identifiable"]:
                met.add("design:fit_gates")
        return met & set(self.ids), rank


def _requests(space: TimingSpace, unmet: set[str], targets: dict[str, dict], shapes: tuple[str, ...]) -> list[dict]:
    """Probe requests that could satisfy ``unmet``: per class in the backend's shapes, plus regimes."""
    requests = []
    for cls in space.classes:
        if cls.status != "required":
            continue
        prefixes = (f"variant:{cls.name}/", f"points:{cls.name}.", f"identifiable:{cls.name}.")
        # Any new distinct row can help the global gate, so it asks every class.
        if "design:fit_gates" in unmet or any(
            key.startswith(prefixes) or key == f"presence:{cls.name}" for key in unmet
        ):
            for shape in shapes:
                requests.append(
                    {"class": cls.name, "instructions": list(cls.instructions), "state": cls.state, "shape": shape}
                )
    high = {q: t["max"] for q, t in targets.items() if f"range_max:{q}" in unmet}
    low = {q: t["min"] for q, t in targets.items() if f"range_min:{q}" in unmet}
    if high:
        requests.append({"class": None, "shape": "regime", "regime": high})
    if low:
        requests.append({"class": None, "shape": "regime", "regime": low})
    return requests


def request_id(request: dict) -> str:
    name = request.get("class") or "regime"
    digest = canonical_sha(request)[:10]
    return f"{name}-{request['shape']}-{digest}"


def select(
    space: TimingSpace,
    labels: list,
    domain: dict,
    backend,
    *,
    budget: dict[str, int] | None = None,
    max_probes: int = 64,
) -> dict:
    """The selection document: chosen probes with what each covers, and every remaining gap."""
    targets = D.targets(domain, budget or {})
    requirements = Requirements(space, targets)
    quantities = list(targets)
    existing = [make_row(space, label.result, label.source, label.decoded, quantities) for label in labels]
    local = [max([1.0] + [abs(r.features.get(c) or 0.0) for r in existing]) for c in requirements.columns]
    met_before = requirements.evaluate(existing, local, [])
    unmet = set(requirements.ids) - met_before

    # Realize every candidate that could help once, up front, so column scales are global.
    shapes = tuple(getattr(backend, "SHAPES", ("full", "partial")))
    candidates: list[Row] = []
    refused: list[dict] = []
    for request in _requests(space, unmet, targets, shapes):
        try:
            realized = backend.realize(request, space)
        except Unrealizable as exc:
            refused.append({"request": request, "reason": str(exc)})
            continue
        decoded = decode(space, realized.commands)
        row = make_row(space, request_id(request), "probe", decoded, quantities)
        row.request, row.realized = request, realized
        candidates.append(row)
    pool = existing + candidates
    scales = [max([1.0] + [abs(r.features.get(c) or 0.0) for r in pool]) for c in requirements.columns]
    complete_existing = [
        [r.features[c] for c in requirements.columns]
        for r in existing
        if all(r.features.get(c) is not None for c in requirements.columns)
    ]
    basis = _basis(complete_existing, scales)
    met, rank = requirements.evaluate_with_rank(existing, scales, basis, rank_rows=[])
    chosen: list[Row] = []
    while len(chosen) < max_probes:
        # Most newly met requirements first; then the largest rank increase, because joint
        # identifiability can need several rows before any single column's unit vector is reached.
        best, best_key, best_gain, best_rank = None, None, set(), rank
        for row in candidates:
            if row in chosen:
                continue
            got, got_rank = requirements.evaluate_with_rank(
                existing + chosen + [row], scales, basis, rank_rows=chosen + [row]
            )
            gain = got - met
            key = (len(gain), got_rank - rank, -row.realized.cost)
            if best_key is None or key > best_key:
                best, best_key, best_gain, best_rank = row, key, gain, got_rank
        identifiability_open = any(k.startswith(("identifiable:", "design:")) for k in set(requirements.ids) - met)
        if best is None or not (best_gain or (best_rank > rank and identifiability_open)):
            break
        best.covers = sorted(best_gain) or [f"rank:{best_rank}"]
        chosen.append(best)
        met |= best_gain
        rank = best_rank
    unmet_after = [key for key in requirements.ids if key not in met]

    matrix = [
        [r.features[c] for c in requirements.columns]
        for r in existing + chosen
        if all(r.features.get(c) is not None for c in requirements.columns)
    ]
    check = (
        design_check(matrix, maximum_condition=MAXIMUM_CONDITION)
        if matrix
        else {"identifiable": False, "reason": "no complete label rows"}
    )
    explanations = parameter_explanations(space)
    for key in unmet_after:
        kind, _, subject = key.partition(":")
        owner = subject.partition("/")[0].partition(".")[0]
        reasons = [r["reason"] for r in refused if r["request"].get("class") == owner]
        explanations.append(
            {
                "subject": subject,
                "status": f"unmet_{kind}",
                "reason": "; ".join(sorted(set(reasons))) or "no realizable candidate request satisfies it",
            }
        )
    for quantity, bounds in targets.items():
        if bounds["clipped"]:
            explanations.append(
                {
                    "subject": f"range_max:{quantity}",
                    "status": "budget_clipped",
                    "reason": f"domain maximum {bounds['domain_max']} exceeds the simulation budget {bounds['max']}; "
                    "predictions above the budget extrapolate",
                }
            )
    probes = []
    for row in chosen:
        probes.append(
            {
                "id": row.id,
                "request": row.request,
                "description": row.realized.description,
                "relative_cost": row.realized.cost,  # the backend's tie-break estimate, not a prediction
                "covers": [
                    {
                        "requirement": key,
                        "meaning": requirements.text.get(
                            key,
                            "raises the design rank: no single row reaches an unmet column alone, several together do",
                        ),
                    }
                    for key in row.covers
                ],
                "predicted": {
                    "classes": dict(row.decoded.counts),
                    "variants": {f"{c}/{i}": n for (c, i), n in row.decoded.variants.items()},
                    "aggregates": row.aggregates,
                    "commands": row.decoded.commands,
                },
                "commands_sha256": hashlib.sha256(
                    json.dumps([[c.name, c.rs1, c.rs2] for c in row.realized.commands]).encode()
                ).hexdigest(),
            }
        )
    return {
        "schema": SCHEMA,
        "target": space.target,
        "identity": {
            **space.identity,
            "domain_sha256": canonical_sha(domain),
            "labels_sha256": canonical_sha(sorted({(label.archive_sha256, label.result) for label in labels})),
        },
        "existing": {
            "labels": len(existing),
            "complete_rows": len(complete_existing),
            "requirements_met": len(met_before),
        },
        "domain_targets": targets,
        "domain": {k: v for k, v in domain.items() if k != "unquantified"},
        "domain_unquantified": len(domain.get("unquantified") or ()),
        "design_columns": requirements.columns,
        "requirements": [
            {
                "id": key,
                "meaning": requirements.text[key],
                "met_by_existing": key in met_before,
                "met_after_selection": key in met,
            }
            for key in requirements.ids
        ],
        "probes": probes,
        "unmet": [{"id": key, "meaning": requirements.text[key]} for key in unmet_after],
        "refused_requests": refused,
        "design_check": check,
        "unidentifiable": explanations,
        "gaps": len(unmet_after),
    }
