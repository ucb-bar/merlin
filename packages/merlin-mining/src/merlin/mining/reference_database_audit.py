"""Audit the input/output boundary of offline reference preprocessing, not experiment outcomes."""

from __future__ import annotations

import json
from pathlib import Path

from .reference_database_analysis import ADVICE_KEYS


def audit(root, artifacts, occurrences, summaries=()):
    root = Path(root)
    for artifact in artifacts:
        if artifact["origin"]["kind"] not in (
            "archive_member",
            "public_repository",
            "upstream_derived_public_artifact",
        ):
            raise ValueError("reference origin is not an explicitly selected original/public artifact")
        if not artifact["path"].startswith("originals/"):
            raise ValueError("original reference outside original collection")

    def reject_advice(value):
        if isinstance(value, dict):
            if ADVICE_KEYS.intersection(value):
                raise ValueError("optimization/advice field in bottom-up analysis")
            for item in value.values():
                reject_advice(item)
        elif isinstance(value, list):
            for item in value:
                reject_advice(item)

    for occurrence in occurrences:
        reject_advice(json.loads((root / occurrence["analysis"]).read_bytes()))
        if occurrence["numerical_equivalence"] != "unknown" or occurrence["performance"] != "unknown":
            raise ValueError("offline preprocessing cannot qualify numerics or performance")
    reject_advice(summaries)
    for path in (root / "mappings").rglob("*.json"):
        reject_advice(json.loads(path.read_bytes()))
    return dict(
        schema="merlin.reference_database_boundary_audit.v1",
        status="PASS",
        original_artifacts=len(artifacts),
        analyzed_occurrences=len(occurrences),
        input_boundary=(
            "exact selected user archives, pinned upstream source/derived references, selected ISA/RTL definitions"
        ),
        candidate_compiler_inputs=False,
        hidden_case_inputs=False,
        grading_answer_inputs=False,
        measured_winner_inputs=False,
        compiler_transformations_selected=False,
        observation_advice_fields_absent=True,
        exact_byte_dedup_only=True,
        numerical_or_performance_qualification=False,
        limitation="Boundary audit of this deterministic builder; not a proof of arbitrary tool/dependency behavior.",
        future_experiment_requirements=[
            "Select this bank explicitly; do not change an existing frozen run.",
            "Give both arms the same underlying original reference selection and public target definitions.",
            "Declare which arm receives precomputed CCA, record preprocessing cost, and keep author budgets matched.",
            "Keep candidate matching, optimization decisions and public-shape experiments inside the author workflow.",
            "Keep hidden evaluation sealed until author freeze; never call name/byte matches semantic equivalence.",
        ],
    )
