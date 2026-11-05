"""Strict public archive unions; no split assignment, fitting or qualification."""

from __future__ import annotations

import copy

from . import contracts as C

MAX_ARCHIVES = 16


def join(datasets: tuple[dict, ...]) -> dict:
    """Retain every case and record under their original source assignments.

    Each component must already be reconstructed by original archive intake.
    Duplicate executions are refused rather than counted as new independent
    variants. Missing observations remain missing records of declared cases.
    """
    if type(datasets) is not tuple or not 2 <= len(datasets) <= MAX_ARCHIVES:
        raise C.ContractError("joint intake requires 2..16 original public archives")
    for document in datasets:
        C.dataset(document)
        if document["schema"] != C.DATASET or document["profile"] != "phase1-public":
            raise C.ContractError("joint intake requires original public Phase 1 evidence")
    first = datasets[0]
    if any(
        (document["profile"], document["domain"], document["feature_names"])
        != (first["profile"], first["domain"], first["feature_names"])
        for document in datasets[1:]
    ):
        raise C.ContractError("joint archives differ in exact profile, model, window or feature domain")
    if (
        sum(len(document["cases"]) for document in datasets) > C.MAX_CASES
        or sum(len(document["records"]) for document in datasets) > C.MAX_RECORDS
    ):
        raise C.ContractError("joint archive roster exceeds existing dataset bounds")
    result = copy.deepcopy(first)
    result["cases"] = [copy.deepcopy(case) for document in datasets for case in document["cases"]]
    result["records"] = [copy.deepcopy(record) for document in datasets for record in document["records"]]
    families = {}
    for case in result["cases"]:
        source = case["source"]["source_sha256"]
        if families.setdefault(source, case["mechanism_family"]) != case["mechanism_family"]:
            raise C.ContractError("one original source changed mechanism family across archives")
    # Includes exact-source/split and workload/group consistency, duplicate
    # case/record/execution refusal, labels, reports and model identities.
    return C.dataset(result)
