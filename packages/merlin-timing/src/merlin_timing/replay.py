"""Freeze new native-model observation requests without relabeling old reports."""

from __future__ import annotations

import copy
from pathlib import Path

from . import archive as A
from . import contracts as C

SCHEMA = "merlin.spike_model_replay.v1"


def prepare(plan: dict, model_path: str | Path, executable: str | Path, libraries: list[str | Path]) -> dict:
    """Retain original labels and reviews; a new model needs new native reports.

    This prepares data for the existing archive-attach path. It neither executes
    code nor certifies model semantics, prices, the observer or target adapters.
    Library ordering is part of the native observer's loading contract.
    """
    if plan.get("schema") not in (A.PLAN, A.PLAN_V2):
        raise C.ContractError("replay requires a prepared archive observation plan")
    original = C.dataset(plan["dataset_template"])
    if original["records"] or original["profile"] != "phase1-public":
        raise C.ContractError("replay requires an empty public archive dataset template")
    before = A.file_identity(model_path)
    model = C.read(model_path)
    if model.get("schema") != "merlin.annotated_spike_model.v1" or model.get("profile") != original["profile"]:
        raise C.ContractError("replay requires the selected public native model schema")
    expected = {key: value for key, value in original["domain"].items() if key != "target_model_sha256"}
    if model.get("domain") != expected:
        raise C.ContractError("replay cannot change original hardware, label, observer, scope or memory domain")
    if model.get("scope") != {"kind": "counter_pc_pair"}:
        raise C.ContractError("replay requires the original explicit counter-PC window policy")
    if not isinstance(libraries, list) or not 1 <= len(libraries) <= 64:
        raise C.ContractError("replay requires an ordered bounded library roster")
    selected_libraries = [{"path": str(Path(path).absolute()), "sha256": A.file_identity(path)} for path in libraries]
    if model.get("libraries") != [row["sha256"] for row in selected_libraries]:
        raise C.ContractError("selected provider libraries differ from the model's exact load order")
    observer = {"path": str(Path(executable).absolute()), "sha256": A.file_identity(executable)}
    if A.file_identity(model_path) != before:
        raise C.ContractError("selected native model changed while preparing replay")
    if before == original["domain"]["target_model_sha256"]:
        raise C.ContractError("new-model replay requires a distinct immutable model artifact")
    rebound = copy.deepcopy(plan)
    rebound["dataset_template"]["domain"]["target_model_sha256"] = before
    C.dataset(rebound["dataset_template"])
    observations = rebound.get("observations")
    if not isinstance(observations, list) or len(observations) > C.MAX_RECORDS:
        raise C.ContractError("replay observations exceed the declared count bound")
    cases = {case["case_id"]: case for case in original["cases"]}
    seen = set()
    for row in observations:
        case = cases[row["case_id"]]
        if row["record_id"] in seen or row["label"]["execution_sha256"] != C.execution_identity(case):
            raise C.ContractError("replay observation repeats or changes its original execution")
        seen.add(row["record_id"])
        A._path(row["elf_member"])
        scope = row["scope_binding"]
        if scope is not None:
            if any(scope.get(key) != case[key] for key in ("elf_sha256", "input_sha256", "scope_sha256")):
                raise C.ContractError("replay scope differs from the original ELF, input or window")
            C.sha(scope.get("evidence_sha256"), "original counter review")
    unavailable = [row["record_id"] for row in observations if row["scope_binding"] is None]
    return {
        "schema": SCHEMA,
        "original_plan_sha256": C.digest(plan),
        "model": {"path": str(Path(model_path).absolute()), "sha256": before},
        "observer": observer,
        "libraries": selected_libraries,
        "observation_plan": rebound,
        "observation_plan_sha256": C.digest(rebound),
        "replayable_records": len(observations) - len(unavailable),
        "unavailable_counter_reviews": unavailable,
        "launch_authority": False,
        "prediction_authority": "NONE",
        "launches": [],
        "required_before_execution": [
            "Verify executable feature-engine/source identity and functional/native provider compatibility.",
            "Reverify archive, extracted ELF, model and ordered libraries against their frozen digests.",
            "Use each original scope_binding and input digest; preserve every source split and repeat.",
            "Acquire the ordinary shared CPU grant; write fresh reports outside immutable inputs.",
        ],
        "required_before_ranking": [
            "Complete resource assignments, dependency/completion semantics and explicit timing prices.",
            "Collect fresh native reports with this exact model; old count reports cannot be attached.",
            "Pass held-out event validation and capacity-domain extrapolation checks independently.",
        ],
    }
