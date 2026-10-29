"""Conservative development timing scope from original completed build products.

Pass names, source filenames and candidate declarations cannot omit a member.
A member is unchanged only after fresh original compile/Spike evidence and a
reopened LLVM/object/ELF producer chain agree under the same frozen inputs.
Partial timing supplies no promotion, plateau or final correctness receipt.
"""

from __future__ import annotations

import time
from pathlib import Path

from merlin.common import invocation_record as I
from merlin_experiments.phase1.feedback.candidate_observations import _reopen_product, reopen_native_products

from . import ordinary_calibration as CAL
from . import ordinary_fast_feedback as FAST
from . import ordinary_measurement_selection as SELECT
from .contracts import StageGateError, document_sha256, mapping_file, sha256_file

ACTION = "evaluation-submit-impact"
SCHEMA = "merlin.ordinary_impact_selection.v1"
MAX_RECORDS = 512


def _producer_identity(records):
    """Effective executable/dependency bytes, independent of private output paths."""
    if not records or any(not (row.get("executable") or {}).get("sha256") for row in records):
        raise ValueError("impact producer toolchain bytes are unknown")
    return sorted(
        {
            document_sha256(
                {
                    "stage": row["stage"],
                    "executable": (row.get("executable") or {}).get("sha256"),
                    "dependencies": sorted(pin["sha256"] for pin in row.get("dependencies", ())),
                }
            )
            for row in records
        }
    )


def _products(root, compiled, grade):
    """Reopen actual role pins and successful producers, never feedback paths."""
    if root.is_symlink() or root.resolve() != root:
        raise ValueError("impact product workspace is indirect")
    if compiled.get("status") != "observed":
        raise ValueError("impact compiled products are unknown")
    join = compiled.get("artifact_join") or {}
    if join.get("kind") == "original_native_model_products":
        native = grade["candidate_native_execution"]
        products, paths = reopen_native_products(
            root,
            emission=grade.get("candidate_emission"),
            elf_pin=native.get("elf"),
            source=native.get("source"),
            build_artifacts=native.get("build_artifacts"),
            candidate=native.get("candidate"),
        )
        records = [I.verify(path) for path in paths]
        identity = {role: sha256_file(path) for role, path in products.items()}
        if identity["source"] != join["source_sha256"] or identity["command_buffer"] != join["command_buffer_sha256"]:
            raise ValueError("impact native source/emission no longer matches")
    elif join.get("kind") == "original_operator_products":
        llvm, elf = FAST._unique(root, "lowered.llvm.mlir"), FAST._unique(root, FAST.PACKAGE_ELF_NAME)
        for path, digest in ((llvm, compiled["llvm_mlir_sha256"]), (elf, compiled["elf_sha256"])):
            _reopen_product({"path": str(path), "sha256": digest}, root=root)
        selected, records = [], []
        for index, path in enumerate(root.rglob("invocation.json")):
            if index >= MAX_RECORDS or path.is_symlink() or path.resolve() != path or path.stat().st_size > 2 << 20:
                raise ValueError("impact invocation inventory exceeds its original bounded owner")
            row = mapping_file(path)
            if row.get("stage") in {"llvm_translation", "object", "elf"} and FAST.successful_record(row):
                records.append((path, row))

        def pins(row, role):
            values = row.get(role, ())
            if len(values) > 4096:
                raise ValueError("impact producer pin inventory exceeds its budget")
            return {(pin["path"], pin["sha256"]) for pin in values}

        by_output = {"llvm_translation": {}, "object": {}}
        for path, row in records:
            if row["stage"] in by_output:
                for pin in pins(row, "outputs"):
                    by_output[row["stage"]].setdefault(pin, []).append((path, row))
        llvm_pin, elf_pin = (str(llvm), sha256_file(llvm)), (str(elf), sha256_file(elf))
        for link_path, link in records:
            if link.get("stage") != "elf" or elf_pin not in pins(link, "outputs"):
                continue
            objects = {path: row for pin in pins(link, "inputs") for path, row in by_output["object"].get(pin, ())}
            for object_path, obj in objects.items():
                obj_pins = pins(obj, "outputs") & pins(link, "inputs")
                if obj.get("stage") != "object" or len(obj_pins) != 1:
                    continue
                translations = {
                    path: row for pin in pins(obj, "inputs") for path, row in by_output["llvm_translation"].get(pin, ())
                }
                for translation_path, translation in translations.items():
                    ll_pins = pins(translation, "outputs") & pins(obj, "inputs")
                    if (
                        translation.get("stage") != "llvm_translation"
                        or len(ll_pins) != 1
                        or llvm_pin not in pins(translation, "inputs")
                    ):
                        continue
                    paths = (translation_path, object_path, link_path)
                    for path in paths:
                        I.verify(path)
                    for path, digest in (*ll_pins, *obj_pins):
                        _reopen_product({"path": path, "sha256": digest}, root=root)
                    selected.append((next(iter(ll_pins))[1], next(iter(obj_pins))[1], paths))
        identities = {(ll, obj) for ll, obj, _ in selected}
        if len(identities) != 1:
            raise ValueError("impact requires one actual LLVM/object/ELF producer chain")
        ll, obj = next(iter(identities))
        identity = {"emitted": llvm_pin[1], "elf": elf_pin[1], "llvm_ir": ll, "object": obj}
        records = [I.verify(path) for path in sorted({path for _, _, paths in selected for path in paths})]
    else:
        raise ValueError("impact has no original product join")
    if identity["emitted"] != compiled["llvm_mlir_sha256"] or identity["elf"] != compiled["elf_sha256"]:
        raise ValueError("impact LLVM/ELF product changed")
    return {"products": identity, "toolchain": _producer_identity(records)}


def _member(root, cell, index):
    identities = []
    for arm in ("baseline", "candidate"):
        value, pin = CAL._cheap_observation(root, cell, arm, index)
        if value["functional_status"] != "PASS":
            raise ValueError("impact functional screening is incomplete")
        grade = mapping_file(Path(pin[0])).get("original_grade")
        if not isinstance(grade, dict):
            raise ValueError("impact original grade is missing")
        identities.append(_products(root / f"m{index:03d}_{arm}", value["compiled"], grade))
    return identities[0] == identities[1]


def plan(document, *, roster, cheap_root, deadline):
    """Host-only consumer: unknown/missing/mutated evidence broadens checks."""
    cells = {(row["family"], row["capsule"]): row for row in document["cells"]} if document else {}
    rows, selected = [], []
    for index, member in enumerate(roster):
        state = "UNKNOWN"
        if time.monotonic() < deadline and member in cells:
            try:
                state = "unchanged_products" if _member(cheap_root, cells[member], index) else "changed_products"
            except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError, StageGateError):
                state = "UNKNOWN"
        if state != "unchanged_products":
            selected.append(member)
        rows.append({"family": member[0], "capsule": member[1], "impact": state})
    return {
        "schema": SCHEMA,
        "members": rows,
        "selected": [list(row) for row in selected],
        "basis": "original fresh compile/Spike, exact emitted and linked products, frozen input/toolchain/profile",
        "unknown_policy": "include member",
        "promotion": "DIAGNOSTIC_ONLY",
        "unchanged_meaning": "identical observed executable products; no new hardware or performance claim",
    }


def submit(coordinator, revision_id, *, timeout_s):
    """Use live original queue evidence; retained jobs and source owners stay fixed."""
    from .evaluation_pool import ComponentEvaluationPool

    coordinator.admit_submission(timeout_s=timeout_s)
    pool, policy = coordinator.pool, coordinator.pool.policy
    if type(pool) is not ComponentEvaluationPool or not pool.inputs.ordinary:
        raise StageGateError("impact selection requires the original ordinary issued queue")
    deadline = time.monotonic() + timeout_s
    with pool._lock:
        revision = pool.revisions.get(revision_id)
        revision.verify(authority=policy.edit_authority)
        roster = tuple(sorted(policy.validate_search_inputs()))
        observations = pool.observations()
        matches = [
            row
            for row in observations
            if row["revision_id"] == revision_id and row["kind"] == FAST.KIND and row["state"] == "completed"
        ]
        document, root, evidence = None, None, None
        if matches:
            observed = matches[-1]
            job = pool._get(observed["job_id"])
            document = pool.inputs.validate_result(observed["result"], revision, FAST.KIND)
            root = job.root / "rtl"
            evidence = sha256_file(job.root / "feedback.json")
        impact = plan(document, roster=roster, cheap_root=root, deadline=deadline)
        impact.update(
            {
                "revision_id": revision_id,
                "source_sha256": revision.source_sha256,
                "baseline_sha256": pool.baseline_revision.source_sha256,
                "configuration_sha256": policy.search_closure_sha256,
                "cheap_evidence_sha256": evidence,
            }
        )
        # Recheck immutable source/profile and live result before dispatch. No
        # saved plan or author-supplied invalidation list is accepted here.
        pool.inputs.validate()
        if matches and pool.status(matches[-1]["job_id"])["state"] != "completed":
            raise StageGateError("impact selection evidence changed during its original join")
        members = tuple(tuple(row) for row in impact["selected"])
        if not members:
            return {"status": "no_changed_products", "impact": impact, "job": None}
        remaining = int(deadline - time.monotonic())
        if remaining < 1:
            raise StageGateError("impact selection exhausted its original broker budget")
        job = coordinator.submit(revision_id, kind=SELECT.KIND, selected_members=members, timeout_s=remaining)
        return {"status": "queued_diagnostic", "impact": impact, "job": job}
