"""Deterministic retained-call selection and Phase 0 packaging, without execution claims.

The capsule writer belongs to targetgen. This owner binds declared cohorts and
capture bytes, stages one run-owned corpus, and exposes only hidden commitments.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import yaml

from ..corpus.phase_selection import generate_phase_selections

SCHEMA = "merlin.phase0.core_aten_stage.v1"
COHORTS = ("public", "hidden", "host_guard")


def _json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def declaration(recipe: Path) -> dict | None:
    document = yaml.safe_load(recipe.read_bytes()) or {}
    block = document.get("core_aten")
    if block is None:
        return None
    if document.get("capsule_policy") != "derived_only" or not isinstance(block, dict):
        raise ValueError("Core ATen requires an explicit derived_only recipe mapping")
    if any(document.get(key) for key in ("capsules", "sweeps", "performance")):
        raise ValueError("Core ATen stage cannot mix authored capsule or sweep membership")
    return block


def _path(recipe: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Core ATen input requires an explicit path")
    lexical = (recipe.parent / value).absolute()
    if lexical != lexical.resolve(strict=True):
        raise ValueError("Core ATen inputs must be ordinary paths without aliases")
    return lexical


def selection(recipe: Path) -> tuple[list[dict], dict[str, str]]:
    """Read only explicitly selected cases/captures, never discover another cohort."""
    from merlin.targetgen.core_aten_capture import case_capture_name, descriptor_specialization_matches

    block = declaration(recipe)
    if block is None or set(block.get("cohorts", {})) != set(COHORTS):
        raise ValueError("Core ATen stage requires public, hidden and host_guard declarations")
    workers = block.get("workers", 1)
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Core ATen workers must be an integer in 1..8")
    paths: dict[str, str] = {}
    overlay = _path(recipe, block.get("overlay"))
    paths[str(overlay)] = _sha(overlay)
    flow = block.get("numeric_flow")
    if flow is not None:
        if flow.get("policy") != "spec_int8":
            raise ValueError("numeric_flow requires an explicitly supported conformant policy")
        for key in ("software_spec", "host_capabilities", "execution_provider", "rtl_facts"):
            source = _path(recipe, flow[key])
            paths[str(source)] = _sha(source)
        package = _path(recipe, flow["device_package"])
        for source in sorted(package.rglob("*.py")):
            if not any(part in {"out", "__pycache__", ".git"} for part in source.relative_to(package).parts):
                paths[str(source)] = _sha(source)
    batches, identities = [], set()
    retained_public, retained_private = [], []
    for cohort in COHORTS:
        rows = block["cohorts"][cohort]
        if cohort == "host_guard" and rows == [] and block.get("allow_empty_host_guards") is True:
            continue
        if not isinstance(rows, list) or not rows:
            raise ValueError("every Core ATen cohort must declare at least one case selection")
        for row in rows:
            if "retained_corpus" in row:
                from merlin.targetgen.capsule_common import load_capsule
                from merlin.targetgen.core_aten_capsules import load_private

                retained = _path(recipe, row["retained_corpus"])
                if retained.name != cohort or set(row) != {"retained_corpus"}:
                    raise ValueError("retained capsules must preserve their source category")
                entries = []
                for member in sorted(retained.glob("*/capsule.yaml")):
                    capsule = load_capsule(member.parent)
                    load_private(capsule)
                    name = capsule["name"]
                    if name in identities or capsule["label"] != ("hidden" if cohort == "hidden" else "public"):
                        raise ValueError("retained capsule identity or visibility conflicts")
                    identities.add(name)
                    answers = retained.parent / "_private" / cohort / name
                    (retained_private if cohort == "hidden" else retained_public).append(
                        (str(member.parent), member.parent)
                    )
                    retained_private.append((str(answers), answers))
                    for owner in (member.parent, answers):
                        for source in sorted(owner.rglob("*")):
                            if source.is_symlink() or not (source.is_file() or source.is_dir()):
                                raise ValueError("retained capsule contains indirect or special entries")
                            if source.is_file():
                                paths[str(source)] = _sha(source)
                    entries.append(dict(id=name, source=str(member.parent), answers=str(answers)))
                if not entries:
                    raise ValueError("retained corpus is empty")
                batches.append(dict(cohort=cohort, retained=entries))
                continue
            lane = row.get("lane", "host-guard" if cohort == "host_guard" else "device")
            if lane not in {"device", "host-guard", "any", "admission", "truth"} or (
                cohort == "host_guard" and lane not in {"host-guard", "truth"}
            ):
                raise ValueError("invalid selected capsule lane")
            if lane in {"admission", "truth"} and flow is None:
                raise ValueError("admission-derived lanes require numeric_flow")
            prepare_numeric = row.get("prepare_numeric", False)
            if "prepare_numeric" in row and (lane != "any" or type(prepare_numeric) is not bool):
                raise ValueError("prepare_numeric is an explicit boolean for optional-device rows only")
            if prepare_numeric and flow is None:
                raise ValueError("optional numeric preparation requires the existing numeric_flow")
            cases_path = _path(recipe, row.get("cases"))
            captures = _path(recipe, row.get("captures"))
            if not captures.is_dir():
                raise ValueError("capture owner must be an ordinary directory")
            paths[str(cases_path)] = _sha(cases_path)
            corpus = json.loads(cases_path.read_bytes())
            if "cases" not in corpus and isinstance(corpus.get("selected_cases"), list):
                corpus = {**corpus, "cases": corpus["selected_cases"]}
            wanted = row.get("select")
            if not isinstance(wanted, list) or not wanted or len(wanted) != len(set(wanted)):
                raise ValueError("case selection must be a nonempty unique list of case_id or overload")
            by_id = {case.get("case_id", case["overload"]): case for case in corpus["cases"]}
            if len(by_id) != len(corpus["cases"]) or set(wanted) - set(by_id):
                raise ValueError("selected cases are missing or ambiguous")
            cases = [by_id[key] for key in sorted(wanted)]
            denominator_sha256 = corpus.get("denominator_sha256")
            del by_id, corpus
            for case in cases:
                name = case_capture_name(case).replace("-", "_")
                if name in identities:
                    raise ValueError("PUBLIC, HIDDEN and HOST-GUARD capsule identities must be disjoint")
                identities.add(name)
                source = captures / case_capture_name(case)
                if source.is_symlink() or not source.is_dir():
                    raise ValueError("selected capture is missing or indirect")
                capture = json.loads((source / "capture.json").read_bytes())
                if (
                    not (
                        capture.get("status") == "captured_exact"
                        or (
                            capture.get("status") == "captured_lost_exact_provenance"
                            and descriptor_specialization_matches(case, capture)
                        )
                    )
                    or capture.get("overload") != case["overload"]
                    or capture.get("capture_meta", {}).get("opaque") != 0
                    or not capture.get("capture_meta", {}).get("result_contract")
                ):
                    raise ValueError(
                        "selected capture must be exact, nonopaque, overload-bound "
                        "and carry a full-call result contract"
                    )
                # The batch builder may read several ABI/input sidecars. Bind complete
                # selected-directory membership, not only its program/capture receipt.
                for member in sorted(source.rglob("*")):
                    if member.is_symlink() or not (member.is_dir() or member.is_file()):
                        raise ValueError("selected capture contains indirect or special entries")
                    if member.is_file():
                        paths[str(member)] = _sha(member)
            batches.append(
                {
                    "cohort": cohort,
                    "corpus": {"cases": cases, "denominator_sha256": denominator_sha256},
                    "captures": str(captures),
                    **({"lane": lane} if "lane" in row else {}),
                    **({"prepare_numeric": prepare_numeric} if "prepare_numeric" in row else {}),
                }
            )
    if retained_public or retained_private:
        from merlin.targetgen.sandbox.bwrap import _validate_host_sources

        _validate_host_sources(retained_public, retained_private)
    return batches, dict(sorted(paths.items()))


def _selection_sha(recipe: Path, batches: list[dict]) -> str:
    flow = declaration(recipe).get("numeric_flow")
    payload = batches if flow is None else {"batches": batches, "numeric_flow": flow}
    return hashlib.sha256(_json(payload)).hexdigest()


def input_paths(recipe: Path) -> list[Path]:
    """Input inventory used by the existing run/resume receipt owner."""
    _, paths = selection(recipe)
    block = declaration(recipe)
    if block.get("derivation"):
        paths[str(_path(recipe, block["derivation"]))] = ""
    # Bind the writer's implementation with this stage, not an arbitrary callback.
    from merlin.common.paths import module_source_path

    for module in (
        "merlin.targetgen.core_aten_capsules",
        "merlin.targetgen.core_aten_batch",
        "merlin.targetgen.core_aten_capture",
        "merlin.targetgen.core_aten_cases",
        "merlin.targetgen.golden_store",
    ):
        paths[str(module_source_path(module).resolve())] = ""
    return [Path(path) for path in sorted(paths)]


def _requirement(batches: list[dict], receipt: dict) -> dict:
    """Declare finite retained-call obligations without claiming execution coverage."""
    return {
        "schema": "merlin.phase0.core_aten_conformance.v1",
        "target": receipt["target"],
        "scope": receipt["scope"],
        "selection_sha256": receipt["selection_sha256"],
        "cohorts": {
            cohort: {
                "count": sum(_batch_count(b) for b in batches if b["cohort"] == cohort),
                "scored": cohort != "host_guard",
                **(
                    {
                        "unscored_count": sum(
                            _batch_count(b) for b in batches if b["cohort"] == cohort and b.get("lane") == "host-guard"
                        )
                    }
                    if cohort != "host_guard"
                    and any(b["cohort"] == cohort and b.get("lane") == "host-guard" for b in batches)
                    else {}
                ),
            }
            for cohort in COHORTS
        },
        "obligations": [
            "exact nonopaque overload-bound full-call capture",
            "complete selected input bytes and disjoint public/hidden calls",
            "independent owner-only full-call answers excluded from candidate grants",
            "Phase 1 numerical agreement and declared lane execution evidence",
        ],
        "execution_coverage_status": "unverified",
    }


def _batch_count(batch: dict) -> int:
    return len(batch["retained"]) if "retained" in batch else len(batch["corpus"]["cases"])


def verify_synthesis(recipe: Path, requirement: dict, synthesis: dict) -> dict:
    """Recompute membership and receipts; input digests alone do not prove selection."""
    batches, paths = selection(recipe)
    receipt_path = _path(recipe, declaration(recipe).get("derivation"))
    receipt = json.loads(receipt_path.read_bytes())
    expected = {
        "schema": SCHEMA,
        "target": requirement.get("target"),
        "inputs": paths,
        "selection_sha256": _selection_sha(recipe, batches),
        "scope": "retained full-call packaging; no numerical or hardware verdict",
    }
    if receipt != expected or requirement != _requirement(batches, expected):
        raise ValueError("Core ATen conformance selection changed; rerun corpus derive")
    if (
        synthesis.get("core_aten_selection")
        != {
            "schema": SCHEMA,
            "selection_sha256": receipt["selection_sha256"],
            "receipt_sha256": _sha(receipt_path),
        }
        or synthesis.get("capsules") != []
    ):
        raise ValueError("Core ATen selected synthesis differs from the retained-call derivation")
    return {"selection_sha256": receipt["selection_sha256"], "execution_coverage_status": "unverified"}


def derive(recipe: Path, output: Path, *, target: str, descriptor: Path) -> dict:
    batches, paths = selection(recipe)
    output = output.absolute()
    if output.exists():
        raise ValueError("Core ATen derivation output already exists")
    output.mkdir(parents=True, mode=0o700)
    receipt = {
        "schema": SCHEMA,
        "target": target,
        "inputs": paths,
        "selection_sha256": _selection_sha(recipe, batches),
        "scope": "retained full-call packaging; no numerical or hardware verdict",
    }
    (output / "selection.json").write_bytes(_json(receipt))
    (output / "selection.json").chmod(0o600)
    document = yaml.safe_load(recipe.read_bytes())
    # Resolve authored owners once; generated recipe never discovers siblings.
    block = document["core_aten"]
    block["overlay"] = str(_path(recipe, block["overlay"]))
    for rows in block["cohorts"].values():
        for row in rows:
            for key in ("retained_corpus",) if "retained_corpus" in row else ("cases", "captures"):
                row[key] = str(_path(recipe, row[key]))
    block["derivation"] = str(output / "selection.json")
    (output / "recipe.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
    (output / "recipe.yaml").chmod(0o600)
    from .profiles import synthesis_input_identity

    requirement = output / "requirements.yaml"
    requirement.write_text(yaml.safe_dump(_requirement(batches, receipt), sort_keys=False))
    synthesis = output / "synth.yaml"
    synthesis.write_text(
        yaml.safe_dump(
            {
                "provenance": {
                    "selected_inputs": synthesis_input_identity(
                        conformance_spec=requirement, recipe=output / "recipe.yaml", descriptor=descriptor
                    )
                },
                "core_aten_selection": {
                    "schema": SCHEMA,
                    "selection_sha256": receipt["selection_sha256"],
                    "receipt_sha256": _sha(output / "selection.json"),
                },
                "capsules": [],
            },
            sort_keys=False,
        )
    )
    for member in (requirement, synthesis):
        member.chmod(0o600)
    return {
        "schema": SCHEMA,
        "recipe": str(output / "recipe.yaml"),
        "conformance_spec": str(requirement),
        "synth_profile": str(synthesis),
        "selection_sha256": receipt["selection_sha256"],
        "counts": {cohort: sum(_batch_count(b) for b in batches if b["cohort"] == cohort) for cohort in COHORTS},
    }


def generate(
    recipe: Path,
    output: Path,
    *,
    target: str,
    capability_contract: Path | None = None,
    rtl_facts: Path | None = None,
) -> list[Path]:
    from merlin.targetgen.core_aten_capsules import write_capsules

    batches, paths = selection(recipe)
    block = declaration(recipe)
    receipt_path = _path(recipe, block.get("derivation"))
    receipt = json.loads(receipt_path.read_bytes())
    if receipt != {
        "schema": SCHEMA,
        "target": target,
        "inputs": paths,
        "selection_sha256": _selection_sha(recipe, batches),
        "scope": "retained full-call packaging; no numerical or hardware verdict",
    }:
        raise ValueError("Core ATen derivation selection changed; rerun corpus derive")
    output = output.absolute()
    if output != output.resolve():
        raise ValueError("Core ATen output cannot follow a directory alias")
    if output.name != "capsules" or output.parent.name != "phase0":
        raise ValueError("Core ATen output must be <run>/phase0/capsules")
    if output.exists():
        raise ValueError("Core ATen output already exists; select a fresh run")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".core-aten-", dir=output.parent))
    try:
        members, manifests, failures = [], [], []
        program_sets = {cohort: set() for cohort in COHORTS}
        for batch in batches:
            cohort = batch["cohort"]
            category = stage / cohort
            if "retained" in batch:
                from ..corpus.preparation import copy_input

                for row in batch["retained"]:
                    member = category / row["id"]
                    copy_input(Path(row["source"]), member, private=cohort == "hidden")
                    copy_input(Path(row["answers"]), stage / "_private" / cohort / row["id"], private=True)
                    call_digest = hashlib.sha256()
                    for filename in ("capsule.interface.mlir", "inputs.npz", "capsule.pytorch.py", "call.json"):
                        call_digest.update(filename.encode())
                        call_digest.update((member / filename).read_bytes())
                    program_sets[cohort].add(call_digest.hexdigest())
                    members.append(member.relative_to(stage).as_posix())
                manifests.append({"retained": batch["retained"]})
                continue
            lane = batch.get("lane", "host-guard" if cohort == "host_guard" else "device")
            device, scored = lane == "device", lane not in {"host-guard", "no-work"}
            # Each batch uses its own writer manifest; aggregate only our actual members.
            writer_manifest = category / "manifest.json"
            if writer_manifest.exists():
                writer_manifest.unlink()
            try:
                manifest = write_capsules(
                    batch["corpus"],
                    Path(batch["captures"]),
                    category,
                    stage / "_private" / cohort,
                    label="hidden" if cohort == "hidden" else "public",
                    tiers=("L2",),
                    **({"admission_lanes": True} if lane == "admission" else {}),
                    **({"truth_lanes": True} if lane == "truth" else {}),
                    **({"workers": block["workers"]} if block.get("workers", 1) != 1 else {}),
                    **(
                        {"numeric_flow": {**block["numeric_flow"], "target": target}}
                        if (device or lane in {"admission", "truth"} or batch.get("prepare_numeric"))
                        and block.get("numeric_flow")
                        else {}
                    ),
                )
            except ValueError as exc:
                if not getattr(exc, "failures", None):
                    raise
                failures.extend({"cohort": cohort, **row} for row in exc.failures)
                continue
            manifests.append(manifest)
            for row in manifest["capsules"]:
                member_lane = row["lane_expectation"] if lane in {"admission", "truth"} else lane
                device, scored = member_lane == "device", member_lane not in {"host-guard", "no-work"}
                member = category / row["id"]
                declaration_path = member / "capsule.yaml"
                capsule = yaml.safe_load(declaration_path.read_bytes())
                capsule["cohort"] = cohort if scored else "guard"
                capsule["scored"] = scored
                capsule["lane_expectation"] = member_lane
                capsule["semantic"] = {"must_accelerate": device}
                if member_lane == "any":
                    capsule.pop("lanes", None)
                else:
                    capsule["lanes"] = {"require" if device else "forbid": ["on_mesh"]}
                declaration_path.write_text(yaml.safe_dump(capsule, sort_keys=False))
                row.setdefault("digests", {})["capsule.yaml"] = _sha(declaration_path)
                call_digest = hashlib.sha256()
                for filename in ("capsule.interface.mlir", "inputs.npz", "capsule.pytorch.py", "call.json"):
                    call_digest.update(filename.encode())
                    call_digest.update((member / filename).read_bytes())
                program_sets[cohort].add(call_digest.hexdigest())
                members.append(member.relative_to(stage).as_posix())
            writer_manifest.unlink()  # Per-batch evidence belongs to the private ledger.
        if failures:
            failure_path = output.parent / "core-aten-failures.json"
            failure_path.write_bytes(_json({"selection_sha256": receipt["selection_sha256"], "failures": failures}))
            failure_path.chmod(0o600)
            raise ValueError(f"{len(failures)} capsule(s) refused; see {failure_path}")
        if program_sets["public"] & program_sets["hidden"]:
            raise ValueError("hidden and public selected program/input pairs overlap")
        private = stage / "_private"
        (private / "stage.json").write_bytes(_json({"receipt": receipt, "packaging": manifests}))
        for member in private.rglob("*"):
            member.chmod(0o700 if member.is_dir() else 0o600)
        if (stage / "hidden").is_dir():
            (stage / "hidden").chmod(0o700)
            for member in (stage / "hidden").rglob("*"):
                member.chmod(0o700 if member.is_dir() else 0o600)
        public_members = sorted(m for m in members if not m.startswith("hidden/"))
        manifest = {
            "generated_by": "merlin/contract/capsules/generate_corpus.py",
            "generated": public_members,
            "hand_authored": [],
            "held_out": {"n_generated": sum(m.startswith("hidden/") for m in members), "n_hand_authored": 0},
            "phase_corpora": {target: generate_phase_selections(public_members, performance_category="_perf")},
            "core_aten_stage": {
                "schema": SCHEMA,
                "private_ledger_sha256": _sha(private / "stage.json"),
                "scope": receipt["scope"],
            },
        }
        if capability_contract is not None:
            contract = yaml.safe_load(capability_contract.read_bytes())
            if not isinstance(contract, dict) or contract.get("name") != target:
                raise ValueError("selected capability contract names a different target")
            exported = stage / "software" / "contract.json"
            exported.parent.mkdir()
            exported.write_bytes(_json(contract))
            manifest["core_aten_stage"]["capability_contract_sha256"] = _sha(capability_contract)
        if rtl_facts is not None:
            facts = json.loads(rtl_facts.read_bytes())
            payload = facts.get("facts", facts)
            if not isinstance(payload, dict) or payload.get("target") != target:
                raise ValueError("selected facts name a different target")
            exported = stage / "hardware" / "effective-views" / "loaded-facts.json"
            exported.parent.mkdir(parents=True)
            exported.write_bytes(_json(facts if "facts" in facts else {"facts": facts}))
            manifest["core_aten_stage"]["rtl_facts_sha256"] = _sha(rtl_facts)
        (stage / "MANIFEST.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
        stage.rename(output)
        return [output / m for m in sorted(members)]
    finally:
        if stage.exists():
            shutil.rmtree(stage)
