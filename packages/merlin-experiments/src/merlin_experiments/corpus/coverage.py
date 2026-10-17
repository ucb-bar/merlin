"""Read-only conformance coverage for one verified Phase 0 source run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from merlin.targetgen import conformance

from ..spec import SpecError
from .preparation import _members, source_run


def _selected_contract(spec_doc: dict, inputs: dict | None) -> dict | None:
    """Return only a contract whose selected identity matches the requirement."""
    contract = (inputs or {}).get("capability_contract")
    expected = ((spec_doc.get("derivation") or {}).get("phase0_execution") or {}).get("contract_sha256")
    if not isinstance(contract, dict) or not isinstance(expected, str):
        return None
    raw = (json.dumps(contract, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("selected capability contract differs from the frozen requirement")
    return contract


def selected_cohort_coverage(
    spec_doc: dict,
    roots: list[Path],
    *,
    inputs: dict | None = None,
    labels: set[str] | None = None,
    phase: str = "phase1",
) -> dict:
    """Measure pure capsule axes; never borrow ambient target facts for admission.

    Composition needs a compiler-owned, exact-source-bound emitted host/device
    seam and execution evidence. Phase 0 has no such observer. Host-only and
    host-lane placement use the byte-bound selected contract, and memory regimes
    use selected facts.
    """
    from merlin.targetgen.contract.materialize import cert_capsule_cover

    if phase not in {"phase1", "phase2"}:
        raise ValueError("cohort coverage phase must be phase1 or phase2")
    from .accelerator_scope import conformance_for_scope, selected_scope

    scope = selected_scope(inputs)
    spec_doc = conformance_for_scope(spec_doc, inputs)
    labels = {"public", "dev"} if labels is None else set(labels)
    got = cert_capsule_cover(roots, labels=labels, tile_dim=(spec_doc.get("boundaries") or {}).get("tile_edge"))
    have = set(got.get("cells") or [])
    want = {row["cell"] for row in spec_doc.get("cells") or []}
    result = {
        "n_required": len(want),
        "n_covered": len(want & have),
        "uncovered": sorted(want - have),
        "corpus_cells": sorted(have),
        "extra_cells": sorted(have - want),
        "scope": "exact admitted capsule bytes and selected conformance requirements only",
    }
    if scope is not None:
        result["capsule_scope"] = {**scope, "standalone_host_tests": "outside_selected_scope"}
    readers = {
        "shape_geometry": conformance._geometry_gap,
        "scope": conformance._scope_gap,
        "epilogue": conformance._epilogue_gap,
        "groups": conformance._group_gap,
        "carried_state": conformance._carried_state_gap,
        "conv_geometry": conformance._conv_geometry_gap,
    }
    for axis, reader in readers.items():
        if axis == "scope" and phase == "phase2":
            performance = ((spec_doc.get("scope") or {}).get("performance") or {})
            result["source_scope_instances"] = len(
                ((spec_doc.get("scope") or {}).get("typed_required_instances") or {}).get("instances") or []
            )
            if performance.get("schema") != "merlin.phase0.performance_scope.v1":
                result[axis] = {
                    "status": "not_measured",
                    "reason": "selected requirement lacks exact SW/emitter-derived Phase 2 scope",
                }
                continue
            from merlin_experiments.phase0.performance_scope import validate_performance_scope

            performance = validate_performance_scope(spec_doc["scope"])
            if performance.get("status") == "unresolved":
                result[axis] = {
                    "status": "not_measured",
                    "reason": "exact source/SW/emitter Phase 2 scope remains unresolved",
                    "n_unresolved": len(performance.get("unresolved") or []),
                }
                continue
            if performance.get("status") == "no_eligible_chain":
                result[axis] = {
                    "status": "not_applicable", "n_required": 0, "uncovered": [],
                    "n_software_refused": len(performance.get("excluded") or []),
                }
                continue
            required = performance.get("required")
        else:
            required = (spec_doc.get(axis) or {}).get("required")
        result[axis] = (
            reader(required, roots, labels=labels)
            if required is not None
            else {"status": "not_measured", "reason": "selected requirement predates this coverage axis"}
        )
    composition = (spec_doc.get("composition") or {}).get("required")
    if composition is None:
        result["composition"] = {
            "status": "not_measured",
            "phase": "phase0",
            "reason": "selected conformance requirement predates the composition axis",
            "required": None,
        }
    elif not composition:
        result["composition"] = {"status": "not_applicable", "phase": "phase0", "n_required": 0, "uncovered": []}
    else:
        result["composition"] = {
            "status": "not_measured",
            "phase": "phase0",
            "reason": (
                "the Phase 0 source-pool observer has no compiler-owned receipt for emission and "
                "execution of the required host/device composition"
            ),
            "required": composition,
            "phase1_receipt_required": {
                "selected_capture": (
                    "exact captured MLIR bytes and digest bound to the selected application inventory "
                    "and conformance requirement"
                ),
                "selected_capsule": (
                    "admitted capsule label, descriptor-tree digest, and authoritative program bytes/digest"
                ),
                "compiler_execution": (
                    "selected compiler and dependency identity, invocation, input program digest, "
                    "and emitted host/device artifact bytes/digest"
                ),
                "lowering_correspondence": (
                    "compiler-owned mapping from every source operation and typed SSA crossing to "
                    "the emitted host/device routes, including shape, dtype, and value preservation"
                ),
                "execution": (
                    "target-visible execution trace and functional verdict bound to that same emitted artifact"
                ),
            },
        }
    selected_contract = _selected_contract(spec_doc, inputs)
    from merlin.targetgen import boundary

    for axis, reader in (
        ("host_lane", boundary.host_lane_coverage),
        ("host_only", boundary.host_only_coverage),
    ):
        required = (spec_doc.get(axis) or {}).get("families" if axis == "host_only" else "required")
        if required is not None and not required:
            result[axis] = {"status": "not_applicable", "n_required": 0, "uncovered": []}
        elif selected_contract is None:
            result[axis] = {
                "status": "not_measured",
                "reason": "exact selected capability contract is absent from coverage inputs or requirement",
                "required": required,
            }
        else:
            result[axis] = reader(
                spec_doc,
                roots,
                labels=labels,
                capability_contract=selected_contract,
            )
    memory = (spec_doc.get("memory_mapping") or {}).get("required")
    if memory is None:
        result["memory_mapping"] = {
            "status": "not_measured",
            "reason": "selected requirement predates the memory-mapping axis",
        }
    elif not memory:
        result["memory_mapping"] = {"status": "not_applicable", "n_required": 0, "uncovered": []}
    else:
        selected = inputs or {}
        raw = selected.get("raw_facts_utf8")
        selected_digest = (selected.get("evidence") or {}).get("raw_facts_sha256")
        required_digest = ((spec_doc.get("derivation") or {}).get("phase0_execution") or {}).get(
            "raw_facts_sha256"
        )
        if not isinstance(raw, str) or not selected_digest or not required_digest:
            result["memory_mapping"] = {
                "status": "not_measured",
                "reason": "exact selected RTL facts are absent from coverage inputs or requirement",
                "required": memory,
            }
        else:
            digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            if digest != selected_digest or digest != required_digest:
                raise ValueError("selected RTL facts differ from the frozen requirement or coverage inputs")
            from merlin.targetgen import memory_regime as MR

            corpus = MR.corpus_regimes(roots, str(spec_doc.get("target") or ""), labels=labels, facts=json.loads(raw))
            if corpus["capacity_rows"] is None:
                result["memory_mapping"] = {
                    "status": "not_measured",
                    "reason": "selected RTL facts did not resolve an operand-store capacity for the cohort",
                    "required": memory,
                    "facts_sha256": digest,
                }
            else:
                gap = MR.uncovered_regimes({"by_regime": memory}, corpus)
                gap.update(
                    status="ok",
                    covered_by=corpus["by_regime"],
                    region_counts=(spec_doc.get("memory_mapping") or {}).get("region_counts") or {},
                    facts_sha256=digest,
                )
                result["memory_mapping"] = gap
    return result


def _public_category_roots(corpus: Path) -> tuple[list[Path], int]:
    """Resolve the category roots understood by the shared capsule scanners.

    Phase 0 owns ``corpus/<category>/<name>/capsule.yaml``; the scanners take
    category roots and look one directory below each. Validate that no source
    member is silently omitted before asking them to classify coverage.
    """
    members = _members(corpus)
    all_paths = {path.relative_to(corpus).as_posix() for path in corpus.rglob("capsule.yaml")}
    expected_paths = {f"{key}/capsule.yaml" for key in members}
    if all_paths != expected_paths:
        raise SpecError("phase-0 corpus has capsule descriptors outside category/member layout")
    provenance = yaml.safe_load((corpus / "MANIFEST.yaml").read_text(encoding="utf-8"))
    if not isinstance(provenance, dict):
        raise SpecError("phase-0 corpus manifest must be a mapping")
    declared = provenance.get("generated")
    if not isinstance(declared, list) or not declared or any(not isinstance(key, str) for key in declared):
        raise SpecError("phase-0 corpus manifest must declare generated public members")
    public = {key: document for key, (_, document) in members.items() if not key.startswith("hidden/")}
    if len(declared) != len(set(declared)) or set(declared) != set(public):
        raise SpecError("phase-0 corpus manifest does not account for every public capsule")
    hidden = len(members) - len(public)
    held_out = provenance.get("held_out") or {}
    if not isinstance(held_out, dict) or held_out.get("n_generated", 0) != hidden:
        raise SpecError("phase-0 corpus manifest does not account for every hidden capsule")
    n_public = sum(document.get("label") == "public" for document in public.values())
    if not n_public:
        raise SpecError("phase-0 corpus has no public-labelled capsules to measure")
    roots = sorted({corpus / key.split("/", 1)[0] for key in public})
    return roots, n_public


def inspect_run(run_dir: Path, spec_path: Path) -> dict:
    """Measure public source-pool coverage without grading or admitting a cohort.

    The completed run and its frozen inputs are verified before reading capsules.
    The conformance spec is an explicit, separately hashed diagnostic input; this
    command neither changes that reference nor turns a source-pool match into a
    numerical or hardware verdict.
    """
    source = run_dir.expanduser().resolve(strict=True)
    plan, _, corpus = source_run(source)
    selected = spec_path.expanduser().resolve(strict=True)
    if not selected.is_file():
        raise SpecError("conformance spec must be a regular file")
    raw = selected.read_bytes()
    try:
        spec = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise SpecError(f"invalid conformance YAML: {selected}") from exc
    if not isinstance(spec, dict) or not isinstance(spec.get("cells"), list):
        raise SpecError("conformance spec must contain a cells list")
    target = plan["target"]
    if spec.get("target") != target:
        raise SpecError(f"conformance spec target {spec.get('target')!r} differs from run target {target!r}")
    edge = (spec.get("boundaries") or {}).get("tile_edge")
    if edge is not None and (type(edge) is not int or edge < 1):
        raise SpecError("conformance spec tile_edge must be a positive integer or absent")
    category_roots, n_public = _public_category_roots(corpus)
    from merlin_experiments.phase0.coverage_commitment import read_inputs

    inputs = read_inputs(corpus)
    if inputs is not None and inputs.get("target") != target:
        raise SpecError("frozen coverage inputs belong to a different target")
    result = selected_cohort_coverage(spec, category_roots, inputs=inputs, labels={"public"})
    if not result["corpus_cells"]:
        raise SpecError("phase-0 public capsules yielded no classifiable coverage cells")
    return {
        "schema_version": 1,
        "target": target,
        "phase0_run": str(source),
        "corpus": str(corpus),
        "spec": {"path": str(selected), "sha256": hashlib.sha256(raw).hexdigest()},
        "scope": "generated public source pool; not admitted, graded, or certified",
        "n_public_capsules_scanned": n_public,
        "coverage": result,
    }
