"""Withheld source and linked-build witness for an ordered f32 CPU prefix scan.

This proves a closed prepared-MLIR algorithm and binds its exact reviewed host
admission, not frontend or numerical equivalence of the linked image. The width is a selected
compiler premise only after the existing post-build index observation joins it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from merlin.common.digest import is_sha256
from merlin.compiler.frontend.patterns.capture_normalization import normalize_capture_mlir
from merlin.compiler.ir import mlir_query as mq
from merlin.development.backend.qualification.source_witnesses import control_guards as control

# ---- private_ordered_scan_support --------------------------------------------------------------


def begin(
    source: Path, raw_sha: str, normalized_sha: str, receipt_sha: str, selected: Mapping[str, Any] | None
) -> dict:
    """Reparse complete normalized source and prove every tagged ordered scan root."""
    from merlin.development.backend.qualification.source_witnesses.ordered_scan import (
        PENDING,
        SCOPE,
        _digest,
        _need,
        _recognize,
    )

    _need(all(is_sha256(value) for value in (raw_sha, normalized_sha, receipt_sha)), "source identity is malformed")
    raw = source.read_bytes()
    _need(sha256(raw).hexdigest() == raw_sha, "raw source bytes changed")
    try:
        text, normalization = normalize_capture_mlir(raw.decode("utf-8"))
        module = mq.parse(text)
        module.verify()
    except Exception as exc:  # noqa: BLE001 - parser/verifier are a fail-closed trust boundary
        raise ValueError(f"ordered f32 scan: complete source failed verification: {exc}") from exc
    _need(
        normalization["input_sha256"] == raw_sha
        and sha256(text.encode()).hexdigest() == normalized_sha == normalization["output_sha256"],
        "normalized source identity differs",
    )
    parsed = tuple(mq.walk(module))
    # A provenance-tagged f64 carry initialization cannot become an empty
    # witness merely because the loop itself lost its frontend/region tag.
    carry_regions = {
        mq.attr_str(op, "prov.region_id")
        for op in parsed
        if mq.attr_str(op, "prov.aten") == "aten.cumsum.default"
        and mq.op_name(op) == "tensor.empty"
        and any(str(result.type).endswith("xf64>") for result in op.results)
    }
    for region in carry_regions:
        _need(
            isinstance(region, str)
            and sum(
                mq.op_name(op) == "scf.for"
                and mq.attr_str(op, "prov.region_id") == region
                and mq.attr_str(op, "prov.aten") == "aten.cumsum.default"
                for op in parsed
            )
            == 1,
            "tagged wide scan carry has no unique ordered loop",
        )
    roots = [
        (i, op)
        for i, op in enumerate(parsed)
        if mq.op_name(op) == "scf.for" and mq.attr_str(op, "prov.aten") == "aten.cumsum.default"
    ]
    if roots:
        _need(
            isinstance(selected, Mapping) and selected.get("schema") == "merlin.selected-index-lowering.v1",
            "selected index observation is missing",
        )
        bits = selected.get("index_bits")
        _need(type(bits) is int and 2 <= bits <= 128, "selected index width is malformed")
    else:
        bits = None
    occurrences = [_recognize(op, parsed, i, bits) for i, op in roots]
    return {
        "status": PENDING,
        "scope": SCOPE,
        "raw_source_sha256": raw_sha,
        "normalized_source_sha256": normalized_sha,
        "capture_receipt_sha256": receipt_sha,
        "n_source_operations": len(parsed),
        "selected_index_observation": dict(selected) if selected is not None else None,
        "count": len(occurrences),
        "occurrences": occurrences,
        "occurrences_sha256": _digest(occurrences),
        "source_verified": True,
        "admission_verified": False,
        "admissions": [],
        "admissions_sha256": _digest([]),
    }


def admit(
    witness: dict,
    parsed: tuple[Any, ...],
    source_rows: Mapping[int, Mapping[str, Any]],
    software: Mapping[str, Any],
    capability: Mapping[str, Any],
    cap_map: Mapping[str, Any],
    host: Mapping[str, Any],
) -> set[int]:
    """Admit only reviewed algorithm roots, then account for their closed body.

    The inventory marks an scf.for as support, so its ordinary hardware screen
    is not applicable. Here the independent oracle sees the proved *tensor*
    f32 reduction, not the loop's index operands. Unknown or eligible hardware
    is never converted into host support by this accounting step.
    """
    from merlin.compiler.ir import mlir_query as mq
    from merlin.development.backend.qualification.source_witnesses.ordered_scan import (
        PENDING,
        _digest,
        _need,
        _semantic_signature,
    )
    from merlin.development.targetgen.applications import operation_accounting as oa
    from merlin.target.capabilities import semantic_families as sf
    from merlin.target.capabilities.eligibility import (
        RegionDescriptor,
        is_eligible,
        providers_from_contract,
        undetermined_families_from_contract,
    )

    _need(witness.get("status") == PENDING and witness.get("source_verified") is True, "source proof is absent")
    _need(witness.get("admission_verified") is False, "source admission was already recorded")
    occ = witness["occurrences"]
    roots = {item["root_ordinal"] for item in occ}
    all_ordinals = {ordinal for item in occ for ordinal in item["all_ordinals"]}
    _need(len(parsed) == witness["n_source_operations"], "parsed source roster differs")
    _need(
        len(roots) == len(occ) and len(all_ordinals) == sum(len(item["all_ordinals"]) for item in occ),
        "ordered scan source proofs overlap",
    )
    admissions = []
    for item in occ:
        root = item["root_ordinal"]
        row = source_rows.get(root)
        _need(
            isinstance(row, Mapping)
            and row.get("mlir_operation") == "scf.for"
            and row.get("operation") == "aten.cumsum.default"
            and row.get("disposition") == "support_required"
            and root in (row.get("ordinals") or [])
            and set(row.get("ordinals") or []) <= roots,
            "ordered scan root inventory is absent or mixed with unproved loops",
        )
        family = sf.from_prov(mq.attr_str(parsed[root], "prov.family"), row["operation"])
        _need(family == "reduction", "scan family has no canonical reduction meaning")
        _need(item["input_type"] == item["output_type"] == "f32", "scan tensor precision differs")
        refusals = []
        for rank in sorted({1, len(item["input_shape"])}):
            verdict = is_eligible(
                RegionDescriptor(
                    op=row["operation"],
                    family=family,
                    in_dtype=item["input_type"],
                    out_dtype=item["output_type"],
                    rank=rank,
                ),
                dict(cap_map),
                undetermined=undetermined_families_from_contract(dict(capability)),
                providers=providers_from_contract(dict(capability)),
            )
            _need(
                not verdict.eligible
                and not verdict.undetermined
                and verdict.refusal in {"undeclared_family", "input_dtype", "result_dtype"},
                "scan tensor computation is hardware-eligible or eligibility is unresolved",
            )
            refusals.append({"rank": rank, "family": verdict.family, "refusal": verdict.refusal})
        screened = oa.admit_operation_row(
            dict(row),
            software_spec=dict(software),
            capability_contract=dict(capability),
            capability_map=dict(cap_map),
            host_capabilities=dict(host),
            observed=_semantic_signature(item),
            source_operations=tuple(parsed[i] for i in row["ordinals"]),
        )
        sw = [
            decision
            for decision in screened["software_admissions"]
            if decision["status"] == "admitted"
            and decision.get("review_status") == "reviewed"
            and decision.get("placement") == "host"
        ]
        profiles = [
            profile
            for profile in screened["host_admission"]["profiles"]
            if profile["status"] == "admitted" and profile.get("reviewed") is True
        ]
        # The exported accelerator software specification need not duplicate
        # pure-host operations. The exact selected host profile owns this
        # admission; any independently declared host software rule must still
        # be unique and reviewed, never a second source of invented capability.
        _need(software.get("status") == "reviewed", "selected software specification is unreviewed")
        _need(len(sw) <= 1 and len(profiles) == len(host) == 1, "scan has no unique reviewed host admission")
        profile = profiles[0]
        decisions = [
            decision
            for decision in profile["decisions"]
            if decision["status"] == "admitted" and decision.get("review_status") == "reviewed"
        ]
        _need(len(decisions) == 1, "scan host declaration is ambiguous or unreviewed")
        sw_id, host_id = (sw[0]["declaration"] if sw else None), decisions[0]["declaration"]
        sw_rules = [rule for rule in software["operations"] if rule.get("id") == sw_id]
        host_rules = [
            rule for rule in host[profile["profile"]]["capability_spec"]["operations"] if rule.get("id") == host_id
        ]
        _need(
            len(host_rules) == 1
            and (sw_id is None or len(sw_rules) == 1 and row["operation"] in sw_rules[0].get("ops", []))
            and row["operation"] in host_rules[0].get("ops", []),
            "scan requires exact reviewed root declarations, not family-only admission",
        )
        _need(
            is_sha256(profile.get("package_sha256")) and is_sha256(profile.get("capability_spec_sha256")),
            "selected host profile has no exact bytes",
        )
        admissions.append(
            {
                "root_ordinal": root,
                "all_ordinals": item["all_ordinals"],
                "software_declaration": sw_id,
                "host_profile": profile["profile"],
                "host_declaration": host_id,
                "host_package_sha256": profile["package_sha256"],
                "host_spec_sha256": profile["capability_spec_sha256"],
                "host_dtype_strategy": profile["dtype_strategy"],
                "software_record_sha256": _digest(software),
                "capability_record_sha256": _digest(capability),
                "semantic_signature": _semantic_signature(item),
                "hardware_refusals": refusals,
            }
        )
    witness["admissions"] = admissions
    witness["admissions_sha256"] = _digest(admissions)
    witness["admission_verified"] = True
    return all_ordinals


def link(source: Mapping[str, Any], actual_index: Mapping[str, Any], linked_build: Mapping[str, Any]) -> None:
    from merlin.development.backend.qualification.source_witnesses.ordered_scan import (
        _BUILD,
        FIELD,
        LINKED,
        PENDING,
        _need,
    )

    witness = source.get(FIELD)
    _need(
        isinstance(witness, dict)
        and witness.get("status") == PENDING
        and witness.get("source_verified") is True
        and witness.get("admission_verified") is True,
        "source witness or reviewed admission is absent or not pending",
    )
    _need(
        witness.get("selected_index_observation") == source.get("selected_index_observation")
        and control._record_matches_selected(actual_index, witness.get("selected_index_observation")),
        "actual linked index lowering differs from selected premise",
    )  # noqa: PLC2701
    _need(_valid_admissions(witness), "reviewed source algorithm admission is incomplete")
    _need(
        isinstance(linked_build, Mapping)
        and set(linked_build) == _BUILD
        and all(is_sha256(linked_build[key]) for key in _BUILD),
        "linked build tuple is incomplete",
    )
    witness["status"] = LINKED
    witness["linked_build"] = dict(linked_build)


def _valid_admissions(witness: Mapping[str, Any]) -> bool:
    from merlin.development.backend.qualification.source_witnesses.ordered_scan import (
        _ADMISSION,
        _digest,
        _semantic_signature,
    )

    occ, admissions = witness.get("occurrences"), witness.get("admissions")
    if not isinstance(occ, list) or not isinstance(admissions, list):
        return False
    try:
        digest = _digest(admissions)
        occurrence_digest = _digest(occ)
    except (TypeError, ValueError):
        return False
    if (
        witness.get("admission_verified") is not True
        or type(witness.get("count")) is not int
        or witness["count"] != len(occ)
        or occurrence_digest != witness.get("occurrences_sha256")
        or len(admissions) != len(occ)
        or not is_sha256(witness.get("admissions_sha256"))
        or digest != witness["admissions_sha256"]
    ):
        return False
    for proof, record in zip(occ, admissions, strict=True):
        if not isinstance(proof, Mapping) or not isinstance(record, Mapping) or set(record) != _ADMISSION:
            return False
        shape = proof.get("input_shape")
        if not isinstance(shape, list) or not shape or any(type(dim) is not int or dim <= 0 for dim in shape):
            return False
        if (proof.get("input_type"), proof.get("output_type"), proof.get("carry_type")) != ("f32", "f32", "f64"):
            return False
        if (
            type(record["root_ordinal"]) is not int
            or record["root_ordinal"] != proof.get("root_ordinal")
            or not isinstance(record["all_ordinals"], list)
            or any(type(ordinal) is not int for ordinal in record["all_ordinals"])
            or record["all_ordinals"] != proof.get("all_ordinals")
            or _digest(record["semantic_signature"]) != _digest(_semantic_signature(proof))
        ):
            return False
        if any(
            not isinstance(record[key], str) or not record[key].strip()
            for key in ("host_profile", "host_declaration", "host_dtype_strategy")
        ):
            return False
        if record["software_declaration"] is not None and (
            not isinstance(record["software_declaration"], str) or not record["software_declaration"].strip()
        ):
            return False
        if any(
            not is_sha256(record[key])
            for key in ("host_package_sha256", "host_spec_sha256", "software_record_sha256", "capability_record_sha256")
        ):
            return False
        refusals = record["hardware_refusals"]
        if (
            not isinstance(refusals, list)
            or [item.get("rank") for item in refusals if isinstance(item, Mapping)] != sorted({1, len(shape)})
            or any(
                not isinstance(item, Mapping)
                or set(item) != {"rank", "family", "refusal"}
                or type(item["rank"]) is not int
                or item["family"] != "reduction"
                or item["refusal"] not in {"undeclared_family", "input_dtype", "result_dtype"}
                for item in refusals
            )
        ):
            return False
    return True


def linked_complete(source: Mapping[str, Any], entry: Mapping[str, Any], candidate_sha: str) -> bool:
    """Mandatory empty or populated roster joined to the same whole-program ELF."""
    from merlin.development.backend.qualification.source_witnesses.ordered_scan import (
        _BUILD,
        _TOP,
        FIELD,
        LINKED,
        SCOPE,
        _digest,
    )

    if not isinstance(source, Mapping) or not isinstance(entry, Mapping) or not is_sha256(candidate_sha):
        return False
    w = source.get(FIELD)
    if (
        not isinstance(w, Mapping)
        or set(w) != _TOP
        or w.get("status") != LINKED
        or w.get("scope") != SCOPE
        or w.get("source_verified") is not True
        or not _valid_admissions(w)
    ):
        return False
    if any(
        not is_sha256(w.get(key)) for key in ("raw_source_sha256", "normalized_source_sha256", "capture_receipt_sha256")
    ):
        return False
    if (w["raw_source_sha256"], w["normalized_source_sha256"], w["capture_receipt_sha256"]) != (
        source.get("source_sha256"),
        source.get("normalized_source_sha256"),
        source.get("capture_receipt_sha256"),
    ):
        return False
    if (
        type(w.get("n_source_operations")) is not int
        or w["n_source_operations"] < 1
        or w["n_source_operations"] != source.get("n_source_operations")
        or entry.get("source_sha256") != source.get("source_sha256")
    ):
        return False
    occ = w.get("occurrences")
    if (
        not isinstance(occ, list)
        or type(w.get("count")) is not int
        or w["count"] != len(occ)
        or not is_sha256(w.get("occurrences_sha256"))
        or _digest(occ) != w["occurrences_sha256"]
    ):
        return False
    selected = w.get("selected_index_observation")
    if selected != source.get("selected_index_observation") or not control._record_matches_selected(
        entry.get("index_lowering"), selected
    ):  # noqa: PLC2701
        return False
    bits = selected.get("index_bits") if isinstance(selected, Mapping) else None
    if type(bits) is not int or not 2 <= bits <= 128:
        return False
    seen_roots, seen_all = set(), set()
    for item in occ:
        if not isinstance(item, Mapping) or set(item) != {
            "root_ordinal",
            "all_ordinals",
            "input_shape",
            "output_shape",
            "axis_candidates",
            "total",
            "stride",
            "axis_extent",
            "input_type",
            "output_type",
            "carry_type",
            "index_bits_premise",
            "region_id",
        }:
            return False
        root, all_ordinals = item["root_ordinal"], item["all_ordinals"]
        if (
            type(root) is not int
            or root < 0
            or root >= w["n_source_operations"]
            or root in seen_roots
            or not isinstance(all_ordinals, list)
            or root not in all_ordinals
            or not all(type(i) is int and 0 <= i < w["n_source_operations"] for i in all_ordinals)
            or all_ordinals != sorted(set(all_ordinals))
            or seen_all.intersection(all_ordinals)
        ):
            return False
        seen_roots.add(root)
        seen_all.update(all_ordinals)
        shape, axes = item["input_shape"], item["axis_candidates"]
        maximum = min((1 << (bits - 1)) - 1, (1 << 63) - 1)
        if not isinstance(shape, list) or not shape or any(type(dim) is not int or dim <= 0 for dim in shape):
            return False
        expected_axes = [
            axis
            for axis in range(len(shape))
            if math.prod(shape[axis + 1 :]) == item.get("stride") and shape[axis] == item.get("axis_extent")
        ]
        if (
            type(item.get("total")) is not int
            or math.prod(shape) != item["total"]
            or not 0 < item["total"] <= maximum
            or item["output_shape"] != shape
            or not isinstance(axes, list)
            or not axes
            or any(type(axis) is not int or axis < 0 or axis >= len(shape) for axis in axes)
            or axes != expected_axes
            or type(item.get("stride")) is not int
            or type(item.get("axis_extent")) is not int
            or not 0 < item["stride"] <= maximum
            or not 0 < item["axis_extent"] <= maximum
        ):
            return False
        if (
            item["index_bits_premise"] != bits
            or item["input_type"] != "f32"
            or item["output_type"] != "f32"
            or item["carry_type"] != "f64"
            or not isinstance(item["region_id"], str)
            or not item["region_id"]
        ):
            return False
    linked = w.get("linked_build")
    return bool(
        isinstance(linked, Mapping)
        and set(linked) == _BUILD
        and all(is_sha256(linked[key]) for key in _BUILD)
        and linked["candidate_tree_sha256"] == candidate_sha
        and linked["capture_tree_sha256"] == entry.get("capture_tree_sha256")
        and linked["elf_sha256"] == entry.get("elf_sha256")
    )
