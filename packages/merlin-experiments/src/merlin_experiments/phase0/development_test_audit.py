"""Decision coverage and redundancy of the complete public development roster.

This compares source requirements with retained declarations and independently
bound execution/refusal evidence. Shape or occurrence coverage never certifies
physical scheduling, complete reduced models, or performance suitability.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from math import ceil

SCHEMA = "merlin.phase0.public_development_decision_audit.v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _tags(member, tile):
    request = member["request"]
    declaration = request.get("declaration", {})
    pattern, geometry = declaration.get("pattern"), declaration.get("geometry", request.get("shape", {}))
    capacity = request.get("capacity", {})
    tags = set(request.get("purposes", []))
    if capacity:
        tags.add("capacity:" + capacity["family"] + ":" + capacity["regime"])
    if pattern:
        tags.add("pattern:" + pattern)
    if isinstance(geometry, dict) and set(geometry) == {"M", "N", "K"} and type(geometry["M"]) is int:
        m, n, k = (geometry[axis] for axis in ("M", "N", "K"))
        tails = [axis for axis in ("M", "N", "K") if geometry[axis] % tile]
        tags.add("aligned" if not tails else "tail:" + tails[0] if len(tails) == 1 else "compound_tails")
        for axis in ("M", "N", "K"):
            tags.add(f"remainder:{axis}:{geometry[axis] % tile}")
        if m == 1:
            tags.add("gemv")
        if m <= tile:
            tags.add("short_m")
        if n <= tile:
            tags.add("narrow_n")
        if k >= 4 * max(m, n):
            tags.add("deep_k_skew")
        if max(m, n) >= 8 * min(m, n):
            tags.add("aspect_skew")
    if pattern and pattern.startswith("conv_"):
        tags.add("conv:pointwise" if geometry["KH"] == geometry["KW"] == 1 else "conv:spatial")
        if geometry["SH"] > 1 or geometry["SW"] > 1:
            tags.add("conv:strided")
        if geometry["PH"] or geometry["PW"]:
            tags.add("conv:padded")
        if geometry["KH"] >= 7 and geometry["CI"] == 3:
            tags.add("conv:stem")
    return tags, pattern, geometry


_REQUIREMENTS = {
    "source_joint_shape_spectrum": ("public_joint_shape_spectrum", "source_joint_shapes"),
    "gemv": ("gemv", "source_geometry"),
    "short_m": ("short_m", "source_geometry"),
    "narrow_n": ("narrow_n", "source_geometry"),
    "deep_k_skew": ("deep_k_skew", "source_geometry"),
    "aspect_skew": ("aspect_skew", "source_geometry"),
    "aligned": ("aligned", "source_geometry"),
    "independent_m_tail": ("tail:M", "source_geometry"),
    "independent_n_tail": ("tail:N", "source_geometry"),
    "independent_k_tail": ("tail:K", "source_geometry"),
    "compound_tails": ("compound_tails", "source_geometry"),
    "rhs_nk_layout": ("pattern:matmul_rhs_nk", "source_layout"),
    "spatial_convolution": ("conv:spatial", "source_geometry"),
    "pointwise_convolution": ("conv:pointwise", "source_geometry"),
    "strided_convolution": ("conv:strided", "source_geometry"),
    "padded_convolution": ("conv:padded", "source_geometry"),
    "stem_convolution": ("conv:stem", "source_geometry"),
    "gqa_layout": ("GQA_layout", "source_layout_not_physical_ownership"),
    "kv_reuse": ("KV_reuse", "source_state_not_epoch_invalidation"),
    "complete_attention_semantics": ("online_softmax_if_numerically_legal", "pending_selected_numeric_contract"),
    "score_materialization_elimination": (
        "score_materialization_elimination",
        "candidate_mechanism_not_emitted_effect",
    ),
    "resident_rhs_chain": ("pattern:resident_rhs_reuse", "source_reuse_not_device_lease"),
    "prefix_action_chain": ("pattern:vision_prefix_action", "connected_pattern_not_complete_model"),
}

# These obligations cannot be inferred from a large matmul or a source label.
_UNPROVED = {
    "joint_live_a_b_reserved_and_pingpong_knees": "requires actual simultaneous A+B+reserved/buffered placement proof",
    "device_resident_rhs_and_mutation_alias_clobber_controls": (
        "requires checked device lease and invalidation controls"
    ),
    "escaping_im2col_and_source_stride": "requires escaping packed live-out so removal is not assumed legal",
    "packet_width_crossing_independent_of_capacity": "requires byte-pinned packet-width boundary members and execution",
    "bank_crossing_independent_of_capacity": "requires byte-pinned bank boundary members and execution",
    "single_ported_spad_read_write_priority": "requires actual bank arbitration and read/write priority evidence",
    "short_m_physical_row_floor": "GEMV geometry does not establish the selected WS minimum physical issue rows",
    "queue_configuration_fence_overlap": "five source matmuls do not prove queue pressure or legal overlap",
    "kv_epoch_invalidation": "reused KV source state does not prove invalidation after mutation/clobber",
    "complete_reduced_conv_model": "connected transitions omit the complete selected reduced model graph",
    "complete_reduced_decoder_model": "QK/PV mixing omits normalization, rotary, residual and complete MLP semantics",
    "complete_reduced_vision_action_model": "prefix/action patterns omit the complete selected reduced model graph",
    "selected_attention_mask_softmax_pv_contract": "no selected mixed-numeric probability/PV conversion supplied",
    "flashattention_score_escape_control": "complete semantic attention has no whole-score escaping control",
    "bias_activation_pooling_selected_contracts": "source forms and reviewed target operation contracts need admission",
}


def build(cohorts, *, evidence=None, tile=16):
    """Audit every retained member; never drop a refusal or alter membership.

    ``cohorts`` contains original complete manifest documents with caller-pinned
    identities. ``evidence`` is a separately byte-bound mapping by ``label:N``;
    this report copies its status, never promotes a copied verdict to a fresh run.
    """
    if type(tile) is not int or tile < 1 or not cohorts:
        raise ValueError("decision audit needs complete cohorts and a selected positive tile granule")
    evidence = {} if evidence is None else evidence
    if len({c["label"] for c in cohorts}) != len(cohorts):
        raise ValueError("decision audit cohort identities must be unique")
    members, duplicate_groups, padded_groups, spectrum = [], defaultdict(list), defaultdict(list), {}
    pressure = defaultdict(list)
    for cohort in cohorts:
        manifest = cohort["manifest"]
        if manifest.get("scope") != "public_architecture_informed_development_not_validation":
            raise ValueError("decision audit requires the explicit public development source boundary")
        if [m["ordinal"] for m in manifest["members"]] != list(range(len(manifest["members"]))):
            raise ValueError("decision audit requires the complete original member roster")
        for original in manifest["members"]:
            request = original["request"]
            tags, pattern, geometry = _tags(original, tile)
            tags.add(request["basis"])
            identity = f"{cohort['label']}:{original['ordinal']}"
            row = dict(
                identity=identity,
                cohort_sha256=cohort["manifest_sha256"],
                request_sha256=original["request_sha256"],
                source_sha256=original.get("loader_identity", {}).get("sha256"),
                pattern=pattern,
                geometry=geometry,
                tags=sorted(tags),
                original_status=original["status"],
                original_reason=original.get("reason"),
                numerical_policy=original.get("numeric_policy"),
                complete_call_boundary=original.get("performance_boundary"),
                evidence=evidence.get(identity, {"status": "UNAVAILABLE", "reason": "no execution evidence selected"}),
            )
            members.append(row)
            capacity, resources = request.get("capacity", {}), manifest.get("resource_basis") or {}
            if capacity.get("working_set") and resources:
                working = capacity["working_set"]
                ratios = [
                    working.get("operand_bytes", 0) / resources["operand_store_bytes"],
                    working.get("accumulator_bytes", 0) / resources["accumulator_bytes"],
                ]
                pressure[(cohort["label"], capacity["family"], capacity["variant"])].append(
                    dict(member=identity, ratio=max(ratios), working_set=working, regime=capacity["regime"])
                )
            declaration = request.get("declaration")
            if declaration:
                semantic = {k: v for k, v in declaration.items() if k not in {"name", "max_logical_elements"}}
                duplicate_groups[_digest(semantic)].append(identity)
                if set(geometry) == {"M", "N", "K"} and type(geometry["M"]) is int:
                    padded_groups[(pattern, *(ceil(geometry[a] / tile) for a in ("M", "N", "K")))].append(identity)
            if request.get("spectrum_member"):
                point = spectrum.setdefault(
                    request["spectrum_member"],
                    dict(occurrence_proxy=request["occurrence_proxy"], purposes=request["purposes"], members=[]),
                )
                point["members"].append(identity)
    requirements = []
    for requirement, (tag, scope) in _REQUIREMENTS.items():
        selected = [r for r in members if tag in r["tags"]]
        requirements.append(
            dict(
                requirement=requirement,
                scope=scope,
                members=[r["identity"] for r in selected],
                source_statuses=dict(Counter(r["original_status"] for r in selected)),
                execution_or_refusal=[dict(member=r["identity"], evidence=r["evidence"]) for r in selected],
                status="declared_source_members_not_qualification" if selected else "MISSING",
            )
        )
    requirements += [dict(requirement=k, status="UNPROVED", reason=v, members=[]) for k, v in _UNPROVED.items()]
    for axis in ("M", "N", "K"):
        for purpose in ("lower_extent:", "upper_extent:", "above_upper_extent:"):
            selected = [r["identity"] for r in members if purpose + axis in r["tags"]]
            requirements.append(
                dict(requirement=purpose + axis, members=selected, status="declared" if selected else "MISSING")
            )
        for remainder in sorted({0, 1, tile - 1}):
            selected = [r["identity"] for r in members if f"remainder:{axis}:{remainder}" in r["tags"]]
            requirements.append(
                dict(
                    requirement=f"ordered_tail:{axis}:qD+{remainder}",
                    members=selected,
                    status="declared" if selected else "MISSING",
                )
            )
    duplicates = [dict(semantic_sha256=k, members=v) for k, v in duplicate_groups.items() if len(v) > 1]
    brackets = []
    for key, points in sorted(pressure.items()):
        if key[1] not in {"knee", "deep_k", "short_m"}:
            continue
        for boundary in [0.5, 1.0]:
            below = [p for p in points if p["ratio"] <= boundary]
            above = [p for p in points if p["ratio"] > boundary]
            low = max(below, key=lambda p: p["ratio"]) if below else None
            high = min(above, key=lambda p: p["ratio"]) if above else None
            brackets.append(
                dict(
                    group=list(key),
                    boundary=boundary,
                    lower=low,
                    upper=high,
                    paired=low is not None and high is not None,
                    relative_bracket_width=(high["ratio"] - low["ratio"]) / boundary if low and high else None,
                    adjacent_legal_allocation_proven=False,
                    scope="coarse declared working-set ratios; joint live A+B+reserved placement unproved",
                )
            )
    return dict(
        schema=SCHEMA,
        scope="public development requirement audit; not validation, timing or cohort admission",
        adequate_for_full_model_performance_claim=False,
        members=members,
        requirements=requirements,
        capacity_brackets=brackets,
        occurrence_sampling=dict(
            scope="public configuration occurrence proxies; never latency/objective weights",
            distinct_spectrum_points=len(spectrum),
            occurrence_proxy_sum=sum(r["occurrence_proxy"] for r in spectrum.values()),
            points=spectrum,
            unresolved=[
                "expert/default depth provenance",
                "complete operator occurrence distribution",
                "host and attention cost weights",
            ],
        ),
        redundancy=dict(
            exact_semantic_duplicate_groups=duplicates,
            duplicate_members_beyond_first=sum(len(r["members"]) - 1 for r in duplicates),
            duplicate_fraction=sum(len(r["members"]) - 1 for r in duplicates) / len(members),
            same_padded_tuple_groups=[dict(key=list(k), members=v) for k, v in padded_groups.items() if len(v) > 1],
            interpretation=(
                "retain all members; identical padded tuples may intentionally exercise distinct tails or purposes"
            ),
        ),
        no_member_removed=True,
        physical_schedule_proof="UNAVAILABLE from source shapes; requires emitted effects and qualified evaluation",
    )
