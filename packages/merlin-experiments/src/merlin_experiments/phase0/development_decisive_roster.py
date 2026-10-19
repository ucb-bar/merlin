"""Additive source/control requirements with explicit emitted-effect obligations.

This derives public source requests, not physical capacity or queue qualification.
The original development cohort, weights, tolerances and refusals stay unchanged.
"""

from __future__ import annotations

from math import prod

from . import development_controls as C
from . import development_envelopes as E

SCHEMA = "merlin.phase0.decisive_development_controls.v1"


def _positive(value):
    return type(value) is int and value > 0


def requests(*, basis, axis_granules, resource_sha256, placement_assumptions):
    """Freeze adjacent joint-store cases under declared conditional placement.

    Full padded A, B and C are simultaneously live in this *declared* placement.
    A compiler using bounded panels does not exercise this placement merely by
    executing the source. Reservation/copy assumptions are never inferred from
    loop-context or bank counts. Actual effects must witness every interval.
    """
    if (
        not isinstance(basis, dict)
        or not isinstance(axis_granules, dict)
        or set(axis_granules) != {"M", "N", "K"}
        or not all(_positive(v) for v in axis_granules.values())
        or any(
            not _positive(basis.get(k))
            for k in ("operand_store_bytes", "accumulator_bytes", "operand_bytes", "accumulate_bytes")
        )
        or not isinstance(resource_sha256, str)
        or len(resource_sha256) != 64
        or any(c not in "0123456789abcdef" for c in resource_sha256)
        or not isinstance(placement_assumptions, dict)
        or set(placement_assumptions) != {"operand_reserved_bytes", "accumulator_reserved_bytes", "buffer_copies"}
        or any(
            type(placement_assumptions[k]) is not int or placement_assumptions[k] < 0
            for k in ("operand_reserved_bytes", "accumulator_reserved_bytes")
        )
        or not isinstance(placement_assumptions["buffer_copies"], list)
        or not placement_assumptions["buffer_copies"]
        or len(set(placement_assumptions["buffer_copies"])) != len(placement_assumptions["buffer_copies"])
        or any(not _positive(v) or v > 2 for v in placement_assumptions["buffer_copies"])
    ):
        raise ValueError("decisive controls need resource pins, axis granules and explicit placement assumptions")
    gm, gn, gk = (axis_granules[k] for k in ("M", "N", "K"))
    rows = []

    def add(pattern, geometry, requirement, *, controls=None, obligations=(), **metadata):
        declaration = E.declaration(pattern, geometry, ordinal=len(rows))
        declaration["name"] = f"decision_{len(rows):04d}"
        declaration["max_logical_elements"] = 2**30
        if controls is not None:
            declaration.update(schema=C.SCHEMA, controls=controls)
        rows.append(
            {
                "declaration": declaration,
                "basis": "explicit_public_decision_control",
                "family": requirement,
                "role": "development_control",
                "purposes": [requirement],
                "required_emitted_witness": list(obligations),
                "physical_mechanism_status": "UNEXERCISED",
                **metadata,
            }
        )

    for uses in (2, 4, 8):
        add(
            "rhs_versions",
            {"M": [1 + i % 3 for i in range(uses)], "K": 2 * gk + 1, "N": 2 * gn - 1},
            "immutable_rhs_reuse",
            controls={"mode": "shared"},
            obligations=["same immutable storage/version", "bounded RHS device lease", "fewer original RHS DMA bytes"],
        )
    for mode in ("distinct_equal", "distinct_changed", "updated_epoch", "intervening_clobber"):
        add(
            "rhs_versions",
            {"M": [1, gm - 1 if gm > 1 else 1, gm + 1], "K": 2 * gk + 1, "N": 2 * gn - 1},
            "rhs_invalidation",
            controls={"mode": mode},
            obligations=[
                "operand identity and epoch correspondence",
                "clobber ownership",
                "complete live-out equality",
            ],
        )
    for stride in (1, 2):
        for escape in (False, True):
            add(
                "strided_conv_escape",
                {"H": 7, "W": 9, "CI": 3, "CO": gn + 1, "KH": 3, "KW": 3, "SH": 2, "SW": 1, "PH": 1, "PW": 1},
                "strided_padded_packing_escape",
                controls={"source_stride": stride, "escape_packing": escape},
                obligations=[
                    "source stride and virtual padding",
                    "escaping packing remains available",
                    "complete original convolution work",
                ],
            )
    for epoch in ("append_only", "replace_prefix"):
        add(
            "kv_epoch",
            {"steps": 2, "prefix_tokens": 7, "query_tokens": 1, "kv_heads": 1, "query_heads": 3, "head_dim": 7},
            "gqa_kv_epoch",
            controls={"epoch": epoch},
            obligations=[
                "GQA head-group mapping",
                "old and replacement KV epochs",
                "all QK/PV products and state live-outs",
            ],
        )
    for axis, granule in axis_granules.items():
        for remainder in sorted({0, 1, granule - 1}):
            shape = {"M": 2 * gm, "N": 2 * gn, "K": 3 * gk}
            shape[axis] = 2 * granule + remainder
            add(
                "matmul",
                shape,
                "independent_axis_tail",
                obligations=["actual packet dimensions", "complete useful and padded product accounting"],
                axis=axis,
                remainder=remainder,
                granule=granule,
            )
    windows = basis.get("command_windows", {})
    if isinstance(windows, dict) and _positive(windows.get("execute")):
        for count in (windows["execute"] - 1, windows["execute"], windows["execute"] + 1):
            if 1 <= count <= 64:
                add(
                    "resident_rhs_reuse",
                    {"M": [1] * count, "K": gk, "N": gn},
                    "queue_window",
                    obligations=[
                        "actual outstanding command occupancy reaches declared window",
                        "configuration/fence drain and completion",
                        "complete all-operation outputs",
                    ],
                    declared_window=windows["execute"],
                    source_contraction_count=count,
                )
    for copies in placement_assumptions["buffer_copies"]:
        sp = basis["operand_store_bytes"] - placement_assumptions["operand_reserved_bytes"]
        acc = basis["accumulator_bytes"] - placement_assumptions["accumulator_reserved_bytes"]
        n = gn
        m = (acc // (copies * basis["accumulate_bytes"] * n) // gm) * gm
        k = (sp // (copies * basis["operand_bytes"] * (m + n)) // gk) * gk if m else 0
        if min(m, k) < 1:
            rows.append(
                {
                    "status": "source_refused",
                    "basis": "explicit_public_joint_placement",
                    "family": "joint_sp_acc",
                    "reason": "declared reservations leave no complete simultaneous tile",
                    "buffer_copies": copies,
                }
            )
            continue
        for label, reduction in (("last_fit", k), ("first_spill", k + gk)):
            shape = {"M": m, "K": reduction, "N": n}
            operands = copies * (m * reduction + reduction * n) * basis["operand_bytes"]
            accumulators = copies * m * n * basis["accumulate_bytes"]
            add(
                "matmul",
                shape,
                "joint_sp_acc",
                obligations=[
                    "actual joint A+B+reserved live placement",
                    "actual simultaneous accumulator and buffer copies",
                    "nonoverlapping legal store intervals",
                    "spill/reload or bounded panel evidence",
                ],
                joint_boundary={
                    "label": label,
                    "buffer_copies": copies,
                    "operand_bytes_including_reservation": operands + placement_assumptions["operand_reserved_bytes"],
                    "accumulator_bytes_including_reservation": accumulators
                    + placement_assumptions["accumulator_reserved_bytes"],
                    "operand_capacity_bytes": basis["operand_store_bytes"],
                    "accumulator_capacity_bytes": basis["accumulator_bytes"],
                    "simultaneous_full_padded_operands": True,
                    "reservation_origin": "explicit public experimental assumption, not current compiler proof",
                    "actual_placement_proven": False,
                },
            )
    for requirement, reason in (
        ("inplace_alias_mutation", "SSA epoch replacement does not establish in-place alias/storage mutation"),
        ("packet_width_bank_crossings", "source geometry cannot fix physical DMA addresses or private bank mapping"),
        ("dma_compute_overlap", "source independence cannot establish legal asynchronous completion or overlap"),
        (
            "attention_score_mask_softmax_escape",
            "selected complete softmax/probability-conversion numeric contract unavailable",
        ),
    ):
        rows.append(
            {
                "status": "source_refused",
                "basis": "explicit_public_decision_obligation",
                "family": requirement,
                "reason": reason,
                "physical_mechanism_status": "UNEXERCISED",
            }
        )
    return {
        "schema": SCHEMA,
        "scope": "public_development_not_validation",
        "resource_sha256": resource_sha256,
        "axis_granules": axis_granules,
        "placement_assumptions": placement_assumptions,
        "requests": rows,
        "original_cohort_changed": False,
        "qualification": "source controls only; actual effects remain separate",
    }


def matrix(manifest):
    """Complete requirement-to-source/evidence matrix without inferred success."""
    rows = []
    for member in manifest["members"]:
        request = member["request"]
        rows.append(
            {
                "ordinal": member["ordinal"],
                "identity": member["request_sha256"],
                "requirement": request["family"],
                "source_sha256": member.get("loader_identity", {}).get("sha256"),
                "reference_sha256": member.get("reference_identity", {}).get("sha256"),
                "source_status": member["status"],
                "source_reason": member.get("reason"),
                "source_work": member.get("description", {}).get("scalar_products"),
                "source_output_elements": sum(
                    prod(r["shape"]) for r in member.get("description", {}).get("outputs", [])
                ),
                "capture": "NOT_RUN",
                "writer": "NOT_RUN",
                "baseline_execution": "NOT_RUN",
                "actual_effects": "UNEXERCISED",
                "timing": "UNQUALIFIED",
                "public_exclusion": "NOT_ADMITTED",
                "required_emitted_witness": request.get("required_emitted_witness", []),
                "joint_boundary": request.get("joint_boundary"),
            }
        )
    return {
        "schema": "merlin.phase0.decisive_development_matrix.v1",
        "members": rows,
        "required_total": len(rows),
        "physical_qualified": 0,
        "timing_qualified": 0,
        "source_reference_only": True,
    }
