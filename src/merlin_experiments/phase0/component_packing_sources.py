"""Prepare conditional storage probes without an accelerator allocation map."""

from __future__ import annotations

from merlin.targetgen.corpus_spec import dtype_info

from .component_generation import digest


def conditional_lengths(facts, *, numerical_semantics):
    """Match scalar *storage width* to complete local input partitions only.

    This relation does not prove that the original scalar is sent to this port,
    that its signedness is implemented, or that any tensor axis uses these bits.
    Opaque instance outputs remain observations, never original-input mappings.
    """
    try:
        _, dtype, _, integer = dtype_info(numerical_semantics["operand_dtype"])
    except KeyError:
        return [], []
    if not integer or not dtype.startswith("i") or not dtype[1:].isdigit():
        return [], []
    bits = int(dtype[1:])
    matches = [
        partition
        for partition in facts["partitions"]
        if partition["slice_width"] == bits and partition["root"]["kind"] == "module_input"
    ]
    return sorted({row["slice_count"] for row in matches}), matches


def append_sources(*, facts, spec, movement_owners, owner_links, program, unknown, obligations, unknowns):
    """Add ordinary bounded copy sources; retain every missing physical premise.

    Both fresh tensor orientations are generated so no RTL name assigns an axis.
    The observed length and neighboring lengths are conditional packing probes,
    not admitted hardware alignment, capacity, address or physical-tail cases.
    """
    lengths, matches = conditional_lengths(facts, numerical_semantics=spec["numerical_semantics"])
    for partition in matches:
        unknowns.append(
            unknown(
                "packing_mapping",
                digest(partition),
                "local bit partition lacks original scalar-to-command, axis allocation, "
                "capacity and physical-tail proof",
            )
        )
    unknowns.append(
        unknown(
            "packing_domain",
            "original_storage_and_resource_domain",
            "local equal partitions do not cover opaque, incomplete or unrecognized resource paths",
        )
    )
    if not lengths or len(movement_owners) != 1:
        unknowns.append(
            unknown(
                "packing_source",
                "scalar_storage_boundary",
                "conditional storage probes need a complete scalar-width input partition "
                "and unique reviewed movement owner",
            )
        )
        return
    owner = next(iter(movement_owners))
    links = [
        {
            "member": link["member"],
            "operations": [{"source": op, "owner": owner} for op in link["operations"]],
            "effects": [],
        }
        for link in owner_links[owner]
    ]
    for length in lengths:
        values = sorted({max(1, length - 1), length, length + 1})
        for orientation in ("M", "K"):
            other = "K" if orientation == "M" else "M"
            for cohort in ("functional_guard", "withheld_transfer"):
                obligations.append(
                    {
                        "id": "auto_storage_probe_" + digest((owner, length, orientation))[:16] + "_" + cohort,
                        "mandatory": True,
                        "cohort": cohort,
                        "operations": [owner],
                        "effects": [],
                        "expectation": "admitted_program",
                        "frontend": "mlir",
                        "base": {"op": "component_program", "kind": "model_slice", "program": program("movement")},
                        "axes": {
                            orientation: {"kind": "extent", "values": values},
                            other: {"kind": "extent", "values": [1] if cohort == "functional_guard" else [2]},
                        },
                        "interactions": [],
                        "semantic_basis": links,
                    }
                )
