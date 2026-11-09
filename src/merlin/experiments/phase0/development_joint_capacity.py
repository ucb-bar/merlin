"""Exact conditional joint-buffer accounting for retained public contractions.

This proves byte/row capacity of an explicit full-padded-buffer placement. It
does not prove that a compiler realizes that placement, banking, or overlap.
"""

from __future__ import annotations

from . import development_sources as D

SCHEMA = "merlin.phase0.public_joint_store_capacity.v1"


def _positive(value):
    return type(value) is int and 0 < value < 2**63


def _up(value, granule):
    return (value + granule - 1) // granule * granule


def selected_stores(facts, *, operand_memory, accumulator_memory):
    """Select existing address-space roles; never infer them from memory names."""
    if (
        not isinstance(facts, dict)
        or not isinstance(operand_memory, str)
        or not isinstance(accumulator_memory, str)
        or operand_memory == accumulator_memory
    ):
        raise ValueError("joint stores require two explicit distinct memory selections")
    body = facts.get("facts", facts)
    if not isinstance(body, dict) or not isinstance(body.get("memories"), list):
        raise ValueError("joint stores require an explicit memory fact list")
    memories = body["memories"]
    if any(not isinstance(row, dict) for row in memories):
        raise ValueError("joint store memory entries must be fact mappings")
    prefix = "facts." if "facts" in facts else ""
    stores = {}
    for role, name in (("operand", operand_memory), ("accumulator", accumulator_memory)):
        matches = [(index, row) for index, row in enumerate(memories) if row.get("name") == name]
        if len(matches) != 1 or matches[0][1].get("view_kind") != "address_space":
            raise ValueError("selected joint store has no unique address-space view")
        index, row = matches[0]
        bits = row.get("row_bits_rtl")
        row_bytes = row.get("row_bytes")
        if row_bytes is None and _positive(bits) and bits % 8 == 0:
            row_bytes = bits // 8
        width = row.get("elem_bits")
        if (
            any(not _positive(row.get(k)) for k in ("bytes", "banks", "depth"))
            or not _positive(row_bytes)
            or not _positive(width)
            or width % 8
            or row["bytes"] != row["banks"] * row["depth"] * row_bytes
            or row_bytes % (width // 8)
            or (bits is not None and bits != 8 * row_bytes)
        ):
            raise ValueError("selected joint store byte/row/element geometry is inconsistent")
        census = [
            dict(fact_path=f"{prefix}census.physical_memories[{i}]", selected_fact=physical)
            for i, physical in enumerate((body.get("census") or {}).get("physical_memories", []))
            if physical.get("name") == name or physical.get("backs") == name
        ]
        stores[role] = dict(
            name=name,
            capacity_bytes=row["bytes"],
            row_bytes=row_bytes,
            element_bytes=width // 8,
            total_rows=row["banks"] * row["depth"],
            fact_path=f"{prefix}memories[{index}]",
            selected_fact=row,
            capacity_authority="unique selected logical address-space view; physical aliases are not summed",
            physical_census_views=census,
            representation_discrepancies=[
                dict(
                    fact_path=entry["fact_path"],
                    logical_capacity_bytes=row["bytes"],
                    physical_view_bytes=entry["selected_fact"].get("bytes"),
                    resolution="UNKNOWN: preserve original census copies/backing notes; no substitution",
                )
                for entry in census
                if entry["selected_fact"].get("bytes") != row["bytes"]
            ],
            bank_mapping="UNKNOWN: counts do not establish address decoding or port legality",
            independent_ports_and_overlap="UNKNOWN: neither bank count nor byte capacity establishes these",
        )
    return stores


def prove(declaration, *, stores, axis_granules, reserved_bytes):
    """Prove both one-copy and two-copy simultaneous A+B+C placements.

    Axis granules and reservations are explicit experimental premises. Two
    banks/contexts do not establish two legal concurrent buffer copies.
    """
    if (
        not isinstance(axis_granules, dict)
        or set(axis_granules) != {"M", "N", "K"}
        or any(not _positive(x) for x in axis_granules.values())
        or not isinstance(reserved_bytes, dict)
        or set(reserved_bytes) != {"operand", "accumulator"}
        or set(stores) != {"operand", "accumulator"}
    ):
        raise ValueError("joint proof needs explicit axis granules, stores and reservations")
    for role, store in stores.items():
        reserve = reserved_bytes[role]
        if type(reserve) is not int or reserve < 0 or reserve > store["capacity_bytes"] or reserve % store["row_bytes"]:
            raise ValueError("joint reservations must be bounded complete rows")
    description = D.describe(declaration)
    if (
        declaration["pattern"] != "matmul"
        or [x["dtype"] for x in description["inputs"]] != ["int8", "int8"]
        or [x["dtype"] for x in description["outputs"]] != ["int32"]
        or stores["operand"]["element_bytes"] != 1
        or stores["accumulator"]["element_bytes"] != 4
    ):
        raise ValueError("joint proof supports original signed-i8 rank2 dot and raw-i32 output only")
    shape = declaration["geometry"]
    padded = {axis: _up(shape[axis], granule) for axis, granule in axis_granules.items()}
    m, k, n = (padded[x] for x in ("M", "K", "N"))
    sp, acc = stores["operand"], stores["accumulator"]

    def buffers(reduction):
        return {
            "lhs": (m, _up(reduction * sp["element_bytes"], sp["row_bytes"])),
            "rhs": (reduction, _up(n * sp["element_bytes"], sp["row_bytes"])),
            "output": (m, _up(n * acc["element_bytes"], acc["row_bytes"])),
        }

    def placement(copies, reduction):
        layouts = buffers(reduction)
        result = {}
        for role, names in (("operand", ("lhs", "rhs")), ("accumulator", ("output",))):
            store, end, intervals = stores[role], reserved_bytes[role], []
            if end:
                intervals.append(dict(owner="reserved", begin=0, end=end, bytes=end))
            for copy in range(copies):
                for name in names:
                    rows, stride = layouts[name]
                    size = rows * stride
                    intervals.append(
                        dict(
                            owner=name,
                            copy=copy,
                            begin=end,
                            end=end + size,
                            bytes=size,
                            logical_rows=rows,
                            row_stride_bytes=stride,
                        )
                    )
                    end += size
            result[role] = dict(
                intervals=intervals,
                required_bytes=end,
                required_rows=end // store["row_bytes"],
                capacity_bytes=store["capacity_bytes"],
                remaining_bytes=store["capacity_bytes"] - end,
                overflow_bytes=max(0, end - store["capacity_bytes"]),
                fits=end <= store["capacity_bytes"],
                disjoint_linear_intervals=True,
            )
        return result

    variants = {}
    for copies in (1, 2):
        current = placement(copies, k)
        # Exact monotone search includes row-stride rounding, independently of
        # axis granules. A slab-byte division alone is wrong for rectangular rows.
        high = sp["capacity_bytes"] // buffers(axis_granules["K"])["rhs"][1] // axis_granules["K"]
        low = 0
        while low < high:
            middle = (low + high + 1) // 2
            trial = placement(copies, middle * axis_granules["K"])
            if all(row["fits"] for row in trial.values()):
                low = middle
            else:
                high = middle - 1
        maximum = low * axis_granules["K"]
        variants[str(copies)] = dict(
            buffer_copies=copies,
            stores=current,
            fits=all(row["fits"] for row in current.values()),
            limiting_stores=[role for role, row in current.items() if not row["fits"]],
            maximum_aligned_K_same_M_N=maximum,
            next_aligned_K=maximum + axis_granules["K"],
            at_last_fit=bool(maximum) and k == maximum,
            at_first_spill=bool(maximum) and k == maximum + axis_granules["K"],
        )
    regime = "fits_double" if variants["2"]["fits"] else "fits_single" if variants["1"]["fits"] else "spills"
    return dict(
        schema=SCHEMA,
        declaration_sha256=description["declaration_sha256"],
        useful_shape=dict(shape),
        padded_shape=padded,
        useful_products=description["scalar_products"],
        full_padded_products=m * k * n,
        axis_granules=dict(axis_granules),
        reserved_bytes=dict(reserved_bytes),
        regime=regime,
        placements=variants,
        proof_scope="exact conditional linear row allocation of complete padded A+B+C buffers",
        physical_placement_proven=False,
        double_buffer_execution_proven=False,
        required_actual_witnesses=[
            "same complete buffers/copies and reservations simultaneously live",
            "actual decoded allocation addresses and nonaliasing ownership",
            "bank/address decoding, ports, command ordering and completion",
            "spill/reload or bounded-panel realization when full-buffer placement fails",
        ],
        roofline="UNKNOWN: no emitted traffic, issue, host costs or legal overlap supplied",
    )
