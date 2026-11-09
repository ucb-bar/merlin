"""Bounded public counts from an original checked pair, without target data.

The original target checker owns derivation and current-product custody. This
projection grants no source/effect qualification, cycle price or hardware floor.
"""

from .contracts import StageGateError

SCHEMA = "merlin.diagnostic_schedule_metrics.v1"
SCOPE = "original_checked_accelerator_command_stream"
UNITS = {
    "workload_products": "products",
    "output_elements": "elements",
    "compute_commands": "commands",
    "operand_load_commands": "commands",
    "readout_commands": "commands",
    "configuration_commands": "commands",
    "completion_fences": "commands",
    "requested_operand_bytes": "bytes",
    "requested_readout_bytes": "bytes",
    "allocated_scratchpad_bytes": "bytes",
    "allocated_accumulator_bytes": "bytes",
}


def project(value):
    """Accept named scalar counts only; no values, paths or unscoped claims."""
    if (
        type(value) is not dict
        or set(value) != {"schema", "scope", "quantities"}
        or value["schema"] != SCHEMA
        or value["scope"] != SCOPE
        or type(value["quantities"]) is not dict
        or not {"workload_products", "output_elements"} <= set(value["quantities"]) <= set(UNITS)
    ):
        raise StageGateError("checked schedule metrics require the exact public count schema and scope")
    quantities = {}
    for name, counts in value["quantities"].items():
        if (
            type(counts) is not dict
            or set(counts) != {"baseline", "candidate"}
            or any(type(number) is not int or not 0 <= number < 1 << 64 for number in counts.values())
        ):
            raise StageGateError("checked schedule metrics require bounded nonnegative integer pairs")
        if name in {"workload_products", "output_elements"} and (
            counts["baseline"] <= 0 or counts["baseline"] != counts["candidate"]
        ):
            raise StageGateError("checked schedule metrics must preserve complete source products and outputs")
        quantities[name] = {
            **counts,
            "unit": UNITS[name],
            "baseline_minus_candidate": counts["baseline"] - counts["candidate"],
        }
    return dict(
        schema=SCHEMA,
        scope=SCOPE,
        quantities=quantities,
        measured_cycles=None,
        physical_traffic_bytes=None,
        utilization=None,
        attainment=None,
        interpretation=(
            "checked issued work, requests and allocations; no physical traffic, latency or complete roofline"
        ),
    )
