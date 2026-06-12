"""Analytical, architecture-independent cost model.

Evaluates a *plan* (a small set of interface flags: how many times the weight is packed and
loaded, whether the i32 accumulator intermediate is materialised per step, and the dispatch
count) against a Region Pressure Vector and a measurable cost model. Compute cycles are held
constant across plans so the model isolates the *interface* effect — exactly the quantity the
thesis cares about.

All parameters are measurable (``docs/dse.md`` forbids vague low/medium/high knobs):
``dispatch_fixed_cycles, pack_startup_cycles, pack_bytes_per_cycle, dram_bytes_per_cycle``
(plus ``mac_per_cycle`` for the constant compute term, and capacity knobs
``resident_store_bytes, accumulator_entries`` used for legality, not timing).
"""
from __future__ import annotations

import math


def evaluate_cost(rpv: dict, plan: dict, cost_model: dict) -> dict:
    """Return ``{cycles, energy, breakdown}`` for ``plan`` under ``cost_model``.

    ``plan`` keys: ``pack_count``, ``weight_loads``, ``per_step_intermediate`` (bool),
    ``dispatch_count``.
    """
    m = rpv["metrics"]
    cm = cost_model
    steps = max(int(m.get("steps", 1)), 1)

    macs = int(m.get("macs", 0))
    mac_per_cycle = cm.get("mac_per_cycle", 256)
    compute_cycles = math.ceil(macs / mac_per_cycle) if macs else 0

    dispatch_cycles = int(plan["dispatch_count"]) * cm["dispatch_fixed_cycles"]

    pack_bytes = int(m.get("pack_bytes", 0))
    pack_cycles = int(plan["pack_count"]) * (
        cm["pack_startup_cycles"] + (pack_bytes / cm["pack_bytes_per_cycle"] if pack_bytes else 0)
    )

    weight_bytes = int(m.get("weight_bytes", 0))
    io_bytes = steps * (int(m.get("input_bytes_step", 0)) + int(m.get("final_output_bytes_step", 0)))
    intermediate_bytes = (steps * int(m.get("intermediate_i32_bytes_step", 0))
                          if plan.get("per_step_intermediate") else 0)
    dram_bytes = int(plan["weight_loads"]) * weight_bytes + io_bytes + intermediate_bytes
    dram_cycles = dram_bytes / cm["dram_bytes_per_cycle"]

    # Exposing an interface is not free: residency pays a make_resident+evict setup, and
    # accumulator-commit pays for the accumulator/commit unit. These fixed, per-region costs
    # are what a low-reuse region cannot amortise — they are why I2/I3 do not always win.
    setup_cycles = 0
    if plan.get("resident_setup"):
        setup_cycles += cm.get("resident_setup_cycles", 0)
    if plan.get("accumulator_setup"):
        setup_cycles += cm.get("accumulator_setup_cycles", 0)

    cycles = compute_cycles + dispatch_cycles + pack_cycles + dram_cycles + setup_cycles
    return {
        "cycles": cycles,
        # Energy proxy: bandwidth-dominated (DRAM bytes moved). Full energy model is M2.
        "energy": float(dram_bytes),
        "breakdown": {
            "compute_cycles": compute_cycles,
            "dispatch_cycles": dispatch_cycles,
            "pack_cycles": pack_cycles,
            "dram_cycles": dram_cycles,
            "setup_cycles": setup_cycles,
            "dram_bytes": dram_bytes,
        },
    }
