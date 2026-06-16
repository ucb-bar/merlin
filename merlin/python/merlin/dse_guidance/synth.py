"""Synthesize an *analytical* baseline cost and temporal metadata from a workload region.

The exhaustive study needs a baseline cost and temporal view for every supported workload, but
most ``semantic_memory`` regions ship without a measured cost breakdown. Rather than invent
numbers, we derive a baseline from the existing analytical cost model
(``merlin.dse.hardware_space.default_cost_model`` + ``merlin.design_pressure`` RPV metrics) and
tag every component ``analytical``. The evidence machinery then weights these below any measured
input — so a synthesized baseline is honestly marked as the weakest grounding, and a measured
fixture (when present) overrides it.

Component mapping (cost-model cycles -> named components), all tagged ``analytical``::

    compute                       = ceil(macs / mac_per_cycle)
    dma_memory                    = (steps*weight_bytes + io_bytes) / dram_bytes_per_cycle
    packing                       = pack_count_baseline * (startup + pack_bytes/pack_bytes_per_cycle)
    cpu_dispatch                  = dispatch_count * dispatch_fixed_cycles
    intermediate_materialization  = steps*intermediate_i32_bytes_step / dram_bytes_per_cycle

Units are the cost model's cycles (``unit: cycles``), not ms — gap_closure is a ratio, so the
unit does not affect the ranking, and we never relabel cycles as milliseconds.
"""
from __future__ import annotations

import math

from merlin.design_pressure import region as R
from merlin.design_pressure.pressure_vector import compute_rpv
from merlin.dse.hardware_space import default_cost_model


def analytical_baseline_cost(region: dict, target_fraction: float | None = 0.5) -> dict:
    """Build a ``baseline_cost`` doc (cycles, all components ``analytical``) from a region.

    ``target_fraction`` sets a synthetic target at that fraction of the baseline total (so a
    gap exists to rank against); pass ``None`` for no target.
    """
    cm = default_cost_model()
    rpv = compute_rpv(region)
    m = rpv["metrics"]
    steps = max(int(m.get("steps", 1)), 1)

    macs = int(m.get("macs", 0))
    compute = math.ceil(macs / cm.get("mac_per_cycle", 256)) if macs else 0

    weight_bytes = int(m.get("weight_bytes", 0))
    io_bytes = steps * (int(m.get("input_bytes_step", 0)) + int(m.get("final_output_bytes_step", 0)))
    dma_memory = (steps * weight_bytes + io_bytes) / cm["dram_bytes_per_cycle"]

    pack_bytes = int(m.get("pack_bytes", 0))
    pack_count = int(m.get("pack_count_baseline", steps))
    packing = pack_count * (cm["pack_startup_cycles"]
                            + (pack_bytes / cm["pack_bytes_per_cycle"] if pack_bytes else 0))

    cpu_dispatch = int(m.get("dispatch_count", steps)) * cm["dispatch_fixed_cycles"]

    intermediate = steps * int(m.get("intermediate_i32_bytes_step", 0)) / cm["dram_bytes_per_cycle"]

    components = {
        "compute": float(compute),
        "dma_memory": float(dma_memory),
        "packing": float(packing),
        "cpu_dispatch": float(cpu_dispatch),
        "intermediate_materialization": float(intermediate),
        "sync": 0.0,
    }
    total = sum(components.values())
    doc: dict = {
        "workload": region.get("name", "workload"),
        "baseline": {
            "unit": "cycles",
            "total_ms": total,
            "components": {f"{k}_ms": v for k, v in components.items()},
        },
        "metadata_source": {f"{k}_ms": "analytical" for k in components},
    }
    if target_fraction is not None and total > 0:
        doc["target"] = {"total_ms": total * float(target_fraction)}
    return doc


def synth_temporal(region: dict, control_rate_hz: float = 30.0) -> dict:
    """Derive a ``temporal_workload_metadata`` doc from a region's reuse structure.

    Treats the region's weight reuse count as the K-step loop trip count; H defaults to K.
    Loop-invariant state lists the weight when it is immutable and reused (so residency is
    legal under the multi-rate view); the negative control (reuse=1) yields K=1 and no loop.
    """
    reuse = R.rhs_reuse_count(region)
    immutable = not R.rhs_mutable(region)
    K = max(int(reuse), 1)
    H = K

    loop_invariant: list[str] = []
    if immutable and K > 1:
        roles = R.classify_tensors(region)
        loop_invariant.append(roles.get("rhs") or "weights")

    return {
        "workload": region.get("name", "workload"),
        "class": "synthesized_from_region",
        "timing": {"K": K, "H": H, "control_rate_hz": control_rate_hz},
        "regions": [
            {
                "name": "region_step",
                "cadence": "K_times_per_replan" if K > 1 else "once_per_replan",
                "loop_trip_count": K,
                "loop_invariant_state": loop_invariant,
                "loop_carried_state": [],
            }
        ],
    }
