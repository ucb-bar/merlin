"""Region Pressure Vector (RPV): architecture-independent workload pressure.

The RPV is the central M1 artifact: a set of *semantic* facts about a workload region,
organised into the pressure classes drawn from the DeepSeek-V3 hardware co-design checklist
(state, precision, layout, dispatch, overlap, bandwidth, compute). It names no architecture —
the cost model and DSE consume it to decide which software-visible contract is worth exposing.

``compute_rpv`` runs the requested cutpoints, merges their metrics, and additionally exposes a
flat ``facts`` sub-dict whose keys match the ``when`` clauses of the mined policies in
``output/kernels/policy_rules.yaml`` (the bridge to ``merlin.kernels.policy.evaluate_when``).
"""
from __future__ import annotations

from merlin.design_pressure.cutpoints.bufferized import cut_bufferized
from merlin.design_pressure.cutpoints.dispatch import cut_dispatch
from merlin.design_pressure.cutpoints.graph import cut_graph
from merlin.design_pressure.cutpoints.linalg import cut_linalg
from merlin.design_pressure.cutpoints.loop import cut_loop
from merlin.design_pressure.cutpoints.trace import cut_trace

_CUTPOINTS = {
    "graph": cut_graph,
    "linalg": cut_linalg,
    "loop": cut_loop,
    "bufferized": cut_bufferized,
    "dispatch": cut_dispatch,
    "trace": cut_trace,
}

DEFAULT_CUTPOINTS = ["linalg", "loop", "dispatch"]


def compute_rpv(region: dict, cutpoints: list[str] | None = None) -> dict:
    """Compute the Region Pressure Vector for ``region``.

    Returns a dict with:
      * ``metrics``  : the merged metrics across the requested cutpoints (flat).
      * ``classes``  : the same metrics grouped by pressure class.
      * ``facts``    : the flat facts the policy engine keys on.
      * ``cutpoints``: the cutpoints actually run.
    """
    cps = cutpoints or DEFAULT_CUTPOINTS
    metrics: dict = {}
    for name in cps:
        fn = _CUTPOINTS.get(name)
        if fn is None:
            continue
        metrics.update(fn(region))

    return {
        "cutpoints": list(cps),
        "metrics": metrics,
        "classes": _by_class(metrics),
        "facts": _facts(metrics),
    }


def _facts(m: dict) -> dict:
    """Flat facts dict whose keys match the mined policy ``when`` clauses.

    ``K`` is included so the *endorsement* check (K>=256) is evaluable; the synthesizer omits
    it for the *structural* legality check. See ``synthesize.py``.
    """
    return {
        "rhs_reuse_count": m.get("rhs_reuse_count", 1),
        "rhs_mutable": m.get("rhs_mutable", False),
        "K": m.get("K"),
        "op": m.get("op"),
        "has_epilogue": m.get("has_epilogue", False),
        "accumulator_live_across_epilogue": m.get("accumulator_live_across_epilogue", False),
        "dma_compute_overlap_beneficial": m.get("dma_compute_overlap_beneficial", False),
    }


# Which metric keys belong to which architecture-independent pressure class.
_CLASS_KEYS: dict[str, tuple[str, ...]] = {
    "state": ("rhs_reuse_count", "rhs_mutable", "reuse_distance", "distinct_weights",
              "state_bytes_per_step", "weight_immutable"),
    "precision": ("intermediate_i32_bytes", "intermediate_i32_bytes_step",
                  "final_output_bytes", "final_output_bytes_step", "has_epilogue"),
    "layout": ("pack_bytes", "pack_count_baseline", "pack_count_resident",
               "pack_bytes_baseline", "layout_conversions", "layout_convert_resident"),
    "dispatch": ("dispatch_count", "work_per_dispatch", "steps"),
    "overlap": ("dma_compute_overlap_beneficial", "sync_event_count"),
    "bandwidth": ("weight_bytes", "input_bytes_step", "dram_traffic_bytes_baseline",
                  "dram_traffic_bytes_resident"),
    "compute": ("op", "op_mix", "M", "K", "N", "macs", "dtype_dist"),
}


def _by_class(m: dict) -> dict:
    out: dict[str, dict] = {}
    for cls, keys in _CLASS_KEYS.items():
        out[cls] = {k: m[k] for k in keys if k in m}
    return out
