"""Cost-model parameter points (architecture-independent), plus an npu_model bridge.

``default_cost_model`` is the schema-valid default point. ``build_hardware_space`` enumerates the
hardware design space (Cartesian product over ``DEFAULT_HW_GRID``) as cost-model points — consumed by
``dse.report``. ``cost_model_from_npu`` derives a cost-model point from an npu_model ``HardwareConfig``
so the analytical model can be calibrated against the cycle-level simulator at a few points.
"""
from __future__ import annotations


def default_cost_model() -> dict:
    """A measurable default cost-model point (see ``dse_result.schema.yaml``)."""
    return {
        "dispatch_fixed_cycles": 200,
        "pack_startup_cycles": 64,
        "pack_bytes_per_cycle": 16,
        "dram_bytes_per_cycle": 8,
        "resident_store_bytes": 131072,
        "accumulator_entries": 16384,
        # Constant compute term (not a swept interface knob).
        "mac_per_cycle": 256,
        # Cost of exposing each interface's hardware (per region invocation). Non-zero so a
        # low-reuse region cannot amortise them — this is why I2/I3 do not always win.
        "resident_setup_cycles": 2000,
        "accumulator_setup_cycles": 1000,
    }


def cost_model_from_npu(hw) -> dict:
    """Derive the cost-model params from an npu_model ``HardwareConfig``.

    DRAM bandwidth comes from the off-chip link (``offchip_link_width_bits`` bits per beat,
    ``offchip_link_core_cycles_per_beat`` cycles per beat); packing bandwidth from the on-chip
    VMEM bus; dispatch/pack startup from the DMA command-word issue cost; compute throughput
    from the MXU tile (a 32x32x32 tile in ``mxu0_matmul_latency_cycles``).
    """
    link_bytes_per_beat = getattr(hw, "offchip_link_width_bits", 32) / 8
    beat_cycles = getattr(hw, "offchip_link_core_cycles_per_beat", 2)
    dram_bpc = max(link_bytes_per_beat / beat_cycles, 1e-6)

    cmd_words = getattr(hw, "dma_offchip_command_words", 2)
    mxu_lat = getattr(hw, "mxu0_matmul_latency_cycles", 32)
    # 32x32x32 macs per mxu_lat cycles.
    mac_per_cycle = max((32 * 32 * 32) // max(mxu_lat, 1), 1)

    return {
        "dispatch_fixed_cycles": int(cmd_words),
        "pack_startup_cycles": int(cmd_words),
        "pack_bytes_per_cycle": int(getattr(hw, "vmem_bytes_per_cycle", 64)),
        "dram_bytes_per_cycle": dram_bpc,
        "resident_store_bytes": int(getattr(hw, "vmem_size", 1 << 20)),
        "accumulator_entries": 32 * 32,
        "mac_per_cycle": int(mac_per_cycle),
        # make_resident+evict ~ a handful of weight pushes; commit ~ one MXU pop.
        "resident_setup_cycles": int(4 * mxu_lat),
        "accumulator_setup_cycles": int(mxu_lat),
    }


import itertools

# The hardware knobs the hardware-only DSE may sweep (the S_hw space). Values are the
# measurable cost-model params; each combination is one candidate hardware design point.
DEFAULT_HW_GRID: dict[str, list] = {
    "dispatch_fixed_cycles": [50, 200, 1000],
    "dram_bytes_per_cycle": [4, 8, 16, 32],
    "pack_bytes_per_cycle": [8, 16],
    "resident_store_bytes": [0, 65536, 131072, 262144],
}

# Area-proxy coefficients (arbitrary but fixed units). Exposing hardware costs area: a wider
# DRAM bus and a faster pack unit and cheaper dispatch all add area; a resident store and an
# accumulator/commit unit add area only when the chosen interface actually uses them.
_AREA = {
    "base": 1000.0,
    "per_dram_bpc": 120.0,
    "per_pack_bpc": 40.0,
    "dispatch_inv": 30000.0,   # cheaper dispatch (fewer fixed cycles) => more control area
    "per_resident_byte": 0.012,
    "accumulator_unit": 600.0,
}


def build_hardware_space(grid: dict | None = None, base: dict | None = None) -> list[dict]:
    """Enumerate the hardware design space as a list of cost-model dicts (Cartesian product)."""
    g = grid or DEFAULT_HW_GRID
    b = dict(base or default_cost_model())
    keys = list(g)
    points = []
    for combo in itertools.product(*(g[k] for k in keys)):
        cm = dict(b)
        cm.update({k: v for k, v in zip(keys, combo)})
        points.append(cm)
    return points


def area_proxy(cost_model: dict, interface_features=()) -> float:
    """Estimate the silicon area of a (hardware point, interface) pair.

    The resident store and accumulator/commit unit only count when the interface exposes them —
    so an interface that does not use residency does not pay for a resident store.
    """
    feats = set(interface_features or ())
    area = _AREA["base"]
    area += _AREA["per_dram_bpc"] * cost_model.get("dram_bytes_per_cycle", 0)
    area += _AREA["per_pack_bpc"] * cost_model.get("pack_bytes_per_cycle", 0)
    disp = cost_model.get("dispatch_fixed_cycles", 0) or 1
    area += _AREA["dispatch_inv"] / disp
    if "resident_packed_tensor" in feats:
        area += _AREA["per_resident_byte"] * cost_model.get("resident_store_bytes", 0)
    if "accumulator_commit" in feats:
        area += _AREA["accumulator_unit"]
    return round(area, 2)
