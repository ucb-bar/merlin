"""The matrix-unit leg of ``merlin-compile`` (``--matrix-unit/--matrix-config/--matrix-support-target``).

Builds the shipped matrix image and its stand-in twin through
:func:`merlin.runtime.backends.matrix_image.build_matrix_pair`, gates the twin on spike, and -- when the support
provider declares a simulator that implements its unit -- also runs the shipped image. Split out of
:mod:`merlin.compile_cli`, which re-exports these names.
"""

from __future__ import annotations

from pathlib import Path


def _compile_matrix(out: dict, bundle, work, pkg, feats, refs, *, board, harts, iters, warmup, timeout, matrix) -> dict:
    """The matrix-unit leg of ``compile_rvv``: shipped image + stand-in twin, the twin gated on spike."""
    from merlin.llvmlower.impr_features import OPU_MATMUL_NAME
    from merlin.runtime.backends import zephyr_model as zm
    from merlin.runtime.backends.matrix_image import ROUTING_REPORT, build_matrix_pair
    from merlin.runtime.boards import board as _board_desc

    unit, config, support = matrix
    brd = _board_desc(board)
    routing = zm.MatrixRouting(unit=unit, config=config, support_target=support)
    cpus = zm.image_cpus(brd, harts)

    def _pair(fs):
        return build_matrix_pair(
            bundle,
            work,
            board=board,
            harts=harts,
            cpus=cpus,
            routing=routing,
            features=frozenset([*fs, OPU_MATMUL_NAME]),
            rvv_schedule=pkg.schedule_text,
            cflags=pkg.cflags + [flag for flag in zm._CFLAGS_COMMON if flag not in pkg.cflags],
            vlen=brd.vlen,
            iters=iters,
            warmup=warmup,
        )

    try:
        pair = _pair(feats)
    except Exception as exc:  # noqa: BLE001
        # Same fallback as the RVV route: a contraction left on the vector unit is split across harts
        # before the schedule runs, and the un-narrowed register block may not lower on the per-hart tile.
        if harts < 2 or "vector.mask" not in str(exc):
            raise
        from merlin import compile_cli  # the RVV route's feature derivation, shared

        narrowed = compile_cli._workload_features(pkg, bundle, out, harts=harts)
        out["harts_split_block_retry"] = {"tried": list(feats), "used": list(narrowed)}
        pair = _pair(narrowed)
    out.update(
        {
            "status": "compiled",
            "binary": str(pair.ship["elf"]),
            "build_hash": pair.ship.get("build_hash"),
            "twin_binary": str(pair.twin["elf"]),
            "matrix": {
                "unit": unit,
                "config": config,
                "support_target": support,
                "lowering_digest": pair.lowering_digest,
                "unit_instruction_counts": pair.unit_instruction_counts,
                "routing_report": str(pair.ship_dir / ROUTING_REPORT),
                "routed_contractions": pair.report.get("routed_contractions"),
                "opu_mac_share": pair.report.get("opu_mac_share"),
                "not_routed_by_reason": pair.report.get("not_routed_by_reason"),
                # What a caller needs to run either image itself on the same simulated machine -- e.g. the
                # shipped one on spike with the unit's own functional model (examples/lib/chip_flow.py).
                "spike_harts": cpus,
                "ship_ram_bytes": pair.ship.get("ram_bytes"),
                "twin_ram_bytes": pair.twin.get("ram_bytes"),
            },
        }
    )
    import time

    started = time.monotonic()
    res = zm.run_on_spike(
        pair.twin["elf"],
        dram_base=brd.dram_base,
        harts=cpus,
        mem_bytes=pair.twin["ram_bytes"],
        timeout=timeout,
        vlen=brd.vlen,
    )
    twin_wall = time.monotonic() - started
    out["status"] = "ran"
    # NOT the shipped image's cycles: the twin replaces the unit with a scalar loop.
    out["matrix"]["twin_spike_cycles"] = res.get("metrics", {}).get("cycles")
    out["matrix"]["gated_via"] = (
        "scalar stand-in twin on spike: same lowering (lowering_digest) as the shipped image, so the routing, "
        "pack, descriptor ABI and epilogue are checked; the unit's datapath is certified on its RTL"
    )
    if refs:
        # The whole output, as the RVV route grades it -- not the console prefix.
        g = zm._gate(res["output"] if res.get("output") is not None else res["prefix"], refs)
        out["verify"] = {"gate_ok": bool(g.get("ok")), "of": "stand-in twin", **g}
        out["status"] = "verified" if g.get("ok") else "run_mismatch"
    # A trap in the shipped image does not end the simulation (the RTOS's fatal handler spins), so the leg
    # is bounded by the twin's own run: same lowering, same work, a scalar body that is never faster.
    shipped_timeout = int(min(timeout, max(_SHIPPED_TIMEOUT_FLOOR_S, _SHIPPED_TIMEOUT_FACTOR * twin_wall)))
    shipped = _run_shipped_on_device_simulator(pair, routing, brd, cpus=cpus, timeout=shipped_timeout, refs=refs)
    out["matrix"]["shipped_on_simulator"] = shipped
    if shipped.get("status") == "run_mismatch" and out["status"] == "verified":
        # The twin agreeing while the unit's own datapath disagrees is exactly what this leg exists for.
        out["status"] = "run_mismatch"
    return out


#: Wall-clock bound of the shipped image's simulator run, relative to the twin's (see _compile_matrix).
_SHIPPED_TIMEOUT_FACTOR = 10
_SHIPPED_TIMEOUT_FLOOR_S = 300


def _run_shipped_on_device_simulator(pair, routing, brd, *, cpus: int, timeout: int, refs) -> dict:
    """Run the SHIPPED image on a functional simulator that implements the unit, when its provider declares one.

    The provider says, through an optional ``device_simulator(unit=, config=)`` on its matrix_lowering
    plugin, which simulator binary implements the unit and which ISA extensions name it. Absent a
    declaration the leg is ``not_declared``; declared but not present it is ``not_run`` with the reason --
    never a fallback to a simulator without the unit, which would trap or, worse, decode the words as
    something else.
    """
    from merlin.runtime.backends import zephyr_model as zm

    provider = routing.provider()
    declare = getattr(provider, "device_simulator", None)
    if not callable(declare):
        why = f"{routing.support_target} declares no simulator for {routing.unit}"
        return {"status": "not_declared", "reason": why}
    try:
        sim = dict(declare(unit=routing.unit, config=routing.config) or {})
    except Exception as exc:  # noqa: BLE001 -- the leg reports why it did not run
        return {"status": "not_run", "reason": f"{type(exc).__name__}: {exc}"}
    if not sim.get("available"):
        return {"status": "not_run", "reason": str(sim.get("reason") or "simulator unavailable"), **_sim_meta(sim)}
    isa = zm.spike_isa(brd.vlen) + "".join(f"_{ext}" for ext in sim.get("isa_extensions") or ())
    try:
        res = zm.run_on_spike(
            pair.ship["elf"],
            dram_base=brd.dram_base,
            harts=cpus,
            mem_bytes=pair.ship["ram_bytes"],
            timeout=timeout,
            isa=isa,
            spike=Path(sim["binary"]),
        )
    except Exception as exc:  # noqa: BLE001 -- a trap prints no DONE; a hang hits the bound
        return {
            "status": "error",
            "reason": f"{type(exc).__name__}: {str(exc)[:1500]}",
            "isa": isa,
            "timeout_s": timeout,
            **_sim_meta(sim),
        }
    got: dict = {
        "status": "ran",
        "elf": str(pair.ship["elf"]),
        "isa": isa,
        "spike_cycles": res.get("metrics", {}).get("cycles"),
        **_sim_meta(sim),
    }
    if refs:
        g = zm._gate(res["output"] if res.get("output") is not None else res["prefix"], refs)
        got["verify"] = {"gate_ok": bool(g.get("ok")), "of": "shipped image", **g}
        got["status"] = "verified" if g.get("ok") else "run_mismatch"
    return got


def _sim_meta(sim: dict) -> dict:
    return {k: sim[k] for k in ("binary", "binary_sha256", "provenance") if k in sim}
