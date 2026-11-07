"""The matrix-unit leg of ``merlin-compile`` (``--matrix-unit/--matrix-config/--matrix-support-target``).

Builds the shipped matrix image and its stand-in twin through
:func:`merlin.runtime.backends.matrix_image.build_matrix_pair`, gates the twin on spike, and -- when the support
target declares a simulator that implements its unit -- also runs the shipped image. Split out of
:mod:`merlin.compile_cli`, which re-exports these names. ``build_tools/scripts/make_delivery.py`` runs the same
shipped-image leg, so a whole-model delivery is gated exactly like one ``merlin-compile`` image.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


def _compile_matrix(
    out: dict,
    bundle,
    work,
    pkg,
    feats,
    refs,
    *,
    board,
    harts,
    iters,
    warmup,
    timeout,
    matrix,
    matrix_gate="twin",
    int8_compute=True,
    op_profile=False,
) -> dict:
    """The matrix-unit leg of ``compile_rvv``: shipped image + stand-in twin, the twin gated on spike."""
    from merlin.kernels import matrix_kernels
    from merlin.runtime.backends import output_identity
    from merlin.runtime.backends import zephyr_model as zm
    from merlin.runtime.backends.matrix_image import ROUTING_REPORT, NothingRouted, build_matrix_pair
    from merlin.runtime.boards import board as _board_desc

    unit, config, support, *rest = matrix
    kernel_spec = rest[0] if rest else None
    brd = _board_desc(board)
    routing = zm.MatrixRouting(unit=unit, config=config, support_target=support, kernel=kernel_spec)
    route_feature = routing.route_feature_name()
    kernel = matrix_kernels.for_routing(routing)  # a bad name or a missing file fails here, before any build
    cpus = zm.image_cpus(brd, harts)

    def _pair(fs):
        return build_matrix_pair(
            bundle,
            work,
            board=board,
            harts=harts,
            cpus=cpus,
            routing=routing,
            features=frozenset([*fs, *([route_feature] if route_feature is not None else [])]),
            rvv_schedule=pkg.schedule_text,
            cflags=pkg.cflags + [flag for flag in zm._CFLAGS_COMMON if flag not in pkg.cflags],
            vlen=brd.vlen,
            iters=iters,
            warmup=warmup,
            int8_compute=int8_compute,
            profile_ship=op_profile,
        )

    try:
        pair = _pair(feats)
    except NothingRouted as none:
        # Not an error to hide behind "the image carries none of the unit's instructions": the model has
        # nothing the unit takes, and the report says which contractions were declined and why.
        out.update(status="not_routed", reason=str(none))
        out["matrix"] = {
            "unit": unit,
            "config": config,
            "support_target": support,
            "routing_report": str(none.report_path),
            "routed_contractions": 0,
            "opu_mac_share": none.report.get("opu_mac_share"),
            "not_routed_by_reason": none.report.get("not_routed_by_reason"),
        }
        return out
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
                # The kernel every routed call ran, with its file's sha256 (merlin.kernels.matrix_kernels).
                "kernel": kernel.record(),
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
                # The shipped image's memory budget (zephyr_memory): buffers summed, peak, arena, region.
                "ship_memory": pair.ship.get("memory"),
                # WHICH revision of the unit's sources the encodings and geometry were derived from (pins
                # verified, drift listed): a verdict about this image is a verdict about that revision.
                "provenance": (pair.ship.get("matrix_build") or {}).get("provenance"),
            },
        }
    )
    if op_profile:
        table_path = Path(pair.ship.get("op_profile_table") or "")
        if not table_path.is_file():
            raise RuntimeError("profiled shipped image has no operation table")
        out["matrix"]["op_profile_table"] = str(table_path)
        out["matrix"]["op_profile_table_sha256"] = hashlib.sha256(table_path.read_bytes()).hexdigest()
    if matrix_gate == "build":
        out["matrix"].update(
            twin_executed=False,
            shipped_on_simulator={"status": "not_run", "reason": "compile-only matrix gate"},
            gated_via="build and instruction audit only; numerical gates pending",
        )
        out["verification_scope"] = "none; twin and shipped images were not executed"
        return out
    import time

    if matrix_gate == "shipped":
        shipped = _run_shipped_on_device_simulator(pair, routing, brd, cpus=cpus, timeout=timeout, refs=refs, twin=None)
        out["matrix"].update(
            shipped_on_simulator=shipped,
            twin_executed=False,
            independent_control_required=True,
            gated_via="shipped image on the declared unit simulator; independent control comparison required",
        )
        if op_profile:
            _require_profiled_spike(out["matrix"], shipped)
        out["verification_scope"] = "shipped_w8a8_only; independent control identity pending"
        out["status"] = shipped.get("status", "error")
        out["verify"] = shipped.get("verify", {})
        out["output_identity"] = shipped.get("output_identity")
        return out
    if matrix_gate != "twin":
        raise ValueError(f"unknown matrix gate: {matrix_gate}")
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
    # The twin's answer, kept so the shipped image can be compared with it bit for bit without re-running
    # the twin (a whole-model scalar stand-in is the most expensive simulation in the flow).
    out["matrix"]["twin_output_identity"] = output_identity.output_identity(res)
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
    shipped = _run_shipped_on_device_simulator(
        pair, routing, brd, cpus=cpus, timeout=shipped_timeout, refs=refs, twin=res
    )
    out["matrix"]["shipped_on_simulator"] = shipped
    if op_profile:
        _require_profiled_spike(out["matrix"], shipped)
    if shipped.get("status") in SHIPPED_FAILURES and out["status"] == "verified":
        # The twin agreeing while the unit's own datapath disagrees is exactly what this leg exists for.
        out["status"] = shipped["status"]
    return out


def _require_profiled_spike(matrix: dict, shipped: dict) -> None:
    """Require one exact, fully marked Spike inference before planning a profiled ELF."""
    table_bytes = Path(matrix["op_profile_table"]).read_bytes()
    if hashlib.sha256(table_bytes).hexdigest() != matrix["op_profile_table_sha256"]:
        raise RuntimeError("profile table changed after the shipped ELF was built")
    table = json.loads(table_bytes)
    if not isinstance(table, list) or {row["id"] for row in table} != set(range(len(table))):
        raise RuntimeError("profiled shipped image has incomplete operation IDs")
    calls = [
        row["id"]
        for row in table
        if row.get("mlir_op") == "func.call" and str(row.get("callee") or "").startswith("@merlin_opu_")
    ]
    # The sidecar is produced by the same lowering as the table; compare its symbols, not just counts.
    signatures_path = Path(matrix["op_profile_table"]).with_name("opu_signatures.json")
    signatures = json.loads(signatures_path.read_text())
    routed = signatures.get("routed") or []
    if not calls or Counter(table[i]["callee"].removeprefix("@") for i in calls) != Counter(
        row["symbol"] for row in routed
    ):
        raise RuntimeError("profiled OPU call sites differ from routed signatures")
    profile = shipped.get("profile") or {}
    if (
        shipped.get("status") != "verified"
        or profile.get("iters") != 1
        or profile.get("prof_marks") != len(table) + 1
        or profile.get("routed_one_hit") is not True
        or profile.get("rows_valid") is not True
        or profile.get("cycles_reconciled") is not True
        or profile.get("routed_calls") != len(calls)
    ):
        raise RuntimeError("shipped profiled ELF lacks one complete verified Spike inference")


def _profile_spike_result(pair, res: dict) -> dict | None:
    """Summarize the shipped ELF's own Spike counters without embedding a large UART in JSON."""
    table_path = pair.ship.get("op_profile_table")
    if not table_path:
        return None
    table = json.loads(Path(table_path).read_text())
    calls = {
        row["id"]
        for row in table
        if row.get("mlir_op") == "func.call" and str(row.get("callee") or "").startswith("@merlin_opu_")
    }
    rows = {}
    for line in (res.get("console") or "").splitlines():
        if not line.startswith("PROF "):
            continue
        fields = line.split()
        if len(fields) != 4:
            raise RuntimeError("malformed Spike PROF row")
        op_id, ticks, hits = map(int, fields[1:])
        if op_id in rows or op_id < 0 or ticks < 0 or hits < 0:
            raise RuntimeError("duplicate or invalid Spike PROF row")
        rows[op_id] = (ticks, hits)
    metrics = res.get("metrics") or {}
    attributed = metrics.get("prof_total_ticks")
    cycles = metrics.get("cycles")
    printed_ticks = sum(ticks for ticks, _ in rows.values())
    return {
        "iters": metrics.get("iters"),
        "prof_marks": metrics.get("prof_marks"),
        "prof_dump_selected": metrics.get("prof_dump_selected"),
        "routed_calls": len(calls),
        "routed_one_hit": bool(calls) and calls.issubset(rows) and all(rows[i][1] == 1 for i in calls),
        "rows_valid": set(rows).issubset(range(len(table))) and metrics.get("prof_dump_selected") == len(rows),
        "cycles_reconciled": isinstance(attributed, int)
        and isinstance(cycles, int)
        and 0 < sum(rows[i][0] for i in calls if i in rows) <= printed_ticks <= attributed <= cycles,
        "console_sha256": hashlib.sha256((res.get("console") or "").encode()).hexdigest(),
    }


#: Wall-clock bound of the shipped image's simulator run, relative to the twin's (see _compile_matrix).
_SHIPPED_TIMEOUT_FACTOR = 10
_SHIPPED_TIMEOUT_FLOOR_S = 300


#: Shipped-image outcomes that fail the image's gate: it ran on a model of the unit and computed something
#: else, or never produced its output. ``not_declared``/``not_run`` are reported and do not fail it.
SHIPPED_FAILURES = ("run_mismatch", "run_failed")


def _run_shipped_on_device_simulator(pair, routing, brd, *, cpus: int, timeout: int, refs, twin=None) -> dict:
    """Run the SHIPPED image on a functional simulator that implements the unit, when its target declares one.

    A target declares it in one of two ways. Its matrix_lowering plugin's optional
    ``device_simulator(unit=, config=)`` names a simulator binary and the ISA extensions that name the unit
    (Iris: riscv-isa-sim with Zvt). Or its contract's ``runner.spike_extension`` names a spike ``--extlib``
    model of the unit, verified by :mod:`merlin.targetgen.spike_extension` (Kodiak's OPU). Absent both, the
    leg is ``not_declared``; declared but unusable it is ``not_run`` with the reason -- never a fallback to a
    simulator without the unit, which would trap or, worse, decode the words as something else.

    ``twin`` is the stand-in twin's spike result. With it, the shipped image must also match the twin bit for
    bit: they are one lowering with two tile bodies, so any difference is the unit's datapath (or its model).
    """
    from merlin.runtime.backends import output_identity
    from merlin.runtime.backends import zephyr_model as zm

    provider = routing.provider()
    declare = getattr(provider, "device_simulator", None)
    if not callable(declare):
        return _run_shipped_on_spike_extension(pair, routing, brd, cpus=cpus, timeout=timeout, refs=refs, twin=twin)
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
        "output_identity": output_identity.output_identity(res),
        **_sim_meta(sim),
    }
    if profile := _profile_spike_result(pair, res):
        got["profile"] = profile
    output = res.get("output") if res.get("output") is not None else res.get("prefix")
    if output is None or not len(output):
        return {**got, "status": "run_failed", "reason": (res.get("console") or "")[-1500:]}
    if not refs and twin is None:
        return got
    ok = True
    if refs:
        g = zm._gate(output, refs)
        got["verify"] = {"gate_ok": bool(g.get("ok")), "of": "shipped image", **g}
        ok = bool(g.get("ok"))
    if twin is not None:
        ok = _compare_with_twin(got, res, twin) and ok
    got["status"] = "verified" if ok else "run_mismatch"
    return got


def _compare_with_twin(got: dict, res: dict, twin: dict) -> bool:
    """Record whether the shipped image's output equals the twin's to the bit (prefix and whole-tensor hash).

    One lowering, two tile bodies: the generated kernel's scalar body, or for a hand-written kernel Merlin's
    reference loop (merlin.kernels.matrix_kernels). Any difference is the kernel or the unit's datapath.
    """
    import numpy as np

    a, b = res.get("prefix"), twin.get("prefix")
    same_prefix = a is not None and b is not None and len(a) == len(b) and bool(np.array_equal(a, b))
    hashes = (res.get("out_hash"), twin.get("out_hash"))
    same_hash = None if None in hashes else hashes[0] == hashes[1]
    got["bit_exact_vs_twin"] = same_prefix and same_hash is not False
    got["twin_compare"] = {
        "prefix_equal": same_prefix,
        "out_hash_equal": same_hash,
        "elements": 0 if a is None else len(a),
    }
    if not got["bit_exact_vs_twin"] and a is not None and b is not None and len(a) == len(b):
        diff = np.flatnonzero(np.asarray(a) != np.asarray(b))
        if diff.size:
            i = int(diff[0])
            got["twin_compare"]["first_difference"] = {"index": i, "shipped": a[i].item(), "twin": b[i].item()}
            got["twin_compare"]["differing_prefix_elements"] = int(diff.size)
    return bool(got["bit_exact_vs_twin"])


def _sim_meta(sim: dict) -> dict:
    return {k: sim[k] for k in ("binary", "binary_sha256", "provenance") if k in sim}


def _run_shipped_on_spike_extension(pair, routing, brd, *, cpus: int, timeout: int, refs, twin) -> dict:
    """The shipped image on spike with the unit's own functional model, as the target's contract declares it
    (``runner.spike_extension``: extension name, the variable naming the ``.so``, its digest or build inputs).

    The model is resolved by :func:`merlin.targetgen.spike_extension.resolve`, which returns the declared
    bytes or raises; a raise is ``not_run`` with its reason. The model computes what the unit computes and
    says nothing about its timing, so the cycles recorded are spike's, not the unit's.
    """
    from merlin.runtime.backends import spike as spike_backend
    from merlin.runtime.backends import zephyr_model as zm
    from merlin.targetgen import spike_extension as SE

    target = routing.support_target
    try:
        if SE.declared_extension(target) is None:
            return {
                "status": "not_declared",
                "reason": f"{target} declares no simulator for {routing.unit} "
                "(no device_simulator in its plugin, no runner.spike_extension in its contract)",
            }
        # Declared: the defaults are never used; resolve() returns the declared bytes or raises.
        ext = SE.resolve(target, default_library_dir=spike_backend.spike_path().parent, default_extension_name="")
    except SE.SpikeExtensionError as exc:
        return {"status": "not_run", "reason": str(exc)}
    meta = {
        "via": "spike_extension",
        "extension": {"name": ext.extension_name, "extlib": str(ext.extlib), "sha256": ext.sha256},
        "elf": str(pair.ship["elf"]),
    }
    try:
        res = zm.run_on_spike(
            pair.ship["elf"],
            dram_base=brd.dram_base,
            harts=cpus,
            mem_bytes=pair.ship["ram_bytes"],
            timeout=timeout,
            vlen=brd.vlen,
            extension=ext,
        )
    except Exception as exc:  # noqa: BLE001 -- a trap prints no DONE; a hang hits the bound
        return {
            **meta,
            "status": "run_failed",
            "reason": f"{type(exc).__name__}: {str(exc)[:1500]}",
            "timeout_s": timeout,
        }
    from merlin.runtime.backends import output_identity

    got: dict = {**meta, "status": "ran", "spike_cycles": res.get("metrics", {}).get("cycles")}
    got["output_identity"] = output_identity.output_identity(res)
    if profile := _profile_spike_result(pair, res):
        got["profile"] = profile
    output = res.get("output") if res.get("output") is not None else res.get("prefix")
    if output is None or not len(output):
        return {**got, "status": "run_failed", "reason": (res.get("console") or "")[-1500:]}
    ok = True
    if refs:
        g = zm._gate(output, refs)
        got["verify"] = {"gate_ok": bool(g.get("ok")), "of": "shipped image on the unit's spike model", **g}
        ok = bool(g.get("ok"))
    if twin is not None:
        # Same lowering, two tile bodies: on a correct unit the outputs agree to the bit, on the dumped
        # prefix and on the whole-tensor hash.
        ok = _compare_with_twin(got, res, twin) and ok
    got["status"] = "verified" if ok else "run_mismatch"
    return got
