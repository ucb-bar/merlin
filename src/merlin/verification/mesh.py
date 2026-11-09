"""Independent tile and routed-mesh certification using explicitly selected callbacks."""

from __future__ import annotations

from pathlib import Path

from merlin.compiler.selection.capacity import (
    _accumulator_capacity_elems,
    _capacity_fit_tile,
    _operand_store_capacity_elems,
)
from merlin.runtime.mesh_backend import _resolve_oot_mesh_simulator

from .mesh_reference import _accum_rel_tolerance, _reference_on_datapath


def _certify_tile_via_executor(target, mlir, *, m, k, n, binding, timeout, numeric_policy=None, _executor) -> dict:
    """Certify one mesh tile on a target whose endpoint is NOT the OOT-cert path, by EXECUTING it through
    the shared endpoint-aware executor (``run_matmul_on_mesh``) with injected operands and gating the
    device's output against the mathematical reference.

    ``_mesh_verify`` used to call ``oot_runner.certify`` for every target regardless of endpoint kind, so a
    self-hosted-ISA target (``external_backend``) was graded on the RoCC/OOT path it has no backend for and
    every tile came back ``oracle_unavailable`` with "no registered backend" -- while the very same target's
    whole-model run executed fine through the endpoint-aware dispatcher. The oracle choice is now made in
    ONE place for both.

    Returns a ``certify``-shaped record so the caller's loop is unchanged. Honest about strength: this is a
    TWO-way gate (reference == oracle), not the OOT path's three-way (reference == simulate == oracle)."""
    run_matmul_on_mesh = _executor
    import numpy as np

    # The stimulus has to be exactly representable in the operand format -- that is what makes a bit-exact
    # gate possible at all -- but "exactly representable" does not have to mean "small integers", and while
    # it did, this gate was structurally incapable of finding the defect it was supposed to find. Small
    # integers are normal numbers in every format here, so no operand ever landed in the subnormal band,
    # so a datapath that flushes subnormals to zero certified bit-exact on all 15 synthesized tiles while
    # all 15 REAL layers of the same model diverged. The blind spot was in the seeds, not in the gate.
    #
    # A float datapath now draws from its own format's representable set instead -- a spread from the
    # smallest subnormal to near the cap, both signs, with distinct rows, distinct columns and A != A^T so
    # stride/transpose bugs stay visible (corpus_operands, the same synthesis the graded corpus uses). An
    # integer datapath keeps small integers: they ARE its range, and it has no subnormal band to miss.
    if binding.integer:
        rng = np.random.default_rng(0xA71A5)
        A = np.rint(rng.standard_normal((m, k)) * 3).clip(-8, 7).astype(np.float32)
        W = np.rint(rng.standard_normal((k, n)) * 3).clip(-8, 7).astype(np.float32)
        stimulus = "exactly-representable small integers"
    else:
        from ..targetgen import corpus_operands as CO

        A = np.asarray(CO.operand_values((m, k), binding.operand_dtype, salt=0xA7), dtype=np.float32).reshape(m, k)
        W = np.asarray(CO.operand_values((k, n), binding.operand_dtype, salt=0x5E), dtype=np.float32).reshape(k, n)
        stimulus = (
            f"{binding.operand_dtype} representable spread (subnormal..near-cap, both signs), "
            f"distinct rows/cols, asymmetric"
        )
    obs: dict = {}
    try:
        got = run_matmul_on_mesh(
            target,
            A.tolist(),
            W.tolist(),
            operand_dtype=binding.operand_dtype,
            accum_dtype=binding.accum_dtype,
            numeric_policy=numeric_policy,
            package=None,
            timeout=timeout,
            observed=obs,
        )
    except Exception as e:  # noqa: BLE001 — an executor failure is recorded, never a fake pass
        return {
            "status": "fail",
            "oracle": {"kind": obs.get("path"), "result": "error"},
            "failure": {"detail": f"{type(e).__name__}: {str(e)[-300:]}"},
        }
    if got is None:
        return {
            "status": "fail",
            "oracle": {"kind": obs.get("path"), "result": "skipped"},
            "failure": {"detail": f"no reachable mesh oracle for endpoint path {obs.get('path')!r}"},
        }
    ref = A @ W
    dev = np.asarray(got, dtype=np.float32)
    if dev.shape != ref.shape:
        return {
            "status": "fail",
            "oracle": {"kind": obs.get("oracle"), "result": "ran"},
            "failure": {"detail": f"device returned {dev.shape}, reference is {ref.shape}"},
        }
    # STRONGEST GATE FIRST: bit-exact against the target's own declared accumulator. Only if the device
    # reduces in a different order (or the format is unresolvable) do we fall back to a tolerance against
    # the f32 product, and the record says which gate carried the verdict.
    acc_ref = _reference_on_datapath(A, W, binding)
    if acc_ref is not None and np.array_equal(dev, acc_ref):
        return {
            "status": "pass",
            "oracle": {"kind": obs.get("oracle"), "result": "ran", "cycles": None},
            "gate": {
                "kind": f"bit-exact vs {binding.accum_dtype} accumulation",
                "rtol": 0.0,
                "exact": True,
                "f32_max_abs_err": float(np.abs(dev - ref).max()),
                "stimulus": stimulus,
                "does_not_cover": (
                    "operand-precision error on realistic value distributions"
                    if binding.integer
                    else "operand values not exactly representable in "
                    f"{binding.operand_dtype} (the format's own rounding)"
                ),
            },
        }
    rtol = _accum_rel_tolerance(binding.accum_dtype, k)
    if rtol is None:
        return {
            "status": "fail",
            "oracle": {"kind": obs.get("oracle"), "result": "ran"},
            "failure": {
                "detail": f"cannot derive a numeric gate for accumulator dtype "
                f"{binding.accum_dtype!r}: refusing to pick a tolerance"
            },
        }
    ok = (
        np.array_equal(dev, ref)
        if rtol == 0.0
        else bool(np.allclose(dev, ref, rtol=rtol, atol=rtol * float(np.abs(ref).max() or 1.0)))
    )
    rec = {
        "status": "pass" if ok else "fail",
        "oracle": {"kind": obs.get("oracle"), "result": "ran", "cycles": None},
        "gate": {
            "kind": "reference==oracle (f32, tolerance)",
            "rtol": rtol,
            "exact": bool(np.array_equal(dev, ref)),
            "stimulus": stimulus,
            "accum_exact": False if acc_ref is not None else None,
        },
    }
    if not ok:
        rec["failure"] = {
            "detail": f"tile {m}x{k}x{n} diverged from the reference: "
            f"max abs err {float(np.abs(dev - ref).max())} (rtol {rtol})"
        }
    return rec


def _mesh_verify(
    plan: dict,
    *,
    target: str,
    package: str | None,
    timeout: int,
    numeric_policy=None,
    _package_selector,
    _tile_binding,
    _tile_builder,
    _tile_certifier,
) -> dict:
    """Execute each mesh-routed matmul as a single systolic tile on the target's REAL mesh oracle.

    For every op the router placed on the accelerator mesh (``plan["mesh"]``), synthesize its
    ``merlin_iface`` tile capsule (``corpus_spec.build_matmul`` — one ``DxD`` tile at the target's derived
    mesh dimension, in the op's routed operand/accumulate dtype) and run it through the EXISTING
    ``oot_runner.certify`` accelerator path (the same one ``compile_oot`` uses). ``certify`` gates the
    emitted kernel's output bit-exact (integer) / within-tolerance (float) against the command buffer's
    mathematical reference == simulate == the RTL/mesh oracle — so a PASS is proof the matmul executed
    correctly ON the mesh, not merely that a routing plan was produced.

    FAIL-CLOSED: a tile whose oracle is unavailable is recorded ``oracle_unavailable`` (never a silent pass);
    an op with no single-tile synthesizer is recorded honestly and never counted as executed."""
    _default_oot_package = _package_selector
    _mesh_tile_binding = _tile_binding
    tile_builder_op = _tile_builder
    _certify_tile_via_executor = _tile_certifier
    import tempfile

    from ..common.paths import runs_root
    from ..targetgen import corpus_spec as CS
    from ..targetgen import package_runtime as oot_runner
    from ..targetgen.oracle_policy import _SIM_ORACLES
    from ..targetgen.oracle_policy import selected_endpoint as _endpoint_of
    from ..targetgen.oracle_policy import selected_sim_via as _bespoke_sim_via

    pkg_dir = package or _default_oot_package(target)
    out: dict = {
        "n_tiles": 0,
        "n_passed": 0,
        "n_failed": 0,
        "n_unavailable": 0,
        "n_unsynthesizable": 0,
        "package": pkg_dir,
        "per_tile": [],
        # CHEAPEST-FIRST, for a model exactly as for an operator capsule. Every tile runs the
        # target's declared screen tier BEFORE its cert tier, and a tile that fails the screen
        # never reaches the expensive oracle. Without this leg a model capsule declared
        # `required_oracle_tiers: [L0, L1, L2, L3]` and was graded on L3 alone -- it reached the
        # cert tier without earning the tier below it, which is the one ordering every operator
        # capsule obeys.
        "n_screened": 0,
        "n_screen_passed": 0,
        "n_screen_failed": 0,
        "n_screen_unavailable": 0,
    }
    if pkg_dir is None:
        out["status"] = "not_run"
        out["reason"] = f"no default OOT backend package for target {target!r}; pass --mesh-package to name one"
        return out
    # WHICH oracle certifies a tile follows the target's DERIVED endpoint, the same decision
    # ``run_matmul_on_mesh`` makes -- not an assumption that every target is the RoCC/OOT one. A
    # self-hosted-ISA target graded on the OOT path reports "no registered backend" for every tile
    # while its whole-model run executes fine, which is the same fact stated as an unavailability.
    _so = _SIM_ORACLES.get(_bespoke_sim_via(target))
    _endpoint, _ = _endpoint_of(target)
    _via_oot = not (_so is not None and _so.exclusive) and _endpoint in (None, "inline_asm_insn", "upstream_target")
    out["certified_via"] = "oot_cert" if _via_oot else "endpoint_executor"
    # Resolve before even building the OOT package: a conflicting campaign pin is a configuration
    # refusal, not an expensive build followed by a late simulator error.
    sim = _resolve_oot_mesh_simulator(target) if _via_oot else None
    # The screen is named by the TARGET'S CONTRACT (tier_sim minus rtl_tiers), never by a "spike"
    # literal here -- a tier is a fidelity and which simulator answers at it is the contract's business.
    # A target that declares no cheap tier yields None and the screen is reported UNAVAILABLE rather
    # than skipped, because "nobody could tell" and "nothing to tell" are different facts.
    from ..targetgen.oracle_policy import selected_screen_tiers as _screen_tiers_of

    _screens = _screen_tiers_of(target) if _via_oot else ()
    screen_tier, screen_sim = _screens[-1] if _screens else (None, None)
    out["screen_tier"], out["screen_sim"] = screen_tier, screen_sim
    if _via_oot and screen_sim is None:
        out["n_screened"] = None  # absent must never read as 0
        out["screen_reason"] = (
            f"{target} declares no oracle tier below its RTL tiers, so its tiles "
            f"cannot be screened before the cert tier"
        )
    if _via_oot:
        # Build the OOT backend ONCE up front so a broken build is a single honest not_run, not a per-tile
        # storm. certify re-runs an incremental (no-op) build per tile — harmless.
        try:
            oot_runner.build_package(oot_runner.load_package(pkg_dir), timeout=timeout)
        except Exception as e:  # noqa: BLE001 — a build failure is honest not_run, never a fake mesh pass
            out["status"] = "not_run"
            out["reason"] = f"OOT backend build failed: {type(e).__name__}: {str(e)[-400:]}"
            return out

    rr = runs_root(target, "mesh_verify")
    for i, r in enumerate(plan.get("mesh", [])):
        d = r.demand
        op, via = tile_builder_op(d)
        rec: dict = {"op": d.op, "site": d.site}
        if via == "family":
            rec["tile_op_via"] = via  # the weaker provenance stays visible in the record
        if op is None:
            out["n_unsynthesizable"] += 1
            rec.update(status="no_tile_synthesizer", reason=f"no single-tile capsule synthesizer for mesh op {d.op!r}")
            out["per_tile"].append(rec)
            continue
        try:
            # THE OP'S OWN FORMAT, not the model's. A whole-model capture is routed under one declared
            # datapath, and a real capture is mixed: torchAO quantizes Linear weights and leaves Conv2d
            # alone, so `M2_microvit_gemmini` captures 12 contractions in i8 and its one convolution in
            # f32, and `SY_model_resnet50` captures 53 in f32 and its classifier contraction in i8.
            # Synthesizing the odd op out at the model's declared format certifies arithmetic that op
            # does not perform -- and a tile is only built for it at all once the extent reader can read
            # a `linalg.generic` contraction, which is why the format has to be derived first.
            # `tile_fmt` falls back to the declared format when the op's own is unreadable.
            binding = _mesh_tile_binding(target, d.tile_fmt, r.acc, numeric_policy=numeric_policy)
            D = binding.tile_dim

            # compile the matmul LAYER at its REAL extent when the router carried one (rounded up to the
            # mesh dim — the backend tiles it into DxD tiles); else a single DxD tile.
            def _rup(x):
                return D if not x else ((int(x) + D - 1) // D) * D

            M, K, N = (_rup(d.m), _rup(d.k), _rup(d.n)) if (d.m and d.k and d.n) else (D, D, D)
            # A whole layer's weight tile (K·N) + activation (M·K) may exceed the on-chip operand store;
            # shrink to the largest capacity-fit tile (derived from the target's scratchpad memory fact)
            # and record how many such tiles cover the layer. The certified tile is the layer's real
            # repeating unit — a fit tile that passes proves the layer runs on the mesh once tiled. When
            # the capacity fact is absent we keep the untiled extent (prior behavior).
            layer_extent = f"{M}x{K}x{N}"
            # RECORD THE PADDING. `layer_extent` above is the extent AFTER rounding each dim up to the
            # mesh edge, and it was the only extent recorded -- so a layer whose real M is a partial tile
            # was certified at the padded M and reported as though that were the layer. Measured: a model
            # whose every matmul is M=8 on a 16-row mesh had 15/15 tiles "pass" at M=16 while the model
            # execution path declined the real M=8 on all 15, and nothing in the tile record showed the
            # two were different shapes. Keep the declared extent beside the certified one; when they
            # differ, the tile is evidence about a PADDED shape, which is a weaker claim.
            declared_extent = f"{int(d.m)}x{int(d.k)}x{int(d.n)}" if (d.m and d.k and d.n) else None
            padded = bool(declared_extent and declared_extent != layer_extent)
            n_subtiles = 1
            sp_cap = _operand_store_capacity_elems(target, binding.operand_dtype)
            acc_cap = _accumulator_capacity_elems(target, binding.accum_dtype)
            if sp_cap and ((K * N + M * K) > sp_cap or (acc_cap and M * N > acc_cap)):
                M, K, N, n_subtiles = _capacity_fit_tile(M, K, N, D, sp_cap, acc_cap)
            entry = {
                "name": f"mesh_tile_{i}_{op}",
                "op": op,
                "kind": "op",
                "source_role": "mesh_tile_synthesized",
                "source_reference": f"whole-model op {d.op} [{d.site}]: layer {layer_extent} on the "
                f"mesh as {n_subtiles} capacity-fit {M}x{K}x{N} tile(s)",
                "M": M,
                "K": K,
                "N": N,
            }
            capsule, mlir = CS.build(entry, binding)
        except Exception as e:  # noqa: BLE001 — a synthesis failure is recorded, never a fake pass
            out["n_unsynthesizable"] += 1
            rec.update(status="synth_error", reason=f"{type(e).__name__}: {str(e)[-300:]}")
            out["per_tile"].append(rec)
            continue
        # ``sim`` is only the simulator the OOT path was ASKED for; the endpoint executor runs on the
        # target's own oracle and ignores it, so record it only where it is the truth and let the
        # executor stamp ``oracle_kind`` with what actually ran.
        rec.update(
            M=M,
            K=K,
            N=N,
            layer_extent=layer_extent,
            n_subtiles=n_subtiles,
            declared_layer_extent=declared_extent,
            padded_to_mesh_edge=padded,
            operand_dtype=binding.cap_dtype(binding.operand_dtype),
            output_dtype=binding.cap_dtype(binding.accum_dtype),
        )
        if padded:
            rec["evidence_note"] = (
                f"certified at {layer_extent}, but this layer is {declared_extent}; "
                f"a dim was rounded UP to the mesh edge, so this tile is evidence "
                f"that the PADDED shape runs, not the layer's own extent"
            )
        if _via_oot:
            rec["sim"] = sim
        if _via_oot:
            with tempfile.TemporaryDirectory(prefix="mesh_tile_") as td:
                iface = Path(td) / f"{entry['name']}.interface.mlir"
                iface.write_text(mlir, encoding="utf-8")
                # THE SCREEN RUNG, run first and gating the cert rung. Same tile, same package, same
                # three-way gate -- only the oracle is the cheap one. Measured on the gemmini corpus, a
                # tile's functional screen costs ~0.4 s against ~17 s for the cycle-accurate rung, so
                # screening every tile is far cheaper than one wasted cert.
                if screen_sim is not None:
                    try:
                        scr = oot_runner.certify(
                            pkg_dir,
                            iface,
                            runs_root=str(rr),
                            run_id=f"{entry['name']}_screen_{screen_tier}",
                            simulator=screen_sim,
                            target=target,
                            timeout=timeout,
                            require_accelerator_trace=True,
                        )
                    except Exception as e:  # noqa: BLE001 — a screen crash is unavailable, never a pass
                        scr = {
                            "status": "error",
                            "oracle": {"result": "skipped"},
                            "failure": {"detail": f"{type(e).__name__}: {str(e)[-300:]}"},
                        }
                    _so_ = scr.get("oracle") or {}
                    rec["screen"] = {
                        "tier": screen_tier,
                        "sim": screen_sim,
                        "oracle_kind": _so_.get("kind"),
                        "oracle_engine": _so_.get("engine"),
                        "oracle_result": _so_.get("result"),
                        "derived_from_rtl": _so_.get("derived_from_rtl"),
                        "cycle_accurate": _so_.get("cycle_accurate"),
                        "cycles": _so_.get("cycles"),
                        "trace_check": scr.get("trace_check"),
                        "artifact_identity": scr.get("artifact_identity"),
                    }
                    out["n_screened"] += 1
                    if _so_.get("result") == "skipped":
                        out["n_screen_unavailable"] += 1
                        rec["screen"]["status"] = "oracle_unavailable"
                        rec["screen"]["reason"] = (scr.get("failure") or {}).get(
                            "detail"
                        ) or f"{screen_sim} screen oracle unavailable"
                    elif scr.get("status") == "pass":
                        out["n_screen_passed"] += 1
                        rec["screen"]["status"] = "pass"
                    else:
                        out["n_screen_failed"] += 1
                        rec["screen"]["status"] = "fail"
                        rec["screen"]["reason"] = (scr.get("failure") or {}).get("detail")
                    # GATEKEEPING: a tile that did not clear the screen does not get the cert oracle.
                    # Spending the expensive rung on a tile the cheap rung already refused is how a
                    # cert tier comes to stand alone, and refusing here is what makes the screen a gate
                    # rather than a decoration. The tile is counted a FAILURE, not skipped: its layer
                    # is unproven either way, and an excluded tile would leave n_passed == n_tiles.
                    if rec["screen"]["status"] == "oracle_unavailable":
                        # An absent screen oracle proves nothing about the tile; count it UNAVAILABLE so an
                        # environment without the screen is never reported as a compiler failure.
                        out["n_tiles"] += 1
                        out["n_unavailable"] += 1
                        rec["status"] = "oracle_unavailable"
                        rec["reason"] = (
                            f"the {screen_tier} {screen_sim} screen oracle is unavailable: {rec['screen']['reason']}"
                        )
                        out["per_tile"].append(rec)
                        continue
                    if rec["screen"]["status"] != "pass":
                        out["n_tiles"] += 1
                        out["n_failed"] += 1
                        rec["status"] = "screen_not_cleared"
                        rec["reason"] = (
                            f"did not clear the {screen_tier} {screen_sim} screen "
                            f"({rec['screen']['status']}), so the cert tier was not run: "
                            f"{rec['screen'].get('reason')}"
                        )
                        out["per_tile"].append(rec)
                        continue
                res = oot_runner.certify(
                    pkg_dir,
                    iface,
                    runs_root=str(rr),
                    run_id=entry["name"],
                    simulator=sim,
                    target=target,
                    timeout=timeout,
                    require_accelerator_trace=True,
                )
        else:
            res = _certify_tile_via_executor(
                target, mlir, m=M, k=K, n=N, binding=binding, timeout=timeout, numeric_policy=numeric_policy
            )
            rec["gate"] = res.get("gate")
        oracle = res.get("oracle") or {}
        out["n_tiles"] += 1
        rec.update(
            oracle_kind=oracle.get("kind"),
            oracle_engine=oracle.get("engine"),
            oracle_result=oracle.get("result"),
            # Preserve the oracle's own fidelity assertion.  A functional Spike pass and a
            # cycle-accurate RTL pass are both useful, but they cannot carry the same formal
            # whole-model claim merely because this record later lands under a tier named L3.
            derived_from_rtl=oracle.get("derived_from_rtl"),
            cycle_accurate=oracle.get("cycle_accurate"),
            cycles=oracle.get("cycles"),
            trace_check=res.get("trace_check"),
            artifact_identity=res.get("artifact_identity"),
        )
        if oracle.get("result") == "skipped":
            out["n_unavailable"] += 1
            rec["status"] = "oracle_unavailable"
            rec["reason"] = (res.get("failure") or {}).get("detail") or "mesh oracle unavailable in this env"
        elif res.get("status") == "pass":
            out["n_passed"] += 1
            rec["status"] = "pass"
        else:
            out["n_failed"] += 1
            rec["status"] = "fail"
            rec["reason"] = (res.get("failure") or {}).get("detail")
        out["per_tile"].append(rec)

    if out["n_tiles"] == 0:
        out["status"] = "no_mesh_matmuls" if out["n_unsynthesizable"] == 0 else "no_synthesizable_mesh_ops"
    elif out["n_unavailable"] == out["n_tiles"]:
        out["status"] = "oracle_unavailable"
    elif out["n_passed"] == out["n_tiles"]:
        out["status"] = "verified"
    else:
        out["status"] = "partial"
    _gate_note = (
        "the emitted kernel's output is gated (bit-exact int / tolerance float) against the "
        "command buffer's reference == simulate == oracle three-way"
        if _via_oot
        else "the device's output is gated against the mathematical reference with the operands "
        "INJECTED (reference == oracle, a two-way gate: this endpoint has no separate simulate "
        "leg), at a tolerance derived from the accumulator format's mantissa width"
    )
    out["note"] = (
        "each mesh-routed matmul LAYER is verified by EXECUTING its capacity-fit tile on the target mesh "
        f"oracle ({'simulator=' + sim if _via_oot else 'via the endpoint-derived executor'}): the tile is "
        "the largest D-aligned unit whose weight+activation working "
        "set fits the target's DERIVED scratchpad capacity (n_subtiles = how many tile it into the layer); "
        f"{_gate_note}. A fit tile that certifies proves the layer runs on the "
        "mesh once tiled. Each tile is run CHEAPEST-FIRST: it must clear the target's declared screen "
        "tier (" + (f"{screen_tier}/{screen_sim}" if screen_sim else "none declared") + ") before the "
        "cert oracle is spent on it, so a model earns the tier below its cert tier rather than being "
        "graded on the cert tier alone. The remaining gap is the single SPLICED image (all layers' mesh kernels + the "
        "scalar/RVV remainder co-scheduled in one binary with activations handed between layers) — see "
        "compile_model's docstring."
    )
    return out
