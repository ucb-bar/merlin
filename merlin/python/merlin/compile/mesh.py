"""Execute and certify matmul layers on a target's accelerator mesh.

``run_matmul_on_mesh`` runs one layer with its real operands through whichever oracle the target's
endpoint reaches (OOT certification, program oracle, or a bespoke simulator), tiling it when it exceeds
the backend's capacity; ``_mesh_verify`` certifies a synthesized tile per mesh-routed matmul of a route
plan. Both resolve the default backend package (``_default_oot_package``) and the tile binding
(``_mesh_tile_binding``) here, so a test that stands in for either patches THIS module.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .capacity import (
    _accumulator_capacity_elems,
    _capacity_fit_tile,
    _operand_store_capacity_elems,
    capacity_fit,
    declared_primitive_tile,
)
from .mesh_backend import (
    _MESH_RUN_SEQ,
    _built_mesh_package,
    _mesh_invocation_id,
    _mesh_layer_id,
    _refuse,
    _resolve_oot_mesh_simulator,
)
from .mesh_reference import _accum_rel_tolerance, _reference_on_datapath

_MESH_NTILE_WIDTH: dict[tuple, int] = {}  # N-tile width a target's backend accepts


def _default_oot_package(target: str) -> str | None:
    """The conventional OOT backend package for ``target``, or None when the target ships no default (a
    bespoke-layout target must name it via ``--package``). Prefers the agent-submission package
    (``.../targets/<target>/agent_spec_v1_mlir_oot`` — the SAME default ``compile_oot`` resolves, so on-mesh
    tile verification reuses it verbatim) and falls back to a hand-curated REFERENCE backend package
    (``.../reference_v0``) when a target ships one instead of an agent submission (e.g. a SIMT core whose
    deterministic reference lowering is the package). Target-agnostic — both are directory conventions, not
    target literals."""
    from ..common.artifacts import artifacts_dir

    base = artifacts_dir() / "targets" / target
    # Selected by KIND, not merely by the presence of a manifest. ``out/artifacts/targets/<target>/`` also
    # holds CODEGEN packages (schedules/knobs/dialects, e.g. a hand-curated ``hand_v0``), whose manifest is
    # a different artifact entirely. Returning one of those as an OOT backend does not fail cleanly: it
    # gets as far as the package loader and dies on "manifest schema violation: 'artifact_type' is a
    # required property", which reads like a corrupt backend rather than a directory that was never one.
    for pkg_id in ("agent_spec_v1_mlir_oot", "reference_v0"):
        cand = base / pkg_id
        mf = cand / "manifest.yaml"
        if not mf.is_file():
            continue
        try:
            from ..common.yaml import load_yaml

            if not str((load_yaml(mf) or {}).get("artifact_type", "")).strip():
                continue  # not an OOT backend manifest — keep looking
        except Exception:  # noqa: BLE001 — unreadable manifest: let the loader report it
            pass
        return str(cand)
    return None


def _mesh_tile_binding(
    target: str, operand_dtype: str | None, accum_dtype: str | None, requant_output_dtype: str | None = None
):
    """A ``corpus_spec.CorpusBinding`` for synthesizing a single systolic mesh tile of ``target``, with the
    operand/accumulate datapath pinned to what the routed op actually needs (derived, never assumed). The
    tile dim, compare policy, and instruction-class deriver all come from the target's own descriptor +
    manifest via ``derive_binding`` — no target literal, no hand-set dims.

    ``requant_output_dtype`` (the narrow dtype an ``acc_scale`` epilogue requants the i32 accumulator to,
    e.g. i8) is passed only when the caller needs a requant handoff — an int8 matmul chain commits each
    layer's accumulator back to i8 so it can feed the next mesh layer.

    The caller's pins OVERLAY the target's declared NUMERIC ``datapath`` facts; they do not replace them.
    This function used to build the dict from nothing but its own arguments, which meant every numeric
    fact the target declares and this caller does not pin — tolerances, inapplicable tiers,
    ``subnormal_operand_flush`` — silently took a dataclass default instead of the declared value. The
    corpus generator got the real block and the mesh path got defaults, from the same deriver.

    ``numeric_only`` because the profile block also holds corpus-AUTHORING choices (which oracle tiers the
    graded suite demands, its ``must_accelerate`` posture, the requant epilogue's output dtype), and this
    caller is describing hardware, not generating a corpus. Handing it those too would quietly give every
    gemmini tile an i8 requant handoff its caller never asked for."""
    from types import SimpleNamespace

    from ..targetgen import corpus_spec as CS
    from ..targetgen.capsule_runner import _bespoke_sim_via

    te = SimpleNamespace(target=target, sim_via=_bespoke_sim_via(target))
    datapath: dict = CS.profile_datapath(target, numeric_only=True)
    if operand_dtype:
        datapath["operand_dtype"] = operand_dtype
    if accum_dtype:
        datapath["accum_dtype"] = accum_dtype
    if requant_output_dtype:
        datapath["requant_output_dtype"] = requant_output_dtype
    return CS.derive_binding(te, datapath)


def _certify_tile_via_executor(target, mlir, *, m, k, n, binding, timeout) -> dict:
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


def _mesh_verify(plan: dict, *, target: str, package: str | None, timeout: int) -> dict:
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
    import tempfile

    from ..benchharness import runs_root
    from ..targetgen import corpus_spec as CS
    from ..targetgen import oot_runner
    from ..targetgen.capsule_runner import _SIM_ORACLES, _bespoke_sim_via, _endpoint_of

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
    from ..targetgen.capsule_runner import _screen_tiers_of

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
        op = "matmul" if d.op in ("matmul", "linear") else d.op
        rec: dict = {"op": d.op, "site": d.site}
        if op not in CS.BUILDERS:
            out["n_unsynthesizable"] += 1
            rec.update(status="no_tile_synthesizer", reason=f"no single-tile capsule synthesizer for mesh op {d.op!r}")
            out["per_tile"].append(rec)
            continue
        try:
            binding = _mesh_tile_binding(target, d.in_fmt, r.acc)
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
            res = _certify_tile_via_executor(target, mlir, m=M, k=K, n=N, binding=binding, timeout=timeout)
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


def run_matmul_on_mesh(
    target: str,
    A: list,
    W: list,
    *,
    operand_dtype: str | None = None,
    accum_dtype: str | None = None,
    simulator: str | None = None,
    package: str | None = None,
    epilogue: list | None = None,
    acc_scale: float | None = None,
    timeout: int = 900,
    observed: dict | None = None,
    _tiled: bool = False,
    _rescaled: bool = False,
) -> list | None:
    """Execute ``A @ W`` on the target's mesh oracle with the REAL operand values INJECTED (not
    materialized-from-name), and return the mesh's output tensor (nested list) — or ``None`` when this
    target has no reachable mesh path. ``A`` / ``W`` are the layer's real activations / weights.

    ``epilogue`` (e.g. ``["acc_scale"]``) applies the accumulator epilogue the layer commits with; an
    ``acc_scale`` epilogue re-quantizes the i32 accumulator to the target's narrow requant dtype (the
    target-declared ``requant_output_dtype``, e.g. i8 for an integer mesh) by the float ``acc_scale`` +
    saturating cast, so the layer's output is a same-dtype activation that can feed the NEXT mesh layer —
    this is what makes an int8 matmul CHAIN executable on the mesh end to end (the requant handoff).

    TARGET-AGNOSTIC: it dispatches on the target's DERIVED ``endpoint_kind`` (from the contract, never a
    literal) to that target's OWN oracle — a systolic/RoCC target (``inline_asm_insn``) through its generated
    OOT-package cert path, a self-hosted-ISA target (``external_backend``) through the program oracle whose
    RTL cosim mlc DERIVES from the target. The per-target kernel + harness come from the GENERATED package;
    this tool only builds the merlin_iface interface, injects the operands, and reads the output back.
    Fail-closed (``None``) on an unavailable oracle or an endpoint with no mesh path (never a fabricated
    result)."""
    from ..targetgen import corpus_spec as CS
    from ..targetgen.capsule_runner import _SIM_ORACLES, _bespoke_sim_via, _endpoint_of

    endpoint, model_ext = _endpoint_of(target)
    M, K, N = len(A), len(A[0]), len(W[0])
    # An acc_scale requant commits the accumulator back to the operand's narrow dtype so the layer's
    # output can feed the next mesh layer (the int8 chain handoff); tell the binding that narrow dtype.
    requant_out = (operand_dtype or "i8") if (epilogue and "acc_scale" in epilogue) else None
    binding = _mesh_tile_binding(target, operand_dtype, accum_dtype, requant_output_dtype=requant_out)

    # PAD TO THE MESH TILE EDGE. A generated package is entitled to reject a sub-tile extent, and a real
    # model is full of them: every matmul layer of an 8-token sequence has M=8 against a tile edge of 32
    # (atlas) or 16 (gemmini), so the mesh refused all 15 layers and the dispatch runtime silently fell
    # back to the host kernel for every one -- a whole model that reported "on mesh" while running
    # entirely on the CPU. Zero-padding is EXACT for a contraction: the padded rows of A and columns of W
    # contribute 0 to every retained output element, and the K padding multiplies zeros against zeros.
    # The result is sliced back to the true extent, so callers see the shape they asked for.
    _D = int(binding.tile_dim or 1)

    def _up(x):
        return ((int(x) + _D - 1) // _D) * _D if _D > 1 else int(x)

    _m_true, _n_true = M, N  # the extent the caller asked for, restored on the way out
    _Mp, _Kp, _Np = _up(M), _up(K), _up(N)
    _padded = (_Mp, _Kp, _Np) != (M, K, N)
    if _padded:
        import numpy as _np

        _A = _np.zeros((_Mp, _Kp), dtype=_np.float64)
        _A[:M, :K] = _np.asarray(A, dtype=_np.float64)
        _W = _np.zeros((_Kp, _Np), dtype=_np.float64)
        _W[:K, :N] = _np.asarray(W, dtype=_np.float64)
        A, W = _A.tolist(), _W.tolist()
        M, K, N = _Mp, _Kp, _Np

    def _build(m: int, k: int, n: int) -> str:
        e = {
            "name": "mesh_layer",
            "op": "matmul",
            "kind": "op",
            "source_role": "mesh_layer_real_operands",
            "source_reference": f"whole-model matmul layer {m}x{k}x{n} on the mesh with real operands",
            "M": m,
            "K": k,
            "N": n,
            "lhs": "A0",
            "weight": "W",
            "out": "Y0",
        }
        if epilogue:
            e["epilogue"] = list(epilogue)
            if acc_scale is not None:
                e["acc_scale"] = float(acc_scale)
        return CS.build(e, binding)[1]

    mlir = _build(M, K, N)

    # DISPATCH ORDER is deliberate (mirrors capsule_runner.oracle_adapters): a target whose contract
    # DECLARES an EXCLUSIVE bespoke sim (a self-hosted SIMT core graded on its own emitted kernel by its
    # own oracle, e.g. a cyclotron/muon backend) is routed FIRST — its endpoint is ALSO external_backend,
    # but the arc command-buffer program oracle grades the WRONG artifact for a SIMT kernel, so the bespoke
    # executor must take precedence. Only when no exclusive sim is declared does the endpoint kind pick the
    # path. Derived from the contract's sim_via + the _SIM_ORACLES registry — never a target-name branch.
    def _unpad(out):
        """Slice the padded mesh result back to the extent the caller asked for."""
        if out is None or not _padded:
            return out
        return [row[:_n_true] for row in out[:_m_true]]

    so = _SIM_ORACLES.get(_bespoke_sim_via(target))

    def _dispatch(_mlir, _A, _W) -> list | None:
        """One mesh call at whatever extent it is given."""
        if so is not None and so.exclusive:
            if observed is not None:
                observed["path"] = "bespoke_sim"
            return _matmul_via_bespoke_sim(
                target,
                _mlir,
                _A,
                _W,
                package=package,
                timeout=timeout,
                layer_id=_mesh_layer_id(len(_A), len(_A[0]), len(_W[0]), binding, epilogue, acc_scale),
                observed=observed,
            )
        if endpoint in (None, "inline_asm_insn", "upstream_target"):
            if observed is not None:
                observed["path"] = "oot_cert"
            return _matmul_via_oot_cert(
                target,
                _mlir,
                _A,
                _W,
                simulator=simulator,
                package=package,
                layer_id=_mesh_layer_id(len(_A), len(_A[0]), len(_W[0]), binding, epilogue, acc_scale),
                timeout=timeout,
                observed=observed,
            )
        if endpoint == "external_backend":
            if observed is not None:
                observed["path"] = "program_oracle"
            return _matmul_via_program_oracle(
                target,
                _mlir,
                _A,
                _W,
                model_ext=model_ext,
                package=package,
                timeout=timeout,
                operand_dtype=binding.operand_dtype,
                observed=observed,
            )
        return None  # no mesh-execution path derived for this endpoint kind

    # BLOCK A LAYER THAT DOES NOT FIT ON CHIP, rather than declining it. The working set of a matmul is
    # the weight tile K*N plus the activation tile M*K; past the target's scratchpad the mesh returns
    # nothing and the whole layer falls back to the host. Measured on lstmnetvit/gemmini: K and N are each
    # fine alone (1x16x512 and 1x512x16 both run) and together they are not (1x512x512 declines), so two
    # of 37 layers fell back and the model failed its must_accelerate gate at 35/37.
    #
    # `_capacity_fit_tile` already computes the fitting extent from RTL-derived facts, and `_mesh_verify`
    # already uses it to shrink the tile it CERTIFIES. Only the execution path did not, so the tile record
    # said "runs at this shape" about a shape the model never got to run. Same tiler, both paths.
    #
    # Splitting is EXACT on an integer datapath: partial products over a K split sum in the i32
    # accumulator, and an N or M split is independent columns/rows. It is NOT applied when an epilogue is
    # declared -- an acc_scale requant must see the whole accumulation, not each K block -- so such a
    # layer still declines, now with that as the stated reason.
    _cap = _operand_store_capacity_elems(target, binding.operand_dtype)
    _mt, _kt, _nt = M, K, N
    _n_sub = 1
    _tiled_by = None
    _acc_cap = _accumulator_capacity_elems(target, binding.accum_dtype)
    if _cap:
        _mt, _kt, _nt, _n_sub = _capacity_fit_tile(M, K, N, max(1, _D), _cap, _acc_cap)
        _tiled_by = "capacity"
    else:
        # NO CLASSIFIABLE OPERAND STORE -> the capacity tiler cannot even be evaluated, and this branch
        # used to end there: the whole extent went to a backend that may implement one tile, which
        # answers with a program that writes nothing. Fall back to what the backend DECLARES it built.
        # Measured: on the target whose 39 SRAMs mlc declines to classify, `capacity_elems` is None, so
        # the residency tiler was structurally unreachable and every multi-tile layer failed there.
        _decl = declared_primitive_tile(package)
        if _decl:
            _dm, _dk, _dn = (max(1, min(v, e)) for v, e in zip(_decl, (M, K, N)))
            if (_dm, _dk, _dn) != (M, K, N):
                _mt, _kt, _nt = _dm, _dk, _dn
                _n_sub = (-(-M // _mt)) * (-(-K // _kt)) * (-(-N // _nt))
                _tiled_by = "declared_primitive_tile"
    if (_mt, _kt, _nt) != (M, K, N):
        if epilogue:
            if observed is not None:
                _why = (
                    f"exceeds the on-chip working set ({_cap} elems)"
                    if _tiled_by == "capacity"
                    else f"exceeds the backend's declared primitive tile {list(_decl)}"
                )
                observed["decline"] = (
                    f"{M}x{K}x{N} {_why} and declares epilogue {list(epilogue)}; "
                    f"an accumulator epilogue cannot be split across K blocks"
                )
            return None
        import numpy as _np

        _An, _Wn = _np.asarray(A, dtype=_np.float64), _np.asarray(W, dtype=_np.float64)
        _acc = _np.zeros((M, N), dtype=_np.float64)
        _sub_mlir: dict[tuple[int, int, int], str] = {}
        if observed is not None:
            # ATTRIBUTION. The runtime is discharging an obligation the contract places on the TARGET
            # BACKEND. Blocking here keeps whole-model work moving, but a result produced this way is
            # evidence about our runtime plus their backend -- not evidence that their backend handles
            # a layer this size. Unrecorded, it reads as the latter.
            # WHICH tiler engaged is part of the attribution, not a detail: "the layer did not fit"
            # and "the backend only built one tile" are different facts about the backend, and only the
            # second one says the shape space is uncovered.
            observed["blocked"] = {
                "tile": [_mt, _kt, _nt],
                "n_subtiles": _n_sub,
                "capacity_elems": _cap,
                "tiled_by": _tiled_by,
            }
            observed["capacity_fit"] = {
                **capacity_fit(target, M, K, N, binding.operand_dtype, _D, binding.accum_dtype),
                # WHICH TILER CHOSE THIS EXTENT. Derived here: the fitting extent was computed from
                # the target's own RTL-derived operand-store capacity. The other tiler below finds a
                # width by probing until the backend stops refusing, and "derived from a hardware
                # capacity fact" is a much stronger claim than "found by probing" -- the record used
                # to make neither, so a reader could not tell them apart.
                "tile_source": "capacity_fact",
                "discharged_by": "merlin runtime (host-side residency tiling)",
                # WHETHER THE SPLIT IS NUMERICALLY FREE depends on the accumulator, and it is not free
                # everywhere. Over an integer accumulator the K-partials sum exactly, so a blocked layer
                # is bit-identical to an unblocked one. Over a FLOAT accumulator it is not: the hardware
                # would reduce all K terms in its own format, while blocking reduces each K block there
                # and then sums the blocks on the host. That difference is charged to the runtime too,
                # so a float target's blocked layer is not quotable as "what the accelerator computes".
                "split_exact": bool(binding.integer),
                "accum_dtype": binding.accum_dtype,
                "tiled_by": _tiled_by,
                "note": (
                    "the target backend did not satisfy capacity_fit at this extent; the runtime "
                    "split the layer so it could run. A whole-model pass resting on this is a "
                    "statement about the runtime + backend together."
                    if _tiled_by == "capacity"
                    else "the backend DECLARED it implements a single "
                    f"{_mt}x{_kt}x{_nt} tile, so the runtime drove the loop nest over this larger "
                    "extent. The backend lowered one tile; the generalization over M/K/N is the "
                    "RUNTIME'S, and this result is NOT evidence that the backend generalizes."
                )
                + (
                    ""
                    if binding.integer
                    else f" This datapath accumulates in {binding.accum_dtype}, so the split also "
                    f"CHANGES THE REDUCTION ORDER (per-block on the device, cross-block on the "
                    f"host) and the layer's numerics are not what the device alone would give."
                ),
            }
        for m0 in range(0, M, _mt):
            for n0 in range(0, N, _nt):
                for k0 in range(0, K, _kt):
                    a = _An[m0 : m0 + _mt, k0 : k0 + _kt]
                    w = _Wn[k0 : k0 + _kt, n0 : n0 + _nt]
                    shp = (a.shape[0], a.shape[1], w.shape[1])
                    if shp not in _sub_mlir:
                        _sub_mlir[shp] = _build(*shp)
                    part = _dispatch(_sub_mlir[shp], a.tolist(), w.tolist())
                    if part is None:
                        return None  # fail closed: a partial sum is not a result
                    _acc[m0 : m0 + a.shape[0], n0 : n0 + w.shape[1]] += _np.asarray(part, dtype=_np.float64)
        return _unpad(_acc.tolist())

    # EVALUATE THE OBLIGATION ON EVERY MESH PATH, not just one of them. This check used to sit inside
    # the RoCC/oot cert path, so a self-hosted-ISA target -- whose layers leave through the program
    # oracle -- had no obligation evaluated at all, and an oversized layer there declined with the same
    # uninformative "the oracle returned nothing" that gemmini produced before the check existed.
    # Evaluating it here covers bespoke_sim / oot_cert / program_oracle from one place, and only at the
    # FULL extent: the blocked path above already attributes each sub-tile it chose to fit.
    if observed is not None:
        try:
            # THE ACCUMULATOR DTYPE IS PASSED, and leaving it out was not cosmetic. It defaults to
            # 8 bits, so the accumulator term was evaluated at 4x the capacity this target
            # actually has (65536 elements against 16384), and the obligation recorded a fit for
            # layers the tiler standing beside it was already splitting. Measured: (96,64)@(64,512)
            # -- a shape whose own regression docstring records spike aborting on it -- reported
            # holds=True without it and holds=False with it. The binding has carried the real
            # accumulate type all along; the tiler passes it and these two sites did not.
            observed["capacity_fit_check"] = capacity_fit(
                target, M, K, N, binding.operand_dtype, _D, binding.accum_dtype
            )
        except Exception:  # noqa: BLE001 — unresolvable target: no obligation known
            pass
    out = _dispatch(mlir, A, W)
    if out is not None or _tiled:
        return _unpad(out)  # already a tile of a split layer: do not recurse further
    # Past the return above, the mesh DECLINED this extent. Name why, before either tiler is tried:
    # the obligation evaluated a moment ago already predicted it, or it could not be evaluated at all.
    if observed is not None:
        _cf = observed.get("capacity_fit_check") or {}
        if _cf.get("holds") is False:
            observed["contract_violation"] = {
                **_cf,
                "discharged_by": None,
                "detail": (
                    f"the target backend does not satisfy the capacity_fit obligation this "
                    f"interface requires: the contraction needs {_cf['required_elems']} "
                    f"resident elements against a declared capacity of {_cf['capacity_elems']}. "
                    f"The lowering must block for residency, not only tile the iteration space."
                ),
            }
        elif _cf.get("holds") is None:
            # SAY THAT THE OBLIGATION COULD NOT BE EVALUATED. A target that declares no on-chip operand
            # capacity gets `holds: None`, which is correct (never assume it holds) but silent -- and a
            # silent None is how a whole class of failures came to be reported as an unreachable oracle.
            # Recording it means a decline on such a target names the missing fact instead of hiding it.
            observed["capacity_fit_unevaluable"] = {
                **_cf,
                "detail": (
                    "this target declares no on-chip operand-store capacity, so the capacity_fit "
                    "obligation could not be evaluated and this decline is UNATTRIBUTED: it may "
                    "or may not be a residency failure. Declare the operand store in the "
                    "target's RTL facts to make the obligation decidable."
                ),
            }

    # The datapath has a magnitude floor: a layer whose operands are small is refused, and one doubling
    # of either operand flips it. Retry once with both operands rescaled to a power of two near 1.0 and
    # the result divided back. Exact -- a power-of-two factor moves only the exponent -- so this changes
    # which layers run and not what they compute.
    if not _rescaled:
        try:
            import math as _math

            _amax = max((abs(v) for row in A for v in row), default=0.0)
            _wmax = max((abs(v) for row in W for v in row), default=0.0)
            if _amax > 0.0 and _wmax > 0.0:
                _i = -int(_math.floor(_math.log2(_amax)))
                _j = -int(_math.floor(_math.log2(_wmax)))
                if _i or _j:
                    _sa, _sw = 2.0**_i, 2.0**_j
                    _As = [[v * _sa for v in row] for row in A]
                    _Ws = [[v * _sw for v in row] for row in W]
                    _scaled = run_matmul_on_mesh(
                        target,
                        _As,
                        _Ws,
                        operand_dtype=operand_dtype,
                        accum_dtype=accum_dtype,
                        simulator=simulator,
                        package=package,
                        timeout=timeout,
                        epilogue=epilogue,
                        acc_scale=acc_scale,
                        observed=observed,
                        _rescaled=True,
                    )
                    if _scaled is not None:
                        _inv = 1.0 / (_sa * _sw)
                        return _unpad([[v * _inv for v in row] for row in _scaled])
        except Exception:  # noqa: BLE001 — rescale is an optimisation, not a gate
            pass

    # EMPIRICAL TILER -- the fallback, reached only when the DERIVED one above could not act. The two
    # are complementary, not rivals: the blocking path above computes a fitting extent from the target's
    # own RTL-derived operand-store capacity and runs first, but `_operand_store_capacity_elems` returns
    # None for a target that declares no such fact. There is then nothing to derive a tile size FROM, and
    # inventing one would be a target literal in disguise. So find the width this backend ACCEPTS by
    # halving, then run the layer as N-tiles of that width: splitting along N is exact, because
    # C[:, a:b] is A @ W[:, a:b]. Costs a few refused probes ONCE per target.
    D = int(getattr(binding, "tile_dim", 0) or 0)
    if D <= 0:
        return None

    def _probed(extent, n_sub, axis):
        """Record that a PROBE, not a capacity fact, chose this extent.

        Both tilers write `observed["capacity_fit"]`, and the layer record downstream reads
        `tile_source` off it. "Derived from an RTL scratchpad capacity" and "found by halving until the
        backend stopped refusing" are very different provenance claims about the same number.
        """
        if observed is None:
            return
        observed["capacity_fit"] = {
            "holds": None,
            "required_elems": None,
            "capacity_elems": None,
            "tile_source": "probed",
            "tile": list(extent),
            "n_subtiles": int(n_sub),
            "split_axis": axis,
            "discharged_by": "merlin runtime (host-side residency tiling)",
            # An N or M split leaves every output element a single full-K reduction on the device, so
            # unlike the K-blocking the derived tiler does, this stays exact on a float accumulator too.
            "split_exact": True,
            "accum_dtype": binding.accum_dtype,
            "note": (
                f"this target declares no on-chip operand-store capacity, so no tile size could be "
                f"DERIVED; the extent above is the largest this backend accepted when probed by "
                f"halving. The numerics are the device's own (an {axis}-split is exact), but the "
                f"EXTENT is an empirical finding about this backend, not a derived property of the "
                f"hardware -- and the layer ran because the runtime split it."
            ),
        }

    _rows_kw = dict(
        operand_dtype=operand_dtype,
        accum_dtype=accum_dtype,
        simulator=simulator,
        package=package,
        timeout=timeout,
        epilogue=epilogue,
        acc_scale=acc_scale,
        observed=observed,
        M=M,
        record=_probed,
    )
    if N <= D:
        return _unpad(_mesh_rows(target, A, W, **_rows_kw))  # nothing to split along N
    # Keyed by K as well: the accepted width is bounded by the WORKING SET (K*w + M*K), so a width
    # discovered at K=128 is too wide at K=344 and its tiles are refused in turn. Measured: caching
    # without K left 4 of 15 small_llama layers on the host for exactly that reason.
    key = (target, operand_dtype or "", accum_dtype or "", K)
    width = _MESH_NTILE_WIDTH.get(key)
    if width is None:
        w = N
        while w > D:
            w = max(D, (w // 2 // D) * D or D)
            probe = run_matmul_on_mesh(
                target,
                A,
                [row[:w] for row in W],
                operand_dtype=operand_dtype,
                accum_dtype=accum_dtype,
                simulator=simulator,
                package=package,
                timeout=timeout,
                epilogue=epilogue,
                acc_scale=acc_scale,
                observed=observed,
                _tiled=True,
            )
            if probe is not None:
                width = w
                break
        if width is None:
            # Even one tile-dim-wide column is refused, so the M*K term is what does not fit.
            return _unpad(_mesh_rows(target, A, W, **_rows_kw))
        _MESH_NTILE_WIDTH[key] = width
    cols: list[list] = []
    for a in range(0, N, width):
        b = min(a + width, N)
        piece = run_matmul_on_mesh(
            target,
            A,
            [row[a:b] for row in W],
            operand_dtype=operand_dtype,
            accum_dtype=accum_dtype,
            simulator=simulator,
            package=package,
            timeout=timeout,
            epilogue=epilogue,
            acc_scale=acc_scale,
            observed=observed,
            _tiled=True,
        )
        if piece is None:
            return None  # a tile the discovered width should have covered
        cols.append(piece)
    _probed((M, K, width), (N + width - 1) // width, "N")
    return _unpad([sum((c[r] for c in cols), []) for r in range(M)])


def _mesh_rows(
    target,
    A,
    W,
    *,
    operand_dtype,
    accum_dtype,
    simulator,
    package,
    timeout,
    epilogue,
    acc_scale,
    M,
    observed=None,
    record=None,
):
    """Run a refused layer as row blocks. The working set is ``K*N + M*K``; N-tiling shrinks the first
    term and M-tiling the second, so a layer whose N is already at or below the tile dim (a projection
    down to a couple of columns, say) can only be helped by splitting M. Exact: C[a:b, :] is A[a:b, :] @ W.
    """
    if M <= 1:
        return None
    height = M
    while height > 1:
        height = max(1, height // 2)
        probe = run_matmul_on_mesh(
            target,
            A[:height],
            W,
            operand_dtype=operand_dtype,
            accum_dtype=accum_dtype,
            simulator=simulator,
            package=package,
            timeout=timeout,
            epilogue=epilogue,
            acc_scale=acc_scale,
            observed=observed,
            _tiled=True,
        )
        if probe is not None:
            break
    else:
        return None
    rows: list = []
    for a in range(0, M, height):
        piece = run_matmul_on_mesh(
            target,
            A[a : a + height],
            W,
            operand_dtype=operand_dtype,
            accum_dtype=accum_dtype,
            simulator=simulator,
            package=package,
            timeout=timeout,
            epilogue=epilogue,
            acc_scale=acc_scale,
            observed=observed,
            _tiled=True,
        )
        if piece is None:
            return None
        rows.extend(piece)
    if record is not None:
        record((height, len(A[0]) if A else 0, len(W[0]) if W else 0), (M + height - 1) // height, "M")
    return rows


def _matmul_via_oot_cert(
    target, mlir, A, W, *, simulator, package, timeout, layer_id: str = "mesh_layer", observed: dict | None = None
) -> list | None:
    """Systolic/RoCC path: certify the matmul through the target's generated OOT package on its ELF/mesh
    oracle, with the real operands injected (``inputs`` -> the package harness's ``materialize_inputs``).

    The ``capacity_fit`` obligation is evaluated by the caller (``run_matmul_on_mesh``) BEFORE any mesh
    path runs, so that if the backend then aborts we can say which contract predicate it failed instead
    of reporting an unreachable oracle. It lived here once, which left every non-RoCC path unchecked."""
    import tempfile

    from ..benchharness import runs_root
    from ..targetgen import capsule_common as CC
    from ..targetgen import oot_runner

    sim = _resolve_oot_mesh_simulator(target, simulator)
    pkg = package or _default_oot_package(target)
    if pkg is None:
        return None
    _lid = _mesh_invocation_id(layer_id, A, W)
    with tempfile.TemporaryDirectory(prefix=_lid + "_") as td:
        iface = Path(td) / f"{_lid}.interface.mlir"
        iface.write_text(mlir, encoding="utf-8")
        res = oot_runner.certify(
            pkg,
            iface,
            runs_root=str(runs_root(target, "mesh_run")),
            run_id=_lid,
            simulator=sim,
            target=target,
            timeout=timeout,
            inputs={"A0": A, "W": W},
            require_accelerator_trace=True,
        )
    if observed is not None:
        # the cert reports its oracle as a record ({kind, derived_from_rtl, ...}); the simulator that ran
        # it is the other half of the identity, so label with both when the record is present.
        _kind = CC.oracle_kind(res.get("oracle"))
        observed["oracle"] = f"{target}-{_kind}-{sim}" if _kind else f"{target}-{sim}"
        # Structured fidelity and content identity travel with THIS dynamic call.  The string above is
        # retained for existing diagnostics; formal grading consumes these runner-produced records.
        observed["oracle_evidence"] = res.get("oracle")
        observed["trace_check"] = res.get("trace_check")
        observed["artifact_identity"] = res.get("artifact_identity")
        observed["cert_run_id"] = res.get("run_id")
    if res.get("status") != "pass":
        # KEEP THE REASON. Returning a bare None here is why a whole-model fallback could only be
        # described as "unsynthesizable at this shape, or the oracle was unreachable" -- a guess covering
        # two very different causes, offered because the cert's own verdict was discarded one frame down.
        if observed is not None:
            observed["decline_status"] = res.get("status")
            observed["decline"] = (res.get("failure") or {}).get("detail") or res.get("failure")
            # The contract ATTRIBUTION for this decline is applied by the caller, which evaluated the
            # obligation for whichever mesh path ran. Doing it here attributed the RoCC path only.
        return None
    return (res.get("oracle_outputs") or {}).get("Y0")


def _matmul_via_program_oracle(
    target, mlir, A, W, *, model_ext, package, timeout, operand_dtype=None, observed: dict | None = None
) -> list | None:
    """Self-hosted-ISA (external_backend) path: emit the target's kernel from the interface through its
    generated OOT package (target-agnostic ``run_entrypoints``), inject the real operands onto the command
    buffer's leaf tensors, and run on the target's mlc-DERIVED arc cosim via the generic program oracle.
    Returns the output or ``None`` (fail-closed). The kernel/codegen is the GENERATED package's; the operand
    injection + oracle dispatch are target-agnostic. Requires ``model_ext`` (the operand-layout model)."""
    if not model_ext:
        return None
    from ..targetgen import mesh_program_run

    return mesh_program_run.matmul_on_program_oracle(
        target,
        mlir,
        A,
        W,
        model_ext=model_ext,
        package=package or _default_oot_package(target),
        timeout=timeout,
        dtype_hint=operand_dtype,
        observed=observed,
    )


def _matmul_via_bespoke_sim(
    target, mlir, A, W, *, package, timeout, layer_id: str = "mesh_layer", observed: dict | None = None
) -> list | None:
    """Exclusive bespoke-sim path: a self-hosted SIMT core (endpoint ``external_backend``) graded on the
    kernel its OWN generated package emits, by its OWN declared bespoke oracle (e.g. cyclotron via the muon
    backend) — NOT the arc command-buffer program oracle, which would grade the wrong artifact for a SIMT
    kernel. Emits the matmul kernel from the target's generated OOT package via the SAME target-agnostic
    entrypoint runner the grader uses (``capsule_common.run_entrypoints``), INJECTS the real ``A``/``W``
    operands onto the command buffer's leaf tensors (``preload_b64``, decoded harness-side by
    ``muon_harness``), runs on the target's DECLARED exclusive bespoke oracle, and reads the named output
    tensor (``Y0``) back off the device. Target-agnostic: the sim engine + adapters come from the DERIVED
    ``_SIM_ORACLES`` entry (contract ``sim_via``); the kernel/codegen is the generated package's. Fail-closed
    (``None``) on a missing package, an unavailable oracle, or an entrypoint/oracle failure — never a
    fabricated result."""
    import base64
    import tempfile

    from ..benchharness import runs_root
    from ..targetgen import capsule_common as CC
    from ..targetgen import capsule_runner as CR
    from ..targetgen import mesh_program_run as MP
    from ..targetgen.capsule_common import make_run_paths

    pkg = package or _default_oot_package(target)
    if pkg is None:
        return _refuse("no OOT package resolved for this target")
    so = CR._SIM_ORACLES.get(CR._bespoke_sim_via(target))
    if so is None or not so.exclusive:
        return None
    ok, _reason = so.available(target)
    if not ok:
        return None
    adapters = so.adapters(target)  # {tier: adapter}
    # The FUNCTIONAL tier (L2) carries the numeric grade for a SIMT core; fall back to any adapter present.
    run = adapters.get("L2") or next(iter(adapters.values()), None)
    if run is None:
        return None

    with tempfile.TemporaryDirectory(prefix="mesh_bsim_") as td:
        tdp = Path(td)
        cdir = tdp / "cap"
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "capsule.interface.mlir").write_text(mlir, encoding="utf-8")
        _bid = layer_id
        capsule = {
            "name": _bid,
            "kind": "op",
            "interface_mlir": "capsule.interface.mlir",
            "operation": {"op": "matmul", "attributes": {}},
            "__dir__": str(cdir),
            "required_oracle_tiers": ["L2"],
        }
        # A run id per LAYER SHAPE: `mesh_layer` was hardcoded, so every layer of a model wrote into one
        # run dir and overwrote the last, leaving nothing to attribute a failure to afterwards.
        # Unique per INVOCATION, not per shape. A model repeats shapes -- small_llama has two 8x128x128
        # layers and two 8x344x128 -- and a shape-keyed id put each repeat in the first one's directory,
        # where it collided with the previous run's artifacts and the oracle exited 1. The original bug
        # was a single hardcoded "mesh_layer" for every layer; keying by shape fixed only distinct shapes.
        _rid = f"mesh_layer_{len(A)}x{len(A[0]) if A else 0}x{len(W[0]) if W else 0}_{next(_MESH_RUN_SEQ)}"
        paths = make_run_paths(
            runs_root(target, "mesh_bsim"), _rid, suite="mesh", target=target, dtype="prog", benchmark=_rid
        )
        try:
            _built = _built_mesh_package(pkg, timeout)  # built once per process, not once per layer
            _pkg, cb, kernel_text = CC.run_entrypoints(
                _built, pkg, capsule, paths, contract=None, timeout=timeout, fourth_output_name="kernel.cpp"
            )
        except Exception as _e:  # noqa: BLE001 — package can't emit this kernel
            return _refuse(f"run_entrypoints raised {type(_e).__name__}: {str(_e)[:160]}")
        if cb is None or not kernel_text:
            return _refuse("the package emitted no command buffer or kernel text")
        # INJECT the real operands onto the cb's leaf tensors (encoded for each tensor's declared dtype);
        # the muon harness decodes ``preload_b64`` and embeds THESE values instead of the materialized ones.
        operands = {"A0": A, "W": W}
        for tname, tspec in (cb.get("tensors") or {}).items():
            if tspec.get("role") in ("input", "weight", "bias") and tname in operands:
                raw = MP._encode_operand(operands[tname], tspec.get("dtype", "f32"))
                if raw is None:
                    return _refuse(
                        f"operand {tname!r} could not be encoded for dtype "
                        f"{tspec.get('dtype', 'f32')!r} (non-finite or out of range?)"
                    )
                tspec["preload_b64"] = base64.b64encode(raw).decode()
        try:
            res = run(cb, kernel_text, tdp / "oracle", timeout)
        except Exception as _e:  # noqa: BLE001 — oracle unavailable / run failure
            # This exit swallowed the oracle's own exception, so an oracle that CRASHED and an oracle that
            # was simply absent reached the caller identically -- and a layer dying here looked exactly
            # like a backend that cannot emit the extent.
            # Dump the WHOLE thing: the first 200 chars of this exception are a cyclotron diagnostic
            # banner ("rename pool: ... (limit 256)") that is well under its own limit and says nothing
            # about the failure, so a truncated reason reads as an explanation while withholding one.
            _seq = next(_MESH_RUN_SEQ)
            _dump = Path(tempfile.gettempdir()) / f"mesh_refusal_{_seq}.txt"
            try:
                _dump.write_text(f"{type(_e).__name__}: {_e}", encoding="utf-8")
                # Dump the OPERANDS too. A layer that fails inside a model and succeeds standalone differs
                # only in the values it was handed, so without them every hypothesis costs a full model
                # run to test. With them the case replays in seconds.
                import numpy as _np

                _np.savez_compressed(
                    Path(tempfile.gettempdir()) / f"mesh_refusal_{_seq}.npz",
                    A=_np.asarray(A, dtype=_np.float64),
                    W=_np.asarray(W, dtype=_np.float64),
                )
            except Exception as _de:  # noqa: BLE001
                # Say WHY the diagnostic write failed. Swallowing this is the same mistake the reason
                # recording exists to fix: the operand dump silently never appeared and the next run was
                # spent discovering that, not diagnosing the layer.
                try:
                    _dump.write_text(
                        f"{type(_e).__name__}: {_e}\n\n[operand dump failed: {type(_de).__name__}: {_de}]",
                        encoding="utf-8",
                    )
                except Exception:  # noqa: BLE001
                    _dump = None
            return _refuse(
                f"the oracle raised {type(_e).__name__}: {str(_e)[:1500]}{f' [full: {_dump}]' if _dump else ''}"
            )
        if observed is not None:
            observed["oracle"] = CC.oracle_kind(res.get("oracle")) or CR._bespoke_sim_via(target)
        outs = res.get("outputs") or {}
        got = outs.get("Y0") or next(iter(outs.values()), None)
        if got is None:
            return _refuse(f"the oracle ran but produced no output tensor (keys: {sorted(outs)})")
        return got
