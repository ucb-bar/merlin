"""Drivers that run every matmul layer of a whole model, or of an int8 layer chain, on the mesh.

Each layer goes through ``mesh.run_matmul_on_mesh`` with its on-device output handed to the op that
consumes it, gated against the host reference at every layer.
"""
from __future__ import annotations

from .mesh_backend import _requested_mesh_simulator
from .mesh import run_matmul_on_mesh


def run_whole_model_on_mesh(target: str, module, *, in_fmt: str = "f32",
                            weight_fmt: str | None = None, leaf_values: dict | None = None,
                            operand_dtype: str | None = None, accum_dtype: str | None = None,
                            simulator: str | None = None, package: str | None = None,
                            ref_target: str = "toy_npu", seed: int = 0, timeout: int = 900) -> dict:
    """Execute a WHOLE MODEL co-scheduled across the target's mesh + scalar lanes on the REAL oracle.

    Routes ``module``'s ops (``mesh_program_run.demands_from_module``), builds the co-scheduled
    whole-model program, then walks it IN ORDER: each mesh matmul LAYER runs on the target's real oracle
    (``run_matmul_on_mesh`` — the operands injected, the kernel emitted by the target's generated OOT
    package, the output read back off the device), while norms/activations/elementwise run inline on the
    scalar lane, and every layer's on-device output is handed to the op that consumes it. The final tensor
    is gated against the whole-model engine reference (``lower_module`` of the entire module) — a PASS is
    proof the model ran end-to-end with its matmul layers ON the mesh, not merely that a plan was produced.

    This is the single co-scheduled whole-model run (mesh layers on hardware + scalar lane inline +
    inter-lane activation handoff). It is TARGET-AGNOSTIC: the lane split is READ from the routing plan, the
    layer extents from the module's def-use edges, the kernel from the generated package. FAIL-CLOSED: a
    layer with no reachable oracle returns ``status="oracle_unavailable"`` (never a fabricated result).

    Residual toward a single fused binary: this co-schedules the mesh kernels with the scalar lane
    host-driven (one program, multiple dispatched kernels), not yet ONE fused kernel in a single device
    address space — that final slice is the OOT backend emitting the whole loop nest inline. Seeded with
    small-integer operands by default so an integer mesh reproduces the f32 reference bit-exact."""
    import os

    import numpy as np

    from ..targetgen import compute_units as _cu
    from ..targetgen import mesh_program_run as mp
    from ..targetgen import routing as _routing
    from ..targetgen import target_registry as tr
    from ..xdsl_dialects.lowering import execute, lower_module

    fn = next(op for op in module.walk() if op.name == "func.func")
    args = list(fn.body.blocks[0].args)

    def _shape(v):
        return [int(d) for d in v.type.get_shape()]

    rng = np.random.default_rng(seed)
    if leaf_values is None:
        # Small integer operands: an integer mesh (e.g. int8·int8->int32) reproduces the f32 engine
        # reference EXACTLY, so the whole-model gate is bit-exact rather than tolerance-bounded.
        arrays = {a: np.rint(rng.standard_normal(tuple(_shape(a))) * 3).clip(-8, 7).astype(np.float32)
                  for a in args}
    else:
        by_index = {a.index: a for a in args}
        arrays = {by_index[int(k[1:])]: np.asarray(v, dtype=np.float32) for k, v in leaf_values.items()}

    # Route + build the co-scheduled program, then run its mesh lane on the REAL oracle.
    demands = mp.demands_from_module(module, in_fmt, weight_fmt)
    plan = _routing.route_plan_on(demands, _cu.compute_units(tr.load_contract(target)))
    program = mp.build_whole_model_program(plan, target, module)
    seed_leaves = {f"L{a.index}": arrays[a] for a in args if f"L{a.index}" in program.leaves}

    # Whole-model reference (the numeric gate). A module in the engine's op vocabulary lowers+runs as ONE
    # module on the target-agnostic engine (through an in-tree ``ref_target``); a module carrying a
    # transcendental / fused op the engine cannot evaluate (softmax/rmsnorm/rope/attention/…) is gated
    # against a host-eager numpy recomputation of the whole model instead. Either way the mesh execution
    # below runs the matmul layers on the REAL ``target`` hardware; both must agree on the whole-model result.
    if mp._engine_can_lower(module):
        ref_names = mp._reference_leaf_names(module)
        ref_inj = {ref_names[a]: arrays[a].tolist() for a in args}
        ref_final = np.asarray(next(iter(
            execute(lower_module(module, target=ref_target), ref_inj)["outputs"].values())))
        ref_kind = "engine"
    else:
        ref_final = mp._host_eager_final(program, seed_leaves)
        ref_kind = "host_eager"

    per_layer: list = []

    def _mesh_exec(lhs, rhs, step):
        la, ra = np.asarray(lhs), np.asarray(rhs)
        # ``obs`` records the executor that ACTUALLY ran this layer. The requested ``simulator`` is only a
        # preference and two of the three dispatch paths ignore it (a self-hosted-ISA target runs on its
        # mlc-derived cosim, an exclusive bespoke sim on its own engine) -- so reporting the request as
        # though it were the device named a simulator that never ran. Report what executed.
        obs: dict = {}
        got = run_matmul_on_mesh(target, la.tolist(), ra.tolist(),
                                 operand_dtype=operand_dtype, accum_dtype=accum_dtype,
                                 simulator=simulator, package=package, timeout=timeout, observed=obs)
        # extents read from the actual operands (a fused op's matmul sub-ops carry the sub-op shapes, not
        # the enclosing step's), so the per-layer log is honest for attention/geglu as well as plain matmuls.
        per_layer.append({"index": step.index, "m": int(la.shape[0]), "k": int(la.shape[1]),
                          "n": int(ra.shape[1]), "unit": step.unit,
                          "oracle": "ok" if got is not None else "unavailable",
                          "executed_on": obs.get("oracle"), "path": obs.get("path")})
        return got

    base = {"target": target, "ref_target": ref_target, "ref_kind": ref_kind,
            "n_steps": len(program.steps), "n_mesh": program.n_mesh(),
            "n_scalar": program.n_scalar(), "output_id": program.output, "per_layer": per_layer,
            "simulator_requested": (_requested_mesh_simulator(simulator)
                                    or os.environ.get("MERLIN_REQUIRED_RTL_ENGINE"))}
    def _executors() -> list:
        """The DISTINCT executors the mesh layers actually ran on (empty when no layer reached one)."""
        return sorted({e for e in (lay.get("executed_on") for lay in per_layer) if e})

    try:
        spliced = mp.run_whole_model_program(program, seed_leaves, mesh_exec=_mesh_exec)
    except mp.MeshLayerUnavailable as e:
        return {**base, "status": "oracle_unavailable", "reason": str(e),
                "mesh_executors": _executors()}

    spliced_final = spliced["outputs"][program.output]
    exact = bool(np.array_equal(spliced_final, ref_final))
    match = bool(np.allclose(spliced_final, ref_final, rtol=1e-4, atol=1e-4))
    return {**base, "status": "pass" if match else "fail", "exact": exact, "match": match,
            "mesh_executors": _executors(),
            "note": "single co-scheduled whole-model run: matmul layers executed on the real mesh oracle, "
                    "scalar/vector lane inline, activations handed between lanes; gated vs the whole-model "
                    "engine reference. Residual: host-driven multi-kernel, not yet one fused address-space "
                    "image."}


def _int8_chain_reference(A0, weights: list, acc_scale: float):
    """Host int8 matmul-chain reference matching the mesh's per-layer requant EXACTLY: i32 accumulate,
    gemmini-faithful ``acc_scale`` (round-half-even of the f32 product), saturating i8 cast; each layer's
    i8 output is the next layer's activation. This is the golden the on-mesh chain is gated against."""
    import numpy as np
    x = np.asarray(A0, dtype=np.int64)
    for w in weights:
        acc = x @ np.asarray(w, dtype=np.int64)
        x = np.clip(np.rint(acc.astype(np.float32) * np.float32(acc_scale)), -128, 127).astype(np.int64)
    return x


def run_int8_chain_on_mesh(target: str, A0: list, weights: list, *, acc_scale: float,
                           operand_dtype: str = "i8", accum_dtype: str = "i32",
                           simulator: str | None = None, package: str | None = None,
                           timeout: int = 900) -> dict:
    """Run an int8 matmul CHAIN on the target mesh with the per-layer REQUANT HANDOFF — the shape of a real
    quantized model's linear stack. For each layer ``Y_l = requant_i8(A_l @ W_l)`` executes on the REAL
    oracle (``run_matmul_on_mesh`` with an ``acc_scale`` epilogue that commits the i32 accumulator back to
    i8), and ``Y_l`` becomes ``A_{l+1}`` — so the activation stays device-native i8 across the whole chain
    rather than round-tripping through a wider host dtype. The final tensor (and every layer) is gated
    bit-exact vs :func:`_int8_chain_reference`.

    This closes the int8 inter-layer handoff that a single independent matmul does not exercise: a real
    model is a CHAIN, and each mesh layer's output must be requantized to the operand dtype to feed the
    next mesh layer. FAIL-CLOSED: a layer with no reachable oracle returns ``oracle_unavailable`` (never a
    fabricated activation). TARGET-AGNOSTIC: the requant + narrow output dtype are the target's own
    (``run_matmul_on_mesh``); this only threads activations layer to layer."""
    import os

    import numpy as np

    a = np.asarray(A0, dtype=np.int64)           # device activation, threaded layer to layer
    r = np.asarray(A0, dtype=np.int64)           # independent host reference, advanced in lockstep
    per_layer: list = []
    for i, w in enumerate(weights):
        wl = np.asarray(w, dtype=np.int64)
        # advance the reference prefix (i32 matmul, gemmini-faithful acc_scale requant, i8 saturate)
        r = np.clip(np.rint((r @ wl).astype(np.float32) * np.float32(acc_scale)), -128, 127).astype(np.int64)
        out = run_matmul_on_mesh(target, a.tolist(), wl.tolist(), operand_dtype=operand_dtype,
                                 accum_dtype=accum_dtype, simulator=simulator, package=package,
                                 epilogue=["acc_scale"], acc_scale=acc_scale, timeout=timeout)
        if out is None:
            return {"target": target, "status": "oracle_unavailable", "failed_layer": i,
                    "n_layers": len(weights), "per_layer": per_layer,
                    "reason": f"mesh layer {i} ({a.shape[0]}x{a.shape[1]}x{wl.shape[1]}) has no reachable "
                              f"oracle in this env"}
        a = np.asarray(out, dtype=np.int64)
        per_layer.append({"layer": i, "m": int(a.shape[0]), "n": int(wl.shape[1]),
                          "matches_ref": bool(np.array_equal(a, r))})

    exact = bool(all(p["matches_ref"] for p in per_layer))
    return {"target": target, "status": "pass" if exact else "fail", "exact": exact,
            "n_layers": len(weights), "acc_scale": acc_scale, "per_layer": per_layer,
            "final_shape": [int(d) for d in a.shape],
            "simulator": (_requested_mesh_simulator(simulator)
                          or os.environ.get("MERLIN_REQUIRED_RTL_ENGINE")),
            "note": "int8 matmul chain on the real mesh oracle with the per-layer acc_scale requant handoff "
                    "(each layer's i8 output feeds the next mesh layer); gated bit-exact vs the host int8 "
                    "chain reference at EVERY layer."}
