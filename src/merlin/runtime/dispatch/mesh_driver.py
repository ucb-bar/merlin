"""Plumbing for running a layer through a target's out-of-tree backend package.

The per-process package build cache, simulator selection for the OOT certification path, the
diagnostic record of why the last mesh attempt declined (``_MESH_REFUSAL``), and the content-addressed
ids that give each mesh layer and invocation its own artifact directory.
"""

from __future__ import annotations

import itertools
import json
import threading
from pathlib import Path

from merlin.runtime.dispatch.mesh import run_matmul_on_mesh
from merlin.verification.references import mesh_arithmetic as _reference

# ---- mesh_backend ------------------------------------------------------------------------------

_MESH_REFUSAL: dict = {}  # diagnostic only: why the last mesh attempt returned None


def _refuse(reason: str):
    """Record why a mesh attempt is giving up, then give up. Diagnostic only -- callers still
    just see None, which stays the fail-closed contract."""
    _MESH_REFUSAL["reason"] = reason
    return None


_MESH_RUN_SEQ = itertools.count()  # one run dir per mesh-layer invocation
_MESH_PKG_CACHE: dict[str, object] = {}
_MESH_PKG_LOCK = threading.Lock()
_MESH_SIM_CACHE: dict[tuple[str, str | None, str | None], str] = {}
_MESH_SIM_LOCK = threading.Lock()


def _built_mesh_package(pkg_dir: str, timeout: int):
    """The loaded+scanned+built package for ``pkg_dir``, built at most once per process.

    The whole-model mesh path calls this once per matmul layer. Rebuilding per layer is pure waste and
    turns one broken build into a per-layer storm of identical failures.
    """
    key = str(Path(pkg_dir).resolve())
    with _MESH_PKG_LOCK:
        hit = _MESH_PKG_CACHE.get(key)
    if hit is not None:
        return hit
    from merlin.development.targetgen.packaging.loading import build_package, integrity_scan, load_package

    obj = load_package(pkg_dir, contract=None)  # the same sequence run_entrypoints does when pkg is None
    integrity_scan(obj)
    build_package(obj)
    with _MESH_PKG_LOCK:
        _MESH_PKG_CACHE[key] = obj
    return obj


def _requested_mesh_simulator(simulator: str | None = None) -> str | None:
    """Return the caller/legacy mesh-simulator request, without inventing a default."""
    import os

    requested = simulator if simulator is not None else os.environ.get("MERLIN_MESH_SIM")
    return str(requested).strip() if requested is not None and str(requested).strip() else None


def _resolve_oot_mesh_simulator(target: str, simulator: str | None = None) -> str:
    """Resolve the simulator for a real OOT mesh invocation through the L3 policy.

    ``MERLIN_MESH_SIM`` predates dynamic elaborated-RTL engine selection and remains an explicit
    functional-bootstrap override (notably ``spike``).  It must never override a campaign-wide
    ``MERLIN_REQUIRED_RTL_ENGINE`` pin.  With no explicit request, the shared Chipyard L3 policy chooses
    an available equal-fidelity engine; this is intentionally not a hidden Verilator fallback.
    """
    import os

    requested = _requested_mesh_simulator(simulator)
    required = os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip() or None
    if required is not None and requested is not None and requested != required:
        raise RuntimeError(f"required RTL engine {required!r} conflicts with requested mesh simulator {requested!r}")

    # An unpinned explicit request is deliberate (e.g. Spike bootstrap), so preserve it.  A required
    # engine, even when repeated in MERLIN_MESH_SIM, is still availability-checked by the central policy.
    if required is None and requested is not None:
        return requested

    # One whole-model build asks this for every mesh layer. The policy probes simulator
    # availability; repeat that only when the target or caller's constraint changes.
    key = (target, requested, required)
    with _MESH_SIM_LOCK:
        if key in _MESH_SIM_CACHE:
            return _MESH_SIM_CACHE[key]
        from merlin.target.simulators.oracle_policy import select_chipyard_engine as chipyard_l3_selection

        selected = chipyard_l3_selection(target)
        engine = str(selected.get("engine") or "").strip()
        if not engine:
            raise RuntimeError(f"{target}: chipyard L3 policy returned no RTL engine")
        if required is not None and engine != required:
            raise RuntimeError(
                f"{target}: selected RTL engine {engine!r} differs from required RTL engine {required!r}"
            )
        _MESH_SIM_CACHE[key] = engine
        return engine


def _mesh_layer_id(m: int, k: int, n: int, binding, epilogue: list | None, acc_scale: float | None) -> str:
    """A per-layer artifact identity for the mesh run directory.

    Every mesh layer used the SAME run_id ("mesh_layer"), so all of a model's layers wrote to and read
    from one artifact directory and clobbered each other. The failure was not subtle once seen: two
    layers of two different models each received the OTHER's interface --
    ``(8,344)@(344,128)`` was handed ``(288,96)`` and ``(64,288)@(288,96)`` was handed ``(352,128)`` --
    and three further layers all expected one stale ``(128,128)``. A layer that fails this way runs
    perfectly in isolation, which is why it read as a shape bug rather than a collision.

    The id is CONTENT-ADDRESSED on what actually changes the emitted kernel: extent, operand and
    accumulator dtype, and the epilogue (an acc_scale requant emits a different kernel). Two genuinely
    identical layers therefore still share one directory -- small_llama repeats the same extent eight
    times and should compile once -- while two different layers can never collide."""
    parts = [f"{m}x{k}x{n}", binding.cap_dtype(binding.operand_dtype), binding.cap_dtype(binding.accum_dtype)]
    if epilogue:
        parts.append("-".join(str(e) for e in epilogue))
    if acc_scale is not None:
        parts.append(f"s{float(acc_scale):.6g}")
    return "mesh_layer_" + "_".join(parts).replace(".", "p").replace("-", "_")


def _mesh_invocation_id(layer_id: str, A: list, W: list) -> str:
    """A stable run directory for one exact real-operand mesh invocation.

    ``layer_id`` identifies emitted code shape, not an execution.  Repeated layers of the same shape
    can carry different activations/weights and therefore produce different injected interfaces, ELFs
    and traces.  Sharing the shape directory overwrote the first call while its now-unrecoverable digest
    remained in the model ledger.  Bind the exact operands into the id; identical calls may safely reuse
    one directory because their complete compiler/oracle inputs are identical.
    """
    import hashlib

    payload = json.dumps({"A": A, "W": W}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return f"{layer_id}_input_{hashlib.sha256(payload).hexdigest()}"


# Runtime transports retain the caller's actual selected package callback.


def _matmul_via_oot_cert(
    target,
    mlir,
    A,
    W,
    *,
    simulator,
    package,
    timeout,
    layer_id: str = "mesh_layer",
    observed: dict | None = None,
    _package_selector,
) -> list | None:
    """Systolic/RoCC path: certify the matmul through the target's generated OOT package on its ELF/mesh
    oracle, with the real operands injected (``inputs`` -> the package harness's ``materialize_inputs``).

    The ``capacity_fit`` obligation is evaluated by the caller (``run_matmul_on_mesh``) BEFORE any mesh
    path runs, so that if the backend then aborts we can say which contract predicate it failed instead
    of reporting an unreachable oracle. It lived here once, which left every non-RoCC path unchecked."""
    _default_oot_package = _package_selector
    import tempfile

    from merlin.common.paths import runs_root
    from merlin.development.targetgen.corpus import capsule_io as CC
    from merlin.development.targetgen.packaging import loading as oot_runner

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
    target,
    mlir,
    A,
    W,
    *,
    model_ext,
    package,
    timeout,
    operand_dtype=None,
    observed: dict | None = None,
    _package_selector,
) -> list | None:
    """Self-hosted-ISA (external_backend) path: emit the target's kernel from the interface through its
    generated OOT package (target-agnostic ``run_entrypoints``), inject the real operands onto the command
    buffer's leaf tensors, and run on the target's mlc-DERIVED arc cosim via the generic program oracle.
    Returns the output or ``None`` (fail-closed). The kernel/codegen is the GENERATED package's; the operand
    injection + oracle dispatch are target-agnostic. Requires ``model_ext`` (the operand-layout model)."""
    _default_oot_package = _package_selector
    if not model_ext:
        return None
    from merlin.development.targetgen.oracles import program_probes as mesh_program_run

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
    target,
    mlir,
    A,
    W,
    *,
    package,
    timeout,
    layer_id: str = "mesh_layer",
    observed: dict | None = None,
    _package_selector,
) -> list | None:
    """Exclusive bespoke-sim path: a self-hosted SIMT core (endpoint ``external_backend``) graded on the
    kernel its OWN generated package emits, by its OWN declared bespoke oracle (e.g. cyclotron via the SIMT
    target's backend) — NOT the arc command-buffer program oracle, which would grade the wrong artifact for a SIMT
    kernel. Emits the matmul kernel from the target's generated OOT package via the SAME target-agnostic
    entrypoint runner the grader uses (``capsule_common.run_entrypoints``), INJECTS the real ``A``/``W``
    operands onto the command buffer's leaf tensors (``preload_b64``, decoded harness-side by
    the target's harness), runs on the target's DECLARED exclusive bespoke oracle, and reads the named output
    tensor (``Y0``) back off the device. Target-agnostic: the sim engine + adapters come from the DERIVED
    ``_SIM_ORACLES`` entry (contract ``sim_via``); the kernel/codegen is the generated package's. Fail-closed
    (``None``) on a missing package, an unavailable oracle, or an entrypoint/oracle failure — never a
    fabricated result."""
    _default_oot_package = _package_selector
    import base64
    import tempfile

    from merlin.common.paths import runs_root
    from merlin.development.targetgen.corpus import capsule_io as CC
    from merlin.development.targetgen.corpus.capsule_io import make_run_paths
    from merlin.development.targetgen.oracles import program_probes as MP
    from merlin.target.simulators.oracle_policy import _SIM_ORACLES
    from merlin.target.simulators.oracle_policy import selected_sim_via as _bespoke_sim_via

    pkg = package or _default_oot_package(target)
    if pkg is None:
        return _refuse("no OOT package resolved for this target")
    so = _SIM_ORACLES.get(_bespoke_sim_via(target))
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
        # the SIMT harness decodes ``preload_b64`` and embeds THESE values instead of the materialized ones.
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
            observed["oracle"] = CC.oracle_kind(res.get("oracle")) or _bespoke_sim_via(target)
        outs = res.get("outputs") or {}
        got = outs.get("Y0") or next(iter(outs.values()), None)
        if got is None:
            return _refuse(f"the oracle ran but produced no output tensor (keys: {sorted(outs)})")
        return got


for _legacy_function in (
    _refuse,
    _built_mesh_package,
    _requested_mesh_simulator,
    _resolve_oot_mesh_simulator,
    _mesh_layer_id,
    _mesh_invocation_id,
):
    _legacy_function.__module__ = "merlin.runtime.dispatch.mesh_driver"
del _legacy_function


# ---- mesh_model --------------------------------------------------------------------------------
# Existing whole-model and int8-chain mesh runtime driving.

# Preserve the original caller-selected reference hooks without duplicating them.
_int8_chain_policy = _reference._int8_chain_policy
_int8_chain_step = _reference._int8_chain_step


def run_whole_model_on_mesh(
    target: str,
    module,
    *,
    in_fmt: str = "f32",
    weight_fmt: str | None = None,
    leaf_values: dict | None = None,
    operand_dtype: str | None = None,
    accum_dtype: str | None = None,
    numeric_policy: dict | None = None,
    simulator: str | None = None,
    package: str | None = None,
    ref_target: str = "toy_npu",
    seed: int = 0,
    timeout: int = 900,
) -> dict:
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

    from merlin.compiler.dialects.lowering import execute, lower_module
    from merlin.development.targetgen.oracles import program_probes as mp
    from merlin.target.capabilities import compute_units as _cu
    from merlin.target.capabilities import routing as _routing
    from merlin.target.registry import identity as tr

    fn = next(op for op in module.walk() if op.name == "func.func")
    args = list(fn.body.blocks[0].args)

    def _shape(v):
        return [int(d) for d in v.type.get_shape()]

    rng = np.random.default_rng(seed)
    if leaf_values is None:
        # Small integer operands: an integer mesh (e.g. int8·int8->int32) reproduces the f32 engine
        # reference EXACTLY, so the whole-model gate is bit-exact rather than tolerance-bounded.
        arrays = {a: np.rint(rng.standard_normal(tuple(_shape(a))) * 3).clip(-8, 7).astype(np.float32) for a in args}
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
        ref_final = np.asarray(
            next(iter(execute(lower_module(module, target=ref_target), ref_inj)["outputs"].values()))
        )
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
        got = run_matmul_on_mesh(
            target,
            la.tolist(),
            ra.tolist(),
            operand_dtype=operand_dtype,
            accum_dtype=accum_dtype,
            numeric_policy=numeric_policy,
            simulator=simulator,
            package=package,
            timeout=timeout,
            observed=obs,
        )
        # extents read from the actual operands (a fused op's matmul sub-ops carry the sub-op shapes, not
        # the enclosing step's), so the per-layer log is honest for attention/geglu as well as plain matmuls.
        per_layer.append(
            {
                "index": step.index,
                "m": int(la.shape[0]),
                "k": int(la.shape[1]),
                "n": int(ra.shape[1]),
                "unit": step.unit,
                "oracle": "ok" if got is not None else "unavailable",
                "executed_on": obs.get("oracle"),
                "path": obs.get("path"),
            }
        )
        return got

    base = {
        "target": target,
        "ref_target": ref_target,
        "ref_kind": ref_kind,
        "n_steps": len(program.steps),
        "n_mesh": program.n_mesh(),
        "n_scalar": program.n_scalar(),
        "output_id": program.output,
        "per_layer": per_layer,
        "simulator_requested": (_requested_mesh_simulator(simulator) or os.environ.get("MERLIN_REQUIRED_RTL_ENGINE")),
    }

    def _executors() -> list:
        """The DISTINCT executors the mesh layers actually ran on (empty when no layer reached one)."""
        return sorted({e for e in (lay.get("executed_on") for lay in per_layer) if e})

    try:
        spliced = mp.run_whole_model_program(program, seed_leaves, mesh_exec=_mesh_exec)
    except mp.MeshLayerUnavailable as e:
        return {**base, "status": "oracle_unavailable", "reason": str(e), "mesh_executors": _executors()}

    spliced_final = spliced["outputs"][program.output]
    exact = bool(np.array_equal(spliced_final, ref_final))
    match = bool(np.allclose(spliced_final, ref_final, rtol=1e-4, atol=1e-4))
    return {
        **base,
        "status": "pass" if match else "fail",
        "exact": exact,
        "match": match,
        "mesh_executors": _executors(),
        "note": "single co-scheduled whole-model run: matmul layers executed on the real mesh oracle, "
        "scalar/vector lane inline, activations handed between lanes; gated vs the whole-model "
        "engine reference. Residual: host-driven multi-kernel, not yet one fused address-space "
        "image.",
    }


def run_int8_chain_on_mesh(
    target: str,
    A0: list,
    weights: list,
    *,
    acc_scale: float,
    operand_dtype: str = "i8",
    accum_dtype: str = "i32",
    numeric_policy: dict | None = None,
    simulator: str | None = None,
    package: str | None = None,
    timeout: int = 900,
) -> dict:
    """Run an int8 matmul CHAIN on the target mesh with the per-layer REQUANT HANDOFF — the shape of a real
    quantized model's linear stack. For each layer ``Y_l = requant_i8(A_l @ W_l)`` executes on the REAL
    oracle (``run_matmul_on_mesh`` with an ``acc_scale`` epilogue that commits the i32 accumulator back to
    i8), and ``Y_l`` becomes ``A_{l+1}`` — so the activation stays device-native i8 across the whole chain
    rather than round-tripping through a wider host dtype. The final tensor (and every layer) is gated
    bit-exact vs :func:`_int8_chain_reference` only when the selected policy
    explicitly declares the supported product precision, rounding and narrow.

    This closes the int8 inter-layer handoff that a single independent matmul does not exercise: a real
    model is a CHAIN, and each mesh layer's output must be requantized to the operand dtype to feed the
    next mesh layer. FAIL-CLOSED: a layer with no reachable oracle returns ``oracle_unavailable`` (never a
    fabricated activation). TARGET-AGNOSTIC: the requant + narrow output dtype are the target's own
    (``run_matmul_on_mesh``); this only threads activations layer to layer."""
    import os

    import numpy as np

    semantics = _int8_chain_policy(numeric_policy)
    if operand_dtype not in {"i8", "int8"} or accum_dtype not in {"i32", "int32"}:
        raise ValueError("int8 mesh-chain execution requires explicit i8 operands and i32 readout")
    if not weights:
        raise ValueError("int8 mesh-chain execution requires at least one layer")
    a = np.asarray(A0)  # device activation, threaded layer to layer
    r = np.asarray(A0)  # independent host reference, advanced in lockstep
    per_layer: list = []
    for i, w in enumerate(weights):
        wl = np.asarray(w)
        # Refuse an unproven internal partial sum before running a device layer.
        r = _int8_chain_step(r, wl, acc_scale, semantics)
        out = run_matmul_on_mesh(
            target,
            a.tolist(),
            wl.tolist(),
            operand_dtype=operand_dtype,
            accum_dtype=accum_dtype,
            numeric_policy=numeric_policy,
            simulator=simulator,
            package=package,
            epilogue=["acc_scale"],
            acc_scale=acc_scale,
            timeout=timeout,
        )
        if out is None:
            return {
                "target": target,
                "status": "oracle_unavailable",
                "failed_layer": i,
                "n_layers": len(weights),
                "per_layer": per_layer,
                "reason": f"mesh layer {i} ({a.shape[0]}x{a.shape[1]}x{wl.shape[1]}) has no reachable "
                f"oracle in this env",
            }
        observed = np.asarray(out)
        if (
            observed.shape != r.shape
            or observed.dtype.kind not in "iu"
            or np.any(observed < -128)
            or np.any(observed > 127)
        ):
            raise ValueError("int8 mesh-chain oracle returned an incompatible i8 readout")
        a = observed.astype(np.int64)
        per_layer.append(
            {"layer": i, "m": int(a.shape[0]), "n": int(wl.shape[1]), "matches_ref": bool(np.array_equal(a, r))}
        )

    exact = bool(all(p["matches_ref"] for p in per_layer))
    return {
        "target": target,
        "status": "pass" if exact else "fail",
        "exact": exact,
        "n_layers": len(weights),
        "acc_scale": acc_scale,
        "per_layer": per_layer,
        "final_shape": [int(d) for d in a.shape],
        "simulator": (_requested_mesh_simulator(simulator) or os.environ.get("MERLIN_REQUIRED_RTL_ENGINE")),
        "note": "int8 matmul chain on the real mesh oracle with the per-layer acc_scale requant handoff "
        "(each layer's i8 output feeds the next mesh layer); gated bit-exact vs the host int8 "
        "chain reference at EVERY layer.",
    }


def _int8_chain_reference(A0, weights: list, acc_scale: float, *, numeric_policy: dict | None = None):
    """Retain the original selected callbacks at the runtime boundary."""
    from merlin.verification.references.mesh_arithmetic import _int8_chain_reference as implementation

    return implementation(
        A0, weights, acc_scale, numeric_policy=numeric_policy, _policy=_int8_chain_policy, _step=_int8_chain_step
    )


for _legacy_function in (
    run_whole_model_on_mesh,
    run_int8_chain_on_mesh,
    _int8_chain_reference,
):
    _legacy_function.__module__ = "merlin.runtime.dispatch.mesh_driver"
del _legacy_function
