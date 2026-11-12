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
    from merlin.development.targetgen.package_runtime import build_package, integrity_scan, load_package

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
        from merlin.development.targetgen.oracle_policy import select_chipyard_engine as chipyard_l3_selection

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

    from merlin.development.targetgen import capsule_common as CC
    from merlin.development.targetgen import package_runtime as oot_runner

    from ..common.paths import runs_root

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
    from merlin.development.targetgen import mesh_program_run

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

    from merlin.development.targetgen import capsule_common as CC
    from merlin.development.targetgen import mesh_program_run as MP
    from merlin.development.targetgen.capsule_common import make_run_paths
    from merlin.development.targetgen.oracle_policy import _SIM_ORACLES
    from merlin.development.targetgen.oracle_policy import selected_sim_via as _bespoke_sim_via

    from ..common.paths import runs_root

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
    _legacy_function.__module__ = "merlin.runtime.mesh_backend"
del _legacy_function
