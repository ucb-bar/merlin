"""Capsule evaluation: model."""

from __future__ import annotations

from pathlib import Path

from merlin.evaluation.capsule.bundle import _model_runtime_bundle as _model_runtime_bundle
from merlin.evaluation.capsule.identity import _MODEL_NUMERIC_UNSET, CONTRACT_VERSION
from merlin.evaluation.capsule.model_assessment import assess_model_execution
from merlin.evaluation.capsule.model_inputs import (
    _resolve_model_host_lane,
    _resolve_model_numeric_policy,
    _verified_model_host_lane_snapshot,
)


def _grade_model_capsule_inline(
    capsule: dict,
    *,
    target: str | None = None,
    timeout: int,
    package_dir: str | Path | None = None,
    numeric_policy=_MODEL_NUMERIC_UNSET,
    numeric_policy_identity: dict | None = None,
) -> dict:
    """Grade a whole-model (kind == "model") capsule end to end via the target-aware whole-model flow
    (``compile_model``): route each op across the target's compute units (matmul/systolic -> the mesh, the
    rest -> the vector/scalar lane), compile the functional whole model (the scalar/RVV reference,
    numerically correct across every op), and gate its output vs the model's golden — attaching the per-op
    mesh-routing plan for the target. ``MERLIN_MODEL_GRADE_RUN`` selects ``host`` (default; x86 dispatch
    runtime) or ``spike``. Honestly reports ``incomplete`` when the whole-model toolchain (m2m venv /
    clang-23) is absent — never a silent pass."""
    import os
    from pathlib import Path as _P

    attrs = (capsule.get("operation") or {}).get("attributes") or {}
    model, dtype = attrs.get("model"), attrs.get("compile_dtype", "fp32")
    # Which lane executes the model. The default was the host dispatch runtime unconditionally, so a
    # capsule that DEMANDS acceleration was graded on a lane that cannot provide it and passed anyway.
    # Derive it instead: a capsule whose semantic block asserts must_accelerate runs on the mesh lane,
    # everything else keeps the host lane. The env var still overrides, for a deliberate diagnostic run.
    _sem = capsule.get("semantic") or {}
    # WHICH LANE THE MODEL RUNS ON. must_accelerate implies the mesh lane. So does an interop capsule
    # that REQUIRES on_mesh: it withholds must_accelerate on purpose (host work is the behaviour under
    # test), but running it on the host lane means the mesh never executes and its own lane requirement
    # becomes unverifiable -- which is exactly how the first such capsule passed while executing nothing
    # on the accelerator.
    _req_lanes = [str(x) for x in ((capsule.get("lanes") or {}).get("require") or [])]
    # THE TARGET'S OWN MESH IS THE ORACLE WHENEVER WE HAVE A TARGET. This used to key on the capsule's
    # must_accelerate / required lanes, which left a capsule declaring neither graded on the HOST even
    # when a target was named -- and merlin's x86 dispatch runtime is OUR compiler, so however well it
    # does it is evidence about us, never about the submission. Keying on the target subsumes both
    # declarations: a capsule demanding acceleration on a target with no mesh has nothing to run on
    # either way. The env var still overrides for a deliberate diagnostic run.
    run_where = os.environ.get("MERLIN_MODEL_GRADE_RUN") or ("mesh" if target else "host")
    _ = _req_lanes  # still read below by the lane report
    # MESH VERIFICATION IS THE CAPSULE'S DEMAND, NOT AN OPERATOR'S OPT-IN. It EXECUTES each mesh-routed
    # matmul as a single systolic tile on the target's real mesh oracle (compile_model mesh_verify),
    # which is the only evidence that the matmul layers run ON the mesh rather than that a routing plan
    # was produced.
    #
    # It used to be off unless an env var said otherwise, because the oracle build/run is heavy. The
    # cost is real; the default was still wrong. A capsule that declares `must_accelerate`, or that
    # names `on_mesh` in `lanes.require`, is asking for exactly this evidence -- and without it the tier
    # ladder below has nothing to record, so the capstone reports a verdict backed by the functional
    # lane alone. That is `capstone-graded-tiles-not-the-model` with the accelerator left out entirely.
    # Derive the default from what the capsule demands; the env var still overrides in BOTH directions,
    # for a deliberate diagnostic run. The wall-clock cost is bounded by MERLIN_MODEL_BUDGET_S.
    _demands_mesh = bool(_sem.get("must_accelerate")) or "on_mesh" in _req_lanes
    _mv_env = os.environ.get("MERLIN_MESH_VERIFY", "").strip().lower()
    mesh_verify = (_mv_env in ("1", "true", "yes", "on")) if _mv_env else _demands_mesh
    result: dict = {
        "capsule": capsule["name"],
        "kind": "model",
        "label": capsule.get("label"),
        "operation": {"op": "model", "model": model, "dtype": dtype, "run": run_where, "target": target},
        "contract_version": CONTRACT_VERSION,
    }
    if not model:
        result.update(
            status="incomplete",
            failure={
                "plane": "model",
                "category": "RUNNER_CRASH",
                "detail": "model capsule missing operation.attributes.model",
            },
        )
        return result

    # The target submission and the scalar/vector host lane are TWO compiler packages. The former is
    # threaded below as ``mesh_package``; the latter is frozen experiment infrastructure and MUST come
    # from this target's descriptor. Passing None used ``default_package`` (the currently certified RVV
    # champion), which could differ from the package granted read-only to the agent. The resulting
    # boundary/interop verdict then depended on a compiler the experiment neither declared nor exposed.
    _host_package = None
    _host_identity = None
    _host_experiment = None
    if target:
        try:
            if numeric_policy is _MODEL_NUMERIC_UNSET:
                numeric_policy, numeric_policy_identity = _resolve_model_numeric_policy(target)
            _host_experiment, _host_package, _host_identity = _resolve_model_host_lane(target, dtype)
            result["host_lane"] = _host_identity
        except Exception as e:  # noqa: BLE001 — an unpinned host compiler cannot produce a verdict
            result.update(
                status="incomplete",
                failure={
                    "plane": "model",
                    "category": "NOT_RUN_IS_NOT_PASS",
                    "detail": f"frozen host lane unavailable: {type(e).__name__}: {str(e)[:300]}",
                },
            )
            return result
    if numeric_policy is _MODEL_NUMERIC_UNSET:
        numeric_policy = None
    if numeric_policy_identity is not None:
        result["numeric_policy_input"] = numeric_policy_identity
    # The captured linalg (visible grounding) drives the per-op mesh routing when present. Read the name
    # the CAPSULE DECLARES -- every model capsule ships `linalg_mlir: capsule.interface.mlir`, while this
    # looked only for `capsule.linalg.mlir`, so it never found one. `linalg_mlir` was therefore always
    # None, `compile_model`'s `if target and linalg_mlir:` never fired, and the routing plan, the coverage
    # certificate AND the mesh verification were all skipped on every whole-model capsule ever graded --
    # silently, because a skipped block leaves no trace in the result.
    linalg_mlir = None
    linalg_path = None
    cdir = capsule.get("__dir__")
    if cdir:
        for _name in (capsule.get("linalg_mlir"), capsule.get("interface_mlir"), "capsule.linalg.mlir"):
            if not _name:
                continue
            lp = _P(cdir) / str(_name)
            if lp.is_file():
                linalg_path = lp
                linalg_mlir = lp.read_text(encoding="utf-8")
                break
    if target and cdir:
        # The declaration-side composition is derived independently from the capsule interface.  Later
        # the model check compares it with the runtime ledger, so M2's A->H->A seam cannot be credited
        # from a static routing plan and M3's must-accelerate claim cannot pass on synthesized tiles alone.
        from merlin.targetgen import boundary as _boundary

        result["boundary_expectation"] = _boundary.profile_capsule(cdir, target).to_dict()
    from merlin.xdsl_dialects.lowering.passes import MODEL_BOUNDARY_CAPSTONE

    _requires_boundary_plane = MODEL_BOUNDARY_CAPSTONE in (capsule.get("pass_requirements") or ())
    if _requires_boundary_plane and linalg_path is None:
        result.update(
            status="incomplete",
            failure={
                "plane": "pass_obligations",
                "category": "NOT_RUN_IS_NOT_PASS",
                "detail": "model capsule has no readable declared linalg/interface module; "
                "the production boundary passes cannot be exercised",
            },
        )
        return result
    try:
        # This is the production, in-process boundary plane: outline the REAL model, materialize and
        # verify its dispatch DAG, partition it, and attach the C ABI. It is deliberately separate from
        # the OOT package's four subprocess stages, which run below on each mesh tile. A pass-obligation
        # failure is structural and blocks the capsule before expensive oracle work.
        if _requires_boundary_plane:
            from merlin.frontends.linalg_mlir import parse_mlir_file
            from merlin.xdsl_dialects.lowering.passes import run_dialect_plane

            # WITH the target, so a dispatch is a whole compute group. Without it this plane took the
            # single-op baseline while `compile_cli` formed groups over the same model.
            _plane = run_dialect_plane(parse_mlir_file(linalg_path), n_harts=1, target=target)
            result["pass_obligations"] = _plane.stats

        from merlin.cli.compile_workload import compile_model

        # `dtype` is the capsule's compile_dtype (an RVV compile mode). The capsule ALSO declares the
        # exact datapath format in operation.attributes.dtype -- pass that for routing so a demand is
        # matched against the unit's declared format rather than against a compile-mode token.
        _attrs = (capsule.get("operation") or {}).get("attributes") or {}
        # THE PACKAGE UNDER TEST MUST REACH THE MESH ORACLE. `run_capsule` is handed the submission
        # being graded, but this function never took it and called `compile_model(package=None)`, so a
        # whole-model capsule was compiled and mesh-verified against the DEFAULT path -- not against the
        # submission. Every whole-model number that resulted (numeric verdict, layers-on-mesh accounting,
        # tile certification) was therefore a statement about the reference flow, not about the backend
        # the capsule was supposed to be judging. Measured: mesh verification reported
        # `n_tiles: 0, reason: "no default OOT backend package for target"` while a perfectly good
        # submission sat in the caller's hand.
        #
        # It goes to `mesh_package` (the OOT ACCELERATOR backend that certifies tiles), while `package`
        # is the independently frozen RVV host lane -- two things that a single name would conflate.
        _pkg = str(package_dir) if package_dir else None
        _host_package_arg = str(_host_package) if _host_package is not None else None
        # A model capsule is itself the frozen capture.  Resolving a same-named live recapture here
        # let mutable operator state choose the program/golden after launch (and made capsule-local
        # models impossible to run at all).  Materialize the exact runtime ABI from the declared,
        # digest-bound capsule assets and forbid compile_model's recapture resolver on this path.
        with _model_runtime_bundle(capsule, timeout=timeout) as (
            _capture_bundle,
            _bundle_provenance,
            _verify_bundle_unchanged,
        ):
            result["model_runtime_bundle"] = _bundle_provenance
            out = compile_model(
                model,
                dtype,
                target=target,
                run=run_where,
                verify=True,
                package=_host_package_arg,
                auto_capture=False,
                timeout=timeout,
                linalg_mlir=linalg_mlir,
                mesh_verify=mesh_verify,
                mesh_package=_pkg,
                routing_dtype=_attrs.get("dtype"),
                capture_bundle=_capture_bundle,
                numeric_policy=numeric_policy,
                transform_audit="exact",
            )
            _verify_bundle_unchanged()
    except SystemExit as e:  # toolchain/bundle unavailable — honest skip
        result.update(
            status="incomplete",
            failure={
                "plane": "model",
                "category": "NOT_RUN_IS_NOT_PASS",
                "detail": f"whole-model compile/run unavailable: {str(e)[:300]}",
            },
        )
        return result
    except Exception as e:  # noqa: BLE001
        result.update(
            status="incomplete",
            failure={
                "plane": "model",
                "category": "NOT_RUN_IS_NOT_PASS",
                "detail": f"whole-model grade error: {type(e).__name__}: {str(e)[:300]}",
            },
        )
        return result
    # Detect drift across the compile, not only before it. For a bwrap run this re-verifies the same
    # immutable snapshot aggregate and resolves the package inside its repo payload; it NEVER switches
    # to the live worktree. Direct/non-bwrap grading retains its live-tree pre/post check.
    if _host_experiment is not None and _host_identity is not None:
        try:
            _snapshot_repo_after, _snapshot_after = _verified_model_host_lane_snapshot()
            _snapshot_before = _host_identity.get("run_snapshot")
            if (_snapshot_before is None) != (_snapshot_after is None):
                raise ValueError("model host-lane run snapshot context changed during grading")
            if _snapshot_before is not None:
                if any(
                    _snapshot_after.get(key) != _snapshot_before.get(key)
                    for key in ("root", "content_sha256", "manifest_sha256")
                ):
                    result["host_lane_snapshot_after"] = _snapshot_after
                    raise ValueError("model host-lane run snapshot changed during grading")
            # RE-RESOLVE THE SAME LANE, not the default one. The check below compares package digests
            # before and after; resolving a different profile would compare two different packages and
            # report drift that never happened (or, worse, miss real drift by comparing the wrong pair)
            # the moment a target declares more than one precision.
            _, _host_after = _host_experiment.resolve_host_lane(root=_snapshot_repo_after, dtype=dtype)
        except Exception as e:  # noqa: BLE001
            result.update(
                status="incomplete",
                failure={
                    "plane": "model",
                    "category": "NOT_RUN_IS_NOT_PASS",
                    "detail": f"frozen host lane became unreadable during grading: {type(e).__name__}: {str(e)[:300]}",
                },
            )
            return result
        if _host_after["package_sha256"] != _host_identity["package_sha256"]:
            result.update(
                status="incomplete",
                host_lane_after=_host_after,
                failure={
                    "plane": "model",
                    "category": "NOT_RUN_IS_NOT_PASS",
                    "detail": "frozen host-lane package changed during grading "
                    f"({_host_identity['package_sha256'][:12]} -> "
                    f"{_host_after['package_sha256'][:12]}); refusing a "
                    "mixed-compiler verdict",
                },
            )
            return result
    return assess_model_execution(capsule, result, out, target=target, dtype=dtype, run_where=run_where)
