"""Run the independent oracle, compiler and author-tool admission before spend."""

from __future__ import annotations

import sys

import yaml

from merlin.evaluation.capsule import availability as capsule_availability


def check(
    a,
    context,
    *,
    _te,
    run_dir,
    ws,
    bundle,
    _resolved_tool_ids,
    _public_root,
    _bundle_snapshot_record,
    _bundle_manifest_sha256,
):
    """Return the original refusal code, or None after every selected check."""

    _te_pf = _te()
    if context.readback_policy is not None and not a.no_oracle:
        from merlin.targetgen.contract.readback_policy import MEMORY_TRANSPORTS

        from .context import readback_kwargs

        if context.readback_policy in MEMORY_TRANSPORTS:
            # Construct the original full ladder before authoring; a global
            # memory override must be supported by every selected engine.
            capsule_availability.qa_checkpoint_adapters(_te_pf.target, _te_pf.sim_via, **readback_kwargs(context))
    _ora_ok, _ora_why = capsule_availability.oracle_available(_te_pf.target, _te_pf.sim_via)
    (run_dir / "oracle_preflight.yaml").write_text(
        yaml.safe_dump(
            {
                "target": _te_pf.target,
                "sim_via": _te_pf.sim_via,
                "oracle_available": _ora_ok,
                "reason": _ora_why,
                "no_oracle": bool(a.no_oracle),
                "verdict": "GO" if (_ora_ok or a.no_oracle) else "NO_GO",
            },
            sort_keys=False,
        )
    )
    if not _ora_ok and not a.no_oracle:
        print(
            f"NO_GO: {_ora_why} — refusing to launch a gradeable run with no numeric oracle "
            f"(zero tokens spent). Re-run with --no-oracle for an explicit structure-only smoke, or "
            f"set MERLIN_MLC_DIR + build the arc model / model venv.",
            file=sys.stderr,
        )
        return 4
    if not _ora_ok:  # a.no_oracle is set — honest structure-only smoke (see qa_grade/no-oracle path)
        print(
            f"[preflight] oracle unavailable ({_ora_why}); proceeding as an EXPLICIT --no-oracle "
            f"structure-only smoke (NOT gradeable — structural tiers only)."
        )
    else:
        print(f"[preflight] oracle GO: {_ora_why}")

    # Presence of a simulator is not proof that its compiler can build a kernel.
    # Exercise the selected compile-based oracle toolchain before any agent spend;
    # non-compile oracles return an explicit n/a from the shared capability check.
    if not a.no_oracle:
        from merlin.targetgen import runtime_build as _RBpf

        _compiler_ok, _compiler_why = _RBpf.compiler_smoke(_te_pf.sim_via)
        (run_dir / "compiler_smoke.yaml").write_text(
            yaml.safe_dump(
                {
                    "target": _te_pf.target,
                    "sim_via": _te_pf.sim_via,
                    "compiler_ok": _compiler_ok,
                    "reason": _compiler_why,
                },
                sort_keys=False,
            )
        )
        if not _compiler_ok:
            print(
                f"NO_GO: oracle compile toolchain failed: {_compiler_why}. Refusing to launch (zero tokens spent).",
                file=sys.stderr,
            )
            return 4
        print(f"[preflight] compiler smoke: {_compiler_why}")

    # Exercise the assisted tool surface using this run's verified frozen bundle.
    # The receipt is host-private, and a failed probe refuses before authoring.
    from . import tooling_readiness as _TRpf

    if a.qualify_submission:
        _tooling = {
            "status": "not_required",
            "checks": [],
            "scope": "unpaid frozen-submission qualification starts no author or authoring-tool clients; "
            "formal compiler, oracle, isolation and input-integrity gates remain required",
        }
        (run_dir / "tooling_readiness.yaml").write_text(yaml.safe_dump(_tooling, sort_keys=False))
    else:
        _tooling = _TRpf.run(
            context,
            _te_pf,
            ws,
            bundle,
            tuple(_resolved_tool_ids),
            _public_root,
            run_dir,
            _bundle_snapshot_record,
            _bundle_manifest_sha256,
            without_tools=tuple(a.without_tool),
        )
    if _tooling["status"] == "no_go":
        failed = [f"{check['name']}: {check['detail']}" for check in _tooling["checks"] if not check["ok"]]
        print(
            f"NO_GO: selected authoring tools are not ready: {'; '.join(failed)}. "
            "Refusing to launch (zero tokens spent).",
            file=sys.stderr,
        )
        return 4
    print(f"[preflight] tooling readiness: {_tooling['status']}")

    # Oracle availability and code production are separate admission evidence. The
    # selected backend owns its production probe; resource paths never select it.
    # An absent optional probe returns None (not run), not a successful qualification.
    # Explicit selection failures and failed probes refuse before authoring spends tokens.
    if not a.no_oracle:
        _cg_backend = _te_pf.preflight_codegen_backend
        if _cg_backend is None:
            _cg_ok, _cg_why = capsule_availability.codegen_smoke(_te_pf.target)
        else:
            _cg_ok, _cg_why = capsule_availability.codegen_smoke(_te_pf.target, backend_target=_cg_backend)
        (run_dir / "codegen_smoke.yaml").write_text(
            yaml.safe_dump(
                {
                    "target": _te_pf.target,
                    "backend_target": _cg_backend,
                    "codegen_ok": _cg_ok,
                    "reason": _cg_why,
                },
                sort_keys=False,
            )
        )
        # `is False`, not falsy: None means the smoke DID NOT RUN (n/a for this emit path), which is
        # neither a pass nor a NO_GO. Gating on falsiness would refuse every target the smoke does not
        # cover; recording None as True is what let a doomed run look verified.
        if _cg_ok is False:
            print(
                f"NO_GO: codegen smoke failed — the target's emit path cannot produce a runnable "
                f"kernel: {_cg_why}. Refusing to launch (zero tokens spent).",
                file=sys.stderr,
            )
            return 4
        print(f"[preflight] codegen smoke: {_cg_why}")
    return None
