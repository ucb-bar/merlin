"""Capsule evaluation: numerics."""

from __future__ import annotations

from merlin.development.targetgen import capsule_golden as CG
from merlin.development.targetgen.capsule_common import _cat, _flat
from merlin.development.targetgen.package_runtime import CertFailure
from merlin.evaluation.capsule.diagnostics import _absent_outputs as _absent_outputs
from merlin.evaluation.capsule.diagnostics import _readback_refusal as _readback_refusal
from merlin.runtime.backends import base as _backends_mod

DID_NOT_HALT_PLANE = "program_did_not_halt"


def _is_readback_refusal(exc: BaseException) -> bool:
    return _readback_refusal(exc) is not None


def _readback_refusal_detail(exc: BaseException) -> str:
    found = _readback_refusal(exc)
    return str(found) if found is not None else str(exc)


def _out_of_range_reason(sim: str, idx, size) -> str:
    """Actionable text for an on-chip out-of-bounds index, naming the index and the limit."""
    return (
        f"the emitted kernel indexed an on-chip resource OUT OF BOUNDS on {sim}: index {idx} with "
        f"size {size} (valid 0..{size - 1}). This is a kernel address/tiling bug, not a simulator "
        f"fault -- the simulator aborted because the access is illegal. Check every scratchpad and "
        f"accumulator address against the capacities your package compiles against (the ISA header's "
        f"ACC_ROWS and BANK_NUM x BANK_ROWS), including the last tile of a partial or odd-tail shape, "
        f"which is where a one-past-the-end row typically comes from."
    )


def _oracle_timeout_reason(sim: str, msg: str) -> str:
    """The tier ``reason`` for a wall-clock timeout without a correctness verdict.

    A timeout does not establish whether the delay came from the emitted program, the
    simulator, startup, or output transport. It remains distinct from an observed
    cycle-cap ``did not halt`` failure and keeps the ``TESTBENCH_TIMEOUT`` category.
    """
    return (
        f"{sim} exceeded its wall-clock budget; this tier is UNMEASURED and the cause is undetermined. "
        f"No correctness verdict was obtained for this tier: {msg[-200:]}"
    )


def _did_not_halt_reason(msg: str) -> str:
    """The tier ``reason`` for a program that ran to the cap without halting."""
    return f"did not halt (ran to the cycle cap): {msg[-240:]}"


def _did_not_halt_failure(msg: str) -> CertFailure:
    """The agent-facing failure for a program that ran to the cap without halting — one construction
    shared by every endpoint, so the verdict reads identically however the endpoint reported it."""
    return CertFailure(
        DID_NOT_HALT_PLANE,
        _cat("TIMEOUT"),
        f"{msg}; the emitted kernel never reached the ISA's halt/terminate "
        "instruction — emit it as the final instruction on every control path (see the "
        "program-termination contract). Numerics are never checked until the program "
        "halts.",
    )


def _readout_epilogue_capabilities(target: str):
    """Return a backend's readout declaration, or ``None`` when it has none.

    Some out-of-tree targets are executed exclusively through a bespoke oracle adapter and therefore
    have no same-named runtime backend.  Absence is the documented ``unavailable`` state for this
    advisory gate; allowing :func:`get_backend`'s ``KeyError`` to escape instead turns every capsule for
    such a target into a runner crash before its oracle can start.
    """
    try:
        backend = _backends_mod.get_backend(target)
    except KeyError:
        return None
    reader = getattr(backend, "readout_epilogue_capability", None)
    return reader() if callable(reader) else None


def _exact_match(a: dict, b: dict) -> bool:
    # NO OUTPUTS IS NOT A MATCH. `all(...)` over an empty mapping is vacuously True, so two empty
    # output sets used to compare EXACTLY EQUAL -- a kernel that stored nothing, graded against a
    # reference that recorded nothing, read as bit-exact. Same shape as the empty-cohort verdict and
    # the vacuous readiness GO: the absence of evidence presenting itself as evidence. Fail closed.
    if not a or not b:
        return False
    if set(a) != set(b):
        return False
    return all(_flat(a[k]) == _flat(b[k]) for k in a)


def _match_by_policy(a: dict, b: dict, policy: dict | None) -> bool:
    """Output-equality per the capsule's numeric_policy — exact for integer policies (the systolic
    default), tolerance for float. This replaces the per-runner hardcoded exact-vs-atol fork: an i8
    capsule matches exactly (identical to _exact_match), an fp capsule matches within its declared
    atol/rtol (default atol 1e-3)."""
    compare = (policy or {}).get("compare", "exact_int")
    if compare == "per_output":
        return CG.match_per_output(a, b, policy)
    if compare in ("exact_int", "exact"):
        return _exact_match(a, b)
    if set(a) != set(b):
        return False
    atol = float((policy or {}).get("atol", 1e-3))
    rtol = float((policy or {}).get("rtol", 0.0))
    for k in a:
        fa, fb = _flat(a[k]), _flat(b[k])
        if len(fa) != len(fb):
            return False
        for x, y in zip(fa, fb):
            if abs(float(x) - float(y)) > atol + rtol * abs(float(y)):
                return False
    return True


def _bind_positional_output_name(cb: dict, capsule: dict) -> None:
    """Rename the sole explicitly positional output, preserving pointer order.

    This is the existing name adapter, not a source/semantic qualification.
    Multi-output bindings retain their independent protocols; actual command
    interpretation, trace, ABI and numerical checks still run.
    """
    if cb.get("operand_naming") != "positional" and cb.get("interface") != "linalg_positional":
        return
    target = ((capsule.get("operation") or {}).get("attributes") or {}).get("out")
    tensors = cb.get("tensors") or {}
    outputs = [name for name, spec in tensors.items() if spec.get("role") == "output"]
    if not target or len(outputs) != 1 or outputs[0] == target:
        return
    previous = outputs[0]
    if target in tensors:
        raise CertFailure(
            "positional_output_binding",
            _cat("PROTOCOL_VIOLATION"),
            "positional output name collides with another emitted tensor",
        )
    pointer = cb.get("compiler_pointer_abi")
    if pointer is not None:
        arguments = pointer.get("arguments") if isinstance(pointer, dict) else None
        if (
            not isinstance(pointer, dict)
            or pointer.get("version") != 1
            or not isinstance(arguments, list)
            or arguments.count(previous) != 1
            or any(not isinstance(name, str) or name not in tensors for name in arguments)
            or len(set(arguments)) != len(arguments)
        ):
            raise CertFailure(
                "positional_output_binding",
                _cat("PROTOCOL_VIOLATION"),
                "positional output requires a complete consistent compiler pointer declaration",
            )
    # Preserve tensor iteration order: the logical harness may derive pointer
    # order from the emitted roster. Reinsert-at-end would change that ABI.
    cb["tensors"] = {target if name == previous else name: spec for name, spec in tensors.items()}
    for field in ("outputs", "arg_order"):
        if field in cb:
            cb[field] = [target if name == previous else name for name in cb[field]]
    if pointer is not None:
        pointer["arguments"] = [target if name == previous else name for name in pointer["arguments"]]
    for command in cb.get("commands", []):
        for key, value in (command.get("operands") or {}).items():
            if value == previous:
                command["operands"][key] = target


def _absent_output_detail(nrep: dict, sim_name: str, expected: dict, observed: dict) -> str | None:
    """A precise failure detail when the kernel dropped a declared output, else None. Turns the baffling
    "1024 mismatches, 0 error" into "you never wrote Y1" — the exact class of silent no-op the generic
    hint tells the agent to hunt for. Target-agnostic; no answer key crosses this boundary."""
    absent = _absent_outputs(nrep)
    if not absent:
        return None
    n_decl = len(expected)
    n_written = sum(1 for k in expected if k in (observed or {}))
    return (
        f"on {sim_name}, your emitted artifact never wrote output(s) {', '.join(absent)} — "
        f"declared in the interface (a commit op) but ABSENT from the observed DRAM readback "
        f"(your kernel produced {n_written} of {n_decl} declared outputs). This is a DROPPED or "
        f"mis-addressed store, NOT a value error (max_abs_error is 0 because those bytes were never "
        f"produced). Decode your OWN emitted artifact and confirm every declared output gets a store "
        f"to its base address."
    )


def _encoding_divergence_hint(
    trace_check_res: dict | None,
    independent_float: bool,
    cb: dict | None = None,
    capsule: dict | None = None,
    trace: dict | None = None,
) -> str:
    """A mandatory hardware/oracle tier failed while the cheap tiers already passed (numeric fails raise
    earlier), so the defect is in the emitted hardware artifact, NOT the command buffer. Return a
    target-agnostic localization hint so the failure is never an opaque 'oracle != golden' — plus the first
    concrete artifact finding if the decoder produced any (a RoCC trace). For a float/oracle-only target
    (no cheap trace decode) the generic hint still fires. This is what makes the np12r class localizable on
    EVERY endpoint: it names WHERE to look (the artifact encoding), not just THAT hardware disagreed.

    When the decoded trace + command buffer are available, an ADVISORY, FAIL-CLOSED cross-check
    (:mod:`divergence_localizer`) is run first: it diffs the decoded config fields against the agent's OWN
    command buffer + capsule spec and, if it can derive a concrete divergence, names the FIRST op+field
    (never a golden value, never a fix). The oracle stays the arbiter; a field it cannot anchor is simply
    not flagged, so the generic hint always still fires underneath."""
    concrete = ""
    if trace is not None:
        try:
            from merlin.development.targetgen import divergence_localizer as _DL

            concrete = _DL.format_finding(_DL.localize(cb, capsule, trace))
        except Exception:  # noqa: BLE001 — the localizer is advisory; never let it break the failure path
            concrete = ""
    findings = (trace_check_res or {}).get("violations") or []
    if independent_float:
        hint = (
            " Decode your OWN emitted artifact (the disassembler / instruction_trace.json) and verify "
            "every op fires with the operands you intend — a config/compute op that silently no-ops "
            "produces wrong output only on hardware, never in a cheap check."
        )
    else:
        hint = (
            " The command-buffer tiers (numeric + trace) PASSED, so the divergence is in your "
            "emit_target_artifact hardware encoding — some field the command buffer cannot carry (a "
            "config scale, an accumulate/dataflow bit, a readout dtype, a DRAM address). Decode your OWN "
            "emitted artifact (the disassembler / instruction_trace.json) and check each op's operands "
            "against your intent."
        )
    if findings:
        hint += f" Your own artifact check also flagged: {findings[0]}"
    # The concrete cross-check (when derivable) is the most actionable line — lead with it.
    return (concrete + hint) if concrete else hint


def _unexercised_note(unexercised: list[str], exercised: dict) -> dict:
    """``{tier: why it did not run}`` for the declared tiers a whole-model grade did not exercise.

    Reported as a bare LIST, this read as "we skipped three of the four tiers this capsule declares",
    which is indistinguishable from a grade that cut corners. The actual reason is structural: a model
    capsule is graded by the whole-model compile+verify path against its golden, plus the mesh oracle --
    it never enters the per-op tier ladder, and the cheap tiers of that ladder (emit/structure,
    command-buffer reference equality, functional sim of a command buffer) have no whole-model analogue
    to run. A tier that is inapplicable and a tier that was skipped deserve different words.
    """
    ran = set(exercised)
    return {
        t: (
            "no whole-model analogue: this tier grades a per-op command buffer, which a model "
            "capsule does not produce — the model is graded end to end against its golden"
        )
        for t in unexercised
        if t not in ran
    }


def _numeric_when_not_accelerated(st, gate, verify: dict, cos: float, engine: str, measured_on: str) -> dict:
    """The numeric verdict for a model capsule that will NOT be reported as a pass.

    The whole-model gate runs BEFORE the acceleration checks, so by the time one of those rejects the
    capsule the arithmetic has already been measured -- and was being overwritten with
    ``not_compared``, which reads as "we do not know" when in fact we do. That erased the one thing a
    reader most needs from a failing capstone: whether the compiler got the model RIGHT and merely ran it
    in the wrong place, or got it wrong as well. Those call for completely different work.

    ``measured_on`` is carried explicitly so the number can never be mistaken for an accelerator result;
    the capsule verdict stays ``fail`` in every caller."""
    return {
        "status": ("pass" if (st == "verified" and gate) else "fail"),
        "engine": engine,
        "gate": verify or None,
        "cos": (cos if cos else None),
        "measured_on": measured_on,
        "note": (
            "arithmetic only -- this capsule is NOT a pass; see the failure block for what "
            "disqualified it. A correct number computed on the wrong lane is still not an "
            "accelerator result."
        ),
        "model_status": st,
    }


def _numeric_not_compared(engine: str, measured_on: str, verify: dict | None, detail: str) -> dict:
    """The numeric block for a verdict where NOTHING WAS COMPARED against the accelerator.

    The single home for this status, deliberately: it is the honest answer when no declared tier ran,
    and the wrong one everywhere a number does exist -- a measured mismatch reported as `not_compared`
    hides a real defect behind an absence. Every path that DID measure reports its verdict through
    :func:`_numeric_when_not_accelerated` instead."""
    return {
        "status": "not_compared",
        "engine": engine,
        "measured_on": measured_on,
        "gate": verify or None,
        "detail": detail,
    }


def _merge_match_policy(force: dict | None, capsule: dict | None) -> dict | None:
    """Combine a target's ``force_match_policy`` with the capsule's declared ``numeric_policy`` so neither's
    tolerance is silently discarded. The compare MODE comes from the force policy (a float/SIMT target
    grades with tolerance even for an integer-policy capsule); ``atol``/``rtol`` take the LOOSER (max) of
    the two present values — a capsule's bf16-appropriate tolerance is honoured, and the tight global
    default still applies where the capsule declares none. Returns the other policy unchanged when one side
    is absent."""
    if not force:
        return capsule
    if not capsule:
        return force
    if "per_output" in (force.get("compare"), capsule.get("compare")):
        if force != capsule:
            raise ValueError("a force policy cannot replace per-output numerical boundaries")
        return capsule
    merged = dict(force)
    for k in ("atol", "rtol"):
        vals = [float(p[k]) for p in (force, capsule) if p.get(k) is not None]
        if vals:
            merged[k] = max(vals)
    return merged
