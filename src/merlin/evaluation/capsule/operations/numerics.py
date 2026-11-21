"""Bounded exception, simulator and output diagnostics for trusted capsule grading."""

from __future__ import annotations

import traceback as _traceback
from collections.abc import Sequence
from pathlib import Path

from merlin.development.targetgen.corpus.capsule_io import _cat, _flat
from merlin.development.targetgen.grading import golden as CG
from merlin.development.targetgen.packaging.loading import CertFailure
from merlin.evaluation.capsule.evidence.source_custody import _capture_compiler_source, _finish_compiler_source
from merlin.evaluation.capsule.operations import setup_timing as _host_timing
from merlin.runtime.backends import registry as _backends_mod

# ---- diagnostics -------------------------------------------------------------------------------

# ---- diagnostics -------------------------------------------------------------------------------


def _candidate_exception_diagnostic(exc: Exception, *, stage: str) -> dict:
    """Keep a bounded, source-free stack when a model run fails before a receipt.

    The public result already exposes the exception class and message. Frame
    names and line numbers make failures such as RecursionError actionable
    without copying source lines, arguments, or private artifact paths.
    """
    frames = [
        {"module": Path(frame.f_code.co_filename).name, "function": frame.f_code.co_name, "line": line}
        for frame, line in _traceback.walk_tb(exc.__traceback__)
    ]
    return {
        "type": type(exc).__name__,
        "detail": str(exc)[:2000],
        "stage": stage,
        "stack_depth": len(frames),
        "stack": frames if len(frames) <= 20 else frames[:4] + frames[-16:],
    }

    #: Marks the middle a clipped diagnostic dropped; readers comparing details split on it.


ELISION = "[…]"


def _clip(msg: str, budget: int) -> str:
    """Clip a long exception to ``budget`` chars KEEPING BOTH ENDS -- the head names the failure, the tail
    usually carries the operand/shape specifics.

    A bare ``msg[-budget:]`` throws the head away, and the head is where the diagnosis lives. Measured:
    the muon operand-binding error is one long sentence whose useful half is first ("could not derive
    harness operands: <which of three cases>") and whose second half is fixed advisory prose. Tail-only
    clipping rendered five radiance capsules as `cyclotron crash: es (outputs: ['Y0'])` -- the real cause
    truncated to the last two letters of "shapes" -- and they read as unexplained infra failures across
    many runs while the message that would have named them was being discarded every time.
    """
    msg = str(msg)
    if len(msg) <= budget:
        return msg
    head = budget * 2 // 3  # bias to the head: that is the diagnosis
    tail = budget - head - 5
    return f"{msg[:head]} {ELISION} {msg[-tail:]}" if tail > 0 else msg[:budget]


def _readback_refusal(exc: BaseException) -> BaseException | None:
    """The ``ReadbackIntegrityError`` in ``exc``'s cause/context chain, if any.

    The refusal is raised deep in a target's oracle and re-wrapped on the way out (the muon oracle
    raises ``MuonError(str(exc)) from exc``; ``oot_runner`` wraps again). Only the TYPE survives that
    intact, so the chain is walked structurally -- matching the message would break the first time
    the wording changed, which is exactly the class of defect this whole path exists to stop.
    """
    try:
        pass
    except ImportError:  # pragma: no cover - module always present in this tree
        return None
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, ReadbackIntegrityError):
            return cur
        cur = cur.__cause__ or cur.__context__
    return None


def _engine_token(oracle_kind: str | None) -> str | None:
    """The elaborated-RTL ENGINE named inside an oracle kind, or None.

    An adapter reports a kind like ``rtl_gsim`` / ``rtl_verilator`` / ``rtl_gsim_<something>``; the
    engine is one token inside it. Read STRUCTURALLY (split on the separator, test membership against
    the policy's own declared engine vocabulary) rather than by pattern — a pattern that is slightly too
    narrow silently reports "no engine" for a conformant spelling, which here would mean a cert whose
    engine is unrecorded, i.e. the exact thing this field exists to prevent.
    """
    tokens = {t for part in str(oracle_kind or "").split("_") for t in part.split("-")}
    from merlin.target.simulators import engine_policy as _pol

    for engine in _pol.ENGINE_PRIORITY:
        if engine in tokens:
            return engine
    return None


def _record_console(paths, name: str, console) -> tuple[str | None, int | None]:
    """Write a sim console to the capsule's artifacts dir and return ``(filename, bytes)``.

    One writer for all three tier paths, which each open-coded the same two lines and none of which
    recorded the SIZE. Size is not decoration: a console is how an RTL verdict is read, and "the marker
    is absent" and "the console was truncated before the marker" are the same bytes to a reader who
    cannot tell how much there was.
    """
    if console is None:
        return None, None
    if type(console) is bytes:
        fname = f"{name}_console.bin"
        (paths.artifacts_dir / fname).write_bytes(console)
        return fname, len(console)
    text = str(console)
    fname = f"{name}_console.log"
    (paths.artifacts_dir / fname).write_text(text, encoding="utf-8")
    return fname, len(text.encode("utf-8"))


def _oracle_kind(oracle_meta) -> str | None:
    """A filesystem-safe name for the engine an adapter reports, or ``None`` when it reports none.

    Adapters return either a rich ``{"kind": ..., "derived_from_rtl": ...}`` dict or a bare provenance
    string. Both carry the engine's identity; the contract's static ``tier_sim`` map does not, because a
    backend may substitute a different RTL-derived engine at runtime. ``None`` means "the oracle did not
    say", and the caller keeps the declared name -- never a guess.
    """
    kind = oracle_meta.get("kind") if isinstance(oracle_meta, dict) else oracle_meta
    if not isinstance(kind, str) or not kind.strip():
        return None
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in kind.strip())
    return safe or None


def _absent_outputs(nrep: dict) -> list[str]:
    """Outputs the kernel DECLARED (present in the interface/golden) but never wrote — or wrote at the
    wrong length — extracted from :func:`capsule_golden.compare`'s ``per_output``. These structural
    failures are the ones that read as ``mismatch_count > 0`` while ``max_abs_error == 0`` and
    ``first_mismatch is None`` (there is no value to diff — the bytes were never produced), a
    self-contradictory signal unless the missing output is named. Reveals no golden VALUE: only the
    identity of a declared output, which the agent already holds in the interface it was handed."""
    po = nrep.get("per_output") or {}
    return [
        nm
        for nm, d in po.items()
        if isinstance(d, dict)
        and d.get("status") == "fail"
        and str(d.get("reason", "")).startswith(("missing", "length"))
    ]


def _unwritten_output_detail(nrep: dict, sim_name: str) -> str | None:
    """A precise detail when a declared output was READ BACK but holds one constant value, else None.

    Sibling of :func:`_absent_output_detail`, for the case that one misses: there the output key is
    absent from the readback; here it is present and uniformly the buffer's untouched fill, so the
    generic "does not compute the declared operation" is technically true and useless. Measured cost of
    not having this: twelve capsules reported ``functional_mismatch`` for six consecutive rounds with
    ``observed: 0.0`` and ``max_rel_error: 1.0``, while the mismatch COUNT was a function of the
    golden's zero distribution rather than of the kernel -- so the number could not move no matter what
    was emitted, and the agent spent six rounds tuning numerics against an unwritable signal.

    Reveals no golden value: only that the observed side is constant, which the agent can compute from
    its own readback."""
    names = nrep.get("outputs_never_written") or []
    if not names:
        return None
    po = nrep.get("per_output") or {}
    got = {n: (po.get(n) or {}).get("observed_constant") for n in names}
    shown = ", ".join(f"{n} (all {got[n]})" for n in names)
    return (
        f"on {sim_name}, output(s) {shown} came back as a SINGLE CONSTANT value while the expected "
        f"result varies — the store never landed, so what was compared is the buffer's untouched "
        f"fill, not a computed result. This is a WRITEBACK failure, not a numeric one: the mismatch "
        f"count here is set by the reference's own value distribution and will NOT move if you only "
        f"change arithmetic. Decode your emitted artifact and confirm the result is actually stored "
        f"to this output's base address (and that any DMA/commit is awaited before readback)."
    )


def _range_check_bounds(msg: str):
    """(index, size) from a C++ ``_M_range_check`` abort, or (None, None).

    libstdc++ spells it ``__n (which is 1024) >= this->size() (which is 1024)``. Parsed STRUCTURALLY --
    no regex in library code (build_tools/checks/check_no_regex.py) -- by splitting on the literal
    ``(which is `` marker, which is what libstdc++ emits. Returns None rather than guessing when the
    shape does not match, so a changed message degrades to the generic crash text instead of inventing
    a bound.
    """
    if "_M_range_check" not in msg and "out_of_range" not in msg:
        return None, None
    marker = "(which is "
    parts = msg.split(marker)
    if len(parts) < 3:
        return None, None
    vals = []
    for chunk in parts[1:3]:
        head = chunk.partition(")")[0].strip()
        if not head.isdigit():
            return None, None
        vals.append(int(head))
    return vals[0], vals[1]

    # ---- readback_integrity ------------------------------------------------------------------------
    # Structural integrity of a device readback, independent of any numeric tolerance.
    #
    # A transport that reconstructs a buffer out of device state can lose a whole
    # residue class of words -- every second 4-byte word, say -- while leaving every
    # other word bit-exact.  Compared value-by-value against a tolerance scaled to the
    # golden's magnitude, such a buffer can still PASS: measured on one 7232-element
    # capsule, 5172 zeroed elements produced only 988 reported mismatches, because the
    # remaining 4184 goldens were smaller than the absolute tolerance.  A mandatory
    # cert tier can therefore be granted to a buffer half of which was never read
    # back at all.
    #
    # This module refuses that shape STRUCTURALLY.  It needs no golden, no tolerance
    # and no knowledge of the device: it only observes that a periodic subset of the
    # word indices is exactly zero while the rest is not.  No arithmetic kernel
    # produces that; a half-width transport produces exactly that.
    #
    # The rule is deliberately fail-closed.  A buffer whose every word is zero does
    # NOT trip it (the complement carries no evidence, and a wholly zero buffer is
    # already visible to any value comparison); a buffer with one periodic hole and
    # live data elsewhere does.  A legitimately zero column that happens to align
    # with a stride is refused rather than certified -- a false refusal is a question
    # to answer, a false certification is a result nobody can retract.

    #: Word strides worth testing.  These are the transport granularities a
    #: reconstruction can halve or quarter (an 8-byte beat, a 16-byte quarter-line,
    #: a 32-byte line, a 64-byte line over 4-byte words), not a property of any
    #: particular device.


STRIDES: tuple[int, ...] = (2, 4, 8, 16)

#: Below these sizes the pattern is not evidence: a short buffer can hold an
#: all-zero residue class by arithmetic accident.
MIN_WORDS = 16
MIN_CLASS_WORDS = 8


class ReadbackIntegrityError(Exception):
    """A readback whose STRUCTURE disqualifies it, whatever its values say."""


def residue_class_defect(
    words: Sequence[int],
    *,
    strides: Sequence[int] = STRIDES,
    min_words: int = MIN_WORDS,
    min_class_words: int = MIN_CLASS_WORDS,
) -> str | None:
    """Return a named diagnostic if a whole residue class of words is exactly zero.

    ``words`` are the raw readback words as unsigned integers (the bit patterns,
    never floats -- ``-0.0`` and ``0.0`` are different bit patterns and only an
    exactly zero pattern counts).  Returns ``None`` when nothing is wrong.
    """
    total = len(words)
    if total < min_words:
        return None
    nonzero_total = sum(1 for w in words if w)
    if nonzero_total == 0:
        # Nothing survives anywhere: there is no complement to contrast with,
        # and a wholly empty buffer is not this defect.
        return None
    for stride in strides:
        if stride < 2 or total // stride < min_class_words:
            continue
        for residue in range(stride):
            klass = words[residue::stride]
            if len(klass) < min_class_words:
                continue
            if any(klass):
                continue
            complement_nonzero = nonzero_total  # klass is all zero, so all of it is outside
            if complement_nonzero == 0:
                continue
            return (
                "readback_residue_class_zeroed: every 4-byte word at index % "
                f"{stride} == {residue} is exactly zero ({len(klass)} of {len(klass)}), "
                f"while {complement_nonzero} of the remaining {total - len(klass)} words "
                "are non-zero. A periodic hole of this shape is a transport defect, "
                "not an arithmetic result; the readback is refused before any value "
                "is compared."
            )
    return None


def words_from_bytes(raw: bytes) -> list[int]:
    """Split a readback into little-endian 4-byte words, without interpreting them."""
    if len(raw) % 4:
        raise ReadbackIntegrityError(f"readback is {len(raw)} bytes, not a whole number of 4-byte words")
    return [int.from_bytes(raw[i : i + 4], "little") for i in range(0, len(raw), 4)]


def require_intact(raw: bytes, *, transport: str) -> None:
    """Raise :class:`ReadbackIntegrityError` if ``raw`` carries a structural defect."""
    defect = residue_class_defect(words_from_bytes(raw))
    if defect is not None:
        raise ReadbackIntegrityError(f"{transport}: {defect}")


# ---- numerics ----------------------------------------------------------------------------------
# Capsule evaluation: numerics.

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
            from merlin.development.targetgen.oracles import divergence as _DL

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


# ---- cpu ---------------------------------------------------------------------------------------
# Capsule evaluation: cpu.


def _grade_cpu_step(function, /, *args, **kwargs):
    """Run one synchronous grader calculation under the shared compute lease.

    Keep this scope around in-process work only. Compiler and oracle adapters
    launch supervised children that acquire their own slots, so the capsule
    runner must release the slot before invoking or waiting for either one.

    The wait for the lease is the ``grade_cpu.lease`` stage and the calculation
    its own ``grade_cpu.<name>`` stage. A calculation registered with
    :mod:`.grade_offload` runs in a worker process on the leased CPU: grading
    threads share one interpreter lock, so pure-Python engines on many capsules
    otherwise serialize on one core however many CPUs are leased.
    """
    from contextlib import ExitStack

    from merlin.common.process.cpu_pool import shared_compute_pool
    from merlin.development.targetgen.grading import grade_offload

    name = _grade_cpu_stages().get(id(function)) or getattr(function, "__name__", "step").lstrip("_")
    with ExitStack() as lease:
        with _host_timing.stage("grade_cpu.lease"):
            lease.enter_context(shared_compute_pool())
        with _host_timing.stage("grade_cpu." + name):
            return grade_offload.call(function, *args, **kwargs)


def _grade_cpu_stages() -> dict[int, str]:
    """Stage names of the grader's pure calculations, each registered to run out of process.

    Each takes and returns plain data and reads only its arguments and capsule files, which is what
    :func:`.grade_offload.register` requires. Anything not listed stays on the grading thread. Looked
    up per call, so a replaced function (a test's stub) is matched by its own identity, never an old one."""
    from merlin.analysis.programs.work_volume import command_buffer_evidence
    from merlin.common.digest import hash_tree
    from merlin.development.targetgen.grading import grade_offload
    from merlin.runtime.engine.reference import reference_outputs
    from merlin.runtime.engine.simulator import simulate

    named = {
        "golden": CG.golden,
        "l0_reference": reference_outputs,
        "l1_simulate": simulate,
        "numeric_compare": CG.compare,
        "declared_numeric_compare": CG.compare_declared_outputs,
        "policy_match": _match_by_policy,
        "tree_hash": hash_tree,
        "work_volume": command_buffer_evidence,
        # Filesystem-only: hash, copy and re-hash the compiler source into this capsule's evidence.
        "capture_compiler_source": _capture_compiler_source,
        "finish_compiler_source": _finish_compiler_source,
    }
    grade_offload.register(*named.values())
    return {id(f): name for name, f in named.items()}
