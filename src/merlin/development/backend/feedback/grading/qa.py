"""Redacted QA gate for the capsule_bench_v0 raw_baseline pilot.

Grades a candidate ``submission/`` against the PUBLIC pilot capsules (A0/A2/A4/B0) through the
real ladder (L0 reference==simulate, spike, verilator + trace_check), then emits ONLY a redacted
verdict the agent is allowed to see:

    {all_pass, n_passed, n_capsules, integrity_status,
     per_capsule: [{capsule, status, numeric_status, mismatch_count, trace_status,
                    trace_violations:[class-name strings], tiers:{L*:status}, failure_plane,
                    failure_category, highest_tier}],
     first_failure_planes}

It DELIBERATELY omits every answer-bearing value: golden outputs, reference/oracle outputs,
numeric diffs (max_abs_diff / first_mismatch), command buffers, lowered MLIR. The full grading
work tree (which contains numeric_report.yaml etc.) is written under an OPERATOR-ONLY runs_root
that the agent never sees; only the scrubbed verdict crosses back.

The agent uses this as a pass/fail QA signal to iterate against — never as an answer key.

Usage:
  qa_check.py --submission <dir> --out <verdict.json> [--labels public,dev]
              [--runs-root <operator-only tmp>] [--no-oracle] [--timeout 900]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

from merlin.common.artifacts import cache_dir  # noqa: E402 — purgeable work trees
from merlin.development.backend.context import (
    InvocationContext,
    add_context_arguments,
    readback_kwargs,
    resolve_context,
)
from merlin.development.backend.feedback.certification import tier_policy as certification_gates
from merlin.development.backend.feedback.verdicts import (
    summarize as qualification_summary,
)
from merlin.development.targetgen.grading import model_checks as CG  # noqa: E402
from merlin.development.targetgen.grading import submission as _merlin_submission_1
from merlin.development.targetgen.oracles import tier_policy as _TI  # noqa: E402
from merlin.evaluation.capsule.evidence import certificates as capsule_adapters
from merlin.evaluation.capsule.evidence import certificates as capsule_availability

# ---- qa ----------------------------------------------------------------------------------------

_SAFE_NUMERIC = {"status", "policy", "mismatch_count"}


def _original_cohort_complete(score: dict) -> bool:
    """A measured denominator cannot discard an original required obligation."""
    from merlin.development.targetgen.grading.boundary import execution_denominator

    admission = score.get("cohort_admission")
    if not isinstance(admission, dict):
        return False
    admitted = admission.get("n_admitted_capsules")
    if type(admitted) is not int or admitted <= 0 or score.get("n_capsules") != admitted:
        return False
    if type(score.get("n_capsules")) is not int or type(score.get("n_passed")) is not int:
        return False
    if score["n_passed"] != admitted or execution_denominator(score)[1]:
        return False
    for field in (
        "n_not_measured",
        "n_cert_not_measured",
        "n_infrastructure_fault",
        "n_gated_deferred",
        "n_screened_only",
        "n_budget_exhausted",
        "n_incomplete",
        "n_not_gradeable_no_oracle",
        "n_not_graded_ineligible",
    ):
        if type(score.get(field)) is not int or score[field] != 0:
            return False
    rows = score.get("per_capsule")
    if type(rows) is not list or len(rows) != admitted:
        return False
    names = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("status") != "pass"
            or type(row.get("capsule")) is not str
            or not row["capsule"]
        ):
            return False
        names.append(row["capsule"])
    return len(set(names)) == admitted and admission.get("admitted_name_set_sha256") == CG._name_set_sha256(names)


_RC_KEYS = (
    "rc",
    "returncode",
    "return_code",
    "exitcode",
    "exit_code",
    "status",
    # A POSITION is not a value. "first divergence at index=2" tells the agent WHERE its
    # output diverged; the expected/observed values stay scrubbed. Withholding the index
    # left the one actionable bit of a numeric failure unreadable -- a target whose only
    # reachable failure read "250 wrong somewhere" for 11 rounds and never moved.
    "index",
    "at index",
    "first_divergence_index",
)


def _is_ascii_letter(ch: str) -> bool:
    """True for a-zA-Z only. ``str.isalpha`` would also accept digits' unicode cousins and letters from
    scripts that cannot appear in an MLIR token, which is a wider door than this needs."""
    return len(ch) == 1 and (("a" <= ch <= "z") or ("A" <= ch <= "Z"))


_POSITION_WORDS = frozenset({"line", "lines", "col", "column"})

_HARNESS_COUNT_WORDS = frozenset(
    {
        "tile",
        "tiles",
        "tile(s)",
        "lane",
        "lanes",
        "lane(s)",
        "region",
        "regions",
        "region(s)",
        "passed",
        "failed",
        "unavailable",
        "unsynthesizable",
        "screened",
        "certified",
        "declined",
    }
)


def _following_words(text: str, i: int, count: int = 3) -> list[str]:
    """The next *count* whitespace-delimited words from *i*, lowercased, trailing punctuation removed.

    Structural (split, never matched), and bounded so a long dump cannot make this quadratic. A closing
    paren is stripped only when the token does not open one, so ``tile(s)`` survives while
    ``unsynthesizable)`` loses its bracket."""
    words: list[str] = []
    for tok in text[i : i + 200].split()[:count]:
        tok = tok.strip(".,;:").lower()
        if tok.endswith(")") and "(" not in tok:
            tok = tok[:-1]
        words.append(tok)
    return words


def _counts_harness_state(text: str, i: int) -> bool:
    """Whether the digit run ending at *i* counts something the harness observed about itself.

    Skips over the ``of`` and the second number in ``12 of 12 tile(s)``, so both runs of that phrase are
    judged by the noun they share."""
    words = _following_words(text, i)
    j = 0
    while j < len(words) and (words[j] == "of" or words[j].isdigit()):
        j += 1
    return j < len(words) and words[j] in _HARNESS_COUNT_WORDS


def _preceding_word(emitted: str) -> str:
    """The alphabetic word immediately before the trailing whitespace of *emitted*, lowercased.

    ``'File "x.py", line '`` -> ``'line'``. Empty when the run is not preceded by a bare word, which is
    the common case for a value (``expected 42``, ``cos 0.9997`` -> ``expected`` / ``cos``, neither of
    which is a position word, so both still scrub)."""
    j = len(emitted)
    while j and emitted[j - 1].isspace():
        j -= 1
    k = j
    while k and emitted[k - 1].isalpha():
        k -= 1
    return emitted[k:j].lower()


def _is_path_like(emitted: str) -> bool:
    """Does the token just before a ':' look like a file path? Trailing identifier-ish run containing a
    '.' or '/' -- i.e. ``input.interface.mlir`` or ``mlir_oot/<target>_opt.py``, never a bare number."""
    j = len(emitted)
    while j and not emitted[j - 1].isspace() and emitted[j - 1] not in "\"'(),":
        j -= 1
    tok = emitted[j:]
    return ("." in tok or "/" in tok) and any(_is_ascii_letter(c) for c in tok)


def _scrub_numbers(text: str) -> str:
    """Collapse numeric VALUES to '#', while leaving numbers that carry STRUCTURE intact.

    A golden value is a bare number: ``expected 42``, ``cos 0.9997``, ``[1, 2, 3]``. A shape, a dtype, a
    capsule name and a tier label are not values — they are tokens that happen to contain digits, and the
    agent already holds every one of them (shapes and dtypes come from ``capsule.yaml``, which the bundle
    grants; the capsule name rides UNREDACTED in the sibling ``capsule`` field of the same record). So the
    two rules are:

      * a digit run touching an ASCII letter on either side is structure — keep it;
      * a digit run directly after an allowlisted ``<key>=`` is process metadata — keep it;
      * anything else is a candidate value — collapse it to '#'.

    MEASURED (gemmini arm-4, 2026-08-19): the blanket scrub handed Nemotron this parse error —

        %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<#x#xi#>

    The shape and the element type ARE the diagnostic for a compiler task, and both were destroyed to
    protect values that were never in the string. Under the rule above the same error now reads
    ``tensor<16x16xi8>`` while ``expected 42, actual 17`` still scrubs to ``expected #, actual #``.

    Structural, not pattern-matched (repo convention: a too-narrow regex silently drops valid input).
    Walks the string once, deciding each digit run from the characters immediately around it.
    """
    out: list[str] = []
    at_source_location = False  # the previous kept run was a `:line`, so `:col` may follow
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        # A digit run starts here (possibly signed). Decide whether it is an exempt return code by
        # looking BACKWARDS at what was just emitted: an allowlisted key followed by '='.
        start = i
        if ch == "-" and i + 1 < n and text[i + 1].isdigit():
            i += 1
        if text[i].isdigit():
            while i < n and text[i].isdigit():
                i += 1
            if i < n and text[i] == "." and i + 1 < n and text[i + 1].isdigit():
                i += 1
                while i < n and text[i].isdigit():
                    i += 1
            emitted = "".join(out)
            before = text[start - 1] if start else ""
            after = text[i] if i < n else ""
            keep = False
            if emitted.endswith("="):
                key = emitted[:-1]
                # take the trailing identifier-ish run before '='
                j = len(key)
                while j and (key[j - 1].isalnum() or key[j - 1] == "_"):
                    j -= 1
                keep = key[j:].lower() in _RC_KEYS
            if not keep:
                # Structure, not a value: the run is part of a token that also contains letters --
                # ``i8``, ``bf16``, ``16x16``, ``A0_config_smoke``, ``L2``, ``vlen256``. A golden value
                # never touches a letter; it sits alone between separators.
                keep = _is_ascii_letter(before) or _is_ascii_letter(after)
            if not keep and before == ":":
                # A source location: ``input.interface.mlir:12:5``. The line and column of the agent's
                # OWN input cannot echo a golden value, and without them a parse error names a file but
                # not a place. The line is kept when the ':' follows a path-like token; the column is
                # kept because it follows a line we just kept.
                keep = at_source_location or _is_path_like(emitted[:-1])
            if not keep and before == " ":
                # The OTHER spelling of a source location, the one a Python traceback uses:
                #   File ".../<target>_opt.py", line 412
                # MEASURED on a live re-run: every traceback the agent received still arrived as
                # ``line #`` after the rest of this scrub was fixed, so it was told its own compiler
                # raised but never where. A line number in a traceback is a position in the AGENT'S OWN
                # source -- it cannot carry a golden value any more than the column above can.
                keep = _preceding_word(emitted) in _POSITION_WORDS
            if not keep:
                # A count of what the HARNESS saw, not a value from the answer. See _HARNESS_COUNT_WORDS.
                keep = _counts_harness_state(text, i)
            at_source_location = bool(keep and before == ":")
            out.append(text[start:i] if keep else "#")
            continue
        if ch != ":":
            at_source_location = False
        out.append(ch)
        i += 1
    return "".join(out)


DETAIL_LIMIT = 800


def _redact_detail(detail: str | None) -> str | None:
    if not detail:
        return None
    # Scrub concrete numbers (could echo expected/actual values) — the anti-answer-leak guard. The length
    # cap only bounds a runaway oracle dump; it must NOT chop the structured localization hint
    # (_encoding_divergence_hint), whose ACTIONABLE tail — "decode your OWN emitted artifact via the
    # disassembler / instruction_trace.json and check each op's operands" — runs past ~240 chars. Cutting it
    # left the agent told WHERE (the encoding) but not the self-inspection METHOD. 800 fits the full hint
    # while still bounding a pathological dump.
    return _scrub_numbers(detail)[:DETAIL_LIMIT]


def _placement_coverage(result: dict) -> dict | None:
    """The whole-model placement census, redacted to what the agent may act on.

    REDACTION-SAFE by construction: every field is about the SUBMISSION's own emitted program --
    its kernel symbols, the lane each call took, and the refusal clause the compiler's own selector
    wrote. Nothing here is corpus data, a golden, or an expected output. Symbols ride because they
    are the only actionable half: "2 calls went to the host" is a number, while naming the two says
    which lowering to fix. The same class of self-describing text as `declined`.

    Absent (None) for an operator capsule, which has one kernel and no dispatch ledger -- an empty
    census on such a row would read as "nothing fell back" rather than "there was nothing to look
    at".
    """
    from merlin.development.backend.feedback.grading.capsule_feedback import _candidate_native_feedback

    if _candidate_native_feedback(result) is not None:
        # The legacy census describes a different runner-owned program, even
        # when that program's numerical check passed. Do not present it as the
        # submitted whole-program compiler's placement coverage.
        return None
    census = (result.get("model_execution_check") or {}).get("lowering_coverage")
    if not isinstance(census, dict) or not census.get("operations"):
        return None
    return {
        key: census.get(key)
        for key in (
            "operations",
            "on_accelerator",
            "on_host",
            "coverage",
            "host_reasons",
            "unjustified_host_operations",
            "by_family",
        )
    }


def _abandoned_on_budget(reason) -> bool:
    """Did this tier run out of TIME rather than answer wrongly?

    Substring membership over a small vocabulary, matching what the two producers actually write: a
    subprocess timeout ("Command '[...]' timed out after 900 seconds") and the broker's own budget
    wording. No regex -- this repo gates `import re` in library code, and a too-narrow pattern silently
    reclassifies a real abandonment as a defect, which is the failure this whole field exists to fix.
    """
    if not isinstance(reason, str):
        return False
    low = reason.lower()
    return any(marker in low for marker in ("timed out", "time budget", "budget exhausted", "out of budget"))


def _emitted_cost(capsule_result: Path) -> dict | None:
    """What the SUBMISSION's own emitted program costs to execute — the pre-oracle signal the agent
    never saw.

    MEASURED on the g3arm gemmini batch: the six capsules that failed the cert tier were the six
    heaviest DRAM movers in the corpus, the top two at 20,592 and 18,624 movement operations against a
    median of 10 across the other 84. `SY_geometry_squareish_gemm` moved 540,672 bytes in 18,624
    operations -- 29 bytes each, where one 16x16 int8 tile is 256 -- so its lowering moves data
    per-element instead of per-tile. The elaborated-RTL engine then spent its whole 900 s budget
    simulating that traffic and was killed, and the agent's verdict said only
    `tiers: {L3: fail}` with `failure_plane: null` and `failure_detail: null`. The same capsule
    certifies in 0.022 s when lowered tile-wise, so this was never simulator cost: it was a lowering
    defect the harness had already measured, written to disk beside the result, and discarded.

    REDACTION-SAFE by construction. Every field is either a statistic of the agent's OWN emitted
    program (movement count, scratchpad rows touched, whether it closes with a fence) or a hardware
    CAPACITY the arm is already granted through the ISA facts and capability manifest. No corpus
    values, no goldens, no expected outputs. `bytes_per_movement` is the interpretable ratio and is
    derived from two numbers already on the row.
    """
    peaks = {}
    lrep = capsule_result.parent / "generated" / "liveness_report.json"
    if lrep.is_file():
        try:
            peaks = (json.loads(lrep.read_text()) or {}).get("resource_peaks") or {}
        except Exception:  # noqa: BLE001 -- an advisory screen must never break verdict production
            peaks = {}
    if not isinstance(peaks, dict) or not peaks:
        return None
    keep = (
        "dram_movements",
        "dram_unmapped",
        "dram_unknown_provenance",
        "scratchpad_rows_touched",
        "scratchpad_rows_capacity",
        "accumulator_max_row",
        "accumulator_rows_capacity",
        "closes_with_fence",
    )
    out = {k: peaks[k] for k in keep if isinstance(peaks.get(k), (int, float, bool))}
    if not out:
        return None
    # DELIBERATELY NOT a bytes-per-movement ratio. `movement_volume` is declared by the compiler's
    # command buffer (`basis: compiler_command_buffer`) while `dram_movements` is counted from the
    # decoded instruction trace, so dividing one by the other crosses two measurement bases: it
    # produced 0.3 "bytes per movement" for GC7_conv2d_pad_i8, which is not a physical quantity. The
    # movement COUNT is single-basis and already separates the populations -- 18,624 and 20,592 for the
    # two capsules that exhausted the cert budget against a median of 10 -- so it is reported alone,
    # with its basis named.
    out["movements_basis"] = "decoded_instruction_trace"
    return out or None


LIVENESS_SEVERITIES: tuple[str, ...] = ("fault", "stall", "unknown", "warn", "info", "ok")

REFUSING_SEVERITIES: frozenset[str] = frozenset({"fault", "stall", "unknown"})

_LIVENESS_RULES: frozenset[str] = frozenset(
    {
        "scratchpad-address-uninterpretable",
        "scratchpad-capacity",
        "scratchpad-overflow",
        "accumulator-capacity",
        "accumulator-overflow",
        "dram-provenance",
        "dram-provenance-unknown",
        "dram-unmapped",
        "dram-window-unknown",
        "visibility-no-drain",
        "funct-legality",
        "untranscodable-op",
        "vlen-match",
        "medany-span",
        "host-assist",
    }
)


def _liveness_screen(capsule_result: Path) -> dict | None:
    """WOULD THE SUBMISSION'S OWN EMITTED PROGRAM STALL OR FAULT ON SILICON — the advisory verdict the
    harness already computed and then dropped on the floor.

    `capsule_runner` writes ``generated/liveness_report.json`` beside every result: the L2.5 screen's
    per-finding severities plus an aggregate verdict. MEASURED across 3446 reports on disk: 313 ``stall``
    verdicts and 229 ``scratchpad-overflow`` findings, none of which ever reached the agent, so a capsule
    whose lowering walked past the scratchpad depth got an opaque cert failure instead of the rule name
    that had already identified it.

    REDACTION-SAFE by construction, on the same discipline as :func:`_emitted_cost`. Three fields, each
    typed and allowlisted:

    * ``verdict`` — one word from :data:`LIVENESS_SEVERITIES`, or nothing;
    * ``rules`` — ``{severity: {rule slug: count}}``, slugs from :data:`_LIVENESS_RULES` (anything else
      counted under ``other``) and counts as ints. NO message, NO ``where``, NO ``evidence``: those carry
      addresses and values copied out of the program, which is exactly what must not ride here;
    * ``dram_window_bytes`` / ``dram_window_provenance`` — the DERIVED hardware window. The provenance
      sentence is NOT echoed from the report (that file sits in the agent's tree and is writable there):
      it is re-derived from the target's own memory map and included only when its size AGREES with the
      reported one, so the string on the row is provably this repo's derivation and not file content.

    Nothing else survives. An advisory screen must never break verdict production, so every failure —
    absent file, unparseable JSON, hostile shape — returns ``None``.
    """
    lrep = capsule_result.parent / "generated" / "liveness_report.json"
    if not lrep.is_file():
        return None
    try:
        rep = json.loads(lrep.read_text()) or {}
    except Exception:  # noqa: BLE001 -- an advisory screen must never break verdict production
        return None
    if not isinstance(rep, dict):
        return None

    out: dict = {}
    verdict = rep.get("verdict")
    if isinstance(verdict, str) and verdict in LIVENESS_SEVERITIES:
        out["verdict"] = verdict

    rules: dict[str, dict[str, int]] = {}
    for f in rep.get("findings") or []:
        if not isinstance(f, dict):
            continue
        sev = f.get("severity")
        if not (isinstance(sev, str) and sev in LIVENESS_SEVERITIES):
            continue
        rule = f.get("rule")
        slug = rule if (isinstance(rule, str) and rule in _LIVENESS_RULES) else "other"
        bucket = rules.setdefault(sev, {})
        bucket[slug] = bucket.get(slug, 0) + 1
    if rules:
        out["rules"] = rules

    peaks = rep.get("resource_peaks")
    win = peaks.get("dram_window_bytes") if isinstance(peaks, dict) else None
    if isinstance(win, int) and not isinstance(win, bool) and win > 0:
        out["dram_window_bytes"] = win
        # The target name comes out of the report file too, and it is used to LOOK UP a descriptor
        # path. Constrain it to a bare identifier so a crafted `"../.."` cannot aim the derivation at
        # some other file in the tree; a name that fails the check simply yields no provenance.
        tgt = rep.get("target")
        if isinstance(tgt, str) and tgt and all(c.isalnum() or c in "_-" for c in tgt):
            try:
                from merlin.target.isa.machine_model import dram_window_for

                _base, size, why = dram_window_for(tgt)
                if isinstance(why, str) and size == win:
                    out["dram_window_provenance"] = why
            except Exception:  # noqa: BLE001 -- provenance is a nicety; its absence is not a failure
                pass
    return out or None


def _execution_digest_from_result(capsule_result: Path) -> str | None:
    """Best-effort bridge to the shared promotion identity; absence keeps legacy invalidation.

    Callers (this reader and ``agent_selfcheck``) put the value on every verdict row, so this MUST stay
    importable: dropping it turned the self-check into an AttributeError that printed no verdict JSON at
    all, and every consumer -- readiness section G included -- reported ``n=None`` as if the oracle had
    not run."""
    from merlin.development.backend.feedback.grading.capsule_feedback import _candidate_native_feedback

    try:
        result = json.loads(capsule_result.read_text(encoding="utf-8"))
        if _candidate_native_feedback(result) is not None:
            # The legacy identity owner gathers generated/ executables, whereas
            # this candidate's ELF lives in candidate_native/. Until that owner
            # binds the native ELF and selected hardware, use conservative
            # submission invalidation rather than borrow another program's cert.
            return None
        from merlin.development.backend.feedback.certification.promotion_identity_promotion import (
            execution_digest,
        )

        return execution_digest(capsule_result)
    except Exception:  # noqa: BLE001 -- verdict production must not fail for an optional cache key
        return None


def _loop_target_sim_via(context: InvocationContext) -> tuple[str, str]:
    """Resolve (target, sim_via) for the loop-gate oracle from THIS experiment's descriptor (honors the
    MERLIN_TARGET_EXPERIMENT override baked into C.EXP). Falls back to C.TARGET + no bespoke sim if the
    descriptor is absent, so a descriptor-less invocation still grades on the RTL-derived (arc) tier."""
    desc = context.descriptor
    if desc.is_file():
        from merlin.target.registry.descriptor_loading import load_target_experiment

        te = load_target_experiment(desc)
        return te.target, te.sim_via
    return context.target, ""


def run(
    submission: str,
    capsules_root: str,
    runs_root: Path,
    labels: set[str],
    no_oracle: bool,
    timeout: int,
    *,
    context: InvocationContext,
    contract: Path | None = None,
    additional_forbidden: tuple[str, ...] = (),
    defer_pending=None,
    defer_running=True,
    on_capsule_complete=None,
    max_workers: int = 0,
) -> dict:
    # Loop gate = L0+L1+trace + the target's full reachable oracle ladder, screened cheapest-first. The
    # adapters are resolved from the descriptor's target+sim_via via the shared factory, so a non-chipyard
    # target (arc/cyclotron) grades on its own RTL-derived tier with NO target-specific path.
    #
    # This used to keep the FASTEST tier only, reserving the cycle-accurate one for an end-of-run barrier,
    # because per-round verilator across three PARALLEL arms was a CPU storm. Fail-fast changes that sum:
    # the expensive tier is only ever paid for on a capsule the cheap tier could not refute, which early in
    # a run is almost none of them.
    from merlin.development.backend.feedback.grading.capsule_feedback import _per_capsule_from_results

    _target, _sim_via = _loop_target_sim_via(context)
    # The loop tier is chosen from the tiers THESE capsules declare, so the per-round gate always rides a
    # tier the capsule asked for. Without this the loop picks the endpoint's fastest tier, which for an
    # endpoint that exposes an additive cheap tier below its declared gold tier means grading against a
    # tier the capsule never declared.
    from merlin.development.targetgen.corpus.materialization import declared_oracle_tiers as _declared

    _decl = _declared(capsules_root)
    # THE PER-ROUND GATE CERTIFIES; it does not merely screen. Hand the grade the target's whole
    # reachable ladder, not just its cheapest tier. The ladder runs CHEAPEST-MEASURED-FIRST and
    # fail-fasts on the first mandatory tier that refutes a capsule, so this is close to free on a
    # failing submission -- a capsule the cheap tier refutes never reaches the expensive one. What it
    # buys is the other direction, the one that was broken: a capsule that PASSES the screen now goes on
    # to the cert tier in the SAME round, and counts as passed only if the cert tier passes too.
    #
    # Grading the round on the screen alone is unsound in exactly one direction, and it was measured:
    # one submission passed the cheap functional tier on 20 of 20 capsules while the RTL tier passed 1.
    # A screen may eliminate; it may never certify (see merlin.development.targetgen.oracles.tier_policy).
    from merlin.development.backend.campaign import control as joint_control

    _loop_adapters = (
        {}
        if no_oracle
        else (
            capsule_availability.qa_loop_adapters(_target, _sim_via, declared_tiers={"L2"}, **readback_kwargs(context))
            if joint_control.selected()
            else capsule_availability.qa_checkpoint_adapters(_target, _sim_via, **readback_kwargs(context))
        )
    )
    if joint_control.selected() and set(_loop_adapters) - {"L0", "L1", "L2"}:
        raise ValueError("joint background grade selected a physical tier")
    # Refuse ONLY when the endpoint exposes tiers but none of them is declared — substituting one there is
    # the defect. An endpoint that reaches nothing at all is an honestly ABSENT oracle: leave the adapter
    # set empty and let each capsule report its missing tier as unavailable, exactly as before.
    if not no_oracle and not _loop_adapters:
        _reach = sorted(capsule_adapters.oracle_adapters(_target, _sim_via))
        if _reach:
            raise SystemExit(
                f"capsule corpus {capsules_root} declares required oracle tiers {sorted(_decl)} but "
                f"target {_target!r} reaches {_reach} — no declared tier is reachable, so this loop "
                f"cannot grade. Refusing to substitute a tier the capsules never declared."
            )
    if joint_control.selected() is not None:
        previous_defer = defer_pending

        def joint_defer():
            if joint_control.remaining() <= 0:
                return "joint campaign deadline reached"
            # A development grade still in flight at the freeze would otherwise run on inside the
            # final window that the frozen-compiler coverage needs.
            if joint_control.remaining(drain=True) <= 0:
                return "joint campaign freeze/drain boundary reached"
            return previous_defer() if previous_defer is not None else None

        defer_pending = joint_defer

    def completed(result):
        name = result.get("capsule")
        projected = _per_capsule_from_results(runs_root, only_capsule=name).get(name)
        if projected is not None:
            fields = (
                "status",
                "numeric_status",
                "mismatch_count",
                "tiers",
                "trace_status",
                "trace_requirement_status",
                "trace_violations",
                "failure_plane",
                "failure_category",
                "failure_detail",
                "execution_digest",
                "tier_cycles",
                "tier_engines",
                "tier_reuse",
                "expected_refusal",
                "candidate_native_verification",
            )
            on_capsule_complete({"capsule": name, **{k: projected.get(k) for k in fields}})

    from merlin.development.backend.feedback.campaign import correctness as joint_test_scope

    selected_roots, test_scope = joint_test_scope.select(capsules_root, submission=submission)
    score = _merlin_submission_1.grade(
        submission,
        capsules_root=selected_roots,
        runs_root=str(runs_root),
        model_snapshot_root=runs_root.resolve() / ".private_model_sources",
        labels=labels,
        contract=str(contract if contract is not None else context.repo / "src/merlin/resources/contracts"),
        oracle_adapters=_loop_adapters,
        timeout=timeout,
        target=_target,
        no_oracle=no_oracle,
        additional_forbidden=additional_forbidden,
        **({"defer_pending": defer_pending, "defer_running": defer_running} if defer_pending is not None else {}),
        **({"on_capsule_complete": completed} if on_capsule_complete is not None else {}),
        **({"max_workers": max_workers} if max_workers else {}),
    )
    redacted = _per_capsule_from_results(runs_root)

    per_capsule = []
    for pc in score.get("per_capsule", []):
        name = pc["capsule"]
        rich = redacted.get(name, {})
        per_capsule.append(
            {
                "capsule": name,
                "label": pc.get("label"),
                "status": pc.get("status"),
                "numeric_status": rich.get("numeric_status", pc.get("numeric")),
                "mismatch_count": rich.get("mismatch_count"),
                "trace_status": rich.get("trace_status", pc.get("trace")),
                "trace_requirement_status": rich.get("trace_requirement_status"),
                "trace_unavailable_requirements": rich.get("trace_unavailable_requirements", []),
                "trace_violations": rich.get("trace_violations", []),
                "tiers": rich.get("tiers", {})
                if rich.get("candidate_native_verification") is not None
                else pc.get("tiers", {}),
                "tier_cycles": rich.get("tier_cycles", {}),
                "tier_reuse": rich.get("tier_reuse"),
                "failure_plane": rich.get("failure_plane"),
                "failure_category": rich.get("failure_category"),
                "failure_tier": rich.get("failure_tier"),
                "failure_detail": rich.get("failure_detail"),
                "tier_reasons": rich.get("tier_reasons") or {},
                "execution_digest": rich.get("execution_digest"),
            }
        )
        if not rich.get("failure_plane"):
            # A deferred test case writes no result file, so its reason lives only in the grade row.
            per_capsule[-1].update(certification_gates.gated_row_fields(pc, _redact_detail))
        if rich.get("placement_coverage"):
            per_capsule[-1]["placement_coverage"] = rich["placement_coverage"]
        if rich.get("candidate_native_verification") is not None:
            per_capsule[-1]["candidate_native_verification"] = rich["candidate_native_verification"]
        # WHAT THE PROGRAM COST, beside whether it was right. `tier_cycles` already rides this row and
        # nothing interprets it: a number with no floor and no ceiling is not a cost, and an agent
        # reading "L3: 2,400,000 cycles" has no way to tell a good program from a hundredfold-slow
        # one. The plane carries the array's own issue floor and, where the capsule declares one, its
        # ceiling. REPORT phase: it never moves `all_pass`, so this round is scored by the same rule
        # as the last. Redaction-safe -- cycle counts of the agent's OWN program, plus a floor derived
        # from hardware geometry the arm already has through the capability manifest.
        if rich.get("candidate_native_verification") is None and isinstance(pc.get("cost_plane"), dict):
            per_capsule[-1]["cost_plane"] = {
                key: pc["cost_plane"].get(key)
                for key in (
                    "status",
                    "reason",
                    "axis",
                    "phase",
                    "timing_tier",
                    "measured_cycles",
                    "floor_cycles",
                    "floor_basis",
                    "ceiling_cycles",
                    "array_efficiency",
                    "movement_floor_cycles",
                    "movement_basis",
                    "limiter",
                    "blocking",
                )
            }
        if rich.get("declined"):
            per_capsule[-1]["declined"] = rich["declined"]
        if rich.get("expected_refusal"):
            per_capsule[-1]["expected_refusal"] = rich["expected_refusal"]

    n_caps = score.get("n_capsules", 0)
    n_pass = score.get("n_passed", 0)
    if no_oracle:
        # HONEST structure-only smoke: the numeric oracle was NOT run, so a capsule that clears the
        # structural tiers reads back as `not_gradeable_no_oracle`, never `pass` (no numeric pass is ever
        # claimed). The STOP signal is therefore "every capsule is structurally clean" (L0/L1/trace), NOT
        # the numeric `all_pass` — which can never be true here and would make the agent thrash to timeout
        # chasing an unreachable numeric pass (the observed 0/11 failure). `n_passed` reports the
        # structural-clean count so the console + round summary read coherently; the note makes the scope
        # explicit and the per-capsule numeric_status stays withheld.
        n_structural = score.get(
            "n_structural_pass", sum(1 for pc in per_capsule if pc["status"] in ("pass", "not_gradeable_no_oracle"))
        )
        all_pass = bool(n_caps > 0 and score.get("structural_pass", n_structural == n_caps))
        verdict = {
            "qa_gate": "capsule_bench_v0_pilot",
            "gradeable": False,
            "labels_graded": score.get("labels_graded"),
            "all_pass": all_pass,
            "stop_condition": "structural_tiers_pass",
            "n_passed": n_structural,
            "n_structural_pass": n_structural,
            "n_capsules": n_caps,
            "integrity_status": score.get("integrity_status"),
            "highest_tier": score.get("highest_tier"),
            "first_failure_planes": score.get("first_failure_planes", {}),
            "per_capsule": per_capsule,
            "note": (
                "NOT GRADEABLE this run: the numeric/trace oracle is unavailable, so ONLY the L0/L1 "
                "structural tiers are graded (a structure-only smoke). Do NOT chase a numeric pass — "
                "capsules that clear the structural tiers show status `not_gradeable_no_oracle`, "
                "which is the target here, NOT a fixable failure. Fix only real structural failures "
                "(schema/language/L0/L1/trace planes); never hardcode outputs."
            ),
        }
    else:
        verdict = {
            "qa_gate": "capsule_bench_v0_pilot",
            "gradeable": True,
            "labels_graded": score.get("labels_graded"),
            "all_pass": bool(_original_cohort_complete(score) and score.get("structural_evidence_all_pass") is True),
            "original_cohort_complete": _original_cohort_complete(score),
            "structural_evidence_complete": score.get("structural_evidence_all_pass") is True,
            "structural_evidence_scope": score.get("structural_evidence_scope"),
            "trace_requirement_summary": score.get("trace_requirement_summary"),
            "n_passed": n_pass,
            "n_execution_passed": qualification_summary(per_capsule)["execution_passed"],
            "qualification_summary": qualification_summary(per_capsule),
            "n_capsules": n_caps,
            "integrity_status": score.get("integrity_status"),
            "highest_tier": score.get("highest_tier"),
            "first_failure_planes": score.get("first_failure_planes", {}),
            "per_capsule": per_capsule,
            "n_declined": score.get("n_declined", 0),
            # WHAT LEFT THE DENOMINATOR, BY NAME. `n_capsules` counts only the rows actually MEASURED,
            # so a deferred/screened/budget-exhausted row silently shrinks it -- and it shrinks exactly
            # where the arm did worst, which makes two arms' ratios non-comparable. Measured on the
            # g3arm batch: arms 1-3 cleared the whole-model gate so M2_microvit_gemmini and
            # SY_micro_model RAN, FAILED and stayed in a denominator of 97, while arm 4 fell below the
            # gate and those same two rows left its denominator, giving 95 -- so 75/95 read against
            # 93/97 understated the gap. Carrying the names here lets the aggregator intersect cohorts
            # instead of re-deriving them from `per_capsule`.
            "n_not_measured": score.get("n_not_measured", 0),
            "not_measured": score.get("not_measured"),
            "not_measured_status": score.get("not_measured_status"),
            "note": (
                "This is a QA pass/fail signal only. It contains NO reference output values — there is no answer key. "
                "Numerical/native pass counts are screens until applicable structural evidence is complete. "
                "Unavailable original operand/schedule evidence is an infrastructure requirement, "
                "not a numerical defect. "
                "Fix failures by capsule + failure_plane + trace_violations; never hardcode outputs."
                + (
                    f" {score['n_declined']} capsule(s) were DECLINED by your backend: it emitted no "
                    f"program for them. Those are shapes/ops you do not lower — a COVERAGE gap, not a "
                    f"numeric bug. See `declined` on those rows."
                    if score.get("n_declined")
                    else ""
                )
            ),
        }
    # AN INFRASTRUCTURE FAULT IS NOT A ROUND RESULT. `grade` already refuses to call such a run
    # gradeable, but this verdict is what gets ARCHIVED as qa_history/verdict_round_NN.json, fed to the
    # next round as the agent's own failure history by round_brief, and read back by every trajectory
    # and status reporter. Measured: a round-0 grade lost its staged cohort to a sibling
    # materialization's collector and the archive recorded `n_passed: 2, n_capsules: 33,
    # gradeable: true, first_failure_planes: {schema: 31}` for a submission that scored 33/34 -- then
    # handed that to round 1 as 31 structural defects of its own making. Carry the fault through, and
    # never let `gradeable`/`all_pass` claim otherwise.
    if score.get("infrastructure_fault"):
        _inf = score["infrastructure_fault"]
        verdict["gradeable"] = False
        verdict["all_pass"] = False
        verdict["infrastructure_fault"] = {
            "n": _inf.get("n"),
            "of": _inf.get("of"),
            "plane": _inf.get("plane"),
            "capsules": _inf.get("capsules"),
            "detail": _redact_detail(str(_inf.get("detail", ""))),
        }
        verdict["note"] = (
            "THIS ROUND WAS NOT GRADED. " + str(_inf.get("detail", "")) + " The pass/fail counts below "
            "are NOT a measurement of the submission and must not be quoted or compared against another "
            "round. Nothing here is a defect for the agent to fix."
        )

    # top-level integrity failure (K0/K1 fail-closed)
    if "failure" in score:
        verdict["package_failure"] = {
            "plane": score["failure"]["plane"],
            "category": score["failure"]["category"],
            "detail": _redact_detail(score["failure"]["detail"]),
        }
    return joint_test_scope.annotate(verdict, test_scope)


def main(argv: list[str] | None = None, *, context=None) -> int:
    ap = argparse.ArgumentParser()
    add_context_arguments(ap)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--out", required=True, help="path to write the redacted verdict JSON")
    ap.add_argument("--capsules-root", default=None)
    ap.add_argument("--labels", default="public,dev")
    ap.add_argument(
        "--runs-root", default=None, help="OPERATOR-ONLY grading work tree (must NOT be inside the agent workspace)"
    )
    ap.add_argument("--no-oracle", action="store_true", help="L0 + trace only (skip spike/verilator)")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args(argv)
    context = resolve_context(a, ap, context)
    if a.capsules_root is None:
        if context.harness is None:
            ap.error("installed QA requires --capsules-root")
        a.capsules_root = str(context.experiment / "scripts" / "pilot_capsules")

    runs_root = Path(a.runs_root) if a.runs_root else (cache_dir("capsule_bench_qa") / "scratch")
    runs_root.mkdir(parents=True, exist_ok=True)
    labels = set(a.labels.split(","))
    verdict = run(a.submission, a.capsules_root, runs_root, labels, a.no_oracle, a.timeout, context=context)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    # Stamp when this verdict was produced. Without it the only clock a reader has is the
    # file mtime, which does not survive a copy between trees.
    verdict.setdefault("graded_at", _dt.datetime.now(_dt.UTC).isoformat())
    Path(a.out).write_text(json.dumps(verdict, indent=2))
    print(
        f"[qa_check] all_pass={verdict['all_pass']} "
        f"{verdict['n_passed']}/{verdict['n_capsules']} integrity={verdict['integrity_status']}"
    )
    for pc in verdict["per_capsule"]:
        extra = "" if pc["status"] == "pass" else f"  <- plane={pc['failure_plane']} viol={pc['trace_violations']}"
        print(f"    [{pc['status']:10s}] {pc['capsule']}{extra}")
    return 0 if verdict["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
