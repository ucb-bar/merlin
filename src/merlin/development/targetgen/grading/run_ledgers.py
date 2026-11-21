"""Per-round artifact fingerprints — tell an INERT edit apart from a WRONG one.

In an iterative agentic compile loop a graded number that does not move has **four** distinct causes
that all present identically:

1. the agent changed nothing that reaches the graded submission;
2. it changed the submission, but nothing changed in what the emitter *produced* for a capsule;
3. it changed the emitted artifact and the result is equally wrong;
4. it changed the artifact and the **oracle is insensitive** for that capsule.

Without evidence, distinguishing these is guesswork. A measured case: an arm-4 run held 12 capsules at
byte-identical mismatch counts for six consecutive rounds while the agent edited its emitter every round
and self-reported substantive fixes. Three separate mechanisms were proposed from reading intermediate
artifacts by hand, and all three were wrong. The information needed to answer it in one line -- did these
bytes change? -- was never recorded.

This module records it. It compares **bytes**, never content: it does not parse an artifact, does not know
an opcode, an ISA, a stage name or a target name, so it cannot be too narrow and cannot go stale when a
backend changes what it emits. A target that emits three files and one that emits seven both work with no
edit here. Anything unreadable is recorded as ``UNKNOWN`` and surfaced -- never silently skipped.

It is **diagnostic, not a gate**: it reports movement, it never passes or fails a run. A threshold here
would create exactly the incentive this apparatus exists to defeat (make the number move by deleting the
hard case). It is also written **out of band** -- into the run dir, which the sandbox denies -- because
telling the agent "your edit was inert" is *feedback*, and feedback defines an arm. Surfacing it in the
agent's verdict would silently turn arm N into arm N+1 and invalidate cross-run comparison.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

from merlin.common.provenance_claims import UNKNOWN, file_digest

# ---- stage_ledger ------------------------------------------------------------------------------

# What a per-capsule or per-submission fingerprint set can say about one round versus the previous one.
NEW = "new"  # first round this artifact existed
CHANGED = "changed"  # bytes differ from the previous round
UNCHANGED = "unchanged"  # bytes identical to the previous round
ABSENT = "absent"  # existed before, does not now (emission stopped producing it)

# Statuses that are NOT the agent attempting this capsule and getting it wrong, so a frozen artifact
# under one of them is not actionable. A whole-model capsule deferred by its op-pass gate never emitted;
# naming it "failing and frozen" would point at work the agent cannot do yet. Anything NOT listed here
# counts as an attempt -- fail closed, so a status added later is surfaced rather than silently dropped.
NOT_AN_ATTEMPT = frozenset({"pass", "gated", "skipped", "incomplete", "not_gradeable_no_oracle", None})

# The diagnosis for a round, derived from (submission moved?) x (emitted artifacts moved?).
NO_SUBMISSION_CHANGE = "no_submission_change"
EMIT_INSENSITIVE_TO_EDIT = "emit_insensitive_to_edit"
EMIT_MOVED = "emit_moved"
NOTHING_TO_COMPARE = "nothing_to_compare"


def _walk(root: Path) -> list[Path]:
    """Every regular file under *root*, sorted, with build/VCS scratch excluded.

    Scratch dirs are excluded because their churn is not the agent's edit: a ``__pycache__`` rewrite on
    every import would report CHANGED forever and make the signal useless.
    """
    skip = {"__pycache__", ".git", "build", ".pytest_cache", ".mypy_cache"}
    out: list[Path] = []
    if not root.is_dir():
        return out
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if any(part in skip for part in p.relative_to(root).parts):
            continue
        out.append(p)
    return out


def fingerprint_tree(root: "str | Path") -> dict[str, str]:
    """``{relative path: sha256}`` for every file under *root*.

    Relative keys, so the same tree copied to a different scratch dir each round fingerprints the same --
    the graded submission IS copied per round, and an absolute-path key would report CHANGED every time.
    """
    root = Path(root)
    return {str(p.relative_to(root)): file_digest(p) for p in _walk(root)}


def compare(now: Mapping[str, str], before: Mapping[str, str] | None) -> dict[str, str]:
    """Per-path verdict of *now* against *before* (``None`` == no previous round)."""
    if before is None:
        return {k: NEW for k in now}
    out: dict[str, str] = {}
    for k, v in now.items():
        if k not in before:
            out[k] = NEW
        elif before[k] == v and v != UNKNOWN:
            out[k] = UNCHANGED
        else:
            out[k] = CHANGED
    for k in before:
        if k not in now:
            out[k] = ABSENT
    return out


def _moved(verdicts: Mapping[str, str]) -> bool:
    return any(v in (CHANGED, NEW, ABSENT) for v in verdicts.values())


def build(
    *,
    submission_dir: "str | Path",
    emitted_roots: Mapping[str, "str | Path"],
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The ledger for one round.

    ``submission_dir`` is the exact tree that was graded (not the live workspace -- the graded copy, so
    the fingerprint is of what the oracle actually read). ``emitted_roots`` maps a capsule name to the
    directory holding whatever that capsule's emit path produced. ``previous`` is the prior round's
    ledger, or ``None``.

    The ``diagnosis`` separates causes 1-3 above. Cause 4 needs a perturbation probe, which costs oracle
    time and is deliberately not done here; ``emit_moved`` is the state in which to reach for it.
    """
    sub_now = fingerprint_tree(submission_dir)
    sub_prev = (previous or {}).get("submission_files")
    sub_verdicts = compare(sub_now, sub_prev)

    caps: dict[str, Any] = {}
    prev_caps = (previous or {}).get("capsules") or {}
    for name, root in sorted(emitted_roots.items()):
        now = fingerprint_tree(root)
        pv = (prev_caps.get(name) or {}).get("files")
        caps[name] = {"files": now, "verdicts": compare(now, pv), "root": str(root)}

    sub_moved = _moved(sub_verdicts) if sub_prev is not None else None
    any_emit_moved = any(_moved(c["verdicts"]) for c in caps.values()) if prev_caps else None

    if sub_prev is None:
        diagnosis = NOTHING_TO_COMPARE
    elif not sub_moved:
        diagnosis = NO_SUBMISSION_CHANGE
    elif not any_emit_moved:
        diagnosis = EMIT_INSENSITIVE_TO_EDIT
    else:
        diagnosis = EMIT_MOVED

    return {
        "submission_dir": str(submission_dir),
        "submission_files": sub_now,
        "submission_verdicts": sub_verdicts,
        "submission_moved": sub_moved,
        "capsules": caps,
        "emit_moved": any_emit_moved,
        "diagnosis": diagnosis,
        "n_unreadable": sum(1 for v in sub_now.values() if v == UNKNOWN)
        + sum(1 for c in caps.values() for v in c["files"].values() if v == UNKNOWN),
        "note": (
            "Diagnostic only -- never gates a run. 'emit_insensitive_to_edit' means the graded "
            "submission changed while every emitted artifact stayed byte-identical: the edit did "
            "not reach emission, so no numeric verdict could have moved."
        ),
    }


def capsules_that_did_not_move(ledger: Mapping[str, Any]) -> list[str]:
    """Capsules whose emitted artifacts are byte-identical to the previous round."""
    return sorted(
        name for name, c in (ledger.get("capsules") or {}).items() if c.get("verdicts") and not _moved(c["verdicts"])
    )


def failing_and_frozen(ledger: Mapping[str, Any], verdict: Mapping[str, Any]) -> list[str]:
    """Capsules that FAILED this round **and** whose emitted bytes did not change.

    This is the actionable class, and the run-level ``diagnosis`` cannot express it. Replaying a real
    six-round plateau showed why: every round reported ``emit_moved`` because *some* capsule moved, while
    26 of 27 were frozen -- and in the last round the single capsule that moved was one that already
    passed. A run-level boolean says "something changed" and hides that nothing changed where it mattered.

    A capsule in this list cannot possibly produce a different number next round unless the emitter starts
    reaching it, so it is where the agent (or we) must look.
    """
    frozen = set(capsules_that_did_not_move(ledger))
    if not frozen:
        return []
    failing = {
        c.get("capsule")
        for c in (verdict.get("per_capsule") or [])
        if c.get("capsule") and c.get("status") not in NOT_AN_ATTEMPT
    }
    return sorted(frozen & failing)


def frozen_fraction(ledger: Mapping[str, Any]) -> float | None:
    """Share of capsules whose emission did not move (``None`` on the first round).

    Reported instead of a bare boolean so a plateau is visible as it forms rather than after six rounds.
    """
    caps = ledger.get("capsules") or {}
    comparable = [c for c in caps.values() if c.get("verdicts") and c.get("files")]
    if not comparable or ledger.get("emit_moved") is None:
        return None
    return len(capsules_that_did_not_move(ledger)) / len(comparable)


def summarize(ledger: Mapping[str, Any]) -> str:
    """One line for a log or a round banner."""
    caps = ledger.get("capsules") or {}
    still = capsules_that_did_not_move(ledger)
    frac = frozen_fraction(ledger)
    parts = [
        f"diagnosis={ledger.get('diagnosis')}",
        f"submission_moved={ledger.get('submission_moved')}",
        f"capsules={len(caps)}",
        f"unmoved={len(still)}",
    ]
    if frac is not None:
        parts.append(f"frozen={frac:.0%}")
    if ledger.get("n_unreadable"):
        parts.append(f"unreadable={ledger['n_unreadable']}")
    return "stage_ledger: " + " ".join(parts)


# ---- aet_bridge --------------------------------------------------------------------------------
# Non-destructive bridge from a completed capsule-bench run to the shared **aet** telemetry store.
#
# The capsule-bench harness already writes its own per-run telemetry (``cost_time_toolcalls.yaml`` via
# :mod:`merlin.development.targetgen.experiment_tokens`) and its trajectory plots. This bridge does NOT replace any
# of that — it ADDITIONALLY re-parses the same stream-json transcript(s) with aet's parser and records
# the run into aet's canonical ``<run_dir>/logs/metrics.jsonl`` + ``<run_dir>/metrics/trajectory.json``
# so every experiment becomes visible to ``aet spend`` / ``aet plot`` for cross-experiment cost tracking
# and the shared budget ceiling.
#
# Design contract:
#   * **opt-in, then STICKY** — a no-op unless ``MERLIN_AET_SINK=1`` *or* the run directory already
#     carries an aet record from an earlier session (see :func:`aet_sink_enabled`), so default runs are
#     unchanged while a ``--resume`` from another shell keeps recording instead of silently stopping.
#   * **lazy + soft** — aet is imported inside the function; if it is not installed (or anything fails),
#     the bridge warns and returns ``False`` rather than raising, so a telemetry hiccup never fails a run.
#   * **target-agnostic** — the target/suite/method/model/run_id all arrive as arguments from the run's
#     own metadata (never hardcoded here).


def aet_sink_enabled(run_dir: str | Path | None = None) -> bool:
    """True when the aet telemetry sink is on for this run.

    Two ways to be on, and the second one is the point:

    1. the ``MERLIN_AET_SINK`` env var is set (1/true/yes/on) in THIS process, the original opt-in;
    2. ``run_dir`` already HAS an aet record (``logs/metrics.jsonl``) -- i.e. some earlier session of
       this same run opted in.

    Without (2) the opt-in lives only in one process's environment, so a ``--resume`` launched from a
    different shell silently stops recording and the run's telemetry ends wherever the first session
    did. MEASURED on the atlas capsule-bench run of 2026-09-04: the first session wrote
    ``logs/metrics.jsonl`` until 22:34Z and two later ``--resume`` launches wrote nothing at all --
    ``/proc/<pid>/environ`` of the live resumed process carried no ``MERLIN_AET_SINK``. The run itself,
    not an exported variable, is what remembers that it is being recorded.
    """
    if os.environ.get("MERLIN_AET_SINK", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    return run_dir is not None and (Path(run_dir) / "logs" / "metrics.jsonl").is_file()


def _warn(msg: str) -> None:
    print(f"[aet-bridge] {msg}", file=sys.stderr, flush=True)


def _resolve_transcripts(run_dir: Path, transcript_paths) -> list[Path]:
    """The stream-json transcript(s) to feed aet: explicit paths if given, else the run's combined
    ``transcript.jsonl`` (written by the harness), else every per-round transcript under ``rounds/``."""
    if transcript_paths:
        items = transcript_paths if isinstance(transcript_paths, (list, tuple)) else [transcript_paths]
        return [Path(p) for p in items if Path(p).is_file()]
    combined = run_dir / "transcript.jsonl"
    if combined.is_file():
        return [combined]
    rounds = run_dir / "rounds"
    if rounds.is_dir():
        return sorted(rounds.glob("*.transcript.jsonl"))
    return []


def emit_to_aet(
    *,
    run_dir: str | Path,
    run_id: str,
    method: str,
    model: str,
    target: str,
    suite: str = "capsule-bench",
    project: str = "merlin",
    seed: int = 0,
    transcript_paths=None,
    save_trajectory: bool = True,
    billing_mode: str = "metered",
) -> bool:
    """Feed one completed run's telemetry into the shared aet store (additive; never destructive).

    Parses ``transcript_paths`` (or the run's combined transcript) with aet's ``parse_stream`` and
    records token usage, per-model usage, cost and agent-turn count via ``EvalRunLogger`` into
    ``<run_dir>/logs/`` — the layout ``aet spend``/``aet plot`` discover. When ``save_trajectory`` is
    set it also writes the cumulative ``<run_dir>/metrics/trajectory.json``.

    All metadata (``method``=arm, ``model``, ``target``, ``suite``, ``run_id``) comes from the run
    record — nothing target-specific is baked in here. Returns ``True`` on success, ``False`` on any
    soft failure (aet absent, no transcript, parse/log error) after warning.

    ``billing_mode`` guards the shared spend store: a subscription-seat run (a ChatGPT/Codex account)
    is not charged per token, so it logs **cost 0** and keeps its projection in the
    ``cost.subscription_notional_usd`` metric. Otherwise the list-price estimate below would enter
    ``aet spend`` as real money and consume a budget ceiling nobody is actually being billed against.
    """
    run_dir = Path(run_dir)
    transcripts = _resolve_transcripts(run_dir, transcript_paths)
    if not transcripts:
        _warn(f"no stream-json transcript found under {run_dir}; skipping aet sink")
        return False

    try:
        from aet.tracking.claude_stream import parse_stream
        from aet.tracking.run_logger import EvalRunLogger
    except Exception as e:  # aet not installed / import error → soft no-op
        _warn(f"aet unavailable ({e}); install the 'telemetry' extra to enable the sink")
        return False

    try:
        stream_text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in transcripts)
        result = parse_stream(stream_text)

        # Raw (non-cache) input tokens: gen_ai.usage.input_tokens is the *raw* prefill bucket; the
        # cache buckets are logged separately, so the rollup does not double-count.
        raw_input = sum(t.input_tokens for t in result.turn_usage)
        # Cost fallback chain: authoritative CLI total_cost_usd → summed per-model costUSD → a
        # per-turn list-price ESTIMATE. The estimate covers a KILLED / timed-out run whose
        # transcript was truncated before the terminal `result` event, so both authoritative
        # sources are 0/empty; without it such a run records cost=$0 and can silently blow a budget
        # ceiling even though real per-turn usage exists. estimated_cost_usd() returns None for a
        # cost-unavailable model — in that case we leave cost at 0 (never fabricate) but still
        # record the real turn count below.
        cost = result.cost_usd or sum(mu.cost_usd for mu in result.model_usage)
        cost_is_estimate = False
        if not cost:
            est = result.estimated_cost_usd()  # PriceTable.from_env(); per-turn × list price
            if est is not None:
                cost, cost_is_estimate = est, True
        notional = None
        if billing_mode != "metered":
            notional, cost, cost_is_estimate = cost, 0.0, False

        logger = EvalRunLogger.start(
            project=project,
            suite=suite,
            target=target,
            method=method,
            seed=seed,
            run_id=run_id,
            run_path=run_dir,
            tracking_mode="local",
        )
        logger.log_token_usage(
            input_tokens=raw_input,
            output_tokens=result.total_output_tokens,
            cache_creation_tokens=result.total_cache_creation_tokens,
            cache_read_tokens=result.total_cache_read_tokens,
            model=result.model or model,
        )
        # Record the true within-run per-model split (an Opus orchestrator + delegated Sonnet
        # sub-agents + Haiku background all show distinctly), for EVERY run — not just complete
        # ones. per_model_usage() prefers the authoritative result-event ``modelUsage`` when present
        # and falls back to the per-turn split for a truncated/killed run (where ``model_usage`` is
        # empty), so ``per_model.*`` metrics are always written, never lost with the missing result
        # event.
        per_model = result.per_model_usage()
        if per_model:
            logger.log_model_usage(per_model)
        logger.log_cost(cost, model=result.model or model)
        if notional is not None:
            logger.log_param("billing_mode", billing_mode)
            logger.log_metric("cost.subscription_notional_usd", round(notional, 4))
        # TURN COUNT. ``result.num_turns`` is the CLI's own figure taken from the LAST ``result`` event,
        # and it is authoritative only when a CLI actually reported one. Two ways it lies here:
        #   * a driver whose terminator carries no count. The codex driver emits
        #     ``{"type":"result","subtype":"success"}`` with no ``num_turns``, so aet's parser reads
        #     ``int(None or 0) == 0`` and -- because a result event WAS present -- skips its own
        #     "no result event" fallback. MEASURED: the atlas run's final record said
        #     ``aet.agent.num_turns: 0`` for a transcript holding 794 assistant turns, while the mid-run
        #     records (parsed before any result event existed) said 101, 186, 282 ... 634.
        #   * a COMBINED multi-round transcript, where the last result event describes only the last
        #     round and the earlier rounds' turns vanish.
        # Taking the max with the assistant turns actually present in the transcript fixes both without
        # ever discarding a larger authoritative figure, and never invents a turn that is not there.
        observed_turns = len(result.turn_usage)
        logger.log_agent_turns(max(int(result.num_turns or 0), observed_turns))
        # Recorded separately so the two are distinguishable after the fact rather than merged.
        logger.log_metric("aet.agent.assistant_turns", observed_turns)
        if result.session_id:
            logger.log_session_id(result.session_id)
        logger.close()

        if save_trajectory:
            _save_trajectory(run_dir, result, run_id=run_id, model=result.model or model, suite=suite)

        cost_note = " (per-turn ESTIMATE; transcript truncated, no result event)" if cost_is_estimate else ""
        if notional is not None:
            cost_note = f" ({billing_mode}: ${notional:.4f} notional, not billed)"
        _warn(
            f"recorded run {run_id} → {run_dir}/logs "
            f"(cost=${cost:.4f}{cost_note}, "
            f"turns={max(int(result.num_turns or 0), observed_turns)}, "
            f"tools={result.tool_call_count})"
        )
        return True
    except Exception as e:
        _warn(f"failed to record run {run_id} into aet: {e}")
        return False


def _save_trajectory(run_dir: Path, result, *, run_id: str, model: str, suite: str) -> None:
    """Best-effort cumulative RunTrajectory → ``<run_dir>/metrics/trajectory.json`` (soft on error)."""
    try:
        from aet.trajectory.build import append_round
        from aet.trajectory.classify import ActivityClassifier
        from aet.trajectory.model import RunTrajectory

        traj = RunTrajectory(run_id=run_id, source=f"import:{suite}", model=model)
        append_round(traj, result, classifier=ActivityClassifier())
        traj.to_json(run_dir / "metrics" / "trajectory.json")
    except Exception as e:
        _warn(f"trajectory build skipped for {run_id}: {e}")
