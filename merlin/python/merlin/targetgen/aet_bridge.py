"""Non-destructive bridge from a completed capsule-bench run to the shared **aet** telemetry store.

The capsule-bench harness already writes its own per-run telemetry (``cost_time_toolcalls.yaml`` via
:mod:`merlin.targetgen.experiment_tokens`) and its trajectory plots. This bridge does NOT replace any
of that — it ADDITIONALLY re-parses the same stream-json transcript(s) with aet's parser and records
the run into aet's canonical ``<run_dir>/logs/metrics.jsonl`` + ``<run_dir>/metrics/trajectory.json``
so every experiment becomes visible to ``aet spend`` / ``aet plot`` for cross-experiment cost tracking
and the shared budget ceiling.

Design contract:
  * **opt-in** — a no-op unless ``MERLIN_AET_SINK=1`` (see :func:`aet_sink_enabled`), so default runs
    are unchanged and the existing telemetry path is untouched.
  * **lazy + soft** — aet is imported inside the function; if it is not installed (or anything fails),
    the bridge warns and returns ``False`` rather than raising, so a telemetry hiccup never fails a run.
  * **target-agnostic** — the target/suite/method/model/run_id all arrive as arguments from the run's
    own metadata (never hardcoded here).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def aet_sink_enabled() -> bool:
    """True when the aet telemetry sink is opted in via ``MERLIN_AET_SINK`` (1/true/yes/on)."""
    return os.environ.get("MERLIN_AET_SINK", "").strip().lower() in ("1", "true", "yes", "on")


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
) -> bool:
    """Feed one completed run's telemetry into the shared aet store (additive; never destructive).

    Parses ``transcript_paths`` (or the run's combined transcript) with aet's ``parse_stream`` and
    records token usage, per-model usage, cost and agent-turn count via ``EvalRunLogger`` into
    ``<run_dir>/logs/`` — the layout ``aet spend``/``aet plot`` discover. When ``save_trajectory`` is
    set it also writes the cumulative ``<run_dir>/metrics/trajectory.json``.

    All metadata (``method``=arm, ``model``, ``target``, ``suite``, ``run_id``) comes from the run
    record — nothing target-specific is baked in here. Returns ``True`` on success, ``False`` on any
    soft failure (aet absent, no transcript, parse/log error) after warning.
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
        cost = result.cost_usd or sum(mu.cost_usd for mu in result.model_usage)

        logger = EvalRunLogger.start(
            project=project, suite=suite, target=target, method=method, seed=seed,
            run_id=run_id, run_path=run_dir, tracking_mode="local",
        )
        logger.log_token_usage(
            input_tokens=raw_input,
            output_tokens=result.total_output_tokens,
            cache_creation_tokens=result.total_cache_creation_tokens,
            cache_read_tokens=result.total_cache_read_tokens,
            model=result.model or model,
        )
        if result.model_usage:
            logger.log_model_usage(result.model_usage)
        logger.log_cost(cost, model=result.model or model)
        logger.log_agent_turns(result.num_turns)
        if result.session_id:
            logger.log_session_id(result.session_id)
        logger.close()

        if save_trajectory:
            _save_trajectory(run_dir, result, run_id=run_id, model=result.model or model, suite=suite)

        _warn(f"recorded run {run_id} → {run_dir}/logs "
              f"(cost=${cost:.4f}, turns={result.num_turns}, tools={result.tool_call_count})")
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
