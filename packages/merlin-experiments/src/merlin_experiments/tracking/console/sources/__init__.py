"""Console sources: one bounded, read-only reader per concern, each ``read(ctx) -> section``.

Every section is a JSON-compatible dict. Common rules: times are epoch seconds (float); a value that
was not recorded is ``None`` (never 0); ``notes`` lists human-readable limits of what was read;
``source`` names the record path a row came from; private details (capsule names, paths, golden
values) appear only when ``ctx.operator_private``. Strings from records pass through
:func:`..io.clean`. Series are downsampled to at most ``SERIES_POINTS`` points.

Section shapes (the contract the Bubble Tea frontend decodes):

``resources``  mode: "live"|"recorded"|"none"; cores: [{cpu, util_pct, kind, label}];
               busy_cpus; total_cpus; queue: {waiters, leases, gsim_active}; process:
               {cpu_pct, rss_bytes, threads, processes, load1, io}; series: {t, cpu_pct, rss_bytes, load1};
               notes. ``io`` is the run tree's read()/write() bytes per second (page cache included).
               ``kind`` is one of KINDS.
``agents``     agents: [{id, parent_id, name, path, role, depth, state, started_at, last_activity_at,
               active_s, recorded_active_s, turns, commands, failed_commands, current, last_message, model, effort,
               tokens: TOKENS, spans: [[start, end, kind]], stage, round}]; totals: {agents, working,
               subagents, helper_ceiling}; source; notes.
``tokens``     total: TOKENS; series: {t, fresh_input, cache_read, cache_write, output, reasoning}
               (cumulative); per_agent: [{id, name, total, output, share}]; rate_per_min:
               {input, output}; cache_hit_ratio; budget: {total, working, reserve, used_pct}|None;
               rate_limit: {used_percent, window_minutes, resets_at}|None; context_window; notes.
``jobs``       spans: [{lane, label, capsule, start, end, status, source}]; stages: [{stage, count,
               p50_s, p95_s, max_s, running}]; sim: {requests, queued, running, done,
               queue_wait_p50, queue_wait_p95, runtime_p50, runtime_p95}; running: [{stage,
               capsule, age_s, tier}]; notes. ``lane`` is one of LANES.
``events``     events: [{t, kind, severity, agent, text, source}] newest first; notes.
``capsules``   capsules: [{name, cohort, kind, op, attrs, shapes, dtype, source_role, origin:
               {model, program, reference, aten_op, occurrences}, family, axis,
               pass_requirements, required_tiers, tiers: {L0..L3}, status, numeric, cycles:
               {L2, L3}, failure: {plane, category, tier}, timing: {build_s, sim_active_s,
               oracle_wait_s}, path}]; counts: {cohort: {..}, kind: {..}, op: {..}, status: {..}}; notes.
``progress``   grades: [{token, state, owner_state, reason, started_at, updated_at, n_selected,
               n_completed, n_passed, n_failed, n_errors, source}] newest first (``state`` is
               grading|unfinished|done|superseded|stopped); grades_total; published: {updated_at,
               n_selected, n_completed, n_passed}|None (the copy shown to the agent); certificates:
               {L2, L3: {pass, pending, fail}, physical: {required, passed, failed, pending}}|None;
               series_scope; verdict: {stage, tiers_graded, tiers_not_run, all_pass}|None, state, n_selected,
               n_completed, n_passed, n_execution_passed, snapshot_current,
               graded_source, cohorts: [{name, total, all, L2, L3, refusal}], numeric:
               {completed, passed, failed}, series: {t, n_passed, n_completed}, completion, plateau, alerts,
               latest_request; notes.
``perf``       rows: [{capsule, cohort, spike_model_cycles, gsim_cycles, l2_cycles, floor_cycles,
               efficiency_pct, useful_macs, peak_macs_per_cycle, requested_bytes, intensity,
               achieved_macs_per_cycle, model_status, timing_mode, top_resource, resources,
               traffic, divergence_pct, comparable, basis, source}]; summary: {modeled,
               provisional, unknown, best_eff, worst_eff, geo_eff}; peak_macs_per_cycle; notes.
``golden``     references: [{name, model, program_identity, device, cycles, source}];
               comparisons: [{subject, candidate_cycles, reference_cycles, ratio, comparable,
               reason}]; notes. Operator-only.
``passes``     passes: [{name, kind, declared, used, status, boundary, required_by,
               exercised_by, churn, profile_wall_s, source}]; counts: {library_modules,
               catalog_mechanisms, declared, used, required, profiled}; pipeline: [names]; notes.
``stages``     stages: [{id, title, state, basis, actions: {done, failed, running, total},
               capsules: {required, passing}, wall_s, tokens, metrics: {..}, items: [..]}];
               contract; notes. ``state`` is one of STAGE_STATES.

``timeline``   window, lead {think_s, tool_s, wait_s, setup_s, idle_s}, recorded, overlap,
               parallelism, concurrency, markers, verdict; see :mod:`.timeline`.
``machine``    cores [{cpu, util_pct, holders, mine, console}], allocations, experiment_cpus,
               console_cpus; see :mod:`.machine`.

Insight panels (``performance``, ``cases``, ``tools``, ``grader``, ``integrity``, ``attempts``, ``library``,
``certification``, ``focus``, ``cost``; :data:`..model.INSIGHTS`) share one generic shape, so one
frontend renderer draws them all::

    {title, summary: [{label, value, level, key}], table: {columns: [str], rows: [{cells: [str],
     level, key}]}|None, series: [{label, t: [..], v: [..]}], notes, sorts?: [{label, order: [int]}],
     ...source-specific fields}

``value`` and ``cells`` are display text (``"—"`` when not recorded); ``level`` is
ok|info|warn|bad|None; ``key`` is passed to that source's ``inspect``. The optional ``sorts`` are row
orders of the table (indices into ``rows``) that the frontend's sort key cycles; without them the rows
are shown as sent. The other source-specific fields are documented in each module (:mod:`.performance`
and the rest) and are not decoded by the frontend.

TOKENS is ``{input, fresh_input, cache_read, cache_write, output, reasoning, total}``.

``inspect(ctx, key) -> {"title", "sections": [{"title", "lines": [str]}], "paths": [str]}``.
"""

from __future__ import annotations

SERIES_POINTS = 240
KINDS = ("agent", "L2", "L3", "build", "verify", "calibration", "host", "unknown", "idle")
LANES = ("setup", "build", "L2", "L3", "verify", "calibration", "queue")
STAGE_STATES = ("done", "active", "blocked", "pending", "unknown")
TOKEN_FIELDS = ("input", "fresh_input", "cache_read", "cache_write", "output", "reasoning", "total")


def downsample(points: list, limit: int = SERIES_POINTS) -> list:
    """At most ``limit`` points, always keeping the last (the current value)."""
    if len(points) <= limit:
        return list(points)
    step = len(points) / (limit - 1)
    picked = [points[int(i * step)] for i in range(limit - 1)]
    return picked + [points[-1]]


def empty_tokens() -> dict:
    return {name: None for name in TOKEN_FIELDS}


__all__ = ["KINDS", "LANES", "SERIES_POINTS", "STAGE_STATES", "TOKEN_FIELDS", "downsample", "empty_tokens"]
