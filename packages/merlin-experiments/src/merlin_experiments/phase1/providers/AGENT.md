# AGENT.md — packages/merlin-experiments/src/merlin_experiments/phase1/providers

Preserve the native process supervisors, model routing, prompt bytes, raw and arrival-stamped
transcripts, billing declarations and timeout recovery. These are trusted host implementations,
not candidate SDK modules. Provider import may observe its established environment defaults;
it must not choose a target, contact a service or import the native controller.

Codex and OpenCode receive the caller's existing sandbox_command explicitly for bwrap runs.
The `execution` owner holds the existing Phase 1 sandbox composition and provider dispatch,
with immutable ProviderConfig/ExecutionConfig and an explicit late tool-observation callback.
It preserves that prefix composition rather than substituting the different full_argv
prefix. Both callers use bwrap.apply_final_answer_masks after all runtime binds and frozen
grant reapplication, so host-only input aliases and private snapshot metadata stay withheld.
Mount-table tests do not qualify kernel isolation; payload integrity remains the existing
native snapshot verification boundary, not a repeated full-corpus hash on every shell command.
Never recreate a second sandbox policy here. Installed bridge startup requires explicit proxy config
and executable paths; checkout defaults require a physically detected source checkout.
The resource sampler is a standalone stdlib CLI, also runnable with the canonical module name.

Native provider authentication/environment assembly remains outside this owner, including
checkout-specific dependency paths. The batch banner passes the parsed provider and driver to
the same accounting policy as execution. Fake CLI tests qualify process,
transcript and cleanup behavior, not a provider account, kernel sandbox or full installed engine.

Codex `run_round` accepts explicit `codex_binary` and `codex_home_root` inputs.
Phase 2 passes its preflight-selected executable and stage-local home root without
mutating `CODEX_BIN` or the shared artifact cache function. Omitted inputs retain
the existing environment/cache defaults. Initial execution and session continuation
use the same selected executable and per-round home; caller sandbox policy remains
mandatory for bwrap execution. This does not make all provider state thread-safe.
Continuous Codex sessions retry only explicit capacity refusals, at most three times
within the existing wall budget, using the same thread, model and sandbox. Preserve
all failed-turn events and unknown usage. Quota/authentication errors, missing session
identity, round-mode execution and exhausted retry budgets remain non-successes.

Optional explicit credential selection is stat/path only and requires complete
explicit isolated Codex inputs. Never discover the real home in that route. Reject
same-file/hard-link executable and credential selections before executable hashing.
The shared ordinary/canary runtime constructs the same read grants, home mounts and
native permission preflight. Existing per-round homes refuse before client startup.
