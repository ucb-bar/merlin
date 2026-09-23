"""Functional compiler invocation options, independent of target initialization.

The parser is the single owner of CLI defaults. Construct it at invocation time so
explicit process-environment values retain their existing precedence. Parsing does
not discover a repository, select a descriptor, or execute an experiment.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from dataclasses import dataclass

from merlin.targetgen import tool_registry as _TR


@dataclass(frozen=True)
class RunOptions:
    """Parsed invocation values; runtime state and evidence belong to the session."""

    arm: str
    run_id: str
    model: str
    effort: str
    driver: str
    subagent_model: str
    background_model: str
    provider: str
    aws_region: str
    aws_profile: str
    schedule: str
    max_wall_s: int
    max_rounds: int
    min_rounds: int
    plateau_rounds: int
    round_timeout: int
    qa_timeout: int
    sim_max_jobs: int
    model_budget_s: int | None
    continuous: bool
    grade_interval: int
    sandbox: str
    allow_unsandboxed: bool
    no_oracle: bool
    skip_hidden: bool
    experiment: str
    bundle: str
    max_rate_limit_waits: int
    rl_test_reset_epoch: int
    resume: bool
    seal_current: bool
    seed_submission: str
    operator_errata: str
    with_tool: list[str]
    without_tool: list[str]
    account_config_dir: str


def build_parser(
    *, default_arm: str = "raw_baseline", environ: Mapping[str, str] | None = None
) -> argparse.ArgumentParser:
    environment = os.environ if environ is None else environ
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        choices=["raw_baseline", "merlin_assisted", "cpp_merlininfra"],
        default=default_arm,
        help="which arm/bundle to run (default raw_baseline; the QA loop is identical)",
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--effort", default="high")
    # AGENT DRIVER (Claude-Code-like interfaces). auto (default) preserves today's behavior: route by model
    # id — the Bedrock Converse loop for a non-Anthropic id, else the claude CLI.
    ap.add_argument(
        "--driver",
        choices=["auto", "converse", "claudecode", "opencode", "codex"],
        default="auto",
        help="agent driver: auto (route by model id), converse (Bedrock Converse loop), "
        "claudecode (claude CLI; Bedrock via --provider bedrock), opencode (OpenCode CLI), "
        "codex (Codex CLI; ChatGPT auth = subscription_notional cost, never metered)",
    )
    ap.add_argument(
        "--subagent-model",
        default="",
        help="delegate/subagent model (alias or Bedrock id) for tier-within-agent; default per "
        "driver (Anthropic: sonnet; non-Anthropic: qwen-coder)",
    )
    ap.add_argument(
        "--background-model",
        default="",
        help="background/mechanical model (alias or Bedrock id) for chores; default per driver "
        "(Anthropic: haiku; non-Anthropic: nova-lite)",
    )
    # PROVIDER for the agent's `claude` CLI — experiments-only, so the interactive Claude Code keeps the
    # subscription. subscription (default) = the machine's ~/.claude creds; bedrock = Claude Code's own
    # Bedrock mode (CLAUDE_CODE_USE_BEDROCK=1 + AWS creds + a Bedrock inference-profile model id).
    ap.add_argument(
        "--provider",
        choices=["subscription", "bedrock"],
        default="subscription",
        help="model provider for the agent CLI (experiments-only; subscription keeps ~/.claude)",
    )
    ap.add_argument(
        "--aws-region", default=environment.get("AWS_REGION", "us-east-1"), help="AWS region for --provider bedrock"
    )
    ap.add_argument(
        "--aws-profile",
        default=environment.get("AWS_PROFILE", ""),
        help="AWS profile (~/.aws) for --provider bedrock; else the env-var cred chain",
    )
    ap.add_argument(
        "--schedule",
        choices=("rounds", "continuous"),
        default="continuous",
        help="CONTINUOUS (default): the round COUNT stops being a terminator and the run "
        "ends on convergence, plateau, or a declared wall/spend budget. ROUNDS is the "
        "explicit legacy-reproduction mode: the loop is bounded by "
        "--max-rounds, and the run ends when that budget is spent whether or not the "
        "submission was still improving. Per-capsule promotion is unaffected by this "
        "flag: a capsule's cert tier "
        "is enqueued the moment its loop tier passes (tier_promote fires on EVERY "
        "verdict — both brokers and the round grade), never at a round boundary. What "
        "continuous removes is the ARTIFICIAL end, not the grading cadence.\n"
        "Rounds remain the unit of agent invocation and of the artifact layout in both "
        "modes, so every downstream reader (round_NN transcripts, qa_history verdicts, "
        "the cost rollup) is unchanged.",
    )
    ap.add_argument(
        "--max-wall-s",
        type=int,
        default=0,
        help="continuous only (0 = no wall cap): stop after this much ACTIVE agent wall "
        "time. With --schedule continuous and no wall cap and no plateau, the only "
        "terminators left are convergence and the spend ceiling — which is what you "
        "want for a run whose whole point is to finish the work, but say so on purpose.",
    )
    ap.add_argument("--max-rounds", type=int, default=12)
    ap.add_argument(
        "--min-rounds",
        type=int,
        default=0,
        help="OPT-IN (0 = disabled, the default). Refuse the agent's READY_FOR_BARRIER "
        "self-declaration before round N unless it is actually passing. A model that "
        "writes the marker while scoring zero has not converged, it has given up: GLM-5 "
        "did exactly that in four consecutive runs, once after only two rounds, leaving "
        "ten of its twelve rounds unspent. Declining the marker deletes it (the agent "
        "must re-declare) and returns the loop to the agent with the same failing "
        "verdict. Never overrides a genuine all_pass, which always ends the loop.",
    )
    ap.add_argument(
        "--plateau-rounds",
        type=int,
        default=0,
        help="OPT-IN (0 = disabled, the default — never cut a productive run). When set to N, "
        "stop early (not converged) after N consecutive rounds with NO progress: neither "
        "the pass count NOR the total numeric mismatch improved. The mismatch-aware metric "
        "means a run making ANY numeric progress is never stopped (verified: it would not "
        "have fired on the productive glm5 15/20 run, whose flat-pass stretches kept "
        "reducing mismatch). Enable it only for a run you know is pathologically stuck "
        "(re-sending its uncached context each round with zero movement), e.g. N=3-4.",
    )
    ap.add_argument(
        "--round-timeout",
        type=int,
        default=14400,
        help="per-round agent wall cap (s). Default 4h (matches launch_ab_batch): a TIGHT "
        "cap is net-detrimental — it doesn't cut the work (fixed by difficulty), it "
        "just forces more rounds, each adding a full grading pass + context re-read + "
        "rate-limit-boundary exposure, and can cut a productive round mid-fix (rc=124). "
        "The original abc runs used 4h and converged in ~1 productive round.",
    )
    ap.add_argument("--qa-timeout", type=int, default=900)
    ap.add_argument(
        "--sim-max-jobs",
        type=int,
        default=0,
        metavar="N",
        help="how many async oracle jobs the simjob broker may run at once (0 = the "
        "broker's own default of 4). The default was chosen when the cert tier meant "
        "Verilator, which is separately capped at 2 global slots because one instance "
        "holds a core for ~45 min. A cheap cert engine is a different machine problem: "
        "Raise this when the cert engine is cheap; leave it alone when it is not. "
        "Verilator's own global slot budget is unaffected either way.",
    )
    ap.add_argument(
        "--model-budget-s",
        type=int,
        default=None,
        help="wall-clock ceiling for ONE whole-model capsule inside a round grade (s). "
        "Default: --qa-timeout, i.e. the capstone may cost at most what this operator "
        "already said one grading step may. --qa-timeout itself is a PER-STEP "
        "subprocess cap and a whole-model grade makes many such calls, so it cannot "
        "bound the capsule. "
        "0 = no ceiling (an operator certification run wants that; a per-round gate "
        "does not).",
    )
    # NOT default-on, and deliberately so: this is the LEGACY single-session path. It keeps one agent
    # session and re-grades underneath it, but it does NOT run the post-freeze public+hidden L3 grade, so
    # it can report progress and can NEVER report a formal success (it returns 1 and hardcodes
    # formal_complete=False). The certified continuous path is `--schedule continuous`, where the round
    # COUNT is not a terminator: the run stops on EVIDENCE (converged, plateaued) or on a declared budget,
    # and `--max-rounds` is ignored. Pair it with a long `--round-timeout` (e.g. 43200) so each agent
    # session is long and barriers are rare.
    #
    # Historical progress-only sessions ended after ~1.5h at 18/33
    # with `grades=2` and `formal_complete=False` -- the session closed when the agent stopped, well inside
    # a 12h --round-timeout, and no formal verdict was reachable.
    ap.add_argument(
        "--continuous",
        action="store_true",
        help="LEGACY single-session mode: one long-lived agent session with a background "
        "grader every --grade-interval seconds, instead of round relaunches. Reports "
        "progress only -- it does NOT run the post-freeze public+hidden L3 grade and so "
        "can never report a formal success. For the certified continuous path use "
        "`--schedule continuous` with a long --round-timeout.",
    )
    ap.add_argument(
        "--grade-interval",
        type=int,
        default=900,
        help="seconds between background grades in --continuous mode, and between the "
        "in-turn full-ladder grades under `--schedule continuous` (default 900). The "
        "FIRST grade of a turn is never on this interval: it is the fast loop-tier "
        "grade, published as soon as there is a submission, so an agent whose turn "
        "outlives the interval is not working blind. 0 disables the interval grades "
        "(the fast first verdict still lands).",
    )
    # Default sandbox=bwrap: enforced FS isolation (the agent cannot read a masked answer surface at all),
    # AND the driver-side broker tools (async simjob oracle, self-check, arc-model isa_tools, CCA) only
    # start under bwrap. The earlier "bwrap crashes claude" blocker was a sandbox-config bug, now fixed in
    # sandbox/bwrap.base_argv (bind the systemd-resolved dir so DNS works; drop inherited CLAUDE_CODE_*
    # nesting markers; provide an XDG runtime dir) — validated end-to-end by smoke_agent_check. "none" stays
    # available as an escape hatch (golden-masked COPY workspace + post-run transcript audit), but it is
    # detection-not-prevention and must not be used for scored runs.
    ap.add_argument("--sandbox", choices=["bwrap", "none"], default="bwrap")
    ap.add_argument(
        "--allow-unsandboxed",
        action="store_true",
        help="explicitly permit a real (spending) run under --sandbox none; without it a "
        "'none' run is refused (workspace assembly alone does not hide denied paths).",
    )
    ap.add_argument("--no-oracle", action="store_true", help="QA = L0+trace only (fast dev)")
    ap.add_argument("--skip-hidden", action="store_true")
    ap.add_argument(
        "--experiment",
        choices=["full", "realistic"],
        default="full",
        help="'realistic' (abc2): whole-repo + self-check tool + READY_FOR_BARRIER self-pacing "
        "+ TASK_realistic; 'full' (abc1, default): spike-loop then verilator checkpoint",
    )
    ap.add_argument("--bundle", default="", help="override the arm's bundle id (e.g. *_realistic_v0)")
    # Rate-limit awareness: when the org five-hour session budget rejects a round (zero work), sleep
    # until resetsAt and RETRY the same round instead of burning it. Lets an n>=3 sweep span windows
    # unattended. --max-rate-limit-waits caps total resets waited across the run (then stop honestly).
    ap.add_argument("--max-rate-limit-waits", type=int, default=3)
    ap.add_argument(
        "--rl-test-reset-epoch",
        type=int,
        default=0,
        help="TEST ONLY: override the reset epoch to wait toward (verification, not 5h)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="continue an existing run_dir (cross-window robustness) instead of refusing",
    )
    ap.add_argument(
        "--seal-current",
        action="store_true",
        help="resume-only: stop authoring at the last completed checkpoint, then run the "
        "ordinary official public/hidden grade and immutable freeze. This records an "
        "incomplete operator seal; it never reports convergence or bypasses integrity",
    )
    ap.add_argument(
        "--seed-submission",
        default="",
        metavar="DIR",
        help="fresh run only: initialize submission/ from a preserved candidate while the "
        "new run seals its current bundle and records the candidate's exact identity",
    )
    ap.add_argument(
        "--operator-errata",
        default="",
        metavar="FILE",
        help="fresh run only: archive a correction as ERRATA.md before the first round brief "
        "is assembled, with exact provenance in environment.yaml",
    )
    # Use a SECOND subscription account (separate org five-hour budget) by pointing claude at a
    # different config dir. This is the only real way to run two agent arms truly concurrently: the
    # five-hour limit is org-wide and non-overage, so process parallelism on ONE account just
    # time-shares the same bucket; a second account = a second bucket. Log in once with
    # `CLAUDE_CONFIG_DIR=<dir> claude auth login`, then pass --account-config-dir <dir> here.
    ap.add_argument(
        "--with-tool",
        action="append",
        default=[],
        metavar="NAME",
        help=f"ABLATION: grant this arm-gated tool on top of the arm's rung (repeatable). "
        f"Known: {', '.join(_TR.ablatable_tools())}",
    )
    ap.add_argument(
        "--without-tool",
        action="append",
        default=[],
        metavar="NAME",
        help="ABLATION: withhold this arm-gated tool from the arm's rung (repeatable). Pair "
        "with a bundle generated for the same cell so the file grants match the brokers.",
    )
    ap.add_argument(
        "--account-config-dir",
        default=environment.get("CLAUDE_CONFIG_DIR", ""),
        help="CLAUDE_CONFIG_DIR for the agent's claude CLI (a different subscription account)",
    )
    return ap


def parse_options(
    argv: list[str] | None = None,
    *,
    default_arm: str = "raw_baseline",
    environ: Mapping[str, str] | None = None,
) -> RunOptions:
    return RunOptions(**vars(build_parser(default_arm=default_arm, environ=environ).parse_args(argv)))
