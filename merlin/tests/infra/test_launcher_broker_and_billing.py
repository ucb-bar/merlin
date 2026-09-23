"""Two launcher facts that were unreachable from the command line, and one of them lied.

**Broker concurrency.** ``_start_selfcheck_broker`` spawned every broker as ``[python, <broker>,
--ws, <ws>]`` and nothing else, so ``simjob_broker``'s ``--max-jobs`` always fell back to its own
default of 4 no matter what engine was certifying. That default belongs to a Verilator cert tier —
the broker separately caps Verilator at 2 GLOBAL slots because one instance holds a core for ~45
minutes. GSIM is a different machine problem entirely: measured on this host, 10.4 MB RSS, one
thread, ~12 s per gemmini capsule, and 24 concurrent instances returned bit-identical cycle counts
while total throughput rose 13.4x. An operator who wants the cert tier to use the machine had no way
to say so, and no error told them why the tier ran at a sixth of its capacity.

**Billing mode.** ``_billing_mode`` asks the DRIVER module for a declared ``BILLING_MODE`` and calls
everything else metered. ``claudecode`` has no module — it drives the ``claude`` CLI directly — so a
run on the machine's own subscription SEAT was stamped ``metered`` and its tokens priced at
Anthropic list rates, indistinguishable in the ledger from a run on our Bedrock key. That is the
exact defect ``subscription_notional`` was introduced for on the codex seat (a round once booked
``estimated_cost_usd: 17.21`` against an account not billed per token), one driver over. For a
module-less driver the PROVIDER is what knows: ``subscription`` = the seat, ``bedrock`` = our AWS
key. Verified before the fix: ``--driver claudecode --provider subscription`` returned ``metered``,
byte-identical to ``--provider bedrock``.

Both are pinned here because both are invisible when wrong: an under-parallel broker looks like a
slow engine, and a mispriced seat run looks like a budget.
"""

from __future__ import annotations

import sys

import pytest
from merlin_experiments.phase1.providers import execution as EX

from merlin.common.paths import merlin_dir
from merlin.targetgen import tool_registry

_HARNESS = merlin_dir() / "experiments/capsule_bench/harness"
if str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))


@pytest.fixture()
def loop():
    import run_baseline_qa_loop as L

    return L


# --------------------------------------------------------------------- billing
def test_a_seat_run_on_the_claude_cli_is_not_billed_per_token(loop):
    """``--driver claudecode --provider subscription`` must NOT report metered spend."""
    from merlin.targetgen import experiment_tokens as ET

    assert EX.billing_mode("claude-opus-4-8", config=EX.ProviderConfig("claudecode")) == ET.SUBSCRIPTION_NOTIONAL, (
        "a subscription-seat Claude Code run reported metered billing; its tokens will be priced at "
        "list rates and can be spent against a real budget ceiling"
    )


def test_the_same_driver_on_our_bedrock_key_is_still_metered(loop):
    """The other half: ``--provider bedrock`` IS our AWS key and must stay metered.

    Without this the fix would be a blanket 'claudecode is free', which under-reports real spend —
    the mirror failure, and the worse one.
    """
    from merlin.targetgen import experiment_tokens as ET

    assert EX.billing_mode("claude-opus-4-8", config=EX.ProviderConfig("claudecode", "bedrock")) == ET.METERED


def test_a_driver_that_declares_its_own_billing_mode_still_decides(loop):
    """A driver module's own declaration outranks the provider — the provider is only the fallback."""
    from merlin.targetgen import experiment_tokens as ET

    assert EX.billing_mode("claude-opus-4-8", config=EX.ProviderConfig("opencode")) == ET.METERED, (
        "the provider fallback leaked past a driver that declares its own BILLING_MODE"
    )


# ------------------------------------------------------------------- broker
def _spawned_argvs(loop, monkeypatch, tmp_path, *, broker_specs=(), sim_max_jobs=0, **inputs):
    """Capture the argv of every broker ``_start_selfcheck_broker`` would spawn."""
    from merlin_experiments.phase1.feedback import lifecycle as FL

    seen: list[list[str]] = []

    class _FakeProc:
        def __init__(self, argv, **kw):
            seen.append(list(argv))

    monkeypatch.setattr(FL.subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(FL, "stage_client", lambda *a, **k: None)
    monkeypatch.setattr(tool_registry, "brokers_for", lambda *a, **k: broker_specs)
    ws = tmp_path / "ws"
    ws.mkdir()
    FL.start_brokers(
        ws, FL.BrokerConfig(loop.C.CONTEXT, (), loop.SCRIPTS / ".oracle_timing.json", sim_max_jobs, **inputs)
    )
    return seen


def test_sim_max_jobs_reaches_the_simjob_broker_and_only_it(loop, monkeypatch, tmp_path):
    argvs = _spawned_argvs(loop, monkeypatch, tmp_path, sim_max_jobs=12)
    simjob = [a for a in argvs if a[1:3] == ["-m", "merlin_experiments.phase1.brokers.simjob"]]
    others = [a for a in argvs if a not in simjob]
    assert simjob, f"no simjob broker was spawned; got {[a[1] for a in argvs]}"
    for a in simjob:
        assert "--max-jobs" in a and a[a.index("--max-jobs") + 1] == "12", (
            f"--sim-max-jobs never reached the broker: {a}"
        )
    for a in others:
        assert "--max-jobs" not in a, f"--max-jobs was forwarded to a broker whose argparse does not accept it: {a}"


def test_the_default_path_passes_no_max_jobs_at_all(loop, monkeypatch, tmp_path):
    """0 must leave the broker on its own default — the pre-flag behaviour, byte for byte."""
    for a in _spawned_argvs(loop, monkeypatch, tmp_path):
        assert "--max-jobs" not in a, f"an unset --sim-max-jobs still forwarded a value: {a}"


def test_feedback_brokers_receive_distinct_frozen_corpus_policy_and_schema_views(loop, monkeypatch, tmp_path):
    public, policy, contract = (tmp_path / name for name in ("frozen-public", "frozen-policy", "frozen-contract"))
    argvs = _spawned_argvs(
        loop,
        monkeypatch,
        tmp_path,
        capsules_root=public,
        policy_root=policy,
        contract=contract,
        broker_specs=tool_registry.brokers_for(("isa_tools", "cca_tools")),
    )
    feedback = [argv for argv in argvs if argv[2].rsplit(".", 1)[-1] in {"selfcheck", "simjob"}]
    assert len(feedback) == 2
    for argv in feedback:
        assert argv[argv.index("--capsules-root") + 1] == str(public)
        assert argv[argv.index("--policy-capsules-root") + 1] == str(policy)
        assert argv[argv.index("--contract") + 1] == str(contract)
    assert all("--capsules-root" not in argv for argv in argvs if argv not in feedback)


def test_packaged_tool_brokers_use_canonical_modules_and_explicit_isa_context(loop, monkeypatch, tmp_path):
    specs = tool_registry.brokers_for(("isa_tools", "cca_tools"))
    argvs = _spawned_argvs(loop, monkeypatch, tmp_path, broker_specs=specs)
    modules = {argv[2]: argv for argv in argvs if argv[1] == "-m"}
    assert set(modules) == {
        "merlin_experiments.phase1.brokers.isa_tools",
        "merlin_experiments.phase1.brokers.cca",
        "merlin_experiments.phase1.brokers.selfcheck",
        "merlin_experiments.phase1.brokers.simjob",
    }
    isa = modules["merlin_experiments.phase1.brokers.isa_tools"]
    assert isa[isa.index("--descriptor") + 1] == str(loop.C.DESCRIPTOR)
    assert isa[isa.index("--repo") + 1] == str(loop.C.REPO)
    assert "--descriptor" not in modules["merlin_experiments.phase1.brokers.cca"]
    assert all("--max-jobs" not in argv for argv in modules.values())


def test_the_flag_exists_on_the_real_operator_surface():
    """An option nobody can pass is the state this fixes, so assert it on the actual ``--help``.

    Run as a SUBPROCESS deliberately: the thing under test is what an operator typing the command
    gets, and an in-process parser probe would pass just as happily against a parser this module
    never reaches.
    """
    import subprocess
    import sys as _sys

    out = subprocess.run(
        [_sys.executable, str(_HARNESS / "run_baseline_qa_loop.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert "--sim-max-jobs" in out.stdout, (
        "run_baseline_qa_loop exposes no way to set the simjob broker's concurrency; the cert tier "
        "is stuck on the broker's Verilator-era default of 4 whatever engine is certifying"
    )


def test_the_flag_defaults_to_deferring_to_the_broker():
    """0 = 'let the broker choose', never a launcher-side override of it."""
    from merlin_experiments.phase1.options import parse_options

    assert parse_options(["--run-id", "probe"]).sim_max_jobs == 0
    explicit = parse_options(["--run-id", "probe", "--sim-max-jobs", "6"])
    assert type(explicit.sim_max_jobs) is int
    assert explicit.sim_max_jobs == 6
