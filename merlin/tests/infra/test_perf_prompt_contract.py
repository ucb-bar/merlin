"""The performance agent receives a complete, immutable, fail-closed Arm4 task."""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace

import pytest

from merlin.common.paths import merlin_dir


_SRC = merlin_dir() / "experiments/gemmini_perf_bench/scripts/perf_prompt.py"
_SPEC = importlib.util.spec_from_file_location("perf_prompt_under_test", _SRC)
assert _SPEC is not None and _SPEC.loader is not None
PP = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = PP
_SPEC.loader.exec_module(PP)


def _inputs():
    digest = "a" * 64
    manifest_digest = "b" * 64
    corpus_digest = "c" * 64
    host_digest = "d" * 64
    frozen = "inputs/functional/submission"
    submission = "submission"
    workload = "inputs/performance/capsules"
    manifest = "inputs/performance/performance_corpus_manifest.json"
    host = "inputs/host_lane/package"
    host_manifest = "inputs/host_lane/package/manifest.yaml"
    receipts = "evidence/tool_invocations.jsonl"
    cells = tuple(
        PP.PerfCell("PK", capsule, simulator, replicate)
        for capsule in ("pk_k2", "pk_k4")
        for simulator in ("spike", "gsim")
        for replicate in ("r000", "r001")
    )
    return PP.PerfPromptInputs(
        target="gemmini",
        approach="arm4",
        functional_run_id="arm4_functional_exact",
        functional_submission_sha256=digest,
        frozen_functional_path=frozen,
        frozen_functional_sha256=digest,
        submission_path=submission,
        submission_initial_sha256=digest,
        functional_public_capsules=20,
        functional_hidden_capsules=5,
        workload_root=workload,
        workload_manifest=manifest,
        workload_manifest_sha256=manifest_digest,
        workload_capsules_sha256=corpus_digest,
        expected_cells=cells,
        families=(PP.PerfFamily("PK", "PREDICTS", "fixed_extents", "residual bound",
                                "identical emitted work; unpriced demands must cancel", ("K",)),),
        host_lane=PP.HostLaneGrant("rvv", "rvv_int8", host, host_digest, host_manifest,
                                   "compile_model(package=<pinned package>)"),
        tools=(
            PP.ToolGrant("perf-selfcheck", "python3 perf_broker.py request selfcheck",
                         "redacted own-artifact QA", True),
            PP.ToolGrant("cca-contract", "python3 perf_broker.py request cca-contract",
                         "enumerate compiler levers", True),
            PP.ToolGrant("rtl-facts", "python3 perf_broker.py request rtl-facts",
                         "derive target facts", True),
            PP.ToolGrant("gsim", "python3 perf_broker.py request probe-gsim",
                         "harness-owned elaborated-RTL L3 simulator"),
        ),
        allowed_paths=(frozen, submission, workload, manifest, host, host_manifest,
                       "perf_broker.py", receipts, "perf_selfcheck.py", "cca_contract.py", "rtl_facts.py"),
        execution_broker_path="perf_broker.py",
        execution_broker_command="python3 perf_broker.py request",
        broker_receipt_path=receipts,
    )


def test_prompt_is_deterministic_and_carries_every_campaign_boundary() -> None:
    inputs = _inputs()
    first = PP.render_initial_prompt(inputs)
    second = PP.render_initial_prompt(inputs)
    assert first == second
    assert len(PP.prompt_sha256(inputs)) == 64

    required = (
        "Only the Arm4 submission is evaluated",
        inputs.functional_run_id,
        inputs.functional_submission_sha256,
        inputs.frozen_functional_path,
        inputs.workload_manifest_sha256,
        inputs.workload_capsules_sha256,
        "mount-scoped `bwrap`",
        "Network is available",
        "Network isolation is neither claimed nor required",
        "`--clearenv`",
        "underlying tool/simulator outside this broker",
        "broker, its credential/mount/",
        inputs.broker_receipt_path,
        "undeclared files",
        inputs.host_lane.package_id,
        inputs.host_lane.package_sha256,
        "hidden host lowering",
        "Spike is a correctness screen only",
        "GSIM is performance certification",
        "cycles must be recorded as `null`/absent",
        "(family, capsule, simulator, replicate)",
        "measured negative-control result",
        "`merlin.perf.differential` comparability decision",
        "required tool",
        "`GO` requires",
        "NO-GO",
        "does **not** claim",
        "submission/performance/PLAN.md",
        "submission/performance/iteration_notes.md",
        "submission/performance/REPORT.md",
    )
    for phrase in required:
        assert phrase in first
    for cell in inputs.expected_cells:
        assert cell.label in first
    for tool in inputs.tools:
        assert tool.command in first


@pytest.mark.parametrize(
    "mutate, message",
    (
        (lambda value: replace(value, approach="golden"), "only approach='arm4'"),
        (lambda value: replace(value, frozen_functional_sha256="e" * 64),
         "frozen functional bytes"),
        (lambda value: replace(value, submission_initial_sha256="e" * 64),
         "exact functional fork"),
        (lambda value: replace(value, functional_hidden_capsules=0), "positive integer"),
        (lambda value: replace(value, functional_highest_tier="L2"), "passed through L3"),
        (lambda value: replace(value, functional_integrity_status="unknown"), "clean integrity"),
        (lambda value: replace(value, sandbox_engine="none"), "requires broker-only bwrap"),
        (lambda value: replace(value, credentials_absent=False), "requires broker-only bwrap"),
        (lambda value: replace(value, answer_surfaces_masked=False), "requires broker-only bwrap"),
        (lambda value: replace(value, declared_files_only=False), "requires broker-only bwrap"),
        (lambda value: replace(value, broker_clearenv=False), "requires broker-only bwrap"),
        (lambda value: replace(value, direct_candidate_execution_forbidden=False),
         "requires broker-only bwrap"),
        (lambda value: replace(value, expected_cells=()), "zero cells"),
        (lambda value: replace(value, families=()), "no falsifiable family"),
        (lambda value: replace(value, tools=()), "grants no tools"),
        (lambda value: replace(
            value, tools=tuple(replace(tool, required=False) for tool in value.tools)),
         "must be enforced"),
        (lambda value: replace(
            value, allowed_paths=tuple(path for path in value.allowed_paths
                                       if path != value.host_lane.package_path)),
         "omit launch inputs"),
    ),
)
def test_launch_prerequisites_fail_closed(mutate, message: str) -> None:
    with pytest.raises(PP.PerfPromptContractError, match=message):
        PP.render_initial_prompt(mutate(_inputs()))


def test_completion_identity_requires_exact_l2_l3_pairs_without_duplicates() -> None:
    inputs = _inputs()
    without_l3 = tuple(cell for cell in inputs.expected_cells
                       if not (cell.capsule == "pk_k2" and cell.replicate == "r000"
                               and cell.simulator == "gsim"))
    with pytest.raises(PP.PerfPromptContractError, match="both the L2 and L3"):
        PP.render_initial_prompt(replace(inputs, expected_cells=without_l3))
    with pytest.raises(PP.PerfPromptContractError, match="repeats a cell"):
        PP.render_initial_prompt(
            replace(inputs, expected_cells=inputs.expected_cells + (inputs.expected_cells[0],)))
    uneven = tuple(cell for cell in inputs.expected_cells
                   if not (cell.capsule == "pk_k2" and cell.replicate == "r001"))
    with pytest.raises(PP.PerfPromptContractError, match="same exact replicate set"):
        PP.render_initial_prompt(replace(inputs, expected_cells=uneven))


def test_family_and_tool_contracts_cannot_drift_from_the_cells() -> None:
    inputs = _inputs()
    with pytest.raises(PP.PerfPromptContractError, match="exactly cover"):
        PP.render_initial_prompt(replace(
            inputs,
            families=(PP.PerfFamily("PF", "RECOVERS", "unfused", "overlap", "same work"),),
        ))
    duplicate_tool = inputs.tools + (inputs.tools[0],)
    with pytest.raises(PP.PerfPromptContractError, match="duplicate names"):
        PP.render_initial_prompt(replace(inputs, tools=duplicate_tool))
    with pytest.raises(PP.PerfPromptContractError, match="bypass"):
        PP.render_initial_prompt(replace(
            inputs,
            tools=(replace(inputs.tools[0], command="python3 submission/compiler.py"),)
            + inputs.tools[1:],
        ))
    with pytest.raises(PP.PerfPromptContractError, match="must declare fitted parameters"):
        PP.render_initial_prompt(replace(
            inputs,
            families=(replace(inputs.families[0], fitted_parameters=()),),
        ))


def test_prompt_does_not_license_agent_authored_measurements_or_broad_claims() -> None:
    prompt = PP.render_initial_prompt(_inputs())
    assert "Official correctness, cycles, provenance, falsifier, differential, and GO/NO-GO records" in prompt
    assert "Never author or patch a passing verdict yourself" in prompt
    assert "superiority over an unevaluated" in prompt
    assert "end-to-end model speedup from kernel-only cells" in prompt


def test_network_is_available_and_is_not_a_launch_isolation_gate() -> None:
    prompt = PP.render_initial_prompt(_inputs())
    assert "Network is available to the agent and to brokered execution" in prompt
    assert "Network isolation is neither claimed nor required" in prompt
    assert "--unshare-net" not in prompt
    assert "Do not access the network" not in prompt
    assert "network is unshared" not in prompt.lower()


def test_the_prompt_documents_every_summary_field_the_feedback_actually_carries() -> None:
    """A field the document carries and the prompt never names is a field nobody reads.

    Measured: the agent was given declared work, baseline cycles and share-of-achievable per member
    and would have had to multiply, subtract and rank across every row to see that seven members held
    92.4% of the objective. It never did, and three trials of search went into the other 7%. The
    ranking is now computed for it -- this pins that the prompt says so, and that it says what the
    ranking may NOT be read as.
    """
    prompt = PP.render_initial_prompt(_inputs())
    for field in ("summary.recoverable", "recoverable_cycles", "share_of_corpus_cycles",
                  "recoverable_share_of_corpus", "total_recoverable_share"):
        assert field in prompt, f"the feedback carries {field} and the prompt never names it"
    # The two things it must NOT be read as. Both are refusals, and a prompt that reported the
    # ranking without them would read as a licence to work only the top row.
    assert "not a claim that the cycles are reachable" in prompt
    assert "not permission to ignore small members" in prompt
