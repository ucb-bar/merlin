"""The performance agent receives a complete, immutable, fail-closed Arm4 task."""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace

import pytest

from merlin.common.paths import merlin_dir


_SRC = merlin_dir() / "experiments/gemmini_perf_bench/scripts/perf_prompt.py"
if str(_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_SRC.parent))
import perf_pk_claim as PK  # noqa: E402

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
    bundle_manifest = "inputs/functional/snapshot.json"
    capsules = ("PK00_k16", "PK01_k32", "PK02_k64", "PK03_k128")
    cells = tuple(
        PP.PerfCell("PK", capsule, simulator, replicate)
        for capsule in capsules
        for simulator in ("spike", "verilator")
        for replicate in ("r000", "r001", "r002")
    )
    acceptance = PK.supported_acceptance()
    formal_claim = {
        "schema_version": 1,
        "family": "PK",
        "claim": "PREDICTS",
        "status": "READY",
        "declaration": acceptance,
        "cohort": {"replicates": ["r000", "r001", "r002"]},
        "expected_identities": [
            {"family": "PK", "capsule": capsule, "simulator": simulator,
             "replicate": replicate, "tier": "L2" if simulator == "spike" else "L3"}
            for capsule in capsules
            for replicate in ("r000", "r001", "r002")
            for simulator in ("spike", "verilator")
        ],
        "refusal_reasons": [],
    }
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
        functional_bundle_snapshot_manifest=bundle_manifest,
        functional_bundle_snapshot_manifest_sha256="e" * 64,
        functional_bundle_snapshot_sha256="f" * 64,
        workload_root=workload,
        workload_manifest=manifest,
        workload_manifest_sha256=manifest_digest,
        workload_capsules_sha256=corpus_digest,
        expected_cells=cells,
        replicates=3,
        formal_replicate_identities=("r000", "r001", "r002"),
        formal_claim=formal_claim,
        smoke_replicates=1,
        wall_budget_seconds=7200,
        rounds=2,
        round_timeout_seconds=3600,
        max_tool_calls=100,
        tool_timeout_seconds=900,
        families=(PP.PerfFamily("PK", "PREDICTS", "fixed_extents", "residual bound",
                                "identical emitted work; unpriced demands must cancel", ("K",),
                                acceptance),),
        host_lane=PP.HostLaneGrant(
            "rvv", "rvv_int8", host, host_digest, host_manifest,
            "host-owned runner consumes schedule+knobs while candidate handles target regions"),
        e2e_sentinel=PP.E2ESentinel(
            "M2_microvit_gemmini", "inputs/model/M2_microvit_gemmini",
            "inputs/functional/repo/merlin/contract/capsules/model/M2_microvit_gemmini",
            "9" * 64, ("on_mesh", "scalar_rvv_lane"), ("L0", "L1", "L2", "L3")),
        tools=(
            PP.ToolGrant("perf-selfcheck", "python3 /perf-control/perf_tool.py perf-selfcheck",
                         "redacted own-artifact QA", True),
            PP.ToolGrant("cca-contract", "python3 /perf-control/perf_tool.py cca-contract",
                         "enumerate compiler levers", True),
            PP.ToolGrant("rtl-facts", "python3 /perf-control/perf_tool.py rtl-facts",
                         "derive target facts", True),
            PP.ToolGrant("verilator", "python3 /perf-control/perf_tool.py probe-verilator",
                         "harness-owned L3 simulator"),
        ),
        allowed_paths=(frozen, submission, workload, manifest, bundle_manifest,
                       "inputs/model/M2_microvit_gemmini", host, host_manifest,
                       "/perf-control/perf_tool.py", receipts, "perf_selfcheck.py",
                       "cca_contract.py", "rtl_facts.py"),
        execution_broker_path="/perf-control/perf_tool.py",
        execution_broker_command="python3 /perf-control/perf_tool.py",
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
        "Verilator is performance certification",
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
        "ordinary_least_squares_all_L3_replicates",
        "r000, r001, r002",
        "Smoke work is a cheap diagnostic",
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
        (lambda value: replace(value, outer_codex_auth_exception=False),
         "requires broker-only bwrap"),
        (lambda value: replace(value, inner_execution_credentials_absent=False),
         "requires broker-only bwrap"),
        (lambda value: replace(value, answer_surfaces_masked=False), "requires broker-only bwrap"),
        (lambda value: replace(value, declared_files_only=False), "requires broker-only bwrap"),
        (lambda value: replace(value, broker_clearenv=False), "requires broker-only bwrap"),
        (lambda value: replace(value, direct_candidate_execution_forbidden=False),
         "requires broker-only bwrap"),
        (lambda value: replace(value, expected_cells=()), "zero cells"),
        (lambda value: replace(value, wall_budget_seconds=0), "wall budget"),
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
                       if not (cell.capsule == "PK00_k16" and cell.replicate == "r000"
                               and cell.simulator == "verilator"))
    with pytest.raises(PP.PerfPromptContractError, match="both the L2 and L3"):
        PP.render_initial_prompt(replace(inputs, expected_cells=without_l3))
    with pytest.raises(PP.PerfPromptContractError, match="repeats a cell"):
        PP.render_initial_prompt(
            replace(inputs, expected_cells=inputs.expected_cells + (inputs.expected_cells[0],)))
    uneven = tuple(cell for cell in inputs.expected_cells
                   if not (cell.capsule == "PK00_k16" and cell.replicate == "r001"))
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


def test_prompt_distinguishes_outer_codex_auth_from_inner_credential_free_execution() -> None:
    prompt = PP.render_initial_prompt(_inputs())
    assert "outer Codex control plane necessarily has an isolated authentication exception" in prompt
    assert "Only inner brokered execution is credential-free" in prompt
    assert "does not fabricate one" in prompt
    assert "high-fidelity correctness admission sentinel, not a performance cell" in prompt


def test_frozen_acceptance_and_formal_replicates_cannot_drift_or_be_omitted() -> None:
    inputs = _inputs()
    missing = replace(inputs.families[0], acceptance=None)
    with pytest.raises(PP.PerfPromptContractError, match="frozen acceptance"):
        PP.render_initial_prompt(replace(inputs, families=(missing,)))

    drifted_acceptance = dict(inputs.families[0].acceptance)
    drifted_acceptance["thresholds"] = dict(drifted_acceptance["thresholds"])
    drifted_acceptance["thresholds"]["r_squared_min_inclusive"] = 0.9
    with pytest.raises(PP.PerfPromptContractError, match="acceptance drifts"):
        PP.render_initial_prompt(replace(
            inputs, families=(replace(inputs.families[0], acceptance=drifted_acceptance),)))

    with pytest.raises(PP.PerfPromptContractError, match="formal replicates"):
        PP.render_initial_prompt(replace(inputs, replicates=2))
    with pytest.raises(PP.PerfPromptContractError, match="smoke replicates"):
        PP.render_initial_prompt(replace(inputs, smoke_replicates=3))
