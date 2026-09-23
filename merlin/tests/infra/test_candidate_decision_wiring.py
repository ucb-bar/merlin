"""The comparison verdict reaches production, or these go red.

``merlin.perf.candidate_decision`` is the named instrument for "which of these two arms is better".
It existed, it was tested, its docstring recorded two measured incidents it refuses by name -- and it
had ZERO production callers. The phase-2 loop emitted both arms, priced both, wrote
``command_buffer_identical`` and ``lowered_identical`` beside the numbers, and nothing turned any of
it into a verdict. That is not a state a comment can detect: a document with no decision in it looks
exactly like a document about a comparison that had nothing to say.

So the assertions here are written against the WIRING, not the library. Each one names the edit that
turns it red:

* delete ``diagnostics["candidate_decision"] = ...`` from ``analyze_whole_model_emission`` and the
  structural check fails;
* stop binding the emission digests into the decision and the identical-emission case comes back a
  speedup;
* drop the ``decide_measured_totals`` call from ``extract_best_candidate.choose_best`` and a cohort
  whose best total equals its baseline reports a confident ``1.0000x`` again.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import sys

import pytest
from merlin_experiments.phase2 import contracts as CONTRACTS
from merlin_experiments.phase2 import emission_analysis as EA
from merlin_experiments.phase2 import emission_diagnostics as ED
from merlin_experiments.phase2 import stage_inputs as INPUTS

from merlin.common.paths import merlin_dir, module_source_path, repo_root
from merlin.perf import candidate_decision as CD
from merlin.targetgen import oot_runner
from merlin.targetgen.rocc import decode

SCRIPTS = repo_root() / "merlin" / "experiments" / "gemmini_perf_bench" / "scripts"
sys.path.insert(0, str(merlin_dir() / "experiments/gemmini_perf_bench/scripts"))

SOURCE = """builtin.module {
  func.func @forward() -> tensor<2xi32> {
    %0 = arith.constant dense<[3, 7]> : tensor<2xi32>
    func.return %0 : tensor<2xi32>
  }
}"""
LOWERED = """builtin.module {
  llvm.func @kernel(%0: !llvm.ptr) {
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.getelementptr %0[%1] {merlin.global_task = 0 : i64} : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.return
  }
}"""


def _assignments_in(path, function: str) -> set[str]:
    """Every literal subscript key assigned inside ``function``.

    Structural (``ast``) rather than a word search: the key name appears in this file's prose, in
    the production comment beside the call, and in the adapter -- counting those would report dead
    wiring as live, which is the exact failure this file exists about.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return {
                target.slice.value
                for inner in ast.walk(node)
                if isinstance(inner, ast.Assign)
                for target in inner.targets
                if isinstance(target, ast.Subscript)
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            }
    raise AssertionError(f"{path} has no function {function!r}")


def _calls_in(path, function: str) -> set[str]:
    """Every function NAME called inside ``function`` (bare names and attribute tails)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            out = set()
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    func = inner.func
                    if isinstance(func, ast.Name):
                        out.add(func.id)
                    elif isinstance(func, ast.Attribute):
                        out.add(func.attr)
            return out
    raise AssertionError(f"{path} has no function {function!r}")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------------------------
# The in-loop emission comparison
# ------------------------------------------------------------------------------------------------


def _buffer(tag: str = "") -> dict:
    return {
        "abi_version": "0.1",
        "target": "fixture-target",
        "commands": [],
        "tensors": {"result": {"shape": [2], "dtype": "i32", "role": "output"}},
        "kernel_abi": {
            "kind": "whole_program",
            "args": [{"tensor": "result", "access": "write"}],
            "outputs": ["result"],
        },
        "params": {"arm_marker": tag} if tag else {},
    }


def _analysis(tmp_path, monkeypatch, *, candidate_lowered=LOWERED, candidate_marker="", arms=None):
    """Run the real ``analyze_whole_model_emission`` over a controlled emission pair."""
    baseline, candidate, source = (tmp_path / name for name in ("base", "candidate", "source"))
    for path in (baseline, candidate, source):
        path.mkdir()
    (baseline / "compiler.py").write_text("BASE=1\n")
    (candidate / "compiler.py").write_text("CANDIDATE=1\n")
    (source / "capsule.yaml").write_text(json.dumps({"id": "fixture-model"}))
    (source / "capsule.interface.mlir").write_text(SOURCE)
    sentinel = INPUTS.StageE2ESentinel(
        "fixture-model", str(source), str(source), CONTRACTS.exact_tree_record(source)["sha256"], (), ()
    )

    def emit(package, interface, scratch, tag, timeout):
        text = LOWERED if tag == "baseline" else candidate_lowered
        return 0, text, json.dumps(_buffer("" if tag == "baseline" else candidate_marker))

    monkeypatch.setattr(oot_runner, "load_package", lambda path: path)
    monkeypatch.setattr(decode, "decode_module", lambda *a, **kw: {"instructions": []})
    monkeypatch.setattr(
        ED,
        "analyze_command_buffers",
        lambda *a, **kw: {
            "arms": arms
            or {
                "baseline": {"status": "emitted"},
                "candidate": {"status": "emitted"},
            }
        },
    )
    monkeypatch.setattr(EA, "inspect_compiler_package", lambda *_: None)
    seen: list = []

    def guidance(diagnostics, inventory):
        seen.append(diagnostics)
        return {}

    monkeypatch.setattr(EA, "guidance_for_emission_analysis", guidance)
    document = EA.analyze_whole_model_emission(
        baseline,
        candidate,
        sentinel,
        timeout_s=10,
        peak_macs_per_cycle=None,
        achievable_macs_per_cycle=None,
        target="fixture-target",
        contract_root=merlin_dir() / "contract",
        emit_pair_runner=emit,
        host_verifier_policy_sha256="a" * 64,
    )
    return document, seen


def test_the_whole_model_analysis_assigns_a_candidate_decision() -> None:
    """MUTATION: delete ``diagnostics["candidate_decision"] = decide_emitted_pair(...)`` and this
    fails. The behavioural tests below would still pass if the adapter existed and nothing called
    it -- which was the state for the whole life of ``candidate_decision``."""
    owner = module_source_path("merlin_experiments.phase2.emission_analysis")
    assigned = _assignments_in(owner, "analyze_whole_model_emission")
    assert "candidate_decision" in assigned, (
        "nothing in the whole-model analysis writes candidate_decision; candidate_decision.compare "
        "would have no production caller and the loop would emit ratios with no verdict over them"
    )
    assert "decide_emitted_pair" in _calls_in(owner, "analyze_whole_model_emission")


def test_an_identical_emission_cannot_produce_a_speedup_verdict(tmp_path, monkeypatch) -> None:
    """THE INCIDENT. Two different compiler trees that emit the SAME program.

    MUTATION: stop passing the emission digests to ``decide_emitted_pair`` and this comes back as a
    verdict about the movement numbers -- which is what the last campaign's receipt recorded beside
    ``command_buffer_identical: True`` and ``lowered_identical: True``.
    """
    arms = {
        "baseline": {
            "status": "emitted",
            "movement": {"known_bytes": 1000.0, "exact_bytes": 1000.0, "is_lower_bound": False},
        },
        "candidate": {
            "status": "emitted",
            "movement": {"known_bytes": 10.0, "exact_bytes": 10.0, "is_lower_bound": False},
        },
    }
    document, _ = _analysis(tmp_path, monkeypatch, arms=arms)
    assert document["emission"]["lowered_identical"] is True
    decision = document["candidate_decision"]
    assert decision["verdict"] == CD.IDENTICAL_EMISSION
    assert decision["verdict"] != CD.BETTER
    assert decision["decided_by"] == "", "no instrument may be credited for an identical pair"


def test_an_axis_that_did_not_move_cannot_carry_the_verdict(tmp_path, monkeypatch) -> None:
    """Byte-identical traffic beside a differing emission is NOT a data-movement win.

    MUTATION: declare the traffic axis as ``moved=True`` regardless of the readings, or drop the
    metric's ``blind_to``, and this reports BETTER on an axis that never moved.
    """
    arms = {
        "baseline": {
            "status": "emitted",
            "movement": {"known_bytes": 169.52e6, "exact_bytes": 169.52e6, "is_lower_bound": False},
        },
        "candidate": {
            "status": "emitted",
            "movement": {"known_bytes": 169.52e6, "exact_bytes": 169.52e6, "is_lower_bound": False},
        },
    }
    document, _ = _analysis(
        tmp_path, monkeypatch, candidate_lowered=LOWERED + "\n// differing\n", candidate_marker="candidate", arms=arms
    )
    assert document["emission"]["lowered_identical"] is False
    assert document["emission"]["command_buffer_identical"] is False
    decision = document["candidate_decision"]
    assert decision["verdict"] != CD.BETTER
    assert "traffic" not in decision["moved_axes"]


def test_a_real_traffic_reduction_is_decided_and_names_its_instrument(tmp_path, monkeypatch) -> None:
    arms = {
        "baseline": {
            "status": "emitted",
            "movement": {"known_bytes": 169.52e6, "exact_bytes": 169.52e6, "is_lower_bound": False},
        },
        "candidate": {
            "status": "emitted",
            "movement": {"known_bytes": 84.0e6, "exact_bytes": 84.0e6, "is_lower_bound": False},
        },
    }
    document, _ = _analysis(
        tmp_path, monkeypatch, candidate_lowered=LOWERED + "\n// differing\n", candidate_marker="candidate", arms=arms
    )
    decision = document["candidate_decision"]
    assert decision["verdict"] == CD.BETTER
    assert decision["decided_by"] == "command_buffer_movement_volume"
    assert decision["moved_axes"] and "traffic" in decision["moved_axes"]


def test_the_decision_is_in_the_diagnostics_the_agent_brief_is_built_from(tmp_path, monkeypatch) -> None:
    """MUTATION: move the assignment BELOW ``guidance_for_emission_analysis`` and this fails -- the
    verdict would exist in the artifact and never reach the agent that needs it."""
    _, seen = _analysis(tmp_path, monkeypatch)
    assert seen, "the brief was never built"
    assert "candidate_decision" in seen[0]
    assert seen[0]["candidate_decision"]["verdict"] in CD.VERDICTS


def test_the_emission_analysis_never_claims_a_timing_verdict(tmp_path, monkeypatch) -> None:
    """The two timing axes are not declared from a pair nothing executed."""
    document, _ = _analysis(
        tmp_path, monkeypatch, candidate_lowered=LOWERED + "\n// differing\n", candidate_marker="candidate"
    )
    assert document["timing_status"] == "UNMEASURED"
    undeclared = {row["axis"] for row in document["candidate_decision"]["undeclared_axes"]}
    assert {"wall_cycles", "accelerator_time"} <= undeclared


# ------------------------------------------------------------------------------------------------
# The post-hoc "which candidate was best" extraction
# ------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def extractor():
    return _load_script("extract_best_candidate")


def _measurement(call, *, baseline, candidate, engine="gsim", members=("f/a",)):
    return {
        "document": f"call_{call}.json",
        "round": 0,
        "call": call,
        "engine": engine,
        "candidate_sha256": f"{call:064d}",
        "members": tuple(members),
        "n_members": len(members),
        "baseline_total_cycles": baseline,
        "candidate_total_cycles": candidate,
        "snapshot": None,
    }


def test_choose_best_calls_the_named_instrument(extractor) -> None:
    """MUTATION: delete the ``decide_measured_totals`` call and this fails."""
    assert "decide_measured_totals" in _calls_in(SCRIPTS / "extract_best_candidate.py", "choose_best")


def test_a_best_total_equal_to_its_baseline_is_not_a_1x_win(extractor) -> None:
    """A confident ``1.0000x`` is not a result; it is the absence of one."""
    got = extractor.choose_best([_measurement(1, baseline=1000, candidate=1000)])
    assert got["speedup_vs_baseline"] == 1.0
    assert got["decision"]["verdict"] == CD.NO_EFFECT
    assert got["decision"]["verdict"] != CD.BETTER


def test_a_real_reduction_is_BETTER_on_the_certified_engine(extractor) -> None:
    got = extractor.choose_best(
        [_measurement(1, baseline=1000, candidate=900), _measurement(2, baseline=1000, candidate=500)]
    )
    assert got["certified_timing_engine"] is True
    assert got["decision"]["verdict"] == CD.BETTER
    assert got["decision"]["decided_by"] == extractor.CERTIFIED_TIMING_ENGINE


def test_a_functional_only_engine_is_not_ranked_on(extractor) -> None:
    """It prices every accelerator command at one cycle, so it flatters exactly the offloading the
    search is doing. MUTATION: pass ``blind_to=()`` for every engine and this reports BETTER."""
    engine = sorted(extractor.FUNCTIONAL_ONLY_ENGINES)[0]
    got = extractor.choose_best([_measurement(1, baseline=1000, candidate=500, engine=engine)])
    assert got["certified_timing_engine"] is False
    assert got["decision"]["verdict"] != CD.BETTER
    assert got["decision"]["decision"]["metrics"][0]["blind_to"]


def test_a_cohort_mixing_engines_records_no_instrument(extractor) -> None:
    got = extractor.choose_best(
        [_measurement(1, baseline=1000, candidate=900), _measurement(2, baseline=1000, candidate=500, engine="other")]
    )
    assert got["cohort_engine"] is None
    assert got["decision"]["verdict"] == CD.UNKNOWN
