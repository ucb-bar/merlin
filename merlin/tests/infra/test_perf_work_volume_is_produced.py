"""The perf runners' compute axis must be PRODUCED, not merely read.

`run_perf_bench._resource_bindings` names a member's ``compute`` resource axis from two keys on the
grade -- ``work_volume`` and ``command_buffer_artifact``. Both readers (the fixed bench and the
paired bench) copied them off the grade; `capsule_runner.run_capsule` wrote neither. So the axis was
unreachable, and nothing said so: `_resource_bindings` returns ``{}`` for an absent key exactly as it
does for a genuinely unpriceable member, `_measurement_identity` files its refusal into a list the
paired path never reads, and the run goes on to report cycles per member with no compute axis to
attribute them to. Loud only in ONE configuration (the fixed bench under ``--hardware-counters`` with
verilator, where `_link_counter_passes` escalates the same refusal into a `CampaignGateError`);
silent in every other, the paired bench included.

These tests hold the wiring from both ends: the producer emits the keys the readers actually read,
the numbers are `work_volume`'s own and not a second notion of work, an unpriceable member yields an
explicit UNKNOWN rather than a zero or an absent key, and the reader accepts exactly the first and
refuses the second.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir
from merlin.perf import work_volume as WV
from merlin.targetgen import capsule_runner as CR

_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location(
    "run_perf_bench_work_volume_under_test", _SCRIPTS / "run_perf_bench.py")
assert _SPEC is not None and _SPEC.loader is not None
FIXED = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = FIXED
_SPEC.loader.exec_module(FIXED)

_M, _K, _N = 16, 32, 8


def _runner_config():
    """A grading config whose shape, not whose identity, matters here (no target facts read)."""
    from merlin.targetgen.runner_config import RunnerConfig

    return RunnerConfig(target="synthetic-endpoint", suite="synthetic-capsule-bench",
                        dtype="i8xi8_i32", fourth_output_name="kernel.S", tier_sim={},
                        rtl_tiers=frozenset(), oracle_tiers=(), perf_fields=(), trace_gate=None)


def _capsule() -> dict:
    """A capsule complete enough that the run reaches the L0/L1 floor and finalizes normally."""
    return {"name": "WV_synthetic_matmul", "kind": "model_slice", "label": "dev",
            "operation": {"op": "matmul", "attributes": {"lhs": "X", "weight": "W", "out": "Y"}},
            "inputs": [{"name": "X", "shape": [_M, _K], "dtype": "i8", "role": "input",
                        "fill": "iota"},
                       {"name": "W", "shape": [_K, _N], "dtype": "i8", "role": "weight",
                        "fill": "iota"}],
            "numeric_policy": {"compare": "exact_int"}}


def _countable_buffer() -> dict:
    """A program whose every command has a work-counting rule, so its total is EXACT."""
    return {"abi_version": "0.1", "target": "synthetic",
            "tensors": {"X": {"shape": [_M, _K], "dtype": "i8", "role": "input"},
                        "W": {"shape": [_K, _N], "dtype": "i8", "role": "weight"},
                        "Y": {"shape": [_M, _N], "dtype": "i32", "role": "output"}},
            "commands": [
                {"opcode": "MATMUL", "operands": {"lhs": "X", "rhs": "W", "dst": "acc"}},
                {"opcode": "COMMIT", "operands": {"src": "acc", "dst": "Y"},
                 "attributes": {"epilogue": [], "output_dtype": "i32"}}]}


def _uncountable_buffer() -> dict:
    """The same program plus one command the counter has no rule for: the TOTAL becomes UNKNOWN."""
    cb = _countable_buffer()
    cb["commands"].insert(1, {"opcode": "SOME_UNMODELLED_COMPUTE",
                              "operands": {"src": "acc", "dst": "acc"}, "attributes": {}})
    return cb


def _grade(tmp_path, monkeypatch, cb, *, run_id: str = "wv") -> dict:
    """Drive the real grading path with the build stubbed out; `cb` is what the compiler emitted."""
    monkeypatch.setattr(CR, "run_entrypoints",
                        lambda *a, **k: (object(), copy.deepcopy(cb), "# kernel.S (stub)\n"))
    return CR.run_capsule(_capsule(), "unused-package", runs_root=tmp_path, run_id=run_id,
                          config=_runner_config(), oracle_adapters={})


def _measurement(grade: dict) -> dict:
    """Exactly the projection `run_arm4` / `_run_arm4_engines` make onto their result row."""
    work = grade.get("work_volume") if isinstance(grade.get("work_volume"), dict) else {}
    row = {"approach": "arm4", "status": grade.get("status"), "work_volume": work}
    if isinstance(grade.get("command_buffer_artifact"), dict):
        row["command_buffer_artifact"] = dict(grade["command_buffer_artifact"])
    return row


# ---------------------------------------------------------------------------------------------
# the keys exist at all, and they are the keys the readers read
# ---------------------------------------------------------------------------------------------
def test_the_readers_read_the_keys_the_producer_writes(tmp_path, monkeypatch):
    """Both halves of the contract, named once, so a rename on either side breaks a test."""
    grade = _grade(tmp_path, monkeypatch, _countable_buffer())
    assert "work_volume" in grade, "the grade carries no work volume; the compute axis is unreachable"
    assert "command_buffer_artifact" in grade, "the grade carries no raw command-buffer receipt"
    source = Path(_SCRIPTS / "run_perf_bench.py").read_text(encoding="utf-8")
    for key in ("work_volume", "command_buffer_artifact"):
        assert f'grade.get("{key}")' in source, f"the fixed bench no longer reads {key}"
    paired = Path(_SCRIPTS / "run_paired_perf_bench.py").read_text(encoding="utf-8")
    for key in ("work_volume", "command_buffer_artifact"):
        assert f'grade.get("{key}")' in paired, f"the paired bench no longer reads {key}"


# ---------------------------------------------------------------------------------------------
# a member whose work CAN be counted produces work_volume's own number -- not a second one
# ---------------------------------------------------------------------------------------------
def test_a_countable_member_reports_the_counter_s_own_total(tmp_path, monkeypatch):
    cb = _countable_buffer()
    grade = _grade(tmp_path, monkeypatch, cb)
    counted = WV.work_from_command_buffer(cb)
    assert counted.exact_macs == _M * _K * _N, "the fixture is not exactly countable"
    work = grade["work_volume"]
    assert work["exact_macs"] == counted.exact_macs, \
        "the grade's total is not work_volume's -- two notions of work that can silently disagree"
    assert work["basis"] == counted.basis and work["unit"] == counted.unit
    assert work["refusals"] == []


def test_the_receipt_is_the_bytes_the_total_was_counted_from(tmp_path, monkeypatch):
    """The reader re-digests the embedded buffer; both digests must be the one the counter used."""
    cb = _countable_buffer()
    grade = _grade(tmp_path, monkeypatch, cb)
    artifact, work = grade["command_buffer_artifact"], grade["work_volume"]
    embedded = artifact["command_buffer"]
    assert FIXED._canonical_sha256(embedded) == work["artifact_sha256"]
    assert artifact["artifact_sha256"] == work["artifact_sha256"]
    assert WV.work_from_command_buffer(embedded).exact_macs == work["exact_macs"]
    assert isinstance(artifact["compiler_provenance"], str) and artifact["compiler_provenance"].strip()


def test_the_reader_now_names_a_compute_axis(tmp_path, monkeypatch):
    """The end the whole change exists for: `_resource_bindings` yields the axis."""
    grade = _grade(tmp_path, monkeypatch, _countable_buffer())
    bindings = FIXED._resource_bindings(_measurement(grade))
    assert "compute" in bindings, "a countable member still has no compute axis to attribute cycles to"
    assert bindings["compute"]["resource"] == "compute:compiler_command_buffer:macs"
    assert bindings["compute"]["derived_from_tool"] is True
    assert grade["work_volume"]["artifact_sha256"] in bindings["compute"]["provenance"]


def test_the_measurement_identity_no_longer_refuses_the_program(tmp_path, monkeypatch):
    """The refusal `_link_counter_passes` escalates must be gone for a priceable member."""
    grade = _grade(tmp_path, monkeypatch, _countable_buffer())
    _, refusals = FIXED._measurement_identity(
        package_before="a" * 64, package_after="a" * 64, inputs_before="b" * 64,
        inputs_after="b" * 64, work_volume=grade["work_volume"], toolchain_shas={"tool": "rev"},
        target="synthetic-endpoint", expected_package_sha256="a" * 64, rtl_facts_sha256="c" * 64)
    assert refusals == [], refusals


# ---------------------------------------------------------------------------------------------
# MUTATION: a member whose work CANNOT be counted must not produce a number -- and must not vanish
# ---------------------------------------------------------------------------------------------
def test_an_uncountable_member_produces_an_explicit_unknown_not_a_zero(tmp_path, monkeypatch):
    cb = _uncountable_buffer()
    grade = _grade(tmp_path, monkeypatch, cb, run_id="wv_unknown")
    work = grade["work_volume"]
    assert work["exact_macs"] is None, "unpriceable work was given a number"
    assert work["exact_macs"] != 0, "a zero here reads as 'this program does no work' on a perf bench"
    assert work["is_lower_bound"] is True
    assert work["refusals"], "the total is UNKNOWN and nothing says why"
    assert any("SOME_UNMODELLED_COMPUTE" in reason for reason in work["refusals"]), work["refusals"]
    # the key is still THERE: an absent key reads as "no compute axis applies", which is a lie
    assert "command_buffer_artifact" in grade
    assert grade["command_buffer_artifact"]["command_buffer"] is not None


def test_the_reader_refuses_a_compute_axis_for_an_uncountable_member(tmp_path, monkeypatch):
    grade = _grade(tmp_path, monkeypatch, _uncountable_buffer(), run_id="wv_unknown_reader")
    assert "compute" not in FIXED._resource_bindings(_measurement(grade))


def test_a_run_that_produced_no_buffer_still_says_so(tmp_path, monkeypatch):
    """The failure mode the absent key hid: nothing was emitted, so nothing can be priced."""

    def _explode(*_a, **_k):
        raise RuntimeError("the submission's entrypoints did not produce a command buffer")

    monkeypatch.setattr(CR, "run_entrypoints", _explode)
    grade = CR.run_capsule(_capsule(), "unused-package", runs_root=tmp_path, run_id="wv_nobuffer",
                           config=_runner_config(), oracle_adapters={})
    assert grade["status"] == "error", grade.get("failure")
    work = grade["work_volume"]
    assert work["exact_macs"] is None and work["is_lower_bound"] is True
    assert WV.NO_COMMAND_BUFFER_REFUSAL in work["refusals"]
    artifact = grade["command_buffer_artifact"]
    assert artifact["command_buffer"] is None and artifact["artifact_sha256"] is None
    assert artifact["refusal"] == WV.NO_COMMAND_BUFFER_REFUSAL
    assert "compute" not in FIXED._resource_bindings(_measurement(grade))


# ---------------------------------------------------------------------------------------------
# NON-VACUITY: the reader's acceptance above is not something it does for any input at all
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("corrupt", [
    pytest.param(lambda row: row["work_volume"].__setitem__("exact_macs", 0), id="zero_total"),
    pytest.param(lambda row: row["work_volume"].__setitem__("artifact_sha256", "0" * 64),
                 id="receipt_digest_disagrees"),
    pytest.param(lambda row: row["command_buffer_artifact"].__setitem__("artifact_sha256", "0" * 64),
                 id="raw_buffer_digest_disagrees"),
    pytest.param(lambda row: row["command_buffer_artifact"].__setitem__("command_buffer", None),
                 id="raw_buffer_withheld"),
    pytest.param(lambda row: row["work_volume"].__setitem__("exact_macs", _M * _K * _N + 1),
                 id="total_disagrees_with_the_program"),
    pytest.param(lambda row: row["work_volume"].__setitem__("basis", ""), id="unnamed_basis"),
])
def test_the_compute_axis_disappears_when_the_evidence_is_corrupted(tmp_path, monkeypatch, corrupt):
    """Each mutation is one way the pair could disagree; the reader must name no axis for any."""
    grade = _grade(tmp_path, monkeypatch, _countable_buffer(), run_id="wv_mutation")
    row = _measurement(grade)
    assert "compute" in FIXED._resource_bindings(row), "the unmutated row does not bind -- vacuous"
    corrupt(row)
    assert "compute" not in FIXED._resource_bindings(row)


def test_the_helper_refuses_a_buffer_that_is_not_serialisable_evidence():
    """A buffer that cannot be digested cannot be a receipt -- and must not crash the result write."""
    work, artifact = WV.command_buffer_evidence(
        {"tensors": {}, "commands": [], "opaque": object()}, compiler_provenance="test")
    assert work["exact_macs"] is None and artifact["command_buffer"] is None
    assert artifact["refusal"] and "priced" in artifact["refusal"]
