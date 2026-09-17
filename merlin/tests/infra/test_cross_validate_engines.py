"""A second elaborated-RTL engine may certify nothing until it is shown to agree with the established
one — and the shape of that proof is what these tests pin.

The load-bearing case is the third state. Two-state agreement (agree/disagree) has to put an engine that
could not run into one of them, and every version of that choice is wrong: "disagree" blames the engine
for the harness, "agree" adopts a second oracle on no evidence at all. The latter is the failure this
repo keeps paying for — a check that could not run reporting success — so it gets a test that says so by
name.

The second load-bearing case is WHAT agreement is made of. A shared verdict is not evidence: two engines
can both "pass" while reading back different bytes. The strong channel is the run's own result readback
(the shared ``OUT`` protocol lines), digested per output tensor, and the tests below pin that a byte
divergence under a shared verdict is a DISAGREE and that missing byte evidence is UNDETERMINABLE.
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

from merlin.common.paths import merlin_dir

_SCRIPT = merlin_dir() / "experiments/capsule_bench/harness/cross_validate_engines.py"


@pytest.fixture(scope="module")
def X():
    """Import the harness script by path — it lives outside any importable package.

    Registered in ``sys.modules`` BEFORE it is executed: the module uses ``from __future__ import
    annotations``, so ``dataclasses`` resolves each field's annotation by looking its module up there,
    and an unregistered module makes every ``@dataclass`` in the file raise at class-creation time.
    """
    name = "_cross_validate_engines"
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _capsule(name, tiers=("L0", "L1", "L3")):
    return {"name": name, "required_oracle_tiers": list(tiers), "__dir__": f"/corpus/{name}"}


def _run(X, engine, verdict="pass", chk=("0x1234",), wall=1.0, ran=True):
    return X.EngineRun(engine, ran, verdict, tuple(chk), {}, wall, "ok")


def _run_bytes(X, engine, outputs, verdict="pass", chk=(), wall=1.0, note=""):
    """An engine run whose evidence is the OUTPUT TENSORS it read back, digested as the tool digests
    them. ``outputs=None`` is an engine that produced no byte evidence at all."""
    digests = () if outputs is None else X.digest_outputs(outputs)
    return X.EngineRun(engine, True, verdict, tuple(chk), {}, wall, "ok", 0, digests, note)


# --------------------------------------------------------------------------------------------------
# Planning — pure: no subprocess, no simulator, no filesystem
# --------------------------------------------------------------------------------------------------
def test_plan_pairs_each_capsule_with_both_engines(X):
    plan = X.plan_cross_validation([_capsule("B1"), _capsule("A2")], target="t",
                                   reference_engine="verilator", candidate_engine="gsim",
                                   artifacts_root="/art")
    assert [e.capsule for e in plan.entries] == ["A2", "B1"]          # deterministic, reviewable order
    assert all(e.engines == ("verilator", "gsim") for e in plan.included())
    assert plan.total_engine_runs == 4                               # one ELF, two runs, per capsule
    assert plan.included()[0].artifact_dir.endswith("/art/A2")


def test_plan_touches_no_filesystem(X, tmp_path):
    """The artifact directory is COMPUTED, never opened: a plan must be reviewable before any simulator
    time is bought, and a plan that stats the tree cannot be built from a corpus listing alone."""
    missing = tmp_path / "definitely-absent"
    plan = X.plan_cross_validation([_capsule("A2")], target="t", reference_engine="verilator",
                                   candidate_engine="gsim", artifacts_root=missing)
    assert plan.included()[0].included is True
    assert not missing.exists()


def test_plan_excludes_capsules_that_never_reach_the_tier(X):
    """A capsule declaring only L0/L1/L2 is never certified at L3, so two L3 engines agreeing on it says
    nothing about how it is graded. Excluded WITH the reason — never silently dropped, and never counted
    into the agreement total."""
    plan = X.plan_cross_validation([_capsule("A2", tiers=("L0", "L1", "L2")), _capsule("B1")],
                                   target="t", reference_engine="verilator",
                                   candidate_engine="gsim", artifacts_root="/art")
    assert [e.capsule for e in plan.included()] == ["B1"]
    (excluded,) = plan.excluded()
    assert excluded.capsule == "A2" and "L3" in excluded.reason and excluded.engine_runs == 0


def test_undeclared_tier_can_be_included_but_only_deliberately(X):
    """Relaxing the rung is an EXPLICIT argument, not the default. The engines do execute the same ELF on
    such a capsule, so agreement there is real evidence about the ENGINES — it is simply not evidence
    about a rung the capsule is never certified at, and the two must not be conflated by default."""
    caps = [_capsule("A2", tiers=("L0", "L1", "L2"))]
    strict = X.plan_cross_validation(caps, target="t", reference_engine="verilator",
                                     candidate_engine="gsim", artifacts_root="/art")
    relaxed = X.plan_cross_validation(caps, target="t", reference_engine="verilator",
                                      candidate_engine="gsim", artifacts_root="/art",
                                      require_declared_tier=False)
    assert strict.included() == () and len(relaxed.included()) == 1


def test_plan_deduplicates_capsule_names(X):
    """Corpora are discovered from several roots; a duplicate would be counted twice in 'the engines
    agreed on N capsules'."""
    plan = X.plan_cross_validation([_capsule("A2"), _capsule("A2")], target="t",
                                   reference_engine="verilator", candidate_engine="gsim",
                                   artifacts_root="/art")
    assert len(plan.included()) == 1
    assert "duplicate" in plan.excluded()[0].reason


def test_plan_excludes_a_capsule_with_no_artifact_directory(X):
    plan = X.plan_cross_validation([_capsule("A2")], target="t", reference_engine="verilator",
                                   candidate_engine="gsim", artifacts={})
    assert plan.included() == ()
    assert "same ELF" in plan.excluded()[0].reason


def test_plan_refuses_to_compare_an_engine_with_itself(X):
    with pytest.raises(ValueError, match="comparing an engine with itself"):
        X.plan_cross_validation([_capsule("A2")], target="t", reference_engine="gsim",
                                candidate_engine="gsim", artifacts_root="/art")


# --------------------------------------------------------------------------------------------------
# CHK parsing — prefix/token matched, never positional, never coerced
# --------------------------------------------------------------------------------------------------
def test_chk_is_token_matched_through_an_interleaved_console(X):
    console = ("%Warning: System has stack size 12500 kb\n"
               "OUT Y0 1 1 42\n"
               "some-other-writer CHK 0x00c0ffee\n"
               "CHECK not-a-chk-value\n"
               "METRIC cycles 1090\n"
               "DONE\n")
    assert X.parse_chk(console) == ("0x00c0ffee",)               # CHECK is a different token


def test_chk_values_are_kept_as_raw_strings(X):
    """Not coerced. '0x10' and '16' are different bytes, and a numeric coercion would make two consoles
    that printed different things compare equal — the silent agreement this whole script exists to
    prevent."""
    assert X.parse_chk("CHK 0x10\n") != X.parse_chk("CHK 16\n")
    assert X.parse_chk("CHK 007\n") == ("007",)


def test_chk_absent_is_an_empty_reading_not_a_zero(X):
    assert X.parse_chk("OUT Y0 1 1 1\nDONE\n") == ()
    assert X.parse_chk("") == ()


# --------------------------------------------------------------------------------------------------
# Agreement — three states, and the third is not a shade of the first
# --------------------------------------------------------------------------------------------------
def test_identical_results_agree(X):
    c = X.compare_runs(_run(X, "verilator", wall=2700.0), _run(X, "gsim", wall=115.0), capsule="A2")
    assert c.agreement == X.AGREE
    assert c.chk_match is True
    assert c.speed_ratio == pytest.approx(23.478, abs=1e-3)      # candidate wall vs reference wall


def test_a_missing_engine_is_undeterminable_never_agreement(X):
    """The rule the script is built around. The reference PASSED; the candidate never ran. That is not
    agreement — it is no evidence — and calling it agreement adopts a second oracle for free."""
    c = X.compare_runs(_run(X, "verilator", verdict="pass"),
                       X.unavailable_run("gsim", "gsim reports unavailable for this backend"),
                       capsule="A2")
    assert c.agreement == X.UNDETERMINABLE
    assert c.agreement != X.AGREE
    assert c.chk_match is None
    assert "gsim" in c.reason


def test_an_engine_the_backend_does_not_know_is_undeterminable(X):
    """``available('gsim')`` RAISES on a backend with no such branch. Recorded as unavailability with the
    reason (as ``rtl_engine_policy.select`` treats a raising probe), and so undeterminable."""
    class _Raises:
        def available(self, engine):
            raise RuntimeError(f"unknown simulator {engine!r}")

    run = X.run_on_engine("gsim", "/tmp/x.elf", backend=_Raises(), grade=lambda o: True)
    assert run.ran is False and run.verdict == X.DID_NOT_RUN
    assert "unknown simulator" in run.detail
    c = X.compare_runs(_run(X, "verilator"), run, capsule="A2")
    assert c.agreement == X.UNDETERMINABLE


def test_both_engines_failing_is_agreement_about_the_oracles(X):
    """The claim under test is that the two oracles say the same thing — not that the capsule passes."""
    c = X.compare_runs(_run(X, "verilator", verdict="fail"), _run(X, "gsim", verdict="fail"),
                       capsule="A2")
    assert c.agreement == X.AGREE


def test_differing_verdicts_disagree(X):
    c = X.compare_runs(_run(X, "verilator", verdict="pass"), _run(X, "gsim", verdict="fail"),
                       capsule="A2")
    assert c.agreement == X.DISAGREE
    assert "verdicts differ" in c.reason


def test_differing_chk_disagrees_even_when_both_verdicts_pass(X):
    """Byte equality, no tolerance. Two engines over the same elaborated design running the same ELF must
    print the same check value; a pass/pass with different bytes is exactly the silent divergence a second
    oracle must never be allowed to hide."""
    c = X.compare_runs(_run(X, "verilator", chk=("0x00c0ffee",)),
                       _run(X, "gsim", chk=("0x00c0fffe",)), capsule="A2")
    assert c.agreement == X.DISAGREE
    assert c.chk_match is False


def test_chk_that_differs_only_in_spelling_disagrees(X):
    """No numeric coercion anywhere on the comparison path."""
    c = X.compare_runs(_run(X, "verilator", chk=("0x10",)), _run(X, "gsim", chk=("16",)), capsule="A2")
    assert c.agreement == X.DISAGREE and c.chk_match is False


def test_one_sided_chk_disagrees(X):
    """An engine that printed no check value cannot corroborate one that did."""
    c = X.compare_runs(_run(X, "verilator", chk=("0x1",)), _run(X, "gsim", chk=()), capsule="A2")
    assert c.agreement == X.DISAGREE
    assert "cannot corroborate" in c.reason


def test_no_chk_on_either_console_is_undeterminable_by_default(X):
    c = X.compare_runs(_run(X, "verilator", chk=()), _run(X, "gsim", chk=()), capsule="A2")
    assert c.agreement == X.UNDETERMINABLE
    assert c.chk_match is None
    # ...and waiving the checksum evidence is an EXPLICIT act, recorded in the reason.
    waived = X.compare_runs(_run(X, "verilator", chk=()), _run(X, "gsim", chk=()),
                            capsule="A2", require_chk=False)
    assert waived.agreement == X.AGREE and waived.chk_match is None
    assert "waived" in waived.reason


def test_speed_ratio_is_absent_when_an_engine_did_not_run(X):
    """Not 0.0, not inf: an engine that did not run took no measurable time, and either number would read
    as a speed."""
    c = X.compare_runs(_run(X, "verilator"), X.unavailable_run("gsim", "absent"), capsule="A2")
    assert c.speed_ratio is None


# --------------------------------------------------------------------------------------------------
# Execution — the same ELF reaches both engines, through the backend's own seam
# --------------------------------------------------------------------------------------------------
class _FakeBackend:
    """Stands in for the target backend at exactly the three seams the script uses."""

    def __init__(self, consoles, unavailable=()):
        self.consoles, self.unavailable = consoles, set(unavailable)
        self.saw: list[tuple[str, str]] = []                     # (engine, elf) pairs actually run

    def available(self, simulator):
        if simulator not in self.consoles and simulator not in self.unavailable:
            raise RuntimeError(f"unknown simulator {simulator!r}")
        return simulator not in self.unavailable

    def run_elf(self, elf, simulator="verilator", timeout=600):
        self.saw.append((simulator, str(elf)))
        return self.consoles[simulator]

    def parse_output(self, text):
        from merlin.runtime.backends.base import parse_console
        return parse_console(text, strip_warnings=True, tolerant_metric=True)


def _console(value=42, chk="0x2a"):
    return f"OUT Y0 1 1 {value}\nCHK {chk}\nMETRIC cycles 1090\nDONE\n"


def _plan_one(X):
    return X.plan_cross_validation([_capsule("A2")], target="t", reference_engine="verilator",
                                   candidate_engine="gsim", artifacts_root="/art")


def test_cross_validate_runs_one_elf_on_both_engines(X, tmp_path):
    backend = _FakeBackend({"verilator": _console(), "gsim": _console()})
    plan = _plan_one(X)
    got = X.cross_validate(plan, workdir=tmp_path, backend=backend,
                           build_elf=lambda e: (tmp_path / "A2.elf", {"tensors": {}}),
                           grader=lambda cb: (lambda outputs: outputs == {"Y0": [[42]]}),
                           log=lambda _m: None)
    assert [e for e, _ in backend.saw] == ["verilator", "gsim"]
    assert len({elf for _, elf in backend.saw}) == 1             # the SAME ELF, not two compilations
    assert got[0].agreement == X.AGREE and got[0].reference.counters == {}


def test_cross_validate_reports_a_console_divergence_as_disagree(X, tmp_path):
    backend = _FakeBackend({"verilator": _console(42, "0x2a"), "gsim": _console(43, "0x2b")})
    got = X.cross_validate(_plan_one(X), workdir=tmp_path, backend=backend,
                           build_elf=lambda e: (tmp_path / "A2.elf", {}),
                           grader=lambda cb: (lambda outputs: outputs == {"Y0": [[42]]}),
                           log=lambda _m: None)
    assert got[0].agreement == X.DISAGREE


def test_a_failed_shared_build_is_undeterminable_not_a_disagreement(X, tmp_path):
    """A build failure is a fact about the harness, not a verdict about whether the engines agree."""
    def _boom(entry):
        raise FileNotFoundError("command_buffer.json not found")

    got = X.cross_validate(_plan_one(X), workdir=tmp_path, backend=_FakeBackend({}),
                           build_elf=_boom, log=lambda _m: None)
    assert got[0].agreement == X.UNDETERMINABLE
    assert "could not build the shared ELF" in got[0].reason


def test_counter_lines_are_read_through_the_shared_reader(X, tmp_path):
    from merlin.perf.hw_counters import COUNTER_MARKER
    console = f"OUT Y0 1 1 42\nCHK 0x2a\n{COUNTER_MARKER} LD_CYCLES 7\nMETRIC cycles 9\nDONE\n"
    backend = _FakeBackend({"verilator": console, "gsim": console})
    got = X.cross_validate(_plan_one(X), workdir=tmp_path, backend=backend,
                           build_elf=lambda e: (tmp_path / "A2.elf", {}),
                           grader=lambda cb: (lambda outputs: True), log=lambda _m: None)
    assert got[0].candidate.counters == {"LD_CYCLES": 7}


# --------------------------------------------------------------------------------------------------
# Summary + exit codes
# --------------------------------------------------------------------------------------------------
def _cmp(X, name, agreement):
    ref = _run(X, "verilator")
    cand = {X.AGREE: _run(X, "gsim"),
            X.DISAGREE: _run(X, "gsim", verdict="fail"),
            X.UNDETERMINABLE: X.unavailable_run("gsim", "absent")}[agreement]
    return X.compare_runs(ref, cand, capsule=name)


def test_any_disagreement_fails_the_run(X):
    plan = _plan_one(X)
    s = X.summarize(plan, [_cmp(X, "A2", X.AGREE), _cmp(X, "B1", X.DISAGREE)])
    assert s.exit_code == X.EXIT_DISAGREE != X.EXIT_OK
    assert len(s.disagreed) == 1 and len(s.agreed) == 1


def test_undeterminable_is_reported_separately_from_disagreement(X):
    s = X.summarize(_plan_one(X), [_cmp(X, "A2", X.AGREE), _cmp(X, "B1", X.UNDETERMINABLE)])
    assert [c.capsule for c in s.undeterminable] == ["B1"]
    assert s.disagreed == ()
    assert s.exit_code == X.EXIT_INCOMPLETE                       # incomplete, and it says which kind
    text = X.render(s)
    assert "UNDETERMINABLE B1" in text and "NOT counted as agreement" in text


def test_a_run_that_proved_nothing_does_not_exit_zero(X):
    """Zero agreements and zero disagreements is the shape a green report takes when nothing ran."""
    assert X.summarize(_plan_one(X), []).exit_code == X.EXIT_INCOMPLETE


def test_all_agree_exits_zero(X):
    s = X.summarize(_plan_one(X), [_cmp(X, "A2", X.AGREE), _cmp(X, "B1", X.AGREE)])
    assert s.exit_code == X.EXIT_OK
    assert s.median_speed_ratio == 1.0


def test_allow_incomplete_waives_undeterminable_but_never_a_disagreement(X):
    und = X.summarize(_plan_one(X), [_cmp(X, "A2", X.AGREE), _cmp(X, "B1", X.UNDETERMINABLE)],
                      allow_incomplete=True)
    assert und.exit_code == X.EXIT_OK
    dis = X.summarize(_plan_one(X), [_cmp(X, "B1", X.DISAGREE)], allow_incomplete=True)
    assert dis.exit_code == X.EXIT_DISAGREE


# --------------------------------------------------------------------------------------------------
# Output-byte evidence — what agreement is actually MADE of
# --------------------------------------------------------------------------------------------------
def test_outputs_are_digested_per_tensor_and_sorted(X):
    """Per tensor, not one digest over everything: a divergence has to be able to NAME the output that
    diverged, and a single rolled-up digest can only say 'something'."""
    digests = X.digest_outputs({"Y1": [[3, 4]], "Y0": [[1, 2]]})
    assert [d.name for d in digests] == ["Y0", "Y1"]              # deterministic, name-sorted
    assert all(d.digest.startswith(X.OUTPUT_DIGEST_ALGO + ":") for d in digests)
    assert (digests[0].rows, digests[0].cols, digests[0].elements) == (1, 2, 2)


def test_a_digest_carries_no_output_VALUES(X):
    """The digest is the comparable. This tool must never republish result data — not the capsule's
    expected values (it never reads them) and not the engines' own tensors either."""
    (d,) = X.digest_outputs({"Y0": [[1234567, 7654321]]})
    blob = repr(d.to_dict())
    assert "1234567" not in blob and "7654321" not in blob


def test_reshaped_outputs_are_a_different_digest(X):
    """Same values, different shape, is a different result — never a silent match."""
    a = X.digest_outputs({"Y0": [[1, 2], [3, 4]]})[0]
    b = X.digest_outputs({"Y0": [[1, 2, 3, 4]]})[0]
    assert a.digest != b.digest


def test_int_and_float_outputs_do_not_collide(X):
    """Backends parse OUT values with ``int`` or ``float`` depending on the target's dtype; 1 and 1.0 are
    different readings and must not hash alike. Floats go through their IEEE bit pattern, so -0.0 is not
    0.0 either — a canonicalization that loses a distinction is how different bytes compare equal."""
    assert X.digest_outputs({"Y": [[1]]})[0].digest != X.digest_outputs({"Y": [[1.0]]})[0].digest
    assert X.digest_outputs({"Y": [[0.0]]})[0].digest != X.digest_outputs({"Y": [[-0.0]]})[0].digest


def test_a_tensor_that_cannot_be_canonicalized_raises_rather_than_hashing_a_repr(X):
    """Fail closed. An element whose byte identity is unknown yields NO digest, so the comparison has no
    byte evidence and says so — never a partial digest that two engines might match on by accident."""
    with pytest.raises(ValueError, match="cannot be canonicalized"):
        X.digest_outputs({"Y0": [["not-a-number"]]})
    with pytest.raises(ValueError, match="ragged"):
        X.digest_outputs({"Y0": [[1, 2], [3]]})


def test_identical_output_bytes_agree_on_the_strong_channel(X):
    c = X.compare_runs(_run_bytes(X, "verilator", {"Y0": [[1, 2]]}, wall=2700.0),
                       _run_bytes(X, "gsim", {"Y0": [[1, 2]]}, wall=115.0), capsule="A2")
    assert c.agreement == X.AGREE
    assert c.bytes_match is True and c.evidence == X.EV_OUTPUT_BYTES
    assert c.speed_ratio == pytest.approx(23.478, abs=1e-3)


def test_differing_output_bytes_disagree_and_name_the_tensor(X):
    """The whole point of a per-tensor digest: the report says WHICH output diverged."""
    c = X.compare_runs(_run_bytes(X, "verilator", {"Y0": [[1, 2]], "Y1": [[9]]}),
                       _run_bytes(X, "gsim", {"Y0": [[1, 3]], "Y1": [[9]]}), capsule="A2")
    assert c.agreement == X.DISAGREE and c.bytes_match is False
    assert "Y0" in c.reason and "1 of 2 tensor" in c.reason
    assert "Y1:" not in c.reason                                  # the tensor that matched is not blamed


def test_a_shared_verdict_does_not_hide_a_byte_divergence(X):
    """The failure this channel exists to catch. Both engines PASS, both consoles print the same CHK —
    and they read back different bytes. Verdict-only agreement calls that AGREE."""
    c = X.compare_runs(_run_bytes(X, "verilator", {"Y0": [[42]]}, chk=("0x2a",)),
                       _run_bytes(X, "gsim", {"Y0": [[43]]}, chk=("0x2a",)), capsule="A2")
    assert c.agreement == X.DISAGREE
    assert c.reference.verdict == c.candidate.verdict == X.PASS   # ...the verdicts agreed


def test_matching_bytes_with_a_diverging_chk_still_disagrees(X):
    """The console CHK stays a second channel once the bytes match: a difference the console can see is
    never quietly dropped just because the readback happened to match."""
    c = X.compare_runs(_run_bytes(X, "verilator", {"Y0": [[42]]}, chk=("0x2a",)),
                       _run_bytes(X, "gsim", {"Y0": [[42]]}, chk=("0xff",)), capsule="A2")
    assert c.agreement == X.DISAGREE and "consoles differ" in c.reason


def test_one_sided_output_bytes_disagree(X):
    """An engine that read back nothing cannot corroborate the tensors the other read back."""
    c = X.compare_runs(_run_bytes(X, "verilator", {"Y0": [[1]]}),
                       _run_bytes(X, "gsim", None, note="the run printed no OUT tensor"), capsule="A2")
    assert c.agreement == X.DISAGREE
    assert "cannot corroborate" in c.reason and "gsim" in c.reason


def test_missing_byte_evidence_is_undeterminable_and_never_agreement(X):
    """THE rule, restated for the byte channel. Neither engine read back an output tensor and neither
    console carried a CHK: there is nothing to compare, so nothing was proven. Both engines PASSED —
    and a pass is not evidence that they computed the same thing. A check that could not run must never
    report success."""
    c = X.compare_runs(_run_bytes(X, "verilator", None, note="the run printed no OUT tensor"),
                       _run_bytes(X, "gsim", None, note="the run printed no OUT tensor"), capsule="A2")
    assert c.agreement == X.UNDETERMINABLE
    assert c.agreement != X.AGREE
    assert c.bytes_match is None and c.evidence == X.EV_NONE
    assert "no output tensor" in c.reason
    # ...and it keeps a run from exiting 0 on evidence it never had.
    assert X.summarize(_plan_one(X), [c]).exit_code == X.EXIT_INCOMPLETE


def test_undigestible_outputs_are_undeterminable_not_agreement(X):
    """Fail closed end to end: outputs the digester refused leave the run with no byte evidence, the
    reason is carried, and the verdict is UNDETERMINABLE rather than a quiet pass on the verdicts."""
    class _Weird:
        def available(self, engine):
            return True

        def run_elf(self, elf, simulator="verilator", timeout=600):
            return "DONE\n"

        def parse_output(self, text):
            return {"Y0": [["not-a-number"]]}, {}

    runs = [X.run_on_engine(e, "/x.elf", backend=_Weird(), grade=lambda o: True)
            for e in ("verilator", "gsim")]
    assert all(r.ran and r.outputs == () for r in runs)
    assert "could not be digested" in runs[0].output_note
    c = X.compare_runs(runs[0], runs[1], capsule="A2")
    assert c.agreement == X.UNDETERMINABLE
    assert "could not be digested" in c.reason


def test_waiving_byte_evidence_is_explicit_and_recorded_as_the_weak_channel(X):
    c = X.compare_runs(_run_bytes(X, "verilator", None), _run_bytes(X, "gsim", None),
                       capsule="A2", require_chk=False)
    assert c.agreement == X.AGREE and c.evidence == X.EV_VERDICT_ONLY
    assert "waived" in c.reason


def test_chk_remains_the_fallback_when_no_tensor_was_read_back(X):
    """The console channel is kept, not replaced: a capsule whose ELF prints a CHK and no OUT tensor is
    still comparable, and the record says which channel carried the verdict."""
    c = X.compare_runs(_run_bytes(X, "verilator", None, chk=("0xc0ffee",)),
                       _run_bytes(X, "gsim", None, chk=("0xc0ffee",)), capsule="A2")
    assert c.agreement == X.AGREE and c.evidence == X.EV_CONSOLE_CHK
    assert c.chk_match is True and c.bytes_match is None


# --------------------------------------------------------------------------------------------------
# Byte evidence end to end, through the backend's own seams
# --------------------------------------------------------------------------------------------------
def _console_no_chk(value=42):
    return f"OUT Y0 1 1 {value}\nMETRIC cycles 1090\nDONE\n"


def test_engine_run_digests_the_readback_the_grader_sees(X, tmp_path):
    backend = _FakeBackend({"verilator": _console_no_chk(), "gsim": _console_no_chk()})
    got = X.cross_validate(_plan_one(X), workdir=tmp_path, backend=backend,
                           build_elf=lambda e: (tmp_path / "A2.elf", {}),
                           grader=lambda cb: (lambda outputs: True), log=lambda _m: None)
    (c,) = got
    assert c.agreement == X.AGREE and c.evidence == X.EV_OUTPUT_BYTES
    assert [d.name for d in c.reference.outputs] == ["Y0"]
    assert c.reference.outputs == c.candidate.outputs
    assert c.to_dict()["reference"]["outputs"][0]["digest"].startswith(X.OUTPUT_DIGEST_ALGO + ":")


def test_engines_that_both_pass_on_different_bytes_disagree_end_to_end(X, tmp_path):
    """A permissive grader passes both consoles — exactly the situation where a verdict comparison is
    blind. The readback is not."""
    backend = _FakeBackend({"verilator": _console_no_chk(42), "gsim": _console_no_chk(43)})
    (c,) = X.cross_validate(_plan_one(X), workdir=tmp_path, backend=backend,
                            build_elf=lambda e: (tmp_path / "A2.elf", {}),
                            grader=lambda cb: (lambda outputs: True), log=lambda _m: None)
    assert c.reference.verdict == c.candidate.verdict == X.PASS
    assert c.agreement == X.DISAGREE and "Y0" in c.reason


def test_a_console_with_no_tensor_and_no_chk_is_undeterminable_end_to_end(X, tmp_path):
    """The measured shape of the problem this change fixes: both engines pass, and there is no evidence
    that they computed the same thing. It stays UNDETERMINABLE and the run does not exit 0."""
    console = "METRIC cycles 1090\nDONE\n"
    backend = _FakeBackend({"verilator": console, "gsim": console})
    plan = _plan_one(X)
    got = X.cross_validate(plan, workdir=tmp_path, backend=backend,
                           build_elf=lambda e: (tmp_path / "A2.elf", {}),
                           grader=lambda cb: (lambda outputs: True), log=lambda _m: None)
    assert got[0].agreement == X.UNDETERMINABLE
    summary = X.summarize(plan, got)
    assert summary.exit_code == X.EXIT_INCOMPLETE
    assert summary.agreed == ()


def test_the_report_says_which_channel_each_agreement_rests_on(X, tmp_path):
    """'The engines agreed on N capsules' means something different per channel; the summary separates
    them so the weakest cannot be cited as the strongest."""
    strong = X.compare_runs(_run_bytes(X, "verilator", {"Y0": [[1]]}),
                            _run_bytes(X, "gsim", {"Y0": [[1]]}), capsule="A2")
    weak = X.compare_runs(_run_bytes(X, "verilator", None), _run_bytes(X, "gsim", None),
                          capsule="B1", require_chk=False)
    s = X.summarize(_plan_one(X), [strong, weak])
    assert s.evidence_census[X.EV_OUTPUT_BYTES] == 1
    assert s.evidence_census[X.EV_VERDICT_ONLY] == 1
    assert s.to_dict()["agreement_evidence"][X.EV_OUTPUT_BYTES] == 1
    text = X.render(s)
    assert "on output bytes 1" in text and "out-bytes" in text
