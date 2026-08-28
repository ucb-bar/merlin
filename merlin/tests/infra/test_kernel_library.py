"""The reuse ladder decides how much the kernel-generation arm is charged, so it must not be generous.

Every rule here fails in the same direction if it regresses: the kernel arm gets credited with reuse
it did not achieve, its cost curve flattens, and the crossover the study reports moves in favour of
the compiler. The load-bearing one is the demotion test -- a kernel with baked-in dimensions must
cost tokens, not be recorded as free.
"""
import json

import pytest

from merlin.benchharness import kernel_library as KL


def _entry(sig="matmul|contraction||f32,f32,f32|3|v2^16", config="C0", axes=(), cycles=100,
           family="contraction", regime="v2^16", path=None):
    return KL.Entry(signature=sig, config_id=config, kernel_path=path or f"/k/{sig}_{config}.mlir",
                    family=family, regime=regime, parametric_axes=tuple(axes), cycles=cycles)


SIG = "matmul|contraction||f32,f32,f32|3|v2^16"


# --- the ladder ---------------------------------------------------------------------------------

def test_an_exact_match_is_free_and_calls_no_model():
    lib = KL.KernelLibrary()
    lib.add(_entry())
    d = lib.propose(SIG, "C0")
    assert d.level == KL.EXACT and d.llm_called is False
    assert d.is_free_reuse() is True


def test_an_exact_match_is_still_re_evaluated_not_assumed():
    """Free means no model call, never an unverified pass."""
    lib = KL.KernelLibrary()
    lib.add(_entry())
    assert "exists" in lib.propose(SIG, "C0").rationale


def test_a_new_signature_costs_a_full_generation():
    lib = KL.KernelLibrary()
    d = lib.propose("something|else||f32|2|v2^4", "C0")
    assert d.level == KL.NEW and d.llm_called is True


def test_a_nearby_kernel_seeds_a_warm_start_and_still_costs_a_call():
    lib = KL.KernelLibrary()
    lib.add(_entry(sig=SIG, family="contraction", regime="v2^16"))
    d = lib.propose("conv|contraction||f32,f32,f32|4|v2^16", "C0",
                    family="contraction", regime="v2^16")
    assert d.level == KL.WARM_START
    assert d.llm_called is True, "a delta call is still a call"
    assert d.matched_entry is not None, "it must say what it was seeded from"


def test_an_entry_claiming_no_axes_is_not_offered_for_a_different_config():
    """Silence is not a promise of generality."""
    lib = KL.KernelLibrary()
    lib.add(_entry(config="C0", axes=()))
    d = lib.propose(SIG, "C2", config_axes=("m", "n"))
    assert d.level != KL.PARAMETRIC


def test_an_entry_must_claim_every_differing_axis():
    lib = KL.KernelLibrary()
    lib.add(_entry(config="C0", axes=("m",)))
    d = lib.propose(SIG, "C2", config_axes=("m", "k"))
    assert d.level != KL.PARAMETRIC, "claiming m does not cover a change in k"


def test_a_claimed_parametric_reuse_is_proposed_but_unconfirmed():
    lib = KL.KernelLibrary()
    lib.add(_entry(config="C0", axes=("m", "n", "k")))
    d = lib.propose(SIG, "C2", config_axes=("m", "n"))
    assert d.level == KL.PARAMETRIC
    assert d.pending_confirmation is True
    assert d.is_free_reuse() is False, "an unconfirmed claim must not count as reuse yet"


# --- the load-bearing rule ------------------------------------------------------------------------

def test_a_kernel_that_fails_the_new_config_is_demoted_and_charged():
    """A dim-baked kernel must cost tokens.

    Granting L1 on signature agreement alone would record its failure as free reuse -- fabricating
    precisely the advantage this module exists to measure honestly.
    """
    lib = KL.KernelLibrary()
    lib.add(_entry(config="C0", axes=("m", "n", "k")))
    d = lib.propose(SIG, "C2", config_axes=("m",))
    lib.confirm(d, passed=False)
    assert d.level == KL.NEW
    assert d.demoted_from == KL.PARAMETRIC
    assert d.llm_called is True
    assert d.is_free_reuse() is False


def test_a_kernel_that_survives_the_new_config_is_free_reuse():
    lib = KL.KernelLibrary()
    lib.add(_entry(config="C0", axes=("m", "n", "k")))
    d = lib.propose(SIG, "C2", config_axes=("m",))
    lib.confirm(d, passed=True)
    assert d.level == KL.PARAMETRIC and d.confirmed is True
    assert d.is_free_reuse() is True and d.llm_called is False


# --- metrics ---------------------------------------------------------------------------------

def test_generalization_depth_counts_passes_not_attempts():
    """Compiling is not surviving."""
    lib = KL.KernelLibrary()
    lib.add(_entry(config="C0", axes=("m", "n", "k")))
    lib.confirm(lib.propose(SIG, "C1", config_axes=("m",)), passed=True)
    lib.confirm(lib.propose(SIG, "C2", config_axes=("n",)), passed=False)
    assert lib.generalization_depth() == pytest.approx(0.5)


def test_generalization_depth_is_none_when_nothing_was_proposed():
    """No L1 attempted is not the same finding as every L1 failing."""
    assert KL.KernelLibrary().generalization_depth() is None


def test_a_pending_decision_is_excluded_from_the_reuse_rate():
    lib = KL.KernelLibrary()
    lib.add(_entry(config="C0", axes=("m",)))
    lib.propose(SIG, "C2", config_axes=("m",))     # left pending on purpose
    assert lib.reuse_rate() is None, "nothing has settled yet"


def test_the_reuse_rate_counts_only_settled_free_decisions():
    lib = KL.KernelLibrary()
    lib.add(_entry(config="C0", axes=("m",)))
    lib.propose(SIG, "C0")                                         # L0, free
    lib.confirm(lib.propose(SIG, "C2", config_axes=("m",)), passed=False)   # demoted
    assert lib.reuse_rate() == pytest.approx(0.5)


def test_the_matrix_reports_the_level_that_served_each_cell():
    lib = KL.KernelLibrary()
    lib.add(_entry(config="C0", axes=("m",)))
    lib.propose(SIG, "C0")
    lib.confirm(lib.propose(SIG, "C1", config_axes=("m",)), passed=True)
    m = lib.matrix()
    assert m[SIG]["C0"] == KL.EXACT and m[SIG]["C1"] == KL.PARAMETRIC


# --- the store ---------------------------------------------------------------------------------

def test_lookup_is_by_signature_not_by_configuration():
    """A config-keyed store is a lookup table and would measure nothing."""
    lib = KL.KernelLibrary()
    lib.add(_entry(config="C0"))
    assert lib.find(SIG) and lib.find(SIG, "C9") == []


def test_candidate_order_is_deterministic_and_prefers_the_faster_kernel():
    lib = KL.KernelLibrary()
    lib.add(_entry(config="C1", cycles=500, path="/k/slow.mlir"))
    lib.add(_entry(config="C2", cycles=100, path="/k/fast.mlir"))
    assert lib.find(SIG)[0].kernel_path == "/k/fast.mlir"


def test_an_unmeasured_kernel_does_not_outrank_a_measured_one():
    lib = KL.KernelLibrary()
    lib.add(_entry(config="C1", cycles=None, path="/k/unknown.mlir"))
    lib.add(_entry(config="C2", cycles=900, path="/k/known.mlir"))
    assert lib.find(SIG)[0].kernel_path == "/k/known.mlir"


# --- persistence and audit ------------------------------------------------------------------------

def test_the_decision_log_round_trips(tmp_path):
    lib = KL.KernelLibrary(tmp_path)
    lib.add(_entry(config="C0", axes=("m",)))
    lib.confirm(lib.propose(SIG, "C1", config_axes=("m",)), passed=True)
    lib.write()
    back = KL.KernelLibrary.read(tmp_path)
    assert len(back.entries) == 1 and len(back.decisions) == 1
    assert back.decisions[0].confirmed is True


def test_the_written_summary_is_json(tmp_path):
    lib = KL.KernelLibrary(tmp_path)
    lib.add(_entry())
    lib.propose(SIG, "C0")
    lib.write()
    assert json.loads((tmp_path / "reuse_summary.json").read_text())["by_level"][KL.EXACT] == 1


def test_a_ledger_that_disagrees_with_the_decision_log_is_surfaced():
    """Produced independently, so disagreement means one is wrong -- and silence would hide it."""
    lib = KL.KernelLibrary()
    lib.propose("a|f||f32|2|v2^4", "C0")        # L3, expects one call
    a = KL.audit_against_ledger(lib.decisions, ledger_calls=3)
    assert a["agrees"] is False and a["discrepancy"] == 2
    assert KL.audit_against_ledger(lib.decisions, ledger_calls=1)["agrees"] is True


def test_writing_without_a_root_is_refused():
    with pytest.raises(ValueError):
        KL.KernelLibrary().write()
