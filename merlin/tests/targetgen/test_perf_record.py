"""The performance record: provenance is required, diagnostics cannot become evidence, gaps stay gaps.

The suite these tests run against is an external, separately-pinned measurement artifact. When it is
not present the tests SKIP with the reason — a check that could not run is reported as not-run, never
as a pass.
"""
from __future__ import annotations

import json

import pytest

from merlin.common import provenance as prov
from merlin.perf import record as R
from merlin.perf.term import (UNKNOWN, UNKNOWN_TOKEN, PerformanceTerm, Provenance,
                              UnknownValueError, Validity)

#: The declared built artifact these records are about, and the pins it was elaborated from. Named
#: HERE, at the test edge, rather than in library code: `merlin/python/merlin/**` must not contain a
#: target-name literal, and the record builder threads all of this in as parameters.
ARTIFACT = "atlas_arc_cycle_suite"
PINS = ("atlas_npu", "atlas_npu_model", "atlas_fp_units")
TARGET = "atlas"
#: bucket : op-stream family : the metadata key holding the unit's structural dimension.
UNIT_MODELS = (R.UnitModel(bucket="mxu", family="MatrixSystolic", dim_key="mxu_dim"),)

_DIGEST_STUB = R.DigestTriple(sources="0" * 64, artifacts={"a": "1" * 64}, pins={"p": "2" * 40})


def _prov(kind: str = "measured", evidence=("unit_activity",)) -> Provenance:
    return Provenance(kind=kind, evidence=tuple(evidence))


def _validity() -> Validity:
    return Validity(validated_regime="one run to halt")


# ---------------------------------------------------------------------------------------------
# fixtures: the live measurement artifact
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def suite_path():
    try:
        arts = prov.load_artifacts()
    except prov.PinsError as exc:                                    # pragma: no cover
        pytest.skip(f"pin registry unreadable: {exc}")
    if ARTIFACT not in arts:                                         # pragma: no cover
        pytest.skip(f"no built artifact {ARTIFACT!r} declared in the pin registry")
    path = arts[ARTIFACT].resolve()
    if path is None or not path.is_file():                           # pragma: no cover
        pytest.skip(f"the measured cycle suite is not on this host ({path}); DID NOT RUN")
    return path


@pytest.fixture(scope="module")
def suite(suite_path):
    return R.load_suite(suite_path)


@pytest.fixture(scope="module")
def records(suite, suite_path):
    digest = R.read_digest_triple(pin_names=PINS, artifact_names=[ARTIFACT], sources=[suite_path])
    return {r.kernel: r for r in R.build_records(suite, target=TARGET, digest=digest,
                                                 unit_models=UNIT_MODELS)}


# ---------------------------------------------------------------------------------------------
# 1. the digest triple is required from the first record written
# ---------------------------------------------------------------------------------------------


def test_a_record_without_a_digest_raises():
    """Not a warning. A result that cannot say what hardware it is about is uncitable, and the
    provenance cannot be reconstructed afterwards."""
    with pytest.raises(R.MissingDigestError):
        R.PerformanceRecord(kernel="k", target=TARGET, digest=None)


@pytest.mark.parametrize("kwargs, why", [
    ({"sources": "", "artifacts": {"a": "1" * 64}, "pins": {"p": "2" * 40}}, "no source digest"),
    ({"sources": "0" * 64, "artifacts": {}, "pins": {"p": "2" * 40}}, "no built artifact"),
    ({"sources": "0" * 64, "artifacts": {"a": "1" * 64}, "pins": {}}, "no pin"),
    ({"sources": "0" * 63, "artifacts": {"a": "1" * 64}, "pins": {"p": "2" * 40}}, "short digest"),
    ({"sources": "0" * 64, "artifacts": {"a": UNKNOWN_TOKEN}, "pins": {"p": "2" * 40}},
     "an UNKNOWN artifact digest"),
    ({"sources": "0" * 64, "artifacts": {"a": "1" * 64}, "pins": {"p": "2" * 7}},
     "an abbreviated commit"),
])
def test_an_incomplete_digest_triple_raises(kwargs, why):
    with pytest.raises(R.MissingDigestError):
        R.DigestTriple(**kwargs)


def test_a_record_whose_digest_was_removed_never_reaches_the_disk(tmp_path):
    rec = R.PerformanceRecord(kernel="k", target=TARGET, digest=_DIGEST_STUB)
    rec.digest = None                                    # simulate a caller stripping it after build
    with pytest.raises(R.MissingDigestError):
        rec.write(tmp_path / "k.json")
    assert not (tmp_path / "k.json").exists()


def test_the_emitted_digest_triple_verifies_against_the_live_checkouts(records):
    digest = next(iter(records.values())).digest
    assert len(digest.sources) == 64
    assert set(digest.pins) == set(PINS)
    assert digest.artifacts[ARTIFACT] == prov.verify_artifact(ARTIFACT).digest
    for name, commit in digest.pins.items():
        assert commit == prov.pin(name).commit
        assert len(commit) == 40
    # Declared local edits are drift-by-design on these pins, so the record must CARRY that note
    # rather than presenting a bare commit as if it described the bytes.
    assert digest.notes, "pins with declared local edits must record that the bytes differ"


# ---------------------------------------------------------------------------------------------
# 2. an UNKNOWN term cannot be read as 0.0
# ---------------------------------------------------------------------------------------------


def test_overlap_is_unknown_not_zero(records):
    """The activity buckets PARTITION the cycles, so they cannot measure overlap. Recording 0 would
    be a wrong number wearing a measurement's clothes."""
    for kernel, rec in records.items():
        term = rec.terms["overlap_cycles"]
        assert term.is_unknown, kernel
        assert term.value is UNKNOWN
        assert term.value != 0 and term.value != 0.0
        assert term.unknown_reason
        with pytest.raises(UnknownValueError):
            _ = float(term.value)
        with pytest.raises(UnknownValueError):
            _ = term.value + rec.terms["total_cycles"].value
        with pytest.raises(UnknownValueError):
            _ = float(term.value or 0)                 # the exact idiom that would publish a zero
        # It is still bounded: the Amdahl cap is derivable even though the value is not.
        assert term.bounds.lower == 0
        assert term.bounds.upper is not UNKNOWN


def test_the_activity_buckets_partition_so_overlap_is_unmeasurable(records):
    """The reason overlap is UNKNOWN, asserted as data: bucket sum - total is a constant on all 21."""
    residuals = {k: rec.terms["activity_partition_residual"].value for k, rec in records.items()}
    assert len(records) == 21
    assert set(residuals.values()) == {1}, residuals


def test_unknown_terms_survive_the_json_round_trip(records, tmp_path):
    rec = records["smolvla_matmul_k_chain"]
    path = rec.write(tmp_path / "k.json")
    back = R.PerformanceRecord.from_dict(json.loads(path.read_text()))
    assert back.to_dict() == rec.to_dict()
    for name, term in rec.terms.items():
        if term.is_unknown:
            assert back.terms[name].value is UNKNOWN
            assert back.terms[name].unknown_reason == term.unknown_reason


# ---------------------------------------------------------------------------------------------
# 3. the composed prediction reproduces the measured numbers exactly
# ---------------------------------------------------------------------------------------------


def test_composed_prediction_reproduces_the_measured_unit_busy_exactly(records):
    """`busy = per drained result: structural fill + the compute delays the program schedules`.

    Zero fitting: the fill is 2*D-2 from the design's own dimension, the delays are the program's.
    """
    assert records["matmul"].terms["predicted_busy_cycles.mxu"].value == 158
    assert records["matmul"].terms["busy_cycles.mxu"].value == 158
    for kernel in ("smolvla_matmul", "smolvla_fused_matmul_bias", "gemma_attention"):
        assert records[kernel].terms["predicted_busy_cycles.mxu"].value == 158, kernel
    for kernel in ("gemma_mlp", "smolvla_attention"):
        assert records[kernel].terms["predicted_busy_cycles.mxu"].value == 316, kernel


def test_the_prediction_is_exact_or_honestly_unknown_on_every_kernel(records):
    exact, unknown = [], []
    for kernel, rec in records.items():
        pred = rec.terms["predicted_busy_cycles.mxu"]
        if pred.is_unknown:
            unknown.append(kernel)
            continue
        assert pred.value == rec.terms["busy_cycles.mxu"].value, kernel
        exact.append(kernel)
    assert len(exact) == 20
    assert unknown == ["smolvla_matmul_k_chain"]


def test_the_k_accumulate_point_is_recorded_measured_and_predicted_unknown(records):
    """284 is RECORDED as the measurement; it is not fitted, and the law is not bent to reach it.

    The single-tile law extends naively to `fill + 2*completion = 254`, which is wrong by 30. Two K
    points show the law is wrong without showing what is right, so the prediction refuses.
    """
    rec = records["smolvla_matmul_k_chain"]
    measured = rec.terms["busy_cycles.mxu"]
    assert measured.value == 284
    assert measured.provenance.kind == "measured"

    pred = rec.terms["predicted_busy_cycles.mxu"]
    assert pred.is_unknown
    assert pred.value is not None and pred.value != 254 and pred.value != 0
    with pytest.raises(UnknownValueError):
        _ = int(pred.value)
    assert "accumulate" in pred.unknown_reason
    assert pred.validity.escalate_when == "more than one compute op per drained result"
    # The structural floor still holds even where the law does not: one drained result, one fill.
    assert pred.bounds.lower == 62
    assert measured.value > pred.bounds.lower


def test_the_fill_and_the_roles_are_derived_not_named(suite):
    """No mnemonic, no opcode and no geometry is written down here: all three come from the data."""
    streams = [k["op_stream"] for k in suite["kernels"].values()]
    delay = R.derive_delay_mnemonic(streams)
    roles = R.derive_unit_roles(streams, UNIT_MODELS[0].family, delay)
    assert roles.compute != roles.drain
    assert roles.compute_delay > 0
    assert R.fill_cycles("systolic_2d", suite["_meta"][UNIT_MODELS[0].dim_key]) == 62
    with pytest.raises(R.FillLawError):
        R.fill_cycles("guessed", 32)


def test_an_ambiguous_corpus_refuses_rather_than_guessing():
    """Fail closed: a stream that does not identify its delay marker gets no schedule, not a zero."""
    with pytest.raises(ValueError, match="delay marker"):
        R.derive_delay_mnemonic([[["U", "a", 3], ["U", "b", 4]]])
    with pytest.raises(ValueError, match="no ops of family"):
        R.derive_unit_roles([[["U", "a", 0], ["S", "delay", 5]]], "OTHER", "delay")


# ---------------------------------------------------------------------------------------------
# 4. the peer cost model is a diagnostic and can never source a term
# ---------------------------------------------------------------------------------------------


def test_peer_model_numbers_are_diagnostics_not_terms(records):
    for kernel, rec in records.items():
        assert "peer_model_cycles" in rec.diagnostics, kernel
        assert "peer_model_unit_stats" in rec.diagnostics, kernel
        assert rec.sources[R.SRC_PEER_MODEL].role == R.DIAGNOSTIC
        assert not rec.sources[R.SRC_PEER_MODEL].citable
        for name in rec.terms:
            assert "peer" not in name, kernel
        for term in rec.terms.values():
            assert R.SRC_PEER_MODEL not in term.provenance.evidence


def test_a_term_citing_a_diagnostic_source_is_rejected(records):
    rec = records["matmul"]
    bad = PerformanceTerm(name="total_cycles_from_the_peer_model",
                          value=rec.diagnostics["peer_model_cycles"].value, unit="cycles",
                          provenance=_prov("measured", (R.SRC_PEER_MODEL,)),
                          validity=_validity())
    with pytest.raises(R.DiagnosticSourceError):
        rec.add_term(bad)
    assert "total_cycles_from_the_peer_model" not in rec.terms


def test_the_peer_model_disagrees_with_the_measured_truth_by_up_to_three_times(records):
    """The reason it is disqualified as evidence, asserted rather than asserted-about.

    TWO CORRECTIONS to the number this task was handed, both measured from the artifact:

    * The (3972 vs 1273) pair is ``gemma_rms_norm``, not ``smolvla_rms_norm`` -- whose peer/truth
      pair is (4895 vs 3282), a 1.49x disagreement.
    * "up to 3x" UNDERSTATES it. The worst disagreements are the elementwise kernels at 4.92x
      (3387 vs 688), i.e. the short kernels where the peer model's fixed start-up term dominates.
      gemma_rms_norm's 3.12x is the worst among the LONG kernels, which is probably where the 3x
      came from.

    Being non-uniform is what disqualifies it: a model wrong by a constant factor could be
    corrected, whereas one that is exact on some kernels (1.005x) and 4.9x out on others cannot
    source anything.
    """
    ratios = {}
    for kernel, rec in records.items():
        peer = rec.diagnostics["peer_model_cycles"].value
        truth = rec.terms["total_cycles"].value
        assert peer is not UNKNOWN, kernel
        ratios[kernel] = peer / truth
    worst = max(ratios, key=ratios.get)
    assert worst.endswith("elementwise_add")
    assert records[worst].diagnostics["peer_model_cycles"].value == 3387
    assert records[worst].terms["total_cycles"].value == 688
    assert ratios[worst] == pytest.approx(4.92, abs=0.01)

    assert records["gemma_rms_norm"].diagnostics["peer_model_cycles"].value == 3972
    assert records["gemma_rms_norm"].terms["total_cycles"].value == 1273
    assert ratios["gemma_rms_norm"] > 3.0
    assert ratios["smolvla_rms_norm"] == pytest.approx(1.49, abs=0.01)

    # Non-uniform: near-exact on some kernels, 4.9x out on others.
    assert min(ratios.values()) < 1.01
    assert max(ratios.values()) > 4.9


def test_an_absent_diagnostic_is_unknown_not_zero(records):
    """The artifact carries no per-execution-unit statistics. Absent must not read as 'all zero'."""
    for kernel, rec in records.items():
        stats = rec.diagnostics["peer_model_unit_stats"]
        assert stats.value is UNKNOWN, kernel
        with pytest.raises(UnknownValueError):
            _ = float(stats.value)
        assert stats.to_dict()["value"] == UNKNOWN_TOKEN


def test_a_diagnostic_may_not_be_declared_citable():
    rec = R.PerformanceRecord(kernel="k", target=TARGET, digest=_DIGEST_STUB,
                              sources={"s": R.Source(id="s", role=R.CITABLE)})
    with pytest.raises(ValueError, match="cannot leak into a term"):
        rec.add_diagnostic(R.Diagnostic(name="d", value=1, unit="cycles", source="s"))


# ---------------------------------------------------------------------------------------------
# 5. the record as a whole
# ---------------------------------------------------------------------------------------------


def test_every_kernel_gets_a_schema_valid_record(records):
    assert len(records) == 21
    for kernel, rec in records.items():
        obj = rec.to_dict()
        R.validate_record(obj)                                   # raises ContractViolation if not
        assert obj["target"] == TARGET
        assert obj["kernel"] == kernel
        assert obj["digest"]["sources"]
        assert rec.terms["total_cycles"].value > 0
        assert rec.workload["footprint_bytes"] is not None


def test_moved_bytes_are_the_bytes_moved_not_the_bytes_needed(records):
    """Beats times beat width, from the measurement; a bound built on algorithmic bytes would be
    optimistic by the transfer-amplification factor."""
    rec = records["matmul"]
    assert rec.terms["moved_bytes"].value == 65536
    assert "MOVED" in rec.terms["moved_bytes"].validity.weak_regime


def test_terms_are_named_identically_across_kernels(records):
    """Inner names shared across the suite (and across targets) so a diff is trivial."""
    shapes = {frozenset(rec.terms) for rec in records.values()}
    assert len(shapes) == 1, shapes


# ---------------------------------------------------------------------------------------------
# 6. the anti-overfit bar: the same code, a different target, different correct answers
# ---------------------------------------------------------------------------------------------

#: A synthetic second target that shares NOTHING lexical with the first: different mnemonics,
#: a different delay marker, a different structural-dimension key, a different beat width, and a
#: differently-named activity bucket. If any of those were baked into the module rather than
#: derived, this fixture would fail rather than quietly agree.
_OTHER_TARGET_SUITE = {
    "_meta": {"beat_bytes": 16, "pe_side": 8},
    "kernels": {
        "one_tile": {
            "npu_cycles": 100, "footprint_bytes": 256,
            "op_stream": [["Q", "push.w", 0], ["S", "stall", 7], ["Q", "mac", 0], ["S", "stall", 40],
                          ["Q", "pop", 0], ["S", "stall", 7]],
            "arc": {"truth": 90, "dma_busy": 30, "q": 54, "none": 7, "reads": 8, "writes": 8,
                    "halt_reason": 1}},
        "accumulated": {
            "npu_cycles": 200, "footprint_bytes": 512,
            "op_stream": [["Q", "push.w", 0], ["S", "stall", 7], ["Q", "mac", 0], ["S", "stall", 40],
                          ["Q", "mac.acc", 0], ["S", "stall", 40], ["Q", "pop", 0], ["S", "stall", 7]],
            "arc": {"truth": 150, "dma_busy": 60, "q": 84, "none": 7, "reads": 16, "writes": 16,
                    "halt_reason": 1}},
    },
}


def test_the_same_code_produces_different_correct_answers_on_another_target():
    """A tool that only works where it was written is manual overfitting with extra steps."""
    recs = {r.kernel: r for r in R.build_records(
        _OTHER_TARGET_SUITE, target="toy_npu", digest=_DIGEST_STUB,
        unit_models=[R.UnitModel(bucket="q", family="Q", dim_key="pe_side")])}

    # A different geometry gives a different fill (2*8-2 = 14), so a different -- and exact -- answer.
    assert R.fill_cycles("systolic_2d", 8) == 14
    assert recs["one_tile"].terms["predicted_busy_cycles.q"].value == 54 == 14 + 40
    assert recs["one_tile"].terms["busy_cycles.q"].value == 54
    # A different beat width gives different moved bytes.
    assert recs["one_tile"].terms["moved_bytes"].value == (8 + 8) * 16

    # And the same regime boundary still refuses, on mnemonics it has never seen.
    assert recs["accumulated"].terms["predicted_busy_cycles.q"].is_unknown
    assert recs["accumulated"].terms["busy_cycles.q"].value == 84


def test_the_derivation_is_lexically_independent_of_the_first_target():
    streams = [k["op_stream"] for k in _OTHER_TARGET_SUITE["kernels"].values()]
    assert R.derive_delay_mnemonic(streams) == "stall"
    roles = R.derive_unit_roles(streams, "Q", "stall")
    assert (roles.compute, roles.drain, roles.compute_delay) == ("mac", "pop", 40)
