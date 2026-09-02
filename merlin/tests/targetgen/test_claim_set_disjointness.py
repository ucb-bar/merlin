"""The generalization claim must be measured over models that did not help build the corpus.

`conformance.required_cells` derives the requirement from what real captures CONTAIN; the synthesized
corpus covers those cells; coverage is then reported over captured models. One capture doing both jobs
makes the claim circular -- the corpus was built from the model it is said to generalize to. Measured
before the split was declared: the coverage gate fed EVERY bundle under `out/artifacts/recaptures/`
into the derivation, the four claim models included, and lstmnetvit was already in both roles.

Two properties are separate on purpose. Disjointness is FORM: no bundle in both sets. Independence is
SUBSTANCE: the requirement derived from the derivation set alone is the same one. Form without
substance is satisfiable by holding out nothing that mattered; substance is the property a reader of
the claim actually needs.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from merlin.common.paths import artifacts_dir, repo_root
from merlin.targetgen import claim_models as CM


def _gate():
    p = repo_root() / "build_tools" / "scripts" / "check_claim_set_disjointness.py"
    spec = importlib.util.spec_from_file_location("_claim_gate", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_claim_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_declaration_names_the_four_claim_models():
    assert set(CM.claim_models()) == {"resnet50_v1_5", "lstmnetvit", "smolvla", "tiny_llama"}
    assert CM.exclusion_rule(), "the standard a reviewer applies must be stated, not implied"
    assert "claim_captures" in CM.forbidden_sources(), (
        "a name is not enough to exclude a model; the artifact classes carrying its facts must be named")


def test_matching_is_on_token_boundaries_not_substrings():
    """Substring matching is wrong in BOTH directions, and each direction has a real example.

    Too loose: `small_llama_fp32_consistent` shares the token `llama` with the claim model
    `tiny_llama`, and excluding it would silently shrink the derivation set by a model nobody held out.
    Too tight is the mirror: a bundle whose model matches must be caught however it is suffixed.
    """
    assert CM.model_of("tiny_llama_fp32_full") == "tiny_llama"
    assert CM.model_of("lstmnetvit_int8_consistent") == "lstmnetvit"
    assert CM.model_of("resnet50_v1_5_fp32_consistent") == "resnet50_v1_5"

    assert CM.model_of("small_llama_fp32_consistent") is None, "shares a token, is not a claim model"
    assert CM.model_of("gemma2_2b_int8_full") is None
    assert CM.model_of("lstmnetvit2_fp32") is None, "a longer first token is a different model"
    assert CM.model_of("tiny_llamaX_fp32") is None


def test_the_partition_puts_every_bundle_on_exactly_one_side():
    caps = {"gemma2_2b_int8_full": Path("a"), "tiny_llama_fp32_full": Path("b"),
            "small_llama_int8_consistent": Path("c"), "smolvla_fp32_consistent": Path("d")}
    derivation, claim = CM.partition(caps)
    assert set(derivation) | set(claim) == set(caps)
    assert not (set(derivation) & set(claim))
    assert set(claim) == {"tiny_llama_fp32_full", "smolvla_fp32_consistent"}


def test_an_uncaptured_claim_model_is_reported_with_no_bundles():
    """Not omitted. A claim measured over a model nobody captured reads exactly like a passing one."""
    covered = CM.covered_claim_models(["gemma2_2b_int8_full", "tiny_llama_fp32_full"])
    assert set(covered) == set(CM.claim_models()), "every declared model must appear"
    assert covered["tiny_llama"] == ["tiny_llama_fp32_full"]
    assert covered["smolvla"] == [], "an uncaptured claim model is an empty list, not a missing key"


def test_the_coverage_gate_derives_from_the_derivation_set_only():
    """The behavioural half: the gate that produces a requirement must not read a held-out model."""
    gate_path = repo_root() / "build_tools" / "scripts" / "check_conformance_coverage.py"
    spec = importlib.util.spec_from_file_location("_cov_gate", gate_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_cov_gate"] = mod
    spec.loader.exec_module(mod)
    if not (artifacts_dir() / "recaptures").is_dir():
        pytest.skip("no capture store")

    derivation = mod._captures()
    everything = mod._captures(include_claim_models=True)
    assert derivation, "the derivation set must not be empty"
    assert not [n for n in derivation if CM.is_claim_bundle(n)], (
        f"held-out models reached requirement derivation: "
        f"{[n for n in derivation if CM.is_claim_bundle(n)]}")
    assert len(everything) > len(derivation), (
        "the unfiltered view must actually include the claim models, or the filter is untested")


def test_the_gate_reports_the_requirement_as_independent_of_the_held_out_models():
    """Independence is the substantive property, and it is MEASURED, not assumed.

    It happens to hold on this corpus -- the derivation captures already contain every cell the claim
    models do -- so declaring the split costs no requirement. That stops being true the moment a claim
    model is the only capture carrying some family, which is exactly why it is a gate.
    """
    gate = _gate()
    bundles = gate._bundles()
    if not bundles:
        pytest.skip("no capture store")
    targets = gate._spec_targets()
    if not targets:
        pytest.skip("no target ships a conformance spec")
    for t in targets:
        row = gate.audit(t, bundles)
        if row["status"] in ("no_captures", "unverifiable"):
            pytest.skip(f"{t}: {row.get('detail')}")
        assert row["status"] == "ok", f"{t}: {row.get('detail')}"
        assert row["requirement_is_independent"] is True
        assert row["cells_depending_on_a_claim_model"] == []
        assert row["n_cells_derivation_only"] > 0, (
            "a derivation that yields no cell has established nothing")


def test_holding_out_everything_is_not_a_pass():
    """The trivial way to satisfy disjointness must be rejected, not congratulated."""
    gate = _gate()
    only_claim = {"tiny_llama_fp32_full": Path("a"), "smolvla_fp32_consistent": Path("b")}
    row = gate.audit("gemmini", only_claim)
    assert row["status"] == "empty_derivation"
    assert "means nothing" in row["detail"]


def test_a_missing_capture_store_is_unverifiable_not_clean():
    gate = _gate()
    row = gate.audit("gemmini", {})
    assert row["status"] == "no_captures"
    assert "establishe" in row["detail"], "the report must say that nothing was established"


def test_the_conv_derivation_gap_is_recorded_rather_than_hidden():
    """Holding out the only convolution-dominated capture removes the conv branch. Say so.

    The honest response is a different derivation model, not re-admitting resnet50 -- and the two
    hand-authored gemmini conv capsules are the stand-in, which is a fact worth being able to find.
    """
    gaps = CM.known_derivation_gaps()
    assert gaps, "the split's known cost must be declared"
    conv = [g for g in gaps if g.get("shape_class") == "convolution"]
    assert conv, "the convolution gap is the one this split creates"
    assert conv[0].get("stood_in_by"), "name what currently covers it"


def test_an_unresolvable_contract_is_not_an_empty_requirement():
    """"Nothing admitted" has two causes and they license opposite actions.

    "This target's manifest admits no family a capture contains" is a final answer. "This target has no
    generated contract to read" is a missing artifact, and a requirement derived from it is UNKNOWN,
    not empty. `conformance.admitted` returns {} for both, which is fine only if callers can tell them
    apart -- and none could. Measured: saturn_opu and saturn_opu_rvv have no
    out/artifacts/targets/<t>/contracts/target_contract.yaml at all, derived zero cells, and this very
    gate reported both as `ok` with `independent=True`, because nothing can depend on a held-out model
    when nothing is required.
    """
    from merlin.targetgen import conformance as CF

    adm, why = CF.admitted_with_reason("gemmini")
    assert adm and why == "resolved"

    missing, why_missing = CF.admitted_with_reason("definitely_not_a_target")
    assert missing == {}
    assert why_missing.startswith("unresolvable:"), (
        f"an unresolvable contract must say so, got {why_missing!r}")


def test_a_zero_cell_derivation_is_never_reported_ok():
    """Independence is vacuously true over an empty requirement, so `ok` there is a false pass.

    This is the "a check that could not run reported success" failure this repo has paid for more than
    once -- and this gate shipped with it until saturn_opu showed 0 cells and a clean verdict.
    """
    gate = _gate()
    rep = gate.audit("definitely_not_a_target", {"gemma2_2b_int8_full": Path("a")})
    assert rep["status"] != "ok", f"a target with no contract must not be ok: {rep}"
    assert rep["status"] in ("contract_unresolved", "no_requirement", "unverifiable")
    assert rep.get("detail"), "the reason must travel with the verdict"


def test_the_written_specs_cite_no_held_out_model():
    """The artifact, not just the derivation: a tracked spec naming a claim model IS the circularity.

    gemmini's spec was derived before the split and cited lstmnetvit 28 times and tiny_llama 17 times
    as the evidence for its requirement. The cells were the same either way -- which is the
    independence result -- but the tracked evidence pointed at held-out models.
    """
    spec_dir = repo_root() / "merlin" / "contract" / "capsules" / "conformance"
    specs = sorted(spec_dir.glob("*.yaml"))
    if not specs:
        pytest.skip("no conformance spec is tracked")
    for spec in specs:
        text = spec.read_text(encoding="utf-8")
        cited = [m for m in CM.claim_models() if m in text]
        # `resnet50_v1_5` also matches a bundle label; check the model tokens, which is what a
        # citation would contain.
        assert not cited, (
            f"{spec.name} cites held-out model(s) {cited} as requirement evidence; regenerate it with "
            f"--write so the requirement is derived from the derivation set alone")
