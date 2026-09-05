"""Claim-bearing PK analysis is predeclared, exact-cell, and L3-only."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

from merlin.common.paths import repo_root


SCRIPTS = repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"
sys.path.insert(0, str(SCRIPTS))
import perf_pk_claim as PK  # noqa: E402


#: The replicate identities the shipped contract declares. Read from the analyzer rather than
#: written out here: this file previously hardcoded a triple, so changing the contract meant editing
#: the assertion that was supposed to be checking it.
REPLICATE_IDS = tuple(PK._ACCEPTANCE_BASE["replicates"]["identities"])


def _descriptor(index: int, k: int) -> dict:
    return {
        "name": f"PK{index:02d}_k{k}",
        "kind": "model_slice",
        "label": "dev",
        "inputs": [
            {"name": "W", "role": "weight", "shape": [k, 16], "dtype": "i8"},
            {"name": "A0", "role": "input", "shape": [16, k], "dtype": "i8"},
        ],
        "operation": {
            "op": "matmul",
            "attributes": {
                "lhs": "A0", "weight": "W", "out": "Y0", "epilogue": [],
                "output_dtype": "i32",
            },
        },
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
        "required_oracle_tiers": ["L0", "L1", "L2", "L3"],
        "performance": {
            "level": "L1_tile",
            "family": "PK",
            "lever": "reduction_depth",
            "claim": "PREDICTS",
            "comparand": {
                "kind": "fitted_prediction", "against": "measured_cycles_same_member",
            },
            "falsifier": {
                "observation": "residual_cycles_by_K",
                "fires_when": "residuals_are_not_bounded_after_rate_and_intercept_fit",
                "negative_control": "fixed_M_and_N_across_all_K_points",
            },
            "acceptance": PK.supported_acceptance(),
        },
    }


@pytest.fixture
def descriptors() -> list[dict]:
    return [_descriptor(index, k) for index, k in enumerate((16, 32, 64, 128))]


def _rows(descriptors: list[dict]) -> list[dict]:
    timing_simulator = descriptors[0]["performance"]["acceptance"]["evidence"][
        "timing_simulator"]
    oracle_kind = descriptors[0]["performance"]["acceptance"]["evidence"][
        "timing_oracle_kind"]
    rows = []
    for descriptor in descriptors:
        k = descriptor["inputs"][1]["shape"][1]
        for replicate in REPLICATE_IDS:
            common = {
                "approach": "arm4",
                "correct": True,
                "tier_status": "pass",
                "grade_status": "pass",
                "numeric_status": "pass",
            }
            rows += [
                {
                    **common,
                    "identity": {
                        "family": "PK", "capsule": descriptor["name"],
                        "simulator": "spike", "replicate": replicate,
                    },
                    "tier": "L2", "purpose": "correctness_screen", "citable": False,
                    "cycles": None,
                },
                {
                    **common,
                    "identity": {
                        "family": "PK", "capsule": descriptor["name"],
                        "simulator": timing_simulator, "replicate": replicate,
                    },
                    "tier": "L3", "purpose": "performance_certification", "citable": True,
                    "cycles": 5 * k + 100,
                    "oracle": {"kind": oracle_kind, "derived_from_rtl": True,
                               "fidelity": "elaborated_rtl"},
                    "timing_identity": {
                        "simulator": timing_simulator, "oracle_kind": oracle_kind,
                        "fidelity": "elaborated_rtl",
                        "simulator_binary_sha256": "a" * 64,
                        "elaborated_firrtl_sha256": "b" * 64,
                        "program_elf_sha256": "c" * 64,
                    },
                },
            ]
    return rows


def _row(rows: list[dict], capsule: str, simulator: str, replicate: str = "r000") -> dict:
    return next(row for row in rows if row["identity"] == {
        "family": "PK", "capsule": capsule,
        "simulator": simulator, "replicate": replicate,
    })


def test_profile_predeclares_the_exact_supported_contract():
    profile = yaml.safe_load(
        (repo_root() / "merlin/contract/capsules/profiles/_perf.yaml").read_text())
    sweep = next(row for row in profile["sweeps"] if row["id"] == "PK")
    assert sweep["base"]["performance"]["acceptance"] == PK.supported_acceptance()
    assert sweep["axes"] == {"M": ["tile"], "N": ["tile"],
                             "K": ["tile", "2*tile", "4*tile", "8*tile"]}


def test_real_gemmini_generator_emits_preflight_ready_pk_descriptors(tmp_path):
    capsules = repo_root() / "merlin/contract/capsules"
    sys.path.insert(0, str(capsules))
    import generate_corpus as generator
    from merlin.targetgen import corpus_spec
    from merlin.targetgen.target_experiment import load_target_experiment

    profile = generator.load_profile("gemmini")
    target = load_target_experiment(
        repo_root() / "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml")
    binding = corpus_spec.derive_binding(target, profile.get("datapath", {}))
    entries = generator.expand_sweeps(
        profile, binding, trait_facts=generator._performance_facts("gemmini"))
    pk_entries = [entry for entry in entries
                  if (entry.get("performance") or {}).get("family") == "PK"]
    written = [generator._write_capsule(entry, binding, tmp_path) for entry in pk_entries]
    generated = [yaml.safe_load((Path(path) / "capsule.yaml").read_text()) for path in written]
    assert PK.preflight_pk_claim(generated)["status"] == "READY"


def test_preflight_records_fixed_cohort_and_the_declared_replicates(descriptors):
    result = PK.preflight_pk_claim(list(reversed(descriptors)))
    assert result["status"] == "READY"
    assert result["cohort"] == {
        "negative_control": "fixed_M_and_N_across_all_K_points",
        "operation": "matmul", "M": 16, "N": 16,
        "operand_dtype": "i8", "accum_dtype": "i32", "epilogue": [],
        "capsules": ["PK00_k16", "PK01_k32", "PK02_k64", "PK03_k128"],
        "K_values": [16, 32, 64, 128], "K_multipliers_of_tile": [1, 2, 4, 8],
        "replicates": list(REPLICATE_IDS),
    }
    assert len(result["expected_identities"]) == 4 * len(REPLICATE_IDS) * 2
    assert {row["simulator"] for row in result["expected_identities"]} == {"spike", "gsim"}


def test_verilator_is_supported_only_when_frozen_contract_selects_it(descriptors):
    selected = copy.deepcopy(descriptors)
    for descriptor in selected:
        descriptor["performance"]["acceptance"] = PK.supported_acceptance("verilator")
    preflight = PK.preflight_pk_claim(selected)
    assert preflight["status"] == "READY"
    assert {row["simulator"] for row in preflight["expected_identities"]} == {
        "spike", "verilator"}
    result = PK.analyze_pk_claim(selected, _rows(selected))
    assert result["status"] == "ESTABLISHED"
    assert result["evidence"]["timing_source"] == "verilator_L3_only"
    assert result["evidence"]["timing_oracle_kind"] == "rtl_verilator"


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda row: row["timing_identity"].__setitem__(
            "simulator_binary_sha256", "not-a-digest"), "simulator_binary_sha256"),
        (lambda row: row["timing_identity"].__setitem__(
            "elaborated_firrtl_sha256", "d" * 64), "one exact elaborated FIRRTL"),
        (lambda row: row["oracle"].__setitem__("kind", "rtl_verilator"),
         "selected elaborated-RTL oracle"),
    ],
)
def test_timing_identity_is_exact_and_bound_to_selected_gsim(
        descriptors, mutate, reason):
    rows = _rows(descriptors)
    mutate(_row(rows, "PK00_k16", "gsim"))
    result = PK.analyze_pk_claim(descriptors, rows)
    assert result["status"] == "REFUSED"
    assert reason in result["refusal_reasons"][0]


def test_exact_affine_positive_l3_evidence_establishes_pk(descriptors):
    result = PK.analyze_pk_claim(descriptors, _rows(descriptors))
    assert result["status"] == "ESTABLISHED"
    assert result["evidence"] == {
        "l2_correctness_rows_validated": len(descriptors) * len(REPLICATE_IDS),
        "l2_cycles_consumed": 0,
        "l3_positive_cycle_rows_consumed": len(descriptors) * len(REPLICATE_IDS),
        "timing_source": "gsim_L3_only",
        "timing_simulator": "gsim",
        "timing_oracle_kind": "rtl_gsim",
        "timing_fidelity": "elaborated_rtl",
        "simulator_binary_sha256": "a" * 64,
        "elaborated_firrtl_sha256": "b" * 64,
        "program_elf_sha256_by_observation": [
            {"capsule": descriptor["name"], "replicate": replicate,
             "sha256": "c" * 64}
            for descriptor in descriptors for replicate in REPLICATE_IDS
        ],
        "arm": "arm4",
    }
    assert result["fit"]["rate_cycles_per_K_element"] == pytest.approx(5.0)
    assert result["fit"]["intercept_cycles"] == pytest.approx(100.0)
    assert result["fit"]["r_squared"] == pytest.approx(1.0)
    assert result["fit"]["max_absolute_residual_cycles"] == pytest.approx(0.0)
    assert result["fit"]["rmse_cycles"] == pytest.approx(0.0)
    assert all(result["checks"].values())


def test_complete_non_affine_l3_evidence_refutes_instead_of_refusing(descriptors):
    rows = _rows(descriptors)
    for replicate in REPLICATE_IDS:
        _row(rows, "PK02_k64", "gsim", replicate)["cycles"] += 300
    result = PK.analyze_pk_claim(descriptors, rows)
    assert result["status"] == "REFUTED"
    assert result["refusal_reasons"] == []
    assert result["refutation_reasons"]
    assert result["fit"]["max_absolute_residual_cycles"] > 8
    assert not all(result["checks"].values())


def test_small_integer_cycle_phase_residual_stays_inside_predeclared_floor(descriptors):
    rows = _rows(descriptors)
    for replicate in REPLICATE_IDS:
        _row(rows, "PK01_k32", "gsim", replicate)["cycles"] += 4
    result = PK.analyze_pk_claim(descriptors, rows)
    assert result["status"] == "ESTABLISHED"
    assert 0 < result["fit"]["max_absolute_residual_cycles"] <= 8
    assert result["checks"]["every_residual_within_max_of_8_cycles_and_3_percent"] is True


def test_complete_constant_cycle_evidence_refutes_the_required_positive_rate(descriptors):
    rows = _rows(descriptors)
    for row in rows:
        if row["identity"]["simulator"] == "gsim":
            row["cycles"] = 100
    result = PK.analyze_pk_claim(descriptors, rows)
    assert result["status"] == "REFUTED"
    assert result["checks"]["positive_rate"] is False
    assert result["fit"]["r_squared"] is None


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda ds: ds.pop(), "exactly four"),
        (lambda ds: ds[2]["inputs"][1]["shape"].__setitem__(0, 8), "fixed field 'M'"),
        (lambda ds: ds[1]["inputs"][0].__setitem__("dtype", "i16"), "share one declared dtype"),
        (lambda ds: ds[3]["inputs"][1]["shape"].__setitem__(1, 96), "contracted K"),
        (lambda ds: ds[0]["performance"]["acceptance"]["thresholds"].__setitem__(
            "r_squared_min_inclusive", 0.9), "unsupported acceptance"),
        (lambda ds: ds[0]["performance"]["acceptance"]["thresholds"].__setitem__(
            "slope_cycles_per_K_element_min_exclusive", False), "unsupported acceptance"),
    ],
)
def test_malformed_or_non_cohort_descriptors_are_refused(descriptors, mutate, reason):
    changed = copy.deepcopy(descriptors)
    mutate(changed)
    result = PK.preflight_pk_claim(changed)
    assert result["status"] == "REFUSED"
    assert reason in result["refusal_reasons"][0]


def test_wrong_k_point_is_refused_even_when_four_points_remain(descriptors):
    changed = copy.deepcopy(descriptors)
    changed[-1]["name"] = "PK03_k256"
    changed[-1]["inputs"][0]["shape"][0] = 256
    changed[-1]["inputs"][1]["shape"][1] = 256
    result = PK.preflight_pk_claim(changed)
    assert result["status"] == "REFUSED"
    assert "exactly K=" in result["refusal_reasons"][0]


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda rows: rows.pop(), f"exactly {4 * len(REPLICATE_IDS) * 2}"),
        (lambda rows: rows.__setitem__(-1, copy.deepcopy(rows[-2])), "repeats identity"),
        (lambda rows: _row(rows, "PK00_k16", "spike").__setitem__("correct", False),
         "not a correct successful"),
        (lambda rows: _row(rows, "PK00_k16", "gsim").__setitem__("cycles", 0),
         "positive integer cycles"),
        (lambda rows: _row(rows, "PK00_k16", "gsim").__setitem__("cycles", True),
         "positive integer cycles"),
        (lambda rows: _row(rows, "PK00_k16", "gsim").__setitem__("tier", "L2"),
         "evidence semantics"),
        (lambda rows: _row(rows, "PK00_k16", "spike").__setitem__("cycles", 999999),
         "exposes citable cycles"),
        (lambda rows: _row(rows, "PK00_k16", "gsim").__setitem__("approach", "arm3"),
         "Arm4 L2/L3"),
        (lambda rows: _row(rows, "PK00_k16", "gsim", REPLICATE_IDS[-1])["identity"].__setitem__(
            "replicate", "r003"), "undeclared identity"),
    ],
)
def test_missing_duplicate_wrong_or_malformed_l2_l3_rows_are_refused(
        descriptors, mutate, reason):
    rows = _rows(descriptors)
    mutate(rows)
    result = PK.analyze_pk_claim(descriptors, rows)
    assert result["status"] == "REFUSED"
    assert result["fit"] is None
    assert reason in result["refusal_reasons"][0]


def test_rows_for_another_declared_family_are_never_consumed(descriptors):
    rows = _rows(descriptors)
    rows.append({"identity": {"family": "PF", "capsule": "PF00",
                              "simulator": "gsim", "replicate": "r000"},
                 "cycles": -999999})
    result = PK.analyze_pk_claim(descriptors, rows)
    assert result["status"] == "ESTABLISHED"
    assert result["fit"]["n_observations"] == len(descriptors) * len(REPLICATE_IDS)


def test_result_is_independent_of_input_order(descriptors):
    rows = _rows(descriptors)
    forward = PK.analyze_pk_claim(descriptors, rows)
    reverse = PK.analyze_pk_claim(list(reversed(descriptors)), list(reversed(rows)))
    assert reverse == forward
