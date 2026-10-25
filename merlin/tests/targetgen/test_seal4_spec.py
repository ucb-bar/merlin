"""Reviewed scheme declarations preserve v1 numerics and historical input custody."""

import importlib
import subprocess
from copy import deepcopy

import pytest
import yaml
from test_quant_recipe import _scheme_facet

from merlin.common.paths import repo_root
from merlin.targetgen import quant_recipe as QR
from merlin.targetgen.quantization_spec import resolve_quantization_schemes
from merlin.targetgen.software_spec import load_software_spec, readout_exactness_gate, validate_readout
from merlin.targetgen.target_experiment import declared_held_out_inputs, load_target_experiment

ROOT = repo_root()
TARGET = ROOT / "examples/gemmini/target"


def test_baseline_scheme_and_implicit_legacy_recipe_keep_identical_numerics():
    spec = load_software_spec(TARGET / "software-spec.yaml")
    path = {"readout": [{"kind": "scale", "granularity": "per_command", "columns_per_scale": 3}]}
    resolved, _ = resolve_quantization_schemes(
        spec, {"output_path": path, "dtype": "int8", "max_k_extension": 64, "weight_panel_width": 3}
    )
    baseline = resolved["schemes"][0]
    assert baseline["id"] == "q_pow2_tensor"
    assert baseline["scale_value"] == "power_of_two"
    assert baseline["readout_strategy"] == spec["numerical_semantics"]["readout"]["strategy"] == "exact_rational_host"
    assert baseline["weight_scale_granularity"] == baseline["activation_scale_granularity"] == "tensor"
    assert baseline["operand_sum_rounding"] == "single_rounding"
    assert baseline["exactness_gate"] == readout_exactness_gate("exact_rational_host")
    domain = spec["numerical_semantics"]["calibration_domain"]
    before = deepcopy(spec)
    legacy = QR.derive(_scheme_facet(3), calibration_domain=domain).to_dict()
    explicit = QR.derive(_scheme_facet(3), scheme=baseline, calibration_domain=domain).to_dict()
    for key in (
        "weight",
        "activation",
        "calibration_domain",
        "integer_sum_policy",
        "readout_numerics",
        "families",
        "bias_domain",
    ):
        assert explicit[key] == legacy[key]
    assert legacy["calibration_domain"]["scale_policy"] == "enclosing_power_of_two"
    assert explicit["recipe_sha256"] == legacy["recipe_sha256"]
    assert "finest" not in legacy["why"]["weight.granularity"]
    assert spec == before


def test_new_scheme_recipes_select_declared_observer_and_block_geometry():
    spec = load_software_spec(TARGET / "software-spec.yaml")
    context = {
        "output_path": {"readout": [{"kind": "scale", "granularity": "per_command", "columns_per_scale": 3}]},
        "dtype": "int8",
        "max_k_extension": 64,
        "weight_panel_width": 3,
    }
    resolved, _ = resolve_quantization_schemes(spec, context)
    for scheme in resolved["schemes"][1:]:
        assert scheme["graded"] is True
        assert scheme["weight_scale_granularity"] != "column"
        assert scheme["exactness_gate"] == readout_exactness_gate("f32_product_rne")
        if scheme["operand_sum_rounding"] == "per_operand":
            assert scheme["residual_scope"] == "every_residual_add"
            continue  # This capture requires explicit per-operand requantization nodes.
        recipe = QR.derive(
            _scheme_facet(3), scheme=scheme, calibration_domain=spec["numerical_semantics"]["calibration_domain"]
        ).to_dict()
        assert recipe["calibration_domain"]["scale_policy"] == "declared_observer"
        if scheme["weight_scale_granularity"] == "column_block":
            assert recipe["weight"]["channels_per_scale"] == 3


def test_int8_readout_has_complete_f32_definition_and_both_exactness_gates():
    spec = load_software_spec(TARGET / "software-spec.yaml")
    transfer = next(t for t in spec["transfer_contracts"]["declarations"] if t["id"] == "int8_readout")
    semantics = transfer["semantics"]
    for strategy in ("exact_rational_host", "f32_product_rne"):
        assert semantics["exactness_gate"][strategy] == readout_exactness_gate(strategy)
    definition = semantics["readout_definitions"]["f32_product_rne"]
    assert validate_readout(definition) == definition
    assert definition["exactness_bridge"] == "prove_fp32_products"


def test_historical_spec_retains_exact_parent_bytes_and_software_pin():
    historical = ROOT / "examples/gemmini/phase0/history/software-spec-seals1-3.yaml"
    parent = subprocess.check_output(["git", "show", "4b402e0c5:examples/gemmini/target/software-spec.yaml"], cwd=ROOT)
    assert historical.read_bytes() == parent
    current = (TARGET / "software-spec.yaml").read_bytes()
    assert current.splitlines()[:4] == parent.splitlines()[:4]
    assert (
        yaml.safe_load(current)["execution_provider_revision"] == yaml.safe_load(parent)["execution_provider_revision"]
    )
    variants = yaml.safe_load(current)["quantization"]["methodology"]["model_variants"]
    assert {row["model"] for row in variants}.isdisjoint({"resnet50", "tiny_llama"})
    old = load_software_spec(historical)
    assert {"resnet50", "tiny_llama"} <= {row["model"] for row in old["quantization"]["methodology"]["model_variants"]}


def test_held_out_inputs_are_masked_under_a_broad_source_grant():
    from merlin.targetgen.sandbox import bwrap as BW
    from merlin.targetgen.sandbox.answer_surfaces import answer_surfaces

    te = load_target_experiment(TARGET / "descriptor.yaml")
    required = {ROOT / path for path in te.held_out_inputs}
    surfaces = [s for s in answer_surfaces(te) if s.label.startswith("held-out-input:")]
    assert {s.path for s in surfaces} == required
    assert ROOT / "merlin/contract/capsules/_model_layers" in required
    assert ROOT / "merlin/contract/capsules/MANIFEST.yaml" in required
    assert ROOT / "examples/gemmini/phase0/history" in required
    broad = ["--ro-bind", str(ROOT), str(ROOT)]
    assert {s.path for s in BW.coverage_gap(broad, surfaces)} == required
    assert BW.coverage_gap(BW.apply_answer_masks(broad, surfaces), surfaces) == []
    assert all(not s.grantable for s in surfaces)


@pytest.mark.parametrize("value", ["one", ["/absolute"], ["../escape"], [3], ["x", "x"], [""]])
def test_malformed_exclusions_fail_closed(value):
    with pytest.raises(ValueError):
        declared_held_out_inputs({"answer_surfaces": {"held_out_inputs": value}})


def test_legacy_descriptors_have_no_new_exclusions():
    assert declared_held_out_inputs({}) == ()


def test_generated_only_is_the_release_default_and_retirement_overlay_is_preserved():
    import inspect

    from merlin_experiments.corpus.release import prepare

    assert inspect.signature(prepare).parameters["generated_only"].default is True
    retired = yaml.safe_load((ROOT / "examples/gemmini/phase0/retirements.yaml").read_bytes())
    entries = str(retired)
    for stem in ("G_conv2d_c3x224x224_k7x7s2_n64_", "G_matmul_m3136k64n64_", "G_matmul_m1k2048n1000_"):
        assert stem in entries


def test_composition_truth_is_withheld_from_compiler_authors():
    from merlin.common.access import declared_modules

    assert "merlin.targetgen.functional_truth_compositions" in declared_modules("grader")
    per_op = importlib.import_module("merlin.targetgen.functional_truth")
    # The existing standalone Core ATen selection rule is untouched.
    assert per_op.T["max_pool2d_with_indices.default"](None).family == "X_select_sort"


def test_historical_capsule_copies_cannot_reenter_public_release(tmp_path):
    from merlin_experiments.corpus.publication import require_public_capsules
    from merlin_experiments.spec import SpecError

    te = load_target_experiment(TARGET / "descriptor.yaml")
    capsule = next((ROOT / "merlin/contract/capsules/_model_layers").glob("*/capsule.yaml"))
    copied = tmp_path / "renamed_directory"
    copied.mkdir()
    (copied / "capsule.yaml").write_bytes(capsule.read_bytes())
    with pytest.raises(SpecError, match="explicitly excluded historical input"):
        require_public_capsules(te, [copied])
