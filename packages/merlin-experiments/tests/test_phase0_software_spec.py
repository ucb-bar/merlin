"""Coarse checks of authored spec -> profile selection -> independent arithmetic."""

from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase0 import numerics, profiles

from merlin.targetgen import software_spec as SS
from merlin.targetgen.corpus_spec import CorpusBinding


def _spec():
    return {
        "schema": SS.SCHEMA,
        "target": "test_device",
        "status": "unreviewed",
        "capability_contract": {"name": "test_device", "compute_units": []},
        "numerical_semantics": {
            "model": {"engine": "specir_fp_reduce"},
            "operand_dtype": "fp8_e4m3",
            "accumulator_dtype": "bf16",
            "readout_dtype": "bf16",
            "product_rounding": "accumulator_format",
            "rounding": "rne",
            "reduction_order": "index_sequential",
            "reduction_cadence": "per_step",
            "subnormal_operand_flush": False,
        },
        "operations": [{"id": "matrix", "placement": "accelerator", "signature": {"ranks": [2]}}],
        "evidence": {"unknowns": ["unqualified example"]},
    }


def test_gemmini_software_spec_does_not_admit_integer_shift_as_fused_readout():
    from merlin_experiments.phase0.software_screen import screen_entry

    from merlin.targetgen.spec_fact_drift import FactField, resolve_spec

    path = Path(__file__).resolve().parents[3] / "examples/gemmini/target/software-spec.yaml"
    authored = SS.load_software_spec(path, target="gemmini")
    entry = {
        "op": "matmul",
        "kind": "isa",
        "epilogue": ["requant"],
        "operand_dtype": "int8",
        "accum_dtype": "i32",
        "placement": "accelerator",
        "layout": "row_major_contiguous",
    }
    # The fused readout declaration is filled from facts; the authored bytes alone cannot screen it.
    with pytest.raises(ValueError, match="resolved spec"):
        screen_entry(authored, entry)
    # The readout stages the gemmini readout facet reports (none is an integer shift).
    fused = {
        "placement": FactField(("fused_accelerator",), True, ()),
        "composed_with": FactField(("contraction",), True, ()),
        "dtypes": FactField(("int32", "int8"), True, ()),
        "epilogues": FactField(("acc_scale", "bias_add", "relu"), True, ()),
        "scale_granularity": FactField(("tensor",), True, ()),
    }
    standalone = {"placement": FactField(("accelerator",), True, ()), "dtypes": FactField(("int8",), True, ())}
    fused_operand_sum = {
        **fused,
        "composed_with": FactField(("residual_add",), True, ()),
        "dtypes": FactField(("int8",), True, ()),
    }
    facts = {
        "forms": {
            "elementwise_map": {"fused": fused, "standalone": standalone, "fused_operand_sum": fused_operand_sum},
            # The contract's standalone sum reduction (accumulate-on-load) backs `reduction_standalone`.
            "reduction": {
                "fused": {**fused, "composed_with": FactField(("contraction", "movement"), True, ())},
                "standalone": standalone,
            },
            "movement": {"standalone": standalone},
        },
        "quantization": {"contraction": FactField(("quantizable",), True, ())},
    }
    spec, record = resolve_spec(authored, facts)
    # These minimal facts resolve the operation screen; the newly declared
    # scheme/composition axes still require their own output-path/resource facts.
    assert all(row["status"] == "resolved" for row in record["declarations"])
    assert record["status"] == "unresolved"
    assert record["unresolved"] and all(
        "scheme" in row or "composition" in row or row.get("field", "").startswith("schemes.")
        for row in record["unresolved"]
    )
    decision = screen_entry(spec, entry)
    assert decision["status"] == "unsupported"
    assert any(row["role"] == "epilogue" and row["status"] == "unsupported" for row in decision["decisions"])


def test_generation_screens_mixed_host_and_device_entries_with_their_writer_bindings():
    from merlin_experiments.phase0.generation import _screen_selected_entry
    from merlin_experiments.phase0.software_screen import screen_entry

    binding = CorpusBinding(
        target="fixture",
        tile_dim=4,
        operand_dtype="i8",
        accum_dtype="i32",
        integer=True,
        tiers=["L0"],
        compare="exact_int",
    )
    package = "a" * 64
    capabilities = {
        "host": {
            "package_sha256": package,
            "capability_spec_sha256": "b" * 64,
            "dtype_strategy": "fixture",
            "capability_spec": {
                "schema": "merlin.host_capabilities.v1",
                "status": "reviewed",
                "compiler": {"package_sha256": package, "dtype_strategy": "fixture"},
                "operations": [
                    {
                        "id": "host_f32_bmm",
                        "ops": ["aten.bmm.default"],
                        "placement": "host",
                        "signature": {
                            "operand_dtypes": ["f32"],
                            "accumulator_dtype": "f32",
                            "ordered_operand_dtypes": ["f32", "f32", "f32"],
                        },
                    }
                ],
                "evidence": {},
            },
        }
    }
    host = {
        "op": "batch_matmul",
        "frontend_op": "aten.bmm.default",
        "operand_dtype": "f32",
        "generalization": {"must_accelerate": False, "eligible": False},
    }
    stale = screen_entry(
        {},
        host,
        defaults={"operand_dtype": binding.operand_dtype, "accumulator_dtype": binding.accum_dtype},
        host_capabilities=capabilities,
    )
    assert stale["status"] == "unsupported"
    assert "accum_dtype" in stale["profiles"][0]["decisions"][0]["reason"]
    decision = _screen_selected_entry({}, host, binding=binding, host_capabilities=capabilities)
    assert decision["status"] == "unknown"  # Ordered source operands still need the emitted program.
    assert "accum_dtype" not in decision["reason"]
    assert (
        _screen_selected_entry({}, {**host, "operand_dtype": "i8"}, binding=binding, host_capabilities=capabilities)[
            "status"
        ]
        == "unsupported"
    )
    assert (
        _screen_selected_entry({}, {**host, "accum_dtype": "i32"}, binding=binding, host_capabilities=capabilities)[
            "status"
        ]
        == "unsupported"
    )  # An explicit conflicting entry cannot be silently repaired.
    accelerator_spec = {
        "status": "reviewed",
        "operations": [
            {
                "id": "fixture_i8_matmul",
                "ops": ["matmul"],
                "placement": "accelerator",
                "signature": {"operand_dtypes": ["i8"], "accumulator_dtype": "i32"},
            }
        ],
    }
    device = {"op": "matmul", "operand_dtype": "i8", "placement": "accelerator"}
    assert _screen_selected_entry(accelerator_spec, device, binding=binding)["status"] == "admitted"
    assert (
        _screen_selected_entry(accelerator_spec, {**device, "accum_dtype": "f32"}, binding=binding)["status"]
        == "unsupported"
    )


def test_recipe_selects_only_contained_oot_software_spec(tmp_path, monkeypatch):
    provider = tmp_path / "provider"
    (provider / "contracts").mkdir(parents=True)
    (provider / "provider.yaml").write_text(
        "schema: merlin.provider.v1\nid: selected\ntarget: test_device\nrole: support\n"
        "contract: contracts/target_contract.yaml\n"
    )
    (provider / "contracts/target_contract.yaml").write_text("name: test_device\n")
    selected = provider / "contracts/software-spec.yaml"
    selected.write_text("schema: merlin.software_spec.v1\ntarget: test_device\n")
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("software_spec: {provider: test_device, resource: contracts/software-spec.yaml}\n")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(provider))
    assert SS.software_spec_path_for_recipe(recipe) == selected
    recipe.write_text("software_spec: {provider: test_device, resource: ../outside.yaml}\n")
    with pytest.raises(ValueError, match="escapes provider root"):
        SS.software_spec_path_for_recipe(recipe)


def test_versioned_selection_preserves_bytes_and_rejects_incoherent_semantics(tmp_path):
    path = tmp_path / "software-spec.yaml"
    doc = _spec()
    path.write_text(yaml.safe_dump(doc))
    loaded = SS.load_software_spec(path, target="test_device")
    identity = SS.software_spec_identity(path, loaded)
    assert identity["status"] == "unreviewed"
    assert SS.numerical_datapath(loaded)["accum_dtype"] == "bf16"
    projected = SS.capability_contract(loaded)
    projected["name"] = "mutation"
    assert loaded["capability_contract"]["name"] == "test_device"
    with pytest.raises(ValueError, match="target differs"):
        SS.load_software_spec(path, target="other_device")
    path.write_text(path.read_text() + "\n# byte identity matters\n")
    assert SS.software_spec_identity(path)["sha256"] != identity["sha256"]
    for key, bad in (("rounding", "unspecified"), ("reduction_order", "unknown"), ("subnormal_operand_flush", 1)):
        changed = deepcopy(doc)
        changed["numerical_semantics"][key] = bad
        path.write_text(yaml.safe_dump(changed))
        with pytest.raises(ValueError, match=key):
            SS.load_software_spec(path)


def test_profile_uses_explicit_spec_and_frozen_override_without_hidden_discovery(tmp_path):
    spec = tmp_path / "software-spec.yaml"
    spec.write_text(yaml.safe_dump(_spec()))
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("software_spec: software-spec.yaml\ndatapath: {compare: tolerance_float}\n")
    template = tmp_path / "performance.yaml"
    template.write_text("sweeps: []\n")
    hidden = tmp_path / "hidden.yaml"
    hidden.write_text("invalid YAML that must not be read: [")
    kwargs = {"recipe": recipe, "performance_template": template, "hidden_profile": hidden, "include_holdouts": False}
    selected = profiles.load_profile("test_device", **kwargs)
    assert selected["datapath"]["numerical_semantics"]["rounding"] == "rne"
    assert selected["_software_spec_identity"] == SS.software_spec_identity(spec)
    frozen = tmp_path / "frozen.yaml"
    frozen.write_bytes(spec.read_bytes())
    spec.unlink()
    selected = profiles.load_profile("test_device", software_spec=frozen, **kwargs)
    assert selected["_software_spec_path"] == str(frozen)
    conformance, descriptor = tmp_path / "conformance.yaml", tmp_path / "descriptor.yaml"
    conformance.write_text("name: test_device\n")
    descriptor.write_text("workload_spec: {}\n")
    identity = profiles.synthesis_input_identity(
        recipe=recipe,
        descriptor=descriptor,
        conformance_spec=conformance,
        software_spec=frozen,
    )
    assert identity["software_spec_sha256"] == SS.software_spec_identity(frozen)["sha256"]
    legacy_identity = {key: value for key, value in identity.items() if key != "software_spec_sha256"}
    synth = tmp_path / "synth.yaml"
    synth.write_text(yaml.safe_dump({"provenance": {"selected_inputs": legacy_identity}}))
    with pytest.raises(ValueError, match="software_spec_sha256 changed"):
        profiles.verify_selected_synthesis(
            synth,
            recipe=recipe,
            descriptor=descriptor,
            conformance_spec=conformance,
            software_spec=frozen,
        )
    recipe.write_text("software_spec: software-spec.yaml\ndatapath: {subnormal_operand_flush: true}\n")
    with pytest.raises(ValueError, match="recipe conflicts"):
        profiles.load_profile("test_device", software_spec=frozen, **kwargs)


def test_selected_float_arithmetic_is_forwarded_and_legacy_is_explicit(monkeypatch):
    calls = []
    dtype = SimpleNamespace(exp_bits=8, mant_bits=7)
    dtypes = SimpleNamespace(
        FP8_E4M3=dtype,
        BF16=dtype,
        decode_float_exact=lambda value, fmt: Fraction(value),
        decode_float=lambda value, fmt: float(value),
        round_to_format=lambda value, fmt, rm: int(value),
    )

    def reduce(addends, fmt, **options):
        calls.append(options)
        return sum(addends)

    monkeypatch.setattr(numerics, "_specir", lambda **kwargs: (dtypes, reduce))
    monkeypatch.setattr(numerics, "_det_fp8", lambda *args: ([1, 2, 3, 4], [1, 2, 3, 4]))
    binding = CorpusBinding(
        target="test_device",
        tile_dim=2,
        operand_dtype="fp8_e4m3",
        accum_dtype="bf16",
        integer=False,
        tiers=["L0"],
        compare="tolerance_float",
    )
    entry = {"name": "matrix", "op": "matmul", "M": 2, "N": 2, "K": 2}
    legacy = numerics._float_golden(entry, binding)
    assert numerics.float_semantics(entry, binding)["selection_status"] == "legacy_compatibility"
    selected = {**entry, "numerical_semantics": _spec()["numerical_semantics"]}
    assert numerics._float_golden(selected, binding) == legacy
    selected["numerical_semantics"] = {
        **selected["numerical_semantics"],
        "rounding": "rtz",
        "reduction_order": "tree",
        "reduction_cadence": "single_final",
    }
    numerics._float_golden(selected, binding)
    assert calls[-1] == {"order": "tree", "cadence": "single_final", "rm": "rtz"}
    selected["numerical_semantics"]["accumulator_dtype"] = "f32"
    selected["numerical_semantics"]["readout_dtype"] = "f32"
    with pytest.raises(ValueError, match="corpus operand/accumulator binding"):
        numerics._float_golden(selected, binding)


def test_cross_phase_numeric_policy_requires_verified_spec_snapshot(tmp_path, monkeypatch):
    from merlin_experiments.corpus.numeric_policy import load_declared_numeric_policy

    from merlin.targetgen.sandbox import bwrap

    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    repo = tmp_path / "repo"
    recipe, spec = repo / "phase0/recipe.yaml", repo / "target/software-spec.yaml"
    recipe.parent.mkdir(parents=True)
    spec.parent.mkdir(parents=True)
    recipe.write_text("software_spec: ../target/software-spec.yaml\ndatapath: {compare: tolerance_float, atol: 0.01}\n")
    spec.write_text(yaml.safe_dump(_spec()))
    experiment = SimpleNamespace(target="test_device", numeric_profile="phase0/recipe.yaml")
    bundle = {"allowed": [], "host_inputs": [{"path": "phase0/recipe.yaml"}, {"path": "target/software-spec.yaml"}]}
    workspace = tmp_path / "run/workspace"
    workspace.mkdir(parents=True)
    try:
        bwrap.materialize_bundle_inputs(workspace, bundle, repo=repo)
        [frozen_recipe] = bwrap.snapshot_input_paths(workspace, bundle, [recipe], repo=repo)
        recipe.unlink()
        spec.unlink()
        with pytest.raises(ValueError, match="verified frozen input"):
            load_declared_numeric_policy(experiment, repo=repo, frozen_profile=frozen_recipe)

        def resolve(source):
            return bwrap.snapshot_input_paths(workspace, bundle, [source], repo=repo)[0]

        observed, identity = load_declared_numeric_policy(
            experiment,
            repo=repo,
            frozen_profile=frozen_recipe,
            frozen_resolver=resolve,
        )
        assert observed["atol"] == 0.01
        assert observed["numerical_semantics"]["reduction_cadence"] == "per_step"
        assert identity["software_spec"]["source"] == "frozen-input"
        frozen_spec = resolve(spec)
        frozen_spec.chmod(0o600)
        frozen_spec.write_text(frozen_spec.read_text() + "\n# changed\n")
        with pytest.raises(RuntimeError, match="content verification"):
            load_declared_numeric_policy(
                experiment,
                repo=repo,
                frozen_profile=frozen_recipe,
                frozen_resolver=resolve,
            )
    finally:
        bwrap.remove_bundle_snapshot(workspace)


def test_operation_admission_checks_signature_and_never_ignores_unknowns():
    spec = _spec()
    spec["status"] = "reviewed"
    spec["operations"] = [
        {
            "id": "matrix",
            "ops": ["matmul"],
            "placement": "accelerator",
            "signature": {
                "operand_dtypes": ["fp8_e4m3"],
                "ranks": [2],
                "layouts": ["row_major"],
                "shape_bounds": {"K": {"min": 1, "max": 64, "multiple_of": 2}},
            },
        }
    ]
    signature = {"operand_dtype": "fp8_e4m3", "rank": 2, "layout": "row_major", "dimensions": {"K": 32}}
    assert SS.admit_operation(spec, "matmul", signature, "accelerator")["status"] == "admitted"
    for changed in ({"operand_dtype": "int8"}, {"rank": 4}, {"dimensions": {"K": 33}}):
        assert SS.admit_operation(spec, "matmul", {**signature, **changed}, "accelerator")["status"] == "unsupported"
    assert SS.admit_operation(spec, "lstm", signature, "accelerator")["status"] == "unsupported"
    assert SS.admit_operation(spec, "matmul", signature, "host")["status"] == "unsupported"
    assert SS.admit_operation(spec, "matmul", {"rank": 2}, "accelerator")["status"] == "unknown"
    spec["operations"][0]["signature"]["restriction"] = "requires independent alias analysis"
    assert SS.admit_operation(spec, "matmul", signature, "accelerator")["status"] == "unknown"


def test_explicit_operation_selector_cannot_widen_through_family_or_id():
    spec = _spec()
    spec["status"] = "reviewed"
    spec["operations"] = [
        {
            "id": "reviewed_explicit",
            "ops": ["aten.explicit"],
            "families": ["elementwise_map"],
            "placement": "host",
            "signature": {"family": "elementwise_map", "ranks": [2]},
        }
    ]
    observed = {"family": "elementwise_map", "rank": 2}
    assert SS.admit_operation(spec, "aten.explicit", observed, "host")["status"] == "admitted"
    assert SS.admit_operation(spec, "aten.synthetic_elementwise", observed, "host")["status"] == "unsupported"

    # A matching row identifier is not an escape from that row's explicit ops list.
    spec["operations"][0]["id"] = "aten.synthetic_elementwise"
    assert SS.admit_operation(spec, "aten.synthetic_elementwise", observed, "host")["status"] == "unsupported"

    # An actual family-only declaration is still available as fallback.
    spec["operations"][0]["id"] = "reviewed_explicit"
    spec["operations"].append(
        {
            "id": "reviewed_family",
            "families": ["elementwise_map"],
            "placement": "host",
            "signature": {"family": "elementwise_map", "ranks": [2]},
        }
    )
    assert SS.admit_operation(spec, "aten.synthetic_elementwise", observed, "host")["status"] == "admitted"
    assert SS.admit_operation(spec, "aten.explicit", {**observed, "rank": 3}, "host")["status"] == "unsupported"


def test_synthesis_producer_binds_explicit_software_and_hardware_selection(tmp_path, monkeypatch):
    import importlib.util

    from merlin_experiments.phase0 import evidence

    from merlin.common.paths import repo_root

    module_spec = importlib.util.spec_from_file_location(
        "software_spec_synthesis",
        repo_root() / "build_tools/scripts/synth_capsule_corpus.py",
    )
    producer = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(producer)
    recipe, spec = tmp_path / "recipe.yaml", tmp_path / "software-spec.yaml"
    descriptor, requirement = tmp_path / "descriptor.yaml", tmp_path / "requirement.yaml"
    recipe.write_text("datapath: {}\n")
    spec.write_text(yaml.safe_dump(_spec()))
    descriptor.write_text("target: test_device\nworkload_spec: {}\n")
    requirement.write_text("cells: []\n")
    declaration = SimpleNamespace(
        target="test_device", recipe=recipe, descriptor=descriptor, conformance_spec=requirement
    )
    captured = SimpleNamespace(
        target="test_device",
        software_spec=_spec(),
        status="diagnostic",
        raw_facts_sha256="a" * 64,
        views_json=b'{"status":"diagnostic"}',
    )
    observed = {}

    def select(target, **inputs):
        observed.update(inputs)
        return captured

    monkeypatch.setattr(producer, "for_target", lambda target: declaration)
    monkeypatch.setattr(producer, "_workload_spec", lambda target: {})
    monkeypatch.setattr(evidence, "select_evidence", select)
    monkeypatch.setattr(producer, "synthesize", lambda *args, **kwargs: {"capsules": [], "provenance": {}})
    result = producer.synth_for("test_device", software_spec=spec, rtl_facts=tmp_path / "facts.json")
    assert result["status"] == "ok"
    assert observed["software_spec"] == spec
    assert observed["facts_path"] == tmp_path / "facts.json"
    provenance = result["provenance"]
    assert provenance["selected_inputs"]["software_spec_sha256"] == SS.software_spec_identity(spec)["sha256"]
    assert provenance["hardware_evidence"]["status"] == "diagnostic"
    assert provenance["software_spec"]["status"] == "unreviewed"
    requirement.write_text("cells: [{cell: contraction/bf16/aligned, family: contraction, dtype: bf16}]\n")
    conflict = producer.synth_for("test_device", software_spec=spec, rtl_facts=tmp_path / "facts.json")
    assert conflict["status"] == "invalid_synthesis_inputs"
    assert "contraction/bf16/aligned" in conflict["detail"]


def test_mx_conformance_rejects_legacy_accelerator_formats():
    import importlib.util

    from merlin.common.paths import repo_root

    root = repo_root()
    module_spec = importlib.util.spec_from_file_location(
        "mx_software_spec_synthesis", root / "build_tools/scripts/synth_capsule_corpus.py"
    )
    producer = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(producer)
    software = SS.load_software_spec(root / "examples/mx_gemmini/target/software-spec.yaml", target="mx_gemmini")
    old_requirement = yaml.safe_load(
        (root / "experiments/reference-data/phase0/conformance/mx_gemmini.yaml").read_text()
    )
    conflicts = producer._software_cell_conflicts(old_requirement, software)
    assert "contraction/bf16/aligned" in conflicts
    assert "contraction/i8/aligned" in conflicts
    assert "contraction/mxfp4/aligned" not in conflicts


def _f32_readout():
    return {
        "strategy": "f32_product_rne",
        "accumulator_conversion": {"to": "f32", "rounding": "half_even"},
        "multiplier": {"dtype": "f32", "formation": "rne_of_exact_rational"},
        "product": {"dtype": "f32", "rounding": "half_even"},
        "narrowing": {"rounding": "half_even", "saturate": "declared_dtype"},
        "activation_order": "from_facts",
        "exactness_bridge": "prove_fp32_products",
    }


@pytest.mark.parametrize("field", list(_f32_readout()))
def test_f32_readout_requires_every_arithmetic_field(field):
    declaration = _f32_readout()
    del declaration[field]
    with pytest.raises(ValueError):
        SS.validate_readout(declaration)


@pytest.mark.parametrize(
    "field,value",
    [
        ("multiplier", {"dtype": "f32", "formation": "stepwise_f32"}),
        ("product", {"dtype": "f64", "rounding": "half_even"}),
        ("narrowing", {"rounding": "half_away", "saturate": "declared_dtype"}),
        ("activation_order", ["activation", "scale"]),
        ("exactness_bridge", "sampled_outputs"),
    ],
)
def test_f32_readout_refuses_weaker_numerical_contracts(field, value):
    declaration = _f32_readout()
    declaration[field] = value
    with pytest.raises(ValueError):
        SS.validate_readout(declaration)


def _composition_spec():
    spec = _spec()
    spec["numerical_semantics"] = {
        "model": {"engine": "integer_reference"},
        "operand_dtype": "int8",
        "accumulator_dtype": "i32",
        "readout_dtype": "i32",
        "subnormal_operand_flush": False,
        "readout": _f32_readout(),
        "quantized_native_fallback": {"readout": "full_int32_then_host_readout_per_scheme"},
    }
    spec["compositions"] = {
        "schema": "merlin.software_spec.compositions.v1",
        "status": "authored_pending_operator_review",
        "grading": "placement_and_instruction_audit",
        "chains": [
            {
                "id": "readout",
                "producer": "contraction",
                "accumulator_stages": {"preload": "optional"},
                "readout_stages": "from_output_path",
                "exits": ["requant", "accumulator"],
                "dataflows": "from_facts",
                "constraints": [
                    "activation_before_scale_requires_nonnegative_multiplier",
                    "accumulator_exit_applies_no_readout_stage",
                    {"os_dataflow_requires": "mesh_partial_sum_bound_from_constant_operands"},
                    {"pool_window_within_register_fields": "from_facts"},
                    {"pool_padding": "from_facts"},
                ],
            },
            {
                "id": "sum",
                "producer": "operand_sum",
                "readout_stages": "from_output_path",
                "exits": ["requant", "accumulator"],
                "constraints": [{"operand_multiplier_in": "(0, 1]"}, "load_minimum_not_negated"],
            },
            {
                "id": "window",
                "producer": ["movement", "contraction"],
                "reduction_kinds": ["max", "sum"],
                "constraints": [
                    {"max_requires_store_path_pooling": "from_facts"},
                    "scratchpad_source_pooling_fenced",
                    {"padded_max_requires": "producer_lower_bound_or_derived_pad_value"},
                ],
            },
            {
                "id": "extension",
                "producer": "contraction",
                "extension": "constant_integer_operand",
                "constraints": ["coefficient_integer_exact", {"k_extension_within": "from_facts"}],
            },
            {
                "id": "siblings",
                "producer": "contraction",
                "multiplicity": {"min": 2},
                "constraints": ["per_output_readout_independent", {"shared_input_resident": "from_facts"}],
            },
        ],
    }
    return spec


def _composition_facts():
    from merlin.targetgen.spec_fact_drift import FactField

    def fact(value):
        return FactField((value,), True, ("structural fixture evidence",))

    return {
        "families": {
            family: {"placement": fact("accelerator")} for family in ("contraction", "operand_sum", "movement")
        },
        "composition": {
            "output_path": {
                "schema": "output_path_v1",
                "accumulator": {"dtype": "i32"},
                "accumulator_stages": [{"kind": "preload", "dtype": "i32"}],
                "readout": [
                    {"kind": "activation", "domain": "accumulator"},
                    {"kind": "convert"},
                    {"kind": "scale", "granularity": "per_command", "columns_per_scale": 7},
                    {"kind": "requant"},
                    {"kind": "pool", "padding": "zero", "reductions": ["max"]},
                ],
                "exits": {"requant": True, "accumulator": True},
            },
            "dataflows": FactField(("weight_stationary", "output_stationary"), True, ("dataflow selector",)),
            "pool_window_within_register_fields": fact({"size_bits": 3, "stride_bits": 2}),
            "max_requires_store_path_pooling": fact(True),
            "k_extension_within": fact({"max_k": 23, "partial_sum_bits": 19}),
            "shared_input_resident": fact({"capacity_bytes": 8192}),
        },
    }


def test_family_compositions_resolve_all_readouts_exits_and_user_rulings():
    from merlin.targetgen.spec_fact_drift import resolve_spec

    spec = _composition_spec()
    original = deepcopy(spec)
    SS.validate_software_spec(spec)
    resolved, record = resolve_spec(spec, _composition_facts())
    assert record["status"] == "resolved"
    assert spec == original
    rows = {row["id"]: row for row in resolved["composition_resolution"]["chains"]}
    assert all(row["placement"] == "fused_accelerator" for row in rows.values())
    assert all(row["grading"] == "placement_and_instruction_audit" for row in rows.values())
    assert rows["readout"]["readout_stages"] == ["activation", "convert", "scale", "requant", "pool"]
    assert rows["readout"]["exit_stages"]["accumulator"] == []
    assert rows["extension"]["required_when"] == {"residual_coefficient": "integer_constant"}
    assert rows["window"]["grading_scope"] == "composition_truth_only"
    assert rows["siblings"]["shared_input"] == "one_immutable_input"
    assert resolved["readout_resolution"]["activation_order"] == rows["readout"]["readout_stages"]
    # Definition describes admission, never claims an executed arithmetic proof.
    assert SS.readout_exactness_gate("f32_product_rne") == {
        "definitional": "f32_product_rne",
        "requires": "decoded_multiplier_bits_equal_formed_multiplier",
    }
    assert SS.readout_exactness_gate("exact_rational_host")["proof"] == "prove_fp32_products"


@pytest.mark.parametrize("missing", ["output_path", "dataflows", "k_extension_within", "shared_input_resident"])
def test_composition_missing_facts_are_unknown_and_listed(missing):
    from merlin.targetgen.spec_fact_drift import resolve_spec

    facts = _composition_facts()
    del facts["composition"][missing]
    resolved, record = resolve_spec(_composition_spec(), facts)
    assert record["status"] == "unresolved"
    assert record["unresolved"]
    assert all(item["value"] == "UNKNOWN" for item in record["unresolved"])
    assert any(row["status"] == "unresolved" for row in resolved["composition_resolution"]["chains"])


def test_underived_pool_padding_and_unproved_producer_do_not_grant_fusion():
    from merlin.targetgen.spec_fact_drift import FactField, resolve_spec

    facts = _composition_facts()
    facts["composition"]["output_path"]["readout"][-1]["padding"] = "unknown"
    facts["families"]["contraction"]["placement"] = FactField(("accelerator",), False, ("incomplete",))
    resolved, record = resolve_spec(_composition_spec(), facts)
    assert record["status"] == "unresolved"
    assert all(
        row["placement"] == "unknown"
        for row in resolved["composition_resolution"]["chains"]
        if row["producer"] == "contraction"
    )
    assert any("pool" in item["field"] for item in record["unresolved"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.__setitem__("grading", "bytes_only"),
        lambda b: b["chains"][0].__setitem__("producer", "conv2d"),
        lambda b: b["chains"][0].__setitem__("readout_stages", ["relu"]),
        lambda b: b["chains"][0].__setitem__("ops", ["conv2d"]),
        lambda b: b["chains"][0].__setitem__("dataflows", ["OS"]),
        lambda b: b["chains"].append(deepcopy(b["chains"][0])),
        lambda b: b["chains"][3].__setitem__("constraints", ["coefficient_integer_exact"]),
        lambda b: b["chains"][4].__setitem__("constraints", ["per_output_readout_independent"]),
        lambda b: b["chains"][4].__setitem__("multiplicity", {"min": 1}),
        lambda b: b["chains"][0].__setitem__("constraints", ["accumulator_exit_applies_no_readout_stage"]),
    ],
)
def test_composition_vocabulary_constraints_and_grading_fail_closed(mutate):
    block = _composition_spec()["compositions"]
    mutate(block)
    with pytest.raises(ValueError):
        SS.validate_compositions(block)


def test_composition_stage_order_is_checked_against_selected_path():
    from merlin.targetgen.spec_fact_drift import resolve_spec

    spec = _composition_spec()
    spec["compositions"]["chains"][0]["readout_stages"] = ["requant", "convert"]
    with pytest.raises(ValueError, match="output path order"):
        resolve_spec(spec, _composition_facts())


def test_legacy_rational_readout_and_fallback_remain_valid():
    spec = _composition_spec()
    del spec["compositions"]
    spec["numerical_semantics"]["readout"] = {"strategy": "exact_rational_host", "requant_rounding": "half_even"}
    spec["numerical_semantics"]["quantized_native_fallback"]["readout"] = "full_int32_then_exact_host_rne"
    assert SS.validate_software_spec(spec)["numerical_semantics"] == spec["numerical_semantics"]
    spec["numerical_semantics"]["quantized_native_fallback"]["readout"] = "guess"
    with pytest.raises(ValueError, match="fallback.readout"):
        SS.validate_software_spec(spec)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_requires_store_path_pooling", False),
        ("shared_input_resident", {"capacity_bytes": 0}),
        ("k_extension_within", {"max_k": True, "partial_sum_bits": 20}),
        ("pool_window_within_register_fields", {"size_bits": 3}),
    ],
)
def test_false_or_malformed_composition_facts_cannot_establish_fusion(field, value):
    from merlin.targetgen.spec_fact_drift import FactField, resolve_spec

    facts = _composition_facts()
    facts["composition"][field] = FactField((value,), True, ("selected evidence",))
    _, record = resolve_spec(_composition_spec(), facts)
    assert record["status"] == "unresolved"
    assert any(field in item["field"] for item in record["unresolved"])


def test_bare_authored_composition_values_cannot_replace_facts():
    from merlin.targetgen.spec_fact_drift import resolve_spec

    facts = _composition_facts()
    facts["composition"]["dataflows"] = ["weight_stationary", "output_stationary"]
    _, record = resolve_spec(_composition_spec(), facts)
    assert any(item["field"] == "dataflows" for item in record["unresolved"])


def test_unavailable_exit_and_readout_stage_remain_unresolved():
    from merlin.targetgen.spec_fact_drift import resolve_spec

    spec, facts = _composition_spec(), _composition_facts()
    spec["compositions"]["chains"][0]["readout_stages"] = ["pool"]
    facts["composition"]["output_path"]["readout"].pop()
    facts["composition"]["output_path"]["exits"]["accumulator"] = False
    _, record = resolve_spec(spec, facts)
    assert record["status"] == "unresolved"
    assert any(item["field"] == "readout_stages" for item in record["unresolved"])
    assert any(item["field"] == "exits.accumulator" for item in record["unresolved"])
