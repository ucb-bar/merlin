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

    path = Path(__file__).resolve().parents[3] / "examples/gemmini/target/software-spec.yaml"
    spec = SS.load_software_spec(path, target="gemmini")
    decision = screen_entry(
        spec,
        {
            "op": "matmul",
            "kind": "isa",
            "epilogue": ["requant"],
            "operand_dtype": "int8",
            "accum_dtype": "i32",
            "placement": "accelerator",
            "layout": "row_major_contiguous",
        },
    )
    assert decision["status"] == "unsupported"
    assert any(row["role"] == "epilogue" and row["status"] == "unsupported" for row in decision["decisions"])


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
