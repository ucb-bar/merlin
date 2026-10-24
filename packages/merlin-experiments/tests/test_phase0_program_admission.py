"""A written capsule is screened from its own program, with the rule captured operations are judged by."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase0 import program_admission

from merlin.targetgen.contract.interface_emit import emit_interface_mlir
from merlin.targetgen.software_spec import admit_operation, validate_software_spec

_CONTRACT = {
    "name": "fixture",
    "compute_units": [
        {
            "name": "array",
            "kind": "systolic",
            "dtypes": ["int8"],
            "ops": ["matmul"],
            "semantic_capabilities": [
                {"family": "contraction", "dtypes": ["int8"], "ranks": [2, 3, 4]},
                {
                    "family": "elementwise_map",
                    "dtypes": ["int8"],
                    "result_dtypes": ["i32"],
                    "composed_with": ["contraction"],
                },
            ],
        }
    ],
}
_SPEC = {
    "schema": "merlin.software_spec.v1",
    "target": "fixture",
    "status": "reviewed",
    "numerical_semantics": {
        "model": {"engine": "integer_reference"},
        "operand_dtype": "int8",
        "accumulator_dtype": "i32",
        "readout_dtype": "i32",
        "subnormal_operand_flush": False,
        "overflow": "wrap_internal_mac",
    },
    "operations": {
        "contraction": {
            "placement": "accelerator",
            "operand_dtypes": ["int8"],
            "accumulator_dtype": "i32",
            "ranks": [2, 3, 4],
            "layouts": ["row_major_contiguous"],
            "tails": ["zero_pad_valid_window"],
            "broadcasting": ["none", "independent_batches"],
            "aliasing": ["disjoint_inputs_outputs"],
        },
        "elementwise_map": {
            "placement": "fused_accelerator",
            "dtypes": ["int8", "i32"],
            "composed_with": ["contraction"],
            "epilogues": ["relu", "acc_scale", "bias"],
            "scale_granularity": ["tensor"],
        },
    },
}


_HOST = {
    "profile": {
        "capability_spec": {
            "schema": "merlin.host_capabilities.v1",
            "status": "reviewed",
            "compiler": {"package_sha256": "a" * 64, "dtype_strategy": "lane"},
            "operations": [
                {
                    "id": "host_float_map",
                    "ops": ["aten.mul.Tensor"],
                    "placement": "host",
                    "signature": {"dtypes": ["f32"]},
                }
            ],
            "evidence": {},
        },
        "package_sha256": "a" * 64,
        "capability_spec_sha256": "b" * 64,
        "dtype_strategy": "lane",
    }
}


def _evidence(**override):
    spec = validate_software_spec(dict(_SPEC, **override), "fixture")
    return SimpleNamespace(
        software_spec=spec, contract=_CONTRACT, host_capabilities=_HOST, whole_program_admission=None
    )


def _capsule(tmp_path, name, commands, tensors, text=None, **extra):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "capsule.interface.mlir").write_text(
        text
        or emit_interface_mlir({"abi_version": "0.1", "target": "fixture", "tensors": tensors, "commands": commands})
    )
    capsule = {"name": name, "kind": "layer", "interface_mlir": "capsule.interface.mlir", **extra}
    (directory / "capsule.yaml").write_text(yaml.safe_dump(capsule))
    return capsule, directory


_BIAS_ADD = """module attributes {merlin_iface.version = "0.1", merlin_iface.target = "fixture", merlin_iface.abi_version = "0.1"} {
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<16x16xi32>
  %B = merlin_iface.tensor {name = "B", role = "bias"} : tensor<16xi32>
  %Y0 = merlin_iface.bias_add %X, %B {name = "Y0", output_dtype = "i32"} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16x16xi32>
}
"""

_RESIDUAL_RELU = """module attributes {merlin_iface.version = "0.1", merlin_iface.target = "fixture", merlin_iface.abi_version = "0.1"} {
  %X0 = merlin_iface.tensor {name = "X0", role = "input"} : tensor<256x8xi8>
  %X1 = merlin_iface.tensor {name = "X1", role = "input"} : tensor<256x8xi8>
  %Y0 = merlin_iface.residual_add %X0, %X1 {name = "Y0", epilogue = ["relu"], lhs_scale = 1.0625 : f32, rhs_scale = 0.3125 : f32, output_dtype = "i8"} : (tensor<256x8xi8>, tensor<256x8xi8>) -> tensor<256x8xi8>
}
"""


_TENSORS = {
    "W": {"shape": [32, 16], "dtype": "i8", "role": "weight"},
    "A0": {"shape": [16, 32], "dtype": "i8", "role": "input"},
}
_FUSED = [
    {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"}, "attributes": {"layout": "packed_rhs"}},
    {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A0", "rhs": "W_res", "dst": "acc0"}},
    {
        "opcode": "COMMIT",
        "operands": {"src": "acc0", "dst": "Y0"},
        "attributes": {"epilogue": ["relu"], "output_dtype": "i32"},
    },
    {"opcode": "EVICT", "operands": {"handle": "W_res"}},
]


@pytest.mark.parametrize("operation", ["matmul", "conv2d"])
@pytest.mark.parametrize("gains", [[0.5] * 4 + [0.25] * 4 + [0.75], [0.5, 0.25, 0.75] * 3])
def test_column_readout_admits_only_screened_logical_scalar_components(operation, gains):
    from merlin.targetgen.interface_observations import command_rows

    attrs = {"epilogue": ["bias", "acc_scale", "relu"], "output_dtype": "i8", "acc_scale_by_column": gains, "bias": "B"}
    tensors = {
        "A0": {"shape": [3, 4], "dtype": "i8", "role": "input"},
        "W": {"shape": [4, 9], "dtype": "i8", "role": "weight"},
        "Y0": {"shape": [3, 9], "dtype": "i8", "role": "output"},
        "B": {"shape": [9], "dtype": "i32", "role": "bias"},
    }
    commands = [
        {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A0", "rhs": "W", "dst": "acc0"}},
        {"opcode": "COMMIT", "operands": {"src": "acc0", "dst": "Y0"}, "attributes": attrs},
    ]
    if operation == "conv2d":
        tensors["A0"]["shape"] = [1, 1, 3, 4]
        commands = [
            {
                "opcode": "CONV2D",
                "operands": {"ifm": "A0", "weight": "W", "dst": "Y0"},
                "attributes": {
                    **attrs,
                    "kernel": [1, 1, 4, 9],
                    "stride": [1, 1],
                    "padding": [0, 0, 0, 0],
                    "dilation": [1, 1],
                    "layout": "nhwc",
                },
            }
        ]
    cb = {"abi_version": "0.1", "target": "fixture", "tensors": tensors, "commands": commands}
    evidence = _evidence()
    evidence.contract = deepcopy(_CONTRACT)
    evidence.contract["compute_units"][0]["semantic_capabilities"][1]["result_dtypes"] = ["i8"]
    text = emit_interface_mlir(cb)
    entries = program_admission.account_interface_text(text, target="fixture", evidence=evidence)
    stages = [entry for entry in entries if entry["operation"] in {"acc_scale", "relu"}]
    assert len(stages) == 2
    assert all(
        row["composed_observation"]["scale_granularity"] == "channel"
        for row in command_rows(cb)
        if row["operation"] in {"acc_scale", "relu"}
    )
    for stage in stages:
        admitted = stage["accelerator_admission"]
        assert admitted["status"] == "admitted", admitted
        proof = admitted["logical_partition"]
        assert proof["physical_implementation"] == "unverified"
        assert [c for part in proof["parts"] for c in range(part["start"], part["stop"])] == list(range(9))
    # Missing/incorrect original dimensions never gain scalar admission.
    tensors["W"]["shape"][1] = 8
    refused = program_admission.account_interface_text(emit_interface_mlir(cb), target="fixture", evidence=evidence)
    assert all(
        entry["accelerator_admission"]["status"] != "admitted"
        for entry in refused
        if entry["operation"] in {"acc_scale", "relu"}
    )


def test_a_written_interface_program_is_admitted_from_its_observations(tmp_path):
    capsule, directory = _capsule(tmp_path, "fused", _FUSED, _TENSORS)
    screen = program_admission.screen_written(capsule, directory, target="fixture", evidence=_evidence())
    assert screen["status"] == "admitted", screen["reason"]
    assert screen["placements"] == {"accelerator": 2}
    # The same program under an unreviewed declaration set is not admitted.
    unreviewed = program_admission.screen_written(
        capsule, directory, target="fixture", evidence=_evidence(status="unreviewed")
    )
    assert unreviewed["status"] == "unknown"


@pytest.mark.parametrize("shape", [[2, 3], [1, 4], [3, 5]])
def test_movement_uses_the_actual_copy_form_and_observed_layout(tmp_path, shape):
    evidence = _evidence(
        operations={
            **_SPEC["operations"],
            "movement": {"placement": "accelerator", "dtypes": ["int8"], "layouts": ["row_major_contiguous"]},
        }
    )
    evidence.contract = deepcopy(_CONTRACT)
    capability = {"family": "movement", "dtypes": ["int8"], "forms": ["copy"], "layouts": ["row_major_contiguous"]}
    evidence.contract["compute_units"][0]["semantic_capabilities"].append(capability)
    tensor_type = "tensor<" + "x".join(map(str, shape)) + "xi8>"
    program = (
        'module attributes {merlin_iface.version = "0.1", merlin_iface.target = "fixture", '
        'merlin_iface.abi_version = "0.1"} {\n'
        f'  %X = merlin_iface.tensor {{name = "X", role = "input"}} : {tensor_type}\n'
        '  %Y = merlin_iface.movement %X {name = "Y", semantic = "mvin_mvout", output_dtype = "i8"} '
        f": ({tensor_type}) -> {tensor_type}\n"
        "}\n"
    )
    capsule, directory = _capsule(
        tmp_path,
        "transfer",
        None,
        None,
        text=program,
    )
    screen = program_admission.screen_written(capsule, directory, target="fixture", evidence=evidence)
    assert screen["status"] == "admitted", screen["reason"]
    assert screen["placements"] == {"accelerator": 1}
    capability["forms"] = ["permutation"]
    assert program_admission.screen_written(capsule, directory, target="fixture", evidence=evidence)["status"] == (
        "unsupported"
    )
    capability["forms"] = ["copy"]
    capability["layouts"] = ["column_major"]
    assert program_admission.screen_written(capsule, directory, target="fixture", evidence=evidence)["status"] == (
        "unsupported"
    )


def test_a_standalone_part_of_a_fused_only_family_is_refused(tmp_path):
    capsule, directory = _capsule(tmp_path, "part", None, None, text=_BIAS_ADD)
    screen = program_admission.screen_written(capsule, directory, target="fixture", evidence=_evidence())
    assert screen["status"] == "unsupported" and "composition" in screen["reason"]


def test_standalone_admission_does_not_license_a_fused_stage(tmp_path):
    spec = deepcopy(_SPEC)
    spec["operations"]["elementwise_map_standalone"] = {
        "placement": "accelerator",
        "families": ["elementwise_map"],
        "dtypes": ["int8"],
    }
    contract = deepcopy(_CONTRACT)
    contract["compute_units"][0]["semantic_capabilities"].append({"family": "elementwise_map", "dtypes": ["int8"]})
    evidence = SimpleNamespace(
        software_spec=validate_software_spec(spec, "fixture"),
        contract=contract,
        host_capabilities=_HOST,
        whole_program_admission=None,
    )
    capsule, directory = _capsule(tmp_path, "fused-map", None, None, text=_RESIDUAL_RELU)
    screen = program_admission.screen_written(capsule, directory, target="fixture", evidence=evidence)
    assert screen["status"] == "unsupported" and "composition" in screen["reason"]


def test_written_operand_sum_uses_selected_numeric_facts_and_refuses_bad_scales(tmp_path):
    evidence = _evidence()
    evidence.software_spec["numerical_semantics"]["readout"] = {
        "acc_scale_rounding": "half_even",
        "narrowing": "saturate_to_declared_dtype",
    }
    evidence.software_spec["operations"].append(
        {
            "id": "sum_readout",
            "families": ["elementwise_map"],
            "placement": "fused_accelerator",
            "signature": {"dtypes": ["int8"], "composed_with": ["residual_add"], "epilogues": ["relu"]},
            "derived_from_facts": {"form": "fused_operand_sum", "family": "elementwise_map"},
        }
    )
    evidence.readout_facets = [
        {
            "operand_sum": {
                "operands": 2,
                "operand_dtype": "i8",
                "operand_rounding": "half_even",
                "operand_saturates": True,
                "scale_dtype": "f32",
            },
            "readouts": [{"selector": "i8", "applies": ["acc_scale", "relu"]}],
            "scale": {"dtype": "f32", "granularities": ["tensor"]},
        }
    ]
    selected = _RESIDUAL_RELU.replace("output_dtype", "bound_lsb = 2 : i64, output_dtype")
    capsule, directory = _capsule(tmp_path, "selected", None, None, text=selected)
    screen = program_admission.screen_written(capsule, directory, target="fixture", evidence=evidence)
    assert screen["numeric_screens"][0]["status"] == "within_bound"
    assert screen["numeric_screens"][0]["pairs_checked"] == 65536

    raw = selected.replace('epilogue = ["relu"]', "epilogue = []")
    capsule, directory = _capsule(tmp_path, "raw", None, None, text=raw)
    raw_screen = program_admission.screen_written(capsule, directory, target="fixture", evidence=evidence)
    assert raw_screen["numeric_screens"][0]["form"] == "standalone_operand_sum"
    assert raw_screen["numeric_screens"][0]["status"] == "within_bound"

    bad = selected.replace("1.0625 : f32", "10.0 : f32").replace("0.3125 : f32", "5.1 : f32")
    capsule, directory = _capsule(tmp_path, "bad", None, None, text=bad)
    screen = program_admission.screen_written(capsule, directory, target="fixture", evidence=evidence)
    assert screen["status"] == "unsupported"
    assert screen["numeric_screens"][0]["max_error_lsb"] == 5

    bias = selected.replace('epilogue = ["relu"]', 'epilogue = ["bias_add"]')
    capsule, directory = _capsule(tmp_path, "bias", None, None, text=bias)
    screen = program_admission.screen_written(capsule, directory, target="fixture", evidence=evidence)
    assert screen["numeric_screens"][0]["status"] == "unknown"


def test_reviewed_operand_sum_stage_requires_its_concrete_numeric_witness(tmp_path):
    evidence = _evidence()
    evidence.software_spec["numerical_semantics"]["readout"] = {
        "acc_scale_rounding": "half_even",
        "narrowing": "saturate_to_declared_dtype",
    }
    evidence.software_spec["operations"].append(
        {
            "id": "sum_readout",
            "families": ["elementwise_map"],
            "placement": "fused_accelerator",
            "numerical_contract": "operand_sum_exhaustive_i8_v1",
            "signature": {"dtypes": ["int8"], "composed_with": ["residual_add"], "epilogues": ["relu"]},
            "derived_from_facts": {"form": "fused_operand_sum", "family": "elementwise_map"},
        }
    )
    evidence.readout_facets = [
        {
            "operand_sum": {
                "operands": 2,
                "operand_dtype": "i8",
                "operand_rounding": "half_even",
                "operand_saturates": True,
                "scale_dtype": "f32",
            },
            "readouts": [{"selector": "i8", "applies": ["acc_scale", "relu"]}],
            "scale": {"dtype": "f32", "granularities": ["tensor"]},
        }
    ]
    signature = {
        "family": "elementwise_map",
        "operand_dtype": "i8",
        "readout_dtype": "i8",
        "composed_with": ["residual_add"],
        "epilogues": ["relu"],
        "scale_granularity": "tensor",
    }
    declaration = {**evidence.software_spec, "operations": [evidence.software_spec["operations"][-1]]}
    no_witness = admit_operation(declaration, "relu", signature, "fused_accelerator")
    assert no_witness["status"] == "unknown" and "numerical_contract" in no_witness["reason"]

    selected = _RESIDUAL_RELU.replace("output_dtype", "bound_lsb = 2 : i64, output_dtype")
    capsule, directory = _capsule(tmp_path, "selected-with-contract", None, None, text=selected)
    screen = program_admission.screen_written(capsule, directory, target="fixture", evidence=evidence)
    assert screen["numeric_screens"][0]["pairs_checked"] == 65536
    entries = program_admission.account_interface(
        directory / "capsule.interface.mlir", target="fixture", evidence=evidence
    )
    [relu] = [row for row in entries if row["operation"] == "relu"]
    [software] = [row for row in relu["software_admissions"] if row["declaration"] == "sum_readout"]
    assert software["status"] == "admitted"
    assert relu["observed_admission_signature"]["numeric_screen"]["status"] == "within_bound"

    bad = selected.replace("1.0625 : f32", "10.0 : f32").replace("0.3125 : f32", "5.1 : f32")
    capsule, directory = _capsule(tmp_path, "bad-with-contract", None, None, text=bad)
    screen = program_admission.screen_written(capsule, directory, target="fixture", evidence=evidence)
    assert screen["status"] == "unsupported"
    assert screen["numeric_screens"][0]["max_error_lsb"] == 5
    entries = program_admission.account_interface(
        directory / "capsule.interface.mlir", target="fixture", evidence=evidence
    )
    [relu] = [row for row in entries if row["operation"] == "relu"]
    [software] = [row for row in relu["software_admissions"] if row["declaration"] == "sum_readout"]
    assert software["status"] == "unsupported"


def test_software_spec_refuses_unknown_numerical_contracts():
    spec = deepcopy(_SPEC)
    spec["operations"]["elementwise_map"]["numerical_contract"] = "unverified"
    with pytest.raises(ValueError, match="unsupported numerical contract"):
        validate_software_spec(spec, "fixture")


def test_a_host_only_probe_may_not_contain_accelerator_work(tmp_path):
    capsule, directory = _capsule(
        tmp_path, "probe", _FUSED, _TENSORS, semantic={"must_accelerate": False, "eligible": False}
    )
    screen = program_admission.screen_written(capsule, directory, target="fixture", evidence=_evidence())
    assert screen["status"] == "unsupported" and "host-only" in screen["reason"]


def test_a_whole_program_capsule_is_held_to_its_derivation_inventory():
    screen = {"status": "admitted", "placements": {"accelerator": 2, "host": 5}}
    saved = {
        "applications": {
            "app": {"capture_sha256": "c" * 64, "status": "admitted", "placements": {"accelerator": 2, "host": 5}}
        }
    }
    evidence = SimpleNamespace(whole_program_admission=saved)
    matched = program_admission._check_saved_inventory(screen, {"capture_sha256": "c" * 64}, evidence)
    assert matched["status"] == "admitted" and matched["saved_inventory"]["status"] == "matched"
    moved = program_admission._check_saved_inventory(
        {**screen, "placements": {"accelerator": 1, "host": 6}}, {"capture_sha256": "c" * 64}, evidence
    )
    assert moved["status"] == "unsupported"
    absent = program_admission._check_saved_inventory(screen, {"capture_sha256": "d" * 64}, evidence)
    assert absent["status"] == "unknown"


_INT_MATMUL = """"builtin.module"() ({
  "func.func"() <{function_type = (tensor<16x32xi8>, tensor<32x16xi8>) -> tensor<16x16xi32>, sym_name = "forward"}> ({
  ^bb0(%x: tensor<16x32xi8>, %w: tensor<32x16xi8>):
    %c = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %e = "tensor.splat"(%c) : (i32) -> tensor<16x16xi32>
    %r = "linalg.generic"(%x, %w, %e) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb1(%p: i8, %q: i8, %s: i32):
      %0 = "arith.extsi"(%p) : (i8) -> i32
      %1 = "arith.extsi"(%q) : (i8) -> i32
      %2 = "arith.muli"(%0, %1) : (i32, i32) -> i32
      %3 = "arith.addi"(%s, %2) : (i32, i32) -> i32
      "linalg.yield"(%3) : (i32) -> ()
    }) {prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.family = "contraction"} : (tensor<16x32xi8>, tensor<32x16xi8>, tensor<16x16xi32>) -> tensor<16x16xi32>
    "func.return"(%r) : (tensor<16x16xi32>) -> ()
  }) : () -> ()
}) : () -> ()
"""


def test_the_derivation_inventory_records_each_capture_placements(tmp_path):
    from merlin.targetgen.application_inventory import application_demand_inventory

    capture = tmp_path / "model.mlir"
    capture.write_text(_INT_MATMUL)
    inventory = application_demand_inventory(
        {"app": capture}, "fixture", detailed=True, capability_contract=_CONTRACT, include_graph=True
    )
    document = program_admission.derive_inventory(
        inventory, target="fixture", evidence=_evidence(), captures={"app": capture}
    )
    row = document["applications"]["app"]
    assert row["status"] == "admitted" and row["placements"] == {"accelerator": 1}
    assert document["sha256"] == program_admission.operations_digest(document)
    with pytest.raises(ValueError, match="exact application roster"):
        program_admission.derive_inventory(inventory, target="fixture", evidence=_evidence(), captures={})
    capture.write_text(_INT_MATMUL + "\n")
    with pytest.raises(ValueError, match="inventoried bytes"):
        program_admission.derive_inventory(inventory, target="fixture", evidence=_evidence(), captures={"app": capture})


def test_only_a_definite_refusal_of_a_non_probe_entry_is_final_before_writing():
    refused, unknown = {"status": "unsupported"}, {"status": "unknown"}
    assert program_admission.entry_refusal_is_final({"op": "bias_add"}, refused)
    assert not program_admission.entry_refusal_is_final({"op": "matmul"}, unknown)
    probe = {"op": "permute", "generalization": {"must_accelerate": False, "eligible": False}}
    assert not program_admission.entry_refusal_is_final(probe, refused)


@pytest.mark.parametrize(
    ("screen", "fragment"),
    [
        ({"status": "admitted", "placements": {"host": 9}}, "host=9"),
        ({"status": "admitted", "placements": {}}, "no placed operation"),
        ({"status": "admitted", "placements": {"accelerator": 0, "host": 1}}, "Merlin-supplied"),
        ({"status": "admitted"}, "no frozen per-operation placement"),
        (None, "no frozen per-operation placement"),
        ({"placements": {"accelerator": "1"}}, "malformed"),
    ],
)
def test_a_program_without_accelerator_placement_is_not_publishable(screen, fragment):
    reason = program_admission.accelerator_placement_refusal(screen)
    assert reason is not None and fragment in reason


def test_a_mixed_program_with_accelerator_placement_is_publishable():
    assert program_admission.accelerator_placement_refusal({"placements": {"accelerator": 1, "host": 15}}) is None


def test_generation_removes_and_records_a_written_member_without_accelerator_placement(tmp_path):
    from merlin_experiments.phase0.generation import withhold_unaccelerated

    host_only = tmp_path / "model_slices" / "host_member"
    host_only.mkdir(parents=True)
    (host_only / "capsule.yaml").write_text("name: host_member\n")
    record = withhold_unaccelerated("host_member", {"status": "admitted", "placements": {"host": 2}}, host_only)
    assert record == {
        "capsule": "host_member",
        "placements": {"host": 2},
        "reason": program_admission.accelerator_placement_refusal({"placements": {"host": 2}}),
    }
    assert not host_only.exists()

    mixed = tmp_path / "model" / "mixed"
    mixed.mkdir(parents=True)
    assert withhold_unaccelerated("mixed", {"status": "admitted", "placements": {"accelerator": 1}}, mixed) is None
    # A refused screen leaves through the diagnostic path, not this one.
    assert withhold_unaccelerated("mixed", {"status": "unsupported", "placements": {}}, mixed) is None
    assert mixed.is_dir()


_GUARD = {"scored": False, "cohort": "guard", "semantic": {"must_accelerate": False}}


def test_unscored_host_guards_are_published_but_other_host_capsules_are_not(tmp_path):
    import yaml
    from merlin_experiments.corpus.publication import require_accelerator_placement
    from merlin_experiments.phase0.generation import withhold_unaccelerated
    from merlin_experiments.spec import SpecError

    from merlin.targetgen.owned_host_leaf_execution import selected_host_leaf

    host = {"status": "admitted", "decisions": [{"placement": "host", "status": "admitted"}], "placements": {"host": 1}}
    root = tmp_path / "public"
    for name, extra in (("guard", _GUARD), ("scored", {"scored": True, "cohort": "public"})):
        (root / name).mkdir(parents=True)
        document = {"name": name, "label": "public", "kind": "model_slice", "software_screen": host, **extra}
        (root / name / "capsule.yaml").write_text(yaml.safe_dump(document))
    with pytest.raises(SpecError, match=r"^1 public capsule\(s\).*scored: no operation"):
        require_accelerator_placement([root])
    (root / "scored" / "capsule.yaml").unlink()
    (root / "scored").rmdir()
    assert require_accelerator_placement([root]) == 1
    # A guard declaring required accelerator work is not a host guard.
    accelerated = {**_GUARD, "semantic": {"must_accelerate": True}}
    (root / "guard" / "capsule.yaml").write_text(
        yaml.safe_dump({"name": "guard", "label": "public", "software_screen": host, **accelerated})
    )
    with pytest.raises(SpecError):
        require_accelerator_placement([root])

    written = tmp_path / "model_slices" / "guard"
    written.mkdir(parents=True)
    assert withhold_unaccelerated("guard", host, written, capsule=_GUARD) is None and written.is_dir()
    # A guard keeps its own host grading instead of the provided-host-infrastructure route.
    assert selected_host_leaf({"kind": "model_slice", "software_screen": host})
    assert not selected_host_leaf({"kind": "model_slice", "software_screen": host, **_GUARD})


_ACT_ACT = ("aligned", "decode", "k_tail")


@pytest.mark.parametrize("mesh_rows", [8, 16, 32])
def test_activation_contraction_members_are_generated_exact_and_released(tmp_path, mesh_rows):
    """Two runtime int8 activations contracted to i32 and reshaped, sized from RTL facts and released."""
    from merlin_experiments.corpus.publication import require_accelerator_placement
    from merlin_experiments.phase0.sweeps import _resolve_flat_extents
    from merlin_experiments.phase0.writer import _write_capsule

    from merlin.targetgen import corpus_spec, corpus_synth, golden_store
    from merlin.targetgen.capsule_inputs import materialize_capsule_leaves

    facts = {"facts": {"arrays": [{"name": "mesh", "rows": mesh_rows, "cols": mesh_rows}]}}
    tile = corpus_spec._tile_dim("fixture", _CONTRACT, operand="int8", facts=facts)
    assert tile == mesh_rows
    requirement = {
        "target": "fixture",
        "cells": [
            {
                "cell": "contraction/i8/aligned",
                "family": "contraction",
                "dtype": "i8",
                "alignment": "aligned",
                "basis": "observed",
            }
        ],
        "boundaries": {"extent_probes": [{"boundary": "tile_edge", "edge": tile, "points": [tile - 1, tile]}]},
    }
    members = {entry["name"]: entry for entry in corpus_synth.synthesize(requirement)["capsules"]}
    binding = corpus_spec.CorpusBinding(
        target="fixture",
        tile_dim=tile,
        operand_dtype="int8",
        accum_dtype="i32",
        integer=True,
        tiers=["L0"],
        compare="exact_int",
    )
    semantics = {
        **_SPEC["numerical_semantics"],
        "internal_arithmetic": {
            "signed_operand_bits": 8,
            "mac_result_bits": 32,
            "full_operation_overflow_policy": "bounded_exact_requires_each_partial_sum",
        },
    }
    root = tmp_path / "generated"
    extents = {}
    for label in _ACT_ACT:
        name = f"SY_act_act_i8_{label}"
        entry = _resolve_flat_extents({**members[name], "numerical_semantics": semantics}, binding)
        extents[label] = (entry["M"], entry["K"], entry["N"])
        directory = Path(_write_capsule(entry, binding, root))
        capsule = yaml.safe_load((directory / "capsule.yaml").read_text())
        # Both operands are runtime activations: no weight role, no scale stream.
        assert [row["role"] for row in capsule["inputs"]] == ["input", "input"]
        assert {row["dtype"] for row in capsule["inputs"]} == {"i8"}
        screen = program_admission.screen_written(capsule, directory, target="fixture", evidence=_evidence())
        assert screen["status"] == "admitted", screen["reason"]
        assert screen["placements"] == {"accelerator": 1}
        assert program_admission.accelerator_placement_refusal(screen) is None
        # Exact integer reference, recomputed here from the declared leaves.
        M, K, N = extents[label]
        leaves = materialize_capsule_leaves(capsule)
        a, b = leaves["A0"].data, leaves["B0"].data
        product = [sum(int(a[i * K + k]) * int(b[k * N + j]) for k in range(K)) for i in range(M) for j in range(N)]
        golden = golden_store.load_golden(directory)["outputs"]["Y0"]
        assert [value for row in golden for value in row] == product
        assert (len(golden), len(golden[0])) == (N, M)
        capsule["software_screen"] = screen
        (directory / "capsule.yaml").write_text(yaml.safe_dump(capsule, sort_keys=False))
    # The release gate refuses zero-accelerator public capsules; these three pass it.
    assert require_accelerator_placement([root]) == len(_ACT_ACT)
    assert extents == {
        "aligned": (tile, 2 * tile, 2 * tile),
        "decode": (1, 2 * tile, 2 * tile),
        "k_tail": (tile, 2 * tile - 1, 2 * tile),
    }
    assert extents["k_tail"][1] % tile != 0


@pytest.mark.parametrize("mesh_rows", [8, 16, 32])
def test_tail_isolation_members_are_generated_with_accelerator_placement(tmp_path, mesh_rows):
    """A ragged-K matmul under aligned M/N and a single-row GEMV with a ragged N, sized from RTL facts."""
    from merlin_experiments.phase0.sweeps import resolve_extent

    from merlin.targetgen import corpus_spec, corpus_synth

    facts = {"facts": {"arrays": [{"name": "mesh", "rows": mesh_rows, "cols": mesh_rows}]}}
    tile = corpus_spec._tile_dim("fixture", _CONTRACT, operand="int8", facts=facts)
    assert tile == mesh_rows
    requirement = {
        "target": "fixture",
        "cells": [
            {
                "cell": "contraction/i8/aligned",
                "family": "contraction",
                "dtype": "i8",
                "alignment": "aligned",
                "basis": "observed",
            }
        ],
        "boundaries": {"extent_probes": [{"boundary": "tile_edge", "edge": tile, "points": [tile - 1, tile]}]},
    }
    members = {entry["name"]: entry for entry in corpus_synth.synthesize(requirement)["capsules"]}
    binding = corpus_spec.CorpusBinding(
        target="fixture",
        tile_dim=tile,
        operand_dtype="int8",
        accum_dtype="i32",
        integer=True,
        tiers=["L0"],
        compare="exact_int",
    )
    extents = {}
    for name in ("SY_contraction_i8_k_tail", "SY_contraction_i8_gemv_n_tail"):
        entry = members[name]
        assert entry["label"] == "public" and entry["generalization"]["generalization_axis"] == "tail_isolation"
        extents[name] = {axis: resolve_extent(entry[axis], tile) for axis in ("M", "K", "N")}
        capsule, program = corpus_spec.build_matmul({**entry, **extents[name]}, binding)
        directory = tmp_path / name
        directory.mkdir()
        (directory / "capsule.interface.mlir").write_text(program)
        screen = program_admission.screen_written(capsule, directory, target="fixture", evidence=_evidence())
        assert screen["status"] == "admitted", screen["reason"]
        assert screen["placements"].get("accelerator", 0) >= 1
        assert program_admission.accelerator_placement_refusal(screen) is None
    k_tail = extents["SY_contraction_i8_k_tail"]
    assert k_tail["M"] % tile == 0 and k_tail["N"] % tile == 0 and k_tail["K"] % tile != 0
    gemv = extents["SY_contraction_i8_gemv_n_tail"]
    assert gemv["M"] == 1 and gemv["N"] % tile != 0 and gemv["K"] % tile == 0
