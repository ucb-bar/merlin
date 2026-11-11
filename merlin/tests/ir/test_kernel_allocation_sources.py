"""Independent source-capsule protocol fixtures, without numerical admission."""

import hashlib
import json

import pytest

from merlin.frontends.kernel_allocation_evidence import checked_allocation_sources


def record(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


def fixture(*, normalized=False, descriptor=False):
    original, generic, bound = b"original source", b"generic source", b"bound source"
    arguments = {
        "schema": "t2m.source-allocation-arguments.v2" if descriptor else "t2m.source-allocation-arguments.v1",
        "source_entry": "kernel",
        "arguments": [],
    }
    files = {
        "allocation-original.ttir": original,
        "allocation-input.mlir": generic,
        "allocation-arguments.json": encoded(arguments),
        "allocation-source-stderr.log": b"",
    }
    if normalized:
        files["allocation-normalized.ttir"] = b"normalized source"
        intake = {
            "schema": "t2m.intake-pass.v1",
            "triton_version": "3.7.1",
            "target_backend": "not-invoked",
            "source_sha256": record(original)["sha256"],
            "output_sha256": record(files["allocation-normalized.ttir"])["sha256"],
            "pipeline": ["triton-rewrite-tensor-descriptor-to-pointer", "canonicalize"],
            "normalized_opcodes": ["tt.func"],
        }
        files["allocation-intake.json"] = encoded(intake)
        files["allocation-inventory.json"] = encoded(
            {"normalization_passes": intake["pipeline"], "normalized_opcodes": intake["normalized_opcodes"]}
        )
    stage = {
        "schema": "t2m.source-stage-custody.v1",
        "source_stage_schema": "t2m.triton-shared-stage.v1",
        "stage": "source",
        "stage_completed": True,
        "grants_semantics": False,
        "flags": [
            "--mlir-disable-threading",
            "--verify-each",
            "--mlir-print-op-generic",
            "--mlir-print-debuginfo",
            "--mlir-print-local-scope",
        ],
        "artifacts": {
            "source.ttir": record(files["allocation-normalized.ttir"] if normalized else original),
            "lowered.mlir": record(generic),
            "stderr.log": record(b""),
        },
        "optimizer": {"sha256": "1" * 64, "bytes": 128},
    }
    files["allocation-source-stage.json"] = encoded(stage)
    capsule = {
        "schema": "t2m.source-allocation-capsule.v1",
        "normalized": normalized,
        "artifacts": {name: record(data) for name, data in files.items()},
        "grants_semantics": False,
    }
    binding = {
        "schema": "t2m.source-allocation-bindings.v1",
        "source_entry": "kernel",
        "allocation_entry": "state",
        "grants_semantics": False,
        "original_source_sha256": record(original)["sha256"],
        "bound_source_sha256": record(bound)["sha256"],
        "source_capsule": capsule,
    }
    if descriptor:
        binding["descriptor_bindings"] = {
            "schema": arguments["schema"],
            "source_entry": "kernel",
            "allocation_entry": "state",
            "grants_semantics": False,
        }
    return {"spec": {"entry": "state"}, "source_argument_binding": binding}, files, bound


@pytest.mark.parametrize("normalized,descriptor", [(False, False), (True, False), (True, True)])
def test_checked_source_capsules_copy_only_the_explicit_compiler_roster(normalized, descriptor):
    trace, files, bound = fixture(normalized=normalized, descriptor=descriptor)
    selected = []

    def checked(name):
        selected.append(name)
        return files[name]

    assert checked_allocation_sources(checked, trace, bound) == files
    assert set(selected) == set(files)
    assert all(name.startswith("allocation-") for name in selected)


@pytest.mark.parametrize(
    "change",
    (
        "capsule_schema",
        "authority",
        "marker",
        "roster",
        "original",
        "bound",
        "entry",
        "arguments",
        "stage_input",
        "stage_flags",
        "stage_evaluator",
        "optimizer",
        "intake_input",
        "intake_output",
        "intake_inventory",
        "intake_evaluator",
        "descriptor_schema",
        "descriptor_authority",
        "drop_capsule",
    ),
)
def test_rehashed_source_capsule_inconsistencies_refuse(change):
    trace, files, bound = fixture(normalized=True, descriptor=True)
    binding = trace["source_argument_binding"]
    capsule = binding["source_capsule"]
    if change == "capsule_schema":
        capsule["schema"] = "unknown"
    elif change == "authority":
        capsule["grants_semantics"] = True
    elif change == "marker":
        capsule["normalized"] = 1
    elif change == "roster":
        capsule["artifacts"]["inputs.npz"] = record(b"evaluator")
    elif change == "original":
        files["allocation-original.ttir"] = b"changed original"
    elif change == "bound":
        bound = b"changed bound"
    elif change == "entry":
        binding["allocation_entry"] = "other"
    elif change == "drop_capsule":
        binding.pop("source_capsule")
    elif change.startswith("descriptor_"):
        binding["descriptor_bindings"]["schema" if change.endswith("schema") else "grants_semantics"] = (
            "unknown" if change.endswith("schema") else True
        )
    else:
        name = (
            "allocation-arguments.json"
            if change == "arguments"
            else "allocation-source-stage.json"
            if change.startswith(("stage_", "optimizer"))
            else "allocation-intake.json"
        )
        data = json.loads(files[name])
        if change == "arguments":
            data["source_entry"] = "other"
        elif change == "stage_input":
            data["artifacts"]["source.ttir"] = record(b"wrong")
        elif change == "stage_flags":
            data["flags"].append("--canonicalize")
        elif change == "stage_evaluator":
            data["reference_sha256"] = "2" * 64
        elif change == "optimizer":
            data["optimizer"]["sha256"] = "g" * 64
        elif change == "intake_input":
            data["source_sha256"] = "2" * 64
        elif change == "intake_output":
            data["output_sha256"] = "2" * 64
        elif change == "intake_inventory":
            data["pipeline"].append("other")
        elif change == "intake_evaluator":
            data["reference_sha256"] = "2" * 64
        files[name] = encoded(data)
    if "source_capsule" in binding and change != "roster":
        capsule["artifacts"] = {name: record(data) for name, data in files.items()}
    selected = []

    def checked(name):
        selected.append(name)
        return files[name]

    with pytest.raises(ValueError, match="source|descriptor|optimizer"):
        checked_allocation_sources(checked, trace, bound)
    assert "inputs.npz" not in selected


def test_legacy_source_bindings_keep_their_original_custody_scope():
    trace, _, bound = fixture()
    trace["source_argument_binding"].pop("source_capsule")
    assert (
        checked_allocation_sources(
            lambda _: pytest.fail("legacy reader requested an absent source capsule"), trace, bound
        )
        == {}
    )
