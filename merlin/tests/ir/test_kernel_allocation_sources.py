"""Independent source-capsule protocol fixtures, without numerical admission."""

import hashlib
import json

import pytest

from merlin.frontends.kernel_allocation_evidence import checked_allocation_sources


def record(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


def fixture(*, normalized=False, descriptor=False, protected=False):
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
    if protected:
        files.update(
            {
                "allocation-generic.mlir": b"complete original generic source",
                "allocation-protected.ttir": b"executable protected source",
            }
        )
        files["allocation-native-print.json"] = encoded(
            {
                "schema": "t2m.native-source-print.v1",
                "triton_version": "3.7.1",
                "flags": ["--mlir-print-op-generic", "--mlir-print-local-scope", "--mlir-print-debuginfo"],
                "operations": 2,
                "normalization": "not-performed",
                "target_backend": "not-invoked",
                "native_source_verified": True,
                "native_generic_roundtrip_verified": True,
                "grants_semantics": False,
                "source_sha256": record(original)["sha256"],
                "output_sha256": record(files["allocation-generic.mlir"])["sha256"],
                "implementation": {
                    "binary_sha256": "3" * 64,
                    "binary_bytes": 128,
                    "printer_options": "MLIR-registerAsmPrinterCLOptions; LLVM-C-LLVMParseCommandLineOptions",
                    "source_reference_revision": "4" * 40,
                    "source_reference_is_build_receipt": False,
                },
            }
        )
        files["allocation-protection.json"] = encoded(
            {
                "schema": "t2m.source-protection.v1",
                "source_entry": "kernel",
                "fp8_overflow": "per-cast",
                "grants_semantics": False,
                "artifacts": {
                    "original": record(original),
                    "generic": record(files["allocation-generic.mlir"]),
                    "protected": record(files["allocation-protected.ttir"]),
                },
                "source_graph": {"operations": [{"opcode": "tt.func"}], "blocks": [], "attributes": "complete"},
                "fp8_conversions": [],
                "assembly_conversions": [],
            }
        )
    initial = files["allocation-protected.ttir"] if protected else original
    if normalized:
        files["allocation-normalized.ttir"] = b"normalized source"
        intake = {
            "schema": "t2m.intake-pass.v1",
            "triton_version": "3.7.1",
            "target_backend": "not-invoked",
            "source_sha256": record(initial)["sha256"],
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
            "source.ttir": record(files["allocation-normalized.ttir"] if normalized else initial),
            "lowered.mlir": record(generic),
            "stderr.log": record(b""),
        },
        "optimizer": {"sha256": "1" * 64, "bytes": 128},
    }
    files["allocation-source-stage.json"] = encoded(stage)
    capsule = {
        "schema": "t2m.source-allocation-capsule.v2" if protected else "t2m.source-allocation-capsule.v1",
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
@pytest.mark.parametrize("protected", [False, True])
def test_checked_source_capsules_copy_only_the_explicit_compiler_roster(normalized, descriptor, protected):
    trace, files, bound = fixture(normalized=normalized, descriptor=descriptor, protected=protected)
    selected = []

    def checked(name):
        selected.append(name)
        return files[name]

    assert checked_allocation_sources(checked, trace, bound) == files
    assert set(selected) == set(files)
    assert all(name.startswith("allocation-") for name in selected)


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize(
    "change",
    [
        "generic",
        "protected",
        "original",
        "entry",
        "authority",
        "evaluator",
        "printer-source",
        "printer-flags",
        "printer-roundtrip",
        "printer-authority",
        "printer-evaluator",
        "printer-binary",
        "normalizer-input",
        "downgrade",
        "missing",
    ],
)
def test_rehashed_protected_source_chain_refuses_inconsistent_links(normalized, change):
    trace, files, bound = fixture(normalized=normalized, protected=True)
    capsule = trace["source_argument_binding"]["source_capsule"]
    name = "allocation-native-print.json" if change.startswith("printer-") else "allocation-protection.json"
    report = json.loads(files[name])
    if change in ("generic", "protected", "original"):
        report["artifacts"][change] = record(b"different compiler source")
    elif change == "entry":
        report["source_entry"] = "other"
    elif change in ("authority", "printer-authority"):
        report["grants_semantics"] = True
    elif change in ("evaluator", "printer-evaluator"):
        report["reference_sha256"] = "5" * 64
    elif change == "printer-source":
        report["source_sha256"] = "5" * 64
    elif change == "printer-flags":
        report["flags"].append("--canonicalize")
    elif change == "printer-roundtrip":
        report["native_generic_roundtrip_verified"] = False
    elif change == "printer-binary":
        report["implementation"]["binary_sha256"] = "g" * 64
    elif change == "downgrade":
        capsule["schema"] = "t2m.source-allocation-capsule.v1"
    elif change == "missing":
        del files["allocation-protected.ttir"]
    else:
        name = "allocation-intake.json" if normalized else "allocation-source-stage.json"
        report = json.loads(files[name])
        if normalized:
            report["source_sha256"] = record(files["allocation-original.ttir"])["sha256"]
        else:
            report["artifacts"]["source.ttir"] = record(files["allocation-original.ttir"])
    files[name] = encoded(report)
    capsule["artifacts"] = {name: record(data) for name, data in files.items()}
    with pytest.raises(ValueError, match="source|protection|roster"):
        checked_allocation_sources(files.__getitem__, trace, bound)


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
