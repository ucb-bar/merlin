"""Actual pinned planner/fusion qualification; kernel and host refusals stay distinct."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.verify import lean_graph_regions as checker
from merlin.verify import lean_graph_regions_capture as producer
from merlin.verify import lean_graph_regions_fusion as fusion
from merlin.verify import lean_graph_regions_mlir as compiler
from merlin.verify import lean_graph_regions_projection as projection


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


@pytest.fixture(scope="module")
def packet():
    root = repo_root()
    output = Path(
        os.environ.get(
            "MERLIN_GRAPH_REGIONS_QUALIFICATION",
            str(root / "out/artifacts/verification/graph-regions" / ("qualification-" + uuid.uuid4().hex)),
        )
    )
    output.mkdir(parents=True, exist_ok=False)
    reference = Path("/home/jeremy/Projects/merlin-graph-proof-intake")
    oracle = Path("/home/jeremy/Projects/model2MLIR/.venv/bin/python")
    scopes = {
        "observed": producer.capture(reference, output / "observed"),
        "declined": producer.capture(reference, output / "declined", realize=True),
        "independent": producer.capture(reference, output / "independent", shape=(2, 11), stages=2),
        "realized": producer.capture(
            reference, output / "realized", observed=False, realize=True, suffix_observer=True
        ),
    }
    validity = {}
    plans = {}
    for name, scope in scopes.items():
        for role in ("source", "retained_source"):
            validity[name, role] = compiler.compiler_verify(
                Path(scope[role]["path"]), output / f"valid-{name}-{role}", python=oracle
            )
            assert validity[name, role]["oracle"]["valid"] is True
        plans[name] = checker.verify_plan(
            Path(scope["source"]["path"]),
            Path(scope["plan"]["path"]),
            scope["selected_ordinals"],
            output / f"plan-{name}",
            retained_source=Path(scope["retained_source"]["path"]),
        )
        assert plans[name]["accepted"] is True
    realized = fusion.verify_fusion(
        output / "realized/CAPTURE.json", output / "realized/fused-body.json", output / "fusion-realized"
    )
    assert realized["accepted"] is True
    return dict(output=output, scopes=scopes, plans=plans, realized=realized, oracle=oracle, validity=validity)


def test_actual_complete_boundaries(packet):
    observed = packet["scopes"]["observed"]
    assert packet["scopes"]["declined"]["realization"]["status"] == "declined"  # escaped P
    assert len(json.loads(Path(observed["plan"]["path"]).read_text())["outputs"]) == 3
    independent = packet["scopes"]["independent"]
    assert independent["shape"] == [2, 11] and len(independent["selected_ordinals"]) == 2
    graph = packet["plans"]["realized"]["projection"]
    assert len(graph["nodes"]) == 4  # unchanged observer follows selected three-stage region
    assert packet["realized"]["actual_correspondence_instantiated"] is True
    assert packet["realized"]["native_implementation_contract_proved"] is False
    assert packet["realized"]["fresh_native_reproduction"]["artifacts_compared"] >= 15


def mutate_plan(plan, fault):
    if fault == "drop-producer-observer":
        plan["outputs"].pop(0)
    elif fault == "drop-second-output":
        plan["outputs"].pop()
    elif fault == "drop-second-consumer":
        plan["stages"].pop()
    elif fault == "wrong-same-typed-input":
        plan["stages"][0]["inputs"][0] = 1
    elif fault == "wrong-same-typed-result":
        plan["stages"][1]["outputs"] = plan["stages"][2]["outputs"][:]
    elif fault == "wrong-semantic-body":
        plan["stages"][0]["semanticKey"] = "0" * 64
    elif fault == "wrong-ordinal":
        plan["stages"][0]["ordinal"] += 1
    elif fault == "wrong-type":
        plan["stages"][0]["operandTypes"][0] = "tensor<3x5xi8>"
    elif fault == "input-order":
        plan["inputs"].reverse()
    elif fault == "output-order":
        plan["outputs"].reverse()
    elif fault == "early-overwrite":
        plan["lifetimes"][2][1] = 1
    elif fault == "wrong-value-identity":
        plan["values"][0] = plan["values"][1]
    elif fault == "out-of-range-index":
        plan["stages"][0]["inputs"][0] = 999
    else:
        raise AssertionError(fault)


@pytest.mark.parametrize(
    "fault",
    [
        "drop-producer-observer",
        "drop-second-output",
        "drop-second-consumer",
        "wrong-same-typed-input",
        "wrong-same-typed-result",
        "wrong-semantic-body",
        "wrong-ordinal",
        "wrong-type",
        "input-order",
        "output-order",
        "early-overwrite",
        "wrong-value-identity",
        "out-of-range-index",
    ],
)
def test_kernel_plan_mutations(packet, fault):
    scope = packet["scopes"]["observed"]
    plan = json.loads(Path(scope["plan"]["path"]).read_text())
    mutate_plan(plan, fault)
    output = packet["output"] / ("mutation-" + fault)
    output.mkdir()
    proposal = save(output / "plan.json", plan)
    # All submitted programs are the unchanged actual compiler-verified programs;
    # only the actual proposed graph representation is mutated in these cases.
    result = checker.verify_plan(
        Path(scope["source"]["path"]),
        proposal,
        scope["selected_ordinals"],
        output / "kernel",
        retained_source=Path(scope["retained_source"]["path"]),
    )
    assert result["accepted"] is False
    save(
        output / "QUALIFICATION.json",
        dict(
            rejection_layer="Lean checkBoundary",
            source_validity=packet["validity"]["observed", "source"],
            evidence=checker.file_ref(output / "kernel/EVIDENCE.json"),
        ),
    )


@pytest.mark.parametrize("fault", ["unchanged-return-observer", "unchanged-suffix-binding"])
def test_compiler_valid_changed_untouched_observers(packet, fault):
    name = "realized" if fault == "unchanged-suffix-binding" else "observed"
    scope = packet["scopes"][name]
    module = projection.parse_source(Path(scope["source"]["path"]).read_text())
    function = next(iter(module.body.block.ops))
    if name == "observed":
        term = tuple(function.body.block.ops)[-1]
        term.operands = (term.operands[1], *term.operands[1:])
    else:
        nodes = [op for op in module.walk() if op.name == "linalg.generic"]
        nodes[-1].operands = (nodes[0].results[0], *nodes[-1].operands[1:])
    module.verify()
    output = packet["output"] / fault
    output.mkdir()
    after = output / "after.mlir"
    after.write_text(projection.ir_text(module))
    valid = compiler.compiler_verify(after, output / "compiler", python=packet["oracle"])
    assert valid["oracle"]["valid"] is True
    result = checker.verify_plan(
        Path(scope["source"]["path"]),
        Path(scope["plan"]["path"]),
        scope["selected_ordinals"],
        output / "kernel",
        retained_source=after,
    )
    assert result["accepted"] is False


@pytest.mark.parametrize("fault", ["drop-second-output", "wrong-same-typed-input", "drop-second-consumer"])
def test_integrated_kernel_realization_mutations(packet, fault):
    body = json.loads((packet["output"] / "realized/fused-body.json").read_text())
    if fault == "drop-second-output":
        body["outputs"].pop()
    elif fault == "wrong-same-typed-input":
        body["body"][0]["operands"][0] = body["body"][0]["operands"][1]
    else:
        body["body"].pop()
    output = packet["output"] / ("fusion-" + fault)
    output.mkdir()
    result = fusion.verify_fusion(
        packet["output"] / "realized/CAPTURE.json", save(output / "body.json", body), output / "kernel"
    )
    assert result["accepted"] is False


def test_authoritative_compiler_refuses_invalid_ssa(packet):
    text = Path(packet["scopes"]["observed"]["source"]["path"]).read_text()
    text = text.replace('"func.return"(', '"func.return"(%never_defined,', 1)
    source = packet["output"] / "invalid-ssa.mlir"
    source.write_text(text)
    assert (
        compiler.compiler_verify(source, packet["output"] / "invalid-ssa", python=packet["oracle"])["oracle"]["valid"]
        is False
    )


@pytest.mark.parametrize("fault", ["initializer-read", "partial-division", "overflow-flag"])
def test_explicitly_unsupported_semantics(packet, fault):
    module = projection.parse_source(Path(packet["scopes"]["observed"]["source"]["path"]).read_text())
    node = next(op for op in module.walk() if op.name == "linalg.generic")
    block = node.body.block
    if fault == "initializer-read":
        tuple(block.ops)[-1].operands = (block.args[-1],)
        text = projection.ir_text(module)
    elif fault == "partial-division":
        text = projection.ir_text(module).replace('"arith.addi"', '"arith.divsi"', 1)
        text = text.replace(" <{overflowFlags = #arith.overflow<none>}>", "", 1)
    else:
        text = projection.ir_text(module).replace("#arith.overflow<none>", "#arith.overflow<nsw>", 1)
    output = packet["output"] / ("unsupported-" + fault)
    output.mkdir()
    source = output / "source.mlir"
    source.write_text(text)
    assert compiler.compiler_verify(source, output / "compiler", python=packet["oracle"])["oracle"]["valid"] is True
    with pytest.raises(projection.UnsupportedGraphRegion):
        projection.project(projection.parse_source(text))


@pytest.mark.parametrize("role", ["source_original", "retained_source_original", "plan_original"])
@pytest.mark.parametrize("fault", ["substituted", "missing", "boolean-size"])
def test_original_snapshot_binding_refusals(packet, role, fault):
    receipt = copy.deepcopy(packet["plans"]["observed"])
    if fault == "substituted":
        other = packet["plans"]["independent"][role]
        receipt[role] = other
    elif fault == "boolean-size":
        receipt[role]["bytes"] = True
    else:
        receipt[role]["path"] = str(packet["output"] / "absent-original")
    with pytest.raises((ValueError, FileNotFoundError)):
        checker.validate_bindings(receipt)


@pytest.mark.parametrize(
    "fault",
    [
        "claim",
        "semantic-source",
        "tool",
        "stale-proof",
        "projection",
        "forged-source-pair",
        "rehashed-proof",
        "rehashed-audit",
        "consistent-source-substitution",
        "worker-profile",
    ],
)
def test_receipt_authority_refusals(packet, fault):
    receipt = copy.deepcopy(packet["plans"]["observed"])
    if fault == "claim":
        receipt["approved_pass_verification"] = True
    elif fault == "semantic-source":
        receipt["identity"]["sources"]["GraphRegions"] = receipt["source_original"]
    elif fault == "tool":
        receipt["identity"]["lean"] = receipt["source_original"]
    elif fault == "stale-proof":
        receipt["kernel_proof"]["sha256"] = "0" * 64
    elif fault == "projection":
        receipt["projection"]["observations"].pop()
    elif fault in ("rehashed-proof", "rehashed-audit"):
        receipt["kernel_proof" if fault == "rehashed-proof" else "kernel_audit"] = receipt["source_original"]
    elif fault == "worker-profile":
        receipt["identity"]["lean_arguments"] = []
    elif fault == "consistent-source-substitution":
        other = packet["plans"]["independent"]
        for key in (
            "source_original",
            "retained_source_original",
            "plan_original",
            "snapshots",
            "projection",
            "after_projection",
            "selection",
        ):
            receipt[key] = other[key]
    else:
        other = packet["plans"]["independent"]
        receipt["source_original"] = other["source_original"]
        receipt["snapshots"][0] = other["snapshots"][0]
    with pytest.raises(ValueError):
        checker.replay_plan(receipt, packet["output"] / ("rejected-replay-" + fault))


@pytest.mark.parametrize("fault", ["rehashed-proof", "rehashed-audit", "approved-claim", "swapped-capture"])
def test_fusion_receipt_refusals(packet, fault):
    receipt = copy.deepcopy(packet["realized"])
    if fault == "approved-claim":
        receipt["approved_pass_verification"] = True
    elif fault == "swapped-capture":
        receipt["capture"] = checker.file_ref(packet["output"] / "observed/CAPTURE.json")
    else:
        receipt["proof" if fault == "rehashed-proof" else "audit"] = receipt["capture"]
    with pytest.raises(ValueError):
        fusion.replay_fusion(receipt, packet["output"] / ("fusion-rejected-" + fault))


@pytest.mark.parametrize("fault", ["image", "specification", "swapped-source-association", "revision", "context"])
def test_rehashed_native_artifact_refusals(packet, fault):
    origin = packet["output"] / "realized"
    output = packet["output"] / ("rehashed-native-" + fault)
    shutil.copytree(origin, output)
    scope = json.loads((output / "CAPTURE.json").read_text().replace(str(origin), str(output)))
    original_scope = packet["scopes"]["realized"]
    # Relocate submitted artifacts while preserving the original selected-input
    # locations, which are part of the native context identity.
    for key in ("selected_fixture_origin", "selected_inputs"):
        scope[key] = original_scope[key]
    if fault in ("image", "specification"):
        name = "actual-image-0.bin" if fault == "image" else "selected-fixture/software.yaml"
        path = output / name
        path.write_bytes(path.read_bytes() + (b"\x00" if fault == "image" else b"\n# substituted\n"))
        scope["artifacts"][name] = checker.file_ref(path)
        if fault == "image":
            scope["realization"]["actual_images"][0] = checker.file_ref(path)
    elif fault == "revision":
        scope["production_revision"] = "0" * 40
    elif fault == "context":
        scope["realization"]["source_passes"][0]["before"]["context"]["execution"] = "0" * 64
    else:
        scope["source"] = packet["scopes"]["independent"]["source"]
    save(output / "CAPTURE.json", scope)
    with pytest.raises(ValueError):
        fusion.verify_fusion(output / "CAPTURE.json", output / "fused-body.json", output / "check")


@pytest.mark.parametrize("kind", ["plan-positive", "plan-refused", "fusion-positive", "fusion-refused"])
def test_fresh_process_replay(packet, kind):
    names = {
        "plan-positive": "plan-observed/EVIDENCE.json",
        "plan-refused": "unchanged-return-observer/kernel/EVIDENCE.json",
        "fusion-positive": "fusion-realized/EVIDENCE.json",
        "fusion-refused": "fusion-drop-second-output/kernel/EVIDENCE.json",
    }
    receipt = packet["output"] / names[kind]
    destination = packet["output"] / ("fresh-process-" + kind)
    module = "lean_graph_regions_fusion" if kind.startswith("fusion") else "lean_graph_regions"
    function = "replay_fusion" if kind.startswith("fusion") else "replay_plan"
    code = (
        f"import json,sys; from pathlib import Path; from merlin.verify.{module} import {function}; "
        f"r={function}(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]));print(json.dumps(r['accepted']))"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(receipt), str(destination)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=repo_root(),
    )
    save(
        packet["output"] / ("fresh-process-" + kind + ".json"),
        dict(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            original_receipt=checker.file_ref(receipt),
        ),
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) is kind.endswith("positive")
