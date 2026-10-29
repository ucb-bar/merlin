"""Bind a live native fusion artifact's logical body to the checked graph contract.

Kernel proofs establish logical fusion and complete-context composition. Native
source/effect projection remains trusted Python; declared implementation-law truth
is not inferred from saved flags, graph equality, or tested sample outputs.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from merlin.verify import lean_graph_regions as base
from merlin.verify import lean_graph_regions_capture as producer
from merlin.verify import lean_graph_regions_dependencies as dependencies
from merlin.verify import lean_graph_regions_projection as projection

FUSION_ROOTS = (
    "logicalFusion_law",
    "checkFusion_preserves_observers",
    "checked_native_contract",
    "checkRealization_sound",
    "checkRealization_native",
    "inhabited_domain",
)


def verify_fusion(capture: Path, fused_body: Path, destination: Path) -> dict:
    """Recompute source graph and kernel-check the actual serialized native body."""
    destination.mkdir(parents=True, exist_ok=False)
    originals, snapshots = [], []
    for original, name in ((capture, "capture.json"), (fused_body, "fused-body.json")):
        reference = base.file_ref(original)
        snapshot = destination / name
        snapshot.write_bytes(original.read_bytes())
        copied = base.file_ref(snapshot)
        if (copied["sha256"], copied["bytes"]) != (reference["sha256"], reference["bytes"]):
            raise ValueError("fusion original/snapshot content differs")
        originals.append(reference)
        snapshots.append(copied)
    scope = json.loads((destination / "capture.json").read_text())
    if scope.get("schema") != "merlin.graph-regions.capture.v1":
        raise ValueError("unsupported capture protocol")
    if scope.get("realization", {}).get("status") != "applied":
        raise projection.UnsupportedGraphRegion("no admitted native realization in this capture")
    # Source/tool/producer references are fresh evidence bindings, not semantic
    # premises. A new native process must reissue and revalidate its own product.
    references = [
        scope[k] for k in ("source", "plan", "retained_source", "generator", "projector", "python_executable")
    ]
    references += [r["file"] for r in scope["production_sources"]]
    references += scope["realization"]["actual_images"]
    if type(scope.get("artifacts")) is not dict or not scope["artifacts"]:
        raise ValueError("complete native artifact inventory required")
    references += list(scope["artifacts"].values())
    if type(scope.get("selected_inputs")) is not dict or set(scope["selected_inputs"]) != {
        "facts.json",
        "software.yaml",
    }:
        raise ValueError("original selected native inputs required")
    references += list(scope["selected_inputs"].values())
    if scope.get("public_synthetic_fixture") is not None:
        references.append(scope["public_synthetic_fixture"])
    for record in references:
        base.fresh_ref(record)
    native_replay = producer.reproduce_native(scope, destination / "fresh-native")
    law_ref = scope["artifacts"].get("declared-composition-law.json")
    law = json.loads(base.fresh_ref(law_ref).read_text())
    premises = (
        "complete_projected_effects",
        "synchronous_private_composition",
        "suppressed_intermediate_transfers_no_faults",
        "suppressed_intermediate_transfers_unobserved",
    )
    if any(type(law.get(k)) is not bool or law[k] is not True for k in premises):
        raise projection.UnsupportedGraphRegion("native composition declaration is unavailable")
    native = json.loads((destination / "fused-body.json").read_text())
    if type(native) is not dict or set(native) != {"body", "inputs", "outputs"}:
        raise ValueError("complete logical fusion body required")
    node_keys = {"ordinal", "semanticKey", "operands", "results", "operandTypes", "resultTypes"}
    if type(native["body"]) is not list or any(type(n) is not dict or set(n) != node_keys for n in native["body"]):
        raise ValueError("complete ordered primitive bindings required")
    # First qualify the actual source/retained source/planner boundary, including
    # untouched observers. No native realization status substitutes for this check.
    boundary = base.verify_plan(
        Path(scope["source"]["path"]),
        Path(scope["plan"]["path"]),
        scope["selected_ordinals"],
        destination / "boundary",
        retained_source=Path(scope["retained_source"]["path"]),
    )
    source = projection.parse_source(Path(scope["source"]["path"]).read_text())
    graph, _, _ = projection.project(source)
    selected = scope["selected_ordinals"]
    region = [node for node in graph["nodes"] if node["ordinal"] in selected]
    last = next(i for i, node in enumerate(graph["nodes"]) if node["ordinal"] == selected[-1])
    suffix = graph["nodes"][last + 1 :]
    boundary_terms = (destination / "boundary/query.lean").read_text().rsplit("#reduce ", 1)[0]
    boundary_terms = boundary_terms.removeprefix("import Merlin.GraphRegions\nopen Merlin.GraphRegions\n")
    text = (
        "import Merlin.GraphRegionsFusion\nopen Merlin.GraphRegions\n"
        + boundary_terms
        + "def actualRegion : List Node := ["
        + ",".join(base._node(n) for n in region)
        + "]\n"
        + "def actualSuffix : List Node := ["
        + ",".join(base._node(n) for n in suffix)
        + "]\n"
        + "def actualObservers : List Nat := "
        + base._nats(graph["observations"])
        + "\n"
        + "def actualFusion : FusedBody := ⟨["
        + ",".join(base._node(n) for n in native["body"])
        + "],"
        + base._nats(native["inputs"])
        + ","
        + base._nats(native["outputs"])
        + "⟩\n"
    )
    lean, environment, identity = base.build(destination / "fusion-build", fusion=True)
    identity["host_sources"]["lean_graph_regions_fusion"] = base.file_ref(Path(__file__))
    identity["host_sources"]["lean_graph_regions_capture"] = base.file_ref(Path(producer.__file__))
    start = next(i for i, node in enumerate(graph["nodes"]) if node["ordinal"] == selected[0])
    check = (
        f"checkRealization actualBefore actualAfter {start} {len(selected)} "
        f"{base._nats(selected)} actualPlan actualFusion"
    )
    query = destination / "query.lean"
    query.write_text(text + "#reduce " + check + "\n")
    verdict = base._run([str(lean), str(query)], destination, "kernel-reduce", environment).stdout.strip()
    if verdict not in ("true", "false"):
        raise ValueError("kernel did not produce a fusion verdict")
    proof = destination / "proof.lean"
    specialization = ""
    if verdict == "true":
        specialization = (
            "theorem actualCorrespondence (meaning : ∀ node, TypedAction node Value typing)\n"
            "    (initial : Merlin.StateEffects.State Value) (domain : WellTyped typing initial) :\n"
            "    actualBefore.Valid ∧ actualBefore = actualAfter ∧\n"
            "    actualBefore.observations.map (executeNodes meaning "
            f"(actualBefore.nodes.drop ({start} + {len(selected)}))\n"
            "      (actualFusion.execute meaning "
            f"(executeNodes meaning (actualBefore.nodes.take {start}) initial))) =\n"
            "      actualBefore.observations.map (executeNodes meaning actualBefore.nodes initial) :=\n"
            "  checkRealization_sound (before := actualBefore) (after := actualAfter) "
            f"(start := {start}) (count := {len(selected)}) "
            f"(selection := {base._nats(selected)}) (plan := actualPlan) (fused := actualFusion) "
            "actualFusionVerdict meaning initial domain\n"
            "#print axioms actualCorrespondence\n"
        )
    proof.write_text(
        text
        + f"theorem actualFusionVerdict : {check} = {verdict} := by decide\n"
        + "#print axioms actualFusionVerdict\n"
        + specialization
        + "".join(f"#print axioms {name}\n" for name in FUSION_ROOTS)
    )
    audit = base._run([str(lean), str(proof)], destination, "kernel-proof", environment).stdout
    for block in audit.split("depends on axioms:")[1:]:
        names = [n.strip() for n in block.partition("[")[2].partition("]")[0].split(",") if n.strip()]
        if not set(names) <= base.ALLOWED_AXIOMS:
            raise ValueError("unapproved fusion theorem dependency")
    expected_audits = len(FUSION_ROOTS) + 1 + (verdict == "true")
    if audit.count("depends on axioms:") + audit.count("does not depend on any axioms") != expected_audits:
        raise ValueError("incomplete fusion theorem audit")
    for record in references:
        base.fresh_ref(record)
    for original, snapshot in zip(originals, snapshots, strict=True):
        if base.fresh_ref(original).read_bytes() != base.fresh_ref(snapshot).read_bytes():
            raise ValueError("fusion original/snapshot actual bytes differ")
    result = {
        "schema": "merlin.graph-regions.kernel-fusion-evidence.v1",
        "accepted": boundary["accepted"] and verdict == "true",
        "boundary": base.file_ref(destination / "boundary/EVIDENCE.json"),
        "capture": originals[0],
        "fused_body": originals[1],
        "snapshots": snapshots,
        "identity": identity,
        "fresh_native_reproduction": native_replay,
        "proof": base.file_ref(proof),
        "audit": base.file_ref(destination / "kernel-proof.stdout"),
        "theorem": "Merlin.GraphRegions.checkRealization_sound",
        "native_contract_theorem": "Merlin.GraphRegions.checkRealization_native",
        "actual_correspondence_instantiated": verdict == "true",
        "logical_local_law_proved": True,
        "native_implementation_contract_proved": False,
        "projection_trusted_python": True,
        "composition_declarations_provisional": True,
        "shared_pipeline_acceptance_issued": False,
        "approved_pass_verification": False,
        "native_or_hardware_fidelity_proved": False,
    }
    (destination / "EVIDENCE.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def replay_fusion(receipt: dict, destination: Path | None = None) -> dict:
    """Fresh production and kernel execution; serialized receipts are descriptive."""
    expected = {
        "schema": "merlin.graph-regions.kernel-fusion-evidence.v1",
        "theorem": "Merlin.GraphRegions.checkRealization_sound",
        "native_contract_theorem": "Merlin.GraphRegions.checkRealization_native",
        "logical_local_law_proved": True,
        "native_implementation_contract_proved": False,
        "projection_trusted_python": True,
        "composition_declarations_provisional": True,
        "shared_pipeline_acceptance_issued": False,
        "approved_pass_verification": False,
        "native_or_hardware_fidelity_proved": False,
    }
    if any(type(receipt.get(k)) is not type(v) or receipt[k] != v for k, v in expected.items()):
        raise ValueError("changed scoped fusion claim")
    if type(receipt.get("accepted")) is not bool or type(receipt.get("actual_correspondence_instantiated")) is not bool:
        raise ValueError("Boolean fusion verdict required")
    originals = [receipt.get(k) for k in ("capture", "fused_body")]
    snapshots = receipt.get("snapshots")
    if type(snapshots) is not list or len(snapshots) != 2:
        raise ValueError("complete fusion snapshot inventory required")
    for original, snapshot in zip(originals, snapshots, strict=True):
        if base.fresh_ref(original).read_bytes() != base.fresh_ref(snapshot).read_bytes():
            raise ValueError("fusion original/snapshot actual bytes differ")
    for key in ("proof", "audit", "boundary"):
        base.fresh_ref(receipt.get(key))
    for key in ("sources", "objects", "host_sources"):
        if type(receipt.get("identity", {}).get(key)) is not dict or not receipt["identity"][key]:
            raise ValueError("complete fusion identity required")
        for record in receipt["identity"][key].values():
            base.fresh_ref(record)
    base.fresh_ref(receipt["identity"].get("lean"))
    if receipt["identity"]["lean"]["sha256"] != base.LEAN_SHA:
        raise ValueError("changed fusion Lean tool")
    if receipt["identity"].get("axiom_policy") != sorted(base.ALLOWED_AXIOMS):
        raise ValueError("changed fusion axiom policy")
    if receipt["identity"].get("lean_arguments") != ["-j1"]:
        raise ValueError("changed fusion Lean execution profile")
    dependencies.validate(receipt["identity"].get("toolchain_dependencies"), base.fresh_ref)
    boundary = json.loads(base.fresh_ref(receipt["boundary"]).read_text())
    target = destination or Path(receipt["proof"]["path"]).parent / ("replay-" + uuid.uuid4().hex)
    target.mkdir(parents=True, exist_ok=False)
    base.replay_plan(boundary, target / "boundary-replay")
    repeated = verify_fusion(base.fresh_ref(originals[0]), base.fresh_ref(originals[1]), target / "fusion-replay")
    old_dependencies = receipt["identity"]["toolchain_dependencies"]
    if repeated["identity"]["toolchain_dependencies"]["sha256"] != old_dependencies["sha256"]:
        raise ValueError("replayed fusion core dependency differs")
    for key in ("accepted", "actual_correspondence_instantiated"):
        if repeated[key] is not receipt[key]:
            raise ValueError("replayed fusion verdict differs")
    for key in ("proof", "audit"):
        if base.fresh_ref(receipt[key]).read_bytes() != base.fresh_ref(repeated[key]).read_bytes():
            raise ValueError("replayed fusion proof/audit bytes differ")
    for key in ("sources", "objects", "host_sources"):
        old, new = receipt["identity"][key], repeated["identity"][key]
        if set(old) != set(new) or any(
            (ref["sha256"], ref["bytes"]) != (new[name]["sha256"], new[name]["bytes"]) for name, ref in old.items()
        ):
            raise ValueError("replayed fusion dependency differs")
    if repeated["fresh_native_reproduction"] != receipt["fresh_native_reproduction"]:
        # Reproduction paths differ; every non-path content identity must match.
        old, new = receipt["fresh_native_reproduction"], repeated["fresh_native_reproduction"]
        if {k: v for k, v in old.items() if k != "reproduced_capture"} != {
            k: v for k, v in new.items() if k != "reproduced_capture"
        }:
            raise ValueError("replayed native reproduction claim differs")
        old_capture = json.loads(base.fresh_ref(old["reproduced_capture"]).read_text())
        new_capture = json.loads(base.fresh_ref(new["reproduced_capture"]).read_text())
        if base.content_identity(old_capture) != base.content_identity(new_capture):
            raise ValueError("saved native reproduction content/claim differs")
    for original, snapshot in zip(originals, snapshots, strict=True):
        if base.fresh_ref(original).read_bytes() != base.fresh_ref(snapshot).read_bytes():
            raise ValueError("fusion original/snapshot changed during replay")
    return repeated
