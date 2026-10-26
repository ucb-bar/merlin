"""Offline source-only binding of retained successful v1 i8 literals.

No compiler/exporter/provider/oracle invocation, SDK/runtime access or old
qualification retry. The original literal proof is copied intact into a new
namespace and extended by the new actual-declaration/raw-execution bridge.
Historical compiler validity and parser/serializer trust remain historical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BASE = Path("out/artifacts/verification/graph-regions-declared")
POSITIVE = (26321, "a4abc04170b9ecdc1041dd7d427b4ce517258fbb7d855db82f99a48b63c2835f")
NEGATIVE = (1430, "560b2ca05a3f05d197e7385791a895bc63c29ae0cd7b34cb463b57a39c7f2086")
EVIDENCE = (1165077, "4ee6584c9b835ecbb9e11319ead14524fc4bdfbe971ad3f8208596934c7b50fa")
NEGATIVE_EVIDENCE = (8195, "69c0039e95b43cade29a5a9dcdb4f2638efa79335c724fa8eca08297a58dda08")
NAMESPACE = "Merlin.GraphRegionsDeclared.Capture.RetainedI8"


def identity(data):
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def read_bound(root, pin):
    if set(pin) != {"path", "bytes", "sha256"} or type(pin["bytes"]) is not int:
        raise ValueError("strict source reference required")
    path = Path(pin["path"]).resolve(strict=True)
    if not path.is_relative_to(root / BASE):
        raise ValueError("only owned retained artifact data may be read")
    data = path.read_bytes()
    if identity(data) != {key: pin[key] for key in ("bytes", "sha256")}:
        raise ValueError("retained original content differs: " + str(path))
    return data


def copy_data(root, pin, target):
    data = read_bound(root, pin)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(data)
    if target.read_bytes() != data:
        raise ValueError("original-to-snapshot actual bytes differ")
    return dict(original=pin, snapshot=dict(path=str(target), **identity(data)))


def original_evidence(root, relative, expected):
    """Expectations come from unchanged qualification-v1/EXTERNAL-RETENTION.

    A coherently rehashed replacement receipt cannot select new originals while
    keeping the old pinned Lean literal. No SDK/runtime expectations are read or
    refreshed by this source-data check.
    """
    pin = dict(path=str(root / BASE / relative), bytes=expected[0], sha256=expected[1])
    return pin, json.loads(read_bound(root, pin))


EXTENSION = r'''
open Merlin.GraphRegionsDeclared.Capture

def actualDeclarations (item : CallStage) : DeclarationPair :=
  if item.call.instruction = "produce" then
    ⟨scalar0_source, scalar0_selected, "source", "forward", none, none⟩
  else if item.call.instruction = "add" then
    ⟨scalar1_source, scalar1_selected, "source", "forward", none, none⟩
  else ⟨scalar2_source, scalar2_selected, "source", "forward", none, none⟩

theorem actual_declarations_checked : checkDeclarations items actualDeclarations = true := by
  apply List.all_eq_true.mpr
  intro item member
  simp only [items, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · exact scalar0_selected_accepted
  · exact scalar1_selected_accepted
  · exact scalar2_selected_accepted

theorem actual_kernel_execution {M : Type} (invoke : BoundCall → M → Option M) (memory : M) :
    runActualKernel [schema0, schema1, schema2] actualKernel invoke memory = runCalls invoke items memory :=
  checked_kernel_execution actual_kernel_decoded invoke memory

/-- This extends the retained ACTUAL source/plan/call example. The unknown
surrounding operations keep explicit total/pure Actions. The actual declaration
and raw-call facts are proved; native fidelity, completion, frame, initialized
packing and publication are still premises. No captured-parent grammar or
frozen operational semantics is silently substituted for this broader source. -/
theorem actual_declared_complete_observers {α M : Type} [Inhabited α]
    (primitive : Primitive → List α → α)
    (fallback : ∀ node, TypedAction node (Tensor 141 α) (fun _ _ => True))
    (read : M → Window → Option (Tensor 141 α)) (invoke : BoundCall → M → Option M)
    (contracts : ∀ item ∈ items,
      DeclaredInstructionContract read invoke primitive item.call (actualDeclarations item).declaration)
    (pack : StateEffects.State (Tensor 141 α) → M)
    (initialized : ∀ entry, Refines read addresses initialKnown entry (pack entry))
    (native : StateEffects.Action
      ⟨liveIns ((actualGraph.nodes.drop 4).take 3),
        liveOuts ((actualGraph.nodes.drop 4).take 3) (actualGraph.nodes.drop 7) actualGraph.observations⟩
      (Tensor 141 α))
    (publication : ∀ entry final,
      runActualKernel [schema0, schema1, schema2] actualKernel invoke (pack entry) = some final →
      ∀ slot ∈ liveOuts ((actualGraph.nodes.drop 4).take 3) (actualGraph.nodes.drop 7) actualGraph.observations,
        read final (addresses slot) = some (native.step entry slot))
    (initial : StateEffects.State (Tensor 141 α)) :
    SourceBinding actualBefore actualRetained sites actualGraph ∧
    (∀ entry, ∃ final,
      runActualKernel [schema0, schema1, schema2] actualKernel invoke (pack entry) = some final ∧
      Refines read addresses (finalKnown initialKnown items)
        (executeStages primitive (items.map CallStage.stage) entry) final) ∧
    (let meaning := registry primitive fallback (items.map CallStage.stage)
     actualGraph.observations.map (executeNodes meaning (actualGraph.nodes.drop 7)
       (native.step (executeNodes meaning (actualGraph.nodes.take 4) initial))) =
       actualGraph.observations.map (executeNodes meaning actualGraph.nodes initial)) := by
  refine ⟨actual_generic_binding, ?_, ?_⟩
  · intro entry
    exact checked_actual_kernel_refines actual_kernel_decoded actual_declarations_checked primitive
      addresses initialKnown actual_calls_valid contracts (initialized entry)
  · exact checked_calls_preserve_observers primitive fallback actual_realization items actual_boundary
      addresses initialKnown (checkSequence_sound actual_calls_valid) actual_nodes_unique read invoke
      (checked_declaration_contracts actual_declarations_checked contracts) pack initialized
      actual_output_coverage native
      (by
        intro entry final ran
        exact publication entry final ((actual_kernel_execution invoke (pack entry)).trans ran))
      initial

theorem missing_second_consumer_refused :
    checkKernel { actualKernel with calls := actualKernel.calls.take 2 }
      originalKernel.name originalKernel.arguments [schema0, schema1, schema2]
      [bound0, bound1, bound2] = false := by decide

theorem original_virtual_kernel_not_physical :
    decodeCalls [schema0, schema1, schema2] originalKernel.calls = none := by decide
'''


def generate(root, destination, receipt):
    root = root.resolve(strict=True)
    summary_path = root / BASE / "paper-evidence/QUALIFIED-EXAMPLE-BINDINGS.json"
    summary = json.loads(summary_path.read_bytes())
    positive = summary["references_from_original_retained_receipt"]["proof"]
    negative = summary["negative"]["proof"]
    for pin, expected in ((positive, POSITIVE), (negative, NEGATIVE)):
        if (pin["bytes"], pin["sha256"]) != expected:
            raise ValueError("retained literal source pin differs from the unchanged successful v1 cut")
    before = read_bound(root, positive).decode()
    refusal = read_bound(root, negative).decode()
    marker = "open Merlin MerlinVerify Merlin.GraphRegions Merlin.GraphRegionsDeclared\n"
    if before.count(marker) != 1:
        raise ValueError("exact pinned proof namespace insertion unavailable")
    prefix = "import Merlin.GraphRegionsDeclaredEvaluation\n"
    if not refusal.startswith(prefix):
        raise ValueError("exact pinned negative proof import unavailable")
    text = (
        "/- OFFLINE retained actual i8 literals; new pure DEVELOPMENT, not new capture/native qualification.\n"
        "Generated by lean_graph_regions_declared_pure_retained.py; original proof retained unchanged. -/\n"
        "import Merlin.GraphRegionsDeclaredCaptureKernel\n"
        + before.replace(marker, "namespace " + NAMESPACE + "\n" + marker, 1)
        + EXTENSION
        + "\nnamespace Negative\n"
        + refusal.removeprefix(prefix)
        + "\nend Negative\nend " + NAMESPACE + "\n"
    )
    receipt.mkdir(parents=True, exist_ok=False)
    inputs = {
        "positive-proof": copy_data(root, positive, receipt / "inputs/positive.lean"),
        "negative-proof": copy_data(root, negative, receipt / "inputs/negative.lean"),
    }
    evidence_pin, evidence = original_evidence(
        root, "qualification-v1-originals/verified-i8/EVIDENCE.json", EVIDENCE
    )
    evidence_path = Path(evidence_pin["path"])
    if evidence_path != root / summary["original_receipt"]:
        raise ValueError("summary cannot redirect the pinned original receipt")
    evidence_bytes = read_bound(root, evidence_pin)
    if evidence["proof"] != positive:
        raise ValueError("original receipt proof differs from the pinned literal source")
    (receipt / "inputs/original-evidence.json").write_bytes(evidence_bytes)
    for name, pin in evidence["originals"].items():
        if name.endswith(".mlir") or name in ("actual-output-kernel.json", "original-provider-kernel.json"):
            if name not in evidence["snapshots"]:
                raise ValueError("missing retained source snapshot")
            original = read_bound(root, pin)
            if original != read_bound(root, evidence["snapshots"][name]):
                raise ValueError("historical original/snapshot actual bytes differ: " + name)
            inputs["original/" + name] = copy_data(root, pin, receipt / "inputs/original" / name)
    for role, pin in evidence["exports"].items():
        exported = json.loads(read_bound(root, pin))
        source_name = {"before": "source.mlir", "retained": "retained-source.mlir"}.get(role, role + ".mlir")
        source = evidence["originals"][source_name]
        if exported["source"] != {key: source[key] for key in ("bytes", "sha256")}:
            raise ValueError("historical exported AST/source association differs: " + role)
        inputs["export/" + role] = copy_data(root, pin, receipt / "inputs/export" / (role + ".json"))
    negative_pin, negative_evidence = original_evidence(
        root, "qualification-v1-originals/scalar-wrong_operand/EVIDENCE.json", NEGATIVE_EVIDENCE
    )
    if negative_evidence["proof"] != negative:
        raise ValueError("negative receipt proof differs from the pinned negative literal")
    inputs["negative-evidence"] = copy_data(root, negative_pin, receipt / "inputs/negative-evidence.json")
    for role, pin in negative_evidence["originals"].items():
        if read_bound(root, pin) != read_bound(root, negative_evidence["snapshots"][role]):
            raise ValueError("negative original/snapshot actual bytes differ")
        inputs["negative/" + role] = copy_data(root, pin, receipt / "inputs/negative" / (role + ".mlir"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with destination.open("x") as stream:
            stream.write(text)
    if destination.read_bytes() != text.encode():
        raise ValueError("generated literal source transport differs")
    (receipt / "Generated.lean").write_bytes(text.encode())
    generator_bytes = Path(__file__).read_bytes()
    (receipt / "generator.py").write_bytes(generator_bytes)
    record = dict(
        schema="merlin.graph-regions.offline-retained-literal-source.v2",
        inputs=inputs,
        summary=dict(path=str(summary_path), **identity(summary_path.read_bytes())),
        original_evidence=evidence_pin,
        generator=dict(path=str(Path(__file__).resolve()), **identity(generator_bytes)),
        generator_snapshot=dict(path=str(receipt / "generator.py"), **identity(generator_bytes)),
        generated=dict(path=str(destination), **identity(text.encode())),
        generated_snapshot=dict(path=str(receipt / "Generated.lean"), **identity(text.encode())),
        sdk_runtime_reads=0,
        compiler_provider_oracle_invocations=0,
        old_profile_promoted=False,
        new_capture_native_qualification=False,
        trust=["historical authoritative export and serializer", "host source association/transport"],
    )
    (receipt / "BINDING.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    arguments = parser.parse_args()
    result = generate(arguments.root, arguments.destination, arguments.receipt)
    print(json.dumps({"generated": result["generated"], "inputs": len(result["inputs"])}))
