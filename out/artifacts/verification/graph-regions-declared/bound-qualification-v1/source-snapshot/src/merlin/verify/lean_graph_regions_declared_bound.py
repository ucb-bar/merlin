"""Additive original-ABI/complete-after-effect binding over immutable v1.

The 46-case v1 sources and capsules are never edited. Stronger evidence imports
their fixed leaves, validates every post-pass effect field and independently
exports the actual post-pass adapter function buffers.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from merlin.common.paths import artifacts_dir
from merlin.verify import lean_graph_regions as base
from merlin.verify import lean_graph_regions_declared_ast as ast
from merlin.verify import lean_graph_regions_declared_verify as v1
from merlin.verify.lean_graph_regions_declared_check import save
from merlin.verify.lean_graph_regions_declared_context import build, select
from merlin.verify.lean_graph_regions_declared_packet import load, packet, strings, window

ROOTS = v1.ROOTS + ["actual_strengthened_complete_observers"]


def fixed_v1():
    root = artifacts_dir() / "verification/graph-regions-declared/qualification-v1"
    status = load(root, "STATUS.json")
    if status["engineering_qualified"] is not True or status["fixed_cases"] != 46:
        raise ValueError("expected fixed v1 cut unavailable")
    profile = load(root, "PROFILE.json")
    for row in profile["source_roster"].values():
        if base.fresh_ref(row["original"]).read_bytes() != base.fresh_ref(row["snapshot"]).read_bytes():
            raise ValueError("immutable v1 source differs")
    for ref in status["original_receipts"]:
        receipt = json.loads(base.fresh_ref(ref).read_text())
        v1.validate(receipt)
    return base.file_ref(root / "SEAL.json")


def build_bound(destination):
    prior = fixed_v1()
    selected = select(destination / "context")
    lean, environment, identity = build(
        destination / "build", selected, ["Merlin.GraphRegionsDeclaredBridge", "Merlin.GraphRegionsDeclaredEffects"]
    )
    save(destination / "FIXED-V1.json", prior)
    return lean, environment, identity


def boolean(value):
    if type(value) is not bool:
        raise ValueError("strict effect Boolean required")
    return str(value).lower()


def effect_term(effect, backings):
    if set(effect) != {"semantic_key", "accesses", "completion", "token", "waits", "exact_copy", "elementwise"}:
        raise ValueError("complete post-pass effect fields required")
    elementwise = effect["elementwise"]
    if set(elementwise) != {"function_ir", "elements", "stages", "byte_orders", "readout"}:
        raise ValueError("complete post-pass elementwise fields required")
    function = elementwise["function_ir"]
    if set(function) != {"encoding", "hex", "bytes"}:
        raise ValueError("complete post-pass scalar bytes required")
    accesses = []
    for access in effect["accesses"]:
        if set(access) != {"space", "start", "size", "writes"}:
            raise ValueError("complete post-pass access required")
        accesses.append(f"⟨{window(access, backings)},{boolean(access['writes'])}⟩")
    readout = elementwise["readout"]
    return (
        f"⟨{ast.text(effect['semantic_key'])},[{','.join(accesses)}],{ast.text(effect['completion'])},"
        f"{ast.optional(effect['token'], ast.text)},{strings(effect['waits'])},{boolean(effect['exact_copy'])},"
        f"{ast.natural(elementwise['elements'])},{strings(elementwise['byte_orders'])},{strings(elementwise['stages'])},"
        f"{ast.optional(None if readout is None else json.dumps(readout, sort_keys=True), ast.text)},"
        f"{ast.text(function['encoding'])},{ast.natural(function['bytes'])},"
        f"{ast.array(list(bytes.fromhex(function['hex'])), ast.natural)}⟩"
    )


def bindings_term(bindings, backings):
    if set(bindings) != {"inputs", "outputs", "byte_orders", "external_buffers_disjoint"}:
        raise ValueError("complete original ABI bindings required")
    for access in bindings["inputs"] + bindings["outputs"]:
        if set(access) != {"space", "start", "size"}:
            raise ValueError("complete original ABI window required")
    return (
        f"⟨{strings([v['space'] for v in bindings['inputs']])},"
        f"{ast.array(bindings['inputs'], lambda value: window(value, backings))},"
        f"{strings([v['space'] for v in bindings['outputs']])},"
        f"{ast.array(bindings['outputs'], lambda value: window(value, backings))},"
        f"{strings(bindings['byte_orders'])},{boolean(bindings['external_buffers_disjoint'])}⟩"
    )


def strong_query(snapshot, documents, after_documents, actual_bytes):
    text, model = packet(snapshot, documents)
    backings = model["backings"]
    bindings = load(snapshot, "original-provider-bindings.json")
    effects = load(snapshot, "actual-output-effects.json")
    shape = load(snapshot, "CAPTURE.json")["case"]["shape"]
    start = model["source_generic_indices"][0]
    text = "import Merlin.GraphRegionsDeclaredEffects\n" + text
    text += f"\ndef originalBindings : ExternalBindings := {bindings_term(bindings, backings)}\n"
    text += f"""theorem actual_original_bindings : checkOriginalBindings
    ((actualGraph.nodes.drop {start}).take 3) (actualGraph.nodes.drop {start + 3}) actualGraph.observations
    addresses {ast.array(shape, ast.natural)} stage0.program.output originalKernel.arguments
      originalBindings = true := by
  have checked := actual_realization
  simp only [checkRealization, checkFusion, Bool.and_eq_true, decide_eq_true_eq] at checked
  simp only [checkOriginalBindings, decide_eq_true_eq, originalBindingsValid]
  rw [← checked.2.1.2, ← checked.2.2]
  repeat first | apply And.intro | rfl | decide
"""
    for i, effect in enumerate(effects):
        text += f"def completeEffect{i} : CompleteEffect := {effect_term(effect, backings)}\n"
        text += f"def afterAdapterBytes{i} : List UInt8 := {ast.array(list(actual_bytes[i]), ast.natural)}\n"
        text += f"def afterAdapter{i} : Module := {ast.module(after_documents[i])}\n"
        (function,) = after_documents[i]["module"]["ops"]
        (source_function,) = documents[f"stage-{i}-source"]["module"]["ops"]
        source_attrs, attrs = source_function["attrs"], function["attrs"]
        text += f"theorem actual_effect{i} : checkCompleteEffect {model['elements']} bound{i} "
        text += f"completeEffect{i} afterAdapterBytes{i} = true := by\n"
        text += "  apply checkCompleteEffect_iff.mpr\n  constructor\n"
        text += "  · exact ⟨rfl,rfl,rfl,rfl,rfl,rfl,rfl⟩\n"
        text += "  · exact ⟨rfl,rfl,rfl,rfl,rfl,rfl,rfl⟩\n"
        text += f"theorem actual_after_adapter{i} : checkScalar scalar{i}_source afterAdapter{i} "
        text += f"{ast.text(source_attrs['sym_name']['value'])} {ast.text(attrs['sym_name']['value'])} "
        text += f"{ast.optional(source_attrs.get('sym_visibility', {}).get('value'), ast.text)} "
        text += f"{ast.optional(attrs.get('sym_visibility', {}).get('value'), ast.text)} program{i} = true := by\n"
        text += "  apply checkScalar_iff.mpr\n  exact ⟨rfl,rfl,by decide⟩\n"
    text += """
theorem actual_after_common_execution {α : Type} (primitive : Primitive → List α → α) (arguments : Environment α) :
    (ScalarExecution primitive scalar0_source arguments (program0.evaluate primitive arguments) ∧
      ScalarExecution primitive afterAdapter0 arguments (program0.evaluate primitive arguments)) ∧
    (ScalarExecution primitive scalar1_source arguments (program1.evaluate primitive arguments) ∧
      ScalarExecution primitive afterAdapter1 arguments (program1.evaluate primitive arguments)) ∧
    (ScalarExecution primitive scalar2_source arguments (program2.evaluate primitive arguments) ∧
      ScalarExecution primitive afterAdapter2 arguments (program2.evaluate primitive arguments)) :=
  ⟨checkScalar_sound actual_after_adapter0 primitive arguments,
   checkScalar_sound actual_after_adapter1 primitive arguments,
   checkScalar_sound actual_after_adapter2 primitive arguments⟩
"""
    elements = model["elements"]
    text += f"""
theorem actual_strengthened_complete_observers {{α M : Type}} [Inhabited α]
    (primitive : Primitive → List α → α)
    (fallback : ∀ node, TypedAction node (Tensor {elements} α) (fun _ _ => True))
    (read : M → Window → Option (Tensor {elements} α)) (invoke : BoundCall → M → Option M)
    (contracts : ∀ item ∈ items, InstructionContract read invoke primitive item.call item.stage.program)
    (pack : StateEffects.State (Tensor {elements} α) → M)
    (initialized : ∀ entry, Refines read addresses initialKnown entry (pack entry))
    (native : StateEffects.Action
      ⟨liveIns ((actualGraph.nodes.drop {start}).take 3),
       liveOuts ((actualGraph.nodes.drop {start}).take 3)
         (actualGraph.nodes.drop {start + 3}) actualGraph.observations⟩ (Tensor {elements} α))
    (publication : ∀ entry final, runCalls invoke items (pack entry) = some final →
      ∀ slot ∈ liveOuts ((actualGraph.nodes.drop {start}).take 3)
          (actualGraph.nodes.drop {start + 3}) actualGraph.observations,
        read final (addresses slot) = some (native.step entry slot))
    (initial : StateEffects.State (Tensor {elements} α)) :
    checkOriginalBindings ((actualGraph.nodes.drop {start}).take 3) (actualGraph.nodes.drop {start + 3})
      actualGraph.observations addresses {ast.array(shape, ast.natural)} stage0.program.output
      originalKernel.arguments originalBindings = true ∧
    checkCompleteEffect {elements} bound0 completeEffect0 afterAdapterBytes0 = true ∧
    checkCompleteEffect {elements} bound1 completeEffect1 afterAdapterBytes1 = true ∧
    checkCompleteEffect {elements} bound2 completeEffect2 afterAdapterBytes2 = true ∧
    (∀ arguments,
      (ScalarExecution primitive scalar0_source arguments (program0.evaluate primitive arguments) ∧
       ScalarExecution primitive afterAdapter0 arguments (program0.evaluate primitive arguments)) ∧
      (ScalarExecution primitive scalar1_source arguments (program1.evaluate primitive arguments) ∧
       ScalarExecution primitive afterAdapter1 arguments (program1.evaluate primitive arguments)) ∧
      (ScalarExecution primitive scalar2_source arguments (program2.evaluate primitive arguments) ∧
       ScalarExecution primitive afterAdapter2 arguments (program2.evaluate primitive arguments))) ∧
    SourceBinding actualBefore actualRetained sites actualGraph ∧
    checkKernel actualKernel originalKernel.name originalKernel.arguments
      [schema0,schema1,schema2] [bound0,bound1,bound2] = true ∧
    (let meaning := registry primitive fallback (items.map CallStage.stage)
     actualGraph.observations.map (executeNodes meaning (actualGraph.nodes.drop {start + 3})
       (native.step (executeNodes meaning (actualGraph.nodes.take {start}) initial))) =
       actualGraph.observations.map (executeNodes meaning actualGraph.nodes initial)) := by
  have previous := actual_complete_observers primitive fallback read invoke contracts pack initialized
    native publication initial
  exact ⟨actual_original_bindings, actual_effect0, actual_effect1, actual_effect2,
    (fun arguments => actual_after_common_execution primitive arguments), previous.1, previous.2.1,
    previous.2.2.2.2.2⟩
#print axioms actual_strengthened_complete_observers
"""
    return text, model


def verify_bound(capture, destination, *, identity_file, fixed_oracles):
    destination.mkdir(parents=True, exist_ok=False)
    fixed = fixed_v1()
    receipt = v1.verify(capture, destination / "base", identity_file=identity_file, fixed_oracles=fixed_oracles)
    documents = {role: json.loads(base.fresh_ref(ref).read_text()) for role, ref in receipt["exports"].items()}
    snapshot = destination / "base/capture"
    effects = load(snapshot, "actual-output-effects.json")
    if len(effects) != 3:
        raise ValueError("exactly the producer and both consumer effects required")
    after_documents, actual_bytes, after_refs, oracles = [], [], [], []
    for index, effect in enumerate(effects):
        data = bytes.fromhex(effect["elementwise"]["function_ir"]["hex"])
        path = destination / f"after-adapter-{index}.mlir"
        path.write_bytes(data)
        result = v1.oracle(
            "mlir",
            path,
            destination / f"after-adapter-{index}-oracle",
            python=Path(json.loads(base.fresh_ref(receipt["identity"]).read_text())["context"]["oracle_invocation"]),
            fixed=fixed_oracles["mlir-scalar"],
        )
        actual_bytes.append(data)
        after_documents.append(load(destination / f"after-adapter-{index}-oracle", "export.json"))
        after_refs.append(base.file_ref(path))
        if after_documents[-1]["source"] != {key: after_refs[-1][key] for key in ("sha256", "bytes")}:
            raise ValueError("post-pass adapter export does not bind its actual bytes")
        oracles.append(result)
    text, model = strong_query(snapshot, documents, after_documents, actual_bytes)
    proof = destination / "Check.lean"
    proof.write_text(text)
    identity = json.loads(base.fresh_ref(receipt["identity"]).read_text())
    env = os.environ | {"LEAN_PATH": str((identity_file.parent / "compiled").resolve())}
    command = [identity["context"]["fixed_identity"]["lean"]["path"], "-j1", str(proof.resolve())]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
    save(
        destination / "process.json",
        dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr),
    )
    result.check_returncode()
    save(destination / "AUDIT.json", v1.audit(result.stdout, ROOTS))
    v1.validate(receipt)
    v1.fresh_identity(identity)
    strong = dict(
        schema="merlin.graph-regions.declared-boundary-strong.v1",
        base=base.file_ref(destination / "base/EVIDENCE.json"),
        fixed_v1=fixed,
        generator=base.file_ref(Path(__file__)),
        after_buffers=after_refs,
        after_oracles=oracles,
        after_exports=[base.file_ref(destination / f"after-adapter-{i}-oracle/export.json") for i in range(3)],
        proof=base.file_ref(proof),
        process=base.file_ref(destination / "process.json"),
        audit=base.file_ref(destination / "AUDIT.json"),
        independently_reviewed=False,
        native_fidelity_proved=False,
    )
    save(destination / "EVIDENCE.json", strong)
    return strong


def validate_bound(receipt):
    if receipt["schema"] != "merlin.graph-regions.declared-boundary-strong.v1":
        raise ValueError("wrong stronger receipt schema")
    if receipt["fixed_v1"] != fixed_v1() or receipt["generator"] != base.file_ref(Path(__file__)):
        raise ValueError("strong checker/fixed-v1 source association differs")
    original = json.loads(base.fresh_ref(receipt["base"]).read_text())
    v1.validate(original)
    documents = {role: json.loads(base.fresh_ref(ref).read_text()) for role, ref in original["exports"].items()}
    snapshot = Path(original["snapshots"]["CAPTURE.json"]["path"]).parent
    effects = load(snapshot, "actual-output-effects.json")
    if any(len(receipt[key]) != 3 for key in ("after_buffers", "after_exports", "after_oracles")) or len(effects) != 3:
        raise ValueError("exactly three post-pass scalar buffers/exports/oracles required")
    after_documents, actual_bytes = [], []
    for index, (buffer, exported, oracle) in enumerate(
        zip(receipt["after_buffers"], receipt["after_exports"], receipt["after_oracles"], strict=True)
    ):
        data = base.fresh_ref(buffer).read_bytes()
        if data != bytes.fromhex(effects[index]["elementwise"]["function_ir"]["hex"]):
            raise ValueError("actual after-effect/scalar bytes association differs")
        document = json.loads(base.fresh_ref(exported).read_text())
        if document["source"] != {key: buffer[key] for key in ("sha256", "bytes")}:
            raise ValueError("actual post-pass scalar/export association differs")
        if (
            oracle["source"] != buffer
            or oracle["valid"] is not True
            or oracle["host"] != original["fixed_oracles"]["mlir-scalar"]
        ):
            raise ValueError("after-adapter source/oracle association differs")
        for reference in v1.host_refs(oracle["host"]):
            base.fresh_ref(reference)
        after_documents.append(document)
        actual_bytes.append(data)
    text, _ = strong_query(snapshot, documents, after_documents, actual_bytes)
    if base.fresh_ref(receipt["proof"]).read_text() != text:
        raise ValueError("strong proof does not bind actual source/call/effect bytes")
    process = json.loads(base.fresh_ref(receipt["process"]).read_text())
    checked = json.loads(base.fresh_ref(receipt["audit"]).read_text())
    identity = json.loads(base.fresh_ref(original["identity"]).read_text())
    command = [identity["context"]["fixed_identity"]["lean"]["path"], "-j1", receipt["proof"]["path"]]
    if process["command"] != command or process["returncode"] != 0 or v1.audit(process["stdout"], ROOTS) != checked:
        raise ValueError("strong Lean proof/process/exact-root association differs")
    return True


def replay_bound(receipt, destination):
    validate_bound(receipt)
    original = json.loads(base.fresh_ref(receipt["base"]).read_text())
    capture = Path(original["originals"]["CAPTURE.json"]["path"]).parent
    current = verify_bound(
        capture, destination, identity_file=Path(original["identity"]["path"]), fixed_oracles=original["fixed_oracles"]
    )
    validate_bound(current)
    for field in ("proof", "audit"):
        if base.fresh_ref(receipt[field]).read_bytes() != base.fresh_ref(current[field]).read_bytes():
            raise ValueError("strong fresh replay differs")
    save(
        destination / "REPLAY.json",
        dict(
            identical_checked_content=True,
            native_execution_run=False,
            original=receipt["proof"],
            current=current["proof"],
        ),
    )
    return current
