"""Kernel-check actual buffer reuse in an explicit private total-allocator model.

Fresh full parsing/SSA/initializer/context projection is trusted Python. No raw
SCF/frozen-evaluator/native/allocator-failure or whole-pipeline proof is implied.
The independently held selected contract digest prevents a coherently rehashed
replacement proof/tool/model selection from becoming this SAME qualified policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .lean_graph_loops import LeanSelection, _reference
from .lean_graph_loops import _audit as _single_line_audit
from .lean_graph_loops_reuse_projection import project_exports
from .lean_graph_loops_unroll import _list, _nat, _string

ALLOCATOR_MODE = "total_private_unobservable"
REUSE_MODULES = (
    "Merlin.StateEffects",
    "Merlin.ResourceGeometry",
    "Merlin.GraphLoopsReuse",
    "Merlin.GraphLoopsReuseProtocol",
    "Merlin.GraphLoopsReuseSemantics",
    "Merlin.GraphLoopsReuseGeometry",
    "Merlin.GraphLoopsReuseLifetime",
    "Merlin.GraphLoopsReuseModel",
    "Merlin.GraphLoopsReuseCountermodels",
)
REUSE_ROOTS = (
    "GraphReuseObligation.valid",
    "GraphReuseObligation.accepted",
    "GraphReuseObligation.complete",
    "GraphReuseObligation.contextual",
    "Merlin.GraphLoopsReuse.Witness.protocol",
    "Merlin.GraphLoopsReuse.Witness.source_inhabited",
    "Merlin.GraphLoopsReuse.Witness.target_inhabited",
    "Merlin.GraphLoopsReuse.Related.read",
    "Merlin.GraphLoopsReuse.Related.allocate",
    "Merlin.GraphLoopsReuse.Related.initialize",
    "Merlin.GraphLoopsReuse.Related.store",
    "Merlin.GraphLoopsReuse.Related.free",
    "Merlin.GraphLoopsReuse.step_forward",
    "Merlin.GraphLoopsReuse.step_backward",
    "Merlin.GraphLoopsReuse.execution_forward",
    "Merlin.GraphLoopsReuse.execution_backward",
    "Merlin.GraphLoopsReuse.checked_observations",
    "Merlin.GraphLoopsReuse.Geometry.every_cell_covered",
    "Merlin.GraphLoopsReuse.Geometry.rectangular_initializes",
    "Merlin.GraphLoopsReuse.LifetimesValid.not_simultaneously_live",
    "Merlin.GraphLoopsReuse.Model.check_sound",
    "Merlin.GraphLoopsReuse.Model.checked_initializer_coverage",
    "Merlin.GraphLoopsReuse.Model.checked_complete_observations",
    "Merlin.GraphLoopsReuse.Model.checked_context",
    "Merlin.GraphLoopsReuse.Countermodels.resource_exhaustion",
    "Merlin.GraphLoopsReuse.Countermodels.pointer_identity",
    "Merlin.GraphLoopsReuse.Countermodels.allocation_call_budget",
    "Merlin.GraphLoopsReuse.Countermodels.stale_cell",
    "Merlin.GraphLoopsReuse.Countermodels.overlapping_live_values",
)


def _audit(stdout: str, roots: tuple[str, ...]) -> dict:
    """Normalize ONLY pretty-printer continuation whitespace, then use the
    immutable strict root/standard3 reader. Never discard names or diagnostics.
    """
    lines = iter(stdout.splitlines())
    normalized = []
    for line in lines:
        if line.startswith("'") and " depends on axioms: [" in line and not line.rstrip().endswith("]"):
            block = [line.rstrip()]
            for continuation in lines:
                if continuation.startswith("'") or "error:" in continuation:
                    raise ValueError("unterminated or interrupted transitive axiom report")
                block.append(continuation.strip())
                if continuation.rstrip().endswith("]"):
                    break
            else:
                raise ValueError("unterminated transitive axiom report")
            line = " ".join(block)
        normalized.append(line)
    return _single_line_audit("\n".join(normalized), roots)


@dataclass(frozen=True)
class ReuseSelection:
    proof: LeanSelection
    source: Path
    revision: str
    producer_manifest: Path
    producer_manifest_sha256: str
    contract: Path
    contract_sha256: str

    def dependencies(self):
        contract_bytes = self.contract.read_bytes()
        if hashlib.sha256(contract_bytes).hexdigest() != self.contract_sha256:
            raise ValueError("originally selected independent reuse contract changed")
        contract = json.loads(contract_bytes)
        proof = {key: str(value) if isinstance(value, Path) else value for key, value in vars(self.proof).items()}
        expected = dict(
            proof=proof,
            source=str(self.source.resolve()),
            revision=self.revision,
            producer_manifest=str(self.producer_manifest.resolve()),
            producer_manifest_sha256=self.producer_manifest_sha256,
            allocator_mode=ALLOCATOR_MODE,
        )
        if contract["selected_context"] != expected:
            raise ValueError("replacement proof/tool/import/source selection differs from the frozen reuse contract")
        data = self.producer_manifest.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.producer_manifest_sha256:
            raise ValueError("originally selected producer/parser manifest changed")
        selected = json.loads(data)
        revision = subprocess.run(
            ["git", "-C", str(self.source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        if selected["producer_revision"] != self.revision or revision != self.revision:
            raise ValueError("originally selected producer revision changed")
        rows = selected["producer_source"] + selected["xdsl_loaded_source"]
        if not selected["producer_source"] or not selected["xdsl_loaded_source"]:
            raise ValueError("actual loaded producer/parser closure missing")
        for row in rows:
            if _reference(Path(row["path"])) != {key: row[key] for key in ("path", "sha256", "bytes")}:
                raise ValueError("originally selected producer/parser code changed")
        build = json.loads(self.proof.build_record.read_bytes())
        if {row["module"] for row in build["attempts"]} != set(REUSE_MODULES):
            raise ValueError("complete reuse memory/protocol/geometry/model/countermodel closure required")
        return dict(
            proof=self.proof.dependencies(),
            contract=_reference(self.contract, contract_bytes),
            producer_manifest=_reference(self.producer_manifest, data),
            producer_source=selected["producer_source"],
            xdsl_loaded_source=selected["xdsl_loaded_source"],
            revision=self.revision,
            allocator_policy_is_admitted_premise=True,
        )


def obligation(projection):
    if (
        projection["allocator_mode"] != ALLOCATOR_MODE
        or projection["initializer_semantics"] != "partial_complete_rectangular_pure"
    ):
        raise ValueError("explicit supported allocator/initializer semantics required")

    def object_literal(row):
        if set(row) != {"slot", "shape", "elementBits", "alignment"}:
            raise ValueError("complete actual allocation geometry required")
        return (
            "⟨"
            + ", ".join(
                (_nat(row["slot"]), _list(row["shape"], _nat), _nat(row["elementBits"]), _nat(row["alignment"]))
            )
            + "⟩"
        )

    def instruction(row):
        kind = row["kind"]
        if (
            kind in ("allocate", "release")
            and set(row) == {"kind", "object", "retained"}
            and type(row["retained"]) is bool
        ):
            return f"(.{kind} {_nat(row['object'])} {str(row['retained']).lower()})"
        if kind in ("initialize", "load", "store", "dimension") and set(row) == {"kind", "object", "key"}:
            return f"(.{kind} {_nat(row['object'])} {_string(row['key'])})"
        if kind == "retained" and set(row) == {"kind", "key"}:
            return f"(.retained {_string(row['key'])})"
        raise ValueError("unknown/unmodeled logical instruction")

    def initializer(pair):
        index, epoch = pair
        row = epoch["initializer"]
        controls = _list(row["controls"], lambda triple: "(" + ", ".join(_nat(value) for value in triple) + ")")
        return (
            "⟨"
            + ", ".join(
                (
                    _nat(index),
                    controls,
                    _list(row["loop_ivs"], _nat),
                    _list(row["indices"], _nat),
                    _string(row["body"]),
                    _string(row["target_body"]),
                )
            )
            + "⟩"
        )

    def lifetime(pair):
        index, row = pair
        return (
            "⟨"
            + ", ".join(
                (
                    _nat(index),
                    _nat(row["allocation"]),
                    _nat(row["initialization"]),
                    _nat(row["release"]),
                    str(row["retainedAllocation"]).lower(),
                    str(row["retainedRelease"]).lower(),
                )
            )
            + "⟩"
        )

    fields = [
        "bits := " + _nat(projection["bits"]),
        "objects := " + _list(projection["objects"], object_literal),
        "instructions := " + _list(projection["instructions"], instruction),
        "initializers := " + _list(list(enumerate(projection["epochs"])), initializer),
        "lifetimes := " + _list(list(enumerate(projection["epochs"])), lifetime),
        "sourceContext := " + _string(projection["sourceContext"]),
        "targetContext := " + _string(projection["targetContext"]),
        "allocatorMode := " + _string(projection["allocator_mode"]),
        "initializerMode := " + _string(projection["initializer_semantics"]),
    ]
    literal = "{\n  " + "\n  ".join(fields) + "\n}"
    roots = "\n".join("#print axioms " + name for name in REUSE_ROOTS)
    return f"""import Merlin.GraphLoopsReuseCountermodels
set_option autoImplicit false
set_option maxHeartbeats 1000000
set_option maxRecDepth 8192
open Merlin GraphLoopsReuse
namespace GraphReuseObligation
def actual : Model.Projection := {literal}
theorem valid : Model.Valid actual := by
  simp [Model.Valid, Model.ValidObject, Model.CompatibleObjects, Model.Initializer.Valid,
    Model.InitializerBound, actual, Geometry.Positive, Geometry.volume, LifetimesValid,
    Lifetime.Valid, isInitializer, objectAt, slotOf]
  all_goals decide
theorem accepted : Model.check actual = true := by
  unfold Model.check
  apply Bool.and_eq_true_iff.mpr
  exact ⟨by simp only [valid, decide_true], by cbv⟩
theorem complete {{Value Public : Type}} (primitives : Primitives Value Public)
    (source target : Memory Value Public) (same : source.observed = target.observed) (outcome : Public) :
    Model.outcomes actual primitives false source outcome ↔ Model.outcomes actual primitives true target outcome :=
  Model.checked_complete_observations accepted primitives source target same outcome
theorem contextual {{Value Public Outcome : Type}} (primitives : Primitives Value Public)
    (source target : Memory Value Public) (same : source.observed = target.observed)
    (context : String → (Public → Prop) → Outcome) :
    context actual.sourceContext (Model.outcomes actual primitives false source) =
      context actual.targetContext (Model.outcomes actual primitives true target) :=
  Model.checked_context accepted primitives source target same context
end GraphReuseObligation
{roots}
"""


def verify_reuse_files(
    before_path: Path,
    after_path: Path,
    *,
    selection: ReuseSelection,
    output: Path,
    index_bits: int,
    max_nodes: int,
    allocator_model: str,
):
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _verify(
            before_path,
            after_path,
            selection=selection,
            output=output,
            index_bits=index_bits,
            max_nodes=max_nodes,
            allocator_model=allocator_model,
        )
    except Exception as error:
        (output / "REFUSAL.json").write_text(
            json.dumps(
                dict(
                    error=str(error),
                    type=type(error).__name__,
                    lean_memory_relation_checked=False,
                    python_projection_trusted=True,
                    pipeline_receipt_issued=False,
                ),
                indent=2,
            )
            + "\n"
        )
        raise


def _verify(before_path, after_path, *, selection, output, index_bits, max_nodes, allocator_model):
    if allocator_model != ALLOCATOR_MODE:
        raise ValueError(
            "explicit total/private/unobservable allocator model is required; failures and identity are unproved"
        )
    first, second = before_path.read_bytes(), after_path.read_bytes()
    originals = [_reference(before_path, first), _reference(after_path, second)]
    dependencies = selection.dependencies()
    from . import (
        lean_graph_loops,
        lean_graph_loops_capture,
        lean_graph_loops_reuse_capture,
        lean_graph_loops_reuse_projection,
        lean_graph_loops_segment_capture,
        lean_graph_loops_segment_projection,
        lean_graph_loops_unroll,
    )

    owner_paths = [
        Path(module.__file__).resolve()
        for module in (
            lean_graph_loops,
            lean_graph_loops_capture,
            lean_graph_loops_segment_capture,
            lean_graph_loops_segment_projection,
            lean_graph_loops_unroll,
            lean_graph_loops_reuse_capture,
            lean_graph_loops_reuse_projection,
        )
    ] + [
        Path(__file__).resolve(),
        Path(__file__).with_name("lean_graph_input_admission.py").resolve(),
        Path(sys.executable).resolve(),
    ]
    owners = [_reference(path) for path in owner_paths]
    before, after = output / "before.mlir", output / "after.mlir"
    before.write_bytes(first)
    after.write_bytes(second)
    fresh = output / "actual-exports.json"
    env = os.environ.copy()
    env.update(PYTHONPATH=str(selection.source / "src"), PYTHONDONTWRITEBYTECODE="1")
    command = [
        sys.executable,
        str(Path(lean_graph_loops_reuse_capture.__file__).resolve()),
        "--source",
        str(selection.source.resolve()),
        "--revision",
        selection.revision,
        "--before",
        str(before.resolve()),
        "--after",
        str(after.resolve()),
        "--output",
        str(fresh.resolve()),
    ]
    parsed = subprocess.run(command, capture_output=True, text=True, check=False, env=env, timeout=120)
    (output / "projection.stdout").write_text(parsed.stdout)
    (output / "projection.stderr").write_text(parsed.stderr)
    if parsed.returncode:
        raise ValueError("actual parsing/SSA export refused; malformed IR is separate from semantic mutation rejection")
    exported = json.loads(fresh.read_bytes())
    export_ref = _reference(fresh)
    if exported["before"] != _reference(before, first) or exported["after"] != _reference(after, second):
        raise ValueError("fresh actual trees are not bound to the same input buffers")
    for field in ("producer_source", "xdsl_loaded_source"):
        known = {row["path"]: row for row in dependencies[field]}
        if not exported[field] or any(known.get(row["path"]) != row for row in exported[field]):
            raise ValueError("fresh actual loaded parser/producer differs from selected source")
    projection = project_exports(
        exported["before_export"], exported["after_export"], index_bits=index_bits, max_nodes=max_nodes
    )
    model = output / "model.json"
    model.write_text(json.dumps(projection, sort_keys=True, indent=2) + "\n")
    model_ref = _reference(model)
    query = output / "Obligation.lean"
    query.write_text(obligation(projection))
    query_ref = _reference(query)
    proof = selection.proof
    env.update(LEAN_PATH=str(proof.library), LEAN_SYSROOT=str(proof.executable.parent.parent), LEAN_NUM_THREADS="2")
    lean_command = [str(proof.executable), "-DautoImplicit=false", "-DrelaxedAutoImplicit=false", str(query.resolve())]
    checked = subprocess.run(lean_command, capture_output=True, text=True, check=False, env=env, timeout=120)
    (output / "stdout.log").write_text(checked.stdout)
    (output / "stderr.log").write_text(checked.stderr)
    record = dict(
        schema="merlin.graph.actual_buffer_reuse_memory_check.v1",
        originals=originals,
        dependencies=dependencies,
        trusted_decoder_sources=owners,
        fresh_exports=export_ref,
        model=model_ref,
        query=query_ref,
        projection=projection,
        command=lean_command,
        projection_command=command,
        returncode=checked.returncode,
        lean_memory_relation_checked=False,
        python_projection_trusted=True,
        allocator_mode=allocator_model,
        allocator_policy_is_admitted_premise=True,
        initializer_semantics_explicit=True,
        source_success_inferred=False,
        untrusted_witness_or_plan_used=False,
        raw_SCF_bridge_proved=False,
        allocator_failure_or_pointer_identity_proved=False,
        LeanMLIR_selected=False,
        pipeline_receipt_issued=False,
        numerical_gates_discharged=False,
        renderer_or_hardware_proved=False,
    )
    if (
        dependencies != selection.dependencies()
        or owners != [_reference(path) for path in owner_paths]
        or originals != [_reference(before_path), _reference(after_path)]
        or export_ref != _reference(fresh)
        or model_ref != _reference(model)
        or query_ref != _reference(query)
        or before.read_bytes() != first
        or after.read_bytes() != second
    ):
        record["refusal"] = "actual inputs/tree/model/query or selected source/tool/proof changed during checking"
    elif checked.returncode:
        record["refusal"] = "Lean refused actual live-epoch/initializer/slot/lifetime/observation obligations"
    else:
        record["transitive_axioms"] = _audit(checked.stdout, REUSE_ROOTS)
        record["lean_memory_relation_checked"] = True
    (output / "CHECK.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    return record
