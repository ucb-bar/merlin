"""Separate pending-capacity mode over the immutable delivered loop/storage mode.

Every old source/tool/object identity is reopened. The new check binds the whole
actual event stream and the complete selected admission declaration; target truth,
Python projection, numerical validity and hardware/runtime conformance stay explicit.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from merlin.sched.ir import parse_kernel

from .lean_graph_loops import LeanSelection, _audit, _reference, project_pair
from .lean_graph_loops_storage import STORAGE_ROOTS, ProducerSelection, storage_obligation, verify_storage_files


def admission_digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class LegacySeal:
    manifest: Path
    sha256: str

    def dependencies(self) -> list[dict]:
        data = self.manifest.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.sha256:
            raise ValueError("originally selected v1 source/runtime seal changed")
        record = json.loads(data)
        entries = [
            record["patch"],
            *record["sources"],
            *record["proof_dependencies"],
            *record["primary_python_ir_dependencies"],
            *record["reports"],
        ]
        result = [_reference(self.manifest, data)]
        for row in entries:
            expected = {key: row[key] for key in ("path", "sha256", "bytes")}
            if _reference(Path(row["path"])) != expected:
                raise ValueError("delivered v1 source/runtime/report changed; no replacement allowed")
            result.append(expected)
        return result


@dataclass(frozen=True)
class CapacitySelection:
    legacy: LegacySeal
    base_proof: LeanSelection
    proof: LeanSelection
    producer: ProducerSelection
    admission_sha256: str

    def dependencies(self) -> dict:
        if (
            type(self.admission_sha256) is not str
            or len(self.admission_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.admission_sha256)
        ):
            raise ValueError("select the original complete admission declaration digest")
        modules = {row["module"] for row in json.loads(self.proof.build_record.read_bytes())["attempts"]}
        expected = {
            "Merlin.StateEffects",
            "Merlin.ResourceGeometry",
            "Merlin.GraphLoops",
            "Merlin.GraphLoopsStorage",
            "Merlin.GraphLoopsBound",
            "Merlin.GraphLoopsBounds",
            "Merlin.GraphLoopsEpochs",
            "Merlin.GraphLoopsCapacity",
        }
        if modules != expected:
            raise ValueError("select the complete additive capacity proof closure")
        return {
            "legacy": self.legacy.dependencies(),
            "base_proof": self.base_proof.dependencies(),
            "capacity_proof": self.proof.dependencies(),
            "producer": self.producer.dependencies(),
            "admission_sha256": self.admission_sha256,
        }


CAPACITY_ROOTS = (
    "GraphLoopCapacityObligation.accepted",
    "GraphLoopCapacityObligation.bounded",
    "Merlin.GraphLoops.Capacity.check_sound",
    "Merlin.GraphLoops.Capacity.execute_sound",
    "Merlin.GraphLoops.Capacity.Trace.erase",
    "Merlin.GraphLoops.Capacity.Trace.state_bounds",
    "Merlin.GraphLoops.Capacity.Trace.event",
)


def capacity_obligation(projection: dict, effects: dict, *, initialized_external: tuple[str, ...]) -> str:
    limit = effects["admission"]["max_pending"]
    if type(limit) is not int or limit <= 0:
        raise ValueError("selected pending capacity must be an actual positive integer")
    base = storage_obligation(projection, effects, initialized_external=initialized_external)
    roots = "\n".join("#print axioms " + root for root in CAPACITY_ROOTS)
    return (
        "import Merlin.GraphLoopsCapacity\n"
        + base
        + f"""
namespace GraphLoopCapacityObligation
open GraphLoopStorageObligation
def limit : Nat := {limit}
theorem accepted : Capacity.check source actual allocations external initialized events limit = true := by cbv
theorem bounded : ∃ final, Capacity.Trace limit (external ++ allocations.map Allocation.region)
    (allocations.map Allocation.region) ⟨initialized, [], []⟩ (events.flatten.map Event.effect) final ∧
    final.pending = [] := (Capacity.check_sound accepted).2.2
end GraphLoopCapacityObligation
{roots}
"""
    )


def verify_capacity_files(
    before_path: Path,
    after_path: Path,
    *,
    selection: CapacitySelection,
    output: Path,
    index_bits: int,
    max_dynamic_calls: int,
    effects_path: Path | None = None,
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _verify(
            before_path,
            after_path,
            selection=selection,
            output=output,
            index_bits=index_bits,
            max_dynamic_calls=max_dynamic_calls,
            effects_path=effects_path,
        )
    except Exception as error:
        (output / "REFUSAL.json").write_text(
            json.dumps(
                {
                    "error": str(error),
                    "type": type(error).__name__,
                    "lean_capacity_checked": False,
                    "complete_boundary_checked": False,
                    "pipeline_receipt_issued": False,
                },
                indent=2,
            )
            + "\n"
        )
        raise


def _verify(before_path, after_path, *, selection, output, index_bits, max_dynamic_calls, effects_path):
    before_path, after_path = before_path.resolve(strict=True), after_path.resolve(strict=True)
    before_bytes, after_bytes = before_path.read_bytes(), after_path.read_bytes()
    originals = [_reference(before_path, before_bytes), _reference(after_path, after_bytes)]
    effects_bytes = effects_path.read_bytes() if effects_path is not None else None
    effects_ref = _reference(effects_path, effects_bytes) if effects_path is not None else None
    dependencies = selection.dependencies()
    copies = output / "inputs"
    copies.mkdir()
    before_copy, after_copy = copies / "before.kernel", copies / "after.kernel"
    before_copy.write_bytes(before_bytes)
    after_copy.write_bytes(after_bytes)
    effects_copy = None
    if effects_bytes is not None:
        effects_copy = copies / "effects.json"
        effects_copy.write_bytes(effects_bytes)
    base = verify_storage_files(
        before_copy,
        after_copy,
        selection=selection.base_proof,
        producer=selection.producer,
        output=output / "v1-check",
        index_bits=index_bits,
        max_dynamic_calls=max_dynamic_calls,
        effects_path=effects_copy,
    )
    record = {
        "schema": "merlin.graph.loop_capacity_check.v1",
        "originals": originals,
        "effects": effects_ref,
        "dependencies": dependencies,
        "v1_check": _reference(output / "v1-check/CHECK.json"),
        "lean_capacity_checked": False,
        "complete_boundary_checked": False,
        "python_projection_trusted": True,
        "target_capacity_and_effect_truth_premises": True,
        "concurrency_checked_by_selected_python_admission": True,
        "pipeline_receipt_issued": False,
        "numerical_gates_discharged": False,
        "renderer_or_hardware_proved": False,
    }
    if not base["lean_storage_checked"]:
        record["refusal"] = "unchanged v1 complete trace/storage check refused"
        (output / "CHECK.json").write_text(json.dumps(record, indent=2) + "\n")
        return record
    from . import lean_graph_loops_capacity_capture

    owners = [Path(__file__).resolve(), Path(lean_graph_loops_capacity_capture.__file__).resolve()]
    owner_refs = [_reference(path) for path in owners]
    producer = selection.producer
    fresh = output / "fresh-capacity.json"
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(producer.source / "src"), PYTHONDONTWRITEBYTECODE="1")
    command = [
        sys.executable,
        str(owners[1]),
        "--source",
        str(producer.source),
        "--revision",
        producer.revision,
        "--example",
        producer.example,
        "--iterations",
        str(producer.iterations),
        "--before",
        str(before_copy.resolve()),
        "--after",
        str(after_copy.resolve()),
        "--output",
        str(fresh.resolve()),
        "--index-bits",
        str(index_bits),
        "--max-dynamic-calls",
        str(max_dynamic_calls),
    ]
    projected = subprocess.run(command, capture_output=True, text=True, check=False, env=environment, timeout=120)
    (output / "projection.stdout").write_text(projected.stdout)
    (output / "projection.stderr").write_text(projected.stderr)
    if projected.returncode:
        raise ValueError("fresh actual capacity/admission projection refused; logs retained")
    fresh_bytes = fresh.read_bytes()
    fresh_ref = _reference(fresh, fresh_bytes)
    effects = json.loads(fresh_bytes)
    previous = json.loads((output / "v1-check/fresh-effects.json").read_bytes())
    for key, original_copy, data in (("before", before_copy, before_bytes), ("after", after_copy, after_bytes)):
        if effects[key] != _reference(original_copy, data) or previous[key] != _reference(
            output / "v1-check" / (key + ".kernel"), data
        ):
            raise ValueError("capacity and v1 projections do not name their exact checked input buffers")
    for key in (
        "blocks",
        "objects",
        "banks",
        "external_windows",
        "private_ownership",
        "adapter_identities",
        "producer_source",
    ):
        if effects[key] != previous[key]:
            raise ValueError("capacity admission is not bound to the same actual v1 event stream")
    if admission_digest(effects["admission"]) != selection.admission_sha256:
        raise ValueError("complete actual admission differs from the originally selected assumptions")
    if effects_bytes is not None and json.loads(effects_bytes)["admission"] != effects["admission"]:
        raise ValueError("saved admission differs from the complete actual selected admission")
    projection = project_pair(
        parse_kernel(before_bytes.decode()),
        parse_kernel(after_bytes.decode()),
        index_bits=index_bits,
        max_dynamic_calls=max_dynamic_calls,
    )
    query = output / "Obligation.lean"
    query_bytes = capacity_obligation(projection, effects, initialized_external=producer.initialized_external).encode()
    query.write_bytes(query_bytes)
    query_ref = _reference(query, query_bytes)
    proof = selection.proof
    environment.update(
        LEAN_PATH=str(proof.library), LEAN_SYSROOT=str(proof.executable.parent.parent), LEAN_NUM_THREADS="2"
    )
    lean_command = [str(proof.executable), "-DautoImplicit=false", "-DrelaxedAutoImplicit=false", str(query.resolve())]
    checked = subprocess.run(lean_command, capture_output=True, text=True, check=False, env=environment, timeout=120)
    (output / "stdout.log").write_text(checked.stdout)
    (output / "stderr.log").write_text(checked.stderr)
    record.update(
        query=query_ref,
        fresh_projection=fresh_ref,
        command=lean_command,
        projection_command=command,
        returncode=checked.returncode,
        trusted_capacity_sources=owner_refs,
        source_admission=effects["source_admission"],
        selected_admission=effects["admission"],
    )
    if (
        dependencies != selection.dependencies()
        or originals != [_reference(before_path), _reference(after_path)]
        or fresh_ref != _reference(fresh)
        or query_ref != _reference(query)
        or owner_refs != [_reference(path) for path in owners]
        or before_copy.read_bytes() != before_bytes
        or after_copy.read_bytes() != after_bytes
        or effects_ref != (_reference(effects_path) if effects_path is not None else None)
        or record["v1_check"] != _reference(output / "v1-check/CHECK.json")
        or any(_reference(Path(base[key]["path"])) != base[key] for key in ("query", "fresh_projection"))
    ):
        record["refusal"] = "original inputs or selected capacity/v1 identities changed during checking"
    elif checked.returncode:
        record["refusal"] = "Lean refused complete pending-capacity admission"
    else:
        record["transitive_axioms"] = _audit(checked.stdout, STORAGE_ROOTS + CAPACITY_ROOTS)
        record["lean_capacity_checked"] = True
        record["complete_boundary_checked"] = (
            base["complete_boundary_checked"] and effects["source_admission"]["accepted"]
        )
        if not record["complete_boundary_checked"]:
            record["refusal"] = "selected source concurrency/admission gate refused"
    (output / "CHECK.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
