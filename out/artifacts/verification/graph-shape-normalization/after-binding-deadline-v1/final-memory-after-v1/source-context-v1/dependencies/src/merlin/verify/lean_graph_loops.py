"""Kernel-checked complete counted-to-closed call correspondence.

The existing Kernel parser and this projection are trusted Python. Acceptance
here checks the stated relation on actual input objects/files; it does not issue
an accepted compiler-pipeline receipt or prove the instruction adapter or C.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from merlin.sched.ir import (
    NULL,
    Call,
    Const,
    Expr,
    Kernel,
    Loop,
    Ptr,
    Scaled,
    Select,
    Sum,
    Var,
    check_structure,
    parse_kernel,
)

from .lean_graph_loops_capture import _call


def _reference(path: Path, data: bytes | None = None) -> dict:
    if data is None:
        data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _projection_python_paths() -> list[Path]:
    """The actual loaded primary IR namespace, owners and interpreter identity.

    Python's standard library/runtime remains trusted, independently of this
    source inventory. Include the re-export initializer and every loaded IR leaf.
    """
    from . import lean_graph_loops_capture

    paths = {
        Path(__file__).resolve(),
        Path(lean_graph_loops_capture.__file__).resolve(),
        Path(sys.executable).resolve(),
    }
    for name, module in tuple(sys.modules.items()):
        if name == "merlin.sched.ir" or name.startswith("merlin.sched.ir."):
            location = getattr(module, "__file__", None)
            if location is not None:
                paths.add(Path(location).resolve())
    return sorted(paths)


def _name(value: str) -> str:
    if type(value) is not str or not value or any(ord(char) < 32 or ord(char) > 126 for char in value):
        raise ValueError("loop projection requires nonempty printable ASCII names")
    return json.dumps(value)


def _integer(value: int) -> str:
    if type(value) is not int:
        raise ValueError("loop indices must be actual integers")
    return f"({value})"


def _expression(value: dict) -> str:
    kind = value["kind"]
    if kind == "constant":
        return f"(.constant {_integer(value['value'])})"
    if kind == "variable":
        return f"(.variable {_name(value['name'])})"
    if kind == "sum":
        result = "(.constant 0)"
        for term in value["terms"]:
            result = f"(.add {result} {_expression(term)})"
        return result
    if kind == "scaled":
        return f"(.scaled {_expression(value['value'])} {_integer(value['factor'])})"
    if kind == "select":
        return (
            f"(.select {_name(value['name'])} {_integer(value['equals'])} "
            f"{_expression(value['yes'])} {_expression(value['no'])})"
        )
    raise ValueError("unsupported loop expression")


def _operand(value: dict, *, closed: bool) -> str:
    kind = value["kind"]
    if kind == "null":
        return ".null"
    expression = value["expression"] if kind == "integer" else value["offset"]
    if closed:
        if expression["kind"] != "constant":
            raise ValueError("realized call has a nonconstant operand")
        text = _integer(expression["value"])
    else:
        text = _expression(expression)
    if kind == "integer":
        return f"(.integer {text})"
    if kind == "pointer":
        return f"(.pointer {_name(value['space'])} {text})"
    raise ValueError("unsupported loop operand")


def _call_literal(call: dict, *, closed: bool) -> str:
    names = [name for name, _argument in call["arguments"]]
    if len(set(names)) != len(names):
        raise ValueError("duplicate operand names cannot define an exact call binding")
    arguments = ", ".join(
        f"({_name(name)}, {_operand(argument, closed=closed)})" for name, argument in call["arguments"]
    )
    return f"⟨{_name(call['instruction'])}, [{arguments}]⟩"


def _calls(calls: list[dict], *, closed: bool) -> str:
    return "[\n  " + ",\n  ".join(_call_literal(call, closed=closed) for call in calls) + "\n]"


def project_pair(before: Kernel, after: Kernel, *, index_bits: int, max_dynamic_calls: int) -> dict:
    """Read the complete actual IR; candidate counts/trace metadata are unused."""
    if type(before) is not Kernel or type(after) is not Kernel or check_structure(before) or check_structure(after):
        raise ValueError("loop correspondence requires structurally valid actual Kernels")
    if (before.name, before.args, before.attrs) != (after.name, after.args, after.attrs) or before.attrs:
        raise ValueError("loop materialization changed its source boundary or has unknown attributes")
    if len(before.body) != 1 or type(before.body[0]) is not Loop or any(type(call) is not Call for call in after.body):
        raise ValueError("supported boundary is one counted call loop to one complete closed call sequence")
    loop = before.body[0]
    if any(type(call) is not Call for call in loop.body):
        raise ValueError("nested loops need a separately checked complete domain")
    if type(loop.extent) is not int or loop.extent <= 0 or not loop.body:
        raise ValueError("empty or noninteger loop domains do not qualify this boundary")
    if type(index_bits) is not int or index_bits not in (32, 64):
        raise ValueError("select an explicit signed 32-bit or 64-bit index domain")
    if type(max_dynamic_calls) is not int or max_dynamic_calls <= 0:
        raise ValueError("an explicit positive complete-domain work budget is required")
    needed = loop.extent * len(loop.body)
    if needed > max_dynamic_calls:
        raise ValueError(f"complete loop needs {needed} calls, budget is {max_dynamic_calls}; no prefix checked")
    if len(after.body) != needed:
        raise ValueError(f"realized trace has {len(after.body)} calls; complete domain requires {needed}")
    forms = Const, Var, Sum, Scaled, Select
    body = [_call(call, forms, Ptr, NULL, Expr) for call in loop.body]
    target = [_call(call, forms, Ptr, NULL, Expr) for call in after.body]
    return {
        "variable": loop.var,
        "extent": loop.extent,
        "bits": index_bits,
        "body": body,
        "target": target,
        "dynamic_calls": needed,
    }


def obligation(projection: dict) -> str:
    source = (
        f"⟨{_name(projection['variable'])}, {projection['extent']}, {projection['bits']}, "
        f"{_calls(projection['body'], closed=False)}⟩"
    )
    target = _calls(projection["target"], closed=True)
    return f"""import Merlin.GraphLoops
set_option autoImplicit false
open Merlin GraphLoops
def source : Counted := {source}
def actual : List ClosedCall := {target}
theorem accepted : check source actual = true := by cbv
theorem complete_count : actual.length = source.extent * source.body.length := complete_call_count accepted
theorem preserves {{Value : Type}} (run : ClosedCall → StateEffects.State Value → StateEffects.State Value)
    (initial : StateEffects.State Value) :
    ∃ blocks, expand source = some blocks ∧
      blocks.foldl (fun state block => StateEffects.execute (block.map run) state) initial =
        StateEffects.execute (actual.map run) initial := checked_execution accepted run initial
#print axioms accepted
#print axioms complete_count
#print axioms preserves
#print axioms Merlin.GraphLoops.check_sound
#print axioms Merlin.GraphLoops.complete_iterations
#print axioms Merlin.GraphLoops.checked_execution
"""


def _audit(stdout: str, roots: tuple[str, ...] | None = None) -> dict:
    allowed = {"propext", "Classical.choice", "Quot.sound"}
    roots = (
        roots
        if roots is not None
        else (
            "accepted",
            "complete_count",
            "preserves",
            "Merlin.GraphLoops.check_sound",
            "Merlin.GraphLoops.complete_iterations",
            "Merlin.GraphLoops.checked_execution",
        )
    )
    result = {}
    for root in roots:
        marker = f"'{root}'"
        if stdout.count(marker) != 1:
            raise ValueError("missing or ambiguous transitive Lean axiom report")
        tail = stdout.partition(marker)[2].split("\n", 1)[0].removesuffix("\r")
        if tail == " does not depend on any axioms":
            axioms = []
        elif tail.startswith(" depends on axioms: [") and tail.endswith("]"):
            names = tail.partition("[")[2].partition("]")[0]
            axioms = [name.strip() for name in names.split(",") if name.strip()]
        else:
            raise ValueError("unexpected Lean axiom report")
        if not set(axioms) <= allowed:
            raise ValueError(f"unapproved axiom at {root}: {axioms}")
        result[root] = axioms
    return result


@dataclass(frozen=True)
class LeanSelection:
    executable: Path
    sha256: str
    library: Path
    build_record: Path
    build_sha256: str

    def dependencies(self) -> list[dict]:
        if _reference(self.executable)["sha256"] != self.sha256:
            raise ValueError("selected Lean executable changed")
        build_bytes = self.build_record.read_bytes()
        if hashlib.sha256(build_bytes).hexdigest() != self.build_sha256:
            raise ValueError("originally selected proof/semantic dependency record changed")
        build = json.loads(build_bytes)
        if build["compiler"]["sha256"] != self.sha256 or build["native_reflection"]:
            raise ValueError("proof build differs from selected kernel-only toolchain")
        dependencies = [_reference(self.executable), _reference(self.build_record)]
        for attempt in build["attempts"]:
            if attempt["returncode"] != 0:
                raise ValueError("proof dependency build did not pass")
            snapshot = Path(attempt["source"]["path"])
            original = Path(build["source_root"]) / Path(*attempt["module"].split(".")).with_suffix(".lean")
            if original.read_bytes() != snapshot.read_bytes() or _reference(snapshot) != attempt["source"]:
                raise ValueError("proof source differs from its actual compiled snapshot")
            target = Path(attempt["object"]["path"])
            expected = self.library / Path(*attempt["module"].split(".")).with_suffix(".olean")
            if target.resolve() != expected.resolve():
                raise ValueError("selected import search path differs from the pinned proof objects")
            if _reference(target) != attempt["object"]:
                raise ValueError("compiled proof dependency changed")
            dependencies.extend((_reference(original), _reference(target)))
        return dependencies


def verify_files(
    before_path: Path,
    after_path: Path,
    *,
    selection: LeanSelection,
    output: Path,
    index_bits: int,
    max_dynamic_calls: int,
) -> dict:
    """Check fresh actual files, preserving refusal logs; saved reports grant no authority."""
    before_path, after_path = before_path.resolve(strict=True), after_path.resolve(strict=True)
    before_bytes, after_bytes = before_path.read_bytes(), after_path.read_bytes()
    before_ref, after_ref = _reference(before_path, before_bytes), _reference(after_path, after_bytes)
    before, after = parse_kernel(before_bytes.decode()), parse_kernel(after_bytes.decode())
    projection = project_pair(before, after, index_bits=index_bits, max_dynamic_calls=max_dynamic_calls)
    dependencies = selection.dependencies()
    python_paths = _projection_python_paths()
    python_dependencies = [_reference(path) for path in python_paths]
    output.mkdir(parents=True, exist_ok=False)
    query = output / "Obligation.lean"
    query_bytes = obligation(projection).encode()
    query.write_bytes(query_bytes)
    query_ref = _reference(query, query_bytes)
    environment = os.environ.copy()
    environment.update(
        LEAN_PATH=str(selection.library), LEAN_SYSROOT=str(selection.executable.parent.parent), LEAN_NUM_THREADS="2"
    )
    command = [str(selection.executable), "-DautoImplicit=false", "-DrelaxedAutoImplicit=false", str(query.resolve())]
    result = subprocess.run(command, text=True, capture_output=True, env=environment, check=False, timeout=120)
    (output / "stdout.log").write_text(result.stdout)
    (output / "stderr.log").write_text(result.stderr)
    record = {
        "schema": "merlin.graph.loop_relation_check.v1",
        "before": before_ref,
        "after": after_ref,
        "projection": projection,
        "query": query_ref,
        "command": command,
        "dependencies": dependencies,
        "returncode": result.returncode,
        "trusted_python_dependencies": python_dependencies,
        "lean_kernel_relation_checked": False,
        "pipeline_receipt_issued": False,
        "python_projection_trusted": True,
        "numerical_gates_discharged": False,
        "renderer_or_hardware_proved": False,
    }
    if (
        before_ref != _reference(before_path)
        or after_ref != _reference(after_path)
        or dependencies != selection.dependencies()
        or python_dependencies != [_reference(path) for path in python_paths]
        or query_ref != _reference(query)
    ):
        record["refusal"] = "actual IR or proof/tool dependencies changed during checking"
    elif result.returncode:
        record["refusal"] = "Lean refused complete source/result call correspondence"
    else:
        record["transitive_axioms"] = _audit(result.stdout)
        record["lean_kernel_relation_checked"] = True
    (output / "CHECK.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    return record
