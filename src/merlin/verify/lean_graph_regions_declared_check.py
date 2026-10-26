"""Actual-buffer scalar queries for the hash-free declared-call checker.

Proposals are untrusted hints. Lean checks every exported operation and property.
Compiler validity, parser trust, content binding and proof acceptance are separate.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.verify import lean_graph_regions as base
from merlin.verify import lean_graph_regions_declared_ast as ast
from merlin.verify import lean_graph_regions_mlir as compiler


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def export(source: Path, destination: Path, python: Path):
    exporter = repo_root() / "src/merlin/verify/lean/mlir_export.py"
    refs = dict(source=base.file_ref(source), exporter=base.file_ref(exporter), python=base.file_ref(python))
    command = [str(python), "-B", str(exporter), str(source), str(destination)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    save(
        destination.with_suffix(".process.json"),
        dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr, inputs=refs),
    )
    result.check_returncode()
    for reference in refs.values():
        base.fresh_ref(reference)
    document = json.loads(destination.read_text())
    if document["source"] != {k: refs["source"][k] for k in ("sha256", "bytes")}:
        raise ValueError("export does not bind the actual source bytes")
    return document, refs


def propose(document):
    """Canonical scalar hint; unsupported input is unavailable, never a proof."""
    (function,) = document["module"]["ops"]
    (region,) = function["regions"]
    (block,) = region["blocks"]
    widths = [value["type"]["width"] for value in block["args"]]
    instructions = []
    for op in block["ops"][:-1]:
        (result,) = op["results"]
        width = result["type"]["width"]
        name = op["name"]
        if name in {"arith.addi", "arith.subi", "arith.muli"}:
            kind = {"arith.addi": "add", "arith.subi": "sub", "arith.muli": "mul"}[name]
            primitive = f"(.binary {width} .{kind})"
        elif name == "arith.constant":
            primitive = f"(.constant {width} {ast.signed(op['attrs']['value']['value'])})"
        elif name in {"arith.trunci", "arith.extsi", "arith.extui"}:
            (source,) = op["operands"]
            kind = {"arith.trunci": "trunc", "arith.extsi": "signed", "arith.extui": "unsigned"}[name]
            primitive = f"(.cast {widths[source]} {width} .{kind})"
        else:
            raise ValueError("unsupported scalar operation: " + name)
        instructions.append(f"⟨{primitive},{ast.array(op['operands'], ast.natural)}⟩")
        widths.append(width)
    (returned,) = block["ops"][-1]["operands"]
    (result_type,) = function["attrs"]["function_type"]["value"]["results"]
    model = (
        f"⟨{ast.array([value['type']['width'] for value in block['args']], ast.natural)},"
        f"[{','.join(instructions)}],{returned},{result_type['width']}⟩"
    )
    visibility = function["attrs"].get("sym_visibility", {}).get("value")
    return model, function["attrs"]["sym_name"]["value"], visibility


def query(source, declaration, *, expected: bool, program=None):
    model, source_name, source_visibility = propose(source)
    (function,) = declaration["module"]["ops"]
    declaration_name = function["attrs"]["sym_name"]["value"]
    declaration_visibility = function["attrs"].get("sym_visibility", {}).get("value")
    return (
        f"""import Merlin.GraphRegionsDeclaredEvaluation
open MerlinVerify Merlin.GraphRegionsDeclared
def actualSource : Module := {ast.module(source)}
def actualDeclaration : Module := {ast.module(declaration)}
def proposedProgram : Program := {program or model}
theorem actual_acceptance : checkScalar actualSource actualDeclaration
  {ast.text(source_name)} {ast.text(declaration_name)}
  {ast.optional(source_visibility, ast.text)} {ast.optional(declaration_visibility, ast.text)}
  proposedProgram = {str(expected).lower()} := by
"""
        + (
            "  apply checkScalar_iff.mpr\n  exact ⟨rfl, rfl, by decide⟩\n"
            if expected
            else "  apply checkScalar_refuse_trace\n  decide\n"
        )
        + (
            """
theorem actual_common_execution {α : Type} (meaning : Primitive → List α → α)
    (arguments : Environment α) :
    ScalarExecution meaning actualSource arguments (proposedProgram.evaluate meaning arguments) ∧
    ScalarExecution meaning actualDeclaration arguments (proposedProgram.evaluate meaning arguments) :=
  checkScalar_sound actual_acceptance meaning arguments
#print axioms actual_common_execution
"""
            if expected
            else "#print axioms actual_acceptance\n"
        )
    )


def scalar_pair(
    source: Path, declaration: Path, destination: Path, *, selected, lean, environment, expected=True, program=None
):
    destination.mkdir(parents=True, exist_ok=False)
    originals, snapshots, documents, exports, validity = {}, {}, {}, {}, {}
    for role, path in (("source", source), ("declaration", declaration)):
        originals[role] = base.file_ref(path)
        snapshot = destination / (role + ".mlir")
        snapshot.write_bytes(base.fresh_ref(originals[role]).read_bytes())
        snapshots[role] = base.file_ref(snapshot)
        validity[role] = compiler.compiler_verify(
            snapshot, destination / (role + "-compiler"), python=Path(selected["oracle_invocation"])
        )
        if validity[role]["oracle"]["valid"] is not True:
            raise ValueError("malformed compiler MLIR is not a semantic refusal")
        documents[role], exports[role] = export(
            snapshot, destination / (role + ".json"), Path(selected["oracle_invocation"])
        )
    proof = destination / "Check.lean"
    proof.write_text(query(documents["source"], documents["declaration"], expected=expected, program=program))
    command = [str(lean), "-j1", str(proof.resolve())]
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=60)
    save(
        destination / "process.json",
        dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr),
    )
    for role in originals:
        if base.fresh_ref(originals[role]).read_bytes() != base.fresh_ref(snapshots[role]).read_bytes():
            raise ValueError("original/snapshot content association changed")
    receipt = dict(
        schema="merlin.graph-regions.declared-scalar-development.v1",
        originals=originals,
        snapshots=snapshots,
        exports=exports,
        validity=validity,
        expected=expected,
        lean_succeeded=result.returncode == 0,
        proof=base.file_ref(proof),
        process=base.file_ref(destination / "process.json"),
        qualified=False,
        source_operational_bridge="UNQUALIFIED; not imported",
        native_fidelity_proved=False,
    )
    save(destination / "EVIDENCE.json", receipt)
    result.check_returncode()
    return receipt
