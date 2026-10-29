"""Complete actual constant-prefix configuration hoist with a kernel sound checker.

The admitted operation model and constant-overwrite law stay visible in the
theorem. This development leaf issues no shared broker/public acceptance.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from merlin.common.owned_process import run_owned
from merlin.common.paths import repo_root

from .lean_graph_state import (
    LEAN,
    _naturals,
    _write,
    audit_claim,
    file_ref,
    fresh,
    host_runtime,
    prepare,
    retain_sources,
)


def emit_query(projection):
    footprints = projection["footprints"]
    if type(footprints) is not list or not footprints or len(footprints) > 4096:
        raise ValueError("complete bounded static configuration roster required")
    cells = projection["cells"]
    if type(cells) is not list or any(type(cell) is not str for cell in cells) or len(set(cells)) != len(cells):
        raise ValueError("complete unique state-cell roster required")
    effects = []
    for row in footprints:
        reads, writes = _naturals(row["reads"]), _naturals(row["writes"])
        if max([*reads, *writes], default=-1) >= len(cells):
            raise ValueError("configuration footprint refers to absent state cell")
        effects.append("⟨" + str(reads) + "," + str(writes) + "⟩")
    move = projection["move"]
    for key in ("earlier", "body", "later", "fields"):
        _naturals(move[key])
    _naturals([move["setup"], move["extent"]])
    if move["extent"] > 10**9:
        raise ValueError("explicit counted-loop extent budget exceeded")

    def code(commands):
        result = []
        for item in commands:
            if set(item) == {"call"}:
                _naturals([item["call"]])
                result.append(".call " + str(item["call"]))
            elif set(item) == {"loop", "calls"}:
                _naturals([item["loop"], *item["calls"]])
                result.append(".loop " + str(item["loop"]) + " " + str(item["calls"]))
            else:
                raise ValueError("unknown actual command shape")
        return "[" + ",".join(result) + "]"

    values = projection["field_values"]
    if any(type(pair) is not list or len(pair) != 2 or type(pair[1]) is not int for pair in values):
        raise ValueError("exact integer setup values required")
    _naturals([pair[0] for pair in values])
    if [pair[0] for pair in values] != move["fields"] or len(set(move["fields"])) != len(move["fields"]):
        raise ValueError("complete unique setup field/value binding required")
    configured = "\n".join(f"  | {key} => literal ({value})" for key, value in values)
    fields = ",".join(f"{key} := {move[key]}" for key in ("earlier", "setup", "body", "extent", "later", "fields"))
    return (
        "import Merlin.GraphStateConfiguration\nnamespace Merlin.GraphStateConfigurationInstance\n"
        "open Merlin.StateEffects Merlin.GraphState\nopen Merlin.GraphState.Configuration\n"
        "def footprints : List Footprint := [" + ",".join(effects) + "]\n"
        "def before : List Command := " + code(projection["before"]) + "\n"
        "def after : List Command := " + code(projection["after"]) + "\n"
        "def move : Hoist := {" + fields + "}\n"
        "def configured {Value : Type} (literal : Int → Value) : State Value\n" + configured + "\n  | _ => literal 0\n"
        "theorem accepted : check footprints before after move = true := by decide\n"
        "theorem actual_configuration {Value : Type} (run : Nat → Nat → State Value → State Value)\n"
        "    (model : ∀ index, Models footprints (fun id => run id index)) (literal : Int → Value)\n"
        "    (binding : ∀ index, run move.setup index = put move.fields (configured literal))\n"
        "    (initial : State Value) : program run before initial = program run after initial :=\n"
        "  check_sound accepted run model (configured literal) binding initial\n"
        "#print axioms actual_configuration\nend Merlin.GraphStateConfigurationInstance\n"
    )


def check_boundary(before, after, destination, *, reference_root, contract, original_contract_sha256=None):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    originals = [file_ref(path) for path in (before, after, contract)]
    if original_contract_sha256 is not None and originals[2]["sha256"] != original_contract_sha256:
        raise ValueError("configuration contract differs from original selected semantic identity")
    retained = []
    for record, name in zip(originals, ("before.mk", "after.mk", "contract.json"), strict=True):
        path = destination / name
        path.write_bytes(Path(record["path"]).read_bytes())
        identity = file_ref(path)
        if any(identity[key] != record[key] for key in ("sha256", "bytes")):
            raise ValueError("configuration inputs changed during retention")
        retained.append(identity)
    root = repo_root() / "src/merlin/verify"
    owned = [
        file_ref(root / name)
        for name in (
            "lean_graph_state.py",
            "lean_graph_state_capture.py",
            "lean_graph_state_configuration.py",
            "lean_graph_state_configuration_capture.py",
            "lean/Merlin/GraphStateAudit.lean",
        )
    ]
    command = [
        sys.executable,
        str(root / "lean_graph_state_configuration_capture.py"),
        "--reference",
        str(reference_root),
        "--source",
        retained[0]["path"],
        "--target",
        retained[1]["path"],
        "--contract",
        retained[2]["path"],
        "--destination",
        str(destination / "projection"),
    ]
    with (destination / "projection.stdout").open("x") as out, (destination / "projection.stderr").open("x") as err:
        produced = run_owned(
            command,
            cwd=reference_root,
            env={**os.environ, "PYTHONPATH": str(Path(reference_root) / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=out,
            stderr=err,
            timeout=60,
            check=False,
        )
    result = {
        "schema": "merlin.graph-state.configuration-boundary.v1",
        "inputs": originals,
        "host_runtime": host_runtime(),
        "retained_inputs": retained,
        "public_acceptance_issued": False,
        "source_contract_approved": False,
        "native_fidelity_proved": False,
        "numerical_gates_discharged": False,
        "full_LeanMLIR_route": "unavailable: no matching selected manifest",
        "pipeline_broker_integration": "pending agreed shared interface",
    }
    if produced.returncode:
        result.update(status="unavailable", reason="pinned configuration projection process failed")
    else:
        capture = json.loads((destination / "projection/capture.json").read_text())
        if not capture["projection_checked"]:
            result.update(status="unsupported", reason=capture["reason"])
        else:
            frozen = prepare()
            query = destination / "GraphStateConfigurationInstance.lean"
            query.write_text(emit_query(capture))
            with (destination / "kernel.stdout").open("x") as out, (destination / "kernel.stderr").open("x") as err:
                checked = run_owned(
                    [str(LEAN), str(query), "-o", str(query.with_suffix(".olean"))],
                    cwd=destination,
                    env={**os.environ, "LEAN_PATH": str(frozen["lib"])},
                    stdout=out,
                    stderr=err,
                    timeout=60,
                    check=False,
                )
            audit = (
                audit_claim(
                    destination,
                    frozen,
                    "GraphStateConfigurationInstance",
                    [
                        "Merlin.GraphState.Configuration.hoist_nonempty",
                        "Merlin.GraphState.Configuration.check_sound",
                        "Merlin.GraphStateConfigurationInstance.accepted",
                        "Merlin.GraphStateConfigurationInstance.actual_configuration",
                    ],
                )
                if checked.returncode == 0
                else {"status": "not_run"}
            )
            sources = [*owned, *capture["producer_sources"], *frozen["sources"]]
            closure = retain_sources(destination, sources)
            if not fresh(
                [
                    *originals,
                    *retained,
                    *sources,
                    *frozen["files"],
                    frozen["manifest"],
                    frozen["lean"],
                    *audit.get("imported_objects", []),
                    *result["host_runtime"]["support_sources"],
                    result["host_runtime"]["python"],
                    *capture["external_runtime"]["loaded_sources"],
                    capture["external_runtime"]["python"],
                ]
            ):
                raise ValueError("configuration IR/source/proof/tool identity changed during check")
            result.update(
                status="graph_model_checked"
                if checked.returncode == 0 and audit["status"] == "checked"
                else "unavailable",
                theorem="Merlin.GraphStateConfigurationInstance.actual_configuration",
                existing_root="Merlin.StateEffects.Action.frame",
                checker_root="Merlin.GraphState.Configuration.check_sound",
                query=file_ref(query),
                proof_profile=frozen["manifest"],
                source_closure=closure,
                transitive_audit=audit,
                projection=file_ref(destination / "projection/capture.json"),
                assumptions=capture["assumptions"],
                producer_verification=capture["producer_verification"],
                producer_external_runtime=capture["external_runtime"],
            )
    _write(destination / "report.json", result)
    return result
