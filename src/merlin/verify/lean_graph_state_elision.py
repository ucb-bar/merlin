"""Complete actual Kernel elision with a proved invariant/partial-outcome checker.

Pure admission is explicit and provisional. Retained calls may fail; the theorem
preserves error and terminal state. This leaf is not shared broker authority.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from merlin.common.owned_process import run_owned
from merlin.common.paths import repo_root

from . import lean_graph_state as stable
from .lean_graph_state_elision_kernel import check_query, prepare


def integer(value, *, natural=False):
    if type(value) is not int or natural and value < 0:
        raise ValueError("exact actual integer value/index required; bool/float refused")
    return str(value) if natural else "(" + str(value) + ")"


def pairs(values):
    if type(values) is not list or any(type(pair) is not list or len(pair) != 2 for pair in values):
        raise ValueError("complete ordered field/value pairs required")
    return "[" + ",".join("(" + integer(key, natural=True) + "," + integer(value) + ")" for key, value in values) + "]"


def effect(value):
    if value is None:
        return "none"
    if type(value) is not dict or set(value) != {"reads", "writes", "clobbers", "ordered", "pureWrite"}:
        raise ValueError("complete typed original effect record required")
    if type(value["ordered"]) is not bool or type(value["pureWrite"]) is not bool:
        raise ValueError("exact effect booleans required")
    return (
        "some {reads := "
        + pairs(value["reads"])
        + ",writes := "
        + pairs(value["writes"])
        + ",clobbers := "
        + str(stable._naturals(value["clobbers"]))
        + ",ordered := "
        + str(value["ordered"]).lower()
        + ",pureWrite := "
        + str(value["pureWrite"]).lower()
        + "}"
    )


def atom(value):
    if type(value) is not dict or set(value) != {"id", "token", "effect", "eraseAllowed"}:
        raise ValueError("complete actual call identity/effect/admission needed")
    if type(value["token"]) is not str or not value["token"] or type(value["eraseAllowed"]) is not bool:
        raise ValueError("bit-exact full call token and exact erasure eligibility required")
    return (
        "{id := "
        + integer(value["id"], natural=True)
        + ",token := "
        + json.dumps(value["token"], ensure_ascii=False)
        + ",effect := "
        + effect(value["effect"])
        + ",eraseAllowed := "
        + str(value["eraseAllowed"]).lower()
        + "}"
    )


def definitions(projection):
    if (
        type(projection["configuration_fields"]) is not list
        or len(projection["configuration_fields"]) > 256
        or any(type(field) is not str for field in projection["configuration_fields"])
        or len(set(projection["configuration_fields"])) != len(projection["configuration_fields"])
    ):
        raise ValueError("complete bounded unique actual configuration field domain required")
    for key in ("before", "after"):
        if type(projection[key]) is not list or len(projection[key]) > 4096:
            raise ValueError("complete bounded source/output call roster required")
    return (
        "def fields : List Nat := List.range " + str(len(projection["configuration_fields"])) + "\n"
        "def before : List Atom := [" + ",".join(atom(value) for value in projection["before"]) + "]\n"
        "def after : List Atom := [" + ",".join(atom(value) for value in projection["after"]) + "]\n"
    )


def emit_query(projection):
    return (
        "import Merlin.GraphStateElision\nset_option maxRecDepth 65536\nset_option maxHeartbeats 4000000\n"
        "namespace Merlin.GraphStateElisionInstance\n"
        "open Merlin.StateEffects Merlin.GraphState.Elision\n"
        + definitions(projection)
        + "theorem accepted : check fields before after = true := by decide\n"
        "theorem actual_elision : Preserves fields before after := check_sound accepted\n"
        "theorem full_observations {Value Error Observation : Type} (literal : Int → Value)\n"
        "    (run : Nat → State Value → Except (Error × State Value) (State Value))\n"
        "    (models : ∀ atom ∈ before, Admitted fields literal run atom) (initial : State Value)\n"
        "    (observe : Except (Error × State Value) (State Value) → Observation) :\n"
        "    observe (execute run before initial) = observe (execute run after initial) :=\n"
        "  congrArg observe (actual_elision Value Error literal run models initial)\n"
        "#print axioms actual_elision\n#print axioms full_observations\nend Merlin.GraphStateElisionInstance\n"
    )


def sources():
    root = repo_root() / "src/merlin/verify"
    paths = [*root.glob("lean_graph_state_elision*.py"), *(root / "lean/Merlin").glob("GraphStateElision*.lean")]
    paths += [root / "lean_graph_state.py", root / "lean/Merlin/GraphStateAudit.lean"]
    paths += [root / f"lean/Merlin/{name}.lean" for name in stable.SOURCES]
    return [stable.file_ref(path) for path in sorted(set(paths))]


def check_boundary(
    before, after, destination, *, reference_root, contract, original_contract_sha256, expected_originals=None
):
    before, after, contract = (Path(path).resolve() for path in (before, after, contract))
    original = [stable.file_ref(path) for path in (before, after)]
    selected_contract = stable.file_ref(contract)
    if selected_contract["sha256"] != original_contract_sha256:
        raise ValueError("selected decoder/admission differs from original contract identity")
    if expected_originals is not None and original != expected_originals:
        raise ValueError("actual source/result originals differ from selected original buffers")
    selected_sources = sources()
    runtime = stable.host_runtime()
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=False)
    snapshots = []
    for identity, name in zip([*original, selected_contract], ("before.mk", "after.mk", "contract.json"), strict=True):
        target = root / name
        target.write_bytes(Path(identity["path"]).read_bytes())
        actual = stable.file_ref(target)
        if any(actual[key] != identity[key] for key in ("sha256", "bytes")):
            raise ValueError("original changed while retaining exact checked snapshot")
        snapshots.append(actual)
    script = repo_root() / "src/merlin/verify/lean_graph_state_elision_capture.py"
    with (root / "projection.stdout").open("x") as out, (root / "projection.stderr").open("x") as err:
        result = run_owned(
            [
                sys.executable,
                str(script),
                "--reference",
                str(reference_root),
                "--source",
                snapshots[0]["path"],
                "--target",
                snapshots[1]["path"],
                "--contract",
                snapshots[2]["path"],
                "--destination",
                str(root / "projection"),
            ],
            cwd=reference_root,
            env={**os.environ, "PYTHONPATH": str(Path(reference_root) / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=out,
            stderr=err,
            timeout=60,
        )
    report = {
        "schema": "merlin.graph-state-elision.boundary.v1",
        "originals": original,
        "snapshots": snapshots,
        "contract": selected_contract,
        "public_acceptance_issued": False,
        "admission_approved": False,
        "native_fidelity_proved": False,
        "source_success_proved": False,
        "poison_freedom_proved": False,
        "numerical_gates_discharged": False,
        "full_LeanMLIR_route": "unavailable: no matching selected manifest",
    }
    if result.returncode:
        report.update(status="unavailable", reason="pinned actual producer/projection process failed")
    else:
        projection = json.loads((root / "projection/projection.json").read_text())
        if not projection["projection_checked"]:
            report.update(status="unsupported", reason=projection["reason"])
        else:
            checked = check_query(emit_query(projection), root / "kernel")
            profile = prepare()
            dependencies = [
                *selected_sources,
                *projection["producer_sources"],
                *profile["sources"],
                *profile["base_objects"],
                *profile["files"],
                profile["lean"],
                profile["manifest"],
                profile["base_manifest"],
                runtime["python"],
                *runtime["support_sources"],
                projection["producer_external_runtime"]["python"],
                *projection["producer_external_runtime"]["loaded_sources"],
                *checked["audit"].get("imported_objects", []),
            ]
            copied = stable.retain_sources(root, selected_sources + projection["producer_sources"])
            if not stable.fresh([*original, selected_contract, *snapshots, *dependencies]):
                raise ValueError("actual originals/snapshots/code/decoder/proof/tool/import changed during check")
            for actual, saved in zip([*original, selected_contract], snapshots, strict=True):
                if Path(actual["path"]).read_bytes() != Path(saved["path"]).read_bytes():
                    raise ValueError("actual original and checked snapshot bytes differ")
            report.update(
                status=checked["status"],
                kernel=stable.file_ref(root / "kernel/report.json"),
                projection=stable.file_ref(root / "projection/projection.json"),
                dependencies=dependencies,
                source_closure=copied,
                host_runtime=runtime,
                genuine_changed_boundary=projection["genuine_changed_boundary"],
                removed_calls=projection["removed_calls"],
                read_establishment=projection["read_establishment"],
                assumptions=projection["trust"],
                theorem="Merlin.GraphStateElisionInstance.actual_elision",
                relation="same complete successful state OR same error and terminal failed state; "
                "arbitrary initial configuration",
            )
    stable._write(root / "report.json", report)
    return report
