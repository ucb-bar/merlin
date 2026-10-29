"""Kernel-checked actual connected-region plans under explicit pure-action semantics.

This leaf extends Merlin.StateEffects through GraphRegions. It issues descriptive
scoped evidence, not the shared pipeline's live acceptance authority. The xDSL
projection and native realization/operational bridges remain separate trust gaps.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.verify import lean_graph_regions_dependencies as dependencies
from merlin.verify import lean_graph_regions_projection as projection

LEAN_SHA = "bf8d54e4714cc4b03d3f6bb34c83b7202b87e49c8bfcbff6895c085bb90ceb38"
STATE_EFFECTS_SHA = "f5bfccb9d8141a7979a350699d1cffd06eb5f8bc9272841fafe77e64f375b176"
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
ROOTS = (
    "external_read_retained",
    "execute_frame",
    "execute_typed",
    "execute_locality",
    "checkPlan_decode",
    "checkPlan_preserves_context",
    "checkPlan_interface",
    "checkPlan_publishes",
    "suffix_congruence",
    "replace_region_preserves_observers",
    "split_nodes",
    "checkBoundary_sound",
)


def file_ref(path: Path) -> dict:
    path = path.resolve(strict=True)
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if any(
        getattr(before, k) != getattr(after, k) for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        raise ValueError(f"file changed while being read: {path}")
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def fresh_ref(record) -> Path:
    if (
        type(record) is not dict
        or set(record) != {"path", "sha256", "bytes"}
        or type(record["path"]) is not str
        or not Path(record["path"]).is_absolute()
        or type(record["bytes"]) is not int
        or record["bytes"] < 0
        or type(record["sha256"]) is not str
        or len(record["sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in record["sha256"])
    ):
        raise ValueError("malformed strict file reference")
    path = Path(record["path"])
    if file_ref(path) != record:
        raise ValueError("stale file reference")
    return path


def _nat(value):
    if type(value) is not int or value < 0:
        raise ValueError("natural-number fields must not contain booleans or negative values")
    return str(value)


def _nats(values):
    if type(values) is not list or len(values) > 10000:
        raise ValueError("bounded list required")
    return "[" + ",".join(_nat(v) for v in values) + "]"


def _strings(values):
    if type(values) is not list or any(type(s) is not str or not s.isascii() or len(s) > 4096 for s in values):
        raise ValueError("bounded ASCII strings required")
    return "[" + ",".join(json.dumps(s) for s in values) + "]"


def _key(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("exact semantic source hash required")
    return json.dumps(value)


def _node(node):
    return (
        "⟨"
        + ",".join(
            (
                _nat(node["ordinal"]),
                _key(node["semanticKey"]),
                _nats(node["operands"]),
                _nats(node["results"]),
                _strings(node["operandTypes"]),
                _strings(node["resultTypes"]),
            )
        )
        + "⟩"
    )


def _plan(plan):
    if type(plan) is not dict or set(plan) != {"values", "stages", "inputs", "outputs", "lifetimes"}:
        raise ValueError("complete exact plan fields required")
    stages = []
    for stage in plan["stages"]:
        if type(stage) is not dict or set(stage) != {
            "ordinal",
            "semanticKey",
            "inputs",
            "outputs",
            "operandTypes",
            "resultTypes",
        }:
            raise ValueError("complete exact stage binding required")
        stages.append(
            "⟨"
            + ",".join(
                (
                    _nat(stage["ordinal"]),
                    _key(stage["semanticKey"]),
                    _nats(stage["inputs"]),
                    _nats(stage["outputs"]),
                    _strings(stage["operandTypes"]),
                    _strings(stage["resultTypes"]),
                )
            )
            + "⟩"
        )
    lives = []
    for pair in plan["lifetimes"]:
        if type(pair) is not list or len(pair) != 2 or type(pair[0]) is not int or pair[0] < -1:
            raise ValueError("exact logical lifetime pair required")
        lives.append(f"({pair[0]},{_nat(pair[1])})")
    return (
        "⟨"
        + ",".join(
            (
                _nats(plan["values"]),
                "[" + ",".join(stages) + "]",
                _nats(plan["inputs"]),
                _nats(plan["outputs"]),
                "[" + ",".join(lives) + "]",
            )
        )
        + "⟩"
    )


def _run(command, work: Path, label: str, environment):
    if command[0].endswith("/lean"):
        # Explicit deterministic execution profile. No proof or recursion limit
        # is raised; SIGSEGVs observed under the default worker selection remain
        # preserved failures rather than being retried automatically.
        command = [command[0], "-j1", *command[1:]]
    result = subprocess.run(command, cwd=repo_root(), env=environment, capture_output=True, text=True, timeout=120)
    (work / f"{label}.stdout").write_text(result.stdout)
    (work / f"{label}.stderr").write_text(result.stderr)
    (work / f"{label}.json").write_text(json.dumps({"command": command, "returncode": result.returncode}) + "\n")
    result.check_returncode()
    return result


def build(work: Path, *, fusion: bool = False) -> tuple[Path, dict, dict]:
    work.mkdir(parents=True, exist_ok=True)
    root = repo_root()
    lean = Path.home() / ".elan/toolchains/leanprover--lean4---v4.35.0/bin/lean"
    if file_ref(lean)["sha256"] != LEAN_SHA:
        raise ValueError("selected Lean executable differs; no dependency download is permitted")
    sources = root / "src/merlin/verify/lean"
    if file_ref(sources / "Merlin/StateEffects.lean")["sha256"] != STATE_EFFECTS_SHA:
        raise ValueError("shared StateEffects source differs from the coordinated immutable intake")
    output = work / "compiled"
    (output / "Merlin").mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["LEAN_PATH"] = str(output)
    modules = ("StateEffects", "GraphRegions", "GraphRegionsFusion") if fusion else ("StateEffects", "GraphRegions")
    for module in modules:
        _run(
            [
                str(lean),
                f"--root={sources}",
                "-o",
                str(output / f"Merlin/{module}.olean"),
                str(sources / f"Merlin/{module}.lean"),
            ],
            work,
            "build-" + module,
            environment,
        )
    identity = {
        "lean": file_ref(lean),
        "sources": {m: file_ref(sources / f"Merlin/{m}.lean") for m in modules},
        "objects": {m: file_ref(output / f"Merlin/{m}.olean") for m in modules},
        "host_sources": {
            m: file_ref(root / f"src/merlin/verify/{m}.py")
            for m in ("lean_graph_regions", "lean_graph_regions_projection")
        },
        "LeanMLIR_route": "unavailable: no explicitly selected matching manifest",
        "axiom_policy": sorted(ALLOWED_AXIOMS),
        "lean_arguments": ["-j1"],
        "toolchain_dependencies": dependencies.inventory(lean, work / "toolchain-dependencies.json", file_ref),
    }
    identity["host_sources"]["lean_graph_regions_dependencies"] = file_ref(Path(dependencies.__file__))
    return lean, environment, identity


def verify_plan(
    source: Path, proposed: Path, selection: list[int], destination: Path, *, retained_source: Path
) -> dict:
    """Reparse actual source, then prove the recomputed plan verdict by kernel reduction."""
    destination.mkdir(parents=True, exist_ok=False)
    originals, snapshots = [], []
    for path, name in ((source, "source.mlir"), (retained_source, "retained-source.mlir"), (proposed, "plan.json")):
        original = file_ref(path)
        target = destination / name
        target.write_bytes(path.read_bytes())
        snapshot = file_ref(target)
        if (original["sha256"], original["bytes"]) != (snapshot["sha256"], snapshot["bytes"]):
            raise ValueError("original/snapshot content identity differs")
        originals.append(original)
        snapshots.append(snapshot)
    module = projection.parse_source((destination / "source.mlir").read_text())
    projected, _, _ = projection.project(module)
    retained = projection.parse_source((destination / "retained-source.mlir").read_text())
    after, _, _ = projection.project(retained)
    nodes = projected["nodes"]
    if (
        type(selection) is not list
        or not 2 <= len(selection) <= 64
        or len(set(selection)) != len(selection)
        or any(type(v) is not int or v < 0 for v in selection)
    ):
        raise ValueError("explicit bounded ordered source region required")
    region = [n for n in nodes if n["ordinal"] in selection]
    if [n["ordinal"] for n in region] != selection:
        raise ValueError("selection differs from current ordered source nodes")
    start = next(i for i, node in enumerate(nodes) if node["ordinal"] == selection[0])
    plan = json.loads((destination / "plan.json").read_text())
    contexts = [projection.ir_text(m) for m in (module, retained)]
    if any(not text.isascii() for text in contexts):
        raise ValueError("initial graph evidence supports ASCII IR rendering")
    text = (
        "import Merlin.GraphRegions\nopen Merlin.GraphRegions\n"
        + "def actualBefore : Graph := ⟨["
        + ",".join(_node(n) for n in nodes)
        + "],"
        + _nats(projected["functionArguments"])
        + ","
        + _nats(projected["observations"])
        + ","
        + _strings([contexts[0][i : i + 64] for i in range(0, len(contexts[0]), 64)])
        + "⟩\n"
        + "def actualAfter : Graph := ⟨["
        + ",".join(_node(n) for n in after["nodes"])
        + "],"
        + _nats(after["functionArguments"])
        + ","
        + _nats(after["observations"])
        + ","
        + _strings([contexts[1][i : i + 64] for i in range(0, len(contexts[1]), 64)])
        + "⟩\n"
        + "def actualPlan : Plan := "
        + _plan(plan)
        + "\n"
    )
    check = f"checkBoundary actualBefore actualAfter {start} {len(selection)} {_nats(selection)} actualPlan"
    lean, environment, identity = build(destination)
    query = destination / "query.lean"
    query.write_text(text + "#reduce " + check + "\n")
    reduced = _run([str(lean), str(query)], destination, "kernel-reduce", environment).stdout.strip()
    if reduced not in ("true", "false"):
        raise ValueError("kernel reduction did not produce one Boolean verdict")
    proof = destination / "proof.lean"
    proof.write_text(
        text
        + f"theorem actualVerdict : {check} = {reduced} := by decide\n"
        + "#print axioms actualVerdict\n"
        + "".join(f"#print axioms {name}\n" for name in ROOTS)
    )
    answer = _run([str(lean), str(proof)], destination, "kernel-proof", environment)
    # Reject any transitive axiom outside std3. The exact raw audit output remains.
    for block in answer.stdout.split("depends on axioms:")[1:]:
        content = block.partition("[")[2].partition("]")[0]
        axioms = [name.strip() for name in content.split(",") if name.strip()]
        if not set(axioms) <= ALLOWED_AXIOMS:
            raise ValueError("unapproved transitive theorem axiom")
    if (
        answer.stdout.count("depends on axioms:") + answer.stdout.count("does not depend on any axioms")
        != len(ROOTS) + 1
    ):
        raise ValueError("incomplete transitive theorem audit output")
    for original, snapshot in zip(originals, snapshots, strict=True):
        left, right = fresh_ref(original), fresh_ref(snapshot)
        if left.read_bytes() != right.read_bytes():
            raise ValueError("original/snapshot actual bytes differ")
    for records in (identity["sources"], identity["objects"], identity["host_sources"]):
        for record in records.values():
            fresh_ref(record)
    result = {
        "schema": "merlin.graph-regions.kernel-plan-evidence.v1",
        "accepted": reduced == "true",
        "claim": "ordered plan/source execution correspondence under explicit typed total pure Action semantics",
        "identity": identity,
        "source_original": originals[0],
        "retained_source_original": originals[1],
        "plan_original": originals[2],
        "snapshots": snapshots,
        "selection": selection,
        "projection": projected,
        "after_projection": after,
        "theorem": "Merlin.GraphRegions.checkBoundary_sound",
        "existing_anchor": "Merlin.StateEffects.Action/execute",
        "kernel_proof": file_ref(proof),
        "kernel_audit": file_ref(destination / "kernel-proof.stdout"),
        "projection_trusted_python": True,
        "source_to_LeanMLIR_operational_bridge_proved": False,
        "supported_realization_composition_proved": False,
        "shared_pipeline_acceptance_issued": False,
        "native_or_hardware_fidelity_proved": False,
        "approved_pass_verification": False,
    }
    (destination / "EVIDENCE.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def validate_bindings(receipt: dict) -> None:
    """Check every original/snapshot association with strict refs and actual bytes."""
    originals = [receipt.get(k) for k in ("source_original", "retained_source_original", "plan_original")]
    snapshots = receipt.get("snapshots")
    if type(snapshots) is not list or len(snapshots) != len(originals):
        raise ValueError("complete before/after/plan snapshot inventory required")
    for original, snapshot in zip(originals, snapshots, strict=True):
        left, right = fresh_ref(original), fresh_ref(snapshot)
        if (original["sha256"], original["bytes"]) != (snapshot["sha256"], snapshot["bytes"]):
            raise ValueError("original/snapshot content identity differs")
        if left.read_bytes() != right.read_bytes():
            raise ValueError("original/snapshot actual bytes differ")
        fresh_ref(original)
        fresh_ref(snapshot)
    for key in ("kernel_proof", "kernel_audit"):
        fresh_ref(receipt.get(key))


def content_identity(value):
    """Normalize location only after validating every embedded strict file ref."""
    if type(value) is dict:
        if set(value) == {"path", "sha256", "bytes"}:
            fresh_ref(value)
            return {"sha256": value["sha256"], "bytes": value["bytes"]}
        return {key: content_identity(item) for key, item in value.items()}
    if type(value) is list:
        return [content_identity(item) for item in value]
    return value


def replay_plan(receipt: dict, destination: Path | None = None) -> dict:
    """Rebuild and reprove from fresh actual artifacts; saved status grants no authority."""
    expected = {
        "schema": "merlin.graph-regions.kernel-plan-evidence.v1",
        "theorem": "Merlin.GraphRegions.checkBoundary_sound",
        "existing_anchor": "Merlin.StateEffects.Action/execute",
        "projection_trusted_python": True,
        "source_to_LeanMLIR_operational_bridge_proved": False,
        "supported_realization_composition_proved": False,
        "shared_pipeline_acceptance_issued": False,
        "native_or_hardware_fidelity_proved": False,
        "approved_pass_verification": False,
        "claim": "ordered plan/source execution correspondence under explicit typed total pure Action semantics",
    }
    if any(type(receipt.get(k)) is not type(v) or receipt[k] != v for k, v in expected.items()):
        raise ValueError("changed scoped graph claim")
    if type(receipt.get("accepted")) is not bool:
        raise ValueError("Boolean verdict required")
    validate_bindings(receipt)
    identity = receipt.get("identity", {})
    dependencies.validate(identity.get("toolchain_dependencies"), fresh_ref)
    fresh_ref(identity.get("lean"))
    if identity["lean"]["sha256"] != LEAN_SHA or identity.get("axiom_policy") != sorted(ALLOWED_AXIOMS):
        raise ValueError("changed selected tool/axiom policy")
    if identity.get("lean_arguments") != ["-j1"]:
        raise ValueError("changed selected Lean execution profile")
    for key in ("sources", "objects", "host_sources"):
        if type(identity.get(key)) is not dict or not identity[key]:
            raise ValueError("missing semantic/source identities")
        for record in identity[key].values():
            fresh_ref(record)
    source = fresh_ref(receipt["source_original"])
    retained = fresh_ref(receipt["retained_source_original"])
    plan = fresh_ref(receipt["plan_original"])
    for path, key in ((source, "projection"), (retained, "after_projection")):
        actual, _, _ = projection.project(projection.parse_source(path.read_text()))
        if actual != receipt.get(key):
            raise ValueError("saved projection differs from actual complete IR")
    target = destination or Path(receipt["kernel_proof"]["path"]).parent / ("replay-" + uuid.uuid4().hex)
    repeated = verify_plan(source, plan, receipt["selection"], target, retained_source=retained)
    if repeated["accepted"] is not receipt["accepted"]:
        raise ValueError("kernel replay verdict differs")
    for key in ("kernel_proof", "kernel_audit"):
        if fresh_ref(receipt[key]).read_bytes() != fresh_ref(repeated[key]).read_bytes():
            raise ValueError("replayed kernel proof/audit bytes differ")
    if repeated["identity"]["toolchain_dependencies"]["sha256"] != identity["toolchain_dependencies"]["sha256"]:
        raise ValueError("replayed core/toolchain dependency differs")
    for key in ("sources", "objects", "host_sources"):
        if set(repeated["identity"][key]) != set(identity[key]):
            raise ValueError("replayed dependency domain differs")
        for name, record in identity[key].items():
            actual = repeated["identity"][key][name]
            if (actual["sha256"], actual["bytes"]) != (record["sha256"], record["bytes"]):
                raise ValueError("replayed semantic/tool identity differs")
    validate_bindings(receipt)
    return repeated
