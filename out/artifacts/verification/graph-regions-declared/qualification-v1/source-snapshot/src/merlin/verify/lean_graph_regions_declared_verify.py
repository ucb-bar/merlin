"""Replayable actual-buffer declared-call evidence, separate from approval.

Every original must still exist and equal its snapshot bytes. Strict references
and all source/role associations are checked even after coherent metadata rehash.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.verify import lean_graph_regions as base
from merlin.verify import lean_graph_regions_dependencies as dependencies
from merlin.verify.lean_graph_regions_declared_check import save
from merlin.verify.lean_graph_regions_declared_packet import packet

ROOTS = [
    "actual_kernel_decoded",
    "actual_calls_valid",
    "actual_generic_binding",
    "actual_complete_observers",
    "actual_nonzero_witness",
]


def audit(stdout, roots):
    found = {}
    for line in stdout.splitlines():
        if line.endswith(" does not depend on any axioms"):
            name = line.removesuffix(" does not depend on any axioms").strip("'")
            if name in found:
                raise ValueError("duplicate exact-root axiom output")
            found[name] = []
            continue
        if " depends on axioms: " not in line:
            continue
        name, _, payload = line.partition(" depends on axioms: ")
        name = name.strip("'")
        if name in found or not payload.startswith("[") or not payload.endswith("]"):
            raise ValueError("malformed or duplicate exact-root axiom output")
        axioms = [value.strip() for value in payload[1:-1].split(",") if value.strip()]
        if not set(axioms) <= base.ALLOWED_AXIOMS:
            raise ValueError("non-standard axiom dependency")
        found[name] = axioms
    if set(found) != set(roots):
        raise ValueError("missing or unexpected exact-root axiom audit")
    return found


def fresh_identity(identity):
    context = identity["context"]
    for ref in [
        context["fixed_identity"]["lean"],
        *context["fixed_syntax"].values(),
        *identity["project_objects"].values(),
        identity["builder"],
    ]:
        base.fresh_ref(ref)
    dependencies.validate(context["fixed_identity"]["toolchain_dependencies"], base.fresh_ref)
    for row in identity["actual_import_closure"].values():
        base.fresh_ref(row["source"])
        if "Std/Time/Format/Basic" in row["source"]["path"]:
            raise ValueError("affected dependency selected")


def host_refs(host):
    return [host["python"], host["configuration"], *host["modules"].values(), *host["mapped"].values()]


def oracle(mode, source, destination, *, python, reference=None, fixed=None):
    if fixed is not None:
        for ref in host_refs(fixed):
            base.fresh_ref(ref)
    worker = repo_root() / "src/merlin/verify/lean_graph_regions_declared_oracle.py"
    command = [str(python), "-B", str(worker), mode, str(source.resolve()), str(destination.resolve())]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if mode == "kernel":
        if reference is None:
            raise ValueError("pinned reference context required")
        environment["PYTHONPATH"] = str(reference / "src")
    else:
        environment.pop("PYTHONPATH", None)
        command += ["--exporter", str(repo_root() / "src/merlin/verify/lean/mlir_export.py")]
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=90)
    save(
        destination.with_suffix(".process.json"),
        dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr),
    )
    result.check_returncode()
    receipt = json.loads((destination / "ORACLE.json").read_text())
    if (
        receipt["source"] != base.file_ref(source)
        or receipt["worker"] != base.file_ref(worker)
        or receipt["valid"] is not True
    ):
        raise ValueError("oracle source/code association differs")
    if fixed is not None and receipt["host"] != fixed:
        raise ValueError("actual oracle host closure differs from fixed context")
    for ref in host_refs(receipt["host"]):
        base.fresh_ref(ref)
    return receipt


def capture_files(capture):
    """Full source/call/selected contracts, not just the expected proof inputs."""
    return {
        path.relative_to(capture).as_posix(): base.file_ref(path)
        for path in sorted(capture.rglob("*"))
        if path.is_file()
    }


def validate_associations(originals, snapshots):
    if (
        type(originals) is not dict
        or type(snapshots) is not dict
        or originals.keys() != snapshots.keys()
        or not originals
    ):
        raise ValueError("complete exact original/snapshot role map required")
    for role in originals:
        original = base.fresh_ref(originals[role])
        snapshot = base.fresh_ref(snapshots[role])
        if original.read_bytes() != snapshot.read_bytes():
            raise ValueError("original/snapshot actual content differs for " + role)


def original_capture_associations(originals):
    root = base.fresh_ref(originals["CAPTURE.json"]).parent
    if capture_files(root) != originals:
        raise ValueError("original role paths/contents are incomplete or substituted")
    capture = load_capture(root)
    for field, relative in (
        ("source", "source.mlir"),
        ("retained_source", "retained-source.mlir"),
        ("original_provider_kernel", "original-provider-kernel.json"),
        ("host_intake", "HOST-INTAKE.json"),
    ):
        if capture[field] != originals[relative]:
            raise ValueError("capture-to-original content association differs")
    base.fresh_ref(capture["generator"])
    expected_generator = repo_root() / "src/merlin/verify/lean_graph_regions_declared_capture.py"
    if capture["generator"] != base.file_ref(expected_generator):
        raise ValueError("capture generator identity differs")
    host = json.loads(base.fresh_ref(originals["HOST-INTAKE.json"]).read_text())
    for ref in [
        host["python"],
        host["configuration"],
        *host["loaded_modules"].values(),
        *host["mapped_host_files"].values(),
    ]:
        base.fresh_ref(ref)
    for row in capture["stages"]:
        for field, suffix in (
            ("scalar_source", "source"),
            ("adapter_declaration", "adapter"),
            ("selected_declaration", "selected"),
        ):
            if row[field] != originals[f"stage-{row['index']}-{suffix}.mlir"]:
                raise ValueError("capture scalar declaration/source roles were substituted")


def verify(capture: Path, destination: Path, *, identity_file: Path, fixed_oracles: dict):
    destination.mkdir(parents=True, exist_ok=False)
    identity_ref = base.file_ref(identity_file)
    identity = json.loads(base.fresh_ref(identity_ref).read_text())
    fresh_identity(identity)
    originals = capture_files(capture)
    original_capture_associations(originals)
    snapshot = destination / "capture"
    snapshot.mkdir()
    for relative, ref in originals.items():
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base.fresh_ref(ref).read_bytes())
    snapshots = capture_files(snapshot)
    validate_associations(originals, snapshots)
    selected = identity["context"]
    roles = {"before": "source.mlir", "retained": "retained-source.mlir"} | {
        f"stage-{index}-{role}": f"stage-{index}-{role}.mlir"
        for index in range(3)
        for role in ("source", "adapter", "selected")
    }
    documents, exports, oracles = {}, {}, {}
    for role, relative in roles.items():
        target = destination / (role + "-oracle")
        oracles[role] = oracle(
            "mlir",
            snapshot / relative,
            target,
            python=Path(selected["oracle_invocation"]),
            fixed=fixed_oracles["mlir-full" if role in {"before", "retained"} else "mlir-scalar"],
        )
        documents[role] = json.loads((target / "export.json").read_text())
        exports[role] = base.file_ref(target / "export.json")
        if documents[role]["source"] != {key: snapshots[relative][key] for key in ("sha256", "bytes")}:
            raise ValueError("authoritative export does not bind its actual source")
    oracles["kernel"] = oracle(
        "kernel",
        snapshot / "actual-output-kernel.json",
        destination / "kernel-oracle",
        python=Path(sys.executable),
        reference=Path(load_capture(snapshot)["reference_root"]),
        fixed=fixed_oracles["kernel"],
    )
    text, model = packet(snapshot, documents)
    proof = destination / "Check.lean"
    proof.write_text(text)
    save(destination / "MODEL.json", model)
    compiled = identity_file.parent / "compiled"
    environment = os.environ | {"LEAN_PATH": str(compiled.resolve())}
    command = [selected["fixed_identity"]["lean"]["path"], "-j1", str(proof.resolve())]
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=90)
    save(
        destination / "process.json",
        dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr),
    )
    result.check_returncode()
    checked = audit(result.stdout, ROOTS)
    save(destination / "AUDIT.json", checked)
    validate_associations(originals, snapshots)
    fresh_identity(identity)
    receipt = dict(
        schema="merlin.graph-regions.declared-boundary.v1",
        originals=originals,
        snapshots=snapshots,
        identity=identity_ref,
        oracles=oracles,
        exports=exports,
        fixed_oracles=fixed_oracles,
        proof=base.file_ref(proof),
        model=base.file_ref(destination / "MODEL.json"),
        process=base.file_ref(destination / "process.json"),
        audit=base.file_ref(destination / "AUDIT.json"),
        retention_policy="all",
        qualification="development; separate frozen roster required",
        source_operational_bridge="UNQUALIFIED; not imported",
        independently_reviewed=False,
        primitive_fidelity_proved=False,
        native_execution_run=False,
        full_model_verified=False,
    )
    save(destination / "EVIDENCE.json", receipt)
    return receipt


def load_capture(path):
    return json.loads((path / "CAPTURE.json").read_text())


def validate(receipt):
    if receipt["schema"] != "merlin.graph-regions.declared-boundary.v1" or receipt["retention_policy"] != "all":
        raise ValueError("wrong receipt schema or retention policy")
    validate_associations(receipt["originals"], receipt["snapshots"])
    original_capture_associations(receipt["originals"])
    identity = json.loads(base.fresh_ref(receipt["identity"]).read_text())
    fresh_identity(identity)
    for field in ("proof", "model", "process", "audit"):
        base.fresh_ref(receipt[field])
    process = json.loads(Path(receipt["process"]["path"]).read_text())
    expected_command = [identity["context"]["fixed_identity"]["lean"]["path"], "-j1", receipt["proof"]["path"]]
    if process["command"] != expected_command:
        raise ValueError("Lean process proof/tool/content association differs")
    if process["returncode"] != 0 or audit(process["stdout"], ROOTS) != json.loads(
        Path(receipt["audit"]["path"]).read_text()
    ):
        raise ValueError("unaccepted or different proof audit")
    expected_roles = {"before", "retained", "kernel"} | {
        f"stage-{index}-{role}" for index in range(3) for role in ("source", "adapter", "selected")
    }
    if set(receipt["oracles"]) != expected_roles or set(receipt["exports"]) != expected_roles - {"kernel"}:
        raise ValueError("complete exact source/declaration/kernel oracle roles required")
    documents = {}
    for role, evidence in receipt["oracles"].items():
        expected = (
            "actual-output-kernel.json"
            if role == "kernel"
            else {"before": "source.mlir", "retained": "retained-source.mlir"}.get(role, role + ".mlir")
        )
        if evidence["source"] != receipt["snapshots"][expected] or evidence["valid"] is not True:
            raise ValueError("oracle actual source/role association differs")
        profile = "kernel" if role == "kernel" else "mlir-full" if role in {"before", "retained"} else "mlir-scalar"
        if evidence["host"] != receipt["fixed_oracles"][profile]:
            raise ValueError("oracle fixed host association differs")
        for ref in host_refs(evidence["host"]):
            base.fresh_ref(ref)
        if role != "kernel":
            document = json.loads(base.fresh_ref(receipt["exports"][role]).read_text())
            if document["source"] != {key: receipt["snapshots"][expected][key] for key in ("sha256", "bytes")}:
                raise ValueError("export/actual snapshot source association differs")
            documents[role] = document
    snapshot_root = Path(receipt["snapshots"]["CAPTURE.json"]["path"]).parent
    actual_snapshot_paths = capture_files(snapshot_root)
    if actual_snapshot_paths != receipt["snapshots"]:
        raise ValueError("snapshot paths/roles are incomplete or substituted")
    generated, model = packet(snapshot_root, documents)
    if (
        Path(receipt["proof"]["path"]).read_text() != generated
        or json.loads(Path(receipt["model"]["path"]).read_text()) != model
    ):
        raise ValueError("proof/model not bound to actual complete input bytes")
    return True


def replay(receipt, destination):
    """One actual fresh proof/oracle execution; never promote failed profiles."""
    validate(receipt)
    original_root = Path(receipt["originals"]["CAPTURE.json"]["path"]).parent
    current = verify(
        original_root,
        destination,
        identity_file=Path(receipt["identity"]["path"]),
        fixed_oracles=receipt["fixed_oracles"],
    )
    for field in ("originals", "identity", "fixed_oracles"):
        if current[field] != receipt[field]:
            raise ValueError("fresh replay selected different actual input/context")
    for field in ("proof", "model", "audit"):
        if base.fresh_ref(current[field]).read_bytes() != base.fresh_ref(receipt[field]).read_bytes():
            raise ValueError("fresh replay proof/model/audit content differs")
    validate(current)
    save(
        destination / "REPLAY.json",
        dict(
            original=base.file_ref(Path(receipt["proof"]["path"]).parent / "EVIDENCE.json"),
            current=base.file_ref(destination / "EVIDENCE.json"),
            identical_checked_content=True,
            native_execution_run=False,
            independently_reviewed=False,
        ),
    )
    return current
