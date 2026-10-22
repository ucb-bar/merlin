"""Reopen exact retained public layer roles without guessing by basename.

These are byte and producer joins only. Numerical, source equivalence, hardware
execution and timing qualification remain with their original independent owners.
"""

import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath

from merlin.common import invocation_record as I
from merlin.common.tree_hash import hash_tree
from merlin.targetgen import executed_program as E
from merlin.targetgen.contract.readback_policy import canonical_sha256


def _mapping(value):
    if not isinstance(value, dict):
        raise ValueError("original execution role must be a mapping")
    return value


def _preflight_retained(root):
    """Bound the complete namespace before the original collector reads payloads."""
    namespace, files, total, records, entries = E._root(root), {}, 0, [], 0
    stack = [(namespace, 0)]
    while stack:
        directory, depth = stack.pop()
        if directory.is_symlink() or directory.resolve(strict=True) != directory or depth > 2:
            raise ValueError("retained execution namespace is indirect or too deep")
        with os.scandir(directory) as children:
            for child in children:
                entries += 1
                if entries > 4096:
                    raise ValueError("retained execution namespace exceeds its entry budget")
                path, info = Path(child.path), child.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    stack.append((path, depth + 1))
                elif stat.S_ISREG(info.st_mode):
                    if info.st_nlink != 1 or info.st_size > 64 << 20:
                        raise ValueError("retained execution member is aliased or exceeds its byte budget")
                    total += info.st_size
                    files[path] = info.st_size
                    if total > 128 << 20:
                        raise ValueError("retained execution namespace exceeds its byte budget")
                    if (child.name == E.RECORD_NAME
                            or child.name.startswith("unretained_") and child.name.endswith(".json")):
                        records.append(path)
                else:
                    raise ValueError("retained execution namespace contains a linked or special member")
    if len(records) > 64:
        raise ValueError("retained execution record roster exceeds its budget")
    for path in records:
        if files[path] > 64 << 10:
            raise ValueError("retained execution record exceeds its metadata budget")
        with path.open("rb") as stream:
            raw = stream.read((64 << 10) + 1)
        if len(raw) > 64 << 10:
            raise ValueError("retained execution record exceeds its metadata budget")
        record = json.loads(raw)
        if not isinstance(record, dict):
            raise ValueError("retained execution record is not a mapping")
        members = _mapping(record.get("compiler_artifacts") or {}).get("members", {})
        if not isinstance(members, dict) or len(members) > 64:
            raise ValueError("retained execution role roster is malformed")
        for pin in [record.get("elf"), record.get("harness"), *members.values()]:
            if pin is None:
                continue
            if not isinstance(pin, dict) or not isinstance(pin.get("path"), str):
                raise ValueError("retained execution role pin is malformed")
            relative = PurePosixPath(pin["path"])
            target = root / relative
            if (relative.is_absolute() or ".." in relative.parts or str(relative) != pin["path"]
                    or target not in files):
                raise ValueError("retained execution member escapes its bounded namespace")


def reopen_execution_products(root, grade, *, compiler=None, candidate_sha256=None):
    return _reopen_execution_products(root, grade, compiler=compiler,
        candidate_sha256=candidate_sha256, tier_name="L2", kinds={"layer", "model_slice"})


def reopen_operator_tier_products(root, grade, *, tier):
    """Reopen a measured public operator tier; this alone issues no verdict."""
    if tier not in {"L2", "L3"}:
        raise ValueError("operator product reopening requires an explicit execution tier")
    return _reopen_execution_products(root, grade, tier_name=tier,
        kinds={"isa", "layer", "model_slice"})


def _reopen_execution_products(root, grade, *, compiler=None, candidate_sha256=None,
                              tier_name, kinds):
    from .candidate_observations import _reopen_product

    root = Path(root)
    if root.is_symlink() or root.resolve(strict=True) != root:
        raise ValueError("original execution product root is indirect")
    if not isinstance(grade, dict) or grade.get("kind") not in kinds or grade.get("label") not in {"public", "dev"}:
        raise ValueError("execution diagnostics require an original public layer or model-slice grade")
    source = _mapping(grade.get("compiler_source") or {})
    tier = _mapping(_mapping(grade.get("tiers") or {}).get(tier_name) or {})
    if not isinstance(source, dict) or not isinstance(tier, dict):
        raise ValueError("original source or functional role is not a mapping")
    simulator = tier.get("sim_provenance") or {}
    if not isinstance(simulator, dict) or not isinstance(source.get("artifacts"), dict):
        raise ValueError("original simulator or source role roster is not a mapping")
    if (source.get("status") != "stable" or tier.get("status") != "pass"
            or tier.get("measured_now") is not True or not simulator.get("engine") or not simulator.get("sha256")):
        raise ValueError("execution diagnostics require the current original functional role")
    if candidate_sha256 is not None and candidate_sha256 != source.get("candidate_sha256"):
        raise ValueError("execution diagnostics selected another compiler source")
    from merlin_experiments.phase2.contracts import StageGateError
    from merlin_experiments.phase2.retained_revisions import source_inventory

    def bounded_source_tree(path):
        try:
            # Bound the whole traversal, including .pyc files that hash_tree
            # reads and generated subtrees that its rglob still traverses.
            source_inventory(path, max_files=8192, max_bytes=32 << 20)
        except StageGateError as error:
            raise ValueError("original compiler source inventory is unavailable") from error
        return hash_tree(path)["sha256"]

    expected = source.get("candidate_sha256")
    if (not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected)
            or source.get("snapshot") != "compiler-source/" + expected):
        raise ValueError("original source snapshot identity is absent or malformed")
    snapshot = root / source["snapshot"]
    if bounded_source_tree(snapshot) != expected:
        raise ValueError("original compiler source snapshot changed")
    if compiler is not None:
        package = Path(compiler)
        if str(package) != source.get("package") or bounded_source_tree(package) != expected:
            raise ValueError("execution diagnostics lost the original compiler source")
    _preflight_retained(root)
    records = E.collect(root)
    if records != source.get("executed_programs"):
        raise ValueError("original executed program roster changed")
    selected = [row for row in records if row.get("schema") == E.SCHEMA_V2
                and row.get("status") == "retained" and row.get("engine") == simulator["engine"]
                and row.get("engine_sha256") == simulator["sha256"]]
    if len(selected) != 1:
        raise ValueError("original functional execution is absent or ambiguous")
    execution = selected[0]

    def member(pin):
        relative = PurePosixPath(pin["path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) != pin["path"]:
            raise ValueError("original execution member escapes its generated owner")
        return _reopen_product({**pin, "path": str(root / relative), "size_bytes": pin["bytes"]}, root=root)

    artifacts = execution["compiler_artifacts"]
    if not isinstance(artifacts, dict) or artifacts.get("status") != "retained":
        raise ValueError("original completed-call compiler artifacts are unavailable")
    members = artifacts["members"]
    if set(members) != {"build", "command_buffer", "console", "invocation", "kernel_object", "lowered_mlir"}:
        raise ValueError("original retained compiler role roster changed")
    paths = {name: member(pin) for name, pin in members.items()}
    elf, harness = member(execution["elf"]), member(execution["harness"])
    invocation = _reopen_product(artifacts["original_execution_invocation"])
    observed = I.verify(invocation)
    returned = (observed.get("returncode") == 0 if observed.get("kind") == "subprocess"
                else observed.get("outcome") == "returned" if observed.get("kind") == "python_call" else False)
    if (observed.get("stage") != "execution" or observed.get("status") != "completed"
            or not returned
            or not any(row.get("sha256") == execution["elf"]["sha256"] for row in observed.get("inputs", ()))):
        raise ValueError("original execution producer does not bind this executable")
    if paths["invocation"].read_bytes() != invocation.read_bytes():
        raise ValueError("retained execution invocation differs from its original producer")
    if members["console"]["sha256"] != observed["stdout"]["sha256"]:
        raise ValueError("retained console differs from its original execution producer")
    build = _mapping(json.loads(paths["build"].read_bytes()))
    command_buffer = _mapping(json.loads(paths["command_buffer"].read_bytes()))
    if (build.get("status") != "completed" or build.get("elf_sha256") != execution["elf"]["sha256"]
            or build.get("harness_sha256") != execution["harness"]["sha256"]
            or build.get("kernel_object_sha256") != members["kernel_object"]["sha256"]
            or build.get("command_buffer_sha256") != canonical_sha256(command_buffer)):
        raise ValueError("retained effective command buffer or linked program differs from its completed build")
    if (build.get("build_identity_sha256") != execution.get("build_identity_sha256")
            or build.get("build_identity_sha256") != canonical_sha256(
                {key: value for key, value in build.items() if key != "build_identity_sha256"})
            or build.get("readback_policy") != execution.get("readback_policy")):
        raise ValueError("retained completed-build identity or readback policy changed")
    source_pin = (source.get("artifacts") or {}).get("input.interface.mlir")
    lowered_pin = (source.get("artifacts") or {}).get("lowered.llvm.mlir")
    source_path = _reopen_product({"path": str(root / "input.interface.mlir"), "sha256": source_pin}, root=root)
    if lowered_pin is not None:
        original_lowered = _reopen_product(
            {"path": str(root / "lowered.llvm.mlir"), "sha256": lowered_pin}, root=root)
        if hashlib.sha256(paths["lowered_mlir"].read_bytes()).hexdigest() != lowered_pin:
            raise ValueError("retained lowered source differs from the original compiler artifacts")
        if original_lowered.read_bytes() != paths["lowered_mlir"].read_bytes():
            raise ValueError("retained lowered compiler role differs from the original pinned role")
    raw_cb_pin = source["artifacts"].get("command_buffer.json")
    raw_cb = _reopen_product({"path": str(root / "command_buffer.json"), "sha256": raw_cb_pin}, root=root)
    return {"source": source_path, "emitted": paths["lowered_mlir"], "compiler_command_buffer": raw_cb,
            "command_buffer": paths["command_buffer"], "object": paths["kernel_object"], "elf": elf,
            "caller": harness}, {
        "kind": "original_retained_functional_layer_products",
        "effective_command_buffer": "completed-call canonical buffer; raw compiler declarations remain separate",
        "source_sha256": source_pin, "elf_sha256": execution["elf"]["sha256"],
        "raw_compiler_command_buffer_sha256": raw_cb_pin,
        "effective_command_buffer_sha256": members["command_buffer"]["sha256"],
        "execution_invocation_sha256": artifacts["original_execution_invocation"]["sha256"],
        "qualification": "byte custody only; no numerical, source-equivalence, target or timing verdict",
    }
