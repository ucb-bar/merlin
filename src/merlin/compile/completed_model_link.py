"""Revalidate a completed link after an exact original input tree is restored.

The failed compiler receipt remains failed. A separate record permits transport
of its already completed objects; it supplies no live effects, new rewriting,
execution, numerical verdict or instruction admission.
"""

from __future__ import annotations

import json
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.compile.model_execution_inputs import strict_tree_sha256
from merlin.llvmlower.compilation_recipe import verify_completed_recipe

SCHEMA = "merlin.completed_model_link_revalidation.v1"
_FAILURE = "BaremetalModelError: selected device package changed during compilation/execution"


def _pin(path):
    path = Path(path)
    if (not path.is_absolute() or path.resolve() != path
            or any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file()):
        raise ValueError("completed link needs original direct file inputs")
    return dict(path=str(path), sha256=sha256_file(path), bytes=path.stat().st_size)


def _reopen(record):
    actual = _pin(record["path"])
    if actual["sha256"] != record["sha256"] or ("bytes" in record and actual["bytes"] != record["bytes"]):
        raise ValueError("completed link original file changed")
    return Path(actual["path"])


def _reopen_nested(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            _reopen(value)
        for child in value.values():
            _reopen_nested(child)
    elif isinstance(value, list):
        for child in value:
            _reopen_nested(child)


def _original_policy(selected, receipt):
    from merlin.llvmlower.host_graph_policy import read_host_graph_policy

    software = _reopen(selected["software_spec"])
    if selected["target"] != receipt["inputs"]["target"]:
        raise ValueError("completed link permissions name another selected target")
    choices = {k: selected[k] for k in (
        "enabled", "minimum_transpose_bytes_saved", "maximum_added_operations", "physical_layout",
        "maximum_layout_assignments", "maximum_layout_permutations", "connected_layout")}
    if "column_reduction_layout" in selected:
        choices["column_reduction_layout"] = selected["column_reduction_layout"]
    policy = read_host_graph_policy(software, target=selected["target"], entry=selected["context"]["entry"], **choices)
    policy.revalidate()
    if json.loads(json.dumps(policy.record())) != selected:
        raise ValueError("completed link selected original permissions changed")
    return policy


def _layout_permissions(receipt, recipe):
    """Replay the original pure source transform; restore no runtime effects."""
    selected = receipt["inputs"]["device"].get("pre_group_layout")
    products = receipt["output"].get("pre_group_layout")
    if selected is None:
        if products is not None:
            raise ValueError("completed link layout has no original selected permissions")
        return None
    if not isinstance(products, dict) or not products:
        raise ValueError("completed link lacks original layout replay products")
    policy = _original_policy(selected, receipt)
    for role, record in products.items():
        expected = recipe.get("preparation", {}).get("pre_group_layout_" + role)
        if expected is None or any(expected[k] != record[k] for k in ("path", "sha256")):
            raise ValueError("completed link layout replay differs from its completed recipe")
        _reopen(record)
    from merlin.llvmlower.graph_optimizations import recheck_host_graph_artifacts

    root = Path(receipt["output"]["elf"]).parent / "pre_group_layout"
    if recheck_host_graph_artifacts(root, policy) != products:
        raise ValueError("completed link original layout source or pure replay changed")
    return selected


def _storage_permissions(receipt, recipe):
    """Read permissions and the original successful, recipe-bound replay products.

    This consumes immutable compiled artifacts. It never reconstructs borrowed
    lifetime owners or invokes the allocator from serialized effects.
    """
    selected = receipt["inputs"]["device"].get("intermediate_storage")
    products = receipt["output"].get("intermediate_storage")
    if selected is None:
        if products is not None:
            raise ValueError("completed link storage has no original selected permissions")
        return None
    policy = _original_policy(selected, receipt)
    software = policy.software_path
    if not isinstance(products, dict):
        raise ValueError("completed link selected storage permissions changed")
    if set(products) != {"input", "selected", "report", "software_spec", "policy"}:
        raise ValueError("completed link lacks original storage replay products")
    for role, record in products.items():
        expected = recipe.get("preparation", {}).get("intermediate_storage_" + role)
        if expected is None or any(expected[k] != record[k] for k in ("path", "sha256")):
            raise ValueError("completed link storage replay differs from its completed recipe")
        _reopen(record)
    if json.loads(_reopen(products["policy"]).read_text()) != selected:
        raise ValueError("completed link stored permission selection changed")
    if _reopen(products["software_spec"]).read_bytes() != software.read_bytes():
        raise ValueError("completed link stored software permissions changed")
    return selected


def _check(failed_pin):
    receipt = json.loads(_reopen(failed_pin).read_text())
    inputs, output = receipt.get("inputs", {}), receipt.get("output", {})
    if (receipt.get("schema") != "merlin.baremetal-saved-model.v1" or receipt.get("status") != "failed"
            or receipt.get("failure") != _FAILURE or inputs.get("run") != "none"
            or not isinstance(inputs.get("device"), dict)):
        raise ValueError("completed link revalidation requires the precise final package-tree refusal")
    unsupported = {"source_optimizations", "source_recomputation", "shared_source_lowerings", "materialized_inputs",
                   "readonly_input_indices", "graph_optimizations", "pre_group_layout", "op_profile", "group_profile"}
    if any(inputs.get(key) for key in unsupported) or any(key in output for key in (
        "source_optimizations", "source_recomputation", "host_graph_optimization",
        "shared_source_preparation", "op_profile_table", "group_profile", "console", "outputs_npz")):
        raise ValueError("completed link has deferred checks outside the supported transport scope")
    device = inputs["device"]
    trees = {}
    for role, path, expected in (
        ("capture", inputs["capture"], inputs["capture_tree"]),
        ("host_package", inputs["package"], inputs["package_tree"]),
        ("device_package", device["package"], device["package_tree"]),
    ):
        actual = strict_tree_sha256(Path(path))
        if actual != expected:
            raise ValueError("completed link original input tree has not been restored exactly")
        trees[role] = dict(path=path, **actual)
    for path_key in ("board_catalog", "dts"):
        _reopen(dict(path=inputs[path_key], sha256=inputs[path_key + "_sha256"]))
    elf = _reopen(dict(path=output["elf"], sha256=output["elf_sha256"]))
    recipe_pin = output["compilation_recipe"]
    recipe = verify_completed_recipe(_reopen(recipe_pin), executable=elf,
                                     expected_recipe_sha256=recipe_pin["sha256"])
    _reopen_nested(output)
    permissions = _storage_permissions(receipt, recipe)
    layout_permissions = _layout_permissions(receipt, recipe)
    return receipt, recipe, dict(restored_input_trees=trees, executable=_pin(elf),
                                recipe=recipe_pin, selected_storage_permissions=permissions,
                                **({"selected_layout_permissions": layout_permissions}
                                   if layout_permissions is not None else {}))


def finalize(failed_receipt, destination, *, restoration_evidence):
    """Write a separate completed-object transport record after exact revalidation."""
    failed_pin, evidence = _pin(failed_receipt), _pin(restoration_evidence)
    _, _, checked = _check(failed_pin)
    record = dict(schema=SCHEMA, status="completed_link_inputs_revalidated", failed_receipt=failed_pin,
                  restoration_evidence=evidence, implementation=_pin(Path(__file__).resolve()), **checked,
                  scope="completed object transport only; original compiler failure retained; no live effects restored")
    destination = Path(destination)
    with destination.open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    return record


def verify(record_pin):
    """Reopen all original inputs/products again before ordinary object transport."""
    record = json.loads(_reopen(record_pin).read_text())
    if record.get("schema") != SCHEMA or record.get("status") != "completed_link_inputs_revalidated":
        raise ValueError("not a completed link revalidation record")
    _reopen(record["restoration_evidence"])
    if record["implementation"] != _pin(Path(__file__).resolve()):
        raise ValueError("completed link revalidation implementation changed")
    receipt, recipe, checked = _check(record["failed_receipt"])
    if any(record.get(key) != value for key, value in checked.items()):
        raise ValueError("completed link original inputs or products changed")
    return receipt, recipe, [record[role] for role in ("failed_receipt", "restoration_evidence", "implementation")]
