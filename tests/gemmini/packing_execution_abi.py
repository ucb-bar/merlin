"""Reopen the selected descriptor bridge from actual ordinary model products."""

import json
from pathlib import Path

from packing_execution_inputs import actual_observation, pin

from merlin.common import invocation_record
from merlin.llvmlower.compilation_recipe import verify_completed_recipe
from merlin.llvmlower.device_offload import SIDECAR_NAME


def descriptor_abi(elf):
    work = elf.parent
    report_path, recipe_path = work / "lower/descriptor-result-identity.json", work / "lower/lowering_recipe.json"
    report, lowering = json.loads(report_path.read_bytes()), json.loads(recipe_path.read_bytes())
    assert report["schema"] == "merlin.descriptor_result_preparation.v1"
    assert report["binding_stage"] == "after_automatic_source_passes_before_preprocessing"
    sidecar = json.loads((work / SIDECAR_NAME).read_bytes())
    assert report["rewrites"] == sum(row["calls"] for row in report["calls"]) == len(sidecar["routed"])
    assert lowering["descriptor_result_identity"]["report"]["sha256"] == pin(report_path)["sha256"]
    shim_path, shim = actual_observation(work / "device", "device_shim_compilation")
    invocation_record.verify(shim_path)
    source, compiled = work / "device/device_shim.c", work / "device/device_shim.o"
    effects_path = work / "device/descriptor-result-effects.json"
    effects = json.loads(effects_path.read_bytes())
    assert effects["schema"] == "merlin.borrowed_buffer_calls.v1"
    selected_effects = {row["wrapper_symbol"]: row for row in effects["calls"]}
    assert set(selected_effects) == {row["wrapper_symbol"] for row in report["selection"]}
    for row in report["selection"]:
        call = selected_effects[row["wrapper_symbol"]]
        assert (call["borrowed_symbol"], call["result_argument"], call["effects"]) == (
            row["borrowed_symbol"],
            row["result_argument"],
            row["borrowed_effects"],
        )
        assert all(
            call["effects"][key] is True for key in ("no_retained_aliases", "no_releases", "complete_before_return")
        )
    assert shim["inputs"] == [pin(source)] and shim["outputs"] == [pin(compiled)]
    assert source.read_text().count("/* merlin.descriptor_result_identities.v1 ") == 1
    llvm = (work / "lower/model.ll").read_text()
    for row in report["calls"]:
        assert "call void @_mlir_ciface_" + row["borrowed_symbol"] + "(" in llvm
        assert row["protected_empty_roots"] == row["calls"]
    compilation_path = work / "compilation_recipe.json"
    compilation_pin = pin(compilation_path)
    compilation = verify_completed_recipe(
        compilation_path, executable=elf, expected_recipe_sha256=compilation_pin["sha256"]
    )
    bound = {row["path"]: row["sha256"] for row in compilation["preparation"].values()}
    paths = [
        report_path,
        recipe_path,
        source,
        compiled,
        shim_path,
        effects_path,
        *(Path(report[name]["path"]) for name in ("original_source", "selected_source")),
    ]
    assert all(bound[str(path.resolve())] == pin(path)["sha256"] for path in paths)
    assert all(bound[row["path"]] == row["sha256"] for row in effects["original_pins"])
    return {
        "report": report,
        "artifacts": [pin(path) for path in (*paths, compilation_path)],
        "shim_invocation": pin(shim_path),
        "compilation_recipe": compilation_pin,
    }


def recheck_descriptor_abi(call):
    """Reopen earlier products after every later original callback has finished."""
    record = call["descriptor_abi"]
    recipe, elf = record["compilation_recipe"], call["elf"]
    verify_completed_recipe(Path(recipe["path"]), executable=Path(elf["path"]), expected_recipe_sha256=recipe["sha256"])
    receipt = record["shim_invocation"]
    assert pin(Path(receipt["path"])) == receipt
    invocation_record.verify(receipt["path"])
