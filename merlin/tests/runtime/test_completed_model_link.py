"""Exact input restoration permits completed-object transport, retaining failure."""

import json
import subprocess
from pathlib import Path

import pytest

from merlin.compile import completed_model_link as C
from merlin.compile.model_execution_inputs import strict_tree_sha256
from merlin.llvmlower.compilation_recipe import CompilationRecipe
from merlin.model_check.chain_image_objects import completed_program


@pytest.fixture
def original(tmp_path, request):
    roots = {}
    for name in ("capture", "host", "device"):
        root = tmp_path / name
        root.mkdir()
        (root / "source.py").write_text("# original " + name)
        roots[name] = root
    build = tmp_path / "build"
    build.mkdir()
    tool, producer, source = (build / p for p in ("tool", "producer.py", "source.c"))
    tool.write_text("#!/bin/sh\nexit 0\n")
    tool.chmod(0o755)
    producer.write_text("# actual producer\n")
    source.write_text("int value(void) { return 7; }\n")
    recipe = CompilationRecipe(build, producer=producer)
    storage, layout = None, None
    if getattr(request, "param", False) is True:
        from merlin.llvmlower.host_graph_policy import read_host_graph_policy
        from merlin.llvmlower.host_optimization_request import POLICY_SCHEMA
        from merlin.llvmlower.host_storage_policy import prepare, recheck

        software = tmp_path / "software.json"
        software.write_text(json.dumps(dict(schema="merlin.software_spec.v1", status="reviewed", target="neutral",
            host_optimization=dict(schema=POLICY_SCHEMA, permitted_stages=["llvm"],
                permitted_optimizations=["static_arena"], sequential_execution=True,
                successful_allocations=True, non_reentrant_entry=True))))
        policy = read_host_graph_policy(software, target="neutral", entry="forward")
        ll = build / "input.ll"
        ll.write_text("define void @forward() {\n  ret void\n}\n")
        prepare(ll, policy, build / "storage", recipe)
        storage = policy.record(), recheck(build / "storage", policy)
    if getattr(request, "param", None) == "layout":
        from merlin.frontends.linalg_mlir import parse_mlir_text
        from merlin.llvmlower.graph_optimizations import recheck_host_graph_artifacts
        from merlin.llvmlower.host_graph_policy import prepare_group_layout, read_host_graph_policy
        from merlin.llvmlower.host_optimization_request import POLICY_SCHEMA

        software = tmp_path / "software.json"
        software.write_text(json.dumps(dict(schema="merlin.software_spec.v1", status="reviewed", target="neutral",
            host_optimization=dict(schema=POLICY_SCHEMA, permitted_stages=["logical_host"],
                permitted_optimizations=["propagate_layout"], independent_element_order_unobserved=True))))
        policy = read_host_graph_policy(software, target="neutral", entry="forward", column_reduction_layout=True)
        module = parse_mlir_text("module {func.func @forward(%x:tensor<2x3xi8>)->tensor<2x3xi8> {return %x:tensor<2x3xi8>}}")
        prepare_group_layout(module, policy, build / "pre_group_layout")
        products = recheck_host_graph_artifacts(build / "pre_group_layout", policy)
        for role, pin in products.items():
            recipe.bind_preparation("pre_group_layout_" + role, Path(pin["path"]))
        layout = policy.record(), products

    def emit(output, data):
        def run(argv):
            output.write_bytes(data)
            return subprocess.CompletedProcess(argv, 0)
        return run

    objects = [build / name for name in ("model.o", "model_call.o", "model_main.o", "weights_blob.o")]
    for obj in objects:
        recipe.run([tool, "-c", source, "-o", obj], runner=emit(obj, obj.name.encode()), inputs=[source], output=obj)
    elf = build / "model.elf"
    recipe.run([tool, *objects, "-o", elf], runner=emit(elf, b"actual completed link"), inputs=objects, output=elf)
    recipe.completed(elf)
    catalog, dts, sidecar = (tmp_path / name for name in ("catalog", "dts", "sidecar.json"))
    for path in (catalog, dts, sidecar):
        path.write_text("original")
    receipt = dict(schema="merlin.baremetal-saved-model.v1", status="failed", failure=C._FAILURE,
        inputs=dict(run="none", target="neutral", capture=str(roots["capture"]), capture_tree=strict_tree_sha256(roots["capture"]),
            package=str(roots["host"]), package_tree=strict_tree_sha256(roots["host"]),
            device=dict(package=str(roots["device"]), package_tree=strict_tree_sha256(roots["device"])),
            board_catalog=str(catalog), board_catalog_sha256=C._pin(catalog)["sha256"],
            dts=str(dts), dts_sha256=C._pin(dts)["sha256"]),
        output=dict(elf=str(elf), elf_sha256=C._pin(elf)["sha256"], compilation_recipe=C._pin(recipe.path),
            device_sidecar=C._pin(sidecar), host_artifacts=dict(object=C._pin(objects[0]))))
    if storage is not None:
        receipt["inputs"]["device"]["intermediate_storage"] = storage[0]
        receipt["output"]["intermediate_storage"] = storage[1]
    if layout is not None:
        receipt["inputs"]["device"]["pre_group_layout"] = layout[0]
        receipt["output"]["pre_group_layout"] = layout[1]
    receipt = json.loads(json.dumps(receipt))
    failed, evidence, recovery = (tmp_path / name for name in ("failed.json", "restoration.json", "recovery.json"))
    failed.write_text(json.dumps(receipt))
    evidence.write_text('{"extra_generated_artifacts_preserved":true}')
    return dict(failed=failed, evidence=evidence, recovery=recovery, roots=roots, objects=objects, elf=elf,
                source=source, sidecar=sidecar, recipe=recipe.path, receipt=receipt)


def finalize(original):
    return C.finalize(original["failed"], original["recovery"], restoration_evidence=original["evidence"])


def test_restored_link_uses_original_recipe_and_preserves_failed_record(original):
    before = original["failed"].read_bytes()
    record = finalize(original)
    assert record["status"] == "completed_link_inputs_revalidated"
    program = completed_program(C._pin(original["recovery"]))
    assert program["receipt"]["status"] == "failed"
    assert program["receipt"]["failure"] == C._FAILURE
    assert set(program["by_name"]) == {p.name for p in original["objects"]}
    assert len(program["completed_link_revalidation_pins"]) == 3
    assert original["failed"].read_bytes() == before
    with pytest.raises(ValueError, match="completed ordinary"):
        completed_program(C._pin(original["failed"]))


@pytest.mark.parametrize("member", ["device", "capture", "host", "elf", "objects", "source", "sidecar", "recipe", "evidence"])
def test_revalidation_reopens_every_original_tree_product_and_evidence(original, member):
    finalize(original)
    if member in original["roots"]:
        path = original["roots"][member] / "new-output"
    elif member == "objects":
        path = original[member][0]
    else:
        path = original[member]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        C.verify(C._pin(original["recovery"]))


@pytest.mark.parametrize("change", ["source", "failure", "run", "source_optimizations", "pre_group_layout"])
def test_unrestored_or_unsupported_finalization_refuses(original, change):
    receipt = original["receipt"]
    if change == "source":
        (original["roots"]["device"] / "source.py").write_text("changed executable input")
    elif change == "failure":
        receipt["failure"] = "unrelated failure"
    elif change == "run":
        receipt["inputs"]["run"] = "native"
    elif change == "pre_group_layout":
        receipt["inputs"]["device"][change] = {"unreplayed": True}
    else:
        receipt["inputs"][change] = {"unreplayed": True}
    original["failed"].write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        finalize(original)
    assert not original["recovery"].exists()


@pytest.mark.parametrize("original", [True], indirect=True)
def test_original_storage_permissions_and_products_are_read_without_restoring_effects(original, monkeypatch):
    from merlin.llvmlower import host_storage_policy

    def no_new_allocator(*args, **kwargs):
        raise AssertionError("completed transport must never reconstruct effects or rerun allocation")

    monkeypatch.setattr(host_storage_policy, "_optimize", no_new_allocator)
    record = finalize(original)
    assert record["selected_storage_permissions"] == original["receipt"]["inputs"]["device"]["intermediate_storage"]
    C.verify(C._pin(original["recovery"]))
    software = Path(record["selected_storage_permissions"]["software_spec"]["path"])
    software.write_text(software.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        C.verify(C._pin(original["recovery"]))


@pytest.mark.parametrize("original", ["layout"], indirect=True)
def test_original_layout_replays_complete_pure_source_and_preserves_failure(original):
    before = original["failed"].read_bytes()
    record = finalize(original)
    assert record["selected_layout_permissions"] == original["receipt"]["inputs"]["device"]["pre_group_layout"]
    assert record["selected_layout_permissions"]["column_reduction_layout"] is True
    assert completed_program(C._pin(original["recovery"]))["receipt"]["status"] == "failed"
    assert original["failed"].read_bytes() == before


@pytest.mark.parametrize("original", ["layout"], indirect=True)
def test_layout_requires_live_original_pure_replay(original, monkeypatch):
    from merlin.llvmlower.host_graph_policy import HostGraphPolicy

    original_optimize = HostGraphPolicy.optimize

    def changed_source(self, source):
        selected, report = original_optimize(self, source)
        return selected + "\n// changed current transformation\n", report

    monkeypatch.setattr(HostGraphPolicy, "optimize", changed_source)
    with pytest.raises(ValueError, match="selection changed"):
        finalize(original)


@pytest.mark.parametrize("original", ["layout"], indirect=True)
def test_layout_products_must_be_bound_by_original_completed_recipe(original):
    receipt = original["receipt"]
    receipt["output"]["pre_group_layout"]["unobserved"] = C._pin(original["source"])
    original["failed"].write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="completed recipe"):
        finalize(original)
