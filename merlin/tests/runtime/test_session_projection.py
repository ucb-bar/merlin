"""Projected entry proofs must reach every session caller and logical state slot."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from test_entry_weight_projection import CONTRACT, MODEL, bundle

from merlin.common import mlir_query
from merlin.llvmlower import session_bundle
from merlin.llvmlower.entry_weight_projection import (
    apply_entry_weight_projection,
    module_text,
    plan_entry_weight_projection,
    prepare_entry_weight_projection,
    rename_projected_entry,
)
from merlin.llvmlower.session_projection import (
    ProgramEntryProjection,
    read_program_entry_projection,
)


def fixture(tmp_path):
    root = tmp_path / "session"
    selections = {}
    for name in ("prefill", "decode"):
        parent = root / "stages" / name
        parent.mkdir(parents=True)
        capture = bundle(parent, session=name == "decode")
        if name == "decode":
            path = capture / "session_contract.yaml"
            child = yaml.safe_load(path.read_text())
            child.update(stages=[name], steps=3)
            path.write_text(yaml.safe_dump(child))
        prepared = parent / "prepared"
        prepared.mkdir()
        source = prepared / "source.mlir"
        source.write_text(MODEL)
        _, plan = prepare_entry_weight_projection(source, prepared, capture, CONTRACT)
        selections[name] = read_program_entry_projection(
            model_dir=capture, prepared_dir=prepared, source=source,
            receipt=prepared / "entry_weight_projection/entry_weight_projection.json")
    contract = {
        "version": 2, "kind": "autoregressive_decode", "paper_ready": False,
        "stages": ["prefill", "decode"],
        "stage_schedule": [{"name": name, "steps": steps, "execution": "compiled", "timed": True}
                           for name, steps in (("prefill", 1), ("decode", 3))],
        "programs": [{"name": name, "bundle": f"stages/{name}/capture", "steps": steps}
                     for name, steps in (("prefill", 1), ("decode", 3))],
        "bindings": [{"name": "state", "from": {"program": "prefill", "output_index": 1},
                      "to": {"program": "decode", "input_arg": 3}}],
        "states": [{"name": "state"}], "streams": [],
        "quality": {"scope": "trajectory", "program": "decode"},
    }
    (root / "session_contract.yaml").write_text(yaml.safe_dump(contract))
    return root, selections


def test_namespaced_projection_keeps_exact_arguments_body_and_original_source(tmp_path):
    root, selections = fixture(tmp_path)
    selected = selections["prefill"]
    original = mlir_query.parse(selected.source)
    before = module_text(original)
    renamed, plan = rename_projected_entry(original, selected.projection, "renamed")
    assert module_text(original) == before
    assert plan.contract == replace(CONTRACT, entry_symbol="renamed")
    assert plan.original_to_projected == (None, 0, 1, 2)
    projected = apply_entry_weight_projection(renamed, plan)
    assert [op.sym_name.data for op in projected.body.block.ops if op.name == "func.func"] == ["renamed"]
    projected.verify()


def test_projected_session_generates_compact_layout_and_remaps_public_input_access(tmp_path):
    root, selections = fixture(tmp_path)
    output = tmp_path / "generated"
    info = session_bundle.generate(root, output, materialize_weights=False, entry_projections=selections)
    assert [row["weights_bytes"] for row in info["programs"]] == [20, 20]
    for index, name in enumerate(("prefill", "decode")):
        stage = output / f"stage_{index}_{name}"
        source = (stage / "stage_adapter.c").read_text()
        assert "case 1: return 0;" in source  # original stream slot
        assert "case 3: return 2;" in source  # original state slot
        assert "case 0:" not in source  # deleted immutable parameter
        assert source.count("arg = original_input_arg(arg);") == 2
        selected = mlir_query.parse(stage / "model.renamed.mlir")
        entry = next(op for op in selected.body.block.ops if op.name == "func.func")
        assert entry.sym_name.data == f"merlin_stage_{index}_{name}"
    scheduler = (output / "merlin_session.c").read_text()
    assert "merlin_stage_1_input(3)" in scheduler  # logical binding remains captured index
    shared = session_bundle.materialize_shared_weights(
        session_bundle.load(root), info, tmp_path / "shared", entry_projections=selections)
    assert shared["bytes"] == 20
    assert (tmp_path / "shared/weights.bin").read_bytes() == np.array([1, 3, 5, 7, 9], np.float32).tobytes()


@pytest.mark.parametrize("mutation", ["remap", "selected", "prepared", "weights"])
def test_saved_projection_is_rederived_and_refuses_mutated_sources(tmp_path, mutation):
    root, selections = fixture(tmp_path)
    selection = selections["prefill"]
    receipt = selection.prepared_dir / "entry_weight_projection/entry_weight_projection.json"
    data = json.loads(receipt.read_text())
    if mutation == "remap":
        data["original_to_projected"] = [0, 1, 2, None]
        receipt.write_text(json.dumps(data))
    elif mutation == "selected":
        path = selection.prepared_dir / "entry_weight_projection/projected.mlir"
        path.write_text(path.read_text() + "\n// changed\n")
    elif mutation == "prepared":
        selection.source.write_text(MODEL.replace("arith.addf", "arith.mulf"))
    else:
        path = root / "stages/prefill/capture/weights.safetensors"
        path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        read_program_entry_projection(model_dir=root / "stages/prefill/capture",
                                      prepared_dir=selection.prepared_dir,
                                      source=selection.source, receipt=receipt)


def test_missing_projection_cannot_materialize_a_compacted_layout(tmp_path):
    root, selections = fixture(tmp_path)
    info = session_bundle.generate(root, tmp_path / "generated", materialize_weights=False,
                                   entry_projections=selections)
    with pytest.raises(ValueError, match="selected entry projection"):
        session_bundle.materialize_shared_weights(session_bundle.load(root), info, tmp_path / "shared")


def test_shared_projected_weights_rederive_bytes_before_reusing_storage(tmp_path):
    import os

    root, selections = fixture(tmp_path)
    info = session_bundle.generate(root, tmp_path / "generated", materialize_weights=False,
                                   entry_projections=selections)
    session = session_bundle.load(root)
    first = session_bundle.materialize_shared_weights(session, info, tmp_path / "first",
                                                      entry_projections=selections)
    source = Path(first["path"])
    result = session_bundle.materialize_shared_weights(session, info, tmp_path / "reused",
                                                       entry_projections=selections, weights_source=source)
    assert os.path.samefile(source, result["path"]) and first["sha256"] == result["sha256"]
    assert source.read_bytes() == np.array([1, 3, 5, 7, 9], np.float32).tobytes()
    source.write_bytes(np.array([1, 3, 5, 7, 10], np.float32).tobytes())
    with pytest.raises(ValueError, match="derived bytes"):
        session_bundle.materialize_shared_weights(session, info, tmp_path / "refused",
                                                   entry_projections=selections, weights_source=source)


def test_all_stage_proofs_are_reopened_before_sharing_one_weight_blob(tmp_path):
    root, selections = fixture(tmp_path)
    info = session_bundle.generate(root, tmp_path / "generated", materialize_weights=False,
                                   entry_projections=selections)
    selections["decode"].source.write_text(MODEL.replace("arith.addf", "arith.mulf"))
    with pytest.raises(ValueError, match="source or proof changed"):
        session_bundle.materialize_shared_weights(session_bundle.load(root), info, tmp_path / "shared",
                                                   entry_projections=selections)
    assert not (tmp_path / "shared").exists()


def test_unknown_program_and_entry_symbol_collision_refuse(tmp_path):
    root, selections = fixture(tmp_path)
    with pytest.raises(ValueError, match="unknown program"):
        session_bundle.generate(root, tmp_path / "generated", entry_projections={"unknown": selections["prefill"]})
    selected = selections["prefill"]
    # A changed module refuses the original proof before any rename.
    changed = mlir_query.parse(MODEL.replace("func.func @forward", "func.func @collision"))
    with pytest.raises(ValueError):
        rename_projected_entry(changed, selected.projection, "collision")
    module = mlir_query.parse(MODEL[:-2] + 'func.func private @collision()\n}\n')
    proof = plan_entry_weight_projection(module, entry_contract=CONTRACT,
                                         argument_table=selected.projection.argument_table)
    with pytest.raises(ValueError, match="collides"):
        rename_projected_entry(module, proof, "collision")
    assert isinstance(selected, ProgramEntryProjection)


def test_direct_internal_callers_are_renamed_and_reprojected(tmp_path):
    _, selections = fixture(tmp_path)
    module = mlir_query.parse(MODEL[:-2] + '''
      func.func @caller(%x:tensor<5xf32>) -> tensor<5xf32> {
        %r,%s=func.call @forward(%x,%x,%x,%x):
          (tensor<5xf32>,tensor<5xf32>,tensor<5xf32>,tensor<5xf32>)->(tensor<5xf32>,tensor<5xf32>)
        func.return %r:tensor<5xf32>
      }
    }''')
    table = selections["prefill"].projection.argument_table
    proof = plan_entry_weight_projection(module, entry_contract=CONTRACT, argument_table=table)
    renamed, new = rename_projected_entry(module, proof, "stage_entry")
    result = apply_entry_weight_projection(renamed, new)
    call, = [op for op in result.walk() if op.name == "func.call"]
    assert call.callee.root_reference.data == "stage_entry" and len(call.arguments) == 3
    assert new.direct_calls_rewritten == proof.direct_calls_rewritten == 1
