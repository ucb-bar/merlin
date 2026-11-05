"""Frozen current SSA must agree with its exact source programs and region."""

import hashlib
from copy import deepcopy
from dataclasses import replace

import pytest
from test_connected_spatial_host import fixture

from merlin.compile.owned_spatial_source import _source_graph
from merlin.targetgen.corpus_spec import CorpusBinding
from merlin.xdsl_dialects._common import text


def documents(tmp_path):
    module, rewrite, *_ = fixture()
    from merlin.frontends.linalg_mlir import parse_mlir_text

    module = parse_mlir_text(text(module, generic=True).replace("2x5x5x3xi8", "1x5x5x3xi8")
                             .replace("50x17xi8", "25x17xi8").replace("18x17xi8", "9x17xi8"))
    entries = deepcopy(rewrite.entries)
    entries["producer"]["batch"] = 1
    entries["pool"]["M"] = 25
    buffers = deepcopy(rewrite.call_buffers)
    for rows in buffers.values():
        for row in rows:
            row["shape"] = {tuple((2, 5, 5, 3)): [1, 5, 5, 3], (50, 17): [25, 17], (18, 17): [9, 17]}.get(
                tuple(row["shape"]), row["shape"]
            )
    for name, entry in entries.items():
        entry.update(name=name, kind="op", source_role="handauthored_compiler_test", source_reference="original SSA")
    rewrite = replace(
        rewrite, granularity="group", entries=entries, signatures={name: () for name in entries}, call_buffers=buffers,
        programs={name: {"schema": "group_program_v1", "entry": deepcopy(entry)} for name, entry in entries.items()},
    )
    from merlin.llvmlower.connected_spatial import extract_connected_spatial_region

    region = extract_connected_spatial_region(module, rewrite, ("producer", "pool"))
    module, rewrite = region.module, region.rewrite
    (tmp_path / "region.mlir").write_text(text(module, generic=True))
    rewrite.write_sidecar(tmp_path)
    def pin(name):
        return {"file": name, "sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()}
    selected = {
        "schema": "merlin.owned_spatial_source.v2",
        "source_graph": pin("region.mlir"), "source_rewrite": pin("device_signatures.json"),
    }
    source = {
        "name": "original_region", "kind": "model_slice", "source_role": "handauthored_compiler_test",
        "source_reference": "original SSA", "op": "spatial_producer_pool", "components": list(entries.values()),
    }
    binding = CorpusBinding(rewrite.device, 16, "int8", "i32", True, ["L2"], "exact_int", requant_output_dtype="i8")
    return module, rewrite, selected, source, binding


def test_normal_reader_consumes_current_ssa_without_reconstructing_a_graph(tmp_path, monkeypatch):
    import merlin.llvmlower.connected_spatial as owner

    module, rewrite, selected, source, binding = documents(tmp_path)

    def reconstruct(*args):
        raise AssertionError("v2 must not reconstruct a source graph")

    monkeypatch.setattr(owner, "public_spatial_source_graph", reconstruct)
    actual, result, receipt = _source_graph(tmp_path, selected, source, binding)
    assert text(actual, generic=True) == text(module, generic=True)
    assert result.entries == rewrite.entries
    assert receipt["schema"] == "captured_spatial_source_graph_v1"


@pytest.mark.parametrize("mutation", ["statement", "extra_code", "wrong_target"])
def test_source_pins_cannot_authorize_different_statements_extra_code_or_target(tmp_path, mutation):
    module, rewrite, selected, source, binding = documents(tmp_path)
    if mutation == "statement":
        source["components"][0]["acc_scale"] = 0.25
    elif mutation == "wrong_target":
        rewrite = replace(rewrite, device="different_target")
        rewrite.write_sidecar(tmp_path)
        path = tmp_path / "device_signatures.json"
        selected["source_rewrite"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        from xdsl.dialects import arith

        block = module.body.block.first_op.body.block
        block.insert_op_before(arith.ConstantOp.from_int_and_width(7, 32), block.last_op)
        path = tmp_path / "region.mlir"
        path.write_text(text(module, generic=True))
        selected["source_graph"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        _source_graph(tmp_path, selected, source, binding)
