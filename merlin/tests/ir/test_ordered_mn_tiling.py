"""Exact outer matrix tiling and row-major boundary placement."""

from __future__ import annotations

from dataclasses import replace

import pytest

from merlin.semantic_compiler.model import IndexMap, KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.ordered_k_tiling import partition_ordered_matmul_k
from merlin.semantic_compiler.ordered_mn_tiling import (
    OrderedMNTilingError,
    assemble_row_major_mn_outputs,
    materialize_row_major_mn_inputs,
    partition_ordered_matmul_mn,
)


def _source(*, m: int = 4, n: int = 6, k: int = 4) -> KernelRequest:
    maps = (
        IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((0, 1, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
    )
    return KernelRequest(
        (
            SemanticNode("a", "input", (), TensorType((m, k), "i8", "raw-byte-copy"), effect="input"),
            SemanticNode("w", "input", (), TensorType((n, k), "i8", "raw-byte-copy"), effect="input"),
            SemanticNode("c", "matmul", ("a", "w"), TensorType((m, n), "i32", "ordered-rne"),
                         index_maps=maps),
        ),
        ("c",), ("external",), (("a", "external"), ("w", "external")),
        "synthetic-target", lowering_policy="diagnostic", source_identity="public-matrix",
    )


def test_mn_tiles_and_k_tiles_preserve_each_ordered_output() -> None:
    source = _source()
    plan = partition_ordered_matmul_mn(source, tile_m=2, tile_n=3,
                                       qualified_policy="ordered-rne")
    assert len(plan.tiles) == 4
    assert [(tile.m_start, tile.n_start) for tile in plan.tiles] == [
        (0, 0), (0, 3), (2, 0), (2, 3),
    ]
    assert plan.record()["source_request_digest"] == source.digest()
    assert all(tile.request.source_identity == source.source_identity for tile in plan.tiles)
    assert all(tile.request.nodes[-1].type.shape == (2, 3) for tile in plan.tiles)
    assert all(partition_ordered_matmul_k(
        tile.request, tile_k=2, qualified_policy="ordered-rne",
    ).tile_count == 2 for tile in plan.tiles)

    inputs = {"a": bytes(range(1, 17)), "w": bytes(range(17, 41))}
    packed = materialize_row_major_mn_inputs(plan, inputs, element_bytes=1)
    assert packed[1]["a"] == packed[0]["a"]
    assert packed[2]["w"] == packed[0]["w"]
    tile_outputs = []
    for tile_inputs in packed:
        tile_outputs.append(b"".join(
            sum(tile_inputs["a"][row * 4 + k] * tile_inputs["w"][col * 4 + k]
                for k in range(4)).to_bytes(2, "little")
            for row in range(2) for col in range(3)
        ))
    assembled = assemble_row_major_mn_outputs(plan, tile_outputs, element_bytes=2)
    expected = b"".join(
        sum(inputs["a"][row * 4 + k] * inputs["w"][col * 4 + k]
            for k in range(4)).to_bytes(2, "little")
        for row in range(4) for col in range(6)
    )
    assert assembled == expected


def test_mn_slice_uses_byte_width_and_preserves_row_stride() -> None:
    plan = partition_ordered_matmul_mn(
        _source(m=4, n=6, k=2), tile_m=2, tile_n=3, qualified_policy="ordered-rne",
    )
    inputs = {
        "a": b"".join(value.to_bytes(2, "little") for value in range(8)),
        "w": b"".join(value.to_bytes(2, "little") for value in range(12)),
    }
    packed = materialize_row_major_mn_inputs(plan, inputs, element_bytes=2)
    assert packed[2]["a"] == b"".join(value.to_bytes(2, "little") for value in range(4, 8))
    assert packed[1]["w"] == b"".join(value.to_bytes(2, "little") for value in range(6, 12))


def test_mn_tail_padding_preserves_valid_output_cells() -> None:
    source = _source(m=5, n=4, k=4)
    plan = partition_ordered_matmul_mn(
        source, tile_m=2, tile_n=3, qualified_policy="ordered-rne",
        tail_pad_element=b"\x00",
    )
    assert [(tile.m_start, tile.n_start) for tile in plan.tiles] == [
        (0, 0), (0, 3), (2, 0), (2, 3), (4, 0), (4, 3),
    ]
    assert plan.record()["schema"] == "merlin.ordered_mn_tiling.v2"
    assert plan.record()["tail_pad_element_hex"] == "00"
    assert plan.record()["tiles"][-1]["valid_m"] == 1
    assert plan.record()["tiles"][-1]["valid_n"] == 1

    inputs = {"a": bytes(range(1, 21)), "w": bytes(range(21, 37))}
    packed = materialize_row_major_mn_inputs(plan, inputs, element_bytes=1)
    assert packed[-1]["a"][-4:] == bytes(4)
    assert packed[-1]["w"][-8:] == bytes(8)
    tiles = [b"".join(
        sum(boundary["a"][row * 4 + k] * boundary["w"][col * 4 + k]
            for k in range(4)).to_bytes(2, "little")
        for row in range(2) for col in range(3)
    ) for boundary in packed]
    assert assemble_row_major_mn_outputs(plan, tiles, element_bytes=2) == b"".join(
        sum(inputs["a"][row * 4 + k] * inputs["w"][col * 4 + k]
            for k in range(4)).to_bytes(2, "little")
        for row in range(5) for col in range(4)
    )


def test_mn_single_partial_tile_keeps_one_result() -> None:
    plan = partition_ordered_matmul_mn(
        _source(m=1, n=1, k=2), tile_m=2, tile_n=3,
        qualified_policy="ordered-rne", tail_pad_element=b"\x00",
    )
    assert len(plan.tiles) == 1
    packed = materialize_row_major_mn_inputs(
        plan, {"a": b"\x02\x03", "w": b"\x05\x07"}, element_bytes=1,
    )
    assert packed == ({"a": b"\x02\x03\x00\x00",
                       "w": b"\x05\x07\x00\x00\x00\x00"},)
    assert assemble_row_major_mn_outputs(
        plan, ((31).to_bytes(2, "little") + bytes(10),), element_bytes=2,
    ) == (31).to_bytes(2, "little")


def test_mn_rejects_unqualified_tails_policy_and_reordered_maps() -> None:
    with pytest.raises(OrderedMNTilingError, match="qualified pad"):
        partition_ordered_matmul_mn(_source(m=5), tile_m=2, tile_n=3,
                                    qualified_policy="ordered-rne")
    with pytest.raises(OrderedMNTilingError, match="element width"):
        plan = partition_ordered_matmul_mn(
            _source(m=5), tile_m=2, tile_n=3,
            qualified_policy="ordered-rne", tail_pad_element=b"\x00\x00",
        )
        materialize_row_major_mn_inputs(
            plan, {"a": bytes(20), "w": bytes(24)}, element_bytes=1,
        )
    with pytest.raises(OrderedMNTilingError, match="policy differs"):
        partition_ordered_matmul_mn(_source(), tile_m=2, tile_n=3,
                                    qualified_policy="different-policy")
    source = _source()
    changed = replace(source.nodes[-1], index_maps=source.nodes[-1].index_maps[::-1])
    with pytest.raises(OrderedMNTilingError, match="maps"):
        partition_ordered_matmul_mn(replace(source, nodes=(*source.nodes[:-1], changed)),
                                    tile_m=2, tile_n=3, qualified_policy="ordered-rne")


def test_mn_record_and_runtime_reject_missing_or_changed_regions() -> None:
    plan = partition_ordered_matmul_mn(_source(), tile_m=2, tile_n=3,
                                       qualified_policy="ordered-rne")
    changed = replace(plan.tiles[1], n_start=0)
    with pytest.raises(OrderedMNTilingError, match="source region"):
        replace(plan, tiles=(plan.tiles[0], changed, *plan.tiles[2:]))
    with pytest.raises(OrderedMNTilingError, match="source inputs"):
        materialize_row_major_mn_inputs(plan, {"a": bytes(16)}, element_bytes=1)
    with pytest.raises(OrderedMNTilingError, match="source bytes"):
        materialize_row_major_mn_inputs(plan, {"a": bytes(15), "w": bytes(24)},
                                        element_bytes=1)
    with pytest.raises(OrderedMNTilingError, match="output tile bytes"):
        assemble_row_major_mn_outputs(plan, [bytes(12)] * 3, element_bytes=2)
    with pytest.raises(OrderedMNTilingError, match="output tile bytes"):
        assemble_row_major_mn_outputs(plan, [bytes(12)] * 3 + [bytes(11)],
                                      element_bytes=2)


def test_selected_native_mn_then_k_records_replay_the_exact_digest_chain(tmp_path, monkeypatch):
    import hashlib
    import json
    from types import SimpleNamespace

    from merlin.compiler.build.native_records import publish_ordered_records
    from merlin.compiler.frontend.tiling import prepare_ordered_k, prepare_ordered_mn
    from merlin.verification.native_records import check_ordered_records

    source = _source()
    seen = []

    class Binding:
        def prepare_ordered_mn(self, request, tile_m, tile_n):
            seen.append(request.digest())
            return partition_ordered_matmul_mn(request, tile_m=tile_m, tile_n=tile_n,
                                               qualified_policy="ordered-rne")

        def prepare_ordered_k(self, request, tile_k):
            seen.append(request.digest())
            return partition_ordered_matmul_k(request, tile_k=tile_k, qualified_policy="ordered-rne")

    args = SimpleNamespace(ordered_m_tile=2, ordered_n_tile=3, outer_tile_index=2, ordered_k_tile=2)
    outer, mn_record = prepare_ordered_mn(args, Binding(), source)
    selected, k_record = prepare_ordered_k(args, Binding(), outer)
    assert seen == [source.digest(), outer.digest()]
    assert selected.source_identity == source.source_identity
    assert json.loads(k_record)["source_request_digest"] == outer.digest()
    assert json.loads(mn_record)["selected_request_digest"] == outer.digest()
    manifest = publish_ordered_records(
        tmp_path, {"stage_artifacts": {}}, source_request=source, k_source_request=outer,
        ordered_mn_record=mn_record, ordered_k_record=k_record,
    )
    assert manifest["outer_source_request_digest"] == source.digest()
    assert manifest["source_request_digest"] == outer.digest()
    options = dict(source_request=source, k_source_request=outer,
                   ordered_mn_record=mn_record, ordered_k_record=k_record)
    check_ordered_records(tmp_path, manifest, **options)
    with pytest.raises(ValueError, match="unrequested ordered-K"):
        check_ordered_records(tmp_path, manifest, **{**options, "ordered_k_record": None})
    (tmp_path / "ordered-mn-tiling.json").write_bytes(mn_record + b" ")
    with pytest.raises(ValueError, match="ordered-M/N partition differs"):
        check_ordered_records(tmp_path, manifest, **options)
    with pytest.raises(ValueError, match="both geometry"):
        prepare_ordered_mn(SimpleNamespace(ordered_m_tile=2), Binding(), source)
    with pytest.raises(ValueError, match="invalid ordered M/N"):
        prepare_ordered_mn(
            SimpleNamespace(ordered_m_tile=2, ordered_n_tile=3, outer_tile_index=True), Binding(), source,
        )

    # Exercise the actual public commands and artifact checker with an explicit
    # synthetic emitter; this proves orchestration, not target execution.
    from merlin.semantic_compiler import snapshot as snapshot_module
    from merlin.semantic_compiler import target_binding
    from merlin.targetgen import cli

    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    (snapshot_root / "manifest.json").write_bytes(b"{}")
    profile = SimpleNamespace(target_identity=source.target_identity, digest=lambda: "selected-profile")
    snapshot = SimpleNamespace(root=snapshot_root, profile=profile)

    class EmittingBinding(Binding):
        def profile(self):
            return profile

        def compile(self, actual_snapshot, request, *, destination, **options):
            assert actual_snapshot is snapshot and request.digest() == selected.digest()
            destination.mkdir()
            binary, execution = b"\x13\x00\x00\x00", b"{}"
            (destination / "program.bin").write_bytes(binary)
            (destination / "execution_plan.json").write_bytes(execution)
            manifest = {
                "engine": "merlin_native", "request_digest": request.digest(),
                "target_identity": request.target_identity, "compile_mode": "diagnostic",
                "native_snapshot_sha256": hashlib.sha256(b"{}").hexdigest(),
                "binary_sha256": hashlib.sha256(binary).hexdigest(),
                "execution_plan_sha256": hashlib.sha256(execution).hexdigest(), "stage_artifacts": {},
            }
            (destination / "manifest.json").write_text(json.dumps(manifest))
            return manifest

    monkeypatch.setattr(snapshot_module, "open_native_snapshot", lambda path: snapshot)
    monkeypatch.setattr(target_binding, "load_native_target_binding", lambda name: EmittingBinding())
    request_path, abi_path = tmp_path / "source.json", tmp_path / "abi.json"
    request_path.write_text(json.dumps(source.record()))
    abi_path.write_text(json.dumps({"fixed_inputs": {"a": 0, "w": 1}, "fixed_outputs": [2]}))
    common = ["--engine", "merlin_native", "--support", "synthetic", "--snapshot", str(snapshot_root),
              "--request", str(request_path), "--ordered-m-tile", "2", "--ordered-n-tile", "3",
              "--outer-tile-index", "2", "--ordered-k-tile", "2", "--mode", "diagnostic"]
    artifact, status = tmp_path / "compiled", tmp_path / "replay.json"
    assert cli.main(["native-compile", *common, "--target-source", str(tmp_path),
                     "--abi", str(abi_path), "--out", str(artifact)]) == 0
    assert (artifact / "ordered-mn-tiling.json").read_bytes() == mn_record
    assert (artifact / "ordered-k-tiling.json").read_bytes() == k_record
    assert cli.main(["native-verify", *common, "--artifact", str(artifact), "--out", str(status)]) == 0
    assert json.loads(status.read_bytes())["status"] == "verified_artifacts"
    changed = list(common)
    changed[changed.index("--outer-tile-index") + 1] = "1"
    assert cli.main(["native-verify", *changed, "--artifact", str(artifact), "--out", str(status)]) == 2
    assert "partition differs" in json.loads(status.read_bytes())["reason"]
