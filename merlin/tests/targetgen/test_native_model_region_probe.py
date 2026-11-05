"""Model-region diagnostics retain SSA bindings and later compile attempts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.frontends.compile_inputs import StagedCompileInputs
from merlin.targetgen import cli
from merlin.tests.ir.test_precision_intake import SOURCE


def _run(tmp_path: Path, monkeypatch, *, source: str = SOURCE, decisions=None,
         blocked_static: bool = False, batched_decisions=None, failed_batched_tile=None,
         max_batched_tiles: int = 256, named_tile_shape=(2, 4),
         failed_named_tile=None, max_named_tiles: int = 256,
         abi_pool: bool = False, first_roles=None, reject_unindexed_first: bool = False,
         duplicate_window_for: str | None = None, abi_base: bool = True):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "program.mlir").write_text(source)
    digest = hashlib.sha256(source.encode()).hexdigest()
    names = decisions or ("first", "second")
    policy = {"source_sha256": digest, "matmul_regions": {name: {} for name in names}}
    if batched_decisions is not None:
        policy["batched_matmul_regions"] = batched_decisions
    staged = StagedCompileInputs(
        program=source,
        entry="work",
        precision_policy=policy,
        precision_policy_document=b"{}",
        original_source_sha256=digest,
        program_sha256=digest,
        precision_policy_sha256=hashlib.sha256(b"{}").hexdigest(),
    )
    monkeypatch.setattr("merlin.frontends.compile_inputs.load_compile_inputs", lambda _: staged)
    profile = SimpleNamespace(target_identity="synthetic-model-probe", digest=lambda: "selected-profile")
    monkeypatch.setattr(
        "merlin.semantic_compiler.snapshot.open_native_snapshot", lambda _: SimpleNamespace(profile=profile)
    )
    def prepare_tile(_staged, region_id, *, tile_index=None):
        if reject_unindexed_first and region_id == "first" and tile_index is None:
            from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError

            raise LinalgBridgeError("source contraction shape exceeds selected target tile")
        from merlin.semantic_compiler.precision_intake import (
            check_source_bound_f32_matmuls,
            inventory_source_f32_batched_matmuls,
        )

        regions = (
            *check_source_bound_f32_matmuls(
                source, entry="work", policy=policy, original_source_sha256=digest,
            ),
            *inventory_source_f32_batched_matmuls(source, entry="work"),
        )
        region = next(item for item in regions if item.region_id == region_id)
        assert tile_index is None or tile_index >= 0
        is_batched = hasattr(region, "batch_shape")
        tile_shape = (2, 4) if is_batched else named_tile_shape
        m_tiles = (region.result_shape[-2] + tile_shape[0] - 1) // tile_shape[0]
        n_tiles = (region.result_shape[-1] + tile_shape[1] - 1) // tile_shape[1]
        position = (tile_index or 0) % (m_tiles * n_tiles)
        batch_index = ((tile_index or 0) // (m_tiles * n_tiles),) if is_batched else None
        output = SimpleNamespace(
            source_type=SimpleNamespace(shape=region.result_shape),
            source_value=region.result_source,
            source_batch_index=batch_index or (),
            source_offsets=(position // n_tiles * tile_shape[0], position % n_tiles * tile_shape[1]),
            target_axes_from_source=(0, 1),
            target_type=SimpleNamespace(shape=tile_shape),
        )
        roles = first_roles if region_id == "first" and first_roles is not None else ("lhs", "rhs")
        return SimpleNamespace(
            region_id=region_id, batch_index=batch_index, output=output,
            inputs=tuple(SimpleNamespace(source_role=role) for role in roles),
            request=SimpleNamespace(outputs=("result",)),
        )

    binding = SimpleNamespace(profile=lambda: profile, prepare_precision_tile=prepare_tile)
    monkeypatch.setattr(
        "merlin.semantic_compiler.target_binding.load_native_target_binding", lambda _: binding,
    )
    attempted = []

    def compile_region(args):
        tile_index = getattr(args, "diagnostic_tile_index", None)
        attempted.append(args.diagnostic_region if tile_index is None else (args.diagnostic_region, tile_index))
        if tile_index == failed_batched_tile and tile_index is not None:
            return {"status": "infeasible_candidate", "reason": "tile allocation declined"}
        if (args.diagnostic_region, tile_index) == failed_named_tile:
            return {"status": "infeasible_candidate", "reason": "named tile allocation declined"}
        if args.diagnostic_region == "first" and tile_index is None:
            return {"status": "unsupported_semantics", "reason": "first descriptor declined"}
        artifact = Path(args.out)
        artifact.mkdir(parents=True)
        (artifact / "program.bin").write_bytes(b"\x13\x00\x00\x00")
        if tile_index is not None:
            recorded_index = 0 if args.diagnostic_region == duplicate_window_for and tile_index == 1 else tile_index
            plan = prepare_tile(staged, args.diagnostic_region, tile_index=recorded_index)
            boundary = plan.output
            window = [
                min(extent, source_extent - offset)
                for extent, source_extent, offset in zip(
                    boundary.target_type.shape, boundary.source_type.shape[-2:],
                    boundary.source_offsets, strict=True,
                )
            ]
            (artifact / "precision-plan.json").write_text(json.dumps({
                "region_id": args.diagnostic_region,
                "request_digest": "second-request",
                **({"batch_index": list(plan.batch_index)} if plan.batch_index is not None else {}),
                "output": {
                    "source_batch_index": list(boundary.source_batch_index),
                    "source_offsets": list(boundary.source_offsets),
                    "source_window_shape": window,
                    "source_type": {"shape": list(boundary.source_type.shape)},
                    "source_value": boundary.source_value.record(),
                    "target_type": {"shape": list(boundary.target_type.shape)},
                    "target_axes_from_source": list(boundary.target_axes_from_source),
                },
            }))
        return {"status": "emitted", "request_digest": "second-request", "binary_sha256": "binary",
                "constant_inputs": ([{"status": "blocked", "role": "rhs"}] if blocked_static else [])}

    monkeypatch.setattr(cli, "_native_compile_report", compile_region)
    output = tmp_path / "probe"
    abi = tmp_path / "abi.json"
    abi.write_text('{"fixed_inputs": {}, "fixed_outputs": []}\n')
    pool_path = tmp_path / "abi-pool.json"
    if abi_pool:
        pool_path.write_text(json.dumps({"input_slots": [3, 4, 5, 6], "output_slots": [35]}) + "\n")
    args = argparse.Namespace(
        out=str(output),
        compile_inputs=str(inputs),
        engine="merlin_native",
        support="selected-provider",
        snapshot="snapshot",
        abi=str(abi) if abi_base else None,
        abi_pool=str(pool_path) if abi_pool else None,
        search_limits=None,
        max_named_tiles=max_named_tiles,
        max_batched_tiles=max_batched_tiles,
        target_source="selected-source",
    )
    code = cli._cmd_native_probe_model_regions(args)
    report = json.loads((output / "manifest.json").read_text())
    return code, report, attempted, output


def test_region_probe_preserves_model_gaps_and_continues_after_refusal(tmp_path: Path, monkeypatch) -> None:
    code, report, attempted, output = _run(tmp_path, monkeypatch)
    assert code == 2 and report["status"] == "incomplete"
    assert attempted == ["first", "second"]
    assert report["region_status_counts"] == {"emitted": 1, "unsupported_semantics": 1}
    assert report["static_input_status_counts"] == {"not_emitted": 1, "packed_or_runtime": 1}
    assert report["whole_model_executed"] is False and report["composition_status"] == "not_constructed"
    assert report["source_operation_count"] > report["region_count"] == 2
    assert report["source_contraction_count"] == 2
    assert report["batched_contraction_count"] == 0
    assert report["unrouted_operation_count"] > 0
    assert report["regions"][0]["artifact"] is None
    assert report["regions"][1]["artifact"] == "regions/region-0001"
    bridge = report["host_bridge"]
    assert bridge["status"] == "typed_declarations_only"
    assert bridge["linked"] is False and bridge["executed"] is False
    assert bridge["strict_native_eligible"] is False and bridge["required_native_unmet"] == ["first"]
    assert bridge["external_symbols"] == ["work$kernel_1__rsecond"]
    assert bridge["c_interface_symbols"] == bridge["external_symbols"]
    assert bridge["entry_c_interface_symbol"] == "work"
    bridge_path = output / bridge["path"]
    assert hashlib.sha256(bridge_path.read_bytes()).hexdigest() == bridge["sha256"]
    from merlin.frontends.linalg_mlir import parse_mlir_file

    outlined_host = parse_mlir_file(bridge_path)
    functions = {
        operation.sym_name.data: operation
        for operation in outlined_host.body.block.ops
        if operation.name == "func.func"
    }
    assert functions["work$kernel_0__rfirst"].body.blocks
    assert not functions["work$kernel_1__rsecond"].body.blocks
    assert "llvm.emit_c_interface" in functions["work$kernel_1__rsecond"].attributes
    assert "llvm.emit_c_interface" in functions["work"].attributes
    assert "llvm.emit_c_interface" not in functions["work$kernel_0__rfirst"].attributes
    assert bridge["bindings"][1]["input_types"] == ["tensor<2x4xf32>", "tensor<4x2xf32>"]
    assert bridge["bindings"][1]["result_types"] == ["tensor<2x2xf32>"]
    assert bridge["bindings"][1]["static_inputs_status"] == "packed_or_runtime"
    first_result = report["regions"][0]["result_source"]
    second_lhs = report["regions"][1]["lhs_source"]
    assert first_result == second_lhs
    assert (output / "source-graph.json").is_file()
    semantic = report["source_semantic_graph"]
    assert semantic["status"] == "translated_source_only"
    assert semantic["unqualified_node_count"] == 2
    from merlin.semantic_compiler.model import KernelRequest

    semantic_path = output / semantic["path"]
    assert hashlib.sha256(semantic_path.read_bytes()).hexdigest() == semantic["sha256"]
    request = KernelRequest.from_record(json.loads(semantic_path.read_text()))
    assert request.digest() == semantic["request_digest"]
    assert request.source_identity == report["staged_program_sha256"]
    assert tuple(semantic["output_ids"]) == request.outputs
    assert len(request.nodes) == semantic["node_count"]
    stitching = json.loads((output / "stitching.json").read_text())
    assert stitching["status"] == "diagnostic_unexecutable"
    assert report["boundary_crossing_count"] == len(stitching["candidate_boundary_crossings"])
    assert "candidate_to_candidate" in {edge["direction"] for edge in stitching["candidate_boundary_crossings"]}
    kernels = [step for step in stitching["composition_plan"]["functions"][0]["steps"] if step["kind"] == "kernel"]
    assert [step["request_digest"] for step in kernels] == [None, "second-request"]


def test_region_probe_attempts_named_output_tiles_without_claiming_composition(
    tmp_path: Path, monkeypatch,
) -> None:
    code, report, attempted, _ = _run(
        tmp_path, monkeypatch, named_tile_shape=(1, 2), failed_named_tile=("first", 1),
    )
    assert code == 2 and report["status"] == "incomplete"
    assert attempted == ["first", "second", *(('first', index) for index in range(4)),
                         *(('second', index) for index in range(2))]
    assert report["named_tile_count"] == 6
    assert report["named_tile_status_counts"] == {"emitted": 5, "infeasible_candidate": 1}
    assert report["regions"][0]["tiling_status"] == "isolated_tile_failure"
    assert report["regions"][1]["tiling_status"] == "isolated_tiles_emitted"
    assert report["composition_status"] == "not_constructed"
    assert report["whole_model_executed"] is False


def test_region_probe_limits_named_tile_work(tmp_path: Path, monkeypatch) -> None:
    code, report, attempted, _ = _run(
        tmp_path, monkeypatch, named_tile_shape=(1, 2), max_named_tiles=3,
    )
    assert code == 2 and attempted == ["first", "second", ("second", 0), ("second", 1)]
    assert report["regions"][0]["tiling_status"] == "resource_limit"
    assert report["regions"][0]["tile_count"] == 4
    assert report["named_tile_count"] == 2


def test_region_probe_derives_exact_tile_abis_from_declared_slot_pool(tmp_path: Path, monkeypatch) -> None:
    code, report, attempted, output = _run(
        tmp_path, monkeypatch, abi_pool=True, abi_base=False,
        first_roles=("lhs__k0", "rhs__k0", "lhs__k1", "rhs__k1"),
    )
    assert code == 2 and attempted == ["first", "second"]
    assert report["abi_sha256"] is None
    assert report["abi_pool_sha256"] == hashlib.sha256((tmp_path / "abi-pool.json").read_bytes()).hexdigest()
    first = json.loads((output / report["regions"][0]["abi_path"]).read_text())
    second = json.loads((output / report["regions"][1]["abi_path"]).read_text())
    assert first == {
        "fixed_inputs": {"lhs__k0": 3, "rhs__k0": 4, "lhs__k1": 5, "rhs__k1": 6},
        "fixed_outputs": [35],
    }
    assert second == {"fixed_inputs": {"lhs": 3, "rhs": 4}, "fixed_outputs": [35]}


def test_region_probe_refuses_both_exact_abi_and_slot_pool(tmp_path: Path, monkeypatch) -> None:
    # The pool would select every tile ABI while the report still recorded
    # the exact ABI's digest, so the two inputs are mutually exclusive.
    code, report, attempted, _ = _run(tmp_path, monkeypatch, abi_pool=True, abi_base=True)
    assert code == 2 and attempted == []
    assert report["status"] == "compile_error"
    assert "exactly one of --abi or --abi-pool" in report["reason"]
    with pytest.raises(SystemExit) as exited:
        cli.main([
            "native-probe-model-regions", "--engine", "merlin_native", "--support", "selected",
            "--snapshot", "snapshot", "--compile-inputs", "inputs", "--abi", "abi.json",
            "--abi-pool", "pool.json", "--target-source", "source", "--out", str(tmp_path / "unused"),
        ])
    assert exited.value.code == 2
    assert not (tmp_path / "unused").exists()


def test_region_probe_keeps_source_shape_refusal_with_derived_tile_abis(tmp_path: Path, monkeypatch) -> None:
    code, report, attempted, _ = _run(
        tmp_path, monkeypatch, abi_pool=True, abi_base=False, named_tile_shape=(1, 2),
        reject_unindexed_first=True,
    )
    assert code == 2 and attempted[0] == "second"
    assert report["regions"][0]["status"] == "unsupported_semantics"
    assert report["regions"][0]["tiling_status"] == "isolated_tiles_emitted"
    assert report["named_tile_status_counts"] == {"emitted": 6}
    assert report["host_bridge"]["external_symbols"] == [
        "work$kernel_0__rfirst", "work$kernel_1__rsecond",
    ]
    assert report["host_bridge"]["bindings"][0]["selected_route"] == "tiled_native"
    assert len(report["host_bridge"]["bindings"][0]["tile_request_digests"]) == 4
    assert report["host_bridge"]["required_native_unmet"] == []
    assert report["whole_model_executed"] is False

    from merlin.targetgen.contract.model_host_bridge import diagnostic_host_bridge

    changed = [dict(tile) for tile in report["named_tiles"]]
    changed[1]["tile_index"] = 0
    with pytest.raises(ValueError, match="complete selected source tile grid"):
        diagnostic_host_bridge(SOURCE, "work", report["regions"], changed)


@pytest.mark.parametrize("batched", [False, True])
def test_region_probe_rejects_duplicate_emitted_source_window(tmp_path: Path, monkeypatch, batched: bool) -> None:
    region_id = "batch_matmul#0" if batched else "first"
    _, report, _, _ = _run(
        tmp_path, monkeypatch,
        source=_batched_source() if batched else SOURCE,
        batched_decisions={region_id: {}} if batched else None,
        named_tile_shape=(1, 2) if not batched else (2, 4),
        duplicate_window_for=region_id,
    )
    tiles = report["batched_tiles"] if batched else report["named_tiles"]
    region = report["batched_contractions"][0] if batched else report["regions"][0]
    assert any(tile["status"] == "compile_error" and "twice" in tile["reason"] for tile in tiles)
    assert region["status" if batched else "tiling_status"] == "isolated_tile_failure"


def _batched_source() -> str:
    return SOURCE.replace(
        "%c: tensor<4x2xf32>)",
        "%c: tensor<4x2xf32>, %ba: tensor<2x3x4xf32>, %bb: tensor<2x4x5xf32>)",
    ).replace(
        "func.return %second : tensor<2x2xf32>",
        """%batch_empty = tensor.empty() : tensor<2x3x5xf32>
    %batch_init = linalg.fill ins(%zero : f32)
      outs(%batch_empty : tensor<2x3x5xf32>) -> tensor<2x3x5xf32>
    %batch = linalg.generic {
      indexing_maps = [affine_map<(b,m,n,k)->(b,m,k)>,
                       affine_map<(b,m,n,k)->(b,k,n)>,
                       affine_map<(b,m,n,k)->(b,m,n)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%ba, %bb : tensor<2x3x4xf32>, tensor<2x4x5xf32>)
      outs(%batch_init : tensor<2x3x5xf32>) {
      ^bb0(%av: f32, %bv: f32, %acc: f32):
        %product = arith.mulf %av, %bv : f32
        %sum = arith.addf %acc, %product : f32
        linalg.yield %sum : f32
    } -> tensor<2x3x5xf32>
    func.return %second : tensor<2x2xf32>""",
    )


def test_region_probe_counts_unselected_batched_contractions(tmp_path: Path, monkeypatch) -> None:
    code, report, attempted, _ = _run(tmp_path, monkeypatch, source=_batched_source())
    assert code == 2 and attempted == ["first", "second"]
    assert report["region_count"] == 2
    assert report["source_contraction_count"] == 3
    assert report["batched_contraction_count"] == 1
    (batch,) = report["batched_contractions"]
    assert batch["batch_shape"] == [2]
    assert batch["lhs_shape"] == [2, 3, 4] and batch["rhs_shape"] == [2, 4, 5]
    assert batch["scalar_add_order"] == "accumulator-product"
    assert batch["status"] == "unrouted_precision_and_tiling"
    assert any(row["operation_id"] == batch["operation_id"] and row["route"] == "native_region_candidate"
               for row in report["source_operations"])
    assert report["source_semantic_graph"]["unqualified_node_count"] == 3
    assert "batch_matmul#0" in report["host_bridge"]["required_native_unmet"]


@pytest.mark.parametrize("failed_tile", [None, 1])
def test_region_probe_attempts_declared_batched_tiles_without_claiming_composition(
    tmp_path: Path, monkeypatch, failed_tile: int | None,
) -> None:
    code, report, attempted, _ = _run(
        tmp_path, monkeypatch, source=_batched_source(),
        batched_decisions={"batch_matmul#0": {}}, failed_batched_tile=failed_tile,
    )
    assert code == 2 and report["status"] == "incomplete"
    assert attempted[:2] == ["first", "second"]
    assert attempted[2:] == [("batch_matmul#0", index) for index in range(8)]
    assert report["batched_tile_count"] == 8
    assert report["batched_tile_status_counts"] == (
        {"emitted": 8} if failed_tile is None else {"emitted": 7, "infeasible_candidate": 1}
    )
    (batch,) = report["batched_contractions"]
    assert batch["status"] == ("isolated_tiles_emitted" if failed_tile is None else "isolated_tile_failure")
    assert batch["tile_count"] == 8
    assert report["source_operations"][batch["operation_ordinal"]]["route"] == "native_region_candidate"
    bridge = report["host_bridge"]
    (binding,) = [row for row in bridge["bindings"] if row["region_id"] == "batch_matmul#0"]
    assert binding["input_types"] == ["tensor<2x3x4xf32>", "tensor<2x4x5xf32>"]
    assert binding["result_types"] == ["tensor<2x3x5xf32>"]
    if failed_tile is None:
        assert binding["selected_route"] == "tiled_native"
        assert len(binding["tile_request_digests"]) == 8
        assert binding["symbol"] in bridge["external_symbols"]
        assert "batch_matmul#0" not in bridge["required_native_unmet"]
    else:
        assert binding["selected_route"] == "retained_host"
        assert binding["symbol"] not in bridge["external_symbols"]
        assert "batch_matmul#0" in bridge["required_native_unmet"]
    assert report["whole_model_executed"] is False
    assert report["composition_status"] == "not_constructed"


def test_batched_bridge_checks_exact_source_and_works_without_rank_two_selection(
    tmp_path: Path, monkeypatch,
) -> None:
    from merlin.targetgen.contract.model_host_bridge import diagnostic_host_bridge

    source = _batched_source()
    _, report, _, _ = _run(
        tmp_path, monkeypatch, source=source, batched_decisions={"batch_matmul#0": {}},
    )
    batch = report["batched_contractions"]
    tiles = report["batched_tiles"]
    bridge, _ = diagnostic_host_bridge(
        source, "work", [], batched_regions=batch, batched_tiles=tiles,
    )
    assert [row["region_id"] for row in bridge["bindings"]] == ["batch_matmul#0"]
    assert bridge["bindings"][0]["selected_route"] == "tiled_native"

    changed = copy.deepcopy(batch)
    changed[0]["operation_ordinal"] -= 1
    with pytest.raises(ValueError, match="batched host bridge regions differ"):
        diagnostic_host_bridge(
            source, "work", [], batched_regions=changed, batched_tiles=tiles,
        )


@pytest.mark.parametrize("failed_tile", [None, 1])
def test_batched_host_dataflow_checks_zero_start_and_source_routes(
    tmp_path: Path, monkeypatch, failed_tile: int | None,
) -> None:
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.targetgen.contract.model_host_object import (
        _check_entry_dataflow,
        _check_omitted_initializer,
        _source_route_accounting,
    )

    source = _batched_source()
    _, report, _, output = _run(
        tmp_path, monkeypatch, source=source,
        batched_decisions={"batch_matmul#0": {}}, failed_batched_tile=failed_tile,
    )
    bridge = report["host_bridge"]
    dataflow = _check_entry_dataflow(
        source, (output / bridge["path"]).read_text(), "work",
        bridge["external_symbols"],
    )
    assert dataflow["native_zero_initialization_obligation_count"] == (
        2 if failed_tile is None else 1
    )
    accounting = _source_route_accounting(report, bridge, dataflow)
    batch_ordinal = report["batched_contractions"][0]["operation_ordinal"]
    assert accounting["operations"][batch_ordinal]["state"] == (
        "native_external_unexecuted" if failed_tile is None
        else "required_native_retained_host"
    )

    changed = parse_mlir_text(source.replace(
        "%batch_init = linalg.fill ins(%zero : f32)",
        "%one = arith.constant 1.0 : f32\n    %batch_init = linalg.fill ins(%one : f32)",
    ))
    generic = next(op for op in changed.walk() if op.name == "linalg.generic")
    with pytest.raises(ValueError, match="nonzero or unknown initializer"):
        _check_omitted_initializer(generic, generic, selected=True)


def test_region_probe_reports_batched_work_limit_before_tile_compilation(tmp_path: Path, monkeypatch) -> None:
    code, report, attempted, _ = _run(
        tmp_path, monkeypatch, source=_batched_source(),
        batched_decisions={"batch_matmul#0": {}}, max_batched_tiles=7,
    )
    assert code == 2 and attempted == ["first", "second"]
    assert report["batched_tile_count"] == 0
    assert report["batched_contractions"][0]["status"] == "resource_limit"
    assert report["batched_contractions"][0]["tile_count"] == 8


@pytest.mark.parametrize("variant", ["transposed_rhs", "nonzero_init"])
def test_region_probe_lists_noncanonical_batched_contractions_as_unrouted(
    tmp_path: Path, monkeypatch, variant: str,
) -> None:
    # An inventory-only probe must not refuse the whole model because one
    # rank-3 contraction is not the canonical batched matmul.
    if variant == "transposed_rhs":
        source = _batched_source().replace(
            "%bb: tensor<2x4x5xf32>", "%bb: tensor<2x5x4xf32>",
        ).replace(
            "affine_map<(b,m,n,k)->(b,k,n)>", "affine_map<(b,m,n,k)->(b,n,k)>",
        ).replace(
            "tensor<2x3x4xf32>, tensor<2x4x5xf32>", "tensor<2x3x4xf32>, tensor<2x5x4xf32>",
        )
    else:
        source = _batched_source().replace(
            "%batch_init = linalg.fill ins(%zero : f32)",
            "%one = arith.constant 1.0 : f32\n    %batch_init = linalg.fill ins(%one : f32)",
        )
    code, report, attempted, _ = _run(tmp_path, monkeypatch, source=source)
    assert code == 2 and attempted == ["first", "second"]
    assert report["status"] == "incomplete"
    assert report["region_count"] == 2
    assert report["batched_contraction_count"] == 0
    generics = [row for row in report["source_operations"] if row["mlir_operation"] == "linalg.generic"]
    assert len(generics) == 1 and generics[0]["route"] == "unrouted"
    assert generics[0]["unrouted_reason"]

    selected_tmp = tmp_path / "selected"
    selected_tmp.mkdir()
    code, report, attempted, _ = _run(
        selected_tmp, monkeypatch, source=source, batched_decisions={"batch_matmul#0": {}},
    )
    assert code == 2 and attempted == []
    assert report["status"] == "unsupported_semantics"


def test_region_probe_rejects_missing_batched_precision_decision(tmp_path: Path, monkeypatch) -> None:
    code, report, attempted, _ = _run(
        tmp_path, monkeypatch, source=_batched_source(), batched_decisions={},
    )
    assert code == 2 and attempted == []
    assert report["status"] == "unsupported_semantics"
    assert "missing=['batch_matmul#0']" in report["reason"]


def test_blocked_static_input_keeps_image_but_not_host_call(tmp_path: Path, monkeypatch) -> None:
    code, report, attempted, output = _run(tmp_path, monkeypatch, blocked_static=True)
    assert code == 2 and attempted == ["first", "second"]
    assert report["region_status_counts"] == {"emitted": 1, "unsupported_semantics": 1}
    assert report["static_input_status_counts"] == {"blocked": 1, "not_emitted": 1}
    assert report["regions"][1]["artifact"] == "regions/region-0001"
    assert report["regions"][1]["static_inputs_status"] == "blocked"
    bridge = report["host_bridge"]
    assert bridge["external_symbols"] == []
    assert bridge["c_interface_symbols"] == []
    assert bridge["required_native_unmet"] == ["first", "second"]
    assert bridge["bindings"][1]["native_status"] == "emitted"
    assert bridge["bindings"][1]["static_inputs_status"] == "blocked"
    from merlin.frontends.linalg_mlir import parse_mlir_file

    outlined = parse_mlir_file(output / bridge["path"])
    functions = {op.sym_name.data: op for op in outlined.body.block.ops if op.name == "func.func"}
    assert functions["work$kernel_1__rsecond"].body.blocks
    stitching = json.loads((output / "stitching.json").read_text())
    kernels = [step for step in stitching["composition_plan"]["functions"][0]["steps"]
               if step["kind"] == "kernel"]
    assert kernels[1]["compiler_support"] == "emitted_blocked_static_input"


@pytest.mark.parametrize("blocked", [False, True])
def test_tile_grid_needs_packed_static_inputs_for_a_native_route(
    tmp_path: Path, monkeypatch, blocked: bool,
) -> None:
    # A complete tile grid is a native route only when every tile's static
    # weights are packed, exactly as for a direct region.
    code, report, _, _ = _run(
        tmp_path, monkeypatch, named_tile_shape=(1, 2), blocked_static=blocked,
    )
    assert code == 2
    second = report["regions"][1]
    assert second["tiling_status"] == "isolated_tiles_emitted"
    tiles = [tile for tile in report["named_tiles"] if tile["region_id"] == "second"]
    assert len(tiles) == 2
    expected = "blocked" if blocked else "packed_or_runtime"
    assert all(tile["static_inputs_status"] == expected for tile in tiles)
    binding = report["host_bridge"]["bindings"][1]
    assert binding["region_id"] == "second"
    if blocked:
        assert binding["selected_route"] == "retained_host"
        assert binding["tile_request_digests"] == []
    else:
        assert binding["selected_route"] == "direct_native"


def test_region_id_cannot_choose_an_artifact_path(tmp_path: Path, monkeypatch) -> None:
    source = SOURCE.replace('prov.region_id = "first"', 'prov.region_id = "../first"')
    code, report, attempted, output = _run(tmp_path, monkeypatch, source=source, decisions=("../first", "second"))
    assert code == 2 and attempted == ["../first", "second"]
    assert report["regions"][1]["artifact"] == "regions/region-0001"
    assert not (tmp_path / "first").exists()
    assert (output / "regions" / "region-0001" / "program.bin").is_file()


def test_missing_source_decision_stops_before_emission(tmp_path: Path, monkeypatch) -> None:
    code, report, attempted, output = _run(tmp_path, monkeypatch, decisions=("first",))
    assert code == 2 and report["status"] == "unsupported_semantics"
    assert "missing=['second']" in report["reason"]
    assert attempted == [] and not (output / "regions").exists()


def test_host_bridge_rejects_a_stale_compiled_operation_id(tmp_path: Path, monkeypatch) -> None:
    from merlin.targetgen.contract.model_host_bridge import diagnostic_host_bridge

    _, report, _, _ = _run(tmp_path, monkeypatch)
    diagnostic_host_bridge(SOURCE, "work", report["regions"])
    stale = [dict(row) for row in report["regions"]]
    stale[1]["operation_id"] = stale[0]["operation_id"]
    with pytest.raises(ValueError, match="operation IDs differ"):
        diagnostic_host_bridge(SOURCE, "work", stale)
    with pytest.raises(ValueError, match="operation IDs differ"):
        diagnostic_host_bridge(SOURCE + "\n", "work", report["regions"])


def test_region_probe_keeps_ordered_multiple_model_outputs(tmp_path: Path, monkeypatch) -> None:
    source = SOURCE.replace(
        ") -> tensor<2x2xf32> {",
        ") -> (tensor<2x4xf32>, tensor<2x2xf32>) {",
    ).replace(
        "func.return %second : tensor<2x2xf32>",
        "func.return %first, %second : tensor<2x4xf32>, tensor<2x2xf32>",
    )
    code, report, attempted, output = _run(tmp_path, monkeypatch, source=source)
    assert code == 2 and attempted == ["first", "second"]
    outputs = report["model_outputs"]
    assert len(outputs) == 1
    source_results = [row["result_value_ids"][0] for row in report["source_operations"] if row["region_id"] is not None]
    assert outputs[0]["operand_value_ids"] == source_results
    semantic = report["source_semantic_graph"]
    assert semantic["status"] == "translated_source_only"
    request = json.loads((output / semantic["path"]).read_text())
    assert semantic["output_ids"] == request["outputs"]
    assert len(request["outputs"]) == 2
    assert request["outputs"][0] != request["outputs"][1]


def test_host_entry_dataflow_checks_source_operands_and_call_ownership(tmp_path: Path, monkeypatch) -> None:
    from xdsl.printer import Printer

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.targetgen.contract.model_host_object import (
        _check_entry_dataflow,
        _source_route_accounting,
    )

    _, report, _, output = _run(tmp_path, monkeypatch)
    bridge_text = (output / report["host_bridge"]["path"]).read_text()
    selected = report["host_bridge"]["external_symbols"]
    checked = _check_entry_dataflow(SOURCE, bridge_text, "work", selected)
    assert checked["schema"] == "merlin.host_entry_dataflow_check.v2"
    assert checked["status"] == "source_to_host_body_checked"
    assert checked["operation_count"] == report["source_operation_count"]
    assert checked["native_external_call_count"] == 1
    assert checked["host_defined_call_count"] == 1
    assert checked["host_linalg_body_correspondence_checked_count"] == 1
    assert checked["omitted_initializer_operand_count"] == 2
    assert checked["host_initializer_check_count"] == 1
    assert checked["native_zero_initialization_obligation_count"] == 1
    assert checked["initialization_semantics"] == "source_drop_and_host_initializer_checked"
    assert checked["native_reset_semantics"] == "not_checked"
    accounting = _source_route_accounting(report, report["host_bridge"], checked)
    assert accounting["operation_count"] == report["source_operation_count"]
    assert accounting["state_counts"]["native_external_unexecuted"] == 1
    assert accounting["state_counts"]["required_native_retained_host"] == 1
    assert sum(accounting["state_counts"].values()) == accounting["operation_count"]
    stale_probe = dict(report)
    stale_probe["source_operations"] = [dict(row) for row in report["source_operations"]]
    stale_probe["source_operations"][0]["mlir_operation"] = "test.mutated"
    with pytest.raises(ValueError, match="changed operation order or names"):
        _source_route_accounting(stale_probe, report["host_bridge"], checked)

    nonzero = SOURCE.replace("0.000000e+00", "1.000000e+00")
    nonzero_bridge = bridge_text.replace("0.000000e+00", "1.000000e+00")
    with pytest.raises(ValueError, match="drops a nonzero or unknown initializer"):
        _check_entry_dataflow(nonzero, nonzero_bridge, "work", selected)

    from xdsl.dialects.builtin import Float32Type, FloatAttr

    host_module = parse_mlir_text(bridge_text)
    retained = next(op for op in host_module.body.block.ops
                    if op.name == "func.func" and op.sym_name.data == "work$kernel_0__rfirst")
    constant = next(op for op in retained.body.block.ops if op.name == "arith.constant")
    constant.properties["value"] = FloatAttr(1.0, Float32Type())
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(host_module)
    with pytest.raises(ValueError, match="host body lost positive-zero initialization"):
        _check_entry_dataflow(SOURCE, stream.getvalue(), "work", selected)

    changed_body = parse_mlir_text(bridge_text)
    retained = next(op for op in changed_body.body.block.ops
                    if op.name == "func.func" and op.sym_name.data == "work$kernel_0__rfirst")
    matmul = next(op for op in retained.body.block.ops if op.name == "linalg.matmul")
    arithmetic = next(op for op in matmul.regions[0].block.ops if op.name == "arith.addf")
    arithmetic.name = "arith.subf"
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(changed_body)
    with pytest.raises(ValueError, match="changed its Linalg computation"):
        _check_entry_dataflow(SOURCE, stream.getvalue(), "work", selected)

    # Equal-shaped operands keep the call type valid after a bad swap, so the
    # wiring check rather than the signature check must catch the mutation.
    uniform = SOURCE
    for shape in ("2x3", "3x4", "4x2", "2x4"):
        uniform = uniform.replace(f"tensor<{shape}xf32>", "tensor<2x2xf32>")
    uniform_root = tmp_path / "uniform"
    uniform_root.mkdir()
    _, uniform_report, _, uniform_output = _run(uniform_root, monkeypatch, source=uniform)
    uniform_bridge = (uniform_output / uniform_report["host_bridge"]["path"]).read_text()
    module = parse_mlir_text(uniform_bridge)
    entry = next(op for op in module.body.block.ops
                 if op.name == "func.func" and op.sym_name.data == "work")
    call = next(op for op in entry.body.block.ops
                if op.name == "func.call" and len(op.operands) == 2)
    call.operands[0] = call.operands[1]
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    with pytest.raises(ValueError, match="changed SSA operands"):
        _check_entry_dataflow(uniform, stream.getvalue(), "work", uniform_report["host_bridge"]["external_symbols"])
    with pytest.raises(ValueError, match="unselected host call has no compiled body"):
        _check_entry_dataflow(SOURCE, bridge_text, "work", [])


def test_host_entry_dataflow_preserves_repeated_ssa_input() -> None:
    from xdsl.printer import Printer

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.targetgen.contract.model_host_object import _check_entry_dataflow
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    source = """"builtin.module"() ({
      "func.func"() <{function_type = (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>, sym_name = "work"}> ({
      ^bb0(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>):
        %empty = "tensor.empty"() : () -> tensor<2x2xf32>
        %result = "linalg.generic"(%a, %a, %empty) <{
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
          operandSegmentSizes = array<i32: 2, 1>} > ({
        ^bb1(%x: f32, %y: f32, %out: f32):
          %product = "arith.mulf"(%x, %y) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "linalg.yield"(%product) : (f32) -> ()
        }) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        "func.return"(%result) : (tensor<2x2xf32>) -> ()
      }) : () -> ()
    }) : () -> ()
    """

    def printed(module) -> str:
        stream = StringIO()
        Printer(stream=stream, print_generic_format=True).print_op(module)
        return stream.getvalue()

    bridge = outline_dispatches(parse_mlir_text(source), forward="work")
    assert len(bridge.dispatches) == 1
    checked = _check_entry_dataflow(source, printed(bridge.module), "work", [])
    assert checked["host_defined_call_count"] == 1
    assert checked["checked_operand_edges"] == 2  # one call input, one return
    assert checked["omitted_initializer_operand_count"] == 1

    changed_body = outline_dispatches(parse_mlir_text(source), forward="work").module
    host = next(op for op in changed_body.body.block.ops
                if op.name == "func.func" and op.sym_name.data != "work")
    generic = next(op for op in host.body.block.ops if op.name == "linalg.generic")
    generic.operands[1] = generic.operands[2]
    with pytest.raises(ValueError, match="changed its Linalg operands"):
        _check_entry_dataflow(source, printed(changed_body), "work", [])

    changed_call = outline_dispatches(parse_mlir_text(source), forward="work").module
    entry = next(op for op in changed_call.body.block.ops
                 if op.name == "func.func" and op.sym_name.data == "work")
    call = next(op for op in entry.body.block.ops if op.name == "func.call")
    call.operands[0] = entry.body.block.args[1]
    with pytest.raises(ValueError, match="changed SSA operands"):
        _check_entry_dataflow(source, printed(changed_call), "work", [])
