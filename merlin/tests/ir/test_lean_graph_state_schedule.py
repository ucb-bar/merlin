"""Actual registered scheduling routes and complete graph-context witnesses."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.verify.lean_graph_state_schedule import intake, check_boundary

REFERENCE = repo_root().parent / "merlin-graph-proof-intake"


def scalar_source(width=8, *, partial=False, multiple=False):
    t = f"i{width}"
    before = f"%provided = func.call @provider(%x) : ({t}) -> {t}" if partial else ""
    value = "%provided" if partial else "%a"
    extra = f"""func.func @second(%x:{t}, %bias:{t}) -> ({t},{t}) {{
      %a = arith.subi %x, %bias : {t}
      %b = arith.muli %bias, %bias : {t}
      func.call @watch(%a,%b) : ({t},{t}) -> ()
      func.return %a,%b : {t},{t}
    }}""" if multiple else ""
    return f"""builtin.module {{
      func.func private @provider({t}) -> {t}
      func.func private @watch({t},{t})
      func.func private @other({t},{t})
      func.func @main(%x:{t},%bias:{t}) -> ({t},{t}) {{
        {before}
        %a = arith.addi %x,%x : {t}
        %b = arith.muli %bias,%bias : {t}
        func.call @watch(%a,%b) : ({t},{t}) -> ()
        func.return {value},%b : {t},{t}
      }}
      {extra}
      func.func @untouched(%x:{t})->{t} {{func.return %x:{t}}}
    }}"""


def nested_source(width=16):
    t = f"i{width}"
    return f"""builtin.module {{
      func.func private @provider({t},index)->{t}
      func.func private @watch({t},{t})
      func.func @main(%x:{t},%bias:{t})->{t} {{
        %lb=arith.constant 0:index
        %ub=arith.constant 3:index
        %step=arith.constant 1:index
        %out=scf.for %iv=%lb to %ub step %step iter_args(%acc=%x)->({t}) {{
          %a=arith.addi %bias,%bias:{t}
          %b=arith.muli %x,%x:{t}
          %inner=scf.for %j=%lb to %ub step %step iter_args(%carry=%acc)->({t}) {{
            %c=arith.addi %bias,%bias:{t}
            %d=arith.muli %x,%x:{t}
            %sum=arith.addi %carry,%d:{t}
            func.call @watch(%sum,%c):({t},{t})->()
            %v=func.call @provider(%sum,%j):({t},index)->{t}
            scf.yield %v:{t}
          }}
          func.call @watch(%a,%b):({t},{t})->()
          scf.yield %inner:{t}
        }}
        func.return %out:{t}
      }}
      func.func @untouched(%x:{t})->{t} {{func.return %x:{t}}}
    }}"""


def same_block_source():
    return """builtin.module {
      func.func private @watch(i8,i8)
      func.func @main(%x:i8,%bias:i8)->(i8,i8) {
        %a=arith.addi %x,%x:i8
        %b=arith.muli %bias,%bias:i8
        %c=arith.subi %x,%bias:i8
        func.call @watch(%a,%b):(i8,i8)->()
        func.return %c,%b:i8,i8
      }
      func.func @untouched(%x:i8)->i8{func.return %x:i8}
    }"""


def same_block_orders(root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "explicit-orders.json"
    path.write_text(json.dumps([
        {"block_index": 1, "order": [1, 0, 2, 3, 4]},
        {"block_index": 1, "order": [2, 0, 1, 3, 4]},
    ], indent=2) + "\n")
    return path


def produce(root, text, *, chain=False, settings=None, explicit_orders=None):
    root.mkdir(parents=True, exist_ok=True)
    source = root / "input.mlir"
    source.write_text(text)
    return intake(REFERENCE, root / "producer", source=source, chain=chain, settings=settings, explicit_orders=explicit_orders)


def checked(root, capture, *, settings=None, mode="graph"):
    return check_boundary([r["path"] for r in capture["buffers"]], [r["path"] for r in capture["order_buffers"]],
                          capture["journal"]["path"], root / "checked", REFERENCE, settings=settings, mode=mode)


@pytest.mark.parametrize("width,partial", [(8, False), (64, True), (256, False)])
def test_actual_scalar_reorder(tmp_path, width, partial):
    capture = produce(tmp_path, scalar_source(width, partial=partial))
    assert capture["each_stage_changed"] == [True]
    report = checked(tmp_path, capture)
    assert report["status"] == "conditional_graph_checked", report
    proof = json.loads(Path(report["proof"]["path"]).read_text())
    assert not proof["audit"]["raw"]["violations"]


def test_actual_multiple_functions_and_current_orders(tmp_path):
    settings = {"count": 2}
    capture = produce(tmp_path, scalar_source(multiple=True), settings=settings)
    assert capture["each_order_changed"] == [True, True]
    assert len({row["block_index"] for row in capture["requests"]}) == 2
    report = checked(tmp_path, capture, settings=settings)
    assert report["status"] == "conditional_graph_checked", report


def test_actual_later_order_on_same_current_block(tmp_path):
    settings = {"count": 2}
    capture = produce(tmp_path, same_block_source(), settings=settings, explicit_orders=same_block_orders(tmp_path))
    assert capture["each_order_changed"] == [True, True]
    assert capture["requests"][0]["block_index"] == capture["requests"][1]["block_index"]
    assert capture["requests"][0]["witness"]["operation_sha256"] != capture["requests"][1]["witness"]["operation_sha256"]
    report = checked(tmp_path, capture, settings=settings)
    assert report["status"] == "conditional_graph_checked", report


@pytest.mark.parametrize("empty", [False, True])
def test_actual_nested_captures_multiple_orders(tmp_path, empty):
    settings = {"count": 2, "minimum_depth": 2}
    text = nested_source().replace("constant 3:index", "constant 0:index") if empty else nested_source()
    capture = produce(tmp_path, text, settings=settings)
    assert capture["each_order_changed"] == [True, True]
    report = checked(tmp_path, capture, settings=settings)
    assert report["status"] == "conditional_graph_checked", report


@pytest.mark.parametrize("width,factor,tile,bits", [(8, 2, 3, 32), (32, 3, 2, 64), (64, 4, 2, 64)])
def test_actual_seven_buffer_all_passes_changed(tmp_path, width, factor, tile, bits):
    # Reuse the sealed source fixture, never its old tests/oracles/replays.
    from test_lean_graph_state_integer import chain_source

    settings = {"factor": factor, "tile": tile, "index_bits": bits}
    capture = produce(tmp_path, chain_source(width), chain=True, settings=settings)
    assert capture["each_stage_changed"] == [True] * 6
    report = checked(tmp_path, capture, settings=settings)
    assert report["status"] == "conditional_graph_checked", report


def test_actual_seven_buffer_initialized_provider(tmp_path):
    from test_lean_graph_state_integer import chain_source

    capture = produce(tmp_path, chain_source(), chain=True)
    assert capture["each_stage_changed"] == [True] * 6
    report = checked(tmp_path, capture, mode="initialized-symbolic-provider-v1")
    assert report["status"] == "conditional_graph_checked", report
    assert report["model"]["success"] == {"values": [17], "world": 32}
    assert report["model"]["failure"] == {"error": 17, "world": 8}


def test_actual_standalone_initialized_provider(tmp_path):
    capture = produce(tmp_path, scalar_source(partial=True))
    report = checked(tmp_path, capture, mode="initialized-scalar-provider-v1")
    assert report["status"] == "conditional_graph_checked", report
    assert report["model"]["success"] == {"values": [17, 25], "world": 9}
    assert report["model"]["failure"] == {"error": 17, "world": 8}
