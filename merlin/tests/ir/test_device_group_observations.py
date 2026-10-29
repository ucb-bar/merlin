"""A multi-group rewrite shares shape observations without changing its statements."""

import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from merlin.common import mlir_query as mq
from merlin.compile.analysis import contraction_shapes
from merlin.llvmlower.device_offload import rewrite_groups_to_device
from merlin.xdsl_dialects.lowering import compute_groups as CG
from merlin.xdsl_dialects.lowering import group_command as GC


class Oracle:
    def ask(self, *, family, in_dtype, **kwargs):
        return (
            CG.Admission(True, units=("unit0",))
            if family == "contraction" and in_dtype == "int8"
            else CG.Admission(False, "undeclared", "not available")
        )

    def unit_for(self, *args):
        return "unit0"

    def absorbs(self, *args):
        return CG.Admission(False, "undeclared", "no readout")


def module():
    shapes = ((3, 7, 5), (11, 13, 17), (2, 19, 23))
    arguments = []
    body = []
    returns = []
    types = []
    for i, (m, k, n) in enumerate(shapes):
        arguments.extend((f"%a{i}: tensor<{m}x{k}xi8>", f"%b{i}: tensor<{k}x{n}xi8>"))
        output = f"tensor<{m}x{n}xi32>"
        body.append(f"""%zero{i} = arith.constant 0 : i32
          %empty{i} = tensor.empty() : {output}
          %init{i} = linalg.fill ins(%zero{i} : i32) outs(%empty{i} : {output}) -> {output}
          %result{i} = linalg.generic {{indexing_maps=[
          affine_map<(d0,d1,d2)->(d0,d2)>,affine_map<(d0,d1,d2)->(d2,d1)>,affine_map<(d0,d1,d2)->(d0,d1)>],
          iterator_types=["parallel","parallel","reduction"]}}
          ins(%a{i},%b{i}:tensor<{m}x{k}xi8>,tensor<{k}x{n}xi8>) outs(%init{i}:{output}) {{
          ^bb0(%lhs:i8,%rhs:i8,%acc:i32):
            %wide_a=arith.extsi %lhs:i8 to i32
            %wide_b=arith.extsi %rhs:i8 to i32
            %product=arith.muli %wide_a,%wide_b:i32
            %sum=arith.addi %product,%acc:i32
            linalg.yield %sum:i32
          }} -> {output}""")
        returns.append(f"%result{i}")
        types.append(output)
    source = (
        "module {func.func @forward("
        + ",".join(arguments)
        + ") -> ("
        + ",".join(types)
        + ") {"
        + "\n".join(body)
        + "\n return "
        + ",".join(returns)
        + ":"
        + ",".join(types)
        + "}}"
    )
    result = mq.parse(source)
    result.verify()
    return result


def test_source_statement_observation_is_shared_within_one_rewrite(monkeypatch):
    original = contraction_shapes.contraction_extents
    calls = []

    def observed(module):
        calls.append(module)
        return original(module)

    monkeypatch.setattr(contraction_shapes, "contraction_extents", observed)
    source = module()
    result = rewrite_groups_to_device(source, "neutral", select=lambda _: True, oracle=Oracle(), weight_args={1, 3, 5})
    source.verify()
    assert result.moved == 3
    assert {(e["M"], e["K"], e["N"]) for e in result.entries.values()} == {(3, 7, 5), (11, 13, 17), (2, 19, 23)}
    assert len(calls) <= 2  # Formation and the rewrite each observe once, never per group.


def test_indexed_rewrite_is_identical_to_fresh_statement_observation(monkeypatch):
    indexed = module()
    result = rewrite_groups_to_device(indexed, "neutral", select=lambda _: True, oracle=Oracle(), weight_args={1, 3, 5})
    original = GC.program

    def uncached(group, **kwargs):
        kwargs["extents"] = None
        return original(group, **kwargs)

    monkeypatch.setattr(GC, "program", uncached)
    fresh = module()
    reference = rewrite_groups_to_device(
        fresh, "neutral", select=lambda _: True, oracle=Oracle(), weight_args={1, 3, 5}
    )
    assert result.to_sidecar() == reference.to_sidecar()
    assert str(indexed) == str(fresh)


@pytest.mark.parametrize("logical", [True, False])
def test_preparation_is_read_once_while_source_operands_are_still_attached(tmp_path, monkeypatch, logical):
    from merlin.llvmlower import device_shim
    from merlin.xdsl_dialects.lowering import group_prepack as GP

    manifest, header, chunks, offset = {}, {}, [], 0
    for arg, k, n in ((1, 7, 5), (3, 13, 17), (5, 19, 23)):
        name = f"weight{arg}"
        raw = np.ones((k, n), dtype=np.int8).tobytes()
        header[name] = {"dtype": "I8", "shape": [k, n], "data_offsets": [offset, offset + len(raw)]}
        manifest[str(arg)] = {"kind": "param", "weight": name}
        chunks.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    (tmp_path / "weights.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))
    (tmp_path / "weights.manifest.json").write_text(json.dumps(manifest))
    original, observations = GP.prepack, []

    def observe(groups, *args, **kwargs):
        observations.append([group.root.parent is not None for group in groups])
        return original(groups, *args, **kwargs)

    monkeypatch.setattr(GP, "prepack", observe)
    monkeypatch.setattr(device_shim, "kernel_abi_for", lambda _target: SimpleNamespace(version=2) if logical else None)
    source = module()
    result = rewrite_groups_to_device(
        source, "neutral", select=lambda _: True, oracle=Oracle(), weight_args={1, 3, 5}, capture=tmp_path
    )
    source.verify()
    assert result.moved == 3
    assert observations == [[True, True, True]]
    assert result.prepack["device_weights"] == 3
    assert result.prepack["skipped"] == []
    assert {row["group"] for row in result.prepack["groups"]} == {0, 1, 2}


def test_column_gains_are_part_of_the_actual_kernel_identity():
    from merlin.llvmlower.device_offload import _kernel_identity

    entry = {"op": "matmul", "M": 3, "K": 7, "N": 5, "epilogue": ["acc_scale"],
             "acc_scale_by_column": [0.5] * 4 + [0.25]}
    assert _kernel_identity(entry) != _kernel_identity({**entry, "acc_scale_by_column": [0.5] * 5})
    assert _kernel_identity(entry) == _kernel_identity({**entry, "name": "another_source_group"})


def test_terminal_residual_keeps_layout_evidence_from_original_producers(tmp_path, monkeypatch):
    from logical_device_target import install
    from test_device_shim_logical_abi import _Capture, _write_capture

    install(monkeypatch)
    capture = _Capture()
    shape = [1, 4, 3, 3]
    x = capture.arg("x", shape, "i8")
    weights = {name: capture.arg(name, [4, 4, 1, 1], "i8") for name in ("wa", "wb")}
    biases = {name: capture.arg(name, [4], "f32") for name in ("ba", "bb")}
    a, _, _ = capture.conv(x, 4, 3, 3, 4, 1, 1, 0, weights["wa"], biases["ba"], .5, .25, .5)
    b, _, _ = capture.conv(a, 4, 3, 3, 4, 1, 1, 0, weights["wb"], biases["bb"], .5, .25, .5)
    result = capture.residual(a, b, shape, .5, .5, .5)
    text = capture.module(result, shape)
    stored = {name: np.ones((4, 4, 1, 1), dtype=np.int8) for name in weights}
    stored.update({name: np.zeros(4, dtype=np.float32) for name in biases})
    root = tmp_path / "capture"
    _write_capture(root, text, stored)
    source = mq.parse(text)
    rewrite = rewrite_groups_to_device(
        source, "gemmini", select=lambda _: True, capture=root / "model.mlir", require_complete_groups=True,
    )
    source.verify()
    assert rewrite.moved == 3
    assert sorted(rewrite.entries[row.symbol]["op"] for row in rewrite.routed) == ["conv2d", "conv2d", "residual_add"]
    assert rewrite.skipped == ()
