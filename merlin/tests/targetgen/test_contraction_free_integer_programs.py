"""A quantized program with no contraction is an integer program, and reaches the device as one.

WHY THIS EXISTS. The core-aten int8 flow integerized a capture only when a contraction was admitted, and
every receipt check required at least one integerized contraction. A quantized relu, pad, cat or pool
-- annotated by the recipe quantizer, rewritten exactly -- was therefore refused at the receipt
("independently executed integer partition differs", "no i8xi8->i32 integer contraction") and kept as an
fp32 host program, and nothing could route it to a provider that lowers it. These pin the pieces: the
marked contraction-free receipt, the prospective screen that decides to recapture, and the provider
region offload that binds, rewrites and links such programs.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from merlin.capture.integerization import (
    NON_CONTRACTION_ANNOTATIONS,
    contraction_partition,
    non_contraction_annotations,
    non_contraction_program,
)
from merlin.llvmlower import device_build
from merlin.targetgen.integerization_export import receipt_export_complete


def _receipt(**changes):
    receipt = {
        "schema": "m2m.pt2e-integerize.v1",
        "quantized_contractions_seen": 0,
        "quantized_contractions_integerized": 0,
        "quantized_contractions_remaining": 0,
        "quantized_by_kind": {
            kind: {"seen": 0, "integerized": 0, "remaining": 0}
            for kind in ("linear", "conv2d", "matmul", "unsupported")
        },
        "refusals": [],
        "integer_program": "non_contraction",
        "non_contraction_annotations": {key: 0 for key in NON_CONTRACTION_ANNOTATIONS} | {"annotated_affine_maps": 1},
        "golden_agreement": {"status": "passed"},
    }
    receipt.update(changes)
    return receipt


def test_a_marked_zero_census_receipt_is_a_contraction_free_program():
    receipt = _receipt()
    assert non_contraction_program(receipt)
    assert contraction_partition(receipt) == {"seen": 0, "integerized": 0, "preserved": 0}
    assert receipt_export_complete(receipt, contraction_partition(receipt))


@pytest.mark.parametrize(
    "changes",
    [
        {"integer_program": None},
        {"non_contraction_annotations": {key: 0 for key in NON_CONTRACTION_ANNOTATIONS}},
        {"quantized_contractions_seen": 1},
        {"refusals": [{"kind": "matmul", "node": "mm"}]},
    ],
)
def test_anything_but_the_exact_marked_form_still_needs_a_contraction(changes):
    receipt = _receipt(**changes)
    assert not non_contraction_program(receipt)
    with pytest.raises(ValueError):
        contraction_partition(receipt)


def test_the_annotation_census_comes_from_the_quantizer_stats():
    assert non_contraction_annotations({"annotated_movements": 2}) == {
        "annotated_sums": 0,
        "annotated_means": 0,
        "annotated_affine_maps": 0,
        "annotated_movements": 2,
    }
    assert non_contraction_annotations({"annotated_contractions": 3}) is None
    assert non_contraction_annotations(None) is None


def _offload(tmp_path: Path, bindings):
    def callback(prepared, workdir):
        workdir.mkdir(parents=True)
        files = {name: workdir / name for name in ("model.mlir", "shim.c", "kernels.o", "manifest.json")}
        files["model.mlir"].write_text("module {}\n")
        files["shim.c"].write_text("void f(void) {}\n")
        files["kernels.o"].write_bytes(b"\x7fELF")
        files["manifest.json"].write_text(json.dumps({"bindings": bindings, "declines": []}))
        return files["model.mlir"], files["shim.c"], files["kernels.o"], files["manifest.json"]

    return device_build.DeviceRouting(
        device="t", package_dir=tmp_path, operand_dtype="i8", accum_dtype="i32", region_offload=callback
    )


def test_region_offload_records_and_verifies_what_the_link_adds(tmp_path):
    routing = _offload(tmp_path, [{"root": 3, "symbol": "k0"}])
    source = tmp_path / "source.mlir"
    source.write_text("module {}\n")
    rewritten = device_build.apply_region_offload(routing, source, tmp_path / "work")
    sidecar = device_build.load_region_offload(tmp_path / "work")
    assert Path(sidecar["model"]) == rewritten and sidecar["routed"] == [{"root": 3, "symbol": "k0"}]
    Path(sidecar["shim"]).write_text("void g(void) {}\n")
    with pytest.raises(ValueError, match="shim changed"):
        device_build.load_region_offload(tmp_path / "work")


def test_region_offload_refuses_an_empty_binding_or_a_mixed_route(tmp_path):
    source = tmp_path / "source.mlir"
    source.write_text("module {}\n")
    with pytest.raises(ValueError, match="bound no region"):
        device_build.apply_region_offload(_offload(tmp_path, []), source, tmp_path / "a")
    mixed = copy.replace(_offload(tmp_path, [{"root": 1}]), catalog_builder=lambda *_: None)
    with pytest.raises(ValueError, match="exclusive"):
        device_build.apply_region_offload(mixed, source, tmp_path / "b")


def test_the_bundle_routing_hands_the_provider_its_package(monkeypatch, tmp_path):
    from merlin.llvmlower import toolchain
    from merlin.targetgen.core_aten_device import region_routing_for_bundle

    seen = {}

    def provider_offload(prepared, workdir, *, package, llvm_bin):
        seen.update(prepared=prepared, package=package, llvm_bin=llvm_bin)
        return ("m", "s", "o", "j")

    monkeypatch.setattr(toolchain, "llvm_install", lambda: tmp_path / "llvm")
    routing = region_routing_for_bundle("t", tmp_path, provider_offload)
    assert routing.region_offload(tmp_path / "x.mlir", tmp_path / "w") == ("m", "s", "o", "j")
    assert seen == {"prepared": tmp_path / "x.mlir", "package": tmp_path, "llvm_bin": tmp_path / "llvm" / "bin"}
