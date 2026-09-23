"""Routing contractions to a matrix unit inside the whole-model build.

The build is where the two halves of the path have to agree: the IR rewrite decides which contractions
move, and the object build defines the symbols those calls need. A disagreement between them is a link
error a long way from its cause, so these tests assert the agreement rather than either half.

They also pin the fail-closed direction. Enabling the feature with nothing to route to would produce a
model that grades correctly and reports a capability it never used, which is the failure mode that is
hardest to notice and easiest to cite.
"""

from __future__ import annotations

import builtins
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.llvmlower.declaration_access import unpatched_declarations
from merlin.llvmlower.impr_features import OPU_MATMUL_NAME
from merlin.runtime.backends.zephyr_model import MatrixRouting, prepare_for_lowering

SIDECAR_NAME = "opu_signatures.json"  # historical adapter artifact, not a generic-core default

#: A rank-2 int8 contraction in the form the int8 rewrite leaves behind. Two extents: one that fills a
#: 32-lane tile in both directions and one that does not, so a tile-filling selector has something to
#: decline.
_MODEL = """
builtin.module {
  func.func @forward(%a: tensor<64x32xi8>, %b: tensor<32x64xi8>,
                     %c: tensor<8x32xi8>, %d: tensor<32x8xi8>) -> tensor<64x64xi32> {
    %z = arith.constant 0 : i32
    %e0 = tensor.empty() : tensor<64x64xi32>
    %f0 = linalg.fill ins(%z : i32) outs(%e0 : tensor<64x64xi32>) -> tensor<64x64xi32>
    %big = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                            affine_map<(d0, d1, d2) -> (d2, d1)>,
                                            affine_map<(d0, d1, d2) -> (d0, d1)>],
                           iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%a, %b : tensor<64x32xi8>, tensor<32x64xi8>)
        outs(%f0 : tensor<64x64xi32>) {
    ^bb0(%x: i8, %y: i8, %acc: i32):
      %xe = arith.extsi %x : i8 to i32
      %ye = arith.extsi %y : i8 to i32
      %m = arith.muli %xe, %ye : i32
      %s = arith.addi %acc, %m : i32
      linalg.yield %s : i32
    } -> tensor<64x64xi32>
    %e1 = tensor.empty() : tensor<8x8xi32>
    %f1 = linalg.fill ins(%z : i32) outs(%e1 : tensor<8x8xi32>) -> tensor<8x8xi32>
    %small = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                              affine_map<(d0, d1, d2) -> (d2, d1)>,
                                              affine_map<(d0, d1, d2) -> (d0, d1)>],
                             iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%c, %d : tensor<8x32xi8>, tensor<32x8xi8>)
        outs(%f1 : tensor<8x8xi32>) {
    ^bb0(%x: i8, %y: i8, %acc: i32):
      %xe = arith.extsi %x : i8 to i32
      %ye = arith.extsi %y : i8 to i32
      %m = arith.muli %xe, %ye : i32
      %s = arith.addi %acc, %m : i32
      linalg.yield %s : i32
    } -> tensor<8x8xi32>
    func.return %big : tensor<64x64xi32>
  }
}
"""


def _model(tmp_path: Path) -> Path:
    p = tmp_path / "model.mlir"
    p.write_text(_MODEL, encoding="utf-8")
    return p


class TestTheRoutingIsInertUnlessAsked:
    def test_without_the_feature_nothing_is_routed_and_no_sidecar_appears(self, tmp_path):
        # The whole-model build must be byte-identical when the feature is off; a sidecar left behind
        # would make a later build think something had been routed.
        prepared, _feats = prepare_for_lowering(_model(tmp_path), tmp_path, features=frozenset(), blocking=False)
        assert "merlin_opu_gemm_i8" not in prepared.read_text()
        assert not (tmp_path / SIDECAR_NAME).exists()

    def test_the_feature_without_a_routing_target_is_refused(self, tmp_path):
        # Silently not routing would be indistinguishable from a feature that did nothing, and the model
        # would grade correctly while reporting a capability it never used.
        with pytest.raises(ValueError, match="no `matrix=` routing"):
            prepare_for_lowering(_model(tmp_path), tmp_path, features=frozenset({OPU_MATMUL_NAME}), blocking=False)


@pytest.fixture
def synthetic_matrix(tmp_path, monkeypatch):
    root = tmp_path / "selected-provider"
    (root / "contracts").mkdir(parents=True)
    (root / "contracts/target_contract.yaml").write_text(
        "name: synthetic_matrix_support\nplugin: {matrix_lowering: matrix.py}\n"
    )
    (root / "matrix.py").write_text("""from merlin.llvmlower import int8_contractions as shared
calls = []
def geometry(*, unit, config):
    calls.append(("geometry", unit, config))
    return (32, 128)
def selector(tile_edge):
    calls.append(("selector", tile_edge))
    return shared.tile_filling_selector(tile_edge)
def rewrite_prepared_file(prepared, work, *, select, tile_edge):
    calls.append(("rewrite", tile_edge))
    return shared.rewrite_prepared_file(prepared, work, select=select, tile_edge=tile_edge,
        symbol_prefix="synthetic_matrix_call", sidecar_name="synthetic_matrix.json")
def load_signatures(work):
    return shared.load_sidecar(work, "synthetic_matrix.json")
def build_object(*args, **kwargs):
    raise AssertionError("no native object build in metadata/preparation tests")
""")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root))
    monkeypatch.setenv("MERLIN_TARGETS_DIR", str(tmp_path / "empty-references"))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("matrix test launched native tooling"))
    original_import = builtins.__import__

    def no_opu(name, globals=None, locals=None, fromlist=(), level=0):
        if any(part in name or part in (fromlist or ()) for part in ("passes_opu", "opu_shim")):
            pytest.fail("generic matrix preparation imported OPU")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", no_opu)
    try:
        yield (
            MatrixRouting(support_target="synthetic_matrix_support", unit="independent_unit", config="explicit_config"),
            root,
        )
    finally:
        for name in tuple(sys.modules):
            if name == "merlin._oot_targets.synthetic_matrix_support" or name.startswith(
                "merlin._oot_targets.synthetic_matrix_support."
            ):
                sys.modules.pop(name)


def test_synthetic_matrix_preparation_uses_explicit_provider(synthetic_matrix, tmp_path):
    matrix, root = synthetic_matrix
    provider = matrix.provider()
    assert Path(provider.__file__) == root / "matrix.py"
    prepared, _ = prepare_for_lowering(
        _model(tmp_path),
        tmp_path,
        features=frozenset({OPU_MATMUL_NAME}),
        blocking=False,
        matrix=matrix,
    )
    signatures = provider.load_signatures(tmp_path)
    assert signatures == {"synthetic_matrix_call_0": (64, 64, 32)}
    assert (tmp_path / "synthetic_matrix.json").is_file()
    assert not (tmp_path / SIDECAR_NAME).exists()
    text = prepared.read_text()
    assert "func.call @synthetic_matrix_call_0" in text
    assert unpatched_declarations(text, signatures) == ()
    assert ("geometry", "independent_unit", "explicit_config") in provider.calls
    assert ("selector", 32) in provider.calls
    assert ("rewrite", 32) in provider.calls


def test_matrix_provider_geometry_and_selector_are_caller_owned(synthetic_matrix):
    matrix, _ = synthetic_matrix
    assert matrix.tile_edge() == 32
    selector = matrix.selector()
    assert selector(SimpleNamespace(parallel=(64, 64), reduction=(32,)))
    assert not selector(SimpleNamespace(parallel=(8, 8), reduction=(32,)))


def test_explicit_selector_bypasses_provider_selector(synthetic_matrix):
    from dataclasses import replace

    matrix, _ = synthetic_matrix

    def explicit(shape):
        return False

    assert replace(matrix, select=explicit).selector() is explicit


def test_matrix_provider_refuses_implicit_selection(synthetic_matrix, monkeypatch):
    from merlin.targetgen.plugins import PluginError

    matrix, _ = synthetic_matrix
    monkeypatch.delenv("MERLIN_TARGET_PATH")
    with pytest.raises(PluginError, match="explicit MERLIN_TARGET_PATH"):
        matrix.provider()


def test_matrix_provider_refuses_missing_plugin(synthetic_matrix):
    from merlin.targetgen.plugins import PluginError

    matrix, root = synthetic_matrix
    (root / "contracts/target_contract.yaml").write_text("name: synthetic_matrix_support\n")
    with pytest.raises(PluginError, match="matrix_lowering"):
        matrix.provider()


def test_matrix_routing_requires_support_identity():
    with pytest.raises(TypeError, match="support_target"):
        MatrixRouting(unit="fixture_unit", config="fixture_config")


@pytest.mark.parametrize("field", ["support_target", "unit", "config"])
@pytest.mark.parametrize("value", ["", "  ", None, 7])
def test_matrix_routing_requires_nonempty_string_fields(field, value):
    arguments = {"support_target": "fixture", "unit": "fixture_unit", "config": "fixture_config"}
    arguments[field] = value
    with pytest.raises(ValueError, match=field):
        MatrixRouting(**arguments)


@pytest.mark.parametrize(
    "missing", ["geometry", "selector", "rewrite_prepared_file", "load_signatures", "build_object"]
)
def test_matrix_provider_requires_complete_callable_interface(synthetic_matrix, missing):
    from merlin.targetgen.plugins import PluginError

    matrix, root = synthetic_matrix
    with (root / "matrix.py").open("a") as stream:
        stream.write(f"\n{missing} = None\n")
    with pytest.raises(PluginError, match=missing):
        matrix.provider()


@pytest.mark.parametrize("backend_name,entry", [("spike_model", "build"), ("zephyr_model", "build_app")])
def test_build_refuses_unselected_matrix_before_output_or_native_work(tmp_path, monkeypatch, backend_name, entry):
    import importlib

    from merlin.targetgen.plugins import PluginError

    monkeypatch.delenv("MERLIN_TARGET_PATH", raising=False)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("build launched native tooling"))
    backend = importlib.import_module("merlin.runtime.backends." + backend_name)
    work = tmp_path / "uncreated-work"
    model = tmp_path / "absent-model"
    matrix = MatrixRouting(support_target="unselected_synthetic_provider", unit="fixture_unit", config="fixture_config")
    with pytest.raises(PluginError, match="explicit MERLIN_TARGET_PATH"):
        getattr(backend, entry)(model, work, matrix=matrix)
    assert not work.exists()
    assert not model.exists()


@pytest.mark.parametrize(
    "change", ["missing_matrix", "support_target", "unit", "config", "signatures", "missing_receipt"]
)
def test_matrix_handoff_refuses_changed_or_unbound_routing(synthetic_matrix, tmp_path, change):
    from dataclasses import replace

    from merlin.runtime.backends.zephyr_model import load_matrix_signatures

    matrix, _ = synthetic_matrix
    prepare_for_lowering(
        _model(tmp_path),
        tmp_path,
        features=frozenset({OPU_MATMUL_NAME}),
        blocking=False,
        matrix=matrix,
    )
    expected = {"synthetic_matrix_call_0": (64, 64, 32)}
    assert load_matrix_signatures(tmp_path, matrix) == expected
    receipt = tmp_path / "matrix_routing.json"
    document = json.loads(receipt.read_text())
    assert document == {
        "schema": "merlin.matrix-routing.v1",
        "support_target": matrix.support_target,
        "unit": matrix.unit,
        "config": matrix.config,
        "signatures": {"synthetic_matrix_call_0": [64, 64, 32]},
    }
    if change == "missing_matrix":
        matrix = None
    elif change in ("support_target", "unit", "config"):
        matrix = replace(matrix, **{change: "different"})
    elif change == "signatures":
        sidecar = tmp_path / "synthetic_matrix.json"
        signatures = json.loads(sidecar.read_text())
        signatures["signatures"]["synthetic_matrix_call_0"] = [8, 8, 32]
        sidecar.write_text(json.dumps(signatures))
    else:
        receipt.unlink()
    with pytest.raises(ValueError):
        load_matrix_signatures(tmp_path, matrix)


def test_matrix_handoff_without_receipt_or_signatures_is_empty(synthetic_matrix, tmp_path):
    from merlin.runtime.backends.zephyr_model import load_matrix_signatures

    matrix, _ = synthetic_matrix
    assert load_matrix_signatures(tmp_path, matrix) == {}
    assert load_matrix_signatures(tmp_path, None) == {}


class TestTheGeometryHasOneSource:
    """The selector's tile edge and the compiled kernel's tile edge must be the same number.

    Deriving one from the caller's ``vlen`` and the other from the configuration's Scala would be two
    statements of the same fact, and a disagreement would be silent: the selector would choose
    contractions for a geometry the kernel does not have.
    """

    @pytest.fixture
    def routing(self):
        from merlin.common.paths import env as _env

        if not _env("MERLIN_CHIPYARD"):
            pytest.skip("needs the hardware checkout ($MERLIN_CHIPYARD)")
        return MatrixRouting(support_target="saturn", unit="saturn_opu", config="OPUV256D128ShuttleConfig")

    def test_the_edge_comes_from_the_named_configuration(self, routing):
        assert routing.tile_edge() == 32

    def test_a_wider_configuration_gives_a_wider_edge(self, routing):
        from dataclasses import replace

        assert replace(routing, config="OPUV512D256ShuttleConfig").tile_edge() == 64

    def test_the_default_selector_declines_a_contraction_narrower_than_a_tile(self, routing):
        select = routing.selector()

        class _Big:
            parallel, reduction = (64, 64), (32,)

        class _Small:
            parallel, reduction = (8, 8), (32,)

        assert select(_Big()) and not select(_Small())

    def test_a_supplied_selector_overrides_the_default(self, routing):
        # This is the seam the cost model and the e-graph plug into; nothing here decides profitability.
        from dataclasses import replace

        assert replace(routing, select=lambda _s: False).selector()(object()) is False


class TestTheRewriteAndTheSidecarAgree:
    @pytest.fixture
    def routed(self, tmp_path):
        from merlin.common.paths import env as _env

        if not _env("MERLIN_CHIPYARD"):
            pytest.skip("needs the hardware checkout ($MERLIN_CHIPYARD)")
        matrix = MatrixRouting(support_target="saturn", unit="saturn_opu", config="OPUV256D128ShuttleConfig")
        prepared, _feats = prepare_for_lowering(
            _model(tmp_path),
            tmp_path,
            features=frozenset({OPU_MATMUL_NAME}),
            blocking=False,
            matrix=matrix,
        )
        return prepared, matrix.provider().load_signatures(tmp_path)

    def test_only_the_tile_filling_contraction_moves(self, routed):
        prepared, sigs = routed
        text = prepared.read_text()
        assert text.count("func.call @merlin_opu_gemm_i8") == 1
        assert list(sigs.values()) == [(64, 64, 32)]

    def test_every_symbol_the_module_calls_is_in_the_sidecar(self, routed):
        # THE agreement: a call with no sidecar entry is a link error at image-build time, a long way
        # from the rewrite that created it.
        prepared, sigs = routed
        text = prepared.read_text()
        for sym in sigs:
            assert f"func.call @{sym}" in text
        for line in text.splitlines():
            if "func.call @merlin_opu_gemm_i8" in line:
                called = line.split("func.call @", 1)[1].split("(", 1)[0].strip()
                assert called in sigs, f"{called} is called but not recorded for the build"

    def test_the_declarations_keep_their_access_attributes(self, routed):
        # Without these one-shot-bufferize copies the weight operand of every routed contraction. The
        # printer drops them silently, so the rewrite repairs the text and refuses to write it if it
        # cannot -- this is the assertion that the repair happened.
        prepared, sigs = routed
        text = prepared.read_text()
        assert unpatched_declarations(text, sigs) == ()
