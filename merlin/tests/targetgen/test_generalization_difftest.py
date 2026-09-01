"""The generalization differential-test materializers (Phase-D measurement loop).

Each capability-derived probe must materialize into a self-contained capsule (interface MLIR + numpy
CPU-reference golden) that LOADS and EMITS a kernel through the reference backend — that is the loop that
turns the probe generators into an actual unseen-workload recall number. These are $0 checks (materialize +
emit only, no oracle run); the numeric recall itself is produced by running the harness module against
cyclotron. Skips cleanly when the radiance descriptor / reference package is not present in this checkout.
"""
from __future__ import annotations

import os
import sys

import pytest

from merlin.common.paths import repo_root

_DESC = repo_root() / "merlin/experiments/capsule_bench/targets/radiance/target_experiment.yaml"
_PKG = repo_root() / "out/artifacts/targets/radiance/reference_v0"


@pytest.fixture(scope="module")
def gd():
    if not _DESC.is_file() or not _PKG.exists():
        pytest.skip("radiance descriptor / reference_v0 package not present")
    # Undone when the module finishes. Setting os.environ directly leaked this descriptor into every
    # later test in the process -- the same defect that made test_model_grade.py's eight mesh-verdict
    # guards fail in the full suite while passing in isolation. It is latent here rather than active
    # only because the fixture skips when reference_v0 is absent.
    mp = pytest.MonkeyPatch()
    mp.setenv("MERLIN_TARGET_EXPERIMENT", str(_DESC))
    mp.syspath_prepend(str(repo_root() / "merlin/experiments/capsule_bench/harness"))
    import generalization_difftest as G  # noqa: PLC0415
    yield G
    mp.undo()


@pytest.mark.parametrize("family,probe_name,op", [
    ("contraction", "contraction.tile", "matmul"),
    ("normalization", "normalization.tile", "rmsnorm"),
    ("softmax", "softmax.tile", "softmax"),
    ("attention", "attention.tile", "attention_qk"),
])
def test_materializer_produces_emittable_capsule(gd, family, probe_name, op):
    from merlin.targetgen import capability_probes as CP
    from merlin.targetgen import eligibility as EL
    from merlin.targetgen.capsule_common import load_capsule
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir
    from merlin.runtime.backends.base import get_backend

    probes = {p.name: p for p in CP.synthesize(EL.capability_map_for_target(gd.TARGET))}
    assert probe_name in probes, f"{probe_name} not in the synthesized probe set"
    cdir = gd.FAMILY_MAT[family](probes[probe_name], seed=7)
    assert cdir is not None
    cap = load_capsule(str(cdir), contract=str(gd.CONTRACT))            # schema-valid capsule
    assert cap["operation"]["op"] == op
    cb = parse_interface_mlir((cdir / "capsule.interface.mlir").read_text())
    mlir = get_backend("muon").muon_codegen_mlir.emit_kernel_mlir(cb, target=gd.TARGET)
    assert "llvm.func @" in mlir                                         # reaches an emitted kernel
    gold = __import__("yaml").safe_load((cdir / "golden.yaml").read_text())
    assert gold["outputs"]["Y0"], "numpy CPU-reference output must be present"


def test_all_four_core_families_are_materializable(gd):
    # the reference backend covers the transformer core via a clean single-op grammar today
    assert set(gd.FAMILY_MAT) == {"contraction", "normalization", "softmax", "attention"}
