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


@pytest.fixture
def synthetic_gd(tmp_path, monkeypatch):
    import importlib.util
    from types import SimpleNamespace

    common = SimpleNamespace(
        TARGET="fixture",
        REPO=tmp_path,
        RUNS=tmp_path / "runs",
        REPORTS=tmp_path / "reports",
        DESCRIPTOR=tmp_path / "explicit.yaml",
    )
    monkeypatch.setitem(sys.modules, "_common", common)
    path = repo_root() / "merlin/experiments/capsule_bench/harness/generalization_difftest.py"
    spec = importlib.util.spec_from_file_location("synthetic_generalization_difftest", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_numeric_policy_uses_explicit_descriptor_and_returned_verified_experiment(synthetic_gd, monkeypatch):
    from types import SimpleNamespace

    from merlin.targetgen import capsule_runner, corpus_spec

    experiment = SimpleNamespace(target="fixture")
    observed = []

    def resolve(target, *, descriptor):
        assert target == "fixture"
        assert descriptor == synthetic_gd.C.DESCRIPTOR
        return {"compare": "exact_int", "operand_dtype": "i16", "accum_dtype": "i64"}, {}, experiment

    def binding(selected, policy):
        observed.append(selected)
        assert selected is experiment
        assert policy["operand_dtype"] == "i16"
        return SimpleNamespace(operand_dtype="i16", accum_dtype="i64", cap_dtype=lambda value: value)

    monkeypatch.setattr(capsule_runner, "resolve_numeric_policy", resolve)
    monkeypatch.setattr(corpus_spec, "derive_binding", binding)
    assert synthetic_gd.datapath_policy() == {
        "compare": "exact_int",
        "exact": True,
        "operand_dtype": "i16",
        "acc_dtype": "i64",
    }
    assert observed == [experiment]


def test_numeric_policy_omission_and_frozen_verification_failure_refuse(synthetic_gd, monkeypatch):
    from merlin.targetgen import capsule_runner

    monkeypatch.setattr(capsule_runner, "resolve_numeric_policy", lambda *args, **kwargs: (None, None, None))
    with pytest.raises(ValueError, match="numeric_profile"):
        synthetic_gd.datapath_policy()

    def failed(*args, **kwargs):
        raise ValueError("frozen verification failed")

    monkeypatch.setattr(capsule_runner, "resolve_numeric_policy", failed)
    with pytest.raises(ValueError, match="frozen verification failed"):
        synthetic_gd.datapath_policy()


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


@pytest.mark.parametrize(
    "family,probe_name,op",
    [
        ("contraction", "contraction.tile", "matmul"),
        ("normalization", "normalization.tile", "rmsnorm"),
        ("softmax", "softmax.tile", "softmax"),
        ("attention", "attention.tile", "attention_qk"),
    ],
)
def test_materializer_produces_emittable_capsule(gd, family, probe_name, op):
    from merlin.runtime.backends.base import get_backend
    from merlin.targetgen import capability_probes as CP
    from merlin.targetgen import eligibility as EL
    from merlin.targetgen.capsule_common import load_capsule
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir

    probes = {p.name: p for p in CP.synthesize(EL.capability_map_for_target(gd.TARGET))}
    assert probe_name in probes, f"{probe_name} not in the synthesized probe set"
    cdir = gd.FAMILY_MAT[family](probes[probe_name], seed=7)
    assert cdir is not None
    cap = load_capsule(str(cdir), contract=str(gd.CONTRACT))  # schema-valid capsule
    assert cap["operation"]["op"] == op
    cb = parse_interface_mlir((cdir / "capsule.interface.mlir").read_text())
    mlir = get_backend("muon").muon_codegen_mlir.emit_kernel_mlir(cb, target=gd.TARGET)
    assert "llvm.func @" in mlir  # reaches an emitted kernel
    gold = __import__("yaml").safe_load((cdir / "golden.yaml").read_text())
    assert gold["outputs"]["Y0"], "numpy CPU-reference output must be present"


def test_all_four_core_families_are_materializable(gd):
    # the reference backend covers the transformer core via a clean single-op grammar today
    assert set(gd.FAMILY_MAT) == {"contraction", "normalization", "softmax", "attention"}
