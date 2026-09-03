"""Grouping an application's real shapes by what the compiler must DO with them.

The corpus gives every synthesized capsule one of two shapes (`extents_for` returns `{M: tile, K:
2*tile, N: tile}` or the partial variant, for the whole corpus). Real models do not look like that:
across six captures there are 757 contraction regions carrying 5.5 billion multiply-accumulates,
and their shapes are things like `(8,2048,32000)` -- a vocabulary-32000 LM head -- or
`(256,2304,196)`, a ResNet convolution lowered to a GEMM.

Emitting a capsule per distinct shape would weight a one-off the same as a shape appearing 313
times; a top-N cut would rest on a threshold nobody can defend. So regions group by BEHAVIOUR, and
these tests pin that the grouping is by the axes that decide compiler behaviour, that the
representative is a shape the application really contains, and that the mass is reported rather than
assumed.

The synthetic modules below are deliberate: parsing a real capture takes minutes, and every property
here is about the grouping rather than about any one model.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import applications as APP

_TARGET = "gemmini"


def _module(*matmuls: tuple[int, int, int]) -> str:
    """A linalg module holding one `linalg.matmul` per (M, K, N)."""
    body = []
    for i, (m, k, n) in enumerate(matmuls):
        body.append(f"""
  func.func @f{i}(%a: tensor<{m}x{k}xf32>, %b: tensor<{k}x{n}xf32>) -> tensor<{m}x{n}xf32> {{
    %z = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<{m}x{n}xf32>
    %f = linalg.fill ins(%z : f32) outs(%e : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
    %o = linalg.matmul ins(%a, %b : tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>)
                       outs(%f : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
    return %o : tensor<{m}x{n}xf32>
  }}""")
    return "module {" + "\n".join(body) + "\n}\n"


def _classify(tmp_path, *matmuls, name="app"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    p = d / "model.mlir"
    p.write_text(_module(*matmuls), encoding="utf-8")
    return APP.classify_capture(p, _TARGET)


def test_shapes_that_share_a_behaviour_share_a_capsule(tmp_path):
    """The whole reason to group. Two contractions that tile the same way, spill the same way and
    carry the same arithmetic exercise one code path between them, so one capsule covers both."""
    got = _classify(tmp_path, (64, 64, 64), (128, 128, 128))
    keys = {e.region_class.key() for e in got}
    assert len(keys) == 1, f"expected one behavioural class, got {sorted(keys)}"
    assert got[0].multiplicity == 2


def test_shapes_that_differ_in_behaviour_do_not(tmp_path):
    """A tile-aligned square and an off-edge skinny shape are different work for a compiler --
    different tail handling, different geometry -- so they must not collapse together."""
    got = _classify(tmp_path, (64, 64, 64), (7, 4096, 4096))
    assert len({e.region_class.key() for e in got}) == 2


def test_the_representative_is_a_shape_the_application_really_contains(tmp_path):
    """The property that makes the capsule evidence about the application rather than about a
    heuristic. Nothing here is allowed to invent a geometry."""
    shapes = {(64, 64, 64), (128, 64, 64)}
    got = _classify(tmp_path, *shapes)
    for ev in got:
        assert (ev.m, ev.k, ev.n) in shapes


def test_the_heaviest_member_represents_its_class(tmp_path):
    """Every member exercises the same behaviour by construction, so the one carrying the most work
    is the one whose cost and numerics are worth reproducing."""
    got = _classify(tmp_path, (32, 64, 64), (128, 64, 64))
    heavy = [e for e in got if e.multiplicity == 2]
    if not heavy:
        pytest.skip("the two shapes did not land in one class on this target")
    assert heavy[0].m == 128


def test_alignment_is_unknown_rather_than_aligned_without_a_tile_edge():
    """"There is no edge" and "it lines up with the edge" are different facts, and only one of them
    says anything about whether tails were exercised."""
    assert APP._alignment(16, 16, 16, None) == "unknown"
    assert APP._alignment(16, 16, 16, 16) == "aligned"
    assert APP._alignment(15, 16, 16, 16) == "partial"


def test_a_capture_with_no_readable_contraction_is_evidence_for_nothing(tmp_path):
    """The same rule `conformance.observed` follows: an unreadable model is evidence neither for nor
    against a requirement, so it yields no class rather than an empty-shaped one."""
    d = tmp_path / "broken"
    d.mkdir()
    (d / "model.mlir").write_text("this is not MLIR", encoding="utf-8")
    assert APP.classify_capture(d / "model.mlir", _TARGET) == []


def test_an_unreadable_capture_is_reported_not_skipped(tmp_path):
    """A capture that could not be read must be named. Silently dropping it makes a requirement
    derived from three models indistinguishable from one derived from four."""
    d = tmp_path / "broken2"
    d.mkdir()
    (d / "model.mlir").write_text("nonsense", encoding="utf-8")
    out = APP.classify_captures({"broken2": d / "model.mlir"}, _TARGET)
    # An unparseable module yields no contraction rather than raising, so it lands as zero classes;
    # what must never happen is a silent claim of coverage over it.
    assert out["n_classes"] == 0
    assert out["total_work"] == 0
    assert out["work_coverage"] is None, "no work observed is not full coverage"


def test_the_axis_states_its_basis(tmp_path):
    out = APP.classify_captures({"a": (tmp_path / "x" / "model.mlir")}, _TARGET)
    assert "grouped by what the compiler must DO" in out["axis_basis"]


def test_merging_two_applications_sums_the_mass_and_keeps_the_heavier_shape(tmp_path):
    """Two applications exhibiting the same behaviour is more evidence for it, not two behaviours."""
    a = tmp_path / "a"
    a.mkdir()
    (a / "model.mlir").write_text(_module((64, 64, 64)), encoding="utf-8")
    b = tmp_path / "b"
    b.mkdir()
    (b / "model.mlir").write_text(_module((128, 64, 64)), encoding="utf-8")
    out = APP.classify_captures({"a": a / "model.mlir", "b": b / "model.mlir"}, _TARGET)
    merged = [c for c in out["classes"] if c["multiplicity"] == 2]
    if not merged:
        pytest.skip("the two shapes did not land in one class on this target")
    assert merged[0]["M"] == 128, "the heavier representative survives the merge"


# ------------------------------------------------------------------ sizing and tier assignment

def _evidence(m, k, n, *, mult=1, geometry="squareish_gemm"):
    return APP.ClassEvidence(
        region_class=APP.RegionClass("contraction", "i8", "aligned", "spills", 2, geometry),
        m=m, k=k, n=n, batch=1, multiplicity=mult, work=m * k * n * mult,
        work_complete=True, source="app")


def _fit_or_skip():
    from merlin.targetgen import cert_cost as CC
    fit = CC.fit_for("gemmini")
    if fit is None:
        pytest.skip("no measured certification history in this checkout")
    return fit


def test_a_certifiable_class_yields_a_clamped_capsule_and_the_real_one_extending_it():
    """The two-capsule split the whole design turns on: something small enough to certify
    cycle-accurately, and the application's own shape resting on it."""
    fit = _fit_or_skip()
    caps, refusal = APP.size_class(_evidence(256, 64, 784), target="gemmini",
                                   budget_s=300.0, tile=16, fit=fit)
    assert refusal is None
    tiers = [c.tier for c in caps]
    assert tiers == ["L3", "L2"]
    cert, large = caps
    assert cert.extends is None
    assert large.extends == cert.region_class.key(), "an L2 capsule must name what it rests on"
    assert (large.m, large.k, large.n) == (256, 64, 784), "the L2 capsule is the application's shape"
    assert cert.m <= 256 and cert.n <= 784, "the certified capsule is a clamp, never a growth"


def test_k_is_never_clamped_because_k_is_what_makes_the_class():
    """Accumulation, residency and spill behaviour live in the reduction depth. A shallower K is a
    different behavioural class, so clamping it would silently answer a different question."""
    fit = _fit_or_skip()
    caps, refusal = APP.size_class(_evidence(256, 64, 784), target="gemmini",
                                   budget_s=300.0, tile=16, fit=fit)
    assert refusal is None
    assert all(c.k == 64 for c in caps)


def test_a_class_whose_reduction_depth_alone_blows_the_budget_is_refused_by_name():
    """Measured on a real class: a vocabulary-32000 LM head at K=2048 needs 32768 elements for a
    single tile of parallel extent. There is no size of that class this budget can certify, and
    saying so is the point -- dropping it would make an unaffordable behaviour look absent."""
    fit = _fit_or_skip()
    caps, refusal = APP.size_class(_evidence(8, 2048, 32000), target="gemmini",
                                   budget_s=300.0, tile=16, fit=fit)
    assert caps == []
    assert refusal and "K=2048" in refusal


def test_no_cost_model_means_no_capsule_rather_than_a_convention_sized_one():
    """A tile-convention fallback suggests itself and is wrong. The convention is K=2*tile; this
    class's K is whatever the application carries, so clamping only the parallel extents would not
    be 'the size the corpus already certifies', and clamping K would change the class. A target must
    certify something before application capsules are admitted for it — otherwise the large L2
    capsule rests on a sibling nobody could size."""
    caps, refusal = APP.size_class(_evidence(256, 64, 784), target="definitely_not_a_target",
                                   budget_s=300.0, tile=16, fit=None)
    assert caps == []
    assert refusal and "no measured certification history" in refusal


def test_a_class_already_small_enough_yields_one_capsule_not_two():
    """No L2 extension when the application's own shape is already certifiable — a second capsule at
    the identical shape would be a duplicate wearing a weaker tier."""
    fit = _fit_or_skip()
    caps, refusal = APP.size_class(_evidence(16, 32, 16), target="gemmini",
                                   budget_s=600.0, tile=16, fit=fit)
    assert refusal is None
    assert [c.tier for c in caps] == ["L3"]


def test_the_certified_size_carries_the_evidence_it_was_sized_by():
    """A reader must be able to tell a capsule sized by measurement from one sized by a default,
    which is why there is no default."""
    fit = _fit_or_skip()
    caps, _ = APP.size_class(_evidence(256, 64, 784), target="gemmini",
                             budget_s=300.0, tile=16, fit=fit)
    basis = caps[0].basis
    assert basis["sized_by"] == "measured_cost_model"
    assert basis["budget_s"] == 300.0
    assert basis["fitted_seconds"] is not None and basis["fitted_seconds"] <= 300.0
    assert basis["cost_fit"]["n_samples"] >= 5
    assert basis["clamped_from"] == [256, 64, 784]


def test_the_certified_size_is_whole_tiles():
    """A capsule whose parallel extents are not tile multiples is testing the tail path by accident
    rather than by declaration — the alignment axis is where that belongs."""
    fit = _fit_or_skip()
    caps, refusal = APP.size_class(_evidence(999, 64, 999), target="gemmini",
                                   budget_s=300.0, tile=16, fit=fit)
    if refusal:
        pytest.skip("this class is not certifiable at this budget")
    assert caps[0].m % 16 == 0 and caps[0].n % 16 == 0
