"""Rank and operand layout: the two things a `(family, dtype, alignment)` cell cannot say.

`capability_probes` has always turned a target's declared capabilities into region descriptors -- `batch`
into a rank-3 region, `transpose` and each declared layout into an operand-layout variant. Nothing outside
the fuzzer read them, so a target could DECLARE that a unit batches and never be asked for a batched
region by anything that grades it.

These tests hold the axis honest in both directions: where a writer exists the requirement produces a
capsule, and where none does it produces a NAMED hole rather than silence. The second half is the point.
A rank-3 requirement quietly met by a rank-2 capsule, or dropped because nothing could build it, is an
uncovered point that reads as covered.
"""
from __future__ import annotations

import pytest
import yaml

from merlin.common.paths import merlin_dir
from merlin.targetgen import conformance as CF
from merlin.targetgen import corpus_synth as CS

_TARGETS = ["gemmini", "atlas", "radiance", "mx_gemmini"]


def _spec(target: str) -> dict:
    p = merlin_dir() / "contract/capsules/conformance" / f"{target}.yaml"
    if not p.is_file():
        pytest.skip(f"no tracked conformance spec for {target}")
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    doc.setdefault("shape_generalization", CF._shape_axis(target))
    return doc


@pytest.mark.parametrize("target", _TARGETS)
def test_the_axis_carries_only_what_a_cell_cannot_express(target):
    """Shape CORNERS and dtypes are already required under another name -- alignment and the cells
    themselves. Requiring them again would inflate the requirement with points already covered, which is
    how a coverage number goes up without any more of the hardware being exercised."""
    required = (CF._shape_axis(target).get("required") or [])
    for req in required:
        assert req["axis"] in ("rank", "layout")
        if req["axis"] == "rank":
            assert int(req["rank"]) >= 3, req
        else:
            assert req["layout"], req


@pytest.mark.parametrize("target", _TARGETS)
def test_every_declared_region_is_either_a_capsule_or_a_named_hole(target):
    """The whole contract of the axis. A declared capability that produces neither is the silent case
    this exists to end.

    The holes have their own provenance key, apart from the CELL holes: every cell every target requires
    currently has a writer and a standing test asserts it, so folding a declared-but-unbuildable batched
    region into that list would have made it fail for a reason it was never about."""
    doc = _spec(target)
    required = (doc.get("shape_generalization") or {}).get("required") or []
    if not required:
        pytest.skip(f"{target} declares no batched or layout region")
    res = CS.synthesize(doc)
    made = {e["name"] for e in res["capsules"]
            if (e.get("semantic") or {}).get("generalization_axis") in ("rank", "layout")}
    holes = " ".join(res["provenance"].get("shape_regions_no_writer_can_express") or ())
    for req in required:
        probe = req["probe"]
        expected = f"{CS.SYNTH_PREFIX}_{req['axis']}_{probe.replace('.', '_')}"
        assert expected in made or probe in holes, (
            f"{target}: declared region {probe!r} produced neither a capsule nor a reported hole")


def test_a_batched_region_is_refused_where_its_golden_cannot_grade_the_dtype():
    """The one batched builder emits a BLOCK-SCALED contraction, and its golden grades exactly ONE
    format. Both narrowings are needed and each was measured: without the first, a rank-3 entry was
    chosen on every target that declares batching and rejected downstream; without the second, an mxfp4
    cell built its interface and then failed in the golden, leaving a capsule directory with no golden
    in it."""
    pool = CS.available_ops()
    assert CS.op_for_shape("contraction", admitted_ops=pool, dtype="mxfp8", rank=3) == "gemv_batched"
    assert CS.op_for_shape("contraction", admitted_ops=pool, dtype="mxfp4", rank=3) is None
    assert CS.op_for_shape("contraction", admitted_ops=pool, dtype="i8", rank=3) is None
    assert CS.op_for_shape("contraction", admitted_ops=pool, dtype="bf16", rank=3) is None


def test_the_single_format_golden_map_agrees_with_the_golden_itself():
    """`_SINGLE_FORMAT_GOLDEN` is a claim about an engine, so it is checked against it. A golden that
    widened or narrowed its accepted format without this map following would put the axis back to
    choosing an op that cannot grade the cell."""
    import inspect
    import sys

    from merlin.common.paths import repo_root
    sys.path.insert(0, str(repo_root() / "merlin" / "contract" / "capsules"))
    import generate_corpus as GC

    for op, want in CS._SINGLE_FORMAT_GOLDEN.items():
        src = "".join(inspect.getsource(fn) for name, fn in vars(GC).items()
                      if callable(fn) and op in name and name.startswith("_golden"))
        src = src or inspect.getsource(GC)
        assert want in src, f"{op}'s golden no longer mentions {want!r}; the map has drifted"


def test_a_shape_region_takes_its_extents_from_the_builder_not_the_probe():
    """The axis asks whether the unit can be asked for a batched region at all; WHICH extent is the
    alignment axis's question. The probe's tile-relative shape is not necessarily legal for the op that
    expresses the region -- the batched golden needs its contraction dim a multiple of 32 where the probe
    offers one tile -- so passing it through built an interface that then failed in the golden."""
    doc = _spec("mx_gemmini")
    entries = [e for e in CS.synthesize(doc)["capsules"]
               if (e.get("semantic") or {}).get("generalization_axis") == "rank"]
    if not entries:
        pytest.skip("mx_gemmini synthesizes no batched capsule in this checkout")
    for e in entries:
        assert not ({"M", "K", "N"} & set(e)), (
            f"{e['name']} pins extents the axis has no business choosing: "
            f"{ {k: e[k] for k in ('M', 'K', 'N') if k in e} }")
        assert e.get("B", 1) > 1, "a batched region must say how many batches it wants"


def test_a_transposed_layout_has_no_writer_and_says_so():
    """Not a wiring gap. The interface declares a contraction's weight [K, N] while a quantized
    `nn.Linear` stores it [N, K]; the shape check refuses the pair rather than loosening, and closing it
    means teaching the builder the transposed-RHS layout. Until then the axis reports it."""
    pool = CS.available_ops()
    for dtype in ("i8", "bf16", "mxfp4"):
        assert CS.op_for_shape("contraction", admitted_ops=pool, dtype=dtype,
                               layout="transposed") is None


def test_the_batched_op_set_is_what_the_dialect_actually_emits():
    """`_BATCHED_OPS` is a claim about the builders, so it is checked against them rather than restated.
    A builder added or renamed without updating the set would otherwise make the rank axis quietly stop
    finding a writer."""
    import inspect

    from merlin.targetgen import corpus_spec as CSPEC

    emits = {name for name, fn in CSPEC.BUILDERS.items()
             if "matmul_batched" in inspect.getsource(fn)}
    assert emits == set(CS._BATCHED_OPS), (
        f"builders emitting the dialect's batched op are {sorted(emits)}, but the rank axis looks for "
        f"{sorted(CS._BATCHED_OPS)}")


def test_the_batched_capsule_really_carries_a_rank_3_operand():
    """A rank-3 requirement met by a rank-2 capsule is the failure this axis exists to prevent, so the
    capsule is BUILT and its operands inspected rather than trusted from the entry."""
    import sys

    from merlin.common.paths import repo_root
    sys.path.insert(0, str(repo_root() / "merlin" / "contract" / "capsules"))
    import generate_corpus as GC
    from merlin.targetgen import corpus_spec as CSPEC
    from merlin.targetgen.corpora import descriptor_path
    from merlin.targetgen.target_experiment import load_target_experiment

    target = "mx_gemmini"
    doc = _spec(target)
    entries = [e for e in CS.synthesize(doc)["capsules"]
               if (e.get("semantic") or {}).get("generalization_axis") == "rank"]
    if not entries:
        pytest.skip(f"{target} synthesizes no batched capsule in this checkout")
    prof = GC.load_profile(target)
    binding = CSPEC.derive_binding(load_target_experiment(descriptor_path(target)),
                                   prof.get("datapath") or {})
    cap, mlir = CSPEC.build(GC._resolve_flat_extents(entries[0], binding), binding)
    assert "matmul_batched" in mlir, "the capsule must reach the dialect's batched contraction"
    ranks = {len(i["shape"]) for i in cap["inputs"] if i.get("role") in ("input", "weight")}
    assert ranks == {3}, f"batched capsule operands have ranks {ranks}, not rank 3"
    assert cap["operation"]["attributes"].get("batch", 1) > 1


# ------------------------------------------------------------------ the host lane, at full width

@pytest.mark.parametrize("target", _TARGETS)
def test_the_host_lane_axis_keys_on_the_pair_not_the_family(target):
    """`host_only` carries families with NO admitted dtype at all, which on a narrow-format target is
    almost nothing. `host_lane` carries every (family, dtype) the captures contain and the manifest does
    not admit -- which is most of a real model. Measured on gemmini: four admitted families, every one of
    them also present at f32, 17k regions the requirement demanded no capsule of."""
    doc = _spec(target)
    hl = (doc.get("host_lane") or {}).get("required") or []
    if not hl:
        pytest.skip(f"{target} admits every (family, dtype) its captures contain")
    admitted = set((doc.get("host_lane") or {}).get("admitted_pairs") or ())
    for pair in hl:
        assert f"{pair['family']}/{pair['dtype']}" not in admitted, (
            f"{pair} is admitted; the host lane is what the hardware may NOT take")
        assert pair["n_regions"] > 0, "a pair nothing was observed at is not host-lane work"


@pytest.mark.parametrize("target", _TARGETS)
def test_a_host_lane_capsule_is_written_by_the_frontend_or_reported(target):
    """An op with a `merlin_iface` builder cannot serve this axis at any dtype: that dialect is the
    ACCELERATOR's interface, so every program expressible in it classifies as accelerator work and the
    boundary gate reads exactly that file. Measured: a matmul capsule declaring genuine f32 operands
    still came out `A`, while `gelu` and `reduce_sum` -- which have no iface builder, so their interface
    stays linalg -- came out `H`."""
    from merlin.targetgen.corpus_spec import BUILDERS

    doc = _spec(target)
    res = CS.synthesize(doc)
    emitted = [e for e in res["capsules"]
               if (e.get("semantic") or {}).get("generalization_axis") == "host_lane"]
    for e in emitted:
        assert e["op"] not in BUILDERS, (
            f"{e['name']} uses {e['op']!r}, which has an iface builder; its capsule would assert the "
            f"accelerator interface for work that must stay off the accelerator")
        assert e.get("source") == "pytorch"
        assert (e.get("lanes") or {}).get("forbid") == ["on_mesh"]
    # Whatever could not be written is reported by name, never dropped.
    required = {f"{p['family']}/{p['dtype']}" for p in ((doc.get("host_lane") or {}).get("required") or [])}
    narrow = {str(f) for f in ((doc.get("host_only") or {}).get("families") or ())}
    holes = " ".join(res["provenance"].get("host_only_unsynthesizable") or ())
    made = {f"{e['op']}" for e in emitted}
    for key in sorted(required):
        fam = key.split("/")[0]
        if fam in narrow:
            continue                               # the narrow axis carries this family
        assert key in holes or made, f"{target}: {key} produced neither a capsule nor a reported hole"
