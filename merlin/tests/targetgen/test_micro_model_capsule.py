"""The whole-model capsule that is small enough to RUN at the cycle-accurate tier.

Three whole-model capsules exist and none of them can be afforded at the RTL tier — one is 497 KB of
interface MLIR against 6.6 MB of weights — so the corpus's strongest claim (the compiler takes a real
network end to end) is the one claim only a functional oracle ever checked. ``M2_microvit_gemmini`` is
the answer to that: the same architecture shape at the target's own tile edge.

What these tests pin is not "the file exists". It is that the capsule discharges a requirement DERIVED
from the target rather than one its author happened to think of:

  * every ``(family, dtype)`` cell the capability manifest ADMITS is present, and the ones the target can
    execute standalone are actually eligible — an admitted capability a whole-model capsule never reaches
    is a claim nothing tested;
  * the families real captures contain that the target does NOT admit are present too, on the host lane,
    and INTERLEAVED — the composition must contain ``A->H->A``, an accelerator region, a host island and
    another accelerator region, which is the placement decision a whole-model compiler gets wrong and the
    one shape no other capsule in the corpus exercises;
  * every region names a family, because an unnamed region is dropped from the composition and quietly
    shrinks the thing being measured;
  * the footprint stays inside the budget that makes the RTL tier affordable at all — a model capsule on
    another target was blocked outright by a generated ``main.c`` of element-wise tensor initialisers;
  * the golden is not a degenerate answer. A quantized micro-network is one bad scale away from emitting
    all zeros, and an all-zero golden passes every tolerance check while proving nothing.
"""
from __future__ import annotations

import json

import pytest
import yaml

from merlin.common.paths import artifacts_dir, merlin_dir, repo_root
from merlin.targetgen import boundary as BD
from merlin.targetgen import conformance as CF
from merlin.targetgen import eligibility as EL
from merlin.targetgen import micro_model as MM
from merlin.targetgen import model_coverage as MC

TARGET = "gemmini"
CAPSULE = merlin_dir() / "contract/capsules/model/M2_microvit_gemmini"
REFERENCE = merlin_dir() / "contract/capsules/model/M1_lstmnetvit_gemmini"
PROFILE = merlin_dir() / "contract/capsules/profiles" / f"{TARGET}.yaml"

#: What the cycle-accurate tier can afford. Not a style rule: the RTL tier's cost is dominated by how
#: much data the harness has to materialise, and these are the two files that carry it. Both are far
#: under the reference capsule, which is separately asserted so the margin cannot silently erode.
MAX_INTERFACE_BYTES = 200_000
MAX_WEIGHT_BYTES = 200_000


def _capsule() -> dict:
    return yaml.safe_load((CAPSULE / "capsule.yaml").read_text(encoding="utf-8"))


def _regions():
    return MC.regions_from_module(MC.load_module(CAPSULE / "capsule.interface.mlir"))


def _profile_entry() -> dict:
    doc = yaml.safe_load(PROFILE.read_text(encoding="utf-8")) or {}
    for entry in doc.get("capsules") or ():
        if entry.get("name") == CAPSULE.name:
            return entry
    raise AssertionError(f"{CAPSULE.name} is not declared in {PROFILE}; it must be generated, not typed")


#: A capsule's golden and its externalized weights are answer surfaces: they are gitignored, so they are
#: present in a generated checkout and ABSENT in a clean clone or a sandbox worktree. Tests that read
#: them skip there rather than erroring, because "the answer key is masked" is not a capsule defect.
_HAS_GOLDEN = (CAPSULE / "golden.yaml").is_file()
_HAS_WEIGHTS = ((CAPSULE / "capsule.weights.safetensors").is_file()
                and (REFERENCE / "capsule.weights.safetensors").is_file())


def _real_captures() -> dict:
    root = artifacts_dir() / "recaptures"
    if not root.is_dir():
        return {}
    return {d.name: d / "model.mlir" for d in sorted(root.iterdir()) if (d / "model.mlir").is_file()}


# ---------------------------------------------------------------------------------------------------
# the derived inventory
# ---------------------------------------------------------------------------------------------------
def _present_cells() -> set:
    return {(r.resolved_family(), r.in_dtype) for r in _regions()}


def _missing_cells() -> list:
    present = _present_cells()
    return sorted((fam, dt) for fam, dtypes in CF.admitted(TARGET).items() for dt in dtypes
                  if (fam, dt) not in present)


#: The families this capsule was BUILT to reach, and which it must never stop reaching. Kept separate
#: from the full admitted set below so that a target growing a new capability shows up as a visible,
#: named gap rather than silently weakening what is already proven.
_DESIGNED_FOR = ("contraction", "elementwise_map", "movement")


def test_the_capability_cells_the_model_was_built_for_stay_reached():
    present = _present_cells()
    admitted = CF.admitted(TARGET)
    missing = [(fam, dt) for fam in _DESIGNED_FOR for dt in admitted.get(fam, ())
               if (fam, dt) not in present]
    assert not missing, (
        f"{CAPSULE.name} no longer reaches {missing}; the manifest declares the hardware computes those "
        f"cells, so a whole-model capsule that never touches one leaves that claim untested")


def test_every_admitted_capability_cell_is_present_in_the_model():
    """One layer per ``(family, dtype)`` the manifest admits. Derived from the target, not chosen.

    This is a hard functional gate: once the target admits a cell, the whole-model capstone must reach
    it with the admitted dtype rather than carrying a permanent deferral."""
    assert not _missing_cells(), (
        f"{CAPSULE.name} never reaches {_missing_cells()}")


def test_the_families_the_target_can_run_standalone_are_actually_eligible():
    """Present is not the same as reachable. A family the target admits WITHOUT a composition
    requirement must have at least one region the eligibility oracle sends to the accelerator; a family
    it admits only fused (this target declares ``elementwise_map`` that way) must not be asserted
    standalone, because an eligible-looking standalone region would be a demand no compiler can meet."""
    cap_map = EL.capability_map_for_target(TARGET)
    regions = _regions()
    admitted = CF.admitted(TARGET)
    for family in _DESIGNED_FOR:            # the deferred cells are named by the xfail above
        dtypes = admitted.get(family, ())
        assert dtypes, f"{family!r} is no longer admitted by {TARGET}; the model was built around it"
        cap = cap_map.get(family)
        assert cap is not None, f"{family!r} is admitted but absent from the capability map"
        hits = [r for r in regions
                if r.resolved_family() == family and r.in_dtype in dtypes]
        assert hits, f"no {family}/{dtypes} region in {CAPSULE.name}"
        eligible = [r for r in hits if EL.is_eligible(r, cap_map).eligible]
        if cap.composed_with:
            assert not eligible, (
                f"{family} is declared reachable only fused with {list(cap.composed_with)}, so a "
                f"standalone {family} region must NOT read as eligible")
        else:
            assert eligible, (
                f"{CAPSULE.name} has {len(hits)} {family} region(s) but none the target can execute; "
                f"the capability is present in the model and unreachable on the hardware")


def test_the_host_work_is_work_this_target_genuinely_refuses():
    """Host-lane work is not filler, and it is not a routing choice either: every region that lands on
    the host must be one this target's OWN capability map declines, with the decline naming a reason
    (no such family, a dtype it does not compute, or a composition it only offers fused). A capsule
    whose host island is really acceleratable work would be proving a seam that does not exist."""
    cap_map = EL.capability_map_for_target(TARGET)
    reasons: dict = {}
    for region in _regions():
        family = region.resolved_family()
        if family is None:
            continue
        verdict = EL.is_eligible(region, cap_map)
        if not verdict.eligible:
            reasons.setdefault(family, verdict.reason)
    assert reasons, "nothing in the model runs on the host, so it proves no seam at all"
    for family, reason in sorted(reasons.items()):
        assert reason and "unrecognized" not in reason, (
            f"{family} landed on the host for reason {reason!r}, which is not a statement about the "
            f"hardware; host placement must be something the target declared")


@pytest.mark.skipif(not _real_captures(), reason="no real model captures on this checkout")
def test_the_derived_inventory_is_discharged_layer_for_layer():
    """The full derivation — accelerator cells from the manifest, host families from what real captures
    actually contain — checked row by row against the capsule."""
    spec = MM.spec(TARGET, _real_captures())
    present = {(r.resolved_family(), r.in_dtype) for r in _regions()}
    families = {f for f, _ in present}
    unmet = []
    for layer in spec.layers:
        if layer.side == MM.ACCELERATOR:
            want = (layer.family, "int8" if layer.dtype == "i8" else layer.dtype)
            if want not in present and layer.family not in families:
                unmet.append(layer.key())
        elif layer.family not in families:
            unmet.append(layer.key())
    assert not unmet, f"derived layers with no counterpart in {CAPSULE.name}: {unmet}"


def test_extents_are_multiples_of_the_targets_own_tile_edge():
    """Sized against the hardware's geometry, so the model is minimal for THIS target rather than
    minimal for one shape somebody typed."""
    edge = CF.boundaries(TARGET).tile_edge
    assert edge, f"{TARGET} declares no tile edge; the capsule's extents cannot be justified"
    for spec in _capsule()["inputs"]:
        trailing = [int(d) for d in spec["shape"] if int(d) > 1]
        assert trailing, f"input {spec['name']} has no extent to check"
        assert all(d % edge == 0 for d in trailing), (
            f"input {spec['name']} shape {spec['shape']} is not a multiple of the tile edge {edge}")


# ---------------------------------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------------------------------
def test_the_composition_contains_the_seam_nothing_else_exercises():
    profile = BD.profile_capsule(CAPSULE, TARGET)
    assert profile.kind != BD.UNKNOWN, f"the capsule's interface could not be read: {profile.detail}"
    patterns = BD.patterns_in_sequence(_sequence())
    assert BD.A_H_A in patterns, (
        f"{CAPSULE.name} composes as {profile.kind} with patterns {sorted(patterns)}; the point of the "
        f"capsule is the accelerator -> host island -> accelerator seam, and it is not there")
    assert profile.accel_segments >= 2 and profile.host_segments >= 2, (
        f"host work must sit BETWEEN accelerator work, not around it "
        f"(accel_segments={profile.accel_segments}, host_segments={profile.host_segments})")


def _sequence() -> list:
    cap_map = EL.capability_map_for_target(TARGET)
    out = []
    for region in _regions():
        family = region.resolved_family()
        if family is None:
            out.append("?")
            continue
        out.append(BD.ACCEL if (family in cap_map and EL.is_eligible(region, cap_map).eligible)
                   else BD.HOST)
    return out


def test_no_region_goes_unnamed():
    """An unresolved region is DROPPED from the composition, so it silently shrinks what is measured."""
    profile = BD.profile_capsule(CAPSULE, TARGET)
    assert profile.n_unresolved == 0, (
        f"{profile.n_unresolved} region(s) name no semantic family; each one is invisible to the "
        f"boundary axis and to the coverage certificate")


# ---------------------------------------------------------------------------------------------------
# affordable at the cycle-accurate tier
# ---------------------------------------------------------------------------------------------------
def test_it_demands_the_cycle_accurate_tier():
    cap = _capsule()
    assert "L3" in (cap.get("required_oracle_tiers") or []), (
        "the capsule exists to be run at the cycle-accurate tier; not requiring it makes the whole "
        "point optional")
    assert cap.get("label") == "public"
    assert (cap.get("lanes") or {}).get("require"), (
        "a model capsule carries no must_accelerate, so lanes.require is the only thing that can demand "
        "both lanes carried work")


def test_the_required_lanes_are_ones_this_target_can_populate():
    from merlin.targetgen.routing import reachable_lanes

    want = set((_capsule().get("lanes") or {}).get("require") or ())
    have = set(reachable_lanes(TARGET))
    assert want <= have, (
        f"lanes {sorted(want - have)} cannot be populated on {TARGET} (reachable: {sorted(have)}); a "
        f"required lane the router can put nothing on is unpassable however good the backend is")


def test_the_interface_fits_the_tier_it_asks_for():
    iface = (CAPSULE / "capsule.interface.mlir").stat().st_size
    ref_iface = (REFERENCE / "capsule.interface.mlir").stat().st_size
    assert iface <= MAX_INTERFACE_BYTES, f"interface MLIR {iface} B over budget"
    assert iface * 3 < ref_iface, (
        f"interface MLIR {iface} B is not decisively smaller than {REFERENCE.name}'s {ref_iface} B")


@pytest.mark.skipif(not _HAS_WEIGHTS, reason="externalized weights are masked here")
def test_the_weight_footprint_fits_the_tier_it_asks_for():
    """The RTL tier's real cost is the data the harness materialises, and the weights carry most of it."""
    weights = (CAPSULE / "capsule.weights.safetensors").stat().st_size
    ref_weights = (REFERENCE / "capsule.weights.safetensors").stat().st_size
    assert weights <= MAX_WEIGHT_BYTES, f"weights {weights} B over budget"
    assert weights * 20 < ref_weights, (
        f"weights {weights} B is not decisively smaller than {REFERENCE.name}'s {ref_weights} B")


# ---------------------------------------------------------------------------------------------------
# self-contained, and not degenerate
# ---------------------------------------------------------------------------------------------------
def test_the_network_is_defined_in_the_capsule_itself():
    """The reference capsule imports its network from an external checkout named by an env var, so a
    clean clone cannot rebuild it (and its capture is currently broken for an unrelated reason nobody
    can reach). This one must depend on nothing but torch."""
    entry = _profile_entry()
    loader = entry.get("loader")
    assert loader, f"{CAPSULE.name} must name its own loader rather than an out-of-tree workload"
    assert (repo_root() / loader).resolve() == (CAPSULE / "capsule.pytorch.py").resolve(), (
        f"the profile's loader {loader!r} must be the capsule's own capsule.pytorch.py")
    src = (CAPSULE / "capsule.pytorch.py").read_text(encoding="utf-8")
    assert "def get_model_and_inputs" in src
    for forbidden in ("os.environ", "sys.path.insert", "nn.LSTM("):
        assert forbidden not in src, (
            f"{forbidden} in capsule.pytorch.py: the network must be defined inline (nn.LSTM in "
            f"particular is what torch.export refuses on the reference capsule)")


@pytest.mark.skipif(not _HAS_GOLDEN, reason="the golden is masked here")
def test_the_golden_is_not_a_degenerate_answer():
    """A quantized micro-network is one bad scale away from emitting all zeros, and an all-zero golden
    satisfies every tolerance check while proving that nothing computed anything."""
    golden = yaml.safe_load((CAPSULE / "golden.yaml").read_text(encoding="utf-8"))
    values = _flat(golden["outputs"][_capsule()["operation"]["attributes"]["out"]])
    assert values, "the golden carries no output values"
    assert any(v != 0 for v in values), "the golden output is all zeros"
    assert len(set(values)) >= 3, (
        f"the golden output takes only {len(set(values))} distinct value(s); the network collapsed")


def _flat(nested) -> list:
    if isinstance(nested, (list, tuple)):
        return [v for item in nested for v in _flat(item)]
    return [float(nested)]


def test_the_capsule_validates_against_the_capsule_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((merlin_dir() / "contract/schemas/capsule.schema.json").read_text("utf-8"))
    jsonschema.validate(_capsule(), schema)


# --- emitting the model, not just specifying it ------------------------------------------------------
# `spec` derived what a minimal whole-model capsule must contain and stopped there, so the one capsule
# small enough for the cycle-accurate tier was hand-authored and merely CHECKED against the derivation.

def _spec(layers, extent=32, tile=16):
    from merlin.targetgen import micro_model as MM

    return MM.MicroModelSpec(target="t", layers=list(layers), extent=extent, tile_edge=tile)


def _layer(family, side, op="generic"):
    from merlin.targetgen import micro_model as MM

    return MM.LayerRequirement(family=family, dtype="i8", side=side, op=op)


def test_the_emitted_source_parses_and_exposes_the_loader_contract():
    import ast

    from merlin.targetgen import micro_model as MM

    src = MM.emit_pytorch(_spec([_layer("contraction", MM.ACCELERATOR, "matmul"),
                                 _layer("normalization", MM.HOST),
                                 _layer("elementwise_map", MM.ACCELERATOR)]))
    tree = ast.parse(src)
    fns = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "get_model_and_inputs" in fns, "the loader contract every capture path calls"
    assert "forward" in fns


def test_every_layer_in_the_inventory_reaches_the_forward():
    """A layer quietly dropped changes the composition the capsule was written to exercise, so the
    capsule would then test a different shape than its name claims."""
    from merlin.targetgen import micro_model as MM

    layers = [_layer("contraction", MM.ACCELERATOR, "matmul"),
              _layer("normalization", MM.HOST),
              _layer("reduction", MM.ACCELERATOR),
              _layer("movement", MM.ACCELERATOR)]
    src = MM.emit_pytorch(_spec(layers))
    for layer in layers:
        assert f"{layer.side}: {layer.family}" in src, f"{layer.family} never reached the forward"


def test_the_host_layer_lands_in_the_interior():
    """The composition a whole-model compiler actually gets wrong is the round trip, not a host prefix:
    it is where keeping an intermediate resident and paying to move it out differ."""
    from merlin.targetgen import micro_model as MM

    layers = [_layer("contraction", MM.ACCELERATOR, "matmul"),
              _layer("normalization", MM.HOST),
              _layer("elementwise_map", MM.ACCELERATOR)]
    src = MM.emit_pytorch(_spec(layers))
    body = src[src.index("def forward"):]
    sides = [ln.split(":")[0].split("# ")[-1] for ln in body.splitlines() if ln.strip().startswith("# ")]
    assert sides[0] == MM.ACCELERATOR and sides[-1] == MM.ACCELERATOR
    assert MM.HOST in sides[1:-1]


def test_the_extent_comes_from_the_spec_so_geometry_is_never_assumed():
    from merlin.targetgen import micro_model as MM

    src = MM.emit_pytorch(_spec([_layer("contraction", MM.ACCELERATOR, "matmul")], extent=128))
    assert "E = 128" in src


def test_a_spec_with_no_derived_extent_raises_rather_than_defaulting():
    """A default width here would be a geometry this repo does not have."""
    from merlin.targetgen import micro_model as MM

    with pytest.raises(MM.UnwritableLayer):
        MM.emit_pytorch(_spec([_layer("contraction", MM.ACCELERATOR, "matmul")], extent=0))
    with pytest.raises(MM.UnwritableLayer):
        MM.emit_pytorch(_spec([]))


def test_a_family_no_statement_expresses_raises_naming_it():
    from merlin.targetgen import micro_model as MM

    with pytest.raises(MM.UnwritableLayer):
        MM.emit_pytorch(_spec([_layer("no_such_family", MM.ACCELERATOR)]))
