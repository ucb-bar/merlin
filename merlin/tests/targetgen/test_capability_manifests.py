"""Tests for the target-agnostic capability-manifest deriver (rvv/mx_gemmini/radiance/atlas) + routing.

There are NO per-target manifest dicts in core: each manifest is derived by ``manifest_for(name)`` from
the target's ``contracts/residual.yaml`` side-input + family defaults (+ RTL facts for ``atlas``). These
tests pin that the derive path reproduces the residual field-for-field and that ``MANIFESTS`` /
``write_all`` iterate the DISCOVERED targets, not a hardcoded list.
"""
from __future__ import annotations

from merlin.targetgen import capability_manifests as cm
from merlin.targetgen import compute_units as cu
from merlin.targetgen import families as fam
from merlin.targetgen import routing as rt
from merlin.targetgen.target_experiment import _primary_kind


def test_manifests_are_schema_valid():
    for name in cm.MANIFESTS:
        cm.validate(cm.MANIFESTS[name]())   # raises on any problem


def test_manifests_are_discovered_not_a_hardcoded_list():
    # MANIFESTS is derived from the residuals shipped in the target packages, not a literal name map.
    assert sorted(cm.MANIFESTS) == cm.discovered_targets()
    assert {"rvv", "mx_gemmini", "radiance", "atlas"} <= set(cm.discovered_targets())


def test_prototype_manifests_reproduce_residual_plus_inert_family_defaults():
    """OV1 regression: rvv/mx_gemmini/radiance are all-residual prototypes (no RTL facts). The derived
    manifest must reproduce the residual field-for-field and add ONLY the inert, family-derived fields
    the loader used to default (endpoint_kind + runner.suite + runtime.backends) — proving the retired
    hardcoded dicts are byte-reproduced from the residual side-input."""
    for name in ("rvv", "mx_gemmini", "radiance"):
        residual = cm._load_residual(name)
        assert "facts_source" not in residual                # a prototype grounds nothing from RTL
        m = cm.manifest_for(name)
        # every residual field is reproduced verbatim (runtime only GAINS an inert 'backends' key)
        for key, val in residual.items():
            if key == "runtime":
                assert m[key] == {**val, "backends": ["simulator"]}
            else:
                assert m[key] == val, f"{name}.{key} drifted from the residual"
        # the only NEW top-level keys are the family-derived defaults, matching the compute-unit kind
        assert set(m) - set(residual) == {"endpoint_kind", "runner"}
        prof = fam.family_profile(_primary_kind(cu.compute_units(m)))
        assert m["endpoint_kind"] == prof.endpoint_kind_default
        assert m["runner"]["suite"] == f"{name}-capsule-bench"


def test_atlas_manifest_reproduced_from_residual_and_facts():
    """OV1 regression for the facts-grounded target: atlas ships ``facts_source: rtl`` in its residual,
    so ``manifest_for`` layers the RTL facts (mesh/encoding/endpoint) onto the intent+prose residual —
    the same derive path, no hand-authored contract, no per-target builder."""
    m = cm.manifest_for("atlas")
    residual = cm._load_residual("atlas")
    assert residual.pop("facts_source") == "rtl"
    # residual intent/prose is preserved verbatim (facts only AUGMENT mesh/encoding/capacities)
    for key in ("name", "family", "features", "provenance", "compute_units"):
        assert m[key] == residual[key], f"atlas.{key} drifted from the residual"
    # runner intent (model_ext/fourth_output_name) is preserved; only the inert suite default is added
    assert {k: m["runner"][k] for k in residual["runner"]} == residual["runner"]
    assert m["runner"]["suite"] == "atlas-capsule-bench"
    # facts-grounded fields the residual deliberately omits
    assert m["endpoint_kind"] == "external_backend"              # 14-bit decode -> self-hosted ISA
    assert m["capabilities"]["mesh"] == {"rows": 32, "cols": 32}  # from the facts mesh array
    assert len(m["encoding"]["legal_funct"]) == 42               # from the decode table


def test_endpoint_from_facts_covers_rocc_and_self_hosted_isa():
    """endpoint_kind derivation (pure/hermetic): a RoCC ``funct_decode_table`` grounds inline_asm_insn
    (all funct <= 0x7f) vs external_backend (wider); a ``self_hosted_isa`` interface (own instruction
    encoding, no RoCC funct) grounds external_backend — the SIMT analog. No signal -> None (family
    default). No target-name test anywhere."""
    ef = cm._endpoint_from_facts
    assert ef({"interfaces": [{"name": "funct_decode_table", "legal_funct": [0, 3, 126]}]}) == "inline_asm_insn"
    assert ef({"interfaces": [{"name": "funct_decode_table", "legal_funct": [0, 9943]}]}) == "external_backend"
    assert ef({"interfaces": [{"name": "self_hosted_isa", "encoding_bits": 64,
                               "instruction_classes": ["FMA", "TMC"]}]}) == "external_backend"
    # a self_hosted_isa carrying no instruction encoding is not a groundable signal -> None
    assert ef({"interfaces": [{"name": "self_hosted_isa", "instruction_classes": []}]}) is None
    assert ef({"interfaces": []}) is None


def test_radiance_and_mx_gemmini_endpoints_are_derived_not_defaulted():
    """Both endpoints are DERIVED, not the family default (simt AND systolic both default to
    inline_asm_insn): radiance -> external_backend from the SIMT self-hosted ISA (facts_source: simt);
    mx_gemmini -> inline_asm_insn from gemmini's RoCC decode table (facts_source: rtl + facts_target:
    gemmini), which also grounds mesh 16x16 while the MX dtypes stay put (not gemmini int8)."""
    import pytest
    try:
        rad = cm.manifest_for("radiance")
        mxg = cm.manifest_for("mx_gemmini")
    except Exception as e:  # noqa: BLE001 — SIMT introspect / mlc facts unavailable in this env
        pytest.skip(f"manifest derivation unavailable: {type(e).__name__}: {e}")
    assert rad["endpoint_kind"] == "external_backend"            # SIMT self-hosted, NOT the simt default
    assert mxg["endpoint_kind"] == "inline_asm_insn"             # RoCC decode, derived
    assert mxg["capabilities"]["mesh"] == {"rows": 16, "cols": 16}   # from gemmini's facts mesh array
    mxpe = next(u for u in mxg["compute_units"] if u["name"] == "mx_pe")
    assert {"mxfp4", "mxfp6", "mxfp8"} <= set(mxpe["dtypes"])    # MX dtypes preserved, not int8-only


def _units(name):
    return cu.compute_units(cm.MANIFESTS[name]())


def test_rvv_accepts_regular_formats_rejects_low_bit():
    units = _units("rvv")
    ok = rt.route([
        rt.OpDemand("matmul", "int8", "int8"),
        rt.OpDemand("matmul", "fp16", "fp16"),
        rt.OpDemand("matmul", "bf16", "bf16"),
    ], units)
    assert rt.is_fully_routed(ok)
    # RVV has no fp4/fp6/native-fp8 datapath -> honest gaps.
    for fmt in ("mxfp4", "mxfp6", "fp4_e2m1", "fp8_e4m3"):
        res = rt.route([rt.OpDemand("matmul", fmt, fmt)], units)
        assert res[0].gap is not None, fmt


def test_mx_gemmini_accepts_low_bit_and_mixed():
    units = _units("mx_gemmini")
    ok = rt.route([
        rt.OpDemand("matmul", "mxfp4", "mxfp4"),
        rt.OpDemand("matmul", "mxfp6", "mxfp6"),
        rt.OpDemand("matmul", "mxfp8", "mxfp8"),
        rt.OpDemand("matmul", "int8", "int8"),
    ], units)
    assert rt.is_fully_routed(ok)


def test_cross_target_contrast():
    # The same fp4 matmul: gap on RVV, routed on mx_gemmini — the whole point.
    d = [rt.OpDemand("matmul", "mxfp4", "mxfp4")]
    assert rt.route(d, _units("rvv"))[0].gap is not None
    assert rt.route(d, _units("mx_gemmini"))[0].unit == "mx_pe"


def test_write_and_route_target(tmp_path):
    # Writing to a temp base and resolving via a plugged-in path proves the end-to-end plumbing.
    import os
    from merlin.targetgen import target_registry as tr

    cm.write_all(base_root=tmp_path)
    os.environ["MERLIN_TARGET_PATH"] = os.pathsep.join(str(tmp_path / n) for n in cm.MANIFESTS)
    try:
        assert tr.resolve("mx_gemmini").kind == "external"
        res = rt.route_target([rt.OpDemand("matmul", "mxfp6", "mxfp6")], "mx_gemmini")
        assert res[0].unit == "mx_pe" and res[0].acc == "f32"
    finally:
        os.environ.pop("MERLIN_TARGET_PATH", None)


def test_radiance_composes_mx_gemmini():
    # radiance's SIMT cluster CONTAINS the gemmini-mx PE: effective dtypes = regular floats + MX.
    units = cu.compute_units(cm.manifest_for("radiance"))
    simt = next(u for u in units if u.name == "simt_cluster")
    eff = cu.effective(simt, units)
    assert {"fp16", "bf16", "fp32"} <= set(eff.dtypes)          # SIMT regular floats
    assert {"mxfp4", "mxfp6", "mxfp8"} <= set(eff.dtypes)       # via the contained gemmini-mx PE
    # the contained unit is the exact gemmini-mx PE (standalone OR embedded)
    assert cm.manifest_for("mx_gemmini")["compute_units"][0]["name"] == "mx_pe"


def test_radiance_oot_package_discovers_and_routes(tmp_path, monkeypatch):
    from merlin.targetgen import target_registry as tr

    root = cm.write_oot_target("radiance", tmp_path / "radiance")
    assert (root / "contracts" / "target_contract.yaml").is_file()
    assert (root / "contracts" / "dialect_plan.yaml").is_file()
    assert (root / "AGENT.md").is_file()

    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root))
    info = tr.resolve("radiance")
    assert info.kind == "external"
    # plugin block points at the OOT dialect + lowering (Merlin reads, never executes)
    assert info.plugin()["dialect_module"] == "radiance_mlir.dialect"
    # routes both regular floats (SIMT) and low-bit MX (contained gemmini-mx PE)
    assert rt.route_target([rt.OpDemand("matmul", "fp16", "fp16")], "radiance")[0].unit == "simt_cluster"
    assert rt.route_target([rt.OpDemand("matmul", "mxfp6", "mxfp6")], "radiance")[0].unit == "simt_cluster"


def test_dialect_plan_derived_from_units():
    plan = cm.dialect_plan_from_manifest(cm.manifest_for("radiance"))
    assert plan["target"] == "radiance" and plan["dialect_name"] == "radiance"
    assert {t["name"] for t in plan["types"]} == {"simt_cluster_tensor", "mx_pe_tensor"}
    assert {o["name"] for o in plan["ops"]} >= {"matmul", "elementwise"}
