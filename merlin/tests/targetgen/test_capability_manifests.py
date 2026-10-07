"""Tests for the target-agnostic capability-manifest deriver (rvv/mx_gemmini/radiance/atlas) + routing.

There are NO per-target manifest dicts in core: each manifest is derived by ``manifest_for(name)`` from
the target's ``contracts/residual.yaml`` side-input + family defaults (+ RTL facts for ``atlas``). These
tests pin that the derive path reproduces the residual field-for-field and that ``MANIFESTS`` /
``write_all`` iterate the DISCOVERED targets, not a hardcoded list.
"""

from __future__ import annotations

import external_sources
import pytest

from merlin.targetgen import capability_manifests as cm
from merlin.targetgen import compute_units as cu
from merlin.targetgen import families as fam
from merlin.targetgen import routing as rt
from merlin.targetgen.rtl.facts import load_facts
from merlin.targetgen.target_experiment import _primary_kind

pytestmark = pytest.mark.target("radiance", "mx_gemmini", "atlas")


@external_sources.requires_rtl("atlas")
def test_manifests_are_schema_valid():
    for name in cm.MANIFESTS:
        cm.validate(cm.MANIFESTS[name]())  # raises on any problem


def test_manifests_are_discovered_not_a_hardcoded_list():
    # MANIFESTS is derived from the residuals shipped in the target packages, not a literal name map.
    assert sorted(cm.MANIFESTS) == cm.discovered_targets()
    assert {"rvv", "mx_gemmini", "radiance", "atlas"} <= set(cm.discovered_targets())


def test_prototype_manifests_reproduce_residual_plus_inert_family_defaults():
    """OV1 regression: an all-residual prototype (no RTL facts) has its derived manifest reproduce the
    residual field-for-field, adding ONLY the inert, family-derived fields the loader used to default
    (endpoint_kind + runner.suite + runtime.backends) — proving the retired hardcoded dicts are
    byte-reproduced from the residual side-input. rvv is the remaining pure prototype; mx_gemmini and
    radiance have since graduated to facts-grounded (facts_source rtl/simt derives their endpoint + mesh
    — see test_radiance_and_mx_gemmini_endpoints_are_derived_not_defaulted)."""
    for name in ("rvv",):
        residual = cm._load_residual(name)
        assert "facts_source" not in residual  # a prototype grounds nothing from RTL
        m = cm.manifest_for(name)
        # every residual field is reproduced verbatim (runtime only GAINS an inert 'backends' key)
        for key, val in residual.items():
            if key == "runtime":
                assert m[key] == {**val, "backends": ["simulator"]}
            else:
                assert m[key] == val, f"{name}.{key} drifted from the residual"
        # the only NEW top-level keys are the family-derived defaults, matching the compute-unit kind,
        # the joined dialect/instruction operation registry, and the capability AUDIT the deriver records
        # every run. These are derivation
        # RESULTS, not curated content: they are deliberately absent from the residual so a stale
        # committed evidence block can never masquerade as fresh.
        assert set(m) - set(residual) == {
            "endpoint_kind",
            "runner",
            "operation_capabilities",
            "capability_evidence",
            "semantic_capabilities_derived",
            "semantic_capabilities_unknown",
            "unmapped_observations",
        }
        prof = fam.family_profile(_primary_kind(cu.compute_units(m)))
        assert m["endpoint_kind"] == prof.endpoint_kind_default
        assert m["runner"]["suite"] == f"{name}-capsule-bench"


@external_sources.requires_rtl("atlas")
def test_atlas_manifest_reproduced_from_residual_and_facts():
    """Atlas intent survives fact derivation without promoting a decoder field to an ISA."""
    m = cm.manifest_for("atlas")
    residual = cm._load_residual("atlas")
    assert residual.pop("facts_source") == "rtl"
    # Residual intent/prose is preserved; observed decode values remain facts, not executable encoding.
    for key in ("name", "family", "features", "provenance"):
        assert m[key] == residual[key], f"atlas.{key} drifted from the residual"
    # compute_units is ADDITIVE rather than verbatim: the generator synthesizes a unit for each engine
    # the target's own evidence reaches and the residual never declared. Every declared unit must still
    # survive UNCHANGED -- authored intent always wins -- and anything extra must carry the rung and the
    # observation that justify it, so a derived unit is never mistaken for a reviewed one.
    declared = {u["name"]: u for u in residual["compute_units"]}
    for unit in m["compute_units"]:
        if unit["name"] in declared:
            assert unit == declared[unit["name"]], f"declared unit {unit['name']} was edited"
        else:
            assert unit.get("derived_from"), f"synthesized unit {unit['name']} carries no evidence"
    # runner intent (model_ext/fourth_output_name) is preserved; only the inert suite default is added
    assert {k: m["runner"][k] for k in residual["runner"]} == residual["runner"]
    assert m["runner"]["suite"] == "atlas-capsule-bench"
    # The default selector may serve an older unverified cache that lacks mesh
    # evidence. Assert exact projection of the selected facts, never a geometry
    # borrowed from a different, explicitly selected source bundle.
    selected = load_facts("atlas")
    assert m["endpoint_kind"] == "unresolved"
    assert m["capabilities"].get("mesh") == cm._mesh_from_facts(cm._facts_body(selected))
    assert "legal_funct" not in m.get("encoding", {})


def test_endpoint_from_facts_covers_rocc_and_self_hosted_isa():
    """A field-local comparison set is not an endpoint; an explicit interface is."""
    ef = cm._endpoint_from_facts
    assert ef({"interfaces": [{"name": "funct_decode_table", "legal_funct": [0, 3, 126]}]}) is None
    assert ef({"interfaces": [{"name": "funct_decode_table", "legal_funct": [0, 9943]}]}) is None
    assert (
        ef(
            {
                "interfaces": [
                    {
                        "name": "funct_decode_table",
                        "scope": "complete_rocc_funct7",
                        "complete_isa": True,
                        "custom_opcode": 123,
                        "legal_funct": [0, 3, 126],
                    }
                ]
            }
        )
        == "inline_asm_insn"
    )
    assert (
        ef({"interfaces": [{"name": "self_hosted_isa", "encoding_bits": 64, "instruction_classes": ["FMA", "TMC"]}]})
        == "external_backend"
    )
    # a self_hosted_isa carrying no instruction encoding is not a groundable signal -> None
    assert ef({"interfaces": [{"name": "self_hosted_isa", "instruction_classes": []}]}) is None
    assert ef({"interfaces": []}) is None


def test_rocc_endpoint_uses_command_transport_not_observed_funct_width():
    observed = {
        "name": "funct_decode_table",
        "legal_funct": [87, 9943],
        "scope": "observed_decode_field",
        "complete_isa": False,
        "custom_opcode": 123,
    }
    assert cm._endpoint_from_facts({"interfaces": [observed]}) is None
    assert cm._endpoint_from_facts({"interfaces": [{"name": "rocc_cmd"}, observed]}) == "inline_asm_insn"


def test_required_executable_facts_cannot_fall_back_to_family_endpoint():
    residual = {"compute_units": [{"name": "core", "kind": "simt", "ops": ["matmul"], "dtypes": ["fp32"]}]}
    unknown = cm.derive_manifest({"target": "unbound", "facts_source": "simt"}, {}, residual=residual)
    assert unknown["endpoint_kind"] == "unresolved"
    assert unknown["endpoint_resolution"]["status"] == "unverified"


@external_sources.requires_ext("chipyard")
def test_radiance_and_mx_gemmini_endpoints_are_derived_not_defaulted():
    """Use selected transport evidence; a missing SIMT provider cannot become RoCC by default."""
    from merlin.targetgen.rtl import mlc_bridge

    rad = cm.manifest_for("radiance")
    mxg = cm.manifest_for("mx_gemmini")
    gemmini_facts = cm._facts_body(load_facts("gemmini"))
    interfaces = gemmini_facts.get("interfaces") or []
    assert any(i.get("name") == "rocc_cmd" for i in interfaces)
    assert any(i.get("name") == "funct_decode_table" and i.get("custom_opcode") == 123 for i in interfaces)
    assert mxg["endpoint_kind"] == "inline_asm_insn"  # command transport + selected custom slot
    assert mxg["capabilities"].get("mesh") == cm._mesh_from_facts(gemmini_facts)
    mxpe = next(u for u in mxg["compute_units"] if u["name"] == "mx_pe")
    assert {"mxfp4", "mxfp6", "mxfp8"} <= set(mxpe["dtypes"])  # MX dtypes preserved, not int8-only
    simt = mlc_bridge.simt_facts("radiance")
    if simt:
        assert cm._endpoint_from_facts(cm._facts_body(simt)) == "external_backend"
        assert rad["endpoint_kind"] == "external_backend"
        assert rad["capabilities"]["simt"]["lanes_per_warp"] == 16
        assert rad["memory_model"].get("shared_memory_bytes") == 131072
    else:
        assert rad["endpoint_kind"] == "unresolved"
        assert rad["endpoint_resolution"]["status"] == "unverified"
        assert "simt" not in rad.get("capabilities", {})


def _units(name):
    return cu.compute_units(cm.MANIFESTS[name]())


def test_rvv_accepts_regular_formats_rejects_low_bit():
    units = _units("rvv")
    ok = rt.route(
        [
            rt.OpDemand("matmul", "int8", "int8"),
            rt.OpDemand("matmul", "fp16", "fp16"),
            rt.OpDemand("matmul", "bf16", "bf16"),
        ],
        units,
    )
    assert rt.is_fully_routed(ok)
    # RVV has no fp4/fp6/native-fp8 datapath -> honest gaps.
    for fmt in ("mxfp4", "mxfp6", "fp4_e2m1", "fp8_e4m3"):
        res = rt.route([rt.OpDemand("matmul", fmt, fmt)], units)
        assert res[0].gap is not None, fmt


@external_sources.requires_ext("chipyard")
def test_mx_gemmini_accepts_low_bit_and_mixed():
    units = _units("mx_gemmini")
    ok = rt.route(
        [
            rt.OpDemand("matmul", "mxfp4", "mxfp4"),
            rt.OpDemand("matmul", "mxfp6", "mxfp6"),
            rt.OpDemand("matmul", "mxfp8", "mxfp8"),
            rt.OpDemand("matmul", "int8", "int8"),
        ],
        units,
    )
    assert rt.is_fully_routed(ok)


@external_sources.requires_ext("chipyard")
def test_cross_target_contrast():
    # The same fp4 matmul: gap on RVV, routed on mx_gemmini — the whole point.
    d = [rt.OpDemand("matmul", "mxfp4", "mxfp4")]
    assert rt.route(d, _units("rvv"))[0].gap is not None
    assert rt.route(d, _units("mx_gemmini"))[0].unit == "mx_pe"


@external_sources.requires_rtl("atlas")
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


def test_target_resolution_is_read_only_and_materialization_is_explicit(tmp_path, monkeypatch):
    """Resolution never derives or deletes a contract; generation needs a new, explicit destination."""
    from merlin.targetgen import target_registry as tr

    name = "synth_materialize_boundary"
    destination = tmp_path / "generated" / name
    calls = []

    def write_support_package(selected, root):
        calls.append((selected, root))
        contracts = root / "contracts"
        contracts.mkdir(parents=True)
        (contracts / "target_contract.yaml").write_text(f"name: {selected}\n", encoding="utf-8")
        return root

    monkeypatch.delenv("MERLIN_TARGET_PATH", raising=False)
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    monkeypatch.setenv("MERLIN_TARGETS_DIR", str(tmp_path / "references"))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setattr(cm, "write_oot_target", write_support_package)

    missing = tr.resolve(name)
    assert not missing.contract_path.exists()
    assert calls == []

    info = tr.materialize(name, destination=destination)
    assert calls == [(name, destination)]
    assert info.kind == "external"
    assert info.contract_path == destination / "contracts" / "target_contract.yaml"
    assert info.load_contract() == {"name": name}
    with pytest.raises(FileExistsError, match="requires a new destination"):
        tr.materialize(name, destination=destination)
    assert calls == [(name, destination)]


@external_sources.requires_ext("chipyard")
def test_radiance_composes_mx_gemmini():
    # radiance's SIMT cluster CONTAINS the gemmini-mx PE: effective dtypes = regular floats + MX.
    units = cu.compute_units(cm.manifest_for("radiance"))
    simt = next(u for u in units if u.name == "simt_cluster")
    eff = cu.effective(simt, units)
    assert {"fp16", "bf16", "fp32"} <= set(eff.dtypes)  # SIMT regular floats
    assert {"mxfp4", "mxfp6", "mxfp8"} <= set(eff.dtypes)  # via the contained gemmini-mx PE
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
