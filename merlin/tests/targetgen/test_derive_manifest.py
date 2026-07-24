"""Tests for the target-agnostic ``derive_manifest`` — a capability manifest reconstructed from CIRCT
facts + family defaults + a small residual (intent/prose) side-input.

The load-bearing proof: ``derive_manifest`` reproduces gemmini's tracked ``target_contract.yaml``
field-by-field, filling the FACTS-derivable fields (mesh / capacities / datapath dtypes+accumulate /
legal-funct codes) from ``facts.json`` while carrying every intent/prose field through the residual.
Hermetic: facts are read from the committed pin via an explicit path (no mlc regeneration).
"""
from __future__ import annotations

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import capability_manifests as cm
from merlin.targetgen import compute_units as cu
from merlin.targetgen import target_registry as tr
from merlin.targetgen.rtl.facts import load_facts

# The facts-grounded fields the deriver LAYERS onto the residual; every other field must reproduce the
# tracked contract verbatim.
_FACTS_FIELDS = {"capabilities", "memory_model", "compute_units", "encoding", "legality"}


@pytest.fixture()
def gemmini_tracked():
    return tr.resolve("gemmini").load_contract()


@pytest.fixture()
def gemmini_facts():
    pin = repo_root() / "merlin" / "targets" / "gemmini" / "contracts" / "rtl_facts" / "facts.json"
    assert pin.is_file(), f"committed gemmini facts pin missing: {pin}"
    return load_facts("gemmini", explicit=pin)


def test_derive_reproduces_gemmini_facts_fields(gemmini_tracked, gemmini_facts):
    # residual = the stripped tracked contract (the proven residual shape).
    m = cm.derive_manifest("gemmini", gemmini_facts, residual=gemmini_tracked)
    body = gemmini_facts["facts"]

    # --- FACTS: capabilities.mesh from the mesh array ---
    arr = next(a for a in body["arrays"] if a["name"] == "mesh")
    assert m["capabilities"]["mesh"] == {"rows": arr["rows"], "cols": arr["cols"]} == {"rows": 16, "cols": 16}

    # --- FACTS: memory capacities from the memory list ---
    for mem in body["memories"]:
        assert m["memory_model"][f"{mem['name']}_bytes"] == mem["bytes"]
    assert m["memory_model"]["scratchpad_bytes"] == 262144
    assert m["memory_model"]["accumulator_bytes"] == 65536
    # ...layered onto (not replacing) the residual's memory_model intent
    assert m["memory_model"]["resident"] is True and m["memory_model"]["accumulators"] is True

    # --- FACTS: compute-unit dtypes + accumulate from datapaths ---
    dp_in = next(d for d in body["datapaths"] if d["name"] == "input")
    dp_acc = next(d for d in body["datapaths"] if d["name"] == "accumulator")
    unit = m["compute_units"][0]
    assert unit["dtypes"] == ["int8"]                       # i8 datapath -> int8 quant-format name
    assert unit["accumulate"] == [{"in": "int8", "weight": "int8", "acc": dp_acc["dtype"]}]
    assert dp_in["dtype"] == "i8" and dp_acc["dtype"] == "i32"

    # --- FACTS: encoding CODES from the funct_decode_table interface ---
    fdt = next(i for i in body["interfaces"] if i["name"] == "funct_decode_table")
    assert m["encoding"]["custom_opcode"] == fdt["custom_opcode"] == 123
    assert m["encoding"]["funct3"] == fdt["funct3"] == 3
    assert m["encoding"]["legal_funct"] == fdt["legal_funct"]


def test_derive_carries_residual_intent_and_prose(gemmini_tracked, gemmini_facts):
    m = cm.derive_manifest("gemmini", gemmini_facts, residual=gemmini_tracked)

    # Every NON-facts field reproduces the tracked contract byte-for-byte (name/family/features/runner/
    # runtime/oracle_ladder/obligations/promises/prototype_scope/notes/...).
    for key, val in gemmini_tracked.items():
        if key in _FACTS_FIELDS:
            continue
        assert m[key] == val, f"non-facts field {key!r} drifted"

    # The encoding ABI sub-block (residual) is preserved; the codes (facts) are ADDED on top.
    for k, v in gemmini_tracked["encoding"].items():
        assert m["encoding"][k] == v
    assert set(m["encoding"]) - set(gemmini_tracked["encoding"]) == {"custom_opcode", "funct3", "legal_funct"}

    # The compute-unit INTENT (name/kind/ops/scaling/requant) comes straight from the residual.
    tracked_u, derived_u = gemmini_tracked["compute_units"][0], m["compute_units"][0]
    for k in ("name", "kind", "ops", "scaling", "requant"):
        assert derived_u[k] == tracked_u[k]


def test_derive_fills_schema_required_defaults(gemmini_tracked, gemmini_facts):
    # The stripped residual omits `legality`; the deriver must supply it so the output is schema-valid.
    assert "legality" not in gemmini_tracked
    m = cm.derive_manifest("gemmini", gemmini_facts, residual=gemmini_tracked)
    assert m["legality"] == []
    cm.validate(m)   # raises on any schema / compute-unit problem


def test_derive_accepts_hand_stub_facts_for_non_arc_target():
    # Non-arc targets have no arc — their "facts" are hand literals (a bare {datapaths,...} stub with
    # no `facts` wrapper). The deriver must accept it and default runner/endpoint from the family.
    hand_stub = {"datapaths": [{"name": "input", "dtype": "i8"},
                               {"name": "accumulator", "dtype": "i32"}]}
    residual = {
        "version": "0.1", "family": "cpu_vector", "status": "prototype",
        "compute_units": [{"name": "vector", "kind": "vector", "ops": ["matmul", "elementwise"],
                           "dtypes": ["fp32", "fp16", "bf16", "int8"], "scaling": "per_channel",
                           "accumulate": [{"in": "fp32", "weight": "fp32", "acc": "f32"}],
                           "requant": {"ref": "x"}}],
        "capabilities": {"ops": ["matmul"]}, "memory_model": {"resident": False},
    }
    m = cm.derive_manifest("toy_vec", hand_stub, residual=residual)
    cm.validate(m)
    # vector family -> upstream_target endpoint + a name-derived suite (no mesh: the stub has no array)
    assert m["endpoint_kind"] == "upstream_target"
    assert m["runner"]["suite"] == "toy_vec-capsule-bench"
    assert "mesh" not in m["capabilities"]
    # the reviewed multi-format matrix is AUGMENTED (int8 already present), never clobbered to one dtype
    assert m["compute_units"][0]["dtypes"] == ["fp32", "fp16", "bf16", "int8"]
    assert m["compute_units"][0]["accumulate"] == [{"in": "fp32", "weight": "fp32", "acc": "f32"}]


def test_derive_synthesizes_unit_from_kind_when_residual_has_none():
    # A truly minimal residual (no compute_units) + a descriptor kind -> a synthesized primary unit.
    stub = {"arrays": [{"name": "mesh", "rows": 8, "cols": 8}],
            "datapaths": [{"name": "input", "dtype": "i8"}, {"name": "accumulator", "dtype": "i32"}]}
    m = cm.derive_manifest({"target": "mini", "kind": "systolic"}, stub, residual={"version": "0.1"})
    cm.validate(m)
    assert m["capabilities"]["mesh"] == {"rows": 8, "cols": 8}
    u = m["compute_units"][0]
    assert u["kind"] == "systolic" and u["dtypes"] == ["int8"]
    assert u["accumulate"] == [{"in": "int8", "weight": "int8", "acc": "i32"}]


def test_existing_manifests_path_unchanged():
    # derive_manifest is ADDITIVE — the hand-written MANIFESTS builders + load path still work.
    for name in cm.MANIFESTS:
        cm.validate(cm.MANIFESTS[name]())
    from merlin.targetgen.target_experiment import load_capability_manifest
    for target in ("atlas", "gemmini"):
        assert load_capability_manifest(target).kind == "systolic"
