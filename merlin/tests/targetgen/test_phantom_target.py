"""Phantom-target acid test: adding a NEW hardware target is DATA ONLY (a descriptor + fake facts + a
plugin module), never an edit to shared library code.

``phantom5`` is a SYNTHETIC 5th target (fixtures under ``merlin/tests/fixtures/phantom_target/phantom5``)
modeling a NOVEL accelerator that shares no code-path assumption with the four shipped targets: an
**fp16** systolic engine (float, not gemmini int8), an **8x8** mesh (not 16x16), and a RoCC wired to
custom **slot 1** (major opcode ``0x2b``) rather than gemmini's slot 3 (``0x7b``). If the target-agnostic
deriver/router/oracle-discovery are truly un-overfit, all of the following fall out of the fixture DATA
with zero shared-code change — and this test drives exactly those paths.

The headline proof: the SAME ``circt_introspect._rocc_custom_opcode`` slot->opcode derivation yields
``0x2b`` for phantom5's slot 1 and ``0x7b`` for gemmini's slot 3 — the opcode is slot-derived, never a
baked gemmini constant.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import yaml

from merlin.common.paths import merlin_dir
from merlin.targetgen import capability_manifests as cm
from merlin.targetgen import compute_units as cu
from merlin.targetgen import routing
from merlin.targetgen.rtl import circt_introspect as ci
from merlin.targetgen.target_experiment import _derived_dtype_token

_FIXTURE = merlin_dir() / "tests" / "fixtures" / "phantom_target" / "phantom5"
_CONTRACT = _FIXTURE / "contracts" / "target_contract.yaml"
_FACTS = _FIXTURE / "contracts" / "rtl_facts" / "facts.json"


def _load_contract() -> dict:
    return yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))


def _load_facts() -> dict:
    return json.loads(_FACTS.read_text(encoding="utf-8"))


def _funct_table(facts: dict) -> dict:
    return next(i for i in facts["facts"]["interfaces"] if i["name"] == "funct_decode_table")


def _stamp_opcode_from_slot(facts: dict, monkeypatch) -> int:
    """Mimic the ONE thing circt_introspect does during facts extraction that our hand-authored fixture
    facts cannot: stamp the RoCC major opcode DERIVED from the contract's ``rocc_custom_slot``. We call
    the REAL derivation (``_rocc_custom_opcode``) — not a literal — pointing it at the fixture contract,
    so the opcode that lands in the facts is honestly slot-derived, exactly as it would be on a real
    onboarding. Returns the derived opcode."""
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(_CONTRACT))
    opcode = ci._rocc_custom_opcode("phantom5")
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    _funct_table(facts)["custom_opcode"] = opcode
    return opcode


# --------------------------------------------------------------------------- slot -> opcode derivation


def test_rocc_opcode_is_slot_derived_not_a_gemmini_literal(monkeypatch):
    """Proves de-overfit 2a: the RoCC major opcode is DERIVED from the target's ``rocc_custom_slot`` via
    the RISC-V standard slot->opcode map, NOT a baked gemmini ``0x7b``. The SAME ``_rocc_custom_opcode``
    code returns ``0x2b`` (custom-1) for phantom5's slot 1 and ``0x7b`` (custom-3) for gemmini's slot 3 —
    the headline proof that the opcode is a fact of the plugged-in target, not a code literal."""
    # gemmini (real reference contract, slot 3) — read with NO env override.
    assert ci._rocc_custom_opcode("gemmini") == 0x7B == 123
    # phantom5 (fixture contract, slot 1) — point the contract resolver at the fixture.
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(_CONTRACT))
    assert ci._rocc_custom_opcode("phantom5") == 0x2B == 43
    # the mapping itself is the RISC-V standard custom-0/1/2/3 encoding, not per-target.
    assert ci._RISCV_CUSTOM_OPCODES == {0: 0x0B, 1: 0x2B, 2: 0x5B, 3: 0x7B}


def test_undeclared_slot_fails_closed(monkeypatch, tmp_path):
    """Proves the fail-closed contract of 2a: a target that declares NO ``rocc_custom_slot`` gets a
    ``None`` opcode (UNKNOWN, surfaced) — never a silent fallback to gemmini's custom-3."""
    noslot = tmp_path / "target_contract.yaml"
    noslot.write_text(yaml.safe_dump({"name": "noslot", "encoding": {"addr_len": 32}}), encoding="utf-8")
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(noslot))
    assert ci._rocc_custom_opcode("noslot") is None


# --------------------------------------------------------------------------- derive_manifest (data-only)


def test_derive_manifest_is_pure_data_for_a_novel_target(monkeypatch):
    """Proves the core claim: a NOVEL fp16 8x8 slot-1 target's whole capability manifest is DERIVED from
    its fixture DATA (facts + contract residual) by the SAME ``derive_manifest`` every target uses — no
    shared-code edit. Asserts every facts-grounded field lands: systolic kind, inline_asm_insn endpoint
    (from the <=0x7f funct table, 2a's decode-width signal), the 8x8 mesh (NOT 16), the fp16-derived run
    dtype token (2b), and — the headline — ``custom_opcode == 0x2b`` (slot-1 derived, not gemmini 0x7b)."""
    facts = _load_facts()
    opcode = _stamp_opcode_from_slot(facts, monkeypatch)   # honest slot->opcode, mimics introspection
    assert opcode == 0x2B

    m = cm.derive_manifest({"target": "phantom5"}, facts, residual=_load_contract())
    cm.validate(m)

    units = cu.compute_units(m)
    # kind: systolic, from the residual's primary compute unit.
    assert units[0].kind == "systolic"
    # endpoint: inline_asm_insn, DERIVED from the funct table (all legal functs <= 0x7f => RoCC .insn).
    assert m["endpoint_kind"] == "inline_asm_insn"
    # mesh: 8x8, DERIVED from the facts ``mesh`` array — proving the 16 in gemmini's contract is not baked.
    assert m["capabilities"]["mesh"] == {"rows": 8, "cols": 8}
    # run dtype token (2b): DERIVED from the fp16 primary accumulate rule, since runner.dtype is omitted.
    token = _derived_dtype_token(units)
    assert token == "fp16xfp16_f32" and "fp16" in token
    # HEADLINE: the encoding opcode is the slot-1 derivation (0x2b/43), NOT gemmini's slot-3 0x7b/123.
    assert m["encoding"]["custom_opcode"] == 0x2B == 43
    assert m["encoding"]["custom_opcode"] != 0x7B          # explicitly NOT the gemmini opcode
    assert m["encoding"]["rocc_custom_slot"] == 1          # residual ABI fact carried through
    assert m["encoding"]["legal_funct"] == [0, 2, 4, 6]    # facts codes


def test_derive_manifest_grounds_float_datapath_not_int8(monkeypatch):
    """Proves the datapath dtypes/accumulate are FACTS, not a baked int8 default: the fp16 input + f32
    accumulator facts ground the primary unit's ``dtypes``/``accumulate`` — the exact place gemmini would
    read int8/i32 — with no code that assumes one target's numeric format."""
    facts = _load_facts()
    _stamp_opcode_from_slot(facts, monkeypatch)
    m = cm.derive_manifest({"target": "phantom5"}, facts, residual=_load_contract())
    primary = cu.compute_units(m)[0]
    assert primary.dtypes == ("fp16",)                     # fp16, not int8
    assert [{"in": r.inp, "weight": r.weight, "acc": r.acc} for r in primary.accumulate] == \
        [{"in": "fp16", "weight": "fp16", "acc": "f32"}]


# --------------------------------------------------------------------------- routing (pure data lookup)


def test_routing_splits_ops_across_lanes_by_data_only(monkeypatch):
    """Proves routing is a pure lookup over the derived contract's compute units, target-agnostic: a
    matmul routes to the systolic MESH unit and an elementwise ``add`` routes to the in-contract
    scalar/vector lane — no shared-code branch on the target. The lane assignment falls out of the
    fixture's two compute units, nothing more."""
    facts = _load_facts()
    _stamp_opcode_from_slot(facts, monkeypatch)
    units = cu.compute_units(cm.derive_manifest({"target": "phantom5"}, facts, residual=_load_contract()))

    plan = routing.route_plan_on(
        [routing.OpDemand("matmul", "fp16", "fp16"), routing.OpDemand("add", "fp16")], units)
    # matmul -> the systolic mesh; add (elementwise, unsupported on the mesh) -> the vector lane.
    assert [r.unit for r in plan["mesh"]] == ["systolic_mesh"]
    assert [r.unit for r in plan["fallback"]] == ["vector_lane"]
    assert plan["scalar_rvv"] == []                        # nothing gapped: both ops found a legal lane

    # And the matched matmul rule carries the fp16 accumulate: the router reads the fact, not a default.
    matmul = plan["mesh"][0]
    assert matmul.acc == "f32" and matmul.gap is None


# --------------------------------------------------------------------------- plugin.sim_oracle discovery
#
# Discovery mutates process-global state (capsule_runner._SIM_ORACLES + sys.modules), so each case runs
# in a FRESH interpreter via subprocess with a tailored MERLIN_TARGET_PATH — mirroring
# test_oot_backend_discovery. Running it in-process would leak the fixture oracle into every other test.

_ORACLE_PROBE = """
import json
import merlin.targetgen.capsule_runner as cr
cr._ensure_sim_oracles_discovered()
so = cr._SIM_ORACLES.get("phantomsim")
print(json.dumps({
    "engines": sorted(cr._SIM_ORACLES),
    "phantom_present": "phantomsim" in cr._SIM_ORACLES,
    "exclusive": getattr(so, "exclusive", None),
    "available": list(so.available("phantom5")) if so else None,
}))
"""


def _run_probe(target_path: str | None) -> dict:
    env = dict(os.environ)
    env.pop("MERLIN_TARGET_PATH", None)
    if target_path is not None:
        env["MERLIN_TARGET_PATH"] = target_path
    proc = subprocess.run([sys.executable, "-c", _ORACLE_PROBE], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, f"subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    return json.loads([ln for ln in proc.stdout.splitlines() if ln.strip()][-1])


def test_sim_oracle_wires_in_from_the_target_path_with_zero_core_edit():
    """Proves de-overfit 2d END-TO-END: a target contributes a bespoke-sim ORACLE as DATA (its contract's
    ``plugin.sim_oracle`` module), and ``_ensure_sim_oracles_discovered`` loads it via the SAME OOT plugin
    discovery — so ``phantomsim`` appears in ``_SIM_ORACLES`` with no edit to the core registry literal.
    The distinct engine name uniquely attributes the entry to the fixture; its fail-closed ``available``
    proves the registered adapter is the fixture's own."""
    res = _run_probe(str(_FIXTURE))
    assert res["phantom_present"] is True, res["engines"]
    assert res["exclusive"] is True
    assert res["available"] == [False, "phantom sim: test fixture, not runnable"]
    # additive — the in-tree built-in engines are untouched.
    assert "chipyard" in res["engines"] and "cyclotron" in res["engines"]


def test_sim_oracle_absent_without_the_target_path():
    """Sanity/negative: with MERLIN_TARGET_PATH unset, the fixture oracle is NOT present — discovery is
    honest (data-driven), not a hardcoded fallback. Proves the built-ins never silently gain phantomsim."""
    res = _run_probe(None)
    assert res["phantom_present"] is False, res["engines"]
    assert "chipyard" in res["engines"]                    # the core registry is still intact
