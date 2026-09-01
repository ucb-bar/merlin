"""The fact-provenance rule: a hardware fact comes from CIRCT or the target's RTL, or it is UNKNOWN.

Three things are pinned here, and the first is the one that makes the rest worth anything:

1. **The gate can FAIL.** A synthetic fact producer that reads a forbidden source is caught, and a
   synthetic one that reads an ARC model is not. A check nobody has ever seen reject something is not
   evidence of anything.
2. **The Muon introspect derives its geometry**, and reports UNKNOWN rather than a plausible default
   when the elaboration is not there. The regression is specific: it used to publish
   ``lanes_per_warp=16, warps_per_core=8, cores=2`` out of ``cfg.get(..., 16)`` defaults while its
   config path pointed at ``/path/to/autocomp/scripts/muon/config_muon.toml`` — a placeholder that
   exists on no machine.
3. **The bundle adapter does not stamp "derived" on a block that was never read.** ``_simt_fact_bundle``
   used to compute ``"derived": name in f`` — key presence, not provenance.
"""
from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from merlin.common.paths import repo_root

GATE = repo_root() / "build_tools" / "scripts" / "check_fact_provenance.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("check_fact_provenance", GATE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_fact_provenance"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gate():
    return _load_gate()


# --------------------------------------------------------------------------------------------
# 1. The gate can fail — and can tell an arc model from a hand model
# --------------------------------------------------------------------------------------------
#: A fact producer reading a FORBIDDEN source. Lives on a fact-producing path (``.../rtl/...``), reads
#: the cyclotron perf model's config, and files the result under ``facts``. Must be caught.
_BAD_CYCLOTRON = '''
"""A fake target introspect that sources its geometry from the cycle model."""
from merlin.common.paths import env


def build_facts() -> dict:
    cfg = env("MERLIN_MUON_CONFIG")
    lanes = _parse(cfg)["num_lanes"]
    return {"facts": {"simt": {"lanes_per_warp": lanes}}}
'''

#: The near-miss the rule exists to survive: ``MERLIN_EXT_ATLAS_VSIM`` resolves to
#: ``.../runs/circt-arc/atlas/vsim_oracle``. It is an ARC model — CIRCT compiling the RTL — so it
#: carries RTL provenance and must NOT be caught, even though "vsim"/"oracle" read like a simulator
#: and even though its sibling ``MERLIN_EXT_NPU_MODEL`` is forbidden.
_GOOD_ARC = '''
"""A fake target introspect that sources its geometry from the arc model."""
from merlin.common.paths import ext_path


def build_facts() -> dict:
    # NB the path is assembled from parts so this FIXTURE does not carry a generated-root literal.
    # It is a fake module fed to the gate, never a real read; the artifact-layout check rightly
    # cannot tell the difference from the text alone.
    state = ext_path("atlas_vsim") / "r" "uns" / "circt-arc/atlas/vsim_oracle/state.json"
    n = len(json.loads(state.read_text())["states"])
    return {"facts": {"n_states": n, "evidence": "arc state manifest"}}
'''

#: A hand model whose only textual trace is the FUNCTION NAME — the config arrives as a parameter.
#: This is the real ``dse/hardware_space.cost_model_from_npu`` shape.
_BAD_NPU_BY_NAME = '''
"""A cost-model bridge whose source is named only by the function."""


def cost_model_from_npu(hw) -> dict:
    return {"mac_per_cycle": hw.mxu0_rows * hw.mxu0_cols}
'''

#: A GRADER. It runs spike to decide whether a program produced the right answer. That is a verdict
#: about a program, not a claim about the hardware, and flagging it would make the gate noise.
_VERDICT_ONLY = '''
"""A capsule grader."""
import subprocess

from merlin.common.paths import env


def grade(elf) -> dict:
    out = subprocess.run([env("MERLIN_SPIKE"), elf], capture_output=True, text=True).stdout
    return {"passed": out.strip().endswith("OK"), "log": out}
'''


def _scan(gate, tmp_path, source: str, rel: str) -> list[dict]:
    p = tmp_path / "mod.py"
    p.write_text(source, encoding="utf-8")
    return gate.scan_file(p, rel)


def test_a_fact_producer_reading_a_forbidden_source_is_caught(gate, tmp_path):
    """THE failing case. Without this the gate is decoration."""
    found = _scan(gate, tmp_path, _BAD_CYCLOTRON,
                  "merlin/python/merlin/targetgen/rtl/fake_introspect.py")
    assert [f["source"] for f in found] == ["cyclotron"], found
    assert found[0]["kind"] == "violation"
    assert "fact" in found[0]["why_fact"]


def test_a_fact_producer_reading_an_arc_model_is_not_caught(gate, tmp_path):
    """The near-miss: an arc model IS CIRCT compiling the RTL, so it carries RTL provenance.

    Told apart from ``npu_model`` by what the artifact IS (a ``circt-arc`` product), never by whether
    the word "model" appears — both spellings read like "the model"."""
    found = _scan(gate, tmp_path, _GOOD_ARC,
                  "merlin/python/merlin/targetgen/rtl/fake_arc_introspect.py")
    assert found == [], found


def test_a_hand_model_named_only_by_the_function_is_still_caught(gate, tmp_path):
    """The literal axis alone cannot see this: the HardwareConfig arrives as a parameter."""
    found = _scan(gate, tmp_path, _BAD_NPU_BY_NAME, "merlin/python/merlin/dse/fake_space.py")
    assert [f["source"] for f in found] == ["npu_model"], found
    assert found[0]["how"] == "function name"


def test_a_grader_running_spike_for_a_verdict_is_not_caught(gate, tmp_path):
    """Fact vs verdict. A grader legitimately runs an ISS; it is not claiming anything about the HW."""
    found = _scan(gate, tmp_path, _VERDICT_ONLY, "merlin/python/merlin/runtime/fake_grader.py")
    assert found == [], found


def test_an_inline_cross_check_annotation_demotes_a_hit(gate, tmp_path):
    src = _BAD_CYCLOTRON.replace(
        'cfg = env("MERLIN_MUON_CONFIG")',
        'cfg = env("MERLIN_MUON_CONFIG")  # fact-source-ok: compared against the derived value only')
    found = _scan(gate, tmp_path, src, "merlin/python/merlin/targetgen/rtl/fake_introspect.py")
    assert [f["kind"] for f in found] == ["cross_check"], found


def test_the_ratchet_is_scoped_per_source_so_one_debt_cannot_excuse_another(gate):
    """A blanket per-file entry would let a module that has accepted its cyclotron debt start reading
    npu_model unnoticed. Keys are ``(path, source)``, and every entry carries a reason."""
    ratchet = gate.load_ratchet()
    assert ratchet, "the ratchet must be readable"
    for (path, source), reason in ratchet.items():
        assert source in gate.FORBIDDEN_SOURCES, f"{path}: unknown source id {source!r}"
        assert len(reason) > 40, f"{path}::{source} carries no real reason: {reason!r}"


def test_every_ratchet_entry_is_a_finding_the_gate_actually_produces(gate):
    """A ratchet entry the scan can never emit is worse than no entry: it reads as accounted-for debt
    while the gate is blind to it. Every entry must correspond to a live finding."""
    live = {(f["path"], f["source"]) for f in gate.findings()
            if f["kind"] in ("violation", "ratcheted")}
    stale = set(gate.load_ratchet()) - live
    assert not stale, f"ratchet entries the gate no longer produces (delete them): {sorted(stale)}"


def test_the_repo_has_no_unratcheted_fact_provenance_violation(gate):
    viol = [f for f in gate.findings() if f["kind"] == "violation"]
    assert not viol, "\n".join(f"{f['path']}:{f['lineno']} [{f['source']}] {f['literal']!r}"
                               for f in viol)


def test_the_gate_json_mode_is_machine_readable(gate, capsys):
    assert gate.main(["--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert set(doc) == {"violations", "ratcheted", "cross_checks", "n_ratchet_entries"}


def test_fail_on_any_actually_fails_while_the_default_reports(gate):
    """Reporting-only by default; the failing behaviour is opt-in and must really fail."""
    assert gate.main([]) == 0
    assert gate.main(["--fail-on-any"]) == 1     # the ratcheted debt is real and non-empty


# --------------------------------------------------------------------------------------------
# 2. The Muon introspect: derived, or UNKNOWN — never a plausible default
# --------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def introspect():
    from merlin.runtime.backends.base import get_backend
    return get_backend("muon").muon_introspect


def test_absent_elaboration_yields_unknown_not_a_default(introspect, monkeypatch):
    """The regression. Point every input at a path that cannot exist and confirm nothing is invented.

    Before this change the same conditions produced ``lanes_per_warp=16, warps_per_core=8, cores=2,
    threads_per_core=128`` and a ``32 GFLOP/s`` peak, under an evidence string that quoted a config
    file no code had opened."""
    monkeypatch.setenv("MERLIN_CHIPYARD", "/nonexistent/chipyard")
    monkeypatch.setenv("MERLIN_MUON_CONFIG", "/nonexistent/config.toml")
    facts = introspect.build_facts()

    simt = facts["facts"]["simt"]
    assert facts["inputs"]["rtl_present"] is False
    for key in ("lanes_per_warp", "warps_per_core", "cores", "threads_per_core"):
        assert simt[key] is None, f"{key} was invented as {simt[key]!r}"
    for key in ("lanes_per_warp", "warps_per_core", "cores"):
        assert simt["provenance"][key]["state"] in ("absent", "undeterminable")
    # The block-level state the bundle adapter reads: a block is only derived when EVERY part of it
    # was read, so a partial reading can never be rendered to an agent as grounded.
    assert simt["state"] in ("absent", "undeterminable")

    fp = facts["facts"]["fp_datapath"]
    assert fp["peak_flops_per_cycle"] is None and fp["peak_gflops"] is None and fp["clock_hz"] is None
    assert facts["facts"]["shared_memory"]["bytes_per_cluster"] is None
    assert facts["facts"]["registers"]["arch_max"] is None


def test_the_perf_model_is_recorded_as_a_cross_check_never_as_the_source(introspect):
    """The cyclotron config may confirm a derived fact; it may not supply one."""
    facts = introspect.build_facts()
    assert facts["inputs"]["perf_model_role"] == "cross-check only (not a fact source)"
    method = facts["generator"]["method"]
    assert "CROSS-CHECK, never as a fact source" in method
    for rec in facts["facts"]["simt"]["provenance"].values():
        xc = rec.get("cross_check")
        if xc is not None:
            assert "NOT RTL" in xc["source"]


@pytest.mark.skipif(not (repo_root() / ".env").exists(), reason="no .env: no elaboration to read")
def test_geometry_is_read_out_of_the_elaboration_when_it_is_present(introspect):
    """Every derived number must quote the RTL token it came from — the property the old evidence
    string only claimed. Skipped where no elaboration is reachable; the point is that WHEN it is
    reachable, the numbers move with the RTL."""
    facts = introspect.build_facts()
    if not facts["inputs"]["rtl_present"]:
        pytest.skip("MERLIN_CHIPYARD resolves, but the elaboration tree is not present")
    prov = facts["facts"]["simt"]["provenance"]
    assert prov["lanes_per_warp"]["state"] == "derived"
    assert "tmask" in prov["lanes_per_warp"]["evidence"]        # one mask bit per lane
    assert prov["warps_per_core"]["state"] == "derived"
    assert "perWarp" in prov["warps_per_core"]["evidence"]      # one counter set per warp
    assert prov["cores"]["state"] == "derived"
    assert "instance tree" in prov["cores"]["evidence"]
    assert facts["facts"]["simt"]["state"] == "derived"
    # ...and the bundle the agent is shown must AGREE that it is grounded. It did not, for one commit:
    # the state lived only in `provenance`, so the block carried no top-level `state` and the SIMT
    # renderer printed "Execution geometry: unavailable" over a fully derived geometry.
    from merlin.targetgen.rtl import mlc_bridge
    assert "Execution geometry**: lanes_per_warp=" in mlc_bridge.render_fact_bundle_for("muon")
    # Shared-memory capacity summed from the RTL's OWN SRAM macros, not a 128*1024 literal.
    smem = facts["facts"]["shared_memory"]
    assert smem["state"] == "derived" and "mems.conf" not in smem["evidence"]
    assert "depth" in smem["evidence"] and "width" in smem["evidence"]
    # The clock comes from the elaborated device tree, not a module-level CLOCK_HZ literal.
    assert "fixed-clock" in facts["facts"]["fp_datapath"]["clock_evidence"]


@pytest.mark.skipif(not (repo_root() / ".env").exists(), reason="no .env: no elaboration to read")
def test_a_cross_check_disagreement_is_surfaced_not_swallowed(introspect):
    """The elaborated RadianceMuonConfig holds ONE MuonCore; the cyclotron config declares
    ``num_cores = 2``. A perf model configured for a different core count is a finding, so the
    disagreement is recorded beside the derived value rather than resolved by picking one."""
    facts = introspect.build_facts()
    if not facts["inputs"]["rtl_present"] or facts["cross_check"]["state"] != "derived":
        pytest.skip("no elaboration and/or no perf-model config on this machine")
    cores = facts["facts"]["simt"]["provenance"]["cores"]
    xc = cores.get("cross_check")
    assert xc is not None and xc["key"] == "num_cores"
    # Whatever the two say, the DERIVED value is the one published and `agrees` states the truth.
    assert xc["agrees"] == (facts["facts"]["simt"]["cores"] == xc["value"])


# --------------------------------------------------------------------------------------------
# 3. The bundle adapter: "derived" means read, not present
# --------------------------------------------------------------------------------------------
def test_a_block_with_no_state_is_not_counted_as_derived():
    """``"derived": name in f`` was the bug: key presence proves the extractor ran, never that it read
    anything. An introspect that declares no state gets no benefit of the doubt."""
    from merlin.targetgen.rtl import mlc_bridge

    rec = mlc_bridge._simt_field("simt", {"lanes_per_warp": 16, "cores": 2}, "fake_introspect")
    assert rec["derived"] is False
    assert "declares no `state`" in rec["evidence"]


def test_state_absent_and_undeterminable_are_both_not_derived():
    from merlin.targetgen.rtl import mlc_bridge

    for state in ("absent", "undeterminable"):
        rec = mlc_bridge._simt_field("simt", {"cores": None, "state": state, "evidence": "x"}, "i")
        assert rec["derived"] is False and rec["state"] == state
    ok = mlc_bridge._simt_field("simt", {"cores": 1, "state": "derived", "evidence": "y"}, "i")
    assert ok["derived"] is True


def test_simt_facts_publishes_nothing_that_was_not_derived(monkeypatch):
    """The manifest deriver grounds ``capabilities.simt`` from this. With no elaboration it must get
    nothing rather than a shape full of defaults."""
    from merlin.targetgen.rtl import mlc_bridge

    monkeypatch.setenv("MERLIN_CHIPYARD", "/nonexistent/chipyard")
    monkeypatch.setenv("MERLIN_MUON_CONFIG", "/nonexistent/config.toml")
    monkeypatch.setenv("MERLIN_MLC_DIR", "/nonexistent/mlc")
    mlc_bridge.clear_caches() if hasattr(mlc_bridge, "clear_caches") else None
    body = mlc_bridge.simt_facts("muon").get("facts", {})
    assert "simt" not in body, body
    assert "memories" not in body, body
