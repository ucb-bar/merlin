"""Phase 4 — the onboarding proof: a target brings up its full scaffolding from {RTL + a capability
manifest + a descriptor} with ZERO target-specific code in the core generators. Exercised on radiance
(SIMT), the target most different from gemmini, so the whole shared-infra pipeline is validated for a
non-systolic accelerator.

Also guards the overfit smell test: the core generators carry no hardcoded target NAME.
"""
from __future__ import annotations

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.target_experiment import load_target_experiment, load_capability_manifest
from merlin.targetgen.runner_config import runner_config_from_manifest
from merlin.targetgen.generate_prompt import prompt_slots, render_prompt
from merlin.targetgen.generate_bundles import generate_bundles

_RAD_DESC = "merlin/experiments/capsule_bench/targets/radiance/target_experiment.yaml"
_SATURN_DESC = "merlin/experiments/capsule_bench/targets/saturn_opu/target_experiment.yaml"
_SATURN_RESID = "merlin/experiments/capsule_bench/targets/saturn_opu/manifest_residual.yaml"


def _load_radiance(monkeypatch):
    # radiance is an out-of-tree target discovered via MERLIN_TARGET_PATH — the documented plug-in path.
    monkeypatch.setenv("MERLIN_TARGET_PATH", "out/artifacts/targets")
    try:
        return load_target_experiment(_RAD_DESC), load_capability_manifest("radiance")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"radiance OOT target not resolvable in this checkout: {e}")


def test_saturn_opu_onboards_end_to_end_as_command_buffer_spatial_target(monkeypatch, tmp_path):
    # A SPATIAL (OuterProductUnit) target — NON-systolic and ISA-LESS (command_buffer, not RoCC .insn) —
    # onboards from {RTL facts + a GENERATED contract + a descriptor} with zero target-specific core code:
    # the analog of the radiance proof for an accelerator class with no command ISA at all. Gated on the
    # OPU arc artifacts (skip when absent); the contract is derived, never hand-authored.
    import yaml
    from merlin.targetgen import capability_manifests as cm
    from merlin.common.yaml import write_yaml
    try:
        from merlin.targetgen.rtl import spatial_introspect as si
        facts = si.build_fact_bundle("saturn_opu_mxv256d128")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"saturn_opu OPU facts not available in this checkout: {e}")
    # build_fact_bundle never raises — it returns an honest all-unavailable bundle (n_derived==0) when
    # mlc / the OPU arc artifacts cannot be read. The end-to-end proof (tile geometry -> the prompt's
    # outer-product framing, datapaths -> the dtype matrix) genuinely needs those grounded facts, so
    # gate on them here rather than deriving from a hollow bundle (the docstring's "skip when absent").
    if not facts.get("n_derived"):
        pytest.skip(f"saturn_opu OPU facts not grounded in this checkout: {facts.get('reason')}")
    descriptor = yaml.safe_load(open(_SATURN_DESC))
    residual = yaml.safe_load(open(_SATURN_RESID))
    manifest = cm.derive_manifest(descriptor, facts, residual=residual)
    # FACTS grounded onto the residual: multi-format datapaths + the command_buffer endpoint
    assert manifest["endpoint_kind"] == "command_buffer"
    # saturn's RTL names no 8-bit float format, so the fp8 sub-format identity is UNKNOWN. The actionable
    # dtype list keeps only registry-known formats (int8); the unnamed float datapath is SURFACED (never a
    # fabricated fp8_e4m3/fp8_e5m2 pair conjured from io_op_altfmt_ presence, never silently dropped).
    assert manifest["compute_units"][0]["dtypes"] == ["int8"]
    assert manifest["compute_units"][0]["unnamed_float_datapaths"] == ["float8"]
    assert "encoding" not in manifest                       # no RoCC funct decode
    # generate the OOT contract into a tmp target path + discover it (radiance-style MERLIN_TARGET_PATH)
    write_yaml(tmp_path / "saturn_opu_oot" / "contracts" / "target_contract.yaml", manifest,
               header="test-generated saturn_opu contract")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(tmp_path))
    te = load_target_experiment(_SATURN_DESC)
    m = load_capability_manifest("saturn_opu_mxv256d128")
    assert m.kind == "spatial" and m.endpoint_kind == "command_buffer" and m.encoding_required is False
    p = render_prompt(te, m, "full", "raw_baseline").lower()
    assert "command buffer" in p and "outer-product" in p          # spatial/command-buffer framing
    assert "rocc" not in p and "gemmini" not in p                  # no systolic/RoCC leakage
    assert "emitted module defines" not in p                       # not a false .insn-module claim


def test_radiance_onboards_end_to_end_from_manifest(monkeypatch):
    te, m = _load_radiance(monkeypatch)
    # SIMT routing comes entirely from the manifest's compute-unit kind via the family registry
    assert m.kind == "simt"
    assert m.encoding_required is False and m.trace_gate is None   # no .insn encoding / no RoCC trace gate

    cfg = runner_config_from_manifest(m)
    assert cfg.trace_gate is None                                 # the runner skips the RoCC trace stage
    assert cfg.perf_fields == ("flops", "gflops", "pct_fp_peak")  # SIMT perf headline

    s = prompt_slots(te, m)
    assert s["tool_stem"] == "radiance-opt" and s["kernel_symbol"] == "radiance_kernel"  # derived, not gemmini
    assert "gemmini" not in s["endpoint_desc"].lower() and "rocc" not in s["endpoint_desc"].lower()

    # 4 ladder rungs + the eqsat modality arm added in 6c5c4274. Asserted as a count of the arms
    # this target actually generates, which is the property under test: onboarding a target from
    # its manifest alone yields the whole roster, not a subset.
    assert len(generate_bundles(te)) == 5


def test_core_generators_carry_no_hardcoded_target_name():
    """families / runner_config / generate_prompt must be target-name-free (kinds + slots only) — the
    overfit smell test. (capsule_runner keeps a gemmini default for back-compat and is excluded.)"""
    names = ("gemmini", "radiance", "muon", "atlas", "mx_gemmini")
    for rel in ("merlin/python/merlin/targetgen/families.py",
                "merlin/python/merlin/targetgen/runner_config.py",
                "merlin/python/merlin/targetgen/generate_prompt.py"):
        text = (repo_root() / rel).read_text()
        # strip line comments: only flag a target name used as a quoted code literal
        code = "\n".join(ln.split("#", 1)[0] for ln in text.splitlines())
        for n in names:
            assert f'"{n}"' not in code and f"'{n}'" not in code, f"{rel} hardcodes target name {n!r}"
